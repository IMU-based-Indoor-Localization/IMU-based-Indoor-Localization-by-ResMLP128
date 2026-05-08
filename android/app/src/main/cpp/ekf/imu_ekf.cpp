/**
 * imu_ekf.cpp
 * ===========
 * SC-EKF 구현체. imu_ekf.h 참조.
 * Python 원본: src/tracker/scekf.py
 */

#include "imu_ekf.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace imu_ekf {

// ─────────────────────────────────────────────────────────────────────────────
// 수학 유틸리티 구현
// ─────────────────────────────────────────────────────────────────────────────

Mat3 get_rotation_from_gravity(const Vec3& acc) {
    // Python: rot_2vec(acc, ig_w) where ig_w = [0,0,1]
    Vec3 ig_w(0.0, 0.0, 1.0);
    Vec3 a_norm = acc.normalized();
    Vec3 axis   = a_norm.cross(ig_w);
    double s    = axis.norm();
    double c    = a_norm.dot(ig_w);
    if (s < 1e-10) {
        // 이미 정렬됨 → Identity
        if (c > 0.0) return Mat3::Identity();
        // 반대 방향 (acc ≈ -[0,0,1]): 180° 회전이 필요.
        // -I 는 det = -1 → SO(3) 원소가 아님 → 수정.
        // X축 기준 180° 회전: diag(1, -1, -1)
        Mat3 R180 = Mat3::Zero();
        R180(0, 0) =  1.0;
        R180(1, 1) = -1.0;
        R180(2, 2) = -1.0;
        return R180;
    }
    Mat3 K = hat(axis / s);
    return Mat3::Identity() + s * K + (1.0 - c) * K * K;
}

// ─────────────────────────────────────────────────────────────────────────────
// IMU 전파
// ─────────────────────────────────────────────────────────────────────────────

PropagateResult propagate_rvt_and_jac(
        const Mat3& R_k, const Vec3& v_k, const Vec3& p_k,
        const Vec3& bg_k, const Vec3& ba_k,
        const Vec3& gyr, const Vec3& acc,
        const Vec3& g, double dt) {

    // 편향 보정된 IMU 값
    Vec3 omega_c = gyr - bg_k;        // 보정된 각속도
    Vec3 acc_c   = acc - ba_k;        // 보정된 가속도

    // 회전 전파
    Vec3 dtheta = omega_c * dt;
    Mat3 dR     = mat_exp(dtheta);
    Mat3 Rd     = R_k * dR;

    // 속도/위치 전파
    Vec3 dv_w = R_k * acc_c * dt;
    Vec3 dp_w = 0.5 * dv_w * dt;
    Vec3 vd   = v_k + dv_w + g * dt;
    Vec3 pd   = p_k + v_k * dt + dp_w + g * 0.5 * dt * dt;

    // 선형화 야코비안 A (15×15)
    Mat15 A = Mat15::Identity();
    // 속도에 대한 자세 오차 영향: ∂dv/∂dθ = -hat(dv_w)
    A.block<3,3>(3, 0) = -hat(dv_w);
    // 위치에 대한 자세 오차 영향: ∂dp/∂dθ = -hat(dp_w)
    A.block<3,3>(6, 0) = -hat(dp_w);
    // 위치에 대한 속도 영향
    A.block<3,3>(6, 3) = Mat3::Identity() * dt;
    // 자이로 편향 → 자세 오차
    A.block<3,3>(0, 9)  = -Rd * Jr_exp(dtheta) * dt;
    // 가속도 편향 → 속도 오차
    A.block<3,3>(3, 12) = -R_k * dt;
    // 가속도 편향 → 위치 오차
    A.block<3,3>(6, 12) = -0.5 * R_k * dt * dt;

    return {Rd, vd, pd, A};
}

// ─────────────────────────────────────────────────────────────────────────────
// 공분산 전파
// ─────────────────────────────────────────────────────────────────────────────

MatXX propagate_covariance(
        const MatXX& A_aug, const MatXX& B_aug,
        double dt, const MatXX& Sigma,
        const Eigen::Matrix<double,6,6>& W,
        const MatXX& Q) {

    // Σ_{k+1} = A Σ A^T  +  B W B^T · dt  +  Q · dt
    MatXX Sigma_new = A_aug * Sigma * A_aug.transpose()
                    + B_aug * W * B_aug.transpose() * dt
                    + Q * dt;

    // 대칭화 (수치 오류 누적 방지)
    return 0.5 * (Sigma_new + Sigma_new.transpose());
}

// ─────────────────────────────────────────────────────────────────────────────
// ScEkf 구현
// ─────────────────────────────────────────────────────────────────────────────

ScEkf::ScEkf(const FilterConfig& cfg) : cfg_(cfg) {
    build_noise_matrices();
}

void ScEkf::build_noise_matrices() {
    // IMU 측정 노이즈 공분산 W (6×6 대각)
    W_.setZero();
    double sna2 = cfg_.sigma_na * cfg_.sigma_na;
    double sng2 = cfg_.sigma_ng * cfg_.sigma_ng;
    W_.block<3,3>(0,0) = Mat3::Identity() * sng2;  // 자이로 노이즈
    W_.block<3,3>(3,3) = Mat3::Identity() * sna2;  // 가속도 노이즈

    // 프로세스 노이즈 Q_15 (편향 랜덤워크, 15×15 구조)
    // 초기 Q_; 전파 시 전체 크기(15+6N)로 확장하여 사용
    Q_ = MatXX::Zero(15, 15);
    Q_.block<3,3>(9,9)  = Mat3::Identity() * (cfg_.ita_bg * cfg_.ita_bg); // 자이로 편향
    Q_.block<3,3>(12,12)= Mat3::Identity() * (cfg_.ita_ba * cfg_.ita_ba); // 가속도 편향
}

void ScEkf::reset_covariance() {
    // 15×15 초기 공분산
    Vec15 diag_15;
    double va = cfg_.init_attitude_sigma * cfg_.init_attitude_sigma;
    double vy = cfg_.init_yaw_sigma      * cfg_.init_yaw_sigma;
    double vv = cfg_.init_vel_sigma      * cfg_.init_vel_sigma;
    double vp = cfg_.init_pos_sigma      * cfg_.init_pos_sigma;
    double vbg= cfg_.init_bg_sigma       * cfg_.init_bg_sigma;
    double vba= cfg_.init_ba_sigma       * cfg_.init_ba_sigma;

    // Roll/Pitch 분산 = va, Yaw 분산 = vy
    diag_15 << va, va, vy,
               vv, vv, vv,
               vp, vp, vp,
               vbg,vbg,vbg,
               vba,vba,vba;

    Sigma_ = MatXX::Zero(15, 15);
    Sigma_.diagonal() = diag_15;
}

void ScEkf::initialize(int64_t t_us, const Vec3& acc,
                        const Vec3& ba_init, const Vec3& bg_init) {
    Mat3 R = get_rotation_from_gravity(acc);
    initialize_with_state(t_us, R, Vec3::Zero(), Vec3::Zero(), ba_init, bg_init);
}

void ScEkf::initialize_with_state(int64_t t_us, const Mat3& R,
                                    const Vec3& v, const Vec3& p,
                                    const Vec3& ba_init, const Vec3& bg_init) {
    state_.R  = R;
    state_.v  = v;
    state_.p  = p;
    state_.ba = ba_init;
    state_.bg = bg_init;
    state_.t_us = t_us;
    state_.si_Rs.clear();
    state_.si_ps.clear();
    state_.si_timestamps_us.clear();

    reset_covariance();
    initialized_ = true;
    first_update_ = true;
}

void ScEkf::propagate(const Vec3& acc, const Vec3& gyr,
                       int64_t t_us, int64_t t_augmentation_us) {
    if (!initialized_) return;

    const Vec3 g(0.0, 0.0, -cfg_.g_norm);
    int N = state_.N();
    double dt_us = static_cast<double>(t_us - state_.t_us);
    double dt    = dt_us * 1e-6;

    if (dt <= 0.0) return;

    auto res = propagate_rvt_and_jac(
        state_.R, state_.v, state_.p,
        state_.bg, state_.ba,
        gyr, acc, g, dt);

    // ── 노이즈 입력 행렬 B (15×6) ──────────────────────────────
    Eigen::Matrix<double,15,6> B = Eigen::Matrix<double,15,6>::Zero();
    B.block<3,3>(0, 0) = -res.A.block<3,3>(0, 9);   // gyro noise → θ
    B.block<3,3>(3, 3) = -res.A.block<3,3>(3, 12);  // acc noise → v
    B.block<3,3>(6, 3) = -res.A.block<3,3>(6, 12);  // acc noise → p

    bool do_augment = (t_augmentation_us >= 0);

    if (do_augment) {
        // 클론 타임스탬프까지 부분 적분
        double dtd = static_cast<double>(t_augmentation_us - state_.t_us) * 1e-6;
        auto resd = propagate_rvt_and_jac(
            state_.R, state_.v, state_.p,
            state_.bg, state_.ba,
            gyr, acc, g, dtd);

        // JA (6×15): 클론 위치/자세를 현재 오차 상태로 선형화
        Eigen::Matrix<double,6,15> JA = Eigen::Matrix<double,6,15>::Zero();
        JA.block<3,15>(0, 0) = resd.A.block<3,15>(0, 0);   // dθ
        JA.block<3,15>(3, 0) = resd.A.block<3,15>(6, 0);   // dp

        int new_sz = 15 + 6 * (N + 1);
        int old_sz = 15 + 6 * N;

        MatXX A_aug = MatXX::Zero(new_sz, old_sz);
        if (N > 0) A_aug.block(0, 0, 6*N, 6*N) = MatXX::Identity(6*N, 6*N);
        A_aug.block<6,15>(6*N,     old_sz - 15) = JA;
        A_aug.block<15,15>(6*N+6,  old_sz - 15) = res.A;

        // B_aug
        Eigen::Matrix<double,6,6> BJ = Eigen::Matrix<double,6,6>::Zero();
        BJ.block<3,3>(0,0) = -resd.A.block<3,3>(0, 9);
        BJ.block<3,3>(3,3) = -resd.A.block<3,3>(6, 12);

        MatXX B_aug = MatXX::Zero(new_sz, 6);
        B_aug.block<6,6>(6*N,   0) = BJ;
        B_aug.block<15,6>(6*N+6,0) = B;

        // Q 확장
        MatXX Q_aug = MatXX::Zero(new_sz, new_sz);
        Q_aug.block<15,15>(new_sz-15, new_sz-15) = Q_.block<15,15>(0,0);

        Sigma_ = propagate_covariance(A_aug, B_aug, dt, Sigma_, W_, Q_aug);

        // 클론 추가
        state_.si_Rs.push_back(resd.R);
        state_.si_ps.push_back(resd.p);
        state_.si_timestamps_us.push_back(t_augmentation_us);

    } else {
        int sz = 15 + 6 * N;

        MatXX A_aug = MatXX::Identity(sz, sz);
        A_aug.block<15,15>(sz-15, sz-15) = res.A;

        MatXX B_aug = MatXX::Zero(sz, 6);
        B_aug.block<15,6>(sz-15, 0) = B;

        MatXX Q_aug = MatXX::Zero(sz, sz);
        Q_aug.block<15,15>(sz-15, sz-15) = Q_.block<15,15>(0,0);

        Sigma_ = propagate_covariance(A_aug, B_aug, dt, Sigma_, W_, Q_aug);
    }

    // 상태 갱신
    state_.R    = res.R;
    state_.v    = res.v;
    state_.p    = res.p;
    state_.t_us = t_us;

    // ── 속도 클램핑 (실내 환경 안전망) ────────────────────────────
    // IMU 적분 발산 방지: 속도가 실내 최대 허용값(5m/s)을 넘으면 방향은 유지한 채 크기만 제한.
    // 정상 보행/달리기 중에는 발동하지 않음; 발산 시 수렴을 돕는 최후 안전망.
    constexpr double MAX_INDOOR_SPEED = 5.0;  // m/s
    double spd = state_.v.norm();
    if (spd > MAX_INDOOR_SPEED) {
        state_.v *= MAX_INDOOR_SPEED / spd;
    }
}

void ScEkf::update(const Vec3& meas, const Mat3& meas_cov,
                    int64_t t_begin_us, int64_t t_end_us) {
    if (!initialized_) return;

    // 타임스탬프로 클론 인덱스 검색
    auto find_idx = [&](int64_t ts) -> int {
        auto it = std::find(state_.si_timestamps_us.begin(),
                            state_.si_timestamps_us.end(), ts);
        if (it == state_.si_timestamps_us.end())
            throw std::runtime_error("update: timestamp not found in past states");
        return static_cast<int>(std::distance(state_.si_timestamps_us.begin(), it));
    };

    int begin_idx = find_idx(t_begin_us);
    int end_idx   = find_idx(t_end_us);

    if (begin_idx >= end_idx)
        throw std::runtime_error("update: begin_idx >= end_idx");

    // 측정 공분산 스케일
    Mat3 R_meas = cfg_.meascov_scale * meas_cov;
    R_meas = 0.5 * (R_meas + R_meas.transpose());

    // begin 클론의 회전에서 Z축 회전 추출 (yaw-only 관측 모델)
    const Mat3& Ri = state_.si_Rs[begin_idx];

    // ZYX (= extrinsic XYZ) 오일러 각도 직접 추출 — Python scekf.py 와 동일
    //   R = Rz(ψ) * Ry(θ) * Rx(φ)
    //   ψ (yaw)   = atan2(R[1,0], R[0,0])
    //   θ (pitch) = atan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
    // 이전 코드: eulerAngles(0,1,2) = intrinsic XYZ → Python/Kotlin 과 다른 각도값 → 발산
    double ri_z = std::atan2(Ri(1, 0), Ri(0, 0));
    double ri_y = std::atan2(-Ri(2, 0),
                             std::sqrt(Ri(0,0)*Ri(0,0) + Ri(1,0)*Ri(1,0)));

    // Yaw 회전 행렬
    Mat3 Ri_z;
    Ri_z << std::cos(ri_z), -std::sin(ri_z), 0.0,
            std::sin(ri_z),  std::cos(ri_z), 0.0,
            0.0,             0.0,            1.0;

    // 예측값: R_z^T * (p_end - p_begin)
    Vec3 pred = Ri_z.transpose() * (state_.si_ps[end_idx] - state_.si_ps[begin_idx]);

    // 특이점 체크
    if (std::abs(std::cos(ri_y)) < 1e-5) return; // singularity → skip

    // 관측 행렬 H (3 × (15 + 6N))
    int N = state_.N();
    int sz = 15 + 6 * N;
    MatXX H = MatXX::Zero(3, sz);

    H.block<3,3>(0, 6*begin_idx + 3) = -Ri_z.transpose();
    H.block<3,3>(0, 6*end_idx   + 3) =  Ri_z.transpose();

    // H 에서 rotation 부분 (yaw 야코비안)
    Eigen::Matrix<double,3,3> Hz;
    double cy = std::cos(ri_y), sy = std::sin(ri_y);
    double cz = std::cos(ri_z), sz_val = std::sin(ri_z);
    double tan_y = sy / cy;
    Hz << 0, 0, 0,
          0, 0, 0,
          cz * tan_y, sz_val * tan_y, 1.0;

    Vec3 dp_end_begin = state_.si_ps[end_idx] - state_.si_ps[begin_idx];
    H.block<3,3>(0, 6*begin_idx) =
        Ri_z.transpose() * hat(dp_end_begin) * Hz;

    // 이노베이션 공분산 S  (MatXX로 선언 — H*Σ*H^T가 dynamic MatXX 반환)
    MatXX S = H * Sigma_ * H.transpose() + R_meas;

    Vec3 innov = meas - pred;

    // ── 절대 이노베이션 게이트 (물리 단위) ──────────────────────────
    // meascov_scale 이 크면 S도 커져 Mahalanobis NSE 가 항상 작아짐 →
    // 통계 게이트가 쓸모없어지는 문제 해결.
    // 실내 최대속도 ~5m/s × ~1s 윈도우 = 5m → 이노베이션 > 6m 는 물리적으로 불가 → 건너뜀.
    // 좌표 변환 오류/네트워크 이상 출력으로 인한 급격한 발산 방어.
    constexpr double MAX_INNOV_NORM = 6.0;  // m
    if (innov.norm() > MAX_INNOV_NORM) return;

    // Mahalanobis gating (chi^2 임계값 11.345, ν=3, p=0.99)
    MatXX S_inv = S.inverse();
    // 1×1 행렬 → scalar: (0,0) 으로 명시 추출 (Eigen은 암묵 변환 없음)
    double nse = (innov.transpose() * S_inv * innov)(0, 0);
    if (nse > 11.345) {
        if (cfg_.mahalanobis_fail_scale <= 0.0) return; // 업데이트 건너뜀
        R_meas = cfg_.mahalanobis_fail_scale * R_meas;
        S = H * Sigma_ * H.transpose() + R_meas;
        S_inv = S.inverse();
    }

    // 칼만 이득  K = Σ H^T S^{-1}  (sz×3)
    MatXX K = Sigma_ * H.transpose() * S_inv;

    // 상태 보정
    VecX delta_X = K * innov;
    apply_correction(delta_X);

    // 공분산 갱신 (Joseph form for numerical stability)
    MatXX I_KH = MatXX::Identity(sz, sz) - K * H;
    MatXX R_meas_dyn = R_meas;  // Mat3 → MatXX 변환 (K * fixed_Mat3 곱 타입 모호성 방지)
    Sigma_ = I_KH * Sigma_ * I_KH.transpose()
           + K * R_meas_dyn * K.transpose();
    Sigma_ = 0.5 * (Sigma_ + Sigma_.transpose());

    first_update_ = false;
}

void ScEkf::apply_position_hold(const Vec3& p_anchor, double sigma_pos) {
    if (!initialized_) return;

    int N        = state_.N();
    int sz       = 15 + 6 * N;
    int pos_base = 6 * N + 6;  // 오차 상태 벡터에서 위치 dp 시작 인덱스
                                // 상태 배치: [clone×N | dθ(3) dv(3) dp(3) dbg(3) dba(3)]

    // 앵커와 현재 위치 차이가 이미 충분히 작으면 건너뜀 (수치 노이즈 방지)
    Vec3 innov = p_anchor - state_.p;
    if (innov.norm() < 1e-6) return;

    // H: 위치 3 성분을 선택하는 3×sz 관측 행렬
    //   H[0:3, pos_base:pos_base+3] = I₃
    MatXX H = MatXX::Zero(3, sz);
    H.block<3, 3>(0, pos_base) = Mat3::Identity();

    // R_pos: 측정 노이즈 (sigma_pos² × I₃)
    Mat3 R_pos = (sigma_pos * sigma_pos) * Mat3::Identity();

    // 이노베이션 공분산 S = H Σ H^T + R_pos
    MatXX S = H * Sigma_ * H.transpose() + R_pos;

    // 칼만 이득 K = Σ H^T S^{-1}
    MatXX K = Sigma_ * H.transpose() * S.inverse();

    // 상태 보정 (위치 오차 외에 속도·자세·편향 교차 공분산도 함께 보정됨)
    VecX delta_X = K * innov;
    apply_correction(delta_X);

    // 공분산 갱신 (Joseph form — 수치 안정성)
    MatXX I_KH = MatXX::Identity(sz, sz) - K * H;
    MatXX R_dyn = R_pos;
    Sigma_ = I_KH * Sigma_ * I_KH.transpose()
           + K * R_dyn * K.transpose();
    Sigma_ = 0.5 * (Sigma_ + Sigma_.transpose());
}

void ScEkf::apply_zupt(double sigma_zupt) {
    if (!initialized_) return;

    int N   = state_.N();
    int sz  = 15 + 6 * N;
    int vel_base = 6 * N + 3;  // 오차 상태 벡터에서 속도 시작 인덱스

    // 현재 속도가 이미 충분히 작으면 건너뜀 (수치 노이즈 방지)
    if (state_.v.norm() < 1e-6) return;

    // H: 속도 3개 성분을 선택하는 3×sz 관측 행렬
    MatXX H = MatXX::Zero(3, sz);
    H.block<3, 3>(0, vel_base) = Mat3::Identity();

    // R_zupt: 측정 노이즈 (sigma² × I₃)
    Mat3 R_zupt = (sigma_zupt * sigma_zupt) * Mat3::Identity();

    // 이노베이션: "측정값 0" − "현재 속도 추정값"
    Vec3 innov = -state_.v;

    // 이노베이션 공분산 S = H Σ H^T + R_zupt
    MatXX S = H * Sigma_ * H.transpose() + R_zupt;

    // 칼만 이득 K = Σ H^T S^{-1}
    MatXX K = Sigma_ * H.transpose() * S.inverse();

    // 상태 보정
    VecX delta_X = K * innov;
    apply_correction(delta_X);

    // 공분산 갱신 (Joseph form)
    MatXX I_KH = MatXX::Identity(sz, sz) - K * H;
    MatXX R_dyn = R_zupt;
    Sigma_ = I_KH * Sigma_ * I_KH.transpose()
           + K * R_dyn * K.transpose();
    Sigma_ = 0.5 * (Sigma_ + Sigma_.transpose());
}

void ScEkf::apply_yaw_update(double yaw_meas, double sigma_yaw) {
    if (!initialized_) return;

    int N          = state_.N();
    int sz         = 15 + 6 * N;
    int theta_base = 6 * N;   /