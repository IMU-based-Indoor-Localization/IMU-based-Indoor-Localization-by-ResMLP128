#pragma once
/**
 * imu_ekf.h
 * =========
 * Stochastic-Cloning Extended Kalman Filter (SC-EKF) for IMU-based indoor
 * localization. C++ port of src/tracker/scekf.py (Python / numba).
 *
 * 상태 벡터 (오차 공간, 15-dim):
 *   dθ [0:3]  – 회전 오차 (so3)
 *   dv [3:6]  – 속도 오차 (m/s)
 *   dp [6:9]  – 위치 오차 (m)
 *   dbg[9:12] – 자이로 편향 오차 (rad/s)
 *   dba[12:15]– 가속도 편향 오차 (m/s²)
 *
 * 과거 상태(클론) 하나당 6-dim [dθ, dp] 가 앞에 붙음.
 * 전체 공분산 크기 = (15 + 6*N) × (15 + 6*N)
 *
 * 의존 라이브러리:
 *   Eigen 3.4+  (헤더 온리)
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace imu_ekf {

// ─────────────────────────────────────────────────────────────────────────────
// 타입 별칭
// ─────────────────────────────────────────────────────────────────────────────
using Mat3  = Eigen::Matrix3d;
using Vec3  = Eigen::Vector3d;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using MatXX = Eigen::MatrixXd;
using VecX  = Eigen::VectorXd;

// ─────────────────────────────────────────────────────────────────────────────
// 수학 유틸리티
// ─────────────────────────────────────────────────────────────────────────────

/** 3-벡터 → 왜대칭 행렬 (skew-symmetric / hat operator) */
inline Mat3 hat(const Vec3& v) {
    Mat3 S;
    S <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return S;
}

/** SO(3) 지수 맵 (Rodrigues): so3 벡터 → 회전 행렬 */
inline Mat3 mat_exp(const Vec3& phi) {
    double angle = phi.norm();
    if (angle < 1e-10) return Mat3::Identity();
    Vec3 axis = phi / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

/** SO(3) 로그 맵: 회전 행렬 → so3 벡터 */
inline Vec3 mat_log(const Mat3& R) {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

/** SO(3) 지수 맵의 우-야코비안 Jr(φ) */
inline Mat3 Jr_exp(const Vec3& phi) {
    double angle = phi.norm();
    if (angle < 1e-10) return Mat3::Identity();
    Mat3 Phi = hat(phi);
    return Mat3::Identity()
         - ((1.0 - std::cos(angle)) / (angle * angle)) * Phi
         + ((angle - std::sin(angle)) / (angle * angle * angle)) * Phi * Phi;
}

/**
 * 중력 방향(acc 측정)으로부터 초기 회전 행렬을 계산.
 * ig_w = [0,0,1]^T (월드 z축 = 중력 방향)
 */
Mat3 get_rotation_from_gravity(const Vec3& acc);

// ─────────────────────────────────────────────────────────────────────────────
// IMU 전파 (1 스텝)
// ─────────────────────────────────────────────────────────────────────────────
struct PropagateResult {
    Mat3  R;    ///< 갱신된 회전
    Vec3  v;    ///< 갱신된 속도
    Vec3  p;    ///< 갱신된 위치
    Mat15 A;    ///< 15×15 선형화 야코비안
};

/**
 * propagate_rvt_and_jac
 * IMU 1 스텝을 적분하고 오차 전파 야코비안을 반환한다.
 *
 * @param R_k   현재 회전 행렬
 * @param v_k   현재 속도 (m/s)
 * @param p_k   현재 위치 (m)
 * @param bg_k  현재 자이로 편향 (rad/s)
 * @param ba_k  현재 가속도 편향 (m/s²)
 * @param gyr   자이로 측정값 (rad/s)
 * @param acc   가속도 측정값 (m/s²)
 * @param g     중력 벡터 (m/s²), 예: [0,0,-9.81]
 * @param dt    시간 간격 (s)
 */
PropagateResult propagate_rvt_and_jac(
    const Mat3& R_k, const Vec3& v_k, const Vec3& p_k,
    const Vec3& bg_k, const Vec3& ba_k,
    const Vec3& gyr, const Vec3& acc,
    const Vec3& g, double dt);

// ─────────────────────────────────────────────────────────────────────────────
// 공분산 전파
// ─────────────────────────────────────────────────────────────────────────────

/**
 * propagate_covariance
 * SC-EKF 공분산을 1 스텝 전파한다.
 *
 * Σ_{k+1} = A_aug Σ_k A_aug^T + B_aug W B_aug^T dt + Q_aug dt
 *
 * @param A_aug  전체 상태 야코비안 ((15+6N)×(15+6N_prev))
 * @param B_aug  노이즈 입력 행렬
 * @param dt     시간 간격 (s)
 * @param Sigma  현재 공분산
 * @param W      IMU 측정 노이즈 공분산 (6×6)
 * @param Q      프로세스 노이즈 (random walk, 전체 크기에 맞게 확장됨)
 */
MatXX propagate_covariance(
    const MatXX& A_aug, const MatXX& B_aug,
    double dt, const MatXX& Sigma,
    const Eigen::Matrix<double,6,6>& W,
    const MatXX& Q);

// ─────────────────────────────────────────────────────────────────────────────
// 필터 파라미터
// ─────────────────────────────────────────────────────────────────────────────
struct FilterConfig {
    double sigma_na             = std::sqrt(1e-3);   ///< 가속도 노이즈 (m/s²)
    double sigma_ng             = std::sqrt(1e-4);   ///< 자이로 노이즈 (rad/s)
    double ita_ba               = 1e-4;              ///< 가속도 편향 랜덤워크
    double ita_bg               = 1e-6;              ///< 자이로 편향 랜덤워크
    double init_attitude_sigma  = 10.0 / 180.0 * M_PI; ///< 초기 자세 불확도 (rad)
    double init_yaw_sigma       = 0.1  / 180.0 * M_PI; ///< 초기 yaw 불확도 (rad)
    double init_vel_sigma       = 1.0;               ///< 초기 속도 불확도 (m/s)
    double init_pos_sigma       = 0.001;             ///< 초기 위치 불확도 (m)
    double init_bg_sigma        = 0.0001;            ///< 초기 자이로 편향 불확도
    double init_ba_sigma        = 0.02;              ///< 초기 가속도 편향 불확도
    double g_norm               = 9.81;              ///< 중력 크기 (m/s²)
    double meascov_scale        = 1.0;               ///< 측정 공분산 스케일 팩터
    double mahalanobis_fail_scale = 0.0;             ///< 0 이면 마할라노비스 실패 시 업데이트 생략
};

// ─────────────────────────────────────────────────────────────────────────────
// 필터 상태
// ─────────────────────────────────────────────────────────────────────────────
struct FilterState {
    Mat3 R;             ///< 현재 회전 행렬 (월드←바디)
    Vec3 v;             ///< 현재 속도 (월드 프레임)
    Vec3 p;             ///< 현재 위치 (월드 프레임)
    Vec3 ba;            ///< 가속도 편향
    Vec3 bg;            ///< 자이로 편향
    int64_t t_us{-1};   ///< 현재 타임스탬프 (마이크로초)

    // Stochastic Cloning 과거 상태
    std::vector<Mat3>    si_Rs;
    std::vector<Vec3>    si_ps;
    std::vector<int64_t> si_timestamps_us;

    int N() const { return static_cast<int>(si_Rs.size()); }
};

// ─────────────────────────────────────────────────────────────────────────────
// SC-EKF 메인 클래스
// ─────────────────────────────────────────────────────────────────────────────
class ScEkf {
public:
    explicit ScEkf(const FilterConfig& cfg = FilterConfig{});

    // ── 초기화 ──────────────────────────────────────────────
    /** 첫 번째 가속도 측정값에서 초기 회전을 추정하여 필터를 초기화 */
    void initialize(int64_t t_us, const Vec3& acc,
                    const Vec3& ba_init, const Vec3& bg_init);

    /** 외부에서 알려진 상태로 직접 초기화 */
    void initialize_with_state(int64_t t_us, const Mat3& R,
                                const Vec3& v, const Vec3& p,
                                const Vec3& ba_init, const Vec3& bg_init);

    // ── 전파 ────────────────────────────────────────────────
    /**
     * IMU 1 샘플을 전파한다.
     * @param acc             가속도 측정값 (m/s²)
     * @param gyr             자이로 측정값 (rad/s)
     * @param t_us            현재 타임스탬프 (μs)
     * @param t_augmentation_us 클론을 추가할 타임스탬프 (≥0 이면 추가, -1 이면 미추가)
     */
    void propagate(const Vec3& acc, const Vec3& gyr,
                   int64_t t_us, int64_t t_augmentation_us = -1);

    // ── 측정 갱신 ───────────────────────────────────────────
    /**
     * 네트워크 변위 예측을 측정값으로 EKF를 갱신한다.
     * @param meas         예측 변위 [3×1] (월드 프레임)
     * @param meas_cov     측정 공분산 [3×3]
     * @param t_begin_us   윈도우 시작 타임스탬프 (μs)  ← si_timestamps_us에 있어야 함
     * @param t_end_us     윈도우 끝 타임스탬프 (μs)
     */
    void update(const Vec3& meas, const Mat3& meas_cov,
                int64_t t_begin_us, int64_t t_end_us);

    // ── 주변화 ──────────────────────────────────────────────
    /** cut_idx 이전의 과거 상태를 주변화(marginalize)한다. */
    void marginalize(int cut_idx);

    /** 현재 EKF 상태에 존재하는 클론(과거 상태 사본) 수를 반환. */
    int clone_count() const { return state_.N(); }

    // ── 파라미터 동적 조정 ──────────────────────────────────
    /** 분류기 결과에 따라 meascov_scale 을 실시간으로 변경 */
    void set_meascov_scale(double scale) { cfg_.meascov_scale = scale; }

    /**
     * 분류기 결과에 따라 IMU 프로세스 노이즈를 실시간으로 변경.
     * W_ 행렬을 즉시 재구성하므로 다음 propagate() 부터 적용.
     * @param sigma_na  가속도 노이즈 표준편차 (m/s²)
     * @param sigma_ng  자이로 노이즈 표준편차 (rad/s)
     */
    void set_process_noise(double sigma_na, double sigma_ng) {
        cfg_.sigma_na = sigma_na;
        cfg_.sigma_ng = sigma_ng;
        double sna2 = sigma_na * sigma_na;
        double sng2 = sigma_ng * sigma_ng;
        W_.block<3,3>(0,0) = Mat3::Identity() * sng2;
        W_.block<3,3>(3,3) = Mat3::Identity() * sna2;
    }

    /**
     * ZUPT (Zero Velocity UPdate): 정지 상태에서 속도를 0으로 제약.
     *
     * 관측 모델:  v_measured = 0  (정지 → 속도는 영)
     * H_zupt = [0 | I₃ | 0 | 0 | 0]  (오차 상태 중 속도 블록 선택)
     * 이노베이션 = 0 − v_current
     *
     * 효과:
     *   1. EKF 속도를 빠르게 0 으로 되돌림 (IMU 적분 드리프트 억제)
     *   2. 교차 공분산을 통해 장기적으로 가속도계 바이어스(ba) 추정 보조
     *
     * @param sigma_zupt  속도 측정 노이즈 표준편차 (m/s, 기본 0.05)
     */
    void apply_zupt(double sigma_zupt = 0.05);

    /**
     * Position Hold: 정지 진입 시 기록한 앵커 위치로 절대 위치를 제약.
     *
     * 관측 모델: p_measured = p_anchor  (정지 → 위치는 고정)
     *   H_pos = [0 | 0 | I₃ | 0 | 0]  (오차 상태 중 위치 블록 선택)
     *   이노베이션 = p_anchor − p_current
     *
     * ZUPT(속도=0)와 함께 사용하면 위치·속도 복합 고정 효과.
     * 가속도계 바이어스(ba) 적분에 의한 드리프트를 직접 차단.
     *
     * @param p_anchor   정지 확정 시점의 EKF 위치 (월드 프레임, m)
     * @param sigma_pos  위치 측정 노이즈 표준편차 (m, 기본 0.01 = 1 cm)
     */
    void apply_position_hold(const Vec3& p_anchor, double sigma_pos = 0.01);

    /**
     * [P9] Hard State Freeze: EKF 측정 모델 우회하여 직접 상태·공분산 고정.
     *
     * apply_position_hold() + apply_zupt() 조합은 칼만 게인이 충분히 크지 않으면
     * 위치 드리프트를 막지 못한다. 이 함수는 EKF 업데이트를 완전히 우회하여
     * 직접 state_.p = p_anchor, state_.v = 0 으로 설정하고
     * 공분산 Σ[v,v], Σ[p,p] 블록을 거의 0 으로 압축(1e-8)한다.
     *
     * 정지 상태에서 매 IMU 프레임마다 호출하면 가속도계 바이어스 적분에 의한
     * 위치·속도 발산을 완전히 차단한다.
     *
     * @param p_anchor  정지 확정 시점의 EKF 위치 (월드 프레임, m)
     */
    void freeze_static_state(const Vec3& p_anchor);

    /**
     * [P9d] Thaw Static State: STATIC→MOVING 전환 시 공분산 해동.
     *
     * freeze_static_state() 는 Σ[v,v], Σ[p,p] 를 1e-8 로 압축한다.
     * 이 상태에서 MOVING 이 시작되면 propagation 으로 1초 후
     *   Σ[p,p] ≈ σ_na²·T³/3 ≈ 3e-4 → 칼만 게인 K ≈ 0.016 (1.6%)
     * → 측정값이 사실상 무시 → IMU 바이어스 적분 지배 → 발산.
     *
     * 이 함수는 STATIC 종료 시 한 번 호출하여 Σ[v,v], Σ[p,p] 를
     * 합리적인 불확실성 값으로 복원한다.
     *   Σ[v,v] = 0.01 m²/s²  (std = 0.1 m/s — 정지 후 속도 불확실성)
     *   Σ[p,p] = 0.01 m²     (std = 0.1 m  — 정지 후 위치 불확실성)
     * → K ≈ 0.01/(0.01+0.02) ≈ 0.33 → 33% 보정 → 측정값 반영 정상화
     */
    void thaw_static_state();

    /**
     * Yaw 절대값 업데이트: Android TYPE_ROTATION_VECTOR 의 yaw 를 EKF 에 주입.
     *
     * 관측 모델: ψ_measured = ψ_current + δψ
     *   H_yaw = ∂ψ/∂dθ (1×sz):  오차 상태 dθ 중 z 성분이 yaw 에 해당.
     *
     * 이노베이션 게이트: |δψ| > 45° 시 자기 간섭(Magnetic disturbance) 으로 판단 → 건너뜀.
     *
     * @param yaw_meas   EKF 월드 프레임 기준 측정 yaw (rad).
     *                   = atan2(R_rv[1,0], R_rv[0,0]) - yaw_rv_at_init
     * @param sigma_yaw  측정 노이즈 표준편차 (rad, 기본 10° ≈ 0.1745)
     */
    void apply_yaw_update(double yaw_meas,
                          double sigma_yaw = 10.0 / 180.0 * M_PI);

    // ── 상태 조회 ───────────────────────────────────────────
    bool        is_initialized()  const { return initialized_; }
    const FilterState& state()    const { return state_; }
    const MatXX&