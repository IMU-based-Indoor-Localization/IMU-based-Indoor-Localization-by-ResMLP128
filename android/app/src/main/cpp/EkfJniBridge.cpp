/**
 * EkfJniBridge.cpp
 * ================
 * Kotlin (JVM) ↔ C++ SC-EKF 브리지.
 * JNI 함수 이름은 com.imulocal.EkfBridge 클래스와 대응.
 */

#include <jni.h>
#include <android/log.h>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include "ekf/imu_ekf.h"

#define TAG "ImuEkf"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ── 전역 필터 인스턴스 (단일 인스턴스 가정) ────────────────────
static std::unique_ptr<imu_ekf::ScEkf> g_ekf;

/**
 * EKF 전역 뮤텍스
 * ──────────────────────────────────────────────────────────────
 * propJob(100Hz propagate)과 inferJob(~1Hz update/marginalize)이
 * 동일한 g_ekf 객체를 동시에 접근할 수 있음.
 * marginalize() 가 Sigma_ 크기를 줄이는 도중 propagate() 가
 * 이전 크기로 행렬곱 시도 시 Eigen assertion 실패(SIGABRT).
 * 모든 JNI 함수 진입 시 이 뮤텍스를 잠가 직렬화함.
 */
static std::mutex g_ekf_mutex;

extern "C" {

// ─────────────────────────────────────────────────────────────
// JNI: 필터 생성 / 초기화
// ─────────────────────────────────────────────────────────────

/**
 * 필터 파라미터로 ScEkf 인스턴스를 생성한다.
 * params 배열 순서:
 *   [0] sigma_na  [1] sigma_ng  [2] ita_ba  [3] ita_bg
 *   [4] init_attitude_sigma  [5] init_yaw_sigma
 *   [6] init_vel_sigma  [7] init_pos_sigma
 *   [8] init_bg_sigma   [9] init_ba_sigma
 *   [10] g_norm  [11] meascov_scale
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeCreate(
        JNIEnv* env, jclass /*cls*/, jdoubleArray params) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    jdouble* p = env->GetDoubleArrayElements(params, nullptr);
    imu_ekf::FilterConfig cfg;
    cfg.sigma_na            = p[0];
    cfg.sigma_ng            = p[1];
    cfg.ita_ba              = p[2];
    cfg.ita_bg              = p[3];
    cfg.init_attitude_sigma = p[4];
    cfg.init_yaw_sigma      = p[5];
    cfg.init_vel_sigma      = p[6];
    cfg.init_pos_sigma      = p[7];
    cfg.init_bg_sigma       = p[8];
    cfg.init_ba_sigma       = p[9];
    cfg.g_norm              = p[10];
    cfg.meascov_scale       = p[11];
    env->ReleaseDoubleArrayElements(params, p, JNI_ABORT);

    g_ekf = std::make_unique<imu_ekf::ScEkf>(cfg);
    LOGI("ScEkf created");
}

/**
 * 첫 번째 가속도 샘플로 필터를 초기화한다.
 * @param t_us   타임스탬프 (마이크로초)
 * @param acc    double[3] {ax, ay, az}
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeInitialize(
        JNIEnv* env, jclass /*cls*/,
        jlong t_us, jdoubleArray acc_arr) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (!g_ekf) { LOGE("nativeInitialize: ekf not created"); return; }

    jdouble* a = env->GetDoubleArrayElements(acc_arr, nullptr);
    imu_ekf::Vec3 acc(a[0], a[1], a[2]);
    env->ReleaseDoubleArrayElements(acc_arr, a, JNI_ABORT);

    g_ekf->initialize(static_cast<int64_t>(t_us),
                      acc,
                      imu_ekf::Vec3::Zero(),
                      imu_ekf::Vec3::Zero());
    LOGI("ScEkf initialized at t=%" PRId64, (int64_t)t_us);
}

// ─────────────────────────────────────────────────────────────
// JNI: IMU 전파
// ─────────────────────────────────────────────────────────────

/**
 * IMU 1 샘플을 전파한다.
 * @param acc_arr        double[3] {ax, ay, az}  (m/s²)
 * @param gyr_arr        double[3] {gx, gy, gz}  (rad/s)
 * @param t_us           현재 타임스탬프 (μs)
 * @param t_augment_us   클론 삽입 타임스탬프 (-1 이면 미삽입)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativePropagate(
        JNIEnv* env, jclass /*cls*/,
        jdoubleArray acc_arr, jdoubleArray gyr_arr,
        jlong t_us, jlong t_augment_us) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (!g_ekf || !g_ekf->is_initialized()) return;

    jdouble* a = env->GetDoubleArrayElements(acc_arr, nullptr);
    jdouble* g = env->GetDoubleArrayElements(gyr_arr, nullptr);

    imu_ekf::Vec3 acc(a[0], a[1], a[2]);
    imu_ekf::Vec3 gyr(g[0], g[1], g[2]);

    env->ReleaseDoubleArrayElements(acc_arr, a, JNI_ABORT);
    env->ReleaseDoubleArrayElements(gyr_arr, g, JNI_ABORT);

    g_ekf->propagate(acc, gyr,
                     static_cast<int64_t>(t_us),
                     static_cast<int64_t>(t_augment_us));
}

// ─────────────────────────────────────────────────────────────
// JNI: 측정 갱신
// ─────────────────────────────────────────────────────────────

/**
 * 네트워크 변위 예측으로 EKF를 갱신한다.
 * @param meas_arr     double[3] {dx, dy, dz}
 * @param cov_arr      double[9] 3×3 공분산 (row-major)
 * @param t_begin_us   윈도우 시작 타임스탬프 (μs)
 * @param t_end_us     윈도우 끝 타임스탬프 (μs)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeUpdate(
        JNIEnv* env, jclass /*cls*/,
        jdoubleArray meas_arr, jdoubleArray cov_arr,
        jlong t_begin_us, jlong t_end_us) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (!g_ekf || !g_ekf->is_initialized()) return;

    jdouble* m = env->GetDoubleArrayElements(meas_arr, nullptr);
    jdouble* c = env->GetDoubleArrayElements(cov_arr, nullptr);

    imu_ekf::Vec3 meas(m[0], m[1], m[2]);
    imu_ekf::Mat3 cov;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cov(i, j) = c[i * 3 + j];

    env->ReleaseDoubleArrayElements(meas_arr, m, JNI_ABORT);
    env->ReleaseDoubleArrayElements(cov_arr,  c, JNI_ABORT);

    try {
        g_ekf->update(meas, cov,
                      static_cast<int64_t>(t_begin_us),
                      static_cast<int64_t>(t_end_us));
    } catch (const std::exception& e) {
        LOGE("nativeUpdate error: %s", e.what());
    }
}

// ─────────────────────────────────────────────────────────────
// JNI: 상태 조회
// ─────────────────────────────────────────────────────────────

/**
 * 현재 위치와 표준편차를 반환한다.
 * @return double[6] {px, py, pz, sx, sy, sz}
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_imulocal_EkfBridge_nativeGetPosition(
        JNIEnv* env, jclass /*cls*/) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    jdoubleArray result = env->NewDoubleArray(6);
    if (!g_ekf || !g_ekf->is_initialized()) return result;

    imu_ekf::Vec3 p   = g_ekf->position();
    imu_ekf::Vec3 std = g_ekf->position_std();

    jdouble buf[6] = {p(0), p(1), p(2), std(0), std(1), std(2)};
    env->SetDoubleArrayRegion(result, 0, 6, buf);
    return result;
}

/**
 * 현재 속도를 반환한다.
 * @return double[3] {vx, vy, vz}
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_imulocal_EkfBridge_nativeGetVelocity(
        JNIEnv* env, jclass /*cls*/) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    jdoubleArray result = env->NewDoubleArray(3);
    if (!g_ekf || !g_ekf->is_initialized()) return result;

    imu_ekf::Vec3 v = g_ekf->velocity();
    jdouble buf[3] = {v(0), v(1), v(2)};
    env->SetDoubleArrayRegion(result, 0, 3, buf);
    return result;
}

// ─────────────────────────────────────────────────────────────
// JNI: EKF 파라미터 동적 변경
// ─────────────────────────────────────────────────────────────

/** 분류기 결과에 따라 meascov_scale 을 갱신 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeSetMeasCovScale(
        JNIEnv* /*env*/, jclass /*cls*/, jdouble scale) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf) g_ekf->set_meascov_scale(scale);
}

/**
 * 분류기 결과에 따라 IMU 프로세스 노이즈(Q/W)를 실시간 갱신.
 * Context-Aware Adaptive EKF: 논문 §4.3.2 Q 조정 방향 구현.
 * @param sigma_na  가속도 노이즈 표준편차 (m/s²)
 * @param sigma_ng  자이로 노이즈 표준편차 (rad/s)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeSetProcessNoise(
        JNIEnv* /*env*/, jclass /*cls*/, jdouble sigma_na, jdouble sigma_ng) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf) g_ekf->set_process_noise(sigma_na, sigma_ng);
}

// ─────────────────────────────────────────────────────────────
// JNI: 주변화 / 리셋
// ─────────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeMarginalize(
        JNIEnv* /*env*/, jclass /*cls*/, jint cut_idx) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf) g_ekf->marginalize(static_cast<int>(cut_idx));
}

JNIEXPORT jboolean JNICALL
Java_com_imulocal_EkfBridge_nativeIsInitialized(
        JNIEnv* /*env*/, jclass /*cls*/) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    return (g_ekf && g_ekf->is_initialized()) ? JNI_TRUE : JNI_FALSE;
}

// ─────────────────────────────────────────────────────────────
// [P9c] JNI: 클론 전체 플러시 (STATIC→MOVING 전환 시 stale 클론 제거)
// ─────────────────────────────────────────────────────────────

/**
 * EKF 내부의 모든 클론(과거 상태 사본)을 제거.
 * STATIC 중 marginalize가 호출되지 않아 stale 클론이 남아있을 때 사용.
 * marginalize(N-1)을 호출하면 인덱스 0..N-1 전부 삭제됨.
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeFlushClones(
        JNIEnv* /*env*/, jclass /*cls*/) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf && g_ekf->is_initialized()) {
        int N = g_ekf->clone_count();
        if (N > 0) g_ekf->marginalize(N - 1);
    }
}

// ─────────────────────────────────────────────────────────────
// JNI: 클론 회전 행렬 반환 (좌표 변환용)
// ─────────────────────────────────────────────────────────────

/**
 * t_begin 클론의 회전 행렬(world←body)을 double[9] (row-major) 로 반환.
 * LocalizationViewModel.transformWindowToWorldFrame() 에서 yaw 추출용으로 호출.
 * 클론이 없거나 EKF 미초기화 시 빈 배열 반환.
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_imulocal_EkfBridge_nativeGetCloneRotation(
        JNIEnv* env, jclass /*cls*/, jlong t_us) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    jdoubleArray result = env->NewDoubleArray(0);  // 기본: 빈 배열
    if (!g_ekf || !g_ekf->is_initialized()) return result;

    const imu_ekf::FilterState& st = g_ekf->state();
    auto it = std::find(st.si_timestamps_us.begin(),
                        st.si_timestamps_us.end(),
                        static_cast<int64_t>(t_us));
    if (it == st.si_timestamps_us.end()) {
        LOGE("nativeGetCloneRotation: ts=%" PRId64 " not found (N=%d)",
             (int64_t)t_us, st.N());
        return result;
    }

    int idx = static_cast<int>(std::distance(st.si_timestamps_us.begin(), it));
    const imu_ekf::Mat3& R = st.si_Rs[idx];

    result = env->NewDoubleArray(9);
    jdouble buf[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            buf[i * 3 + j] = R(i, j);
    env->SetDoubleArrayRegion(result, 0, 9, buf);
    return result;
}

// ─────────────────────────────────────────────────────────────
// JNI: 자이로 편향 반환 (좌표 변환 내부 자이로 적분 보정용)
// ─────────────────────────────────────────────────────────────

/**
 * 현재 EKF 자이로 편향 추정값을 double[3]으로 반환.
 * LocalizationViewModel.transformWindowToWorldFrame() 에서 Rs_bofbi 적분 보정에 사용.
 * EKF 미초기화 시 [0,0,0] 반환.
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_imulocal_EkfBridge_nativeGetGyrBias(
        JNIEnv* env, jclass /*cls*/) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    jdoubleArray result = env->NewDoubleArray(3);
    jdouble buf[3] = {0.0, 0.0, 0.0};

    if (g_ekf && g_ekf->is_initialized()) {
        const imu_ekf::Vec3& bg = g_ekf->state().bg;
        buf[0] = bg(0);
        buf[1] = bg(1);
        buf[2] = bg(2);
    }
    env->SetDoubleArrayRegion(result, 0, 3, buf);
    return result;
}

// ─────────────────────────────────────────────────────────────
// JNI: ZUPT (Zero Velocity UPdate)
// ─────────────────────────────────────────────────────────────

/**
 * 정지 상태에서 속도를 0 으로 제약하는 EKF 업데이트.
 * 가속도계 바이어스 추정도 점진적으로 보조함.
 *
 * @param sigma_zupt  속도 측정 노이즈 표준편차 (m/s)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeApplyZupt(
        JNIEnv* /*env*/, jclass /*cls*/, jdouble sigma_zupt) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf && g_ekf->is_initialized())
        g_ekf->apply_zupt(static_cast<double>(sigma_zupt));
}

// ─────────────────────────────────────────────────────────────
// JNI: Position Hold (정지 앵커 위치 고정)
// ─────────────────────────────────────────────────────────────

/**
 * 정지 확정 시 기록한 앵커 위치로 EKF 위치를 제약.
 * ZUPT(속도=0)와 병행하면 위치·속도 복합 고정 효과.
 * 가속도계 바이어스 적분에 의한 위치 드리프트를 직접 차단함.
 *
 * @param px, py, pz  앵커 위치 (월드 프레임, m)
 * @param sigma_pos   측정 노이즈 표준편차 (m)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeApplyPositionHold(
        JNIEnv* /*env*/, jclass /*cls*/,
        jdouble px, jdouble py, jdouble pz, jdouble sigma_pos) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf && g_ekf->is_initialized()) {
        imu_ekf::Vec3 anchor(px, py, pz);
        g_ekf->apply_position_hold(anchor, static_cast<double>(sigma_pos));
    }
}

// ─────────────────────────────────────────────────────────────
// [P9] JNI: Hard State Freeze (EKF 측정 우회 직접 고정)
// ─────────────────────────────────────────────────────────────

/**
 * [P9] 정지 상태에서 매 IMU 프레임마다 호출.
 * EKF 칼만 업데이트를 완전히 우회하여 state_.p, state_.v 를 직접 고정하고
 * 공분산 v/p 블록을 압축(1e-8)한다.
 * apply_position_hold() + apply_zupt() 조합의 칼만 게인 부족 문제를 근본 해결.
 *
 * @param px, py, pz  정지 확정 시점에 기록한 앵커 위치 (월드 프레임, m)
 */
JNIEXPORT void JNICALL
Java_com_imulocal_EkfBridge_nativeFreezeStaticState(
        JNIEnv* /*env*/, jclass /*cls*/,
        jdouble px, jdouble py, jdouble pz) {

    std::lock_guard<std::mutex> lock(g_ekf_mutex);
    if (g_ekf && g_ekf->is_initialized()) {
        imu_ekf::Vec3 anchor(px, py, pz);
        g_ekf->freeze_static_state(anchor);
    }
}

// ─────────────────────────────────────────────────────�