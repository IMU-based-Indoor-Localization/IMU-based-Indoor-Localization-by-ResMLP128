package com.imulocal

import kotlin.math.sqrt

/**
 * EkfBridge.kt
 * ============
 * C++ SC-EKF 와 Kotlin 사이의 JNI 브리지.
 *
 * Context-Aware Adaptive EKF (논문 §4.3.2) 구현:
 *   - 분류 확률 벡터(clsProb[7])를 이용한 소프트 스위칭(soft-switching)
 *   - Q (σ_na) 와 R (meascov_scale) 을 분류 확률 가중 평균으로 연속 조정
 *   - Q_adaptive = Σ p_k · Q^(k),  R_adaptive = Σ p_k · R^(k)
 *
 * 클래스 인덱스 (train.py LABEL_REMAP 기준):
 *   0=handbag  1=handheld  2=pocket  3=running  4=slow_walk  5=trolley  6=unknown
 */
object EkfBridge {

    init {
        System.loadLibrary("imu_ekf_jni")
    }

    // ── 기본 EKF 파라미터 ──────────────────────────────────────
    // Python filter_batch.py / main_filter.py 기본값과 동일
    //
    // [P14 → P15 롤백 이력]
    //   P14 에서 meascov_scale 기본값을 10.0 → 1.0 으로 낮추고
    //   R_SCALE_PER_CLASS 를 Python ekf_tune.py grid-search 결과(0.001~1.0)와
    //   동기화하였으나, 어플리케이션 환경에서 EKF 가 발산하여 궤적이
    //   사라지는 부작용이 발생하였다. 원인은 Python grid-search 가 GT 자세를
    //   사용한 환경에서 도출된 값이라 측정값이 거의 GT 수준 정확도인 반면,
    //   어플리케이션은 EKF 자세 추정 오차로 인해 측정 좌표계가 학습 분포에서
    //   어긋난다는 점이다. 작은 R_scale 이 부정확한 측정값을 K≈1 로 받아들여
    //   양의 피드백 발산을 일으켰다. P15 에서 P13 이전 디자인으로 복원한다.
    private val DEFAULT_PARAMS = doubleArrayOf(
        sqrt(1e-3),              // [0]  sigma_na  — 가속도 노이즈 (m/s²)
        sqrt(1e-4),              // [1]  sigma_ng  — 자이로 노이즈 (rad/s)
        1e-4,                    // [2]  ita_ba    — 가속도 편향 랜덤워크
        1e-6,                    // [3]  ita_bg    — 자이로 편향 랜덤워크
        10.0 / 180.0 * Math.PI, // [4]  init_attitude_sigma (rad)
        0.1  / 180.0 * Math.PI, // [5]  init_yaw_sigma (rad)
        1.0,                     // [6]  init_vel_sigma (m/s)
        0.001,                   // [7]  init_pos_sigma (m)
        0.0001,                  // [8]  init_bg_sigma (rad/s)
        0.02,                    // [9]  init_ba_sigma (m/s²)
        9.81,                    // [10] g_norm (m/s²)
        10.0                     // [11] meascov_scale 기본값 — P15: 1.0 → 10.0 원복
    )

    // ── Context-Aware Adaptive EKF 파라미터 테이블 ────────────
    // 논문 §4.3.2 Table 기반 설계 원칙:
    //   Running    : Q↑ R↑   (격한 움직임 → 모델 불확실성 및 측정 노이즈 증가)
    //   Pocket     : Q↓ R↓   (몸에 밀착 → 규칙적 패턴, 낮은 외부 노이즈)
    //   Handbag    : Q↑ R↑   (진자 운동 패턴, 중간보다 약간 높음)
    //   Handheld   : Q 기준, R↑ (손 움직임에 의한 불규칙 노이즈 → 기준값)
    //   Slow Walk  : Q↓ R↓   (느리고 규칙적인 패턴 → 예측 신뢰도 높음)
    //   Trolley    : Q↓ R↓   (기계적으로 안정적인 움직임)
    //   Unknown    : Q↑ R↑↑  (분류 불가 → 보수적 처리)
    //
    // 인덱스: 0=handbag 1=handheld 2=pocket 3=running 4=slow_walk 5=trolley 6=unknown
    //
    // [P15] P14 의 Python 동기화 값(0.001~1.0)이 발산을 유발하여 P13 이전 값으로 원복.
    //   Python grid-search 는 GT 자세 기반 환경에서 튜닝된 결과로 측정값이 거의
    //   GT 수준 정확도임을 전제로 한다. 어플리케이션은 EKF 자세 추정 오차로 인해
    //   측정 좌표계가 학습 분포에서 어긋나므로 더 큰 R 로 노이즈를 흡수해야 한다.
    //
    // R (meascov_scale): handheld=10.0 기준 ─────────────────────
    private val R_SCALE_PER_CLASS = doubleArrayOf(
        15.0,   // 0 handbag    — 진자 운동, 중간↑
        10.0,   // 1 handheld   — 기준값 (Python batch_runner 기본값)
         5.0,   // 2 pocket     — 안정적, 측정 신뢰 ↓
        10.0,   // 3 running    — [P47-D] 50→10 (handheld 기준값 동일). 도보가 running 으로 분류되어도 측정 신뢰 회복
         5.0,   // 4 slow_walk  — 매우 규칙적, 측정 신뢰 ↓
         7.0,   // 5 trolley    — 기계적 안정, 측정 약간 신뢰 ↓
       100.0    // 6 unknown    — 분류 불가, 보수적 ↑↑↑
    )

    // Q (sigma_na): sigma_na_base = sqrt(1e-3) ≈ 0.03162 기준 ──
    // 자이로(sigma_ng)는 휴대 상태에 무관하므로 고정 유지
    private val SIGMA_NA_BASE = sqrt(1e-3)   // ≈ 0.031623 m/s²
    private val SIGMA_NG_BASE = sqrt(1e-4)   // ≈ 0.010000 rad/s (불변)

    // [P15] P14 의 통일값(모두 SIGMA_NA_BASE)을 P13 이전의 클래스별 가중치로 원복.
    private val SIGMA_NA_PER_CLASS = doubleArrayOf(
        SIGMA_NA_BASE * 1.5,   // 0 handbag    — 중간↑ (×1.5)
        SIGMA_NA_BASE * 1.0,   // 1 handheld   — 기준
        SIGMA_NA_BASE * 0.5,   // 2 pocket     — 안정적 ↓ (×0.5)
        SIGMA_NA_BASE * 3.0,   // 3 running    — 격한 움직임 ↑↑ (×3.0)
        SIGMA_NA_BASE * 0.5,   // 4 slow_walk  — 안정적 ↓ (×0.5)
        SIGMA_NA_BASE * 0.5,   // 5 trolley    — 기계적 안정 ↓ (×0.5)
        SIGMA_NA_BASE * 2.0    // 6 unknown    — 보수적 ↑ (×2.0)
    )

    // ── 공개 API ───────────────────────────────────────────────

    fun create(params: DoubleArray = DEFAULT_PARAMS) {
        nativeCreate(params)
    }

    fun initialize(tUs: Long, acc: DoubleArray) {
        nativeInitialize(tUs, acc)
    }

    fun propagate(acc: DoubleArray, gyr: DoubleArray,
                  tUs: Long, tAugmentUs: Long = -1L) {
        nativePropagate(acc, gyr, tUs, tAugmentUs)
    }

    fun update(meas: DoubleArray, cov: DoubleArray,
               tBeginUs: Long, tEndUs: Long) {
        nativeUpdate(meas, cov, tBeginUs, tEndUs)
    }

    /** 현재 위치와 표준편차를 반환. [px, py, pz, sx, sy, sz] */
    fun getPosition(): DoubleArray = nativeGetPosition()

    /** 현재 속도를 반환. [vx, vy, vz] */
    fun getVelocity(): DoubleArray = nativeGetVelocity()

    /**
     * Context-Aware Adaptive EKF: 분류 확률 벡터로 Q/R 소프트 스위칭.
     *
     * 논문 §4.3.2 수식:
     *   R_adaptive = Σ p_k · R^(k)
     *   Q_adaptive = Σ p_k · Q^(k)
     *
     * 폴백 처리:
     *   sum(clsProb) < 0.01 (분류기 없는 EKF_NEW 모델, cls_prob=zeros)
     *   → handheld 기준값(index=1) 100% 적용 — 고정 파라미터 EKF와 동등
     *
     * @param clsProb 분류 헤드의 softmax 출력 [7] (합=1 또는 zeros)
     */
    fun applySoftSwitching(clsProb: FloatArray) {
        val n   = minOf(clsProb.size, R_SCALE_PER_CLASS.size)
        val sum = clsProb.take(n).sum()

        val prob: FloatArray = if (sum < 0.01f) {
            // 분류기 없는 모델(EKF_NEW) → handheld 기준값으로 폴백
            FloatArray(7).also { it[1] = 1.0f }
        } else {
            clsProb
        }

        // 확률 가중 평균: R_adaptive, sigma_na_adaptive
        var rScale = 0.0
        var sigNA  = 0.0
        for (i in 0 until n) {
            rScale += prob[i] * R_SCALE_PER_CLASS[i]
            sigNA  += prob[i] * SIGMA_NA_PER_CLASS[i]
        }

        nativeSetMeasCovScale(rScale)
        nativeSetProcessNoise(sigNA, SIGMA_NG_BASE)
    }

    fun marginalize(cutIdx: Int) = nativeMarginalize(cutIdx)

    /**
     * [P9c] STATIC→MOVING 전환 시 EKF 내 모든 stale 클론을 제거.
     * STATIC 중 marginalize가 호출되지 않아 오래된 클론이 남아있으면
     * update() 이노베이션이 폭발하는 문제를 방지.
     */
    fun flushClones() = nativeFlushClones()

    /**
     * [P9d] Thaw Static State: STATIC→MOVING 전환 시 공분산 해동.
     * freeze_static_state() 로 1e-8 로 압축된 Σ[v,v], Σ[p,p] 를
     * 0.01 m² 로 복원 → 칼만 게인 K≈0.33 → 측정값 정상 반영.
     * flushClones() 직후 호출.
     */
    fun thawStaticState() = nativeThawStaticState()

    fun isInitialized(): Boolean = nativeIsInitialized()

    /**
     * t_begin 클론의 회전 행렬(world←body)을 DoubleArray(9)로 반환 (row-major).
     * 좌표 변환(transformWindowToWorldFrame)에서 yaw 추출용.
     */
    fun getCloneRotation(tBeginUs: Long): DoubleArray = nativeGetCloneRotation(tBeginUs)

    /**
     * 현재 EKF 자이로 편향 추정값을 DoubleArray(3)으로 반환 [bgx, bgy, bgz] (rad/s).
     * 좌표 변환(transformWindowToWorldFrame)에서 Rs_bofbi 적분 보정에 사용.
     */
    fun getGyrBias(): DoubleArray = nativeGetGyrBias()

    /**
     * ZUPT (Zero Velocity UPdate): 정지 상태에서 "속도 = 0" 측정값을 EKF에 주입.
     * @param sigmaZupt 속도 측정 노이즈 표준편차 (m/s, 기본 0.05)
     */
    fun applyZupt(sigmaZupt: Double = 0.05) {
        nativeApplyZupt(sigmaZupt)
    }

    /**
     * Position Hold: 정지 진입 시 기록한 앵커 위치를 EKF 절대 위치 측정값으로 주입.
     * ZUPT(속도=0)와 함께 사용하면 위치·속도 복합 고정 효과.
     * 가속도계 바이어스 적분에 의한 정지 중 드리프트를 근본적으로 차단.
     *
     * @param px, py, pz  정지 확정 시점의 EKF 위치 (월드 프레임, m)
     * @param sigmaPos    위치 측정 노이즈 표준편차 (m, 기본 0.01 = 1 cm)
     */
    fun applyPositionHold(px: Double, py: Double, pz: Double,
                          sigmaPos: Double = 0.01) {
        nativeApplyPositionHold(px, py, pz, sigmaPos)
    }

    /**
     * [P9] Hard State Freeze: EKF 측정 모델 완전 우회, 상태 직접 고정.
     *
     * apply_position_hold() + apply_zupt() 조합은 init_pos_sigma(0.001)에 비해
     * sigma_pos(0.01)가 10× 크면 칼만 게인 K≈0.01 → 보정이 거의 이루어지지 않음.
     * 이 함수는 칼만 게인 없이 직접 state_.p = p_anchor, state_.v = 0 으로 고정하고
     * Σ[v,v], Σ[p,p] 블록을 1e-8 로 압축한다.
     *
     * @param px, py, pz  정지 확정 시점의 EKF 위치 (월드 프레임, m)
     */
    fun freezeStaticState(px: Double, py: Double, pz: Double) {
        nativeFreezeStaticState(px, py, pz)
    }

    /**
     * Yaw 업데이트: Android TYPE_ROTATION_VECTOR 의 절대 yaw 를 EKF 에 주입.
     *
     * yaw_meas 는 EKF 월드 프레임 기준으로 계산해야 함:
     *   yaw_meas = atan2(R_rv[1,0], R_rv[0,0])  (측정)
     *            - atan2(R_rv_init[1,0], R_rv_init[0,0])  (초기화 시점 오프셋)
     *
     * 이노베이션 |δψ| > 45° 이면 자기 간섭으로 판단하여 C++ 내부에서 자동 스킵.
     *
     * @param yawMeas   EKF 프레임 yaw 측정값 (rad)
     * @param sigmaYaw  측정 노이즈 표준편차 (rad, 기본 10° = 0.1745 rad)
     */
    fun applyYawUpdate(yawMeas: Double,
                       sigmaYaw: Double = 10.0 / 180.0 * Math.PI) {
        nativeApplyYawUpdate(yawMeas, sigmaYaw)
    }

    // ── Native 선언 ────────────────────────────────────────────
    @JvmStatic external fun nativeCreate(params: DoubleArray)
    @JvmStatic external fun nativeInitialize(tUs: Long, acc: DoubleArray)
    @JvmStatic external fun nativePropagate(acc: DoubleArray, gyr: DoubleArray, tUs: Long, tAugmentUs: Long)
    @JvmStatic external fun nativeUpdate(meas: DoubleArray, cov: DoubleArray, tBeginUs: Long, tEndUs: Long)
    @JvmStatic external fun nativeGetPosition(): DoubleArray
    @JvmStatic external fun nativeGetVelocity(): DoubleArray
    @JvmStatic external fun nativeSetMeasCovScale(scale: Double)
    @JvmStatic external fun nativeSetProcessNoise(sigmaNA: Double, sigmaNG: Double)
    @JvmStatic external fun nativeMarginalize(cutIdx: Int)
    @JvmStatic external fun nativeIsInitialized(): Boolean
    @JvmStatic external fun nativeGetCloneRotation(tUs: Long): DoubleArray
    @JvmStatic external fun nativeGetGyrBias(): DoubleArray
    @JvmStatic external fun nativeApplyZupt(sigma: Double)
    @JvmStatic external fun nativeApplyYawUpdate(yawMeas: Double, sigmaYaw: Double)
    @JvmStatic external fun nativeApplyPositionHold(px: Double, py: Double, pz: Double, sigmaPos: Double)
    @JvmStatic external fun nativeFreezeStaticState(px: Double, py: Double, pz: Double)
    @JvmStatic external fun nativeFlushClones()
    @JvmStatic external fun nativeThawStaticState()
}
