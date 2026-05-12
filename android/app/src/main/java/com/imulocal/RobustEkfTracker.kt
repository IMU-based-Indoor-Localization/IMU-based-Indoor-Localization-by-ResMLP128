package com.imulocal

import android.util.Log
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * RobustEkfTracker.kt — Stage 3
 * ==================================================================
 * 단방향(One-Way) 아키텍처의 종착점: 강건한 상태 추정기.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │ 설계 원칙 — 단방향 종착점                                   │
 *  │                                                              │
 *  │  ‣ 자신의 상태(p, v, R)를 Stage 1 / Stage 2 에 절대         │
 *  │    역으로 전달하지 않는다                                    │
 *  │  ‣ Stage 1 의 rotMat 을 자세 anchor 로 사용 — EKF 자체      │
 *  │    yaw 표류 차단                                              │
 *  │  ‣ Stage 2 의 측정값이 비정상이면 무시 (Innovation Gate)    │
 *  │  ‣ 분류 확률 + RotVec 정확도로 Adaptive Covariance          │
 *  └─────────────────────────────────────────────────────────────┘
 *
 * ── 상태 ──────────────────────────────────────────────────────
 *   p[3]  : world frame 위치 (m)
 *   v[3]  : world frame 속도 (m/s)
 *   R[9]  : world←body 회전 (Stage 1 anchor, EKF 추정 안 함)
 *
 *   per-axis 공분산 (3축 독립 가정으로 단순화):
 *     varP[i]  : Var(p_i)
 *     varV[i]  : Var(v_i)
 *     covPV[i] : Cov(p_i, v_i)
 *
 * ── 입력 ──────────────────────────────────────────────────────
 *   propagate(sample) : 매 100Hz Stage 1 의 WorldSample
 *   update(output)    : Stage 2 의 InferenceOutput (윈도우 단위)
 *
 * ── 동작 ──────────────────────────────────────────────────────
 *   1. Propagate (물리 적분):
 *        a_world = sample.worldLinAcc  (이미 중력 제거됨, world frame)
 *        v += a · dt
 *        p += v · dt + 0.5 · a · dt²
 *        공분산: F·Σ·Fᵀ + Q·dt  (per-axis [p, v] 2×2)
 *        자세: R_anchor = sample.rotMat
 *
 *   2. Window anchor:
 *        첫 propagate 또는 update 직후 (p_now, ts_now) 캐시.
 *        다음 update 의 예측 변위 계산 기준점.
 *
 *   3. Update (네트워크 측정 보정):
 *        a. dispLocal → dispWorld  (windowStartYaw 회전 복원)
 *        b. predDisp = p_now − p_windowStart
 *        c. innov = dispWorld − predDisp
 *        d. 절대 게이트: ‖innov‖ > MAX_INNOV_NORM → reject
 *        e. Adaptive R = (exp(logVar)·clip) × classRScale × accInflate
 *        f. Mahalanobis 게이트: χ²(ν=3, p=0.99) = 11.345
 *           초과 시 R inflate 또는 reject
 *        g. per-axis Kalman update: K = σ_p/(σ_p + R), p += K·innov
 *        h. 공분산 갱신: Σ_p ← (1 − K)·Σ_p
 *        i. Window anchor 재설정
 *
 * ── 출력 ──────────────────────────────────────────────────────
 *   getPosition()    : DoubleArray[3]  최종 world frame 위치 (m)
 *   getVelocity()    : DoubleArray[3]  최종 world frame 속도 (m/s)
 *   getOrientation() : DoubleArray[9]  최근 자세 R_world←body (row-major)
 *   getYawRad()      : Double           atan2(R[1,0], R[0,0])
 *   getPositionStd() : DoubleArray[3]  σ_p per-axis
 *
 * ── 단방향 흐름의 위치 ─────────────────────────────────────────
 *       [Stage 1] AbsoluteSensorNode    ─┐
 *           │ WorldSample (raw 센서)     │ Stage 3 가 두 입력 모두
 *           ▼                            │ polling, 자기 상태는
 *       [Stage 2] StatelessInferenceNode │ 어디로도 전송 안 함
 *           │ InferenceOutput            │
 *           ▼                            │
 *       [Stage 3] RobustEkfTracker  ←────┘    ← 이 파일 (종착점)
 *           │ position, velocity, orientation
 *           ▼
 *       UI (MainActivity, TrackView)
 */
class RobustEkfTracker {

    // ── 동시성 보호 ──────────────────────────────────────────────
    // propJob(5ms) 와 inferJob(50ms) 가 동시에 propagate/update 를 호출하므로
    // 공유 상태(p, v, R, varP/covPV/varV, windowAnchorP)는 lock 으로 보호한다.
    // getter 들도 일관된 스냅샷 반환을 위해 동일 lock 사용.
    private val lock = Any()

    companion object {
        private const val TAG = "Stage3.EkfTracker"

        /** 1초 윈도우의 절대 이노베이션 게이트 (m). 보행 속도 상한 기반. */
        private const val MAX_INNOV_NORM = 6.0

        /** Mahalanobis χ² 임계값 (자유도 3, 신뢰도 0.99). NIST 표 기준. */
        private const val MAHAL_CHI2_THRESHOLD = 11.345

        /** Mahalanobis 실패 시 R 인플레이트 배수 (0 이면 reject). */
        private const val MAHAL_FAIL_SCALE = 10.0

        /** 측정 공분산 하한 (m²) — 학습 모델의 과도한 자신감 방어. */
        private const val MIN_MEAS_VARIANCE = 0.05

        /** 측정 공분산 상한 (m²) — 거의 무한대 측정의 수치 안정성. */
        private const val MAX_MEAS_VARIANCE = 100.0

        /** 가속도 process noise σ (m/s²). Python sigma_na = √1e-3 와 동일. */
        private const val SIGMA_A = 0.031623

        /** 분류 확률 합이 이 값 미만이면 handheld(인덱스 1) 폴백. */
        private const val CLASS_PROB_FLOOR = 0.01f

        /** RotVec 정확도 < 2 (UNRELIABLE/LOW) 일 때 R 추가 인플레이트. */
        private const val LOW_ROTACC_INFLATE = 5.0

        /** 분류별 R scale (P15 디자인 유지, 보수적). */
        private val CLASS_R_SCALE = doubleArrayOf(
            15.0,   // 0 handbag    — 진자 운동
            10.0,   // 1 handheld   — 기준
             5.0,   // 2 pocket     — 안정적
            50.0,   // 3 running    — 격한 움직임
             5.0,   // 4 slow_walk  — 규칙적
             7.0,   // 5 trolley    — 기계적 안정
           100.0    // 6 unknown    — 보수적
        )

        /** 속도 클램프 (m/s) — 실내 환경에서 발산 방어망. */
        private const val MAX_INDOOR_SPEED = 5.0

        // ── ZUPT (Zero-Velocity Update) 파라미터 (P22) ─────────────
        // 정지 감지 → velocity 강제 0 + 모델 update skip 으로 정지 시 발산 차단.
        /** 정지 감지용 worldLinAcc norm 임계 (m/s²). 슬라이딩 평균이 이 값 미만이면 정지 후보. */
        private const val STILL_LIN_ACC_THRESHOLD = 0.20
        /** 정지 감지용 worldGyr norm 임계 (rad/s). */
        private const val STILL_GYR_THRESHOLD = 0.05
        /** 정지 감지 슬라이딩 윈도우 길이 (100Hz 기준 sample 개수 — 0.5초). */
        private const val STILL_WINDOW_SIZE = 50
        /** 정지 진입 hysteresis (ms) — false-enter 방지. */
        private const val STILL_ENTER_HOLD_MS = 500L
        /** 정지 해제 hysteresis (ms) — false-unfreeze 방지. */
        private const val STILL_EXIT_HOLD_MS = 300L
        /** 정지 시 varV 축소 비율 (한 step 당). 1 step 마다 50% 축소. */
        private const val STILL_VARV_DECAY = 0.5
        /** 정지 시 varV 최저 한계 (m/s)². 너무 작으면 수치 불안정. */
        private const val STILL_VARV_FLOOR = 1e-4
    }

    // ── 상태 변수 ────────────────────────────────────────────────
    private val p = DoubleArray(3)            // 위치 (m, world)
    private val v = DoubleArray(3)            // 속도 (m/s, world)
    private val R = DoubleArray(9).also { it[0] = 1.0; it[4] = 1.0; it[8] = 1.0 }  // 자세 (Stage 1 anchor)

    // per-axis 공분산 (3축 독립 근사)
    private val varP  = DoubleArray(3) { 0.001 * 0.001 }  // 초기 위치 σ = 1 mm
    private val varV  = DoubleArray(3) { 1.0 * 1.0 }      // 초기 속도 σ = 1 m/s
    private val covPV = DoubleArray(3) { 0.0 }

    // ── 타임스탬프 ───────────────────────────────────────────────
    private var lastPropTsUs: Long = -1L
    private var initialized: Boolean = false

    // ── 윈도우 시작 시점 anchor (SC-EKF 클론의 단순화) ───────────
    private var windowAnchorP:  DoubleArray? = null
    private var windowAnchorTs: Long = -1L

    // ── 진단 통계 ────────────────────────────────────────────────
    private var totalUpdates  = 0L
    private var rejectedInnov = 0L
    private var rejectedMahal = 0L
    @Volatile private var lastInnovNorm = 0.0
    @Volatile private var lastMahal     = 0.0

    // ── ZUPT (Zero-Velocity Update) 상태 (P22) ─────────────────────
    // 슬라이딩 윈도우 norm 버퍼 (모두 0 초기화 — 쓰레기값 방지)
    private val linAccNormBuf = DoubleArray(STILL_WINDOW_SIZE)
    private val gyrNormBuf    = DoubleArray(STILL_WINDOW_SIZE)
    private var normBufIdx    = 0
    private var normBufFilled = false

    // 정지 후보 ↔ 확정 전이용 hysteresis 카운터
    @Volatile private var isStationary: Boolean = false
    private var candidateChangeStartMs: Long = 0L

    // 진단 통계
    private var stationaryUpdatesSkipped: Long = 0L
    private var zuptApplications: Long = 0L
    @Volatile private var lastLinAccMean: Double = 0.0
    @Volatile private var lastGyrMean:    Double = 0.0

    /**
     * 트래커 초기화 / 재시작. p, v 를 0 으로, R 을 identity 로 되돌린다.
     * Stage 1 의 첫 샘플 직후 호출하는 것이 일반적이다.
     */
    fun reset() = synchronized(lock) {
        for (i in 0..2) { p[i] = 0.0; v[i] = 0.0; varP[i] = 0.001 * 0.001; varV[i] = 1.0; covPV[i] = 0.0 }
        for (i in 0..8) R[i] = if (i == 0 || i == 4 || i == 8) 1.0 else 0.0
        lastPropTsUs = -1L
        initialized = false
        windowAnchorP = null
        windowAnchorTs = -1L
        totalUpdates = 0L
        rejectedInnov = 0L
        rejectedMahal = 0L

        // ZUPT 상태 초기화 (P22)
        for (k in 0 until STILL_WINDOW_SIZE) {
            linAccNormBuf[k] = 0.0
            gyrNormBuf[k]    = 0.0
        }
        normBufIdx     = 0
        normBufFilled  = false
        isStationary   = false
        candidateChangeStartMs   = 0L
        stationaryUpdatesSkipped = 0L
        zuptApplications         = 0L
        lastLinAccMean = 0.0
        lastGyrMean    = 0.0

        Log.i(TAG, "RobustEkfTracker reset")
    }

    /**
     * Stage 1 의 매 WorldSample 도착 시 호출 — 물리 적분 + 자세 anchor.
     *
     * 첫 호출에서 lastPropTsUs 만 캐시하고 다음 호출부터 적분 시작.
     */
    fun propagate(sample: AbsoluteSensorNode.WorldSample) = synchronized(lock) {
        // 자세는 항상 Stage 1 의 rotMat 으로 anchor (단방향 흐름의 핵심)
        for (i in 0..8) R[i] = sample.rotMat[i].toDouble()

        if (lastPropTsUs < 0L) {
            lastPropTsUs = sample.ts_us
            // 첫 propagate 시점에 윈도우 anchor 도 함께 설정
            if (windowAnchorP == null) {
                windowAnchorP = p.copyOf()
                windowAnchorTs = sample.ts_us
            }
            initialized = true
            return
        }

        val dt = (sample.ts_us - lastPropTsUs) * 1e-6
        lastPropTsUs = sample.ts_us

        // 비정상 dt 방어 (음수 / 너무 큼)
        if (dt <= 0.0 || dt > 0.1) {
            Log.v(TAG, "비정상 dt=$dt, propagate 건너뜀")
            return
        }

        // ── 적분: a 는 world frame, 중력 이미 제거됨 ────────────────
        val a = sample.worldLinAcc
        val g = sample.worldGyr

        for (i in 0..2) {
            val ai = a[i].toDouble()
            // 상태 갱신
            p[i] += v[i] * dt + 0.5 * ai * dt * dt
            v[i] += ai * dt

            // 공분산 전파 (per-axis [p, v] 2×2):
            //   F = [[1, dt], [0, 1]]
            //   Σ_new = F Σ Fᵀ + G·σ_a²·dt·Gᵀ
            //   G = [0.5·dt², dt]ᵀ  →  G·G·dt 는 process noise 기여
            // 결과:
            //   var_p ← var_p + 2·dt·cov_pv + dt²·var_v + (σ_a·dt²/2)²·dt? 단순화
            //   cov_pv ← cov_pv + dt·var_v + σ_a²·dt²/2·dt
            //   var_v ← var_v + σ_a²·dt
            val sigmaA2 = SIGMA_A * SIGMA_A
            val newVarP  = varP[i] + 2 * dt * covPV[i] + dt * dt * varV[i] + sigmaA2 * (dt * dt * dt) / 3.0
            val newCovPV = covPV[i] + dt * varV[i] + sigmaA2 * (dt * dt) / 2.0
            val newVarV  = varV[i] + sigmaA2 * dt
            varP[i]  = newVarP
            covPV[i] = newCovPV
            varV[i]  = newVarV
        }

        // ── ★ ZUPT (P22): 정지 감지 + velocity 강제 0 ──────────────
        updateStationaryState(a, g)
        if (isStationary) {
            // velocity 를 0 으로 강제, varV 빠르게 축소 (정지 확신)
            for (i in 0..2) {
                v[i] = 0.0
                varV[i] = (varV[i] * STILL_VARV_DECAY).coerceAtLeast(STILL_VARV_FLOOR)
                covPV[i] = 0.0
            }
            zuptApplications++
        }

        // ── 속도 클램프 (실내 발산 안전망) ──────────────────────────
        val speed = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if (speed > MAX_INDOOR_SPEED) {
            val s = MAX_INDOOR_SPEED / speed
            v[0] *= s; v[1] *= s; v[2] *= s
            Log.w(TAG, "속도 클램프: ${"%.2f".format(speed)} → $MAX_INDOOR_SPEED m/s")
        }
    }

    /**
     * 정지 상태 갱신 (P22 — ZUPT 의 핵심).
     * 슬라이딩 윈도우의 worldLinAcc / worldGyr 평균 norm 이 임계 이하인 상태가
     * STILL_ENTER_HOLD_MS 이상 유지되면 isStationary=true 로 진입,
     * 임계 이상인 상태가 STILL_EXIT_HOLD_MS 이상 유지되면 해제.
     */
    private fun updateStationaryState(a: FloatArray, gyr: FloatArray) {
        val aNorm = sqrt(
            (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).toDouble()
        )
        val gNorm = sqrt(
            (gyr[0] * gyr[0] + gyr[1] * gyr[1] + gyr[2] * gyr[2]).toDouble()
        )
        linAccNormBuf[normBufIdx] = aNorm
        gyrNormBuf[normBufIdx]    = gNorm
        normBufIdx = (normBufIdx + 1) % STILL_WINDOW_SIZE
        if (normBufIdx == 0) normBufFilled = true

        // 버퍼가 절반도 안 찼으면 판단 보류 (초기 transient 보호)
        val validN = if (normBufFilled) STILL_WINDOW_SIZE else normBufIdx
        if (validN < STILL_WINDOW_SIZE / 2) return

        var aSum = 0.0
        var gSum = 0.0
        for (k in 0 until validN) {
            aSum += linAccNormBuf[k]
            gSum += gyrNormBuf[k]
        }
        val aMean = aSum / validN
        val gMean = gSum / validN
        lastLinAccMean = aMean
        lastGyrMean    = gMean

        val candidateStill = (aMean < STILL_LIN_ACC_THRESHOLD && gMean < STILL_GYR_THRESHOLD)
        val nowMs = System.currentTimeMillis()

        if (candidateStill == isStationary) {
            // 같은 상태 유지 — hysteresis 카운터 리셋
            candidateChangeStartMs = nowMs
        } else {
            // 상태 전이 후보 — required hold 이상 유지되면 확정
            val held = nowMs - candidateChangeStartMs
            val requiredHold = if (candidateStill) STILL_ENTER_HOLD_MS else STILL_EXIT_HOLD_MS
            if (held >= requiredHold) {
                isStationary = candidateStill
                candidateChangeStartMs = nowMs
                Log.i(
                    TAG,
                    "ZUPT 상태 전이: stationary=$isStationary " +
                    "(aMean=${"%.3f".format(aMean)} m/s², gMean=${"%.4f".format(gMean)} rad/s)"
                )
            }
        }
    }

    /**
     * Stage 2 의 InferenceOutput 으로 위치를 보정한다.
     *
     * @return true = 정상 update / false = 게이트에 의해 reject
     */
    fun update(output: StatelessInferenceNode.InferenceOutput): Boolean = synchronized(lock) {
        if (!initialized) {
            Log.v(TAG, "초기화 전, update 건너뜀")
            return@synchronized false
        }
        val anchor = windowAnchorP
        if (anchor == null) {
            Log.v(TAG, "windowAnchor 없음, update 건너뜀")
            return@synchronized false
        }

        // ── ★ P22 ZUPT: 정지 상태에서는 모델 update 누적 차단 ──────
        // 정지 구간에 trolley/handheld 등 잘못된 분류로 disp 가 누적되어
        // 30초 동안 수십 m 발산하던 문제를 차단. anchor 만 갱신해 위치 고정.
        if (isStationary) {
            stationaryUpdatesSkipped++
            advanceWindowAnchor(output.tsEndUs)
            Log.v(TAG, "ZUPT: stationary update skip (cls=${output.className})")
            return@synchronized false
        }

        // ── 1. dispLocal → dispWorld (yaw0 으로 회전 복원) ─────────
        val yaw0 = output.windowStartYaw
        val cosY = cos(yaw0)
        val sinY = sin(yaw0)
        val dispW = doubleArrayOf(
             cosY * output.dispLocal[0] - sinY * output.dispLocal[1],
             sinY * output.dispLocal[0] + cosY * output.dispLocal[1],
             output.dispLocal[2].toDouble()
        )

        // ── 2. 예측 변위 ─────────────────────────────────────────────
        val predDisp = doubleArrayOf(p[0] - anchor[0], p[1] - anchor[1], p[2] - anchor[2])

        // ── 3. Innovation ───────────────────────────────────────────
        val innov = doubleArrayOf(
            dispW[0] - predDisp[0],
            dispW[1] - predDisp[1],
            dispW[2] - predDisp[2]
        )
        val innovNorm = sqrt(innov[0]*innov[0] + innov[1]*innov[1] + innov[2]*innov[2])
        lastInnovNorm = innovNorm

        // ── 4. 절대 이노베이션 게이트 ───────────────────────────────
        if (innovNorm > MAX_INNOV_NORM) {
            rejectedInnov++
            Log.w(TAG, "Innovation gate reject: ‖innov‖=${"%.2f".format(innovNorm)}m " +
                  "> $MAX_INNOV_NORM m")
            advanceWindowAnchor(output.tsEndUs)
            return@synchronized false
        }

        // ── 5. Adaptive R 계산 ──────────────────────────────────────
        val r = computeAdaptiveR(output)

        // ── 6. Mahalanobis 게이트 (per-axis 근사) ─────────────────────
        // NSE = Σᵢ innov_i² / (var_p_i + r_i)
        var nse = 0.0
        for (i in 0..2) {
            val s = varP[i] + r[i]
            if (s < 1e-12) continue
            nse += innov[i] * innov[i] / s
        }
        lastMahal = nse

        if (nse > MAHAL_CHI2_THRESHOLD) {
            if (MAHAL_FAIL_SCALE <= 0.0) {
                rejectedMahal++
                Log.w(TAG, "Mahalanobis reject: NSE=${"%.2f".format(nse)} > $MAHAL_CHI2_THRESHOLD")
                advanceWindowAnchor(output.tsEndUs)
                return@synchronized false
            } else {
                // R inflate 후 진행
                for (i in 0..2) r[i] *= MAHAL_FAIL_SCALE
                Log.w(TAG, "Mahalanobis inflate: NSE=${"%.2f".format(nse)} → R ×$MAHAL_FAIL_SCALE")
            }
        }

        // ── 7. per-axis Kalman update ───────────────────────────────
        // 측정 모델: meas = p_now (windowAnchor 와의 차를 측정으로 본 뒤
        //            target = anchor + dispWorld 로 절대 위치 측정 환산)
        // → H = I_3 (per-axis 단순)
        for (i in 0..2) {
            val s = varP[i] + r[i]
            if (s < 1e-12) continue
            val k = varP[i] / s          // 위치 Kalman gain
            p[i] += k * innov[i]         // 상태 보정
            varP[i] = (1.0 - k) * varP[i]  // 공분산 축소
            // p-v cross-correlation 도 같은 비율로 축소 (근사)
            covPV[i] *= (1.0 - k)
            // v 자체는 직접 측정되지 않지만 cross-correlation 으로 약하게 보정
            //   K_v = covPV / s  (per-axis)
            val kv = covPV[i] / s
            v[i] += kv * innov[i]
        }

        totalUpdates++

        // ── 8. 윈도우 anchor 재설정 (다음 윈도우의 시작점) ──────────
        advanceWindowAnchor(output.tsEndUs)

        Log.v(TAG, "Update OK | innov=${"%.3f".format(innovNorm)}m " +
              "nse=${"%.2f".format(nse)} cls=${output.className} " +
              "rotAcc=${output.rotAccuracyStart}")
        return@synchronized true
    }

    /**
     * 다음 윈도우의 anchor 를 현재 위치/시각으로 갱신.
     */
    private fun advanceWindowAnchor(tsEndUs: Long) {
        windowAnchorP  = p.copyOf()
        windowAnchorTs = tsEndUs
    }

    /**
     * Adaptive Measurement Covariance:
     *   R_diag[i] = clip(exp(logVar_i), MIN, MAX) · class_R_scale · acc_inflate
     */
    private fun computeAdaptiveR(output: StatelessInferenceNode.InferenceOutput): DoubleArray {
        // dispLogVar → variance (clip)
        val sig2 = DoubleArray(3) { i ->
            val lv = output.dispLogVar[i].toDouble().coerceAtLeast(-4.0)
            exp(lv).coerceIn(MIN_MEAS_VARIANCE, MAX_MEAS_VARIANCE)
        }

        // 분류별 R scale soft switching
        val probSum = output.classProb.sum()
        val rScale = if (probSum < CLASS_PROB_FLOOR) {
            CLASS_R_SCALE[1]   // handheld fallback (분류기 없거나 신뢰도 낮음)
        } else {
            var s = 0.0
            for (i in 0..6) s += output.classProb[i] * CLASS_R_SCALE[i]
            s
        }

        // RotVec 정확도 게이팅: < 2 (UNRELIABLE/LOW) 이면 측정 신뢰도 추가 하향
        val accInflate = if (output.rotAccuracyStart < 2) LOW_ROTACC_INFLATE else 1.0

        return DoubleArray(3) { i -> sig2[i] * rScale * accInflate }
    }

    // ── 출력 API (lock 으로 일관된 스냅샷 반환) ──────────────────
    fun getPosition():    DoubleArray = synchronized(lock) { p.copyOf() }
    fun getVelocity():    DoubleArray = synchronized(lock) { v.copyOf() }
    fun getOrientation(): DoubleArray = synchronized(lock) { R.copyOf() }

    /** ZYX yaw (rad) — Stage 1 의 rotMat 기준. */
    fun getYawRad(): Double = synchronized(lock) { atan2(R[3], R[0]) }

    /** per-axis 위치 표준편차 [σx, σy, σz] (m). */
    fun getPositionStd(): DoubleArray = synchronized(lock) {
        DoubleArray(3) { i -> sqrt(varP[i]) }
    }

    /** 진단용 통계. */
    fun getDiagnostics(): String = synchronized(lock) {
        "updates=$totalUpdates rejInnov=$rejectedInnov rejMahal=$rejectedMahal " +
        "stillSkip=$stationaryUpdatesSkipped zupt=$zuptApplications still=$isStationary " +
        "aMean=${"%.3f".format(lastLinAccMean)} gMean=${"%.4f".format(lastGyrMean)} " +
        "lastInnov=${"%.3f".format(lastInnovNorm)}m lastNSE=${"%.2f".format(lastMahal)}"
    }

    fun isInitialized(): Boolean = synchronized(lock) { initialized }

    /** 현재 ZUPT 가 활성(정지 감지) 상태인가. UI 표시 / 로깅용 (P22). */
    fun isStationary(): Boolean = synchronized(lock) { isStationary }
}
