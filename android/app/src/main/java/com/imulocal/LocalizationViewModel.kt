package com.imulocal

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * LocalizationViewModel.kt
 * ========================
 * 앱의 핵심 측위 파이프라인.
 *
 *  1. ImuCollector  → 100Hz IMU 수집
 *  2. InferenceEngine → 20Hz 네트워크 추론 (변위 + 분류)
 *  3. EkfBridge (JNI) → SC-EKF 상태 갱신
 *  4. UI 스레드로 위치/경로 방출
 *
 * ── 버그 수정 사항 ─────────────────────────────────────────
 *  ① 타임스탬프: System.currentTimeMillis() 완전 제거.
 *    모든 클론/갱신 타임스탬프는 SensorEvent.timestamp(부팅 경과 ns→μs) 사용.
 *
 *  ② 샘플 스킵: windowReady.collect + getLatestSample() 방식 → 폐기.
 *    propJob 은 drainPropagateQueue() 로 5ms 마다 모든 100Hz 샘플 처리.
 *
 *  ③ 추론 타이밍: delay(50ms) 고정 → 경과 시간 보정 방식으로 교체.
 *    루프 시작시각 기록 후 runInferStep() 완료 뒤 나머지 시간만 대기.
 *
 *  ④ 클론 쌍: SC-EKF update() 는 t_begin, t_end 양쪽 모두
 *    si_timestamps_us 에 존재해야 함 → 매 추론마다 t_end 클론 삽입,
 *    cloneChannel 에서 ~1초 이전 클론을 t_begin 으로 탐색.
 *
 * ── P3/P4 수정 ─────────────────────────────────────────────
 *  P3: cloneHistory(ArrayDeque + synchronized) → Channel<Long>.
 *      propJob 이 Channel 에 send, inferJob 이 로컬 history 에 drain.
 *      두 코루틴이 공유 뮤텍스 없이 통신 → 락 경쟁 완전 제거.
 *  P4: CLONE_SETTLE_MS 20ms → 30ms (기기 부하 시 타이밍 여유 증가).
 *
 * ── P1 수정 ────────────────────────────────────────────────
 *  transformWindowToWorldFrame() 에서 자이로 적분 시
 *  EKF bg 편향을 차감하여 Python scekf.py 동작과 일치.
 *
 * ── P5 수정 (정지 노이즈 / 이동→정지 프리즈) ─────────────────
 *  원인: pendingCloneTs.set() 이 정지 판정보다 먼저 실행되어
 *        정지 중에도 클론이 C++ EKF 에 20Hz 로 삽입되지만
 *        marginalize() 는 호출되지 않음 → 클론 무한 누적.
 *        누적된 클론이 propagate() 의 O(N²) 행렬 연산을 폭증시켜
 *        Default 디스패처 스레드 포화 → 앱 프리즈 유발.
 *  수정:
 *    ① runInferStep 에서 정지 판정을 pendingCloneTs.set() 보다 앞으로 이동.
 *    ② STATIC 브랜치: pendingCloneTs = -1L 로 신규 클론 차단.
 *    ③ STATIC 브랜치: Channel 잔여 클론 drain 후 C++ 클론 전부 주변화.
 *       → 정지 상태에서 EKF 상태 벡터를 최소 크기로 유지.
 *
 * ── P6 수정 (코루틴 yield 보장) ──────────────────────────────
 *  원인: inferJob 루프에서 elapsedMs ≥ INFER_INTERVAL_MS 이면
 *        delay() 를 호출하지 않아 Default 스레드를 양보하지 않음.
 *  수정: delay(remaining.coerceAtLeast(1L)) 로 항상 최소 1ms yield.
 */
class LocalizationViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "LocalizationVM"

        /** 추론 루프 목표 주기: 20Hz */
        private const val INFER_INTERVAL_MS  = 50L

        /** 클론 삽입 대기 시간 (propJob 처리 여유 확보).
         *  P4 수정: 20ms → 30ms (기기 부하 시 미삽입 빈도 감소) */
        private const val CLONE_SETTLE_MS    = 30L

        /** propJob 드레인 폴링 주기 */
        private const val PROP_POLL_MS       = 5L

        /**
         * 네트워크 윈도우 지속 시간 (μs).
         * 100샘플 × 10,000μs/샘플 = 1,000,000μs 이지만
         * getWindow() 는 first.ts 를 반환하므로
         * last.ts − first.ts ≈ 99 × 10,000 = 990,000μs.
         */
        private const val WINDOW_DURATION_US = 990_000L

        /** cloneChannel 버퍼 크기 (약 2초분 × 20Hz = 40개) */
        private const val MAX_CLONE_HISTORY  = 40

        /** t_begin 탐색 허용 오차 ±200ms */
        private const val CLONE_MATCH_TOL_US = 200_000L

        /**
         * 정적 상태 판정 임계값.
         * body frame 자이로 3축 RMS 가 이 값 미만이면 정지로 판단 → EKF 업데이트 건너뜀.
         * 정지 MEMS 자이로 노이즈 ≈ 0.003-0.01 rad/s RMS.
         * 보행 자이로 ≈ 0.1-0.5 rad/s RMS.
         * [P9 조정] 0.03 → 0.08 rad/s:
         * 실기기/에뮬레이터 MEMS 자이로 정지 노이즈가 0.03을 초과하는 경우가 많음.
         * STATIC 앵커 로그가 전혀 찍히지 않을 경우(정지 미감지) 이 값을 올린다.
         * 0.08 rad/s ≈ 4.6°/s — 느린 걷기(보통 0.15+ rad/s)와 충분히 구분됨.
         */
        private const val STATIC_GYR_RMS_THRESHOLD = 0.08f  // rad/s

        /**
         * 1-초 윈도우당 최대 허용 변위 (m).
         * [P9d] 실내 일반 보행 최대속도 ≈ 2 m/s → 1초 ≈ 2m.
         * 2m 초과 시 네트워크 이상 출력 또는 좌표 변환 오류로 판단 → 건너뜀.
         * (기존 6.0m 는 너무 관대 — 잘못된 측정값이 EKF 를 발산시킴)
         */
        private const val MAX_DISP_PER_WINDOW_M = 2.0

        /**
         * [P9d] 측정 공분산 최솟값 (바닥 설정).
         * 네트워크가 과도하게 자신감 있는 예측을 할 때 EKF 가 맹목적으로 따라가는 것을 방지.
         * exp(-4) ≈ 0.018 m² → 0.1 m² (std = 0.316 m) 로 하향.
         * K = 0.01/(0.01+0.1) ≈ 0.09 → 과도한 보정 억제.
         */
        private const val MIN_MEAS_COV = 0.05  // m² (std ≈ 0.224 m)

        // ── [아이디어 3] Hysteresis 상태 머신 파라미터 ────────────────
        /**
         * MOVING → STATIC 전환에 필요한 연속 정지 프레임 수.
         * 5프레임 × 50ms = 250ms 연속 정지해야 STATIC 으로 확정.
         * 값을 높이면 이동→정지 반응이 느려지지만 오판정 감소.
         */
        private const val STATIC_CONFIRM_FRAMES = 5

        /**
         * STATIC → MOVING 전환에 필요한 연속 이동 프레임 수.
         * 3프레임 × 50ms = 150ms 연속 이동해야 MOVING 으로 확정.
         * 값을 낮추면 보행 시작 반응이 빨라지지만 오판정 가능성 증가.
         */
        private const val MOVING_CONFIRM_FRAMES = 3

        // ── [아이디어 5] EKF 속도 게이트 ─────────────────────────────
        /**
         * 모델 only 궤적 누적을 허용하는 최소 EKF 속도 (m/s).
         * EKF 갱신 후 속도가 이 값 미만이면 dead-reckoning 을 차단.
         * 5 cm/s: MEMS 정지 드리프트(≈1-3 cm/s) 의 약 2-3배 → 안전 여유.
         */
        private const val MODEL_VELOCITY_GATE = 0.05  // m/s

        // ── Yaw drift 보정 파라미터 ──────────────────────────────────
        /**
         * TYPE_ROTATION_VECTOR yaw 측정 노이즈 표준편차 (rad).
         * 실내 자기 간섭 환경을 고려하여 10° (0.1745 rad) 로 보수적으로 설정.
         * 지자기 노이즈가 적은 환경이면 5° 로 줄여도 무방.
         */
        private const val YAW_SIGMA_RAD = 10.0 / 180.0 * Math.PI

        // ── [P10] 최초 이동 발산 방지 파라미터 ─────────────────────────
        /**
         * 추론 윈도우의 동적 구간 최소 비율 (0.0~1.0).
         * WINDOW_SIZE=100 샘플(1초) 중 이 비율 이상이 gyr > threshold 여야 추론 실행.
         *
         * 최초 이동 시 윈도우는 [정지 85%][이동 15%] 혼합 → 네트워크 출력 오염.
         * 0.5(50%) 요구 → 이동 시작 후 ~0.5초 후부터 추론 허용.
         * cloneHistory 요구(~1초)와 맞물려 혼합 윈도우 업데이트를 실질적으로 차단.
         */
        private const val MIN_DYNAMIC_FRACTION = 0.5f

        /**
         * EKF 업데이트 후 속도 크기 상한 (m/s).
         * 실내 최고 보행속도 ≈ 3 m/s. 초과 시 EKF 발산으로 판단 → ZUPT 강제 적용.
         * 발산 감지 후 안전망(reactive divergence recovery).
         */
        private const val MAX_POST_UPDATE_SPEED = 3.0  // m/s
    }

    // ── 의존 컴포넌트 ────────────────────────────────────────────
    val imuCollector = ImuCollector(application)
    val inferEngine  = InferenceEngine(application)

    // ── UI 상태 ──────────────────────────────────────────────────
    data class LocalizationState(
        val isRunning:         Boolean = false,
        val position:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val posStd:            Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val velocity:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val carryMode:         String  = "unknown",
        val carryProb:         Float   = 0f,
        val trackPoints:       List<Pair<Double, Double>> = emptyList(),  // EKF 궤적
        val modelTrackPoints:  List<Pair<Double, Double>> = emptyList(),  // 모델 only 궤적
        val inferLatency:      Long    = 0L
    )

    private val _state = MutableStateFlow(LocalizationState())
    val state: StateFlow<LocalizationState> = _state

    // ── 내부 상태 ────────────────────────────────────────────────
    private var inferJob: Job? = null
    private var propJob:  Job? = null
    private val trackPoints      = mutableListOf<Pair<Double, Double>>()
    private val modelTrackPoints = mutableListOf<Pair<Double, Double>>()

    /** 모델 단독 누적 위치 (EKF 없이 displacement 합산) */
    private var modelPosX = 0.0
    private var modelPosY = 0.0

    // ── [아이디어 3] Hysteresis 상태 머신 ────────────────────────────
    private enum class MotionState { STATIC, MOVING }

    /** 확정된 현재 운동 상태. 초기값 STATIC (시작 시 정지 가정). */
    private var motionState = MotionState.STATIC

    /** MOVING 상태에서 연속으로 gyrRms < threshold 인 프레임 수 (→ STATIC 전환 카운터). */
    private var staticCandidateCount  = 0

    /** STATIC 상태에서 연속으로 gyrRms ≥ threshold 인 프레임 수 (→ MOVING 전환 카운터). */
    private var movingCandidateCount  = 0

    /**
     * [P9] Hard State Freeze 앵커 위치.
     * STATIC 첫 프레임에 EKF 위치를 기록, 이후 STATIC 기간 내내
     * freezeStaticState(앵커)를 호출하여 EKF 상태를 직접 고정.
     * MOVING 프레임 또는 reset() 시 null 로 초기화.
     */
    private var staticAnchorPos: DoubleArray? = null

    // ── [아이디어 5] 직전 EKF 갱신 후 속도 크기 (m/s) ───────────────
    /** EKF update() 직후 조회한 속도 노름 — 다음 스텝 모델 only 게이팅에 사용. */
    private var prevEkfVelNorm = 0.0

    // ── 원점 복귀 자동 경로 클리어 ────────────────────────────────
    /** 출발 후 이 거리(m) 이상 멀어진 적이 있어야 복귀 감지를 활성화. */
    private val AWAY_THRESHOLD_M   = 1.5
    /** 원점으로부터 이 거리(m) 이내로 들어오면 경로 클리어. */
    private val RETURN_PROXIMITY_M = 0.5
    /** 출발 후 AWAY_THRESHOLD_M 이상 멀어진 적이 있는지 여부. */
    private var wasAwayFromOrigin  = false

    // ── Yaw drift 보정 ────────────────────────────────────────────
    /**
     * EKF 초기화 시점의 TYPE_ROTATION_VECTOR yaw (rad).
     * Double.NaN = 미초기화 또는 rotVecSensor 없음.
     *
     * 보정 공식:  yaw_meas = yaw_rv_current − yaw_rv_at_init
     *   → EKF 월드 프레임 기준 상대 yaw (초기화 시점 기준 0)
     */
    private var yawRvAtInit = Double.NaN

    /**
     * inferJob → propJob 단방향 신호:
     *  inferJob 이 원하는 끝 클론 센서 타임스탬프를 기록.
     *  propJob 이 해당 ts 이상의 샘플 처리 시 클론 삽입 후 -1 로 리셋.
     */
    private val pendingCloneTs      = AtomicLong(-1L)

    /**
     * propJob → inferJob 단방향 신호:
     *  가장 최근에 삽입된 클론의 실제 센서 ts.
     */
    private val lastInsertedCloneTs = AtomicLong(-1L)

    /**
     * P3 수정: propJob → inferJob 클론 히스토리 채널.
     *
     * propJob 이 클론 삽입 시 ts 를 Channel 에 trySend.
     * inferJob 이 runInferStep() 시작 시 Channel 을 drain 하여
     * 자신만의 localCloneHistory(ArrayDeque) 에 적재.
     * → synchronized 블록 없이 단방향 메시지 패싱으로 안전하게 통신.
     *
     * DROP_OLDEST: 버퍼가 가득 찰 경우 오래된 클론 ts 를 자동 제거
     * (가장 오래된 것은 이미 주변화되어 필요 없음).
     */
    private val cloneChannel = Channel<Long>(
        capacity       = MAX_CLONE_HISTORY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // ── 측위 시작 ────────────────────────────────────────────────
    fun start() {
        if (_state.value.isRunning) return
        Log.i(TAG, "측위 시작")

        viewModelScope.launch(Dispatchers.IO) {
            // 모델 로드
            if (!inferEngine.isLoaded()) {
                try { inferEngine.load() }
                catch (e: Exception) { Log.e(TAG, "모델 로드 실패: ${e.message}") }
            }

            // EKF 생성
            EkfBridge.create()

            // IMU 수집 시작
            imuCollector.start()

            // ── ① EKF 전파 루프 (5ms 폴링, 모든 100Hz 샘플 처리) ──
            propJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val samples = imuCollector.drainPropagateQueue()
                    for (sample in samples) {
                        val tsUs = sample.ts_us
                        val acc  = doubleArrayOf(
                            sample.acc[0].toDouble(),
                            sample.acc[1].toDouble(),
                            sample.acc[2].toDouble()
                        )
                        val gyr  = doubleArrayOf(
                            sample.gyr[0].toDouble(),
                            sample.gyr[1].toDouble(),
                            sample.gyr[2].toDouble()
                        )

                        if (!EkfBridge.isInitialized()) {
                            EkfBridge.initialize(tsUs, acc)
                            // EKF 초기화 직후 Rotation Vector yaw 오프셋 기록
                            val rvYaw = imuCollector.getLatestYawRad()
                            if (!rvYaw.isNaN()) {
                                yawRvAtInit = rvYaw.toDouble()
                                Log.i(TAG, "Yaw 오프셋 초기화: ${"%.1f".format(Math.toDegrees(yawRvAtInit))}°")
                            }
                            continue
                        }

                        // 클론 삽입 여부: inferJob 이 예약한 ts 이상이면 삽입
                        val pending = pendingCloneTs.get()
                        val augTs: Long = if (pending >= 0L && tsUs >= pending) {
                            // compareAndSet 으로 중복 삽입 방지
                            if (pendingCloneTs.compareAndSet(pending, -1L)) tsUs else -1L
                        } else {
                            -1L
                        }

                        EkfBridge.propagate(acc, gyr, tsUs, augTs)

                        if (augTs >= 0L) {
                            lastInsertedCloneTs.set(augTs)
                            // P3 수정: synchronized 대신 Channel.trySend 로 비락킹 전달
                            cloneChannel.trySend(augTs)
                            Log.v(TAG, "클론 삽입 ts=$augTs → Channel 전달")
                        }
                    }
                    delay(PROP_POLL_MS)
                }
            }

            // ── ② 추론 루프 (경과 시간 보정 20Hz) ─────────────────
            inferJob = viewModelScope.launch(Dispatchers.Default) {
                // P3: inferJob 전용 클론 히스토리 — 이 코루틴만 읽기/쓰기
                val localCloneHistory = ArrayDeque<Long>()

                while (isActive) {
                    val loopStart = System.nanoTime()
                    runInferStep(localCloneHistory)
                    val elapsedMs = (System.nanoTime() - loopStart) / 1_000_000L
                    val remaining = INFER_INTERVAL_MS - elapsedMs
                    // P6 수정: 항상 최소 1ms yield → 스레드 기아 방지
                    delay(remaining.coerceAtLeast(1L))
                }
            }

            _state.value = _state.value.copy(isRunning = true)
        }
    }

    // ── 측위 정지 ────────────────────────────────────────────────
    fun stop() {
        inferJob?.cancel(); inferJob = null
        propJob?.cancel();  propJob  = null
        imuCollector.stop()
        pendingCloneTs.set(-1L)
        lastInsertedCloneTs.set(-1L)
        // Channel 비우기 (재시작 시 오래된 ts 로 오동작 방지)
        while (cloneChannel.tryReceive().isSuccess) { /* drain */ }
        _state.value = _state.value.copy(isRunning = false)
        Log.i(TAG, "측위 정지")
    }

    // ── 초기화 ──────────────────────────────────────────────────
    fun reset() {
        stop()
        trackPoints.clear()
        modelTrackPoints.clear()
        modelPosX = 0.0
        modelPosY = 0.0
        // [아이디어 3] 상태 머신 초기화
        motionState          = MotionState.STATIC
        staticCandidateCount = 0
        movingCandidateCount = 0
        // [P8] Position Hold 앵커 초기화
        staticAnchorPos      = null
        // [아이디어 5] 속도 게이트 초기화
        prevEkfVelNorm       = 0.0
        // 원점 복귀 감지 초기화
        wasAwayFromOrigin    = false
        // Yaw drift 보정 초기화
        yawRvAtInit          = Double.NaN
        _state.value = LocalizationState()
    }

    // ── 추론 + EKF 갱신 스텝 ────────────────────────────────────
    /**
     * suspend 함수: delay(CLONE_SETTLE_MS) 포함.
     * inferJob 코루틴 내에서만 호출.
     *
     * @param localCloneHistory inferJob 전용 클론 히스토리 (락 불필요).
     */
    private suspend fun runInferStep(localCloneHistory: ArrayDeque<Long>) {
        if (!EkfBridge.isInitialized()) return
        if (!inferEngine.isLoaded())    return

        // ① 추론 윈도우 확보 (최소 100 샘플 필요)
        val (window, _) = imuCollector.getWindow() ?: return

        // ② [P5→P7] Hysteresis 정지 판정 (클론 예약보다 먼저 수행).
        //
        //  P5: 정지 판정 후 클론 차단 + 전체 주변화 → 프리즈 해결, 단 위치 드리프트.
        //  P7: 정지 판정 후에도 클론 정상 삽입 → zero-disp EKF update + 주변화 + ZUPT.
        //      위치 제약(std≈1cm) + 속도 제약(ZUPT) 복합 적용 → 드리프트 해소.
        //
        //  STATIC → MOVING : gyrRms ≥ threshold 가 MOVING_CONFIRM_FRAMES 연속
        //  MOVING → STATIC : gyrRms <  threshold 가 STATIC_CONFIRM_FRAMES  연속
        val gyrRms        = computeGyrRms(window)
        val isStaticFrame = gyrRms < STATIC_GYR_RMS_THRESHOLD
        // [진단] 정지 감지 여부 실시간 확인 — 필요 시 주석 처리
        Log.v(TAG, "gyrRms=${"%.4f".format(gyrRms)} thr=$STATIC_GYR_RMS_THRESHOLD static=$isStaticFrame state=$motionState")

        val currentlyStatic: Boolean = when (motionState) {
            MotionState.STATIC -> {
                if (isStaticFrame) {
                    // 정지 유지 — 이동 후보 카운터 리셋
                    movingCandidateCount = 0
                    true
                } else {
                    // 이동 후보 누적
                    movingCandidateCount++
                    staticCandidateCount = 0
                    if (movingCandidateCount >= MOVING_CONFIRM_FRAMES) {
                        motionState = MotionState.MOVING
                        movingCandidateCount = 0
                        Log.d(TAG, "상태 전환: STATIC → MOVING " +
                              "(gyrRms=${"%.4f".format(gyrRms)} rad/s, " +
                              "velNorm=${"%.3f".format(prevEkfVelNorm)} m/s)")
                        // [P9c] STATIC→MOVING 전환 시 stale 클론 이중 플러시
                        // ① Kotlin localCloneHistory: STATIC 이전 타임스탬프 제거
                        //    → findBeginClone() 이 stale tBegin 을 찾지 못하게 함
                        // ② C++ EKF 내부 클론: marginalize 미호출로 남은 오래된 클론 제거
                        //    → update(tBegin, tEnd) 에서 존재하지 않는 클론 참조 방지
                        // 두 플러시 후 ~1초간 tBegin 미확보 → update() 자동 스킵
                        // → 새 클론 ~1초 누적 후 정상 재개
                        localCloneHistory.clear()
                        EkfBridge.flushClones()
                        EkfBridge.thawStaticState()
                        Log.d(TAG, "STATIC→MOVING: 클론 이중 플러시+공분산 해동 — stale 발산 방지")
                        false   // 이번 프레임부터 추론 진행
                    } else {
                        // 아직 MOVING 미확정 → 정지로 유지
                        true
                    }
                }
            }
            MotionState.MOVING -> {
                if (!isStaticFrame) {
                    // 이동 유지 — 정지 후보 카운터 리셋
                    staticCandidateCount = 0
                    false
                } else {
                    // 정지 후보 누적
                    staticCandidateCount++
                    movingCandidateCount = 0
                    if (staticCandidateCount >= STATIC_CONFIRM_FRAMES) {
                        motionState = MotionState.STATIC
                        staticCandidateCount = 0
                        Log.d(TAG, "상태 전환: MOVING → STATIC " +
                              "(gyrRms=${"%.4f".format(gyrRms)} rad/s, " +
                              "velNorm=${"%.3f".format(prevEkfVelNorm)} m/s)")
                        true    // 이번 프레임부터 ZUPT 적용
                    } else {
                        // 아직 STATIC 미확정 → 이동으로 유지 (추론 계속)
                        false
                    }
                }
            }
        }

        if (currentlyStatic) {
            // [P9] 정지 상태: Hard State Freeze (EKF 측정 우회 직접 고정)
            //
            // 문제 원인 (P8 실패 이유):
            //   apply_position_hold(sigma=0.01) + apply_zupt() 조합은
            //   init_pos_sigma=0.001 → Σ[p,p]=1e-6 m²,  R_pos=(0.01)²=1e-4 m²
            //   → 칼만 게인 K = 1e-6 / (1e-6 + 1e-4) ≈ 0.01 → 보정 1% 미만
            //   → 사실상 위치 고정 불가 → 발산 지속.
            //
            // P9 해결책:
            //   freezeStaticState() — EKF 측정 모델 완전 우회:
            //   ① state_.p = p_anchor  (직접 위치 고정)
            //   ② state_.v = Vec3::Zero()  (직접 속도 0)
            //   ③ Σ[v,v], Σ[p,p] 블록 행·열 전부 0, 대각만 1e-8 (교차 공분산 제거)
            //   → 칼만 게인 없이 완전 고정 → 가속도계 바이어스 적분 드리프트 100% 차단

            // ① 클론 삽입 차단 (P5 방식 유지)
            pendingCloneTs.set(-1L)

            // ② 앵커 기록 (STATIC 기간 중 첫 프레임에만 실행)
            if (staticAnchorPos == null) {
                staticAnchorPos = EkfBridge.getPosition().take(3).toDoubleArray()
                Log.d(TAG, "STATIC 앵커 설정[P9]: " +
                      "(${"%.3f".format(staticAnchorPos!![0])}, " +
                      "${"%.3f".format(staticAnchorPos!![1])}, " +
                      "${"%.3f".format(staticAnchorPos!![2])}) m")
            }

            // ③ Hard State Freeze: 위치·속도 직접 고정 + 공분산 압축
            staticAnchorPos?.let { anchor ->
                EkfBridge.freezeStaticState(anchor[0], anchor[1], anchor[2])
            }

            // ④ yaw 보정 (자이로 편향 보조 — 회전 드리프트 억제)
            applyRotVecYaw("STATIC")
            return
        }

        // MOVING 브랜치 진입 시 앵커 해제 (다음 STATIC 때 새로 기록)
        staticAnchorPos = null

        // ─── 이동(MOVING) 경로 ────────────────────────────────────────

        // ③ 끝 클론 예약 — 현재 최신 센서 ts 기준 (wall-clock 사용 금지)
        val tEndTarget = imuCollector.getLatestSample()?.ts_us ?: return
        pendingCloneTs.set(tEndTarget)

        // ④ propJob 이 클론을 삽입할 때까지 대기 (P4: 20ms → 30ms)
        delay(CLONE_SETTLE_MS)

        // ⑤ 실제 삽입된 끝 클론 ts 확인
        //    tEnd < tEndTarget 이면 아직 미삽입 → 이번 스텝 건너뜀
        val tEnd = lastInsertedCloneTs.get()
        if (tEnd < tEndTarget) return

        // ⑥ P3: Channel 에서 새 클론 ts 를 localCloneHistory 로 drain
        var newTs = cloneChannel.tryReceive().getOrNull()
        while (newTs != null) {
            localCloneHistory.addLast(newTs)
            if (localCloneHistory.size > MAX_CLONE_HISTORY) localCloneHistory.removeFirst()
            newTs = cloneChannel.tryReceive().getOrNull()
        }

        // ⑦ 시작 클론 탐색 (~1초 이전, localCloneHistory 에서 가장 가까운 항목)
        val tBegin = findBeginClone(tEnd, localCloneHistory)
        if (tBegin < 0L || tBegin >= tEnd) return

        // ⑦-P10: 윈도우 동적 비율 게이팅
        //   최초 이동 시 추론 윈도우(1초)는 [정지 데이터 85%][이동 15%] 혼합됨.
        //   네트워크는 이 혼합 윈도우에서 오염된 변위를 예측 → EKF 발산.
        //   윈도우 내 gyr > threshold 인 샘플 비율이 MIN_DYNAMIC_FRACTION 미만이면 건너뜀.
        //   cloneHistory 요구(~1초)와 맞물려 혼합 윈도우 업데이트를 이중으로 차단.
        val dynamicFrac = computeDynamicFraction(window)
        if (dynamicFrac < MIN_DYNAMIC_FRACTION) {
            Log.v(TAG, "윈도우 동적 비율 부족 (${"%.2f".format(dynamicFrac)} < $MIN_DYNAMIC_FRACTION) — 업데이트 스킵")
            return
        }

        // ⑧ body frame → gravity-aligned world frame 좌표 변환
        //    t_begin 클론의 회전 행렬로 yaw를 제거한 월드 프레임으로 변환.
        //    네트워크는 이 프레임에서 학습되었음 (Python dataset.py acc_ga / gyr_ga).
        val R_begin = EkfBridge.getCloneRotation(tBegin)
        val worldWindow = if (R_begin.size == 9) {
            transformWindowToWorldFrame(window, R_begin)
        } else {
            Log.w(TAG, "클론 회전 없음 (tBegin=$tBegin) — body frame 그대로 사용 (정확도 저하)")
            window
        }

        // ⑩ 네트워크 추론 실행
        val inferStart = System.currentTimeMillis()
        val result = try {
            inferEngine.infer(worldWindow)
        } catch (e: Exception) {
            Log.w(TAG, "추론 실패: ${e.message}")
            return
        }
        val inferLatency = System.currentTimeMillis() - inferStart

        // ⑪ Context-Aware Adaptive EKF: 분류 확률 벡터로 Q/R 소프트 스위칭
        //    논문 §4.3.2: R_adaptive = Σ p_k·R^(k), Q_adaptive = Σ p_k·Q^(k)
        //    EKF_NEW 모델(분류기 없음)은 clsProb=zeros → handheld 기준값 폴백
        EkfBridge.applySoftSwitching(result.clsProb)

        // ⑫ 변위 측정값 + 공분산 구성
        val meas = doubleArrayOf(
            result.disp[0].toDouble(),
            result.disp[1].toDouble(),
            result.disp[2].toDouble()
        )
        val cov = buildCovMatrix(result.dispCov)

        // ⑪ t_begin 인덱스 (주변화 기준) — localCloneHistory 에서 직접 탐색
        val beginIdx = localCloneHistory.indexOfFirst { it == tBegin }

        // ⑪-post: 비정상 변위 필터링
        //   좌표 변환 오류 또는 네트워크 이상 출력 시 물리적으로 불가능한 변위 제거.
        //   MAX_DISP_PER_WINDOW_M(6.0m) 초과 = 실내 최대속도(~5m/s) × 1s 초과 → 건너뜀.
        val dispNorm = sqrt(meas[0] * meas[0] + meas[1] * meas[1] + meas[2] * meas[2])
        if (dispNorm > MAX_DISP_PER_WINDOW_M) {
            Log.w(TAG, "비정상 변위 (${"%.2f".format(dispNorm)}m) — EKF 업데이트 건너뜀")
            return
        }

        // ⑪-model: [아이디어 5] 모델 단독 위치 누적 — EKF 속도 게이팅 적용.
        //
        //  직전 EKF 갱신 후 속도 크기(prevEkfVelNorm)가 MODEL_VELOCITY_GATE 미만이면
        //  누적을 차단한다.
        //  ┌─ 이유: 네트워크는 정지 시에도 non-zero displacement 를 출력(바이어스).
        //  │        EKF 속도가 충분히 작으면 실제로 정지 중임을 의미 → 누적 억제.
        //  └─ prevEkfVelNorm: 직전 스텝의 EKF update() 후 속도를 저장해 둔 값.
        //     (이번 스텝 update() 전 속도이므로 1 스텝 지연 — 허용 가능한 오차)
        if (prevEkfVelNorm >= MODEL_VELOCITY_GATE) {
            modelPosX += meas[0]
            modelPosY += meas[1]
            modelTrackPoints.add(Pair(modelPosX, modelPosY))
            if (modelTrackPoints.size > 5000) modelTrackPoints.removeAt(0)
        } else {
            Log.v(TAG, "모델 only 누적 게이팅 " +
                  "(velNorm=${"%.3f".format(prevEkfVelNorm)} m/s < $MODEL_VELOCITY_GATE)")
        }

        // ⑫ EKF 측정 갱신 — t_begin, t_end 모두 si_timestamps_us 에 존재해야 함
        try {
            EkfBridge.update(meas, cov, tBegin, tEnd)
        } catch (e: Exception) {
            Log.w(TAG, "EKF update 실패: ${e.message}")
            return
        }

        // ⑫-P10: 사후 속도 안전망 (Reactive Divergence Recovery)
        //   EKF update() 후 속도가 MAX_POST_UPDATE_SPEED 초과 시 발산 판정.
        //   강제 ZUPT(sigma=0.01 m/s, 거의 하드 리셋 수준)로 속도를 0으로 압축.
        //   이후 EKF 는 정상 상태로 복귀 — 궤적 점프는 발생하나 지속 발산은 방지.
        val postUpdateVel = EkfBridge.getVelocity()
        val postUpdateSpeed = sqrt(
            postUpdateVel[0] * postUpdateVel[0] +
            postUpdateVel[1] * postUpdateVel[1] +
            postUpdateVel[2] * postUpdateVel[2]
        )
        if (postUpdateSpeed > MAX_POST_UPDATE_SPEED) {
            Log.w(TAG, "발산 감지: 속도 ${"%.2f".format(postUpdateSpeed)} m/s > ${MAX_POST_UPDATE_SPEED} — ZUPT 강제 적용")
            EkfBridge.applyZupt(0.01)
        }

        // ⑬ 주변화: tBegin 포함 그 이전 클론 모두 제거
        //    C++ marginalize(idx) 는 0..idx 포함 삭제 (rm = idx+1)
        if (beginIdx >= 0) {
            EkfBridge.marginalize(beginIdx)
            val rm = (beginIdx + 1).coerceAtMost(localCloneHistory.size)
            repeat(rm) { if (localCloneHistory.isNotEmpty()) localCloneHistory.removeFirst() }
        }

        // ⑭ Yaw drift 보정 (이동 중): EKF update 직후 주입
        applyRotVecYaw("MOVING")

        // ⑮ 위치/속도 조회 → UI 갱신
        val pos = EkfBridge.getPosition()  // [px, py, pz, sx, sy, sz]
        val vel = EkfBridge.getVelocity()  // [vx, vy, vz]

        // [아이디어 5] 다음 스텝의 모델 only 게이팅을 위해 현재 속도 크기 저장
        prevEkfVelNorm = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])

        trackPoints.add(Pair(pos[0], pos[1]))
        if (trackPoints.size > 5000) trackPoints.removeAt(0)

        _state.value = _state.value.copy(
            position          = Triple(pos[0], pos[1], pos[2]),
            posStd            = Triple(pos[3], pos[4], pos[5]),
            velocity          = Triple(vel[0], vel[1], vel[2]),
            carryMode         = result.className,
            carryProb         = result.clsProb.maxOrNull() ?: 0f,
            trackPoints       = trackPoints.toList(),
            modelTrackPoints  = modelTrackPoints.toList(),
            inferLatency      = inferLatency
        )
    }

    // ── 헬퍼: 시작 클론 탐색 ─────────────────────────────────────
    /**
     * localCloneHistory 에서 (tEndUs - WINDOW_DURATION_US) 에 가장 가까운 항목 반환.
     * 허용 오차 CLONE_MATCH_TOL_US 초과 시 -1L 반환.
     * inferJob 전용 history 를 직접 참조하므로 락 불필요.
     */
    private fun findBeginClone(tEndUs: Long, history: ArrayDeque<Long>): Long {
        if (history.isEmpty()) return -1L
        val target   = tEndUs - WINDOW_DURATION_US
        var best     = history[0]
        var bestDiff = abs(best - target)
        for (ts in history) {
            val diff = abs(ts - target)
            if (diff < bestDiff) { bestDiff = diff; best = ts }
        }
        return if (bestDiff <= CLONE_MATCH_TOL_US) best else -1L
    }

    // ── 헬퍼: 변위 공분산 행렬 구성 ────────────────────────────
    /**
     * 네트워크 출력 log-variance → variance 변환 후 3×3 대각 행렬(row-major).
     *
     * Python 원본(meas_source_torchscript.py):
     *   meas_cov[meas_cov < -4] = -4   ← exp(-4) ≈ 0.018 m² 가 최소 분산
     *
     * 이전 코드는 variance를 1e-6 까지 허용해 과도한 신뢰 가능성이 있었음.
     * log-variance를 -4 에서 클리핑 후 exp 적용으로 Python 동작과 일치.
     */
    private fun buildCovMatrix(dispCov: FloatArray): DoubleArray {
        val cov = DoubleArray(9) { 0.0 }
        for (i in 0 until 3) {
            val logVar = dispCov[i].toDouble().coerceAtLeast(-4.0)  // Python: clip < -4
            // [P9d] 최솟값 MIN_MEAS_COV 로 바닥 설정:
            // 네트워크가 과도하게 자신감 있을 때 EKF 가 잘못된 측정값을 맹목적으로 따라가는 것 방지
            cov[i * 3 + i] = exp(logVar).coerceIn(MIN_MEAS_COV, 100.0)
        }
        return cov
    }

    // ── 헬퍼: 자이로 RMS 계산 (정적 상태 판정용) ─────────────────
    /**
     * body frame window 에서 자이로(ch 3-5) 3축 합산 RMS 를 반환 (rad/s).
     * 정지 판정에 사용: 이 값이 STATIC_GYR_RMS_THRESHOLD 미만이면 정적.
     *
     * 계산: sqrt( mean_over_t( gx²+gy²+gz² ) )
     * = sqrt( (Σ(gx²+gy²+gz²)) / N )
     */
    private fun computeGyrRms(window: FloatArray): Float {
        val N = ImuCollector.WINDOW_SIZE
        var sumSq = 0f
        for (t in 0 until N) {
            val gx = window[3 * N + t]
            val gy = window[4 * N + t]
            val gz = window[5 * N + t]
            sumSq += gx * gx + gy * gy + gz * gz
        }
        return sqrt(sumSq / N)
    }

    /**
     * [P10] 추론 윈도우에서 동적 샘플(gyr > threshold)의 비율을 반환.
     *
     * 최초 이동 시 윈도우는 정지/이동 혼합 → 네트워크 추론 오염 → 발산.
     * 이 값이 MIN_DYNAMIC_FRACTION 미만이면 추론을 건너뛰어 혼합 입력을 차단.
     *
     * @return 0.0(전부 정지) ~ 1.0(전부 이동)
     */
    private fun computeDynamicFraction(window: FloatArray): Float {
        val N = ImuCollector.WINDOW_SIZE
        var dynamicCount = 0
        for (t in 0 until N) {
            val gx = window[3 * N + t]
            val gy = window[4 * N + t]
            val gz = window[5 * N + t]
            val mag = sqrt((gx * gx + gy * gy + gz * gz).toDouble()).toFloat()
            if (mag >= STATIC_GYR_RMS_THRESHOLD) dynamicCount++
        }
        return dynamicCount.toFloat() / N
    }

    // ── 헬퍼: body frame → gravity-aligned world frame 좌표 변환 ──
    /**
     * Python imu_tracker.py 의 get_displacement_and_cov_from_network() 에 해당하는 변환.
     *
     * 알고리즘:
     *  1. t_begin 의 EKF 회전 R_begin (world←body) 에서 yaw 를 제거 → R_yawfree
     *  2. EKF 자이로 편향 bg 를 가져와 보정된 자이로로 상대 회전 Rs_bofbi[t] 적분
     *     → Python scekf.py 의 (net_gyr[j] − bg) × dt 와 동일
     *  3. Rs_net[t] = R_yawfree @ Rs_bofbi[t]
     *  4. 각 샘플: linAcc_w = Rs_net[t] @ linAcc_body,  gyr_w = Rs_net[t] @ gyr_body
     *
     * @param window   FloatArray[6×100] channel-major, body frame (ch0-2: linAcc, ch3-5: gyr)
     * @param R