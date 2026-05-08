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
         * 0.03 rad/s: 정지 노이즈(0.01) 3× → 충분한 여유; 느린 보행(0.05+)은 오감지하지 않음.
         * 이전 0.05 는 느린 보행 초기 구간을 억제 → 시작 지연 유발.
         */
        private const val STATIC_GYR_RMS_THRESHOLD = 0.03f  // rad/s

        /**
         * 1-초 윈도우당 최대 허용 변위 (m).
         * 실내 최대 이동속도 ≈ 4-5 m/s → 1초 ≈ 5m.
         * 6m 초과 시 좌표 변환 오류 또는 네트워크 이상 출력으로 판단 → 건너뜀.
         */
        private const val MAX_DISP_PER_WINDOW_M = 6.0

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

        // ② [P5 수정] Hysteresis 정지 판정을 클론 예약보다 먼저 수행.
        //
        //  수정 전: pendingCloneTs.set() → delay → 판정 → STATIC 이면 return
        //           → 정지 중에도 20Hz 로 클론이 C++ EKF 에 삽입됨.
        //             marginalize() 는 호출 안 되므로 클론 수 무한 증가.
        //             → propagate() 의 O(N²) 연산 폭증 → 프리즈 유발.
        //
        //  수정 후: 먼저 판정 → STATIC 이면 클론 예약 없이 즉시 ZUPT 처리.
        //
        //  STATIC → MOVING : gyrRms ≥ threshold 가 MOVING_CONFIRM_FRAMES 연속
        //  MOVING → STATIC : gyrRms <  threshold 가 STATIC_CONFIRM_FRAMES  연속
        val gyrRms        = computeGyrRms(window)
        val isStaticFrame = gyrRms < STATIC_GYR_RMS_THRESHOLD

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
            // [P5] 정지 상태: 클론 삽입 차단 + C++ 클론 정리 + ZUPT

            // ① 신규 클론 예약 취소 → propJob 이 추가 클론을 삽입하지 않음
            pendingCloneTs.set(-1L)

            // ② Channel 잔여 클론을 localCloneHistory 로 drain
            //    (이전 MOVING 스텝이나 race 로 남은 ts 포함)
            var ch = cloneChannel.tryReceive().getOrNull()
            while (ch != null) {
                localCloneHistory.addLast(ch)
                if (localCloneHistory.size > MAX_CLONE_HISTORY) localCloneHistory.removeFirst()
                ch = cloneChannel.tryReceive().getOrNull()
            }

            // ③ C++ EKF 에 남은 클론 전부 주변화 → 상태 벡터 최소화
            //    정지 중에는 클론이 불필요. 주변화 후 propagate() 부하 O(15²) 로 복귀.
            val n = localCloneHistory.size
            if (n > 0) {
                try {
                    EkfBridge.marginalize(n - 1)
                } catch (e: Exception) {
                    Log.w(TAG, "STATIC 클론 주변화 실패: ${e.message}")
                }
                localCloneHistory.clear()
            }

            Log.d(TAG, "정적 상태 확정 (gyrRms=${"%.4f".format(gyrRms)} rad/s) — ZUPT 적용")
            EkfBridge.applyZupt()
            applyRotVecYaw("STATIC")
            return
        }

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
        //   MAX_DISP_PER_WINDOW_M(6.0m) 초과 = �