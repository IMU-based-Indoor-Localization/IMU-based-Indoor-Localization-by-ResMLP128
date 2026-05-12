package com.imulocal

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * LocalizationViewModel.kt — 단방향 컨트롤러 (Stage 1 → Stage 2 → Stage 3)
 * ==================================================================
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │ 책임 — 순수 데이터 라우터                                   │
 *  │                                                              │
 *  │  ‣ 3 Stage 인스턴스의 lifecycle 관리 (start/stop/reset)     │
 *  │  ‣ 코루틴 3개로 단방향 데이터 흐름 (propJob/inferJob/uiJob)│
 *  │  ‣ Stage 3 출력 → LocalizationState 방출 → UI              │
 *  │                                                              │
 *  │ 비-책임 — 모두 Stage 1/2/3 내부에 위임                       │
 *  │  ‣ 좌표 변환 / yaw 추출 / 회전 복원                          │
 *  │  ‣ EKF 적분 / 게이팅 / Adaptive R                            │
 *  │  ‣ 분류 / 정규화 / 모델 호출                                 │
 *  └─────────────────────────────────────────────────────────────┘
 *
 * ── 단방향 코루틴 구성 ────────────────────────────────────────
 *   propJob (Dispatchers.Default, 5 ms 폴링):
 *       stage1.drainPropagateQueue() → stage3.propagate(sample)
 *
 *   inferJob (Dispatchers.Default, 50 ms = 20 Hz 목표):
 *       stage1.getWindowSamples() → stage2.infer(window)
 *                                 → stage3.update(output)
 *       마지막 carryMode/carryProb/inferLatency 캐시
 *
 *   uiJob (Dispatchers.Default, 100 ms = 10 Hz):
 *       stage3.getPosition()/getVelocity()/getPositionStd()
 *       → trackPoints 누적 (워밍업 3초 후) → _state 갱신
 *
 * ── 단방향 흐름 보장 ──────────────────────────────────────────
 *   stage3 의 상태는 어떤 코루틴에서도 stage1, stage2 로 전달되지 않는다.
 *   세 Stage 모두 자체 lock / immutable 인터페이스로 동시 접근에 안전하다.
 */
class LocalizationViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "Controller"

        /** propJob 폴링 주기 (Stage 3.propagate 호출 간격) */
        private const val PROP_POLL_MS    = 5L

        /** inferJob 목표 주기 (Stage 2 + Stage 3.update 호출 간격) */
        private const val INFER_INTERVAL_MS = 50L

        /** uiJob 갱신 주기 (LocalizationState 방출 간격) */
        private const val UI_INTERVAL_MS  = 100L

        /** 워밍업 — 시작 후 이 시간 동안 trackPoints 적재 보류 */
        private const val WARMUP_MS       = 3_000L

        /** trackPoints 최대 크기 (메모리 보호) */
        private const val MAX_TRACK_POINTS = 5_000
    }

    // ── 3 Stage 노드 ───────────────────────────────────────────────
    private val stage1 = AbsoluteSensorNode(application)
    private val inferenceEngine = InferenceEngine(application)
    private val stage2 = StatelessInferenceNode(inferenceEngine)
    private val stage3 = RobustEkfTracker()

    // ── UI 상태 (MainActivity / TrackView 가 구독) ─────────────────
    data class LocalizationState(
        val isRunning:         Boolean = false,
        val position:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val posStd:            Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val velocity:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val carryMode:         String  = "unknown",
        val carryProb:         Float   = 0f,
        val trackPoints:       List<Pair<Double, Double>> = emptyList(),
        val modelTrackPoints:  List<Pair<Double, Double>> = emptyList(),
        val inferLatency:      Long    = 0L,
        // ── 실시간 영점 보정 진행 상태 (Stage 1 AbsoluteSensorNode 가 소스) ──
        val calibrating:       Boolean = false, // true 면 UI 안내 카드 표시
        val calibProgress:     Float   = 0f,    // 0.0 → 1.0
        val calibDone:         Boolean = false  // 한 번이라도 완료되었는지
    )

    private val _state = MutableStateFlow(LocalizationState())
    val state: StateFlow<LocalizationState> = _state

    // ── 내부 상태 ─────────────────────────────────────────────────
    private var propJob:  Job? = null
    private var inferJob: Job? = null
    private var uiJob:    Job? = null

    private val trackPoints = mutableListOf<Pair<Double, Double>>()
    private var startTimeMs: Long = 0L

    // inferJob 이 마지막 추론 결과를 캐시 → uiJob 이 읽어 UI 에 반영
    @Volatile private var lastCarryMode:    String = "unknown"
    @Volatile private var lastCarryProb:    Float  = 0f
    @Volatile private var lastInferLatency: Long   = 0L

    // ── start ────────────────────────────────────────────────────
    fun start() {
        if (_state.value.isRunning) return
        startTimeMs = System.currentTimeMillis()
        Log.i(TAG, "측위 시작 (단방향 파이프라인 P20)")

        viewModelScope.launch(Dispatchers.IO) {
            // 1. 모델 로드 (1회만)
            if (!inferenceEngine.isLoaded()) {
                try {
                    inferenceEngine.load()
                } catch (e: Exception) {
                    Log.e(TAG, "모델 로드 실패: ${e.message}")
                    return@launch
                }
            }

            // 2. Stage 3 reset
            stage3.reset()
            synchronized(trackPoints) { trackPoints.clear() }
            lastCarryMode = "unknown"
            lastCarryProb = 0f
            lastInferLatency = 0L

            // 3. Stage 1 시작 — 필수 센서 미지원 시 조기 종료
            if (!stage1.start()) {
                Log.e(TAG, "Stage 1 시작 실패 — 필수 센서(rotVec/acc/gyr) 미지원")
                return@launch
            }

            // 4. propJob — Stage 1 → Stage 3.propagate (5ms 폴링)
            propJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val samples = stage1.drainPropagateQueue()
                    for (s in samples) stage3.propagate(s)
                    delay(PROP_POLL_MS)
                }
            }

            // 5. inferJob — Stage 1 → Stage 2 → Stage 3.update (20Hz 목표)
            inferJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val loopStart = System.nanoTime()
                    val window = stage1.getWindowSamples()
                    if (window != null) {
                        val inferStart = System.currentTimeMillis()
                        val output = stage2.infer(window)
                        val inferMs = System.currentTimeMillis() - inferStart
                        if (output != null) {
                            stage3.update(output)
                            lastCarryMode    = output.className
                            lastCarryProb    = output.classProb.maxOrNull() ?: 0f
                            lastInferLatency = inferMs
                        }
                    }
                    val elapsedMs = (System.nanoTime() - loopStart) / 1_000_000L
                    val remaining = INFER_INTERVAL_MS - elapsedMs
                    delay(remaining.coerceAtLeast(1L))
                }
            }

            // 6. uiJob — Stage 3 출력 + Stage 1 캘리브레이션 진행도 → LocalizationState (10Hz)
            uiJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    // ── 캘리브레이션 진행도는 항상 갱신 (stage3 미초기화 단계에서도) ──
                    // (이름은 LocalizationState 필드와 shadowing 회피용으로 prefix 추가)
                    val curCalibrating = stage1.isCalibrating()
                    val curCalibProg   = stage1.getCalibrationProgress()
                    val curCalibDone   = stage1.isCalibrationDone()

                    if (stage3.isInitialized()) {
                        val p  = stage3.getPosition()
                        val v  = stage3.getVelocity()
                        val sd = stage3.getPositionStd()

                        // 워밍업 이후 + 캘리브레이션 완료 이후만 trackPoints 적재
                        val elapsed = System.currentTimeMillis() - startTimeMs
                        if (elapsed >= WARMUP_MS && !curCalibrating) {
                            synchronized(trackPoints) {
                                trackPoints.add(Pair(p[0], p[1]))
                                if (trackPoints.size > MAX_TRACK_POINTS) trackPoints.removeAt(0)
                            }
                        }

                        val snapshot = synchronized(trackPoints) { trackPoints.toList() }

                        _state.value = _state.value.copy(
                            position      = Triple(p[0], p[1], p[2]),
                            posStd        = Triple(sd[0], sd[1], sd[2]),
                            velocity      = Triple(v[0], v[1], v[2]),
                            carryMode     = lastCarryMode,
                            carryProb     = lastCarryProb,
                            trackPoints   = snapshot,
                            inferLatency  = lastInferLatency,
                            calibrating   = curCalibrating,
                            calibProgress = curCalibProg,
                            calibDone     = curCalibDone
                        )
                    } else {
                        // stage3 가 아직 초기화 안 됨 (대개 캘리브레이션 진행 단계)
                        _state.value = _state.value.copy(
                            calibrating   = curCalibrating,
                            calibProgress = curCalibProg,
                            calibDone     = curCalibDone
                        )
                    }
                    delay(UI_INTERVAL_MS)
                }
            }

            _state.value = _state.value.copy(isRunning = true)
        }
    }

    // ── stop ─────────────────────────────────────────────────────
    fun stop() {
        propJob?.cancel();  propJob  = null
        inferJob?.cancel(); inferJob = null
        uiJob?.cancel();    uiJob    = null
        stage1.stop()
        _state.value = _state.value.copy(isRunning = false)
        Log.i(TAG, "측위 정지 — 진단: ${stage3.getDiagnostics()}")
    }

    // ── reset ────────────────────────────────────────────────────
    fun reset() {
        stop()
        synchronized(trackPoints) { trackPoints.clear() }
        stage3.reset()
        lastCarryMode    = "unknown"
        lastCarryProb    = 0f
        lastInferLatency = 0L
        _state.value = LocalizationState()
        Log.i(TAG, "리셋")
    }

    override fun onCleared() {
        super.onCleared()
        stop()
        inferenceEngine.release()
    }
}
