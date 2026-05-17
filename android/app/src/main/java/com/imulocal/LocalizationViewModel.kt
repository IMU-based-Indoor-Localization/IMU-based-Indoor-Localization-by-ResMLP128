package com.imulocal

import android.app.Application
import android.location.Location
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.naver.maps.geometry.LatLng
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * LocalizationViewModel.kt — 단방향 컨트롤러 (Stage 1 → Stage 2 → Stage 3)
 * ==================================================================
 *
 * P23 변경사항:
 *   - GPS 경로 / IMU-only 경로 / LLIO(EKF) 경로 세 가지 동시 추적
 *   - GpsTracker 통합: GPS 첫 수신 시점에 세 경로 앵커링
 *   - IMU-only 누적: Stage 2 output의 dispLocal을 world frame으로 적분
 *   - meterOffsetToLatLng: 미터 오프셋 → 위경도 (앵커 GPS 좌표 기준)
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │ 단방향 코루틴 구성                                           │
 *  │   propJob  (5ms)  : Stage1 → Stage3.propagate              │
 *  │   inferJob (50ms) : Stage1 → Stage2 → Stage3.update        │
 *  │                     + IMU-only 누적                          │
 *  │   uiJob    (100ms): Stage3 출력 + GPS + 세 경로 → State     │
 *  └─────────────────────────────────────────────────────────────┘
 */
class LocalizationViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "Controller"
        private const val PROP_POLL_MS      = 5L
        private const val INFER_INTERVAL_MS = 50L
        private const val UI_INTERVAL_MS    = 100L
        private const val WARMUP_MS         = 3_000L
        private const val MAX_TRACK_POINTS  = 5_000
    }

    // ── 3 Stage 노드 ───────────────────────────────────────────────
    private val stage1          = AbsoluteSensorNode(application)
    private val inferenceEngine = InferenceEngine(application)
    private val stage2          = StatelessInferenceNode(inferenceEngine)
    private val stage3          = RobustEkfTracker()

    // ── GPS 트래커 ────────────────────────────────────────────────
    private val gpsTracker = GpsTracker(application)

    // ── UI 상태 ────────────────────────────────────────────────────
    data class LocalizationState(
        val isRunning:     Boolean = false,
        val position:      Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val posStd:        Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val velocity:      Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val carryMode:     String  = "unknown",
        val carryProb:     Float   = 0f,
        val inferLatency:  Long    = 0L,
        // 캘리브레이션
        val calibrating:   Boolean = false,
        val calibProgress: Float   = 0f,
        val calibDone:     Boolean = false,
        // ── 세 경로 (GPS 첫 수신 후부터 누적) ─────────────────────
        val gpsPath:       List<LatLng> = emptyList(),   // GPS 실제 경로  (파랑)
        val imuOnlyPath:   List<LatLng> = emptyList(),   // IMU-only 경로  (주황)
        val llioPath:      List<LatLng> = emptyList(),   // LLIO(EKF) 경로 (초록)
        val currentGps:    LatLng?      = null,          // 현재 GPS 위치 (지도 중심용)
        val gpsReady:      Boolean      = false,         // GPS 첫 수신 완료 여부
        // 하위 호환 (TrackView 등 기존 코드 대비)
        val trackPoints:   List<Pair<Double, Double>> = emptyList()
    )

    private val _state = MutableStateFlow(LocalizationState())
    val state: StateFlow<LocalizationState> = _state

    // ── 코루틴 핸들 ───────────────────────────────────────────────
    private var propJob:  Job? = null
    private var inferJob: Job? = null
    private var uiJob:    Job? = null

    // ── 경로 누적 버퍼 ────────────────────────────────────────────
    private val gpsPoints  = mutableListOf<LatLng>()
    private val imuOnlyPts = mutableListOf<LatLng>()
    private val llioPts    = mutableListOf<LatLng>()
    private var startTimeMs = 0L

    // ── GPS 앵커 (GPS 첫 수신 시점에 확정) ───────────────────────
    @Volatile private var gpsAnchorLat: Double = 0.0
    @Volatile private var gpsAnchorLng: Double = 0.0
    @Volatile private var ekfAnchorPos:      DoubleArray? = null
    @Volatile private var imuOnlyAnchorPos:  DoubleArray? = null
    @Volatile private var gpsAnchored: Boolean = false

    // ── IMU-only 누적 위치 (inferJob 이 갱신) ────────────────────
    private val imuOnlyPos  = DoubleArray(3)
    private val imuOnlyLock = Any()

    // ── inferJob 캐시 ─────────────────────────────────────────────
    @Volatile private var lastCarryMode:    String = "unknown"
    @Volatile private var lastCarryProb:    Float  = 0f
    @Volatile private var lastInferLatency: Long   = 0L

    // ── start ────────────────────────────────────────────────────
    fun start() {
        if (_state.value.isRunning) return
        startTimeMs = System.currentTimeMillis()
        Log.i(TAG, "측위 시작 (P23 — GPS + IMU-only + LLIO)")

        viewModelScope.launch(Dispatchers.IO) {
            if (!inferenceEngine.isLoaded()) {
                try { inferenceEngine.load() }
                catch (e: Exception) {
                    Log.e(TAG, "모델 로드 실패: ${e.message}")
                    return@launch
                }
            }

            stage3.reset()
            clearPaths()

            if (!stage1.start()) {
                Log.e(TAG, "Stage 1 시작 실패 — 필수 센서 미지원")
                return@launch
            }

            // GPS 수신 시작
            gpsTracker.start { location -> onGpsLocation(location) }

            // propJob
            propJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val samples = stage1.drainPropagateQueue()
                    for (s in samples) stage3.propagate(s)
                    delay(PROP_POLL_MS)
                }
            }

            // inferJob
            inferJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val loopStart = System.nanoTime()
                    val window    = stage1.getWindowSamples()
                    if (window != null) {
                        val inferStart = System.currentTimeMillis()
                        val output     = stage2.infer(window)
                        val inferMs    = System.currentTimeMillis() - inferStart
                        if (output != null) {
                            stage3.update(output)
                            lastCarryMode    = output.className
                            lastCarryProb    = output.classProb.maxOrNull() ?: 0f
                            lastInferLatency = inferMs

                            // IMU-only 누적: dispLocal(yaw-free) → world frame 적분
                            val cosY = cos(output.windowStartYaw)
                            val sinY = kotlin.math.sin(output.windowStartYaw)
                            val dx = cosY * output.dispLocal[0].toDouble() - sinY * output.dispLocal[1].toDouble()
                            val dy = sinY * output.dispLocal[0].toDouble() + cosY * output.dispLocal[1].toDouble()
                            val dz = output.dispLocal[2].toDouble()
                            synchronized(imuOnlyLock) {
                                imuOnlyPos[0] += dx
                                imuOnlyPos[1] += dy
                                imuOnlyPos[2] += dz
                            }
                        }
                    }
                    val elapsedMs = (System.nanoTime() - loopStart) / 1_000_000L
                    delay((INFER_INTERVAL_MS - elapsedMs).coerceAtLeast(1L))
                }
            }

            // uiJob
            uiJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val curCalibrating = stage1.isCalibrating()
                    val curCalibProg   = stage1.getCalibrationProgress()
                    val curCalibDone   = stage1.isCalibrationDone()

                    if (stage3.isInitialized()) {
                        val p  = stage3.getPosition()
                        val v  = stage3.getVelocity()
                        val sd = stage3.getPositionStd()

                        val elapsed = System.currentTimeMillis() - startTimeMs
                        if (elapsed >= WARMUP_MS && !curCalibrating && gpsAnchored) {
                            val ekfAnchor = ekfAnchorPos ?: DoubleArray(3)
                            val imuAnchor = imuOnlyAnchorPos ?: DoubleArray(3)

                            // LLIO 경로
                            val llioLatLng = meterOffsetToLatLng(
                                gpsAnchorLat, gpsAnchorLng,
                                p[0] - ekfAnchor[0], p[1] - ekfAnchor[1]
                            )
                            // IMU-only 경로
                            val imuNow = synchronized(imuOnlyLock) { imuOnlyPos.copyOf() }
                            val imuLatLng = meterOffsetToLatLng(
                                gpsAnchorLat, gpsAnchorLng,
                                imuNow[0] - imuAnchor[0], imuNow[1] - imuAnchor[1]
                            )

                            synchronized(llioPts) {
                                llioPts.add(llioLatLng)
                                if (llioPts.size > MAX_TRACK_POINTS) llioPts.removeAt(0)
                            }
                            synchronized(imuOnlyPts) {
                                imuOnlyPts.add(imuLatLng)
                                if (imuOnlyPts.size > MAX_TRACK_POINTS) imuOnlyPts.removeAt(0)
                            }
                        }

                        val llioSnap = synchronized(llioPts)   { llioPts.toList()   }
                        val imuSnap  = synchronized(imuOnlyPts) { imuOnlyPts.toList() }
                        val gpsSnap  = synchronized(gpsPoints)  { gpsPoints.toList() }

                        _state.value = _state.value.copy(
                            position      = Triple(p[0], p[1], p[2]),
                            posStd        = Triple(sd[0], sd[1], sd[2]),
                            velocity      = Triple(v[0], v[1], v[2]),
                            carryMode     = lastCarryMode,
                            carryProb     = lastCarryProb,
                            inferLatency  = lastInferLatency,
                            calibrating   = curCalibrating,
                            calibProgress = curCalibProg,
                            calibDone     = curCalibDone,
                            llioPath      = llioSnap,
                            imuOnlyPath   = imuSnap,
                            gpsPath       = gpsSnap,
                            currentGps    = gpsSnap.lastOrNull(),
                            gpsReady      = gpsAnchored,
                            trackPoints   = llioSnap.map { Pair(it.latitude, it.longitude) }
                        )
                    } else {
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

    // ── GPS 콜백 (메인 스레드) ─────────────────────────────────────
    private fun onGpsLocation(location: Location) {
        val latLng = LatLng(location.latitude, location.longitude)

        synchronized(gpsPoints) {
            gpsPoints.add(latLng)
            if (gpsPoints.size > MAX_TRACK_POINTS) gpsPoints.removeAt(0)
        }

        // GPS 첫 수신 + EKF 초기화 완료 시 앵커 확정
        if (!gpsAnchored && stage3.isInitialized()) {
            gpsAnchorLat     = location.latitude
            gpsAnchorLng     = location.longitude
            ekfAnchorPos     = stage3.getPosition()
            imuOnlyAnchorPos = synchronized(imuOnlyLock) { imuOnlyPos.copyOf() }
            gpsAnchored      = true
            Log.i(TAG, "GPS 앵커 확정: lat=$gpsAnchorLat lng=$gpsAnchorLng")
        }

        // 현재 GPS 즉시 반영 (지도 중심 이동)
        _state.value = _state.value.copy(
            currentGps = latLng,
            gpsReady   = gpsAnchored
        )
    }

    // ── stop ─────────────────────────────────────────────────────
    fun stop() {
        propJob?.cancel();  propJob  = null
        inferJob?.cancel(); inferJob = null
        uiJob?.cancel();    uiJob    = null
        stage1.stop()
        gpsTracker.stop()
        _state.value = _state.value.copy(isRunning = false)
        Log.i(TAG, "측위 정지 — 진단: ${stage3.getDiagnostics()}")
    }

    // ── reset ────────────────────────────────────────────────────
    fun reset() {
        stop()
        clearPaths()
        stage3.reset()
        lastCarryMode    = "unknown"
        lastCarryProb    = 0f
        lastInferLatency = 0L
        _state.value = LocalizationState()
        Log.i(TAG, "리셋")
    }

    private fun clearPaths() {
        synchronized(gpsPoints)   { gpsPoints.clear()   }
        synchronized(imuOnlyPts)  { imuOnlyPts.clear()  }
        synchronized(llioPts)     { llioPts.clear()     }
        synchronized(imuOnlyLock) {
            imuOnlyPos[0] = 0.0; imuOnlyPos[1] = 0.0; imuOnlyPos[2] = 0.0
        }
        gpsAnchored      = false
        ekfAnchorPos     = null
        imuOnlyAnchorPos = null
    }

    /**
     * 미터 단위 오프셋 → LatLng 근사 변환.
     *   dlat = dy / 111320
     *   dlng = dx / (111320 * cos(lat_rad))
     */
    private fun meterOffsetToLatLng(
        anchorLat: Double, anchorLng: Double,
        dx: Double, dy: Double
    ): LatLng {
        val dlat = dy / 111320.0
        val dlng = dx / (111320.0 * cos(anchorLat * PI / 180.0))
        return LatLng(anchorLat + dlat, anchorLng + dlng)
    }

    override fun onCleared() {
        super.onCleared()
        stop()
        inferenceEngine.release()
    }
}
