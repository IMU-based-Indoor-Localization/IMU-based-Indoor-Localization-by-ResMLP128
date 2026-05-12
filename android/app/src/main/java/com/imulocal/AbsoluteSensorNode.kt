package com.imulocal

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * AbsoluteSensorNode.kt — Stage 1
 * ==================================================================
 * 단방향(One-Way) 아키텍처의 최하단 노드: 절대 좌표계 센서 전처리기.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │ 설계 원칙 — EKF 의 존재를 전혀 모른다                       │
 *  │                                                              │
 *  │  ‣ 오직 Android SensorManager 만 의존                       │
 *  │  ‣ 출력만 있고 외부로부터 입력을 받지 않음                  │
 *  │  ‣ TYPE_ROTATION_VECTOR 의 절대 회전 행렬을 GT 대용으로     │
 *  │    사용해 학습 시 GT 쿼터니언 분포에 근접한 입력을 만든다   │
 *  │  ‣ 자세 추정 오차가 누적되지 않음 (지자기·중력 융합으로     │
 *  │    안드로이드가 매 콜백 절대 회전을 제공)                   │
 *  └─────────────────────────────────────────────────────────────┘
 *
 * ── 입력 ──────────────────────────────────────────────────────
 *   TYPE_ACCELEROMETER       : raw accel (중력 포함, body frame)
 *   TYPE_LINEAR_ACCELERATION : linear accel (중력 제거됨, body frame)
 *   TYPE_GYROSCOPE           : raw gyro (rad/s, body frame)
 *   TYPE_ROTATION_VECTOR     : 절대 자세 (지자기·가속도·자이로 융합)
 *
 * ── 동작 ──────────────────────────────────────────────────────
 *   1. TYPE_ROTATION_VECTOR 콜백 시 getRotationMatrixFromVector() 로
 *      3×3 회전 행렬 R = R_world←body 캐시.
 *   2. acc / linAcc / gyro 도착 시 캐시된 R 로 즉시 world frame 회전.
 *        a_world = R @ a_body
 *        g_world = R @ g_body
 *   3. 100Hz 리샘플링 후 WorldSample 을 ringBuffer 에 적재.
 *
 * ── 출력 (Stage 2 가 polling) ─────────────────────────────────
 *   WorldSample
 *     ts_us          : SensorEvent.timestamp / 1000  (μs)
 *     worldAcc[3]    : R @ raw_acc       (m/s², 중력 포함)
 *     worldLinAcc[3] : R @ linear_acc    (m/s², 중력 제거)
 *     worldGyr[3]    : R @ raw_gyro      (rad/s)
 *     rotMat[9]      : R_world←body row-major
 *     rotAccuracy    : SENSOR_STATUS_*   (지자기 품질 0=UNRELIABLE … 3=HIGH)
 *
 *   getWindow()        — 최근 100 샘플을 channel-major FloatArray(600) 로
 *                        (학습 acc_ga + gyr_ga 와 동일한 6채널 레이아웃)
 *   getWindowSamples() — 최근 100 샘플을 List<WorldSample> 로 (회전 행렬 포함)
 *
 * ── 단방향 흐름의 위치 ─────────────────────────────────────────
 *   [Stage 1] AbsoluteSensorNode      ← 이 파일
 *       │ World Frame 윈도우 (yaw 포함)
 *       ▼
 *   [Stage 2] 추론 어댑터              ← 윈도우 시작 yaw 제거 + 정규화 + 모델 호출
 *       │ disp_local (yaw-free local frame), disp_cov, cls_prob
 *       ▼
 *   [Stage 3] EKF 게이트웨이           ← 단방향 측정값 주입 + AEKF 튜닝
 *       │ EKF 절대 위치 / 속도
 *       ▼
 *   UI (MainActivity, TrackView)
 */
class AbsoluteSensorNode(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "Stage1.AbsNode"

        /** 출력 윈도우 샘플링 주파수 (Hz) */
        const val TARGET_HZ   = 100

        /** Stage 2 가 한 번에 가져가는 윈도우 길이 (1초 분) */
        const val WINDOW_SIZE = 100

        /** ringBuffer 용량 (윈도우 2 개 분 = 2 초) */
        const val BUFFER_SIZE = WINDOW_SIZE * 2

        /** getWindow() 가 반환하는 채널 수: worldLinAcc(3) + worldGyr(3) */
        const val CHANNEL_NUM = 6

        /**
         * 실시간 영점 보정(Auto-Calibration) 지속 시간(ms).
         * 시작 직후 이 시간 동안 모든 입력을 영점 누적용으로만 사용하고,
         * ringBuffer / propagateQueue 에는 적재하지 않는다.
         */
        const val CALIBRATION_DURATION_MS: Long = 2_000L

        /** 표준 중력 가속도 (m/s²) — 가속도계 raw bias 추출 시 사용 */
        const val STANDARD_GRAVITY: Float = 9.80665f
    }

    /**
     * Stage 1 출력 단위.
     * @param ts_us       SensorEvent.timestamp (μs)
     * @param worldAcc    raw accel rotated to world frame (중력 포함) m/s²
     * @param worldLinAcc linear accel rotated to world frame (중력 제거) m/s²
     * @param worldGyr    raw gyro rotated to world frame (rad/s)
     * @param rotMat      이 샘플 시점의 R_world←body (row-major 9개)
     * @param rotAccuracy TYPE_ROTATION_VECTOR 정확도 (0..3)
     */
    data class WorldSample(
        val ts_us:        Long,
        val worldAcc:     FloatArray,
        val worldLinAcc:  FloatArray,
        val worldGyr:     FloatArray,
        val rotMat:       FloatArray,
        val rotAccuracy:  Int
    )

    // ── 센서 핸들 ────────────────────────────────────────────────
    private val sm     = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sAcc   = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val sLin   = sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val sGyr   = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val sRotV  = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    // ── 최신 센서 캐시 ────────────────────────────────────────────
    @Volatile private var accBody    = FloatArray(3)
    @Volatile private var linAccBody = FloatArray(3)
    @Volatile private var gyrBody    = FloatArray(3)
    @Volatile private var rotMat     = FloatArray(9).also { it[0] = 1f; it[4] = 1f; it[8] = 1f }
    @Volatile private var rotAccuracy = 0
    @Volatile private var hasAcc:   Boolean = false
    @Volatile private var hasGyr:   Boolean = false
    @Volatile private var hasRotV:  Boolean = false

    // ── 윈도우 링 버퍼 (Stage 2 의 getWindow / getWindowSamples 용) ──
    private val ringBuffer = ArrayBlockingQueue<WorldSample>(BUFFER_SIZE)

    // ── propagate 큐 (Stage 3 의 매 100Hz propagate 용) ─────────────
    // propJob 이 drainPropagateQueue() 로 소비할 때까지 모든 신규 샘플 보관.
    // drain 시 비워지며 ringBuffer 와는 독립적으로 운영된다.
    private val propagateQueue = ConcurrentLinkedQueue<WorldSample>()

    // 100Hz 리샘플링 제어
    private var lastSampleTsUs: Long = -1L
    private val sampleIntervalUs: Long = 1_000_000L / TARGET_HZ  // 10,000 μs

    // ── 실시간 영점 보정 (Auto-Calibration) ────────────────────────
    // 모든 배열·카운터는 명시적으로 0 으로 초기화 (쓰레기값 방지).
    // 누적용 sum 은 정밀도 확보를 위해 Double 로 처리한 뒤 Float 평균으로 변환.
    @Volatile private var calibrating: Boolean = false
    @Volatile private var calibrationDone: Boolean = false
    @Volatile private var calibProgress: Float = 0f          // 0.0 → 1.0
    private var calibStartElapsedMs: Long = 0L
    private var calibCount: Int = 0
    private val calibAccSum:    DoubleArray = DoubleArray(3) // {0.0, 0.0, 0.0}
    private val calibLinAccSum: DoubleArray = DoubleArray(3)
    private val calibGyrSum:    DoubleArray = DoubleArray(3)
    private val calibRotSum:    DoubleArray = DoubleArray(9) // 평균 자세 (raw bias 보정용)

    // 확정된 영점(Bias) — 모두 0 으로 초기화되어 보정 전이라도 (값 - 0) = 값 으로 안전 동작
    private val accBias:    FloatArray = FloatArray(3) // {0f, 0f, 0f}
    private val linAccBias: FloatArray = FloatArray(3)
    private val gyrBias:    FloatArray = FloatArray(3)

    // ── 시작 / 종료 ───────────────────────────────────────────────
    /**
     * 모든 필요 센서를 등록한다. TYPE_ACCELEROMETER, TYPE_GYROSCOPE,
     * TYPE_ROTATION_VECTOR 가 필수이며 미지원 시 start() 가 false 반환.
     * TYPE_LINEAR_ACCELERATION 은 선택(없으면 worldLinAcc=0).
     *
     * @return 필수 센서가 모두 존재해 등록되었으면 true
     */
    fun start(): Boolean {
        if (sAcc == null) {
            Log.e(TAG, "TYPE_ACCELEROMETER 미지원 — Stage 1 시작 불가")
            return false
        }
        if (sGyr == null) {
            Log.e(TAG, "TYPE_GYROSCOPE 미지원 — Stage 1 시작 불가")
            return false
        }
        if (sRotV == null) {
            Log.e(TAG, "TYPE_ROTATION_VECTOR 미지원 — Stage 1 시작 불가 " +
                       "(이 노드는 절대 회전이 없으면 동작할 수 없음)")
            return false
        }
        clear()
        // ★ 캘리브레이션 모드로 진입 — 첫 필수센서 도착 시점에 calibStartElapsedMs 설정
        calibrating  = true
        calibrationDone = false
        calibProgress  = 0f
        sm.registerListener(this, sAcc,  SensorManager.SENSOR_DELAY_FASTEST)
        sm.registerListener(this, sGyr,  SensorManager.SENSOR_DELAY_FASTEST)
        sm.registerListener(this, sRotV, SensorManager.SENSOR_DELAY_FASTEST)
        sLin?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        Log.i(TAG, "Stage 1 시작 (acc, gyr, rotVec 등록; linAcc=${sLin != null}) — " +
                   "캘리브레이션 ${CALIBRATION_DURATION_MS}ms 진입")
        return true
    }

    fun stop() {
        sm.unregisterListener(this)
        clear()
        Log.i(TAG, "Stage 1 중지")
    }

    private fun clear() {
        ringBuffer.clear()
        propagateQueue.clear()
        lastSampleTsUs = -1L
        hasAcc = false; hasGyr = false; hasRotV = false
        rotAccuracy = 0

        // 캘리브레이션 상태 / 누적값 / 영점 모두 0 으로 초기화
        calibrating  = false
        calibrationDone = false
        calibProgress  = 0f
        calibStartElapsedMs = 0L
        calibCount = 0
        for (i in 0..2) {
            calibAccSum[i] = 0.0
            calibLinAccSum[i] = 0.0
            calibGyrSum[i] = 0.0
            accBias[i]    = 0f
            linAccBias[i] = 0f
            gyrBias[i]    = 0f
        }
        for (i in 0..8) calibRotSum[i] = 0.0
    }

    // ── SensorEventListener ──────────────────────────────────────
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        if (sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            rotAccuracy = accuracy
            Log.d(TAG, "RotVec 정확도: $accuracy " +
                  when (accuracy) {
                      0 -> "(UNRELIABLE)"; 1 -> "(LOW)"
                      2 -> "(MEDIUM)";     3 -> "(HIGH)"
                      else -> "(?)"
                  })
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val tsUs = event.timestamp / 1000L

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accBody[0] = event.values[0]
                accBody[1] = event.values[1]
                accBody[2] = event.values[2]
                hasAcc = true
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                linAccBody[0] = event.values[0]
                linAccBody[1] = event.values[1]
                linAccBody[2] = event.values[2]
                // 별도 ready 플래그 없음 (선택 센서)
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyrBody[0] = event.values[0]
                gyrBody[1] = event.values[1]
                gyrBody[2] = event.values[2]
                hasGyr = true
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                // R = R_world←body (Android 컨벤션: device-to-world 회전 행렬)
                SensorManager.getRotationMatrixFromVector(rotMat, event.values)
                hasRotV = true
            }
        }

        // 필수 센서 셋이 모두 도착해야 첫 출력 생성
        if (!hasAcc || !hasGyr || !hasRotV) return

        // ── ★ 캘리브레이션 분기 — ringBuffer / propagateQueue 적재 금지 ──
        // 시작 직후 CALIBRATION_DURATION_MS 동안 body frame 입력을 평균 내어
        // 영점(Bias) 을 추정한다. 이 구간에서는 어떤 출력도 만들지 않는다.
        if (calibrating) {
            if (calibStartElapsedMs == 0L) {
                calibStartElapsedMs = SystemClock.elapsedRealtime()
            }
            // 누적 — DoubleArray 로 부동소수 오차 최소화
            // Kotlin 은 Float→Double 암묵 변환을 하지 않으므로 toDouble() 명시 필수.
            calibAccSum[0]    += accBody[0].toDouble();    calibAccSum[1]    += accBody[1].toDouble();    calibAccSum[2]    += accBody[2].toDouble()
            calibLinAccSum[0] += linAccBody[0].toDouble(); calibLinAccSum[1] += linAccBody[1].toDouble(); calibLinAccSum[2] += linAccBody[2].toDouble()
            calibGyrSum[0]    += gyrBody[0].toDouble();    calibGyrSum[1]    += gyrBody[1].toDouble();    calibGyrSum[2]    += gyrBody[2].toDouble()
            for (i in 0..8) calibRotSum[i] += rotMat[i].toDouble()
            calibCount++

            val elapsed = SystemClock.elapsedRealtime() - calibStartElapsedMs
            calibProgress = (elapsed.toFloat() / CALIBRATION_DURATION_MS.toFloat()).coerceIn(0f, 1f)

            if (elapsed >= CALIBRATION_DURATION_MS) {
                performWarmup()
            }
            return
        }

        // 100Hz 리샘플링 (sampleIntervalUs 마다 1 회 emit)
        if (lastSampleTsUs < 0) lastSampleTsUs = tsUs
        if (tsUs - lastSampleTsUs < sampleIntervalUs) return
        lastSampleTsUs = tsUs

        // ── 핵심: 캐시된 절대 회전 R 로 즉시 회전 ───────────────────
        // R 은 row-major: index = row*3 + col
        //   row 0: [R0 R1 R2]
        //   row 1: [R3 R4 R5]
        //   row 2: [R6 R7 R8]
        val r = rotMat

        // ★ 영점(Bias) 제거 — body frame 에서 먼저 빼고 world 로 회전
        val accDb0 = accBody[0] - accBias[0]
        val accDb1 = accBody[1] - accBias[1]
        val accDb2 = accBody[2] - accBias[2]
        val linDb0 = linAccBody[0] - linAccBias[0]
        val linDb1 = linAccBody[1] - linAccBias[1]
        val linDb2 = linAccBody[2] - linAccBias[2]
        val gyrDb0 = gyrBody[0] - gyrBias[0]
        val gyrDb1 = gyrBody[1] - gyrBias[1]
        val gyrDb2 = gyrBody[2] - gyrBias[2]

        val worldAcc = FloatArray(3)
        worldAcc[0] = r[0]*accDb0 + r[1]*accDb1 + r[2]*accDb2
        worldAcc[1] = r[3]*accDb0 + r[4]*accDb1 + r[5]*accDb2
        worldAcc[2] = r[6]*accDb0 + r[7]*accDb1 + r[8]*accDb2

        val worldLinAcc = FloatArray(3)
        worldLinAcc[0] = r[0]*linDb0 + r[1]*linDb1 + r[2]*linDb2
        worldLinAcc[1] = r[3]*linDb0 + r[4]*linDb1 + r[5]*linDb2
        worldLinAcc[2] = r[6]*linDb0 + r[7]*linDb1 + r[8]*linDb2

        val worldGyr = FloatArray(3)
        worldGyr[0] = r[0]*gyrDb0 + r[1]*gyrDb1 + r[2]*gyrDb2
        worldGyr[1] = r[3]*gyrDb0 + r[4]*gyrDb1 + r[5]*gyrDb2
        worldGyr[2] = r[6]*gyrDb0 + r[7]*gyrDb1 + r[8]*gyrDb2

        val sample = WorldSample(
            ts_us       = tsUs,
            worldAcc    = worldAcc,
            worldLinAcc = worldLinAcc,
            worldGyr    = worldGyr,
            rotMat      = r.copyOf(),
            rotAccuracy = rotAccuracy
        )

        // ringBuffer 가 가득 차면 가장 오래된 것 제거 후 추가
        while (ringBuffer.size >= BUFFER_SIZE) ringBuffer.poll()
        ringBuffer.offer(sample)

        // propagate 큐 (Stage 3 가 소비할 때까지 보관, 500개 넘으면 오래된 것 제거)
        propagateQueue.offer(sample)
        while (propagateQueue.size > 500) propagateQueue.poll()
    }

    // ── Stage 2 가 polling 으로 가져가는 API ─────────────────────
    /**
     * 최근 WINDOW_SIZE(100) 샘플을 channel-major FloatArray 로 반환.
     *
     * 채널 레이아웃 (학습 데이터 dataset.py acc_ga + gyr_ga 와 동일한 6채널):
     *   ch 0-2 : worldLinAcc (m/s², 중력 제거, world frame)
     *   ch 3-5 : worldGyr    (rad/s,          world frame)
     *
     * 주의: 출력은 world frame (yaw 포함). 학습 입력은 윈도우 시작점 yaw 가
     * 제거된 yaw-free local frame 이므로, Stage 2 에서 추가로 yaw 제거를
     * 적용해야 학습 분포와 일치한다 (단방향 흐름 유지를 위해 여기서는
     * 의도적으로 yaw 를 남겨둔다).
     *
     * @return Pair(FloatArray[600], 윈도우 시작 ts_us) 또는 샘플 부족 시 null
     */
    fun getWindow(): Pair<FloatArray, Long>? {
        val snap = ringBuffer.toArray().filterIsInstance<WorldSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(CHANNEL_NUM * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            flat[0 * WINDOW_SIZE + t] = s.worldLinAcc[0]
            flat[1 * WINDOW_SIZE + t] = s.worldLinAcc[1]
            flat[2 * WINDOW_SIZE + t] = s.worldLinAcc[2]
            flat[3 * WINDOW_SIZE + t] = s.worldGyr[0]
            flat[4 * WINDOW_SIZE + t] = s.worldGyr[1]
            flat[5 * WINDOW_SIZE + t] = s.worldGyr[2]
        }
        return Pair(flat, recent.first().ts_us)
    }

    /**
     * 최근 WINDOW_SIZE 샘플을 WorldSample 객체 리스트로 반환.
     * 회전 행렬과 정확도까지 보존하므로 Stage 2 에서 윈도우 시작 yaw 추출,
     * Stage 3 에서 EKF 자세 anchor 등에 직접 활용 가능하다.
     */
    fun getWindowSamples(): List<WorldSample>? {
        val snap = ringBuffer.toArray().filterIsInstance<WorldSample>()
        if (snap.size < WINDOW_SIZE) return null
        return snap.takeLast(WINDOW_SIZE)
    }

    /**
     * propagateQueue 에 쌓인 모든 신규 샘플을 반환하고 큐를 비운다.
     * Stage 3.propagate 가 5ms 폴링으로 호출하여 100Hz 적분에 사용한다.
     * ringBuffer (윈도우용) 는 영향받지 않는다.
     */
    fun drainPropagateQueue(): List<WorldSample> {
        if (propagateQueue.isEmpty()) return emptyList()
        val out = ArrayList<WorldSample>(16)
        var s = propagateQueue.poll()
        while (s != null) {
            out.add(s)
            s = propagateQueue.poll()
        }
        return out
    }

    /** ringBuffer 의 가장 최근 1개 샘플 (없으면 null) */
    fun getLatestSample(): WorldSample? {
        val snap = ringBuffer.toArray().filterIsInstance<WorldSample>()
        return snap.lastOrNull()
    }

    /** 현재 ringBuffer 에 적재된 샘플 수 (디버그용) */
    fun bufferedCount(): Int = ringBuffer.size

    /** 현재 TYPE_ROTATION_VECTOR 정확도 — 0=UNRELIABLE … 3=HIGH */
    fun getRotVecAccuracy(): Int = rotAccuracy

    /** 모든 필수 센서가 한 번이라도 수신되었는지 (start 후 초기 안정화 확인용) */
    fun isReady(): Boolean = hasAcc && hasGyr && hasRotV

    // ──────────────────────────────────────────────────────────────
    // 실시간 영점 보정 (Auto-Calibration) 공개 API
    // ──────────────────────────────────────────────────────────────

    /** 현재 캘리브레이션 누적 중이면 true (start 직후 ~ performWarmup 호출 전까지) */
    fun isCalibrating(): Boolean = calibrating

    /** 한 번이라도 캘리브레이션이 끝났는지 (UI 안내 숨김 판단용) */
    fun isCalibrationDone(): Boolean = calibrationDone

    /** 0.0(시작) → 1.0(완료) 진행률. UI ProgressBar 바인딩용. */
    fun getCalibrationProgress(): Float = calibProgress

    /**
     * 추정된 영점(Bias) 스냅샷.
     *  first  : accBias    (body frame, m/s², 중력 성분 제거된 raw bias)
     *  second : linAccBias (body frame, m/s²)
     *  third  : gyrBias    (body frame, rad/s)
     *
     * 캘리브레이션 완료 전이면 모두 0 벡터.
     */
    fun getBiasSnapshot(): Triple<FloatArray, FloatArray, FloatArray> =
        Triple(accBias.copyOf(), linAccBias.copyOf(), gyrBias.copyOf())

    /**
     * 동적 영점 추정 함수.
     * onSensorChanged 의 캘리브레이션 분기가 CALIBRATION_DURATION_MS 경과
     * 시점에 1 회 호출한다.
     *
     *  ‣ gyrBias    : 정지 평균이 곧 bias (회전 없으면 0 이어야 함)
     *  ‣ linAccBias : Android TYPE_LINEAR_ACCELERATION 은 중력 제거된 출력이라
     *                 정지 평균이 곧 bias
     *  ‣ accBias    : 정지 평균 ≈ R^T·[0,0,g]_world + bias  →
     *                 bias = mean(accBody) − R_meanᵀ·[0,0,g]
     *                 (row-major R 의 경우 body frame 중력 = (R[6]g, R[7]g, R[8]g))
     *
     * 호출 후 calibrating=false, calibrationDone=true, calibProgress=1f 로 전환,
     * 100Hz 리샘플링 타이머도 리셋해 캘리브레이션 동안 누적된 시간차를 무효화한다.
     */
    private fun performWarmup() {
        if (calibCount <= 0) {
            Log.w(TAG, "캘리브레이션 누적 샘플 0개 — 영점을 0 으로 유지")
        } else {
            val n = calibCount.toDouble()

            // 자이로 — 정지 평균 = bias
            gyrBias[0] = (calibGyrSum[0] / n).toFloat()
            gyrBias[1] = (calibGyrSum[1] / n).toFloat()
            gyrBias[2] = (calibGyrSum[2] / n).toFloat()

            // 선형 가속도 — 정지 평균 = bias (중력은 Android 가 이미 제거함)
            linAccBias[0] = (calibLinAccSum[0] / n).toFloat()
            linAccBias[1] = (calibLinAccSum[1] / n).toFloat()
            linAccBias[2] = (calibLinAccSum[2] / n).toFloat()

            // 가속도계 raw — 평균 자세에서 본 body-frame 중력을 빼서 bias 만 추출
            val meanR2x = (calibRotSum[6] / n).toFloat() // R[2,0]
            val meanR2y = (calibRotSum[7] / n).toFloat() // R[2,1]
            val meanR2z = (calibRotSum[8] / n).toFloat() // R[2,2]
            val gBody0 = meanR2x * STANDARD_GRAVITY      // R_meanᵀ·[0,0,g] = 3rd row of R_mean
            val gBody1 = meanR2y * STANDARD_GRAVITY
            val gBody2 = meanR2z * STANDARD_GRAVITY
            accBias[0] = (calibAccSum[0] / n).toFloat() - gBody0
            accBias[1] = (calibAccSum[1] / n).toFloat() - gBody1
            accBias[2] = (calibAccSum[2] / n).toFloat() - gBody2

            Log.i(TAG,
                "캘리브레이션 완료 (n=$calibCount, elapsed=${SystemClock.elapsedRealtime() - calibStartElapsedMs}ms)" +
                "\n  gyrBias    = [${"%.5f".format(gyrBias[0])}, ${"%.5f".format(gyrBias[1])}, ${"%.5f".format(gyrBias[2])}] rad/s" +
                "\n  linAccBias = [${"%.4f".format(linAccBias[0])}, ${"%.4f".format(linAccBias[1])}, ${"%.4f".format(linAccBias[2])}] m/s²" +
                "\n  accBias    = [${"%.4f".format(accBias[0])}, ${"%.4f".format(accBias[1])}, ${"%.4f".format(accBias[2])}] m/s² (중력 제외)"
            )
        }

        calibrating  = false
        calibrationDone = true
        calibProgress  = 1f
        // 캘리브레이션 동안 누적된 시간차 무효화 — 다음 첫 샘플부터 100Hz 시작
        lastSampleTsUs = -1L
    }
}
