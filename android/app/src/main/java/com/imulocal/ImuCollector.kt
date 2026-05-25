package com.imulocal

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ImuCollector.kt
 * ===============
 * Android SensorManager 에서 가속도계 + 자이로스코프를 100Hz 로 수집한다.
 *
 * ── 센서 구성 ───────────────────────────────────────────────────
 *  - TYPE_ACCELEROMETER       : 중력 포함 가속도 → EKF 전파용 (sample.acc)
 *  - TYPE_LINEAR_ACCELERATION : 중력 제거 가속도 → 네트워크 입력용 (sample.linAcc)
 *  - TYPE_GYROSCOPE           : 각속도 → EKF + 네트워크 공용 (sample.gyr)
 *
 * ── 두 가지 버퍼 ──────────────────────────────────────────────
 *  1. propagateQueue  : EKF 전파용. 모든 100Hz 샘플을 순서대로 보관.
 *                       drainPropagateQueue() 로 한 번에 소비.
 *  2. ringBuffer      : 추론 윈도우용. 최근 WINDOW_SIZE 샘플 유지.
 *                       getWindow() 로 channel-major 배열 반환.
 *
 * ── getWindow() 채널 레이아웃 ──────────────────────────────────
 *  ch 0-2 : linAcc (선형 가속도, 중력 없음)  m/s²   ← 학습 데이터와 일치
 *  ch 3-5 : gyr    (각속도)                 rad/s
 *
 * ── 타임스탬프 ────────────────────────────────────────────────
 *  SensorEvent.timestamp (부팅 후 경과 나노초) → μs 변환.
 *  System.currentTimeMillis() 는 일체 사용하지 않음.
 */
class ImuCollector(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "ImuCollector"
        const val TARGET_HZ   = 100
        const val WINDOW_SIZE = 100
        const val BUFFER_SIZE = WINDOW_SIZE * 2
        const val CHANNEL_NUM = 6

        /**
         * [P21-ish] 실시간 영점 보정 지속 시간 (ms).
         * 시작 후 이 시간 동안 linAcc + gyr 평균을 bias 로 추정,
         * 이후 모든 샘플에서 bias 차감 후 propagateQueue / ringBuffer 에 적재.
         * 캘리브레이션 중에는 어떠한 샘플도 큐에 들어가지 않으므로
         * EkfBridge 초기화도 자연스럽게 지연된다.
         */
        const val CALIBRATION_DURATION_MS: Long = 2_000L

        // ─────────────────────────────────────────────────────────────
        // [P33-A1 재도입] Butterworth 2차 LPF (fs=100Hz, fc=12Hz)
        // ─────────────────────────────────────────────────────────────
        // scipy.signal.butter(2, 12, btype='low', fs=100) 결과:
        //   b = [0.09131490, 0.18262980, 0.09131490]
        //   a = [1.0, -0.98240579, 0.34766539]
        // Direct Form I: y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
        //
        // 적용 위치: ringBuffer 의 linAcc/gyr 만 (네트워크 입력용).
        // propagateQueue 의 raw acc/gyr 는 LPF 없이 EKF propagate 로 전달
        //   → propagate 적분 정확도 유지 (HANDOFF P38 §P35 교훈 반영).
        //
        // iPhone Core Motion 의 내부 필터링을 흉내내어 분류기 OOD 완화.
        // HANDOFF P38 §P33-A1: 단방향 단계에서 unknown 100% → 84% 효과 확인됨.
        private const val LPF_B0: Float =  0.09131490f
        private const val LPF_B1: Float =  0.18262980f
        private const val LPF_B2: Float =  0.09131490f
        private const val LPF_A1: Float = -0.98240579f
        private const val LPF_A2: Float =  0.34766539f
    }

    /**
     * @param acc       TYPE_ACCELEROMETER [ax, ay, az] m/s²  (중력 포함, RAW — EKF 전파용)
     * @param gyr       TYPE_GYROSCOPE     [gx, gy, gz] rad/s  (RAW — EKF 전파용)
     * @param linAcc    TYPE_LINEAR_ACCELERATION [lx, ly, lz] m/s² (RAW — bias 차감 후)
     * @param linAccLpf [P33-A1] linAcc 의 Butterworth fc=12Hz LPF 버전 (네트워크 입력용)
     * @param gyrLpf    [P33-A1] gyr 의 Butterworth fc=12Hz LPF 버전 (네트워크 입력용)
     * @param rotMat    [P53] 샘플 시점 TYPE_ROTATION_VECTOR 회전행렬 (device→world, row-major 9).
     *                  RotVec dead-reckoning 의 per-timestep gravity-aligned 변환에 사용.
     *                  rotVec 미수신 시 단위행렬.
     */
    data class ImuSample(
        val ts_us:     Long,
        val acc:       FloatArray,   // [ax, ay, az]  m/s²  RAW (gravity included)  → EKF
        val gyr:       FloatArray,   // [gx, gy, gz]  rad/s RAW                     → EKF
        val linAcc:    FloatArray,   // [lx, ly, lz]  m/s²  RAW (no gravity)        → 진단용
        val linAccLpf: FloatArray,   // [lx, ly, lz]  m/s²  LPF                     → 네트워크
        val gyrLpf:    FloatArray,   // [gx, gy, gz]  rad/s LPF                     → 네트워크
        val rotMat:    FloatArray    // [9] device→world 회전행렬 (row-major)        → RotVec DR
    )

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelSensor   = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroSensor    = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    /** 중력 제거 가속도 — 네트워크 입력용. 기기에 없으면 null (대부분 기기에 존재) */
    private val linAccSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    /**
     * 절대 방위 융합 센서 (가속도+자이로+지자기 융합).
     * Yaw drift 보정에 사용. 기기에 없으면 null (대부분의 안드로이드 기기에 존재).
     */
    private val rotVecSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    // ── 최신 센서값 캐시 ─────────────────────────────────────────
    @Volatile private var latestAcc    = FloatArray(3)
    @Volatile private var latestGyr    = FloatArray(3)
    @Volatile private var latestLinAcc = FloatArray(3)   // 없으면 zero 유지
    @Volatile private var latestAccTs  = -1L
    @Volatile private var latestGyrTs  = -1L

    /**
     * TYPE_ROTATION_VECTOR 에서 추출한 최신 yaw.
     *
     * 계산: SensorManager.getRotationMatrixFromVector() → 3×3 행렬 → yaw = atan2(R[1,0], R[0,0])
     * 초기값 Float.NaN — rotVecSensor 가 없거나 아직 수신 전이면 NaN 유지.
     */
    @Volatile private var latestYawRad: Float = Float.NaN

    /**
     * [P53] 최신 TYPE_ROTATION_VECTOR 회전행렬 (device→world, row-major 9-element).
     *
     * RotVec dead-reckoning 의 per-timestep gravity-aligned 프레임 변환에 사용.
     * 자력계 융합 절대 자세 → yaw drift 없음 (EKF clone rotation 과 달리).
     * 초기값 단위행렬 — rotVec 미수신 구간에서는 회전 없음(body frame) 으로 graceful degrade.
     */
    @Volatile private var latestRotMat: FloatArray =
        floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

    /**
     * [P11] TYPE_ROTATION_VECTOR 지자기계 정확도.
     *
     * Android SensorManager 상수:
     *   SENSOR_STATUS_UNRELIABLE  = 0  (자기 캘리브레이션 안됨)
     *   SENSOR_STATUS_ACCURACY_LOW    = 1
     *   SENSOR_STATUS_ACCURACY_MEDIUM = 2
     *   SENSOR_STATUS_ACCURACY_HIGH   = 3
     *
     * 초기값 0(UNRELIABLE): 첫 정확도 콜백 전까지 yaw 보정을 막아 잘못된 주입 방지.
     * onAccuracyChanged 에서 TYPE_ROTATION_VECTOR 의 정확도만 추적.
     */
    @Volatile private var rotVecAccuracy: Int = 0  // 0 = SENSOR_STATUS_UNRELIABLE

    // ── EKF 전파용 큐 (소비 후 비워짐) ──────────────────────────
    private val propagateQueue = ConcurrentLinkedQueue<ImuSample>()

    // ── 추론 윈도우용 링 버퍼 (최근 200 개 유지) ─────────────────
    private val ringBuffer = ArrayBlockingQueue<ImuSample>(BUFFER_SIZE)

    // ── [P70-3] no-preprocess 버퍼 — 모든 전처리 BYPASS ──────────
    //   - 100Hz 리샘플 X (네이티브 콜백마다 즉시 적재 ≈ 200-250Hz on Galaxy)
    //   - P21 영점 보정 X (latestLinAcc/latestGyr 그대로 — bias 미차감)
    //   - LPF X (linAccLpf/gyrLpf 자리에도 raw 동일 값 채움)
    //   - 캘리브 기간 중에도 적재 (calibrating 분기 *앞*에서 push)
    //   학술용: 전처리 작업의 효과를 시각적으로 보여주기 위해
    //   getNoPreprocWindow() / getNoPreprocRotMatWindow() 로 별도 추론 경로 입력.
    //   크기 BUFFER_SIZE*3 = 600 → 네이티브 250Hz 기준 약 2.4 초 보관 (윈도우 1초 충분).
    private val noPreprocBuffer = ArrayBlockingQueue<ImuSample>(BUFFER_SIZE * 3)

    // [P70-5] noPreprocBuffer push 비율 측정 (5초마다 로그) — 의도한 100Hz 확인용
    @Volatile private var noPreprocPushCount: Long = 0
    @Volatile private var noPreprocFirstTsUs: Long = -1L
    @Volatile private var noPreprocLastLogTsUs: Long = -1L

    // [P70-6] noPreprocBuffer 의 100Hz 리샘플 게이트 — ringBuffer 의 lastSampleTs 와 *분리*.
    //   학술 시연: norm OFF + calib OFF + (100Hz 정상) 조합으로 norm 부재 효과를 정량 관찰.
    //   이전 (P70-3..5) 의 네이티브 ≈250Hz 입력은 모델이 100Hz 로 해석 → 윈도우당 400ms
    //   motion 만 담겨 norm OFF ×2.8 증폭이 시간압축 ×0.4 와 상쇄됐음. 본 변경으로 시간
    //   압축 효과 제거 → norm OFF 효과가 그대로 가시화.
    private var lastNoPreprocTs: Long = -1L

    // ── 최신 샘플 빠른 접근 ────────────────────────────────────
    @Volatile private var _latestSample: ImuSample? = null

    // 100Hz 리샘플링 제어
    private var lastSampleTs     = -1L
    private val sampleIntervalUs = 1_000_000L / TARGET_HZ  // 10,000 μs

    // 윈도우 준비 알림 (추론 루프 트리거용)
    private val _windowReady = MutableSharedFlow<Unit>(extraBufferCapacity = 4)
    val windowReady: SharedFlow<Unit> = _windowReady.asSharedFlow()

    // ── [P21-ish] 실시간 영점 보정 상태 ──────────────────────────
    // 시작 시 2 초간 linAcc + gyr 평균을 bias 로 추정.
    // 이후 모든 sample 에 bias 차감을 적용해 큐에 적재한다.
    // acc (중력 포함) 는 bias 추정 제외 — 정지 자세에 따라 다르며 EKF 가 자체 추정.
    @Volatile private var calibrating: Boolean = false
    @Volatile private var calibrationDone: Boolean = false
    @Volatile private var calibProgress: Float = 0f
    private var calibStartElapsedMs: Long = 0L
    private var calibCount: Int = 0
    private val calibLinAccSum: DoubleArray = DoubleArray(3)
    private val calibGyrSum:    DoubleArray = DoubleArray(3)

    private val linAccBias: FloatArray = FloatArray(3)
    private val gyrBias:    FloatArray = FloatArray(3)

    // ── [P33-A1 재도입] LPF state (Direct Form I, 6 채널) ───────
    // 채널 인덱스: 0=linAcc.x, 1=linAcc.y, 2=linAcc.z, 3=gyr.x, 4=gyr.y, 5=gyr.z
    private val lpfX1: FloatArray = FloatArray(6)  // x[n-1]
    private val lpfX2: FloatArray = FloatArray(6)  // x[n-2]
    private val lpfY1: FloatArray = FloatArray(6)  // y[n-1]
    private val lpfY2: FloatArray = FloatArray(6)  // y[n-2]

    /**
     * [P33-A1] Direct Form I 한 step. 채널 state 갱신 후 LPF 출력 반환.
     * y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
     */
    private fun lpfStep(ch: Int, xn: Float): Float {
        val yn = LPF_B0 * xn +
                 LPF_B1 * lpfX1[ch] +
                 LPF_B2 * lpfX2[ch] -
                 LPF_A1 * lpfY1[ch] -
                 LPF_A2 * lpfY2[ch]
        lpfX2[ch] = lpfX1[ch]
        lpfX1[ch] = xn
        lpfY2[ch] = lpfY1[ch]
        lpfY1[ch] = yn
        return yn
    }

    /** [P33-A1] LPF state 전체 0 으로 초기화 (start/stop 시 호출). */
    private fun resetLpfState() {
        for (i in 0..5) {
            lpfX1[i] = 0f; lpfX2[i] = 0f
            lpfY1[i] = 0f; lpfY2[i] = 0f
        }
    }

    // ── [P45-Replay] 재생 thread 상태 ───────────────────────────
    @Volatile private var replayMode: Boolean = false
    @Volatile private var replayThread: Thread? = null
    private val replayStop: AtomicBoolean = AtomicBoolean(false)

    /**
     * @return 현재 인스턴스가 replay 모드 (CSV 재생) 인지 여부.
     */
    fun isReplayMode(): Boolean = replayMode

    // ── 시작 / 종료 ─────────────────────────────────────────────
    /**
     * @param replayMode true 이면 SensorManager 등록을 *건너뛴다*.
     *                   대신 별도 호출되는 [startReplay] 가 CSV 를 읽어
     *                   [processSensorData] 를 직접 호출한다.
     */
    fun start(replayMode: Boolean = false) {
        this.replayMode = replayMode
        replayStop.set(false)

        lastSampleTs  = -1L
        latestAccTs   = -1L
        latestGyrTs   = -1L
        latestLinAcc  = FloatArray(3)
        latestYawRad  = Float.NaN
        latestRotMat  = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)  // [P53] 단위행렬 초기화
        rotVecAccuracy = 0  // 재시작 시 UNRELIABLE 로 초기화
        propagateQueue.clear()
        ringBuffer.clear()
        noPreprocBuffer.clear()                 // [P70-3]
        noPreprocPushCount = 0L                 // [P70-5]
        noPreprocFirstTsUs = -1L                // [P70-5]
        noPreprocLastLogTsUs = -1L              // [P70-5]
        lastNoPreprocTs = -1L                   // [P70-6] 100Hz gate 리셋
        _latestSample = null

        // ── [P21-ish] 캘리브레이션 시작 상태 ──────────────────────
        calibrating         = true
        calibrationDone     = false
        calibProgress       = 0f
        calibStartElapsedMs = 0L
        calibCount          = 0
        for (i in 0..2) {
            calibLinAccSum[i] = 0.0
            calibGyrSum[i]    = 0.0
            linAccBias[i] = 0f
            gyrBias[i]    = 0f
        }
        // [P33-A1] LPF state 초기화
        resetLpfState()

        if (replayMode) {
            Log.i(TAG, "[P45-Replay] CSV 재생 모드 — SensorManager 등록 생략. startReplay() 호출 필요")
            // replay 모드에서도 rotVec 정확도는 CSV 안에서 받지 못하므로
            // 임의로 HIGH(3) 으로 가정 → yaw 보정이 NaN 으로 막히지 않도록.
            rotVecAccuracy = 3
            return
        }

        Log.i(TAG, "[P21] 영점 보정 ${CALIBRATION_DURATION_MS}ms 시작 — 기기 정지 유지 필요")

        sensorManager.registerListener(this, accelSensor,  SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroSensor,   SensorManager.SENSOR_DELAY_FASTEST)
        if (linAccSensor != null) {
            sensorManager.registerListener(this, linAccSensor, SensorManager.SENSOR_DELAY_FASTEST)
            Log.i(TAG, "TYPE_LINEAR_ACCELERATION 등록 완료")
        } else {
            Log.w(TAG, "TYPE_LINEAR_ACCELERATION 없음 — linAcc=0 으로 대체 (네트워크 정확도 저하)")
        }
        if (rotVecSensor != null) {
            // UI 주파수(20Hz)면 충분 — SENSOR_DELAY_GAME ≈ 50Hz
            sensorManager.registerListener(this, rotVecSensor, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "TYPE_ROTATION_VECTOR 등록 완료 — Yaw drift 보정 활성화")
        } else {
            Log.w(TAG, "TYPE_ROTATION_VECTOR 없음 — Yaw drift 보정 비활성화")
        }
        Log.i(TAG, "IMU 수집 시작")
    }

    fun stop() {
        // ── [P45-Replay] CSV 재생 thread 종료 신호 + join ─────────
        if (replayMode) {
            replayStop.set(true)
            try {
                replayThread?.join(500L)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
            replayThread = null
            replayMode = false
        } else {
            sensorManager.unregisterListener(this)
        }
        propagateQueue.clear()
        noPreprocBuffer.clear()                 // [P70-3]

        // [P33-A1] LPF state 리셋
        resetLpfState()

        // [P21-ish] 캘리브레이션 상태 리셋
        calibrating         = false
        calibrationDone     = false
        calibProgress       = 0f
        calibStartElapsedMs = 0L
        calibCount          = 0
        for (i in 0..2) {
            calibLinAccSum[i] = 0.0
            calibGyrSum[i]    = 0.0
            linAccBias[i] = 0f
            gyrBias[i]    = 0f
        }

        Log.i(TAG, "IMU 수집 종료")
    }

    // ── SensorEventListener ──────────────────────────────────────
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // [P11] TYPE_ROTATION_VECTOR 정확도 추적 — 지자기계 품질 게이팅에 사용
        if (sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            rotVecAccuracy = accuracy
            Log.d(TAG, "RotVec 정확도 변경: $accuracy " +
                  when (accuracy) {
                      0 -> "(UNRELIABLE)"
                      1 -> "(LOW)"
                      2 -> "(MEDIUM)"
                      3 -> "(HIGH)"
                      else -> "(?)"
                  })
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        processSensorData(event.sensor.type, event.timestamp / 1000L, event.values)
    }

    /**
     * [P45-Replay] 센서 1 샘플 처리의 *본질 로직* (실시간 / CSV 재생 공용).
     *
     * 실시간 모드: onSensorChanged(event) → 본 함수.
     * Replay  모드: startReplay(csv) → CSV 1 라인 → 본 함수.
     *
     * @param sensorType Sensor.TYPE_ACCELEROMETER / GYROSCOPE / LINEAR_ACCELERATION / ROTATION_VECTOR
     * @param tsUs       해당 샘플 타임스탬프 (μs, monotonic — boot time 기준)
     * @param values     센서 값 배열 — rotVec 만 size>=4 (스칼라 w 포함), 그 외 size=3
     */
    fun processSensorData(sensorType: Int, tsUs: Long, values: FloatArray) {
        when (sensorType) {
            Sensor.TYPE_ACCELEROMETER       -> { latestAcc    = values.copyOf(3); latestAccTs = tsUs }
            Sensor.TYPE_GYROSCOPE           -> { latestGyr    = values.copyOf(3); latestGyrTs = tsUs }
            Sensor.TYPE_LINEAR_ACCELERATION -> { latestLinAcc = values.copyOf(3) }
            // linAcc 는 별도 타임스탬프 동기 불요: TYPE_ACCELEROMETER 와 동일 타임베이스에서
            // 파생되므로 최신값 사용으로 충분
            Sensor.TYPE_ROTATION_VECTOR -> {
                // 3×3 회전 행렬(row-major, device→world) 로 변환
                val rotMat = FloatArray(9)
                // SensorManager.getRotationMatrixFromVector 는 rotation vector 의 3 또는 4 원소 형태 지원.
                // CSV 에서 4 원소로 들어올 수 있으므로 그대로 전달.
                android.hardware.SensorManager.getRotationMatrixFromVector(rotMat, values)
                // yaw = atan2(R[1,0], R[0,0])  (EKF 와 동일한 ZYX Euler yaw 공식)
                //   rotMat[row*3+col] → R[1,0]=rotMat[3], R[0,0]=rotMat[0]
                latestYawRad = Math.atan2(rotMat[3].toDouble(), rotMat[0].toDouble()).toFloat()
                // [P53] RotVec dead-reckoning 용 전체 회전행렬 보관 (device→world).
                latestRotMat = rotMat
            }
        }

        // 가속도 + 자이로 모두 수신된 이후에만 샘플링
        if (latestAccTs < 0 || latestGyrTs < 0) return

        // ── [P70-6] no-preprocess 버퍼 적재 — TYPE_ACC + 100Hz 게이트 ──
        //   처음 P70-3 은 모든 sensorType 콜백마다 push (3× 중복 버그).
        //   P70-5 에서 TYPE_ACCELEROMETER 만 통과시켜 ≈250Hz 네이티브 rate 확보.
        //   P70-6 에서 *100Hz 리샘플만 복구* — norm/calib OFF 는 유지.
        //     ⮡ 이전 시간압축 효과 (×0.4) 가 norm OFF 증폭 (×2.8) 을 상쇄해서
        //        궤적이 오히려 작아지던 현상 제거 → norm OFF 효과만 정상 가시화.
        //   linAcc/gyr 는 여전히 RAW (bias 미차감) 유지 — calib 효과 분리 관찰.
        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            // [P70-6] 100Hz 리샘플 — 마지막 push 로부터 10ms (sampleIntervalUs) 경과 시에만.
            //   ringBuffer 의 lastSampleTs 와는 독립 변수 (gate 시점이 다름).
            val shouldPush = if (lastNoPreprocTs < 0L) {
                lastNoPreprocTs = tsUs
                true
            } else if (tsUs - lastNoPreprocTs >= sampleIntervalUs) {
                lastNoPreprocTs = tsUs
                true
            } else {
                false
            }

            if (shouldPush) {
                val rawSample = ImuSample(
                    ts_us      = tsUs,
                    acc        = latestAcc.clone(),
                    gyr        = latestGyr.clone(),       // RAW (bias 미차감)
                    linAcc     = latestLinAcc.clone(),    // RAW (bias 미차감)
                    linAccLpf  = latestLinAcc.clone(),    // LPF 자리에 raw 동일 값 (LPF 미적용)
                    gyrLpf     = latestGyr.clone(),       // 동일
                    rotMat     = latestRotMat
                )
                while (noPreprocBuffer.size >= BUFFER_SIZE * 3) noPreprocBuffer.poll()
                noPreprocBuffer.offer(rawSample)

                // [P70-5] push 비율 측정 — 5초마다 실제 Hz 로그 (P70-6 후 ≈100Hz 기대)
                noPreprocPushCount++
                if (noPreprocFirstTsUs < 0L) {
                    noPreprocFirstTsUs = tsUs
                    noPreprocLastLogTsUs = tsUs
                } else if (tsUs - noPreprocLastLogTsUs > 5_000_000L) {  // 5 초
                    val elapsedSec = (tsUs - noPreprocFirstTsUs) / 1_000_000.0
                    val hz = noPreprocPushCount / elapsedSec
                    Log.i(TAG, "[P70-6] noPreprocBuffer push rate = ${"%.1f".format(hz)} Hz " +
                            "(count=$noPreprocPushCount, elapsed=${"%.1f".format(elapsedSec)}s, " +
                            "buffer size=${noPreprocBuffer.size})  ← 100Hz 기대")
                    noPreprocLastLogTsUs = tsUs
                }
            }
        }

        // ── [P21-ish] 캘리브레이션 분기 ───────────────────────────
        // 시작 후 CALIBRATION_DURATION_MS 동안 linAcc + gyr 평균을 누적해
        // bias 를 추정. 이 기간에는 propagateQueue / ringBuffer 미적재
        // → EkfBridge 초기화·추론 모두 자연스럽게 지연됨.
        if (calibrating) {
            if (calibStartElapsedMs == 0L) {
                calibStartElapsedMs = android.os.SystemClock.elapsedRealtime()
            }
            calibLinAccSum[0] += latestLinAcc[0].toDouble()
            calibLinAccSum[1] += latestLinAcc[1].toDouble()
            calibLinAccSum[2] += latestLinAcc[2].toDouble()
            calibGyrSum[0]    += latestGyr[0].toDouble()
            calibGyrSum[1]    += latestGyr[1].toDouble()
            calibGyrSum[2]    += latestGyr[2].toDouble()
            calibCount++

            val elapsed = android.os.SystemClock.elapsedRealtime() - calibStartElapsedMs
            calibProgress = (elapsed.toFloat() / CALIBRATION_DURATION_MS.toFloat()).coerceIn(0f, 1f)

            if (elapsed >= CALIBRATION_DURATION_MS) {
                performWarmup()
            }
            return
        }

        // 100Hz 리샘플링
        if (lastSampleTs < 0) lastSampleTs = tsUs
        if (tsUs - lastSampleTs < sampleIntervalUs) return
        lastSampleTs = tsUs

        // [P21-ish] linAcc + gyr 에 bias 차감 적용. acc 는 그대로 유지 (EKF 자체 추정).
        val linAccCorr = FloatArray(3)
        linAccCorr[0] = latestLinAcc[0] - linAccBias[0]
        linAccCorr[1] = latestLinAcc[1] - linAccBias[1]
        linAccCorr[2] = latestLinAcc[2] - linAccBias[2]
        val gyrCorr = FloatArray(3)
        gyrCorr[0] = latestGyr[0] - gyrBias[0]
        gyrCorr[1] = latestGyr[1] - gyrBias[1]
        gyrCorr[2] = latestGyr[2] - gyrBias[2]

        // [P33-A1] LPF 적용 (bias 차감된 값에 대해, 네트워크 입력용)
        // 채널 인덱스: 0..2 = linAcc x/y/z, 3..5 = gyr x/y/z
        val linAccLpf = FloatArray(3)
        linAccLpf[0] = lpfStep(0, linAccCorr[0])
        linAccLpf[1] = lpfStep(1, linAccCorr[1])
        linAccLpf[2] = lpfStep(2, linAccCorr[2])
        val gyrLpf = FloatArray(3)
        gyrLpf[0] = lpfStep(3, gyrCorr[0])
        gyrLpf[1] = lpfStep(4, gyrCorr[1])
        gyrLpf[2] = lpfStep(5, gyrCorr[2])

        val sample = ImuSample(
            ts_us      = tsUs,
            acc        = latestAcc.clone(),  // RAW (gravity 포함) → EKF propagate
            gyr        = gyrCorr,            // RAW bias 차감 후 → EKF propagate
            linAcc     = linAccCorr,         // RAW bias 차감 후 → 진단용
            linAccLpf  = linAccLpf,          // LPF → 네트워크 입력 (getWindow)
            gyrLpf     = gyrLpf,             // LPF → 네트워크 입력 (getWindow)
            rotMat     = latestRotMat        // [P53] 샘플 시점 회전행렬 → RotVec DR
        )

        // ① EKF 전파 큐 (500 개 초과 시 오래된 것 제거)
        propagateQueue.offer(sample)
        while (propagateQueue.size > 500) propagateQueue.poll()

        // ② 윈도우 링 버퍼
        while (ringBuffer.size >= BUFFER_SIZE) ringBuffer.poll()
        ringBuffer.offer(sample)

        // ③ 최신 샘플 캐시
        _latestSample = sample

        // 윈도우가 가득 차면 알림
        if (ringBuffer.size >= WINDOW_SIZE) _windowReady.tryEmit(Unit)
    }

    // ── EKF 전파용 ───────────────────────────────────────────────
    /**
     * propagateQueue 에 쌓인 모든 샘플을 반환하고 큐를 비운다.
     * EKF 전파에는 sample.acc (중력 포함) 와 sample.gyr 를 사용할 것.
     */
    fun drainPropagateQueue(): List<ImuSample> {
        if (propagateQueue.isEmpty()) return emptyList()
        val result = ArrayList<ImuSample>(16)
        var s = propagateQueue.poll()
        while (s != null) {
            result.add(s)
            s = propagateQueue.poll()
        }
        return result
    }

    // ── 추론용 ───────────────────────────────────────────────────
    /**
     * 최근 WINDOW_SIZE 샘플을 channel-major FloatArray(600) 로 반환.
     *
     * 채널 레이아웃 (Python 학습 데이터와 동일한 형식):
     *   ch 0-2 : linAcc  body frame (중력 없음) — LocalizationViewModel 에서 world frame 으로 회전됨
     *   ch 3-5 : gyr     body frame             — 동일하게 회전됨
     *
     * @return Pair(FloatArray[600], 윈도우_첫_샘플_ts_us) or null
     */
    fun getWindow(): Pair<FloatArray, Long>? {
        val snap = ringBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(CHANNEL_NUM * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            // [P33-A1] 네트워크 입력은 LPF 버전 사용 (iPhone Core Motion 흉내)
            // propagate 는 다른 경로 (drainPropagateQueue → s.acc / s.gyr RAW) 이므로 영향 없음
            flat[0 * WINDOW_SIZE + t] = s.linAccLpf[0]   // ch 0: linear acc x (LPF)
            flat[1 * WINDOW_SIZE + t] = s.linAccLpf[1]   // ch 1: linear acc y (LPF)
            flat[2 * WINDOW_SIZE + t] = s.linAccLpf[2]   // ch 2: linear acc z (LPF)
            flat[3 * WINDOW_SIZE + t] = s.gyrLpf[0]      // ch 3: gyr x (LPF)
            flat[4 * WINDOW_SIZE + t] = s.gyrLpf[1]      // ch 4: gyr y (LPF)
            flat[5 * WINDOW_SIZE + t] = s.gyrLpf[2]      // ch 5: gyr z (LPF)
        }
        return Pair(flat, recent.first().ts_us)
    }

    /**
     * [P41] raw gyr (P21 bias 차감 후, LPF 미적용) 만 channel-major FloatArray(300) 로 반환.
     *
     * R_all[t] 자이로 적분 처리용 — Python `_get_imu_samples_for_network` 의 `net_gyr` 와 동등.
     * getWindow() 는 LPF 적용된 gyrLpf 반환 (네트워크 입력용). 자이로 적분에는 LPF 가
     * 위상 지연 + 진폭 감쇠로 누적 오차 야기하므로 *raw (P21 후, LPF 전)* 사용이 정확.
     *
     * 주의: 여기 반환되는 gyr 는 *이미 P21 정적 bias 차감*된 상태. R_all[t] 사용 시
     * EKF s_bg 추가 차감 *금지* (이중 차감 발산 야기, P41 세션에서 검증됨).
     *
     * 채널: ch 0 gx, ch 1 gy, ch 2 gz (rad/s)
     * @return FloatArray[3 × WINDOW_SIZE] or null (윈도우 미충족 시)
     */
    fun getRawGyrWindow(): FloatArray? {
        val snap = ringBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(3 * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            flat[0 * WINDOW_SIZE + t] = s.gyr[0]
            flat[1 * WINDOW_SIZE + t] = s.gyr[1]
            flat[2 * WINDOW_SIZE + t] = s.gyr[2]
        }
        return flat
    }

    /**
     * [P53] 최근 WINDOW_SIZE 샘플을 LPF 미적용 raw 6채널 window 로 반환.
     *
     * getWindow() 는 12Hz Butterworth LPF 버전(linAccLpf/gyrLpf)을 반환하나, RotVec
     * dead-reckoning 은 오프라인 하니스(offline_eval.py)와 입력 분포를 정합시키기 위해
     * raw linAcc/gyr 를 사용한다 — 하니스가 검증한 입력 = CSV raw linAcc/gyr.
     * 둘 다 P21 정적 bias 차감은 완료된 상태.
     *
     * 채널: ch0-2 linAcc(raw), ch3-5 gyr(raw).
     * @return FloatArray[6 × WINDOW_SIZE] or null (윈도우 미충족 시)
     */
    fun getRawWindow(): FloatArray? {
        val snap = ringBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(CHANNEL_NUM * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            flat[0 * WINDOW_SIZE + t] = s.linAcc[0]
            flat[1 * WINDOW_SIZE + t] = s.linAcc[1]
            flat[2 * WINDOW_SIZE + t] = s.linAcc[2]
            flat[3 * WINDOW_SIZE + t] = s.gyr[0]
            flat[4 * WINDOW_SIZE + t] = s.gyr[1]
            flat[5 * WINDOW_SIZE + t] = s.gyr[2]
        }
        return flat
    }

    /**
     * [P70-3] no-preprocess 버퍼에서 최근 WINDOW_SIZE 샘플을 raw 6채널 window 로 반환.
     *
     * getRawWindow() 와의 차이:
     *   - 100Hz 리샘플 *없음* — 네이티브 센서 콜백 그대로 (Galaxy 약 200-250Hz 추정).
     *     → 모델 입력 분포가 학습 (100Hz) 와 어긋남 → 가시적 성능 저하 기대.
     *   - P21 영점 보정 *없음* — bias 차감 미적용 → 정지 시 비영점 출력.
     *   - LPF *없음* — 어차피 getRawWindow 도 LPF 미적용이지만, 본 함수는 ImuSample.linAcc
     *     자체가 RAW (bias 미차감) 라는 차이.
     *
     * 채널: ch0-2 linAcc(raw), ch3-5 gyr(raw).  레이아웃: channel-major (학습 동일).
     * @return FloatArray[6 × WINDOW_SIZE] or null (윈도우 미충족 시)
     */
    fun getNoPreprocWindow(): FloatArray? {
        val snap = noPreprocBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(CHANNEL_NUM * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            flat[0 * WINDOW_SIZE + t] = s.linAcc[0]
            flat[1 * WINDOW_SIZE + t] = s.linAcc[1]
            flat[2 * WINDOW_SIZE + t] = s.linAcc[2]
            flat[3 * WINDOW_SIZE + t] = s.gyr[0]
            flat[4 * WINDOW_SIZE + t] = s.gyr[1]
            flat[5 * WINDOW_SIZE + t] = s.gyr[2]
        }
        return flat
    }

    /**
     * [P70-3] no-preprocess 버퍼의 동일 100 샘플에 대응하는 rotMat window 반환.
     * gravity-aligned 변환에 사용 (모델은 world frame 입력을 기대 — 좌표변환 단계는 유지).
     */
    fun getNoPreprocRotMatWindow(): FloatArray? {
        val snap = noPreprocBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(9 * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            System.arraycopy(s.rotMat, 0, flat, t * 9, 9)
        }
        return flat
    }

    /**
     * [P70-3] no-preprocess 버퍼의 가장 최근 샘플 (timestamp 참조용).
     */
    fun getNoPreprocLatestSample(): ImuSample? = noPreprocBuffer.peek()?.let {
        // ArrayBlockingQueue.peek 은 head (가장 오래된) 반환 — 가장 최근은 toArray().last
        val snap = noPreprocBuffer.toArray().filterIsInstance<ImuSample>()
        snap.lastOrNull()
    }

    /**
     * [P53] 최근 WINDOW_SIZE 샘플의 회전행렬을 channel-major FloatArray(9 × WINDOW_SIZE) 로 반환.
     *
     * RotVec dead-reckoning 의 per-timestep gravity-aligned 변환용. getWindow() 와 *동일한*
     * takeLast(WINDOW_SIZE) 스냅샷 규칙 → 두 호출 사이 1-2 샘플 어긋남은 무시 가능 (10-20ms).
     *
     * 레이아웃: rotMat[t] 는 flat[t*9 .. t*9+8] (row-major device→world).
     * @return FloatArray[9 × WINDOW_SIZE] or null (윈도우 미충족 시)
     */
    fun getRotMatWindow(): FloatArray? {
        val snap = ringBuffer.toArray().filterIsInstance<ImuSample>()
        if (snap.size < WINDOW_SIZE) return null
        val recent = snap.takeLast(WINDOW_SIZE)
        val flat = FloatArray(9 * WINDOW_SIZE)
        for ((t, s) in recent.withIndex()) {
            System.arraycopy(s.rotMat, 0, flat, t * 9, 9)
        }
        return flat
    }

    /** 가장 최근 샘플 (EKF 초기화, 타임스탬프 참조용) */
    fun getLatestSample(): ImuSample? = _latestSample

    /**
     * TYPE_ROTATION_VECTOR 에서 추출한 최신 yaw (rad).
     *
     * EKF 와 동일한 공식: atan2(R[1,0], R[0,0])  (ZYX Euler, 수학 양방향 반시계 양수)
     * rotVecSensor 가 없거나 아직 수신 전이면 Float.NaN 반환.
     */
    fun getLatestYawRad(): Float = latestYawRad

    /**
     * [P11] TYPE_ROTATION_VECTOR 지자기계 정확도 반환.
     *
     * 0=UNRELIABLE, 1=LOW, 2=MEDIUM, 3=HIGH.
     * 초기값 0 — rotVecSensor 가 없거나 아직 onAccuracyChanged 수신 전.
     * 게이팅 기준: 2(MEDIUM) 이상일 때만 yaw 보정을 허용.
     */
    fun getRotVecAccuracy(): Int = rotVecAccuracy

    // ──────────────────────────────────────────────────────────────
    // [P21-ish] 실시간 영점 보정 공개 API
    // ──────────────────────────────────────────────────────────────

    /** 현재 캘리브레이션 진행 중인지 여부 */
    fun isCalibrating(): Boolean = calibrating

    /** 캘리브레이션 완료 여부 (한 번이라도 끝났는지) */
    fun isCalibrationDone(): Boolean = calibrationDone

    /** 캘리브레이션 진행률 (0.0 ~ 1.0). UI 게이팅용. */
    fun getCalibrationProgress(): Float = calibProgress

    /** 추정된 (linAccBias, gyrBias) 스냅샷 — 진단/로깅용 */
    fun getBiasSnapshot(): Pair<FloatArray, FloatArray> =
        Pair(linAccBias.copyOf(), gyrBias.copyOf())

    /**
     * 캘리브레이션 누적 종료 — 평균을 bias 로 확정하고
     * calibrating=false / calibrationDone=true 로 상태 전환.
     * 호출 직후 다음 sample 부터 정상 수집·bias 차감 적용 시작.
     */
    private fun performWarmup() {
        if (calibCount <= 0) {
            Log.w(TAG, "[P21] 캘리브레이션 누적 샘플 0 개 — bias 를 0 으로 유지")
        } else {
            val n = calibCount.toDouble()
            linAccBias[0] = (calibLinAccSum[0] / n).toFloat()
            linAccBias[1] = (calibLinAccSum[1] / n).toFloat()
            linAccBias[2] = (calibLinAccSum[2] / n).toFloat()
            gyrBias[0]    = (calibGyrSum[0]    / n).toFloat()
            gyrBias[1]    = (calibGyrSum[1]    / n).toFloat()
            gyrBias[2]    = (calibGyrSum[2]    / n).toFloat()

            Log.i(TAG,
                "[P21] 캘리브레이션 완료 (n=$calibCount, " +
                "elapsed=${android.os.SystemClock.elapsedRealtime() - calibStartElapsedMs}ms)" +
                "\n  linAccBias = [${"%.4f".format(linAccBias[0])}, ${"%.4f".format(linAccBias[1])}, ${"%.4f".format(linAccBias[2])}] m/s²" +
                "\n  gyrBias    = [${"%.5f".format(gyrBias[0])}, ${"%.5f".format(gyrBias[1])}, ${"%.5f".format(gyrBias[2])}] rad/s"
            )
        }

        calibrating     = false
        calibrationDone = true
        calibProgress   = 1f
        // 캘리브레이션 종료 직후 첫 sample 이 곧바로 흐르도록 lastSampleTs 재초기화
        lastSampleTs = -1L
    }

    // ──────────────────────────────────────────────────────────────
    // [P45-Replay] CSV 재생 (ImuTestActivity 기록 CSV → processSensorData)
    // ──────────────────────────────────────────────────────────────

    /**
     * 별도 thread 가 CSV 를 *기록 당시 타임스탬프 간격* 으로 재생한다.
     *
     * 전제: [start] 가 `replayMode=true` 로 먼저 호출되어 SensorManager 미등록 + 캘리브 상태로 진입.
     * 본 함수가 그 위에 CSV 를 흘려보내 [processSensorData] 를 호출.
     *
     * CSV 형식 (ImuTestActivity 기록):
     * ```
     * sensor,ts_ns,x,y,z,w
     * acc,1234567890,0.012,-9.811,0.034,0.0
     * gyr,1234567892,0.0011,-0.0023,0.0008,0.0
     * linAcc,1234567893,0.011,-0.001,0.024,0.0
     * rotVec,1234567895,0.012,0.034,-0.067,0.997
     * ```
     *
     * @param csvFile     재생할 CSV. 헤더 1 줄 + ts 오름차순 데이터 라인.
     * @param speedFactor 재생 속도 배수 (1.0=실시간, 2.0=2배속, 0.5=절반속). 디버깅 편의.
     * @param onFinished  CSV 끝 또는 [stop] 호출 시 main thread 와 무관하게 호출되는 콜백.
     */
    fun startReplay(
        csvFile: File,
        speedFactor: Double = 1.0,
        onFinished: () -> Unit = {}
    ) {
        if (!replayMode) {
            Log.e(TAG, "[P45-Replay] start(replayMode=true) 호출 전에 startReplay 호출됨 — 무시")
            return
        }
        if (!csvFile.exists()) {
            Log.e(TAG, "[P45-Replay] CSV 파일 없음: ${csvFile.absolutePath}")
            onFinished()
            return
        }

        val thread = Thread({
            var lineCount = 0L
            var firstTsNs = -1L
            val tStartElapsedMs = android.os.SystemClock.elapsedRealtime()

            try {
                BufferedReader(FileReader(csvFile)).use { br ->
                    val header = br.readLine()  // "sensor,ts_ns,x,y,z,w"
                    if (header == null) {
                        Log.e(TAG, "[P45-Replay] CSV 비어있음: ${csvFile.name}")
                        return@use
                    }

                    var line = br.readLine()
                    while (line != null && !replayStop.get()) {
                        val parts = line.split(',')
                        if (parts.size < 5) { line = br.readLine(); continue }

                        val sensorTag = parts[0]
                        val tsNsParsed = parts[1].toLongOrNull()
                        if (tsNsParsed == null) { line = br.readLine(); continue }
                        val tsNs      = tsNsParsed
                        val x         = parts[2].toFloatOrNull() ?: 0f
                        val y         = parts[3].toFloatOrNull() ?: 0f
                        val z         = parts[4].toFloatOrNull() ?: 0f
                        val w         = if (parts.size >= 6) parts[5].toFloatOrNull() ?: 0f else 0f

                        // sensor 문자열 → Sensor.TYPE_* 매핑
                        val sensorType = when (sensorTag) {
                            "acc"    -> Sensor.TYPE_ACCELEROMETER
                            "gyr"    -> Sensor.TYPE_GYROSCOPE
                            "linAcc" -> Sensor.TYPE_LINEAR_ACCELERATION
                            "rotVec" -> Sensor.TYPE_ROTATION_VECTOR
                            else     -> { line = br.readLine(); continue }
                        }

                        // 첫 라인의 ts 를 wall clock 기준점으로 사용
                        if (firstTsNs < 0L) firstTsNs = tsNs

                        // 기록 시점 경과 (ms) = (현재 라인 ts - 첫 ts) / 1e6 / speedFactor
                        val csvElapsedMs = ((tsNs - firstTsNs).toDouble() / 1_000_000.0 / speedFactor).toLong()
                        val wallElapsedMs = android.os.SystemClock.elapsedRealtime() - tStartElapsedMs
                        val sleepMs = csvElapsedMs - wallElapsedMs
                        if (sleepMs > 1L) {
                            try { Thread.sleep(sleepMs) }
                            catch (e: InterruptedException) {
                                Thread.currentThread().interrupt()
                                break
                            }
                        }

                        // 센서 값 배열 — rotVec 만 4 원소 (w 포함), 그 외 3 원소
                        val values = if (sensorType == Sensor.TYPE_ROTATION_VECTOR) {
                            floatArrayOf(x, y, z, w)
                        } else {
                            floatArrayOf(x, y, z)
                        }

                        // tsUs 는 CSV 의 ts_ns 를 그대로 변환 (boot-time monotonic 가정)
                        processSensorData(sensorType, tsNs / 1000L, values)

                        lineCount++
                        if (lineCount % 1000L == 0L) {
                            Log.i(TAG, "[P45-Replay] 진행 ${lineCount} 라인 (csv ${csvElapsedMs}ms / wall ${wallElapsedMs}ms)")
                        }

                        line = br.readLine()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "[P45-Replay] 재생 중 오류: ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                val totalMs = android.os.SystemClock.elapsedRealtime() - tStartElapsedMs
                Log.i(TAG, "[P45-Replay] 재생 종료 — 처리 ${lineCount} 라인, wall ${totalMs}ms, stopFlag=${replayStop.get()}")
                onFinished()
            }
        }, "ImuReplayThread")

        replayThread = thread
        thread.isDaemon = true
        thread.start()
        Log.i(TAG, "[P45-Replay] 재생 thread 시작: ${csvFile.name} (speed=${speedFactor}x)")
    }
}
