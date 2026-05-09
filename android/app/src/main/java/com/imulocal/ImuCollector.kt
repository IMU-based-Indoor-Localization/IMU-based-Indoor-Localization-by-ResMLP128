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
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentLinkedQueue

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
    }

    /**
     * @param acc    TYPE_ACCELEROMETER [ax, ay, az] m/s²  (중력 포함 — EKF 전파용)
     * @param gyr    TYPE_GYROSCOPE     [gx, gy, gz] rad/s
     * @param linAcc TYPE_LINEAR_ACCELERATION [lx, ly, lz] m/s² (중력 제거 — 네트워크 입력용)
     */
    data class ImuSample(
        val ts_us:  Long,
        val acc:    FloatArray,   // [ax, ay, az]  m/s²  (includes gravity) — EKF용
        val gyr:    FloatArray,   // [gx, gy, gz]  rad/s
        val linAcc: FloatArray    // [lx, ly, lz]  m/s²  (no gravity)      — 네트워크용
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

    // ── 최신 샘플 빠른 접근 ────────────────────────────────────
    @Volatile private var _latestSample: ImuSample? = null

    // 100Hz 리샘플링 제어
    private var lastSampleTs     = -1L
    private val sampleIntervalUs = 1_000_000L / TARGET_HZ  // 10,000 μs

    // 윈도우 준비 알림 (추론 루프 트리거용)
    private val _windowReady = MutableSharedFlow<Unit>(extraBufferCapacity = 4)
    val windowReady: SharedFlow<Unit> = _windowReady.asSharedFlow()

    // ── 시작 / 종료 ─────────────────────────────────────────────
    fun start() {
        lastSampleTs  = -1L
        latestAccTs   = -1L
        latestGyrTs   = -1L
        latestLinAcc  = FloatArray(3)
        latestYawRad  = Float.NaN
        rotVecAccuracy = 0  // 재시작 시 UNRELIABLE 로 초기화
        propagateQueue.clear()
        ringBuffer.clear()
        _latestSample = null

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
        sensorManager.unregisterListener(this)
        propagateQueue.clear()
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
        val tsUs = event.timestamp / 1000L   // ns → μs

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER       -> { latestAcc    = event.values.clone(); latestAccTs = tsUs }
            Sensor.TYPE_GYROSCOPE           -> { latestGyr    = event.values.clone(); latestGyrTs = tsUs }
            Sensor.TYPE_LINEAR_ACCELERATION -> { latestLinAcc = event.values.clone() }
            // linAcc 는 별도 타임스탬프 동기 불요: TYPE_ACCELEROMETER 와 동일 타임베이스에서
            // 파생되므로 최신값 사용으로 충분
            Sensor.TYPE_ROTATION_VECTOR -> {
                // 3×3 회전 행렬(row-major, device→world) 로 변환
                val rotMat = FloatArray(9)
                android.hardware.SensorManager.getRotationMatrixFromVector(rotMat, event.values)
                // yaw = atan2(R[1,0], R[0,0])  (EKF 와 동일한 ZYX Euler yaw 공식)
                //   rotMat[row*3+col] → R[1,0]=rotMat[3], R[0,0]=rotMat[0]
                latestYawRad = Math.atan2(rotMat[3].toDouble(), rotMat[0].toDouble()).toFloat()
            }
        }

        // 가속도 + 자이로 모두 수신된 이후에만 샘플링
        if (latestAccTs < 0 || latestGyrTs < 0) return

        // 100Hz 리샘플링
        if (lastSampleTs < 0) lastSampleTs = tsUs
        if (tsUs - lastSampleTs < sampleIntervalUs) return
        lastSampleTs = tsUs

        val sample = ImuSample(
            ts_us  = tsUs,
            acc    = latestAcc.clone(),
            gyr    = latestGyr.clone(),
            linAcc = latestLinAcc.clone()
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
            flat[0 * WINDOW_SIZE + t] = s.linAcc[0]   // ch 0: linear acc x
            flat[1 * WINDOW_SIZE + t] = s.linAcc[1]   // ch 1: linear acc y
            flat[2 * WINDOW_SIZE + t] = s.linAcc[2]   // ch 2: linear acc z
            flat[3 * WINDOW_SIZE + t] = s.gyr[0]      // ch 3: gyr x
            flat[4 * WINDOW_SIZE + t] = s.gyr[1]      // ch 4: gyr y
            flat[5 * WINDOW_SIZE + t] = s.gyr[2]      // ch 5: gyr z
        }
        return Pair(flat, recent.first().ts_us)
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
}
