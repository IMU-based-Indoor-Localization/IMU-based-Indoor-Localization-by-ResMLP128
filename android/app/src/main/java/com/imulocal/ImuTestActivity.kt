package com.imulocal

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MenuItem
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.IOException
import kotlin.math.sqrt

/**
 * ImuTestActivity.kt
 * ==================
 * 가속도계 + 자이로스코프 원시값을 실시간으로 화면에 표시한다.
 * 모델 파일이나 EKF 없이 센서 수신 여부만 독립적으로 검증한다.
 *
 * [P40 CSV 기록 기능 추가]
 *   목적: OxIOD raw (iPhone, Apple Core Motion) 와 Android raw IMU 의 통계 비교용 데이터셋 확보.
 *   - 시작 시 4종 센서 (acc / gyr / linAcc / rotVec) 를 모두 등록.
 *   - 매 SensorEvent 마다 long-format CSV 한 줄 기록 (sensor,ts_ns,x,y,z,w).
 *   - 정지 시 파일 절대경로 + 라인 수를 Toast 로 안내.
 *   - 저장 위치: getExternalFilesDir(null)/imu_csv/imu_record_<epoch_ms>.csv
 *   - adb pull 명령:
 *       adb pull /sdcard/Android/data/com.imulocal/files/imu_csv/ D:\imu_csv_android\
 *
 *   CSV 형식 (long, pandas pivot 으로 wide 변환 용이):
 *       sensor,ts_ns,x,y,z,w
 *       acc,1234567890,0.012,-9.811,0.034,0.0
 *       gyr,1234567892,0.0011,-0.0023,0.0008,0.0
 *       linAcc,1234567893,0.011,-0.001,0.024,0.0
 *       rotVec,1234567895,0.012,0.034,-0.067,0.997
 */
class ImuTestActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor:  Sensor? = null
    // [P40 CSV] OxIOD raw 와 채널 매칭을 위해 LinearAcceleration / RotationVector 도 함께 수집.
    //          EKF/InferenceEngine 동작과는 완전히 독립 — 본 Activity 의 CSV 기록 전용.
    private var linAccSensor: Sensor? = null
    private var rotVecSensor: Sensor? = null

    // 현재값
    private val accVal = FloatArray(3)
    private val gyrVal = FloatArray(3)

    // 샘플링 속도 계산
    private var sampleCount  = 0L
    private var lastRateCalcMs = 0L
    private var currentHz   = 0.0

    // 윈도우(100샘플) 도달 여부
    private var totalSamples = 0L

    // [P40 CSV] 기록 상태 — startCollection 에서 열고 stopCollection 에서 닫는다.
    private var csvWriter: BufferedWriter? = null
    private var csvFile:   File? = null
    private var csvLineCount = 0L

    private val handler = Handler(Looper.getMainLooper())
    private var running  = false

    // ── 뷰 참조 ──────────────────────────────────────────────────
    private lateinit var tvStatus:       TextView
    private lateinit var tvAcc:          TextView
    private lateinit var tvAccNorm:      TextView
    private lateinit var tvGyr:          TextView
    private lateinit var tvGyrNorm:      TextView
    private lateinit var tvSampleRate:   TextView
    private lateinit var tvSampleCount:  TextView
    private lateinit var tvWindowReady:  TextView
    // [P58] 분류기(휴대 방식) 표시 — MainActivity 측위 ViewModel state 구독.
    private lateinit var tvCarryMode:    TextView
    private lateinit var btnStart:       MaterialButton
    private lateinit var btnStop:        MaterialButton

    // ── UI 갱신 Runnable (200ms 주기) ────────────────────────────
    private val uiUpdater = object : Runnable {
        override fun run() {
            if (!running) return
            refreshUi()
            handler.postDelayed(this, 200)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_imu_test)

        // 뒤로가기 버튼 활성화
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "IMU 센서 진단"

        tvStatus      = findViewById(R.id.tvStatus)
        tvAcc         = findViewById(R.id.tvAcc)
        tvAccNorm     = findViewById(R.id.tvAccNorm)
        tvGyr         = findViewById(R.id.tvGyr)
        tvGyrNorm     = findViewById(R.id.tvGyrNorm)
        tvSampleRate  = findViewById(R.id.tvSampleRate)
        tvSampleCount = findViewById(R.id.tvSampleCount)
        tvWindowReady = findViewById(R.id.tvWindowReady)
        tvCarryMode   = findViewById(R.id.tvCarryMode)
        btnStart      = findViewById(R.id.btnTestStart)
        btnStop       = findViewById(R.id.btnTestStop)

        // [P58] MainActivity 측위 ViewModel 의 state 를 구독해 분류기 출력 표시.
        //  - 측위가 켜지지 않았으면 sharedInstance 가 null 이거나 isRunning=false →
        //    "측위 미실행" 안내 유지.
        //  - Activity 가 STARTED 상태일 때만 collect (메모리 누수 방지).
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                val vm = LocalizationViewModel.sharedInstance
                if (vm == null) {
                    tvCarryMode.text = "측위 미실행 — 메인 화면에서 시작하세요"
                    return@repeatOnLifecycle
                }
                vm.state.collectLatest { s ->
                    tvCarryMode.text = if (s.isRunning) {
                        String.format(
                            "%s  (%d%%)",
                            s.carryMode,
                            (s.carryProb * 100).toInt()
                        )
                    } else {
                        "측위 미실행 — 메인 화면에서 시작하세요"
                    }
                }
            }
        }

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelSensor   = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor    = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        // [P40 CSV] 추가 센서 (OxIOD raw 와 동일 채널 매핑 — Apple Core Motion 의 linAcc/rotVec 대응)
        linAccSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        rotVecSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        // 센서 없을 때 경고
        if (accelSensor == null) tvStatus.text = "⚠ 가속도계 없음"
        if (gyroSensor  == null) tvStatus.text = "⚠ 자이로스코프 없음"
        // [P40 CSV] 보조 센서 부재는 비치명적 — CSV 에 해당 종류만 빠진 채 기록됨.
        if (linAccSensor == null) tvStatus.text = "⚠ LinearAcceleration 센서 없음 (CSV 에 linAcc 미기록)"
        if (rotVecSensor == null) tvStatus.text = "⚠ RotationVector 센서 없음 (CSV 에 rotVec 미기록)"

        btnStart.setOnClickListener { startCollection() }
        btnStop.setOnClickListener  { stopCollection()  }
    }

    private fun startCollection() {
        if (running) return
        running        = true
        sampleCount    = 0L
        totalSamples   = 0L
        lastRateCalcMs = System.currentTimeMillis()

        // [P40 CSV] 기록 파일 준비 — getExternalFilesDir(null) 은 앱 전용 외부 저장소(/sdcard/Android/data/...).
        //          앱 권한 없이 쓸 수 있고, adb pull 로 PC 에 가져오기 좋다.
        csvLineCount = 0L
        try {
            val dir = File(getExternalFilesDir(null), "imu_csv")
            if (!dir.exists()) dir.mkdirs()
            val name = "imu_record_${System.currentTimeMillis()}.csv"
            csvFile = File(dir, name)
            csvWriter = csvFile!!.bufferedWriter()
            // long-format 헤더 — acc/gyr/linAcc 는 w=0.0 으로 채움, rotVec 만 4 축 사용.
            csvWriter!!.write("sensor,ts_ns,x,y,z,w\n")
        } catch (e: IOException) {
            csvWriter = null
            csvFile   = null
            Toast.makeText(this, "CSV 파일 열기 실패: ${e.message}", Toast.LENGTH_LONG).show()
        }

        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroSensor,  SensorManager.SENSOR_DELAY_FASTEST)
        // [P40 CSV] 추가 센서 등록 — UI/EKF 와 무관, CSV 기록 전용.
        linAccSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        rotVecSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }

        btnStart.isEnabled = false
        btnStop.isEnabled  = true
        tvStatus.text      = "● 수집 중..."
        tvStatus.setTextColor(0xFF388E3C.toInt())  // 초록

        handler.post(uiUpdater)
    }

    private fun stopCollection() {
        if (!running) return
        running = false
        sensorManager.unregisterListener(this)
        handler.removeCallbacks(uiUpdater)

        // [P40 CSV] 파일 마무리 — flush() 후 close(). 실패해도 UI 갱신은 진행.
        val path  = csvFile?.absolutePath
        val lines = csvLineCount
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (e: IOException) {
            // 디스크 풀 등 → 부분 기록은 이미 디스크에 있으므로 무시.
        }
        csvWriter = null
        if (path != null) {
            Toast.makeText(
                this,
                "CSV 저장 완료\n$path\n총 $lines 줄",
                Toast.LENGTH_LONG
            ).show()
        }

        btnStart.isEnabled = true
        btnStop.isEnabled  = false
        tvStatus.text      = "■ 정지됨  (총 ${totalSamples}샘플, CSV ${lines}줄)"
        tvStatus.setTextColor(0xFF757575.toInt())  // 회색
    }

    // ── SensorEventListener ──────────────────────────────────────
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent) {
        // (1) 화면 표시용 가속도/자이로 캐싱 + 가속도 샘플 카운트 (기존 동작 그대로)
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accVal[0] = event.values[0]
                accVal[1] = event.values[1]
                accVal[2] = event.values[2]
                totalSamples++
                sampleCount++
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyrVal[0] = event.values[0]
                gyrVal[1] = event.values[1]
                gyrVal[2] = event.values[2]
            }
        }

        // (2) [P40 CSV] long-format 한 줄 기록 — 4 종 센서 모두 처리.
        //     ts_ns 는 event.timestamp (boot-time nanoseconds, monotonic).
        //     rotVec 는 일부 단말에서 values.size == 4 (스칼라 w 포함) — 안전하게 size 체크.
        val writer = csvWriter
        if (writer != null) {
            val tag = when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER       -> "acc"
                Sensor.TYPE_GYROSCOPE           -> "gyr"
                Sensor.TYPE_LINEAR_ACCELERATION -> "linAcc"
                Sensor.TYPE_ROTATION_VECTOR     -> "rotVec"
                else                            -> null
            }
            if (tag != null) {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                val w = if (event.values.size >= 4) event.values[3] else 0.0f
                try {
                    writer.write("$tag,${event.timestamp},$x,$y,$z,$w\n")
                    csvLineCount++
                } catch (e: IOException) {
                    // 한 번 실패하면 더 이상 쓰지 않음 (디스크 풀 등) — Toast 는 stop 시 한 번만.
                    csvWriter = null
                }
            }
        }

        // (3) Hz 계산 (1초 주기) — 기존 동작 그대로
        val now = System.currentTimeMillis()
        val elapsed = now - lastRateCalcMs
        if (elapsed >= 1000L) {
            currentHz      = sampleCount * 1000.0 / elapsed
            sampleCount    = 0L
            lastRateCalcMs = now
        }
    }

    // ── UI 갱신 ──────────────────────────────────────────────────
    private fun refreshUi() {
        // 가속도계
        tvAcc.text = String.format(
            "X: %+7.3f    Y: %+7.3f    Z: %+7.3f",
            accVal[0], accVal[1], accVal[2]
        )
        val aNorm = sqrt((accVal[0]*accVal[0] + accVal[1]*accVal[1] + accVal[2]*accVal[2]).toDouble())
        val accOk = aNorm in 8.5..11.0
        tvAccNorm.text = String.format("|a| = %.3f m/s²  (%s)", aNorm, if (accOk) "✓ 정상" else "⚠ 비정상")
        tvAccNorm.setTextColor(if (accOk) 0xFF388E3C.toInt() else 0xFFD32F2F.toInt())

        // 자이로스코프
        tvGyr.text = String.format(
            "X: %+7.3f    Y: %+7.3f    Z: %+7.3f",
            gyrVal[0], gyrVal[1], gyrVal[2]
        )
        val gNorm = sqrt((gyrVal[0]*gyrVal[0] + gyrVal[1]*gyrVal[1] + gyrVal[2]*gyrVal[2]).toDouble())
        tvGyrNorm.text = String.format("|ω| = %.4f rad/s", gNorm)

        // 샘플링 통계
        tvSampleRate.text  = String.format("수신 속도: %.1f Hz  (목표: 100Hz)", currentHz)
        // [P40 CSV] 누적 샘플 옆에 CSV 라인 수 표시 — 실시간 기록 확인용.
        tvSampleCount.text = "누적 샘플: $totalSamples   |   CSV: ${csvLineCount}줄"

        // 윈도우 상태
        if (totalSamples >= 100) {
            tvWindowReady.text = "100샘플 윈도우: ✓ 준비됨"
            tvWindowReady.setTextColor(0xFF388E3C.toInt())
        } else {
            tvWindowReady.text = "100샘플 윈도우: 수집 중... (${totalSamples}/100)"
            tvWindowReady.setTextColor(0xFF757575.toInt())
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish(); return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopCollection()
    }
}
