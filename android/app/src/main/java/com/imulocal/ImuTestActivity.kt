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
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import kotlin.math.sqrt

/**
 * ImuTestActivity.kt
 * ==================
 * 가속도계 + 자이로스코프 원시값을 실시간으로 화면에 표시한다.
 * 모델 파일이나 EKF 없이 센서 수신 여부만 독립적으로 검증한다.
 */
class ImuTestActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor:  Sensor? = null

    // 현재값
    private val accVal = FloatArray(3)
    private val gyrVal = FloatArray(3)

    // 샘플링 속도 계산
    private var sampleCount  = 0L
    private var lastRateCalcMs = 0L
    private var currentHz   = 0.0

    // 윈도우(100샘플) 도달 여부
    private var totalSamples = 0L

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
        btnStart      = findViewById(R.id.btnTestStart)
        btnStop       = findViewById(R.id.btnTestStop)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelSensor   = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor    = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // 센서 없을 때 경고
        if (accelSensor == null) tvStatus.text = "⚠ 가속도계 없음"
        if (gyroSensor  == null) tvStatus.text = "⚠ 자이로스코프 없음"

        btnStart.setOnClickListener { startCollection() }
        btnStop.setOnClickListener  { stopCollection()  }
    }

    private fun startCollection() {
        if (running) return
        running       = true
        sampleCount   = 0L
        totalSamples  = 0L
        lastRateCalcMs = System.currentTimeMillis()

        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroSensor,  SensorManager.SENSOR_DELAY_FASTEST)

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

        btnStart.isEnabled = true
        btnStop.isEnabled  = false
        tvStatus.text      = "■ 정지됨  (총 ${totalSamples}샘플)"
        tvStatus.setTextColor(0xFF757575.toInt())  // 회색
    }

    // ── SensorEventListener ──────────────────────────────────────
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent) {
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

        // Hz 계산 (1초 주기)
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
        tvSampleCount.text = "누적 샘플: $totalSamples"

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
