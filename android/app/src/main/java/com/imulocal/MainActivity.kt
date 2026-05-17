package com.imulocal

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.naver.maps.geometry.LatLng
import com.naver.maps.map.CameraAnimation
import com.naver.maps.map.CameraUpdate
import com.naver.maps.map.MapFragment
import com.naver.maps.map.NaverMap
import com.naver.maps.map.OnMapReadyCallback
import com.naver.maps.map.overlay.PolylineOverlay
import com.imulocal.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.math.sqrt

/**
 * MainActivity.kt (P23)
 * =====================
 * 네이버 지도 위에 세 경로를 실시간으로 표시한다.
 *
 *  경로 색상:
 *    파랑 #1565C0 : GPS 실제 경로
 *    주황 #E65100 : IMU-only (모델 변위 누적)
 *    초록 #2E7D32 : LLIO — 모델 + EKF 보정
 */
class MainActivity : AppCompatActivity(), OnMapReadyCallback {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: LocalizationViewModel

    // ── 네이버 지도 ───────────────────────────────────────────────
    private var naverMap: NaverMap? = null
    private val gpsPolyline  = PolylineOverlay()
    private val imuPolyline  = PolylineOverlay()
    private val llioPolyline = PolylineOverlay()

    /** GPS 첫 수신 시 카메라를 한 번만 이동 */
    private var cameraMovedOnce = false

    // ── 위치 권한 런처 ────────────────────────────────────────────
    private val locationPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        val granted = grants[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                      grants[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        if (!granted) {
            Toast.makeText(this, "위치 권한 없음 — GPS 경로는 표시되지 않습니다.", Toast.LENGTH_LONG).show()
        }
        viewModel.start()   // 권한 없어도 IMU/LLIO 동작
    }

    // ── onCreate ─────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[LocalizationViewModel::class.java]

        // 네이버 지도 프래그먼트 초기화
        val mapFragment = supportFragmentManager.findFragmentById(R.id.map) as MapFragment?
            ?: MapFragment.newInstance().also {
                supportFragmentManager.beginTransaction().add(R.id.map, it).commit()
            }
        mapFragment.getMapAsync(this)

        // 폴리라인 스타일
        setupPolylines()

        // 버튼 이벤트
        binding.btnStart.setOnClickListener { requestLocationAndStart() }
        binding.btnStop.setOnClickListener  { viewModel.stop() }
        binding.btnReset.setOnClickListener {
            viewModel.reset()
            cameraMovedOnce = false
            clearPolylines()
        }

        // State 구독
        lifecycleScope.launch {
            viewModel.state.collectLatest { s ->
                updateButtons(s.isRunning)
                updateInfoCard(s)
                updateCalibCard(s)
                updatePolylines(s)
            }
        }
    }

    // ── 지도 준비 완료 ────────────────────────────────────────────
    override fun onMapReady(map: NaverMap) {
        naverMap = map
        map.uiSettings.isLocationButtonEnabled = true

        // 폴리라인을 지도에 붙임
        gpsPolyline.map  = map
        imuPolyline.map  = map
        llioPolyline.map = map
    }

    // ── 폴리라인 스타일 설정 ──────────────────────────────────────
    private fun setupPolylines() {
        gpsPolyline.apply  { color = Color.parseColor("#1565C0"); width = 8 }
        imuPolyline.apply  { color = Color.parseColor("#E65100"); width = 6 }
        llioPolyline.apply { color = Color.parseColor("#2E7D32"); width = 8 }
    }

    // ── State → UI 반영 ──────────────────────────────────────────
    private fun updateButtons(isRunning: Boolean) {
        binding.btnStart.isEnabled = !isRunning
        binding.btnStop.isEnabled  =  isRunning
    }

    private fun updateInfoCard(s: LocalizationViewModel.LocalizationState) {
        binding.tvPosition.text = String.format(
            "[EKF] x=%.3f  y=%.3f  z=%.3f m",
            s.position.first, s.position.second, s.position.third
        )
        binding.tvStd.text = String.format(
            "σ  x=%.4f  y=%.4f  z=%.4f m",
            s.posStd.first, s.posStd.second, s.posStd.third
        )
        val speed = sqrt(
            s.velocity.first  * s.velocity.first  +
            s.velocity.second * s.velocity.second +
            s.velocity.third  * s.velocity.third
        )
        binding.tvVelocity.text  = String.format("속도  |v|=%.2f m/s", speed)
        binding.tvCarryMode.text = "휴대 방식: ${s.carryMode}  (${(s.carryProb * 100).toInt()}%)"
        binding.tvLatency.text   = "추론 지연: ${s.inferLatency} ms"
    }

    private fun updateCalibCard(s: LocalizationViewModel.LocalizationState) {
        if (s.calibrating) {
            binding.calibCard.visibility = View.VISIBLE
            val pct = (s.calibProgress * 100f).toInt().coerceIn(0, 100)
            binding.calibProgress.progress = pct
            binding.tvCalibPercent.text = String.format("%d %%", pct)
        } else {
            binding.calibCard.visibility = View.GONE
        }
    }

    private fun updatePolylines(s: LocalizationViewModel.LocalizationState) {
        val map = naverMap ?: return

        if (s.gpsPath.size >= 2) {
            gpsPolyline.map    = map
            gpsPolyline.coords = s.gpsPath
        }
        if (s.imuOnlyPath.size >= 2) {
            imuPolyline.map    = map
            imuPolyline.coords = s.imuOnlyPath
        }
        if (s.llioPath.size >= 2) {
            llioPolyline.map    = map
            llioPolyline.coords = s.llioPath
        }

        // GPS 첫 수신 시 카메라 한 번 이동
        if (!cameraMovedOnce && s.gpsReady && s.currentGps != null) {
            map.moveCamera(
                CameraUpdate.scrollTo(s.currentGps).animate(CameraAnimation.Easing, 800)
            )
            cameraMovedOnce = true
        }
    }

    private fun clearPolylines() {
        gpsPolyline.map  = null
        imuPolyline.map  = null
        llioPolyline.map = null
        naverMap?.let {
            gpsPolyline.map  = it
            imuPolyline.map  = it
            llioPolyline.map = it
        }
    }

    // ── 위치 권한 요청 ────────────────────────────────────────────
    private fun requestLocationAndStart() {
        val fine   = Manifest.permission.ACCESS_FINE_LOCATION
        val coarse = Manifest.permission.ACCESS_COARSE_LOCATION
        val ok = ContextCompat.checkSelfPermission(this, fine)   == PackageManager.PERMISSION_GRANTED ||
                 ContextCompat.checkSelfPermission(this, coarse) == PackageManager.PERMISSION_GRANTED
        if (ok) viewModel.start()
        else    locationPermLauncher.launch(arrayOf(fine, coarse))
    }

    // ── 메뉴 ─────────────────────────────────────────────────────
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_imu_test -> {
                startActivity(Intent(this, ImuTestActivity::class.java))
                true
            }
            R.id.action_export -> exportPath().let { true }
            else -> super.onOptionsItemSelected(item)
        }
    }

    // ── 경로 CSV 내보내기 ─────────────────────────────────────────
    private fun exportPath() {
        val s = viewModel.state.value
        if (s.gpsPath.isEmpty() && s.llioPath.isEmpty()) {
            Toast.makeText(this, "경로 데이터가 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }
        val csv = buildString {
            appendLine("type,lat,lng")
            s.gpsPath.forEach     { appendLine("gps,${it.latitude},${it.longitude}")  }
            s.imuOnlyPath.forEach { appendLine("imu,${it.latitude},${it.longitude}")  }
            s.llioPath.forEach    { appendLine("llio,${it.latitude},${it.longitude}") }
        }
        val file = java.io.File(getExternalFilesDir(null), "track_${System.currentTimeMillis()}.csv")
        file.writeText(csv)
        Toast.makeText(this, "저장: ${file.name}", Toast.LENGTH_LONG).show()
    }
}
