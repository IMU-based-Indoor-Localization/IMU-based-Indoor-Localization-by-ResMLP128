package com.imulocal

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.imulocal.databinding.ActivityMainBinding
import com.naver.maps.map.CameraAnimation
import com.naver.maps.map.CameraPosition
import com.naver.maps.map.CameraUpdate
import com.naver.maps.map.MapFragment
import com.naver.maps.map.NaverMap
import com.naver.maps.map.OnMapReadyCallback
import com.naver.maps.map.overlay.Marker
import com.naver.maps.map.overlay.PolylineOverlay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * MainActivity.kt
 * ===============
 * 측위 시작/정지, 실시간 경로 표시, 상태 정보 표시.
 */
class MainActivity : AppCompatActivity(), OnMapReadyCallback {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: LocalizationViewModel

    // [P68-4] Naver Map + PATH_B 궤적 polyline (파랑)
    private var naverMap: NaverMap? = null
    private val pathPolyline = PolylineOverlay()
    private var lastPathSize: Int = 0
    private var cameraMoved: Boolean = false

    // [P68-6] 시작점 marker (빨간 핀) — long-press 로 위치 이동
    private val startMarker = Marker().apply {
        position = LocalizationViewModel.DEFAULT_ANCHOR
        captionText = "시작점"
        captionTextSize = 12f
    }

    // [P73] EKF 궤적 polyline (보라 점선) — 기존 EKF 모드를 부가 궤적으로.
    //   PATH_B (파랑) 와 *병렬* 실행. 토글 ON 시 EKF measurement 흐름 진행 +
    //   ekfPosJob (ViewModel) 이 EkfBridge.getPosition 을 50ms 마다 누적.
    //   학술 시연 종료 후 제거 예정.
    private val ekfPolyline = PolylineOverlay().apply {
        color = android.graphics.Color.parseColor("#9C27B0")  // 보라 (Material Purple 500)
        width = 9
        setPattern(24, 12)
    }
    private var ekfVisible: Boolean = false
    private var lastEkfSize: Int = 0

    // [P74] 재생 슬라이더 + 현재 위치 마커.
    //   sliderProgress: 0~100 (퍼센트). 100 = 라이브, 100 미만 = 사용자 scrub.
    //   cursor marker: PATH_B polyline 의 슬라이더 위치에 표시.
    //   색 — 시작점(녹색)/PATH_B(파랑)/EKF(보라)/raw(녹색)/지도(베이지) 모두와 구분되는
    //   핑크 A400 (#F50057). iconTintColor 로 기본 핀 아이콘에 틴트.
    //   학술 시연 종료 후 제거 예정.
    private val cursorMarker = Marker().apply {
        captionText = "현재"
        captionTextSize = 11f
        iconTintColor = android.graphics.Color.parseColor("#F50057")  // 핑크 A400
    }
    private var sliderProgress: Int = 100
    private var sliderUserControlled: Boolean = false

    // [P68-5] 표시 모드 토글 (false=격자 TrackView 기본, true=Naver Map)
    private var isMapMode: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        viewModel = ViewModelProvider(this)[LocalizationViewModel::class.java]

        // [P68-4] MapFragment 획득 → onMapReady 콜백 등록
        val mf = supportFragmentManager.findFragmentById(R.id.map) as MapFragment?
            ?: MapFragment.newInstance().also {
                supportFragmentManager.beginTransaction().add(R.id.map, it).commit()
            }
        mf.getMapAsync(this)

        // [P68-4] polyline 초기 스타일 (실제 map 연결은 onMapReady 에서)
        pathPolyline.color = Color.parseColor("#1565C0")  // 파랑
        pathPolyline.width = 10

        // ── 버튼 이벤트 ─────────────────────────────────────────
        binding.btnStart.setOnClickListener { requestAndStart() }
        binding.btnStop.setOnClickListener  { viewModel.stop()  }
        binding.btnReset.setOnClickListener {
            viewModel.reset()
            // [P68-5] 두 view 모두 초기화 (어느 모드든 대응)
            binding.trackView.clearPath()
            pathPolyline.map = null
            naverMap?.let { pathPolyline.map = it }
            lastPathSize = 0
            // [P73] EKF 궤적 polyline 도 초기화
            ekfPolyline.map = null
            if (ekfVisible) naverMap?.let { ekfPolyline.map = it }
            lastEkfSize = 0
            // [P74] cursor 마커 초기화 + 슬라이더 라이브 모드 복귀
            cursorMarker.map = null
            sliderUserControlled = false
            sliderProgress = 100
            binding.replaySeekBar.progress = 100
        }

        // [P74] 슬라이더 리스너 — 사용자가 만지면 scrub 모드, 100% 도달하면 라이브 복귀
        binding.replaySeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                sliderProgress = progress
                sliderUserControlled = (progress < 100)
                updateCursorMarker(viewModel.state.value.pathLatLng)
                updateReplayTimeLabel(viewModel.state.value.pathLatLng)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        // [P45-Replay] Replay 버튼 — 단말의 imu_csv/replay/latest.csv 를 재생
        binding.btnReplay.setOnClickListener { startReplay() }

        // ── 상태 관찰 ────────────────────────────────────────────
        lifecycleScope.launch {
            viewModel.state.collectLatest { s ->
                // 버튼 활성화 제어
                binding.btnStart.isEnabled = !s.isRunning
                binding.btnStop.isEnabled  =  s.isRunning

                // [P57] 위치 텍스트 — 경로 B(RotVec DR + PDR-hybrid) 누적 위치.
                // 경로 A(EKF) 토글 활성화 시에도 같은 필드를 공유.
                binding.tvPosition.text = String.format(
                    "[위치] x=%.3f  y=%.3f  z=%.3f m",
                    s.position.first, s.position.second, s.position.third
                )
                // 불확실도
                binding.tvStd.text = String.format(
                    "σ  x=%.4f  y=%.4f  z=%.4f m",
                    s.posStd.first, s.posStd.second, s.posStd.third
                )
                // 속도
                binding.tvVelocity.text = String.format(
                    "속도  |v|=%.2f m/s",
                    Math.sqrt(
                        s.velocity.first  * s.velocity.first +
                        s.velocity.second * s.velocity.second +
                        s.velocity.third  * s.velocity.third
                    )
                )
                // [P58] 휴대 방식(분류기 출력) 은 IMU 측정 진단 화면으로 이동.
                //  메뉴 → IMU 센서 진단 에서 측위 실행 중 확인 가능.

                // 추론 latency
                binding.tvLatency.text = "추론 지연: ${s.inferLatency} ms"

                // ── 실시간 영점 보정 UI 피드백 ─────────────────────────
                // 캘리브레이션 진행 중에만 카드 표시, 완료 시 자동 숨김.
                if (s.calibrating) {
                    binding.calibCard.visibility = View.VISIBLE
                    val pct = (s. calibProgress * 100f).toInt().coerceIn(0, 100)
                    binding.calibProgress.progress = pct
                    binding.tvCalibPercent.text = String.format("%d %%", pct)
                } else {
                    binding.calibCard.visibility = View.GONE
                }

                // [P68-5] 두 view 동시 갱신 — 토글 visibility 와 무관.
                //   격자 TrackView (P66 1m 격자, 미터 좌표)
                binding.trackView.updatePaths(s.trackPoints, emptyList())
                //   지도 polyline (P68-3 LatLng 변환, size 변화 시만 GL 갱신)
                if (s.pathLatLng.size >= 2 && s.pathLatLng.size != lastPathSize) {
                    naverMap?.let {
                        pathPolyline.map    = it
                        pathPolyline.coords = s.pathLatLng
                    }
                    lastPathSize = s.pathLatLng.size
                }
                // [P73] EKF 궤적 polyline — visible 일 때만 갱신
                if (ekfVisible && s.pathLatLngEkf.size >= 2 && s.pathLatLngEkf.size != lastEkfSize) {
                    naverMap?.let {
                        ekfPolyline.map    = it
                        ekfPolyline.coords = s.pathLatLngEkf
                    }
                    lastEkfSize = s.pathLatLngEkf.size
                }
                // [P74] cursor 마커 + 시간 라벨 갱신 (지도 모드 + 데이터 존재 시)
                if (isMapMode) {
                    updateCursorMarker(s.pathLatLng)
                    updateReplayTimeLabel(s.pathLatLng)
                }
            }
        }
    }

    // [P68-4 / fix] Naver Map 준비 콜백
    override fun onMapReady(map: NaverMap) {
        naverMap = map
        // [P68-5 fix] 카메라 이동 — CameraPosition 단일 설정 (이전 moveCamera 2회 호출 race
        //   해소). onMapReady 시점에 fragment view 가 measure 안 됐어도 cameraPosition 직접
        //   설정은 즉시 반영됨.
        moveCameraToAnchor()
        // 이미 누적된 path 가 있으면 즉시 표시
        pathPolyline.map = map
        viewModel.state.value.pathLatLng.let { pts ->
            if (pts.size >= 2) {
                pathPolyline.coords = pts
                lastPathSize = pts.size
            }
        }

        // [P68-6] 시작점 마커 표시 + long-press 로 위치 이동
        startMarker.position = viewModel.currentAnchor
        startMarker.map = map
        map.setOnMapLongClickListener { _, latLng ->
            if (viewModel.state.value.isRunning) {
                Toast.makeText(this,
                    "측위 중에는 시작점 변경 불가 — [정지] 후 다시 시도",
                    Toast.LENGTH_SHORT).show()
            } else {
                viewModel.setAnchor(latLng)
                startMarker.position = latLng
                map.cameraPosition = CameraPosition(latLng, map.cameraPosition.zoom)
                Toast.makeText(this,
                    "시작점 이동: %.6f, %.6f".format(latLng.latitude, latLng.longitude),
                    Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** [P68-5 fix / P68-6] 현재 anchor 로 카메라 이동 — long-press 변경 후도 따라감. */
    private fun moveCameraToAnchor() {
        naverMap?.let { m ->
            m.cameraPosition = CameraPosition(viewModel.currentAnchor, 18.0)
            cameraMoved = true
        }
    }

    private fun requestAndStart() {
        // Android IMU 는 별도 권한 불필요이나, 미래 호환을 위해 구조 유지
        viewModel.start()
    }

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
            R.id.action_export -> {
                exportPath()
                true
            }
            R.id.action_toggle_view -> {
                toggleView(item)
                true
            }
            R.id.action_toggle_ekf -> {
                toggleEkf(item)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    /**
     * [P76] "전처리 OFF 궤적" 토글 (보라 점선).
     *   라벨상 "전처리 OFF" 라고 표시되지만 내부 동작은 [P73] EKF measurement flow 그대로.
     *   ON: enableEkfTrajectory=true → PATH_B 와 *병렬* 로 EKF measurement 흐름 실행 +
     *       ekfPosJob 이 50ms 마다 getPosition 누적.
     *   OFF: EKF measurement 흐름 skip (propagation 만 계속), polyline 갱신 멈춤.
     *   학술 시연 종료 후 제거 예정.
     */
    private fun toggleEkf(menuItem: MenuItem) {
        ekfVisible = !ekfVisible
        LocalizationViewModel.enableEkfTrajectory = ekfVisible
        val map = naverMap
        if (ekfVisible && map != null) {
            val pts = viewModel.state.value.pathLatLngEkf
            if (pts.size >= 2) {
                ekfPolyline.coords = pts
                lastEkfSize = pts.size
            }
            ekfPolyline.map = map
            menuItem.title = "전처리 OFF 궤적 숨김"
        } else {
            ekfPolyline.map = null
            menuItem.title = "전처리 OFF 궤적 표시"
        }
    }

    /**
     * [P68-5] 격자 (TrackView) ↔ 지도 (MapFragment) 토글.
     *  두 view 모두 같은 state 를 받으므로 visibility 만 전환.
     *  메뉴 항목의 title 도 동기 갱신 — 다음 누름 동작을 안내.
     */
    private fun toggleView(menuItem: MenuItem) {
        isMapMode = !isMapMode
        if (isMapMode) {
            // [P68-5 fix] TrackView 를 INVISIBLE 로 → Map fragment(아래쪽 z-order)가 보임.
            //   GONE 사용 시 layout 재계산으로 map view 가 resize 되어 카메라 reset 가능.
            binding.trackView.visibility = View.INVISIBLE
            findViewById<View>(R.id.map)?.visibility = View.VISIBLE
            menuItem.title = "표시 모드: 지도 → 격자"
            // [P74] 슬라이더도 같이 보이게
            binding.replayControls.visibility = View.VISIBLE
            // 지도 모드로 전환 시 카메라 강제 재이동 (안전망)
            moveCameraToAnchor()
            // cursor 마커 즉시 갱신
            updateCursorMarker(viewModel.state.value.pathLatLng)
            updateReplayTimeLabel(viewModel.state.value.pathLatLng)
        } else {
            binding.trackView.visibility = View.VISIBLE
            findViewById<View>(R.id.map)?.visibility = View.INVISIBLE
            menuItem.title = "표시 모드: 격자 → 지도"
            // [P74] 격자 모드에서는 슬라이더 숨김 (지도 마커도 자동으로 안 보임)
            binding.replayControls.visibility = View.GONE
        }
    }

    /**
     * [P74] cursor 마커 갱신 — PATH_B polyline 의 슬라이더 위치에 표시.
     *   sliderUserControlled=false 면 항상 polyline 끝 (라이브 추적).
     *   true 면 sliderProgress (0~100) 비율로 인덱스 계산.
     */
    private fun updateCursorMarker(pathPts: List<com.naver.maps.geometry.LatLng>) {
        val map = naverMap
        if (map == null || pathPts.size < 2) {
            cursorMarker.map = null
            return
        }
        val idx = if (sliderUserControlled) {
            ((sliderProgress / 100.0) * (pathPts.size - 1)).toInt().coerceIn(0, pathPts.size - 1)
        } else {
            pathPts.size - 1  // 라이브: 끝
        }
        cursorMarker.position = pathPts[idx]
        if (cursorMarker.map == null) cursorMarker.map = map
    }

    /**
     * [P74] 시간 라벨 갱신 — 슬라이더 위치를 가상의 시간 (인덱스 × ~50ms) 으로 표시.
     *   total = polyline 길이 × inferInterval (≈50ms).
     */
    private fun updateReplayTimeLabel(pathPts: List<com.naver.maps.geometry.LatLng>) {
        if (pathPts.size < 2) {
            binding.tvReplayTime.text = "t = 0.00s / 0.00s   (대기)"
            return
        }
        val totalSec = pathPts.size * 0.05  // ≈50ms per polyline 점 (inferInterval)
        val curIdx = if (sliderUserControlled) {
            ((sliderProgress / 100.0) * (pathPts.size - 1)).toInt().coerceIn(0, pathPts.size - 1)
        } else {
            pathPts.size - 1
        }
        val curSec = curIdx * 0.05
        val mode = if (sliderUserControlled) "scrub" else "라이브"
        binding.tvReplayTime.text = String.format(
            "t = %.2fs / %.2fs   (%s)",
            curSec, totalSec, mode
        )
    }

    /**
     * [P45-Replay] 단말의 `imu_csv/replay/latest.csv` 파일을 viewModel 에 넘겨 재생 시작.
     *
     * push 흐름:
     *   PC: .\tools\push_replay.ps1 -Latest
     *      → adb push <local.csv> /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
     *   단말: 본 함수가 그 파일을 읽어 viewModel.start(replayCsv=file) 호출.
     *
     * 파일 없으면 Toast 안내 + 시작 안 함.
     */
    private fun startReplay() {
        val replayDir  = java.io.File(getExternalFilesDir(null), "imu_csv/replay")
        val replayFile = java.io.File(replayDir, "latest.csv")
        if (!replayFile.exists()) {
            Toast.makeText(
                this,
                "재생할 CSV 가 없습니다.\nPC: .\\tools\\push_replay.ps1 -Latest 로 push 후 다시 시도하세요.\n경로: ${replayFile.absolutePath}",
                Toast.LENGTH_LONG
            ).show()
            return
        }
        val sizeKb = replayFile.length() / 1024
        Toast.makeText(this, "Replay 시작: ${replayFile.name} (${sizeKb} KB)", Toast.LENGTH_SHORT).show()
        viewModel.start(replayCsv = replayFile)
    }

    private fun exportPath() {
        val points = viewModel.state.value.trackPoints
        if (points.isEmpty()) {
            Toast.makeText(this, "경로 데이터가 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }
        // [P71] EKF 모드 비교 기능 제거 — 항상 PATH_B 단일 경로.
        val csv = buildString {
            appendLine("# mode=PATH_B")
            appendLine("# n_points=${points.size}")
            appendLine("x_m,y_m")
            points.forEach { (x, y) -> appendLine("$x,$y") }
        }
        val file = java.io.File(
            getExternalFilesDir(null),
            "track_PATH_B_${System.currentTimeMillis()}.csv"
        )
        file.writeText(csv)
        Toast.makeText(this, "경로 저장: ${file.name}", Toast.LENGTH_LONG).show()
    }
}
