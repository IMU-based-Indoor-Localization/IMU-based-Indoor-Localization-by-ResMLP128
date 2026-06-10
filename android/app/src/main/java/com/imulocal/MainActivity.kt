package com.imulocal

import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.KeyEvent
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.EditText
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
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

    // [P78] "전처리 OFF" 궤적 polyline (보라 점선) — 논문 §3.3 ablation 시각화.
    //   PATH_B (파랑) 와 *병렬* 실행. 토글 ON 시 window-start-only 회전 + 동일 후처리.
    //   변수명 ekfPolyline / ekfVisible 은 P73 → P78 의미 재정의 (라벨만 변경).
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

    // [P84] 평면도 오버레이 모드 (격자 ↔ 평면도). 체크포인트 기록 모드 플래그.
    private var isFloorPlanMode: Boolean = false
    private var checkpointMode: Boolean = false

    // [P87] 절대 GT 마크 — 웨이포인트 통과 시 (est_x, est_y, t_ms) 기록. 볼륨키/버튼.
    private val gtMarks = mutableListOf<Triple<Double, Double, Long>>()
    /** 외부에서 고른 평면도를 복사 보관하는 영구 경로 — 재시작 시 자동 로드. */
    private val floorPlanFile: java.io.File
        get() = java.io.File(java.io.File(getExternalFilesDir(null), "floorplan"), "current.png")

    // [P84] SAF 문서 선택기 — 갤러리/파일에서 평면도 PNG/JPG 선택 → 앱 영역에 복사 후 로드.
    private val openFloorPlanLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) importFloorPlan(uri)
        }

    companion object {
        // [P79-1→P88 재활성화] 네이버 지도 스위치.
        //   true: MapFragment init + 메뉴 '표시 모드: 격자↔지도' 사용 가능 (작품 기본 화면).
        //   지도 모드는 제품 화면 — GT 마크 버튼(P87, 측정용)은 지도 모드에서 숨김.
        private const val MAP_ENABLED = true
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        viewModel = ViewModelProvider(this)[LocalizationViewModel::class.java]

        // [P68-4] MapFragment 획득 → onMapReady 콜백 등록
        // [P79-1] MAP_ENABLED=false 면 지도 init 전체 skip (네이버 네트워크 호출 0).
        if (MAP_ENABLED) {
            val mf = supportFragmentManager.findFragmentById(R.id.map) as MapFragment?
                ?: MapFragment.newInstance().also {
                    supportFragmentManager.beginTransaction().add(R.id.map, it).commit()
                }
            mf.getMapAsync(this)
        }

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
            // [P84] 평면도 오버레이 초기화 — 궤적·체크포인트만 클리어
            binding.floorPlanView.clearPath()
            binding.floorPlanView.setMode(FloorPlanView.Mode.NONE)
            loadFloorPlanCalib()   // [P88c] 초기화 직후 보정 자동 복원 (안전망)
            checkpointMode = false
            gtMarks.clear()   // [P87] GT 마크 초기화
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
        // [P87] 마크 버튼 (웨이포인트 통과) — 볼륨키로도 가능 (onKeyDown)
        binding.btnMark.setOnClickListener { addMark() }

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
                //   [P84] 평면도 오버레이도 동일 궤적 갱신 (보정돼 있으면 정렬되어 그려짐)
                binding.floorPlanView.updateTrajectory(s.trackPoints)
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

        // [P84] 평면도 오버레이 초기화 — 저장된 평면도 로드 + 보정/체크포인트 콜백
        setupFloorPlan()
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

    // [P87] 측위 중 볼륨키(UP/DOWN) = GT 마크 (eyes-free). 측위 중이 아니면 기본 볼륨 동작.
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if ((keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)
            && viewModel.state.value.isRunning) {
            addMark()
            return true   // 소비 → 볼륨 UI 안 뜸
        }
        return super.onKeyDown(keyCode, event)
    }

    // [P87] 측위 중 볼륨키 up 도 소비 (key-up 시 볼륨 변경/비프 방지)
    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if ((keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)
            && viewModel.state.value.isRunning) {
            return true
        }
        return super.onKeyUp(keyCode, event)
    }

    /** [P87] 웨이포인트 통과 마크 — 현재 추정 위치(est)와 시각을 기록. */
    private fun addMark() {
        val s = viewModel.state.value
        if (!s.isRunning) {
            Toast.makeText(this, "측위 중에만 마크 가능 — [시작] 후 사용", Toast.LENGTH_SHORT).show()
            return
        }
        val x = s.position.first; val y = s.position.second
        gtMarks.add(Triple(x, y, System.currentTimeMillis()))
        Toast.makeText(this, "마크 #${gtMarks.size}  est(%.2f, %.2f)".format(x, y), Toast.LENGTH_SHORT).show()
    }

    /** [P87] GT 마크 CSV 내보내기 — idx,est_x_m,est_y_m,t_ms. 웨이포인트 좌표는 오프라인 매칭. */
    private fun exportMarks() {
        if (gtMarks.isEmpty()) {
            Toast.makeText(this, "마크가 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }
        val sb = StringBuilder()
        sb.appendLine("# GT marks (P87) — est position at waypoint passage")
        sb.appendLine("# n=${gtMarks.size}")
        sb.appendLine("idx,est_x_m,est_y_m,t_ms")
        gtMarks.forEachIndexed { i, (x, y, t) -> sb.appendLine("${i + 1},$x,$y,$t") }
        val file = java.io.File(getExternalFilesDir(null), "marks_${System.currentTimeMillis()}.csv")
        file.writeText(sb.toString())
        Toast.makeText(this, "마크 저장: ${file.name} (${gtMarks.size}개)", Toast.LENGTH_LONG).show()
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
            R.id.action_toggle_unitfix -> { toggleUnitFix(item); true }
            R.id.action_toggle_view -> {
                toggleView(item)
                true
            }
            // [P88b] 지도 표시 경로 회전 — 누적 경로 재변환 후 polyline 즉시 갱신
            R.id.action_rotate_map -> {
                val deg = viewModel.rotateMapPath()
                val pts = viewModel.state.value.pathLatLng
                naverMap?.let { m ->
                    if (pts.size >= 2) {
                        pathPolyline.coords = pts
                        pathPolyline.map = m
                        lastPathSize = pts.size
                    }
                }
                if (isMapMode) updateCursorMarker(pts)
                Toast.makeText(this, "지도 경로 회전: ${deg}° (표시 전용 — 측정 데이터 불변)", Toast.LENGTH_SHORT).show()
                true
            }
            R.id.action_toggle_ekf -> {
                toggleEkf(item)
                true
            }
            // [P84] 평면도 오버레이
            R.id.action_toggle_floorplan -> { toggleFloorPlanView(item); true }
            R.id.action_load_floorplan -> { openFloorPlanLauncher.launch(arrayOf("image/*")); true }
            R.id.action_calib_floorplan -> { toggleAlignFloorPlan(item); true }
            R.id.action_toggle_checkpoint -> { toggleCheckpointMode(item); true }
            R.id.action_export_checkpoint -> { exportCheckpoints(); true }
            R.id.action_export_marks -> { exportMarks(); true }
            else -> super.onOptionsItemSelected(item)
        }
    }

    /**
     * [P78] "전처리 OFF 궤적" 토글 (보라 점선).
     *   논문 §3.3 의 "윈도우 시작 회전 하나만으로 근사" ablation 시각화.
     *   ON: enableEkfTrajectory=true → PATH_B 와 *병렬* 로 두 번째 inference 실행,
     *       단 입력은 transformWindowRotVecWindowStartOnly (per-sample 회전 무시) 사용.
     *       후처리 (adaptive scale + RotVec heading + clamp) 는 PATH_B 와 동일.
     *   OFF: ablation inference skip, polyline 갱신 멈춤.
     *   변수명 (ekfVisible, enableEkfTrajectory) 는 P73 잔재 — 의미만 ablation 으로 재정의.
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
            // [P88] 평면도(측정용)와 상호 배타 — 지도 진입 시 평면도 숨김
            binding.floorPlanView.visibility = View.INVISIBLE
            isFloorPlanMode = false
            findViewById<View>(R.id.map)?.visibility = View.VISIBLE
            menuItem.title = "표시 모드: 지도 → 격자"
            // [P74] 슬라이더도 같이 보이게
            binding.replayControls.visibility = View.VISIBLE
            // [P88] 지도 모드 = 제품 화면 — GT 마크 버튼(측정용) 숨김
            binding.btnMark.visibility = View.GONE
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
            // [P88] 격자(측정) 모드 복귀 — 마크 버튼 다시 표시
            binding.btnMark.visibility = View.VISIBLE
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

    /**
     * [P85] 단위보정(A) 토글 — InferenceEngine.USE_OOD_FIX on/off.
     *   ON: linAcc÷9.81(g단위) + 적응스케일 1.0× (과대보정 제거). replay/측위 A/B 비교용.
     *   토글 후에는 [초기화] 하고 측위/Replay 를 다시 실행해야 새 설정이 궤적에 반영됨.
     */
    private fun toggleUnitFix(item: MenuItem) {
        InferenceEngine.USE_OOD_FIX = !InferenceEngine.USE_OOD_FIX
        val on = InferenceEngine.USE_OOD_FIX
        item.title = if (on) "단위보정(A): ON" else "단위보정(A): OFF"
        Toast.makeText(
            this,
            if (on) "단위보정(A) ON — [초기화] 후 측위/Replay 다시 실행"
            else "단위보정(A) OFF (기존 동작) — [초기화] 후 다시 실행",
            Toast.LENGTH_LONG
        ).show()
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

    // ───────────────────────── [P84] 평면도 오버레이 ─────────────────────────

    /** 저장된 평면도 로드 + FloorPlanView 콜백 연결. onCreate 말미 1회 호출. */
    private fun setupFloorPlan() {
        loadFloorPlanFromDisk()
        loadFloorPlanCalib()   // [P88c] 저장된 보정 복원 (앱 재시작에도 유지)
        binding.floorPlanView.onScalePointsReady = { showScaleDistanceDialog() }
        binding.floorPlanView.onCalibrationReady = {
            saveFloorPlanCalib()   // [P88c] 보정 완료 즉시 영구 저장
            Toast.makeText(this, "보정 완료 — 궤적이 평면도에 정렬됨 (저장됨)", Toast.LENGTH_SHORT).show()
        }
        binding.floorPlanView.onCheckpointAdded = { idx ->
            Toast.makeText(this, "체크포인트 C${idx + 1} 기록", Toast.LENGTH_SHORT).show()
        }
    }

    /** 앱 영역에 보관된 평면도(current.png)가 있으면 디코드해 표시. */
    private fun loadFloorPlanFromDisk(): Boolean {
        val f = floorPlanFile
        if (!f.exists()) return false
        val bmp = BitmapFactory.decodeFile(f.absolutePath) ?: return false
        binding.floorPlanView.setFloorPlan(bmp)
        return true
    }

    /** SAF 로 고른 이미지(uri)를 앱 영역에 복사 후 로드. (재시작에도 유지) */
    private fun importFloorPlan(uri: Uri) {
        try {
            val dir = floorPlanFile.parentFile
            if (dir != null && !dir.exists()) dir.mkdirs()
            contentResolver.openInputStream(uri)?.use { input ->
                floorPlanFile.outputStream().use { output -> input.copyTo(output) }
            }
            if (loadFloorPlanFromDisk()) {
                Toast.makeText(this, "평면도 불러오기 완료 (${floorPlanFile.length() / 1024} KB)", Toast.LENGTH_SHORT).show()
                if (!isFloorPlanMode) Toast.makeText(this, "메뉴 → '표시 모드: 격자 → 평면도' 로 전환하세요", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "이미지 디코드 실패", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "평면도 불러오기 오류: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // [P88c] 평면도 보정 영구 저장/복원 — SharedPreferences("floorplan").
    private fun saveFloorPlanCalib() {
        val c = binding.floorPlanView.getCalibration() ?: return
        getSharedPreferences("floorplan", MODE_PRIVATE).edit()
            .putString("calib", c.joinToString(","))
            .apply()
    }

    private fun loadFloorPlanCalib() {
        val s = getSharedPreferences("floorplan", MODE_PRIVATE).getString("calib", null) ?: return
        val c = s.split(",").mapNotNull { it.toDoubleOrNull() }.toDoubleArray()
        if (c.size == 6) binding.floorPlanView.setCalibration(c)
    }

    /** 격자(TrackView) ↔ 평면도(FloorPlanView) 토글. 지도 모드와 독립. */
    private fun toggleFloorPlanView(menuItem: MenuItem) {
        isFloorPlanMode = !isFloorPlanMode
        if (isFloorPlanMode) {
            if (!binding.floorPlanView.hasFloorPlan() && !loadFloorPlanFromDisk()) {
                Toast.makeText(this, "먼저 '평면도 불러오기'로 이미지를 선택하세요", Toast.LENGTH_LONG).show()
            }
            binding.trackView.visibility = View.INVISIBLE
            // [P88] 지도(제품 화면)와 상호 배타 — 평면도(측정용) 진입 시 지도 숨김
            findViewById<View>(R.id.map)?.visibility = View.INVISIBLE
            binding.replayControls.visibility = View.GONE
            isMapMode = false
            binding.btnMark.visibility = View.VISIBLE   // 측정 모드 — 마크 버튼 표시
            binding.floorPlanView.visibility = View.VISIBLE
            menuItem.title = "표시 모드: 평면도 → 격자"
        } else {
            binding.floorPlanView.visibility = View.INVISIBLE
            binding.trackView.visibility = View.VISIBLE
            menuItem.title = "표시 모드: 격자 → 평면도"
        }
    }

    // [P88e] 직접 정렬 모드 토글 — 보정 절차(거리/방향 탭) 대체.
    //   ON: 한 손가락=궤적 이동, 두 손가락=회전/크기. OFF: 변환 영구 저장.
    private var alignMode = false

    private fun toggleAlignFloorPlan(item: MenuItem) {
        if (!isFloorPlanMode) {
            Toast.makeText(this, "먼저 평면도 모드로 전환하세요", Toast.LENGTH_SHORT).show()
            return
        }
        if (!alignMode && viewModel.state.value.trackPoints.size < 2 && !binding.floorPlanView.isCalibrated()) {
            Toast.makeText(this, "궤적이 없습니다 — 측위/Replay 후 정렬하세요", Toast.LENGTH_SHORT).show()
            return
        }
        alignMode = !alignMode
        if (alignMode) {
            binding.floorPlanView.setMode(FloorPlanView.Mode.ALIGN)
            item.title = "평면도 정렬 완료 (저장)"
            Toast.makeText(this, "한 손가락=이동 · 두 손가락=회전/크기 — 끝나면 메뉴 '정렬 완료'", Toast.LENGTH_LONG).show()
        } else {
            binding.floorPlanView.setMode(FloorPlanView.Mode.NONE)
            saveFloorPlanCalib()
            item.title = "평면도 정렬 (드래그·핀치·회전)"
            Toast.makeText(this, "정렬 저장됨 — 초기화/재시작 후에도 유지", Toast.LENGTH_SHORT).show()
        }
    }

    /** [P88e 이후 미사용 — 구 2점 보정 진입점, 메뉴에서 분리됨] */
    private fun startFloorPlanCalibration() {
        if (!isFloorPlanMode) {
            Toast.makeText(this, "먼저 평면도 모드로 전환하세요", Toast.LENGTH_SHORT).show()
            return
        }
        if (viewModel.state.value.trackPoints.size < 2) {
            Toast.makeText(this, "보정 기준 궤적이 없습니다 — 측위/Replay 후 시도하세요", Toast.LENGTH_LONG).show()
            return
        }
        binding.floorPlanView.setMode(FloorPlanView.Mode.CALIBRATE)
        Toast.makeText(this, "①거리를 아는 두 점을 탭하세요 (다음 화면에서 실거리 입력)", Toast.LENGTH_LONG).show()
    }

    /** ①스케일 단계 — 방금 찍은 두 점의 실제 거리(m) 입력 → px/m 확정, ②단계로 진행. */
    private fun showScaleDistanceDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "예: 11.2"
        }
        AlertDialog.Builder(this)
            .setTitle("① 스케일: 실제 거리(m)")
            .setMessage("방금 찍은 두 점의 실제 거리를 입력하세요.\n(예: 강의실 한 칸 = 11.2 m — 도면 치수선 참고)")
            .setView(input)
            .setPositiveButton("다음") { _, _ ->
                val v = input.text.toString().toDoubleOrNull()
                if (v != null && v > 0) {
                    binding.floorPlanView.setScaleFromRealDistance(v)
                    Toast.makeText(this, "②출발 지점 → 처음 걸어간 방향을 탭하세요", Toast.LENGTH_LONG).show()
                } else {
                    binding.floorPlanView.cancelCalibration()
                    Toast.makeText(this, "거리 입력이 올바르지 않아 보정을 취소했습니다", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("취소") { _, _ -> binding.floorPlanView.cancelCalibration() }
            .setCancelable(false)
            .show()
    }

    /** 체크포인트 기록 모드 토글. ON 동안 평면도 탭 = 체크포인트(참위치) 기록. */
    private fun toggleCheckpointMode(menuItem: MenuItem) {
        if (!isFloorPlanMode) {
            Toast.makeText(this, "먼저 평면도 모드로 전환하세요", Toast.LENGTH_SHORT).show()
            return
        }
        checkpointMode = !checkpointMode
        if (checkpointMode) {
            binding.floorPlanView.setMode(FloorPlanView.Mode.CHECKPOINT)
            menuItem.title = "체크포인트 기록 중지"
            Toast.makeText(this, "랜드마크 지날 때 평면도를 탭하세요", Toast.LENGTH_LONG).show()
        } else {
            binding.floorPlanView.setMode(FloorPlanView.Mode.NONE)
            menuItem.title = "체크포인트 기록 시작"
        }
    }

    /**
     * 체크포인트 CSV 내보내기. 보정 transform 으로 참위치(image-px)→world(m) 환산,
     * 탭 순간 궤적 추정위치(est)와 함께 오차(ATE)까지 기록.
     * 컬럼: idx,t_ms,est_x_m,est_y_m,true_x_m,true_y_m,err_m
     */
    private fun exportCheckpoints() {
        val cps = binding.floorPlanView.getCheckpoints()
        if (cps.isEmpty()) {
            Toast.makeText(this, "체크포인트가 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }
        val calibrated = binding.floorPlanView.isCalibrated()
        val sb = StringBuilder()
        sb.appendLine("# floorplan checkpoints")
        sb.appendLine("# calibrated=$calibrated  n=${cps.size}")
        sb.appendLine("idx,t_ms,est_x_m,est_y_m,true_x_m,true_y_m,err_m,label")
        val errs = mutableListOf<Double>()
        cps.forEachIndexed { i, cp ->
            val tw = binding.floorPlanView.imageToWorld(cp.imgX, cp.imgY)
            val tx = tw?.first; val ty = tw?.second
            val err = if (tw != null) Math.hypot(tx!! - cp.estX, ty!! - cp.estY) else Double.NaN
            if (!err.isNaN()) errs.add(err)
            sb.appendLine(
                "${i + 1},${cp.tMs},${cp.estX},${cp.estY}," +
                "${tx ?: ""},${ty ?: ""},${if (err.isNaN()) "" else err},${cp.label}"
            )
        }
        if (errs.isNotEmpty()) {
            val mean = errs.average()
            val rmse = Math.sqrt(errs.sumOf { it * it } / errs.size)
            sb.appendLine("# checkpoint ATE: mean=${"%.3f".format(mean)} m  rmse=${"%.3f".format(rmse)} m")
        }
        val file = java.io.File(getExternalFilesDir(null), "checkpoints_${System.currentTimeMillis()}.csv")
        file.writeText(sb.toString())
        val ateMsg = if (errs.isNotEmpty()) "  ATE=%.2fm".format(errs.average()) else "  (미보정)"
        Toast.makeText(this, "체크포인트 저장: ${file.name}$ateMsg", Toast.LENGTH_LONG).show()
    }
}
