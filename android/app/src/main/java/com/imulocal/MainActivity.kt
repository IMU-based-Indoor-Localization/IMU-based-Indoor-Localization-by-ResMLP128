package com.imulocal

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.imulocal.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * MainActivity.kt
 * ===============
 * 측위 시작/정지, 실시간 경로 표시, 상태 정보 표시.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: LocalizationViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        viewModel = ViewModelProvider(this)[LocalizationViewModel::class.java]

        // ── 버튼 이벤트 ─────────────────────────────────────────
        binding.btnStart.setOnClickListener { requestAndStart() }
        binding.btnStop.setOnClickListener  { viewModel.stop()  }
        binding.btnReset.setOnClickListener {
            viewModel.reset()
            binding.trackView.clearPath()   // TrackView 내부 두 경로 모두 초기화
        }
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

                // [P57] 경로 B 단일 궤적 — modelTrackPoints 미사용(범례 단일화).
                binding.trackView.updatePaths(s.trackPoints, emptyList())
            }
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
            R.id.action_ekf_mode -> {
                showEkfModeDialog()
                true
            }
            R.id.action_replay_pick -> {
                showReplayPicker()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    /**
     * [P62] Replay CSV 선택 다이얼로그.
     *  - getExternalFilesDir(null)/imu_csv/ 하위 (재귀) 모든 .csv 파일 자동 나열.
     *  - 최근 수정 순 정렬. 파일명 + 상대경로 + 크기 표시.
     *  - 선택 시 viewModel.start(replayCsv=file) 호출.
     *
     * 포함 대상 (둘 다 같은 다이얼로그):
     *   - imu_csv/imu_record_*.csv   (ImuTestActivity 가 저장)
     *   - imu_csv/replay/latest.csv  (adb push 한 파일)
     *   - imu_csv/**/*.csv           (사용자가 임의 위치에 둔 파일)
     */
    private fun showReplayPicker() {
        val baseDir = java.io.File(getExternalFilesDir(null), "imu_csv")
        val files = mutableListOf<java.io.File>()
        fun scan(d: java.io.File) {
            if (!d.isDirectory) return
            d.listFiles()?.forEach {
                if (it.isDirectory) scan(it)
                else if (it.extension.equals("csv", ignoreCase = true)) files.add(it)
            }
        }
        scan(baseDir)
        if (files.isEmpty()) {
            Toast.makeText(
                this,
                "imu_csv/ 디렉토리에 CSV 파일이 없습니다.\n" +
                "PC: adb push ... 로 추가하거나\n" +
                "IMU 진단에서 수집 시작 후 정지하여 저장하세요.\n" +
                "경로: ${baseDir.absolutePath}",
                Toast.LENGTH_LONG
            ).show()
            return
        }
        files.sortByDescending { it.lastModified() }
        val labels = files.map { f ->
            val sizeKb = f.length() / 1024
            val rel = f.relativeTo(baseDir).path
            "$rel  (${sizeKb} KB)"
        }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Replay CSV 선택 (${files.size}개)")
            .setItems(labels) { _, which ->
                val file = files[which]
                Toast.makeText(
                    this,
                    "Replay 시작: ${file.name}",
                    Toast.LENGTH_SHORT
                ).show()
                viewModel.start(replayCsv = file)
            }
            .setNegativeButton("취소", null)
            .show()
    }

    /**
     * [P60] EKF 모드 선택 다이얼로그.
     *  PATH_B       : 데모 기본 (RotVec DR + PDR-hybrid, EKF 미사용).
     *  EKF_CURRENT  : 경로 A — 단말 현재 cfg (DEFAULT_PARAMS).
     *  EKF_TLIO     : 경로 A — TLIO 논문 §V-D/§V-E cfg (TLIO_PARAMS).
     *
     * 측위 실행 중 변경 시: 다음 [시작] 부터 새 모드 적용 (현재 세션 영향 없음).
     */
    private fun showEkfModeDialog() {
        val modes  = LocalizationViewModel.EkfMode.values()
        val labels = arrayOf(
            "PATH_B (데모 기본, RotVec DR)",
            "EKF_CURRENT (경로 A, 단말 cfg)",
            "EKF_TLIO (경로 A, TLIO 논문 cfg)"
        )
        val cur = LocalizationViewModel.ekfMode.ordinal
        AlertDialog.Builder(this)
            .setTitle("EKF 모드 (비교용)")
            .setSingleChoiceItems(labels, cur) { dlg, which ->
                LocalizationViewModel.ekfMode = modes[which]
                Toast.makeText(this,
                    "다음 [시작] 부터 적용: ${modes[which].name}",
                    Toast.LENGTH_SHORT).show()
                dlg.dismiss()
            }
            .setNegativeButton("닫기", null)
            .show()
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
        // [P60] 헤더에 모드 메타 + 파일명에 mode 토큰 — 외부 비교 도구가 자동 인식.
        val mode = LocalizationViewModel.ekfMode.name
        val csv = buildString {
            appendLine("# mode=$mode")
            appendLine("# n_points=${points.size}")
            appendLine("x_m,y_m")
            points.forEach { (x, y) -> appendLine("$x,$y") }
        }
        val file = java.io.File(
            getExternalFilesDir(null),
            "track_${mode}_${System.currentTimeMillis()}.csv"
        )
        file.writeText(csv)
        Toast.makeText(this, "경로 저장: ${file.name}", Toast.LENGTH_LONG).show()
    }
}
