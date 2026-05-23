package com.imulocal

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
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
                // [P57] 휴대 방식 — 표시 전용(위치 계산엔 사용 안 함).
                // 데모는 HANDHELD 자세 한정. docs/HANDOFF_P56.md §8 참고.
                binding.tvCarryMode.text =
                    "휴대 방식: ${s.carryMode}  (${(s.carryProb * 100).toInt()}%) — 표시 전용"

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
            else -> super.onOptionsItemSelected(item)
        }
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
        // CSV 형식으로 저장
        val csv = buildString {
            appendLine("x_m,y_m")
            points.forEach { (x, y) -> appendLine("$x,$y") }
        }
        val file = java.io.File(getExternalFilesDir(null), "track_${System.currentTimeMillis()}.csv")
        file.writeText(csv)
        Toast.makeText(this, "경로 저장: ${file.name}", Toast.LENGTH_LONG).show()
    }
}
