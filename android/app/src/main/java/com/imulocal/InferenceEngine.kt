package com.imulocal

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

/**
 * InferenceEngine.kt
 * ==================
 * PyTorch Mobile Lite 모델을 로드하고 추론을 실행한다.
 *
 * 모델 입력:  FloatTensor [1, 6, 100]  (배치=1, 채널=6, 길이=100)
 * 모델 출력:  Tuple(disp[3], disp_cov[3], cls_prob[7])
 *
 * 전처리: (raw_imu - mean) / std  (norm_params.pt 값 사용)
 */
class InferenceEngine(private val context: Context) {

    companion object {
        private const val TAG = "InferenceEngine"
        private const val MODEL_ASSET = "imu_model.ptl"
        private const val META_ASSET  = "model_meta.json"
        const val INPUT_LEN     = 100
        const val INPUT_CHANNEL = 6
        const val OUTPUT_DISP   = 3
        const val OUTPUT_COV    = 3
        const val OUTPUT_CLS    = 7

        // ─────────────────────────────────────────────────────────
        // [P46-OOD] Android linAcc (m/s²) → OxIOD acc (g) 단위 변환
        //
        // 학습 데이터 단위 (P44 §1.1 진단):
        //   OxIOD acc col[4:7] = g 단위 (gravity 제거 후 userAcceleration), norm 1.0
        //   Android TYPE_LINEAR_ACCELERATION = m/s² 단위
        //   → 9.81× 차이
        //
        // 적용 위치: normalize 직전, channel 0-2 (linAcc) 만 /9.81.
        //          channel 3-5 (gyr) 는 rad/s 그대로 (Android = OxIOD 동일 단위).
        //
        // P44 R3 시도 시 (out_tlio_6ch_128 잘못된 norm 사용 시점) 86% unknown 으로
        // 악화됐었으나, *out_classifier2 진짜 norm* 이 적용된 현재는 결과 재검증 필요.
        //
        // 토글 OFF: 기존 동작 (linAcc m/s² 그대로 normalize).
        // 토글 ON:  학습 분포 매칭 시도.
        private const val USE_OOD_FIX = false  // P47-OOD rollback: norm std 매칭 성공(1.06)했으나 실측 ekf 5.75m로 악화 → 단위 fix 단독으로 부족. P47-D (R=10) baseline 유지
        private const val GRAVITY = 9.81f

        // ─────────────────────────────────────────────────────────
        // [P47-LPF] Butterworth 2nd-order Low-Pass Filter (anti-aliasing)
        //
        // 진단 (P47 FFT 분석): Android raw IMU 의 5-10Hz 대역 power 가 학습 데이터
        //   (OxIOD = iPhone Core Motion anti-aliased) 대비 *25배* 큼 (acc z 기준).
        //   → handheld 0% / running 22% / unknown 71% 의 OoD 원인 후보 1순위.
        //
        // 적용: window normalize 직전 모든 6채널에 동일 LPF 적용.
        //       cutoff fc=5Hz, sample rate fs=100Hz, Q=1/sqrt(2) (Butterworth).
        //       보행 fundamental (1-3Hz) 96~99% 보존, 5-10Hz 약 3배 감쇠 (Python 사전 검증).
        //
        // 토글 OFF: 기존 동작 (raw window 그대로 normalize).
        // 토글 ON:  5Hz LPF 적용 후 normalize → 학습 spectral 분포 매칭 시도.
        private const val USE_LPF = false  // P47-LPF rollback: jumpgate_008 ekf 19.6m 발산 → 가설 기각
        // Direct Form II Transposed biquad 계수 (Python scipy butter(2, 5/50) 등가)
        private const val LPF_B0 =  0.020083f
        private const val LPF_B1 =  0.040167f
        private const val LPF_B2 =  0.020083f
        private const val LPF_A1 = -1.561018f
        private const val LPF_A2 =  0.641352f
    }

    // 첫 추론 시 1회 진단 로그 출력용
    @Volatile private var oodDiagLogged: Boolean = false

    private var module: Module? = null

    // 정규화 파라미터 (채널별 mean/std)
    private var normMean = FloatArray(INPUT_CHANNEL) { 0f }
    private var normStd  = FloatArray(INPUT_CHANNEL) { 1f }

    // ── 초기화 ──────────────────────────────────────────────────
    fun load() {
        val modelFile = copyAssetToCache(MODEL_ASSET)
        module = LiteModuleLoader.load(modelFile.absolutePath)
        loadNormParams()
        Log.i(TAG, "Model loaded: $MODEL_ASSET")
    }

    fun isLoaded() = module != null

    // ── LPF (Butterworth biquad, channel-wise) ─────────────────
    /**
     * [P47-LPF] 6채널 window 에 동일 biquad LPF 를 적용.
     * Direct Form II Transposed (수치 안정성). 채널별 filter state 독립.
     * window 입력 시점마다 state 초기화 → transient ~2 sample (무시 가능).
     */
    private fun applyLPF(window: FloatArray): FloatArray {
        val out = FloatArray(window.size)
        for (ch in 0 until INPUT_CHANNEL) {
            var s1 = 0f
            var s2 = 0f
            for (t in 0 until INPUT_LEN) {
                val idx = ch * INPUT_LEN + t
                val v = window[idx]
                val y = LPF_B0 * v + s1
                s1 = LPF_B1 * v - LPF_A1 * y + s2
                s2 = LPF_B2 * v - LPF_A2 * y
                out[idx] = y
            }
        }
        return out
    }

    // ── 추론 ────────────────────────────────────────────────────
    /**
     * IMU 윈도우 데이터로 추론을 실행한다.
     *
     * @param window  Float 배열, 크기 = INPUT_CHANNEL × INPUT_LEN
     *                layout: [ch0_t0, ch0_t1, ..., ch5_t(N-1)]  (channel-major)
     * @return InferenceResult
     */
    fun infer(window: FloatArray): InferenceResult {
        requireNotNull(module) { "모델이 로드되지 않았습니다. load() 를 먼저 호출하세요." }
        require(window.size == INPUT_CHANNEL * INPUT_LEN)

        // [P47-LPF] 5Hz Butterworth LPF 를 normalize 직전에 적용 (anti-aliasing)
        val srcWindow = if (USE_LPF) applyLPF(window) else window

        // [P46-OOD] linAcc /= 9.81 (channel 0-2) + 정규화
        // OOD fix OFF 시: 기존 동작 (m/s² 그대로 정규화)
        // OOD fix ON  시: linAcc 를 g 단위로 변환 후 정규화 → OxIOD 학습 분포 매칭 시도
        val normalized = FloatArray(srcWindow.size)
        for (ch in 0 until INPUT_CHANNEL) {
            // channel 0-2 (linAcc) 만 9.81 로 나눔. channel 3-5 (gyr) 는 그대로.
            val unitScale = if (USE_OOD_FIX && ch < 3) GRAVITY else 1f
            for (t in 0 until INPUT_LEN) {
                val idx = ch * INPUT_LEN + t
                normalized[idx] = (srcWindow[idx] / unitScale - normMean[ch]) / normStd[ch]
            }
        }

        // 첫 추론 시 1회 진단 — raw vs LPF 후 vs normalized 통계
        if (!oodDiagLogged) {
            oodDiagLogged = true
            val rawMeanCh = FloatArray(INPUT_CHANNEL)
            val rawStdCh  = FloatArray(INPUT_CHANNEL)
            val lpfMeanCh = FloatArray(INPUT_CHANNEL)
            val lpfStdCh  = FloatArray(INPUT_CHANNEL)
            val normMeanCh = FloatArray(INPUT_CHANNEL)
            val normStdCh  = FloatArray(INPUT_CHANNEL)
            for (ch in 0 until INPUT_CHANNEL) {
                var s = 0f; var s2 = 0f
                var ls = 0f; var ls2 = 0f
                var ns = 0f; var ns2 = 0f
                for (t in 0 until INPUT_LEN) {
                    val v  = window[ch * INPUT_LEN + t]
                    val lv = srcWindow[ch * INPUT_LEN + t]
                    val nv = normalized[ch * INPUT_LEN + t]
                    s += v;   s2 += v * v
                    ls += lv; ls2 += lv * lv
                    ns += nv; ns2 += nv * nv
                }
                rawMeanCh[ch]  = s / INPUT_LEN
                rawStdCh[ch]   = kotlin.math.sqrt((s2 / INPUT_LEN - rawMeanCh[ch] * rawMeanCh[ch]).coerceAtLeast(0f))
                lpfMeanCh[ch]  = ls / INPUT_LEN
                lpfStdCh[ch]   = kotlin.math.sqrt((ls2 / INPUT_LEN - lpfMeanCh[ch] * lpfMeanCh[ch]).coerceAtLeast(0f))
                normMeanCh[ch] = ns / INPUT_LEN
                normStdCh[ch]  = kotlin.math.sqrt((ns2 / INPUT_LEN - normMeanCh[ch] * normMeanCh[ch]).coerceAtLeast(0f))
            }
            Log.i(TAG, "[P47-DIAG] USE_OOD_FIX=$USE_OOD_FIX  USE_LPF=$USE_LPF (fc=5Hz, fs=100Hz)")
            Log.i(TAG, "  raw  mean=${rawMeanCh.toList().map { "%.4f".format(it) }}  std=${rawStdCh.toList().map { "%.4f".format(it) }}")
            Log.i(TAG, "  lpf  mean=${lpfMeanCh.toList().map { "%.4f".format(it) }}  std=${lpfStdCh.toList().map { "%.4f".format(it) }}  (LPF 후 — 5-10Hz 감쇠)")
            Log.i(TAG, "  norm mean=${normMeanCh.toList().map { "%.4f".format(it) }}  std=${normStdCh.toList().map { "%.4f".format(it) }}  (≈0 / ≈1 면 학습 분포 매칭)")
        }

        // Tensor 생성 [1, 6, 100]
        val inputTensor = Tensor.fromBlob(
            normalized,
            longArrayOf(1L, INPUT_CHANNEL.toLong(), INPUT_LEN.toLong())
        )

        // 추론 — forward() 실패 시 fallback 결과 반환
        val output = try {
            module!!.forward(IValue.from(inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "forward() 실패 (모델 shape 불일치 등): ${e.javaClass.simpleName}: ${e.message}")
            return InferenceResult.fallback()
        }

        // 출력 파싱 — 모델 출력 개수에 따라 유연하게 처리
        return try {
            val tuple   = output.toTuple()
            val disp    = tuple.getOrNull(0)?.toTensor()?.dataAsFloatArray ?: FloatArray(OUTPUT_DISP)
            val dispCov = tuple.getOrNull(1)?.toTensor()?.dataAsFloatArray ?: FloatArray(OUTPUT_COV) { 1f }
            // 모델이 분류 출력을 포함하지 않는 경우 기본값(handheld=1.0) 사용
            val clsProb = if (tuple.size >= 3) {
                tuple[2].toTensor().dataAsFloatArray
            } else {
                Log.w(TAG, "모델 출력이 ${tuple.size}개 — clsProb 기본값(handheld) 사용")
                FloatArray(OUTPUT_CLS).also { it[2] = 1.0f }  // index 2 = handheld
            }
            val topClass = clsProb.indices.maxByOrNull { clsProb[it] } ?: -1
            InferenceResult(disp, dispCov, clsProb, topClass)
        } catch (e: Exception) {
            Log.e(TAG, "출력 파싱 실패: ${e.javaClass.simpleName}: ${e.message}")
            InferenceResult.fallback()
        }
    }

    // ── 정규화 파라미터 로드 ─────────────────────────────────────
    private fun loadNormParams() {
        // norm_params.pt 는 torch.save({"mean": ..., "std": ...}) 형식
        // 여기서는 Python 에서 미리 float 배열 txt 로 변환한다고 가정.
        // 실제 배포 시에는 별도 변환 스크립트로 norm_mean.bin / norm_std.bin 생성 권장.
        try {
            val meanFile = context.assets.open("norm_mean.txt")
            val stdFile  = context.assets.open("norm_std.txt")
            normMean = meanFile.bufferedReader().readLine()
                .split(",").map { it.trim().toFloat() }.toFloatArray()
            normStd  = stdFile.bufferedReader().readLine()
                .split(",").map { it.trim().toFloat() }.toFloatArray()
            meanFile.close()
            stdFile.close()
            Log.i(TAG, "Norm params loaded: mean=${normMean.toList()}")
        } catch (e: Exception) {
            Log.w(TAG, "norm params 로드 실패, 기본값(0/1) 사용: ${e.message}")
        }
    }

    // ── 유틸: assets → 캐시 복사 ───────────────────────────────
    private fun copyAssetToCache(assetName: String): File {
        val file = File(context.cacheDir, assetName)
        // 항상 새로 복사 — 압축 asset은 openFd()로 크기 비교 불가
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        Log.i(TAG, "Asset copied to cache: $assetName (${file.length()} bytes)")
        return file
    }

    fun release() {
        module?.destroy()
        module = null
    }
}

// ── 추론 결과 데이터 클래스 ───────────────────────────────────────
data class InferenceResult(
    val disp:     FloatArray,   // [dx, dy, dz]  (m)
    val dispCov:  FloatArray,   // [σx², σy², σz²]
    val clsProb:  FloatArray,   // [7] 확률 분포
    val topClass: Int           // argmax 클래스 인덱스
) {
    val className: String get() = CLASS_NAMES.getOrElse(topClass) { "unknown" }

    companion object {
        // train.py LABEL_REMAP {-1→6, 1→0, 2→1, 3→2, 4→3, 5→4, 6→5} 기준
        // 0=handbag 1=handheld 2=pocket 3=running 4=slow_walk 5=trolley 6=unknown
        val CLASS_NAMES = listOf(
            "handbag", "handheld", "pocket", "running", "slow_walk", "trolley", "unknown"
        )

        /** 추론 실패 시 EKF update 를 건너뛸 수 있도록 zero displacement + high cov 반환 */
        fun fallback() = InferenceResult(
            disp     = FloatArray(3),               // [0, 0, 0] — 이동 없음
            dispCov  = FloatArray(3) { 1e6f },      // 매우 큰 불확실도 → EKF 가중치 ≈ 0
            clsProb  = FloatArray(7).also { it[6] = 1.0f },  // unknown (index 6)
            topClass = 6
        )
    }
}
