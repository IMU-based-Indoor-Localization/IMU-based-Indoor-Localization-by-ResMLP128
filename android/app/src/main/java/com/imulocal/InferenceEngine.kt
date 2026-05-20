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
    }

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

        // 정규화
        val normalized = FloatArray(window.size)
        for (ch in 0 until INPUT_CHANNEL) {
            for (t in 0 until INPUT_LEN) {
                val idx = ch * INPUT_LEN + t
                normalized[idx] = (window[idx] - normMean[ch]) / normStd[ch]
            }
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
