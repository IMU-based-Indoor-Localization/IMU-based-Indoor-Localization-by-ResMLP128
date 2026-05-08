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

        // 추론  ── forward() 는 IValue 를 받음, Tensor 직접 전달 불가
        val output = module!!.forward(IValue.from(inputTensor))

        // 출력 파싱 (IValue tuple → 3개의 Tensor)
        val tuple = output.toTuple()
        val disp    = tuple[0].toTensor().dataAsFloatArray   // [3]
        val dispCov = tuple[1].toTensor().dataAsFloatArray   // [3]
        val clsProb = tuple[2].toTensor().dataAsFloatArray   // [7]

        val topClass = clsProb.indices.maxByOrNull { clsProb[it] } ?: -1

        return InferenceResult(disp, dispCov, clsProb, topClass)
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
        if (file.exists()) return file
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
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
    }
}
