package com.imulocal

import android.util.Log
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * StatelessInferenceNode.kt — Stage 2
 * ==================================================================
 * 단방향(One-Way) 아키텍처의 두 번째 노드: 상태 비저장 네트워크 추론기.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │ 설계 원칙 — Pure Function                                   │
 *  │                                                              │
 *  │  ‣ 이전 상태나 EKF 의 존재를 전혀 모른다                    │
 *  │  ‣ infer(window) 한 번 호출이 완전히 독립적                 │
 *  │  ‣ 내부 mutable 상태 없음 (InferenceEngine 의존성만 주입)   │
 *  │  ‣ 같은 입력 → 항상 같은 출력                                │
 *  └─────────────────────────────────────────────────────────────┘
 *
 * ── 입력 ──────────────────────────────────────────────────────
 *   List<AbsoluteSensorNode.WorldSample> (Stage 1 의 출력, 100 샘플)
 *     · worldLinAcc, worldGyr  — yaw 포함된 world frame
 *     · rotMat                — 매 샘플 시점의 R_world←body
 *
 * ── 동작 ──────────────────────────────────────────────────────
 *   1. 윈도우 시작 샘플(t=0)의 rotMat 에서 yaw0 추출:
 *        yaw0 = atan2(R[1,0], R[0,0])
 *      (ZYX 오일러 yaw — 학습 dataset.py 와 동일한 컨벤션)
 *
 *   2. 모든 샘플의 worldLinAcc / worldGyr 에 R_yaw_inv 를 적용해
 *      yaw-free local frame 으로 변환 (학습 시 acc_ga / gyr_ga 와 동일한 좌표계).
 *        R_yaw_inv = | cos(yaw0)  sin(yaw0)  0|
 *                    |-sin(yaw0)  cos(yaw0)  0|
 *                    | 0          0          1|
 *        v_ga = R_yaw_inv · v_world
 *
 *   3. channel-major FloatArray[6 × 100] 윈도우 생성:
 *        ch 0-2 : acc_ga (m/s², 중력 제거)
 *        ch 3-5 : gyr_ga (rad/s)
 *
 *   4. InferenceEngine.infer() 호출 (정규화·텐서화·모델 forward 위임).
 *
 *   5. InferenceOutput 으로 포장해 반환.
 *
 * ── 출력 ──────────────────────────────────────────────────────
 *   InferenceOutput
 *     dispLocal[3]       : 윈도우 동안의 변위 (yaw-free local frame, m)
 *     dispLogVar[3]      : log-variance (3 채널) — Stage 3 에서 cov 변환
 *     classProb[7]       : 휴대 자세 분류 softmax 확률
 *     topClass           : argmax 인덱스
 *     className          : 휴대 자세 이름
 *     windowStartYaw     : yaw0 (rad) — Stage 3 가 world frame 으로 복원할 때 사용
 *     tsBeginUs/EndUs    : 윈도우 경계 타임스탬프
 *     rotAccuracyStart   : Stage 1 시작 샘플의 RotVec 정확도 (0..3)
 *
 * ── 단방향 흐름의 위치 ─────────────────────────────────────────
 *       [Stage 1] AbsoluteSensorNode
 *           │ List<WorldSample>
 *           ▼
 *       [Stage 2] StatelessInferenceNode      ← 이 파일
 *           │ InferenceOutput
 *           ▼
 *       [Stage 3] EKF 게이트웨이 + AEKF       ← 다음
 *
 * ── 비-책임 사항 ───────────────────────────────────────────────
 *   · 윈도우 폴링 주기 결정 — 호출자(컨트롤러) 책임
 *   · 정지/이동 판정      — Stage 3 또는 별도 모듈 책임
 *   · 좌표 변환 후 누적   — Stage 3 책임
 *   · 모델 로드/언로드    — InferenceEngine 책임
 */
class StatelessInferenceNode(private val engine: InferenceEngine) {

    companion object {
        private const val TAG = "Stage2.InferNode"
        const val WINDOW_SIZE  = AbsoluteSensorNode.WINDOW_SIZE   // 100
        const val CHANNEL_NUM  = AbsoluteSensorNode.CHANNEL_NUM   // 6
    }

    /**
     * Stage 2 의 출력 단위. 불변(immutable) — 모든 필드는 read-only.
     */
    data class InferenceOutput(
        val dispLocal:        FloatArray,   // [Δx, Δy, Δz] m, yaw-free local frame
        val dispLogVar:       FloatArray,   // [σ²x_log, σ²y_log, σ²z_log] (log)
        val classProb:        FloatArray,   // [7] 휴대 자세 분류 확률
        val topClass:         Int,          // argmax 인덱스
        val className:        String,       // 휴대 자세 라벨
        val windowStartYaw:   Double,       // yaw0 (rad) — yaw-free → world 복원용
        val tsBeginUs:        Long,         // 윈도우 시작 ts
        val tsEndUs:          Long,         // 윈도우 끝 ts
        val rotAccuracyStart: Int           // 윈도우 시작 시점 RotVec 정확도 (0..3)
    )

    /**
     * 윈도우 한 개에 대해 추론을 수행한다. 순수 함수.
     *
     * @param windowSamples Stage 1 의 List<WorldSample>, 정확히 WINDOW_SIZE 개여야 함
     * @return InferenceOutput, 또는 입력 길이 불일치/엔진 미로드 시 null
     */
    fun infer(windowSamples: List<AbsoluteSensorNode.WorldSample>): InferenceOutput? {
        if (windowSamples.size != WINDOW_SIZE) {
            Log.w(TAG, "윈도우 길이 불일치: ${windowSamples.size} (기대 $WINDOW_SIZE)")
            return null
        }
        if (!engine.isLoaded()) {
            Log.w(TAG, "InferenceEngine 미로드 — 추론 건너뜀")
            return null
        }

        // ── 1. 윈도우 시작 yaw 추출 ────────────────────────────────
        // Android getRotationMatrixFromVector 결과는 R_world←body (row-major).
        //   index = row*3 + col  →  R[1,0] = rotMat[3], R[0,0] = rotMat[0]
        // ZYX 오일러 yaw = atan2(R[1,0], R[0,0])  (학습 dataset.py 와 동일 컨벤션)
        val startRot = windowSamples[0].rotMat
        val yaw0 = atan2(startRot[3].toDouble(), startRot[0].toDouble())
        val cosZ = cos(yaw0)
        val sinZ = sin(yaw0)

        // ── 2. yaw-free local frame 윈도우 생성 ─────────────────────
        // R_yaw_inv (z축 -yaw 회전):
        //   [ cos(yaw0)  sin(yaw0)  0 ]
        //   [-sin(yaw0)  cos(yaw0)  0 ]
        //   [ 0          0          1 ]
        // 적용:  v_ga[x] =  cosZ·v_w[x] + sinZ·v_w[y]
        //        v_ga[y] = -sinZ·v_w[x] + cosZ·v_w[y]
        //        v_ga[z] =  v_w[z]
        val flat = FloatArray(CHANNEL_NUM * WINDOW_SIZE)
        for (t in 0 until WINDOW_SIZE) {
            val s = windowSamples[t]
            val lx = s.worldLinAcc[0].toDouble()
            val ly = s.worldLinAcc[1].toDouble()
            val lz = s.worldLinAcc[2].toDouble()
            val gx = s.worldGyr[0].toDouble()
            val gy = s.worldGyr[1].toDouble()
            val gz = s.worldGyr[2].toDouble()

            // ch 0-2: acc_ga
            flat[0 * WINDOW_SIZE + t] = ( cosZ * lx + sinZ * ly).toFloat()
            flat[1 * WINDOW_SIZE + t] = (-sinZ * lx + cosZ * ly).toFloat()
            flat[2 * WINDOW_SIZE + t] = lz.toFloat()
            // ch 3-5: gyr_ga
            flat[3 * WINDOW_SIZE + t] = ( cosZ * gx + sinZ * gy).toFloat()
            flat[4 * WINDOW_SIZE + t] = (-sinZ * gx + cosZ * gy).toFloat()
            flat[5 * WINDOW_SIZE + t] = gz.toFloat()
        }

        // ── 3. 추론 (InferenceEngine 이 정규화 + 텐서화 + forward 수행) ──
        val raw = try {
            engine.infer(flat)
        } catch (e: Exception) {
            Log.w(TAG, "InferenceEngine.infer() 실패: ${e.message}")
            return null
        }

        // ── 4. 결과 포장 (방어적 복사) ──────────────────────────────
        return InferenceOutput(
            dispLocal        = raw.disp.copyOf(),
            dispLogVar       = raw.dispCov.copyOf(),
            classProb        = raw.clsProb.copyOf(),
            topClass         = raw.topClass,
            className        = raw.className,
            windowStartYaw   = yaw0,
            tsBeginUs        = windowSamples.first().ts_us,
            tsEndUs          = windowSamples.last().ts_us,
            rotAccuracyStart = windowSamples.first().rotAccuracy
        )
    }
}
