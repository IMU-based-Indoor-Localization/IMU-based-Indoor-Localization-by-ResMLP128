package com.imulocal

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.ceil

/**
 * TrackView.kt
 * ============
 * 실내 이동 경로를 실시간으로 캔버스에 그리는 커스텀 뷰.
 *
 * 궤적 표시 (파랑 #1565C0 = 측위 궤적):
 *   - 경로 B(RotVec DR, 현재 기본): 추정 궤적 1개. modelPoints 는 비어 있음.
 *   - 경로 A(EKF, USE_ROTVEC_DR=false): EKF 궤적 + 모델 궤적(주황) 2개 비교 표시.
 *
 * 표시 궤적의 합집합 Bounding Box 로 자동 스케일링.
 */
class TrackView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ── [P66] 1m 고정 스케일 — 좁은 시연 (~5×5m) 직관 ↑ ───────────
    companion object {
        /** 1m = N 픽셀 (고정). 화면 가로 ~1080px → 표시 가능 영역 약 21m. */
        const val PX_PER_M = 50f
    }

    // ── 페인트 ────────────────────────────────────────────────────
    private val ekfPathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color       = Color.parseColor("#1565C0")   // 파랑: EKF 보정
        strokeWidth = 4f
        style       = Paint.Style.STROKE
        strokeJoin  = Paint.Join.ROUND
        strokeCap   = Paint.Cap.ROUND
    }
    private val modelPathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color       = Color.parseColor("#E65100")   // 주황: 모델 only
        strokeWidth = 3f
        style       = Paint.Style.STROKE
        strokeJoin  = Paint.Join.ROUND
        strokeCap   = Paint.Cap.ROUND
        alpha       = 200
    }
    private val ekfDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1565C0")
        style = Paint.Style.FILL
    }
    private val modelDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E65100")
        style = Paint.Style.FILL
    }
    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#388E3C")         // 초록: 공통 시작점
        style = Paint.Style.FILL
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color       = Color.parseColor("#E0E0E0")
        strokeWidth = 1f
        style       = Paint.Style.STROKE
    }

    // ── 범례 페인트 ───────────────────────────────────────────────
    private val legendBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 255, 255, 255)
        style = Paint.Style.FILL
    }
    private val legendLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 6f
        style       = Paint.Style.STROKE
        strokeCap   = Paint.Cap.ROUND
    }
    private val legendTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.parseColor("#212121")
        textSize = 28f
    }
    // [P65] 격자 1칸의 m 단위 표시 (자동 스케일링이라 가변)
    private val gridInfoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.parseColor("#616161")
        textSize = 26f
    }

    // ── 데이터 ────────────────────────────────────────────────────
    private var ekfPoints:   List<Pair<Double, Double>> = emptyList()
    private var modelPoints: List<Pair<Double, Double>> = emptyList()

    // ── 외부 API ─────────────────────────────────────────────────
    /**
     * @param ekf   EKF 보정 궤적 (파랑)
     * @param model 모델 단독 궤적 (주황)
     */
    fun updatePaths(
        ekf:   List<Pair<Double, Double>>,
        model: List<Pair<Double, Double>>
    ) {
        ekfPoints   = ekf
        modelPoints = model
        invalidate()
    }

    fun clearPath() {
        ekfPoints   = emptyList()
        modelPoints = emptyList()
        invalidate()
    }

    // ── 그리기 ───────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w   = width.toFloat()
        val h   = height.toFloat()

        canvas.drawColor(Color.WHITE)

        // 두 궤적 모두 점이 1개 미만이면 격자만 그리고 종료
        val hasEkf   = ekfPoints.size   >= 2
        val hasModel = modelPoints.size >= 2

        if (!hasEkf && !hasModel) {
            // [P66] 시작점도 없으면 화면 중앙 (0,0) 기준 격자만
            drawGrid(canvas, w, h, w / 2f, h / 2f)
            drawLegend(canvas, w, h)
            canvas.drawText("격자 ≈ 1.00 m/칸", 12f, h - 12f, gridInfoPaint)
            return
        }

        // ── 합집합 Bounding Box ───────────────────────────────────
        val allXs = mutableListOf<Double>()
        val allYs = mutableListOf<Double>()
        if (hasEkf)   { allXs += ekfPoints.map   { it.first }; allYs += ekfPoints.map   { it.second } }
        if (hasModel) { allXs += modelPoints.map  { it.first }; allYs += modelPoints.map { it.second } }

        val minX = allXs.min(); val maxX = allXs.max()
        val minY = allYs.min(); val maxY = allYs.max()
        val centerX = (minX + maxX) / 2.0
        val centerY = (minY + maxY) / 2.0

        // [P66] 1m = PX_PER_M 픽셀 고정 — 좁은 시연 (~5×5m) 직관 ↑.
        //   bbox 중심을 화면 중앙으로 자동 panning. 큰 측정 시 화면 밖 짤림.
        //   PX_PER_M 50 ≈ 화면 가로 19m, 세로 ~18m 표시 가능.
        val scale = PX_PER_M.toDouble()

        // 화면 중앙 = bbox 중심
        val cx0 = w / 2f
        val cy0 = h / 2f

        fun toPixX(x: Double) = (cx0 + (x - centerX) * scale).toFloat()
        fun toPixY(y: Double) = (cy0 - (y - centerY) * scale).toFloat()   // Y축 반전

        // [P66] 격자도 1m 단위로 그림 — 시작점 위치 기준 정렬
        drawGrid(canvas, w, h, toPixX(0.0), toPixY(0.0))

        // ── 모델 only 궤적 (먼저 그려 EKF 위에 덮이지 않게) ─────
        if (hasModel) {
            val path = Path()
            path.moveTo(toPixX(modelPoints[0].first), toPixY(modelPoints[0].second))
            for (i in 1 until modelPoints.size) {
                path.lineTo(toPixX(modelPoints[i].first), toPixY(modelPoints[i].second))
            }
            canvas.drawPath(path, modelPathPaint)
            // 현재 위치 점
            canvas.drawCircle(
                toPixX(modelPoints.last().first),
                toPixY(modelPoints.last().second),
                10f, modelDotPaint
            )
        }

        // ── EKF 궤적 ─────────────────────────────────────────────
        if (hasEkf) {
            val path = Path()
            path.moveTo(toPixX(ekfPoints[0].first), toPixY(ekfPoints[0].second))
            for (i in 1 until ekfPoints.size) {
                path.lineTo(toPixX(ekfPoints[i].first), toPixY(ekfPoints[i].second))
            }
            canvas.drawPath(path, ekfPathPaint)
            // 현재 위치 점
            canvas.drawCircle(
                toPixX(ekfPoints.last().first),
                toPixY(ekfPoints.last().second),
                13f, ekfDotPaint
            )
        }

        // ── 공통 시작점 (초록 원) ─────────────────────────────────
        val startX: Double
        val startY: Double
        if (hasEkf) {
            startX = ekfPoints.first().first
            startY = ekfPoints.first().second
        } else {
            startX = modelPoints.first().first
            startY = modelPoints.first().second
        }
        canvas.drawCircle(toPixX(startX), toPixY(startY), 11f, startPaint)

        drawLegend(canvas, w, h)
        // [P66] 격자 1m 고정 — drawGrid step = PX_PER_M.
        canvas.drawText(
            "격자 = 1.00 m/칸",
            12f, h - 12f, gridInfoPaint
        )
    }

    // ── 범례 (우상단) ─────────────────────────────────────────────
    // [P56] 경로 B(RotVec DR)는 추정 궤적이 하나 — EKF 미사용, 단일 궤적으로 표기.
    private fun drawLegend(canvas: Canvas, w: Float, h: Float) {
        val items = listOf(
            Pair(Color.parseColor("#1565C0"), "측위 궤적"),
            Pair(Color.parseColor("#388E3C"), "시작점")
        )
        val lineLen  = 36f
        val itemH    = 34f
        val padLR    = 12f
        val padTB    = 10f
        val textOff  = lineLen + 8f
        val maxTextW = items.maxOf { legendTextPaint.measureText(it.second) }
        val boxW     = padLR + lineLen + 8f + maxTextW + padLR
        val boxH     = padTB + itemH * items.size + padTB

        val left   = w - boxW - 12f
        val top    = 12f
        canvas.drawRoundRect(RectF(left, top, left + boxW, top + boxH), 8f, 8f, legendBgPaint)

        items.forEachIndexed { idx, (color, label) ->
            val cy = top + padTB + itemH * idx + itemH / 2f
            legendLinePaint.color = color
            canvas.drawLine(left + padLR, cy, left + padLR + lineLen, cy, legendLinePaint)
            legendTextPaint.color = Color.parseColor("#212121")
            canvas.drawText(label, left + padLR + textOff, cy + legendTextPaint.textSize * 0.35f, legendTextPaint)
        }
    }

    /**
     * [P66] 1m 단위 격자 — step = PX_PER_M, 시작점(originPx) 기준 정렬.
     * 5m 마다 굵은 선 + m 단위 라벨로 거리 직관 ↑.
     */
    private fun drawGrid(canvas: Canvas, w: Float, h: Float, originPxX: Float, originPxY: Float) {
        val step = PX_PER_M
        val gridBold = Paint(gridPaint).apply { color = Color.parseColor("#BDBDBD"); strokeWidth = 1.5f }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#9E9E9E"); textSize = 22f
        }
        // x 축 — 좌우로 격자 (origin 기준 −N, +N)
        var i = -ceil((originPxX / step).toDouble()).toInt()
        while (true) {
            val x = originPxX + i * step
            if (x > w) break
            val bold = (i % 5 == 0)
            canvas.drawLine(x, 0f, x, h, if (bold) gridBold else gridPaint)
            if (bold && i != 0) {
                canvas.drawText("${i}m", x + 2f, h - 30f, labelPaint)
            }
            i++
        }
        // y 축 — 상하로 격자
        i = -ceil((originPxY / step).toDouble()).toInt()
        while (true) {
            val y = originPxY + i * step
            if (y > h) break
            val bold = (i % 5 == 0)
            canvas.drawLine(0f, y, w, y, if (bold) gridBold else gridPaint)
            if (bold && i != 0) {
                // Y 는 화면 좌표(아래로 증가) → 실제 m 부호 반전
                canvas.drawText("${-i}m", 4f, y - 4f, labelPaint)
            }
            i++
        }
    }
}
