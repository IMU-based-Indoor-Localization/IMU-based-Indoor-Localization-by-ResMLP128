package com.imulocal

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * TrackView.kt
 * ============
 * 실내 이동 경로를 실시간으로 캔버스에 그리는 커스텀 뷰.
 *
 * 두 궤적을 동시에 표시:
 *   - EKF 궤적   (파랑 #1565C0): 모델 + SC-EKF 보정 위치
 *   - 모델 궤적  (주황 #E65100): 모델 추론값 누적 (EKF 없음)
 *
 * 두 궤적의 합집합 Bounding Box 로 자동 스케일링.
 */
class TrackView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

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
        val pad = 48f

        canvas.drawColor(Color.WHITE)
        drawGrid(canvas, w, h)

        // 두 궤적 모두 점이 1개 미만이면 범례만 그리고 종료
        val hasEkf   = ekfPoints.size   >= 2
        val hasModel = modelPoints.size >= 2

        if (!hasEkf && !hasModel) {
            drawLegend(canvas, w, h)
            return
        }

        // ── 합집합 Bounding Box ───────────────────────────────────
        val allXs = mutableListOf<Double>()
        val allYs = mutableListOf<Double>()
        if (hasEkf)   { allXs += ekfPoints.map   { it.first }; allYs += ekfPoints.map   { it.second } }
        if (hasModel) { allXs += modelPoints.map  { it.first }; allYs += modelPoints.map { it.second } }

        val minX = allXs.min(); val maxX = allXs.max()
        val minY = allYs.min(); val maxY = allYs.max()

        // 최소 표시 범위: 실내 공간 기준 10m × 10m
        // → 작은 오차가 화면 전체를 차지하는 과도한 확대 방지
        // 궤적이 10m 초과 시 자동 축소
        val MIN_RANGE_M = 10.0
        val rangeX = (maxX - minX).coerceAtLeast(MIN_RANGE_M)
        val rangeY = (maxY - minY).coerceAtLeast(MIN_RANGE_M)
        val scaleX = (w - 2 * pad) / rangeX
        val scaleY = (h - 2 * pad) / rangeY
        val scale  = minOf(scaleX, scaleY)

        val offsetX = pad + ((w - 2 * pad) - rangeX * scale) / 2f
        val offsetY = pad + ((h - 2 * pad) - rangeY * scale) / 2f

        fun toPixX(x: Double) = (offsetX + (x - minX) * scale).toFloat()
        fun toPixY(y: Double) = (h - offsetY - (y - minY) * scale).toFloat()   // Y축 반전

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
    }

    // ── 범례 (우상단) ─────────────────────────────────────────────
    private fun drawLegend(canvas: Canvas, w: Float, h: Float) {
        val items = listOf(
            Pair(Color.parseColor("#1565C0"), "모델 + EKF"),
            Pair(Color.parseColor("#E65100"), "모델 only"),
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

    private fun drawGrid(canvas: Canvas, w: Float, h: Float) {
        val step = 80f
        var x = 0f
        while (x <= w) { canvas.drawLine(x, 0f, x, h, gridPaint); x += step }
        var y = 0f
        while (y <= h) { canvas.drawLine(0f, y, w, y, gridPaint); y += step }
    }
}
