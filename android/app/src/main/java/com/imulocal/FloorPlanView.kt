package com.imulocal

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewConfiguration
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * [P84] 평면도 위 궤적 오버레이 뷰 (approach A — 측정/GT용).
 * ==========================================================
 * 기존 Naver Map(MapFragment) 의 "공간 오버레이" 역할을 정적 평면도 이미지로 대체.
 * 알고리즘은 건드리지 않고 *측정/검증*만 쉽게 만든다 (지도-보조 보정 아님).
 *
 * 좌표계 3단:
 *   world(m, ENU: x→동/우, y→북/위)  --[보정 transform]-->  image-px(평면도 비트맵, y↓)
 *   image-px  --[fit transform]-->  view-px(화면, y↓)
 *
 * 2점 보정 (similarity, 반사 포함 처리):
 *   ① 평면도에서 "시작 위치" 탭  → imageA  ↔ worldA = 궤적 첫 점
 *   ② 평면도에서 "끝 위치"   탭  → imageB  ↔ worldB = 궤적 마지막 점
 *   → 원점·회전·스케일 자동 산출. 실거리(m)를 알면 setRealDistance()로 스케일을 참값으로 교정
 *     (이 경우 궤적의 과소예측이 평면도 위에서 "못 미침"으로 그대로 드러남 = GT 신호).
 *
 * 체크포인트 GT:
 *   체크포인트 모드에서 랜드마크를 지날 때 평면도를 탭 → (탭한 image-px = 참위치,
 *   탭 순간 궤적 위치 = 추정위치, timestamp) 저장. 보정 후 image→world 역변환으로
 *   참위치를 world(m)로 환산해 체크포인트 ATE 계산(내보내기는 MainActivity).
 */
class FloorPlanView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class Mode { NONE, CALIBRATE, CHECKPOINT }

    // ── 데이터 ────────────────────────────────────────────────────
    private var bitmap: Bitmap? = null
    private var trackPoints: List<Pair<Double, Double>> = emptyList()
    private val checkpoints = mutableListOf<FloorCheckpoint>()

    // ── 보정(world→image) 파라미터 [P84 방식A] ───────────────────
    //   image = imgO + s·Rot(theta)·world'(원점 이동·y반전).  s=px/m, theta=회전.
    private var calibrated = false
    private var imgOx = 0f; private var imgOy = 0f          // world 원점이 놓이는 image-px
    private var wOx = 0.0;  private var wOy = 0.0           // world 원점(궤적 시작점)
    private var sPxPerM = 50.0                              // image-px per world-meter (①스케일 단계)
    private var theta = 0.0                                 // 회전(rad) (②시작·방향 단계)

    // 보정 단계: 0=대기, 1=스케일(거리 2점), 2=시작·방향(2점)
    private var calibPhase = 0
    private var pendingScalePix = 0.0                       // 스케일 2점의 픽셀거리(실거리 입력 대기)
    // 현재 단계의 탭 누적 (image-px)
    private val calibTaps = mutableListOf<Pair<Float, Float>>()

    var mode: Mode = Mode.NONE
        private set

    // ── [P84] 확대/이동 (pinch zoom + drag pan) ───────────────────
    private var userScale = 1f
    private var userTransX = 0f
    private var userTransY = 0f
    // [P84-fix] 최소 줌 = 1.0(화면맞춤) — fit 이하 과축소로 가독성 떨어지는 문제 방지
    private val scaleDetector = ScaleGestureDetector(context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                val newScale = (userScale * d.scaleFactor).coerceIn(1f, 12f)
                val applied = newScale / userScale
                // 핀치 초점(focus) 아래 지점이 고정되도록 이동량 보정
                userTransX = d.focusX - (d.focusX - userTransX) * applied
                userTransY = d.focusY - (d.focusY - userTransY) * applied
                userScale = newScale
                clampUserTransform()
                invalidate()
                return true
            }
        })

    // [P84-fix] 두 번 탭 = 화면맞춤 초기화 (표시 모드에서만 — 보정/체크포인트 탭과 충돌 방지)
    private val gestureDetector = GestureDetector(context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (mode == Mode.NONE) { resetView(); return true }
                return false
            }
        })
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop
    private var isDragging = false
    private var lastTouchX = 0f; private var lastTouchY = 0f
    private var downTouchX = 0f; private var downTouchY = 0f

    /** 확대/이동 초기화 (새 평면도 로드·두 번 탭 시 호출) — 화면맞춤 정렬로 복귀. */
    fun resetView() { userScale = 1f; userTransX = 0f; userTransY = 0f; invalidate() }

    // [P84-fix] 화면 크기 변경(회전 등) 시 화면맞춤으로 재정렬
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        resetView()
    }

    /**
     * [P84-fix] 사용자 변환 클램프 — 평면도가 화면 밖으로 사라져 정렬을 잃는 문제 방지.
     * 스케일된 이미지가 화면보다 작은 축은 가운데 고정, 큰 축은 가장자리가 화면 안쪽으로
     * 들어오지 못하게 제한(여백 금지).
     */
    private fun clampUserTransform() {
        val f = fitParams()
        val vw = width.toFloat(); val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return
        val dw = f.dst.width() * userScale
        val dh = f.dst.height() * userScale
        var left = userTransX + userScale * f.dst.left
        var top = userTransY + userScale * f.dst.top
        left = if (dw <= vw) (vw - dw) / 2f else left.coerceIn(vw - dw, 0f)
        top = if (dh <= vh) (vh - dh) / 2f else top.coerceIn(vh - dh, 0f)
        userTransX = left - userScale * f.dst.left
        userTransY = top - userScale * f.dst.top
    }

    // ── 콜백 (Activity 가 다이얼로그/토스트 처리) ─────────────────
    /** ①스케일 2점 탭 완료 → Activity 가 실거리(m) 입력 다이얼로그 표시. 인자=두 점 픽셀거리. */
    var onScalePointsReady: ((pixDist: Double) -> Unit)? = null
    /** 보정(②시작·방향까지) 완료 시 호출. */
    var onCalibrationReady: (() -> Unit)? = null
    /** 체크포인트 추가 시 호출 — 인자는 추가된 체크포인트 인덱스. */
    var onCheckpointAdded: ((index: Int) -> Unit)? = null

    // ── 페인트 ────────────────────────────────────────────────────
    private val pathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1565C0"); strokeWidth = 5f
        style = Paint.Style.STROKE; strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND
    }
    private val curDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1565C0"); style = Paint.Style.FILL
    }
    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#388E3C"); style = Paint.Style.FILL
    }
    private val cpTruePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#D50000"); strokeWidth = 5f; style = Paint.Style.STROKE
    }
    private val cpEstPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF6D00"); style = Paint.Style.FILL
    }
    private val errLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#D50000"); strokeWidth = 2.5f; style = Paint.Style.STROKE
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(8f, 6f), 0f)
    }
    private val calibPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#6A1B9A"); strokeWidth = 4f; style = Paint.Style.STROKE
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#212121"); textSize = 26f
    }
    private val bannerBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(210, 33, 33, 33); style = Paint.Style.FILL
    }
    private val bannerText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 30f
    }
    private val hintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#9E9E9E"); textSize = 30f; textAlign = Paint.Align.CENTER
    }

    // ── 외부 API ─────────────────────────────────────────────────
    fun setFloorPlan(bmp: Bitmap?) { bitmap = bmp; resetView(); invalidate() }
    fun hasFloorPlan(): Boolean = bitmap != null

    fun updateTrajectory(points: List<Pair<Double, Double>>) {
        trackPoints = points
        invalidate()
    }

    fun setMode(m: Mode) {
        mode = m
        if (m == Mode.CALIBRATE) { calibPhase = 1; calibTaps.clear() } else { calibPhase = 0 }
        invalidate()
    }

    fun isCalibrated(): Boolean = calibrated
    fun getCheckpoints(): List<FloorCheckpoint> = checkpoints.toList()

    fun clearCheckpoints() { checkpoints.clear(); invalidate() }

    /** Reset 버튼용 — 궤적·체크포인트·보정 모두 클리어(평면도 비트맵은 유지). */
    fun clearPath() {
        trackPoints = emptyList()
        checkpoints.clear()
        calibTaps.clear()
        calibrated = false
        mode = Mode.NONE
        invalidate()
    }

    /** ①스케일 단계: 두 점 실거리(m) 입력 → px/m 확정 후 ②시작·방향 단계로 진행. */
    fun setScaleFromRealDistance(realDistM: Double) {
        if (calibPhase != 1 || realDistM <= 0.0 || pendingScalePix < 1e-6) return
        sPxPerM = pendingScalePix / realDistM
        calibPhase = 2
        calibTaps.clear()
        invalidate()
    }

    /** 보정 취소 (다이얼로그 취소 등). */
    fun cancelCalibration() {
        calibPhase = 0
        calibTaps.clear()
        mode = Mode.NONE
        invalidate()
    }

    /** image-px → world(m) 역변환. 미보정이면 null. */
    fun imageToWorld(ix: Float, iy: Float): Pair<Double, Double>? {
        if (!calibrated) return null
        val dx = (ix - imgOx).toDouble(); val dy = (iy - imgOy).toDouble()
        val rx = dx / sPxPerM; val ry = dy / sPxPerM
        // Rot(-θ)
        val px = cos(theta) * rx + sin(theta) * ry
        val py = -sin(theta) * rx + cos(theta) * ry
        val wx = wOx + px
        val wy = wOy - py            // world y-up ↔ worldO'=(wOx,-wOy)
        return Pair(wx, wy)
    }

    // ── world(m) → image-px (보정 transform) ─────────────────────
    private fun worldToImage(wx: Double, wy: Double): Pair<Float, Float> {
        // worldO' = (wOx, -wOy), p = (wx,-wy) - worldO'
        val px = wx - wOx
        val py = -wy + wOy
        val rx = cos(theta) * px - sin(theta) * py
        val ry = sin(theta) * px + cos(theta) * py
        return Pair((imgOx + sPxPerM * rx).toFloat(), (imgOy + sPxPerM * ry).toFloat())
    }

    // ── image-px ↔ view-px (비트맵 fit-center) ───────────────────
    private data class Fit(val scale: Float, val offX: Float, val offY: Float, val dst: RectF)

    private fun fitParams(): Fit {
        val bmp = bitmap
        val vw = width.toFloat(); val vh = height.toFloat()
        if (bmp == null || bmp.width == 0 || bmp.height == 0) return Fit(1f, 0f, 0f, RectF(0f, 0f, vw, vh))
        val sc = minOf(vw / bmp.width, vh / bmp.height)
        val dw = bmp.width * sc; val dh = bmp.height * sc
        val ox = (vw - dw) / 2f; val oy = (vh - dh) / 2f
        return Fit(sc, ox, oy, RectF(ox, oy, ox + dw, oy + dh))
    }

    private fun imgToViewX(ix: Float, f: Fit) = f.offX + ix * f.scale
    private fun imgToViewY(iy: Float, f: Fit) = f.offY + iy * f.scale
    private fun viewToImgX(vx: Float, f: Fit) = (vx - f.offX) / f.scale
    private fun viewToImgY(vy: Float, f: Fit) = (vy - f.offY) / f.scale

    // ── 그리기 ───────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.parseColor("#FAFAFA"))
        val bmp = bitmap
        if (bmp == null) {
            canvas.drawText("평면도를 불러오세요 (메뉴 → 평면도 불러오기)", width / 2f, height / 2f, hintPaint)
            drawBanner(canvas)
            return
        }

        val f = fitParams()
        // [P84] 사용자 확대/이동 적용 — 이하 도형은 base(fit) 좌표로 그리면 화면에 변환됨
        canvas.save()
        canvas.translate(userTransX, userTransY)
        canvas.scale(userScale, userScale)

        canvas.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), f.dst, null)

        if (calibrated && trackPoints.size >= 2) {
            val path = Path()
            val p0 = worldToImage(trackPoints[0].first, trackPoints[0].second)
            path.moveTo(imgToViewX(p0.first, f), imgToViewY(p0.second, f))
            for (i in 1 until trackPoints.size) {
                val p = worldToImage(trackPoints[i].first, trackPoints[i].second)
                path.lineTo(imgToViewX(p.first, f), imgToViewY(p.second, f))
            }
            canvas.drawPath(path, pathPaint)
            canvas.drawCircle(imgToViewX(p0.first, f), imgToViewY(p0.second, f), 12f, startPaint)
            val pl = worldToImage(trackPoints.last().first, trackPoints.last().second)
            canvas.drawCircle(imgToViewX(pl.first, f), imgToViewY(pl.second, f), 13f, curDotPaint)
        }

        // 체크포인트 (참위치=빨강 X, 보정 시 추정위치까지 점선 + 오차)
        checkpoints.forEachIndexed { idx, cp ->
            val tx = imgToViewX(cp.imgX, f); val ty = imgToViewY(cp.imgY, f)
            val r = 14f
            canvas.drawLine(tx - r, ty - r, tx + r, ty + r, cpTruePaint)
            canvas.drawLine(tx - r, ty + r, tx + r, ty - r, cpTruePaint)
            canvas.drawText("C${idx + 1}", tx + r + 4f, ty + 8f, labelPaint)
            if (calibrated) {
                val est = worldToImage(cp.estX, cp.estY)
                val ex = imgToViewX(est.first, f); val ey = imgToViewY(est.second, f)
                canvas.drawCircle(ex, ey, 7f, cpEstPaint)
                canvas.drawLine(ex, ey, tx, ty, errLinePaint)
                imageToWorld(cp.imgX, cp.imgY)?.let { tw ->
                    val err = hypot(tw.first - cp.estX, tw.second - cp.estY)
                    canvas.drawText(String.format("%.1fm", err), (ex + tx) / 2f, (ey + ty) / 2f - 6f, labelPaint)
                }
            }
        }

        // 보정 진행 중 탭 미리보기
        if (mode == Mode.CALIBRATE) {
            calibTaps.forEachIndexed { i, (ix, iy) ->
                val vx = imgToViewX(ix, f); val vy = imgToViewY(iy, f)
                canvas.drawCircle(vx, vy, 12f, calibPaint)
                val lab = if (calibPhase == 1) (if (i == 0) "거리1" else "거리2")
                          else (if (i == 0) "시작" else "방향")
                canvas.drawText(lab, vx + 14f, vy, labelPaint)
            }
        }

        canvas.restore()

        if (!calibrated && trackPoints.size >= 2) {
            canvas.drawText("보정 필요 — 메뉴 → 평면도 보정", width / 2f, height - 40f, hintPaint)
        }
        drawBanner(canvas)
    }

    private fun drawBanner(canvas: Canvas) {
        val msg = when (mode) {
            Mode.CALIBRATE -> when (calibPhase) {
                1 -> if (calibTaps.isEmpty()) "보정 ①스케일: 거리 아는 두 점 중 첫 점 탭"
                     else "보정 ①스케일: 두 번째 점 탭"
                2 -> if (calibTaps.isEmpty()) "보정 ②위치: 출발한 지점 탭"
                     else "보정 ②방향: 처음 걸어간 방향으로 한 점 더 탭"
                else -> "보정 중"
            }
            Mode.CHECKPOINT -> "체크포인트 모드 — 랜드마크 지날 때 평면도를 탭 (누적 ${checkpoints.size})"
            Mode.NONE -> if (!calibrated && bitmap != null) "표시 모드 — 핀치 확대 · 드래그 이동 · 두 번 탭 = 화면맞춤" else ""
        }
        if (msg.isEmpty()) return
        val pad = 16f
        val tw = bannerText.measureText(msg)
        val h = 52f
        canvas.drawRect(0f, 0f, minOf(width.toFloat(), tw + pad * 2), h, bannerBg)
        canvas.drawText(msg, pad, h - 18f, bannerText)
    }

    // ── 탭/드래그/핀치 처리 ───────────────────────────────────────
    //   핀치(두 손가락)=확대, 드래그(한 손가락 이동)=이동, 탭(이동 없음)=모드별 탭.
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downTouchX = event.x; downTouchY = event.y
                lastTouchX = event.x; lastTouchY = event.y
                isDragging = false
            }
            MotionEvent.ACTION_MOVE -> {
                if (!scaleDetector.isInProgress) {
                    if (!isDragging &&
                        hypot((event.x - downTouchX).toDouble(), (event.y - downTouchY).toDouble()) > touchSlop) {
                        isDragging = true
                    }
                    if (isDragging) {
                        userTransX += event.x - lastTouchX
                        userTransY += event.y - lastTouchY
                        lastTouchX = event.x; lastTouchY = event.y
                        clampUserTransform()
                        invalidate()
                    }
                }
            }
            MotionEvent.ACTION_UP -> {
                if (!isDragging && !scaleDetector.isInProgress) {
                    // 탭 → 화면좌표에서 사용자변환·fit 을 역산해 image-px 로
                    val f = fitParams()
                    val baseX = (event.x - userTransX) / userScale
                    val baseY = (event.y - userTransY) / userScale
                    val ix = (baseX - f.offX) / f.scale
                    val iy = (baseY - f.offY) / f.scale
                    when (mode) {
                        Mode.CALIBRATE -> handleCalibTap(ix, iy)
                        Mode.CHECKPOINT -> handleCheckpointTap(ix, iy)
                        Mode.NONE -> { /* 표시 전용 */ }
                    }
                }
            }
        }
        return true
    }

    private fun handleCalibTap(ix: Float, iy: Float) {
        when (calibPhase) {
            1 -> {  // ①스케일: 거리 아는 두 점
                calibTaps.add(Pair(ix, iy))
                invalidate()
                if (calibTaps.size >= 2) {
                    pendingScalePix = hypot(
                        (calibTaps[1].first - calibTaps[0].first).toDouble(),
                        (calibTaps[1].second - calibTaps[0].second).toDouble()
                    )
                    onScalePointsReady?.invoke(pendingScalePix)   // Activity → 실거리 입력 다이얼로그
                }
            }
            2 -> {  // ②시작·방향
                if (trackPoints.size < 2) return
                calibTaps.add(Pair(ix, iy))
                invalidate()
                if (calibTaps.size >= 2) {
                    computePose()
                    mode = Mode.NONE
                    calibPhase = 0
                    onCalibrationReady?.invoke()
                }
            }
        }
    }

    /** ②시작 위치(탭1)=원점, 시작→방향(탭2) 벡터를 궤적 초기 진행방향과 맞춰 회전 결정. */
    private fun computePose() {
        imgOx = calibTaps[0].first; imgOy = calibTaps[0].second
        wOx = trackPoints.first().first; wOy = trackPoints.first().second
        // 궤적 초기 진행방향 — 시작점에서 1m 이상 떨어진 첫 점(없으면 마지막 점)
        var jx = trackPoints.last().first; var jy = trackPoints.last().second
        for (p in trackPoints) {
            if (hypot(p.first - wOx, p.second - wOy) > 1.0) { jx = p.first; jy = p.second; break }
        }
        val dWx = jx - wOx; val dWy = jy - wOy
        val dIx = (calibTaps[1].first - imgOx).toDouble()
        val dIy = (calibTaps[1].second - imgOy).toDouble()
        if (hypot(dWx, dWy) < 1e-6 || hypot(dIx, dIy) < 1e-6) { calibrated = false; return }
        // world'=(x,-y): 초기방향 각도와 image 방향 각도의 차 = 회전
        theta = atan2(dIy, dIx) - atan2(-dWy, dWx)
        calibrated = true
        calibTaps.clear()
        invalidate()
    }

    private fun handleCheckpointTap(ix: Float, iy: Float) {
        val est = trackPoints.lastOrNull() ?: Pair(0.0, 0.0)
        checkpoints.add(
            FloorCheckpoint(
                imgX = ix, imgY = iy,
                estX = est.first, estY = est.second,
                tMs = System.currentTimeMillis()
            )
        )
        invalidate()
        onCheckpointAdded?.invoke(checkpoints.size - 1)
    }
}

/** [P84] 체크포인트 1건 — image-px(참위치) + 탭 순간 궤적 추정위치 + 시각. */
data class FloorCheckpoint(
    val imgX: Float,
    val imgY: Float,
    val estX: Double,
    val estY: Double,
    val tMs: Long,
    var label: String = ""
)
