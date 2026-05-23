package com.imulocal

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * LocalizationViewModel.kt
 * ========================
 * ?깆쓽 ?듭떖 痢≪쐞 ?뚯씠?꾨씪??
 *
 *  1. ImuCollector  ??100Hz IMU ?섏쭛
 *  2. InferenceEngine ??20Hz ?ㅽ듃?뚰겕 異붾줎 (蹂??+ 遺꾨쪟)
 *  3. EkfBridge (JNI) ??SC-EKF ?곹깭 媛깆떊
 *  4. UI ?ㅻ젅?쒕줈 ?꾩튂/寃쎈줈 諛⑹텧
 *
 * ?? 踰꾧렇 ?섏젙 ?ы빆 ?????????????????????????????????????????
 *  ????꾩뒪?ы봽: System.currentTimeMillis() ?꾩쟾 ?쒓굅.
 *    紐⑤뱺 ?대줎/媛깆떊 ??꾩뒪?ы봽??SensorEvent.timestamp(遺??寃쎄낵 ns?믊펣) ?ъ슜.
 *
 *  ???섑뵆 ?ㅽ궢: windowReady.collect + getLatestSample() 諛⑹떇 ???먭린.
 *    propJob ? drainPropagateQueue() 濡?5ms 留덈떎 紐⑤뱺 100Hz ?섑뵆 泥섎━.
 *
 *  ??異붾줎 ??대컢: delay(50ms) 怨좎젙 ??寃쎄낵 ?쒓컙 蹂댁젙 諛⑹떇?쇰줈 援먯껜.
 *    猷⑦봽 ?쒖옉?쒓컖 湲곕줉 ??runInferStep() ?꾨즺 ???섎㉧吏 ?쒓컙留??湲?
 *
 *  ???대줎 ?? SC-EKF update() ??t_begin, t_end ?묒そ 紐⑤몢
 *    si_timestamps_us ??議댁옱?댁빞 ????留?異붾줎留덈떎 t_end ?대줎 ?쎌엯,
 *    cloneChannel ?먯꽌 ~1珥??댁쟾 ?대줎??t_begin ?쇰줈 ?먯깋.
 *
 * ?? P3/P4 ?섏젙 ?????????????????????????????????????????????
 *  P3: cloneHistory(ArrayDeque + synchronized) ??Channel<Long>.
 *      propJob ??Channel ??send, inferJob ??濡쒖뺄 history ??drain.
 *      ??肄붾（?댁씠 怨듭쑀 裕ㅽ뀓???놁씠 ?듭떊 ????寃쎌웳 ?꾩쟾 ?쒓굅.
 *  P4: CLONE_SETTLE_MS 20ms ??30ms (湲곌린 遺??????대컢 ?ъ쑀 利앷?).
 *
 * ?? P1 ?섏젙 ????????????????????????????????????????????????
 *  transformWindowToWorldFrame() ?먯꽌 ?먯씠濡??곷텇 ??
 *  EKF bg ?명뼢??李④컧?섏뿬 Python scekf.py ?숈옉怨??쇱튂.
 *
 * ?? P5 ?섏젙 (?뺤? ?몄씠利?/ ?대룞?믪젙吏 ?꾨━利? ?????????????????
 *  ?먯씤: pendingCloneTs.set() ???뺤? ?먯젙蹂대떎 癒쇱? ?ㅽ뻾?섏뼱
 *        ?뺤? 以묒뿉???대줎??C++ EKF ??20Hz 濡??쎌엯?섏?留?
 *        marginalize() ???몄텧?섏? ?딆쓬 ???대줎 臾댄븳 ?꾩쟻.
 *        ?꾩쟻???대줎??propagate() ??O(N짼) ?됰젹 ?곗궛????쬆?쒖폒
 *        Default ?붿뒪?⑥쿂 ?ㅻ젅???ы솕 ?????꾨━利??좊컻.
 *  ?섏젙:
 *    ??runInferStep ?먯꽌 ?뺤? ?먯젙??pendingCloneTs.set() 蹂대떎 ?욎쑝濡??대룞.
 *    ??STATIC 釉뚮옖移? pendingCloneTs = -1L 濡??좉퇋 ?대줎 李⑤떒.
 *    ??STATIC 釉뚮옖移? Channel ?붿뿬 ?대줎 drain ??C++ ?대줎 ?꾨? 二쇰???
 *       ???뺤? ?곹깭?먯꽌 EKF ?곹깭 踰≫꽣瑜?理쒖냼 ?ш린濡??좎?.
 *
 * ?? P6 ?섏젙 (肄붾（??yield 蹂댁옣) ??????????????????????????????
 *  ?먯씤: inferJob 猷⑦봽?먯꽌 elapsedMs ??INFER_INTERVAL_MS ?대㈃
 *        delay() 瑜??몄텧?섏? ?딆븘 Default ?ㅻ젅?쒕? ?묐낫?섏? ?딆쓬.
 *  ?섏젙: delay(remaining.coerceAtLeast(1L)) 濡???긽 理쒖냼 1ms yield.
 */
class LocalizationViewModel(application: Application) : AndroidViewModel(application) {

    /**
     * [P60] EKF 비교 모드 — class 본문 nested enum.
     *  PATH_B       : 데모 기본. RotVec DR + PDR-hybrid (EKF 미사용).
     *  EKF_CURRENT  : 경로 A — 단말 현재 cfg (EkfBridge.DEFAULT_PARAMS).
     *  EKF_TLIO     : 경로 A — TLIO 논문 §V-D/§V-E cfg (EkfBridge.TLIO_PARAMS).
     */
    enum class EkfMode { PATH_B, EKF_CURRENT, EKF_TLIO }

    companion object {
        private const val TAG = "LocalizationVM"

        /** 異붾줎 猷⑦봽 紐⑺몴 二쇨린: 20Hz */
        private const val INFER_INTERVAL_MS  = 50L

        /** ?대줎 ?쎌엯 ?湲??쒓컙 (propJob 泥섎━ ?ъ쑀 ?뺣낫).
         *  P4 ?섏젙: 20ms ??30ms (湲곌린 遺????誘몄궫??鍮덈룄 媛먯냼) */
        private const val CLONE_SETTLE_MS    = 100L  // [DEBUG-1] 30 -> 100ms (clone ts mismatch fix)

        /** propJob ?쒕젅???대쭅 二쇨린 */
        private const val PROP_POLL_MS       = 5L

        /**
         * ?ㅽ듃?뚰겕 ?덈룄??吏???쒓컙 (關s).
         * 100?섑뵆 횞 10,000關s/?섑뵆 = 1,000,000關s ?댁?留?
         * getWindow() ??first.ts 瑜?諛섑솚?섎?濡?
         * last.ts ??first.ts ??99 횞 10,000 = 990,000關s.
         */
        private const val WINDOW_DURATION_US = 990_000L

        /** cloneChannel 踰꾪띁 ?ш린 (??2珥덈텇 횞 20Hz = 40媛? */
        private const val MAX_CLONE_HISTORY  = 40

        /** t_begin ?먯깋 ?덉슜 ?ㅼ감 짹200ms */
        private const val CLONE_MATCH_TOL_US = 200_000L  // [DEBUG-2] 1s -> 200ms 환원 (게이팅 추가로 누적 처리)

        /** [DEBUG-2] inferJob update 게이팅 — history span 이 이 시간 이상일 때만 update.
         *  의도: 1초 윈도우 학습 모델과 매칭. 짧은 윈도우 update 시 모델 출력 비현실적 → EKF 발산. */
        private const val MIN_HISTORY_SPAN_US = 800_000L  // 0.8초

        /**
         * ?뺤쟻 ?곹깭 ?먯젙 ?꾧퀎媛?
         * body frame ?먯씠濡?3異?RMS 媛 ??媛?誘몃쭔?대㈃ ?뺤?濡??먮떒 ??EKF ?낅뜲?댄듃 嫄대꼫?.
         * ?뺤? MEMS ?먯씠濡??몄씠利???0.003-0.01 rad/s RMS.
         * 蹂댄뻾 ?먯씠濡???0.1-0.5 rad/s RMS.
         * [P9 議곗젙] 0.03 ??0.08 rad/s:
         * ?ㅺ린湲??먮??덉씠??MEMS ?먯씠濡??뺤? ?몄씠利덇? 0.03??珥덇낵?섎뒗 寃쎌슦媛 留롮쓬.
         * STATIC ?듭빱 濡쒓렇媛 ?꾪? 李랁엳吏 ?딆쓣 寃쎌슦(?뺤? 誘멸컧吏) ??媛믪쓣 ?щ┛??
         * 0.08 rad/s ??4.6째/s ???먮┛ 嫄룰린(蹂댄넻 0.15+ rad/s)? 異⑸텇??援щ텇??
         */
        private const val STATIC_GYR_RMS_THRESHOLD = 0.08f  // rad/s

        /**
         * 1-珥??덈룄?곕떦 理쒕? ?덉슜 蹂??(m).
         * [P9d] ?ㅻ궡 ?쇰컲 蹂댄뻾 理쒕??띾룄 ??2 m/s ??1珥???2m.
         * 2m 珥덇낵 ???ㅽ듃?뚰겕 ?댁긽 異쒕젰 ?먮뒗 醫뚰몴 蹂???ㅻ쪟濡??먮떒 ??嫄대꼫?.
         * (湲곗〈 6.0m ???덈Т 愿? ???섎せ??痢≪젙媛믪씠 EKF 瑜?諛쒖궛?쒗궡)
         */
        private const val MAX_DISP_PER_WINDOW_M = 2.0

        /**
         * [P9d] 痢≪젙 怨듬텇??理쒖넖媛?(諛붾떏 ?ㅼ젙).
         * ?ㅽ듃?뚰겕媛 怨쇰룄?섍쾶 ?먯떊媛??덈뒗 ?덉륫??????EKF 媛 留밸ぉ?곸쑝濡??곕씪媛??寃껋쓣 諛⑹?.
         * exp(-4) ??0.018 m짼 ??0.1 m짼 (std = 0.316 m) 濡??섑뼢.
         * K = 0.01/(0.01+0.1) ??0.09 ??怨쇰룄??蹂댁젙 ?듭젣.
         */
        private const val MIN_MEAS_COV = 0.05  // m짼 (std ??0.224 m)

        // ?? [?꾩씠?붿뼱 3] Hysteresis ?곹깭 癒몄떊 ?뚮씪誘명꽣 ????????????????
        /**
         * MOVING ??STATIC ?꾪솚???꾩슂???곗냽 ?뺤? ?꾨젅????
         * 5?꾨젅??횞 50ms = 250ms ?곗냽 ?뺤??댁빞 STATIC ?쇰줈 ?뺤젙.
         * 媛믪쓣 ?믪씠硫??대룞?믪젙吏 諛섏쓳???먮젮吏吏留??ㅽ뙋??媛먯냼.
         */
        private const val STATIC_CONFIRM_FRAMES = 5

        /**
         * STATIC ??MOVING ?꾪솚???꾩슂???곗냽 ?대룞 ?꾨젅????
         * 3?꾨젅??횞 50ms = 150ms ?곗냽 ?대룞?댁빞 MOVING ?쇰줈 ?뺤젙.
         * 媛믪쓣 ??텛硫?蹂댄뻾 ?쒖옉 諛섏쓳??鍮⑤씪吏吏留??ㅽ뙋??媛?μ꽦 利앷?.
         */
        private const val MOVING_CONFIRM_FRAMES = 3

        // ?? [?꾩씠?붿뼱 5] EKF ?띾룄 寃뚯씠???????????????????????????????
        /**
         * 紐⑤뜽 only 沅ㅼ쟻 ?꾩쟻???덉슜?섎뒗 理쒖냼 EKF ?띾룄 (m/s).
         * EKF 媛깆떊 ???띾룄媛 ??媛?誘몃쭔?대㈃ dead-reckoning ??李⑤떒.
         * 5 cm/s: MEMS ?뺤? ?쒕━?꾪듃(??-3 cm/s) ????2-3諛????덉쟾 ?ъ쑀.
         */
        private const val MODEL_VELOCITY_GATE = 0.05  // m/s

        // ?? Yaw drift 蹂댁젙 ?뚮씪誘명꽣 ??????????????????????????????????
        /**
         * TYPE_ROTATION_VECTOR yaw 痢≪젙 ?몄씠利??쒖??몄감 (rad).
         * ?ㅻ궡 ?먭린 媛꾩꽠 ?섍꼍??怨좊젮?섏뿬 10째 (0.1745 rad) 濡?蹂댁닔?곸쑝濡??ㅼ젙.
         * 吏?먭린 ?몄씠利덇? ?곸? ?섍꼍?대㈃ 5째 濡?以꾩뿬??臾대갑.
         */
        private const val YAW_SIGMA_RAD = 10.0 / 180.0 * Math.PI

        // ?? [P10] ?쒖옉 ?뚮컢??+ 理쒖큹 ?대룞 諛쒖궛 諛⑹? ?뚮씪誘명꽣 ???????????
        /**
         * ?쒖옉 踰꾪듉 ?꾨Ⅸ ??沅ㅼ쟻??洹몃━吏 ?딅뒗 ?뚮컢??湲곌컙 (ms).
         * EKF ??利됱떆 ?ㅽ뻾(諛붿씠?댁뒪 ?섎졃, 怨듬텇???덉젙?? ?대줎 ?꾩쟻)?섎릺
         * ??湲곌컙 ?숈븞? trackPoints ???먯쓣 異붽??섏? ?딅뒗??
         * ??理쒖큹 ?대룞 諛쒖궛???붾㈃???쒖떆?섏? ?딆쓬.
         * ???댄썑 ?덉젙?붾맂 ?곹깭?먯꽌 沅ㅼ쟻 ?쒖떆 ?쒖옉.
         * 3珥? ?대줎 3媛??꾩쟻 + 諛붿씠?댁뒪 珥덇린 ?섎졃 異⑸텇.
         */
        private const val WARMUP_DURATION_MS = 3_000L  // 3珥?

        // ?? [P10] 理쒖큹 ?대룞 諛쒖궛 諛⑹? ?뚮씪誘명꽣 ?????????????????????????
        /**
         * 異붾줎 ?덈룄?곗쓽 ?숈쟻 援ш컙 理쒖냼 鍮꾩쑉 (0.0~1.0).
         * WINDOW_SIZE=100 ?섑뵆(1珥? 以???鍮꾩쑉 ?댁긽??gyr > threshold ?ъ빞 異붾줎 ?ㅽ뻾.
         *
         * 理쒖큹 ?대룞 ???덈룄?곕뒗 [?뺤? 85%][?대룞 15%] ?쇳빀 ???ㅽ듃?뚰겕 異쒕젰 ?ㅼ뿼.
         * 0.5(50%) ?붽뎄 ???대룞 ?쒖옉 ??~0.5珥??꾨???異붾줎 ?덉슜.
         * cloneHistory ?붽뎄(~1珥?? 留욌Ъ???쇳빀 ?덈룄???낅뜲?댄듃瑜??ㅼ쭏?곸쑝濡?李⑤떒.
         */
        private const val MIN_DYNAMIC_FRACTION = 0.5f

        /**
         * EKF ?낅뜲?댄듃 ???띾룄 ?ш린 ?곹븳 (m/s).
         * ?ㅻ궡 理쒓퀬 蹂댄뻾?띾룄 ??3 m/s. 珥덇낵 ??EKF 諛쒖궛?쇰줈 ?먮떒 ??ZUPT 媛뺤젣 ?곸슜.
         * 諛쒖궛 媛먯? ???덉쟾留?reactive divergence recovery).
         */
        // [DEBUG-4 revert] 5.0 → 3.0 복원.
        //   이유: 임계 5.0 시 ZUPT 가 거의 발동 안 함 → 잘못된 측정값 (unknown 클래스 OOD)
        //        도 그대로 따라감 → 분류 안정성/제자리 회전 인식 저하 + dispLocal 변동 큼.
        //   3.0 임계의 ZUPT 진동 (cycle) 은 trade-off 로 수용 — 분류 안정성 + 추적 의미성 우선.
        //   본질적 해결은 IMU CSV 추출 + OxIOD 비교 후 보정 (다음 세션 plan).
        private const val MAX_POST_UPDATE_SPEED = 3.0  // m/s

        // [P41 Dead-Reckoning Bypass] Python 원본 (Notion 2026-05-11) 핵심 통찰:
        //   ekf_tune.py 그리드서치 결과 — trolley 외 모든 클래스에서 Network-only RMSE <<
        //   EKF RMSE. trolley (state 6) 만 EKF 사용. 나머지는 EKF.update 우회 + dispLocal
        //   을 begin yaw 만큼 회전해 _net_pos 에 직접 누적.
        //
        //   롤백 방법: 아래 USE_DEAD_RECKONING_BYPASS = false 로 변경 → 기존 코드 그대로.
        //
        //   클래스 매핑 (Android CLASS_NAMES = InferenceEngine.kt L210 = train.py LABEL_REMAP 후 인덱스):
        //     0 handbag, 1 handheld, 2 pocket, 3 running, 4 slow_walk, 5 trolley, 6 unknown
        //   (이전 코멘트의 "model_meta.json 기준" 라벨은 raw 라벨 — P46-C 정정. 현재 model_meta.json 도 정합 수정됨.)
        //
        //   [P41 ROLLBACK] bypass01 측정에서 trackPoints 발산 → 즉시 롤백.
        //   원인 가설: Bypass 로 EKF.update 우회 후 (a) yaw 추정 정확도 저하 + (b) dispLocal
        //   방향 편향 (-y 일관) 이 EKF cross-coupling 없이 그대로 누적 → 한 방향 큰 흐름.
        //   다음 시도 전 Python 원본 _net_pos 누적 방식의 추가 후처리 (윈도우 평균, 방향
        //   필터 등) 검토 필요.
        private const val USE_DEAD_RECKONING_BYPASS = false
        private val NETWORK_ONLY_CLASSES = setOf(0, 1, 2, 3, 4, 6)   // P46-C 정정: 5 (trolley) 만 EKF.update — 나머지는 Dead-Reckoning Bypass 대상 (USE_DEAD_RECKONING_BYPASS=true 시)

        // [P41 R_all[t] Frame] Python `_get_imu_samples_for_network` 와 동일 처리.
        //   매 시점 자이로 적분 (EKF s_bg 차감) 으로 Rs_bofbi[t] 계산 → 시점별 R[t] 적용.
        //   현재 transformWindowToWorldFrame 의 R_begin 1개 사용 (모든 시점에 같은 회전)
        //   는 윈도우 내 자세 변화를 무시해 OOD 야기. 학습 코드 _window_to_gravity_aligned
        //   도 매 시점 quat 적용 — 동일하게 맞춤.
        //
        //   롤백: 아래 false 로 → 기존 R_begin 1개 방식.
        //
        //   [P41 v2] 1차 시도에서 trackPoints 발산 → 원인 분석 후 재구현:
        //     - raw gyr (P21 bias 차감 후, LPF 미적용) 사용  ← getRawGyrWindow()
        //     - EKF s_bg 추가 차감 안 함 (이중 차감 회피)
        //     - 적분 dt = 0.01s (100Hz 리샘플 후)
        //   롤백: false 로 → 기존 R_begin 1개 방식
        //
        //   [현재 상태] dispCov 형식 수정 (#C) 이 더 안전 + 효과 큼 → 그것부터 우선.
        //   R_all[t] v2 는 dispCov 검증 후 별도 시도.
        //
        //   [P41 v2 ACTIVATE] dispCov 수정 성공 (5m 왕복 → 1.8m 오차, 모델 RMSE 수준)
        //   확인 후 frame mismatch 해소 시도 — 국소 지그재그 (누적/직선 10.6×) 완화 목표.
        //   raw gyr (getRawGyrWindow) 사용 + EKF s_bg 차감 안 함 (v1 발산 회피).
        //   효과 없거나 발산 시 즉시 false 로 1줄 롤백.
        private const val USE_R_ALL_T_FRAME = true

        // ────────────────────────────────────────────────────────────
        // [P46] 점프 방지 게이트 (P42 jumpgate 재구축 — IMU 정합성 추가)
        //
        // 원인 (P42 분석 + P46 cls_replay_001 재확인):
        //   EKF state 의 yaw drift + clone state inconsistency 가 update 시
        //   position 변화로 풀려나오는 현상. cls_replay_001 (07:13:07.7) 에서
        //   meas|xy|=0.75m 인데 ekfPos Y +1.78m 점프 → 측정은 정상 보행,
        //   EKF state 가 outlier.
        //
        // 게이트 조건 (모두 만족 시 발동 — IMU 정합성 검증):
        //   1) |Δekf_pos|xy > JUMP_GATE_POS_M
        //      - EKF.update 전후 위치 xy 변화가 임계 초과
        //   2) window 내 linAcc magnitude peak < JUMP_GATE_ACC_PEAK
        //      - raw IMU 입력은 정상 보행 범위 (큰 가속/충격 없음)
        //      - "IMU 센서 정보로 위치 점프 정당성 검증" — 가속도 ↓ + 위치 ↑ = outlier
        //
        // 발동 시 동작 (P42 방식):
        //   1) applyPositionHold(preX, preY, preZ, 0.01)  — 위치 복귀
        //   2) applyZupt(0.01)                             — 속도 0 강제
        //   3) marginalize + clone history pop
        //   4) UI 갱신 skip (trackPoints 추가 안 함) + return
        //
        // 토글: 임계값을 1e6 으로 키워 사실상 비활성화. ablation 시 false 로 OFF.
        private const val USE_JUMP_GATE = true   // [P52] P48-D 롤백: C++ innov gate (P48-A) 발동 0건 확인, JUMP-GATE 의 우연한 EKF state outlier 차단 효과 복원 (P47-D jumpgate_007 의 3.22m 점프 1건 차단이 1.05m best 결과에 기여)
        /** Δekf_pos xy 임계 (m). 보행 1초 윈도우 최대 ~2m, 단일 inference (~150ms) 면 0.3m 이하 정상. */
        private const val JUMP_GATE_POS_M = 1.5
        /** window 내 raw linAcc body frame magnitude peak 임계 (m/s²). 정상 보행 peak ≤ 8. */
        private const val JUMP_GATE_ACC_PEAK = 8.0

        // ────────────────────────────────────────────────────────────
        // [P53] RotVec Dead-Reckoning — EKF 우회, 모델 disp 직접 적분
        //
        // 오프라인 하니스(src/Network/offline_eval.py) 결론:
        //   - 모델 자체는 정상 — window당 |disp_xy| 0.3~0.55m, OxIOD RMSE 0.89m 로 검증.
        //   - "8~13배 과대추정"(P51 전제)은 모델이 아니라 EKF 내부 발산 (yaw drift → 측정 OoD).
        //   - 모델 출력만 적분한 dead-reckoning → 경로 8~9m(실제 10m), 종점 폐합 ~1m. 정상.
        //
        // USE_ROTVEC_DR=true 동작:
        //   - EKF 클론/update 완전 우회 (propagate 는 무관하게 진행하나 위치엔 미사용).
        //   - 입력 프레임 변환: TYPE_ROTATION_VECTOR 절대 회전행렬을 per-timestep 으로 사용.
        //     EKF clone rotation 의 yaw drift 와 분리 — 하니스 'ga' frame 과 동일 변환.
        //   - 입력 = getRawWindow() (LPF 미적용 raw). 하니스 입력 분포와 정합.
        //   - 1초 비겹침 윈도우마다 1회 모델 disp 를 RotVec 시작 yaw 로 월드 회전 후 누적.
        //     (20Hz 겹침 윈도우를 매번 누적하면 ~20배 과적분 → 비겹침 1Hz 누적 필수.)
        //
        // 토글 OFF: 기존 EKF 경로 (runInferStep 의 클론/update 흐름).
        // const 가 아닌 val — runInferStep 조기 return 이후 코드의 unreachable 경고 회피.
        private val USE_ROTVEC_DR = true

        // ────────────────────────────────────────────────────────────
        // [P54] PDR-hybrid — 모델은 크기, 방향은 rotVec heading
        //
        // latest.csv 진단 (5m 왕복): 모델 disp 의 *크기*(보행 에너지)는 강건하나
        // *방향* 은 Android OoD 로 흩어져 재구성이 랜덤워크화 (구간별 순변위
        // 0.8m·2.2m, 기대 5m·5m). OxIOD 에선 방향 정상(349m RMSE 0.89m) →
        // 방향 단서가 도메인 갭에 취약.
        //
        // USE_PDR_HEADING=true 시: 모델 출력에서 |disp_xy|(크기) 만 취하고,
        // 진행 방향은 TYPE_ROTATION_VECTOR heading(자력계 융합 절대 방위) 사용.
        // 전진 보행 가정 (heading 방향으로 |disp| 만큼 이동).
        //   - handheld(폰을 진행 방향으로 들고 보행) 시나리오에 적합.
        //   - 윈도우 내 heading 변화가 큰 '제자리 회전' 윈도우는 병진 0 으로 누적 제외.
        //
        // 토글 OFF: P53 순수 모델 DR (방향도 모델 disp 사용 — 랜덤워크화).
        private val USE_PDR_HEADING = true

        // ────────────────────────────────────────────────────────────
        // [P55] 20Hz 속도 적분 — 추적 연속성 복원
        //
        // P53/P54 는 1초 비겹침 윈도우마다 1회 누적 → 갱신 1Hz, 게다가 회전 윈도우는
        // 통째 폐기(turn-skip) → 제자리 회전 중 앱이 완전히 멈춤(사용자 보고).
        //
        // P55: 모델 출력(1초 윈도우 변위)을 *속도*(disp/1s)로 환산해 매 추론 틱(20Hz)
        // 마다 dt 만큼 적분. 겹침 윈도우를 속도로 다루므로 과적분 없음(20틱×disp/20초
        // ≈ 실제 1초 변위). 회전 윈도우는 폐기 대신 *속도 감쇠* → 멈추지 않고 추적 유지.
        // 정지 윈도우는 속도 0(위치 고정)이되 UI 는 매 틱 갱신 → 앱이 죽지 않음.
        //
        /** 윈도우 내 heading 변화가 이 값(°) 초과면 제자리 회전으로 보고 속도 감쇠. */
        private const val TURN_YAW_THRESH_DEG = 60.0
        /** 제자리 회전 윈도우의 속도 감쇠 계수 (0=정지, 1=감쇠없음). */
        private const val TURN_SPEED_ATTEN = 0.3
        /** 속도 EMA 평활 계수 (20Hz 틱 jitter 완화). */
        private const val DR_VEL_EMA = 0.25
        /** trackPoint 를 추가하는 최소 이동거리 (m) — 리스트 비대 방지. */
        private const val DR_TRACKPOINT_MIN_MOVE = 0.1

        // ────────────────────────────────────────────────────────────
        // [P57] HANDHELD-only 데모 — 단일 속도 스케일
        //
        // 결정 배경 (docs/HANDOFF_P56.md §8):
        //   - 분류기 자체가 Android 도메인에서 OoD(주로 unknown) → per-class
        //     스위칭의 신뢰 근거 없음.
        //   - rotVec heading = 진행 방향 가정은 handheld 자세에서만 유효.
        //   - 현재 보정 데이터로는 모든 클래스가 균일 ~1.5× → soft-switching
        //     구조는 의미 없는 *가짜 구조* 였음(P56 한계 메모).
        //
        // P57: SPEED_SCALE_PER_CLASS(7) 소프트 스위칭 제거 → 단일 상수로 단순화.
        //   effectiveSpeed = modelSpeed × HANDHELD_SPEED_SCALE
        //   분류기 출력(clsProb, topClass)은 *UI 표시 전용* 으로 유지하되,
        //   위치 계산엔 사용하지 않는다(정합화).
        //
        // 향후 휴대모드별 실측 보정 데이터가 확보되면 그때 다시 가중 스위칭
        // 구조를 복원한다 (git history 의 P56 커밋 참조).
        private const val HANDHELD_SPEED_SCALE = 1.5

        // ────────────────────────────────────────────────────────────
        // [P60] EKF 비교 모드 (단말 토글) — 모드별 EKF 계수 비교용
        //
        // 데모 기본은 PATH_B (RotVec DR + PDR-hybrid, EKF 미사용).
        // 비교 측정 시에는 EKF_CURRENT / EKF_TLIO 중 하나로 전환해 *경로 A
        // (논문 EKF)* 를 활성화하고, EkfBridge.create() 에 모드별 cfg 파라미터
        // 를 전달한다. 식·gate(χ²=11.345, MAX_INNOV_NORM=3.0)는 두 모드 모두
        // 동일하며, cfg 만 다르다(init_vel/init_ba/meascov_scale).
        //
        // 한 번 보행 → exportPath() 로 trackPoints CSV 저장 → 다른 모드로 다시
        // 보행 → 두 CSV 를 tools/overlay_tracks.py 로 한 그래프에 겹쳐 비교.
        // (enum 정의는 외부 Activity 가 `LocalizationViewModel.EkfMode` 로 직접
        //  참조하도록 class 본문 nested 로 둠 — companion 안에 두면
        //  `Companion.EkfMode` 거쳐야 해서 호출부가 복잡해진다.)

        /** 런타임 변경 가능(MainActivity 메뉴). 기본 = 데모 경로 B. */
        @Volatile
        var ekfMode: EkfMode = EkfMode.PATH_B

        // ────────────────────────────────────────────────────────────
        // [P58] Activity 간 ViewModel 공유 — IMU 진단 화면에서 측위 상태 구독용.
        //
        // MainActivity 의 ViewModelProvider 가 만든 인스턴스를 다른 Activity
        // (ImuTestActivity 등) 에서 읽을 수 있도록 약한 참조로 노출한다.
        // 분류기 출력(carryMode/carryProb) 표시를 메인 UI 에서 빼서 진단 쪽으로
        // 이전한 결과(P58).
        //
        // 동시 다중 인스턴스가 만들어지는 시나리오는 없으나(MainActivity 1개),
        // 안전을 위해 init 등록 / onCleared 해제로 lifecycle 매칭.
        @Volatile
        var sharedInstance: LocalizationViewModel? = null
            private set
    }

    init {
        // 가장 최근 생성된 인스턴스를 노출 (Activity 간 state 공유용).
        sharedInstance = this
    }

    override fun onCleared() {
        if (sharedInstance === this) sharedInstance = null
        super.onCleared()
    }

    // ?? ?섏〈 而댄룷?뚰듃 ????????????????????????????????????????????
    val imuCollector = ImuCollector(application)
    val inferEngine  = InferenceEngine(application)

    // ?? UI ?곹깭 ??????????????????????????????????????????????????
    data class LocalizationState(
        val isRunning:         Boolean = false,
        val position:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val posStd:            Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val velocity:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
        val carryMode:         String  = "unknown",
        val carryProb:         Float   = 0f,
        val trackPoints:       List<Pair<Double, Double>> = emptyList(),  // EKF 沅ㅼ쟻
        val modelTrackPoints:  List<Pair<Double, Double>> = emptyList(),  // 紐⑤뜽 only 沅ㅼ쟻
        val inferLatency:      Long    = 0L,
        // [P21-ish] 罹섎━釉뚮젅?댁뀡 UI ?쇰뱶諛???MainActivity ??calibCard ?쒖떆 ?쒖뼱
        val calibrating:       Boolean = false,
        val calibProgress:     Float   = 0f
    )

    private val _state = MutableStateFlow(LocalizationState())
    val state: StateFlow<LocalizationState> = _state

    // ?? ?대? ?곹깭 ????????????????????????????????????????????????
    private var inferJob: Job? = null
    private var propJob:  Job? = null
    private val trackPoints      = mutableListOf<Pair<Double, Double>>()
    private val modelTrackPoints = mutableListOf<Pair<Double, Double>>()

    /** [P10] ?쒖옉 踰꾪듉???꾨Ⅸ ?쒓컖 (System.currentTimeMillis). ?뚮컢???먯젙???ъ슜. */
    private var startTimeMs: Long = 0L

    /** 紐⑤뜽 ?⑤룆 ?꾩쟻 ?꾩튂 (EKF ?놁씠 displacement ?⑹궛) */
    private var modelPosX = 0.0
    private var modelPosY = 0.0

    // [P41 Dead-Reckoning Bypass] Network-only 모드 (trolley 외) 의 누적 위치.
    // EKF 모드 진입 시 EKF.p 로 동기화, NetOnly 모드 진입 시 누적 시작.
    private var netPosX = 0.0
    private var netPosY = 0.0
    /** 직전 추론에서 EKF 모드였는지 — 모드 전환 시 _net_pos 재동기화용. */
    private var prevUsedEkfUpdate = true

    /** [P55] RotVec DR: 마지막 적분 틱의 ts (μs). -1 = 미시작. dt 계산용. */
    private var lastDrTickTs: Long = -1L
    /** [P55] world-frame 속도 추정값 (m/s, EMA 평활). */
    private var drVelX: Double = 0.0
    private var drVelY: Double = 0.0
    /** [P55] DR 틱 카운터 — 주기적 로그용. */
    private var drTickCount: Int = 0

    // ?? [?꾩씠?붿뼱 3] Hysteresis ?곹깭 癒몄떊 ????????????????????????????
    private enum class MotionState { STATIC, MOVING }

    /** ?뺤젙???꾩옱 ?대룞 ?곹깭. 珥덇린媛?STATIC (?쒖옉 ???뺤? 媛??. */
    private var motionState = MotionState.STATIC

    /** MOVING ?곹깭?먯꽌 ?곗냽?쇰줈 gyrRms < threshold ???꾨젅????(??STATIC ?꾪솚 移댁슫??. */
    private var staticCandidateCount  = 0

    /** STATIC ?곹깭?먯꽌 ?곗냽?쇰줈 gyrRms ??threshold ???꾨젅????(??MOVING ?꾪솚 移댁슫??. */
    private var movingCandidateCount  = 0

    /**
     * [P9] Hard State Freeze ?듭빱 ?꾩튂.
     * STATIC 泥??꾨젅?꾩뿉 EKF ?꾩튂瑜?湲곕줉, ?댄썑 STATIC 湲곌컙 ?대궡
     * freezeStaticState(?듭빱)瑜??몄텧?섏뿬 EKF ?곹깭瑜?吏곸젒 怨좎젙.
     * MOVING ?꾨젅???먮뒗 reset() ??null 濡?珥덇린??
     */
    private var staticAnchorPos: DoubleArray? = null

    // ?? [?꾩씠?붿뼱 5] 吏곸쟾 EKF 媛깆떊 ???띾룄 ?ш린 (m/s) ???????????????
    /** EKF update() 吏곹썑 議고쉶???띾룄 ?몃쫫 ???ㅼ쓬 ?ㅽ뀦 紐⑤뜽 only 寃뚯씠?낆뿉 ?ъ슜. */
    private var prevEkfVelNorm = 0.0

    // ?? ?먯젏 蹂듦? ?먮룞 寃쎈줈 ?대━??????????????????????????????????
    /** 異쒕컻 ????嫄곕━(m) ?댁긽 硫?댁쭊 ?곸씠 ?덉뼱??蹂듦? 媛먯?瑜??쒖꽦?? */
    private val AWAY_THRESHOLD_M   = 1.5
    /** ?먯젏?쇰줈遺????嫄곕━(m) ?대궡濡??ㅼ뼱?ㅻ㈃ 寃쎈줈 ?대━?? */
    private val RETURN_PROXIMITY_M = 0.5
    /** 異쒕컻 ??AWAY_THRESHOLD_M ?댁긽 硫?댁쭊 ?곸씠 ?덈뒗吏 ?щ?. */
    private var wasAwayFromOrigin  = false

    // ?? Yaw drift 蹂댁젙 ????????????????????????????????????????????
    /**
     * EKF 珥덇린???쒖젏??TYPE_ROTATION_VECTOR yaw (rad).
     * Double.NaN = 誘몄큹湲고솕 ?먮뒗 rotVecSensor ?놁쓬.
     *
     * 蹂댁젙 怨듭떇:  yaw_meas = yaw_rv_current ??yaw_rv_at_init
     *   ??EKF ?붾뱶 ?꾨젅??湲곗? ?곷? yaw (珥덇린???쒖젏 湲곗? 0)
     */
    private var yawRvAtInit = Double.NaN

    /**
     * inferJob ??propJob ?⑤갑???좏샇:
     *  inferJob ???먰븯?????대줎 ?쇱꽌 ??꾩뒪?ы봽瑜?湲곕줉.
     *  propJob ???대떦 ts ?댁긽???섑뵆 泥섎━ ???대줎 ?쎌엯 ??-1 濡?由ъ뀑.
     */
    private val pendingCloneTs      = AtomicLong(-1L)

    /**
     * propJob ??inferJob ?⑤갑???좏샇:
     *  媛??理쒓렐???쎌엯???대줎???ㅼ젣 ?쇱꽌 ts.
     */
    private val lastInsertedCloneTs = AtomicLong(-1L)

    /**
     * P3 ?섏젙: propJob ??inferJob ?대줎 ?덉뒪?좊━ 梨꾨꼸.
     *
     * propJob ???대줎 ?쎌엯 ??ts 瑜?Channel ??trySend.
     * inferJob ??runInferStep() ?쒖옉 ??Channel ??drain ?섏뿬
     * ?먯떊留뚯쓽 localCloneHistory(ArrayDeque) ???곸옱.
     * ??synchronized 釉붾줉 ?놁씠 ?⑤갑??硫붿떆吏 ?⑥떛?쇰줈 ?덉쟾?섍쾶 ?듭떊.
     *
     * DROP_OLDEST: 踰꾪띁媛 媛??李?寃쎌슦 ?ㅻ옒???대줎 ts 瑜??먮룞 ?쒓굅
     * (媛???ㅻ옒??寃껋? ?대? 二쇰??붾릺???꾩슂 ?놁쓬).
     */
    private val cloneChannel = Channel<Long>(
        capacity       = MAX_CLONE_HISTORY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // ?? 痢≪쐞 ?쒖옉 ????????????????????????????????????????????????
    /**
     * @param replayCsv null 이면 실시간 측위. File 이면 CSV 재생 모드 (P45-Replay).
     *   CSV 형식은 ImuTestActivity 기록 (sensor,ts_ns,x,y,z,w).
     */
    fun start(replayCsv: File? = null) {
        if (_state.value.isRunning) return
        startTimeMs = System.currentTimeMillis()
        val mode = if (replayCsv != null) "Replay(${replayCsv.name})" else "실시간"
        Log.i(TAG, "측위 시작 [$mode] — 워밍업 ${WARMUP_DURATION_MS}ms")
        Log.i(TAG, "痢≪쐞 ?쒖옉 (?뚮컢??${WARMUP_DURATION_MS}ms ?숈븞 沅ㅼ쟻 ?쒖떆 ?듭젣)")

        viewModelScope.launch(Dispatchers.IO) {
            // 紐⑤뜽 濡쒕뱶
            if (!inferEngine.isLoaded()) {
                try { inferEngine.load() }
                catch (e: Exception) { Log.e(TAG, "紐⑤뜽 濡쒕뱶 ?ㅽ뙣: ${e.message}") }
            }

            // IMU ?섏쭛 ?쒖옉 (罹섎━釉뚮젅?댁뀡 吏꾩엯 ??2珥덇컙 sample ?먯뿉 ?ㅼ뼱媛吏 ?딆쓬)
            imuCollector.start(replayMode = replayCsv != null)

            // [P45-Replay] CSV 재생 thread 시작 — 캘리브 polling 보다 *먼저* 시작되어야 함.
            // CSV 처음 2초 = 측정 당시 정지 자세였다고 가정. 자동 캘리브 사용.
            if (replayCsv != null) {
                imuCollector.startReplay(replayCsv) {
                    Log.i(TAG, "[P45-Replay] CSV 끝 도달 — 측위 자동 종료")
                    viewModelScope.launch { stop() }
                }
            }

            // ?? [P21-ish] 罹섎━釉뚮젅?댁뀡 吏꾪뻾瑜?polling ??????????????????
            // EkfBridge ?앹꽦쨌propJob ?쒖옉 *?? ??2 珥덇컙 polling.
            // ImuCollector ??罹섎━釉뚮젅?댁뀡 以??먯뿉 sample ???곸옱?섏? ?딆쑝誘濡?
            // ?댁감??propJob ??利됱떆 ?쒖옉?쇰룄 ?????놁?留? UI 媛 吏꾪뻾瑜좎쓣
            // 蹂댁뿬二쇨린 ?꾪빐 紐낆떆?곸쑝濡??湲?
            _state.value = _state.value.copy(calibrating = true, calibProgress = 0f)
            while (imuCollector.isCalibrating()) {
                _state.value = _state.value.copy(
                    calibrating  = true,
                    calibProgress = imuCollector.getCalibrationProgress()
                )
                delay(50L)
            }
            _state.value = _state.value.copy(calibrating = false, calibProgress = 1f)

            // [P60] EKF 생성 — ekfMode 별 cfg 파라미터 적용.
            //   PATH_B / EKF_CURRENT : DEFAULT_PARAMS (단말 기본)
            //   EKF_TLIO            : TLIO_PARAMS    (논문 §V-D/§V-E 계수)

            // EKF ?앹꽦 (罹섎━釉뚮젅?댁뀡 ?꾨즺 ??
            val ekfCfgEnum = when (ekfMode) {
                EkfMode.EKF_TLIO -> EkfBridge.EkfCfg.TLIO
                else             -> EkfBridge.EkfCfg.CURRENT
            }
            Log.i(TAG, "[P60] EKF cfg = ${ekfMode.name} (${ekfCfgEnum.name})")
            EkfBridge.create(EkfBridge.paramsFor(ekfCfgEnum))

            // ?? ??EKF ?꾪뙆 猷⑦봽 (5ms ?대쭅, 紐⑤뱺 100Hz ?섑뵆 泥섎━) ??
            propJob = viewModelScope.launch(Dispatchers.Default) {
                while (isActive) {
                    val samples = imuCollector.drainPropagateQueue()
                    for (sample in samples) {
                        val tsUs = sample.ts_us
                        val acc  = doubleArrayOf(
                            sample.acc[0].toDouble(),
                            sample.acc[1].toDouble(),
                            sample.acc[2].toDouble()
                        )
                        val gyr  = doubleArrayOf(
                            sample.gyr[0].toDouble(),
                            sample.gyr[1].toDouble(),
                            sample.gyr[2].toDouble()
                        )

                        if (!EkfBridge.isInitialized()) {
                            EkfBridge.initialize(tsUs, acc)
                            // [P11] EKF 珥덇린??吏곹썑 Rotation Vector yaw ?ㅽ봽??湲곕줉.
                            // ?먭린怨??뺥솗?꾧? MEDIUM(2) ?댁긽???뚮쭔 湲곕줉 ????쑝硫?NaN ?좎?.
                            // ???섏쨷???뺥솗?꾧? ?щ씪媛硫?applyRotVecYaw 媛 ?먮룞?쇰줈 二쇱엯 ?쒖옉.
                            // ??珥덇린 ?섎せ??湲곗??먯쑝濡??명븳 yaw ?ㅽ봽???ㅻ쪟 諛⑹?.
                            val rvYaw     = imuCollector.getLatestYawRad()
                            val rvAccInit = imuCollector.getRotVecAccuracy()
                            if (!rvYaw.isNaN() && rvAccInit >= 2) {
                                yawRvAtInit = rvYaw.toDouble()
                                Log.i(TAG, "Yaw ?ㅽ봽??珥덇린?? ${"%.1f".format(Math.toDegrees(yawRvAtInit))}째 (?뺥솗??$rvAccInit/3)")
                            } else {
                                Log.w(TAG, "Yaw ?ㅽ봽??珥덇린??蹂대쪟: ?먭린怨??뺥솗??遺議?($rvAccInit/3) ???뺥솗???뚮났 ???먮룞 ?ㅼ젙")
                            }
                            continue
                        }

                        // [P11] yawRvAtInit 吏??珥덇린??
                        // EKF 珥덇린?????먭린怨??뺥솗?꾧? ??븘 ?ㅽ봽?뗭씠 NaN ??寃쎌슦,
                        // ?뺥솗?꾧? MEDIUM ?댁긽?쇰줈 ?ㅻⅤ硫?洹??쒖젏???ㅽ봽?뗭쓣 湲곕줉.
                        if (yawRvAtInit.isNaN()) {
                            val rvYaw     = imuCollector.getLatestYawRad()
                            val rvAccNow  = imuCollector.getRotVecAccuracy()
                            if (!rvYaw.isNaN() && rvAccNow >= 2) {
                                yawRvAtInit = rvYaw.toDouble()
                                Log.i(TAG, "Yaw ?ㅽ봽??吏??珥덇린?? ${"%.1f".format(Math.toDegrees(yawRvAtInit))}째 (?뺥솗??$rvAccNow/3)")
                            }
                        }

                        // ?대줎 ?쎌엯 ?щ?: inferJob ???덉빟??ts ?댁긽?대㈃ ?쎌엯
                        val pending = pendingCloneTs.get()
                        val augTs: Long = if (pending >= 0L && tsUs >= pending) {
                            // compareAndSet ?쇰줈 以묐났 ?쎌엯 諛⑹?
                            if (pendingCloneTs.compareAndSet(pending, -1L)) tsUs else -1L
                        } else {
                            -1L
                        }

                        EkfBridge.propagate(acc, gyr, tsUs, augTs)

                        if (augTs >= 0L) {
                            lastInsertedCloneTs.set(augTs)
                            Log.i(TAG, "[CLONE-INSERT] propJob inserted ts=$augTs")
                            // P3 ?섏젙: synchronized ???Channel.trySend 濡?鍮꾨씫???꾨떖
                            cloneChannel.trySend(augTs)
                            Log.v(TAG, "?대줎 ?쎌엯 ts=$augTs ??Channel ?꾨떖")
                        }
                    }
                    delay(PROP_POLL_MS)
                }
            }

            // ?? ??異붾줎 猷⑦봽 (寃쎄낵 ?쒓컙 蹂댁젙 20Hz) ?????????????????
            inferJob = viewModelScope.launch(Dispatchers.Default) {
                // P3: inferJob ?꾩슜 ?대줎 ?덉뒪?좊━ ????肄붾（?대쭔 ?쎄린/?곌린
                val localCloneHistory = ArrayDeque<Long>()

                while (isActive) {
                    val loopStart = System.nanoTime()
                    runInferStep(localCloneHistory)
                    val elapsedMs = (System.nanoTime() - loopStart) / 1_000_000L
                    val remaining = INFER_INTERVAL_MS - elapsedMs
                    // P6 ?섏젙: ??긽 理쒖냼 1ms yield ???ㅻ젅??湲곗븘 諛⑹?
                    delay(remaining.coerceAtLeast(1L))
                }
            }

            _state.value = _state.value.copy(isRunning = true)
        }
    }

    // ?? 痢≪쐞 ?뺤? ????????????????????????????????????????????????
    fun stop() {
        inferJob?.cancel(); inferJob = null
        propJob?.cancel();  propJob  = null
        imuCollector.stop()
        pendingCloneTs.set(-1L)
        lastInsertedCloneTs.set(-1L)
        // Channel 鍮꾩슦湲?(?ъ떆?????ㅻ옒??ts 濡??ㅻ룞??諛⑹?)
        while (cloneChannel.tryReceive().isSuccess) { /* drain */ }
        _state.value = _state.value.copy(isRunning = false)
        Log.i(TAG, "痢≪쐞 ?뺤?")
    }

    // ?? 珥덇린????????????????????????????????????????????????????
    fun reset() {
        stop()
        trackPoints.clear()
        modelTrackPoints.clear()
        modelPosX = 0.0
        modelPosY = 0.0
        // [P41 Dead-Reckoning Bypass] 위치 누적 초기화
        netPosX = 0.0
        netPosY = 0.0
        prevUsedEkfUpdate = true
        // [P55] RotVec DR 속도 적분 상태 초기화
        lastDrTickTs = -1L
        drVelX = 0.0
        drVelY = 0.0
        drTickCount = 0
        // [?꾩씠?붿뼱 3] ?곹깭 癒몄떊 珥덇린??
        motionState          = MotionState.STATIC
        staticCandidateCount = 0
        movingCandidateCount = 0
        // [P8] Position Hold ?듭빱 珥덇린??
        staticAnchorPos      = null
        // [?꾩씠?붿뼱 5] ?띾룄 寃뚯씠??珥덇린??
        prevEkfVelNorm       = 0.0
        // ?먯젏 蹂듦? 媛먯? 珥덇린??
        wasAwayFromOrigin    = false
        // Yaw drift 蹂댁젙 珥덇린??
        yawRvAtInit          = Double.NaN
        _state.value = LocalizationState()
    }

    // ?? 異붾줎 + EKF 媛깆떊 ?ㅽ뀦 ????????????????????????????????????
    /**
     * suspend ?⑥닔: delay(CLONE_SETTLE_MS) ?ы븿.
     * inferJob 肄붾（???댁뿉?쒕쭔 ?몄텧.
     *
     * @param localCloneHistory inferJob ?꾩슜 ?대줎 ?덉뒪?좊━ (??遺덊븘??.
     */
    private suspend fun runInferStep(localCloneHistory: ArrayDeque<Long>) {
        if (!EkfBridge.isInitialized()) return
        if (!inferEngine.isLoaded())    return

        // [P53] RotVec dead-reckoning 경로 — EKF 클론/update 흐름 완전 우회.
        // [P60] ekfMode 가 EKF_CURRENT/EKF_TLIO 면 *EKF 경로* 강제 활성화.
        //       PATH_B (기본, 데모) 일 때만 RotVec DR 로 우회.
        if (USE_ROTVEC_DR && ekfMode == EkfMode.PATH_B) {
            runRotVecDrStep()
            return
        }

        // ??異붾줎 ?덈룄???뺣낫 (理쒖냼 100 ?섑뵆 ?꾩슂)
        val (window, _) = imuCollector.getWindow() ?: return

        // ??[P5?뭁7] Hysteresis ?뺤? ?먯젙 (?대줎 ?덉빟蹂대떎 癒쇱? ?섑뻾).
        //
        //  P5: ?뺤? ?먯젙 ???대줎 李⑤떒 + ?꾩껜 二쇰??????꾨━利??닿껐, ???꾩튂 ?쒕━?꾪듃.
        //  P7: ?뺤? ?먯젙 ?꾩뿉???대줎 ?뺤긽 ?쎌엯 ??zero-disp EKF update + 二쇰???+ ZUPT.
        //      ?꾩튂 ?쒖빟(std??cm) + ?띾룄 ?쒖빟(ZUPT) 蹂듯빀 ?곸슜 ???쒕━?꾪듃 ?댁냼.
        //
        //  STATIC ??MOVING : gyrRms ??threshold 媛 MOVING_CONFIRM_FRAMES ?곗냽
        //  MOVING ??STATIC : gyrRms <  threshold 媛 STATIC_CONFIRM_FRAMES  ?곗냽
        val gyrRms        = computeGyrRms(window)
        val isStaticFrame = gyrRms < STATIC_GYR_RMS_THRESHOLD
        // [吏꾨떒] ?뺤? 媛먯? ?щ? ?ㅼ떆媛??뺤씤 ???꾩슂 ??二쇱꽍 泥섎━
        Log.v(TAG, "gyrRms=${"%.4f".format(gyrRms)} thr=$STATIC_GYR_RMS_THRESHOLD static=$isStaticFrame state=$motionState")

        val currentlyStatic: Boolean = when (motionState) {
            MotionState.STATIC -> {
                if (isStaticFrame) {
                    // ?뺤? ?좎? ???대룞 ?꾨낫 移댁슫??由ъ뀑
                    movingCandidateCount = 0
                    true
                } else {
                    // ?대룞 ?꾨낫 ?꾩쟻
                    movingCandidateCount++
                    staticCandidateCount = 0
                    if (movingCandidateCount >= MOVING_CONFIRM_FRAMES) {
                        motionState = MotionState.MOVING
                        movingCandidateCount = 0
                        Log.d(TAG, "?곹깭 ?꾪솚: STATIC ??MOVING " +
                              "(gyrRms=${"%.4f".format(gyrRms)} rad/s, " +
                              "velNorm=${"%.3f".format(prevEkfVelNorm)} m/s)")
                        // [P9c] STATIC?묺OVING ?꾪솚 ??stale ?대줎 ?댁쨷 ?뚮윭??
                        // ??Kotlin localCloneHistory: STATIC ?댁쟾 ??꾩뒪?ы봽 ?쒓굅
                        //    ??findBeginClone() ??stale tBegin ??李얠? 紐삵븯寃???
                        // ??C++ EKF ?대? ?대줎: marginalize 誘명샇異쒕줈 ?⑥? ?ㅻ옒???대줎 ?쒓굅
                        //    ??update(tBegin, tEnd) ?먯꽌 議댁옱?섏? ?딅뒗 ?대줎 李몄“ 諛⑹?
                        // ???뚮윭????~1珥덇컙 tBegin 誘명솗蹂???update() ?먮룞 ?ㅽ궢
                        // ?????대줎 ~1珥??꾩쟻 ???뺤긽 ?ш컻
                        localCloneHistory.clear()
                        EkfBridge.flushClones()
                        EkfBridge.thawStaticState()
                        Log.d(TAG, "STATIC?묺OVING: ?대줎 ?댁쨷 ?뚮윭??怨듬텇???대룞 ??stale 諛쒖궛 諛⑹?")
                        false   // ?대쾲 ?꾨젅?꾨???異붾줎 吏꾪뻾
                    } else {
                        // ?꾩쭅 MOVING 誘명솗?????뺤?濡??좎?
                        true
                    }
                }
            }
            MotionState.MOVING -> {
                if (!isStaticFrame) {
                    // ?대룞 ?좎? ???뺤? ?꾨낫 移댁슫??由ъ뀑
                    staticCandidateCount = 0
                    false
                } else {
                    // ?뺤? ?꾨낫 ?꾩쟻
                    staticCandidateCount++
                    movingCandidateCount = 0
                    if (staticCandidateCount >= STATIC_CONFIRM_FRAMES) {
                        motionState = MotionState.STATIC
                        staticCandidateCount = 0
                        Log.d(TAG, "?곹깭 ?꾪솚: MOVING ??STATIC " +
                              "(gyrRms=${"%.4f".format(gyrRms)} rad/s, " +
                              "velNorm=${"%.3f".format(prevEkfVelNorm)} m/s)")
                        true    // ?대쾲 ?꾨젅?꾨???ZUPT ?곸슜
                    } else {
                        // ?꾩쭅 STATIC 誘명솗?????대룞?쇰줈 ?좎? (異붾줎 怨꾩냽)
                        false
                    }
                }
            }
        }

        if (currentlyStatic) {
            // [P9] ?뺤? ?곹깭: Hard State Freeze (EKF 痢≪젙 ?고쉶 吏곸젒 怨좎젙)
            //
            // 臾몄젣 ?먯씤 (P8 ?ㅽ뙣 ?댁쑀):
            //   apply_position_hold(sigma=0.01) + apply_zupt() 議고빀?
            //   init_pos_sigma=0.001 ??誇[p,p]=1e-6 m짼,  R_pos=(0.01)짼=1e-4 m짼
            //   ??移쇰쭔 寃뚯씤 K = 1e-6 / (1e-6 + 1e-4) ??0.01 ??蹂댁젙 1% 誘몃쭔
            //   ???ъ떎???꾩튂 怨좎젙 遺덇? ??諛쒖궛 吏??
            //
            // P9 ?닿껐梨?
            //   freezeStaticState() ??EKF 痢≪젙 紐⑤뜽 ?꾩쟾 ?고쉶:
            //   ??state_.p = p_anchor  (吏곸젒 ?꾩튂 怨좎젙)
            //   ??state_.v = Vec3::Zero()  (吏곸젒 ?띾룄 0)
            //   ??誇[v,v], 誇[p,p] 釉붾줉 ?됀룹뿴 ?꾨? 0, ?媛곷쭔 1e-8 (援먯감 怨듬텇???쒓굅)
            //   ??移쇰쭔 寃뚯씤 ?놁씠 ?꾩쟾 怨좎젙 ??媛?띾룄怨?諛붿씠?댁뒪 ?곷텇 ?쒕━?꾪듃 100% 李⑤떒

            // ???대줎 ?쎌엯 李⑤떒 (P5 諛⑹떇 ?좎?)
            pendingCloneTs.set(-1L)

            // ???듭빱 湲곕줉 (STATIC 湲곌컙 以?泥??꾨젅?꾩뿉留??ㅽ뻾)
            if (staticAnchorPos == null) {
                staticAnchorPos = EkfBridge.getPosition().take(3).toDoubleArray()
                Log.d(TAG, "STATIC ?듭빱 ?ㅼ젙[P9]: " +
                      "(${"%.3f".format(staticAnchorPos!![0])}, " +
                      "${"%.3f".format(staticAnchorPos!![1])}, " +
                      "${"%.3f".format(staticAnchorPos!![2])}) m")
            }

            // ??Hard State Freeze: ?꾩튂쨌?띾룄 吏곸젒 怨좎젙 + 怨듬텇???뺤텞
            staticAnchorPos?.let { anchor ->
                EkfBridge.freezeStaticState(anchor[0], anchor[1], anchor[2])
            }

            // ??yaw 蹂댁젙 (?먯씠濡??명뼢 蹂댁“ ???뚯쟾 ?쒕━?꾪듃 ?듭젣)
            applyRotVecYaw("STATIC")
            return
        }

        // MOVING 釉뚮옖移?吏꾩엯 ???듭빱 ?댁젣 (?ㅼ쓬 STATIC ???덈줈 湲곕줉)
        staticAnchorPos = null

        // ??? ?대룞(MOVING) 寃쎈줈 ????????????????????????????????????????

        // ?????대줎 ?덉빟 ???꾩옱 理쒖떊 ?쇱꽌 ts 湲곗? (wall-clock ?ъ슜 湲덉?)
        val tEndTarget = imuCollector.getLatestSample()?.ts_us ?: return
        pendingCloneTs.set(tEndTarget)

        // ??propJob ???대줎???쎌엯???뚭퉴吏 ?湲?(P4: 20ms ??30ms)
        delay(CLONE_SETTLE_MS)

        // ???ㅼ젣 ?쎌엯?????대줎 ts ?뺤씤
        //    tEnd < tEndTarget ?대㈃ ?꾩쭅 誘몄궫?????대쾲 ?ㅽ뀦 嫄대꼫?
        val tEnd = lastInsertedCloneTs.get()
        if (tEnd < tEndTarget) return

        // ??P3: Channel ?먯꽌 ???대줎 ts 瑜?localCloneHistory 濡?drain
        var newTs = cloneChannel.tryReceive().getOrNull()
        while (newTs != null) {
            localCloneHistory.addLast(newTs)
            if (localCloneHistory.size > MAX_CLONE_HISTORY) localCloneHistory.removeFirst()
            newTs = cloneChannel.tryReceive().getOrNull()
        }

        // [DEBUG-2] history span 게이팅 — 1초 윈도우 학습 모델과 매칭 위해
        // localCloneHistory 의 oldest 가 tEnd 보다 충분히 이전일 때만 update 진행.
        // 짧은 윈도우 (예: 100ms) update 시 모델 출력이 0 근처로 폭락 → EKF 발산.
        val oldestTs = localCloneHistory.firstOrNull()
        if (oldestTs == null || (tEnd - oldestTs) < MIN_HISTORY_SPAN_US) {
            Log.i(TAG, "[GATE-SPAN] history span ${if (oldestTs != null) tEnd - oldestTs else -1L}us < ${MIN_HISTORY_SPAN_US}us — update skip (누적 대기)")
            return
        }

        // ???쒖옉 ?대줎 ?먯깋 (~1珥??댁쟾, localCloneHistory ?먯꽌 媛??媛源뚯슫 ??ぉ)
        val tBegin = findBeginClone(tEnd, localCloneHistory)
        if (tBegin < 0L || tBegin >= tEnd) return

        // ??P10: ?덈룄???숈쟻 鍮꾩쑉 寃뚯씠??
        //   理쒖큹 ?대룞 ??異붾줎 ?덈룄??1珥???[?뺤? ?곗씠??85%][?대룞 15%] ?쇳빀??
        //   ?ㅽ듃?뚰겕?????쇳빀 ?덈룄?곗뿉???ㅼ뿼??蹂?꾨? ?덉륫 ??EKF 諛쒖궛.
        //   ?덈룄????gyr > threshold ???섑뵆 鍮꾩쑉??MIN_DYNAMIC_FRACTION 誘몃쭔?대㈃ 嫄대꼫?.
        //   cloneHistory ?붽뎄(~1珥?? 留욌Ъ???쇳빀 ?덈룄???낅뜲?댄듃瑜??댁쨷?쇰줈 李⑤떒.
        val dynamicFrac = computeDynamicFraction(window)
        if (dynamicFrac < MIN_DYNAMIC_FRACTION) {
            Log.v(TAG, "?덈룄???숈쟻 鍮꾩쑉 遺議?(${"%.2f".format(dynamicFrac)} < $MIN_DYNAMIC_FRACTION) ???낅뜲?댄듃 ?ㅽ궢")
            return
        }

        // ??body frame ??gravity-aligned world frame 醫뚰몴 蹂??
        //    t_begin ?대줎???뚯쟾 ?됰젹濡?yaw瑜??쒓굅???붾뱶 ?꾨젅?꾩쑝濡?蹂??
        //    ?ㅽ듃?뚰겕?????꾨젅?꾩뿉???숈뒿?섏뿀??(Python dataset.py acc_ga / gyr_ga).
        val R_begin = EkfBridge.getCloneRotation(tBegin)
        val worldWindow = if (R_begin.size == 9) {
            // [YAW-DIAG] yaw drift 진단 — EKF 의 R_begin 에서 추출한 yaw vs RotVec 의 yaw 비교.
            // 보행 시 ekfYaw 가 빠르게 회전하면 dispLocal(body) 의 world 변환 방향이 매번 달라
            // xy 궤적이 지그재그가 됨. rvAcc 가 2/3 미만이면 RotVec 자체가 부정확.
            val ekfYawDeg = Math.toDegrees(atan2(R_begin[3], R_begin[0]))
            val rvYawRad  = imuCollector.getLatestYawRad()
            val rvYawDeg  = if (rvYawRad.isNaN()) Double.NaN else Math.toDegrees(rvYawRad.toDouble())
            val rvAcc     = imuCollector.getRotVecAccuracy()
            Log.i(TAG, "[YAW-DIAG] ekfYaw=${"%.1f".format(ekfYawDeg)}° " +
                       "rvYaw=${"%.1f".format(rvYawDeg)}° rvAcc=$rvAcc/3")
            transformWindowToWorldFrame(window, R_begin)
        } else {
            Log.w(TAG, "?대줎 ?뚯쟾 ?놁쓬 (tBegin=$tBegin) ??body frame 洹몃?濡??ъ슜 (?뺥솗?????")
            window
        }

        // ???ㅽ듃?뚰겕 異붾줎 ?ㅽ뻾
        val inferStart = System.currentTimeMillis()
        val result = try {
            inferEngine.infer(worldWindow)
        } catch (e: Exception) {
            Log.e(TAG, "추론 예외 (catch) [${e.javaClass.simpleName}]: ${e.message}")
            return
        }
        val inferLatency = System.currentTimeMillis() - inferStart

        // ??Context-Aware Adaptive EKF: 遺꾨쪟 ?뺣쪧 踰≫꽣濡?Q/R ?뚰봽???ㅼ쐞移?
        //    ?쇰Ц 짠4.3.2: R_adaptive = 誇 p_k쨌R^(k), Q_adaptive = 誇 p_k쨌Q^(k)
        //    EKF_NEW 紐⑤뜽(遺꾨쪟湲??놁쓬)? clsProb=zeros ??handheld 湲곗?媛??대갚
        EkfBridge.applySoftSwitching(result.clsProb)

        // ??蹂??痢≪젙媛?+ 怨듬텇??援ъ꽦
        val meas = doubleArrayOf(
            result.disp[0].toDouble(),
            result.disp[1].toDouble(),
            result.disp[2].toDouble()
        )
        val cov = buildCovMatrix(result.dispCov)
        // [DEBUG-5 revert] 사용자 평가 = 사진 시점 (DEBUG-5 *없는* 상태) 이 baseline.
        // z drift fix 는 별도 후속 commit 또는 다음 세션 검증 후 재도입 예정.

        // ??t_begin ?몃뜳??(二쇰???湲곗?) ??localCloneHistory ?먯꽌 吏곸젒 ?먯깋
        val beginIdx = localCloneHistory.indexOfFirst { it == tBegin }

        // ??post: 鍮꾩젙??蹂???꾪꽣留?
        //   醫뚰몴 蹂???ㅻ쪟 ?먮뒗 ?ㅽ듃?뚰겕 ?댁긽 異쒕젰 ??臾쇰━?곸쑝濡?遺덇??ν븳 蹂???쒓굅.
        //   MAX_DISP_PER_WINDOW_M(6.0m) 珥덇낵 = ?ㅻ궡 理쒕??띾룄(~5m/s) 횞 1s 珥덇낵 ??嫄대꼫?.
        val dispNorm = sqrt(meas[0] * meas[0] + meas[1] * meas[1] + meas[2] * meas[2])
        // [DEBUG-3] InferenceEngine 출력 진단 — under-prediction 여부 정량 확인
        // 보행 1초 윈도우 GT ≈ 0.5~1.5m. 그보다 훨씬 작으면 학습 분포 OOD 가능성.
        val xyNorm = sqrt(meas[0] * meas[0] + meas[1] * meas[1])
        val topProbInfer = result.clsProb.maxOrNull() ?: 0f
        // [P41 진단] clsProb 의 raw 값 + sum + maxLogit 출력 — softmax 미적용 여부 확정용.
        //   sum 이 1.0 근처 = softmax 적용됨 / sum 이 임의 값 = raw logits.
        //   maxLogit 와 topProbInfer (= max(clsProb)) 가 동일하므로 변수명만 'topLogitInfer' 가 정확.
        val clsSumDbg = result.clsProb.sum()
        Log.i(TAG, "[INFER-OUT] cls=${result.topClass}(${result.className}) " +
                "p=${"%.3f".format(topProbInfer)} " +
                "disp=[${"%.4f".format(meas[0])}, ${"%.4f".format(meas[1])}, ${"%.4f".format(meas[2])}] " +
                "|xy|=${"%.4f".format(xyNorm)}m |3d|=${"%.4f".format(dispNorm)}m " +
                "dispCovRaw=[${"%.3f".format(result.dispCov[0])}, ${"%.3f".format(result.dispCov[1])}, ${"%.3f".format(result.dispCov[2])}] " +
                "clsSum=${"%.3f".format(clsSumDbg)} clsRaw=[${result.clsProb.joinToString(",") { "%.2f".format(it) }}]")
        if (dispNorm > MAX_DISP_PER_WINDOW_M) {
            Log.w(TAG, "鍮꾩젙??蹂??(${"%.2f".format(dispNorm)}m) ??EKF ?낅뜲?댄듃 嫄대꼫?")
            return
        }

        // ??model: [?꾩씠?붿뼱 5] 紐⑤뜽 ?⑤룆 ?꾩튂 ?꾩쟻 ??EKF ?띾룄 寃뚯씠???곸슜.
        //
        //  吏곸쟾 EKF 媛깆떊 ???띾룄 ?ш린(prevEkfVelNorm)媛 MODEL_VELOCITY_GATE 誘몃쭔?대㈃
        //  ?꾩쟻??李⑤떒?쒕떎.
        //  ?뚢? ?댁쑀: ?ㅽ듃?뚰겕???뺤? ?쒖뿉??non-zero displacement 瑜?異쒕젰(諛붿씠?댁뒪).
        //  ??       EKF ?띾룄媛 異⑸텇???묒쑝硫??ㅼ젣濡??뺤? 以묒엫???섎? ???꾩쟻 ?듭젣.
        //  ?붴? prevEkfVelNorm: 吏곸쟾 ?ㅽ뀦??EKF update() ???띾룄瑜???ν빐 ??媛?
        //     (?대쾲 ?ㅽ뀦 update() ???띾룄?대?濡?1 ?ㅽ뀦 吏?????덉슜 媛?ν븳 ?ㅼ감)
        if (prevEkfVelNorm >= MODEL_VELOCITY_GATE) {
            modelPosX += meas[0]
            modelPosY += meas[1]
            modelTrackPoints.add(Pair(modelPosX, modelPosY))
            if (modelTrackPoints.size > 5000) modelTrackPoints.removeAt(0)
        }

        // ??EKF 痢≪젙 媛깆떊 ??t_begin, t_end 紐⑤몢 si_timestamps_us ??議댁옱?댁빞 ??
        // [P41 Dead-Reckoning Bypass] Python 원본 (Notion 2026-05-11) 의 핵심 분기:
        //   trolley (cls 6) 외 모든 클래스에서 EKF.update 우회 → dispLocal 직접 누적.
        //   USE_DEAD_RECKONING_BYPASS = false 로 토글 시 모두 EKF.update (기존 동작).
        val useEkfUpdate = !USE_DEAD_RECKONING_BYPASS ||
                           (result.topClass !in NETWORK_ONLY_CLASSES)

        if (!useEkfUpdate) {
            // Network-only 모드 — EKF.update 우회 + dispLocal 직접 누적.
            // 모드 전환 시 netPos 를 EKF 위치로 동기화 (점프 방지).
            if (prevUsedEkfUpdate) {
                val curP = EkfBridge.getPosition()
                netPosX = curP[0]
                netPosY = curP[1]
                Log.i(TAG, "[BYPASS] EKF→NetOnly 전환 cls=${result.topClass} sync netPos=(${"%.2f".format(netPosX)},${"%.2f".format(netPosY)})")
            }
            // meas (gravity-aligned local frame) → world frame 변위
            //   world = R_z(yaw0) @ local  →  [cos -sin; sin cos] @ [meas0; meas1]
            val yaw0 = atan2(R_begin[3], R_begin[0])
            val cosZ = kotlin.math.cos(yaw0)
            val sinZ = kotlin.math.sin(yaw0)
            val dxWorld = cosZ * meas[0] - sinZ * meas[1]
            val dyWorld = sinZ * meas[0] + cosZ * meas[1]
            netPosX += dxWorld
            netPosY += dyWorld
            Log.i(TAG, "[BYPASS-NET] cls=${result.topClass} dWorld=[${"%.4f".format(dxWorld)},${"%.4f".format(dyWorld)}] netPos=[${"%.3f".format(netPosX)},${"%.3f".format(netPosY)}]")
            prevUsedEkfUpdate = false

            // marginalize 만 호출 (clone history 관리 위해, EKF.propagate 는 계속 진행)
            if (beginIdx >= 0) {
                EkfBridge.marginalize(beginIdx)
                val rm = (beginIdx + 1).coerceAtMost(localCloneHistory.size)
                repeat(rm) { if (localCloneHistory.isNotEmpty()) localCloneHistory.removeFirst() }
            }

            // Yaw 보정은 EKF state.R 위해 계속 호출 (gravity-aligned local frame 정확도 위해)
            applyRotVecYaw("MOVING-NET")

            // UI state 갱신 (netPos 사용)
            val pos = EkfBridge.getPosition()  // EKF state — z 등 용도
            val vel = EkfBridge.getVelocity()
            prevEkfVelNorm = sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2])
            val elapsedMs = System.currentTimeMillis() - startTimeMs
            if (elapsedMs >= WARMUP_DURATION_MS) {
                trackPoints.add(Pair(netPosX, netPosY))   // ★ Net-only 모드 위치
                if (trackPoints.size > 5000) trackPoints.removeAt(0)
            }
            _state.value = _state.value.copy(
                position          = Triple(netPosX, netPosY, pos[2]),
                posStd            = Triple(pos[3], pos[4], pos[5]),
                velocity          = Triple(vel[0], vel[1], vel[2]),
                carryMode         = result.className,
                carryProb         = result.clsProb.maxOrNull() ?: 0f,
                trackPoints       = trackPoints.toList(),
                modelTrackPoints  = modelTrackPoints.toList(),
                inferLatency      = inferLatency
            )
            return  // EKF.update 우회, 이후 코드 (post-update 처리) skip
        }

        // EKF 모드 (trolley 또는 bypass OFF) — 기존 동작.
        // 직전이 NetOnly 였으면 netPos 를 EKF 위치로 동기화 (다음 NetOnly 진입 대비)
        if (!prevUsedEkfUpdate) {
            val curP = EkfBridge.getPosition()
            netPosX = curP[0]
            netPosY = curP[1]
            Log.i(TAG, "[BYPASS] NetOnly→EKF 전환 sync netPos=(${"%.2f".format(netPosX)},${"%.2f".format(netPosY)})")
        }
        prevUsedEkfUpdate = true

        // [P46-JUMP-GATE] EKF.update 직전 위치 백업 + window accPeak 계산.
        // 발동 조건: |Δpos|xy > JUMP_GATE_POS_M  AND  accPeak < JUMP_GATE_ACC_PEAK
        //   → IMU 입력은 정상인데 EKF 위치만 점프 = EKF outlier 확정.
        val jgPreEkfPos: DoubleArray = if (USE_JUMP_GATE) {
            EkfBridge.getPosition().copyOfRange(0, 3)
        } else {
            DoubleArray(0)
        }
        val jgAccPeak: Double = if (USE_JUMP_GATE) computeAccPeak(window) else 0.0

        try {
            Log.i(TAG, "[UPDATE-TRY] tBegin=$tBegin tEnd=$tEnd histSize=${localCloneHistory.size} hist.first=${localCloneHistory.firstOrNull()} hist.last=${localCloneHistory.lastOrNull()}")
            EkfBridge.update(meas, cov, tBegin, tEnd)
        } catch (e: Exception) {
            Log.w(TAG, "EKF update ?ㅽ뙣: ${e.message}")
            return
        }

        // ??P10: ?ы썑 ?띾룄 ?덉쟾留?(Reactive Divergence Recovery)
        //   EKF update() ???띾룄媛 MAX_POST_UPDATE_SPEED 珥덇낵 ??諛쒖궛 ?먯젙.
        //   媛뺤젣 ZUPT(sigma=0.01 m/s, 嫄곗쓽 ?섎뱶 由ъ뀑 ?섏?)濡??띾룄瑜?0?쇰줈 ?뺤텞.
        //   ?댄썑 EKF ???뺤긽 ?곹깭濡?蹂듦? ??沅ㅼ쟻 ?먰봽??諛쒖깮?섎굹 吏??諛쒖궛? 諛⑹?.
        val postUpdateVel = EkfBridge.getVelocity()
        val postUpdateSpeed = sqrt(
            postUpdateVel[0] * postUpdateVel[0] +
            postUpdateVel[1] * postUpdateVel[1] +
            postUpdateVel[2] * postUpdateVel[2]
        )
        // [DEBUG-3] EKF update 직후 위치/속도 진단
        val postUpdatePos = EkfBridge.getPosition()
        Log.i(TAG, "[UPDATE-RES] ekfPos=[${"%.3f".format(postUpdatePos[0])}, ${"%.3f".format(postUpdatePos[1])}, ${"%.3f".format(postUpdatePos[2])}]m " +
                "ekfSpeed=${"%.3f".format(postUpdateSpeed)}m/s " +
                "meas|xy|=${"%.4f".format(sqrt(meas[0]*meas[0]+meas[1]*meas[1]))}m")
        if (postUpdateSpeed > MAX_POST_UPDATE_SPEED) {
            Log.w(TAG, "諛쒖궛 媛먯?: ?띾룄 ${"%.2f".format(postUpdateSpeed)} m/s > ${MAX_POST_UPDATE_SPEED} ??ZUPT 媛뺤젣 ?곸슜")
            EkfBridge.applyZupt(0.01)
        }

        // [P46-JUMP-GATE] update 후 위치 점프 검사.
        //   조건: Δekf_pos xy > 임계 (위치 튐) AND raw linAcc peak < 임계 (IMU 입력 정상)
        //   발동 시: applyPositionHold(pre, 0.01) + applyZupt(0.01) + marginalize +
        //            clone pop + UI 갱신 skip + return.
        //   accPeak 만 큰 경우 (실제 큰 움직임) 는 통과시키되 진단 로그.
        if (USE_JUMP_GATE && jgPreEkfPos.size == 3) {
            val dxPos  = postUpdatePos[0] - jgPreEkfPos[0]
            val dyPos  = postUpdatePos[1] - jgPreEkfPos[1]
            val dxyPos = sqrt(dxPos * dxPos + dyPos * dyPos)
            if (dxyPos > JUMP_GATE_POS_M) {
                if (jgAccPeak < JUMP_GATE_ACC_PEAK) {
                    // EKF outlier 확정 — 발동
                    Log.w(TAG, "[JUMP-GATE] 발동: |Δpos|xy=${"%.3f".format(dxyPos)}m > $JUMP_GATE_POS_M, " +
                            "accPeak=${"%.2f".format(jgAccPeak)}m/s² < $JUMP_GATE_ACC_PEAK → PositionHold+ZUPT")
                    EkfBridge.applyPositionHold(jgPreEkfPos[0], jgPreEkfPos[1], jgPreEkfPos[2], 0.01)
                    EkfBridge.applyZupt(0.01)
                    if (beginIdx >= 0) {
                        EkfBridge.marginalize(beginIdx)
                        val rm = (beginIdx + 1).coerceAtMost(localCloneHistory.size)
                        repeat(rm) { if (localCloneHistory.isNotEmpty()) localCloneHistory.removeFirst() }
                    }
                    prevEkfVelNorm = 0.0
                    return
                } else {
                    // 위치는 튀었으나 가속도가 큰 정상 움직임 — 통과시키되 진단
                    Log.i(TAG, "[JUMP-GATE] 통과: |Δpos|xy=${"%.3f".format(dxyPos)}m > $JUMP_GATE_POS_M 이나 " +
                            "accPeak=${"%.2f".format(jgAccPeak)}m/s² ≥ $JUMP_GATE_ACC_PEAK (실제 큰 움직임)")
                }
            }
        }

        // ??二쇰??? tBegin ?ы븿 洹??댁쟾 ?대줎 紐⑤몢 ?쒓굅
        //    C++ marginalize(idx) ??0..idx ?ы븿 ??젣 (rm = idx+1)
        if (beginIdx >= 0) {
            EkfBridge.marginalize(beginIdx)
            val rm = (beginIdx + 1).coerceAtMost(localCloneHistory.size)
            repeat(rm) { if (localCloneHistory.isNotEmpty()) localCloneHistory.removeFirst() }
        }

        // ??Yaw drift 蹂댁젙 (?대룞 以?: EKF update 吏곹썑 二쇱엯
        applyRotVecYaw("MOVING")

        // ???꾩튂/?띾룄 議고쉶 ??UI 媛깆떊
        val pos = EkfBridge.getPosition()  // [px, py, pz, sx, sy, sz]
        val vel = EkfBridge.getVelocity()  // [vx, vy, vz]

        // [?꾩씠?붿뼱 5] ?ㅼ쓬 ?ㅽ뀦??紐⑤뜽 only 寃뚯씠?낆쓣 ?꾪빐 ?꾩옱 ?띾룄 ?ш린 ???
        prevEkfVelNorm = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])

        // [P10] ?뚮컢??湲곌컙 ?숈븞 沅ㅼ쟻 ?쒖떆 ?듭젣
        // EKF ??怨꾩냽 ?ㅽ뻾(諛붿씠?댁뒪 ?섎졃쨌怨듬텇???덉젙?붋룻겢濡??꾩쟻)?섎릺
        // ?붾㈃?먮뒗 WARMUP_DURATION_MS ?댄썑遺???먯쓣 李띻린 ?쒖옉.
        val elapsedMs = System.currentTimeMillis() - startTimeMs
        if (elapsedMs >= WARMUP_DURATION_MS) {
            trackPoints.add(Pair(pos[0], pos[1]))
            if (trackPoints.size > 5000) trackPoints.removeAt(0)
        }

        _state.value = _state.value.copy(
            position          = Triple(pos[0], pos[1], pos[2]),
            posStd            = Triple(pos[3], pos[4], pos[5]),
            velocity          = Triple(vel[0], vel[1], vel[2]),
            carryMode         = result.className,
            carryProb         = result.clsProb.maxOrNull() ?: 0f,
            trackPoints       = trackPoints.toList(),
            modelTrackPoints  = modelTrackPoints.toList(),
            inferLatency      = inferLatency
        )
    }
    // ── [P55] RotVec Dead-Reckoning 스텝 (20Hz 속도 적분) ──────────
    /**
     * EKF 를 우회하고 모델 출력을 *속도* 로 환산해 매 추론 틱(20Hz)마다 적분한다.
     *
     * P53/P54 의 1초 비겹침 누적 + turn-skip(회전 윈도우 폐기) → 1Hz 갱신 + 회전 중
     * 완전 정지 문제를 해소. 모델 출력(1초 윈도우 변위)을 speed=|disp|/1s 로 환산,
     * 겹침 윈도우를 속도로 다뤄 과적분 없이 매 틱 dt 만큼 적분한다.
     *
     *  1. dt = 직전 틱 이후 경과
     *  2. transformWindowRotVec → gravity-aligned 입력 → 모델 추론 → disp
     *  3. speed = |disp| / 윈도우길이(≈1s)
     *  4. 정지 윈도우 → speed 0 / 제자리 회전 윈도우 → speed 감쇠 (멈춤 아님)
     *  5. 방향 = 최신 rotVec heading (PDR). 속도 EMA 평활 후 dt 적분.
     *  6. UI 는 매 틱 갱신 → 회전·정지 어떤 상황에서도 앱이 멈추지 않음.
     */
    private fun runRotVecDrStep() {
        val latestTs = imuCollector.getLatestSample()?.ts_us ?: return
        val window   = imuCollector.getRawWindow()    ?: return
        val rotMats  = imuCollector.getRotMatWindow() ?: return

        // 적분 dt (초) — 첫 틱은 기준만 잡고 반환
        if (lastDrTickTs < 0L) { lastDrTickTs = latestTs; return }
        var dt = (latestTs - lastDrTickTs) / 1_000_000.0
        lastDrTickTs = latestTs
        if (dt <= 0.0) return
        if (dt > 0.2) dt = 0.2          // 틱 누락 시 큰 점프 방지

        val ws = ImuCollector.WINDOW_SIZE
        fun headingAt(t: Int) =
            atan2(rotMats[t * 9 + 3].toDouble(), rotMats[t * 9].toDouble())

        // 윈도우 내 heading 변화 (제자리 회전 감쇠 판정용)
        var yawSpan = headingAt(ws - 1) - headingAt(0)
        while (yawSpan >  Math.PI) yawSpan -= 2.0 * Math.PI
        while (yawSpan < -Math.PI) yawSpan += 2.0 * Math.PI

        // RotVec gravity-aligned 변환 + 추론
        val worldWindow = transformWindowRotVec(window, rotMats)
        val inferStart  = System.currentTimeMillis()
        val result = try {
            inferEngine.infer(worldWindow)
        } catch (e: Exception) {
            Log.e(TAG, "[DR] 추론 예외: ${e.javaClass.simpleName}: ${e.message}")
            return
        }
        val inferLatency = System.currentTimeMillis() - inferStart

        // 모델 출력(≈1초 윈도우 변위) → 속도(m/s)
        val d0     = result.disp[0].toDouble()
        val d1     = result.disp[1].toDouble()
        val winSec = WINDOW_DURATION_US / 1_000_000.0
        var speed  = sqrt(d0 * d0 + d1 * d1) / winSec

        // [P57] HANDHELD-only 단일 스케일 — 분류기 출력은 표시 전용(아래 carryMode).
        // 모델의 Android 변위 ~30~40% 과소 → 균일 ~1.5× 보정.
        speed *= HANDHELD_SPEED_SCALE

        // 정지 윈도우 → 속도 0 (위치 고정, 단 UI 는 계속 갱신 = 안 멈춤)
        val dynamicFrac = computeDynamicFraction(window)
        val isStatic    = dynamicFrac < MIN_DYNAMIC_FRACTION
        if (isStatic) speed = 0.0
        // 제자리 회전 윈도우 → 속도 감쇠 (병진 적으나 0 은 아님 → 추적 유지)
        val turnDeg = Math.toDegrees(abs(yawSpan))
        val isTurn  = turnDeg > TURN_YAW_THRESH_DEG
        if (isTurn) speed *= TURN_SPEED_ATTEN

        // 진행 방향 — PDR: 최신 rotVec heading / 토글 OFF: 모델 disp 벡터 방향
        val targetVx: Double
        val targetVy: Double
        if (USE_PDR_HEADING) {
            val h = headingAt(ws - 1)
            targetVx = speed * cos(h)
            targetVy = speed * sin(h)
        } else {
            val yaw0 = headingAt(0)
            targetVx = (cos(yaw0) * d0 - sin(yaw0) * d1) / winSec
            targetVy = (sin(yaw0) * d0 + cos(yaw0) * d1) / winSec
        }

        // 속도 EMA 평활 (틱 jitter 완화). 정지면 즉시 0.
        if (isStatic) {
            drVelX = 0.0; drVelY = 0.0
        } else {
            drVelX += DR_VEL_EMA * (targetVx - drVelX)
            drVelY += DR_VEL_EMA * (targetVy - drVelY)
        }

        // 20Hz 연속 적분
        netPosX += drVelX * dt
        netPosY += drVelY * dt
        modelPosX = netPosX
        modelPosY = netPosY

        // 주기적 로그 (~1초마다)
        if (drTickCount++ % 20 == 0) {
            Log.i(TAG, "[DR] cls=${result.topClass}(${result.className}) " +
                    "scale=${"%.2f".format(HANDHELD_SPEED_SCALE)} " +
                    "speed=${"%.2f".format(sqrt(drVelX * drVelX + drVelY * drVelY))}m/s " +
                    "${if (isStatic) "STATIC " else ""}${if (isTurn) "TURN " else ""}" +
                    "netPos=[${"%.2f".format(netPosX)}, ${"%.2f".format(netPosY)}]")
        }

        // trackPoint 는 일정 거리 이상 이동 시에만 추가 (리스트 비대 방지), 워밍업 이후
        // [P56] 경로 B 는 추정 궤적이 하나 → modelTrackPoints 미사용(범례 단일화).
        val elapsedMs = System.currentTimeMillis() - startTimeMs
        if (elapsedMs >= WARMUP_DURATION_MS) {
            val last = trackPoints.lastOrNull()
            val moved = last == null || kotlin.math.hypot(
                netPosX - last.first, netPosY - last.second) >= DR_TRACKPOINT_MIN_MOVE
            if (moved) {
                trackPoints.add(Pair(netPosX, netPosY))
                if (trackPoints.size > 5000) trackPoints.removeAt(0)
            }
        }

        // UI 는 매 틱(20Hz) 갱신 — 정지·회전 중에도 멈추지 않음
        _state.value = _state.value.copy(
            position          = Triple(netPosX, netPosY, 0.0),
            posStd            = Triple(0.0, 0.0, 0.0),
            velocity          = Triple(drVelX, drVelY, 0.0),
            carryMode         = result.className,
            carryProb         = result.clsProb.maxOrNull() ?: 0f,
            trackPoints       = trackPoints.toList(),
            modelTrackPoints  = emptyList(),
            inferLatency      = inferLatency
        )
    }

    // ── [P53] RotVec per-timestep gravity-aligned 프레임 변환 ──────
    /**
     * window 6채널(body frame linAcc+gyr)을 RotVec 절대 회전행렬로 gravity-aligned
     * world frame 으로 변환. 학습 dataset.py `_window_to_gravity_aligned`(frame='ga') 와 동일:
     *   v_ga[t] = R_yaw_inv · R_rotvec[t] · v_body[t]
     *   R_yaw_inv = R_z(-yaw0),  yaw0 = window 시작 RotVec yaw
     *
     * EKF clone rotation 기반 transformWindowToWorldFrame 과 달리 자력계 융합 절대
     * 자세를 쓰므로 yaw drift 가 없다 (하니스 'ga' frame 과 정확히 동일).
     *
     * @param window  channel-major FloatArray[6 × WINDOW_SIZE] (ch0-2 linAcc, ch3-5 gyr)
     * @param rotMats channel-major FloatArray[9 × WINDOW_SIZE] — rotMat[t]=flat[t*9 .. t*9+8]
     */
    private fun transformWindowRotVec(window: FloatArray, rotMats: FloatArray): FloatArray {
        val ws  = ImuCollector.WINDOW_SIZE
        val out = FloatArray(window.size)

        // 윈도우 시작 yaw 제거: R_yaw_inv = R_z(-yaw0)
        val yaw0 = atan2(rotMats[3].toDouble(), rotMats[0].toDouble())
        val cosZ = cos(yaw0)
        val sinZ = sin(yaw0)

        for (t in 0 until ws) {
            val b = t * 9
            val r0 = rotMats[b].toDouble();     val r1 = rotMats[b + 1].toDouble(); val r2 = rotMats[b + 2].toDouble()
            val r3 = rotMats[b + 3].toDouble(); val r4 = rotMats[b + 4].toDouble(); val r5 = rotMats[b + 5].toDouble()
            val r6 = rotMats[b + 6].toDouble(); val r7 = rotMats[b + 7].toDouble(); val r8 = rotMats[b + 8].toDouble()

            // M = R_yaw_inv · R   (R_yaw_inv = [[cosZ, sinZ, 0], [-sinZ, cosZ, 0], [0, 0, 1]])
            val m00 =  cosZ * r0 + sinZ * r3
            val m01 =  cosZ * r1 + sinZ * r4
            val m02 =  cosZ * r2 + sinZ * r5
            val m10 = -sinZ * r0 + cosZ * r3
            val m11 = -sinZ * r1 + cosZ * r4
            val m12 = -sinZ * r2 + cosZ * r5
            val m20 = r6; val m21 = r7; val m22 = r8

            // linAcc (ch 0-2)
            val lx = window[0 * ws + t].toDouble()
            val ly = window[1 * ws + t].toDouble()
            val lz = window[2 * ws + t].toDouble()
            out[0 * ws + t] = (m00 * lx + m01 * ly + m02 * lz).toFloat()
            out[1 * ws + t] = (m10 * lx + m11 * ly + m12 * lz).toFloat()
            out[2 * ws + t] = (m20 * lx + m21 * ly + m22 * lz).toFloat()
            // gyr (ch 3-5)
            val gx = window[3 * ws + t].toDouble()
            val gy = window[4 * ws + t].toDouble()
            val gz = window[5 * ws + t].toDouble()
            out[3 * ws + t] = (m00 * gx + m01 * gy + m02 * gz).toFloat()
            out[4 * ws + t] = (m10 * gx + m11 * gy + m12 * gz).toFloat()
            out[5 * ws + t] = (m20 * gx + m21 * gy + m22 * gz).toFloat()
        }
        return out
    }

    // ── 헬퍼: 시작 클론 탐색 ─────────────────────────────────────
    /**
     * localCloneHistory 에서 (tEndUs - WINDOW_DURATION_US) 에 가장 가까운 항목 반환.
     * 허용 오차 CLONE_MATCH_TOL_US 초과 시 -1L 반환.
     * inferJob 전용 history 를 직접 참조하므로 락 불필요.
     */
    private fun findBeginClone(tEndUs: Long, history: ArrayDeque<Long>): Long {
        if (history.isEmpty()) return -1L
        val target   = tEndUs - WINDOW_DURATION_US
        var best     = history[0]
        var bestDiff = abs(best - target)
        for (ts in history) {
            val diff = abs(ts - target)
            if (diff < bestDiff) { bestDiff = diff; best = ts }
        }
        return if (bestDiff <= CLONE_MATCH_TOL_US) best else -1L
    }

    // ── 헬퍼: 변위 공분산 행렬 구성 ────────────────────────────
    /**
     * 네트워크 출력 log-variance → variance 변환 후 3×3 대각 행렬 (row-major).
     *
     * Python 원본 (meas_source_torchscript.py):
     *   meas_cov[meas_cov < -4] = -4   ← exp(-4) ≈ 0.018 m² 가 최소 분산
     *
     * 추가로 MIN_MEAS_COV 로 바닥 클램프 — 네트워크가 과도하게 자신감 있는 예측을
     * 할 때 EKF 가 맹목적으로 따라가는 것을 방지 (Kalman 게인이 1 에 가까워지지 않도록).
     */
    private fun buildCovMatrix(dispCov: FloatArray): DoubleArray {
        // [P41 #C FIX] 모델 출력 dispCov[i] = pred_log_std (log standard deviation, TLIO 표준)
        //   확정 근거: bypass01 측정에서 dispCovRaw 범위 -6~-3.4 (mean -4.3) = log_std 범위
        //   meas_source_torchscript.py 의 outputs['pred_log_std'] + DiagonalParam.vec2Cov
        //   → variance = exp(2 * log_std)   (NOT exp(log_std))
        //
        //   기존 코드 (`exp(log_std)`) 는 *std 를 variance 로 잘못 사용* → 모든 EKF.update
        //   에서 cov 가 √cov 배 (~7×) 크게 들어가 Kalman gain 작음 → dispLocal 측정값
        //   덜 반영 → under-prediction 일관 발생. 이 fix 가 모든 update 정확도 회복.
        //
        //   롤백 시: `exp(2.0 * logStd)` → `exp(logStd)` 로 1줄.
        val cov = DoubleArray(9) { 0.0 }
        for (i in 0 until 3) {
            val logStd   = dispCov[i].toDouble().coerceAtLeast(-4.0)    // Python: clip < -4
            val variance = kotlin.math.exp(2.0 * logStd).coerceAtLeast(MIN_MEAS_COV)
            cov[i * 3 + i] = variance
        }
        return cov
    }

    // ── 헬퍼: 윈도우 자이로 RMS ────────────────────────────────
    /**
     * 윈도우 (channel-major FloatArray[6 × WINDOW_SIZE]) 의 자이로 채널 (ch 3-5) 의
     * 전체 sample-axis RMS 를 계산. 정지/이동 판정용.
     *
     * 정지 MEMS 자이로 노이즈 ≈ 0.003-0.01 rad/s RMS.
     * 보행 자이로 ≈ 0.1-0.5 rad/s RMS.
     * STATIC_GYR_RMS_THRESHOLD = 0.08 rad/s 가 두 경계의 중간값.
     */
    private fun computeGyrRms(window: FloatArray): Float {
        val ws = ImuCollector.WINDOW_SIZE
        var sumSq = 0.0
        var n = 0
        for (c in 3..5) {
            for (t in 0 until ws) {
                val v = window[c * ws + t].toDouble()
                sumSq += v * v
                n++
            }
        }
        return if (n > 0) kotlin.math.sqrt(sumSq / n).toFloat() else 0f
    }

    // ── 헬퍼: 윈도우 동적 sample 비율 ───────────────────────────
    /**
     * 윈도우 100 샘플 중 자이로 벡터 노름이 STATIC_GYR_RMS_THRESHOLD 를 초과한 비율.
     *
     * [P10] 최초 이동 시 윈도우는 [정지 85%][이동 15%] 혼합 → 네트워크 출력 오염.
     * 이 비율이 MIN_DYNAMIC_FRACTION (0.5) 미만이면 추론을 건너뛴다.
     */
    private fun computeDynamicFraction(window: FloatArray): Float {
        val ws = ImuCollector.WINDOW_SIZE
        val thr = STATIC_GYR_RMS_THRESHOLD.toDouble()
        var dynamicCount = 0
        for (t in 0 until ws) {
            val gx = window[3 * ws + t].toDouble()
            val gy = window[4 * ws + t].toDouble()
            val gz = window[5 * ws + t].toDouble()
            val gNorm = kotlin.math.sqrt(gx * gx + gy * gy + gz * gz)
            if (gNorm > thr) dynamicCount++
        }
        return dynamicCount.toFloat() / ws.toFloat()
    }

    // ── 헬퍼: 윈도우 linAcc magnitude peak ────────────────────────
    /**
     * 윈도우 (channel-major FloatArray[6 × WINDOW_SIZE]) 의 linAcc 채널 (ch 0-2) 의
     * 시점별 magnitude (sqrt(ax² + ay² + az²)) 의 최댓값.
     *
     * 점프 방지 게이트의 IMU 정합성 검증에 사용:
     *  - 정상 보행 peak: ~3-6 m/s²
     *  - 빠른 보행/충격: 10+ m/s²
     *  - 정지: ~0.1-0.3 m/s²
     * peak 가 JUMP_GATE_ACC_PEAK 미만인데 ekf 위치만 점프하면 EKF outlier 확정.
     *
     * 참고: 여기서 window 는 transformWindowToWorldFrame() *전* 의 raw body frame
     * linAcc (Android TYPE_LINEAR_ACCELERATION, m/s²). 좌표계 변환은 magnitude
     * 에 영향 없음 (||R·v|| = ||v||).
     */
    private fun computeAccPeak(window: FloatArray): Double {
        val ws = ImuCollector.WINDOW_SIZE
        var peak = 0.0
        for (t in 0 until ws) {
            val ax = window[0 * ws + t].toDouble()
            val ay = window[1 * ws + t].toDouble()
            val az = window[2 * ws + t].toDouble()
            val norm = kotlin.math.sqrt(ax * ax + ay * ay + az * az)
            if (norm > peak) peak = norm
        }
        return peak
    }

    // ── 헬퍼: TYPE_ROTATION_VECTOR yaw 측정 주입 ─────────────────
    /**
     * Android TYPE_ROTATION_VECTOR 의 yaw 를 EKF 에 주입 (자이로 적분 drift 보정).
     *
     * 보정 공식:
     *   yaw_meas (EKF 월드 프레임 기준) = yaw_rv_current − yaw_rv_at_init
     *
     * 게이팅:
     *   ① yawRvAtInit 가 NaN (오프셋 미초기화 — 자기계 정확도 낮을 때) → skip
     *   ② imuCollector.getLatestYawRad() NaN (rotVecSensor 없음 또는 미수신) → skip
     *   ③ 자기계 정확도 < MEDIUM (2) → skip (UNRELIABLE / LOW 시 부정확)
     *
     * @param label 호출 분기 식별자 ("STATIC" / "MOVING") — 로그 구분용
     */
    private fun applyRotVecYaw(label: String) {
        if (yawRvAtInit.isNaN()) return
        val yawRv = imuCollector.getLatestYawRad()
        if (yawRv.isNaN()) return
        val rvAcc = imuCollector.getRotVecAccuracy()
        if (rvAcc < 2) return

        // [P41 YAW-WRAP FIX] yawRv 와 yawRvAtInit 가 모두 atan2 결과 (±π wrap)
        //   단순 뺄셈은 ±π 경계 통과 시 ±2π fictitious 점프 발생 (실제 회전 N° → -360+N° 또는 +360+N°).
        //   이 잘못된 yawMeas 가 EKF 에 매 추론(20Hz)마다 주입 → state.yaw 가 누적 fictitious drift.
        //   →  실측: walk30s 30초간 -850°, walk_clear 22초간 -358° drift 의 *직접 원인*.
        //   해결: yawMeas 를 ±π 범위로 wrap (실제 회전량만 EKF 에 전달).
        var yawMeas = yawRv.toDouble() - yawRvAtInit
        while (yawMeas >  Math.PI) yawMeas -= 2.0 * Math.PI
        while (yawMeas < -Math.PI) yawMeas += 2.0 * Math.PI
        try {
            EkfBridge.applyYawUpdate(yawMeas, YAW_SIGMA_RAD)
        } catch (e: Exception) {
            Log.w(TAG, "[$label] yaw 주입 실패: ${e.message}")
        }
    }

    // ── 헬퍼: body frame 윈도우 → gravity-aligned world frame ───
    /**
     * 윈도우의 6 채널 (linAcc body + gyr body) 을 t_begin 클론의 회전 행렬로
     * gravity-aligned world frame (= yaw 가 제거된 world frame) 으로 변환.
     *
     * 학습 데이터 (Python dataset.py 의 acc_ga / gyr_ga) 와 동일한 좌표계로
     * 만들어 네트워크 입력 분포 mismatch 를 최소화.
     *
     * 변환:
     *   R_ga_from_body = R_yaw_inv · R_begin
     *   where R_yaw_inv = z-축 (-yaw0) 회전, yaw0 = atan2(R_begin[1,0], R_begin[0,0])
     *
     *   v_ga = R_ga_from_body · v_body
     *
     * row-major 인덱싱: R_begin[row*3 + col] → R[0,0]=R_begin[0], R[1,0]=R_begin[3], …
     *
     * @param window  channel-major FloatArray[6 × WINDOW_SIZE] (ch 0-2 linAcc body, ch 3-5 gyr body)
     * @param R_begin t_begin 시점 EKF 클론의 world←body 회전 (9-element row-major)
     * @return 동일 형식의 yaw-free world frame 윈도우
     */
    private fun transformWindowToWorldFrame(window: FloatArray, R_begin: DoubleArray): FloatArray {
        if (R_begin.size != 9) return window  // 안전 fallback

        // [P41 R_all[t] Frame v2] Python `_get_imu_samples_for_network` 와 동일 처리로 분기.
        //   USE_R_ALL_T_FRAME = true → 매 시점 자이로 적분 R[t] (Python 동일)
        //     v2: raw gyr (P21 차감 후, LPF 안 됨) 사용 — getRawGyrWindow() 호출.
        //     bg 추가 차감 안 함 (이중 차감 회피).
        //   false → 기존 R_begin 1개 방식
        if (USE_R_ALL_T_FRAME) {
            val rawGyr = imuCollector.getRawGyrWindow()
            if (rawGyr != null) {
                return transformWindowR_all(window, R_begin, rawGyr)
            }
            // rawGyr null (윈도우 미충족) → fallback 으로 기존 방식
        }

        val ws = ImuCollector.WINDOW_SIZE
        val out = FloatArray(window.size)

        // R_begin 의 yaw 추출 (ZYX Euler convention)
        val yaw0 = atan2(R_begin[3], R_begin[0])
        val cosZ = cos(yaw0)
        val sinZ = sin(yaw0)

        // R_ga_from_body[i][j] = Σ_k R_yaw_inv[i][k] · R_begin[k][j]
        //   R_yaw_inv = | cos(yaw0)  sin(yaw0)  0 |
        //               |-sin(yaw0)  cos(yaw0)  0 |
        //               | 0          0          1 |
        val r00 =  cosZ * R_begin[0] + sinZ * R_begin[3]
        val r01 =  cosZ * R_begin[1] + sinZ * R_begin[4]
        val r02 =  cosZ * R_begin[2] + sinZ * R_begin[5]
        val r10 = -sinZ * R_begin[0] + cosZ * R_begin[3]
        val r11 = -sinZ * R_begin[1] + cosZ * R_begin[4]
        val r12 = -sinZ * R_begin[2] + cosZ * R_begin[5]
        val r20 = R_begin[6]
        val r21 = R_begin[7]
        val r22 = R_begin[8]

        for (t in 0 until ws) {
            // linAcc (ch 0-2)
            val lx = window[0 * ws + t].toDouble()
            val ly = window[1 * ws + t].toDouble()
            val lz = window[2 * ws + t].toDouble()
            out[0 * ws + t] = (r00 * lx + r01 * ly + r02 * lz).toFloat()
            out[1 * ws + t] = (r10 * lx + r11 * ly + r12 * lz).toFloat()
            out[2 * ws + t] = (r20 * lx + r21 * ly + r22 * lz).toFloat()
            // gyr (ch 3-5)
            val gx = window[3 * ws + t].toDouble()
            val gy = window[4 * ws + t].toDouble()
            val gz = window[5 * ws + t].toDouble()
            out[3 * ws + t] = (r00 * gx + r01 * gy + r02 * gz).toFloat()
            out[4 * ws + t] = (r10 * gx + r11 * gy + r12 * gz).toFloat()
            out[5 * ws + t] = (r20 * gx + r21 * gy + r22 * gz).toFloat()
        }
        return out
    }

    // ── [P41 R_all[t]] body frame → world frame, 매 시점 자이로 적분 ──────
    /**
     * Python `_get_imu_samples_for_network` 와 동일 처리.
     *
     * 단계:
     *  1. R_begin 의 yaw 제거 → R_yawfree (= R_oldest_state_wfb)
     *  2. EKF s_bg 가져옴 (동적 자이로 bias)
     *  3. 매 시점 자이로 적분 — Rs_bofbi[t] = Rs_bofbi[t-1] · exp(skew((gyr-bg)·dt))
     *     (begin time frame 기준 누적 회전)
     *  4. Rs_net_wfb[t] = R_yawfree · Rs_bofbi[t]  (시점별 world frame R)
     *  5. acc_world[t] = Rs_net_wfb[t] · acc_body[t]
     *     gyr_world[t] = Rs_net_wfb[t] · gyr_body[t]
     */
    private fun transformWindowR_all(window: FloatArray, R_begin: DoubleArray, rawGyr: FloatArray): FloatArray {
        val ws = ImuCollector.WINDOW_SIZE
        val out = FloatArray(window.size)
        val dt = 1.0 / ImuCollector.TARGET_HZ.toDouble()   // = 0.01 s (100 Hz)

        // 1. R_yawfree = R_yaw_inv · R_begin
        val yaw0 = atan2(R_begin[3], R_begin[0])
        val cosZ = cos(yaw0)
        val sinZ = sin(yaw0)
        val Ryf = DoubleArray(9)
        Ryf[0] =  cosZ * R_begin[0] + sinZ * R_begin[3]
        Ryf[1] =  cosZ * R_begin[1] + sinZ * R_begin[4]
        Ryf[2] =  cosZ * R_begin[2] + sinZ * R_begin[5]
        Ryf[3] = -sinZ * R_begin[0] + cosZ * R_begin[3]
        Ryf[4] = -sinZ * R_begin[1] + cosZ * R_begin[4]
        Ryf[5] = -sinZ * R_begin[2] + cosZ * R_begin[5]
        Ryf[6] = R_begin[6]; Ryf[7] = R_begin[7]; Ryf[8] = R_begin[8]

        // 2. [P41 v2] EKF s_bg 추가 차감 제거 — rawGyr 가 이미 P21 차감된 상태.
        //    EKF s_bg 추가 차감은 *이중 차감* 으로 1차 시도에서 발산 야기 (롤백됨).

        // 3. 매 시점 자이로 적분 — Rs_bofbi[t] (begin time frame 기준)
        //    Rs_bofbi[0] = I (begin 시점은 정렬됨)
        //    메모리 절약 위해 누적 R 만 유지 (이전 R 만 필요)
        val curR = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   // I

        // 4-5. 각 시점 t 마다 누적 R[t] 적용
        for (t in 0 until ws) {
            // 시점 t 의 누적 R[t] (begin frame → body[t] frame)
            // t=0 에서는 curR = I (begin 자체)
            // 이후 dR(rawGyr[t-1] 적분) 으로 갱신 후 사용
            if (t > 0) {
                // [P41 v2] rawGyr (P21 차감 후, LPF 안 됨) 사용. bg 차감 안 함.
                val gx = rawGyr[0 * ws + (t - 1)].toDouble()
                val gy = rawGyr[1 * ws + (t - 1)].toDouble()
                val gz = rawGyr[2 * ws + (t - 1)].toDouble()
                val dR = matExp(gx * dt, gy * dt, gz * dt)
                val newR = mat3Mul(curR, dR)
                System.arraycopy(newR, 0, curR, 0, 9)
            }

            // Rs_net_wfb[t] = R_yawfree · Rs_bofbi[t] = Ryf · curR
            val rNet = mat3Mul(Ryf, curR)

            // acc_world[t] = rNet · acc_body[t]
            val lx = window[0 * ws + t].toDouble()
            val ly = window[1 * ws + t].toDouble()
            val lz = window[2 * ws + t].toDouble()
            out[0 * ws + t] = (rNet[0] * lx + rNet[1] * ly + rNet[2] * lz).toFloat()
            out[1 * ws + t] = (rNet[3] * lx + rNet[4] * ly + rNet[5] * lz).toFloat()
            out[2 * ws + t] = (rNet[6] * lx + rNet[7] * ly + rNet[8] * lz).toFloat()

            // gyr_world[t] = rNet · gyr_body[t]
            val gx = window[3 * ws + t].toDouble()
            val gy = window[4 * ws + t].toDouble()
            val gz = window[5 * ws + t].toDouble()
            out[3 * ws + t] = (rNet[0] * gx + rNet[1] * gy + rNet[2] * gz).toFloat()
            out[4 * ws + t] = (rNet[3] * gx + rNet[4] * gy + rNet[5] * gz).toFloat()
            out[5 * ws + t] = (rNet[6] * gx + rNet[7] * gy + rNet[8] * gz).toFloat()
        }
        return out
    }

    // ── 헬퍼: Rodrigues' formula — exp(skew([rx, ry, rz])) ─────
    /** 회전 벡터 (axis × angle) → 3×3 rotation matrix (row-major) */
    private fun matExp(rx: Double, ry: Double, rz: Double): DoubleArray {
        val theta = sqrt(rx * rx + ry * ry + rz * rz)
        val R = DoubleArray(9)
        if (theta < 1e-10) {
            R[0] = 1.0; R[4] = 1.0; R[8] = 1.0
            return R
        }
        val invT = 1.0 / theta
        val kx = rx * invT; val ky = ry * invT; val kz = rz * invT
        val s  = sin(theta); val c = cos(theta); val mc = 1.0 - c
        R[0] = c + kx*kx*mc;     R[1] = kx*ky*mc - kz*s; R[2] = kx*kz*mc + ky*s
        R[3] = ky*kx*mc + kz*s;  R[4] = c + ky*ky*mc;    R[5] = ky*kz*mc - kx*s
        R[6] = kz*kx*mc - ky*s;  R[7] = kz*ky*mc + kx*s; R[8] = c + kz*kz*mc
        return R
    }

    // ── 헬퍼: 3×3 행렬 곱 (row-major) ───────────────────────
    /** C = A · B  (모두 9-element row-major DoubleArray) */
    private fun mat3Mul(A: DoubleArray, B: DoubleArray): DoubleArray {
        val C = DoubleArray(9)
        for (i in 0..2) {
            for (j in 0..2) {
                var sum = 0.0
                for (k in 0..2) sum += A[i * 3 + k] * B[k * 3 + j]
                C[i * 3 + j] = sum
            }
        }
        return C
    }
}
