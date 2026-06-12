// KIISE 양식 Word(.docx) 생성 — docx-js. 한글 Malgun Gothic, A4, 여백 위30/아래20/좌우10mm.
// 본문 단단(single column): Word에서 Layout→Columns→Two 로 2단 전환 가능.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        ImageRun, AlignmentType, WidthType, BorderStyle, ShadingType,
        HeadingLevel } = require("docx");

const DIR = __dirname;
const FIG = path.join(DIR, "figures");
const FONT = "Malgun Gothic";
const CW = 10772; // content width DXA (A4 11906 - 567 - 567)

// ---- helpers ----
const T = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, italics: o.it, ...o });
const P = (runs, o = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [T(runs, o)],
  alignment: o.align, spacing: { after: o.after == null ? 120 : o.after, line: 264 },
  ...(o.bullet ? { bullet: { level: 0 } } : {}),
});
const H = (num, title) => new Paragraph({
  spacing: { before: 220, after: 120 },
  children: [T(`${num}. ${title}`, { bold: true, size: 24 })],
});
const SH = (title) => new Paragraph({
  spacing: { before: 140, after: 80 },
  children: [T(title, { bold: true, size: 21 })],
});
const EQ = (txt, n) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [T(txt, { it: true }), T(n ? `   (${n})` : "")],
});
const CAP = (txt) => new Paragraph({   // 캡션 (표는 위/그림은 아래에 배치)
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [T(txt, { size: 18 })],
});

function imgPara(file, w) {
  const buf = fs.readFileSync(path.join(FIG, file));
  // 원본 비율 유지 위해 알려진 크기 사용
  const dims = { "fig2_backbone.png":[2262,991],"fig3_handbag.png":[2100,900],
    "fig4_handheld.png":[2100,900],"fig5_trolley.png":[2100,900],
    "fig6_align.png":[1650,780],"fig7_sensitivity.png":[1650,630],
    "fig8_drifttraj.png":[960,870],"fig9_ekfvsrotvec.png":[1800,750]}[file];
  const h = Math.round(w * dims[1] / dims[0]);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing:{before:80,after:40},
    children: [ new ImageRun({ type:"png", data: buf, transformation:{ width:w, height:h },
      altText:{ title:file, description:file, name:file } }) ] });
}

function makeTable(rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const total = widths.reduce((a,b)=>a+b,0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((cells, ri) => new TableRow({ children: cells.map((c, ci) =>
      new TableCell({ borders, width:{size:widths[ci],type:WidthType.DXA},
        shading: ri===0 ? { fill:"D9E2F3", type:ShadingType.CLEAR } : undefined,
        margins:{top:40,bottom:40,left:80,right:80},
        children:[ new Paragraph({ alignment: ci===0?AlignmentType.LEFT:AlignmentType.CENTER,
          spacing:{after:0,line:240},
          children:[ T(String(c), { size:17, bold: ri===0 }) ] }) ] }) ) })) });
}

// ================= 본문 구성 =================
const body = [];
const push = (...x) => x.forEach(e => body.push(e));

// ---- 제목부 ----
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:40},
  children:[ T("스마트폰 IMU 기반 실내 측위:", {bold:true,size:32}) ]}));
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:120},
  children:[ T("적응형 EKF의 한계와 전처리 중심 해법", {bold:true,size:32}) ]}));
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:60},
  children:[ T("［저자명］", {}), T("°  ［소속/학과］   beoteu@gmail.com", {}) ]}));
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:30},
  children:[ T("Smartphone IMU-based Indoor Localization:", {bold:true,size:22}) ]}));
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:40},
  children:[ T("Limits of Adaptive EKF and a Preprocessing-Centric Solution", {bold:true,size:22}) ]}));
push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:160},
  children:[ T("［Author Name］° ［Affiliation］", {size:18}) ]}));

// ---- 요약 ----
const abstract = "GPS와 Wi-Fi·비콘 인프라가 소실되는 재난 환경의 실내 측위를 위해, 본 논문은 스마트폰 내장 IMU 센서만으로 보행자 위치를 추정하는 인프라 비의존 측위를 다룬다. 기존 TLIO·LLIO 계열은 신경망이 예측한 변위·공분산을 EKF에 제공하되 측정 공분산을 단일 고정값으로 운용하여 휴대 상태별 신호 변화를 반영하지 못한다. 본 연구는 휴대 상태를 인식해 EKF 측정 공분산을 가변 조정하는 상태 기반 적응형 EKF를 설계·평가하고, OxIOD 전체 시퀀스에 대한 전처리 정렬 ablation과 합성 yaw 드리프트 민감도 분석을 수행하였다. 그 결과, 카테고리별 측정 공분산을 그리드 서치로 최적화하더라도 EKF 결합은 트롤리를 제외한 모든 휴대 상태에서 네트워크 단독보다 열등하여, 격한 동작에서 2차원 위치 오차(RMSE)가 약 1.3–1.6 m에서 4.7–7.8 m로 3~5배 악화되었다. 반면 정답 변위를 직접 대입하면 트롤리 오차가 0.37 m까지 회복되어, 병목이 EKF 구조가 아니라 입력 측정값 품질에 있음을 확인하였다. 나아가 중력 정렬을 생략하면 정확도가 2.44배 악화되고, 합성 yaw 드리프트는 변위 방향 예측만 선택적으로 7.7배 붕괴시키되 크기 예측은 1.0배로 보존함을 정량적으로 규명하였다. 이로써 측위 품질을 좌우하는 핵심이 측정 공분산 조정이 아니라 중력 정렬과 입력 좌표계의 yaw 안정성을 포함한 전처리의 정확성임을 실증하고, 전처리를 우선적 해결 방향으로 제시한다.";
push(new Paragraph({ shading:{fill:"F2F2F2",type:ShadingType.CLEAR}, spacing:{after:80,line:260},
  children:[ T("요  약   ", {bold:true}), T(abstract, {size:18}) ] }));
push(P([ T("Keywords  ", {bold:true,size:18}),
  T("Inertial Odometry, Indoor Localization, Extended Kalman Filter, Gravity-Aligned Preprocessing, Out-of-Distribution", {size:18}) ], {after:160}));

// ---- 1. 서론 ----
push(H(1,"서론"));
push(P("실내 보행자 위치 추정은 재난 대응에서 중요한 과제다. GPS는 실내에서 위성 신호가 차폐되어 측위가 어렵고, Wi-Fi·비콘 등 인프라 기반 측위[12]는 인프라가 정상 작동한다는 전제에 의존한다. 지진·화재 시 비콘 인프라가 소실되면 인프라 의존형 측위는 가장 필요한 순간에 기능을 상실한다. 농연 속 소방대원에게는 외부 인프라 없이 단말이 자체 측위하는 인프라 비의존 측위가 유일한 수단이며, 가속도계·자이로만 이용하는 관성 항법(IMU)이 유망한 대안이다."));
push(P("그러나 IMU 측위는 이중 적분 과정의 바이어스·노이즈 누적 오차(drift)가 발산하는 한계가 있다. 전통적 PDR[10]은 이를 완화하나 보폭 모델 오차가 누적된다. 최근에는 IMU로부터 운동 정보를 직접 학습하는 데이터 기반 접근이 주류가 되었다."));
push(P("TLIO[1]·LLIO[2]는 신경망이 예측한 변위·공분산을 EKF에 입력하되 측정 노이즈 공분산 R의 스케일을 단일 고정값으로 운용한다. 이는 휴대 상태에 따른 신호 변화를 반영하지 못한다. 이에 본 연구는 휴대 상태 인식 기반 적응형 EKF[11]를 설계·평가하였으나, OxIOD에서 카테고리별 스케일을 그리드 서치로 최적화해도 트롤리를 제외한 전 카테고리에서 EKF가 네트워크 단독보다 열등하였고 격한 동작에서 3~5배 악화되었다. 정답 변위 대입 진단으로 EKF 구조는 정상이며 병목이 입력 측정값 품질임을 확인하였다. 즉 EKF 추정 yaw의 누적 드리프트로 입력 좌표계가 훈련 분포를 이탈(OOD)하면서 예측 신뢰성이 저하된다. 문제의 본질은 필터 파라미터가 아니라 입력 전처리에 있다."));
push(P("본 논문의 기여는 다음과 같다.", {after:40, bold:false}));
push(P("최적 공분산을 부여해도 트롤리 외 전 카테고리에서 EKF가 네트워크 단독보다 열등함을 정량 규명하여 필터단 적응의 한계를 드러낸다.", {bullet:true, after:40}));
push(P("정답 변위 대입으로 병목이 yaw 드리프트에 의한 입력 OOD임을 진단한다.", {bullet:true, after:40}));
push(P("OxIOD 152개 전 시퀀스 ablation으로 중력 정렬이 정확도를 지배하고(생략 시 2.44배 악화), yaw 드리프트가 변위 방향 예측만 7.7배 붕괴시키되 크기는 보존하며 이 해리가 ResNet1D 백본에서도 재현됨을 확증한다.", {bullet:true, after:40}));
push(P("방향 불변 전처리가 측위 품질의 핵심임을 밝히고 전처리가 가변 파라미터 EKF를 대체하는 우선 해결 방향임을 제시한다.", {bullet:true, after:120}));

// ---- 2. 관련 연구 ----
push(H(2,"관련 연구"));
push(P("RONIN[3]은 단말 자세로 IMU를 중력 기준 안정화 좌표계(heading-agnostic)로 변환하는 방향 불변 전처리를 도입해 입력 전처리가 측위 정확도에 직접 기여함을 보였으나, 회귀 전용 구조로 예측 불확실성을 다루지 못한다. TLIO[1]는 1초 윈도우를 중력 정렬 후 1D-ResNet으로 변위·공분산을 예측해 EKF에 결합한다. LLIO[2]는 경량 ResMLP로 스마트폰급 실시간 추론을 달성한다. 세 연구 모두 절대 yaw 기준이 성립하는 환경을 전제해 중력 정렬을 단일 요소로 취급하며, yaw 기준 부재 시 정렬이 무너지는 상황을 다루지 않는다. 적응형 칼만 필터[11]는 필터단 적응을 추구하나, 본 연구는 해결 지점을 입력 전처리로 이동시킨다."));

// ---- 3. 제안 방법 ----
push(H(3,"제안 방법"));
push(SH("3.1 시스템 설계"));
push(P("제안 시스템은 (1) 전처리, (2) 변위·불확실성·휴대 상태를 동시 예측하는 1D-ResMLP128 다중 헤드 신경망, (3) 출력을 누적하는 상태 추정기로 구성된다(그림 1). 학습은 OxIOD(iPhone)[4] 기반이며 추론은 PyTorch Mobile 모델이 단말에서 실행한다. iPhone 학습 모델과 Android 단말의 도메인 차이로 EKF 경로가 발산해, 단말 기본 동작은 RotVec dead-reckoning 경로로 운용된다."));
push(new Paragraph({ shading:{fill:"F7F7F7",type:ShadingType.CLEAR}, spacing:{before:40,after:20,line:220},
  children:[ T(
"[IMU] Accel·Gyro·RotVec (100Hz) → [전처리: 2초 바이어스 영점보정 · per-sample 중력정렬 · 채널정규화 · 1초(100×6) 윈도우] → [1D-ResMLP128: patch embed→ResMLP×6→mean pool→[B,128]] → {회귀:변위 μ | 회귀:공분산 logσ² | 분류 p∈R⁷} → [상태추정기 A:SC-EKF(15-dim) / B:RotVec DR+PDR-hybrid] → 추정 궤적 (x,y)",
  {size:16}) ]}));
push(CAP("그림 1. 시스템 파이프라인 블록도."));

push(SH("3.2 데이터셋 및 데이터 증강"));
push(P("주 데이터셋은 OxIOD[4](iPhone, 6개 휴대 상태 레이블)이며, 보조로 TLIO golden(283 시퀀스, 약 908,149 샘플)을 추가해 클래스 불균형을 완화한다. 입력은 1초(100 샘플) 윈도우를 stride 10으로 생성한 6채널 신호다. 학습 시 표 1의 세 증강을 적용한다. 특히 Yaw Rotation은 매 윈도우를 일관된 단일 yaw로 회전시켜 윈도우 내부 정렬을 보존하므로, 윈도우 시작 yaw가 누적 이탈해 per-sample 정렬이 어긋나는 yaw 드리프트(4.6절)와는 다른 변형이며 누적 드리프트에 의한 OOD를 면역시키지 못한다."));
push(CAP("표 1. 학습 데이터 증강 기법."));
push(makeTable([
  ["증강 기법","내용"],
  ["Bias Noise","가속도 ±0.2 m/s², 자이로 ±0.05 rad/s 랜덤 바이어스"],
  ["Gravity Pert.","중력 방향 ±5° 수평 랜덤 기울임"],
  ["Yaw Rotation","매 윈도우 랜덤 yaw 회전(방향 불변성 강화)"],
], [3000, 7772]));
push(P("",{after:80}));

push(SH("3.3 전처리"));
push(P("전처리는 바이어스 영점 보정, 중력 정렬 좌표 변환, 채널별 정규화로 구성된다. (1) 시작 약 2초 정지 구간 평균을 바이어스로 차감한다."));
push(EQ("b_a = (1/N_c) Σ a^lin_t ,   b_ω = (1/N_c) Σ ω_t","1"));
push(P("(2) 각 시각 자세 q_t로 body→world 회전 후 윈도우 시작 yaw ψ₀만 제거한다."));
push(EQ("a^ga_t = R_z(ψ₀)⁻¹ R(q_t) a_t ,   ω^ga_t = R_z(ψ₀)⁻¹ R(q_t) ω_t","2"));
push(P("이는 기울기 무관 표현과 절대 방위 불변(heading-agnostic) 입력을 동시 제공한다. ablation(4.6절)에 따르면 시간적 입도 차이는 1.08배로 작고, 품질을 지배하는 것은 중력 정렬의 존재(생략 시 2.44배 악화)와 입력 좌표계 절대 yaw 안정성이다. (3) 채널별 표준화."));
push(EQ("ũ_t = (u_t − μ_norm) ⊘ σ_norm","3"));

push(SH("3.4 회귀 헤드: 변위·불확실성"));
push(P("특징 h∈R¹²⁸로부터 변위 평균 μ∈R³과 대각 로그 분산 logσ²를 산출하고 공분산을 복원한다."));
push(EQ("Σ = diag( exp(logσ²) )","4"));
push(P("초기 30 epoch는 MSE로 안정화 후 가우시안 NLL로 전환한다."));
push(EQ("L_MSE = (1/N) Σ ||y_i − μ_i||²","5"));
push(EQ("L_NLL = ½ Σ [ exp(−logσ_i²)(y_i − μ_i)² + logσ_i² ]","6"));
push(P("logσ²는 [−2,2]로 클램핑한다."));

push(SH("3.5 분류 헤드: 휴대 상태"));
push(EQ("p = softmax( f_cls(h) ),   p ∈ R^K   (K=7)","7"));
push(EQ("w_c ∝ 1/√n_c ,   L_cls = − Σ w_c y_c log p_c","8"));
push(P("배포 변위 모델은 순수 회귀로 운용되며 분류 헤드는 별도 학습한 독립 분류기다."));

push(SH("3.6 상태 추정기: SC-EKF"));
push(P("확률적 클로닝 EKF[8,9]로 15차원 오차 상태와 윈도우 경계 클론을 추정한다. 예측 측정값과 노이즈 공분산은 다음과 같다."));
push(EQ("ẑ = R_z(ψ_b)ᵀ (p_e − p_b)","9"));
push(EQ("R = s · Σ ,   s = meascov_scale","10"));
push(EQ("S = HΣHᵀ + R ,  K = ΣHᵀS⁻¹ ,  δx = K r","11"));
push(EQ("Σ⁺ = (I−KH)Σ(I−KH)ᵀ + KRKᵀ","12"));
push(P("서론의 가변 파라미터 접근은 바로 s를 휴대 상태에 따라 조정하려는 시도였다."));

push(SH("3.7 학습 설정"));
push(P("학습은 단일 NVIDIA GeForce RTX 4070 Ti GPU에서 Anaconda 기반 PyTorch 2.7.1(CUDA 11.8)로 수행하였고, 모델은 PyTorch Mobile로 변환되어 단말 CPU에서 추론된다. 하이퍼파라미터는 표 2와 같다."));
push(CAP("표 2. 학습 하이퍼파라미터(config.json 기준)."));
push(makeTable([
  ["항목","설정값"],
  ["옵티마이저","AdamW[7] (lr 1e-4, wd 1e-4)"],
  ["스케줄","CosineAnnealing (T_max=100), warmup 5"],
  ["배치 / epoch","128 / 100 (early stop 30)"],
  ["Grad clip","max-norm 1.0"],
  ["손실 전환","30 epoch MSE → Gaussian NLL"],
  ["윈도우 / stride","100 샘플(1초) / 10 샘플"],
], [3400, 7372]));
push(P("",{after:80}));

// ---- 4. 실험 ----
push(H(4,"실험"));
push(SH("4.1 실험 설정"));
push(P("OxIOD(train 122/val 15/test 15)와 TLIO golden으로 학습한 모델을 대상으로, 비겹침 앵커 위치의 2차원 RMSE로 측정한다."));
push(EQ("RMSE_XY = √( (1/N) Σ ||p̂_xy − p_xy,GT||² )","13"));
push(P("EKF는 카테고리별 meascov_scale을 [0.001,…,10.0]에서 그리드 서치한다. 이는 평가 시퀀스 오차를 직접 최소화하는 사후(oracle) 최적화로 EKF에 유리한 관대한 상한이며, 그럼에도 트롤리 외 전 카테고리에서 네트워크 단독에 열등하다. 평가는 카테고리당 최장 시퀀스 1개로 수행한다."));

push(SH("4.2 변위 회귀 성능"));
push(P("약 460K 동급에서 ResMLP128이 ResNet1D-Small보다 Test RMSE 0.021 m 우수하고, 10배 큰 Full과의 차이도 0.014 m에 불과하다(표 3, 그림 2)."));
push(CAP("표 3. 변위 회귀 백본의 파라미터 수 대비 Test RMSE."));
push(makeTable([
  ["모델","파라미터","Val RMSE(m)","Test RMSE(m)"],
  ["ResMLP128","459,870","0.1694","0.1381"],
  ["ResNet1D-Small","460,490","0.1875","0.1588"],
  ["ResNet1D-Full","5,031,430","0.1584","0.1239"],
], [3400,2624,2374,2374]));
push(imgPara("fig2_backbone.png", 500));
push(CAP("그림 2. Handbag 세 모델 예측 경로 비교. ResMLP128 1.250 m로 최저 오차."));

push(SH("4.3 휴대 상태 분류 성능"));
push(P("독립 분류기 정확도 98.68%, Handbag–Pocket 오분류 약 0.3%(표 4)."));
push(CAP("표 4. 휴대 상태 분류 성능(독립 분류기)."));
push(makeTable([
  ["클래스","Prec.","Rec.","F1","Support"],
  ["Trolley","1.000","0.999","0.999","895"],
  ["Handbag","0.999","0.993","0.996","2,289"],
  ["Handheld","0.941","0.993","0.966","752"],
  ["Pocket","0.975","0.992","0.984","661"],
  ["Running","1.000","0.970","0.985","1,382"],
  ["Slow Walking","0.976","0.979","0.978","1,223"],
  ["Weighted Avg","0.987","0.987","0.987","7,202"],
], [3172,1900,1900,1900,1900]));
push(P("",{after:80}));

push(SH("4.4 EKF 결합 대 네트워크 단독"));
push(P("트롤리 외 전 카테고리에서 EKF가 열등하며, 격한 동작에서 3~5배, 장거리에서 8배 이상 악화된다(표 5, 그림 3–5)."));
push(CAP("표 5. 카테고리별 EKF 결합 대 네트워크 단독 RMSE_XY 비교."));
push(makeTable([
  ["Category","Best Scale","Net-only(m)","EKF(m)","결과"],
  ["trolley","0.001","1.9764","1.6424","EKF 우위 (+17%)"],
  ["handbag","0.001","1.2018","3.0852","EKF 열등 (2.6×)"],
  ["handheld","0.001","1.5706","7.7783","EKF 열등 (5.0×)"],
  ["pocket","0.001","1.3682","4.8744","EKF 열등 (3.6×)"],
  ["running","0.01","1.3139","4.6835","EKF 열등 (3.6×)"],
  ["slow_walking","1.0","1.3399","3.8677","EKF 열등 (2.9×)"],
  ["large_scale","0.1","2.3559","19.3046","EKF 열등 (8.2×)"],
], [2300,1900,2200,1900,2472]));
push(imgPara("fig3_handbag.png", 440));
push(imgPara("fig4_handheld.png", 440));
push(imgPara("fig5_trolley.png", 440));
push(CAP("그림 3–5. 대표 시퀀스 30초 GT·Net·EKF 궤적(Handbag 0.361→2.418 m, Handheld 0.230→2.802 m, Trolley 0.222→1.341 m)."));

push(SH("4.5 근본 원인 진단: 정답 변위 대체"));
push(P("정답 변위를 직접 대입하면 트롤리가 0.37 m로 회복되어 병목이 입력 측정값 품질임을 입증한다(표 6). running은 큰 yaw 드리프트로 GT 변위마저 잘못된 프레임으로 변환되어 악화된다."));
push(CAP("표 6. 정답 변위 대체 실험(RMSE_XY, m)."));
push(makeTable([
  ["Category","Net","EKF best","GT-Meas EKF"],
  ["trolley","1.98","1.64","0.37"],
  ["handheld","1.57","7.78","4.22"],
  ["pocket","1.37","4.87","2.42"],
  ["running","1.31","4.68","6.75"],
  ["large_scale","2.36","19.30","11.64"],
], [3172,2500,2500,2600]));
push(P("",{after:80}));

push(SH("4.6 전처리 정렬 ablation 및 yaw 드리프트 민감도"));
push(P("baseline 윈도우 오차 0.148 m는 Test RMSE 0.138 m와 정합한다. (a) 정렬 ablation: 회전 정렬 생략(body)은 ATE를 2.44배 악화시키나 시간적 입도 차이는 1.08배로 작다(표 7, 그림 6)."));
push(CAP("표 7. 입력 전처리 정렬 ablation(OxIOD 152 시퀀스 pooled)."));
push(makeTable([
  ["비교","기하평균(95% CI)","OFF 악화","효과"],
  ["body vs ga","2.44× (2.08–2.86)","124/152","대"],
  ["yaw vs ga","1.08× (1.02–1.13)","94/152","소(유의)"],
], [2900,3600,2272,2000]));
push(imgPara("fig6_align.png", 500));
push(CAP("그림 6. 정렬 ablation 궤적. body 발산(54.3 m), ga(2.44)·yaw(2.54) GT 근접."));
push(P("(b) yaw 드리프트 민감도: 예측 저하를 크기·방향 오차로 분해하면 크기는 보존(1.00배)되나 방향만 7.7배 붕괴한다 — 진정한 OOD다(표 8, 그림 7). 누적 yaw 약 10°만으로 1.5배에 도달한다(표 9, 그림 8). 이 해리는 전체 152 시퀀스와 ResNet1D 백본(460K·5.0M)에서도 방향 7.2~7.7배 붕괴·크기 보존으로 재현되어 백본·시퀀스 선택에 무관한 일반 속성이다."));
push(CAP("표 8. yaw 드리프트의 방향/크기 분해(input_only)."));
push(makeTable([
  ["드리프트(°/s)","방향오차(m)","크기오차(m)","출력경로(m)"],
  ["0.0","0.148","0.096","0.148"],
  ["0.5","1.155","0.095","0.148"],
  ["5.0","1.140","0.096","0.148"],
  ["0→5 배수","7.70×","1.00×","1.00×"],
], [2972,2600,2600,2600]));
push(CAP("표 9. 누적 yaw에 따른 윈도우 예측오차(onset)."));
push(makeTable([
  ["드리프트(°/s)","누적 yaw(°)","윈도우오차(m)","baseline 대비"],
  ["0.00","0","0.148","1.00×"],
  ["0.02","11","0.220","1.49×"],
  ["0.05","28","0.339","2.29×"],
  ["0.20","110","0.586","3.96×"],
], [2972,2600,2600,2600]));
push(imgPara("fig7_sensitivity.png", 500));
push(CAP("그림 7. 합성 yaw 드리프트 민감도(8 카테고리 평균). 방향 7.70× 붕괴, 크기 1.00× 보존."));
push(imgPara("fig8_drifttraj.png", 330));
push(CAP("그림 8. 입력 yaw 드리프트 궤적(large_scale_14). 누적 12°·35°로 GT에서 이탈."));
push(P("(c) 절대 yaw 기준(RotVec): SC-EKF 추정 yaw는 말기 약 135°(최대 222°)까지 불규칙 드리프트하여 ~10° 예산을 초과, ATE가 RotVec-DR 대비 5.0배 악화된다(EKF 7.78 m 대 1.57 m). 절대 yaw를 공급하면 Net-only 수준을 회복한다(그림 9)."));
push(imgPara("fig9_ekfvsrotvec.png", 500));
push(CAP("그림 9. 실제 EKF yaw 드리프트 vs RotVec 절대 yaw(handheld_1)."));

push(SH("4.7 실측 Android 도메인 정성적 재현(case study)"));
push(P("배포 단말(Galaxy S23 FE) 실내 보행 3개 시퀀스 사례 검토다(정량 본체는 OxIOD 152 시퀀스). 모델 in-domain 스케일은 정확하며(ATE 경로 길이의 약 1%), 정렬 재현(ga≈window-start ≤0.95 m, body 발산 1.7–15.3 m)과 합성 드리프트 재현(크기 불변, 방향 누적 yaw 비례)이 OxIOD와 일관된다. 본 단말 자이로-only heading은 20–33초에 72–113° 드리프트하여 RotVec 채택을 실측으로 정당화한다."));

// ---- 5. 논의 ----
push(H(5,"논의"));
push(P("정답 변위 대입(트롤리 0.37 m)은 EKF 구조가 정상임을, 병목이 입력 측정값 품질임을 입증한다. 학습 시 입력은 정답 yaw 프레임이나 EKF 운용 시 추정 yaw로 정규화되므로, yaw가 정답에서 벗어나면 입력이 OOD가 되어 예측이 저하된다. 입력·출력 경로 분리에서 방향만 7.7배 붕괴하고 크기는 보존된다는 사실은 이것이 좌표 변환 잔차가 아닌 진정한 OOD임을 보인다. 따라서 핵심은 필터 파라미터가 아니라 입력 프레임 정렬의 정확성이다."));
push(P("한계: (i) 측위 평가가 카테고리당 최장 시퀀스 1개로 수행되어 선택 편향이 있다. (ii) 정렬 ablation 일차 결과는 선행연구 재확인으로 하니스 검증·대조군이다. (iii) 실측 Android 재현은 N=3·단일 기기·단일 자세·상대(RotVec) 기준 소규모 사례 검토로 절대 위치 GT 검증은 향후 과제다. 누적 약 10° 임계는 등속 합성 드리프트·본 모델 기준 경험값이다. 또한 EKF 열등은 (a) 동역학 오버헤드와 (b) yaw 드리프트 OOD로 분해되며, 느린 보행은 (a)로, 고회전은 (b)로 지배된다."));

// ---- 6. 결론 ----
push(H(6,"결론"));
push(P("본 논문은 적응형 EKF를 설계·평가하였으나 카테고리별 공분산 최적화에도 트롤리 외 전 휴대 상태에서 EKF가 네트워크 단독보다 열등함을 확인하였다. 정답 변위 대입 진단은 병목이 입력 측정값 품질, 근본 원인이 절대 yaw 부재의 누적 드리프트에 의한 입력 OOD임을 입증하였다. 전 시퀀스 ablation으로 중력 정렬의 존재(생략 시 2.44배 악화)와 입력 yaw 안정성이 품질을 지배하며 yaw 드리프트가 변위 방향 예측만 7.7배 선택적으로 붕괴시킴을 정량 확증하였다. 이로써 핵심이 측정 공분산 조정이 아니라 입력 전처리의 정확성임을 규명하고, 누적 yaw 약 10° 이내 유지를 정량 목표로 하는 절대 yaw 융합 전처리를 우선적 해결 방향으로 제시한다."));

// ---- 참고문헌 ----
push(H("","참고 문헌"));
const refs = [
"[1] W. Liu et al., “TLIO: Tight Learned Inertial Odometry,” IEEE RA-L, vol. 5, no. 4, pp. 5653–5660, 2020.",
"[2] Y. Wang et al., “LLIO: Lightweight Learned Inertial Odometer,” IEEE IoT-J, vol. 10, no. 3, pp. 2508–2518, 2023.",
"[3] S. Herath, H. Yan, and Y. Furukawa, “RoNIN: Robust Neural Inertial Navigation in the Wild,” in Proc. IEEE ICRA, pp. 3146–3152, 2020.",
"[4] C. Chen et al., “OxIOD: The Dataset for Deep Inertial Odometry,” arXiv:1809.07491, 2018.",
"[5] C. Chen et al., “IONet: Learning to Cure the Curse of Drift in Inertial Odometry,” in Proc. AAAI, vol. 32, no. 1, pp. 6468–6476, 2018.",
"[6] H. Touvron et al., “ResMLP: Feedforward Networks for Image Classification,” IEEE TPAMI, vol. 45, no. 4, pp. 5314–5321, 2023.",
"[7] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. ICLR, 2019.",
"[8] R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” J. Basic Eng., vol. 82, no. 1, pp. 35–45, 1960.",
"[9] S. I. Roumeliotis and J. W. Burdick, “Stochastic Cloning,” in Proc. IEEE ICRA, pp. 1788–1795, 2002.",
"[10] R. Harle, “A Survey of Indoor Inertial Positioning Systems for Pedestrians,” IEEE Commun. Surveys Tuts., vol. 15, no. 3, pp. 1281–1293, 2013.",
"[11] A. H. Mohamed and K. P. Schwarz, “Adaptive Kalman Filtering for INS/GPS,” J. Geodesy, vol. 73, no. 4, pp. 193–203, 1999.",
"[12] P. Bahl and V. N. Padmanabhan, “RADAR: An In-Building RF-Based User Location and Tracking System,” in Proc. IEEE INFOCOM, vol. 2, pp. 775–784, 2000.",
];
refs.forEach(r => push(new Paragraph({ spacing:{after:40,line:240},
  indent:{left:360,hanging:360}, children:[ T(r,{size:18}) ] })));

// ================= 문서 빌드 =================
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: { page: {
      size: { width: 11906, height: 16838 },
      margin: { top: 1701, bottom: 1134, left: 567, right: 567 }, // 30/20/10/10 mm
    } },
    children: body,
  }],
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(DIR, "KIISE_paper.docx"), buf);
  console.log("[OK] KIISE_paper.docx written");
});
