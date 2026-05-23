# PATH_B 채택 근거 — 단말 데모 경로 결정 문서

> 2026-05-23 작성. 참조: `docs/HANDOFF_P56.md` (변경 이력), `README.txt`.
> 본 문서는 *왜* PATH_B 를 단말 데모의 기본 경로로 채택했는지, 그 한계와
> 학술적 위치를 정리한다. 코드 변경 이력은 HANDOFF_P56.md, 사용법은
> README.txt 를 참조한다.

---

## 0. 한 줄 요약

논문 thesis (Context-Aware Adaptive EKF) 의 **EKF 부분이 본 단말 환경에서
발산** 한다는 사실이 진단 도구(P53 오프라인 하니스)로 정량 확정됨. **단말
도메인 갭(OxIOD/iPhone 학습 → Samsung Android 배포)** 이 모델 출력 방향
채널을 OoD 로 만들어 EKF measurement 가 발산원이 되기 때문. 단말 데모를
의미 있게 동작시키는 유일한 경로로 **PDR-hybrid (= RoNIN/TLIO 3D-RONIN
baseline 과 동등)** 인 PATH_B 를 채택했다. 학술 thesis 자체는 **OxIOD
Python 검증 트랙(트랙 A)** 에 별도로 보존된다.

---

## 1. 프로젝트의 두 트랙

본 프로젝트는 다음 두 트랙으로 분리해 보아야 한다.

| 트랙 | 환경 | 알고리즘 | 결과 |
|---|---|---|---|
| **A. 학술 검증** | OxIOD (iPhone Core Motion) | Context-Aware Adaptive **EKF** + multi-head ResMLP128 | thesis 가 의도한 우수성 — Python 시뮬레이션 (`src/View/visualize_comparison.py` + `src/tracker/scekf.py`) 으로 검증 |
| **B. 단말 데모** | Android Samsung | **PATH_B** (RoNIN/3D-RONIN baseline 과 동등) | EKF 우회. 단말 도메인 갭으로 인한 한계 인정 |

두 트랙은 **같은 학습 모델(`out_classifier2`)을 공유** 하지만, 모델 출력의 어느 채널을 어떻게 사용하는지가 다르다.

| 모델 출력 채널 | 트랙 A (OxIOD EKF) | 트랙 B (PATH_B) |
|---|---|---|
| `disp_xy` 방향+크기 | EKF measurement 로 사용 | **크기만** 사용, 방향은 rotVec heading 으로 치환 |
| `disp_z` | EKF measurement | 그대로 적분 |
| `dispLogVar` (공분산) | EKF R 산정에 사용 | 미사용 |
| `clsProb` (휴대모드 7-class) | EKF Q/R soft-switching | 표시 전용 (P57 부터 위치 계산 미사용) |

본 문서는 트랙 B = PATH_B 의 결정 근거를 다룬다. 트랙 A 의 결과는 별도로 정리해 보고한다.

---

## 2. 단말 EKF 발산 진단 — 왜 PATH_B 가 필요했나

### 2.1 P40~P52: 단말 상수 튜닝의 누적 실패

P40 부터 P52 까지 단말 EKF 의 상수(`R_SCALE`, `meascov_scale`, `JUMP_GATE`, `disp cap`, `USE_OOD_FIX`, `USE_LPF` 등) 를 *한 번에 하나씩* 바꿔 replay + logcat 으로 결과 관찰하는 방식이 반복됐다. 한계:

- 모델 오차와 EKF 오차가 *섞여서* 측정 불가능
- 어느 상수 변경이 어떤 효과인지 격리 안 됨
- 결과 메트릭이 합산이라 인과 추적 불가

→ 진단 방법론 자체 변경 필요.

### 2.2 P53: 오프라인 하니스 도입 → EKF 가 발산원 확정

`src/Network/offline_eval.py`: EKF 를 완전히 배제하고 *모델 + 학습 동일 전처리* 만 PC 에서 재현. 결과 (`memory/project_offline_harness.md`):

| 평가 | 결과 |
|---|---|
| OxIOD RMSE | 0.89 m → 모델·norm·전처리 정상 |
| Android `latest.csv` window 당 \|disp_xy\| | 0.3~0.55 m (5 m 왕복 1초 기대 ~0.6 m 와 정합) |
| 모델만 적분 (dead-reckoning) | 경로 8~9 m / 실제 10 m, 종점 폐합 ~1 m |
| g vs m/s² 단위 가설 | 기각 (mean 0.440 vs 0.425, 거의 동일) |

**확정**: P51 의 "8~13배 과대추정" 전제는 *모델이 아니라 EKF 내부 발산* 이었음. 모델 자체는 OxIOD/Android 모두 정상 동작. EKF 가 측정값을 잘못 처리해 발산.

### 2.3 P53 라이브 진단: 모델 *방향* 채널이 Android 에서 OoD

`src/View/analyze_latest_csv.py` 정밀 분석:
- raw 데이터 정상 (180° 회전이 rotVec 누적 yaw −178.5° 로 명확히 잡힘)
- 모델 window 별 \|disp_xy\| 정상
- **그러나 모델 world disp 가 보행 중간에 방향 역전**
- 구간별 순변위 walk1 = 0.83 m / walk2 = 2.22 m (각 5 m 여야 함)
- 폐합 1.8~2.7 m ≈ 0.47×√15 (랜덤워크와 통계적으로 일치)

→ 모델은 *보행 에너지(크기)* 는 잡지만 *이동 방향* 을 Android 도메인 갭으로 잃는다. EKF measurement 의 *방향* 이 OoD 라 EKF 가 잘못된 방향을 흡수 → 발산.

이는 EKF cfg 튜닝(meascov_scale, R_SCALE 등) 으로 해결 불가. **모델 출력에 *없는* 방향 정보 문제**.

### 2.4 RoNIN fine-tuning 시도 → Samsung 전이 실패

`memory/project_ronin_finetune.md`:

- RoNIN 데이터셋(Android 폰 + Tango GT) 으로 회귀 모델 fine-tuning
- RoNIN 도메인 내: per-window err 0.98 → 0.52 m, 크기비 0.59 → 0.77 *개선*
- **사용자 Samsung 폰 실측: |disp| 0.48 → 0.28 m, *더 과소*** — 전이 실패
- 결론: device-domain 갭은 *해당 기기 데이터* 없이는 닫지 못함. RoNIN(Asus 등) → Samsung 전이도 보장 안 됨

→ 모델측 해결책 (학습 데이터 변경/추가) 으로 단말 EKF 를 살리는 길도 닫힘.

---

## 3. PATH_B 의 구성 (P53/P54/P55/P56/P57 누적 결정)

| 단계 | 변경 | 동기 |
|---|---|---|
| P53 | `runRotVecDrStep()` — EKF 클론/update 우회, 모델 disp 만 적분 | EKF 발산 회피 |
| P54 | PDR-hybrid: \|disp_xy\| + rotVec heading | Android OoD 방향 채널 회피 |
| P55 | 20 Hz 속도 적분 (1 Hz 비겹침 → 매 추론 틱 dt 적분) | 회전·정지 중 화면 멈춤 해소 |
| P56 | 클래스별 속도 스케일 soft-switching (구조만) | thesis 와의 명목 정합 |
| P57 | soft-switching 제거 → 단일 `HANDHELD_SPEED_SCALE = 1.5` | 분류기 자체가 Android OoD — 가짜 구조 정리 |
| P58 | 분류기 표시를 IMU 진단 화면으로 이동 | 메인 UI 정합 |

핵심 알고리즘 (P57 이후 확정):
```
매 추론 틱 (20 Hz):
  1) IMU window 100 sample 추출 + rotVec gravity-aligned 변환
  2) 모델 추론 → (disp_xy, disp_z, log_var, cls_prob)
  3) xyMag      = |disp_xy|
     headingW   = rotVec yaw (자력계 융합 절대 yaw)
     speed      = xyMag / 1s × HANDHELD_SPEED_SCALE(1.5)
  4) 정지 윈도우(dynamicFrac < threshold) → speed = 0
     회전 윈도우(|Δyaw_window| > 60°)    → speed × 0.3
  5) velocity_world = speed × (cos headingW, sin headingW, 0) + (0,0,disp_z/1s)
     v_ema = v_ema + 0.25 × (velocity_world - v_ema)
  6) netPos += v_ema × dt
  7) TrackView 갱신 (UI 는 매 틱 = 끊김 없음)
```

---

## 4. PATH_B 의 한계 (정직)

### 4.1 학습 모델의 일부 채널만 사용
- 모델이 학습한 4 채널(disp 방향, disp 크기, 공분산, 분류) 중 *xy 크기* 1 채널만 사용
- *공분산* 은 PATH_B 가 K=1 효과로 동작하므로 무의미
- *분류* 는 표시 전용 (P57 부터 위치 계산 미사용)
- *방향* 은 OoD 라 명시적으로 폐기 + 자력계로 치환
- ⇒ thesis 의 multi-head 모델 학습 contribution 의 1/4 만 활용

### 4.2 자력계 의존
- heading 정확도 = TYPE_ROTATION_VECTOR 의 자력계 융합 결과
- 자기 환경 교란(철골/대형 가전/스피커) 에 취약
- 단말 시연 공간 선정 시 주의 필요

### 4.3 HANDHELD 자세 한정
- rotVec heading = 진행 방향 가정은 *폰을 진행 방향으로 들고 보행* 할 때만 유효
- pocket/handbag/trolley/running 부적합
- 분류기도 Android OoD (대부분 unknown) → 자세 자동 판별 신뢰 불가
- ⇒ 데모 시연 자세를 HANDHELD only 로 명시(README §8.1)

### 4.4 모델 *크기* 도 여전히 과소
- Android 에서 모델 |disp| ~30~40% 과소 (Samsung 기기, latest.csv 진단)
- P56 시점에선 클래스별 가중치 시도 → 실측 데이터 부족으로 효과 무의미
- P57 에서 균일 1.5× 단일 스칼라로 단순화 — *평균 보정* 만
- walk1/walk2 *비대칭 과소* 는 전역 스케일로 못 고침 (모델 per-window 편차)
- per-class 차등 보정은 휴대모드별 실측 데이터 확보 후 가능

### 4.5 EKF 관측성/필터 이론적 우월성 포기
- SC-EKF 의 yaw 관측성 회피(논문 §V-D) 사용 안 함
- adaptive Q/R soft-switching 사용 안 함
- bias 추정, propagation 모델 사용 안 함
- ⇒ thesis 의 핵심 contribution 3 개 중 (b)(c) 미사용 — *학술 데모로서는 약하다*

---

## 5. PATH_B 채택 후의 장점

### 5.1 단말 데모 동작 회복
- P40~P52 의 EKF 발산으로 trackPoints 가 거의 안 그려지거나 임의 방향 흐름 → P53/P54 이후 5 m 왕복이 직선 형태로 복원
- 종점 폐합 1.0 m 수준 (TLIO 3D-RONIN baseline 결과와 정합)
- 회전·정지 중 끊김 없는 연속 추적 (P55)

### 5.2 학술적 baseline 과 정합
- RoNIN (Yan et al., ICRA 2020) 의 핵심 패턴 = "학습 모델 → velocity 크기, AHRS → heading"
- TLIO 논문(Liu et al., RAL 2020) 의 **3D-RONIN baseline** = 우리 PATH_B 와 동일 구조
  - TLIO 논문 §VI: "3D-RONIN: an estimator where the displacements from the same trained network are concatenated in the direction given by an engineered AHRS attitude filter, resembling what smartphones have."
- ⇒ PATH_B 는 학술적으로 인용 가능한 baseline 으로 정당화 가능

### 5.3 정직한 negative result 로서의 가치
- "OxIOD 학습 모델 → Android 단말 배포" 의 도메인 갭이 EKF measurement 신뢰도를 무너뜨림을 *정량적으로 진단* 함 (offline_eval.py + analyze_latest_csv.py)
- RoNIN fine-tuning 으로도 device-domain 전이 실패 — *해당 기기 데이터* 없이는 못 닫는다는 점 실증
- ⇒ "어떤 모델 학습/EKF 튜닝으로도 해결 못 함" 의 측정 가능한 한계
- 부정적 결과지만 *연구 가치* 있음 — 후속 연구의 데이터 수집 전제 조건을 정량화

### 5.4 코드 정합화 (P57/P58)
- soft-switching 가짜 구조 제거 → 단일 스칼라로 단순
- 분류기 출력은 진단 화면 표시 전용으로 분리 → 메인 UI 가 위치 추적만 담당
- 데모 시연 자세 HANDHELD 명시 → 시연 신뢰도 ↑

---

## 6. 학술적 위치 — RoNIN / 3D-RONIN baseline 과 동등

### 6.1 RoNIN (Yan et al., ICRA 2020)
> Hang Yan, Sachini Herath, Yasutaka Furukawa. "Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations and New Methods." ICRA 2020.

- Android 폰 IMU + AHRS attitude filter
- 학습 모델은 **velocity 의 크기만** 회귀
- heading 은 **Android 자체 AHRS** (자력계 융합) 사용
- 결과: 다양한 보행 시나리오에서 robust

PATH_B = 본 패턴의 단말 구현. 우리는 별도 진단(P53/P54)으로 도달했으나, *방법론적으로 RoNIN 과 동등*.

### 6.2 TLIO 3D-RONIN baseline (Liu et al., RAL 2020)
> Wenxin Liu et al. "TLIO: Tight Learned Inertial Odometry." IEEE Robotics and Automation Letters, 2020.

- 본 논문이 *비교 baseline* 으로 둔 3D-RONIN ≈ PATH_B
- TLIO 논문 §VII.B: TLIO(EKF) > 3D-RONIN(PDR) > NET(network-only integration)
- 단 이 우월성은 *학습 분포가 단말 데이터에 정합되는 환경* 에서 — 우리 단말은 그 정합이 깨진 상태

⇒ 우리 단말 환경에선 TLIO 의 EKF 우월성이 적용 안 됨 → PATH_B(3D-RONIN baseline) 가 합리적 회귀점.

### 6.3 발표/논문에서의 기술 권장
```
"단말 데모는 학습 도메인 갭으로 인해 본 thesis 의 SC-EKF 가 발산하는 것을
정량 진단하였고(offline harness + Android CSV 분석), 단말 측 데모는 학술적
baseline 인 RoNIN/TLIO 의 3D-RONIN 구조와 동등한 PDR-hybrid 경로(PATH_B)로
회귀하였다. thesis 의 SC-EKF + Context-Aware Adaptive 우월성은 학습 도메인
(OxIOD) 의 Python 검증 트랙(트랙 A) 에 별도 보존한다."
```

---

## 7. 향후 개선 경로

우선순위 순:

1. **트랙 A 의 결과 정리 + 시각화 산출 (시급)**
   - `src/View/visualize_comparison.py` 의 OxIOD 결과를 figure 로 저장
   - 트랙 B 의 PATH_B 결과와 한 그림에 비교 → 도메인 갭의 정량 증거
   - 보고서/논문에 *학술적 우월성* 의 직접 증거로 인용

2. **자세 사용 방법 구상 (별도 문서)**
   - Notion 2026-05-11 의 "trolley 만 EKF, 나머지 PDR" 결론을 단말에 적용 가능성 탐색
   - 분류기 OoD 회피를 위한 자세 자동 판별 대안
   - 단계적 구현 로드맵
   - → `docs/POSE_SWITCHING_PLAN.md` 에서 다룸

3. **PATH_B 의 *크기* 보정 정밀화**
   - 현재 균일 1.5× → 휴대모드별 실측 보정 (HANDHELD 외 자세 측정 데이터 확보 시)
   - walk 구간 비대칭 보정은 모델 per-window 편차 개선 필요 (학습 측 작업)

4. **자력계 의존도 완화**
   - 시연 공간 자기 환경 사전 측정 도구
   - rotVec accuracy < MEDIUM 시 UI 경고

5. **device-domain 학습 데이터 수집 (장기)**
   - Samsung 단말 자체 데이터 수집 + fine-tuning
   - 모델 방향 채널의 OoD 해소 시 PATH_B 의 학습 모델 활용 채널 확장 가능

---

## 8. 변경 추적

- 본 문서 작성 시점: 2026-05-23 (P61 fix 완료 시점)
- 관련 커밋:
  - `23d3b30` P53 + P54 (RotVec DR + PDR-hybrid)
  - `fc665dc` P55 (20 Hz 속도 적분)
  - `db6db2f` P56 (소프트 스위칭 + 범례 단일화)
  - `a63dcbc` P57 (HANDHELD-only 단일 스칼라 정합화)
  - `ee3b16a` P58 (분류기 표시를 진단 화면으로)
  - `e643a62` P60 (단말 EKF 모드 토글 + 비교 export — 진단용)
  - `d5fa89b` P61 (EKF measurement 를 PDR-hybrid 로 교체 — 진단용)
- 관련 메모리:
  - `memory/project_offline_harness.md` (오프라인 진단 방법론)
  - `memory/project_ronin_finetune.md` (RoNIN 트랙 종결)
- 관련 외부:
  - Notion 2026-05-06 "EKF vs Network-only 비교"
  - Notion 2026-05-07 "분리 구조 권고"
  - Notion 2026-05-11 "NETWORK_ONLY_STATES 확정" (Python 트랙)
