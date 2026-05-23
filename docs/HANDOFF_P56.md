# HANDOFF — P56 (2026-05-22)

> 본 문서는 **P40 이후 ~ 현재(P56)** 까지의 변경 기록·진행 상황·폐기된 시도·
> 그리고 그에 따른 *현재 어플리케이션 결정사항* 을 정리한다. 현재 앱 구조와
> 사용법은 `README.txt` 본문을, P40 이전 상세는 `docs/HANDOFF_P39.md`,
> `docs/HANDOFF_P40.md`, `docs/HANDOFF_P44.md`, `docs/HANDOFF_P46.md` 를 참조.

## 0. 한 줄 결론

본 단말(Android) 데모는 **경로 B (RotVec DR + PDR-hybrid + 클래스별 속도
스케일 소프트 스위칭)** 로 운영한다. 논문의 Context-Aware Adaptive EKF
트랙은 OxIOD 환경에서 유효하며, Android 단말 도메인 갭으로 인한 경로 A
발산은 *센서·기기 도메인 한계* 로 진단·문서화한다.

시연 자세는 **HANDHELD only** 로 한정한다 (자세한 근거는 §6 참고).

---

## 1. P40 이후 핵심 진단 기록

### P40 — 크래시 해결
- SIGBUS → ResNet1DSmall 교체 + asset 캐시 버그 + AGP `noCompress` 수정
- 자세한 기록: `docs/HANDOFF_P40.md`, `memory/project_crash_fix_p40.md`

### P41 ~ P52 — EKF 단발 튜닝 (대부분 실패)
- USE_DEAD_RECKONING_BYPASS, USE_OOD_FIX, USE_LPF, R_SCALE 그리드,
  JUMP-GATE, DISP CAP 등 *증상 억제형* 단말 튜닝 반복.
- 매번 "단말에서 상수 하나 바꿔 replay+logcat 읽기" 방식 — *모델 오차와
  EKF 오차가 섞여 측정 불가능* 했음.
- 결론: 모델측 / EKF측 어디서 발산이 시작되는지 분리 측정이 필요했다.

### P53 ~ P56 — 오프라인 하니스로 분리 진단 후 경로 변경
- 진단 결과(아래 §2)에 따라 EKF 우회 경로(B)를 단말에 구현하고,
  나머지 한계는 데모 스코핑으로 정직하게 문서화하기로 결정.

---

## 2. 오프라인 하니스 진단 결과 (2026-05-22, 결정적 전환점)

도구: `src/Network/offline_eval.py` (EKF 배제, 모델 + 학습 동일 전처리만)

| 평가축 | 결과 |
| --- | --- |
| OxIOD RMSE | 0.89 m → 모델·norm·전처리 검증 통과 |
| Android `latest.csv` 단위/프레임 4 조합 | window 당 \|disp_xy\| 0.3~0.55 m |
| 5 m 왕복 17 초 보행 기대값 | ~0.6 m/s × 1 s = 0.6 m/window |
| g vs m/s² 단위 가설 | mean 0.440 vs 0.425 — 거의 동일, 가설 기각 |
| Dead-Reckoning (모델 출력만 적분) | 경로 8~9 m (실제 10 m), 종점 폐합 ~1 m |

### 핵심 발견 1 — "8~13배 과대추정" 은 모델이 아니다
- 모델 출력 자체는 보행 에너지를 정확히 잡고 있었다.
- P51 disp cap 은 *존재하지 않는 문제* 를 막고 있었음.

### 핵심 발견 2 — 발산원은 EKF 내부 (yaw drift + clone state outlier)
- Network-only 적분이 ~10% 오차로 *그냥 동작* 한다는 게 결정적 증거.
- Notion 2026-05-06 메모("EKF 가 Network-only 보다 열등, yaw drift 원인")
  와 정확히 일치.

### 핵심 발견 3 — Android 에서 모델 *방향* 이 OoD
- 후속 `latest.csv` 분석(`src/View/analyze_latest_csv.py`):
  - raw 데이터는 정상 (180° 회전이 rotVec 누적 yaw -178.5° 로 명확).
  - 그러나 모델 world disp 가 보행 중간에 *방향 역전*.
  - 구간별 순변위 walk1 = 0.83 m / walk2 = 2.22 m (각 5 m 여야 함).
  - 폐합 1.8~2.7 m ≈ 0.47×√15 (랜덤워크와 통계적으로 일치).
- OxIOD 는 동일 코드로 349 m 시퀀스 RMSE 0.89 m → 방향 정상.
- **확정 진단**: 모델은 "보행 *에너지(크기)* 는 잡지만 *이동 방향* 을
  Android 도메인 갭으로 잃는다." EKF / 프레임 / DR 튜닝으로 해결 불가.

---

## 3. P53 — RotVec Dead-Reckoning (EKF 우회)

커밋: `23d3b30` (P53 + P54 + offline_eval.py)

- `ImuCollector.kt`
  - `ImuSample.rotMat: FloatArray` (9-element 회전행렬, per sample)
  - `latestRotMat` (volatile, identity init)
  - `getRawWindow()` : LPF 미적용 6채널 (하니스 입력 분포와 정합)
  - `getRotMatWindow()` : per-sample 회전행렬
- `LocalizationViewModel.kt`
  - 토글 `USE_ROTVEC_DR = true` (기본)
  - `runRotVecDrStep()` 진입 시 EKF 클론/update 완전 우회
  - `transformWindowRotVec()` : per-timestep rotVec gravity-aligned 변환
    (학습 `_window_to_gravity_aligned` 와 동일)
  - 1 초 비겹침 윈도우마다 모델 disp 를 RotVec 시작 yaw 로 월드 회전 후
    `netPos` 에 누적.

빌드 후 Replay 결과:
- EKF 발산은 완전히 사라짐 (경로 ~9 m, 폐합 2.74 m).
- 하지만 (위 §2) "방향" 문제는 RotVec DR 만으로는 해결되지 않음.

---

## 4. P54 — PDR-hybrid (모델 크기 + rotVec heading)

커밋: `23d3b30` (P53 와 동일 커밋에 포함)

- 토글 `USE_PDR_HEADING = true`
- 모델 disp 에서 `|disp_xy|` 만 취하고, 진행 방향은 rotVec heading 사용.
  - `dWorld = |disp_xy| × (cos heading, sin heading)`
  - heading = 윈도우 중앙 rotVec yaw, 전진 보행 가정.
- 윈도우 내 heading 변화 > 60° 인 *제자리 회전* 윈도우는 누적에서 제외.
- 적합 시나리오: **handheld** (폰을 진행 방향으로 들고 보행).
- 효과: walk1·walk2 가 일관된 직선으로 복원. 180° 회전 → 왕복 형태 회복.

한계:
- 크기는 여전히 Android 에서 ~30% 과소 → 별도 스케일 보정 필요(P56).
- pocket/bag 자세에는 부적합 (heading ≠ 진행 방향).

---

## 5. P55 — 20Hz 속도 적분 (추적 연속성 복원)

커밋: `fc665dc`

P54 의 turn-skip(제자리 회전 윈도우 통째 폐기) 가 *제자리 회전 중 앱이
완전히 멈추는* 현상을 유발했음 (사용자 보고). 1 Hz 비겹침 누적도
체감상 끊김의 원인이었다.

수정:
- 모델 출력(1 초 윈도우 변위) → *속도*(disp/1 s) 로 환산.
- 매 추론 틱(20 Hz) 마다 `dt` 만큼 적분 (겹침 윈도우를 속도로 다루므로
  과적분 없음 — 20 틱 × disp/20 ≈ 실제 1 초 변위).
- 회전 윈도우는 폐기 대신 **속도 감쇠** (× `TURN_SPEED_ATTEN = 0.3`).
- 정지 윈도우는 속도 0 (위치 고정, UI 는 매 틱 갱신).
- 새 파라미터:
  - `TURN_YAW_THRESH_DEG = 60.0`
  - `TURN_SPEED_ATTEN = 0.3`
  - `DR_VEL_EMA = 0.25`
  - `DR_TRACKPOINT_MIN_MOVE = 0.1 m`

효과: 회전·정지 어떤 상황에서도 끊김 없는 연속 추적 회복.

---

## 6. P56 — 클래스별 속도 스케일 소프트 스위칭

커밋: `db6db2f`

논문의 Context-Aware soft-switching 형식 `Σ p_k · θ^(k)` 를 DR speed
scale 에 적용:
```
effectiveSpeed = modelSpeed × Σ_k clsProb_k · SPEED_SCALE_PER_CLASS_k
```
경로 B 가 분류 헤드를 *기능적으로* 사용하게 됨 (이전엔 UI 표시뿐).

현재 값 (균일 — *구조만* 들어간 상태):
```kotlin
SPEED_SCALE_PER_CLASS = doubleArrayOf(
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5
)
// 인덱스: 0 handbag 1 handheld 2 pocket 3 running 4 slow_walk 5 trolley 6 unknown
```

한계:
- 휴대모드별 과소비율의 실측 보정 데이터가 없어 *현재 균일 ~1.5×*.
- Android 에서 분류기 자체가 OoD (대부분 unknown) → 시연 시 unknown
  값이 실질적으로 지배.
- per-class 차등은 휴대모드별 측정 데이터 확보 후 산출 예정.

또한 P56 에서 `TrackView` 범례를 단일화:
- 이전: `모델+EKF`, `모델 only` (오해 소지)
- 현재: `측위 궤적`, `시작점` (경로 B 는 단일 궤적)

P56 실측 결과 (latest.csv 5 m 왕복):
- legs (구간별 순변위) 2.3 / 3.2 m → 3.5 / 4.8 m (walk2 ≈ 5 m 회복)
- 폐합: 0.91 m → 1.36 m (asymmetry 증폭 — 알려진 한계)

---

## 7. RoNIN fine-tuning 트랙 — 종결 (미배포)

별도 폴더: `D:\mobile\ronin_finetune\` (메모리 노드:
`memory/project_ronin_finetune.md`)

### 시도
- 모델측 도메인 갭 해소 목적으로 ResMLP 회귀 모델을 **RoNIN
  데이터셋**(Android 폰 + Tango GT) 으로 fine-tuning.
- 분리 구조: 회귀(out_regression)는 RoNIN 으로 적응, 분류기는 OxIOD
  학습본 유지 (RoNIN 에는 7-class 휴대모드 라벨 없음).
- RoNIN train 1+2 추출 → 85 시퀀스 (~12.8 h), CPU 25 epoch 학습.

### 결과
- RoNIN 도메인 내: per-window err 0.98 → 0.52 m, 크기비 0.59 → 0.77.
  *RoNIN 자체에선 명확히 개선됨.*
- 사용자 Samsung 폰 실측: mean |disp| 0.48 → **0.28 m** 로 *더* 과소.
  - latest.csv, clean_walk 모두 동일 경향.
  - **RoNIN(Asus 등 Android) → Samsung 전이 실패.**

### 결론
- 학습 IO 의 device 도메인 갭은 *해당 기기 자체 데이터* 없이는 못
  닫는다는 점이 실증됨.
- 공개 Android 데이터셋으로도 *해당 기기* 전이는 보장되지 않는다.
- **결정**: RoNIN 모델 배포 안 함. RoNIN 트랙 종결.
- 현재 앱은 OxIOD 학습 out_classifier2 를 유지하고, P56 의 균일 1.5×
  전역 스케일로 평균 보정한다 (Samsung 에서 RoNIN 모델보다 덜 나쁨).

---

## 8. 데모 시연 자세 결정: HANDHELD only

위 진단·종결을 종합한 결정:

1. **분류기 자체가 Android 에서 OoD** (대부분 unknown) → handheld 외
   휴대모드 인식 신뢰도 보장 불가.
2. **rotVec heading = 진행 방향** 가정은 handheld 자세에서만 유효
   (pocket/bag/trolley 부적합).
3. **단일 사용 자세로 한정** 해야 향후 per-class 스케일 보정 데이터
   수집 시에도 측정이 신뢰 가능.
4. 논문 thesis 와 모순되지 않음: "Context-Aware Adaptive EKF" 의
   *Context-Aware* 측면(분류 기반 스위칭)은 OxIOD 환경에서 검증된 것이며,
   단말 데모에서는 device-domain 한계를 인정하고 demo scoping 으로 처리.

→ `README.txt` 본문(데모 시연 자세 / §8.1) 에 명시.

---

## 8.5 TLIO 논문 EKF 계수 오프라인 비교 (2026-05-23 추가)

`src/Network/compare_tlio_ekf.py` + `imu_ekf_py.py` 신설.

- 목적: 단말 EKF (imu_ekf.cpp) 식은 그대로 두고 **계수만 TLIO 논문 값** 으로
  바꾼 변형을 같은 IMU+모델 시퀀스에 입력해 궤적 차이를 격리 측정.
- 두 cfg 가 다른 점: `init_vel_sigma 1.0→0.1`, `init_ba_sigma 0.02→0.2`,
  `meascov_scale 1.0→10.0` (TLIO §V-D 끝 — temporal correlation 보정).
- latest.csv 시연 결과: TLIO cfg 가 χ² 게이트 통과율 6.5× 증가
  (2→13 updates), 종점 폐합 1.49 → **1.00 m** 개선. EKF 식·gate·χ² 임계는
  완전히 동일하므로 *계수 효과만* 격리됨.
- 단말 코드 변경 0. 본 도구는 사용자 측 `--android latest.csv` 한 줄로 동작.

## 9. 현재 커밋 상태

```
(P59) compare_tlio_ekf.py + imu_ekf_py.py — TLIO 논문 EKF 계수 오프라인 비교 도구
(P58) carryMode 표시를 메인 UI → IMU 진단 화면으로 이동 (sharedInstance 패턴)
a63dcbc P57  HANDHELD-only 정합화 (soft-switching 제거, 분류기 표시 전용)
4857c93 docs  README 슬림화 + HANDOFF_P56.md 분리
9cda565 docs(README): P56 최신 상태 헤더 추가 (※ 4857c93 으로 대체됨)
b9a137f 정리: out_resnet*/ 학습 산출물 제거
db6db2f P56  소프트 스위칭 + 범례 단일화
fc665dc P55  20Hz 속도 적분 (추적 연속성 복원)
23d3b30 P53 + P54  RotVec DR + PDR-hybrid + 오프라인 진단 하니스
1adfa45 P50 + P51 + P52  C++ 게이트 진단 로그 + disp cap + JUMP-GATE
…(P40 이전 상세는 docs/HANDOFF_P39.md)
```

브랜치: `android` (origin 동기화).

---

## 10. 다음 단계 후보

우선순위 순:

1. **(권장)** 코드를 HANDHELD-only 가정에 맞춰 단순화 (P57 후보)
   - 분류기 의존 코드(soft-switching 구조) 제거하고 단일 handheld
     스케일로 운영하거나, 분류기를 *표시 전용* 으로 명시.
   - UI 에 "HANDHELD 자세로 보행해 주세요" 안내 추가.
2. 휴대모드별 실측 보정 데이터 수집 → per-class 스케일 산출
   (현재 균일 1.5× → 실제 측정 기반 값).
3. 경로 A (논문 EKF) 의 단말 발산 재현 + 분석 노트 정리
   (별도 트랙으로 보존, 본 데모와는 분리).
4. 보행 패턴 다양화 테스트 (8 자, 사각형, 장거리) — handheld 한정.

---

## 11. 관련 메모리 / 외부 문서

- `memory/MEMORY.md` — Memory Index
- `memory/project_crash_fix_p40.md` — P40 크래시 해결 기록
- `memory/project_p45_session.md` — P45 Replay 인프라
- `memory/project_offline_harness.md` — 오프라인 하니스 방법론 + 진단 결과
- `memory/project_ronin_finetune.md` — RoNIN 트랙 (종결)
- Notion: "Context-Aware Adaptive EKF for IMU-based Indoor Localization"
  - 2026-05-06  EKF vs Network-only 비교 결론
  - 2026-05-07  분리 구조(분류기/회귀기) 권고
  - 2026-05-11  ekf_tune 그리드서치 (trolley 외 Network-only 우세)
