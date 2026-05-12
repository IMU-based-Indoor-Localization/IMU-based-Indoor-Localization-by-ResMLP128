# IMU-based Indoor Localization — 프로젝트 컨텍스트 인계 문서

이 문서는 새 Claude 채팅에서 이어서 작업할 수 있도록 작성된 단일 자료입니다. 프로젝트 구조 / 학습 모델 / 수정 이력 / 현재 코드 상태 / 다음 단계 / 주의사항이 모두 포함됩니다.

---

## 1. 프로젝트 개요

- **목적**: 스마트폰 IMU(가속도계 + 자이로 + 지자기)와 1D-ResNet(ResMLP) 모델 + EKF 융합으로 실내 측위
- **기반**: TLIO(MIT) + Oxford-IOD 데이터셋, ResMLP 회귀 헤드 + 7-way 휴대 자세 분류 헤드
- **루트 경로**: `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128`
- **주요 폴더**:
  - `src/` — Python 학습/평가 코드 (그대로 보존, 수정 거의 없음)
  - `android/` — 안드로이드 어플리케이션 (수정 중심)
  - `mobile_assets/` — 모바일 배포용 .ptl / norm_params / model_meta
  - `outputs/` — 학습 결과 (norm_mean.npy, norm_std.npy, best.pth)
  - `수정_이력_보고서.docx` — 전체 P9~P22 수정 이력 (현재 18장, 726 단락)

---

## 2. 사용된 학습 모델

| 폴더 | use_classifier | input_len | std (ch0~2) | 용도 |
|---|---|---|---|---|
| `out_classifier2` | **true** | **100** | **[0.118, 0.120, 0.148]** | **어플리케이션 배포 모델 (정답)** |
| `out_classifier_7way` | true | 200 | [0.46, 0.55, 0.53] | standalone 분류기 (사전학습용) |
| `out_tlio_6ch_128` | false | 200 | [5.76, 6.07, 2.97] | 분류기 없는 회귀 전용 (사전학습용) |
| `out_regression` / `out_resnet*` | false | 100/200 | [0.12, ...] | 다른 변형 모델 |

**중요**: `convert_and_prepare.bat` 가 모델은 `out_classifier2` 를 변환하지만 norm_params 는 `android/prepare_assets.py` 가 하드코딩한 `out_tlio_6ch_128` 의 것을 쓰는 버그가 있었음 (P13 에서 수정). 모델/통계 일치 필수.

---

## 3. 수정 이력 (P9 ~ P20)

각 P 번호는 `수정_이력_보고서.docx` 의 7장 이후 절에 대응. 8장 이후가 이번 세션의 핵심.

| P | 장 | 요약 | 결과 |
|---|---|---|---|
| P9~P11 | 7-4~7-5 | freeze_static_state, P11 자기계 게이팅 등 | 부분 개선 |
| **P12** | **8** | `modelTrack` 누적에 t_begin yaw 회전 복원 (`LocalizationViewModel`) | 모델 only 발산 해결 |
| **P13** | **9** | `prepare_assets.py` NORM 경로 `out_tlio_6ch_128` → `out_classifier2` 수정 + norm_*.txt 재생성 | running collapse 해결 |
| **P14** | **10** | EkfBridge R_SCALE 을 Python `STATE_EKF_PARAMS` (0.001~1.0) 와 동기화 | **실패 — 발산 유발** |
| **P15** | **11** | P14 롤백 — R_SCALE [15,10,5,50,5,7,100] / SIGMA_NA 차등 / meascov=10.0 원복 | 안정 회복 |
| **P16** | **12** | `LocalizationViewModel` 본문 ~960 → 95 줄 stub 으로 철거. 엔진 모듈만 보존 | 단방향 재설계 1/4 |
| **P17** | **13** | `AbsoluteSensorNode.kt` 신규 — Stage 1 절대 좌표계 센서 노드 | 단방향 재설계 2/4 |
| **P18** | **14** | `StatelessInferenceNode.kt` 신규 — Stage 2 순수 함수 추론기 | 단방향 재설계 3/4 |
| **P19** | **15** | `RobustEkfTracker.kt` 신규 — Stage 3 자체 미니 EKF + AEKF + 게이팅 | 단방향 재설계 4/4 |
| **P20** | **16** | `LocalizationViewModel` 컨트롤러 재작성 + propagateQueue + RobustEkfTracker 동기화 | **첫 실행 가능 빌드 완성** |
| **P21** | **17** | 실시간 영점 자동 보정 (Auto-Calibration) — `AbsoluteSensorNode` 캘리브레이션 분기 + `performWarmup` + UI 진행도 | 입력 bias 적분 누적 차단 |
| **P22** | **18** | ZUPT (Zero-Velocity Update) — `RobustEkfTracker` 자체 정지 감지(슬라이딩 norm 평균 + hysteresis) + propagate v=0 강제 + update skip | 정지 30초 발산(13/40m → 한 점 근처 유지) 차단 |

---

## 4. 단방향(One-Way) 아키텍처 — 현재 코드 구조

P9~P15 의 양방향 피드백 발산을 구조적으로 차단하기 위해 P16~P20 으로 재설계됨.

```
[Sensor HW] acc, linAcc, gyr, rotVec
     ▼
[Stage 1] AbsoluteSensorNode  (EKF 존재 모름)
  - TYPE_ROTATION_VECTOR 절대 회전으로 acc/gyr 즉시 world frame 변환
  - propagateQueue (Stage 3 용) + ringBuffer (Stage 2 용)
  - 출력: WorldSample { ts_us, worldAcc, worldLinAcc, worldGyr, rotMat, rotAccuracy }
     ▼
[Stage 2] StatelessInferenceNode  (이전 상태·EKF 모름, Pure Function)
  - 윈도우 시작 yaw 추출 → R_yaw_inv 적용 → yaw-free local frame
  - InferenceEngine.infer() 위임 (1D-ResNet PyTorch Mobile Lite)
  - 출력: InferenceOutput { dispLocal[3], dispLogVar[3], classProb[7], windowStartYaw, ... }
     ▼
[Stage 3] RobustEkfTracker  (자기 상태 역류 안 함, synchronized lock)
  - 자체 미니 EKF: p[3], v[3] 상태, R[9] 은 Stage 1 anchor
  - propagate: worldLinAcc 적분, F·Σ·Fᵀ + Q·dt
  - update: dispLocal→dispWorld 복원, Innovation/Mahalanobis gate, Adaptive R
  - 출력: getPosition / getVelocity / getOrientation / getYawRad / getPositionStd
     ▼
[Controller] LocalizationViewModel  (순수 데이터 라우터, 230 줄)
  - propJob(5ms): stage1.drainPropagateQueue → stage3.propagate
  - inferJob(50ms): stage1.getWindowSamples → stage2.infer → stage3.update
  - uiJob(100ms): stage3 출력 → _state (StateFlow<LocalizationState>)
     ▼
MainActivity / TrackView (외부 API 호환 유지: start/stop/reset/state)
```

**핵심**: Stage 3 의 상태는 어떤 경로로도 Stage 1·2 로 전달되지 않음. EKF 자세 추정 오차가 입력 좌표계에 흘러 들어가던 P9~P15 의 발산 매커니즘이 구조적으로 제거됨.

---

## 5. 핵심 파일 인벤토리 (android/app/src/main/)

### 활성 사용 (단방향 파이프라인)

| 파일 | 줄수 | 역할 |
|---|---:|---|
| `java/com/imulocal/AbsoluteSensorNode.kt` | 339 | Stage 1 — 절대 좌표계 센서 노드 |
| `java/com/imulocal/StatelessInferenceNode.kt` | 170 | Stage 2 — 상태 비저장 추론기 (Pure Function) |
| `java/com/imulocal/RobustEkfTracker.kt` | 529 | Stage 3 — 강건한 상태 추정기 (mini EKF + AEKF + ZUPT) |
| `java/com/imulocal/LocalizationViewModel.kt` | 230 | 단방향 컨트롤러 (propJob/inferJob/uiJob) |
| `java/com/imulocal/InferenceEngine.kt` | 146 | PyTorch Mobile Lite 모델 호출부 |
| `java/com/imulocal/MainActivity.kt` | 119 | UI |
| `java/com/imulocal/TrackView.kt` | (변경 없음) | 궤적 그리기 |
| `assets/imu_model.ptl` | — | 학습된 모델 (out_classifier2 변환본) |
| `assets/norm_mean.txt` | 73B | out_classifier2 정규화 평균 (ch0~5) |
| `assets/norm_std.txt` | 70B | out_classifier2 정규화 표준편차 |
| `assets/model_meta.json` | — | 모델 메타 (input_len=100, output_cls=7 등) |

### 보존되었으나 현재 미사용 (Stage 3 가 자체 EKF 사용)

| 파일 | 줄수 | 비고 |
|---|---:|---|
| `java/com/imulocal/EkfBridge.kt` | 246 | C++ SC-EKF JNI 래퍼 — 향후 자세 anchor API 추가 후 활용 가능 |
| `java/com/imulocal/ImuCollector.kt` | 280 | 기존 IMU 수집 모듈 — Stage 1 으로 대체됨 |
| `cpp/EkfJniBridge.cpp` | 455 | JNI 진입점 12개 함수 |
| `cpp/ekf/imu_ekf.{h,cpp}` | 972 | SC-EKF 코어 + Eigen 수학 |

### Python 학습 코드 (변경 없음, 참조용)

| 파일 | 역할 |
|---|---|
| `src/Network/model_twolayer.py` | TwoLayerModel (extractor + reg + classifier) |
| `src/Network/train.py` | Phase 2 학습 진입점 |
| `src/Trans/dataset.py` | TLIONpySingleDataset, `_window_to_gravity_aligned()` |
| `src/Trans/classification_dataset.py` | 분류 라벨링, `LABEL_REMAP={-1:6, 1:0, ...}` |
| `src/tracker/scekf.py` | Python SC-EKF (참조 구현) |
| `src/tracker/imu_tracker.py` | Python 측위 드라이버 |
| `src/View/visualize_comparison.py` | Python 평가 (GT 기반, oracle) |
| `src/View/ekf_tune.py` | grid-search 튜닝 |
| `android/prepare_assets.py` | 모델 → assets 변환 (P13 수정됨) |

---

## 6. 학습 데이터 클래스 매핑 (중요)

학습 LABEL_REMAP: `{-1:6, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5}`

| 인덱스 | 학습 이름 | 어플리케이션 이름 |
|---:|---|---|
| 0 | handbag | handbag |
| 1 | handheld | handheld |
| 2 | pocket | pocket |
| 3 | running | running |
| 4 | slow | slow_walk |
| 5 | trolley | trolley |
| 6 | noise | unknown |

학습 시 class weights (out_classifier2 train.log 12행): `[1.182, 0.770, 0.996, 1.205, 1.112, 1.178, 0.557]`
→ noise/unknown 의 weight 가 가장 낮음 = **학습 샘플 수가 가장 많음** → 어플리케이션 입력이 학습 분포에서 조금 벗어나면 unknown 으로 collapse 되기 쉬움.

---

## 7. 게이팅·AEKF 파라미터 (RobustEkfTracker.kt)

```kotlin
MAX_INNOV_NORM           = 6.0    // 1초 윈도우 절대 이노베이션 상한 (m)
MAHAL_CHI2_THRESHOLD     = 11.345 // χ²(ν=3, p=0.99) NIST
MAHAL_FAIL_SCALE         = 10.0   // Mahalanobis 실패 시 R 인플레이트 배수
MIN_MEAS_VARIANCE        = 0.05   // 측정 공분산 하한 (m²)
MAX_MEAS_VARIANCE        = 100.0  // 상한 (m²)
SIGMA_A                  = 0.0316 // process noise (m/s², Python √1e-3)
LOW_ROTACC_INFLATE       = 5.0    // RotVec 정확도 < 2 시 R 추가 inflate
MAX_INDOOR_SPEED         = 5.0    // 속도 클램프 (m/s)

CLASS_R_SCALE = [15, 10, 5, 50, 5, 7, 100]
//               handbag, handheld, pocket, running, slow_walk, trolley, unknown

// ── P22 ZUPT (Zero-Velocity Update) 파라미터 ───────────────
STILL_LIN_ACC_THRESHOLD = 0.20  // worldLinAcc 평균 norm < 이 값 → 정지 후보
STILL_GYR_THRESHOLD     = 0.05  // worldGyr  평균 norm < 이 값 → 정지 후보
STILL_WINDOW_SIZE       = 50    // 슬라이딩 윈도우 길이 (0.5초 @100Hz)
STILL_ENTER_HOLD_MS     = 500   // 정지 진입 hysteresis
STILL_EXIT_HOLD_MS      = 300   // 정지 해제 hysteresis
STILL_VARV_DECAY        = 0.5   // 정지 시 varV 한 step 축소 비율
STILL_VARV_FLOOR        = 1e-4  // varV 최저 한계
```

> P14 에서 Python grid-search 값(0.001~1.0)으로 동기화했더니 발산 → P15 에서 위 값으로 원복. Python 값은 GT 자세 환경 전제이므로 어플리케이션에 직접 사용 불가.
>
> P22 에서 정지 시 발산 차단을 위해 ZUPT 도입. trolley 오분류 시 update 누적 차단 + 잔여 가속도 적분 차단 둘 다 처리.

---

## 8. UI 외부 API (MainActivity 가 호출)

```kotlin
viewModel.start()    // 측위 시작
viewModel.stop()     // 측위 정지
viewModel.reset()    // 상태 초기화
viewModel.state      // StateFlow<LocalizationState>
viewModel.state.value.trackPoints  // CSV export
```

`LocalizationState` 필드:
- `isRunning: Boolean`
- `position / posStd / velocity: Triple<Double, Double, Double>`
- `carryMode: String`, `carryProb: Float`
- `trackPoints: List<Pair<Double, Double>>` (EKF 궤적)
- `modelTrackPoints: List<Pair<Double, Double>>` (현재 빈 리스트)
- `inferLatency: Long`

---

## 9. Git 상태 / 커밋 안내

- **현재 브랜치**: `android`
- **`.git/index.lock`** stale (May 8 부터, sandbox 권한으로 삭제 불가) — 사용자가 Windows cmd 에서 처리 필요
- **커밋 대기 파일 (P12~P21)**:
  ```
  android/app/src/main/java/com/imulocal/LocalizationViewModel.kt
  android/app/src/main/java/com/imulocal/AbsoluteSensorNode.kt
  android/app/src/main/java/com/imulocal/StatelessInferenceNode.kt
  android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
  android/app/src/main/java/com/imulocal/EkfBridge.kt
  android/app/src/main/java/com/imulocal/MainActivity.kt          # P21
  android/app/src/main/res/layout/activity_main.xml               # P21
  android/app/src/main/assets/norm_mean.txt
  android/app/src/main/assets/norm_std.txt
  android/prepare_assets.py
  수정_이력_보고서.docx
  PROJECT_CONTEXT.md   (이 파일)
  ```

**권장 커밋 명령** (Windows cmd, 인덱스 lock 해제 후):

```cmd
cd /d D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128
del .git\index.lock
git reset HEAD
git add android/app/src/main/java/com/imulocal/LocalizationViewModel.kt
git add android/app/src/main/java/com/imulocal/AbsoluteSensorNode.kt
git add android/app/src/main/java/com/imulocal/StatelessInferenceNode.kt
git add android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
git add android/app/src/main/java/com/imulocal/EkfBridge.kt
git add android/app/src/main/assets/norm_mean.txt
git add android/app/src/main/assets/norm_std.txt
git add android/prepare_assets.py
git add 수정_이력_보고서.docx
git add PROJECT_CONTEXT.md
git status
git commit -m "[P12~P20] 단방향 아키텍처 완성: Stage1+2+3+Controller" ^
  -m "P12: modelTrack yaw 회전 복원." ^
  -m "P13: norm_params 를 out_classifier2 통계로 일치, prepare_assets.py 경로 수정." ^
  -m "P15: P14 (R_SCALE Python 동기화) 발산 유발 → P13 이전 디자인 원복." ^
  -m "P16: LocalizationViewModel 메인 파이프라인 ~960 줄 → 95 줄 stub 으로 철거. 엔진 모듈 보존." ^
  -m "P17: AbsoluteSensorNode 신규 (Stage 1 절대 좌표계 센서 노드)." ^
  -m "P18: StatelessInferenceNode 신규 (Stage 2 Pure Function 추론기)." ^
  -m "P19: RobustEkfTracker 신규 (Stage 3 자체 미니 EKF + AEKF + 게이팅)." ^
  -m "P20: LocalizationViewModel 컨트롤러 본문 재작성 + propagateQueue + 동기화."
```

---

## 10. 다음 단계 (우선순위 순)

### A. 실기기 검증
1. **빌드 & 설치**: Android Studio 에서 `./gradlew assembleDebug` 또는 GUI 빌드
2. **권한**: 별도 권한 불필요 (IMU 는 정상 접근, ROTATION_VECTOR 도 동일)
3. **테스트 경로**:
   - 직선 보행 10m → EKF 위치가 직선을 그리는지
   - 사각형 보행 10×10m → 회전 시 발산이 없는지
   - 8자 회전 → yaw 추정이 안정적인지
   - 정지 → trackPoints 가 한 점에 머무는지

### B. 진단 로그 활용
- `Logcat` 필터: `Stage1.AbsNode`, `Stage2.InferNode`, `Stage3.EkfTracker`, `Controller`
- `stage3.getDiagnostics()` → `updates / rejInnov / rejMahal / lastInnov / lastNSE`
- RotVec 정확도: `Stage1.AbsNode RotVec 정확도: 2 (MEDIUM)` — 2 이상 권장

### C. AEKF 튜닝 (필요 시)
- `RobustEkfTracker.kt` 의 `CLASS_R_SCALE`, `MIN_MEAS_VARIANCE`, `MAX_INNOV_NORM` 등 조정
- 실측 노이즈 분포 기반 grid search 가능 (Python 코드 참조)
- 분류기가 unknown 으로 자주 collapse 되면 `CLASS_R_SCALE[6]` 을 5~10 으로 낮춰 측정 신뢰도 ↑

### D. 향후 개선 후보
- **EkfBridge / C++ SC-EKF 통합**: Stage 3 의 mini EKF 를 C++ SC-EKF 로 교체하되 자세 anchor API 추가. 더 정교한 stochastic cloning / Joseph form / Mahalanobis 활용 가능.
- **modelTrack 부활**: Stage 3 의 EKF 와 별개로 모델 only 궤적 시각화 (디버그용)
- **분류기 unknown collapse 근본 해결**: 학습 데이터의 noise 비중 줄이거나 입력 정규화 추가 개선
- **속도 expose**: `LocalizationState` 의 velocity 가 UI 에 잘 표시되는지

---

## 11. 알려진 위험 / 주의사항

1. **TYPE_ROTATION_VECTOR 정확도 의존성**: 자기계 캘리브레이션이 부실하면 Stage 1 의 회전이 부정확. 실내 금속/전자장비 근처에서 정확도 저하 가능. Stage 3 의 `LOW_ROTACC_INFLATE = 5.0` 으로 어느 정도 방어.

2. **워밍업 3초**: `WARMUP_MS = 3_000L` — 시작 후 3초 동안 trackPoints 적재 보류 (EKF 안정화 대기). 너무 길면 사용자 경험 저하, 너무 짧으면 초기 발산 표시.

3. **`.git/index.lock` stale**: sandbox 에서 삭제 불가. Windows 측에서 `del .git\index.lock` 필요.

4. **Write 도구 부작용**: 큰 기존 파일을 작은 새 컨텐츠로 덮어쓸 때 null 바이트 trailing garbage 발생 가능. 작업 후 Python 으로 `data.find(b'\x00')` 검증 권장 (이번 세션에서 처리 코드 자동화됨).

5. **EkfBridge / C++ SC-EKF 미사용**: 컴파일에는 포함되지만 Stage 3 에서 호출하지 않음. JNI 라이브러리 로드는 EkfBridge 의 `init { System.loadLibrary("imu_ekf_jni") }` 로 여전히 실행되므로 NDK 빌드는 필요. 빌드 실패 시 EkfBridge 의 init 블록을 try/catch 로 감싸거나 제거 검토.

6. **Stage 1 의 worldLinAcc 가 TYPE_LINEAR_ACCELERATION 의존**: 일부 저가형 기기에서 미지원 가능. 미지원 시 worldLinAcc 가 0 으로 머물러 EKF 가 동작 안 함. `AbsoluteSensorNode.start()` 의 `sLin` null 체크 부분에 폴백 로직 추가 검토 (TYPE_ACCELEROMETER 의 중력 수동 제거).

7. **분류기 unknown 편향**: 학습 데이터의 noise 클래스 비중이 가장 커서 어플리케이션 입력이 학습 분포에서 조금만 벗어나도 unknown 으로 매핑. `CLASS_R_SCALE[6] = 100` 이지만 게이팅이 정상 동작하면 발산 안 함.

---

## 12. 보고서 위치

- `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128\수정_이력_보고서.docx`
- 16장 692 단락, 모든 변경 내역 정리됨

---

## 13. 새 채팅에서 작업 시작 시 안내 멘트 (참고)

> "이 프로젝트는 IMU 기반 실내 측위 어플리케이션입니다. `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128` 에 있고 android 브랜치에서 작업 중입니다. P9~P15 에서 EKF 발산 문제를 겪은 후 P16~P20 으로 단방향 아키텍처(Stage 1 AbsoluteSensorNode → Stage 2 StatelessInferenceNode → Stage 3 RobustEkfTracker → LocalizationViewModel 컨트롤러)로 재설계했습니다. 첫 실행 가능 빌드가 완성된 상태이며, 다음 단계는 실기기 검증과 AEKF 튜닝입니다. `PROJECT_CONTEXT.md` 와 `수정_이력_보고서.docx` 를 먼저 읽어주세요. [현재 작업할 항목 명시]"

---

## 14. 사용자 선호사항

- **한국어 응답** (코드 주석도 한국어)
- 요약 요청 시 **개조식**
- 변경 사항은 **수정_이력_보고서.docx 에 누적**
- 큰 작업은 **TaskCreate / TaskUpdate 로 추적**
- git 커밋은 **사용자가 Windows cmd 에서 직접 실행** (sandbox 권한 제약)
