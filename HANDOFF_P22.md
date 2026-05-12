# IMU-based Indoor Localization — 핸드오프 문서 (P22 시점)

이 문서는 새 Claude 채팅에서 이어서 작업할 수 있도록 준비한 단일 자료다. 기존 `PROJECT_CONTEXT.md` 의 후속 스냅샷이며, P21 (실시간 영점 보정) 과 P22 (ZUPT) 까지 반영되어 있다.

> **새 채팅 시작 시**: 이 문서와 `수정_이력_보고서.docx` 를 먼저 읽고, "현재 P22 까지 완료된 단방향 IMU 측위 어플리케이션을 이어서 작업합니다" 로 시작하면 충분하다.

---

## 1. 현재 상태 (한 줄 요약)

P22 ZUPT (Zero-Velocity Update) 도입까지 완료. 정지 30 초 시험에서 13×40 m 발산 → 한 점 근처 유지로 차단 예상. **실기기 빌드 후 ZUPT 동작 통계 (`stillSkip`, `zupt`) 와 보행 궤적 검증이 다음 단계**.

---

## 2. 프로젝트 개요

- **목적**: 스마트폰 IMU(가속도계 + 자이로 + 지자기)와 1D-ResNet(ResMLP) 모델 + EKF 융합으로 실내 측위
- **기반**: TLIO(MIT) + Oxford-IOD 데이터셋, ResMLP 회귀 헤드 + 7-way 휴대 자세 분류 헤드
- **루트 경로**: `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128`
- **브랜치**: `android`
- **주요 폴더**
  - `src/` — Python 학습/평가 코드 (변경 없음, 참조용)
  - `android/` — 안드로이드 어플리케이션 (모든 P9~P22 변경 집중)
  - `mobile_assets/` — 모바일 배포용 .ptl / norm_params / model_meta
  - `outputs/` — 학습 결과 (norm_mean.npy, norm_std.npy, best.pth)
  - `수정_이력_보고서.docx` — P9~P22 전체 수정 이력 (18 장, 726 단락)
  - `PROJECT_CONTEXT.md` — 기존 컨텍스트 (이 문서의 직전 버전)

---

## 3. 사용된 학습 모델

| 폴더 | use_classifier | input_len | std (ch0~2) | 용도 |
|---|---|---|---|---|
| `out_classifier2` | **true** | **100** | **[0.118, 0.120, 0.148]** | **어플리케이션 배포 모델 (정답)** |
| `out_classifier_7way` | true | 200 | [0.46, 0.55, 0.53] | standalone 분류기 (사전학습용) |
| `out_tlio_6ch_128` | false | 200 | [5.76, 6.07, 2.97] | 분류기 없는 회귀 전용 (사전학습용) |

> P13 에서 `prepare_assets.py` 의 norm_params 경로를 `out_tlio_6ch_128` → `out_classifier2` 로 수정. 모델/통계 일치 필수.

---

## 4. 단방향(One-Way) 아키텍처

P9~P15 의 양방향 피드백 발산을 구조적으로 차단하기 위해 P16~P20 으로 재설계됨.

```
[Sensor HW] acc, linAcc, gyr, rotVec
     ▼
[Stage 1] AbsoluteSensorNode  (EKF 존재 모름)
  - 시작 직후 2 초 실시간 영점 보정 (P21)
  - TYPE_ROTATION_VECTOR 절대 회전으로 acc/gyr 즉시 world frame 변환
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
  - ★ P22 ZUPT: 자체 정지 감지 → v 강제 0 + update skip
  - 출력: getPosition / getVelocity / getOrientation / getYawRad / getPositionStd / isStationary
     ▼
[Controller] LocalizationViewModel  (순수 데이터 라우터, ~250 줄)
  - propJob(5ms): stage1.drainPropagateQueue → stage3.propagate
  - inferJob(50ms): stage1.getWindowSamples → stage2.infer → stage3.update
  - uiJob(100ms): stage3 출력 + stage1 캘리브레이션 진행도 → _state
     ▼
MainActivity / TrackView (외부 API 호환 유지: start/stop/reset/state)
```

**핵심**: Stage 3 의 상태는 어떤 경로로도 Stage 1·2 로 전달되지 않는다.

---

## 5. 수정 이력 (P9 ~ P22)

| P | 장 | 요약 | 결과 |
|---|---|---|---|
| P9~P11 | 7 | freeze_static_state, 자기계 게이팅 | 부분 개선 |
| P12 | 8 | `modelTrack` 누적에 t_begin yaw 회전 복원 | 모델 only 발산 해결 |
| P13 | 9 | `prepare_assets.py` NORM 경로 수정 + norm_*.txt 재생성 | running collapse 해결 |
| P14 | 10 | EkfBridge R_SCALE Python 동기화 | **실패 — 발산 유발** |
| P15 | 11 | P14 롤백 — R_SCALE [15,10,5,50,5,7,100] 원복 | 안정 회복 |
| P16 | 12 | `LocalizationViewModel` 본문 ~960 → 95 줄 stub | 단방향 재설계 1/4 |
| P17 | 13 | `AbsoluteSensorNode.kt` 신규 — Stage 1 절대 좌표계 센서 노드 | 단방향 2/4 |
| P18 | 14 | `StatelessInferenceNode.kt` 신규 — Stage 2 Pure Function 추론기 | 단방향 3/4 |
| P19 | 15 | `RobustEkfTracker.kt` 신규 — Stage 3 자체 미니 EKF + AEKF | 단방향 4/4 |
| P20 | 16 | `LocalizationViewModel` 컨트롤러 재작성 + propagateQueue + 동기화 | **첫 실행 가능 빌드** |
| **P21** | **17** | 실시간 영점 자동 보정 — `AbsoluteSensorNode` 캘리브레이션 분기 + `performWarmup` + UI 진행도 | bias 적분 누적 차단 |
| **P22** | **18** | ZUPT (Zero-Velocity Update) — `RobustEkfTracker` 자체 정지 감지 + propagate v=0 강제 + update skip | 정지 30 초 발산(13/40m) → 한 점 유지 차단 |

---

## 6. P21 — 실시간 영점 보정 (확정 결과 포함)

### 6.1. 메커니즘

- 시작 직후 `CALIBRATION_DURATION_MS = 2_000L` 동안 모든 센서 입력을 누적
- 누적 중에는 `ringBuffer` / `propagateQueue` 적재 안 함 (Stage 2/3 는 영점 보정 전 입력을 보지 못함)
- 2 초 경과 시 `performWarmup()` 호출
  - `gyrBias` = 정지 평균
  - `linAccBias` = 정지 평균 (Android TYPE_LINEAR_ACCELERATION 은 중력 제거됨)
  - `accBias` = `mean(accBody) − R_meanᵀ·[0, 0, g]` (row-major: body-frame 중력 = `R[6..8]·g`)
- 100 Hz 리샘플링 직전 body frame 에서 bias 차감 후 절대 회전 적용

### 6.2. 실측 결과 (정상 동작 확인)

| 항목 | 측정값 | 정상 범위 | 평가 |
|---|---|---|---|
| `n` 누적 샘플 | 2505 | 400~2500 | 상한 (모든 콜백 합산 ~1250 Hz) |
| `elapsed` | 2002 ms | 2000~2030 | 거의 정확 |
| `RotVec` 정확도 | 3 (HIGH) | 2 이상 | 최상위 (자기 환경 우수) |
| `gyrBias` | [0.00139, 0.00134, 0.00180] rad/s | < 0.01 | 정상 범위의 1/5 이내 |
| `linAccBias` | [-0.0086, 0.0173, -0.0176] m/s² | < 0.1 | 정상 범위의 1/5 이내 |
| `accBias` | [-0.0085, 0.0185, -0.0169] m/s² | < 0.5 | 매우 양호 |

### 6.3. 공개 API

```kotlin
stage1.isCalibrating(): Boolean
stage1.isCalibrationDone(): Boolean
stage1.getCalibrationProgress(): Float    // 0.0 → 1.0
stage1.getBiasSnapshot(): Triple<FloatArray, FloatArray, FloatArray>
```

### 6.4. UI 통합

- `LocalizationState` 에 `calibrating`, `calibProgress`, `calibDone` 필드 추가
- `uiJob` 가 stage3 초기화 여부와 무관하게 stage1 진행도 매 100 ms 폴링
- `activity_main.xml` 의 `calibCard` (MaterialCardView, 황색 배경, 기본 gone) + ProgressBar + 퍼센트 텍스트
- `MainActivity` 가 `s.calibrating` 감지 시 카드 표시

---

## 7. P22 — ZUPT (Zero-Velocity Update)

### 7.1. 도입 배경

- P21 캘리브레이션이 정상 동작 (bias 매우 작음) 했음에도 **정지 30 초 시험에서 x=13 m, y=40 m 발산** 관측
- `Controller 측위 정지 — 진단: updates=683 rejInnov=0 rejMahal=0` — **게이트가 한 번도 동작하지 않고 모든 update 통과**
- UI 표시상 `carryMode` 가 trolley(`CLASS_R_SCALE=7`) ↔ unknown(`R=100`) 번갈아 분류
- trolley 구간에서 모델 disp 가 작은 노이즈로 EKF 에 누적 → 속도 `|v|` 1 m/s 까지 키워짐
- 결론: 캘리브레이션은 bias 적분은 차단하지만, **분포 외 정지 입력의 모델 update 누적은 차단 못 함**

### 7.2. 메커니즘

`RobustEkfTracker.kt` 가 자체 입력값(`worldLinAcc`, `worldGyr`) 만으로 정지 감지 — 단방향 원칙 유지.

```
매 propagate():
  aNorm = ‖worldLinAcc‖
  gNorm = ‖worldGyr‖
  → 슬라이딩 윈도우(50 sample = 0.5 s @100 Hz) 평균
  → aMean < 0.20 m/s² AND gMean < 0.05 rad/s 면 candidateStill
  → hysteresis (진입 500 ms / 해제 300 ms) 통과 시 isStationary 토글

isStationary == true:
  - propagate: v[i] = 0, varV[i] *= 0.5 (최저 1e-4), covPV[i] = 0, zuptApplications++
  - update: anchor 만 갱신 후 false 반환, stationaryUpdatesSkipped++
```

### 7.3. 파라미터 (companion object)

```kotlin
STILL_LIN_ACC_THRESHOLD = 0.20  // worldLinAcc 평균 norm 임계 (m/s²)
STILL_GYR_THRESHOLD     = 0.05  // worldGyr  평균 norm 임계 (rad/s)
STILL_WINDOW_SIZE       = 50    // 슬라이딩 윈도우 길이 (0.5초 @100Hz)
STILL_ENTER_HOLD_MS     = 500   // 정지 진입 hysteresis
STILL_EXIT_HOLD_MS      = 300   // 정지 해제 hysteresis
STILL_VARV_DECAY        = 0.5   // 정지 시 varV 한 step 축소 비율
STILL_VARV_FLOOR        = 1e-4  // varV 최저 한계
```

### 7.4. 공개 API / 진단

```kotlin
stage3.isStationary(): Boolean

stage3.getDiagnostics(): String
// 예: "updates=NN rejInnov=NN rejMahal=NN stillSkip=NN zupt=NN still=true|false
//      aMean=X.XXX gMean=X.XXXX lastInnov=X.XXXm lastNSE=X.XX"
```

### 7.5. 검증 시 기대값 (실기기 빌드 후 정지 30 초 재시험)

| 채널 | 기대값 | 비고 |
|---|---|---|
| UI `\|v\|` | 정지 진입 직후 0 으로 떨어져 유지 | 이전: 1 m/s 까지 누적 |
| UI trackPoints | 한 점 근처에서 머무름 | 이전: 13 m × 40 m 발산 |
| Logcat `stillSkip` | 양수 (정지 구간 update skip 횟수) | ZUPT 동작 카운터 |
| Logcat `zupt` | 양수 (propagate ZUPT 적용 횟수) | 1000+ 예상 |
| Logcat `still=true` | true | 정지 확정 상태 |
| `aMean`, `gMean` | < 0.20, < 0.05 | 임계 튜닝 근거 |

---

## 8. 핵심 파일 인벤토리 (`android/app/src/main/`)

### 활성 사용 (단방향 파이프라인)

| 파일 | 줄수 | 역할 |
|---|---:|---|
| `java/com/imulocal/AbsoluteSensorNode.kt` | ~505 | Stage 1 — 절대 좌표계 센서 노드 + 실시간 영점 보정 (P21) |
| `java/com/imulocal/StatelessInferenceNode.kt` | 170 | Stage 2 — 상태 비저장 추론기 (Pure Function) |
| `java/com/imulocal/RobustEkfTracker.kt` | **529** | Stage 3 — 강건한 상태 추정기 (mini EKF + AEKF + ZUPT P22) |
| `java/com/imulocal/LocalizationViewModel.kt` | ~250 | 단방향 컨트롤러 (propJob/inferJob/uiJob) + 캘리브레이션 진행도 |
| `java/com/imulocal/InferenceEngine.kt` | 146 | PyTorch Mobile Lite 모델 호출부 |
| `java/com/imulocal/MainActivity.kt` | ~132 | UI + 캘리브레이션 카드 |
| `java/com/imulocal/TrackView.kt` | (변경 없음) | 궤적 그리기 |
| `res/layout/activity_main.xml` | ~193 | 캘리브레이션 카드 + 트랙뷰 + 상태 카드 + 버튼 |
| `assets/imu_model.ptl` | — | 학습된 모델 (out_classifier2 변환본) |
| `assets/norm_mean.txt`, `norm_std.txt` | 73B/70B | out_classifier2 정규화 통계 |
| `assets/model_meta.json` | — | 모델 메타 (input_len=100, output_cls=7 등) |

### 보존되었으나 현재 미사용

| 파일 | 비고 |
|---|---|
| `java/com/imulocal/EkfBridge.kt` (246) | C++ SC-EKF JNI 래퍼 — 향후 자세 anchor API 추가 후 활용 가능 |
| `java/com/imulocal/ImuCollector.kt` (280) | 기존 IMU 수집 모듈 — Stage 1 으로 대체됨 |
| `java/com/imulocal/ImuTestActivity.kt` | raw 센서 진단 화면 (캘리브레이션 미적용 — raw 검증 용도) |
| `cpp/EkfJniBridge.cpp` (455) | JNI 진입점 12개 함수 |
| `cpp/ekf/imu_ekf.{h,cpp}` (972) | SC-EKF 코어 + Eigen 수학 |

### Python 학습 코드 (변경 없음, 참조용)

| 파일 | 역할 |
|---|---|
| `src/Network/model_twolayer.py` | TwoLayerModel (extractor + reg + classifier) |
| `src/Network/train.py` | Phase 2 학습 진입점 |
| `src/Trans/dataset.py` | TLIONpySingleDataset, `_window_to_gravity_aligned()` |
| `src/Trans/classification_dataset.py` | 분류 라벨링, LABEL_REMAP |
| `src/tracker/scekf.py` | Python SC-EKF (참조 구현) |
| `src/View/visualize_comparison.py` | Python 평가 (GT 기반, oracle) |
| `android/prepare_assets.py` | 모델 → assets 변환 (P13 수정됨) |

---

## 9. 학습 데이터 클래스 매핑

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

학습 시 class weights: `[1.182, 0.770, 0.996, 1.205, 1.112, 1.178, 0.557]`
→ noise/unknown 의 weight 가 가장 낮음 = **학습 샘플 수가 가장 많음** → 분포 밖 입력이 unknown 또는 인접 클래스(trolley 등)로 매핑되기 쉬움. P22 ZUPT 가 이 효과를 차단.

---

## 10. Stage 3 파라미터 전체 (`RobustEkfTracker.kt`)

```kotlin
// ── 게이트 ─────────────────────────────────────────────
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

// ── P22 ZUPT ───────────────────────────────────────────
STILL_LIN_ACC_THRESHOLD = 0.20  // 정지 후보 임계 (m/s²)
STILL_GYR_THRESHOLD     = 0.05  // 정지 후보 임계 (rad/s)
STILL_WINDOW_SIZE       = 50    // 슬라이딩 윈도우 (0.5초 @100Hz)
STILL_ENTER_HOLD_MS     = 500   // 진입 hysteresis
STILL_EXIT_HOLD_MS      = 300   // 해제 hysteresis
STILL_VARV_DECAY        = 0.5
STILL_VARV_FLOOR        = 1e-4
```

> P14 에서 Python grid-search 값(0.001~1.0)으로 동기화했더니 발산 → P15 에서 위 값으로 원복.
> Python 값은 GT 자세 환경 전제이므로 어플리케이션에 직접 사용 불가.

---

## 11. UI 외부 API (MainActivity 가 호출)

```kotlin
viewModel.start()    // 측위 시작
viewModel.stop()     // 측위 정지
viewModel.reset()    // 상태 초기화
viewModel.state      // StateFlow<LocalizationState>
viewModel.state.value.trackPoints  // CSV export
```

`LocalizationState` 필드 (P22 시점):

```kotlin
data class LocalizationState(
    val isRunning:         Boolean = false,
    val position:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
    val posStd:            Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
    val velocity:          Triple<Double, Double, Double> = Triple(0.0, 0.0, 0.0),
    val carryMode:         String  = "unknown",
    val carryProb:         Float   = 0f,
    val trackPoints:       List<Pair<Double, Double>> = emptyList(),
    val modelTrackPoints:  List<Pair<Double, Double>> = emptyList(),
    val inferLatency:      Long    = 0L,
    // P21 캘리브레이션
    val calibrating:       Boolean = false,
    val calibProgress:     Float   = 0f,
    val calibDone:         Boolean = false
)
```

> 향후 P22 ZUPT 상태(`stationary: Boolean`) 를 UI 에 노출하려면 같은 패턴으로 필드 추가.

---

## 12. 검증 가이드 (다음 단계 실행 시 사용)

### 12.1. 빌드 + 설치

- Android Studio Run 버튼 (`Shift + F10`) — 빌드 + APK + 폰 설치 + 자동 실행
- 또는 cmd: `cd /d D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128\android && gradlew assembleDebug installDebug`

### 12.2. Logcat 캡처 (P22 검증 핵심)

```cmd
cd /d D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128
adb logcat -c
adb logcat -v threadtime "Stage1.AbsNode:*" "Stage2.InferNode:*" "Stage3.EkfTracker:*" "Controller:*" "*:S" > logcat_zupt.txt
```

이 상태에서 폰에서 측위 수행 → 측위 정지 → `Ctrl+C`

### 12.3. P21 캘리브레이션 정상 범위표

| 항목 | 정상 범위 |
|---|---|
| `n` 누적 샘플 | 400 ~ 2500 |
| `gyrBias` 각 축 | 절대값 < 0.01 rad/s |
| `linAccBias` 각 축 | 절대값 < 0.1 m/s² |
| `accBias` 각 축 | 절대값 < 0.5 m/s² |
| `elapsed` | 2000~2030 ms |
| `RotVec` 정확도 | 2 이상 (3 권장) |

### 12.4. P22 ZUPT 정상 범위표

| 항목 | 기대값 | 비고 |
|---|---|---|
| `stillSkip` | 양수 | 정지 시 update skip 횟수 |
| `zupt` | 1000+ | propagate ZUPT 적용 횟수 |
| `still` | true (정지 시) | 정지 확정 상태 |
| `aMean` | < 0.20 | 임계 조정 근거 |
| `gMean` | < 0.05 | 임계 조정 근거 |
| 정지 30 초 시 UI `\|v\|` | 0 m/s 유지 | 이전 1 m/s |
| 정지 30 초 시 trackPoints | 한 점 근처 | 이전 13×40 m |

---

## 13. 다음 단계 (우선순위 순)

### A. 실기기 P22 검증 (즉시)

1. **빌드 & 설치**: Android Studio Run 버튼
2. **정지 30 초 시험**: 시작 → 캘리브레이션 카드 사라진 후 워밍업 3 초 대기 → 30 초 정지 → 정지 버튼
3. **Logcat 진단**: `Controller 측위 정지 — 진단:` 줄의 `stillSkip`, `zupt`, `aMean`, `gMean` 확인
4. **UI 진단**: 속도 `|v|` 0 유지 여부, trackPoints 한 점 근처 머무름 여부

### B. 실기기 보행 검증 (PROJECT_CONTEXT §10 A)

1. 직선 보행 10 m → 거의 직선
2. 사각형 보행 10×10 m → 사각형이 닫힘 (loop closure)
3. 8 자 회전 → 한 자리에서 작은 원호
4. 정지 → trackPoints 한 점에 머무름 (P22 검증과 동일)

### C. P22 임계값 미세 튜닝

실측 `aMean`, `gMean` 분포 기반으로:
- `STILL_LIN_ACC_THRESHOLD = 0.20` 조정 (정지 진입이 너무 까다로우면 ↑, 느린 보행에서 잘못 진입하면 ↓)
- `STILL_GYR_THRESHOLD = 0.05` 마찬가지
- `STILL_ENTER_HOLD_MS = 500` (느린 정지 인식 → 줄임, 빠른 false-enter → 늘림)

### D. 향후 개선 후보

- **stationary 상태 UI 노출**: `LocalizationState.stationary` + "정지 감지" 인디케이터
- **EkfBridge / C++ SC-EKF 통합**: Stage 3 의 mini EKF → C++ SC-EKF 로 교체하되 자세 anchor API + ZUPT 인터페이스 유지
- **modelTrack 부활**: Stage 3 의 EKF 와 별개로 모델 only 궤적 시각화 (디버그용)
- **분류기 unknown collapse 근본 해결**: 학습 데이터 noise 비중 축소 또는 입력 정규화 추가 개선
- **Python 평가에 동일 ZUPT 미러링**: 학습/검증 일관성

---

## 14. 알려진 위험 / 한계

### P21 캘리브레이션

1. **사용자가 캘리브레이션 동안 기기를 움직이면** 평균이 편향되어 잘못된 bias 가 확정됨 — UI 안내로 보호
2. **자기 환경 (RotVec 정확도 0/1)** 에서는 raw acc bias 추출 정확도 떨어짐
3. `Kotlin private property "calibrating"` 과 `public function "isCalibrating()"` 는 JVM signature 가 다르게 (각각 `getCalibrating()` / `isCalibrating()`) 의도적으로 분리됨 — 컴파일 충돌 없음

### P22 ZUPT

1. **매우 느린 보행** (가속도 진폭 < `STILL_LIN_ACC_THRESHOLD`) 시 ZUPT 잘못 진입 가능 → 임계 조정 또는 추가 시그널 필요
2. **트롤리 위 핸드폰 거치 이동** 시 진동이 작아 정지로 오인될 수 있음
3. 임계값이 모든 기기에 일률적이지 않아 실측 데이터로 추후 튜닝 필요

### 일반

1. **TYPE_ROTATION_VECTOR 정확도 의존성**: 자기계 캘리브레이션 부실 시 Stage 1 회전 부정확. `LOW_ROTACC_INFLATE = 5.0` 으로 어느 정도 방어
2. **워밍업 3 초** (`WARMUP_MS = 3_000L`): 시작 후 3 초 동안 trackPoints 적재 보류 (EKF 안정화 대기)
3. **`.git/index.lock`** 가 sandbox 권한 부족으로 stale 남기 쉬움 — Windows 측에서 `del .git\index.lock`
4. **`Write` 도구 부작용**: 큰 기존 파일을 작은 새 컨텐츠로 덮어쓸 때 null 바이트 trailing garbage 가능. 작업 후 Python 으로 `data.find(b'\x00')` 검증 권장
5. **EkfBridge / C++ SC-EKF 미사용**: 컴파일에는 포함되지만 Stage 3 에서 호출하지 않음. NDK 빌드는 필요. 빌드 실패 시 EkfBridge 의 init 블록을 try/catch 로 감싸거나 제거 검토
6. **Stage 1 의 worldLinAcc 가 TYPE_LINEAR_ACCELERATION 의존**: 일부 저가형 기기에서 미지원. 미지원 시 worldLinAcc 가 0 으로 머물러 EKF 동작 안 함. `AbsoluteSensorNode.start()` 의 `sLin` null 체크 부분에 폴백 로직 추가 검토
7. **분류기 unknown 편향**: P22 ZUPT 가 정지 구간 영향은 차단했지만 보행 중 분포 외 입력 시 같은 메커니즘 가능 — `CLASS_R_SCALE[6] = 100` 로 보수적 처리 중

---

## 15. Git 상태 / 커밋 안내

- **브랜치**: `android`
- **`.git/index.lock`** 가 sandbox 권한으로 삭제 불가한 경우가 자주 발생 — Windows cmd 에서 처리 필요

**커밋 대기 파일 (P12~P22)**

```
android/app/src/main/java/com/imulocal/LocalizationViewModel.kt
android/app/src/main/java/com/imulocal/AbsoluteSensorNode.kt
android/app/src/main/java/com/imulocal/StatelessInferenceNode.kt
android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
android/app/src/main/java/com/imulocal/EkfBridge.kt
android/app/src/main/java/com/imulocal/MainActivity.kt
android/app/src/main/res/layout/activity_main.xml
android/app/src/main/assets/norm_mean.txt
android/app/src/main/assets/norm_std.txt
android/prepare_assets.py
수정_이력_보고서.docx
PROJECT_CONTEXT.md
HANDOFF_P22.md       (이 파일)
```

**권장 커밋 명령** (Windows cmd, 인덱스 lock 해제 후)

```cmd
cd /d D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128
del .git\index.lock
git reset HEAD
git add android/app/src/main/java/com/imulocal/LocalizationViewModel.kt
git add android/app/src/main/java/com/imulocal/AbsoluteSensorNode.kt
git add android/app/src/main/java/com/imulocal/StatelessInferenceNode.kt
git add android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
git add android/app/src/main/java/com/imulocal/EkfBridge.kt
git add android/app/src/main/java/com/imulocal/MainActivity.kt
git add android/app/src/main/res/layout/activity_main.xml
git add android/app/src/main/assets/norm_mean.txt
git add android/app/src/main/assets/norm_std.txt
git add android/prepare_assets.py
git add 수정_이력_보고서.docx
git add PROJECT_CONTEXT.md
git add HANDOFF_P22.md
git status
git commit -m "[P12~P22] 단방향 아키텍처 + 캘리브레이션 + ZUPT" ^
  -m "P12~P15: modelTrack yaw 복원, norm 통계 일치, R_SCALE 원복." ^
  -m "P16~P20: 단방향 4 Stage 재설계 + 첫 실행 가능 빌드." ^
  -m "P21: 실시간 영점 자동 보정 (AbsoluteSensorNode 2초 워밍업)." ^
  -m "P22: ZUPT (RobustEkfTracker 자체 정지 감지, propagate v=0, update skip)."
```

---

## 16. 사용자 선호사항

- **한국어 응답** (코드 주석도 한국어)
- 요약 요청 시 **개조식**
- 변경 사항은 **수정_이력_보고서.docx 에 누적**
- 큰 작업은 **TaskCreate / TaskUpdate 로 추적**
- git 커밋은 **사용자가 Windows cmd 에서 직접 실행** (sandbox 권한 제약)
- 빌드 실패 시 **첫 에러 줄 + Build 탭 출력** 알려주면 정확히 진단 가능

---

## 17. 새 채팅에서 시작 시 안내 멘트 (참고)

> "이 프로젝트는 IMU 기반 실내 측위 어플리케이션입니다. `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128` 에 있고 `android` 브랜치에서 작업 중입니다. P9~P20 에서 단방향 아키텍처(Stage 1 AbsoluteSensorNode → Stage 2 StatelessInferenceNode → Stage 3 RobustEkfTracker → LocalizationViewModel 컨트롤러) 로 재설계했고, P21 에서 실시간 영점 자동 보정, P22 에서 ZUPT (Zero-Velocity Update) 까지 도입했습니다. 첫 실행 가능 빌드는 P20 시점에 완성되었고 P22 까지 적용된 빌드의 실기기 검증이 다음 단계입니다. `HANDOFF_P22.md` 와 `수정_이력_보고서.docx` 를 먼저 읽어주세요. [현재 작업할 항목 명시]"

---

## 18. 변경된 파일 핵심 코드 위치 (검색 키워드)

| 작업 | 파일 | 핵심 키워드 |
|---|---|---|
| P21 캘리브레이션 상수 | `AbsoluteSensorNode.kt` | `CALIBRATION_DURATION_MS`, `STANDARD_GRAVITY` |
| P21 상태 변수 | `AbsoluteSensorNode.kt` | `calibrating`, `calibAccSum`, `accBias` |
| P21 분기 + warmup | `AbsoluteSensorNode.kt` | `if (calibrating)`, `performWarmup()` |
| P21 ViewModel | `LocalizationViewModel.kt` | `calibrating`, `calibProgress`, `stage1.isCalibrating()` |
| P21 UI | `MainActivity.kt`, `activity_main.xml` | `calibCard`, `tvCalibPercent` |
| P22 ZUPT 상수 | `RobustEkfTracker.kt` | `STILL_LIN_ACC_THRESHOLD`, `STILL_WINDOW_SIZE` |
| P22 ZUPT 상태 | `RobustEkfTracker.kt` | `isStationary`, `linAccNormBuf`, `zuptApplications` |
| P22 ZUPT 적용 | `RobustEkfTracker.kt` | `updateStationaryState()`, `if (isStationary)` |
| P22 진단 확장 | `RobustEkfTracker.kt` | `getDiagnostics()`, `stillSkip`, `zupt`, `still=` |
