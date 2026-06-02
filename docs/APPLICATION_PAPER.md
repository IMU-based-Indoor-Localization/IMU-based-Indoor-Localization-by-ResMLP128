# Application Section — IMU 기반 실내 측위 Android 어플리케이션 (논문용)

> **용도**: 논문 Application/Implementation 섹션 직접 활용 가능한 상세 정리.
> **발표 대본 / Q&A 버전**: `APPLICATION_PRESENTATION.md` 참조.

목차:
- [3.1 시스템 개요](#31-시스템-개요)
- [3.2 센서 수집과 전처리 (ImuCollector)](#32-센서-수집과-전처리-imucollector)
- [3.3 모델 추론 (InferenceEngine)](#33-모델-추론-inferenceengine)
- [3.4 PATH_B Dead-Reckoning 적분](#34-path_b-dead-reckoning-적분)
- [3.5 EKF 비교 경로 (옵션)](#35-ekf-비교-경로-옵션)
- [3.6 시각화 및 사용자 인터페이스](#36-시각화-및-사용자-인터페이스)
- [3.7 검증 및 재현성 인프라](#37-검증-및-재현성-인프라)
- [3.8 설계 결정과 학술적 근거](#38-설계-결정과-학술적-근거)
- [3.9 한계와 향후 작업](#39-한계와-향후-작업)
- [3.10 기술 스택 요약](#310-기술-스택-요약)
- [3.11 디렉토리 구조](#311-디렉토리-구조-참고)

---

## 3.1 시스템 개요

본 어플리케이션은 Android 단말의 관성 센서만으로 GPS-denied 실내 환경에서 보행자의 위치를 실시간 추정하는 시스템이다. 핵심 구성은 다음과 같다:

| 컴포넌트 | 기술 | 주파수 / 주기 |
|---|---|---|
| 센서 수집 | Android SensorManager (acc/gyr/linAcc/rotVec) | 100Hz 리샘플 |
| 자세 추정 | TYPE_ROTATION_VECTOR (sensor fusion) | 50Hz (`SENSOR_DELAY_GAME`) |
| 변위 회귀 | PyTorch Mobile (ResNet-1D 계열) | 20Hz 추론 |
| 위치 적분 | PDR-hybrid (모델 크기 + RotVec 방향) | 20Hz |
| 지도 표시 | Naver Map SDK 3.23.2 | 화면 갱신 동기 |
| EKF 비교 | C++ EkfBridge (JNI) | 5ms propagation |
| 저장/재현 | CSV (`sensor,ts_ns,x,y,z,w`) | 측정 종료 시 / Replay 모드 |

## 3.2 센서 수집과 전처리 (ImuCollector)

### 3.2.1 다중 센서 동기화
- **TYPE_ACCELEROMETER**: raw 가속도 (중력 포함). EKF propagation 입력.
- **TYPE_GYROSCOPE**: 각속도 (rad/s). EKF propagation + 네트워크 입력.
- **TYPE_LINEAR_ACCELERATION**: 중력 제거된 가속도. 네트워크 입력 (학습 데이터와 정합).
- **TYPE_ROTATION_VECTOR**: 가속도+자이로+자력계 융합 절대 자세. dead-reckoning 의 heading 소스.

각 센서는 `SENSOR_DELAY_FASTEST` 로 등록 (≈200-250Hz 네이티브 rate), `processSensorData` 콜백에서 latest 값을 캐시한다. 가속도+자이로가 모두 수신된 이후 100Hz 리샘플 게이트 (`lastSampleTs`) 를 통해 ringBuffer 에 적재.

### 3.2.2 P21 정적 영점 보정
시작 후 2초간 (`CALIBRATION_DURATION_MS=2000`) `latestLinAcc + latestGyr` 평균을 누적하여 bias 추정. 이후 모든 샘플에 bias 를 차감한 값을 ringBuffer 에 저장. 캘리브 기간 중 ringBuffer 미적재 → EKF 초기화 자연스럽게 지연.

### 3.2.3 P33-A1 LPF
12Hz Butterworth 2차 (Direct Form I, scipy.signal.butter(2, 12, btype='low', fs=100)) LPF 를 네트워크 입력용 채널에만 적용 (linAccLpf, gyrLpf). EKF propagation 에는 raw 사용 (적분 정확도 유지).

```kotlin
y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2]
// b = [0.09131, 0.18263, 0.09131], a = [1.0, -0.98241, 0.34767]
```

### 3.2.4 윈도우 추출
- `getRawWindow()`: 100×6 channel-major `[linAcc(3) + gyr(3)]`, LPF 미적용 (RotVec DR 입력).
- `getRotMatWindow()`: 100×9 channel-major (per-sample device→world rotation).
- 두 윈도우는 `takeLast(WINDOW_SIZE)` 동기 스냅샷 (1-2 샘플 어긋남 무시).

## 3.3 모델 추론 (InferenceEngine)

### 3.3.1 모델 사양
- **아키텍처**: ResNet 계열 1D CNN + global pooling + MLP head (`SimplePoolingReg`).
- **입력**: 100×6 channel-major float (linAcc xyz + gyr xyz).
- **출력**: (1) 2D 변위 [dx, dy] in gravity-aligned local frame, (2) 공분산 logits, (3) 7-class 휴대모드 logits (handbag/handheld/pocket/running/slow_walk/trolley/unknown).
- **학습 데이터**: OxIOD (iPhone, Oxford Inertial Odometry Dataset) + TLIO golden (Meta Aria headset). 100Hz 동기. window 1초 비겹침.

### 3.3.2 전처리 파이프라인
```
ringBuffer raw window
   ↓ getRawWindow() [body frame]
   ↓ RotVec gravity-aligned transform (windowed yaw0 normalized)
   ↓ unitScale (linAcc / 9.81, P46 OoD fix)
   ↓ norm_mean / norm_std standardization
모델 입력
```

학습 시와 동일한 *gravity-aligned + window-start yaw normalized* 프레임으로 변환되어, 모델은 절대 방위에 대해 invariant 하다.

### 3.3.3 RotVec 기반 gravity-aligned 변환
- 각 sample 의 `latestRotMat` (device→world, ENU) 를 보관.
- 윈도우 단위 변환 (`transformWindowRotVec`):
  ```
  yaw0 = atan2(R_start[1,0], R_start[0,0])     # window 시작 시점의 yaw
  R_yaw_inv = R_z(-yaw0)
  v_ga[t] = R_yaw_inv · R_rotvec[t] · v_body[t]
  ```
- EKF clone rotation 과 분리 — 자력계 융합 절대 자세이므로 yaw drift 가 없다 (학습 'ga' frame 과 정확히 동일).

## 3.4 PATH_B Dead-Reckoning 적분

PATH_B 는 모델 출력의 *크기* 와 RotVec 의 *방향* 을 결합하는 PDR-hybrid 구조이다.

### 3.4.1 핵심 적분 식
```
per 50ms tick:
  result = inferEngine.infer(world_window)
  raw_xy = |result.disp[0:2]|             # 모델 출력 크기 (1초 변위)
  scale  = adaptiveScale(raw_xy)          # 적응형 스케일 [1, 2.5, 5, 7]
  speed  = raw_xy × scale / 1.0s          # m/s
  heading = headingAt(window_end)         # RotVec yaw, drift-free
  netPos.x += speed × cos(heading) × dt
  netPos.y += speed × sin(heading) × dt
```

### 3.4.2 적응형 스케일 (P67)
모델 출력 magnitude 가 학습 분포 대비 압축되는 saturation 현상을 보정하기 위한 piecewise linear function.

```kotlin
ADAPTIVE_SCALE_THRESH = [0.15, 0.25, 0.40]  // m
ADAPTIVE_SCALE_VALUES = [1.0, 2.5, 5.0, 7.0]

fun adaptiveScale(rawXy: Double): Double = when {
    rawXy >= ADAPTIVE_RAW_OUTLIER -> 1.0   // P67-C: 이상치 fallback
    rawXy <  THRESH[0]            -> VALUES[0]  // 1.0
    rawXy <  THRESH[1]            -> VALUES[1]  // 2.5
    rawXy <  THRESH[2]            -> VALUES[2]  // 5.0
    else                          -> VALUES[3]  // 7.0
}

// 추가 안전망 (P67-C)
ADAPTIVE_RAW_OUTLIER = 0.70     // m — saturation 천장 초과 시 신뢰 안 함
MAX_EFFECTIVE_SPEED = 2.0       // m/s — 보행 물리 상한 clamp
```

### 3.4.3 정지/회전 분기 (Hysteresis)
- `gyrRms` 기반 정지 (STATIC) / 운동 (MOVING) 상태 머신.
- STATIC: `EkfBridge.freezeStaticState(anchor)` 로 위치·속도 강제 동결 + RotVec yaw 보정.
- MOVING → STATIC 진입: 5프레임 (≈250ms) gyrRms < threshold 연속 확인.
- STATIC → MOVING: 3프레임 연속 threshold 초과.

### 3.4.4 P55 20Hz 속도 적분
1초 비겹침 윈도우 단위 누적 (1Hz) 은 회전·정지 중 *완전 멈춤* 현상을 유발 → 20Hz 속도 적분으로 변경. 매 50ms 마다 (윈도우 overlap) 새 model output 으로 속도 갱신 후 dt 만큼 위치 적분 → 부드러운 궤적.

## 3.5 EKF 비교 경로 (옵션)

C++ JNI 기반 SC-EKF (`EkfBridge`) 가 propagation 은 항상 수행하며, 메뉴 토글로 measurement update 흐름을 PATH_B 와 *병렬* 활성화 가능.

- **propagate**: 5ms 단위 (`drainPropagateQueue` → JNI), acc/gyr raw 사용.
- **clone & update**: 1초 윈도우 단위. start/end clone 의 rotation 으로 transformWindowToWorldFrame → 추론 → `EkfBridge.update(tBegin, tEnd, disp, cov)`.
- **결과**: `EkfBridge.getPosition()` 을 50ms 마다 별도 코루틴 (`ekfPosJob`) 으로 read → 보라색 polyline 시각화.

PATH_B 와 EKF 의 *시각적 비교* 로 도메인 격차로 인한 EKF 발산을 학술적으로 demonstrating.

## 3.6 시각화 및 사용자 인터페이스

### 3.6.1 듀얼 뷰
- **격자 모드 (TrackView, 기본)**: 1m 고정 격자, 미터 좌표계, 시작점 (0, 0) 기준.
- **지도 모드 (Naver Map)**: 실세계 위경도, 사용자 long-press 로 anchor 변경 가능.
- 메뉴 토글로 두 view 즉시 전환 (visibility GONE 대신 INVISIBLE 사용 — 지도 fragment re-init 방지).

### 3.6.2 위경도 변환
시작 anchor 기준 미터 → degree 선형 근사 (실내 단거리에 충분):
```kotlin
fun meterOffsetToLatLng(anchor: LatLng, dx: Double, dy: Double): LatLng {
    val dLat = dy / 111_320.0
    val dLng = dx / (111_320.0 * cos(Math.toRadians(anchor.latitude)))
    return LatLng(anchor.latitude + dLat, anchor.longitude + dLng)
}
```

### 3.6.3 재생 슬라이더 (P74)
지도 하단 SeekBar (0-100%) + 핑크 cursor 마커. PATH_B polyline 의 인덱스 비율로 매핑되어 임의 시점 위치를 즉시 확인. 슬라이더 100% = 라이브 추적 (자동), 100% 미만 = scrub 모드 (사용자 고정). 왕복·복잡 경로의 시간 순서를 시각적으로 추적 가능.

## 3.7 검증 및 재현성 인프라

### 3.7.1 측정 기록과 재생 (Replay)
- 모든 측정은 `sensor,ts_ns,x,y,z,w` 단일 CSV 로 단말 저장 (`imu_record_<epoch>.csv`).
- `ImuCollector.startReplay()` 가 동일 형식 CSV 를 *기록 시점 간격으로* 재생 → SensorManager 등록 우회 → 결정론적 재현.
- `latest.csv` (지정 경로) 를 [Replay] 버튼으로 즉시 재생.

### 3.7.2 PC 오프라인 ablation
`src/Network/preproc_ablation.py` 는 동일 Android CSV 입력에 대해 4 조합 (baseline / -calib / +LPF / -norm) 의 PC dead-reckoning 결과를 단일 figure 에 비교. 단말 변경 없이 *입력 전처리 효과만* 격리한 학술 보고용.

### 3.7.3 Cross-platform 변환
`tools/ios_sensorlogger_to_replay.py` 는 iOS Sensor Logger 앱 출력 (4-파일 CSV) 을 Android replay 형식으로 변환:
- 컬럼 순서 (z/y/x) 보정
- Apple gravity 부호 반전 (`Android acc = iOS Accelerometer − iOS Gravity`)
- Orientation quaternion 직접 매핑 (모델이 yaw-invariant 라 reference frame 차이 무관)

OxIOD 학습 모델 입장에서는 iOS 데이터가 오히려 학습 분포에 가까워 cross-platform 일관성 검증에 유용.

## 3.8 설계 결정과 학술적 근거

| 결정 | 근거 |
|---|---|
| PDR-hybrid (RotVec heading + 모델 magnitude) | OoD 진단으로 모델 *방향* 신뢰도 낮음 확인 (P53/P54). RotVec 은 자력계 융합으로 drift-free heading 제공. |
| 적응형 스케일 piecewise linear | 단일 scalar (P56) 는 작은변위 과대 + 큰변위 과소 동시 발생. 임계 기반 분기로 비선형 완화. (P67/B/C) |
| 100Hz 리샘플 필수 | 모델이 100Hz 학습 → 250Hz native input 시 windowed time span 압축 (1s → 400ms) → magnitude × 0.4 왜곡 확인. |
| EKF 보조용 (PATH_B 주력) | 단말에서 EKF measurement 가 발산 (P52). PATH_B 의 *후처리 단순함* 이 OoD 도메인에서 더 강건. |
| 재생 (Replay) 인프라 | 단말 변경 시 동일 측정 즉시 재현 → A/B 비교 가능 (P45). PC ablation 도구의 단말 실측 동등 입력. |

## 3.9 한계와 향후 작업

### 한계
1. **누적 drift**: dead reckoning 의 본질적 한계. 시간 따라 적분 오차 누적, 외부 absolute 보정 부재.
2. **도메인 격차**: 학습 OxIOD/TLIO ≠ 배포 Samsung Galaxy. RoNIN fine-tuning 시도했으나 unseen 단말 전이 실패. PATH_B 의 PDR-hybrid 가 부분 완화.
3. **휴대모드 가정**: 학습은 7-class 다양 자세 라벨 보유, 본 시연은 handheld 자세 권장.
4. **좌표계 차이 시각 표시**: iOS 데이터의 NWU vs Android ENU world frame 차이로 지도 *절대 방향* 90° 회전 가능 (궤적 모양/거리 영향 없음).

### 향후
1. 외부 anchor (BLE beacon, Wi-Fi RTT, 영상 인식) 와 sensor fusion 으로 absolute 보정.
2. Map matching (실내 도면 알고 있을 때) 으로 경로 corner snap.
3. Samsung 자체 데이터 수집으로 도메인-specific fine-tuning.
4. EKF 의 measurement covariance 학습 (TLIO 형식) 으로 동적 신뢰도 가중.

## 3.10 기술 스택 요약

- **클라이언트**: Android (Kotlin, AndroidX, Material Components)
- **추론**: PyTorch Mobile (`org.pytorch:pytorch_android:1.13.x`)
- **지도**: Naver Maps Android SDK 3.23.2
- **EKF**: C++ (JNI), `EkfBridge.create/propagate/update/getPosition`
- **테스팅**: PC Python (offline_eval.py, preproc_ablation.py, ios_sensorlogger_to_replay.py)
- **개발**: Galaxy S23 FE (Android 14), 무선 ADB

## 3.11 디렉토리 구조 (참고)
```
imu_android/
├── android/                      # Android Studio 프로젝트
│   └── app/src/main/
│       ├── java/com/imulocal/
│       │   ├── ImuCollector.kt      # 센서 수집 + 전처리 + 캘리브
│       │   ├── InferenceEngine.kt   # PyTorch Mobile 추론
│       │   ├── LocalizationViewModel.kt  # PATH_B 적분 + EKF 병렬
│       │   ├── MainActivity.kt      # UI + Naver Map + 슬라이더
│       │   └── EkfBridge.kt         # C++ JNI 바인딩
│       └── res/                   # 레이아웃, 메뉴, drawable
├── src/Network/                  # PyTorch 학습/평가 코드
│   ├── train.py, model_resnet_small.py
│   ├── offline_eval.py           # 단말 CSV 오프라인 검증
│   └── preproc_ablation.py       # 전처리 ablation 도구
├── tools/                        # 운영 스크립트
│   ├── collect.ps1               # 단말 → PC CSV pull
│   ├── push_replay.ps1           # PC → 단말 latest.csv push
│   └── ios_sensorlogger_to_replay.py  # iOS → Android 변환
├── mobile_assets/                # 배포 .ptl 모델 + norm_params
└── docs/                         # HANDOFF, RATIONALE, 본 문서
```

---

*문서 작성: 2026-05-25.*
*전체 commit 이력: `git log --oneline android` 참조.*
