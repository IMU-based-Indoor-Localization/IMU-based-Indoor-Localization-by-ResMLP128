================================================================
 IMU-based Indoor Localization (Android demo)
================================================================

스마트폰 IMU + 1D-ResMLP128 모델을 사용한 실시간 실내 측위 데모
앱이다. 학습된 모델을 PyTorch Mobile Lite(.ptl) 로 단말에 올려
직접 추론하고 궤적을 화면에 그린다.

본 문서는 *현재 어플리케이션 구조* 와 *사용법* 만 기술한다.
변경 이력·이전 시도·연구 과정은 docs/HANDOFF_P56.md 에 별도 정리.

논문(thesis):
  "Context-Aware Adaptive EKF for IMU-based Indoor Localization"
   — multi-head 1D-ResMLP (변위 + 공분산 + 휴대모드 분류)
   — 분류 확률로 Q/R 을 soft-switching (Σ p_k · θ^(k))
   — Oxford Inertial Odometry Dataset(OxIOD) 으로 검증

데모 시연 자세:
  실제 단말 시연은 **HANDHELD(폰을 진행 방향으로 들고 보행)** 자세
  하나로 한정한다. iOS/Android 센서 도메인 갭과 RotVec heading
  사용 가정 때문에 다른 휴대 자세(handbag/pocket/trolley/running)
  는 데모 신뢰도 보장 범위 밖이다 — §8 한계 참고.


================================================================
 1. 배포 모델
================================================================

  파일      : android/app/src/main/assets/imu_model.ptl
  구조      : 1D-ResMLP128 (multi-head)
              ├ dispHead   : 1초 IMU 윈도우 → world-free local 변위(xy,z)
              ├ logVarHead : 변위 공분산 (대각, log)
              └ clsHead    : 7-class 휴대모드 분류
  학습      : OxIOD (Oxford Inertial Odometry Dataset, iPhone)
  입력      : 100 샘플 × 6채널 (가속도 3 + 자이로 3), 100Hz, gravity-aligned
  정규화    : assets/norm_mean.txt, assets/norm_std.txt
  분류 7-class (인덱스, src/Network/train_classifier.py LABEL_REMAP):
    0 handbag  1 handheld  2 pocket  3 running
    4 slow_walk  5 trolley  6 unknown


================================================================
 2. 센서 / IMU 입력
================================================================

  - TYPE_ACCELEROMETER         : 100Hz 리샘플(중력 포함)
  - TYPE_GYROSCOPE             : 100Hz 리샘플
  - TYPE_LINEAR_ACCELERATION   : 중력 제거 (참고용 / 가용 시)
  - TYPE_ROTATION_VECTOR       : per-sample 회전행렬 → 자세 융합

  요구 조건: 위 4 센서를 모두 지원하는 실기기. 에뮬레이터에서는
  IMU 데이터가 가짜이므로 측위 동작 검증 불가.


================================================================
 3. 앱 측위 파이프라인 (경로 B, USE_ROTVEC_DR=true 기본)
================================================================

  Android 단말에서는 EKF 경로(경로 A) 가 발산하는 도메인 갭이
  관측되어, 기본 동작은 **RotVec dead-reckoning + PDR-hybrid**
  경로(경로 B) 로 운영한다.

  [Sensor HW]  acc, gyr, linAcc, rotVec
       │
  [ImuCollector] 100Hz 리샘플, 윈도우 큐
   - 시작 2초 자동 영점 보정(자이로/가속도 bias 추정)
   - per-sample rotVec 회전행렬 저장
   - getRawWindow() : LPF 미적용 6채널 (학습 입력 분포 정합)
   - getRotMatWindow() : per-sample 회전행렬
       │
  [InferenceEngine] PyTorch Lite 추론 (20Hz 호출)
   - 입력  : gravity-aligned 100×6 (per-timestep rotVec 변환)
   - 출력  : dispLocal(xy,z) + dispLogVar + clsProb(7)
       │
  [DR 적분: LocalizationViewModel.runRotVecDrStep]
   - 모델 |disp_xy| → "1초 윈도우 변위" → 속도 = |disp|/winSec
   - heading      = 윈도우 중앙 rotVec yaw (PDR-hybrid)
   - 속도 스케일  = HANDHELD_SPEED_SCALE (P57 단일 스칼라, 균일 1.5×)
                    분류기 출력은 표시 전용 — 위치 계산에 미사용.
   - 회전 윈도우  : |Δyaw|>60° 이면 속도 × 0.3 감쇠 (멈춤 방지)
   - 정지 윈도우  : 속도 0
   - 20Hz dt 적분  v_ema = lerp(v, v_new, 0.25)
                  netPos += v_ema · dt
       │
  [UI: TrackView]
   - 측위 궤적(파랑) + 시작점(초록) — 단일 궤적
   - 합집합 bounding box 로 10m×10m 최소 표시 범위 자동 스케일


================================================================
 4. 주요 토글 (LocalizationViewModel.kt companion)
================================================================

  USE_ROTVEC_DR            기본 true   true=경로 B (DR),
                                       false=경로 A (논문 EKF, 단말 발산)
  USE_PDR_HEADING          기본 true   true=heading 은 rotVec,
                                       false=모델 disp 벡터
  TURN_YAW_THRESH_DEG      60.0        제자리 회전 판정 (deg/win)
  TURN_SPEED_ATTEN         0.3         회전 윈도우 속도 감쇠
  DR_VEL_EMA               0.25        속도 EMA 평활
  DR_TRACKPOINT_MIN_MOVE   0.1 m       trackPoint 추가 최소 이동
  HANDHELD_SPEED_SCALE     1.5         HANDHELD 단일 속도 스케일
                                       (P57: per-class 소프트 스위칭
                                        제거, 분류기 출력은 표시 전용)
  WARMUP_DURATION_MS       3000        캘리브 후 워밍업 (궤적 미표시)


================================================================
 5. 폴더 구조
================================================================

  IMU-based-Indoor-Localization-by-ResMLP128/
  ├── android/                         Android 어플리케이션
  │   └── app/src/main/
  │       ├── java/com/imulocal/       Kotlin (ViewModel, Collector,
  │       │                            InferenceEngine, TrackView,
  │       │                            MainActivity, RobustEkfTracker,
  │       │                            AbsoluteSensorNode 등)
  │       ├── cpp/                     C++ SC-EKF (경로 A 백엔드)
  │       ├── res/layout/              UI 레이아웃
  │       └── assets/                  imu_model.ptl, norm_*.txt,
  │                                    model_meta.json
  ├── src/                             Python 학습/평가 코드
  │   ├── Network/                     모델 정의 + 학습/오프라인 평가
  │   │   └── offline_eval.py          ★ 오프라인 진단 하니스
  │   ├── Trans/                       데이터셋 로더(OxIOD/TLIO)
  │   ├── tracker/                     Python SC-EKF (참조 구현)
  │   └── View/
  │       ├── visualize_comparison.py  EKF 시각화/평가
  │       └── analyze_latest_csv.py    Android CSV 구간 분석
  ├── mobile_assets/                   배포용 .ptl / norm_* / model_meta
  ├── docs/                            ★ 변경 이력 / 핸드오프 문서
  │   ├── HANDOFF_P56.md               P40 이후 현재(P56) 까지 진행 기록
  │   ├── HANDOFF_P46.md / P44 / P40 / P39
  │   └── …
  └── README.txt                       (본 문서)


================================================================
 6. 사전 요구사항
================================================================

  Android
  -------
    - Android Studio (Hedgehog 2023.1 이상 권장)
    - Android SDK Platform API 33 이상
    - Android NDK (cpp/ 가 빌드에 포함됨 — 경로 B 만 쓸 거면
                   추후 제거 가능)
    - JDK 17 이상 (Android Studio 번들 JBR 권장)
    - 실기기 (IMU + TYPE_LINEAR_ACCELERATION +
              TYPE_ROTATION_VECTOR 지원)

  Python (학습/평가 시)
  ---------------------
    - Python 3.10+, PyTorch 2.0+
    - numpy, scipy, matplotlib, pandas, tqdm, einops
    - OxIOD 데이터셋 (별도 다운로드)


================================================================
 7. 빌드 및 실행
================================================================

  7.1. 저장소 clone
       git clone <repo-url>
       cd IMU-based-Indoor-Localization-by-ResMLP128

  7.2. Android Studio 에서 프로젝트 열기
       File → Open → "android" 폴더 선택
       (루트가 아니라 "android" 서브폴더를 직접 여는 것이 중요)

       첫 Gradle Sync 가 끝날 때까지 대기.
       SDK / NDK 누락 알림이 뜨면 안내대로 설치.

  7.3. 어플리케이션 assets 확인
       android/app/src/main/assets/ 에 다음 4 개가 있어야 한다.
         imu_model.ptl, norm_mean.txt, norm_std.txt, model_meta.json
       없다면 (학습 결과 재변환):
         cd android
         python prepare_assets.py

  7.4. 실기기 연결 후 빌드 + 설치
       - 폰의 USB / 무선 디버깅 활성화
       - Android Studio 디바이스 드롭다운에서 폰 선택
       - Run (Shift+F10)


================================================================
 8. 어플리케이션 사용
================================================================

  화면 구성:
    - 상단     : 시작 / 정지 / 초기화 / 테스트(센서 로깅)
    - 중앙     : TrackView (측위 궤적)
    - 하단     : 현재 위치 / 속도 / 추론 latency
                (휴대 방식 분류기 출력은 메뉴 → IMU 센서 진단 화면에서 확인)

  기본 흐름:
    1) [시작]  → 캘리브레이션 카드 표시 (폰을 평평히, 가만히 두기)
              → 2 초 자동 영점 보정 → 워밍업 3 초 → 측위 시작
    2) 데모 자세 권장: **HANDHELD** (폰을 진행 방향으로 들고 보행)
       - 이유: §8.1 한계 항목 참고
    3) 5 m 직선 보행 → 정지 → 180° 회전 → 5 m 복귀 등 단순 패턴부터.
    4) [정지]  → 측위 중단, Logcat 에 진단 통계 출력
    5) [초기화] → 모든 상태/트랙/통계 리셋

  CSV 재생 (참고):
    ImuTestActivity 로 기록한 CSV(sensor,ts_ns,x,y,z,w) 를
    LocalizationViewModel.start(replayCsv = File(...)) 로 재생 가능.
    수집-측위 분리 검증에 사용.


  8.1. 알려진 한계 (정직)
  -----------------------
   (a) 휴대 자세 demo 범위: **HANDHELD only**
       - 경로 B 의 진행 방향은 rotVec heading(자력계 융합 절대 방위)
         이므로 "폰이 가리키는 방향 = 진행 방향" 인 handheld 자세에서만
         의미가 있다. pocket/handbag/trolley/running 은 부적합.
       - 분류기도 Android 도메인에서는 OoD(주로 unknown) 라 결과를
         시각화 외에 기능적으로 신뢰하지 않는다.

   (b) 모델 변위 *크기* 과소 (Android)
       - OxIOD(iPhone) 학습 → Samsung Android 단말에서 보행 |disp|
         ~30~40% 과소.
       - 현재 HANDHELD_SPEED_SCALE = 1.5× 단일 상수로 평균 보정 (P57).
       - per-class 차등 보정 필요 시: 휴대모드별 실측 데이터 확보 후
         P56 시점의 가중 스위칭 구조 복원(git history).
       - walk1/walk2 의 *비대칭 과소* 는 전역 스케일로 못 고친다
         (모델 per-window 편차).

   (c) 모델 변위 *방향* OoD (Android)
       - offline_eval.py GT 평가에서 모델 출력 방향이 Android 에선
         랜덤워크화. 경로 B 는 이를 rotVec heading 으로 대체한다.
       - 결과적으로 모델은 "보행 에너지(크기)" 만 사용하는 PDR
         형태로 동작.

   (d) 경로 A (논문 EKF) 는 단말에서 발산
       - C++ SC-EKF 자체는 빌드되어 있고 USE_ROTVEC_DR=false 로
         전환 시 활성화되나, 모델 출력의 Android 도메인 갭으로
         인해 yaw drift / position jump 가 누적된다.
       - 논문 검증은 OxIOD 환경에서 유효하며, 본 단말 데모와는
         별개의 트랙으로 본다.

   (e) RotVec(자력계) 의존성
       - 자기 환경이 나쁘면 heading 자체가 흔들린다. 시연 공간은
         철골/대형 가전/스피커 근접을 피한다.

   (f) 매우 느린 보행(가속도 진폭 < 0.2 m/s²) 은 정지로 오판될 수
       있다.


================================================================
 9. 오프라인 진단 도구 (Python)
================================================================

  9.1. 모델·전처리 격리 평가 (EKF 배제)
       src/Network/offline_eval.py
         - IMU 시퀀스 → 학습 동일 전처리 → 모델 → 윈도우별 disp 재구성
         - OxIOD 통과 시 GT RMSE 산출
         - --diagnose : Android CSV 의 단위/프레임 매트릭스 sweep
       사용 예:
         python src/Network/offline_eval.py \
           --model_dir outputs/out_classifier2 \
           --csv latest.csv

  9.2. Android CSV 구간 분석
       src/View/analyze_latest_csv.py
         - 정지/이동/회전 구간 자동 분할 + rotVec 누적 yaw / 구간별
           순변위 / 폐합 오차 정량.

  9.3. EKF 시각화 (논문 트랙)
       src/View/visualize_comparison.py
         - OxIOD 시퀀스에 대해 Python SC-EKF 추적 + 그래프

  9.4. EKF 계수 비교 (현재 단말 cfg vs TLIO 논문 cfg)
       src/Network/compare_tlio_ekf.py  (+ imu_ekf_py.py)
         - imu_ekf.cpp 식과 1:1 동등한 self-contained Python EKF.
         - 같은 IMU+모델 시퀀스를 두 cfg 변형에 동기 입력 → 두 궤적 비교.
         - TLIO 차이: σ_v 1.0→0.1, σ_ba 0.02→0.2, meascov_scale 1.0→10.0.
       사용 예:
         python src/Network/compare_tlio_ekf.py \
           --model_dir src/Network/out_classifier2 \
           --android latest.csv --plot logs/tlio_compare.png

  9.5. 단말에서 모드별 trackPoints 측정 → 외부 overlay (P60)
       앱: 메뉴 → "EKF 모드 (비교용)" → EKF_CURRENT 선택 → 시작 → 보행 →
            정지 → 메뉴 → 경로 내보내기 (track_EKF_CURRENT_<ts>.csv 저장)
       앱: 다시 EKF_TLIO 로 같은 경로 보행 → 두 번째 CSV 저장
       PC: adb pull ... → 두 CSV 를 한 그래프에 겹치기:
         python tools/overlay_tracks.py \
           track_EKF_CURRENT_*.csv  track_EKF_TLIO_*.csv \
           --out logs/ekf_mode_overlay.png


================================================================
 10. Logcat 으로 동작 검증
================================================================

  주요 태그: LocalizationVM, ImuCollector, InferenceEngine,
            EkfBridge, Controller

  cmd 한 번에 캡처:
    adb logcat -c
    adb logcat -v threadtime \
      "LocalizationVM:*" "ImuCollector:*" "InferenceEngine:*" \
      "EkfBridge:*" "Controller:*" "*:S" > logcat.txt

  기대 로그 흐름:
    측위 시작 [실시간|Replay(...)] — 워밍업 3000ms
    캘리브레이션 완료 (n=..., elapsed=2000ms)
      gyrBias    = [...] rad/s
      linAccBias = [...] m/s²
      accBias    = [...] m/s²
    [DR] tick=NN dt=0.05s |v|=X.XXX m/s speed=Y.YYY heading=ZZZ°
    [DR] turn-skip yawΔ=NN° → speed × 0.3
    [DR] static-window → v=0


================================================================
 11. 변경 이력
================================================================

  변경 이력 / 진행 기록 / 폐기된 시도 / RoNIN 트랙 / 도메인 갭
  결론 등은 다음 문서에 정리되어 있다.

    docs/HANDOFF_P56.md    P40 이후 ~ 현재(P56) 까지의 변경 기록
    docs/HANDOFF_P46.md    P40 ~ P46 핸드오프 스냅샷
    docs/HANDOFF_P44.md    P40 ~ P44 핸드오프 스냅샷
    docs/HANDOFF_P40.md    P40 시점 크래시 해결 기록
    docs/HANDOFF_P39.md    P22 ~ P39 핸드오프 스냅샷

  Git log:
    git log --oneline   (P40 이전 상세 커밋은 docs/HANDOFF_P39 참조)


================================================================
 12. 도움 받기
================================================================

  - 새 채팅 / 새 작업자 합류 시: docs/HANDOFF_P56.md 우선 정독
  - 코드 검색: IDE Find in Files (Ctrl+Shift+F)
              핵심 키워드: USE_ROTVEC_DR, runRotVecDrStep,
                          transformWindowRotVec,
                          HANDHELD_SPEED_SCALE, getRawWindow
