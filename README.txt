================================================================
실제 실행 방법 및 검증 방법은 5~7
  - HANDOFF_P22.md          최신 상태 핸드오프 스냅샷 (P22 시점)
  - 수정_이력_보고서.docx     P9~P22 전체 수정 이력 (18 장, 726 단락)
  - PROJECT_CONTEXT.md       이전 컨텍스트 (직전 버전, 참고용)
================================================================
 1. 프로젝트 한 줄 요약
================================================================

스마트폰 IMU(가속도계+자이로+지자기)와 1D-ResNet 회귀 + 7-way
분류 모델, 자체 미니 EKF 를 결합해 실시간 실내 측위를 수행한다.
TLIO(MIT) 데이터셋 기반으로 학습한 모델을 PyTorch Mobile Lite
로 변환해 Android 어플리케이션에서 직접 추론한다.

핵심 설계 원칙: "단방향 (One-Way) 아키텍처"
  Stage 3 EKF 의 추정값을 Stage 1/2 입력 좌표계로 역류시키지
  않아 자세 추정 오차의 양방향 피드백 발산을 구조적으로 차단.


================================================================
 2. 현재 진행 상황 (P22 시점)
================================================================

  [완료]
  - P9~P15  : 양방향 피드백 발산 분석 및 부분 개선
  - P16~P20 : 단방향 4-Stage 재설계, 첫 실행 가능 빌드
  - P21     : 실시간 영점 자동 보정 (Auto-Calibration)
              앱 시작 2초간 정지 평균으로 bias 확정 후 차감
  - P22     : ZUPT (Zero-Velocity Update)
              Stage 3 자체 정지 감지 + propagate v=0 강제 +
              update skip → 정지 시 발산 차단
  [다음 단계]
  - P22 빌드 실기기 검증 (정지 30초 → 한 점 유지 확인)
  - 보행 검증 (직선 10m / 사각형 10x10 / 8자)
  - ZUPT 임계값 미세 튜닝
  - "stationary" 상태를 UI 에 노출 (선택)
================================================================
 3. 폴더 구조
================================================================

  IMU-based-Indoor-Localization-by-ResMLP128/
  +-- android/                       Android 어플리케이션 (작업 중심)
  |   +-- app/src/main/
  |       +-- java/com/imulocal/     Kotlin 소스 (Stage 1~3, ViewModel, MainActivity)
  |       +-- cpp/                   C++ SC-EKF (현재 미사용, 빌드만 됨)
  |       +-- res/layout/            UI 레이아웃 (activity_main.xml, activity_imu_test.xml)
  |       +-- assets/                imu_model.ptl, norm_mean.txt, norm_std.txt, model_meta.json
  |   +-- prepare_assets.py          학습 모델 → 어플리케이션 assets 변환 스크립트
  +-- src/                           Python 학습/평가 코드
  |   +-- Network/                   ResMLP 모델 정의 + 학습 스크립트
  |   +-- Trans/                     데이터셋 로더, 변환 (TLIO/Oxford)
  |   +-- tracker/                   Python SC-EKF (참조 구현)
  |   +-- View/                      평가/시각화 (visualize_comparison.py)
  |   +-- batch_runner/              일괄 실행 유틸
  |   +-- TLIO_Oxford_Dataset/       (큰 데이터셋 — gitignore 권장)
  +-- outputs/                       학습 결과 (체크포인트, norm_*.npy)
  +-- mobile_assets/                 모바일 배포용 .ptl / norm_params / model_meta
  +-- PROJECT_CONTEXT.md             이전 컨텍스트
  +-- 수정_이력_보고서.docx           ★ 전체 수정 이력
================================================================
 4. 단방향 아키텍처 한눈에 보기
================================================================
  [Sensor HW] acc, linAcc, gyr, rotVec
       v
  [Stage 1] AbsoluteSensorNode (EKF 존재 모름)
    - 시작 직후 2초 자동 영점 보정 (P21)
    - TYPE_ROTATION_VECTOR 절대 회전으로 world frame 변환
    - 출력: WorldSample
       v
  [Stage 2] StatelessInferenceNode (Pure Function, 이전 상태 모름)
    - 윈도우 시작 yaw 제거 -> yaw-free local frame
    - 1D-ResNet 추론 (dispLocal + dispLogVar + classProb)
       v
  [Stage 3] RobustEkfTracker (자기 상태 역류 안 함)
    - 자체 미니 EKF: p[3], v[3]
    - propagate: worldLinAcc 적분, F·Sigma·F^T + Q·dt
    - update: Innovation/Mahalanobis gate, Adaptive R
    - ZUPT (P22): 자체 정지 감지 -> v 강제 0 + update skip
       v
  [Controller] LocalizationViewModel (순수 데이터 라우터)
    - propJob(5ms) / inferJob(50ms) / uiJob(100ms)
       v
  MainActivity / TrackView (start/stop/reset/state)
================================================================
 5. 사전 요구사항
================================================================

  Android 작업
  ------------
    - Android Studio (Hedgehog 2023.1 이상 권장, 현재 작업환경: Panda 4 / 2025.3.4)
    - Android SDK Platform API 33 이상
    - Android NDK (cpp/ 가 빌드에 포함되어 있음 - 추후 제거 가능)
    - JDK 17 이상 (Android Studio 가 번들한 JBR 사용 권장)
    - 안드로이드 실기기 (IMU + TYPE_LINEAR_ACCELERATION + TYPE_ROTATION_VECTOR 지원)
      에뮬레이터는 IMU 데이터가 가짜이므로 측위 동작 검증 불가

  Python 학습/평가 작업 (필요 시)
  ------------------------------
    - Python 3.10+
    - PyTorch 2.0+ (CUDA 11.7+ 권장 / CPU 도 가능)
    - 주요 패키지: numpy, scipy, matplotlib, pandas, tqdm
    - TLIO / Oxford-IOD 데이터셋 (별도 다운로드 필요)


================================================================
 6. 빠른 시작 - Android 어플리케이션 빌드 및 실행
================================================================

  6.1. 저장소 clone
  -----------------
    git clone <repo-url>
    cd IMU-based-Indoor-Localization-by-ResMLP128

  6.2. Android Studio 에서 프로젝트 열기
  -------------------------------------
    File -> Open -> "android" 폴더 선택
    (루트가 아니라 "android" 서브폴더를 직접 여는 것이 중요)

    첫 Gradle Sync 가 끝날 때까지 대기 (5~10분 가능).
    SDK / NDK 누락 알림이 뜨면 안내대로 설치.

  6.3. 어플리케이션 assets 확인
  ----------------------------
    다음 4개 파일이 android/app/src/main/assets/ 에 있어야 한다.
      - imu_model.ptl       (학습된 모델 PyTorch Mobile Lite)
      - norm_mean.txt
      - norm_std.txt
      - model_meta.json

    없다면 (학습 결과 재변환이 필요한 경우):
      cd android
      python prepare_assets.py
    이 명령은 outputs/out_classifier2/ 의 학습 결과를 어플리케이션 assets 로 변환한다.

  6.4. 실기기 연결 후 빌드 + 설치
  -------------------------------
    - 폰의 USB 디버깅 활성화
    - USB 케이블 연결
    - Android Studio 의 디바이스 드롭다운에서 연결된 폰 선택
    - 초록색 Run 버튼 (Shift+F10) 클릭

  6.5. 어플리케이션 사용
  --------------------
    "시작" 버튼 -> 캘리브레이션 카드 표시 (폰을 평평히, 가만히 두기)
                -> 2초 후 카드 사라짐 -> 워밍업 3초 -> 측위 시작
    "정지" 버튼 -> 측위 중단, Logcat 에 진단 통계 출력
    "초기화" 버튼 -> 모든 상태 / 트랙 / 통계 리셋


================================================================
 7. Logcat 으로 동작 검증
================================================================

  주요 태그 필터: Stage1.AbsNode, Stage2.InferNode,
                  Stage3.EkfTracker, Controller

  cmd 에서 한 번에 캡처:
    adb logcat -c
    adb logcat -v threadtime "Stage1.AbsNode:*" "Stage2.InferNode:*" "Stage3.EkfTracker:*" "Controller:*" "*:S" > logcat.txt

  기대 로그 (P21 캘리브레이션)
  ----------------------------
    Stage 1 시작 (acc, gyr, rotVec 등록; linAcc=true) — 캘리브레이션 2000ms 진입
    RotVec 정확도: 3 (HIGH)
    캘리브레이션 완료 (n=2505, elapsed=2002ms)
      gyrBias    = [...] rad/s          (각 축 |값| < 0.01 정상)
      linAccBias = [...] m/s²            (각 축 |값| < 0.1 정상)
      accBias    = [...] m/s² (중력 제외) (각 축 |값| < 0.5 정상)

  기대 로그 (P22 ZUPT)
  --------------------
    ZUPT 상태 전이: stationary=true (aMean=0.0XX, gMean=0.0XX)
    측위 정지 — 진단: updates=NN rejInnov=NN rejMahal=NN
                     stillSkip=NN zupt=NN still=true|false
                     aMean=X.XXX gMean=X.XXXX
                     lastInnov=X.XXXm lastNSE=X.XX

    정지 30초 시: still=true, |v|=0 유지, trackPoints 한 점 근처.


================================================================
 8. Python 학습/평가 (선택)
================================================================

  학습/평가 코드를 만질 일이 없는 어플리케이션 작업자는 이 절을 건너뛰어도 된다.
  자세한 튜닝 파라미터는 src/View/visualize_comparison.py 와
  src/Network/train.py 의 헤더 주석을 직접 참고하라.

  8.1. 학습 데이터셋 준비
  ---------------------
    TLIO / Oxford-IOD 데이터셋을 다음 경로에 배치:
      src/TLIO_Oxford_Dataset/
        oxford_handheld_01/imu0_resampled.npy
        ...
        oxford_large_scale_06/imu0_resampled.npy

  8.2. 평가 (visualize_comparison) — 빠른 1회 실행
  ----------------------------------------------
    out_classifier2 가 어플리케이션 배포 모델 (use_classifier=true, input_len=100).

    python src/View/visualize_comparison.py \
      --data_path  src/TLIO_Oxford_Dataset/oxford_large_scale_6/imu0_resampled.npy \
      --model_path outputs/out_classifier2/checkpoints/best.pth \
      --norm_mean  outputs/out_classifier2/norm_mean.npy \
      --norm_std   outputs/out_classifier2/norm_std.npy

    추가 EKF 파라미터(--meascov_scale, --sigma_na, --sigma_ng, --init_vel_sigma 등)
    는 visualize_comparison.py 의 argparse 정의 또는 STATE_EKF_PARAMS 딕셔너리
    (파일 상단부) 를 참고해서 조정한다.

    [중요] Python EKF 파라미터는 GT 자세 환경 전제다.
    어플리케이션(자기계 의존 자세) 에 그대로 옮기지 말 것.
    어플리케이션 측 튜닝은 android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
    의 CLASS_R_SCALE / STILL_* 상수로 진행한다.

  8.3. 어플리케이션 assets 재변환
  -----------------------------
    학습 결과로부터 어플리케이션 assets (.ptl / .txt) 갱신:
      cd android
      python prepare_assets.py
    out_classifier2 의 norm_mean / norm_std / best.pth 를 자동으로 변환해
    app/src/main/assets/ 에 복사한다.


================================================================
 9. 핵심 파일 위치 (검색 키워드)
================================================================

  Stage 1 (Auto-Calibration P21)
    android/app/src/main/java/com/imulocal/AbsoluteSensorNode.kt
      - companion : CALIBRATION_DURATION_MS, STANDARD_GRAVITY
      - 상태       : calibrating, calibAccSum, accBias, gyrBias, linAccBias
      - 함수       : start(), onSensorChanged(), performWarmup()

  Stage 3 (ZUPT P22)
    android/app/src/main/java/com/imulocal/RobustEkfTracker.kt
      - companion : STILL_LIN_ACC_THRESHOLD = 0.20
                    STILL_GYR_THRESHOLD     = 0.05
                    STILL_WINDOW_SIZE       = 50
                    STILL_ENTER_HOLD_MS     = 500
                    STILL_EXIT_HOLD_MS      = 300
                    CLASS_R_SCALE = [15, 10, 5, 50, 5, 7, 100]
      - 상태       : isStationary, linAccNormBuf, zuptApplications,
                    stationaryUpdatesSkipped
      - 함수       : propagate(), update(), updateStationaryState()

  Controller
    android/app/src/main/java/com/imulocal/LocalizationViewModel.kt
      - LocalizationState : isRunning, position, posStd, velocity,
                           carryMode, carryProb, trackPoints,
                           inferLatency, calibrating, calibProgress, calibDone

  UI
    android/app/src/main/java/com/imulocal/MainActivity.kt
    android/app/src/main/res/layout/activity_main.xml
      - calibCard, calibProgress, tvCalibPercent


================================================================
 10. 알려진 한계 / 주의사항
================================================================

  - TYPE_ROTATION_VECTOR 정확도 (0~3) 가 0/1 이면 Stage 1 절대 회전이 부정확하다.
    LOW_ROTACC_INFLATE = 5.0 으로 어느 정도 방어하지만 자기/전자기 환경에 민감.
  - 매우 느린 보행 (가속도 진폭 < 0.20 m/s²) 시 ZUPT 가 잘못 진입할 수 있다.
    STILL_LIN_ACC_THRESHOLD / STILL_GYR_THRESHOLD 임계값을 실측 데이터로 튜닝하라.
  - 분류기의 noise(unknown) 클래스 학습 비중이 가장 커서 분포 외 입력이
    인접 클래스(trolley 등) 로 매핑되기 쉽다. P22 ZUPT 가 정지 구간은 차단했지만
    보행 중 분포 외 입력은 동일 메커니즘 가능 - CLASS_R_SCALE[6] = 100 으로
    보수적으로 두고 있다.
  - C++ SC-EKF (EkfBridge / cpp/) 는 컴파일에는 포함되지만 Stage 3 가 호출하지 않는다.
    NDK 빌드는 필요하다. 빌드 실패 시 EkfBridge.kt 의 init 블록을 try/catch 로
    감싸거나 cpp 빌드 자체를 끄는 것 검토.
  - TYPE_LINEAR_ACCELERATION 미지원 기기에서는 worldLinAcc 가 0 으로 머문다.
    AbsoluteSensorNode.start() 의 sLin null 체크 부분에 폴백 로직 추가가 추후 과제.


================================================================
 11. 작업 흐름 / 컨벤션
================================================================

  - 브랜치: android (어플리케이션 작업 중심)
  - 한국어 코드 주석 일관 유지
  - 새 변경마다 수정_이력_보고서.docx 에 한 절 (Heading 1) 누적
  - 큰 작업은 P 번호 부여 (현재 다음 번호는 P23)
  - 빌드 실패 시 첫 에러 줄 + Build 탭 출력 보존
  - 정기적으로 PROJECT_CONTEXT.md / HANDOFF_*.md 동기화


================================================================
 12. 도움 받기
================================================================

  - HANDOFF_P22.md 의 §17 "새 채팅 시작 시 안내 멘트" 참고
  - 수정_이력_보고서.docx 를 검색 가능하므로 P 번호로 빠르게 위치 찾기
  - 코드 검색은 IDE 의 Find in Files (Ctrl+Shift+F) - 위 §9 의 키워드 사용
