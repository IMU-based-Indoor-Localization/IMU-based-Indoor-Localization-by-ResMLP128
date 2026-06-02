# Application 발표/Q&A 대본 — IMU 기반 실내 측위 Android

> **용도**: 짧은 학술 발표 + 청중 질문 대응용. 암기/낭독 부담 적은 톤.
> **상세 논문 버전**: `APPLICATION_PAPER.md` 참조.

목차:
1. [짧은 발표 대본 (3~5분)](#1-짧은-발표-대본-35분)
2. [예상 질문 & 답변 (Q&A)](#2-예상-질문--답변-qa)

---

## 1. 짧은 발표 대본 (3-5분)

### 도입 (30초)
"안녕하십니까. 저는 스마트폰 IMU 만으로 GPS 가 닿지 않는 실내 공간에서 사용자의 보행 궤적을 실시간 추정하는 안드로이드 시스템을 개발했습니다.

핵심 아이디어는, 딥러닝 모델이 출력하는 *변위 크기* 와 스마트폰의 자세 융합 센서가 제공하는 *진행 방향* 을 결합하여, 두 측정의 강점을 동시에 활용하는 **PDR-Hybrid 구조** 입니다."

### 시스템 흐름 (1분)
"수집부터 표시까지 4단계로 동작합니다.

1. **수집**: 가속도계·자이로·자력계·중력 센서를 100Hz 로 동기화 수집하고, 시작 2초간 정적 영점 보정으로 센서 bias 를 제거합니다.
2. **추론**: 100샘플(1초) 윈도우를 회전 행렬로 *gravity-aligned* 프레임으로 변환한 뒤, PyTorch Mobile 로 단말에서 직접 추론합니다. 출력은 1초 동안의 2D 변위와 불확실도입니다.
3. **적분**: 모델의 변위 크기에 적응형 스케일을 적용하고, 회전 벡터 센서의 절대 방위와 결합해 20Hz 로 위치를 누적합니다. saturation, 정지·회전 구간은 별도 처리합니다.
4. **표시**: 누적된 위치를 위경도 오프셋으로 변환하여 Naver Map 폴리라인으로 실시간 렌더링합니다. 시작점은 long-press 로 임의 변경 가능합니다."

### 핵심 기여 (1분 30초)
"세 가지 설계 결정이 핵심입니다.

첫째, **PATH_B 라 부르는 RotVec 기반 dead-reckoning + PDR-hybrid 구조** 입니다. 기존 EKF 기반 융합은 모델 출력의 OoD 로 인해 단말에서 발산하는 문제가 있었습니다. 모델이 *크기* 는 신뢰 가능하나 *방향* 은 도메인 격차로 흔들린다는 점을 진단 한 뒤, 방향은 센서 융합으로, 크기만 모델에서 가져오는 구조로 단순화했습니다.

둘째, **적응형 스케일 (Adaptive Scale)** 입니다. OxIOD 학습 모델이 Galaxy 단말에서 systematic 으로 작게 출력하는 saturation 현상을 임계 기반 piecewise 선형 보정으로 완화했습니다. 작은 변위는 1.0배, 중간은 2.5배, 큰 변위는 5.0~7.0배로 비선형 보정하되, 0.7m 초과 이상치는 1.0배로 fallback 하는 안전망을 두었습니다.

셋째, **3단 검증 인프라** 입니다. PC 오프라인 ablation 도구로 전처리별 효과를 격리 비교하고, 단말에는 동일 입력으로 추론을 동시 실행하여 PATH_B 와 EKF 궤적을 병렬 시각화하며, 모든 측정은 CSV 로 기록하여 단말 재생 (replay) 으로 재현 가능합니다. iOS Sensor Logger 데이터도 동일 형식으로 변환되어 cross-platform 검증을 지원합니다."

### 시연 안내 (30초)
"화면에는 시작점에서 출발한 파란 실선이 PATH_B 궤적입니다. 슬라이더로 시점을 자유롭게 스크럽 하여 보행 경과를 추적할 수 있고, '전처리 OFF 궤적' 토글로 EKF 기반 보라색 비교 궤적을 동시 표시할 수 있습니다. 또한 동일 측정의 CSV 를 PC 에서 분석하여 학술적 재현성을 확보했습니다.

감사합니다. 질문 받겠습니다."

---

## 2. 예상 질문 & 답변 (Q&A)

> 자주 받을 질문 카테고리별 정리. 짧은 키답변 + 필요 시 깊이 확장.

### Q1. 모델은 어떤 것을 사용했나요?
**짧게**: ResNet 계열 1D 회귀 모델 (`SimplePoolingReg`) 을 OxIOD (iPhone) 데이터셋과 TLIO golden (Meta Aria 헤드셋) 데이터로 학습. 입력은 100×6 (gravity-aligned linAcc + gyr), 출력은 2D 변위 + 공분산 + 7-class 휴대모드 라벨.

**깊이**: RoNIN 데이터셋으로 Samsung 도메인 fine-tuning 도 시도했으나 unseen 단말 전이가 오히려 악화 (per-window err 0.48→0.27m, 과소 누적). 현재 배포는 원본 OxIOD 학습 모델 + PATH_B 의 P56 전역 스케일 보정으로 결정.

### Q2. EKF 대신 PDR-hybrid 를 선택한 이유는?
**짧게**: 단말에서 EKF measurement update 가 발산했기 때문입니다. 오프라인 하니스로 모델·EKF 를 분리 진단한 결과, 모델은 정상 동작하나 EKF 가 잘못된 방향 measurement 를 흡수하며 발산함을 확인했고, PDR-hybrid (회전 벡터 heading + 모델 크기) 가 가장 안정적이었습니다.

**깊이**: P52~P54 진단 세션에서 (1) EKF 발산은 measurement covariance 가 아닌 input 방향의 OoD 때문, (2) 모델 |xy| 만 사용 시 1m/window 수준 정상 회귀, (3) RotVec 은 자력계 융합으로 yaw drift 없음 — 세 가지 사실을 차례로 입증했습니다.

### Q3. 적응형 스케일은 어떻게 정했나요?
**짧게**: 학습 분포 (OxIOD/TLIO, 약 0.5-1.0m/window) 와 단말 실측 분포 (Galaxy, 약 0.15-0.4m/window) 의 정량 분석으로 임계 [0.15, 0.25, 0.40] m 와 배수 [1.0, 2.5, 5.0, 7.0] 를 결정했습니다.

**깊이**: 작은 정지 노이즈 (raw<0.15) 는 증폭하지 않고, 일반 보행 (0.15~0.40) 은 2.5~5.0배로 saturation 보정하며, raw>0.7 의 이상치 (모델 spike) 는 1.0배로 fallback. 추가로 최대 유효 속도 2.0 m/s clamp 로 발산 안전망. (P67 / P67-B / P67-C 단계적 도입)

### Q4. 좌표계는 어떻게 처리하나요?
**짧게**: 매 윈도우 시작 시점의 yaw 를 정규화한 *gravity-aligned* local 프레임으로 모델 입력을 변환합니다. 모델 출력 (local 프레임 변위) 은 yaw0 만큼 역회전하여 world 프레임에서 적분합니다. 이로 인해 모델은 절대 방위에 대해 invariant 합니다.

**깊이**: 안드로이드 `TYPE_ROTATION_VECTOR` 는 ENU (East-North-Up) world frame 을 제공하나, 모델은 매 window 시작 yaw 가 0 으로 정규화된 프레임에서 학습되었기 때문에 절대 방위 정의에 무관. iOS Sensor Logger 의 NWU frame 데이터도 동일하게 동작 (단 지도 위 *절대 방향* 은 ~90° 회전될 수 있음 — 시각 표시 외 영향 없음).

### Q5. 추론 latency 는?
**짧게**: 100Hz 입력 + 50ms 단위 추론 주기 (20Hz). 단일 inference 약 5-15ms. 백그라운드 코루틴 이라 UI 차단 없음.

### Q6. 배터리/리소스는?
**짧게**: PyTorch Mobile + JNI EKF 모두 CPU 만 사용 (GPU 미사용). 100Hz 센서 콜백 + 20Hz 추론 + 5ms EKF propagation 코루틴. 일반 시연 시 분당 약 2-3% 배터리.

### Q7. 실내 GPS 가 없을 때 시작점은 어떻게 정합니까?
**짧게**: Naver Map 의 사용자 long-press 로 임의 위치를 시작점 anchor 로 설정합니다. 기본값은 국민대 미래관 (37.6109, 126.9963) 이며, runtime 에 변경 가능합니다.

**깊이**: 시작점 변경 시 누적된 (x_m, y_m) 위치는 그대로 두고 `currentAnchor` 만 갱신 → `meterOffsetToLatLng` 가 새 anchor 기준으로 즉시 위경도 재계산. 시작점이 GPS 였다면 적분 누적 오차가 그대로 지도에 반영.

### Q8. iOS 데이터도 처리 가능한가요?
**짧게**: 네. `tools/ios_sensorlogger_to_replay.py` 컨버터로 iOS Sensor Logger 앱의 4-파일 CSV (Accelerometer/Gravity/Gyroscope/Orientation) 를 단일 Android replay 형식으로 변환할 수 있습니다.

**깊이**: 컬럼 순서 (z/y/x 역순), gravity 부호 (Apple 은 user accel 과 gravity 분리, Android 는 raw accel = gravity 포함) 를 자동 보정. 실제로 OxIOD 학습 모델이 iPhone 데이터에 더 적합 (도메인 일치).

### Q9. 정확도는 어느 정도인가요?
**짧게**: 5m 단방향 보행 시 1-2m 누적 drift 수준. 짧은 시연 (1분 내) 에 적합하며, 장시간 누적은 추가 보정 (절대 위치 anchor, GPS hybrid) 없이는 한계가 있습니다.

**깊이**: 본 시스템은 *상대 변위 누적* (dead reckoning) 이므로 적분 오차가 시간에 따라 누적 — 학술 시연용. 실용화 위해서는 (1) 외부 절대 위치 보정 (BLE beacon, Wi-Fi RTT), (2) map matching, (3) loop closure 등이 필요.

### Q10. 학술적 기여는?
**짧게**: 세 가지. (1) 모델 OoD 진단 → PDR-hybrid 구조 설계, (2) 적응형 스케일로 saturation 비선형 완화, (3) PC ablation + 단말 replay + cross-platform 변환 통합 검증 인프라.

---

*문서 작성: 2026-05-25.*
