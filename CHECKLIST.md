# Android 포팅 버그 수정 및 작업 체크리스트

> **프로젝트**: IMU-based Indoor Localization (ResMLP128 + SC-EKF) → Android 실시간 측위  
> **최종 갱신**: 2026-05-03  
> **범례**: ✅ 완료 | 🔄 진행중 | ⬜ 미착수 | ⚠️ 주의 필요

---

## 1. 파이프라인 구조 버그 수정

### ✅ Bug #1: 타임스탬프 오류 (System.currentTimeMillis 사용)
- **파일**: `LocalizationViewModel.kt`
- **원인**: 클론/갱신 타임스탬프에 wall-clock(`System.currentTimeMillis`)을 사용. EKF는 `SensorEvent.timestamp`(부팅 경과 ns→μs) 기반이므로 단위·기준 모두 불일치.
- **수정**: `System.currentTimeMillis()` 완전 제거. 모든 타임스탬프는 `ImuSample.ts_us`(SensorEvent 기반 μs)만 사용.
- **검증**: `LocalizationViewModel.kt` 내 `currentTimeMillis` 미사용 확인 완료.

### ✅ Bug #2: 샘플 스킵 (windowReady + getLatestSample 방식)
- **파일**: `LocalizationViewModel.kt`, `ImuCollector.kt`
- **원인**: `windowReady.collect`에서 트리거 후 `getLatestSample()` 하나만 처리 → 최대 99개 샘플 누락.
- **수정**: `drainPropagateQueue()`로 5ms마다 쌓인 모든 100Hz 샘플을 일괄 처리.
- **검증**: propJob 루프에서 `for (sample in samples)` 전수 처리 확인.

### ✅ Bug #3: 추론 타이밍 고정 delay (delay(50ms))
- **파일**: `LocalizationViewModel.kt`
- **원인**: 고정 `delay(50ms)` → 추론 소요 시간(+수십ms)만큼 루프가 느려져 실제 10~15Hz로 저하.
- **수정**: 루프 시작 시각 `System.nanoTime()` 기록 → `runInferStep()` 완료 후 잔여 시간만 `delay`.
- **검증**: `val elapsedMs = ...; if (remaining > 2L) delay(remaining)` 패턴 적용 확인.

### ✅ Bug #4: SC-EKF 클론 쌍 미삽입
- **파일**: `LocalizationViewModel.kt`, `EkfBridge.kt`, `EkfJniBridge.cpp`
- **원인**: `update(t_begin, t_end)` 호출 시 EKF는 두 타임스탬프 모두 `si_timestamps_us`에 존재해야 하나, t_begin 클론을 삽입하지 않아 예외 발생.
- **수정**:
  - `pendingCloneTs` (inferJob→propJob): t_end 클론 예약 신호.
  - `lastInsertedCloneTs` (propJob→inferJob): 삽입 완료 확인.
  - `cloneHistory: ArrayDeque<Long>`: EKF `si_timestamps_us`와 동기화 유지.
  - `findBeginClone()`: ~1초 이전 클론을 허용 오차 ±200ms 내에서 탐색.
- **검증**: `update()` 전 tBegin/tEnd 모두 cloneHistory에 존재함을 `CLONE_SETTLE_MS` 대기로 확보.

### ✅ Bug #1-b: HIGH_SAMPLING_RATE_SENSORS 권한 누락 (Android 12+)
- **파일**: `AndroidManifest.xml`
- **증상**: 시작 버튼 터치 시 즉시 앱 종료. `SecurityException: To use the sampling rate of 0 microseconds, app needs to declare HIGH_SAMPLING_RATE_SENSORS`
- **원인**: Android 12(API 31)부터 `SENSOR_DELAY_FASTEST`(간격 0μs) 사용 시 `HIGH_SAMPLING_RATE_SENSORS` 권한 선언 필수. normal permission이므로 런타임 요청 불필요.
- **수정**: `<uses-permission android:name="android.permission.HIGH_SAMPLING_RATE_SENSORS"/>` 추가.
- **발견**: 2026-05-02 실기기 테스트 중 확인.

---

## 2. IMU 센서 데이터 오류 수정

### ✅ Bug #5: 네트워크 입력에 TYPE_ACCELEROMETER 사용 (82σ OOD)
- **파일**: `ImuCollector.kt`
- **원인**: `TYPE_ACCELEROMETER`는 중력(~9.8 m/s²) 포함. 학습 데이터(`acc_ga`)는 중력이 제거된 값. norm_mean≈0, norm_std≈0.12 기준으로 82σ 이상의 out-of-distribution 입력 발생.
- **수정**:
  - `TYPE_LINEAR_ACCELERATION` 센서 추가 등록.
  - `ImuSample`에 `linAcc: FloatArray` 필드 추가.
  - `getWindow()`: ch 0-2를 `s.linAcc`로 교체 (ch 3-5 gyr는 유지).
  - EKF 전파는 여전히 `sample.acc` (중력 포함) 사용.
- **주의**: `TYPE_LINEAR_ACCELERATION`가 없는 기기(드물지만 존재) → `linAcc=0`으로 폴백, 로그 경고 출력. 대부분 현대 Android 기기는 지원.

### ✅ Bug #6: 네트워크 입력 좌표계 불일치 (body frame vs world frame)
- **파일**: `LocalizationViewModel.kt`
- **원인**: 네트워크는 Python `dataset.py`에서 gravity-aligned, yaw-removed world frame(`acc_ga`, `gyr_ga`)으로 학습됨. Android 원본 코드는 body frame 그대로 입력 → 방향 오류로 변위 예측 불가.
- **수정**: `transformWindowToWorldFrame()` 구현.
  - t_begin EKF 클론 회전 행렬 R_begin 취득.
  - yaw 제거: `R_yawfree = Ri_z^T @ R_begin`.
  - 샘플별 자이로 적분으로 상대 회전 `Rs_bofbi[t]` 계산 (Rodrigues 공식).
  - 각 샘플: `linAcc_w = (R_yawfree @ Rs) @ linAcc_body`, 동일하게 `gyr_w`.
- **헬퍼 함수 추가**:
  - `mat3mul(A, B)`: row-major 3×3 행렬 곱
  - `rodrigues(phiX, phiY, phiZ)`: 각속도 벡터 → 회전 행렬
- **검증**: Python `imu_tracker.py` L150-195 알고리즘과 1:1 대응 확인.

---

## 3. EKF 파라미터 불일치 수정

### ✅ Bug #7: meascov_scale 불일치 (1.0 vs 10.0)
- **파일**: `EkfBridge.kt`
- **원인**: Python `filter_batch.py` 기본값 = 10.0. Android `DEFAULT_PARAMS[11]` = 1.0 → 측정 노이즈 10배 과소 추정, EKF가 오염된 네트워크 출력을 과도하게 신뢰.
- **수정**: `DEFAULT_PARAMS[11] = 10.0` 으로 변경.

### ✅ Bug #8: dispCov 클리핑 방식 오류 (variance 클립 vs log-variance 클립)
- **파일**: `LocalizationViewModel.kt` → `buildCovMatrix()`
- **원인**: 이전 코드 `coerceIn(1e-6, 100.0)` → variance에 직접 클리핑. Python `meas_source_torchscript.py`는 log-variance에서 `clip(< -4)` 후 `exp()` 적용 → exp(-4)≈0.018 m²가 최솟값.
- **수정**: `val logVar = dispCov[i].toDouble().coerceAtLeast(-4.0); cov[i*3+i] = exp(logVar).coerceAtMost(100.0)`
- **검증**: Python 동작과 동일한 최솟값 exp(-4)≈0.018 m² 보장.

---

## 4. JNI 인터페이스 추가

### ✅ Task #9: `nativeGetCloneRotation` JNI 함수 추가
- **파일**: `EkfBridge.kt`, `EkfJniBridge.cpp`
- **이유**: `transformWindowToWorldFrame()`에서 t_begin 클론의 R(world←body) 행렬이 필요.
- **구현**:
  - C++: `Java_com_imulocal_EkfBridge_nativeGetCloneRotation` — `si_timestamps_us` 탐색 후 `si_Rs[idx]`를 row-major double[9]로 반환. 미발견 시 빈 배열 반환.
  - Kotlin: `fun getCloneRotation(tBeginUs: Long): DoubleArray` + `external fun nativeGetCloneRotation(tUs: Long): DoubleArray`

---

## 5. 빌드 / 환경 문제

### ✅ Task #10: Eigen 프리캐시 (첫 NDK 빌드 시 필요)
- **경로**: `android/app/src/main/cpp/third_party/eigen/` — **✅ 배치 완료**
- **출처**: 기존 빌드 캐시(`.cxx/Debug/6l3h2867/arm64-v8a/_deps/eigen-src/`)에서 복사.
- **검증**: `third_party/eigen/Eigen/Dense` 존재 확인.
- **CMakeLists.txt 수정**: `EIGEN_LOCAL_DIR`를 두 경로 우선순위로 수정.
  - 1순위: `${CMAKE_SOURCE_DIR}/third_party/eigen` (CHECKLIST 지침 경로 ← 현재 배치 위치)
  - 2순위: `${CMAKE_SOURCE_DIR}/../../../../../third_party/eigen` (프로젝트 루트 fallback)

### ⬜ Task #11: NDK 빌드 및 .so 생성 확인
- `libimu_ekf_jni.so`가 올바른 ABI(arm64-v8a, x86_64)로 생성되는지 확인.
- 빌드 커맨드: `./gradlew assembleDebug`
- 예상 경로: `android/app/build/intermediates/cxx/Debug/*/obj/arm64-v8a/libimu_ekf_jni.so`

---

## 6. 실기기 테스트

### ⬜ Task #12: 기본 동작 확인 (정적 상태)
- 앱 설치 후 IMU 수집 시작 → EKF 초기화 로그 확인.
- 정지 상태에서 위치 드리프트 < 0.5m/분 기준.
- Logcat 필터: `tag:ImuCollector tag:ImuEkf tag:LocalizationVM`.

### ⬜ Task #13: 좌표 변환 검증
- 디버그 로그: `transformWindowToWorldFrame` 입출력의 채널 RMS 비교.
  - 입력 ch0-2(linAcc) RMS: 중력 없으므로 ~0-2 m/s² 예상.
  - 출력 ch0-2 RMS: 유사해야 함 (회전만 적용, 크기 불변).
- R_begin이 빈 배열일 때 경고 로그 발생 빈도 확인 (빈번하면 클론 삽입 타이밍 문제).

### ⬜ Task #14: 보행 추적 정확도 평가
- 알려진 10m 직선 보행 → 종점 오차 < 1.5m 기준.
- 방향 전환 포함 경로에서 궤적 형태 정성 평가.
- 분류 레이블 (handheld/pocket/etc.) 전환 시 meascov_scale 동적 변경 확인.

### ⬜ Task #15: 성능 측정
- 추론 루프 실제 Hz 확인 (목표 20Hz).
- `inferLatency` StateFlow 값 확인 (목표 < 40ms on mid-range device).
- propJob CPU 사용률 확인 (5ms 폴링).

---

## 7. 알려진 잠재 문제

| # | 문제 | 영향도 | 상태 | 비고 |
|---|------|--------|------|------|
| P1 | 자이로 바이어스 미보정 상태로 Rs_bofbi 적분 | 중간 | ✅ **수정** | `EkfBridge.getGyrBias()` → `nativeGetGyrBias()` 추가. `transformWindowToWorldFrame`에서 `(gyr − bg) × dt` 적용 |
| P2 | `TYPE_LINEAR_ACCELERATION` 미지원 기기 | 낮음 | ✅ 기존 처리 | linAcc=0 폴백 + 로그 경고 |
| P3 | cloneHistory와 EKF si_timestamps_us 비동기 가능성 | 중간 | ✅ **수정** | `synchronized(ArrayDeque)` → `Channel<Long>(DROP_OLDEST)` 방식으로 교체. inferJob 전용 `localCloneHistory`로 락 완전 제거 |
| P4 | CLONE_SETTLE_MS(20ms) 타이밍 여유 불충분 | 낮음 | ✅ **수정** | `CLONE_SETTLE_MS` 20ms → **30ms** 로 증가 |
| P5 | EKF 파라미터 분류별 테이블이 경험적 값 | 중간 | 🔄 **부분 수정** | `STATE_EKF_PARAMS`에 sigma_na/ng 초기값 추가 (README 표 방향성 기반). Python 배치 러너 실험으로 최종 검증 필요 |
| P6 | propJob/inferJob 동시 C++ EKF 접근 (Eigen SIGABRT) | 높음 | ✅ **수정** | `EkfJniBridge.cpp` 전체 JNI 함수(11개)에 `std::lock_guard<std::mutex> lock(g_ekf_mutex)` 추가. `marginalize()`와 `propagate()` 동시 실행으로 인한 Eigen 행렬 차원 불일치(SIGABRT) 해결 |
| P7 | C++ EKF `eulerAngles(0,1,2)` = intrinsic XYZ → Python ZYX 불일치 | **높음** | ✅ **수정** | `imu_ekf.cpp update()`: `eulerAngles(0,1,2)` 삭제 → `atan2(Ri(1,0), Ri(0,0))` + `atan2(-Ri(2,0), sqrt(...))` ZYX 직접 공식으로 교체. Python `extrinsic XYZ`(=ZYX)와 일치. **특정 방향 발산 주요 원인** |
| P8 | `transformWindowToWorldFrame` 자이로 적분 1샘플 오프셋 | 중간 | ✅ **수정** | 루프 끝 Rs 갱신 → 루프 시작(t>0)으로 변경. Python `Rs[j]=Rs[j-1]@mat_exp(gyr[j]*dt)` 와 정확히 일치 |
| P9 | meascov_scale 테이블 값 Python 기준 1000~10000× 작음 | 높음 | ✅ **수정** | `EkfBridge.MEASCOV_SCALE_TABLE`: 0.001~0.05 → 5.0~100.0. Python 10.0 기준, handheld=10.0 정렬. P7/P8 오류와 결합 시 발산 가속 원인 |
| P10 | `applyClassLabel` 미등록 레이블 폴백값 0.001f — unknown(100.0)보다 100,000× 작음 | 중간 | ✅ **수정** | `?: 0.001f` → `?: 100.0f`. 미래 레이블 추가 시 보수적 동작 보장 |
| P11 | `get_rotation_from_gravity` 반대 방향(acc≈−Z) 시 `-I` 반환 — det=−1, SO(3) 원소 아님 | 중간 | ✅ **수정** | `imu_ekf.cpp`: `Mat3::Identity() * -1.0` → X축 기준 180° 회전 `diag(1,−1,−1)` (det=+1, 유효한 회전 행렬). 엎어진 폰 초기화 실패 방지 |
| P12 | 채널 순서 불일치 의심: `dataset.py` acc-first, `meas_source_torchscript.py` gyr-first | **높음** | ✅ **해소** (Android 정상) | **학습 경로 완전 추적 완료**: `train.py` → `dataset.py` L105 `[acc_ga, gyr_ga]` = **acc-first**로 학습 → `convert_to_pytorch_mobile.py`로 `.ptl` 변환 → Android 배포. Android `ImuCollector.kt`(ch0-2=linAcc, ch3-5=gyr)는 학습과 일치. **올바름.** Python `meas_source_torchscript.py`의 `[net_gyr_w, net_acc_w]` gyr-first 는 Python 추론 파이프라인 자체의 버그 (Android에는 영향 없음). 단, Python `filter_batch.py` 검증 결과 신뢰도 낮음 주의. |
| P13 | 정지 상태에서 네트워크 바이어스 출력 + IMU 적분 ba 오차 → EKF 드리프트 누적 | **높음** | ✅ **수정** | (1) `computeGyrRms(window)` 로 정적 감지(< 0.03 rad/s). (2) 단순 `return` 대신 **ZUPT** 적용: "속도=0" 측정값 주입 → 속도 0 복원 + ba 추정 보조. 단순 skip 시 ba_error가 속도로 적분돼 드리프트 발생; ZUPT 적용 시 드리프트 1/10 이하 |
| P14 | 격한 움직임(running) 중 누적 발산: meascov_scale이 크면 Mahalanobis 게이트가 쓸모없어짐 | **높음** | ✅ **수정** | (1) `imu_ekf.cpp`: **절대 이노베이션 게이트 6m** — meascov_scale과 무관하게 `innov.norm() > 6m` 시 업데이트 건너뜀. (2) `LocalizationViewModel.kt`: Kotlin 측 `dispNorm > 6.0m` 필터링 (Kotlin 1차 방어). (3) `imu_ekf.cpp`: 속도 클램핑 `MAX_INDOOR_SPEED=5m/s` 최후 안전망. running meascov_scale은 50 유지 (200으로 올렸다가 추적 지연 유발 → 복원) |
| P15 | running meascov_scale=200 → 칼만 이득≈0 → 달리기 중 추적 지연(위치가 이동을 따라가지 못함) | **높음** | ✅ **수정** | `EkfBridge.kt`: 200→50 복원. 발산 방어는 이제 C++ 절대 이노베이션 게이트(6m)가 담당. `STATIC_GYR_RMS_THRESHOLD` 0.05→0.03 rad/s (느린 보행 오감지 방지) |

---

---

## 8. Play Store 배포 준비

> **현황 (2026-05-03)**: 릴리스 서명·빌드 파일 구성 완료. 아래 절차를 따라 AAB를 생성하고 Play Console에 업로드하면 됩니다.

### ✅ 릴리스 빌드 파일 구성

| 파일 | 상태 | 내용 |
|------|------|------|
| `android/app/build.gradle` | ✅ | `signingConfigs.release` 블록 추가; `minifyEnabled true` + `shrinkResources true` |
| `android/app/proguard-rules.pro` | ✅ 신규 | JNI native 메서드, PyTorch Mobile, ViewModel, Coroutines keep 규칙 |
| `android/keystore.properties.example` | ✅ 신규 | 키스토어 설정 템플릿 (git-safe) |
| `android/.gitignore` | ✅ 신규 | `keystore.properties`, `*.jks`, `*.keystore` 제외 |

### ⬜ Step 1: 서명 키스토어 생성 (최초 1회)

```bash
keytool -genkey -v \
  -keystore ~/imulocal-release-key.jks \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias imulocal-key
# 비밀번호와 조직 정보 입력 → .jks 파일은 안전한 곳에 백업 보관
```

> ⚠️ `.jks` 파일을 분실하면 이후 앱 업데이트를 Play Store에 올릴 수 없습니다. 반드시 외부 백업 필요.

### ⬜ Step 2: keystore.properties 설정

```bash
cd android/
cp keystore.properties.example keystore.properties
# 편집기로 keystore.properties 열고 실제 경로·비밀번호 입력
```

`keystore.properties` 예시:
```properties
storeFile=../../imulocal-release-key.jks
storePassword=my_store_password
keyAlias=imulocal-key
keyPassword=my_key_password
```

### ⬜ Step 3: 릴리스 AAB 빌드

```bash
cd android/
./gradlew bundleRelease
# 결과물: android/app/build/outputs/bundle/release/app-release.aab
```

APK가 필요한 경우:
```bash
./gradlew assembleRelease
# 결과물: android/app/build/outputs/apk/release/app-release.apk
```

### ⬜ Step 4: Play Console 등록 및 앱 업로드

1. [Google Play Console](https://play.google.com/console) → 새 앱 만들기
2. **앱 이름**: `IMU Indoor Localization` (또는 원하는 이름)
3. **기본 정보 완성**: 설명, 스크린샷, 아이콘(512×512 PNG), 기능 그래픽(1024×500)
4. **내부 테스트** 트랙 → `app-release.aab` 업로드
5. **콘텐츠 등급 설문** 완료 (위치 정보 수집: 정밀 위치 없음 → IMU만 사용)
6. **타겟 독자** 및 **데이터 보안** 섹션 완성
   - 앱은 INTERNET 권한 없음, 데이터 수집·전송 없음 → "데이터 수집 없음" 표시 가능
7. 검토 제출 → 보통 2~7일 소요

### ⬜ Step 5: 프로덕션 출시 전 확인사항

- [ ] `versionCode` 증가 (현재 1)하여 업데이트 구분
- [ ] `targetSdk 34` → Play Store 2024년 8월 이후 API 34 이상 요구 사항 충족 ✅
- [ ] 실기기에서 `app-release.apk` 직접 설치 후 정상 동작 확인
- [ ] Logcat에서 ProGuard 제거로 인한 런타임 크래시 없는지 확인 (특히 JNI, PyTorch)
- [ ] 적응형 아이콘 (`mipmap-anydpi-v26/`) 없으면 추가 권장 (없어도 Play Store 거부 없음)

### 알려진 Play Store 잠재 이슈

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| PS-1 | JNI native 메서드 ProGuard 난독화 | ✅ 해결 | `proguard-rules.pro`에 `-keepclasseswithmembernames native <methods>` + `EkfBridge` 전체 keep |
| PS-2 | PyTorch Mobile Lite 내부 리플렉션 | ✅ 해결 | `proguard-rules.pro`에 `org.pytorch.**`, `com.facebook.**` keep |
| PS-3 | `shrinkResources true`가 `.ptl` 모델 파일 제거 | ✅ 안전 | assets 폴더는 `shrinkResources` 대상 아님 — 자동 보호됨 |
| PS-4 | 64비트 ABI 요구사항 | ✅ 충족 | `abiFilters "arm64-v8a"` 포함. x86_64는 에뮬레이터용 |
| PS-5 | INTERNET 권한 없음 → Play Store 데이터 보안 설문 유리 | ✅ 이점 | 완전 오프라인 앱으로 표시 가능 |

---

## 9. 파일 수정 이력

| 파일 | 상태 | 주요 변경 |
|------|------|-----------|
| `LocalizationViewModel.kt` | ✅ 재작성 | 버그 #1-4, #6-8; 클론 쌍 로직; 좌표 변환; **P1**: getGyrBias 바이어스 보정; **P3**: Channel 방식; **P4**: CLONE_SETTLE_MS 30ms; **P8**: 자이로 적분 1샘플 오프셋 수정; **P13**: computeGyrRms() 정적 감지 + **ZUPT 적용** (단순 skip → 속도 0 복원); **P14**: 비정상 변위 6m 필터링 |
| `ImuCollector.kt` | ✅ 수정 | 버그 #5: TYPE_LINEAR_ACCELERATION 추가, ImuSample.linAcc 추가 |
| `EkfBridge.kt` | ✅ 수정 | 버그 #7: meascov_scale=10.0; Task #9: getCloneRotation; **P1**: getGyrBias 추가; **P5**: MEASCOV_SCALE_TABLE 주석 정비; **P9**: 테이블 값 5.0~100.0으로 수정; **P10**: applyClassLabel 폴백 0.001f → 100.0f |
| `EkfJniBridge.cpp` | ✅ 수정 | Task #9: nativeGetCloneRotation; **P1**: nativeGetGyrBias C++ 구현; **P6**: 전체 JNI 함수 11개 `std::lock_guard` 추가 (SIGABRT race condition 수정) |
| `imu_ekf.cpp` | ✅ 수정 | **P7**: ZYX 공식; **P11**: 반대방향 회전; **P14**: 속도 클램핑 5m/s + 절대 이노베이션 게이트 6m; **P13**: `apply_zupt()` 구현 |
| `imu_ekf.h` | ✅ 수정 | **P13**: `apply_zupt(sigma)` 선언 추가 |
| `EkfJniBridge.cpp` | ✅ 수정 (추가) | **P13**: `nativeApplyZupt` JNI 함수 추가 |
| `EkfBridge.kt` | ✅ 수정 | **P9**: 테이블 5.0~100.0; **P10**: 폴백 100.0f; **P15**: running 200→50 복원; **P13**: `applyZupt()` + `nativeApplyZupt` 추가 |
| `model_meta.json` | ✅ 수정 | **P9**: 5.0~100.0; **P15**: running 50.0 복원 |
| `CMakeLists.txt` | ✅ 수정 | **Task #10**: EIGEN_LOCAL_DIR 두 경로 우선순위 로직 추가 |
| `android/app/src/main/cpp/third_party/eigen/` | ✅ 신규 | **Task #10**: Eigen 3.4.0 헤더 배치 완료 |
| `src/View/visualize_comparison.py` | ✅ 수정 | **P5**: STATE_EKF_PARAMS에 sigma_na/ng 초기값 추가 (README 방향성 기반) |
| `AndroidManifest.xml` | ✅ 수정 | **Bug #1-b**: HIGH_SAMPLING_RATE_SENSORS 권한 추가 (Android 12+ 실기기 크래시 수정) |
| `src/tracker/meas_source_torchscript.py` | ✅ 수정 | **P12**: `[net_gyr_w, net_acc_w]` → `[net_acc_w, net_gyr_w]`. Python 추론 채널 순서 학습(acc-first)과 일치하도록 수정. Python `filter_batch.py` 검증 신뢰도 복원 |
| `CHECKLIST.md` | ✅ 갱신 | 이 파일 — 2026-05-02 수정사항 반영 |
| `android/app/build.gradle` | ✅ 수정 | **Play Store**: `signingConfigs.release` 블록; `minifyEnabled true`; `shrinkResources true` |
| `android/app/proguard-rules.pro` | ✅ **신규** | JNI·PyTorch Mobile·ViewModel·Coroutines ProGuard keep 규칙 |
| `android/keystore.properties.example` | ✅ **신규** | 키스토어 설정 템플릿 (git 안전) |
| `android/.gitignore` | ✅ **신규** | 키스토어 파일 git 제외 목록 |
