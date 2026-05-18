# IMU-based Indoor Localization — 핸드오프 문서 (P39 시점, 양방향 EKF 복원 시험 + IMU CSV 추출 진입 직전)

이 문서는 새 Claude 채팅에서 **Android raw IMU CSV 추출 → OxIOD raw 와 정량 비교 → 옵션 B(iOS)/옵션 C(재학습) 결정** 으로 이어 진행하기 위한 단일 자료다. `HANDOFF_P38.md` 의 후속 스냅샷이며, **P38-A1 이후 → 양방향 P11 EKF 복원 시험 → 핵심 결론 도출** 까지의 흐름을 모두 반영한다.

> **새 채팅 시작 시**: 이 문서 + `HANDOFF_P38.md` 를 먼저 읽고, "P38 롤백 시도 결과 양방향 P11 EKF 가 정상 작동하지만 학습 모델-Android IMU mismatch 가 본질적 한계로 *재확정*되었습니다. 다음 단계로 Android raw IMU CSV 추출 + OxIOD raw 와 비교 분석을 진행할 시점입니다. `D:\EKF_DATASET` 에 OxIOD raw 위치." 로 시작하면 충분하다.

---

## 1. 한 줄 요약

P38-A1 (in-run bias EMA) 후 사용자가 P11 commit (8616232) 으로 롤백 시도 → **commit 자체가 손상된 스냅샷임이 발견** → `android` 브랜치 (P22) 의 cpp/JNI 자산을 가져와 양방향 EKF 복원 → ImuCollector 에 P21 캘리브레이션 이식 → LocalizationViewModel 의 손상 부분 복원 + DEBUG-2 (history span 게이팅) 적용 → **clone-based 1초 윈도우 update 정상 작동 확인** → 그러나 궤적 시험에서 *발산 + unknown 48%* → **iPhone/Android sensor mismatch 가 본질적 한계로 재확정**.

---

## 2. 현재 git 상태

| 항목 | 값 |
|---|---|
| 활성 브랜치 | `rollback-p11-with-p21-calib` |
| 부모 commit | `8616232` (P11 — 손상된 WIP 스냅샷) |
| 미커밋 변경 | 다수 (이번 세션의 모든 작업) |
| 다른 브랜치 보존 | `android` (P22, d58f4999), `p38-checkpoint` (P29~P38, 1c31a4d8), `EKF` |
| 추가 stash | 사용자가 이미 `git stash drop stash@{0}` 처리함 |

**중요**: 8616232 commit 의 핵심 파일 4개가 환경 이전 중 truncate 됨 — `imu_ekf.h` (322줄), `EkfBridge.kt` (232줄), `LocalizationViewModel.kt` (775줄 부근), `MainActivity.kt` (120줄). 모두 `android` 브랜치에서 복원해서 양방향 EKF 구조 재구성됨.

---

## 3. P38 → P39 변경 사항 (모든 작업 상세)

### 3.1 P11 롤백 시도 (실패) 및 결과

- `git checkout -b rollback-p11-with-p21-calib 8616232` — P11 시점으로 새 브랜치 생성
- 8616232 자체가 작업 환경 이전 중 *부분 저장* 된 손상 commit 임이 빌드 에러로 드러남
- HANDOFF P38 §12 의 검증 계획 (Section 12) 는 이 commit 자체가 깨졌으므로 *순수 P11* 검증 불가
- 대안: P11 commit 의 cpp + JNI 만 `android` 브랜치 (P22) 에서 복원 + Kotlin 계층 재작성 결정 (옵션 B)

### 3.2 양방향 EKF 인프라 복원 (사용자 git 명령)

```cmd
# cpp 전체 복원
git checkout android -- android/app/src/main/cpp/

# JNI 인터페이스
git checkout android -- android/app/src/main/java/com/imulocal/EkfBridge.kt

# InferenceEngine (truncate 가능성)
git checkout android -- android/app/src/main/java/com/imulocal/InferenceEngine.kt

# MainActivity (P22 의 단방향용 — 캘리브레이션 카드 UI 포함)
git checkout android -- android/app/src/main/java/com/imulocal/MainActivity.kt

# Layout (calibCard / calibProgress / tvCalibPercent)
git checkout android -- android/app/src/main/res/layout/activity_main.xml
```

→ C++ ScEkf (15-dim error-state + multi-state clone, `imu_ekf.{h,cpp}` 972 줄) + EkfBridge JNI 가 작동 가능 상태로 복원됨.

### 3.3 ImuCollector 에 P21 캘리브레이션 이식 (신규 작성)

`AbsoluteSensorNode` 가 없는 P11 시점이므로, P21 캘리브레이션 정신을 **`ImuCollector.kt` 에 직접 이식**:

| 추가 항목 | 위치 |
|---|---|
| `CALIBRATION_DURATION_MS = 2_000L` 상수 | companion |
| `calibrating` / `calibrationDone` / `calibProgress` state | 멤버 |
| `calibLinAccSum` / `calibGyrSum` / `linAccBias` / `gyrBias` | 멤버 |
| 캘리브레이션 분기 (sample 큐 미적재 + bias 평균) | `onSensorChanged()` 안 |
| linAcc + gyr bias 차감 (acc 는 제외 — EKF 자체 추정) | `onSensorChanged()` 끝부분 |
| `isCalibrating()` / `isCalibrationDone()` / `getCalibrationProgress()` / `getBiasSnapshot()` | public API |
| `performWarmup()` 함수 | private |

**설계 결정**: TYPE_ACCELEROMETER (중력 포함) 는 영점 보정 *제외* — 정지 자세에 따라 다르며 C++ ScEkf 가 자체 bias 추정 (전통적 EKF 동작 보존).

### 3.4 LocalizationViewModel 손상 복원

8616232 의 `LocalizationViewModel.kt` 가 line 775 부근에서 잘려 헬퍼 함수 5 개가 사라진 상태였음. PowerShell truncate (740줄까지 보존) 후 marker 를 헬퍼들로 교체:

| 추가 항목 | 의도 |
|---|---|
| `findBeginClone(tEnd, history)` | localCloneHistory 에서 1초 전 클론 탐색 (재추가 — 같이 잘렸음) |
| `buildCovMatrix(dispCov)` | log-variance → variance, MIN_MEAS_COV 클램프 |
| `computeGyrRms(window)` | 윈도우 ch 3-5 RMS (정지/이동 판정용) |
| `computeDynamicFraction(window)` | gyr norm > threshold 비율 (혼합 윈도우 차단) |
| `applyRotVecYaw(label)` | RotVec yaw → EKF.applyYawUpdate 주입 (정확도 ≥ MEDIUM 게이팅) |
| `transformWindowToWorldFrame(window, R_begin)` | body → gravity-aligned (`R_yaw_inv · R_begin`) |
| 클래스 닫기 `}` | |

또한 `LocalizationState` 에 `calibrating: Boolean` + `calibProgress: Float` 필드 추가, `start()` 안 EkfBridge.create() *직전* 에 캘리브레이션 진행률 polling 루프 추가.

### 3.5 PowerShell truncate 인코딩 손상 (미해결)

PowerShell `Set-Content -Encoding UTF8` 가 사실은 CP949 변환을 일으켜 **한국어 308 글자 손상**. 빌드 / 동작에는 무영향이지만 logcat 한국어가 깨짐. 다음 세션 우선순위 낮은 작업으로 보류.

복원 방법:
- `git diff HEAD -- LocalizationViewModel.kt` 로 차이 확인
- `p38-checkpoint` 의 동일 파일에서 *한국어 부분만* cherry-pick (구조가 달라서 부분 적용)
- 또는 Android Studio 에서 한 줄씩 수동 복원

### 3.6 DEBUG-1: 첫 빌드 후 빌드 통과 시험

**최초 빌드 통과 후 시험 결과**: 모든 `nativeUpdate` 가 `timestamp not found in past states` 로 실패. `nativeGetCloneRotation: ts=... not found (N=26)` 반복. EKF 측정값 반영 0 회 → propagate 적분만으로 1.5 초 만에 19m 발산.

**1차 시도 (DEBUG-1)**:
- `CLONE_SETTLE_MS` 30 → **100ms**
- `CLONE_MATCH_TOL_US` 200ms → **1초**
- propJob 의 클론 삽입 ts 로그를 Verbose → Info (`[CLONE-INSERT]`)
- inferJob 의 update 시도 시 ts 로그 추가 (`[UPDATE-TRY]`)

→ `nativeGetCloneRotation not found` 에러는 *완전히 사라짐*. **그러나 `histSize=2` 가 항상 유지** — 즉 100ms 윈도우 update 만 호출됨 (1초가 아님). 학습 모델 (1초 윈도우 학습) 출력이 0 근처로 폭락.

### 3.7 DEBUG-2: history span 게이팅 (핵심 fix)

**근본 원인 분석**:
- inferJob 매 50ms 마다 cloneChannel drain → 1 ts 추가
- update 호출 후 marginalize → 1 ts 제거
- 결과 history.size 가 1~2 로 정체 → tBegin ≈ tEnd - 100ms (1초가 아님)
- 학습 모델은 1초 윈도우 학습 → 100ms 윈도우 입력에 대해 dispLocal ≈ 0 출력 → EKF 거의 정지

**2차 시도 (DEBUG-2)**:
- `CLONE_MATCH_TOL_US` 1초 → **200ms 환원**
- 새 상수 `MIN_HISTORY_SPAN_US = 800_000L` 추가
- `findBeginClone` 호출 *직전* 에 게이팅:
  ```kotlin
  val oldestTs = localCloneHistory.firstOrNull()
  if (oldestTs == null || (tEnd - oldestTs) < MIN_HISTORY_SPAN_US) {
      Log.i(TAG, "[GATE-SPAN] history span ${...}us < ${MIN_HISTORY_SPAN_US}us — update skip (누적 대기)")
      return
  }
  ```

**효과 확인 (test_debug2.txt)**:
- 시작 직후 `[GATE-SPAN]` 8회 (0 → 729,369us) — 의도대로 누적 대기
- 약 0.85 초 후 정상 update 진입
- 이후 `histSize=8~9` 유지 (학습 모델과 매칭되는 ~1초 윈도우)
- `nativeGetCloneRotation not found` 에러 0회
- `nativeUpdate error` 0회

---

## 4. 시험 결과 — 핵심 결론

### 4.1 양방향 EKF 자체는 정상 작동 (HANDOFF §12 가설 부분 검증)

| 항목 | 결과 |
|---|---|
| 캘리브레이션 (2 초) | UI 정상, n≈2500 sample 누적, bias 적절 (acc < 0.04 m/s², gyr < 0.01 rad/s) |
| EKF clone 매칭 | 100% 성공 (`histSize=8~9`, ts mismatch 0회) |
| 정지 시 위치 안정성 | freezeStaticState 작동, 위치 0 근처 유지 |
| ZUPT (속도 > 3.0 m/s 시 강제) | 작동 |

### 4.2 그러나 궤적 발산 — **본질적 원인 = 모델/센서 mismatch**

**제자리 회전 + 5m 왕복 시험 결과** (시험 캡처):
- 시작점 → 위쪽 **5m 발산** (y=4.75m)
- σ = **3.46m** (불확실도 큼, EKF 가 자신 없어함)
- 휴대방식: **unknown 48%** ← 결정적 단서
- 회전 인식 못함, 왕복 궤적이 시작점으로 안 돌아옴

**진짜 원인 (EKF 가 아님)**:
- 양방향 EKF 가 *정확한 1초 측정값을 받아* 따라가지만, **그 측정값(dispLocal) 자체가 부정확**
- 분류기 unknown 48% = 학습 분포 (iPhone OxIOD) 와 Android IMU 의 형상 mismatch
- HANDOFF P38 §4 의 가설이 **재확인**

### 4.3 가설 검증 결과 정리

| 가설 | 결론 |
|---|---|
| Pre-P16 양방향 EKF 가 단방향 P38 보다 path tracking 좋다 (HANDOFF §12) | ✅ EKF 자체는 정상, 그러나 **추적 향상 미미** |
| 모델-센서 mismatch 가 본질적 한계 (HANDOFF §4) | ✅ **확정 — 양방향 EKF 로도 극복 불가** |
| Section 12 의 옵션 X (Stage 3 만 양방향 복원) | **효과 제한적** — 학습 데이터 / 모델 자체를 바꿔야 함 |

---

## 5. 다음 세션 — Android Raw IMU CSV 추출 + OxIOD 비교

### 5.1 plan 개요

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: ImuTestActivity 확장 (CSV 기록 기능)              │
│   - TYPE_ACCELEROMETER / LINEAR / GYR / ROTATION_VECTOR    │
│   - long-format CSV: sensor, ts_ns, x, y, z, w             │
│   - 위치: /sdcard/Android/data/com.imulocal/files/imu_csv/  │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Step 2: 측정 시나리오 3종 (각 1 CSV)                       │
│   ① 정지 60 초 — noise floor                               │
│   ② 직선 보행 30 초 — dynamic range                        │
│   ③ 360° 회전 30 초 — gyr response                         │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Step 3: adb pull 로 CSV 가져옴                              │
│   D:\SDK\platform-tools\adb.exe pull \                     │
│     /sdcard/Android/data/com.imulocal/files/imu_csv/ \     │
│     D:\imu_csv_android\                                    │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Step 4: Python 통계 비교 스크립트                          │
│   - Android CSV vs D:\EKF_DATASET (OxIOD raw, iPhone)      │
│   - 1초 윈도우 stat: mean/std/skew/kurtosis                │
│   - FFT power spectrum (50Hz 이하 noise spectrum)          │
│   - channel-wise 분포 histogram                            │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Step 5: 결과 따라 분기 결정                                │
│   acc std 4~6× LARGE (P29 측정 재확인)                     │
│     → 옵션 D (LPF 보정) 재검토                              │
│   noise floor 자체 다름 (LPF 로 해결 불가)                  │
│     → 옵션 B (iOS 재구축, 1~2주)                            │
│     → 옵션 C (Android 재학습, 3주~2개월)                    │
└────────────────────────────────────────────────────────────┘
```

### 5.2 ImuTestActivity 확장 설계 (다음 세션 첫 작업)

**파일**: `android/app/src/main/java/com/imulocal/ImuTestActivity.kt`

**추가 항목**:
| # | 변경 |
|---|---|
| 1 | 추가 센서 등록: `TYPE_LINEAR_ACCELERATION`, `TYPE_ROTATION_VECTOR` |
| 2 | 멤버: `csvWriter: BufferedWriter?`, `csvFile: File?`, `linAccVal`, `rotVecVal` |
| 3 | `startCollection()` 안에 CSV 파일 열기 (`getExternalFilesDir(null)/imu_csv/imu_record_<unix_ms>.csv`) |
| 4 | `stopCollection()` 안에 CSV flush + close + Toast 로 파일명 |
| 5 | `onSensorChanged()` 안 각 case 에 CSV 한 줄 쓰기 |
| 6 | UI 에 누적 라인 수 + 파일명 표시 (`tvSampleCount` 확장) |

**CSV 형식 예**:
```csv
sensor,ts_ns,x,y,z,w
acc,1234567890,0.012,-9.811,0.034,0.0
gyr,1234567892,0.0011,-0.0023,0.0008,0.0
linAcc,1234567893,0.011,-0.001,0.024,0.0
rotVec,1234567895,0.012,0.034,-0.067,0.997
```

`w` 칼럼은 rotVec quaternion 의 cos 성분만 채워짐 (다른 센서는 0).

### 5.3 OxIOD raw 위치

사용자 보고: **`D:\EKF_DATASET`**

다음 세션 첫 작업으로 그 디렉토리 구조 확인:
```cmd
dir D:\EKF_DATASET
```

그리고 `src/Trans/dataset.py` 의 raw 데이터 로딩 코드를 보고 정확한 파일 형식 (CSV / NumPy / HDF5) 과 컬럼 매핑 확인.

### 5.4 Python 비교 스크립트 (다음 세션 후반부)

위치: `src/View/imu_oxiod_vs_android.py` (신규 제안)

핵심 plot:
1. Per-channel violin plot (acc x/y/z, gyr x/y/z) — OxIOD vs Android
2. 1초 윈도우 std 분포 비교
3. FFT power spectrum (0-50 Hz) 비교
4. 동일 시나리오 (정지) 의 sample variance time-series

기대 결과 (HANDOFF P38 §4 측정 기준):
- acc std: Android **4.7~6.3× LARGE** 일 것
- gyr std: Android **0.5~0.6× SMALL** 일 것
- iPhone Core Motion 내부 필터링 효과가 정량적으로 보일 것

---

## 6. 적용된 안전장치 인벤토리 (P39 시점)

| 안전장치 | 상태 | 위치 |
|---|---|---|
| C++ ScEkf 15-dim error-state + clone | 활성 (P22 cpp 복원) | cpp/ekf/imu_ekf.{h,cpp} |
| P21 캘리브레이션 (2초 자동 영점, linAcc+gyr) | 활성 (ImuCollector 에 신규 이식) | ImuCollector.kt |
| EKF P5 freezeStaticState (정지 시 hard freeze) | 활성 (P22 복원) | LocalizationViewModel.kt, scekf.cpp |
| Hysteresis 정지/이동 판정 (STATIC/MOVING) | 활성 | LocalizationViewModel.kt |
| P10 워밍업 (3초간 trackPoints 표시 억제) | 활성 | LocalizationViewModel.kt |
| MIN_DYNAMIC_FRACTION (50%) 게이팅 | 활성 | LocalizationViewModel.kt |
| MAX_DISP_PER_WINDOW_M = 2.0 | 활성 | LocalizationViewModel.kt |
| 사후 속도 클램프 (MAX_POST_UPDATE_SPEED = 3.0) + ZUPT | 활성 | LocalizationViewModel.kt |
| Yaw drift 보정 (RotVec accuracy ≥ MEDIUM) | 활성 | LocalizationViewModel.kt |
| **DEBUG-2: MIN_HISTORY_SPAN_US = 800ms 게이팅** | **활성 (신규)** | LocalizationViewModel.kt |
| **DEBUG-1: CLONE_SETTLE_MS = 100ms** | **활성 (30 → 100ms)** | LocalizationViewModel.kt |
| **CLONE_MATCH_TOL_US = 200ms (DEBUG-2 환원)** | 활성 | LocalizationViewModel.kt |
| **[CLONE-INSERT] / [UPDATE-TRY] / [GATE-SPAN] 진단 로그** | 활성 (Info level) | LocalizationViewModel.kt |

---

## 7. 미해결 / 보류 사항

1. **`LocalizationViewModel.kt` 한국어 308 글자 손상** (Section 3.5)
   - PowerShell truncate 시 CP949 변환 이슈
   - 빌드/동작 무영향이지만 logcat 한국어 깨짐
   - 우선순위 낮음, 다음 세션 후반 또는 별도 작업

2. **8616232 commit 자체의 손상** — git 상으로 영원히 그 상태
   - 이 commit 의존 작업은 `rollback-p11-with-p21-calib` 브랜치에서 우회 완료

3. **HANDOFF P38 의 수정_이력_보고서.docx 갱신** (P30~P39 미반영)

---

## 8. 핵심 파일 인벤토리 (P39 시점)

### Android (현재 브랜치 `rollback-p11-with-p21-calib`)
| 파일 | 줄수 | 역할 | P38→P39 변경 |
|---|---:|---|---|
| `cpp/ekf/imu_ekf.h` | 353 | C++ ScEkf 헤더 | 복원 (android 브랜치에서) |
| `cpp/ekf/imu_ekf.cpp` | 972 | C++ ScEkf 본체 | 복원 (android 브랜치에서) |
| `cpp/EkfJniBridge.cpp` | ? | JNI 진입점 | 복원 |
| `EkfBridge.kt` | ? | JNI Kotlin 래퍼 | 복원 |
| `ImuCollector.kt` | ~415 | 100Hz IMU 수집 + P21 캘리브레이션 | **신규 캘리브레이션 이식** |
| `InferenceEngine.kt` | ? | PyTorch Mobile Lite 추론 | 복원 |
| `LocalizationViewModel.kt` | ~880 | propJob + inferJob + EKF 조율 | **헬퍼 5개 + LocalizationState 확장 + DEBUG-1/2** |
| `MainActivity.kt` | 131 | UI 진입점 + 캘리브레이션 카드 | 복원 (P22 의 단방향용) |
| `TrackView.kt` | ? | 궤적 시각화 | 손상 없음 |
| `ImuTestActivity.kt` | 199 | IMU 센서 진단 화면 | **다음 세션: CSV 기록 기능 추가** |
| `res/layout/activity_main.xml` | ? | 메인 레이아웃 (calibCard 포함) | 복원 |

### Python (`src/Trans/`, `src/View/`)
| 파일 | 역할 |
|---|---|
| `dataset.py` | OxIOD raw 로딩, `_window_to_gravity_aligned` 변환 (다음 세션에서 확인) |
| `classification_dataset.py` | LABEL_REMAP, CLASS_NAMES |
| (신규 예정) `src/View/imu_oxiod_vs_android.py` | Android CSV vs OxIOD raw 통계 비교 |

### 데이터
| 위치 | 내용 |
|---|---|
| `D:\EKF_DATASET` | OxIOD raw (iPhone 7+, Apple Core Motion) — 다음 세션 첫 작업으로 구조 확인 |
| `D:\imu_csv_android\` (예정) | Android CSV 시나리오 3종 |
| `D:\test_debug2.txt` | 이번 세션 시험 logcat (DEBUG-2 성공 증거) |

---

## 9. 사용자 환경 (HANDOFF P38 §8 그대로 + 추가)

- **한국어 응답** (코드 주석도 한국어). 요약 요청 시 **개조식**
- 큰 작업은 **TaskCreate / TaskUpdate** 로 추적
- git 커밋은 **사용자가 Windows cmd 에서 직접 실행**
- 빌드 실패 시 **첫 에러 줄 + Build 탭 출력** 알려주면 정확히 진단 가능
- 환경: Windows + `D:\SDK\platform-tools\adb.exe` + Wi-Fi 무선 디버깅 (Samsung 기기, 시리얼 R5CWC2B9J3D)
- **PowerShell 인코딩 주의**: `Set-Content -Encoding UTF8` 가 한국어 손상 일으킬 수 있음 → UTF-8 BOM 없는 명령 필요 시 `[System.IO.File]::WriteAllText` 권장
- `\.git\index.lock` 가 sandbox 권한 부족으로 stale 남을 수 있음 — `del .git\index.lock`

---

## 10. 새 채팅에서 시작 안내 멘트 예시

> "이 프로젝트는 IMU 기반 실내 측위 어플리케이션입니다 (`D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128`, `rollback-p11-with-p21-calib` 브랜치). HANDOFF_P39 에 따라 양방향 EKF (Pre-P16) 복원 시험을 마쳤고, **EKF clone matching 자체는 정상 작동 확인 (DEBUG-2 효과)** 했지만 학습 모델 / Android IMU mismatch 가 본질적 한계로 *재확정* 되었습니다. 다음 단계로 **Android raw IMU CSV 추출 + OxIOD raw 와 정량 비교** 진입할 차례입니다. OxIOD raw 는 `D:\EKF_DATASET` 에 있습니다. `HANDOFF_P39.md` + `HANDOFF_P38.md` 를 먼저 읽고, ImuTestActivity 에 CSV 기록 기능 확장부터 시작해주세요."

---

## 11. 빠른 인덱스 — 핵심 코드 위치 (P39 검색 키워드)

| 작업 | 파일 | 검색 키워드 |
|---|---|---|
| P21 캘리브레이션 (ImuCollector) | ImuCollector.kt | "[P21]", "CALIBRATION_DURATION_MS", "performWarmup" |
| DEBUG-1 CLONE_SETTLE_MS 100ms | LocalizationViewModel.kt | "[DEBUG-1]", "CLONE_SETTLE_MS" |
| DEBUG-2 history span 게이팅 | LocalizationViewModel.kt | "[DEBUG-2]", "MIN_HISTORY_SPAN_US", "[GATE-SPAN]" |
| 진단 로그 (clone insert / update try) | LocalizationViewModel.kt | "[CLONE-INSERT]", "[UPDATE-TRY]" |
| 헬퍼 함수들 (truncate 복원) | LocalizationViewModel.kt | "findBeginClone", "buildCovMatrix", "transformWindowToWorldFrame" |
| ImuTestActivity (다음 세션 확장 대상) | ImuTestActivity.kt | "startCollection", "onSensorChanged" |

---

**작성일**: 2026-05-16  
**마지막 적용 단계**: DEBUG-2 (history span 800ms 게이팅) — 양방향 EKF clone matching 정상 작동 확인  
**다음 작업**: ImuTestActivity 에 CSV 기록 기능 추가 → 시나리오 3종 측정 → OxIOD raw 와 정량 비교 → 옵션 B/C 결정
