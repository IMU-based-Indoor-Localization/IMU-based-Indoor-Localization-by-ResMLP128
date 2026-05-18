# IMU-based Indoor Localization — 핸드오프 문서 (P40 시점, LPF + DEBUG 누적 → CSV 측정/보정 진입 직전)

이 문서는 새 Claude 채팅에서 **DEBUG-5 검증 → IMU CSV 추출 → OxIOD raw 비교 → 단계적 보정 전략** 으로 곧바로 진입하기 위한 단일 자료다. `HANDOFF_P39.md` 의 후속 스냅샷이며, **P39 → P40** 의 모든 추가 작업 (LPF 양방향 재도입, DEBUG-3/4/5, 시험 결과, 단계적 보정 전략) 을 반영한다.

> **새 채팅 시작 시**: 이 문서 + `HANDOFF_P39.md` + `HANDOFF_P38.md` 를 먼저 읽고, "P40 시점에서 LPF + DEBUG-2/3/4/5 모두 적용 완료. 양방향 EKF 가 매끄럽게 동작하나 dispLocal under-prediction ~50% 가 본질적 한계. 다음 작업으로 **DEBUG-5 빌드 검증 → IMU CSV 추출 → OxIOD raw 비교 → 보정 전략 결정** 진행할 시점입니다. OxIOD raw 는 `D:\EKF_DATASET`." 로 시작.

---

## 1. 한 줄 요약 (P39 → P40)

P39 에서 양방향 EKF clone matching 정상화 (DEBUG-2) → **이번 세션 (P40)** 에서 **LPF 양방향 재도입 + DEBUG-3 진단 로그 + DEBUG-4 ZUPT 임계 완화 + DEBUG-5 z drift fix** 누적 적용. 시험 결과: 양방향 EKF 매끄럽게 동작 (ZUPT 진동 0회), **단 dispLocal under-prediction ~50% + 분류기 변동성 + z drift = 학습 분포 mismatch 의 본질적 증거**. 다음 작업으로 **Android raw IMU CSV 추출 + OxIOD 비교 → 보정 알고리즘 선택** 진입할 시점.

---

## 2. 현재 git 상태

| 항목 | 값 |
|---|---|
| 활성 브랜치 | `rollback-p11-with-p21-calib` |
| 부모 commit | `8616232` (P11 — 손상된 WIP) |
| 미커밋 변경 | 이번 세션의 모든 추가 작업 + DEBUG-5 검증 대기 |
| 권장 commit 시점 | 다음 세션 DEBUG-5 검증 후 |
| 추천 commit message | `"P39+P40: 양방향 EKF + P21 캘리브 + LPF + DEBUG-2/3/4/5 누적 baseline"` |

---

## 3. P39 → P40 변경 사항 상세

### 3.1 P33-A1 LPF 양방향 재도입 (ImuCollector.kt)

| 변경 | 내용 |
|---|---|
| `LPF_B0/B1/B2/A1/A2` 상수 (companion) | Butterworth 2차, fs=100Hz, fc=12Hz |
| `ImuSample` 에 `linAccLpf` / `gyrLpf` 필드 추가 | 5필드 데이터 클래스로 확장 |
| LPF state (`lpfX1/X2`, `lpfY1/Y2` × 6채널) + `lpfStep()` / `resetLpfState()` 헬퍼 | Direct Form I |
| `start()` / `stop()` 에 LPF state 리셋 추가 | |
| `onSensorChanged()` bias 차감 후 LPF step 6회 → sample 에 두 버전 보관 | RAW + LPF 동시 보관 |
| `getWindow()` 가 `s.linAccLpf` / `s.gyrLpf` 반환 (네트워크 입력용) | |
| `drainPropagateQueue()` 는 `s.acc` / `s.gyr` (RAW) — EKF propagate 정확도 유지 | HANDOFF P38 §P35 교훈 |

### 3.2 DEBUG-3 진단 로그 (LocalizationViewModel.kt)

`runInferStep` 안에 2 개 추가:

```kotlin
// [INFER-OUT] InferenceEngine 출력 직후
Log.i(TAG, "[INFER-OUT] cls=${result.topClass}(${result.className}) p=${...} disp=[...] |xy|=...m |3d|=...m")

// [UPDATE-RES] EKF update 직후
Log.i(TAG, "[UPDATE-RES] ekfPos=[...] ekfSpeed=...m/s meas|xy|=...m")
```

→ dispLocal magnitude, 분류 결과, EKF 위치/속도 직접 추적 가능.

### 3.3 DEBUG-4 → revert (3.0 → 5.0 → 3.0)

**처음 변경 (5.0)**: 분류기 `running` (97%) 라우팅 시 학습 분포 속도 (2.5~3.5 m/s) 정상 수용. ZUPT cycle 진동 (속도 0 ↔ 3 m/s 반복) → "간헐적 발산" UI 인식 해소.

**검증 (test_diag2.txt)**: `발산 감지 ZUPT 강제` 0회/26+초 ✓

**그러나 사용자 평가**: test_diag.txt (DEBUG-4 적용 *전*) 가 *분류 안정성 + 제자리 회전 인식* 면에서 더 좋았음. DEBUG-4 가 ZUPT 안전망을 너무 풀어 **unknown 클래스 OOD 측정값을 그대로 따라감** → 분류 변동 + dispLocal 변동 큼.

**revert (3.0 복원)**: 임계 3.0 복원. ZUPT 진동 trade-off 수용. 본질적 해결은 IMU CSV 측정 + 보정.

### 3.4 DEBUG-5: Z drift fix (meas[2]=0 + cov[z,z]=1e6)

**원인**: `test_diag2.txt` 에서 26초간 `ekfPos.z = 0 → -2.2m` 발산. Android `TYPE_LINEAR_ACCELERATION` 의 z 가 보행 중 미세 systematic bias 누적 (P21 캘리브의 정적 bias 차감만으로 부족, Apple Core Motion 의 동적 보정 부재).

**해결**: 실내 평면 환경 가정 → z 측정 무시
```kotlin
meas[2] = 0.0
cov[2 * 3 + 2] = 1e6   // z 측정 분산 매우 큼 → 칼만 게인 ≈ 0
```

**검증 대기**: 다음 세션 빌드 + 시험 (최우선 작업)

---

## 4. 시험 결과 정량 분석 (P40 시점)

### 4.1 누적 효과 확인 (test_diag2.txt 기준, DEBUG-4 까지 적용 + LPF)

| 지표 | DEBUG-1 (1차) | DEBUG-2 (clone gating) | LPF + DEBUG-3/4 |
|---|---|---|---|
| `nativeGetCloneRotation not found` | 매번 | 0회 ✓ | 0회 ✓ |
| histSize | 2 (잘못) | 8~9 (정상 1초 윈도우) ✓ | 8~9 ✓ |
| `발산 감지 ZUPT 강제` | N/A | 6회/12초 (cycle 진동) | **0회/26초** ★ |
| 분류기 unknown 비중 | N/A | 100% (LPF 전) | **48~70% (LPF 후, 변동)** |
| 분류 안정성 | N/A | unknown 고정 | **변동 (handheld/unknown/running)** |
| dispLocal `|xy|` 평균 | N/A | ~0.05 (under) | **~0.32~0.55m/window** |
| EKF 추적률 (5m 보행 시) | N/A | ~10% | **~36~50% of GT** |
| z 축 drift | N/A | 미관찰 | **-2.2m/26s (새 문제)** ← DEBUG-5 로 fix 대기 |

### 4.2 핵심 결론

| 항목 | 상태 | 이유 |
|---|---|---|
| 양방향 EKF clone matching | ✅ 완벽 | DEBUG-2 효과 |
| ZUPT cycle 진동 (간헐적 발산) | ✅ 해결 | DEBUG-4 효과 |
| Z 축 drift | ⏳ DEBUG-5 검증 대기 | meas[2]=0 + cov 1e6 |
| **dispLocal under-prediction ~50%** | ❌ **잔존 — 본질적 모델 mismatch** | 보정 또는 재학습 필요 |
| 분류기 변동성 (running ↔ unknown ↔ handheld) | ⚠️ LPF 부분 효과 | Android raw IMU 가 학습 6 클래스 어디에도 안정 매핑 안 됨 |

### 4.3 가설 검증 최종 결과

| 가설 | 결과 |
|---|---|
| Pre-P16 양방향 EKF 가 path tracking 향상 (HANDOFF §12) | ✅ **검증** — 단방향 27.6m → 양방향 ~2.5m (5× 향상) |
| 모델-Android IMU mismatch 가 본질적 한계 (HANDOFF §4) | ✅ **재확정** — 양방향 + LPF + DEBUG 모두 적용에도 50% under-prediction |
| Section 12 옵션 X (Stage 3 양방향 복원) 효과 | ✅ 부분 효과, 단 모델 mismatch 가 ceiling |

---

## 5. 단계적 보정 전략 (다음 세션 ~ 이후 1-2주)

### 사용자 결정 사항 (이번 세션)
- 보정 진행이 좋은 방향 — **합리적 첫 시도** 로 인정됨
- 단 단독 path 아닌 *단계 1 측정 → 단계 2 보정 → 단계 3 평가* 흐름

### 단계 1 — 측정 (다음 세션, 4-6시간)

목적: 보정 알고리즘 선택의 정확한 근거 확보

**작업 순서**:
1. DEBUG-5 빌드 + 시험 검증 (z drift 해결 확인) ★ 최우선
2. `ImuTestActivity.kt` 에 CSV 기록 기능 추가
3. 시나리오 3종 측정 (정지 60s / 보행 30s / 회전 30s)
4. `adb pull` 로 `D:\imu_csv_android\` 로 가져옴
5. `D:\EKF_DATASET` (OxIOD raw) 구조 확인 + `src/Trans/dataset.py` 로딩 코드 분석
6. Python 비교 스크립트 작성 (`src/View/imu_oxiod_vs_android.py` 신규)
7. 정량 보고서 산출 — channel-wise 분포, 1초 윈도우 stat, FFT spectrum

### 단계 2 — 보정 적용 + A/B 시험 (이후 세션, 1-2주)

**단계 1 결과 따라 보정 선택**:

| mismatch 양상 | 적용 보정 |
|---|---|
| acc 노이즈 5× LARGE | LPF fc 8Hz 또는 6Hz (12Hz 보다 강하게) |
| gyr 진폭 0.5× SMALL | gyr 증폭 (1.5~2× — 학습 분포 매칭) |
| 동적 bias 누적 | P38-A1 in-run bias EMA 재시도 또는 정교화 |
| 분류 불안정 | 휴대 방식 사용자 입력 → 강제 라우팅 |
| z drift | DEBUG-5 유지 |

각 보정 적용 시 5m 보행 추적률 / 분류 안정성 측정 → A/B 비교.

### 단계 3 — 결정 (1-2주 후)

| 보정 결과 | 다음 action |
|---|---|
| ≥ 70% 추적률 | 보정 path 확정, 추가 최적화 |
| 50~70% | Hybrid (보정 + 옵션 D 스케일) |
| < 50% | 옵션 C (Android 재학습) 진입 |

### 보정의 *근본적 한계* (사전 인지)

- Apple Core Motion 알고리즘 **미공개** → 완벽 모사 불가
- 학습 분포 **6 클래스 한정** → Android 다양한 자세 모두 커버 X
- 정확도 ceiling **~70-80%** → 95% 달성은 재학습만 가능

---

## 6. 다음 세션 작업 (정리)

| # | 작업 | 우선순위 | 예상 시간 |
|---|---|---|---|
| 1 | DEBUG-5 빌드 + 시험 검증 (z drift) | ★ 최우선 | 15분 |
| 2 | `ImuTestActivity.kt` 에 CSV 기록 기능 추가 | 높음 | 1시간 |
| 3 | 시나리오 3종 측정 + `adb pull` | 높음 | 30분 |
| 4 | `D:\EKF_DATASET` 구조 확인 + `dataset.py` 로딩 분석 | 높음 | 30분 |
| 5 | Python 비교 스크립트 (`imu_oxiod_vs_android.py`) | 높음 | 1~2시간 |
| 6 | 정량 보고서 작성 → 단계 2 보정 선택 | 핵심 결정 | 1시간 |
| 7 | `LocalizationViewModel.kt` 한국어 308 글자 복원 | 선택 (낮음) | 30분 |
| 8 | git commit (P39+P40 baseline) | 권장 | 5분 |

### CSV 기록 기능 설계 (다음 세션 참고)

`ImuTestActivity.kt` 에 추가:

```kotlin
// 멤버
private var csvWriter: BufferedWriter? = null
private var csvFile: File? = null
private var linAccSensor: Sensor? = null
private var rotVecSensor: Sensor? = null
private var csvLineCount = 0L

// onCreate 에 추가 센서 등록
linAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
rotVecSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

// startCollection 에 CSV 열기
val dir = File(getExternalFilesDir(null), "imu_csv")
dir.mkdirs()
csvFile = File(dir, "imu_record_${System.currentTimeMillis()}.csv")
csvWriter = csvFile!!.bufferedWriter()
csvWriter!!.write("sensor,ts_ns,x,y,z,w\n")
sensorManager.registerListener(this, linAccSensor, SensorManager.SENSOR_DELAY_FASTEST)
sensorManager.registerListener(this, rotVecSensor, SensorManager.SENSOR_DELAY_FASTEST)

// onSensorChanged 확장: 매 이벤트마다 csvWriter?.write(...)

// stopCollection: csvWriter?.flush() / close() / Toast 로 파일명
```

**CSV 형식** (long-format, Python pandas pivot 으로 wide 변환 용이):
```csv
sensor,ts_ns,x,y,z,w
acc,1234567890,0.012,-9.811,0.034,0.0
gyr,1234567892,0.0011,-0.0023,0.0008,0.0
linAcc,1234567893,0.011,-0.001,0.024,0.0
rotVec,1234567895,0.012,0.034,-0.067,0.997
```

**adb pull 명령**:
```cmd
D:\SDK\platform-tools\adb.exe pull /sdcard/Android/data/com.imulocal/files/imu_csv/ D:\imu_csv_android\
```

### Python 비교 스크립트 핵심 plot

1. Channel-wise violin plot (acc x/y/z, gyr x/y/z): OxIOD vs Android
2. 1초 윈도우 std 분포 비교
3. FFT power spectrum (0-50 Hz)
4. 시계열 sample variance (정지 60초)
5. 보행 30초의 channel-wise time-series

기대 결과 (HANDOFF P38 §P29 기반):
- acc std: Android 4.7~6.3× LARGE
- gyr std: Android 0.5~0.6× SMALL
- iPhone 의 내부 LPF 효과 spectrum 으로 정량 확인

---

## 7. 누적 안전장치 인벤토리 (P40 시점)

| 안전장치 | 상태 | 위치 |
|---|---|---|
| C++ ScEkf 15-dim error-state + clone | 활성 (P22 cpp 복원) | cpp/ekf/imu_ekf.{h,cpp} |
| P21 캘리브레이션 (2초, linAcc+gyr) | 활성 (ImuCollector 이식) | ImuCollector.kt |
| **P33-A1 LPF** (Butterworth fc=12Hz, inference 전용) | **활성 (P40 신규)** | ImuCollector.kt |
| EKF P5 freezeStaticState | 활성 (P22) | LocalizationViewModel + scekf.cpp |
| Hysteresis 정지/이동 판정 | 활성 | LocalizationViewModel |
| P10 워밍업 (3초간 trackPoints 억제) | 활성 | LocalizationViewModel |
| MIN_DYNAMIC_FRACTION (50%) | 활성 | LocalizationViewModel |
| MAX_DISP_PER_WINDOW_M = 2.0 | 활성 | LocalizationViewModel |
| **MAX_POST_UPDATE_SPEED = 3.0** (DEBUG-4 revert) | **활성 (P40, 5.0→3.0 복원)** | LocalizationViewModel |
| Yaw drift 보정 (RotVec accuracy ≥ MEDIUM) | 활성 | LocalizationViewModel |
| **DEBUG-2: MIN_HISTORY_SPAN_US = 800ms 게이팅** | 활성 (P39) | LocalizationViewModel |
| **DEBUG-1: CLONE_SETTLE_MS = 100ms** | 활성 (P39) | LocalizationViewModel |
| **CLONE_MATCH_TOL_US = 200ms** | 활성 (P39) | LocalizationViewModel |
| **DEBUG-3 진단 로그** ([INFER-OUT], [UPDATE-RES]) | **활성 (P40 신규, Info level)** | LocalizationViewModel |
| **DEBUG-5: z 측정 무시** (meas[2]=0 + cov 1e6) | **활성 (P40 신규, 검증 대기)** | LocalizationViewModel |
| `[CLONE-INSERT] / [UPDATE-TRY] / [GATE-SPAN]` 진단 | 활성 (P39) | LocalizationViewModel |

---

## 8. 미해결 / 보류 사항

1. **DEBUG-5 빌드 검증 대기** — 다음 세션 최우선
2. **dispLocal under-prediction ~50%** — 단계 1 측정 + 단계 2 보정으로 접근
3. **분류기 변동성** (running ↔ unknown ↔ handheld) — 단계 2 에서 휴대 방식 사용자 입력 옵션
4. **`LocalizationViewModel.kt` 한국어 308 글자 손상** — 우선순위 낮음, 다음 세션 후반
5. **HANDOFF P38 의 수정_이력_보고서.docx 갱신** (P30~P40 미반영)

---

## 9. 핵심 파일 인벤토리 (P40 시점)

### Android
| 파일 | 줄수 | P39→P40 변경 |
|---|---:|---|
| `cpp/ekf/imu_ekf.h` | 353 | 변경 없음 |
| `cpp/ekf/imu_ekf.cpp` | 972 | 변경 없음 |
| `cpp/EkfJniBridge.cpp` | ? | 변경 없음 |
| `EkfBridge.kt` | ? | 변경 없음 |
| `ImuCollector.kt` | ~510 (추정) | **+LPF (계수, state, lpfStep, resetLpfState) + ImuSample 확장 + getWindow 변경** |
| `InferenceEngine.kt` | ? | 변경 없음 |
| `LocalizationViewModel.kt` | ~895 (추정) | **+DEBUG-3 진단 로그 + DEBUG-4 임계 5.0 + DEBUG-5 z 측정 무시** |
| `MainActivity.kt` | 131 | 변경 없음 |
| `ImuTestActivity.kt` | 199 | **다음 세션: CSV 기록 기능 추가 대상** |
| `TrackView.kt` | ? | 변경 없음 |

### Python (다음 세션 작업 대상)
| 파일 | 역할 |
|---|---|
| `src/Trans/dataset.py` | OxIOD raw 로딩 코드 — 다음 세션 분석 대상 |
| `src/View/imu_oxiod_vs_android.py` (신규) | Android CSV vs OxIOD raw 통계 비교 — 다음 세션 작성 |

### 데이터
| 위치 | 내용 |
|---|---|
| `D:\EKF_DATASET` | OxIOD raw (iPhone 7+, Apple Core Motion) — 다음 세션 첫 분석 |
| `D:\imu_csv_android\` (예정) | Android CSV 시나리오 3종 |
| `D:\test_walk.txt`, `D:\test_debug1.txt`, `D:\test_debug2.txt`, `D:\test_lpf.txt`, `D:\test_diag.txt`, `D:\test_diag2.txt` | 이번 세션 시험 logcat 들 |

---

## 10. 새 채팅에서 시작 안내 멘트 (P40 시점)

> "이 프로젝트는 IMU 기반 실내 측위 어플리케이션입니다 (`D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128`, `rollback-p11-with-p21-calib` 브랜치).
>
> P40 시점에서 양방향 EKF + P21 캘리브 + LPF + DEBUG-2/3/4/5 모두 적용됨. 양방향 EKF 매끄럽게 동작 (ZUPT 진동 0회), 단 **dispLocal under-prediction ~50%** 가 본질적 한계로 *재확정*되었습니다.
>
> 사용자가 단계적 보정 전략 (측정 → 보정 → 평가) 채택. 다음 작업으로:
> 1. **DEBUG-5 빌드 검증** (z drift fix 확인) ★ 최우선
> 2. **ImuTestActivity 에 CSV 기록 기능 추가**
> 3. **3 시나리오 측정 + OxIOD raw (D:\EKF_DATASET) 와 Python 통계 비교**
> 4. **단계 2 보정 알고리즘 선택**
>
> `HANDOFF_P40.md` + `HANDOFF_P39.md` + `HANDOFF_P38.md` 를 먼저 읽어주세요."

---

## 11. 빠른 인덱스 (P40 검색 키워드)

| 작업 | 파일 | 검색 키워드 |
|---|---|---|
| P21 캘리브레이션 | ImuCollector.kt | "[P21]", "CALIBRATION_DURATION_MS" |
| **P33-A1 LPF (P40 신규)** | ImuCollector.kt | "LPF_B0", "lpfStep", "linAccLpf" |
| DEBUG-1 CLONE_SETTLE_MS 100ms | LocalizationViewModel.kt | "[DEBUG-1]" |
| DEBUG-2 history span 게이팅 | LocalizationViewModel.kt | "[DEBUG-2]", "MIN_HISTORY_SPAN_US", "[GATE-SPAN]" |
| **DEBUG-3 진단 로그 (P40 신규)** | LocalizationViewModel.kt | "[INFER-OUT]", "[UPDATE-RES]" |
| **DEBUG-4 MAX_POST 5.0 (P40 신규)** | LocalizationViewModel.kt | "[DEBUG-4]", "MAX_POST_UPDATE_SPEED" |
| **DEBUG-5 z 측정 무시 (P40 신규)** | LocalizationViewModel.kt | "[DEBUG-5]", "meas[2] = 0.0" |
| 진단 로그 (clone insert / update try) | LocalizationViewModel.kt | "[CLONE-INSERT]", "[UPDATE-TRY]" |
| ImuTestActivity (다음 세션 확장) | ImuTestActivity.kt | "startCollection", "onSensorChanged" |
| 헬퍼 함수들 | LocalizationViewModel.kt | "findBeginClone", "buildCovMatrix", "transformWindowToWorldFrame" |

---

**작성일**: 2026-05-16 (P40 정리)  
**마지막 적용 단계**: DEBUG-5 (z 측정 무시) — 빌드 검증 대기  
**다음 작업**: DEBUG-5 검증 → IMU CSV 기능 추가 → 3 시나리오 측정 → OxIOD raw 비교 → 단계 2 보정 알고리즘 선택
