# HANDOFF P44 — P43+ 세션 결과 정리 (코드 변경 폐기)

작성: 2026-05-20 KST
이전 핸드오프: HANDOFF_P40.md, HANDOFF_P41.md, HANDOFF_P43.md
**본 세션 코드 변경은 모두 폐기됩니다 (앱 빌드 깨짐). 다음 세션은 git clone 후 처음부터 시작.**
본 문서는 *코드 외 정보/사실/실패 시도/권장 작업* 만 정리합니다.

---

## 0. 본 세션의 실패 요약 (재발 방지)

세션 진행 중 누적한 변경 (10+ 토글, norm 파일 교체, replay 모드, OOD fix, FORCED_CLS 등) 이 한 번에 격리 검증 안 된 채 빌드되며 끝에 `SIGBUS BUS_ADRALN at 0x3 in libpytorch_jni_lite.so` (`at::infer_size_dimvector` → `structured_add_Tensor::meta`) 가 발생, 캘리브레이션 직후 강제 종료가 *모든 토글 조합* 에서 재현됨. `noCompress 'ptl'` 추가, `pm clear` + `uninstall` + 단말 재부팅 + Gradle 캐시 nuke 어느 것도 해결하지 못함.

원인 미해결. 다음 세션은 *clean git clone* 에서 다시 시작.

### 다음 세션에서 *반드시 회피*

1. **한 번에 다중 변경 금지** — 한 변경 → 빌드 → 측정 → 검증 → 다음 변경
2. **norm_*.txt 를 Write 도구로 덮어쓰지 말 것** — NUL byte 부산물 가능성. `prepare_assets.py` 또는 `Copy-Item` 사용
3. **InferenceEngine.kt / ImuCollector.kt 의 `Edit` 도구 사용 시 파일 잘림 발생 사례 다수** — 큰 변경은 Write 로 전체 재작성하거나 매우 작은 단위로 Edit
4. **git stash 등 명령 실행 후 `git status` 로 검증 필수** — 명령이 실제 효과 냈는지 확인
5. **bash (Linux mount) 가 Windows 측 파일 변경에 대해 stale 캐시 보일 수 있음** — `Read` 도구는 직접 액세스라 항상 fresh
6. **`Edit` 의 `old_string` 매칭 실패 시 *해당 라인 주변 mojibake* (CP949 깨진 한글) 확인** — 코드의 기존 깨진 한글 주석을 한국어로 *복원 시도하지 말 것*. 그대로 둘 것

---

## 1. 확정된 사실 (다음 세션에서 *재현 검증 후* 활용)

### 1.1 OxIOD 학습 데이터 단위 = g (gravity 제거 후 userAcceleration)

`src/TLIO_Oxford_Dataset/oxford_handheld_1/imu0_resampled.npy` 의 컬럼별 직접 통계:

| 컬럼 | 단위 / 의미 |
|---|---|
| col[1:4] gyr | rad/s (Android 와 동일) |
| col[4:7] acc | **g (gravity 제거됨)** — userAcceleration 정의 |
| col[7:10] gravity | g (norm mean **1.0000** 으로 확정됨) |

검증 명령:
```python
import numpy as np
d = np.load('src/TLIO_Oxford_Dataset/oxford_handheld_1/imu0_resampled.npy')
print(f"gravity norm mean = {np.linalg.norm(d[:,7:10], axis=1).mean():.4f}")  # 1.0000 → g 단위 확정
print(f"acc axis mean = {d[:,4:7].mean(axis=0)}")                              # ≈ 0 → gravity 제거됨
```

→ **Android `TYPE_LINEAR_ACCELERATION` (m/s²) 과 9.81× 단위 차이**. 모델 입력 시 변환 필요.

### 1.2 모델 정규화 통계의 출처

`mobile_assets/norm_params.pt` (모델 .ptl 과 *같은 시점 변환* 된 진짜 정규화) 의 raw binary 추출:

```
mean = [0.00267683, 0.01422549, -0.00353606, -0.02691823, 0.03028979, -0.01410883]
std  = [0.11819147, 0.12025723, 0.14811487, 0.43106490, 0.48224238, 0.61772841]
```

→ `src/outputs/out_classifier2/norm_{mean,std}.npy` 와 *완전 일치*. 즉 **모델 = `out_classifier2` 학습 결과**.

### 1.3 `prepare_assets.py` 의 잘못된 경로

```python
NORM_MEAN = ".../src/outputs/out_tlio_6ch_128/norm_mean.npy"   # ← 잘못, 다른 학습 결과
NORM_STD  = ".../src/outputs/out_tlio_6ch_128/norm_std.npy"
```

→ assets/norm_*.txt 가 *out_tlio_6ch_128 의 값* (mean ≈ [-0.13, -2.83, -3.40], std ≈ [5.76, 6.07, 2.97]) 으로 채워짐 → **모델 학습 정규화와 50× 차이**.

**수정**: `prepare_assets.py` 의 NORM_MEAN/NORM_STD 경로를 `out_classifier2` 로 변경 후 재실행.

### 1.4 CLASS_NAMES 매핑 (코드 실측)

logcat 의 `cls=N(name)` 출력 매핑:

```
0: handbag
1: handheld
2: pocket
3: running
4: slow_walk  (logcat 미관측, 추정)
5: trolley   (logcat 미관측, 추정)
6: unknown
```

→ `model_meta.json` 의 cls_labels (`0=unknown, 1=handbag, ...`) 와 **다름**. 코드/모델 출력이 정확. model_meta.json 은 잘못된 매핑.

학습 코드 주석에 `train.py LABEL_REMAP {-1→6, 1→0, 2→1, 3→2, 4→3, 5→4, 6→5}` 명시되어 있음.

### 1.5 Android 의 Dead-Reckoning Bypass *이미 구현*

`LocalizationViewModel.kt`:
- line 217: `NETWORK_ONLY_CLASSES = setOf(0, 1, 2, 3, 4, 5)` — trolley(5) 외 모두 우회
- line 794-849: Network-only 분기 (R_z(yaw0) @ meas → netPos 누적 + marginalize + applyRotVecYaw)
- line 216: `USE_DEAD_RECKONING_BYPASS = false` — P41 bypass01 측정에서 trackPoints 발산 (-y 방향 편향 누적) → 토글 OFF

→ 메모리의 `reference_p41_confirmed_facts.md` §2/§4 의 "Android 에 반영 안 됨" 은 *오류*. 이미 *구현 완료, 토글 OFF*.

### 1.6 IMU CSV 형식 (replay 입력 호환)

```
헤더: sensor,ts_ns,x,y,z,w
sensor: acc / gyr / linAcc / rotVec
ts_ns: SensorEvent.timestamp (부팅 후 ns)
x,y,z: 센서 값
w: rotVec 의 quaternion 스칼라부 (다른 센서는 0.0)
```

| 센서 | 실효 rate (SENSOR_DELAY_FASTEST) |
|---|---|
| acc, gyr | ~500 Hz |
| linAcc, rotVec | ~125 Hz |

`ImuTestActivity` 가 P40 부터 이 형식으로 raw IMU 기록 가능 (메뉴 → IMU 센서 진단 → 시작).

### 1.7 replay 환경 작동 검증됨 (구현 가능성 확정)

`ImuCollector.kt` 의 `onSensorChanged` 의 핵심 로직 (latest* 갱신 + 캘리브 + 100Hz 리샘플 + 버퍼 push) 을 `processSensorData(sensorType, tsUs, values)` 로 분리하면, 별도 thread 가 CSV 를 ts 순서대로 읽어 `processSensorData` 직접 호출 가능. 같은 EKF 거동 재현.

23.8초 CSV → 23.802초 재생 (Thread.sleep 분해능 ±6ms). 라인 100% 처리.

**replay 단말 위치**: `/sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv`

### 1.8 PowerShell 측정 명령 (tools/ 폴더 없으면 직접 실행)

```powershell
$adb    = "D:\SDK\platform-tools\adb.exe"
$device = "adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp"

# logcat 캡쳐 (UTF-8 필수 — PowerShell 의 `>` 는 기본 UTF-16)
& $adb -s $device logcat -c
& $adb -s $device logcat | Out-File -Encoding utf8 logs\<세션명>.txt
# 측정 종료 후 Ctrl+C

# 단말 → PC CSV 회수
& $adb -s $device pull /sdcard/Android/data/com.imulocal/files/ csv\

# 측정 도구 디렉토리 — commit a37b770 에 tools/ 복원됨
ls tools\
```

### 1.9 ablation 결과 통계 (replay 23.8초, 5m 왕복 측정)

| 설정 | 최종 \|xy\| | meas\|xy\| avg | cls 분포 (idx) |
|---|---|---|---|
| R0  DISPCOV=T R_ALL=F  | 5.530m | 0.323 | 89×3(running) |
| R0' DISPCOV=T R_ALL=F  | 4.988m | 0.332 | 동일 |
| R1  DISPCOV=F R_ALL=F  | 5.263m | 0.335 | 동일 |
| R2  DISPCOV=T R_ALL=T  | 5.450m | 0.336 | 동일 |
| R2v2 DISPCOV=T R_ALL=T | 5.371m | 0.337 | 동일 |
| **R3 OOD=T (÷9.81 + 학습 norm)** | **6.548m** | **0.452** | **86×6(unknown)** |

**Noise floor (R0 vs R0' = 같은 토글 재실행)** = **0.542m**.

→ EKF 측 모든 토글 (dispCov fix, R_all[t], JUMP-GATE) 효과 = noise 범위 내. *유의차 없음*.
→ OOD fix 적용 시 모델 출력 *극적 변화* (cls 86% unknown, disp_y 부호 반전, |xy| +1.0m). 단위 변환은 옳지만 분류기는 *Android raw IMU 패턴을 학습 분포에 못 매칭*.

### 1.10 모델 자체 한계

5m 왕복 측정인데 ekfSpeed avg 1.48 m/s × 23.8s = **35m 누적 추정**. 실제 보행 (~10m) 대비 *3.5× 과대평가*. Notion §5 의 Network-only RMSE ~1.5m 와 일치 — *모델 자체* 의 한계. EKF 측 토글로 해결 불가.

---

## 2. 시도와 결과 (다음 세션에서 같은 시도 반복 방지)

| # | 시도 | 결과 | 비고 |
|---|---|---|---|
| 1 | dispCov fix (`exp(2*log_std)`) 적용/롤백 | 효과 < noise (0.5m) | Notion §3.5 표준, 메모리 ★ 우선 |
| 2 | USE_R_ALL_T_FRAME=true (매 시점 R[t]) | 효과 < noise. R-ALL-T 진입 81회 확인됨 | P41 v1 발산 → v2 (raw gyr + s_bg 차감 없음) 로 안전 작동 |
| 3 | POST_UPDATE_JUMP_GATE_M=1.5 | 발동 0건. 본 CSV 의 1-step max 1.12m | 점프 측정 케이스 (outdoor_jumpgate*) 에선 효과 가능 |
| 4 | USE_DEAD_RECKONING_BYPASS=true | P41 측정에서 trackPoints 발산 → 즉시 롤백 | 이미 코드에 구현, 토글 OFF |
| 5 | USE_OOD_FIX (linAcc m/s² → g, ÷9.81) | 모델 분류 99% running → 86% unknown 변화. 누적 |xy| +1.0m 악화 | 단위 변환 자체는 맞음. 모델 분류기가 Android 패턴 못 인식 |
| 6 | FORCED_CLS=1 (handheld 강제) | SIGBUS crash 발생 시작점 (추정) | 정확한 원인 미해결 |
| 7 | build.gradle 의 noCompress 'ptl' | crash 지속 (해결 X) | 가설 무효 또는 다른 원인과 결합 |
| 8 | pm clear + uninstall + 단말 재부팅 + Gradle nuke | crash 지속 | 환경 차원 미해결 |

---

## 3. 미해결 의문 (다음 세션에서 해결 시도)

### 3.1 `SIGBUS BUS_ADRALN at 0x3` 의 진짜 원인

backtrace:
```
#08 at::infer_size_dimvector(...)
#09 at::TensorIteratorBase::compute_shape(...)
#10 at::TensorIteratorBase::build(...)
#11 at::TensorIteratorBase::build_borrowing_binary_op(...)
#12 at::meta::structured_add_Tensor::meta(...)
```

PyTorch Mobile 의 add 연산에서 broadcast 추론 시 정렬 위반. R3 (06:06) 측정 정상 → R4 (06:30) crash. 코드 차이 = FORCED_CLS 분기. *FORCED_CLS=-1 으로 동일 동작 롤백 후에도 crash 지속* → 원인 불명.

### 3.2 모델 분류기의 Android raw IMU 미인식

OOD fix 적용 시 *정규화 후 분포는 학습 분포 안에 들어옴* (Android linAcc/9.81 → 학습 norm 적용 후 0.86σ). 그런데 분류 86% unknown. 즉 *통계 분포는 매칭* 하지만 *시간적 패턴 (high-freq spectral content)* 이 학습 데이터 (iPhone Core Motion 의 anti-aliased) 와 다름.

### 3.3 model_meta.json 의 매핑 오류

- cls_labels: `{"0":"unknown", "1":"handbag", ...}` — 코드 실측과 *완전히 다름*
- ekf_meascov_scale 도 *idx 0 = unknown 가정* 으로 작성됨

→ model_meta.json 을 코드 매핑으로 정정 필요 (단 EkfBridge 의 R-scale 사용 로직 확인 필수).

---

## 4. 다음 세션 권장 출발점 — *작은 단계, 격리 검증*

1. **git clone + 첫 빌드 검증 (변경 없이)** — `commit a37b770` (P41 baseline + tools 복원) 빌드/측정 정상 동작 확인. *이게 baseline*.

2. **`noCompress 'ptl'` 만 추가 → 측정** — PyTorch Mobile 공식 권장. 영향 검증.

3. **`prepare_assets.py` 수정 → `out_classifier2/norm_*.npy` → assets/norm_*.txt 재생성** — Python 환경에서 진행, Write 도구로 덮어쓰지 말 것.

4. **InferenceEngine 에 `linAcc /= 9.81` 만 추가** (norm 교체 *없이*) → 측정 → 부분 OOD 검증.

5. **위 3+4 동시 적용 → 측정 → R3 결과 (86% unknown) 재현 확인**.

6. **모델 OOD 추가 진단** (3.2):
   - Android raw IMU 의 FFT/PSD vs OxIOD FFT/PSD 비교 (`src/View/imu_oxiod_vs_android.py` 활용)
   - 모델 self-attention 또는 분류 헤드의 어느 feature 가 분류 결정에 사용되는지 ablation

7. **모델 재학습 가능성 검토** — Android 측정 데이터로 fine-tune. `src/Network/out_classifier2/` 의 학습 코드 + Python torch 환경 필요.

---

## 5. 회수된 데이터 (다음 세션에서 *그대로* 활용 가능)

### 측정 CSV (replay 입력)
- `csv/imu_csv/imu_record_1779218552206.csv` — 5/20 04:22, 23.8초, 사용자 측정 (5m 왕복 보고)
- `csv/imu_csv/imu_record_1779067578235.csv` — 5/18, 24.6초

### logcat (분석 자료)
- `logs/clean_walk_30s_replay_001.txt` (R0)
- `logs/clean_walk_30s_replay_R0p_001.txt` (R0')
- `logs/clean_walk_30s_replay_R1_dispcov_off.txt` (R1)
- `logs/clean_walk_30s_replay_R2_rallt_on.txt` (R2)
- `logs/clean_walk_30s_replay_R2v2_001.txt` (R2v2)
- `logs/clean_walk_30s_replay_R3_oodfix_001.txt` (R3)
- `logs/clean_walk_30s_replay_R4_handheld_001.txt` (R4 — SIGBUS 시작)
- `logs/clean_walk_30s_replay_R3v2_clean.txt` (R3v2 — 클린 빌드 후에도 SIGBUS)
- `logs/measure_20260520_033550.txt` (5/18 빌드 실측 + INFER-OUT 정상)
- `logs/outdoor_jumpgate*.txt` (P42 진단)

→ 다음 세션에서 *코드 변경 없이 분석만으로 가치 있음*.

---

## 6. 환경 reference

| 항목 | 값 |
|---|---|
| 프로젝트 루트 | `D:\mobile\IMU-based-Indoor-Localization-by-ResMLP128` |
| 브랜치 | `rollback-p11-with-p21-calib` |
| HEAD commit | `a37b770` "chore: tools 스크립트 복원" |
| adb 경로 | `D:\SDK\platform-tools\adb.exe` |
| 단말 시리얼 | `adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp` |
| 모델 .ptl SHA (assets, mobile_assets 동일) | `9eb93ba82294b89c75be83b75a5fd0fd` |
| PyTorch Mobile 버전 | `org.pytorch:pytorch_android_lite:2.1.0` |
| Naver Map SDK | `com.naver.maps:map-sdk:3.23.2` (인증 키 `f7cl8uy1mq` 코드 직접 박힘 — local.properties NAVER_MAP_CLIENT_ID 와 무관) |

---

## 7. 메모리 업데이트 권장 (다음 세션 자동 로드)

다음 세션 시작 시 다음 메모리 파일 *정정* 권장:

- `reference_p41_confirmed_facts.md` §2/§4: "Android 에 반영 안 됨" → "이미 구현됨 (line 217, 794-849). USE_DEAD_RECKONING_BYPASS=false 로 OFF 상태"
- 추가 메모리: `reference_p44_ood_diagnosis.md` — OxIOD 단위 (g), norm 출처 (out_classifier2), model_meta.json 매핑 오류, prepare_assets.py 잘못된 경로
- `feedback_one_change_at_a_time.md` 신규 — 본 세션 실패 교훈 (한 번에 하나만 변경, 격리 검증)

---

**END OF HANDOFF P44**

다음 세션은 *git clone 후 위 권장 순서대로 차근차근* 진행 권장. 본 문서는 *코드 변경 폐기 후에도 활용 가능한 정보* 만 담음. 본 세션의 실제 코드 변경 (build.gradle noCompress, InferenceEngine OOD fix 코드, ImuCollector replay 코드 등) 은 *모두 폐기* 이며 *재구현 시 본 문서 참고하여 작은 단계로 분리* 진행할 것.
