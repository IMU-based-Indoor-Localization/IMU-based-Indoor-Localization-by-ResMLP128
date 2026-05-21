# tools/ — IMU 측정 환경 스크립트

P45+ 세션 측정 워크플로용 PowerShell 스크립트 모음.

## 사전 조건

- Windows 10/11 + PowerShell 5+
- `D:\SDK\platform-tools\adb.exe` 존재
- 단말 USB 디버깅 + 무선 ADB 활성
- 기본 단말 시리얼: `adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp`
  (다른 단말 사용 시 각 스크립트의 `-DeviceSerial` 인자로 지정)

## 디렉토리 구조

```
imu_android/
├─ tools/                      # 본 스크립트 폴더
│  ├─ collect.ps1              # 측정 회수 (logcat + CSV pull)
│  ├─ push_replay.ps1          # replay CSV 단말 push
│  └─ README.md                # 본 문서
├─ logs/                       # 회수된 logcat (.txt, UTF-8)
└─ csv/
   └─ imu_csv/                 # 회수된 측정 CSV (imu_record_*.csv)
```

## 표준 워크플로

### 1) 실측 측정 (단말에서 직접)

1. Android Studio 또는 사전 빌드한 APK 로 `com.imulocal` 앱 실행
2. 메뉴 → `IMU 센서 진단` → ImuTestActivity 진입
3. **측정 시작** 버튼 → 보행 / 정지 / 흔들기 등 시나리오 수행
4. **측정 종료**
5. 단말 저장 경로:
   `/sdcard/Android/data/com.imulocal/files/imu_csv/imu_record_<epoch_ms>.csv`

### 2) PC 로 회수 (collect.ps1)

```powershell
# 측정 *직전* 에 실행 → logcat 캡쳐 시작 → Enter 까지 대기 → 측정 종료 시 Enter → 자동 회수
.\tools\collect.ps1 -SessionName "baseline_01"
```

회수 결과:
- `logs/baseline_01.txt` — UTF-8 logcat 전체
- `csv/imu_csv/imu_record_*.csv` — 단말의 측정 CSV 전체

### 3) replay 입력 push (push_replay.ps1)

```powershell
# 회수된 CSV 목록 확인
.\tools\push_replay.ps1 -List

# 가장 최근 CSV 를 단말 latest.csv 로 push
.\tools\push_replay.ps1 -Latest

# 특정 CSV 를 push
.\tools\push_replay.ps1 -CsvName "imu_record_1779218552206.csv"
```

단말 push 경로 (HANDOFF_P44 §1.7):
`/sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv`

### 4) replay 실행 (앱 기능 구현 필요)

**현재 P40 baseline 의 working tree 에는 replay 기능이 *구현되어 있지 않음***.
HANDOFF_P44 §1.7 의 설명대로 `ImuCollector.kt` 의 `onSensorChanged` 핵심 로직을 `processSensorData(sensorType, tsUs, values)` 로 분리한 후 별도 thread 가 `latest.csv` 를 ts 순서대로 읽어 호출하는 방식으로 구현. *별도 task* 로 진행.

## 측정 시 권장 시나리오 (R-시리즈 명명)

P44 ablation 표 (HANDOFF §1.9) 와 호환되는 명명:

| 세션명 | 의도 |
|---|---|
| `baseline_NN` | 코드 변경 없는 기본 측정 (P40 baseline 검증) |
| `noise_NN`    | 정지 상태 (sensor noise floor 측정) |
| `R0_NN`       | DISPCOV=T, R_ALL=F (기본 ablation) |
| `R0p_NN`      | R0 재실행 (noise floor 검증) |
| `R1_NN`       | DISPCOV=F (dispCov fix off) |
| `R2_NN`       | R_ALL=T (매 시점 R[t]) |
| `R3_NN`       | OOD fix (linAcc/9.81 + 학습 norm) |

(NN = 01, 02, ... 순번)

## adb 연결 troubleshoot

```powershell
$adb = "D:\SDK\platform-tools\adb.exe"

# 1) 현재 연결 목록
& $adb devices

# 2) 무선 ADB 다시 연결 (단말 IP 확인 후)
& $adb connect 192.168.0.XX:5555

# 3) USB → 무선 전환 (USB 연결 상태에서)
& $adb tcpip 5555
& $adb connect 192.168.0.XX:5555

# 4) 인증 reset (단말 RSA 키 변경 시)
& $adb kill-server
& $adb start-server
```

## 환경 정보 (P45 기준)

| 항목 | 값 |
|---|---|
| 단말 | Samsung r11sksx (Galaxy ?) — Android 16 / BP2A.250605.031.A3 |
| 단말 시리얼 | `adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp` |
| 앱 패키지 | `com.imulocal` |
| adb 경로 | `D:\SDK\platform-tools\adb.exe` |
| 모델 .ptl SHA | `9eb93ba82294b89c75be83b75a5fd0fd` (assets/mobile_assets 동일) |

## P45 logcat 파일 정리 정책

`logs/` 에는 이전 세션의 P45 측정 파일이 있습니다:
- `crash_p45.txt` / `crash_p45_utf8.txt` — P45 crash 로그 (수정 전 데이터)
- `main_p45.txt` / `main_p45_utf8.txt` / `main_p45_app.txt` — P45 측정 logcat
- `noise_p45.txt` / `noise_p45_utf8.txt` — P45 노이즈 측정

**보존**: 디버깅/비교 목적. 새 측정 시에는 `-SessionName` 으로 다른 이름 사용.
**대용량 정리**: `crash_log.txt` (414 MB, project root) 는 별도 정리 task 진행 시 핵심 구간만 추출 후 제거.
