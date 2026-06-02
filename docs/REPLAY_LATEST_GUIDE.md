# `latest.csv` 교체 가이드

> 단말의 Replay 기능이 읽는 `latest.csv` 를 다른 측정 데이터로 교체하는 방법.
> 사용자 직접 수행 또는 다른 Claude 세션 agent 가 수행할 수 있도록 정리.

---

## 0. 핵심 정보 (Claude/AI 가 빠르게 읽을 부분)

```
[단말 측위 앱 패키지]   com.imulocal
[단말 시리얼]          adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp  (무선 ADB)
[ADB 경로 (Windows)]   D:\SDK\platform-tools\adb.exe
[Replay 디렉토리]      /sdcard/Android/data/com.imulocal/files/imu_csv/replay/
[Replay 대상 파일]     /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
[측정 기록 디렉토리]    /sdcard/Android/data/com.imulocal/files/imu_csv/imu_record_<epoch>.csv
[PC 측정 사본]         D:\mobile\imu_android\csv\imu_csv\
[CSV 형식 (Replay)]    sensor,ts_ns,x,y,z,w  (헤더 1줄 + 데이터)
```

CSV 형식 (`ImuCollector.startReplay` 가 읽는 단일 파일):
```
sensor,ts_ns,x,y,z,w
acc,1234567890000,0.012,-9.811,0.034,0.0
gyr,1234567892000,0.0011,-0.0023,0.0008,0.0
linAcc,1234567893000,0.011,-0.001,0.024,0.0
rotVec,1234567895000,0.012,0.034,-0.067,0.997
...
```
- `sensor`: acc / gyr / linAcc / rotVec
- `ts_ns`: epoch nanoseconds (또는 monotonic 나노초 — 차이만 정확하면 OK)
- `x,y,z`: 센서 값 / `w`: rotVec 만 quaternion w. 다른 센서는 0.0.

---

## 1. AI Agent 용 (다른 Claude 세션) — 명령어 ready

### A. 가장 흔한 케이스: 기존 백업 → latest 복원
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell cp \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest_backup_XXX.csv \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

### B. 단말 내 측정 (imu_record_*.csv) → latest 로 활성화
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell cp \
  /sdcard/Android/data/com.imulocal/files/imu_csv/imu_record_<EPOCH>.csv \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

### C. PC 파일 → 단말 latest 로 push
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp \
  push "D:\path\to\my.csv" \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

또는 기존 PowerShell 도구 사용:
```powershell
.\tools\push_replay.ps1 -CsvName "imu_record_1779677257332.csv"    # 특정 파일
.\tools\push_replay.ps1 -Latest                                   # PC csv/imu_csv 중 가장 최근
.\tools\push_replay.ps1 -List                                     # 후보 목록만 표시
```

### D. iOS Sensor Logger 폴더 → 변환 → latest 로 push
```powershell
python D:\mobile\imu_android\tools\ios_sensorlogger_to_replay.py "<ios_folder>" "D:\mobile\imu_android\csv\imu_csv\ios_<tag>.csv"

D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp \
  push "D:\mobile\imu_android\csv\imu_csv\ios_<tag>.csv" \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

### E. 안전: 기존 latest 백업 후 새 파일로 교체 (recommended)
```powershell
# 1) 현재 latest 백업
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell cp \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest_backup_<DATE_OR_TAG>.csv

# 2) 새 파일로 교체 (위 A/B/C/D 중 하나)
# ...

# 3) 결과 확인
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell ls -la \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/
```

### F. ADB 연결 확인
```powershell
D:\SDK\platform-tools\adb.exe devices
# 출력에 "adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp  device" 가 보이면 OK.
# offline 또는 미표시 → 단말에서 wifi ADB 재연결 필요.
```

---

## 1.5 직접 측정 → 즉시 latest 로 교체 (가장 흔한 사용자 워크플로우)

### 핵심 구분 — *두 가지 export 가 있다*
| 기능 | 메뉴 / 버튼 | 출력 파일 | 형식 | replay 입력 가능? |
|---|---|---|---|---|
| **IMU 센서 진단** 측정 | 메뉴 "IMU 센서 진단" → [시작]/[정지] | `imu_record_<epoch>.csv` | `sensor,ts_ns,x,y,z,w` (raw IMU 4센서) | **✅ 가능** |
| **경로 내보내기** | 메뉴 "경로 내보내기" | `track_PATH_B_<ts>.csv` | `x_m,y_m` (PATH_B 궤적 점들만) | ❌ **불가** — IMU raw 가 아닌 *결과* 위치만 |

→ **새 latest.csv 만들려면 반드시 IMU 센서 진단 (ImuTestActivity) 으로 측정**. "경로 내보내기" 는 *분석/시각화* 용 결과 출력일 뿐.

### 단계별 절차

#### Step 1 — 단말에서 측정
1. 앱 메인 화면 → 메뉴 (우상단 ⋮) → **"IMU 센서 진단"** 선택 → `ImuTestActivity` 진입
2. **[시작]** 버튼 누름. 화면에 "● 수집 중..." 표시 + 샘플 카운트 증가 확인 (≈100Hz)
3. 측정하고 싶은 보행/동작 수행 (예: 5m 직선, 왕복, 사각형 등)
4. **[정지]** 버튼 누름 → Toast "CSV 저장 완료: imu_record_<epoch>.csv (N 줄)" 확인
5. 저장 위치 (앱 내부 외부저장소):
   ```
   /sdcard/Android/data/com.imulocal/files/imu_csv/imu_record_<epoch_ms>.csv
   ```

#### Step 2 — 단말 안에서 latest 로 복사 (PC 경유 불필요)

**방법 A — PC + ADB (사용자가 명령어 실행)**:
```powershell
# 1) 가장 최근 측정 파일명 확인
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell ls -t \
  /sdcard/Android/data/com.imulocal/files/imu_csv/ | head -5

# 2) 직전 latest 백업 (옵션, 추천)
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell cp \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest_backup_$(Get-Date -Format yyMMdd_HHmm).csv

# 3) 새 측정 → latest
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell cp \
  /sdcard/Android/data/com.imulocal/files/imu_csv/imu_record_<EPOCH>.csv \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

**방법 B — Android Studio Device File Explorer (코드 없이 GUI)**:
1. Android Studio → `View → Tool Windows → Device Explorer`
2. 경로 이동: `sdcard/Android/data/com.imulocal/files/imu_csv/`
3. 방금 만든 `imu_record_<epoch>.csv` 우클릭 → "Synchronize" (또는 Save As) → PC 임시 폴더 다운로드
4. 같은 트리에서 `imu_csv/replay/` 폴더로 이동
5. 다운로드한 파일을 우클릭 "Upload" → 업로드 후 이름을 `latest.csv` 로 변경 (기존 덮어쓰기)

**방법 C — 단말 내부 파일 관리자 앱 (Material Files 등 `Android/data` 접근 가능 앱)**:
1. `Android/data/com.imulocal/files/imu_csv/` 진입
2. `imu_record_<epoch>.csv` 길게 눌러 "복사"
3. `replay/` 하위 폴더로 이동 → 붙여넣기 → 이름을 `latest.csv` 로 변경

#### Step 3 — 즉시 재생 검증
1. 앱 메인 화면으로 돌아옴 → **[초기화]** 누름
2. **[Replay (latest.csv 재생)]** 누름 → Toast 알림 + polyline 자라기 시작
3. 지도 모드 진입해서 시각 확인 (필요 시 슬라이더로 시점 scrub)

### Step 1.5 단축 — *(현재 없는 기능, future 개선 옵션)*
현 버전엔 ImuTestActivity 에 "이 측정을 즉시 latest 로 저장" 버튼이 없음. 사용자 작업 단축 위해:
- (A) 그 버튼 추가 (코드 5~10 줄, [정지] 직후 자동으로 replay/latest.csv 에 cp)
- (B) MainActivity 의 [Replay] 옆에 "최근 측정 사용" 버튼 (가장 최근 imu_record_*.csv 자동 선택)

원하면 신호 주세요 — 둘 다 단순 변경.

---

## 2. 사용자 직접 수행 (3가지 방법)

### 방법 1 — **Android Studio Device File Explorer (추천)**
1. 단말을 USB 또는 무선 ADB 로 연결.
2. Android Studio 열기 → 메뉴 `View → Tool Windows → Device Explorer` (또는 우하단 "Device Explorer" 아이콘).
3. 좌측 탭에서 단말 선택 → 트리에서 다음 경로 이동:
   `sdcard/Android/data/com.imulocal/files/imu_csv/replay/`
4. 작업:
   - **현재 latest.csv 백업**: 우클릭 → "Synchronize" 로 PC 에 다운로드, 또는 단말 안에서 "Copy" 후 같은 폴더에 paste → `latest_backup_xxx.csv` 로 rename
   - **다른 측정으로 교체**: 상위 폴더 (`imu_csv/`) 의 원하는 `imu_record_*.csv` 우클릭 → "Save As" 로 다운로드 → 다시 `replay/` 폴더에 "Upload" → 이름을 `latest.csv` 로 변경 (기존 것 덮어쓰기)
5. 단말 앱에서 [Replay (latest.csv 재생)] 누르면 새 데이터로 재생.

### 방법 2 — **단말 내 파일 관리자 앱 (Files by Google, "내 파일" 등)**
1. **단말의 일부 파일 관리자는 `Android/data/` 접근이 제한됨** (Android 11+ scoped storage). 다음 중 하나:
   - **Material Files** (오픈소스), **MiXplorer**, **Solid Explorer** 등 `Android/data` 접근 가능한 앱 설치.
   - 또는 PC 경유로 우회 (방법 1 또는 3).
2. 파일 관리자에서 경로 이동: `Internal storage → Android → data → com.imulocal → files → imu_csv → replay`
3. `latest.csv` 우상단 "복사" 또는 "이름변경" 으로 백업.
4. 상위 `imu_csv/` 폴더의 원하는 `imu_record_*.csv` 를 `replay/` 로 복사 → 이름을 `latest.csv` 로 변경.

### 방법 3 — **USB 연결 + Windows 파일 탐색기**
1. 단말을 USB 로 PC 에 연결.
2. 단말 화면에서 "파일 전송 (MTP)" 모드 선택.
3. Windows 파일 탐색기 → "내 PC" 에 단말 이름 표시됨.
4. 다만 **`Android/data` 폴더는 보통 MTP 로 안 보일 수 있음** — 보이지 않으면 방법 1 (Device Explorer) 사용 권장.
5. 보이면 위와 같이 백업 + 교체.

---

## 3. 백업 파일 명명 규칙 (권장)

여러 측정을 보존하려면 의미 있는 이름으로:

| 패턴 | 예시 | 용도 |
|---|---|---|
| `latest_backup_<날짜>.csv` | `latest_backup_5_25.csv` | 시간 기준 |
| `latest_backup_<상황>.csv` | `latest_backup_5m_walk.csv` | 보행 시나리오 기준 |
| `latest_backup_<단말>.csv` | `latest_backup_ios_5_54.csv` | 기기 기준 |
| `latest_backup_<태스크번호>.csv` | `latest_backup_p73.csv` | 개발 마일스톤 |

현재 단말에 존재하는 백업 확인:
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell ls -la \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/
```

---

## 4. 교체 후 검증

### A. 파일 크기/타임스탬프 확인
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell ls -la \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```

### B. 첫 줄 (헤더) 확인
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp shell head -2 \
  /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
```
첫 줄이 `sensor,ts_ns,x,y,z,w` 이면 형식 OK. 다음 줄은 데이터.

### C. 앱에서 재생 + logcat 로 진행 확인
```powershell
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp logcat -c        # 클리어
# 앱에서 [초기화] → [Replay (latest.csv 재생)] 누름
D:\SDK\platform-tools\adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp logcat -d | grep -E "P45-Replay|DR|EkfBridge"
```
정상 패턴: `[P45-Replay] 재생 thread 시작: latest.csv (speed=1.0x)` + `[DR]` 로그 정기 출력.

---

## 5. 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `adb: command not found` | ADB 경로 미설정 | `D:\SDK\platform-tools\adb.exe` 전체 경로로 호출 |
| `device offline` 또는 `unauthorized` | 무선 ADB 끊김 / 인증 필요 | 단말 화면에서 USB 디버깅 허용 / `adb kill-server; adb start-server` |
| `Permission denied` | scoped storage | `Android/data` 는 shell 권한으로만 접근 가능 — 위 `adb shell cp` 사용 (앱 권한 불필요) |
| Replay 누르면 곧바로 끝남 | CSV 빈 파일 / 헤더 잘못 | 헤더 정확히 `sensor,ts_ns,x,y,z,w` 확인. 비어있으면 다시 push |
| `[P45-Replay] CSV 비어있음` 로그 | 단말이 빈 파일 인식 | 파일 크기 확인. 0 바이트면 push 실패 — 다시 시도 |
| 시작점 위치가 다르게 보임 | 자석/방위 보정 안 됨 | 단말 8자 회전 동작으로 자력계 캘리브 → RotVec accuracy HIGH 확보 후 재시도 |
| 궤적이 90° 회전됨 | iOS 데이터의 NWU↔ENU 차이 | 정상 (모양·거리는 보존). 학술적 비교는 동일 영향이라 무관 |

---

## 6. 참고 코드 (단말 측)

- 재생 진입점: `ImuCollector.kt:startReplay()` 
- replay 모드 동기화: `LocalizationViewModel.kt` 의 [P45-Replay] 블록
- UI 트리거: `MainActivity.kt:startReplay()` (메뉴 또는 버튼)
- 단말 권한: scoped storage 우회 위해 `Android/data/com.imulocal` 내부에 저장 (자기 패키지 영역)

---

*문서 작성: 2026-05-25. 단말 시리얼·경로 등 환경 의존 값은 1번 섹션 참조.*
