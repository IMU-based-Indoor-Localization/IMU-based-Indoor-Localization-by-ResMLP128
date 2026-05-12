@echo off
REM ===============================================================
REM  P22 ZUPT 검증용 Logcat 캡처 스크립트
REM  - 사용 방법:
REM    1. 폰을 USB 디버깅 모드로 PC에 연결
REM    2. 이 파일을 더블클릭 (또는 cmd 에서 실행)
REM    3. 어플 시작 -> 캘리브레이션 진행 -> 워밍업 3초 대기
REM       -> 정지 30초 유지 -> 측위 정지 버튼
REM    4. 이 창에서 Ctrl+C 로 캡처 종료
REM    5. logcat_zupt.txt 가 같은 폴더에 생성됨
REM ===============================================================

cd /d "%~dp0"

echo [1/3] adb 디바이스 확인...
adb devices
echo.

echo [2/3] 이전 Logcat 버퍼 초기화...
adb logcat -c
echo.

echo [3/3] Logcat 캡처 시작 - 파일: logcat_zupt.txt
echo       (어플 측위 시작 -> 30초 정지 -> 측위 정지 후, 이 창에서 Ctrl+C)
echo.
adb logcat -v threadtime ^
  "Stage1.AbsNode:*" ^
  "Stage2.InferNode:*" ^
  "Stage3.EkfTracker:*" ^
  "Controller:*" ^
  "*:S" > logcat_zupt.txt
