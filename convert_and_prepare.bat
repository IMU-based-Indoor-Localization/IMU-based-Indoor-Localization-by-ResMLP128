@echo off
chcp 65001 > nul
echo ================================================
echo  IMU 모델 변환 + Android assets 준비
echo ================================================
echo.

cd /d "%~dp0"

:: ── Python 실행파일 자동 탐색 ────────────────────────────────
set PYTHON_CMD=

:: 1순위: py 런처 (Python 공식 설치 시 존재)
where py >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    goto found_python
)

:: 2순위: python 명령 (단, Windows Store 앱인지 확인)
where python >nul 2>&1
if %errorlevel% == 0 (
    python --version >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON_CMD=python
        goto found_python
    )
)

:: 3순위: python3
where python3 >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python3
    goto found_python
)

:: 4순위: Anaconda 기본 경로
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%P (
        set PYTHON_CMD=%%P
        goto found_python
    )
)

echo [오류] Python을 찾을 수 없습니다.
echo        Python이 설치되어 있다면 전체 경로를 직접 입력하세요:
echo        예) C:\Users\사용자\anaconda3\python.exe android\prepare_assets.py
pause
exit /b 1

:found_python
echo Python 경로: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: ── Step 1: 모델 변환 ──────────────────────────────────────────
echo [1/2] PyTorch Mobile 모델 변환 중...
echo       (처음 실행 시 수 분 소요될 수 있습니다)
echo.

%PYTHON_CMD% src\convert_to_pytorch_mobile.py ^
  --model_path  src\outputs\out_classifier2\checkpoints\best.pth ^
  --config_path src\outputs\out_classifier2\config.json ^
  --norm_mean   src\outputs\out_classifier2\norm_mean.npy ^
  --norm_std    src\outputs\out_classifier2\norm_std.npy ^
  --out_dir     mobile_assets ^
  --verify

if %errorlevel% neq 0 (
    echo.
    echo [오류] 모델 변환 실패했습니다.
    echo        아래 명령으로 필요한 패키지를 설치한 뒤 다시 실행하세요:
    echo.
    echo        %PYTHON_CMD% -m pip install torch einops numpy
    echo.
    pause
    exit /b 1
)

:: ── Step 2: assets 복사 ────────────────────────────────────────
echo.
echo [2/2] Android assets 복사 중...

%PYTHON_CMD% android\prepare_assets.py

if %errorlevel% neq 0 (
    echo [오류] assets 복사 실패.
    pause
    exit /b 1
)

echo.
echo ================================================
echo  완료! Android Studio 에서 android\ 폴더를 여세요.
echo ================================================
echo.
pause
