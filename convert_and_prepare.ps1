# ============================================================
# convert_and_prepare.ps1
# 모델 변환 → Android assets 배치 원클릭 스크립트
# 실행: 프로젝트 루트에서  powershell -ExecutionPolicy Bypass -File convert_and_prepare.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  IMU Localization 모델 변환 파이프라인" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── 경로 설정 ──────────────────────────────────────────────
$MODEL_PTH    = "$ROOT\src\outputs\out_tlio_6ch_128\checkpoints\best.pth"
$CONFIG_JSON  = "$ROOT\src\outputs\out_tlio_6ch_128\config.json"
$MOBILE_DIR   = "$ROOT\mobile_assets"
$PT_OUT       = "$MOBILE_DIR\model_torchscript.pt"
$PTL_OUT      = "$MOBILE_DIR\imu_model.ptl"
$SRC_DIR      = "$ROOT\src"

# ── Python 탐색 ────────────────────────────────────────────
$PYTHON = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") { $PYTHON = $cmd; break }
    } catch {}
}
if (-not $PYTHON) {
    Write-Host "❌ Python 3 를 찾지 못했습니다. PATH를 확인하세요." -ForegroundColor Red
    exit 1
}
Write-Host "✔  Python: $(&  $PYTHON --version)" -ForegroundColor Green

# ── 입력 파일 확인 ─────────────────────────────────────────
foreach ($f in @($MODEL_PTH, $CONFIG_JSON)) {
    if (-not (Test-Path $f)) {
        Write-Host "❌ 파일 없음: $f" -ForegroundColor Red
        exit 1
    }
}
New-Item -ItemType Directory -Force -Path $MOBILE_DIR | Out-Null

# ──────────────────────────────────────────────────────────
# Step 1: TorchScript(.pt) 변환
# ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/3] TorchScript 변환 중 ..." -ForegroundColor Yellow
& $PYTHON "$SRC_DIR\convert_model_to_torchscript.py" `
    --model_path  $MODEL_PTH `
    --config_path $CONFIG_JSON `
    --out_dir     $MOBILE_DIR
if ($LASTEXITCODE -ne 0) { Write-Host "❌ TorchScript 변환 실패" -ForegroundColor Red; exit 1 }
Write-Host "✔  $PT_OUT 생성 완료" -ForegroundColor Green

# ──────────────────────────────────────────────────────────
# Step 2: PyTorch Mobile Lite(.ptl) 변환
# ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/3] PyTorch Mobile Lite 변환 중 ..." -ForegroundColor Yellow
& $PYTHON "$SRC_DIR\convert_to_pytorch_mobile.py" `
    --pt_path  $PT_OUT `
    --out_path $PTL_OUT
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Mobile Lite 변환 실패" -ForegroundColor Red; exit 1 }
Write-Host "✔  $PTL_OUT 생성 완료" -ForegroundColor Green

# ──────────────────────────────────────────────────────────
# Step 3: Android assets 배치
# ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/3] Android assets 배치 중 ..." -ForegroundColor Yellow
& $PYTHON "$ROOT\android\prepare_assets.py"
if ($LASTEXITCODE -ne 0) { Write-Host "❌ assets 배치 실패" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ 모든 단계 완료!" -ForegroundColor Green
Write-Host "  Android Studio 에서 android/ 폴더를" -ForegroundColor Green
Write-Host "  열고 Run 하세요." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
