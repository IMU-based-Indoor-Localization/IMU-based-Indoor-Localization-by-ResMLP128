# tools/push_replay.ps1 — 회수한 CSV 를 단말 replay 경로로 push
#
# 사용:
#   # 1) 회수된 CSV 목록 표시
#   .\tools\push_replay.ps1 -List
#
#   # 2) 특정 CSV 를 latest.csv 로 push
#   .\tools\push_replay.ps1 -CsvName "imu_record_1779218552206.csv"
#
#   # 3) 가장 최근 CSV 자동 선택
#   .\tools\push_replay.ps1 -Latest
#
# 단말 경로 (HANDOFF_P44 §1.7):
#   /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv

param(
    [string]$CsvName,
    [switch]$Latest,
    [switch]$List,

    [string]$DeviceSerial = "adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp",
    [string]$AdbPath      = "D:\SDK\platform-tools\adb.exe"
)

$ErrorActionPreference = "Stop"

# ── 경로 ──────────────────────────────────────────────────────
$projectRoot   = Resolve-Path (Join-Path $PSScriptRoot "..")
$localCsvDir   = Join-Path $projectRoot "csv\imu_csv"
$remoteDir     = "/sdcard/Android/data/com.imulocal/files/imu_csv/replay"
$remoteFile    = "$remoteDir/latest.csv"

if (-not (Test-Path $localCsvDir)) {
    throw "csv/imu_csv 디렉토리가 없습니다. 먼저 .\tools\collect.ps1 로 측정 회수하세요: $localCsvDir"
}

# ── 회수된 CSV 목록 ──────────────────────────────────────────
$csvs = Get-ChildItem -Path $localCsvDir -Filter "imu_record_*.csv" |
        Sort-Object -Property LastWriteTime -Descending

if (-not $csvs) {
    throw "회수된 imu_record_*.csv 파일이 없습니다: $localCsvDir"
}

# ── -List: 목록 표시 후 종료 ─────────────────────────────────
if ($List) {
    Write-Host "회수된 CSV (최신 순):" -ForegroundColor Cyan
    $csvs | ForEach-Object {
        $sizeKb = [math]::Round($_.Length / 1KB, 1)
        Write-Host ("  {0,-50} {1,8} KB  {2}" -f $_.Name, $sizeKb, $_.LastWriteTime) -ForegroundColor White
    }
    exit 0
}

# ── 대상 CSV 결정 ────────────────────────────────────────────
$target = $null
if ($Latest) {
    $target = $csvs[0]
    Write-Host "최신 CSV 자동 선택: $($target.Name)" -ForegroundColor Cyan
}
elseif ($CsvName) {
    $target = $csvs | Where-Object { $_.Name -eq $CsvName }
    if (-not $target) {
        Write-Host "지정한 CSV 를 찾지 못함: $CsvName" -ForegroundColor Red
        Write-Host "사용 가능한 목록:" -ForegroundColor Yellow
        $csvs | ForEach-Object { Write-Host "  $($_.Name)" }
        exit 1
    }
}
else {
    Write-Host "사용법:" -ForegroundColor Yellow
    Write-Host "  .\tools\push_replay.ps1 -List"
    Write-Host "  .\tools\push_replay.ps1 -Latest"
    Write-Host "  .\tools\push_replay.ps1 -CsvName <파일명>"
    exit 1
}

# ── adb 연결 확인 ────────────────────────────────────────────
if (-not (Test-Path $AdbPath)) {
    throw "adb.exe 를 찾을 수 없습니다: $AdbPath"
}
$devices = & $AdbPath devices
if ($devices -notmatch $DeviceSerial) {
    Write-Warning "지정 단말 '$DeviceSerial' 가 연결 목록에 없습니다."
    Write-Host $devices
    exit 1
}

# ── push ──────────────────────────────────────────────────────
Write-Host "[1/3] replay 디렉토리 생성: $remoteDir" -ForegroundColor Cyan
& $AdbPath -s $DeviceSerial shell mkdir -p $remoteDir | Out-Null

Write-Host "[2/3] push: $($target.Name) → $remoteFile" -ForegroundColor Cyan
& $AdbPath -s $DeviceSerial push $target.FullName $remoteFile

Write-Host "[3/3] 확인" -ForegroundColor Cyan
$verifyOut = & $AdbPath -s $DeviceSerial shell ls -la $remoteFile
Write-Host "  $verifyOut" -ForegroundColor Gray

Write-Host ""
Write-Host "=== push 완료 ===" -ForegroundColor Green
Write-Host "단말 replay 모드 진입 시 위 latest.csv 가 입력으로 사용됩니다." -ForegroundColor Green
Write-Host "(replay 기능은 ImuCollector.kt 에 별도 구현 필요 — HANDOFF §1.7 참조)" -ForegroundColor DarkYellow
