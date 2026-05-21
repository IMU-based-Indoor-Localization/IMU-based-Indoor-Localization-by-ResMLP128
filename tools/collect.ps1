# tools/collect.ps1 — IMU 측정 회수 스크립트 (PowerShell)
#
# 사용:
#   .\tools\collect.ps1 -SessionName "baseline_01"
#   .\tools\collect.ps1 -SessionName "baseline_01" -DeviceSerial "adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp"
#
# 동작:
#   1. adb 연결 확인
#   2. 단말 logcat 버퍼 클리어
#   3. 백그라운드 logcat 캡쳐 시작 (UTF-8, logs/<세션명>.txt)
#   4. 사용자가 측정 종료할 때까지 대기 (Enter 입력)
#   5. logcat 캡쳐 종료
#   6. 단말 imu_csv 폴더 전체 PC 로 pull (csv/imu_csv/)
#   7. 회수된 파일 목록 요약 출력
#
# 사전조건:
#   - D:\SDK\platform-tools\adb.exe 존재
#   - 단말 USB 디버깅 활성화 + 무선 ADB 연결
#   - com.imulocal 앱에서 ImuTestActivity 진입 후 측정 시작 상태

param(
    [Parameter(Mandatory=$true)]
    [string]$SessionName,

    [string]$DeviceSerial = "adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp",

    [string]$AdbPath = "D:\SDK\platform-tools\adb.exe"
)

$ErrorActionPreference = "Stop"

# ── 경로 ──────────────────────────────────────────────────────
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$logsDir     = Join-Path $projectRoot "logs"
$csvDir      = Join-Path $projectRoot "csv"
$logFile     = Join-Path $logsDir "$SessionName.txt"

if (-not (Test-Path $logsDir)) { New-Item -Path $logsDir -ItemType Directory | Out-Null }
if (-not (Test-Path $csvDir))  { New-Item -Path $csvDir  -ItemType Directory | Out-Null }

# ── 1. adb 연결 확인 ─────────────────────────────────────────
Write-Host "[1/6] adb 연결 확인" -ForegroundColor Cyan
if (-not (Test-Path $AdbPath)) {
    throw "adb.exe 를 찾을 수 없습니다: $AdbPath"
}

$devices = & $AdbPath devices
Write-Host $devices
if ($devices -notmatch $DeviceSerial) {
    Write-Warning "지정 단말 '$DeviceSerial' 가 연결 목록에 없습니다."
    Write-Warning "필요시 'adb connect <IP>:<PORT>' 또는 USB 케이블 연결 확인."
    $continue = Read-Host "그대로 진행할까요? (y/N)"
    if ($continue -ne "y") { exit 1 }
}

# ── 2. logcat 버퍼 클리어 ───────────────────────────────────
Write-Host "[2/6] logcat 버퍼 클리어" -ForegroundColor Cyan
& $AdbPath -s $DeviceSerial logcat -c

# ── 3. 백그라운드 logcat 캡쳐 시작 ─────────────────────────
Write-Host "[3/6] logcat 캡쳐 시작 → $logFile" -ForegroundColor Cyan
# UTF-8 출력 강제 (Out-File 의 기본 UTF-16 회피)
$logcatJob = Start-Job -ScriptBlock {
    param($adb, $serial, $output)
    & $adb -s $serial logcat | Out-File -Encoding utf8 -FilePath $output
} -ArgumentList $AdbPath, $DeviceSerial, $logFile

Start-Sleep -Seconds 1
if ($logcatJob.State -ne "Running") {
    throw "logcat 캡쳐 job 시작 실패: $($logcatJob.State)"
}
Write-Host "  Job ID: $($logcatJob.Id)" -ForegroundColor Gray

# ── 4. 측정 종료 대기 ───────────────────────────────────────
Write-Host ""
Write-Host "[4/6] 단말에서 측정 진행중..." -ForegroundColor Yellow
Write-Host "      측정 완료 후 Enter 키를 누르세요." -ForegroundColor Yellow
Read-Host "      [Enter 대기]"

# ── 5. logcat 캡쳐 종료 ─────────────────────────────────────
Write-Host "[5/6] logcat 캡쳐 종료" -ForegroundColor Cyan
Stop-Job  -Job $logcatJob -ErrorAction SilentlyContinue
Remove-Job -Job $logcatJob -Force -ErrorAction SilentlyContinue

if (Test-Path $logFile) {
    $logSize = (Get-Item $logFile).Length
    Write-Host "  logcat 파일 크기: $([math]::Round($logSize / 1KB, 1)) KB" -ForegroundColor Gray
}

# ── 6. 단말 imu_csv 폴더 pull ───────────────────────────────
Write-Host "[6/6] 단말 imu_csv 폴더 pull → csv/" -ForegroundColor Cyan
& $AdbPath -s $DeviceSerial pull /sdcard/Android/data/com.imulocal/files/imu_csv $csvDir

# ── 요약 ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== 회수 완료 ===" -ForegroundColor Green
$pulledCsvs = Get-ChildItem -Path (Join-Path $csvDir "imu_csv") -Filter "imu_record_*.csv" -ErrorAction SilentlyContinue |
              Sort-Object -Property LastWriteTime -Descending
if ($pulledCsvs) {
    Write-Host "회수된 CSV (최신 순):" -ForegroundColor Green
    $pulledCsvs | Select-Object -First 5 | ForEach-Object {
        $sizeKb = [math]::Round($_.Length / 1KB, 1)
        Write-Host ("  {0,-50} {1,8} KB  {2}" -f $_.Name, $sizeKb, $_.LastWriteTime) -ForegroundColor White
    }
}
Write-Host ""
Write-Host "logcat: $logFile" -ForegroundColor Green
Write-Host "CSV:    $csvDir\imu_csv\" -ForegroundColor Green
Write-Host ""
Write-Host "다음 단계: .\tools\push_replay.ps1 -CsvName <imu_record_xxx.csv>" -ForegroundColor Yellow
