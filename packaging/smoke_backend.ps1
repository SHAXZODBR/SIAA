# Smoke-test the frozen Windows backend (packaging\dist\sentinel-backend\sentinel-backend.exe)
# the way the desktop app starts it: spawn the exe, poll /health, assert status ok|degraded,
# print the model table, kill it. PowerShell twin of packaging/smoke_backend.sh (macOS).
# Used by .github/workflows/build-windows.yml right after the freeze; runs on a developer PC too:
#
#     powershell -ExecutionPolicy Bypass -File packaging\smoke_backend.ps1
#     $env:SENTINEL_MODELS_DIR = 'D:\models'; $env:SMOKE_PORT = '8765'; $env:SMOKE_TIMEOUT = '300'
#
# $env:REQUIRE_VALIDATED = '1' additionally demands status 'ok' (brain_triage loaded) - CI sets it
# when the MODEL_BUNDLE_URL bundle was restored, so a bundle the frozen backend cannot load never
# ships silently. Nothing is written outside a throw-away SENTINEL_DATA_DIR.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Exe = Join-Path $Root 'packaging\dist\sentinel-backend\sentinel-backend.exe'
if (-not (Test-Path -LiteralPath $Exe)) { throw "$Exe not found - run packaging\build_backend.ps1 first" }
$Port = if ($env:SMOKE_PORT) { [int]$env:SMOKE_PORT } else { 8765 }
$TimeoutS = if ($env:SMOKE_TIMEOUT) { [int]$env:SMOKE_TIMEOUT } else { 300 }   # first launch of a FRESH bundle can exceed 120 s (Defender scans every DLL once; a warm launch is much faster)

$DataDir = Join-Path ([System.IO.Path]::GetTempPath()) ('sentinel-smoke-data-' + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
$Log = Join-Path $DataDir 'backend-smoke.log'
$ErrLog = Join-Path $DataDir 'backend-smoke.err.log'

$env:SENTINEL_DEV_INSECURE = '1'                    # /health is public; no admin bootstrap needed
$env:SENTINEL_OFFLINE = '1'                         # hospital posture: the bundle only, never a download
$env:SENTINEL_DATA_DIR = $DataDir
if (-not $env:SENTINEL_MODELS_DIR) { $env:SENTINEL_MODELS_DIR = Join-Path $Root 'models' }
Write-Host "smoke: $Exe --host 127.0.0.1 --port $Port  (models: $env:SENTINEL_MODELS_DIR, data: $DataDir)"

$Proc = Start-Process -FilePath $Exe -ArgumentList @('--host', '127.0.0.1', '--port', "$Port") `
    -PassThru -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
try {
    $Deadline = (Get-Date).AddSeconds($TimeoutS)
    $Health = $null
    while ((Get-Date) -lt $Deadline) {
        if ($Proc.HasExited) { break }
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
            break
        } catch {
            Start-Sleep -Seconds 3
        }
    }
    if ($null -eq $Health) {
        Write-Host '---- backend log (tail) ----'
        Get-Content -Path $Log, $ErrLog -ErrorAction SilentlyContinue | Select-Object -Last 80
        if ($Proc.HasExited) { throw "backend exited (code $($Proc.ExitCode)) before /health answered" }
        throw "/health did not answer within $TimeoutS s"
    }

    Write-Host ('status={0} version={1} offline={2} device={3}' -f $Health.status, $Health.version, $Health.offline, $Health.device)
    foreach ($entry in $Health.models.PSObject.Properties) {
        $m = $entry.Value
        $state = if ($m.loaded) { 'loaded' } else { 'NOT loaded' }
        Write-Host ('  {0,-20} {1,-10} {2}' -f $entry.Name, $state, $m.reason)
    }
    if ($Health.status -notin @('ok', 'degraded')) { throw "unexpected /health status: $($Health.status)" }
    if ($env:REQUIRE_VALIDATED -eq '1' -and $Health.status -ne 'ok') {
        throw "validated triage model was restored but /health is not 'ok' (brain_triage did not load)"
    }
    Write-Host 'SMOKE OK'
} finally {
    if (-not $Proc.HasExited) { Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue }
}
