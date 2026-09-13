# Freeze the backend with PyInstaller on Windows -> packaging\dist\sentinel-backend\sentinel-backend.exe
# Twin of packaging/build_backend.sh (macOS). PyInstaller does not cross-compile: run this ON Windows
# (a developer PC, or the windows-latest job in .github/workflows/build-windows.yml).
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_backend.ps1
#   $env:PYTHON = 'C:\path\to\venv\Scripts\python.exe'   # optional; default: python on PATH
#
# Runs under Windows PowerShell 5.1 and PowerShell 7 (pwsh). Output is kept to WARN level so the
# CI log stays readable; the launcher's existence is the pass/fail gate, same as the .sh twin.
$ErrorActionPreference = 'Stop'
# PowerShell 7.4+ can turn any non-zero native exit code into a terminating error; exit codes
# are checked explicitly below instead (harmless no-op on 5.1).
$PSNativeCommandUseErrorActionPreference = $false

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
$Py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

# Native stderr is never redirected here: Windows PowerShell 5.1 would wrap redirected stderr
# lines in ErrorRecords and, with ErrorActionPreference=Stop, abort on pip's first warning.
& $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller into $Py"
    & $Py -m pip install -q pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed (exit $LASTEXITCODE)" }
}

& $Py -m PyInstaller --noconfirm --clean --log-level WARN packaging/sentinel_backend.spec `
    --distpath packaging/dist --workpath packaging/build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

$Exe = Join-Path $Root 'packaging\dist\sentinel-backend\sentinel-backend.exe'
if (-not (Test-Path -LiteralPath $Exe)) { throw "freeze failed - $Exe not found" }
$BundleDir = Split-Path -Parent $Exe
$Bytes = (Get-ChildItem -LiteralPath $BundleDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("bundle size: {0:N0} MB  ({1})" -f ($Bytes / 1MB), $BundleDir)
