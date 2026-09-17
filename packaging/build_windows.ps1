# ============================================================================
#  Sentinel Medical AI - ONE-SHOT WINDOWS BUILD (fresh Windows 10/11 x64 laptop)
#
#      cd C:\SIAA\sentinel
#      powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
#  Runs as a NORMAL user (no Administrator). Needs internet for pip / npm /
#  electron-builder downloads; the models are NOT downloaded when the kit
#  already ships them in models\ (that is the normal case). What it does:
#
#     1. checks Python 3.11 (64-bit) and Node.js 20 LTS - prints the exact
#        winget / download instruction when one is missing and stops
#     2. creates .\venv and installs requirements-win.txt (CPU torch) + PyInstaller
#     3. freezes the backend -> packaging\dist\sentinel-backend\sentinel-backend.exe
#        (packaging\build_backend.ps1)
#     4. checks models\ (validated triage model, tumor fine-tune, hf\ bundles,
#        xrv\, densenet\); downloads the PUBLIC ones only when something is
#        missing AND the internet is reachable, otherwise tells you what to copy
#     5. smoke-tests the frozen exe (packaging\smoke_backend.ps1, REQUIRE_VALIDATED=1)
#     6. desktop-app: npm ci ; npm run build:win  -> desktop-app\release\*.exe
#     7. prints the installer path + size
#
#  Every line goes to packaging\build_windows.log. On failure the last 40 log
#  lines are printed with a "send this file to the developer" hint.
#  Idempotent: re-running skips the steps that are already done (venv with the
#  right torch, exe newer than src\, models present, node_modules from the
#  lockfile, installer newer than the exe). $env:FORCE_REBUILD = '1' redoes
#  everything; $env:SKIP_SMOKE = '1' skips step 5; $env:PYTHON = 'C:\...\python.exe'
#  picks the interpreter used to CREATE the venv.
#
#  No code signing: Windows SmartScreen shows "Windows protected your PC" the
#  first time the installer runs -> "More info" -> "Run anyway".
#  Works under Windows PowerShell 5.1 (the one every Windows has) and PowerShell 7.
# ============================================================================
$ErrorActionPreference = 'Stop'
# PowerShell 7.4+ can turn any non-zero native exit code into a terminating error;
# exit codes are checked explicitly (harmless no-op on 5.1).
$PSNativeCommandUseErrorActionPreference = $false

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
$Log = Join-Path $Root 'packaging\build_windows.log'
$Force = ($env:FORCE_REBUILD -eq '1')

# Mirror .github/workflows/build-windows.yml
$TorchVersion = '2.11.0'
$TorchVisionVersion = '0.26.0'
$PyInstallerPin = 'pyinstaller==6.22.3'
$PublicModelKeys = 'brain_tumor_class,chest,head_ct,chest_checkpoint,generic_processors'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'                 # download_all_models.py prints non-ASCII; consoles default to cp1251/cp866
$env:PYTHONIOENCODING = 'utf-8'
$env:CSC_IDENTITY_AUTO_DISCOVERY = 'false'   # electron-builder: never look for a certificate
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
# Windows PowerShell 5.1 on an older Windows 10 build may still default to TLS 1.0 for
# Invoke-WebRequest; pypi.org / github.com / huggingface.co require TLS 1.2.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
$script:StepName = 'preflight'

function Banner([string]$Text) {
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host ("  [{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Text)
    Write-Host ('=' * 78)
}

function Step([string]$Name) { $script:StepName = $Name; Banner $Name }

# Run a native command line through cmd.exe with stderr merged into stdout, so
# BOTH streams show up live on the console AND in the transcript log, and no
# stderr line is ever turned into a terminating ErrorRecord (Windows PowerShell
# 5.1 would do that with ErrorActionPreference=Stop and a 2>&1 redirect).
# The command line travels in an environment variable: cmd expands it itself,
# so no PowerShell-version-specific argument quoting is involved.
function Invoke-Native([string]$What, [string]$CmdLine, [string]$WorkDir = $Root) {
    Write-Host ''
    Write-Host ("  > {0}" -f $What)
    Write-Host ('    $ {0}' -f $CmdLine)
    $env:SENTINEL_STEP_CMD = $CmdLine
    Push-Location $WorkDir
    try {
        # /s: cmd strips exactly the outer quotes PowerShell adds around the argument
        & $env:ComSpec /d /s /c "%SENTINEL_STEP_CMD% 2>&1"
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        Remove-Item Env:\SENTINEL_STEP_CMD -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) { throw "$What failed (exit code $code)" }
}

# Output of a native command as a string (for version probes); $null when the
# command is missing, fails or writes nothing. ErrorActionPreference is relaxed
# inside (function scope only): under Windows PowerShell 5.1 a redirected stderr
# line would otherwise become a terminating error.
function Probe([string]$Exe, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return $null }
    try {
        $global:LASTEXITCODE = 0
        $out = & $Exe @Arguments 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $out) { return $null }
        $text = ($out | Out-String).Trim()
        if (-not $text) { return $null }
        return $text
    } catch { return $null }
}

function Q([string]$Path) { return '"' + $Path + '"' }   # quote for a cmd.exe command line

function NewestWrite([string[]]$Paths) {
    $t = [DateTime]::MinValue
    foreach ($p in $Paths) {
        if (-not (Test-Path -LiteralPath $p)) { continue }
        $items = if ((Get-Item -LiteralPath $p).PSIsContainer) { Get-ChildItem -LiteralPath $p -Recurse -File } else { Get-Item -LiteralPath $p }
        foreach ($i in $items) { if ($i.LastWriteTimeUtc -gt $t) { $t = $i.LastWriteTimeUtc } }
    }
    return $t
}

function SizeMB([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $item = Get-Item -LiteralPath $Path
    $bytes = if ($item.PSIsContainer) { (Get-ChildItem -LiteralPath $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum } else { $item.Length }
    if (-not $bytes) { $bytes = 0 }
    return [math]::Round($bytes / 1MB)
}

function Test-Internet {
    foreach ($u in @('https://pypi.org/simple/pip/', 'https://huggingface.co', 'https://github.com')) {
        try {
            $r = Invoke-WebRequest -Uri $u -Method Head -UseBasicParsing -TimeoutSec 8
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { return $true }
        } catch { }
    }
    return $false
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
$Failed = $null
try { Start-Transcript -Path $Log -Append | Out-Null } catch { Write-Host "WARNING: could not start transcript at $Log ($($_.Exception.Message))" }
$Started = Get-Date
try {
    Banner "Sentinel Medical AI - Windows build  ($Root)"
    Write-Host ("  log: {0}" -f $Log)
    Write-Host ('  PowerShell {0}   Windows {1}   64-bit OS: {2}   FORCE_REBUILD={3}' -f $PSVersionTable.PSVersion, [Environment]::OSVersion.Version, [Environment]::Is64BitOperatingSystem, $Force)

    # ---------------------------------------------------------------- preflight
    Step '0/7  preflight (OS, disk, RAM)'
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'a 64-bit Windows is required (this is 32-bit)' }
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        throw "CPU architecture $env:PROCESSOR_ARCHITECTURE is not x64 (ARM laptops are not supported by this build)"
    }
    $drive = (Get-Item -LiteralPath $Root).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host ("  free space on {0}: {1} GB  (need ~15 GB: venv 3 + PyInstaller 4 + node_modules 1 + installer 4)" -f $drive.Root, $freeGB)
    if ($freeGB -lt 8) { throw "only $freeGB GB free on $($drive.Root) - free at least 15 GB and re-run" }
    if ($freeGB -lt 15) { Write-Host '  WARNING: less than 15 GB free - the build may run out of disk in the last step' }
    try {
        $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        Write-Host ("  RAM: {0} GB" -f $ramGB)
        if ($ramGB -lt 16) { Write-Host '  WARNING: 16 GB RAM recommended (PyInstaller + the ViT models are memory hungry)' }
    } catch { }
    if ($Root -match '\s') { Write-Host "  WARNING: the repo path contains spaces ($Root) - C:\SIAA\sentinel is the tested layout" }
    if (-not (Test-Path -LiteralPath (Join-Path $Root 'requirements-win.txt'))) { throw "requirements-win.txt not found in $Root - run this from the repo root of the kit" }

    # ---------------------------------------------------------------- python
    Step '1/7  Python 3.11 (64-bit)'
    $PyHelp = @(
        '',
        '  Python 3.11 (64-bit) was not found. Install it ONCE (internet needed), then open a NEW',
        '  PowerShell window and re-run this script. Either:',
        '',
        '      winget install --id Python.Python.3.11 -e --source winget',
        '',
        '  or download  https://www.python.org/downloads/release/python-3119/',
        '      -> "Windows installer (64-bit)"  -> tick "Add python.exe to PATH" on the first screen.',
        '',
        '  (Another Python version installed? Set  $env:PYTHON = "C:\path\to\python311\python.exe"  and re-run.)',
        ''
    )
    $BasePy = $null
    $candidates = @()
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    $pyLauncher = Probe 'py' @('-3.11', '-c', 'import sys; print(sys.executable)')
    if ($pyLauncher) { $candidates += $pyLauncher.Split("`n")[-1].Trim() }
    $candidates += @('python', 'python3.11', 'python3')
    foreach ($c in $candidates) {
        if (-not $c) { continue }
        # (no double quotes inside -c code: Windows PowerShell 5.1 passes embedded quotes to natives unescaped)
        $v = Probe $c @('-c', "import sys, struct; print('%d.%d.%d %d' % (sys.version_info[:3] + (struct.calcsize('P') * 8,)))")
        if (-not $v) { continue }
        $parts = $v.Split("`n")[-1].Trim().Split(' ')
        if ($parts[0].StartsWith('3.11.') -and $parts[1] -eq '64') {
            $BasePy = if ($c -match '[\\/]') { $c } else { (Get-Command $c).Source }
            Write-Host ("  found Python {0} (64-bit): {1}" -f $parts[0], $BasePy)
            break
        } else {
            Write-Host ("  skipping {0}: Python {1} {2}-bit (need 3.11.x 64-bit)" -f $c, $parts[0], $parts[1])
        }
    }
    if (-not $BasePy) { $PyHelp | ForEach-Object { Write-Host $_ }; throw 'Python 3.11 (64-bit) not found' }

    # ---------------------------------------------------------------- node
    Step '2/7  Node.js 20 LTS + npm'
    $NodeHelp = @(
        '',
        '  Node.js 20 LTS (20.19 or newer) was not found. 22.12+ also works; 24 does NOT (npm ci would',
        '  try to compile better-sqlite3 with Visual Studio). Install it ONCE (internet needed), then',
        '  open a NEW PowerShell window and re-run this script. Either:',
        '',
        '      winget install --id OpenJS.NodeJS.LTS -e --source winget      (installs the current LTS)',
        '',
        '  or download  https://nodejs.org/dist/latest-v20.x/   -> node-v20.x.x-x64.msi  (defaults are fine).',
        ''
    )
    $nodeV = Probe 'node' @('--version')
    if (-not $nodeV) { $NodeHelp | ForEach-Object { Write-Host $_ }; throw 'node not found on PATH' }
    $nv = $nodeV.Trim().TrimStart('v').Split('.')
    $nodeMajor = [int]$nv[0]; $nodeMinor = [int]$nv[1]
    # vite 8 needs 20.19+ / 22.12+; better-sqlite3 11.10 has prebuilt binaries for Node 20 and 22
    # only (ABI 115/127) - with 24 `npm ci` falls back to node-gyp and fails without MSVC.
    $nodeOk = ($nodeMajor -eq 20 -and $nodeMinor -ge 19) -or ($nodeMajor -eq 22 -and $nodeMinor -ge 12)
    if (-not $nodeOk) { $NodeHelp | ForEach-Object { Write-Host $_ }; throw "Node.js $nodeV is not supported by this build (need 20.19+ or 22.12+)" }
    if ($nodeMajor -ne 20) { Write-Host "  NOTE: Node.js $nodeV found - the CI build uses Node 20 LTS; 22 works too" }
    $npmV = Probe 'npm' @('--version')
    if (-not $npmV) { $NodeHelp | ForEach-Object { Write-Host $_ }; throw 'npm not found on PATH (comes with Node.js)' }
    Write-Host ("  node {0}   npm {1}" -f $nodeV.Trim(), $npmV.Trim())

    # ---------------------------------------------------------------- venv + deps
    Step '3/7  Python venv (.\venv) + requirements-win.txt + PyInstaller'
    $Py = Join-Path $Root 'venv\Scripts\python.exe'
    if ($Force -and (Test-Path -LiteralPath (Join-Path $Root 'venv'))) {
        Write-Host '  FORCE_REBUILD=1: removing .\venv'
        Remove-Item -Recurse -Force (Join-Path $Root 'venv')
    }
    if (-not (Test-Path -LiteralPath $Py)) {
        Invoke-Native 'create venv' ((Q $BasePy) + ' -m venv venv')
        if (-not (Test-Path -LiteralPath $Py)) { throw "venv creation did not produce $Py" }
    } else {
        Write-Host "  venv exists: $Py"
    }
    $probeCode = 'import torch, torchvision, transformers, fastapi, uvicorn, pydicom, pylibjpeg, gdcm, monai, torchxrayvision, imageio, cv2, skimage, scipy, PIL, loguru, cryptography, bcrypt, jose, yaml, nibabel, PyInstaller; print(torch.__version__, torchvision.__version__, PyInstaller.__version__)'
    $depState = Probe $Py @('-c', $probeCode)
    $depsOk = $depState -and $depState.Split("`n")[-1].Trim().StartsWith($TorchVersion)
    if ($depsOk) {
        Write-Host ("  dependencies already installed ({0}) - skipping pip" -f $depState.Split("`n")[-1].Trim())
    } else {
        if (-not (Test-Internet)) { throw 'internet is required to install the Python dependencies (pip) - connect and re-run' }
        Invoke-Native 'upgrade pip + wheel' ((Q $Py) + ' -m pip install --upgrade pip wheel')
        Invoke-Native 'pip install -r requirements-win.txt (CPU torch from the PyTorch index; 10-20 min)' ((Q $Py) + ' -m pip install -r requirements-win.txt')
        Invoke-Native "pip install $PyInstallerPin" ((Q $Py) + " -m pip install $PyInstallerPin")
    }
    $tv = Probe $Py @('-c', 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())')
    if (-not $tv) { throw 'torch does not import inside .\venv - see the pip output above' }
    $tv = $tv.Split("`n")[-1].Trim()
    Write-Host ("  torch/torchvision/cuda: {0}" -f $tv)
    if (-not $tv.StartsWith($TorchVersion)) { throw "torch resolved to '$tv', expected $TorchVersion (a requirement re-pinned it; fix requirements-win.txt)" }
    if (-not $tv.Contains(" $TorchVisionVersion")) { throw "torchvision is not $TorchVisionVersion ('$tv')" }
    $imp = Probe $Py @('-c', $probeCode)
    if (-not $imp) { throw 'a runtime package is missing inside .\venv (see requirements-win.txt) - re-run with FORCE_REBUILD=1' }
    $env:PYTHON = $Py

    # ---------------------------------------------------------------- freeze
    Step '4/7  Freeze the backend (PyInstaller -> packaging\dist\sentinel-backend\sentinel-backend.exe)'
    $Exe = Join-Path $Root 'packaging\dist\sentinel-backend\sentinel-backend.exe'
    $srcStamp = NewestWrite @((Join-Path $Root 'src'), (Join-Path $Root 'run_server.py'), (Join-Path $Root 'configs'),
                              (Join-Path $Root 'packaging\sentinel_backend.spec'), (Join-Path $Root 'requirements-win.txt'))
    $exeFresh = (Test-Path -LiteralPath $Exe) -and ((Get-Item -LiteralPath $Exe).LastWriteTimeUtc -gt $srcStamp)
    if ($exeFresh -and -not $Force) {
        Write-Host ("  frozen backend is up to date ({0} MB, {1}) - skipping PyInstaller" -f (SizeMB (Split-Path -Parent $Exe)), $Exe)
    } else {
        if (Test-Path -LiteralPath (Join-Path $Root 'packaging\dist\sentinel-backend')) {
            Write-Host '  removing the previous packaging\dist\sentinel-backend'
            Remove-Item -Recurse -Force (Join-Path $Root 'packaging\dist\sentinel-backend')
        }
        Invoke-Native 'PyInstaller freeze (5-15 min; Defender scanning slows it down)' `
            ('powershell -NoProfile -ExecutionPolicy Bypass -File ' + (Q (Join-Path $Root 'packaging\build_backend.ps1')))
        if (-not (Test-Path -LiteralPath $Exe)) { throw "freeze finished but $Exe is missing" }
        Write-Host '  freeing the PyInstaller work dir (packaging\build)'
        Remove-Item -Recurse -Force (Join-Path $Root 'packaging\build') -ErrorAction SilentlyContinue
    }

    # ---------------------------------------------------------------- models
    Step '5/7  Models (models\ next to the exe; shipped in the kit, never downloaded when present)'
    $Models = Join-Path $Root 'models'
    $Kit = Split-Path -Parent $Root                       # C:\SIAA  (the kit root, if this IS the kit)
    $required = [ordered]@{
        'models\brain_triage_finetuned\MANIFEST.json'      = 'validated Uzbek triage model (VENDOR-ONLY, not downloadable)'
        'models\brain_triage_finetuned\model.safetensors'  = 'validated Uzbek triage model weights'
        'models\brain_finetuned\model.safetensors'         = 'brain tumor fine-tune (VENDOR-ONLY, not downloadable)'
    }
    $public = [ordered]@{
        'models\hf\andrei-teodor__resnet-pretrained-brain-mri\config.json'       = 'brain_tumor_class (public HF)'
        'models\hf\DifeiT__rsna-intracranial-hemorrhage-detection\config.json'   = 'head_ct (public HF)'
        'models\hf\google__vit-base-patch16-224\preprocessor_config.json'        = 'generic ViT processor'
        'models\hf\microsoft__resnet-50\preprocessor_config.json'                = 'generic ResNet processor'
        'models\densenet\best_model.pt'                                          = 'chest (legacy DenseNet wrapper)'
    }
    $missingVendor = @(); $missingPublic = @()
    foreach ($k in $required.Keys) { $ok = Test-Path -LiteralPath (Join-Path $Root $k); Write-Host ("  {0} {1,-72} {2}" -f ($(if ($ok) {'OK     '} else {'MISSING'}), $k, $required[$k])); if (-not $ok) { $missingVendor += $k } }
    foreach ($k in $public.Keys)   { $ok = Test-Path -LiteralPath (Join-Path $Root $k); Write-Host ("  {0} {1,-72} {2}" -f ($(if ($ok) {'OK     '} else {'MISSING'}), $k, $public[$k]));   if (-not $ok) { $missingPublic += $k } }
    $xrvOk = (Test-Path -LiteralPath (Join-Path $Models 'xrv')) -and ((Get-ChildItem -LiteralPath (Join-Path $Models 'xrv') -Filter *.pt -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
    Write-Host ("  {0} {1,-72} {2}" -f ($(if ($xrvOk) {'OK     '} else {'MISSING'}), 'models\xrv\*.pt', 'chest (torchxrayvision weights)'))
    if (-not $xrvOk) { $missingPublic += 'models\xrv\*.pt' }

    if ($missingVendor.Count -gt 0) {
        Write-Host ''
        Write-Host '  The vendor models cannot be downloaded. Copy them from the kit into this repo:'
        Write-Host ("      robocopy {0} {1} /E" -f (Join-Path $Kit 'models'), $Models)
        Write-Host '  (or from the developer: models\brain_triage_finetuned and models\brain_finetuned), then re-run.'
        throw ('missing vendor model files: ' + ($missingVendor -join ', '))
    }
    if ($missingPublic.Count -gt 0) {
        if (Test-Path -LiteralPath (Join-Path $Kit 'models\hf')) {
            Write-Host ''
            Write-Host ("  The kit next to this repo has a models\ folder - copy it first, no internet needed:")
            Write-Host ("      robocopy {0} {1} /E" -f (Join-Path $Kit 'models'), $Models)
        }
        if (Test-Internet) {
            Write-Host '  internet is reachable - downloading ONLY the missing public models (torchxrayvision / HuggingFace)'
            Invoke-Native "download public models ($PublicModelKeys)" `
                ((Q $Py) + ' scripts\download_all_models.py --dest models --only ' + $PublicModelKeys + ' --verify')
        } else {
            Write-Host ''
            Write-Host '  NO internet and public model files are missing. On a PC WITH internet run, inside a copy of this repo:'
            Write-Host ("      python scripts\download_all_models.py --dest models --only {0}" -f $PublicModelKeys)
            Write-Host ("  then copy its models\hf, models\xrv, models\densenet here: {0}" -f $Models)
            throw ('missing public model files (offline): ' + ($missingPublic -join ', '))
        }
    }
    Invoke-Native 'verify the validated model against MANIFEST.json (sha256)' `
        ((Q $Py) + ' packaging\fetch_model_bundle.py --dest models --verify-only --require')
    Write-Host ("  models\ total: {0} MB" -f (SizeMB $Models))

    # ---------------------------------------------------------------- smoke
    Step '6/7  Smoke-test the frozen backend (/health must be ok = brain_triage loaded)'
    $SmokeMarker = Join-Path $Root 'packaging\dist\sentinel-backend\.smoke_ok'
    $smokeFresh = (Test-Path -LiteralPath $SmokeMarker) -and ((Get-Item -LiteralPath $SmokeMarker).LastWriteTimeUtc -gt (Get-Item -LiteralPath $Exe).LastWriteTimeUtc)
    if ($env:SKIP_SMOKE -eq '1') {
        Write-Host '  SKIP_SMOKE=1 - skipped'
    } elseif ($smokeFresh -and -not $Force) {
        Write-Host '  smoke test already passed for this exe - skipping (delete packaging\dist\sentinel-backend\.smoke_ok to redo)'
    } else {
        $env:REQUIRE_VALIDATED = '1'
        $env:SENTINEL_MODELS_DIR = $Models
        if (-not $env:SMOKE_TIMEOUT) { $env:SMOKE_TIMEOUT = '420' }   # first launch: Defender scans ~3,000 DLLs once
        Invoke-Native 'smoke_backend.ps1 (first launch can take 2-5 min - Defender scans every DLL once)' `
            ('powershell -NoProfile -ExecutionPolicy Bypass -File ' + (Q (Join-Path $Root 'packaging\smoke_backend.ps1')))
        Set-Content -Path $SmokeMarker -Value (Get-Date -Format o) -Encoding ASCII
        Remove-Item Env:\REQUIRE_VALIDATED -ErrorAction SilentlyContinue
    }

    # ---------------------------------------------------------------- desktop app
    Step '7/7  Desktop app: npm ci ; npm run build:win (electron-builder NSIS x64)'
    $App = Join-Path $Root 'desktop-app'
    $Release = Join-Path $App 'release'
    $lock = Join-Path $App 'package-lock.json'
    $nmStamp = Join-Path $App 'node_modules\.package-lock.json'
    $nmFresh = (Test-Path -LiteralPath $nmStamp) -and ((Get-Item -LiteralPath $nmStamp).LastWriteTimeUtc -ge (Get-Item -LiteralPath $lock).LastWriteTimeUtc)
    if ($nmFresh -and -not $Force) {
        Write-Host '  node_modules matches package-lock.json - skipping npm ci'
    } else {
        if (-not (Test-Internet)) { throw 'internet is required for npm ci (Electron + node_modules download) - connect and re-run' }
        Invoke-Native 'npm ci (downloads Electron; 2-5 min)' 'npm ci --no-audit --no-fund' $App
    }
    $installers = @()
    if (Test-Path -LiteralPath $Release) { $installers = @(Get-ChildItem -LiteralPath $Release -Filter *.exe -File -ErrorAction SilentlyContinue) }
    $appStamp = NewestWrite @((Join-Path $App 'src'), (Join-Path $App 'electron'), (Join-Path $App 'package.json'), (Join-Path $App 'build'), $Exe, $Models)
    $instFresh = ($installers.Count -gt 0) -and (($installers | Sort-Object LastWriteTimeUtc | Select-Object -Last 1).LastWriteTimeUtc -gt $appStamp)
    if ($instFresh -and -not $Force) {
        Write-Host '  installer is newer than the backend, models and app sources - skipping electron-builder'
    } else {
        if (-not (Test-Internet)) { Write-Host '  WARNING: no internet - electron-builder needs its NSIS tool download on the first build; continuing anyway' }
        # packaging\electron-builder.win.js = package.json "build" + npmRebuild:false - main.js
        # loads no native module, and rebuilding better-sqlite3 for Electron 41 would need MSVC.
        Invoke-Native 'vite build + electron-builder --win (installer compression takes 10-20 min and looks frozen - it is not)' `
            'npm run build:win -- --publish never --config ../packaging/electron-builder.win.js' $App
        $installers = @(Get-ChildItem -LiteralPath $Release -Filter *.exe -File -ErrorAction SilentlyContinue)
        if ($installers.Count -eq 0) { throw "electron-builder finished but no *.exe in $Release" }
    }

    # ---------------------------------------------------------------- done
    Banner 'BUILD OK'
    foreach ($f in ($installers | Sort-Object LastWriteTimeUtc)) {
        Write-Host ("  installer: {0}" -f $f.FullName)
        Write-Host ("  size:      {0:N0} MB" -f ($f.Length / 1MB))
        try { Write-Host ("  sha256:    {0}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash.ToLower()) } catch { }
    }
    Write-Host ("  elapsed:   {0:hh\:mm\:ss}" -f ((Get-Date) - $Started))
    Write-Host ''
    Write-Host '  Next: double-click the installer. It is NOT code-signed, so Windows SmartScreen shows'
    Write-Host '  "Windows protected your PC" -> click "More info" -> "Run anyway". Then follow'
    Write-Host '  docs\windows_laptop_test_ru.md (RU) / docs\windows_laptop_test_en.md (EN): turn Wi-Fi OFF and test.'
    Write-Host ("  log: {0}" -f $Log)
} catch {
    $Failed = $_
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}

if ($Failed) {
    Write-Host ''
    Write-Host ('!' * 78)
    Write-Host ("  BUILD FAILED at step '{0}':" -f $script:StepName)
    Write-Host ("  {0}" -f $Failed.Exception.Message)
    if ($Failed.InvocationInfo -and $Failed.InvocationInfo.ScriptLineNumber) {
        Write-Host ("  (packaging\build_windows.ps1 line {0})" -f $Failed.InvocationInfo.ScriptLineNumber)
    }
    Write-Host ('!' * 78)
    Write-Host '  ---- last 40 lines of packaging\build_windows.log ----'
    try { Get-Content -LiteralPath $Log -Tail 40 -ErrorAction Stop | ForEach-Object { Write-Host ('  | ' + $_) } } catch { Write-Host '  (log not readable)' }
    Write-Host '  -------------------------------------------------------'
    Write-Host ("  >>> Send this file to the developer: {0}" -f $Log)
    Write-Host '  Common fixes: "Cannot create symbolic link" during electron-builder -> Windows Settings -> For developers -> Developer Mode ON, re-run;'
    Write-Host '                "MSBuild / node-gyp" errors -> use Node.js 20 LTS (not 24) and re-run;  disk full -> free 15 GB and re-run.'
    exit 1
}
exit 0
