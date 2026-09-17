#!/usr/bin/env bash
# ============================================================================
#  Assemble the OFFLINE Windows test kit on the developer Mac.
#
#      packaging/make_windows_kit.sh                 # -> /Users/shakhzodbtr/Desktop/Siaa_ai/SIAA_Windows_Kit
#      KIT=/Volumes/USB/SIAA_Windows_Kit packaging/make_windows_kit.sh
#
#  Produces
#      <KIT>/sentinel/            git archive HEAD of this repo (+ uncommitted changes
#                                 overlaid, listed loudly) - no .git, no data/, no models/,
#                                 no node_modules, no venv, no build output
#      <KIT>/sentinel/models/     copy of <KIT>/models (staged beforehand by
#                                 scripts/download_all_models.py --from-cache and a plain
#                                 copy of the vendor models; see docs/windows_laptop_test_en.md)
#      <KIT>/demo_study/          one real brain MRI study (23 .dcm files) for the offline test
#                                 + SIAA_demo_Brain_MRI.dcm when it is on the Desktop
#      <KIT>/README_FIRST.txt     the 3 steps for the laptop
#
#  The kit's models/ must already exist (this script never touches the HF cache or the
#  repo's models/ tree except to READ from them). Re-runnable: sentinel/ and demo_study/
#  are rebuilt from scratch every time, models/ is left as staged.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
KIT="${KIT:-/Users/shakhzodbtr/Desktop/Siaa_ai/SIAA_Windows_Kit}"
DEMO_SRC="${DEMO_SRC:-/Users/shakhzodbtr/Desktop/Siaa_ai/brain_studies_keep/1.3.12.2.1107.5.2.53.190296.30000026071806470847700000031}"
DEMO_DCM="${DEMO_DCM:-/Users/shakhzodbtr/Desktop/SIAA_demo_Brain_MRI.dcm}"
PY="${PYTHON:-/Users/shakhzodbtr/venv/bin/python}"

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
size_of() { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO is not a git checkout (git archive needs HEAD)"
[ -d "$KIT/models" ] || die "$KIT/models is missing - stage the models first (see docs/windows_laptop_test_en.md, 'Preparing the kit')"
for need in brain_triage_finetuned/MANIFEST.json brain_triage_finetuned/model.safetensors brain_finetuned/model.safetensors \
            densenet/best_model.pt monai_bundles/brats_mri_segmentation/models/model.pt \
            hf/andrei-teodor__resnet-pretrained-brain-mri/config.json hf/DifeiT__rsna-intracranial-hemorrhage-detection/config.json \
            hf/google__vit-base-patch16-224/preprocessor_config.json hf/microsoft__resnet-50/preprocessor_config.json; do
    [ -f "$KIT/models/$need" ] || die "staged models incomplete: $KIT/models/$need missing"
done
ls "$KIT"/models/xrv/*.pt >/dev/null 2>&1 || die "staged models incomplete: $KIT/models/xrv/*.pt missing"
[ -d "$DEMO_SRC" ] || die "demo study not found: $DEMO_SRC"

HEAD_SHA="$(git -C "$REPO" rev-parse --short HEAD)"
log "==> kit:   $KIT"
log "==> repo:  $REPO @ $HEAD_SHA"

# ---------------------------------------------------------------- sentinel/ (git archive HEAD)
DEST="$KIT/sentinel"
rm -rf "$DEST"
mkdir -p "$DEST"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$DEST"
log "==> git archive HEAD -> $DEST ($(git -C "$REPO" ls-tree -r --name-only HEAD | wc -l | tr -d ' ') tracked files)"

# Overlay uncommitted work (modified + untracked-but-not-ignored) so the kit equals the
# working tree minus ignored files; deleted-in-worktree files are removed. Listed loudly
# so nobody ships an unnoticed local edit.
OVERLAID=0
while IFS= read -r -d '' entry; do
    xy="${entry:0:2}"; path="${entry:3}"
    case "$xy" in
        R*|C*|*R|*C) IFS= read -r -d '' _old || true ;;   # rename/copy: the NUL-terminated old path follows
    esac
    case "$xy" in
        *D)  rm -f "$DEST/$path"; log "    overlay: removed  $path"; OVERLAID=$((OVERLAID + 1)); continue ;;
    esac
    if [ -f "$REPO/$path" ]; then
        mkdir -p "$DEST/$(dirname "$path")"
        cp -p "$REPO/$path" "$DEST/$path"
        log "    overlay: $xy $path"
        OVERLAID=$((OVERLAID + 1))
    fi
done < <(git -C "$REPO" status --porcelain=v1 -z --untracked-files=all)
if [ "$OVERLAID" -gt 0 ]; then
    log "==> WARNING: $OVERLAID uncommitted path(s) overlaid on top of HEAD (listed above) - the kit is the WORKING TREE, not commit $HEAD_SHA"
else
    log "==> working tree clean - kit equals commit $HEAD_SHA"
fi

# Belt and braces: nothing heavy or private may ride along.
for bad in .git data models desktop-app/node_modules desktop-app/dist desktop-app/release packaging/dist packaging/build venv .venv logs .pytest_cache; do
    [ -e "$DEST/$bad" ] && die "unexpected $bad inside the archived tree"
done
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name '.DS_Store' -delete 2>/dev/null || true
for must in packaging/build_windows.ps1 packaging/build_backend.ps1 packaging/smoke_backend.ps1 packaging/sentinel_backend.spec \
            packaging/fetch_model_bundle.py packaging/electron-builder.win.js requirements-win.txt run_server.py \
            scripts/download_all_models.py desktop-app/package.json desktop-app/package-lock.json desktop-app/build/icon.ico \
            desktop-app/build/installer.nsh desktop-app/build/license.txt docs/windows_laptop_test_en.md docs/windows_laptop_test_ru.md; do
    [ -f "$DEST/$must" ] || die "kit is missing $must (commit it or make sure it is not git-ignored)"
done
[ -n "$(find "$DEST" -name '*.dcm' -print -quit)" ] && die "a .dcm file slipped into the source tree"

# ---------------------------------------------------------------- sentinel/models (copy of the staged bundle)
log "==> models: $KIT/models -> $DEST/models"
mkdir -p "$DEST/models"
# -c = APFS clonefile on the Mac (instant, no extra disk); a real copy on USB/NTFS later.
if ! cp -Rpc "$KIT/models/." "$DEST/models/" 2>/dev/null; then
    cp -Rp "$KIT/models/." "$DEST/models/"
fi
rm -rf "$DEST/models/.empty_hf_cache"
find "$DEST/models" -name '.DS_Store' -delete 2>/dev/null || true
"$PY" - "$DEST/models" <<'EOF'
import json, sys, hashlib
from pathlib import Path
d = Path(sys.argv[1]) / 'brain_triage_finetuned'
man = json.loads((d / 'MANIFEST.json').read_text())
for name, pinned in man['files'].items():
    h = hashlib.sha256((d / name).read_bytes()).hexdigest()
    assert h == pinned, f'{name}: sha256 mismatch after copy'
print(f"    MANIFEST.json re-hashed OK: identity {man['files']['model.safetensors'][:12]}")
EOF

# ---------------------------------------------------------------- demo_study/
DEMO="$KIT/demo_study"
rm -rf "$DEMO"
mkdir -p "$DEMO"
# Only the DICOM files (the source folder also holds .bmp previews); the series sub-folders
# are kept so the study looks exactly like a scanner export.
(cd "$DEMO_SRC" && find . -type f \( -iname '*.dcm' -o -iname '*.dicom' \) -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$DEMO/$(dirname "$f")"
    cp -p "$f" "$DEMO/$f"
done)
N_DCM="$(find "$DEMO" -type f -iname '*.dcm' | wc -l | tr -d ' ')"
[ "$N_DCM" -gt 0 ] || die "no .dcm files copied from $DEMO_SRC"
log "==> demo_study: $N_DCM .dcm files ($(size_of "$DEMO")) from $(basename "$DEMO_SRC")"
if [ -f "$DEMO_DCM" ]; then
    cp -p "$DEMO_DCM" "$KIT/SIAA_demo_Brain_MRI.dcm"
    log "==> single-file demo: $KIT/SIAA_demo_Brain_MRI.dcm ($(size_of "$KIT/SIAA_demo_Brain_MRI.dcm"))"
else
    rm -f "$KIT/SIAA_demo_Brain_MRI.dcm"
    log "==> single-file demo SKIPPED: $DEMO_DCM not present"
fi

# ---------------------------------------------------------------- README_FIRST.txt
# UTF-8 BOM first: the RU half must open correctly in any Windows Notepad.
printf '\xEF\xBB\xBF' > "$KIT/README_FIRST.txt"
cat >> "$KIT/README_FIRST.txt" <<EOF
SIAA / Sentinel Medical AI - WINDOWS TEST KIT  (built $(date '+%Y-%m-%d %H:%M') from commit $HEAD_SHA)
=====================================================================================

  1. Copy this whole folder to the laptop as   C:\\SIAA
     (so that C:\\SIAA\\sentinel\\packaging\\build_windows.ps1 exists).
     Internet is needed ONCE for the build (pip / npm downloads) - the models are inside.

  2. Open PowerShell in C:\\SIAA\\sentinel
     (Explorer: open the folder, Shift + right-click on empty space -> "Open PowerShell window here",
      or type  cd C:\\SIAA\\sentinel  in any PowerShell window).
     Python 3.11 (64-bit) and Node.js 20 LTS must be installed - the script tells you exactly how if not.

  3. Run:
        powershell -ExecutionPolicy Bypass -File packaging\\build_windows.ps1

     45-90 minutes. It ends with the installer path:  desktop-app\\release\\Sentinel Medical AI Setup 1.0.0.exe
     If it fails, send  C:\\SIAA\\sentinel\\packaging\\build_windows.log  to the developer.

  Then install the .exe (SmartScreen: "More info" -> "Run anyway"), turn Wi-Fi OFF and follow
  the checklist:   sentinel\\docs\\windows_laptop_test_ru.md  (RU)   /   sentinel\\docs\\windows_laptop_test_en.md  (EN)
  Test data:       demo_study\\   (one real brain MRI, $N_DCM DICOM files - upload the FOLDER)

-------------------------------------------------------------------------------------
  RU
  1. Скопируйте всю папку на ноутбук как  C:\\SIAA  (должен существовать файл
     C:\\SIAA\\sentinel\\packaging\\build_windows.ps1). Интернет нужен ОДИН раз - для сборки; модели уже внутри.
  2. Откройте PowerShell в папке C:\\SIAA\\sentinel  (Shift + правый клик по пустому месту ->
     "Открыть окно PowerShell здесь"). Нужны Python 3.11 (64-bit) и Node.js 20 LTS - скрипт подскажет, как поставить.
  3. Выполните:   powershell -ExecutionPolicy Bypass -File packaging\\build_windows.ps1
     45-90 минут. В конце - путь к установщику desktop-app\\release\\*.exe.
     Если ошибка - отправьте разработчику файл C:\\SIAA\\sentinel\\packaging\\build_windows.log.
  Дальше: установить .exe, выключить Wi-Fi, пройти чек-лист  sentinel\\docs\\windows_laptop_test_ru.md.

Folder layout:
  sentinel\\            source + build scripts + models\\ (what you build from)
  demo_study\\          test study for the offline check
  models\\              spare copy of the model bundle (identical to sentinel\\models; only needed if that one is damaged)
EOF

# ---------------------------------------------------------------- summary
log ""
log "==> KIT READY: $KIT"
for e in README_FIRST.txt sentinel demo_study models SIAA_demo_Brain_MRI.dcm; do
    if [ -e "$KIT/$e" ]; then printf '    %-28s %s\n' "$e" "$(size_of "$KIT/$e")"; fi
done
printf '    %-28s %s\n' 'sentinel/models (inside)' "$(size_of "$DEST/models")"
printf '    %-28s %s  (source files: %s)\n' 'sentinel/ without models' "$(du -sh -I models "$DEST" 2>/dev/null | awk '{print $1}')" "$(find "$DEST" -type f -not -path "$DEST/models/*" | wc -l | tr -d ' ')"
log "    TOTAL                        $(size_of "$KIT")"
