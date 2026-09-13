#!/bin/zsh
# Freeze the backend with PyInstaller (run on the TARGET OS: macOS here; on Windows use
# packaging/build_backend.ps1 — PyInstaller does not cross-compile).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-/Users/shakhzodbtr/venv/bin/python}"
"$PY" -m pip show pyinstaller >/dev/null 2>&1 || "$PY" -m pip install -q pyinstaller
"$PY" -m PyInstaller --noconfirm --clean packaging/sentinel_backend.spec \
   --distpath packaging/dist --workpath packaging/build 2>&1 | grep -E "INFO: (Building|Copying|Appending)|WARNING|ERROR|completed successfully" | tail -20
# The grep pipeline above hides PyInstaller's exit code — the launcher's existence is the gate.
test -x packaging/dist/sentinel-backend/sentinel-backend || { echo "ERROR: freeze failed — no packaging/dist/sentinel-backend/sentinel-backend" >&2; exit 1; }
du -sh packaging/dist/sentinel-backend | awk '{print "bundle size:", $1}'
