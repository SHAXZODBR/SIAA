#!/usr/bin/env bash
# Smoke-test the frozen backend (packaging/dist/sentinel-backend) the way the desktop app
# starts it: spawn the launcher, poll /health, assert status ok|degraded, print the model
# table, kill it. Bash twin of packaging/smoke_backend.ps1 (Windows). Used by
# .github/workflows/build-windows.yml right after the freeze; runs on a developer Mac too:
#
#     packaging/smoke_backend.sh                                  # models from <repo>/models
#     SENTINEL_MODELS_DIR=/path/models SMOKE_PORT=8765 SMOKE_TIMEOUT=300 packaging/smoke_backend.sh
#
# REQUIRE_VALIDATED=1 additionally demands status 'ok' (brain_triage loaded) — CI sets it when
# the MODEL_BUNDLE_URL bundle was restored, so a bundle the frozen backend cannot load never
# ships silently. Nothing is written outside a throw-away SENTINEL_DATA_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$ROOT/packaging/dist/sentinel-backend/sentinel-backend"
PORT="${SMOKE_PORT:-8765}"
TIMEOUT_S="${SMOKE_TIMEOUT:-300}"   # first launch of a FRESH bundle can exceed 120 s (macOS verifies every dylib once; a warm launch answers in ~20 s)
PY="${PYTHON:-python3}"
[ -x "$EXE" ] || { echo "ERROR: $EXE not found — run packaging/build_backend.sh first" >&2; exit 1; }

DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sentinel-smoke-data.XXXXXX")"
LOG="$DATA_DIR/backend-smoke.log"
HEALTH="$DATA_DIR/health.json"

export SENTINEL_DEV_INSECURE=1                       # /health is public; no admin bootstrap needed
export SENTINEL_OFFLINE=1                            # hospital posture: the bundle only, never a download
export SENTINEL_DATA_DIR="$DATA_DIR"
export SENTINEL_MODELS_DIR="${SENTINEL_MODELS_DIR:-$ROOT/models}"
echo "smoke: $EXE --host 127.0.0.1 --port $PORT  (models: $SENTINEL_MODELS_DIR, data: $DATA_DIR)"

"$EXE" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT

deadline=$(( $(date +%s) + TIMEOUT_S ))
answered=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: backend exited before /health answered" >&2; tail -n 80 "$LOG" >&2; exit 1
  fi
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" -o "$HEALTH" 2>/dev/null; then answered=1; break; fi
  sleep 3
done
if [ "$answered" != 1 ]; then
  echo "ERROR: /health did not answer within ${TIMEOUT_S}s" >&2; tail -n 80 "$LOG" >&2; exit 1
fi

"$PY" - "$HEALTH" "${REQUIRE_VALIDATED:-0}" <<'EOF'
import json, sys
h = json.load(open(sys.argv[1]))
print(f"status={h.get('status')} version={h.get('version')} offline={h.get('offline')} device={h.get('device')}")
for key, m in (h.get('models') or {}).items():
    print(f"  {key:20s} {'loaded' if m.get('loaded') else 'NOT loaded':10s} {m.get('reason', '')}")
assert h.get('status') in ('ok', 'degraded'), f"unexpected /health status: {h.get('status')!r}"
if sys.argv[2] == '1':
    assert h.get('status') == 'ok', "validated triage model was restored but /health is not 'ok' (brain_triage did not load)"
print("SMOKE OK")
EOF
