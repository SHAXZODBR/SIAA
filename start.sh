#!/usr/bin/env bash
# =============================================================================
#  SENTINEL MEDICAL AI — One-command launcher (macOS)
# =============================================================================
#  Opens 3 Terminal tabs:
#    1. Ollama (LLM for reports)
#    2. Sentinel inference server (port 8000)
#    3. Desktop app (Electron + Vite, port 5173)
#
#  Run with: bash start.sh
# =============================================================================

set -e
REPO="/Users/shakhzodbtr/Desktop/Siaa_ai/sentinel"

if [ ! -d "$REPO" ]; then
  echo "✗ Repo not found at $REPO"
  exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Sentinel Medical AI — Launcher"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Ollama in a new Terminal tab
echo "[1/3] Starting Ollama in a new Terminal tab…"
osascript <<EOF
tell application "Terminal"
  activate
  do script "echo '── OLLAMA (Gemma 3:4b for reports) ──'; ollama serve"
end tell
EOF
sleep 2

# 2. Sentinel server in a new Terminal tab
echo "[2/3] Starting Sentinel inference server in a new Terminal tab…"
osascript <<EOF
tell application "Terminal"
  activate
  do script "echo '── SENTINEL SERVER (port 8000) ──'; cd $REPO; source ~/venv/bin/activate; python run_server.py"
end tell
EOF

# 3. Wait for server, then start desktop app in a new Terminal tab
echo "[3/3] Waiting for server, then starting desktop app…"
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "     ✓ Server up"
    break
  fi
  sleep 2
done

osascript <<EOF
tell application "Terminal"
  activate
  do script "echo '── DESKTOP APP (Electron + Vite) ──'; cd $REPO/desktop-app; npm run dev"
end tell
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ All 3 tabs launched"
echo ""
echo "  Endpoints:"
echo "    Inference server: http://127.0.0.1:8000/health"
echo "    Vite dev server:  http://localhost:5173"
echo "    Ollama API:       http://localhost:11434/api/tags"
echo ""
echo "  Electron window opens automatically after Vite is ready (~10s)"
echo ""
echo "  To stop everything later:"
echo "    bash $REPO/stop.sh"
echo "════════════════════════════════════════════════════════════════"
