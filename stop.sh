#!/usr/bin/env bash
# =============================================================================
#  SENTINEL MEDICAL AI — Stop everything
# =============================================================================

echo "Stopping Sentinel…"
pkill -f electron 2>/dev/null
pkill -f Electron 2>/dev/null
pkill -f "vite" 2>/dev/null
pkill -f "concurrently" 2>/dev/null
pkill -f "wait-on" 2>/dev/null
pkill -f "run_server.py" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -x "ollama" 2>/dev/null

sleep 2

echo ""
echo "Port check:"
for port in 8000 5173 11434; do
  pid=$(lsof -ti:$port 2>/dev/null | head -1)
  if [ -n "$pid" ]; then
    echo "  :$port  ✗ STILL RUNNING (PID $pid) — kill -9 $pid"
  else
    echo "  :$port  ✓ free"
  fi
done

echo ""
echo "✓ All stopped."
