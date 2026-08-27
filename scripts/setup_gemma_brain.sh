#!/usr/bin/env bash
# =============================================================================
#  SENTINEL — GEMMA 4 / MEDGEMMA SETUP FOR BRAIN-FIRST DEPLOYMENT
# =============================================================================
#
#  Sets up the LLM that generates radiology reports.
#
#  Three deployment modes:
#
#   (1) CLINIC LOCAL — Ollama + Gemma 4 (recommended for production)
#       100 % offline, patient data never leaves the building.
#       Requires GPU (NVIDIA 6+ GB or Apple Silicon).
#
#   (2) CLINIC LOCAL — HuggingFace + MedGemma 4B (best quality)
#       Best radiology output, requires more GPU memory (~10 GB).
#       Fine-tunable on clinic data via training/finetune_medgemma_reporter.py
#
#   (3) DEV / DEMO — Google AI Studio (cloud, free tier)
#       Use only for development; never patient data.
#
#  Usage:
#       bash scripts/setup_gemma_brain.sh local           # mode 1
#       bash scripts/setup_gemma_brain.sh medgemma        # mode 2
#       bash scripts/setup_gemma_brain.sh cloud           # mode 3
#       bash scripts/setup_gemma_brain.sh status          # check current setup
# =============================================================================

set -e

MODE="${1:-status}"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=================================================================="
echo "  Sentinel — Brain Report Engine Setup"
echo "=================================================================="

case "$MODE" in
  local|gemma)
    # Mode 1: Ollama + Gemma 4
    echo ""
    echo "MODE: Clinic-local Gemma 4 via Ollama (100% offline)"
    echo ""

    # Check Ollama
    if ! command -v ollama >/dev/null 2>&1; then
      echo -e "${YELLOW}⚠ Ollama not installed. Installing now…${NC}"
      if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  → Download from https://ollama.com/download/mac and run the installer"
        echo "  → Re-run this script after installing"
        exit 1
      else
        curl -fsSL https://ollama.com/install.sh | sh
      fi
    else
      echo -e "${GREEN}✓ Ollama installed${NC} ($(ollama --version 2>&1 | head -1))"
    fi

    # Start Ollama
    if ! pgrep -x ollama >/dev/null; then
      echo "  Starting Ollama daemon…"
      ollama serve >/tmp/ollama.log 2>&1 &
      sleep 2
    fi

    # Pull preferred model — try Gemma 4 first, then 3
    for MODEL in gemma4:e4b gemma4:e2b gemma3:4b gemma3:1b; do
      echo "  Pulling ${MODEL}…"
      if ollama pull "$MODEL" 2>&1 | tail -3; then
        echo -e "${GREEN}✓ ${MODEL} ready${NC}"
        SELECTED="$MODEL"
        break
      else
        echo -e "${YELLOW}  ${MODEL} unavailable, trying next…${NC}"
      fi
    done

    if [[ -z "$SELECTED" ]]; then
      echo -e "${RED}✗ No Gemma model could be pulled. Check internet connection.${NC}"
      exit 1
    fi

    # Quick smoke test
    echo ""
    echo "Testing report generation in Russian…"
    RESPONSE=$(ollama run "$SELECTED" "Скажи 'тест' одним словом." 2>&1 | head -1)
    if [[ "$RESPONSE" == *"тест"* ]] || [[ "$RESPONSE" == *"test"* ]]; then
      echo -e "${GREEN}✓ Cyrillic generation works${NC}"
    else
      echo -e "${YELLOW}⚠ Response: $RESPONSE${NC}"
    fi

    # Unset cloud key so backend prefers local
    if [[ -n "$GOOGLE_AI_KEY" ]]; then
      echo ""
      echo -e "${YELLOW}⚠ GOOGLE_AI_KEY is set in your environment.${NC}"
      echo "  For clinic deployment, REMOVE it so reports use local Gemma:"
      echo "    unset GOOGLE_AI_KEY"
      echo "    sed -i.bak '/GOOGLE_AI_KEY/d' ~/.zshrc"
    fi

    echo ""
    echo -e "${GREEN}✓ DONE. Restart the inference server:${NC}"
    echo "    python run_server.py"
    echo "  You should see: 'Ollama ready with $SELECTED'"
    ;;

  medgemma)
    # Mode 2: MedGemma 4B (best radiology quality)
    echo ""
    echo "MODE: MedGemma 4B (best brain-radiology quality)"
    echo "      ~8 GB download, requires ~10 GB GPU memory"
    echo ""

    source ~/venv/bin/activate 2>/dev/null || true

    # Install deps
    echo "  Installing transformers + peft…"
    pip install -q --upgrade transformers peft accelerate

    # Download base MedGemma
    python3 << 'EOF'
import sys
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("  Pulling unsloth/medgemma-4b-it (this takes a while)…")
    processor = AutoProcessor.from_pretrained("unsloth/medgemma-4b-it")
    print("  ✓ Processor downloaded")
    model = AutoModelForImageTextToText.from_pretrained("unsloth/medgemma-4b-it")
    print("  ✓ Model downloaded")
    print()
    print("  Try the BraTS-fine-tuned variant for brain reports:")
    print("    Jesteban247/brats_medgemma")
except Exception as e:
    print(f"  ✗ Download failed: {e}")
    print()
    print("  Common causes:")
    print("    • Repo is gated — login: huggingface-cli login")
    print("    • Out of disk space — need ~10 GB free")
    sys.exit(1)
EOF

    echo ""
    echo -e "${GREEN}✓ DONE. The inference server will pick up MedGemma automatically.${NC}"
    echo "  Restart: python run_server.py"
    echo "  Look for: 'Loaded MedGemma from unsloth/medgemma-4b-it'"
    echo ""
    echo "  To fine-tune on YOUR clinic's reports:"
    echo "    python training/finetune_medgemma_reporter.py --data data/reports.jsonl"
    ;;

  cloud)
    # Mode 3: Google AI Studio (dev only)
    echo ""
    echo "MODE: Google AI Studio (cloud, dev only — DO NOT use with patient data)"
    echo ""
    echo "  1. Get a free API key: https://aistudio.google.com/apikey"
    echo "  2. Add to your shell config:"
    echo "       echo 'export GOOGLE_AI_KEY=\"YOUR_KEY\"' >> ~/.zshrc"
    echo "       source ~/.zshrc"
    echo "  3. Restart server"
    ;;

  status)
    # Check current setup
    echo ""
    echo "Current status:"
    echo ""

    # Ollama
    if pgrep -x ollama >/dev/null 2>&1; then
      MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models', [])))" 2>/dev/null || echo "(none)")
      echo -e "  ${GREEN}✓ Ollama running${NC} — models: $MODELS"
    else
      echo -e "  ${YELLOW}⚠ Ollama NOT running${NC} (run: bash scripts/setup_gemma_brain.sh local)"
    fi

    # MedGemma cache
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    if [[ -d "$HF_CACHE" ]] && ls "$HF_CACHE" 2>/dev/null | grep -q medgemma; then
      echo -e "  ${GREEN}✓ MedGemma cached locally${NC}"
    else
      echo -e "  ${YELLOW}⚠ MedGemma NOT cached${NC} (run: bash scripts/setup_gemma_brain.sh medgemma)"
    fi

    # Cloud key
    if [[ -n "$GOOGLE_AI_KEY" ]]; then
      echo -e "  ${YELLOW}⚠ GOOGLE_AI_KEY is set${NC} — reports may use cloud (DEV mode)"
    else
      echo -e "  ${GREEN}✓ GOOGLE_AI_KEY not set${NC} — reports will use local backend"
    fi

    # Server
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
      LICENSE=$(curl -s http://localhost:8000/license/status | python3 -c "import json,sys; print(json.load(sys.stdin).get('info', {}).get('mode', 'unknown'))" 2>/dev/null)
      echo -e "  ${GREEN}✓ Inference server up${NC} — license mode: $LICENSE"
    else
      echo -e "  ${YELLOW}⚠ Inference server NOT running${NC}"
    fi
    ;;

  *)
    echo "Usage: bash scripts/setup_gemma_brain.sh {local|medgemma|cloud|status}"
    echo ""
    echo "  local      Install Ollama + Gemma 4 (recommended for clinics)"
    echo "  medgemma   Install MedGemma 4B (best radiology quality, GPU-heavy)"
    echo "  cloud      Use Google AI Studio (dev only, never patient data)"
    echo "  status     Show what's currently configured"
    exit 1
    ;;
esac
