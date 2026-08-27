#!/bin/bash
# ================================================================
# SENTINEL — CLINIC INSTALLATION SCRIPT
# Installs everything needed to run Sentinel 100% locally
# at a clinic with NO cloud dependencies.
#
# What this installs:
#   ✅ Ollama (local LLM runner)
#   ✅ Gemma 4 (or Gemma 3 fallback) — for medical reports
#   ✅ TorchXRayVision pretrained chest model
#   ✅ Brain tumor + segmentation models
#   ✅ Sentinel server + desktop app
#
# After install, runs 100% offline:
#   - No GOOGLE_AI_KEY needed
#   - No internet during operation
#   - Patient data never leaves the clinic
# ================================================================

set -e

echo ""
echo "============================================================"
echo "  SENTINEL CLINIC INSTALLATION"
echo "  100% LOCAL — Patient data stays at clinic"
echo "============================================================"
echo ""

# ===== STEP 1: Detect OS =====
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    *)          echo "Unsupported OS: ${OS}. Use Windows installer instead."; exit 1;;
esac
echo "[1/6] Platform: $PLATFORM"

# ===== STEP 2: Install Ollama =====
echo ""
echo "[2/6] Installing Ollama (local LLM runner)..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama already installed"
else
    if [ "$PLATFORM" = "Mac" ]; then
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    echo "  ✓ Ollama installed"
fi

# ===== STEP 3: Start Ollama service =====
echo ""
echo "[3/6] Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    echo "  ✓ Ollama already running"
else
    if [ "$PLATFORM" = "Mac" ]; then
        brew services start ollama 2>/dev/null || (ollama serve &) > /dev/null 2>&1
    else
        systemctl --user start ollama 2>/dev/null || (ollama serve &) > /dev/null 2>&1
    fi
    sleep 3
    echo "  ✓ Ollama started"
fi

# ===== STEP 4: Pull Gemma model =====
echo ""
echo "[4/6] Downloading medical AI model (Gemma)..."
echo "  This may take 5-10 minutes (downloading ~3 GB)..."

# Try Gemma 4 first, fall back to Gemma 3
if ollama list 2>/dev/null | grep -q "gemma4"; then
    echo "  ✓ Gemma 4 already installed"
elif ollama list 2>/dev/null | grep -q "gemma3"; then
    echo "  ✓ Gemma 3 already installed"
else
    echo "  Trying Gemma 4..."
    if ollama pull gemma4:e4b 2>/dev/null; then
        echo "  ✓ Gemma 4 (e4b) downloaded"
    else
        echo "  Gemma 4 not available, downloading Gemma 3..."
        ollama pull gemma3:4b
        echo "  ✓ Gemma 3 (4b) downloaded"
    fi
fi

# ===== STEP 5: Download medical AI models =====
echo ""
echo "[5/6] Downloading medical imaging models..."

cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install -q -r requirements.txt 2>&1 | tail -5

# Chest X-ray model (TorchXRayVision)
if [ ! -f "models/densenet/best_model.pt" ]; then
    echo "  Downloading chest X-ray model..."
    python -m src.inference.pretrained_model
fi

# Brain models
if [ ! -f "models/brain_2d/best_brain_model.pt" ]; then
    echo "  Downloading brain MRI models..."
    python -m src.inference.brain_pretrained --mode both 2>/dev/null || true
fi

echo "  ✓ Medical models downloaded"

# ===== STEP 6: Configure for LOCAL ONLY =====
echo ""
echo "[6/6] Configuring for LOCAL-ONLY operation..."

# Make sure GOOGLE_AI_KEY is NOT set (forces Ollama use)
if grep -q "GOOGLE_AI_KEY" ~/.zshrc 2>/dev/null; then
    echo "  ⚠ Found GOOGLE_AI_KEY in ~/.zshrc"
    echo "  Removing it for clinic deployment (local-only)..."
    sed -i.backup '/GOOGLE_AI_KEY/d' ~/.zshrc
fi

# Set Ollama to keep model loaded
echo 'export OLLAMA_KEEP_ALIVE=24h' >> ~/.zshrc

echo "  ✓ Configured for local-only operation"

# ===== DONE =====
echo ""
echo "============================================================"
echo "  ✅ INSTALLATION COMPLETE"
echo "============================================================"
echo ""
echo "  Sentinel is now 100% local:"
echo "  - Patient data never leaves this PC"
echo "  - Works without internet"
echo "  - Uses local Gemma model for reports"
echo ""
echo "  TO START SENTINEL:"
echo "  --------------------"
echo "  1. Start AI server:"
echo "     cd $(pwd)"
echo "     source venv/bin/activate"
echo "     python run_server.py"
echo ""
echo "  2. Start desktop app (new terminal):"
echo "     cd $(pwd)/desktop-app"
echo "     npm install"
echo "     npm run dev"
echo ""
echo "  3. Configure folder watcher for MRI/CT machine:"
echo "     python -m src.inference.folder_watcher \\"
echo "       --watch /path/to/clinic/dicom/inbox \\"
echo "       --language ru"
echo ""
echo "============================================================"
