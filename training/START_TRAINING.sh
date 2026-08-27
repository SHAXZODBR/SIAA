#!/bin/bash
# ================================================================
# SENTINEL MEDICAL AI — ONE-CLICK TRAINING ON RUNPOD
# ================================================================
#
# BEFORE RUNNING:
#   1. Start a RunPod pod: RTX 4090, PyTorch template, 500GB disk
#   2. Upload your datasets to /workspace/datasets/
#      OR let this script download them
#   3. Upload this entire 'training/' folder to /workspace/training/
#
# THEN RUN:
#   cd /workspace/training && bash START_TRAINING.sh
#
# WHAT HAPPENS:
#   1. Installs all dependencies
#   2. Downloads NIH + RSNA datasets (if not present)
#   3. Preprocesses all images
#   4. Trains DenseNet121 with ALL optimization tricks
#   5. Optimizes thresholds for >90% recall
#   6. Generates evaluation report with plots
#   7. Saves final model ready for deployment
#
# TOTAL TIME: ~10-20 hours on RTX 4090
# TOTAL COST: ~$5-10 on RunPod
# ================================================================

set -e

echo ""
echo "================================================================"
echo "  SENTINEL MEDICAL AI — MASTER TRAINING"
echo "  Target: Recall >= 90% | Specificity >= 80%"
echo "================================================================"
echo ""

# Check GPU
nvidia-smi 2>/dev/null || echo "WARNING: No GPU detected!"

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r requirements_gpu.txt 2>/dev/null
echo "  Done"

# Check for data
DATA_DIR="/workspace/datasets"
if [ -d "$DATA_DIR/nih-chestxray14" ] || [ -d "$DATA_DIR/rsna-pneumonia" ]; then
    echo ""
    echo "[2/3] Datasets found at $DATA_DIR"
    echo "  Skipping download, starting preprocessing + training..."
    echo ""

    python train_master.py \
        --data-dir "$DATA_DIR" \
        --output-dir /workspace/sentinel_training \
        --skip-download
else
    echo ""
    echo "[2/3] No datasets found. Will attempt to download..."
    echo "  If auto-download fails, manually upload your datasets to:"
    echo "  $DATA_DIR/nih-chestxray14/ and $DATA_DIR/rsna-pneumonia/"
    echo ""

    python train_master.py \
        --data-dir "$DATA_DIR" \
        --output-dir /workspace/sentinel_training
fi

echo ""
echo "[3/3] Training complete!"
echo ""
echo "  Your trained model is at:"
echo "    /workspace/sentinel_training/best_model.pt"
echo ""
echo "  Download it to your local machine:"
echo "    scp runpod:/workspace/sentinel_training/best_model.pt ./models/densenet/"
echo ""
echo "  View results:"
echo "    cat /workspace/sentinel_training/results.json"
echo ""
echo "  View plots (download and open):"
echo "    scp -r runpod:/workspace/sentinel_training/plots/ ./plots/"
echo ""
echo "================================================================"
