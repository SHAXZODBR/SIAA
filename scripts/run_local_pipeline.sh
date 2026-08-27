#!/usr/bin/env bash
# =============================================================================
#  SENTINEL — LOCAL DATA PIPELINE (one command)
#  match images -> build labelled set -> fine-tune -> validate on local patients
#
#  Usage:
#     bash scripts/run_local_pipeline.sh "<image folder>" [binary|multiclass]
#  Example:
#     bash scripts/run_local_pipeline.sh "/Users/shakhzodbtr/Desktop/Siaa_ai/МРТ + МСКТ" binary
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/Users/shakhzodbtr/venv/bin/python}"
IMAGES="${1:?give the image folder path}"
MODE="${2:-binary}"
export LOGURU_LEVEL=ERROR

echo "==================== 1/4  MATCH images -> report labels ===================="
"$PY" scripts/match_images_to_reports.py --images "$IMAGES" \
    --labels data/hospital_reports_labels.csv --out data/matched_dataset.csv

echo; echo "==================== 2/4  BUILD labelled training set ($MODE) ===================="
"$PY" scripts/build_training_set.py --images "$IMAGES" \
    --matched data/matched_dataset.csv --out data/local_finetune --mode "$MODE"

# derive class list from the built folders
CLASSES=$(ls data/local_finetune/train 2>/dev/null | paste -sd, -)
echo "  classes: $CLASSES"

echo; echo "==================== 3/4  FINE-TUNE on local data ===================="
"$PY" training/finetune_brain_classifier.py --data data/local_finetune \
    --classes "$CLASSES" --base-model google/vit-base-patch16-224 \
    --epochs 8 --batch-size 32 --unfreeze-last-blocks 4 --balance \
    --device mps --output models/brain_local_finetuned

echo; echo "==================== 4/4  VALIDATE (precision / recall / F1) ===================="
"$PY" scripts/eval_classifier.py --model-dir models/brain_local_finetuned \
    --data data/local_finetune/val --title "LOCAL brain model (real Uzbek patients)"

echo; echo "✓ DONE. Fine-tuned local model in models/brain_local_finetuned/"
echo "  Compare the recall/F1 above to the 12% specificity baseline — that's your gain."
