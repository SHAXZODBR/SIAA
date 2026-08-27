#!/usr/bin/env bash
# =============================================================================
#  SENTINEL — DOWNLOAD ADDITIONAL PRETRAINED BRAIN MODELS  (run by the user)
# =============================================================================
#
#  Sentinel does NOT auto-download these: they are community models from
#  external HuggingFace repos, and pulling+running model weights executes code
#  from those repos. For a medical product that decision is YOURS to make, not
#  the agent's. This script downloads + test-loads each one so you can see which
#  actually work before trusting them. Once cached, the registry uses them
#  automatically.
#
#  ⚠  PROVENANCE WARNING: these are low-reputation community models, UNVALIDATED
#     on your clinic's population. They are wired as 'experimental' — screening
#     hints only, radiologist confirms every read. Do not present as diagnostic.
#
#  Usage:
#     bash scripts/download_brain_models.sh           # vetted set (recommended)
#     bash scripts/download_brain_models.sh --all     # also optional extras
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/shakhzodbtr/venv/bin/python}"
[ -x "$PY" ] || PY="python3"

# Prefer safetensors (no pickle code-execution) where the repo ships it.
export HF_HUB_OFFLINE=0

echo "================================================================"
echo "  Downloading VETTED brain models (Apache-2.0, proper configs)"
echo "================================================================"

# --- WIRED into the brain panel as 'experimental' ---------------------------
VETTED=(
  # Stroke staging on diffusion MRI (BEiT, safetensors, Apache-2.0) -> brain_stroke
  "BTX24/beit-finetuned-stroke-diff-mri"
)

# --- OPTIONAL extras (not wired yet; download to evaluate) -------------------
EXTRAS=(
  "andrei-teodor/vit-base-brain-mri"                  # ViT tumor (alt to current ResNet)
  "BTX24/deit-base-patch16-224-finetuned-stroke-binary"  # stroke binary backup
  "prithivMLmods/Brain3-Anomaly-SigLIP2"              # general normal/abnormal screen
)

LIST=("${VETTED[@]}")
if [ "${1:-}" = "--all" ]; then
  LIST+=("${EXTRAS[@]}")
fi

printf '%s\n' "${LIST[@]}" > /tmp/sentinel_brain_dl.txt
"$PY" scripts/test_load_models.py --file /tmp/sentinel_brain_dl.txt

echo
echo "✓ Done. Models that printed '✓ LOADS' are cached and now active in Sentinel."
echo "  Hemorrhage (CT) is already covered by the existing head_ct model."
echo "  Re-run the brain panel:  POST /analyze/brain  (stroke detector now fires on DWI)."
