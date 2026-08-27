#!/usr/bin/env python3
"""
================================================================================
  SENTINEL MEDICAL AI — DOWNLOAD ALL PRETRAINED MODELS
================================================================================

  Downloads every modality model the registry knows about:
    • Chest X-ray (TorchXRayVision DenseNet121, ~30 MB)
    • Brain MRI 2D Tumor Classifier (HuggingFace, ~90 MB)
    • Head CT Hemorrhage Detection (HuggingFace, ~110 MB)
    • Mammography Screening (HuggingFace, ~85 MB)

  After this script runs, every modality is ready to analyze without internet.

  Usage:
    python scripts/download_all_models.py            # download everything
    python scripts/download_all_models.py chest      # just one
    python scripts/download_all_models.py brain_2d head_ct
================================================================================
"""

import sys
import os
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.model_registry import REGISTRY, get_model


def download_one(key: str) -> bool:
    """Force-load a model so its weights are downloaded and cached."""
    card = REGISTRY.get(key)
    if not card:
        print(f"  ✗ Unknown model: {key}")
        return False

    print(f"\n[{key}] {card.display_name}")
    print(f"  Backend: {card.backend}")
    print(f"  Size: ~{card.download_size_mb} MB")
    print(f"  Classes: {len(card.classes)}")
    print(f"  Source: {card.citation}")
    print(f"  Downloading…", end=' ', flush=True)

    try:
        entry = get_model(key, device='cpu')
        if entry and entry['available']:
            print('✓ READY')
            return True
        else:
            reason = entry['reason'] if entry else 'unknown'
            print(f'✗ FAILED: {reason}')
            return False
    except Exception as e:
        print(f'✗ ERROR: {e}')
        return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    keys = args if args else list(REGISTRY.keys())

    print("=" * 70)
    print("  SENTINEL MEDICAL AI — Pretrained Model Downloader")
    print("=" * 70)
    print(f"  Will download: {', '.join(keys)}")

    # Make sure dependencies are present
    print("\n[deps] Checking dependencies…")
    deps_to_install = []
    try:
        import torchxrayvision  # noqa
        print("  ✓ torchxrayvision")
    except ImportError:
        deps_to_install.append('torchxrayvision')
        print("  ✗ torchxrayvision (will install)")
    try:
        import transformers  # noqa
        print("  ✓ transformers")
    except ImportError:
        deps_to_install.append('transformers')
        print("  ✗ transformers (will install)")

    if deps_to_install:
        print(f"\n  Installing: {' '.join(deps_to_install)}…")
        cmd = f"{sys.executable} -m pip install {' '.join(deps_to_install)}"
        rc = os.system(cmd)
        if rc != 0:
            print("  ✗ Install failed. Run manually and re-try.")
            sys.exit(1)

    # Download each model
    results = {}
    for key in keys:
        results[key] = download_one(key)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    ready, missing = [], []
    for key, ok in results.items():
        card = REGISTRY[key]
        if ok:
            ready.append(card.display_name)
        else:
            missing.append(card.display_name)

    if ready:
        print("\n  ✓ READY:")
        for name in ready:
            print(f"      • {name}")

    if missing:
        print("\n  ✗ NOT READY:")
        for name in missing:
            print(f"      • {name}")
        print("\n  → For HuggingFace models the most common cause is no")
        print("    internet during download or the repo went private. Try:")
        print("      pip install --upgrade transformers huggingface_hub")
        print("      huggingface-cli login   # only if a repo is gated")

    print("\n  Next: restart the inference server. It will pick the right")
    print("        model for each DICOM automatically (chest/brain/head_ct/")
    print("        mammography) based on the DICOM Modality + BodyPart tags.")
    print()


if __name__ == '__main__':
    main()
