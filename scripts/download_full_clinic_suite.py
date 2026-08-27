#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — DOWNLOAD FULL CLINIC SUITE
================================================================================

  Downloads every model needed for production clinic deployment:

    BRAIN (6 models, ~830 MB total)
      • Brain Tumor Classifier 2D       andrei-teodor/resnet-pretrained-brain-mri
      • Brain Tumor Segmentation 3D     anhaltai/swinunetrv2_BraTS2021_mini
      • MedSAM (universal seg)          wanglab/medsam-vit-base
      • Dementia Detector               dhritic9/vit-base-brain-mri-dementia-detection

    CHEST + HEAD CT + MAMMO (4 models, ~525 MB total)
      • Chest X-ray (TorchXRayVision)   200 MB
      • Head CT Hemorrhage              nateraw/rsna-2019-intracranial-hemorrhage
      • Mammography                     Falconsai/breast_cancer_image_detection

    Total cache after download: ~1.4 GB on disk

    LLM (separate, via Ollama):
      • Gemma 4 e4b                     3.3 GB    (run setup_gemma_brain.sh local)

  Tracks each download with:
    - Real-time size progress
    - Fallback chain if a repo is unavailable
    - Final on-disk size report

  Run:
    python scripts/download_full_clinic_suite.py
================================================================================
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors
G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

# Build the download plan FROM the live registry so repos never drift stale.
# (Fix: previously hardcoded dead repos like nateraw/rsna and Falconsai/breast.)
def _build_plan():
    from src.inference.model_registry import REGISTRY
    plan = []
    # Map registry backend → downloader method
    method_for = {'xrv': 'xrv', 'huggingface': 'hf-cls', 'sam': 'hf-sam', 'medgemma': 'medgemma'}
    seen = set()
    for key, card in REGISTRY.items():
        if key == 'brain_2d':       # alias
            continue
        if card.modality in seen:
            continue
        seen.add(card.modality)
        # 3D segmentation uses snapshot (custom code / weights)
        method = 'snapshot' if card.is_3d else method_for.get(card.backend, 'hf-cls')
        # MedGemma is huge (4.5GB) — skip from the default "clinic suite"
        if card.backend == 'medgemma':
            continue
        plan.append((card.modality, card.display_name, method, card.repo_id, list(card.fallback_repos or [])))
    return plan

try:
    PLAN = _build_plan()
except Exception:
    # Fallback static plan if registry import fails (kept in sync with registry)
    PLAN = [
        ('chest', 'Chest X-ray (TorchXRayVision DenseNet121)', 'xrv', None, []),
        ('brain_tumor_class', 'Brain Tumor Classifier 2D', 'hf-cls',
            'andrei-teodor/resnet-pretrained-brain-mri',
            ['BehradG/resnet-18-MRI-Brain', 'Devarshi/Brain_Tumor_Classification']),
        ('brain_tumor_seg_3d', 'Brain Tumor 3D Segmentation (SwinUNETR)', 'snapshot',
            'anhaltai/swinunetrv2_BraTS2021_mini', []),
        ('brain_medsam', 'MedSAM', 'hf-sam', 'wanglab/medsam-vit-base', ['flaviagiammarino/medsam-vit-base']),
        ('brain_dementia', 'Dementia Detector', 'hf-cls', 'dhritic9/vit-base-brain-mri-dementia-detection', []),
        ('chest_tb', 'Tuberculosis Detection', 'hf-cls', 'runaksh/chest_xray_tuberculosis_detection', []),
        ('chest_pneumonia', 'Pneumonia Detection', 'hf-cls', 'nickmuchi/vit-finetuned-chest-xray-pneumonia',
            ['lxyuan/vit-xray-pneumonia-classification']),
        ('head_ct', 'Head CT Hemorrhage', 'hf-cls', 'DifeiT/rsna-intracranial-hemorrhage-detection', []),
        ('covid_ct', 'COVID-19 Detection', 'hf-cls', 'asajjad/lung_ct_covid_binary_classification', []),
        ('mammography', 'Mammography Screening', 'hf-cls', 'ITSheep/breastcancer-ultrasound-ViT', []),
    ]


def get_cache_size() -> float:
    """Return HF cache size in MB."""
    p = Path.home() / '.cache' / 'huggingface'
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob('*'):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def download_xrv() -> tuple[bool, str]:
    """Download TorchXRayVision DenseNet121."""
    try:
        import torchxrayvision as xrv
        m = xrv.models.DenseNet(weights='densenet121-res224-all')
        return True, 'OK'
    except Exception as e:
        return False, str(e)[:200]


def download_hf_classifier(primary: str, fallbacks: list[str]) -> tuple[bool, str]:
    """Try to download a HuggingFace image classifier."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    repos = [primary] + fallbacks
    last_err = ''
    for repo in repos:
        try:
            print(f'      → {repo}', end=' ', flush=True)
            AutoImageProcessor.from_pretrained(repo)
            AutoModelForImageClassification.from_pretrained(repo)
            print(f'{G}✓{NC}')
            return True, repo
        except Exception as e:
            print(f'{R}✗{NC} ({str(e)[:70]})')
            last_err = str(e)[:200]
            continue
    return False, last_err


def download_hf_sam(primary: str, fallbacks: list[str]) -> tuple[bool, str]:
    """Download a SAM-style segmentation model."""
    from transformers import SamModel, SamProcessor
    repos = [primary] + fallbacks
    last_err = ''
    for repo in repos:
        try:
            print(f'      → {repo}', end=' ', flush=True)
            SamProcessor.from_pretrained(repo)
            SamModel.from_pretrained(repo)
            print(f'{G}✓{NC}')
            return True, repo
        except Exception as e:
            print(f'{R}✗{NC} ({str(e)[:70]})')
            last_err = str(e)[:200]
            continue
    return False, last_err


def download_snapshot(primary: str, fallbacks: list[str]) -> tuple[bool, str]:
    """Snapshot-download for non-standard model formats (custom code, .bin only)."""
    from huggingface_hub import snapshot_download
    repos = [primary] + fallbacks
    last_err = ''
    for repo in repos:
        try:
            print(f'      → {repo}', end=' ', flush=True)
            snapshot_download(
                repo_id=repo,
                allow_patterns=['*.json', '*.bin', '*.safetensors', '*.py', '*.txt', '*.md'],
            )
            print(f'{G}✓{NC}')
            return True, repo
        except Exception as e:
            print(f'{R}✗{NC} ({str(e)[:70]})')
            last_err = str(e)[:200]
            continue
    return False, last_err


def main():
    print('=' * 78)
    print(f'  {B}SENTINEL — DOWNLOADING FULL CLINIC SUITE{NC}')
    print('=' * 78)
    print()
    print(f'  Models to download:    {len(PLAN)}')
    print(f'  Estimated total size:  ~1.4 GB cached on disk')
    print(f'  Starting cache size:   {get_cache_size():.0f} MB')
    print()

    # Check deps once
    print(f'  Checking dependencies…', end=' ', flush=True)
    try:
        import torch  # noqa
        import transformers  # noqa
        import torchxrayvision  # noqa
        import huggingface_hub  # noqa
        print(f'{G}✓{NC}')
    except ImportError as e:
        print(f'{R}✗ {e}{NC}')
        sys.exit(1)

    print()
    print('-' * 78)

    results = []
    start = time.time()

    for i, (key, name, kind, primary, fallbacks) in enumerate(PLAN, 1):
        print()
        print(f'  [{i}/{len(PLAN)}] {name}')
        cache_before = get_cache_size()

        if kind == 'xrv':
            ok, info = download_xrv()
        elif kind == 'hf-cls':
            ok, info = download_hf_classifier(primary, fallbacks)
        elif kind == 'hf-sam':
            ok, info = download_hf_sam(primary, fallbacks)
        elif kind == 'snapshot':
            ok, info = download_snapshot(primary, fallbacks)
        else:
            ok, info = False, f'unknown kind: {kind}'

        delta = get_cache_size() - cache_before
        if ok:
            print(f'      {G}DONE{NC}  cache +{delta:.0f} MB  ({info[:60]})')
        else:
            print(f'      {R}FAILED{NC}  {info}')
        results.append((key, name, ok, delta, info))

    elapsed = time.time() - start

    # SUMMARY
    print()
    print('=' * 78)
    print(f'  {B}SUMMARY{NC}')
    print('=' * 78)
    print()

    ok_count = sum(1 for _, _, ok, _, _ in results if ok)
    fail_count = len(results) - ok_count

    for key, name, ok, delta, info in results:
        status = f'{G}✓ READY{NC}' if ok else f'{R}✗ FAILED{NC}'
        print(f'  {status}  {name:50} {delta:>6.0f} MB')

    print()
    print(f'  Successful:        {G}{ok_count}/{len(results)}{NC}')
    if fail_count:
        print(f'  Failed:            {R}{fail_count}/{len(results)}{NC}')
    print(f'  Total cache size:  {get_cache_size():.0f} MB')
    print(f'  Time elapsed:      {elapsed/60:.1f} minutes')
    print()

    if fail_count:
        print(f'  {Y}WARNING{NC}: Some models failed to download.')
        print(f'           Try again with internet, or run with HF_TOKEN set.')
        print(f'           Failed models will fall back to other classes at runtime.')
        print()

    print(f'  {B}Next step:{NC}')
    print(f'    bash scripts/setup_gemma_brain.sh local    # install Ollama + Gemma 4 (3.3 GB)')
    print(f'    DEV_BYPASS_LICENSE=1 python run_server.py  # verify everything loads')
    print()

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
