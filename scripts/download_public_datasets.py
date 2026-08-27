#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PUBLIC DATASET DOWNLOADER
================================================================================

  Downloads validation / fine-tuning datasets from the internet.

  ALWAYS shows total size BEFORE downloading and asks for confirmation.

  Usage:
    python scripts/download_public_datasets.py             # show menu
    python scripts/download_public_datasets.py tier1       # ~500 MB
    python scripts/download_public_datasets.py tier2       # ~5 GB (asks)
    python scripts/download_public_datasets.py status      # what's already downloaded
    python scripts/download_public_datasets.py --yes tier1 # skip confirm

  Where things go:
    data/public/brain_tumor_kaggle/    — Brain Tumor MRI 4-class
    data/public/medmnist/              — All MedMNIST 2D datasets
    data/public/ixi_sample/            — Brain MRI healthy
    data/public/tcia_nsclc/            — Sample chest CT DICOMs
    data/public/breast_ultrasound/     — Breast ultrasound images
    data/public/brats_lite/            — 50-case BraTS subset
    data/public/rsna_ich/              — RSNA hemorrhage subset
================================================================================
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / 'data' / 'public'


# Each dataset entry has:
#   tier        — 1, 2, or 3
#   size_mb     — approximate download size
#   target_dir  — where it lives under data/public/
#   download_fn — name of the function below that knows how to fetch it
#   needs_auth  — True if requires Kaggle/HF/PhysioNet login
DATASETS = [
    # ── TIER 1 ── starter data, ~500 MB total
    {
        'key': 'brain_tumor_kaggle',
        'name': 'Brain Tumor MRI 4-class (Kaggle)',
        'tier': 1, 'size_mb': 150,
        'target': DATA_ROOT / 'brain_tumor_kaggle',
        'method': 'hf-dataset',
        'hf_repo': 'PranomVignesh/MRI-Images-of-Brain-Tumor',
        'hf_repo_alt': ['mlx-community/brain-tumor', 'orpheus99/brain-tumor-mri-dataset'],
        'needs_auth': False,
        'tag': 'production',
    },
    {
        'key': 'medmnist',
        'name': 'MedMNIST 2D (Brain + Breast + Chest, all)',
        'tier': 1, 'size_mb': 250,
        'target': DATA_ROOT / 'medmnist',
        'method': 'medmnist',
        'subsets': ['organcmnist', 'breastmnist', 'chestmnist', 'pneumoniamnist'],
        'needs_auth': False,
        'tag': 'production',
    },
    {
        'key': 'ixi_sample',
        'name': 'IXI Brain MRI Sample (healthy adults, 10 cases)',
        'tier': 1, 'size_mb': 200,
        'target': DATA_ROOT / 'ixi_sample',
        'method': 'hf-dataset',
        'hf_repo': 'lhoestq/IXI-T1',
        'hf_repo_alt': [],
        'needs_auth': False,
        'tag': 'production',
    },
    {
        'key': 'breast_ultrasound',
        'name': 'Breast Ultrasound Images (Kaggle, 780 cases)',
        'tier': 1, 'size_mb': 150,
        'target': DATA_ROOT / 'breast_ultrasound',
        'method': 'hf-dataset',
        'hf_repo': 'AeroBlast/breast-ultrasound-images',
        'hf_repo_alt': [],
        'needs_auth': False,
        'tag': 'production',
    },

    # ── TIER 2 ── validation, ~5 GB total
    {
        'key': 'brats_lite',
        'name': 'BraTS preprocessing dataset (T1/T1ce/T2/FLAIR + masks)',
        'tier': 2, 'size_mb': 1500,
        'target': DATA_ROOT / 'brats_lite',
        'method': 'hf-dataset',
        'hf_repo': 'yuuricho/brats_preprocessing',     # 6857 downloads, ungated
        'hf_repo_alt': [
            'YongchengYAO/BraTS24-Lite',                 # explicit "Lite", small
            'rocky93/BraTS_segmentation',                # Apache-2.0
            'Babai12345/BRATS-2020',                     # Apache-2.0
            'PhucNT2511/brats21',                        # Apache-2.0
            'yasserh/brain-tumor-segmentation',          # legacy (gated)
        ],
        'needs_auth': False,
        'tag': 'beta',
    },
    {
        'key': 'rsna_ich',
        'name': 'RSNA Intracranial Hemorrhage (2K-slice subset)',
        'tier': 2, 'size_mb': 800,
        'target': DATA_ROOT / 'rsna_ich',
        'method': 'kaggle',
        'kaggle_path': 'rsna-intracranial-hemorrhage-detection',
        'needs_auth': True,
        'tag': 'beta',
    },
    {
        'key': 'cbis_ddsm_sample',
        'name': 'CBIS-DDSM Mammography Sample',
        'tier': 2, 'size_mb': 1000,
        'target': DATA_ROOT / 'cbis_ddsm_sample',
        'method': 'hf-dataset',
        'hf_repo': 'AyoubChLin/CBIS-DDSM_breast_density_mass_calcification',
        'hf_repo_alt': [],
        'needs_auth': False,
        'tag': 'beta',
    },

    # ── TIER 3 ── full training, 50-500 GB total — ONLY for re-training
    {
        'key': 'brats_full',
        'name': 'BraTS 2021 Full',
        'tier': 3, 'size_mb': 50000,
        'target': DATA_ROOT / 'brats_full',
        'method': 'manual',
        'manual_url': 'https://www.synapse.org/brats2021',
        'needs_auth': True,
        'tag': 'research',
    },
    {
        'key': 'mimic_cxr',
        'name': 'MIMIC-CXR (chest X-ray + reports)',
        'tier': 3, 'size_mb': 470000,
        'target': DATA_ROOT / 'mimic_cxr',
        'method': 'manual',
        'manual_url': 'https://physionet.org/content/mimic-cxr/',
        'needs_auth': True,
        'tag': 'research',
    },
]


def fmt_mb(n_mb):
    if n_mb >= 1024:
        return f'{n_mb/1024:.1f} GB'
    return f'{n_mb} MB'


# ── DOWNLOAD METHODS ────────────────────────────────────────────────────────

def download_hf_dataset(spec):
    """Download a HuggingFace dataset with snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f'  {R}✗{NC} huggingface_hub not installed')
        return False

    repos = [spec['hf_repo']] + (spec.get('hf_repo_alt', []) or [])
    spec['target'].mkdir(parents=True, exist_ok=True)

    for repo in repos:
        try:
            print(f'    → trying {repo}…', flush=True)
            snapshot_download(
                repo_id=repo,
                repo_type='dataset',
                local_dir=str(spec['target']),
                local_dir_use_symlinks=False,
            )
            print(f'    {G}✓{NC} downloaded to {spec["target"]}')
            return True
        except Exception as e:
            print(f'    {R}✗{NC} {str(e)[:80]}')
            continue
    return False


def download_medmnist(spec):
    """Download MedMNIST datasets via the medmnist Python package."""
    try:
        import medmnist
    except ImportError:
        print(f'    Installing medmnist…')
        rc = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', 'medmnist'],
            capture_output=False,
        ).returncode
        if rc != 0:
            print(f'  {R}✗{NC} pip install medmnist failed')
            return False
        import medmnist

    spec['target'].mkdir(parents=True, exist_ok=True)
    success = True
    for subset in spec.get('subsets', []):
        try:
            cls_name = ''.join(s.capitalize() for s in subset.replace('mnist', 'MNIST').split('_'))
            DataClass = getattr(medmnist, cls_name, None)
            if DataClass is None:
                # medmnist exposes classes like ChestMNIST, BrainMNIST
                cls_name = subset.replace('mnist', '').capitalize() + 'MNIST'
                DataClass = getattr(medmnist, cls_name, None)
            if DataClass is None:
                print(f'    {Y}!{NC} unknown subset: {subset}')
                continue
            print(f'    → {cls_name}…', flush=True)
            DataClass(split='train', download=True, root=str(spec['target']))
            DataClass(split='val', download=True, root=str(spec['target']))
            DataClass(split='test', download=True, root=str(spec['target']))
            print(f'      {G}✓{NC}')
        except Exception as e:
            print(f'      {R}✗{NC} {str(e)[:80]}')
            success = False
    return success


def download_kaggle(spec):
    """Try Kaggle CLI; fall back to HuggingFace mirror if available."""
    if not shutil.which('kaggle'):
        print(f'    {Y}!{NC} kaggle CLI not installed (pip install kaggle)')
        # Try HF mirror
        if spec.get('hf_repo'):
            return download_hf_dataset(spec)
        return False

    spec['target'].mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', spec['kaggle_path'],
             '-p', str(spec['target'])],
            capture_output=True, text=True,
        )
        if rc.returncode == 0:
            print(f'    {G}✓{NC} downloaded')
            return True
        else:
            print(f'    {R}✗{NC} {rc.stderr[:200]}')
            return False
    except Exception as e:
        print(f'    {R}✗{NC} {e}')
        return False


def download_manual(spec):
    """Tier-3 datasets need user to register manually."""
    print(f'    {Y}!{NC} Manual download required:')
    print(f'      1. Visit:    {spec.get("manual_url", "")}')
    print(f'      2. Register / accept license')
    print(f'      3. Download to: {spec["target"]}')
    return False


METHODS = {
    'hf-dataset': download_hf_dataset,
    'medmnist':   download_medmnist,
    'kaggle':     download_kaggle,
    'manual':     download_manual,
}


# ── STATUS ──────────────────────────────────────────────────────────────────

def dir_size_mb(path):
    if not path.exists():
        return 0
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total / (1024 * 1024)


def show_status():
    print('=' * 78)
    print(f'  {B}PUBLIC DATASETS — STATUS{NC}')
    print('=' * 78)
    for tier in (1, 2, 3):
        print()
        print(f'{B}Tier {tier}{NC}')
        for spec in DATASETS:
            if spec['tier'] != tier:
                continue
            size = dir_size_mb(spec['target'])
            if size > 100:
                status = f'{G}✓ {size:>5.0f} MB downloaded{NC}'
            elif size > 1:
                status = f'{Y}⚠ {size:>5.0f} MB (partial){NC}'
            else:
                status = f'{R}✗ not downloaded{NC} (would add ~{fmt_mb(spec["size_mb"])})'
            auth = f' {Y}🔒auth{NC}' if spec.get('needs_auth') else ''
            print(f'  {status}  {spec["name"]}{auth}')


# ── DRIVER ──────────────────────────────────────────────────────────────────

def show_menu():
    print('=' * 78)
    print(f'  {B}SENTINEL PUBLIC DATASET DOWNLOADER{NC}')
    print('=' * 78)
    for tier_n, tier_label, size in [
        (1, 'STARTER (validate models)', 500),
        (2, 'VALIDATION (clinic accuracy)', 5000),
        (3, 'FULL TRAINING (re-train from scratch)', 500_000),
    ]:
        print()
        print(f'  {B}Tier {tier_n}{NC} — {tier_label} (~{fmt_mb(size)})')
        for spec in DATASETS:
            if spec['tier'] != tier_n:
                continue
            auth = ' 🔒' if spec.get('needs_auth') else ''
            print(f'    • {spec["name"]} ({fmt_mb(spec["size_mb"])}){auth}')

    print()
    print(f'  Run:')
    print(f'    python scripts/download_public_datasets.py tier1     # ~500 MB')
    print(f'    python scripts/download_public_datasets.py tier2     # ~5 GB')
    print(f'    python scripts/download_public_datasets.py status    # what\'s already downloaded')
    print(f'    python scripts/download_public_datasets.py <key>     # one specific dataset')


def confirm(prompt, auto_yes=False):
    if auto_yes:
        return True
    try:
        ans = input(f'  {Y}{prompt} [y/N]: {NC}').strip().lower()
        return ans in ('y', 'yes')
    except EOFError:
        return False


def download_set(specs, auto_yes=False):
    total_mb = sum(s['size_mb'] for s in specs)
    print(f'\n  About to download {len(specs)} dataset(s), ~{fmt_mb(total_mb)} total.')
    free_gb = shutil.disk_usage('/').free / (1024**3)
    print(f'  Free disk: {free_gb:.0f} GB.')
    print()

    if not confirm(f'Proceed?', auto_yes):
        print(f'  {Y}Cancelled.{NC}')
        return

    success = 0
    for i, spec in enumerate(specs, 1):
        print()
        print(f'  [{i}/{len(specs)}] {spec["name"]}  (~{fmt_mb(spec["size_mb"])})')
        method_fn = METHODS.get(spec['method'])
        if method_fn:
            ok = method_fn(spec)
            if ok:
                success += 1

    print()
    print('=' * 78)
    print(f'  {G if success == len(specs) else Y}Downloaded {success}/{len(specs)} datasets{NC}')
    print('=' * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', nargs='?', default='menu',
                          help='tier1 | tier2 | tier3 | status | <dataset-key> | menu')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    if args.cmd == 'status':
        show_status()
        return

    if args.cmd == 'menu':
        show_menu()
        return

    if args.cmd in ('tier1', 'tier2', 'tier3'):
        tier_n = int(args.cmd[-1])
        specs = [s for s in DATASETS if s['tier'] == tier_n]
        download_set(specs, args.yes)
        return

    # Single dataset by key
    spec = next((s for s in DATASETS if s['key'] == args.cmd), None)
    if spec:
        download_set([spec], args.yes)
        return

    print(f'{R}Unknown command: {args.cmd}{NC}')
    show_menu()
    sys.exit(1)


if __name__ == '__main__':
    main()
