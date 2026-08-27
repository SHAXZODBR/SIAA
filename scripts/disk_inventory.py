#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — DISK INVENTORY
================================================================================

  Shows EXACTLY what's on your machine for Sentinel + what space each thing
  takes. Run before any download to see the true cost.

  Categories:
    1. AI MODELS         — pretrained weights (HuggingFace cache + Ollama)
    2. PUBLIC DATASETS   — training/validation data downloaded from internet
    3. CLINIC DATA       — your customers' anonymized DICOMs
    4. APP CODE/INSTALLER — the codebase + built installers
    5. OS+DEPS           — Python venv, node_modules, system Python pkgs

  Run:
    python scripts/disk_inventory.py
================================================================================
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    try:
        for root, _, files in os.walk(path, followlinks=False):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total / (1024 * 1024)


def mb(n_mb: float) -> str:
    if n_mb >= 1024:
        return f'{n_mb/1024:.2f} GB'
    return f'{n_mb:.0f} MB'


def section(title: str):
    print()
    print(f'{B}━━━ {title} ━━━{NC}')


def main():
    repo = Path('/Users/shakhzodbtr/Desktop/Siaa_ai/sentinel')
    home = Path.home()

    print('=' * 78)
    print(f'  {B}SENTINEL DISK INVENTORY{NC}  —  what\'s using your MacBook drive')
    print('=' * 78)

    # ── Disk free
    section('Disk free')
    total, used, free = shutil.disk_usage('/')
    print(f'  Total drive:  {total / (1024**3):.0f} GB')
    print(f'  Used:         {used / (1024**3):.0f} GB ({100*used/total:.0f}%)')
    print(f'  Free:         {G}{free / (1024**3):.0f} GB{NC}')

    grand_total_mb = 0.0

    # ── 1. AI MODELS
    section('1. AI MODELS (pretrained weights)')

    # HuggingFace cache
    hf_cache = home / '.cache' / 'huggingface' / 'hub'
    hf_size = dir_size_mb(hf_cache)
    print(f'  HuggingFace cache:        {hf_size:>8.0f} MB    {hf_cache}')

    # List individual HF model dirs
    if hf_cache.exists():
        models = sorted(hf_cache.glob('models--*'))
        for m in models:
            size = dir_size_mb(m)
            name = m.name.replace('models--', '').replace('--', '/')
            tag = f'{Y}STALE{NC}' if size < 1 else ''
            print(f'    • {name:<55} {size:>6.0f} MB  {tag}')
        if not models:
            print(f'    (no models cached yet)')

    # Ollama models
    ollama_dir = home / '.ollama' / 'models'
    ollama_size = dir_size_mb(ollama_dir)
    print(f'  Ollama models:            {ollama_size:>8.0f} MB    {ollama_dir}')
    try:
        r = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines()[1:]:  # skip header
                parts = line.split()
                if parts:
                    print(f'    • {parts[0]:<35}  {parts[2] if len(parts) > 2 else ""}')
    except Exception:
        pass

    # TorchXRayVision cache (in xrv install dir or ~/.cache/torch)
    torch_cache = home / '.cache' / 'torch'
    torch_size = dir_size_mb(torch_cache)
    if torch_size > 0:
        print(f'  Torch hub cache:          {torch_size:>8.0f} MB    {torch_cache}')

    # Custom models in repo
    repo_models = repo / 'models'
    repo_models_size = dir_size_mb(repo_models)
    if repo_models_size > 0:
        print(f'  Repo models/ dir:         {repo_models_size:>8.0f} MB    {repo_models}')

    models_total = hf_size + ollama_size + torch_size + repo_models_size
    grand_total_mb += models_total
    print(f'  {B}AI MODELS SUBTOTAL:       {models_total:>8.0f} MB ({mb(models_total)}){NC}')

    # ── 2. PUBLIC DATASETS
    section('2. PUBLIC DATASETS (training/validation data)')

    dataset_locations = [
        ('Sentinel data/raw/',       repo / 'data' / 'raw'),
        ('Sentinel data/processed/', repo / 'data' / 'processed'),
        ('Sentinel data/test_dicoms/', repo / 'data' / 'test_dicoms'),
        ('Sentinel data/validation/', repo / 'data' / 'validation'),
        ('Sentinel data/public/',    repo / 'data' / 'public'),
        ('MedMNIST cache',           home / '.medmnist'),
        ('Kaggle cache',             home / '.kaggle'),
    ]
    datasets_total = 0.0
    for label, path in dataset_locations:
        size = dir_size_mb(path)
        if size > 0:
            print(f'  {label:<32}  {size:>8.0f} MB  {path}')
            datasets_total += size
    if datasets_total == 0:
        print(f'  {Y}(no public datasets downloaded yet){NC}')
    grand_total_mb += datasets_total
    print(f'  {B}PUBLIC DATASETS SUBTOTAL: {datasets_total:>8.0f} MB ({mb(datasets_total)}){NC}')

    # ── 3. CLINIC DATA (anonymized customer DICOMs)
    section('3. CLINIC DATA (anonymized customer DICOMs)')

    clinic_locations = [
        ('Clinic data dir',     repo / 'data' / 'clinic'),
        ('Doctor corrections',  repo / 'data' / 'doctor_corrections'),
        ('Training corpus',     repo / 'data' / 'training_corpus'),
    ]
    clinic_total = 0.0
    for label, path in clinic_locations:
        size = dir_size_mb(path)
        if size > 0:
            print(f'  {label:<32}  {size:>8.0f} MB  {path}')
            clinic_total += size
    if clinic_total == 0:
        print(f'  {Y}(no clinic data yet — none collected){NC}')
    grand_total_mb += clinic_total
    print(f'  {B}CLINIC DATA SUBTOTAL:     {clinic_total:>8.0f} MB ({mb(clinic_total)}){NC}')

    # ── 4. APP CODE + INSTALLERS
    section('4. APP CODE + INSTALLERS')

    code_locations = [
        ('Repo source (src/)',         repo / 'src'),
        ('Desktop app (desktop-app/)', repo / 'desktop-app' / 'src'),
        ('Training scripts',           repo / 'training'),
        ('Scripts',                    repo / 'scripts'),
        ('Built installers',           repo / 'desktop-app' / 'release'),
        ('Vite build output',          repo / 'desktop-app' / 'dist'),
    ]
    code_total = 0.0
    for label, path in code_locations:
        size = dir_size_mb(path)
        if size > 0:
            print(f'  {label:<32}  {size:>8.0f} MB  {path}')
            code_total += size
    grand_total_mb += code_total
    print(f'  {B}CODE SUBTOTAL:            {code_total:>8.0f} MB ({mb(code_total)}){NC}')

    # ── 5. OS + DEPS
    section('5. OS + DEPS (Python venv, node_modules)')

    deps_locations = [
        ('Python venv',  home / 'venv' / 'lib'),
        ('node_modules', repo / 'desktop-app' / 'node_modules'),
        ('pip cache',    home / 'Library' / 'Caches' / 'pip'),
    ]
    deps_total = 0.0
    for label, path in deps_locations:
        size = dir_size_mb(path)
        if size > 0:
            print(f'  {label:<32}  {size:>8.0f} MB  {path}')
            deps_total += size
    grand_total_mb += deps_total
    print(f'  {B}DEPS SUBTOTAL:            {deps_total:>8.0f} MB ({mb(deps_total)}){NC}')

    # ── GRAND TOTAL
    print()
    print('=' * 78)
    print(f'  {B}GRAND TOTAL (Sentinel-related on disk):  {grand_total_mb:>7.0f} MB ({mb(grand_total_mb)}){NC}')
    print(f'  Free disk after:                              {free/(1024**3):.0f} GB available')
    print('=' * 78)

    # ── 6. WHAT'S NOT YET DOWNLOADED — registered models
    section('6. MODELS REGISTERED BUT NOT YET DOWNLOADED')

    sys.path.insert(0, str(repo))
    try:
        from src.inference.model_registry import REGISTRY
        not_downloaded = []
        for key, card in REGISTRY.items():
            if key == 'brain_2d':
                continue  # alias
            # Heuristic: check if any HF cache dir contains the repo name
            primary = card.repo_id or ''
            if not primary:
                continue
            hf_dir_name = 'models--' + primary.replace('/', '--')
            cached = (hf_cache / hf_dir_name).exists() if hf_cache.exists() else False
            if not cached:
                not_downloaded.append((key, card.display_name, card.download_size_mb, primary))

        if not_downloaded:
            for key, name, size, repo_id in not_downloaded:
                print(f'  • {name:<55}  ~{size:>4} MB  ({repo_id})')
            total_pending = sum(s for _, _, s, _ in not_downloaded)
            print()
            print(f'  Would add: ~{total_pending} MB ({mb(total_pending)})')
        else:
            print(f'  {G}✓ All registered models are downloaded{NC}')
    except Exception as e:
        print(f'  Could not load registry: {e}')

    print()


if __name__ == '__main__':
    main()
