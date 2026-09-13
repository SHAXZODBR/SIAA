#!/usr/bin/env python3
"""
================================================================================
  SENTINEL MEDICAL AI — PREPARE THE OFFLINE MODEL BUNDLE
================================================================================

  A hospital PC has NO internet. This script is the single step, run ONCE on a
  machine WITH internet, that fills the models/ tree the server reads from
  (src/utils/paths.py: MODELS_DIR). Copy models/ to the install afterwards and
  the server never needs the network:

    models/hf/<org>__<name>/            every ModelCard repo_id (snapshot_download)
    models/hf/google__vit-base-patch16-224/, microsoft__resnet-50/   (processor configs only)
    models/xrv/<densenet121-res224-all>.pt   TorchXRayVision chest weights (GitHub release)
    models/densenet/best_model.pt            legacy chest checkpoint (generated from XRV if absent)
    models/hd-bet_params/release_2.0.0/      HD-BET skull-stripping params (zenodo)
    models/monai_bundles/brats_mri_segmentation/   MONAI BraTS bundle
    models/<modality>_finetuned/             the clinic fine-tunes — NOT downloadable
                                             (shipped by the vendor; encrypt with
                                             python -m src.utils.model_crypto encrypt)

  Everything is skipped when already present, so re-running is cheap.

  Usage:
    python scripts/download_all_models.py                 # everything
    python scripts/download_all_models.py --only brain_triage --only chest
    python scripts/download_all_models.py --list          # plan + what is present
    python scripts/download_all_models.py --from-cache    # copy from this machine's HF cache, no network
    python scripts/download_all_models.py --dest /Volumes/USB/models   # build the bundle elsewhere
    python scripts/download_all_models.py --verify        # then test-load every key OFFLINE
================================================================================
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Extra keys beyond the registry (not ModelCards)
EXTRA_KEYS = ('chest_checkpoint', 'hd_bet', 'monai_brats', 'generic_processors')
GENERIC_PROCESSORS = ('google/vit-base-patch16-224', 'microsoft/resnet-50')
# Never pull alternative-framework weights, demos or git noise into the bundle.
HF_IGNORE = ['*.h5', '*.ot', '*.msgpack', '*.tflite', '*.onnx', '*.ckpt', '*.gguf', '*.pth.tar',
             '*.md', '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ipynb', '.gitattributes', 'runs/*', 'logs/*']

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'


def dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / 1e6


def _print(status: str, key: str, what: str, note: str = ''):
    col = {'present': G, 'done': G, 'skip': Y, 'fail': R, 'plan': B}[status]
    print(f"  {col}{status:8s}{NC} {key:20s} {what}{('  — ' + note) if note else ''}")


# ----------------------------------------------------------------------------
# items
# ----------------------------------------------------------------------------

def _wanted(rel: str, allow, ignore) -> bool:
    import fnmatch
    name = rel.split('/')[-1]
    if allow and not any(fnmatch.fnmatch(rel, a) or fnmatch.fnmatch(name, a) for a in allow):
        return False
    return not any(fnmatch.fnmatch(rel, i) or fnmatch.fnmatch(name, i) for i in ignore)


def _copy_from_hub_cache(repo_id: str, dest: Path, allow=None) -> int:
    """Materialise a repo out of this machine's HF hub cache into dest (real
    files, symlinks resolved). snapshot_download(local_dir=..., local_files_only=True)
    does not consult the cache, so this is done by hand. Returns files copied."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(repo_id, local_files_only=True))     # cache lookup only
    n = 0
    for src in snap.rglob('*'):
        if not src.is_file():
            continue
        rel = str(src.relative_to(snap))
        if not _wanted(rel, allow, HF_IGNORE):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src.resolve(), out)
        n += 1
    return n


def hf_snapshot(repo_id: str, dest: Path, from_cache: bool, allow=None) -> tuple[bool, str]:
    """Repo → models/hf/<org>__<name>: snapshot_download (internet) or a copy out
    of the local HF cache (--from-cache). Skipped when config.json is there."""
    if (dest / 'config.json').exists() or (allow and any(dest.glob('*.json'))):
        return True, f'present ({dir_size_mb(dest):.0f} MB)'
    dest.mkdir(parents=True, exist_ok=True)
    if from_cache:
        n = _copy_from_hub_cache(repo_id, dest, allow)
        return n > 0, f'copied {n} files from HF cache ({dir_size_mb(dest):.0f} MB)'
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, local_dir=str(dest), ignore_patterns=HF_IGNORE, allow_patterns=allow)
    return True, f'downloaded ({dir_size_mb(dest):.0f} MB)'


def ensure_xrv_weights(dest_dir: Path, name: str = 'densenet121-res224-all') -> tuple[bool, str]:
    import torchxrayvision as xrv
    fname = os.path.basename(xrv.models.model_urls[name]['weights_url'])
    target = dest_dir / fname
    if target.exists():
        return True, f'present ({dir_size_mb(target):.0f} MB)'
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = Path.home() / '.torchxrayvision' / 'models_data' / fname
    if local.exists():
        shutil.copy2(local, target)
        return True, f'copied from {local.parent} ({dir_size_mb(target):.0f} MB)'
    xrv.models.get_weights(name, cache_dir=str(dest_dir))     # GitHub release download
    return True, f'downloaded ({dir_size_mb(target):.0f} MB)'


def ensure_chest_checkpoint(models_dir: Path, name: str = 'densenet121-res224-all') -> tuple[bool, str]:
    """models/densenet/best_model.pt — the wrapper the server lifespan loads
    (class names + thresholds + XRV metadata). Generated from the XRV weights
    when absent, so the bundle is complete without a training run."""
    ckpt = models_dir / 'densenet' / 'best_model.pt'
    if ckpt.exists():
        return True, f'present ({dir_size_mb(ckpt):.0f} MB)'
    import torch
    import torchxrayvision as xrv
    cache_dir, _ = _xrv_dir(models_dir, name)
    model = xrv.models.DenseNet(weights=name, cache_dir=cache_dir)
    class_names = [p for p in model.pathologies if p]
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'class_names': class_names,
        'optimal_thresholds': [0.5] * len(class_names),
        'image_size': 224,
        'model_name': name,
        'xrv_weights': name,
        'is_pretrained': True,
        'source': 'torchxrayvision',
        'model_state_dict': model.state_dict(),
    }, str(ckpt))
    return True, f'generated from XRV ({dir_size_mb(ckpt):.0f} MB)'


def _xrv_dir(models_dir: Path, name: str):
    import torchxrayvision as xrv
    fname = os.path.basename(xrv.models.model_urls[name]['weights_url'])
    d = models_dir / 'xrv'
    return str(d), (d / fname if (d / fname).exists() else None)


def ensure_hdbet(dest: Path) -> tuple[bool, str]:
    """HD-BET params into models/hd-bet_params/release_2.0.0 (copied from
    ~/hd-bet_params when this machine already has them, else zenodo)."""
    if (dest / 'fold_all' / 'checkpoint_final.pth').exists():
        return True, f'present ({dir_size_mb(dest):.0f} MB)'
    try:
        import HD_BET.paths as hp
    except ImportError:
        return False, 'HD-BET not installed (pip install HD-BET) — skull-stripping is optional'
    local = Path(hp.folder_with_parameter_files)
    if (local / 'fold_all' / 'checkpoint_final.pth').exists():
        shutil.copytree(local, dest, dirs_exist_ok=True)
        return True, f'copied from {local} ({dir_size_mb(dest):.0f} MB)'
    import HD_BET.checkpoint_download as cd
    dest.mkdir(parents=True, exist_ok=True)
    cd.folder_with_parameter_files = str(dest)      # module copy used by the downloader
    cd.maybe_download_parameters()
    return True, f'downloaded ({dir_size_mb(dest):.0f} MB)'


def ensure_monai_brats(bundles_dir: Path) -> tuple[bool, str]:
    dest = bundles_dir / 'brats_mri_segmentation'
    if (dest / 'models' / 'model.pt').exists():
        return True, f'present ({dir_size_mb(dest):.0f} MB)'
    from monai.bundle import download
    bundles_dir.mkdir(parents=True, exist_ok=True)
    download(name='brats_mri_segmentation', bundle_dir=str(bundles_dir))
    ok = (dest / 'models' / 'model.pt').exists()
    return ok, (f'downloaded ({dir_size_mb(dest):.0f} MB)' if ok else 'download did not produce models/model.pt')


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------

def build_plan(registry, models_dir: Path, include_fallbacks: bool) -> list[dict]:
    from src.utils.paths import hf_bundle_dir
    plan = []
    for key, card in registry.items():
        if key == 'brain_2d':
            continue
        if card.backend == 'xrv':
            plan.append({'key': key, 'kind': 'xrv', 'what': 'torchxrayvision densenet121-res224-all → models/xrv/',
                         'size_mb': card.download_size_mb})
            continue
        repos = [card.repo_id] + (card.fallback_repos if include_fallbacks else [])
        repos = [r for r in repos if r]
        if not repos:
            plan.append({'key': key, 'kind': 'local', 'what': f'local-only: models/{card.modality}_finetuned/ (vendor-shipped)',
                         'size_mb': 0})
            continue
        for repo in repos:
            plan.append({'key': key, 'kind': 'hf', 'repo': repo, 'dest': hf_bundle_dir(repo),
                         'what': f'{repo} → {hf_bundle_dir(repo).relative_to(models_dir.parent) if hf_bundle_dir(repo).is_relative_to(models_dir.parent) else hf_bundle_dir(repo)}',
                         'size_mb': card.download_size_mb})
    plan.append({'key': 'generic_processors', 'kind': 'generic', 'what': 'processor configs for repos without one', 'size_mb': 1})
    plan.append({'key': 'chest_checkpoint', 'kind': 'ckpt', 'what': 'models/densenet/best_model.pt (legacy chest wrapper)', 'size_mb': 30})
    plan.append({'key': 'hd_bet', 'kind': 'hdbet', 'what': 'HD-BET params → models/hd-bet_params/release_2.0.0/', 'size_mb': 120})
    plan.append({'key': 'monai_brats', 'kind': 'monai', 'what': 'MONAI brats_mri_segmentation → models/monai_bundles/', 'size_mb': 36})
    return plan


def is_present(item: dict, models_dir: Path) -> bool:
    k = item['kind']
    if k == 'hf':
        return (item['dest'] / 'config.json').exists()
    if k == 'xrv':
        return _xrv_dir(models_dir, 'densenet121-res224-all')[1] is not None
    if k == 'ckpt':
        return (models_dir / 'densenet' / 'best_model.pt').exists()
    if k == 'hdbet':
        return (models_dir / 'hd-bet_params' / 'release_2.0.0' / 'fold_all' / 'checkpoint_final.pth').exists()
    if k == 'monai':
        return (models_dir / 'monai_bundles' / 'brats_mri_segmentation' / 'models' / 'model.pt').exists()
    if k == 'generic':
        from src.utils.paths import hf_bundle_dir
        return all((hf_bundle_dir(g) / 'preprocessor_config.json').exists() for g in GENERIC_PROCESSORS)
    if k == 'local':
        d = models_dir / f"{item['key']}_finetuned"
        return (d / 'config.json').exists()
    return False


def run_item(item: dict, models_dir: Path, from_cache: bool) -> tuple[bool, str]:
    from src.utils.paths import hf_bundle_dir
    k = item['kind']
    if k == 'hf':
        return hf_snapshot(item['repo'], item['dest'], from_cache)
    if k == 'xrv':
        return ensure_xrv_weights(models_dir / 'xrv')
    if k == 'ckpt':
        return ensure_chest_checkpoint(models_dir)
    if k == 'hdbet':
        return ensure_hdbet(models_dir / 'hd-bet_params' / 'release_2.0.0')
    if k == 'monai':
        return ensure_monai_brats(models_dir / 'monai_bundles')
    if k == 'generic':
        notes, all_ok = [], True
        for g in GENERIC_PROCESSORS:
            try:
                _, note = hf_snapshot(g, hf_bundle_dir(g), from_cache, allow=['*.json'])
            except Exception as e:
                all_ok, note = False, f'{type(e).__name__}: {str(e)[:100]}'
            notes.append(f'{g}: {note}')
        return all_ok, '; '.join(notes)
    if k == 'local':
        d = models_dir / f"{item['key']}_finetuned"
        return (d / 'config.json').exists(), ('present' if (d / 'config.json').exists()
                                             else f'not downloadable — vendor ships {d}')
    return False, f'unknown kind {k}'


def verify_offline(models_dir: Path, keys: list[str]) -> int:
    """Test-load each registry key in a SUBPROCESS with SENTINEL_OFFLINE=1 and
    the HF cache hidden, i.e. exactly what the hospital PC will do."""
    code = r'''
import json, sys
from src.inference.model_registry import get_model
out = {}
for k in sys.argv[1:]:
    try:
        e = get_model(k, device="cpu")
        out[k] = {"loaded": bool(e and e.get("available")), "reason": (e or {}).get("reason", "unknown key")}
    except Exception as ex:
        out[k] = {"loaded": False, "reason": f"RAISED {type(ex).__name__}: {ex}"}
print("__VERIFY__" + json.dumps(out))
'''
    env = dict(os.environ)
    env.update({'SENTINEL_OFFLINE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'SENTINEL_MODELS_DIR': str(models_dir), 'HF_HUB_CACHE': str(models_dir / '.empty_hf_cache'),
                'SENTINEL_SKIP_WARMUP': '1'})
    (models_dir / '.empty_hf_cache').mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, '-c', code, *keys], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=1800)
    line = next((l for l in proc.stdout.splitlines() if l.startswith('__VERIFY__')), None)
    if line is None:
        print(f"  {R}verify subprocess failed{NC}\n{proc.stderr[-2000:]}")
        return 1
    res = json.loads(line[len('__VERIFY__'):])
    bad = 0
    print(f"\n{'=' * 70}\n  OFFLINE LOAD CHECK (SENTINEL_OFFLINE=1, HF cache hidden)\n{'=' * 70}")
    for k, r in res.items():
        _print('done' if r['loaded'] else 'fail', k, 'loads offline' if r['loaded'] else r['reason'])
        bad += 0 if r['loaded'] else 1
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare the Sentinel offline model bundle (models/)')
    ap.add_argument('--only', action='append', default=[],
                    help='registry key or one of: ' + ', '.join(EXTRA_KEYS) + ' (repeatable, or comma-separated)')
    ap.add_argument('--dest', default=None, help='bundle root (default: <repo>/models or $SENTINEL_MODELS_DIR)')
    ap.add_argument('--from-cache', action='store_true', help='no network: copy HF repos out of this machine\'s HF cache')
    ap.add_argument('--fallbacks', action='store_true', help='also bundle every card\'s fallback repos')
    ap.add_argument('--list', action='store_true', help='print the plan and what is present; download nothing')
    ap.add_argument('--verify', action='store_true', help='afterwards test-load the keys offline in a subprocess')
    args = ap.parse_args()

    # This script is the ONE place downloads are allowed; decide before `import src`
    # (src/__init__ evaluates SENTINEL_OFFLINE at import).
    if args.dest:
        os.environ['SENTINEL_MODELS_DIR'] = str(Path(args.dest).expanduser().resolve())
    os.environ['SENTINEL_OFFLINE'] = '1' if args.from_cache else '0'
    if args.from_cache:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from src.inference.model_registry import REGISTRY
    from src.utils.paths import MODELS_DIR

    only = [k.strip() for chunk in args.only for k in chunk.split(',') if k.strip()]
    unknown = [k for k in only if k not in REGISTRY and k not in EXTRA_KEYS]
    if unknown:
        print(f"{R}unknown key(s): {', '.join(unknown)}{NC}\n  registry: {', '.join(k for k in REGISTRY if k != 'brain_2d')}"
              f"\n  extras:   {', '.join(EXTRA_KEYS)}")
        return 2

    plan = build_plan(REGISTRY, MODELS_DIR, args.fallbacks)
    if only:
        plan = [it for it in plan if it['key'] in only]

    print('=' * 70)
    print(f"  SENTINEL — offline model bundle → {MODELS_DIR}")
    print(f"  mode: {'copy from local HF cache (no network)' if args.from_cache else 'download (internet required)'}")
    print('=' * 70)

    failed = []
    total_mb = 0.0
    for it in plan:
        present = is_present(it, MODELS_DIR)
        if args.list:
            _print('present' if present else 'plan', it['key'], it['what'],
                   '' if present else f"~{it['size_mb']} MB")
            continue
        try:
            ok, note = run_item(it, MODELS_DIR, args.from_cache)
        except Exception as e:
            ok, note = False, f'{type(e).__name__}: {str(e)[:160]}'
        _print('done' if ok else 'fail', it['key'], it['what'], note)
        if not ok:
            failed.append((it['key'], note))
    if args.list:
        return 0

    for sub in ('hf', 'xrv', 'densenet', 'hd-bet_params', 'monai_bundles'):
        p = MODELS_DIR / sub
        if p.exists():
            mb = dir_size_mb(p)
            total_mb += mb
            print(f"  {sub + '/':18s} {mb:8.0f} MB")
    print(f"  {'bundle total':18s} {total_mb:8.0f} MB  ({MODELS_DIR})")

    if failed:
        print(f"\n  {R}✗ NOT READY:{NC}")
        for k, note in failed:
            print(f"      • {k}: {note}")
        print("\n  → HF repos need internet (or --from-cache on a machine whose HF cache has them);")
        print("    gated repos need `huggingface-cli login`. Re-run: only missing items are fetched.")
    else:
        print(f"\n  {G}✓ bundle complete.{NC} Copy {MODELS_DIR} to the install; the server then runs with")
        print("    SENTINEL_OFFLINE=1 (the default outside dev mode) and never touches the network.")

    rc = 1 if failed else 0
    if args.verify:
        keys = only or [k for k in REGISTRY if k != 'brain_2d' and REGISTRY[k].backend in ('huggingface', 'xrv')]
        keys = [k for k in keys if k in REGISTRY]
        rc = max(rc, verify_offline(MODELS_DIR, keys))
    return rc


if __name__ == '__main__':
    sys.exit(main())
