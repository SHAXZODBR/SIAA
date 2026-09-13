#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — STUDY-LEVEL EVALUATION OF A SLICE CLASSIFIER  (+ release gate)
================================================================================
  Slice-level accuracy understates the clinically-relevant number: a radiologist
  triages a STUDY, not one slice. This script groups val slices by patient
  (filenames are {patient_id}_{slice_idx}.png), aggregates slice probabilities
  per study, and reports study-level sensitivity/specificity/accuracy.

  Aggregations compared:
    mean  — average abnormal-probability across slices
    max   — most-suspicious slice decides ("any slice abnormal → study abnormal")
    vote  — majority vote of slice argmax

  RELEASE GATE (exit code != 0 blocks a ship):
    --min-sens / --min-spec   study-level floor on the gate aggregation
                              (default mean-prob @0.5 — the protocol
                              brain_analysis.py runs in production)
                              → exit 2 when either floor is missed
    MANIFEST.json             sha256 of the weight files pinned next to the
                              model. When present it is verified BEFORE the
                              eval runs → exit 3 on any mismatch / missing file.
                              --write-manifest (re)creates it from the current
                              files. The model.safetensors hash is the one
                              /analyze/study reports as model_identity.sha256_12.
                              Encrypted-at-rest dirs (model.safetensors.enc, see
                              src/utils/model_crypto.py) are supported: the
                              PLAINTEXT sha comes from the .enc.meta.json sidecar
                              (so the pin is the same as for the plaintext dir)
                              and the .enc bytes are pinned as well.

  Usage:
    python scripts/eval_study_level.py --model models/brain_triage_local \
        --data data/local_finetune_bin/val --device mps
    python scripts/eval_study_level.py --model models/brain_triage_finetuned \
        --write-manifest --min-sens 0.85 --min-spec 0.40
================================================================================
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_GATE_FAILED = 2
EXIT_MANIFEST_FAILED = 3

# weight/config files that define a model release (hashed if present)
MANIFEST_FILES = ('config.json', 'preprocessor_config.json',
                  'model.safetensors', 'pytorch_model.bin', 'model.pt')


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_plaintext_sha(model_dir: Path, name: str):
    """plaintext sha256 of <name> when only <name>.enc (+ sidecar) is on disk."""
    enc = model_dir / (name + '.enc')
    meta = model_dir / (name + '.enc.meta.json')
    if enc.exists() and meta.exists():
        try:
            return json.loads(meta.read_text()).get('plaintext_sha256')
        except Exception:
            return None
    return None


def write_manifest(model_dir: Path, manifest_path: Path) -> dict:
    files = {}
    for name in MANIFEST_FILES:
        if (model_dir / name).exists():
            files[name] = sha256_file(model_dir / name)
        elif _sidecar_plaintext_sha(model_dir, name):
            files[name] = _sidecar_plaintext_sha(model_dir, name)          # plaintext pin (stable)
            files[name + '.enc'] = sha256_file(model_dir / (name + '.enc'))  # ciphertext as shipped
    if not files:
        print(f"  {R}✗ no weight/config files to hash in {model_dir}{NC}")
        sys.exit(EXIT_MANIFEST_FAILED)
    manifest = {'model_dir': str(model_dir), 'created_at': datetime.now().isoformat(timespec='seconds'),
                'algorithm': 'sha256', 'files': files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  {G}✓ wrote manifest{NC} {manifest_path}")
    for name, digest in files.items():
        print(f"      {name:28s} {digest[:12]}…")
    return manifest


def check_manifest(model_dir: Path, manifest_path: Path) -> dict:
    """Verify every file pinned in the manifest. Exits 3 on the first problem."""
    try:
        manifest = json.loads(manifest_path.read_text())
        pinned = manifest['files']
    except Exception as e:
        print(f"  {R}✗ unreadable manifest {manifest_path}: {e}{NC}")
        sys.exit(EXIT_MANIFEST_FAILED)
    problems = []
    for name, expected in pinned.items():
        fp = model_dir / name
        if fp.exists():
            actual = sha256_file(fp)
        elif not name.endswith('.enc') and _sidecar_plaintext_sha(model_dir, name):
            actual = _sidecar_plaintext_sha(model_dir, name)   # encrypted at rest: plaintext sha via sidecar
        else:
            problems.append(f"{name}: missing")
            continue
        if actual != expected:
            problems.append(f"{name}: sha256 {actual[:12]}… != pinned {str(expected)[:12]}…")
    if problems:
        print(f"  {R}✗ MANIFEST CHECK FAILED{NC} ({manifest_path})")
        for p in problems:
            print(f"      {p}")
        print(f"  weights on disk are not the release that was validated — refusing to evaluate")
        sys.exit(EXIT_MANIFEST_FAILED)
    print(f"  {G}✓ manifest OK{NC} ({len(pinned)} files match {manifest_path})")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/brain_triage_local')
    ap.add_argument('--data', default='data/local_finetune_bin/val')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--abnormal-class', default='abnormal')
    ap.add_argument('--out', default=None, help='eval JSON path (default: <model>/study_level_eval.json)')
    # release gate
    ap.add_argument('--min-sens', type=float, default=0.85, help='study-level sensitivity floor (exit 2 below)')
    ap.add_argument('--min-spec', type=float, default=0.40, help='study-level specificity floor (exit 2 below)')
    ap.add_argument('--gate-agg', choices=['mean', 'max', 'vote'], default='mean',
                    help='aggregation the gate is scored on (mean = production protocol)')
    ap.add_argument('--gate-threshold', type=float, default=0.5, help='decision threshold for --gate-agg')
    ap.add_argument('--manifest', default=None, help='MANIFEST.json path (default: <model>/MANIFEST.json)')
    ap.add_argument('--write-manifest', action='store_true', help='(re)create the sha256 manifest, then evaluate')
    args = ap.parse_args()

    model_dir = Path(args.model)
    manifest_path = Path(args.manifest) if args.manifest else model_dir / 'MANIFEST.json'
    print(f"\n{'='*64}\n  {B}MODEL MANIFEST{NC}  {model_dir}\n{'='*64}")
    if args.write_manifest:
        manifest = write_manifest(model_dir, manifest_path)
    elif manifest_path.exists():
        manifest = check_manifest(model_dir, manifest_path)
    else:
        manifest = None
        print(f"  {Y}⚠ no manifest at {manifest_path} — weights are unpinned; "
              f"run with --write-manifest to pin this release{NC}")

    # Plaintext or encrypted-at-rest dir — same loader the server uses.
    from src.inference.model_registry import load_hf_classifier_dir
    processor, model = load_hf_classifier_dir(model_dir, args.device)
    id2label = model.config.id2label
    abn_idx = next(i for i, l in id2label.items() if l == args.abnormal_class)

    root = Path(args.data)
    # slice probabilities grouped by (study := patient_id from filename, true label)
    studies: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_slices = 0
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        true_cls = cls_dir.name
        for fp in sorted(cls_dir.glob('*.png')):
            pid = fp.stem.rsplit('_', 1)[0]          # {patient_id}_{slice_idx}
            img = Image.open(fp).convert('RGB')
            with torch.no_grad():
                inputs = processor(images=img, return_tensors='pt').to(args.device)
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
            studies[(pid, true_cls)].append(float(probs[int(abn_idx)]))
            n_slices += 1

    print(f"\n{'='*64}\n  {B}STUDY-LEVEL EVALUATION{NC}  ({len(studies)} studies, {n_slices} slices)\n{'='*64}")

    def score(threshold_fn, name):
        tp = tn = fp_ = fn = 0
        for (pid, true_cls), probs in studies.items():
            pred_abn = threshold_fn(probs)
            is_abn = true_cls == args.abnormal_class
            if is_abn and pred_abn: tp += 1
            elif is_abn and not pred_abn: fn += 1
            elif not is_abn and pred_abn: fp_ += 1
            else: tn += 1
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp_, 1)
        acc = (tp + tn) / max(tp + tn + fp_ + fn, 1)
        bal = (sens + spec) / 2
        print(f"  {name:22s} acc {acc:.3f} | sens {sens:.3f} | spec {spec:.3f} | balanced {bal:.3f}"
              f"   (TP {tp} FN {fn} TN {tn} FP {fp_})")
        return {'name': name, 'acc': acc, 'sens': sens, 'spec': spec, 'balanced': bal,
                'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp_}

    results = []
    results.append(score(lambda ps: sum(ps) / len(ps) >= 0.5, 'mean-prob @0.5'))
    results.append(score(lambda ps: max(ps) >= 0.5, 'max-prob @0.5'))
    results.append(score(lambda ps: sum(p >= 0.5 for p in ps) > len(ps) / 2, 'majority vote'))
    # threshold sweep on mean-prob for the best balanced score
    best = None
    for t in [x / 20 for x in range(2, 19)]:
        r = None
        tp = tn = fp_ = fn = 0
        for (pid, true_cls), probs in studies.items():
            pred_abn = (sum(probs) / len(probs)) >= t
            is_abn = true_cls == args.abnormal_class
            if is_abn and pred_abn: tp += 1
            elif is_abn and not pred_abn: fn += 1
            elif not is_abn and pred_abn: fp_ += 1
            else: tn += 1
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp_, 1)
        bal = (sens + spec) / 2
        if best is None or bal > best[1]:
            best = (t, bal, sens, spec)
    t, bal, sens, spec = best
    print(f"\n  {G}best mean-prob threshold {t:.2f}: balanced {bal:.3f} (sens {sens:.3f} / spec {spec:.3f}){NC}")

    # ---- release gate on the production aggregation ----
    gt = args.gate_threshold
    gate_fns = {
        'mean': lambda ps: sum(ps) / len(ps) >= gt,
        'max': lambda ps: max(ps) >= gt,
        'vote': lambda ps: sum(p >= gt for p in ps) > len(ps) / 2,
    }
    print(f"\n{'='*64}\n  {B}RELEASE GATE{NC}  ({args.gate_agg} @{gt:.2f}; floors sens ≥ {args.min_sens:.2f}, "
          f"spec ≥ {args.min_spec:.2f})\n{'='*64}")
    gate = score(gate_fns[args.gate_agg], f'gate {args.gate_agg} @{gt:.2f}')
    gate_ok = gate['sens'] >= args.min_sens and gate['spec'] >= args.min_spec
    gate.update({'min_sens': args.min_sens, 'min_spec': args.min_spec, 'passed': gate_ok})
    if gate_ok:
        print(f"  {G}✓ GATE PASSED{NC}  sens {gate['sens']:.3f} ≥ {args.min_sens:.2f}, "
              f"spec {gate['spec']:.3f} ≥ {args.min_spec:.2f}")
    else:
        print(f"  {R}✗ GATE FAILED{NC}  sens {gate['sens']:.3f} (floor {args.min_sens:.2f}), "
              f"spec {gate['spec']:.3f} (floor {args.min_spec:.2f})")

    out = Path(args.out) if args.out else model_dir / 'study_level_eval.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'results': results,
                               'best_mean_threshold': {'threshold': t, 'balanced': bal,
                                                       'sens': sens, 'spec': spec},
                               'gate': gate,
                               'n_studies': len(studies), 'n_slices': n_slices,
                               'model': str(model_dir),
                               'manifest': {'path': str(manifest_path),
                                            'files': (manifest or {}).get('files')},
                               'evaluated_at': datetime.now().isoformat(timespec='seconds')},
                              indent=2))
    print(f"  saved → {out}\n")
    if not gate_ok:
        sys.exit(EXIT_GATE_FAILED)


if __name__ == '__main__':
    main()
