#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — STUDY-LEVEL EVALUATION OF A SLICE CLASSIFIER  (+ release gate)
================================================================================
  Slice-level accuracy understates the clinically-relevant number: a radiologist
  triages a STUDY, not one slice. This script groups val slices by patient
  (filenames are {patient_id}_{slice_idx}.png), aggregates slice probabilities
  per study, and reports study-level sensitivity/specificity/accuracy, each
  with a Wilson 95% confidence interval (sens_ci / spec_ci as [lo, hi]).

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
    --compare-with <old dir>  the CURRENT release is evaluated on the SAME val
                              set with the SAME protocol and a 'promotion'
                              block is written into the JSON:
                                promote = gate passed
                                          AND balanced ≥ old_balanced − 0.01
                                          AND sens ≥ old_sens − 0.03
                              (a big sensitivity drop is never accepted for a
                              specificity gain) → exit 4 when promote is false.
                              A gate failure still exits 2 (more specific).

  PREPROCESSING PROTOCOL (identical to production, by construction):
    every val PNG is scored through the SAME callable the app uses —
    src.inference.model_registry._make_hf_predictor — fed the native-resolution
    image (no pre-resize): uint8 → RGB → ViTImageProcessor (bilinear 224).
    brain_analysis._triage_mean_probs hands the registry predictor the
    native-resolution normalized slice, so gate numbers ARE production numbers
    (tests/test_triage_protocol_parity.py pins this). The JSON records it as
    protocol.preprocessing = 'registry_predictor_native_res_bilinear_224'
    together with aggregation / threshold / n_central_slices.

  WHICH PATIENTS (so promote_model.py can prove the gate ran on the release
  val set): the JSON carries n_patients, val_patients_sha256 (sha256 of the
  sorted, newline-joined unique patient ids scored — the same formula
  combine_training_sets.py writes as splits.val.patients_sha256) and
  holdout_covered (every id of --holdout, default data/holdout_patients_v1.txt
  when it exists, was scored; null with --holdout none) + holdout_missing.
  Patient id = file stem up to the first '_' (study keys '<pid>', '<pid>_s2',
  '<pid>[_sN]_<batch>' all map to '<pid>').

  Exit codes: 0 ok · 2 gate failed · 3 manifest failed · 4 not promoted.
  The per-study scoring is pure (score_studies / evaluate_studies) so it can be
  unit-tested without a model.

  Usage:
    python scripts/eval_study_level.py --model models/brain_triage_local \
        --data data/local_finetune_bin/val --device mps
    python scripts/eval_study_level.py --model models/brain_triage_finetuned \
        --write-manifest --min-sens 0.85 --min-spec 0.40
    python scripts/eval_study_level.py --model models/brain_triage_candidate_v2 \
        --data data/train_release_v2/val --write-manifest \
        --compare-with models/brain_triage_finetuned
================================================================================
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_GATE_FAILED = 2
EXIT_MANIFEST_FAILED = 3
EXIT_NOT_PROMOTED = 4

# promotion rule (docs/dev/release_pipeline_contract.md §7)
PROMOTE_MAX_BALANCED_DROP = 0.01
PROMOTE_MAX_SENS_DROP = 0.03

# the production protocol (src/inference/brain_analysis.py::_triage_mean_probs)
PROTOCOL_AGGREGATION = 'mean'
PROTOCOL_THRESHOLD = 0.5
PROTOCOL_N_CENTRAL_SLICES = 5
PROTOCOL_PREPROCESSING = 'registry_predictor_native_res_bilinear_224'
DEFAULT_HOLDOUT = Path(__file__).resolve().parent.parent / 'data' / 'holdout_patients_v1.txt'

# weight/config files that define a model release (hashed if present)
MANIFEST_FILES = ('config.json', 'preprocessor_config.json',
                  'model.safetensors', 'pytorch_model.bin', 'model.pt')

# studies := {(patient_id, true_class): [abnormal-prob per slice]}
StudyProbs = dict[tuple[str, str], list[float]]
ThresholdFn = Callable[[list[float]], bool]


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


def model_identity(model_dir: Path, manifest_path: Optional[Path] = None) -> str:
    """First 12 hex of sha256(model.safetensors) — the pin MANIFEST.json holds and
    /analyze/study reports as model_identity.sha256_12. Prefers the manifest pin,
    else hashes the file (or reads the encrypted-at-rest sidecar)."""
    model_dir = Path(model_dir)
    manifest_path = manifest_path or model_dir / 'MANIFEST.json'
    if manifest_path.exists():
        try:
            pin = json.loads(manifest_path.read_text())['files'].get('model.safetensors')
            if pin:
                return str(pin)[:12]
        except Exception:
            pass
    if (model_dir / 'model.safetensors').exists():
        return sha256_file(model_dir / 'model.safetensors')[:12]
    side = _sidecar_plaintext_sha(model_dir, 'model.safetensors')
    if side:
        return str(side)[:12]
    for name in ('pytorch_model.bin', 'model.pt'):
        if (model_dir / name).exists():
            return sha256_file(model_dir / name)[:12]
    return 'unknown'


# ------------------------------------------------------------------ statistics

def wilson_ci(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score 95% interval for a proportion k/n → [lo, hi]. n == 0 → [0, 1]."""
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


# ------------------------------------------------------------------ scoring (pure)

def score_studies(studies: StudyProbs, threshold_fn: ThresholdFn, name: str = '',
                  abnormal_class: str = 'abnormal') -> dict:
    """Confusion counts + sens/spec/acc/balanced (+ Wilson CIs) of one decision
    rule over {(patient_id, true_class): [slice probs]}. No model needed."""
    tp = tn = fp_ = fn = 0
    for (pid, true_cls), probs in studies.items():
        pred_abn = bool(threshold_fn(probs))
        is_abn = true_cls == abnormal_class
        if is_abn and pred_abn: tp += 1
        elif is_abn and not pred_abn: fn += 1
        elif not is_abn and pred_abn: fp_ += 1
        else: tn += 1
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp_, 1)
    acc = (tp + tn) / max(tp + tn + fp_ + fn, 1)
    bal = (sens + spec) / 2
    return {'name': name, 'acc': acc, 'sens': sens, 'spec': spec, 'balanced': bal,
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp_,
            'sens_ci': wilson_ci(tp, tp + fn), 'spec_ci': wilson_ci(tn, tn + fp_)}


def _print_score(r: dict) -> None:
    print(f"  {r['name']:22s} acc {r['acc']:.3f} | sens {r['sens']:.3f} [{r['sens_ci'][0]:.2f}–{r['sens_ci'][1]:.2f}]"
          f" | spec {r['spec']:.3f} [{r['spec_ci'][0]:.2f}–{r['spec_ci'][1]:.2f}] | balanced {r['balanced']:.3f}"
          f"   (TP {r['tp']} FN {r['fn']} TN {r['tn']} FP {r['fp']})")


def gate_fn(agg: str, threshold: float) -> ThresholdFn:
    """Decision rule for one aggregation ('mean' = production protocol)."""
    fns: dict[str, ThresholdFn] = {
        'mean': lambda ps: sum(ps) / len(ps) >= threshold,
        'max': lambda ps: max(ps) >= threshold,
        'vote': lambda ps: sum(p >= threshold for p in ps) > len(ps) / 2,
    }
    return fns[agg]


def best_mean_threshold(studies: StudyProbs, abnormal_class: str = 'abnormal') -> dict:
    """Sweep the mean-prob threshold 0.10…0.90 for the best balanced score."""
    best = None
    for t in [x / 20 for x in range(2, 19)]:
        r = score_studies(studies, gate_fn('mean', t), f'mean @{t:.2f}', abnormal_class)
        if best is None or r['balanced'] > best['balanced']:
            best = {'threshold': t, 'balanced': r['balanced'], 'sens': r['sens'], 'spec': r['spec']}
    return best


def evaluate_studies(studies: StudyProbs, *, abnormal_class: str = 'abnormal',
                     gate_agg: str = 'mean', gate_threshold: float = 0.5,
                     min_sens: float = 0.85, min_spec: float = 0.40,
                     verbose: bool = True) -> tuple[list[dict], dict, dict]:
    """(results, best_mean_threshold, gate) for one model's study probabilities.
    gate carries min_sens/min_spec/passed like the JSON's 'gate' block."""
    results = [
        score_studies(studies, gate_fn('mean', 0.5), 'mean-prob @0.5', abnormal_class),
        score_studies(studies, gate_fn('max', 0.5), 'max-prob @0.5', abnormal_class),
        score_studies(studies, gate_fn('vote', 0.5), 'majority vote', abnormal_class),
    ]
    if verbose:
        for r in results:
            _print_score(r)
    best = best_mean_threshold(studies, abnormal_class)
    if verbose:
        print(f"\n  {G}best mean-prob threshold {best['threshold']:.2f}: balanced {best['balanced']:.3f} "
              f"(sens {best['sens']:.3f} / spec {best['spec']:.3f}){NC}")
        print(f"\n{'='*64}\n  {B}RELEASE GATE{NC}  ({gate_agg} @{gate_threshold:.2f}; floors sens ≥ {min_sens:.2f}, "
              f"spec ≥ {min_spec:.2f})\n{'='*64}")
    gate = score_studies(studies, gate_fn(gate_agg, gate_threshold),
                         f'gate {gate_agg} @{gate_threshold:.2f}', abnormal_class)
    gate_ok = gate['sens'] >= min_sens and gate['spec'] >= min_spec
    gate.update({'min_sens': min_sens, 'min_spec': min_spec, 'passed': gate_ok})
    if verbose:
        _print_score(gate)
        if gate_ok:
            print(f"  {G}✓ GATE PASSED{NC}  sens {gate['sens']:.3f} ≥ {min_sens:.2f}, "
                  f"spec {gate['spec']:.3f} ≥ {min_spec:.2f}")
        else:
            print(f"  {R}✗ GATE FAILED{NC}  sens {gate['sens']:.3f} (floor {min_sens:.2f}), "
                  f"spec {gate['spec']:.3f} (floor {min_spec:.2f})")
    return results, best, gate


def decide_promotion(old_gate: dict, new_gate: dict, old_model: str = '', new_model: str = '') -> dict:
    """Promotion rule: new gate passed AND balanced ≥ old − 0.01 AND sens ≥ old − 0.03.
    Returns the JSON 'promotion' block."""
    d_sens = new_gate['sens'] - old_gate['sens']
    d_spec = new_gate['spec'] - old_gate['spec']
    d_bal = new_gate['balanced'] - old_gate['balanced']
    reasons = []
    if not new_gate.get('passed', False):
        reasons.append(f"gate failed (sens {new_gate['sens']:.3f} floor {new_gate.get('min_sens', 0):.2f}, "
                       f"spec {new_gate['spec']:.3f} floor {new_gate.get('min_spec', 0):.2f})")
    if d_bal < -PROMOTE_MAX_BALANCED_DROP:
        reasons.append(f"balanced {new_gate['balanced']:.3f} < old {old_gate['balanced']:.3f} − {PROMOTE_MAX_BALANCED_DROP:.2f}")
    if d_sens < -PROMOTE_MAX_SENS_DROP:
        reasons.append(f"sens {new_gate['sens']:.3f} < old {old_gate['sens']:.3f} − {PROMOTE_MAX_SENS_DROP:.2f}")
    promote = not reasons
    keys = ('name', 'acc', 'sens', 'spec', 'balanced', 'tp', 'fn', 'tn', 'fp', 'sens_ci', 'spec_ci',
            'min_sens', 'min_spec', 'passed')
    return {
        'old': {**{k: old_gate[k] for k in keys if k in old_gate}, 'model': old_model},
        'new': {**{k: new_gate[k] for k in keys if k in new_gate}, 'model': new_model},
        'promote': promote,
        'reason': ('new model passes the gate and is not worse than the current release '
                   f'(balanced {d_bal:+.3f}, sens {d_sens:+.3f}, spec {d_spec:+.3f})'
                   if promote else '; '.join(reasons)),
        'delta_sens': d_sens, 'delta_spec': d_spec, 'delta_balanced': d_bal,
        'rule': {'max_balanced_drop': PROMOTE_MAX_BALANCED_DROP, 'max_sens_drop': PROMOTE_MAX_SENS_DROP},
    }


def _print_comparison(promo: dict) -> None:
    o, n = promo['old'], promo['new']
    print(f"\n{'='*64}\n  {B}OLD vs NEW{NC}  (same val set, same protocol)\n{'='*64}")
    print(f"  {'':10s} {'sens':>7s} {'95% CI':>13s} {'spec':>7s} {'95% CI':>13s} {'balanced':>9s} {'TP/FN/TN/FP':>15s}")
    for tag, g in (('old', o), ('new', n)):
        counts = f"{g['tp']}/{g['fn']}/{g['tn']}/{g['fp']}"
        print(f"  {tag:10s} {g['sens']:7.3f} {g['sens_ci'][0]:5.2f}–{g['sens_ci'][1]:<6.2f} "
              f"{g['spec']:7.3f} {g['spec_ci'][0]:5.2f}–{g['spec_ci'][1]:<6.2f} {g['balanced']:9.3f} {counts:>15s}")
    print(f"  {'delta':10s} {promo['delta_sens']:+7.3f} {'':13s} {promo['delta_spec']:+7.3f} {'':13s} "
          f"{promo['delta_balanced']:+9.3f}")
    if promo['promote']:
        print(f"  {G}✓ PROMOTE{NC}  {promo['reason']}")
    else:
        print(f"  {R}✗ NOT PROMOTED{NC}  {promo['reason']}")


# ------------------------------------------------------------------ inference

def patient_id_of(study_key: str) -> str:
    """Study key → hospital patient id. Keys are '<pid>', '<pid>_s2' (build_training_set)
    or '<pid>[_sN]_<batch>' (combine_training_sets); ids never contain '_'."""
    return str(study_key).split('_', 1)[0]


def sha256_ids(ids) -> str:
    """sha256 of the sorted, newline-joined unique ids — the formula
    combine_training_sets.py writes as splits.<split>.patients_sha256."""
    return hashlib.sha256('\n'.join(sorted(set(ids))).encode('utf-8')).hexdigest()


def load_ids(fp: Path) -> set[str]:
    """Patient ids, one per line ('#' comments and blank lines ignored)."""
    return {ln.split('#', 1)[0].strip() for ln in Path(fp).read_text(encoding='utf-8').splitlines()
            if ln.split('#', 1)[0].strip()}


def patients_block(studies: StudyProbs, holdout: Optional[Path]) -> dict:
    """n_patients / val_patients_sha256 / holdout coverage of the scored studies."""
    pids = {patient_id_of(k[0]) for k in studies}
    block = {'n_patients': len(pids), 'val_patients_sha256': sha256_ids(pids)}
    if holdout is not None and Path(holdout).is_file():
        ids = load_ids(Path(holdout))
        missing = sorted(ids - pids)
        block.update({'holdout': {'file': str(holdout), 'sha256': sha256_file(Path(holdout)),
                                  'n_ids': len(ids), 'n_scored': len(ids) - len(missing)},
                      'holdout_covered': not missing, 'holdout_missing': missing})
    else:
        block.update({'holdout': None, 'holdout_covered': None, 'holdout_missing': []})
    return block


def make_predictor(model_dir: Path, device: str = 'cpu'):
    """The registry's own predictor for a classifier dir — the SAME callable
    brain_analysis._triage_mean_probs receives in the app (plaintext or
    encrypted-at-rest dir, same loader, same softmax/sigmoid choice as the
    brain_triage card)."""
    from src.inference import model_registry as mr
    processor, model = mr.load_hf_classifier_dir(Path(model_dir), device)
    card = getattr(mr, 'BRAIN_TRIAGE', None)
    return mr._make_hf_predictor(model, processor, device, source=str(model_dir),
                                 multi_label=getattr(card, 'multi_label', None))


def score_png(predictor, fp: Path) -> dict:
    """Class probs of one training/val PNG through the registry predictor's
    3-D branch: native-resolution uint8 RGB → processor (bilinear 224)."""
    import numpy as np
    from PIL import Image
    with Image.open(fp) as im:
        arr = np.asarray(im.convert('RGB'))
    return predictor(arr)


def collect_study_probs(model_dir: Path, data_root: Path, device: str = 'cpu',
                        abnormal_class: str = 'abnormal', predictor=None) -> tuple[StudyProbs, int]:
    """Score every val slice with the registry predictor (make_predictor) —
    the production callable, fed the native-resolution PNG (protocol
    PROTOCOL_PREPROCESSING). The model is loaded ONCE. Returns (studies, n_slices)."""
    if predictor is None:
        predictor = make_predictor(Path(model_dir), device)
    classes = list(getattr(predictor, 'class_names', []) or [])
    if classes and abnormal_class not in classes:
        raise SystemExit(f"  {R}✗ model {model_dir} has no class '{abnormal_class}' (classes: {classes}){NC}")

    studies: StudyProbs = defaultdict(list)
    n_slices = 0
    root = Path(data_root)
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        true_cls = cls_dir.name
        for fp in sorted(cls_dir.glob('*.png')):
            key = fp.stem.rsplit('_', 1)[0]          # {study key}_{slice_idx}
            probs = score_png(predictor, fp)
            if abnormal_class not in probs:
                raise SystemExit(f"  {R}✗ model {model_dir} has no class '{abnormal_class}' "
                                 f"(classes: {sorted(probs)}){NC}")
            studies[(key, true_cls)].append(float(probs[abnormal_class]))
            n_slices += 1
    return dict(studies), n_slices


def protocol_block(gate_agg: str, gate_threshold: float) -> dict:
    return {'aggregation': gate_agg, 'threshold': gate_threshold,
            'n_central_slices': PROTOCOL_N_CENTRAL_SLICES, 'preprocessing': PROTOCOL_PREPROCESSING}


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
    ap.add_argument('--holdout', default=None,
                    help='locked holdout ids (one per line) whose coverage is recorded as holdout_covered '
                         f"(default {DEFAULT_HOLDOUT.relative_to(DEFAULT_HOLDOUT.parent.parent)} when it exists; "
                         "'none' to skip)")
    ap.add_argument('--manifest', default=None, help='MANIFEST.json path (default: <model>/MANIFEST.json)')
    ap.add_argument('--write-manifest', action='store_true', help='(re)create the sha256 manifest, then evaluate')
    # promotion
    ap.add_argument('--compare-with', default=None, metavar='OLD_MODEL_DIR',
                    help='evaluate the current release on the same val set and decide promotion (exit 4 = no)')
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

    old_dir: Optional[Path] = Path(args.compare_with) if args.compare_with else None
    if old_dir is not None:
        if not old_dir.is_dir():
            print(f"  {R}✗ --compare-with {old_dir}: not a directory{NC}")
            sys.exit(1)
        if old_dir.resolve() == model_dir.resolve():
            print(f"  {Y}⚠ --compare-with is the candidate itself — deltas will be zero{NC}")
        old_manifest = old_dir / 'MANIFEST.json'
        print(f"\n{'='*64}\n  {B}OLD MODEL MANIFEST{NC}  {old_dir}\n{'='*64}")
        if old_manifest.exists():
            check_manifest(old_dir, old_manifest)          # tampered baseline → exit 3
        else:
            print(f"  {Y}⚠ no manifest at {old_manifest} — old weights are unpinned{NC}")

    if args.holdout is None:
        holdout: Optional[Path] = DEFAULT_HOLDOUT if DEFAULT_HOLDOUT.is_file() else None
    elif args.holdout.lower() == 'none':
        holdout = None
    else:
        holdout = Path(args.holdout)
        if not holdout.is_file():
            print(f"  {R}✗ --holdout {holdout}: file not found{NC}")
            sys.exit(1)

    root = Path(args.data)
    studies, n_slices = collect_study_probs(model_dir, root, args.device, args.abnormal_class)
    print(f"\n{'='*64}\n  {B}STUDY-LEVEL EVALUATION{NC}  {model_dir}  ({len(studies)} studies, {n_slices} slices)\n{'='*64}")
    results, best, gate = evaluate_studies(
        studies, abnormal_class=args.abnormal_class, gate_agg=args.gate_agg,
        gate_threshold=args.gate_threshold, min_sens=args.min_sens, min_spec=args.min_spec)
    gate_ok = gate['passed']

    promotion = None
    old_block = None
    if old_dir is not None:
        old_studies, old_n = collect_study_probs(old_dir, root, args.device, args.abnormal_class)
        print(f"\n{'='*64}\n  {B}STUDY-LEVEL EVALUATION (OLD){NC}  {old_dir}  ({len(old_studies)} studies, {old_n} slices)\n{'='*64}")
        old_results, old_best, old_gate = evaluate_studies(
            old_studies, abnormal_class=args.abnormal_class, gate_agg=args.gate_agg,
            gate_threshold=args.gate_threshold, min_sens=args.min_sens, min_spec=args.min_spec)
        if set(old_studies) != set(studies):
            print(f"  {Y}⚠ old/new study sets differ ({len(old_studies)} vs {len(studies)}) — check --data{NC}")
        promotion = decide_promotion(old_gate, gate, str(old_dir), str(model_dir))
        promotion['old']['identity'] = model_identity(old_dir)
        promotion['new']['identity'] = model_identity(model_dir, manifest_path)
        old_block = {'model': str(old_dir), 'identity': promotion['old']['identity'],
                     'results': old_results, 'best_mean_threshold': old_best, 'gate': old_gate,
                     'n_studies': len(old_studies), 'n_slices': old_n}
        _print_comparison(promotion)

    out = Path(args.out) if args.out else model_dir / 'study_level_eval.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'results': results,
               'best_mean_threshold': best,
               'gate': gate,
               'n_studies': len(studies), 'n_slices': n_slices,
               'model': str(model_dir),
               'identity': model_identity(model_dir, manifest_path),
               'data': str(root),
               'protocol': protocol_block(args.gate_agg, args.gate_threshold),
               **patients_block(studies, holdout),
               'manifest': {'path': str(manifest_path),
                            'files': (manifest or {}).get('files')},
               'evaluated_at': datetime.now().isoformat(timespec='seconds')}
    if promotion is not None:
        payload['promotion'] = promotion
        payload['compare_with'] = old_block
    out.write_text(json.dumps(payload, indent=2))
    pb = payload
    if pb['holdout_covered'] is True:
        print(f"  {G}✓ holdout covered{NC}  all {pb['holdout']['n_ids']} ids of {pb['holdout']['file']} scored")
    elif pb['holdout_covered'] is False:
        print(f"  {Y}⚠ holdout NOT covered{NC}  {len(pb['holdout_missing'])} of {pb['holdout']['n_ids']} ids of "
              f"{pb['holdout']['file']} not in {root} — promote_model.py will refuse on a legacy val dir")
    print(f"  {pb['n_patients']} patients, val_patients_sha256 {pb['val_patients_sha256'][:12]}…, "
          f"protocol {pb['protocol']['preprocessing']}")
    print(f"  saved → {out}\n")
    if not gate_ok:
        sys.exit(EXIT_GATE_FAILED)
    if promotion is not None and not promotion['promote']:
        sys.exit(EXIT_NOT_PROMOTED)


if __name__ == '__main__':
    main()
