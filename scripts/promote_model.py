#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PROMOTE A GATED CANDIDATE MODEL INTO THE RELEASE SLOT
================================================================================
  Stage 8 of the release pipeline (docs/dev/release_pipeline_contract.md).
  The ONLY way a new brain-triage model becomes models/brain_triage_finetuned:

    1. refuse unless <candidate>/study_level_eval.json proves the candidate
       earned it UNDER THE RELEASE PROTOCOL — every one of these is checked
       (not just the passed flag, which is only as good as the CLI knobs the
       eval ran with):
         - gate.passed == true AND the gate numbers themselves clear the
           release floors (gate.min_sens ≥ 0.85, gate.min_spec ≥ 0.40,
           gate.sens ≥ 0.85, gate.spec ≥ 0.40);
         - the gate is the production protocol: gate.name 'gate mean @0.50',
           protocol.aggregation 'mean', protocol.threshold 0.50,
           protocol.preprocessing 'registry_predictor_native_res_bilinear_224'
           (brain_analysis._triage_mean_probs — see eval_study_level.py);
         - the eval ran on the release val set: when
           <eval data dir>/../dataset_manifest.json exists (a combined release
           set, stage 5) its splits.val.patients_sha256 must equal the eval's
           val_patients_sha256, the set must have been combined WITH the
           locked holdout (its holdout block: n_ids > 0, every holdout id that
           exists in some batch in val, and — when the locked holdout file
           exists here — the same file sha256), and the eval must have
           checked coverage (holdout_covered true, not skipped with --holdout
           none); otherwise (legacy val dir) the eval must say
           holdout_covered == true for the locked holdout file (--holdout,
           default data/holdout_patients_v1.txt; its sha256 must match the
           file the eval used, when that file exists here);
         - the eval belongs to THESE weights (its manifest pin == the
           candidate's MANIFEST.json pin);
         - promotion.promote == true when a 'promotion' block is present
           (eval ran with --compare-with).
       --force overrides ONLY the last one (the old-vs-new comparison said no:
       a human decision, loudly logged in promotion_log.json). A failed gate,
       a floor below the release floors, a non-production protocol, the wrong
       val set / uncovered holdout or a stale eval are NEVER overridable.
    2. verify the candidate's MANIFEST.json (eval_study_level.check_manifest)
       BEFORE anything moves — a corrupted candidate never displaces a release.
    3. back up the current release → models/brain_triage_finetuned_prev_<old_id>
       (old_id = first 12 hex of sha256(model.safetensors) from its MANIFEST).
    4. move the candidate into place and re-verify the manifest at the new path.
    5. prove the app's registry can load it: a SUBPROCESS with
       SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE=<dest> calls
       src.inference.model_registry.load_hf_classifier_dir(dest, 'cpu') and runs
       one forward pass. Any failure → the move is rolled back (candidate goes
       back to where it was, the backup is restored) and the exit is non-zero.
    6. print old identity → new identity and write promotion_log.json into dest.

  An interrupt (Ctrl-C / SIGTERM / SIGHUP) anywhere in steps 3-5 rolls the move
  back as well: the candidate and the previous release are put back where
  they were, so the orchestrator's --resume can simply run promote again.

  Nothing is copied — directories are moved (rename on the same volume), so a
  343 MB model promotes in well under a second and no plaintext is duplicated.

  Exit codes: 0 promoted · 1 refused / load failed (rolled back) ·
              3 manifest check failed (eval_study_level.EXIT_MANIFEST_FAILED).

  Usage:
    python scripts/promote_model.py --candidate models/brain_triage_candidate_v2
    python scripts/promote_model.py --candidate models/brain_triage_candidate_v2 \
        --dest models/brain_triage_finetuned --force
================================================================================
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package

from eval_study_level import (check_manifest, model_identity, sha256_file,   # noqa: E402
                              EXIT_MANIFEST_FAILED, PROTOCOL_AGGREGATION, PROTOCOL_THRESHOLD,
                              PROTOCOL_PREPROCESSING, DEFAULT_HOLDOUT)

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_REFUSED = 1
DEFAULT_DEST = 'models/brain_triage_finetuned'
# release floors (docs/dev/release_pipeline_contract.md: sens ≥ 0.85, spec ≥ 0.40)
RELEASE_MIN_SENS = 0.85
RELEASE_MIN_SPEC = 0.40
PRODUCTION_GATE_NAME = f'gate {PROTOCOL_AGGREGATION} @{PROTOCOL_THRESHOLD:.2f}'   # 'gate mean @0.50'
LOAD_TIMEOUT_S = 600

# what the subprocess runs: the registry's own loader + one forward pass
_LOAD_PROBE = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from PIL import Image
from src.inference.model_registry import load_hf_classifier_dir
dest = sys.argv[2]
assert os.environ.get('SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE') == dest, 'override env not set'
processor, model = load_hf_classifier_dir(dest, 'cpu')
labels = [str(v) for v in model.config.id2label.values()]
side = int(getattr(model.config, 'image_size', 224) or 224)
img = Image.fromarray(np.zeros((side, side, 3), dtype=np.uint8))
with torch.no_grad():
    probs = torch.softmax(model(**processor(images=img, return_tensors='pt')).logits, dim=-1)[0]
assert probs.shape[-1] == len(labels), 'logit/label mismatch'
print(json.dumps({'labels': labels, 'n_params': sum(p.numel() for p in model.parameters())}))
'''


def refuse(msg: str, code: int = EXIT_REFUSED) -> None:
    print(f"  {R}✗ REFUSED{NC}  {msg}")
    sys.exit(code)


def read_eval(candidate: Path) -> dict:
    fp = candidate / 'study_level_eval.json'
    if not fp.exists():
        refuse(f"{fp} missing — run scripts/eval_study_level.py --model {candidate} --write-manifest first")
    try:
        return json.loads(fp.read_text())
    except Exception as e:
        refuse(f"unreadable {fp}: {e}")
    return {}


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _resolve(path_str: str) -> Path:
    """An eval JSON path as written (absolute, or relative to the cwd the eval
    ran in — normally the repo root)."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def val_set_reasons(ev: dict, holdout: Optional[Path]) -> list[str]:
    """Did the eval run on the release val set? (see module docstring)"""
    reasons = []
    val_sha = ev.get('val_patients_sha256')
    if not val_sha:
        return ["study_level_eval.json has no val_patients_sha256 — re-run scripts/eval_study_level.py "
                "(current version) so the scored patients are recorded"]
    data = ev.get('data')
    ds_manifest = _resolve(str(data)).parent / 'dataset_manifest.json' if data else None
    if ds_manifest is not None and ds_manifest.is_file():
        try:
            want = ((json.loads(ds_manifest.read_text()).get('splits') or {}).get('val') or {}).get('patients_sha256')
        except Exception as e:
            return [f"unreadable {ds_manifest}: {e}"]
        if not want:
            reasons.append(f"{ds_manifest} has no splits.val.patients_sha256 — re-run combine_training_sets.py")
        elif want != val_sha:
            reasons.append(f"eval scored a different patient set (val_patients_sha256 {val_sha[:12]}…) than "
                           f"{ds_manifest} pins for val ({want[:12]}…) — evaluate on {data} as combined")
        reasons.extend(_combined_holdout_reasons(ds_manifest, ev, holdout))
        return reasons
    # legacy val dir: the locked holdout must be fully covered
    if ev.get('holdout_covered') is not True:
        reasons.append(f"holdout_covered is {ev.get('holdout_covered')!r} and {data}/../dataset_manifest.json "
                       f"does not exist — the eval must cover every locked holdout patient "
                       f"(eval_study_level.py --holdout {holdout or DEFAULT_HOLDOUT})")
    elif holdout is not None and holdout.is_file():
        used = (ev.get('holdout') or {}).get('sha256')
        if used != sha256_file(holdout):
            reasons.append(f"the eval checked holdout coverage against a different file "
                           f"({(ev.get('holdout') or {}).get('file')}, sha {str(used)[:12]}…) than the locked "
                           f"holdout {holdout}")
    return reasons


def _combined_holdout_reasons(ds_manifest: Path, ev: dict, holdout: Optional[Path]) -> list[str]:
    """A combined release set must carry the locked holdout in val (contract stage 5), and the
    eval must have checked it — a set combined without --holdout and an eval run with
    --holdout none would otherwise promote on a random val split (review round 2)."""
    reasons: list[str] = []
    if ev.get('holdout_covered') is False:
        reasons.append(f"{len(ev.get('holdout_missing') or [])} locked holdout patients were not scored "
                       f"(holdout_covered false)")
    elif ev.get('holdout_covered') is not True:
        reasons.append(f"holdout_covered is {ev.get('holdout_covered')!r}: the eval did not check the locked "
                       f"holdout (eval_study_level.py --holdout {holdout or DEFAULT_HOLDOUT}, never 'none')")
    try:
        hb = json.loads(ds_manifest.read_text()).get('holdout') or {}
    except Exception:
        hb = {}
    n_ids = hb.get('n_ids') or 0
    present = n_ids - len(hb.get('missing_from_all_batches') or [])
    if not n_ids or not hb.get('sha256'):
        reasons.append(f"{ds_manifest} records no locked holdout (combine_training_sets.py was run without "
                       f"--holdout {holdout or DEFAULT_HOLDOUT}): the gate would score a random val split")
    elif (hb.get('n_in_val') or 0) < present:
        reasons.append(f"{ds_manifest}: only {hb.get('n_in_val')} of {present} locked holdout patients are in val")
    elif holdout is not None and holdout.is_file() and hb.get('sha256') != sha256_file(holdout):
        reasons.append(f"{ds_manifest} was combined with a different holdout file ({hb.get('file')}, sha "
                       f"{str(hb.get('sha256'))[:12]}…) than the locked holdout {holdout}")
    return reasons


def protocol_reasons(ev: dict) -> list[str]:
    """The gate must be the production protocol (gate name, aggregation/threshold, preprocessing).
    Shared with packaging/publish_model_bundle.py so a direct publish checks the same rule."""
    hard: list[str] = []
    gate = ev.get('gate') or {}
    if gate.get('name') != PRODUCTION_GATE_NAME:
        hard.append(f"gate is {gate.get('name')!r}, not the production protocol {PRODUCTION_GATE_NAME!r} "
                    f"(eval_study_level.py --gate-agg / --gate-threshold were changed)")
    proto = ev.get('protocol') or {}
    if proto.get('aggregation') != PROTOCOL_AGGREGATION or _num(proto.get('threshold')) != PROTOCOL_THRESHOLD:
        hard.append(f"protocol aggregation/threshold is {proto.get('aggregation')!r} @{proto.get('threshold')!r}, "
                    f"not {PROTOCOL_AGGREGATION!r} @{PROTOCOL_THRESHOLD}")
    if proto.get('preprocessing') != PROTOCOL_PREPROCESSING:
        hard.append(f"protocol.preprocessing is {proto.get('preprocessing')!r}, not "
                    f"{PROTOCOL_PREPROCESSING!r} — re-run the current eval_study_level.py (it scores through "
                    f"the production predictor)")
    return hard


def eval_verdict(ev: dict, candidate_manifest: Optional[dict],
                 holdout: Optional[Path] = None) -> tuple[list[str], list[str]]:
    """(hard, soft) reasons NOT to promote; both empty = ok.
    hard = gate / floors / protocol / val set / stale eval — never overridable;
    soft = the old-vs-new comparison said no — the ONLY thing --force overrides."""
    hard: list[str] = []
    soft: list[str] = []
    gate = ev.get('gate') or {}
    if gate.get('passed') is not True:
        hard.append(f"gate.passed is {gate.get('passed')!r} (sens {gate.get('sens')}, spec {gate.get('spec')})")
    # the floors the eval was run with must be the release floors, and the numbers must clear them
    for key, floor in (('min_sens', RELEASE_MIN_SENS), ('min_spec', RELEASE_MIN_SPEC)):
        v = _num(gate.get(key))
        if v is None or v < floor - 1e-9:
            hard.append(f"gate.{key} is {gate.get(key)!r} — the release floor is {floor:.2f} "
                        f"(eval_study_level.py --{key.replace('_', '-')} was lowered)")
    for key, floor in (('sens', RELEASE_MIN_SENS), ('spec', RELEASE_MIN_SPEC)):
        v = _num(gate.get(key))
        if v is None or v < floor - 1e-9:
            hard.append(f"gate.{key} {gate.get(key)!r} is below the release floor {floor:.2f}")
    # the gate must be the production protocol
    hard.extend(protocol_reasons(ev))
    # the gate must have run on the release val set
    hard.extend(val_set_reasons(ev, holdout))
    # the eval must be about THESE weights
    eval_pin = ((ev.get('manifest') or {}).get('files') or {}).get('model.safetensors')
    disk_pin = (candidate_manifest or {}).get('files', {}).get('model.safetensors')
    if eval_pin and disk_pin and eval_pin != disk_pin:
        hard.append(f"study_level_eval.json was produced for weights {eval_pin[:12]}… but "
                    f"MANIFEST.json pins {disk_pin[:12]}… — re-run the eval on the candidate")
    if not eval_pin:
        hard.append("study_level_eval.json carries no manifest pin — re-run the eval with --write-manifest")
    # the comparison (the only overridable verdict)
    promo = ev.get('promotion')
    if promo is not None and promo.get('promote') is not True:
        soft.append(f"promotion.promote is {promo.get('promote')!r}: {promo.get('reason')}")
    return hard, soft


def registry_can_load(dest: Path, python: str) -> tuple[bool, str]:
    """Load through the app's registry in a clean subprocess (the same code path
    the server uses at startup); returns (ok, detail)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('SENTINEL_MODEL_DIR_OVERRIDE_')}
    env['SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE'] = str(dest)
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    env.setdefault('LOGURU_LEVEL', 'WARNING')
    try:
        proc = subprocess.run([python, '-c', _LOAD_PROBE, str(ROOT), str(dest)],
                              capture_output=True, text=True, timeout=LOAD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return False, f'registry load timed out after {LOAD_TIMEOUT_S}s'
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:] or ['(no output)']
        return False, (f'registry load failed (exit {proc.returncode}, interpreter {python}):\n      '
                       + '\n      '.join(tail))
    last = (proc.stdout.strip().splitlines() or ['{}'])[-1]
    return True, last


class _signals_raise_interrupt:
    """Turn SIGTERM / SIGHUP into KeyboardInterrupt while the release slot is in flux, so the
    rollback handler runs (the default action would kill the process mid-move)."""

    def __enter__(self):
        import signal
        self._old = {}
        for name in ('SIGTERM', 'SIGHUP'):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    self._old[sig] = signal.signal(sig, self._raise)
                except (ValueError, OSError):      # not the main thread / unsupported
                    pass
        return self

    @staticmethod
    def _raise(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')

    def __exit__(self, *exc):
        import signal
        for sig, old in self._old.items():
            signal.signal(sig, old)
        return False


def _unique_backup(dest: Path, old_id: str) -> Path:
    backup = dest.parent / f"{dest.name}_prev_{old_id}"
    if backup.exists():
        backup = backup.with_name(backup.name + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    return backup


def main():
    ap = argparse.ArgumentParser(description='promote a gated candidate into the release slot')
    ap.add_argument('--candidate', required=True, help='candidate model dir (output of training + eval)')
    ap.add_argument('--dest', default=DEFAULT_DEST, help=f'release slot (default {DEFAULT_DEST})')
    ap.add_argument('--force', action='store_true',
                    help='promote although the old-vs-new comparison said no (promotion.promote false) — '
                         'never overrides a failed gate, floors, protocol or val-set/holdout checks')
    ap.add_argument('--holdout', default=None,
                    help=f'locked holdout ids a legacy (non-combined) val dir must cover '
                         f'(default {DEFAULT_HOLDOUT} when it exists)')
    ap.add_argument('--python', default=sys.executable, help='interpreter for the registry load probe')
    args = ap.parse_args()

    candidate = Path(args.candidate)
    dest = Path(args.dest)
    print(f"\n{'='*64}\n  {B}PROMOTE MODEL{NC}  {candidate} → {dest}\n{'='*64}")
    if not candidate.is_dir():
        refuse(f"candidate {candidate} is not a directory")
    if dest.exists() and not dest.is_dir():
        refuse(f"dest {dest} exists and is not a directory")
    if dest.exists() and candidate.resolve() == dest.resolve():
        refuse("candidate and dest are the same directory")

    # ---- 1. eval verdict --------------------------------------------------
    ev = read_eval(candidate)
    cand_manifest_path = candidate / 'MANIFEST.json'
    if not cand_manifest_path.exists():
        refuse(f"{cand_manifest_path} missing — the release must be pinned (eval with --write-manifest)")
    try:
        cand_manifest = json.loads(cand_manifest_path.read_text())
    except Exception as e:
        refuse(f"unreadable {cand_manifest_path}: {e}")
    holdout = Path(args.holdout) if args.holdout else (DEFAULT_HOLDOUT if DEFAULT_HOLDOUT.is_file() else None)
    if args.holdout and not holdout.is_file():
        refuse(f"--holdout {holdout}: file not found")
    hard, soft = eval_verdict(ev, cand_manifest, holdout)
    reasons = hard + soft
    gate = ev.get('gate') or {}
    print(f"  eval: sens {gate.get('sens', float('nan')):.3f} spec {gate.get('spec', float('nan')):.3f} "
          f"gate.passed={gate.get('passed')} "
          f"promotion.promote={(ev.get('promotion') or {}).get('promote', 'n/a')} "
          f"(evaluated {ev.get('evaluated_at', '?')}, {ev.get('n_studies', '?')} studies)")
    if reasons:
        for r_ in reasons:
            print(f"      - {r_}")
        if hard:
            refuse("candidate did not earn promotion — gate / floors / protocol / val set are not "
                   "overridable (--force only overrides the old-vs-new comparison)"
                   + (" [--force ignored]" if args.force else ""))
        if not args.force:
            refuse("candidate did not earn promotion (the comparison said no; --force overrides that "
                   "ONLY — you own that decision)")
        print(f"  {R}{'!'*64}\n  !! --force: PROMOTING A CANDIDATE THE OLD-vs-NEW COMPARISON REJECTED\n"
              f"  !! (it passed the release gate) — recorded in {dest / 'promotion_log.json'}\n  {'!'*64}{NC}")

    # ---- 2. candidate manifest BEFORE anything moves -----------------------
    print(f"\n  candidate manifest:")
    check_manifest(candidate, cand_manifest_path)                      # exit 3 on mismatch, nothing moved
    new_id = model_identity(candidate, cand_manifest_path)

    # ---- 3. backup current release ----------------------------------------
    old_id = model_identity(dest) if dest.exists() else None
    backup: Optional[Path] = None
    rolled_back = False

    def rollback(why: str) -> None:
        nonlocal rolled_back
        if rolled_back:
            return
        rolled_back = True
        print(f"  {R}✗ {why} — rolling back{NC}")
        if dest.exists() and not candidate.exists():
            shutil.move(str(dest), str(candidate))
            print(f"      candidate restored → {candidate}")
        if backup is not None and backup.exists() and not dest.exists():
            shutil.move(str(backup), str(dest))
            print(f"      previous release restored → {dest}  (identity {old_id})")

    # From the first move until the registry probe has passed, ANY exit — an explicit failure,
    # Ctrl-C during the (up to LOAD_TIMEOUT_S) probe, SIGTERM / SIGHUP — rolls the move back, so
    # an interrupted promote never leaves an unverified model in the release slot.
    with _signals_raise_interrupt():
        try:
            if dest.exists():
                backup = _unique_backup(dest, old_id or 'unknown')
                if old_id == new_id:
                    print(f"  {Y}⚠ candidate has the SAME identity as the current release ({new_id}) — promoting anyway{NC}")
                shutil.move(str(dest), str(backup))
                print(f"  {G}✓ backed up{NC} current release → {backup}  (identity {old_id})")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"  {Y}⚠ no current release at {dest} — first promotion, nothing to back up{NC}")

            # ---- 4. move candidate into place + re-verify ----------------------------
            shutil.move(str(candidate), str(dest))
            print(f"  {G}✓ moved{NC} {candidate} → {dest}")
            print(f"\n  manifest at new path:")
            try:
                check_manifest(dest, dest / 'MANIFEST.json')
            except SystemExit:
                rollback("manifest check failed after the move")
                sys.exit(EXIT_MANIFEST_FAILED)

            # ---- 5. registry load probe ---------------------------------------------
            print(f"\n  registry load probe (SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE={dest}) …")
            ok, detail = registry_can_load(dest, args.python)
            if not ok:
                print(f"      {detail}")
                rollback("the app's registry could not load the promoted dir")
                sys.exit(EXIT_REFUSED)
            print(f"  {G}✓ registry loads it{NC}  {detail}")

        except BaseException as e:
            rollback('promote failed' if isinstance(e, SystemExit)
                     else f'promote interrupted ({type(e).__name__})')
            raise

    # ---- 6. log + summary ---------------------------------------------------
    log = {'promoted_at': datetime.now().isoformat(timespec='seconds'),
           'candidate': str(candidate), 'dest': str(dest),
           'old_identity': old_id, 'new_identity': new_id,
           'backup': str(backup) if backup else None,
           'forced': bool(args.force and reasons), 'refusal_reasons_overridden': reasons if args.force else [],
           'eval': {'gate': gate, 'promotion': ev.get('promotion'), 'evaluated_at': ev.get('evaluated_at'),
                    'n_studies': ev.get('n_studies'), 'data': ev.get('data'),
                    'protocol': ev.get('protocol'), 'n_patients': ev.get('n_patients'),
                    'val_patients_sha256': ev.get('val_patients_sha256'),
                    'holdout_covered': ev.get('holdout_covered')}}
    (dest / 'promotion_log.json').write_text(json.dumps(log, indent=2))
    print(f"\n{'='*64}\n  {G}✓ PROMOTED{NC}  {old_id or '(none)'} → {new_id}\n"
          f"  release: {dest}\n  backup:  {backup or '(none)'}\n"
          f"  next: python scripts/bump_model_release.py --model {dest}\n{'='*64}\n")


if __name__ == '__main__':
    main()
