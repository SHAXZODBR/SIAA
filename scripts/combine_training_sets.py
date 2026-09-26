#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — COMBINE BATCH TRAINING SETS INTO ONE RELEASE SET
================================================================================
  Release pipeline stage 5 (docs/dev/release_pipeline_contract.md).

  Merges several build_training_set.py outputs (data/sets/<batch>/{train,val}/
  <cls>/<pid>_<k>.png) into ONE data/train_release_<tag>/{train,val,tune}/<cls>/
  with a PATIENT-LEVEL split across ALL batches:
    - a patient present in two batches lands on one side only (no leakage);
    - the locked holdout ids (data/holdout_patients_v1.txt) are forced into
      val, so the new gate numbers stay comparable with the old release;
    - TRAIN-LOCK: patients a previous model was TRAINED on are locked to train
      and can NEVER be placed in val/tune — otherwise eval_study_level.py
      --compare-with would score the previous release on its own training
      data. Locked are:
        * every patient under train/ of a LEGACY input set (no studies.csv,
          e.g. data/local_finetune_bin, the set release 0d559766ce58 was
          trained on);
        * studies.csv split=train/tune of an input set that is itself a
          combined RELEASE set (has dataset_manifest.json, i.e. what a
          release was trained on);
        * --train-lock <file|dir>: extra ids (one per line), or a release set
          dir (its train_patients.txt + tune_patients.txt, else its train/
          and tune/ PNGs).
      The split column of a build_training_set.py studies.csv is ADVISORY
      (nothing was trained on it yet; this script re-splits across batches),
      so it does not lock. A patient that is BOTH holdout and train-locked is
      a hard error (the ids are listed);
    - --val-frac applies to the NEW (non-holdout, non-locked) patients only,
      stratified by class, shuffled with --seed (deterministic);
    - tune/: --tune-frac (default 0.10) of the NEW train patients, but at
      least --min-tune-per-class (default 10) patients per class while that
      leaves ≥ 2/3 of the class's new train patients in train — patient-
      disjoint from train and val, for checkpoint selection by the trainer
      (training/finetune_brain_classifier.py --select-split) so the gate set
      (val) is never used to pick an epoch. A handful of tune patients would
      make the kept epoch mostly noise; a tune/ smaller than the minimum is
      warned about;
    - files are symlinked (absolute targets) or copied with --copy.

  Cross-set dedupe by patient — ONE class per patient:
    - the LAST set on the command line that holds a patient decides its class
      (pass the NEWEST batch LAST); conflicts are printed (count + ids) and
      recorded in dataset_manifest.json;
    - EXCEPT a locked holdout patient: the FIRST (oldest) set that holds it
      decides, and only that set's slices go to val — the holdout was locked
      on that study, and a follow-up scan in a newer batch must not replace
      it (the gate numbers stay comparable across releases);
    - slices a NON-deciding set holds under a DIFFERENT class are dropped, not
      relabelled: with no study identity in a legacy set they may be another
      study (a 2025 normal scan and a 2026 tumour follow-up), and moving them
      into the other class would be label noise (counted in the manifest);
    - byte-identical slices of a patient are kept once; in train/ and tune/,
      different slices of the same patient from another batch WITH THE SAME
      class are kept as '<pid>_<batch>_<k>.png';
    - in val/ (the gate set) a patient's slices come from ONE batch — the
      deciding one — because eval_study_level.py keys a study by the file-name
      prefix: a second export would be scored as a second study and double
      the patient's weight in sens/spec;
    - a patient that the deciding set holds under two classes (two studies
      with different labels) cannot be merged automatically: it is dropped
      from the combined set, listed, and recorded in the manifest (fatal if
      it is a holdout patient);
    - the written tree is asserted: no patient under two classes, no patient
      in two splits, every holdout patient in val, no train-locked patient in
      val/tune.

  Writes dataset_manifest.json (per-batch counts, class counts per split,
  UNIQUE n_patients / n_studies, seed, holdout sha256, train-lock counts,
  label conflicts, sorted-patient-list sha256 per split),
  {train,val,tune}_patients.txt, train_lock_applied.txt, and studies.csv (the
  batches' studies.csv rows with `batch`, the final `split` and `cls`;
  `cls_in_batch` keeps the batch's own class).

  Hard failure (exit 1) on any split overlap, a holdout patient in train/tune,
  a train-locked patient in val/tune, a holdout id that is also train-locked,
  a patient under two classes, or an empty train/val. Holdout ids that exist
  in no batch are WARNED (listed) — --strict-holdout turns that into a failure.

  Usage:
    python scripts/combine_training_sets.py --sets data/local_finetune_bin data/sets/new1 \
        --holdout data/holdout_patients_v1.txt --out data/train_release_v2 \
        [--train-lock ids.txt|<release set dir>] [--val-frac 0.2] [--tune-frac 0.1] [--seed 42] [--copy] [--fresh]
================================================================================
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

G, Y, B, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0;31m', '\033[0m'
SRC_SPLITS = ('train', 'val', 'tune')    # build_training_set.py writes train/val; a release set also tune
SPLITS = ('train', 'val', 'tune')        # what this script writes


def patient_id_of(png: Path) -> str:
    """'<pid>_<k>.png' or '<pid>_s2_<k>.png' -> '<pid>' (ids never contain '_')."""
    return png.stem.split('_', 1)[0]


def study_prefix_of(png: Path) -> str:
    """'<pid>_<k>.png' -> '<pid>', '<pid>_s2_<k>.png' -> '<pid>_s2' (the study key eval uses)."""
    return png.stem.rsplit('_', 1)[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: set[str]) -> str:
    return hashlib.sha256('\n'.join(sorted(ids)).encode('utf-8')).hexdigest()


def load_ids(path: Optional[Path]) -> tuple[set[str], str]:
    """Patient ids, one per line ('#' comments allowed) -> (ids, file sha256)."""
    if path is None:
        return set(), ''
    ids = {ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.startswith('#')}
    return ids, sha256_file(path)


load_holdout = load_ids


def scan_set(set_dir: Path) -> list[dict]:
    """Every PNG of one batch: {path, patient_id, cls, src_split}."""
    files: list[dict] = []
    for split in SRC_SPLITS:
        split_dir = set_dir / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for fp in sorted(cls_dir.glob('*.png')):
                files.append({'path': fp, 'patient_id': patient_id_of(fp), 'cls': cls_dir.name, 'src_split': split})
    return files


def read_studies_csv(set_dir: Path) -> list[dict]:
    sc = set_dir / 'studies.csv'
    if not sc.is_file():
        return []
    with open(sc, encoding='utf-8') as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def is_release_set(set_dir: Path) -> bool:
    """A combined release set (this script's output) — what a released model was trained on."""
    return (set_dir / 'dataset_manifest.json').is_file()


def train_locked_ids(files: list[dict], study_rows: list[dict], trained_on: bool = False) -> set[str]:
    """Patients a previous model was trained on, as far as this input set tells.

    Legacy layout (no studies.csv): everything under train/. A release set (trained_on):
    train/ + tune/ PNGs and studies.csv split=train/tune. A build_training_set.py output
    (studies.csv, not trained on): nothing — its split is advisory.
    """
    if study_rows and not trained_on:
        return set()
    locked_splits = ('train', 'tune') if trained_on else ('train',)
    locked = {f['patient_id'] for f in files if f['src_split'] in locked_splits}
    locked |= {(r.get('patient_id') or '').strip() for r in study_rows
               if (r.get('split') or '').strip() in locked_splits}
    locked.discard('')
    return locked


def load_train_lock(path: Optional[Path]) -> tuple[set[str], str]:
    """--train-lock: an id file, or a release set dir -> (ids, sha256 of the sorted ids / file)."""
    if path is None:
        return set(), ''
    if path.is_dir():
        ids: set[str] = set()
        lists = [path / f'{sp}_patients.txt' for sp in ('train', 'tune')]
        if any(fp.is_file() for fp in lists):
            for fp in lists:
                if fp.is_file():
                    ids |= load_ids(fp)[0]
        else:
            ids = {f['patient_id'] for f in scan_set(path) if f['src_split'] in ('train', 'tune')}
        return ids, sha256_ids(ids)
    return load_ids(path)


def resolve_patient_classes(sets: list[dict], first_wins: Optional[set[str]] = None,
                            ) -> tuple[dict[str, str], dict[str, str], list[dict], list[dict]]:
    """ONE class per patient across sets: the LAST set (command-line order) holding a patient decides —
    for the ids in `first_wins` (the locked holdout) the FIRST set that holds them with one class.

    sets: [{'name', 'files'}] in command-line order.
    Returns (winner {pid: cls}, winner_set {pid: set name}, cross-set conflicts, within-set conflicts).
      cross-set conflict: {'patient_id', 'classes': {batch: cls or 'a+b'}, 'winner', 'winner_set'}
      within-set conflict: {'patient_id', 'set', 'classes': [..], 'resolved': bool} — unresolved
        (the deciding set itself holds two classes) -> the patient is DROPPED; resolved -> a later
        set decided and the conflicting set's slices follow the winning class.
    """
    per_set: list[tuple[str, dict[str, set[str]]]] = []
    for s in sets:
        m: dict[str, set[str]] = defaultdict(set)
        for f in s['files']:
            m[f['patient_id']].add(f['cls'])
        per_set.append((s['name'], m))
    history: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for name, m in per_set:
        for pid, clss in m.items():
            history[pid][name] = clss
    winner: dict[str, str] = {}
    winner_set: dict[str, str] = {}
    cross: list[dict] = []
    within: list[dict] = []
    first_wins = first_wins or set()
    for pid in sorted(history):
        hist = history[pid]                      # insertion order == command-line order
        last_name, last_cls = list(hist.items())[-1]
        if pid in first_wins:                    # holdout: the study the holdout was locked on
            single = [(n, c) for n, c in hist.items() if len(c) == 1]
            if single:
                last_name, last_cls = single[0]
        resolved = len(last_cls) == 1
        for name, clss in hist.items():
            if len(clss) > 1:
                within.append({'patient_id': pid, 'set': name, 'classes': sorted(clss),
                               'resolved': resolved and name != last_name})
        if not resolved:
            continue
        winner[pid] = next(iter(last_cls))
        winner_set[pid] = last_name
        if len({frozenset(c) for c in hist.values()}) > 1:
            cross.append({'patient_id': pid, 'classes': {n: '+'.join(sorted(c)) for n, c in hist.items()},
                          'winner': winner[pid], 'winner_set': last_name})
    return winner, winner_set, cross, within


def split_patients(all_pids: set[str], holdout: set[str], pid_classes: dict[str, str] | dict[str, set[str]],
                   val_frac: float, seed: int, locked: Optional[set[str]] = None,
                   tune_frac: float = 0.0, min_tune_per_class: int = 0) -> tuple[set[str], set[str], set[str]]:
    """(train_ids, val_ids, tune_ids).

    Holdout ids present -> val. Train-locked ids -> train (never val, never tune).
    The remaining NEW patients are stratified by class: val_frac of them -> val,
    then tune_frac of the rest -> tune (but at least min_tune_per_class, capped at a third
    of the rest, when tune_frac > 0), the rest -> train. Deterministic for a seed.
    """
    locked = set(locked or ())
    val = {p for p in all_pids if p in holdout}
    new = sorted(all_pids - val - locked)
    rng = random.Random(seed)
    by_key: dict[str, list[str]] = defaultdict(list)
    for p in new:
        c = pid_classes[p]
        by_key['+'.join(sorted(c)) if isinstance(c, set) else str(c)].append(p)
    tune: set[str] = set()
    for key in sorted(by_key):
        group = by_key[key]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        val.update(group[:n_val])
        rest = group[n_val:]
        n_tune = int(round(len(rest) * tune_frac))
        if tune_frac > 0 and min_tune_per_class > 0:
            n_tune = max(n_tune, min(min_tune_per_class, len(rest) // 3))
        tune.update(rest[:n_tune])
    return all_pids - val - tune, val, tune


def place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _fmt_ids(ids, n: int = 20) -> str:
    ids = sorted(ids)
    return ', '.join(ids[:n]) + (' …' if len(ids) > n else '')


def main() -> None:
    ap = argparse.ArgumentParser(description='Combine batch training sets with a patient-level split + locked holdout.')
    ap.add_argument('--sets', nargs='+', required=True,
                    help='build_training_set.py output dirs, OLDEST first: the LAST set wins a label conflict')
    ap.add_argument('--holdout', default=None, help='patient ids (one per line) forced into val')
    ap.add_argument('--train-lock', default=None,
                    help='extra patient ids locked to train (never val/tune): a file (one id per line) or the '
                         'release set dir the current model was trained on (data/train_release_<tag>); added to '
                         'the train/ patients of legacy / release input sets')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-frac', type=float, default=0.2, help='val fraction of NEW (non-holdout, non-locked) patients')
    ap.add_argument('--tune-frac', type=float, default=0.1,
                    help='tune fraction of the NEW train patients (checkpoint-selection split; 0 = no tune/)')
    ap.add_argument('--min-tune-per-class', type=int, default=10,
                    help='at least this many tune patients per class (capped at 1/3 of that class\'s new train '
                         'patients); 0 = plain --tune-frac')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--copy', action='store_true', help='copy PNGs instead of symlinking them')
    ap.add_argument('--fresh', action='store_true', help='remove --out first')
    ap.add_argument('--strict-holdout', action='store_true', help='fail when a holdout id is in no batch')
    args = ap.parse_args()

    set_dirs = [Path(s) for s in args.sets]
    for sd in set_dirs:
        if not sd.is_dir() or not any((sd / sp).is_dir() for sp in SRC_SPLITS):
            print(f'{R}not a training set dir (needs train/ or val/): {sd}{NC}'); sys.exit(1)
    if len({sd.resolve() for sd in set_dirs}) != len(set_dirs):
        print(f'{R}the same set dir was given twice{NC}'); sys.exit(1)
    if len({sd.name for sd in set_dirs}) != len(set_dirs):
        print(f'{R}two sets share the same directory NAME (used as the batch tag): {[sd.name for sd in set_dirs]}{NC}')
        sys.exit(1)
    holdout_path = Path(args.holdout) if args.holdout else None
    if holdout_path is not None and not holdout_path.is_file():
        print(f'{R}holdout file not found: {holdout_path}{NC}'); sys.exit(1)
    lock_path = Path(args.train_lock) if args.train_lock else None
    if lock_path is not None and not lock_path.exists():
        print(f'{R}--train-lock file not found: {lock_path}{NC}'); sys.exit(1)
    if not 0.0 <= args.val_frac < 1.0:
        print(f'{R}--val-frac must be in [0, 1){NC}'); sys.exit(1)
    if not 0.0 <= args.tune_frac < 1.0:
        print(f'{R}--tune-frac must be in [0, 1){NC}'); sys.exit(1)

    out = Path(args.out)
    if out.exists():
        if args.fresh:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            print(f'{R}{out} exists and is not empty — use --fresh to rebuild it{NC}'); sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    holdout, holdout_sha = load_ids(holdout_path)
    lock_file_ids, lock_sha = load_train_lock(lock_path)
    batches: list[dict] = []
    all_files: list[dict] = []
    locked_from_sets: set[str] = set()
    for sd in set_dirs:
        files = scan_set(sd)
        if not files:
            print(f'{R}no PNGs under {sd}/{{train,val}}/<cls>/{NC}'); sys.exit(1)
        study_rows = read_studies_csv(sd)
        for f in files:
            f['batch'] = sd.name
        trained_on = is_release_set(sd)
        set_locked = train_locked_ids(files, study_rows, trained_on)
        locked_from_sets |= set_locked
        kind = 'release set' if trained_on else ('build set' if study_rows else 'legacy layout, no studies.csv')
        batches.append({'name': sd.name, 'path': str(sd), 'files': files, 'studies': study_rows,
                        'kind': kind, 'n_train_locked': len(set_locked)})
        all_files.extend(files)
        print(f'  {sd.name}: {len(files)} slices, {len({f["patient_id"] for f in files})} patients, '
              f'classes {sorted({f["cls"] for f in files})}, train-locked {len(set_locked)} ({kind})')

    # --- train-lock vs holdout: a contradiction, never silently resolved --------
    locked_all = locked_from_sets | lock_file_ids
    both = holdout & locked_all
    if both:
        print(f'{R}FAIL: {len(both)} patients are BOTH in the holdout file and train-locked (under train/ of a '
              f'legacy/release input set or in --train-lock): a previous model was trained on them, so they cannot '
              f'be held out. Remove them from the holdout file or from the lock list.\n'
              f'  ids: {_fmt_ids(both, 50)}{NC}')
        sys.exit(1)

    # --- one class per patient (last set wins; holdout: first set), within-set conflicts dropped -----
    winner, winner_set, cross_conflicts, within_conflicts = resolve_patient_classes(batches, first_wins=holdout)
    dropped_pids = {c['patient_id'] for c in within_conflicts if not c['resolved']}
    all_pids = set(winner)
    locked_present = locked_all & all_pids
    train_ids, val_ids, tune_ids = split_patients(all_pids, holdout, winner, args.val_frac, args.seed,
                                                  locked=locked_all, tune_frac=args.tune_frac,
                                                  min_tune_per_class=args.min_tune_per_class)
    split_of = {p: 'val' for p in val_ids}
    split_of.update({p: 'tune' for p in tune_ids})
    split_of.update({p: 'train' for p in train_ids})
    holdout_missing = sorted(holdout - all_pids)
    pid_batches: dict[str, set[str]] = defaultdict(set)
    for f in all_files:
        pid_batches[f['patient_id']].add(f['batch'])
    multi_batch = {p for p, bs in pid_batches.items() if len(bs) > 1}

    # --- write the tree ---------------------------------------------------
    seen_names: set[tuple[str, str, str]] = set()
    seen_hash: dict[str, set[str]] = defaultdict(set)      # pid -> sha256 of slices already placed
    n_dup = n_renamed = n_dropped_files = n_val_other_batch = n_conflicting_class = 0
    conflicting_class_pids: set[str] = set()
    out_counts: Counter = Counter()
    per_batch_out: dict[str, Counter] = defaultdict(Counter)
    for f in all_files:
        pid = f['patient_id']
        if pid in dropped_pids:
            n_dropped_files += 1; continue
        split, cls = split_of[pid], winner[pid]
        if pid in multi_batch and f['batch'] != winner_set[pid] and f['cls'] != cls:
            # another set's slices under a DIFFERENT class: possibly another study (follow-up scan) —
            # never relabel them into the deciding class (see docstring)
            n_conflicting_class += 1; conflicting_class_pids.add(pid); continue
        if split == 'val' and pid in multi_batch and f['batch'] != winner_set[pid]:
            n_val_other_batch += 1; continue       # gate set: one export per patient (see docstring)
        if pid in multi_batch:                     # only then can the same slice arrive twice
            digest = sha256_file(f['path'])
            if digest in seen_hash[pid]:
                n_dup += 1; continue               # identical slice already present from another batch
            seen_hash[pid].add(digest)
        name = f['path'].name
        key = (split, cls, name)
        if key in seen_names:
            name = f"{study_prefix_of(f['path'])}_{f['batch']}_{f['path'].stem.rsplit('_', 1)[1]}.png"
            key = (split, cls, name)
            n_renamed += 1
        seen_names.add(key)
        place(f['path'], out / split / cls / name, args.copy)
        out_counts[(split, cls)] += 1
        per_batch_out[f['batch']][(split, cls)] += 1
        f['split'] = split

    # --- verify from the WRITTEN tree (the thing the trainer will read) ----
    written: dict[str, set[str]] = {sp: set() for sp in SPLITS}
    written_cls: dict[str, set[str]] = defaultdict(set)
    written_prefixes: dict[str, set[str]] = {sp: set() for sp in SPLITS}
    for sp in SPLITS:
        if not (out / sp).is_dir():
            continue
        for cls_dir in (d for d in (out / sp).iterdir() if d.is_dir()):
            for fp in cls_dir.glob('*.png'):
                if not args.copy and not fp.resolve().is_file():
                    print(f'{R}dangling symlink: {fp}{NC}'); sys.exit(1)
                pid = patient_id_of(fp)
                written[sp].add(pid)
                written_cls[pid].add(cls_dir.name)
                written_prefixes[sp].add(study_prefix_of(fp))
    overlaps = {f'{a} and {b}': written[a] & written[b] for a, b in (('train', 'val'), ('train', 'tune'), ('val', 'tune'))}
    two_classes = {p for p, cs in written_cls.items() if len(cs) > 1}
    holdout_in_val = holdout & written['val']
    holdout_in_train = holdout & (written['train'] | written['tune'])
    locked_in_val = locked_all & (written['val'] | written['tune'])
    written_all = written['train'] | written['val'] | written['tune']

    for sp in SPLITS:
        if written[sp] or sp != 'tune':
            (out / f'{sp}_patients.txt').write_text('\n'.join(sorted(written[sp])) + '\n', encoding='utf-8')
    (out / 'train_lock_applied.txt').write_text('\n'.join(sorted(locked_present)) + '\n', encoding='utf-8')

    # studies.csv: batches' rows + batch + final split + winning class
    study_rows: list[dict] = []
    for b in batches:
        for r in b['studies']:
            r = dict(r); r['batch'] = b['name']
            pid = (r.get('patient_id') or '').strip()
            r['cls_in_batch'] = r.get('cls', '')
            if pid in dropped_pids:
                r['split'] = 'dropped'
            else:
                r['split'] = split_of.get(pid, '')
                r['cls'] = winner.get(pid, r.get('cls', ''))
            study_rows.append(r)
    if study_rows:
        cols = list(dict.fromkeys(k for r in study_rows for k in r))
        with open(out / 'studies.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(study_rows)

    classes = sorted({f['cls'] for f in all_files})
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seed': args.seed, 'val_frac_new_patients': args.val_frac, 'tune_frac_new_train_patients': args.tune_frac,
        'mode': 'copy' if args.copy else 'symlink',
        'classes': classes,
        'set_order': [b['name'] for b in batches],
        'sets': [{'name': b['name'], 'path': b['path'], 'n_files': len(b['files']),
                  'n_patients': len({f['patient_id'] for f in b['files']}),
                  'kind': b['kind'], 'n_train_locked': b['n_train_locked'], 'has_studies_csv': bool(b['studies']),
                  'classes': dict(Counter(f['cls'] for f in b['files'])),
                  'out': {sp: {c: per_batch_out[b['name']].get((sp, c), 0) for c in classes} for sp in SPLITS}}
                 for b in batches],
        'splits': {sp: {'n_files': sum(v for (s, _), v in out_counts.items() if s == sp),
                        'n_patients': len(written[sp]),
                        'classes': {c: out_counts.get((sp, c), 0) for c in classes},
                        'n_study_keys': len(written_prefixes[sp]),   # what eval_study_level.py counts
                        'patients_sha256': sha256_ids(written[sp])} for sp in SPLITS},
        'n_files': sum(out_counts.values()),
        'n_patients': len(written_all),                   # UNIQUE patients in the written tree
        'n_studies': len(written_all),                    # one study per patient after cross-set dedupe
        'n_patients_in_several_batches': len(multi_batch),
        'duplicate_slices_skipped': n_dup, 'slices_renamed_on_collision': n_renamed,
        'val_slices_from_non_winning_batch_skipped': n_val_other_batch,
        'conflicting_class_slices_dropped': {'n_files': n_conflicting_class,
                                             'patient_ids': sorted(conflicting_class_pids),
                                             'rule': 'slices a non-deciding set holds under another class are '
                                                     'dropped, never relabelled'},
        'holdout_decided_by_first_set': True,
        'min_tune_per_class': args.min_tune_per_class,
        'label_conflicts': {'rule': 'the last set on the command line holding the patient wins',
                            'n': len(cross_conflicts),
                            'patient_ids': [c['patient_id'] for c in cross_conflicts], 'details': cross_conflicts},
        'patients_dropped_within_set_conflict': {'n': len(dropped_pids),
                                                 'patient_ids': sorted(dropped_pids), 'details': within_conflicts,
                                                 'n_files_dropped': n_dropped_files},
        'train_lock': {'file': str(lock_path) if lock_path else None, 'sha256': lock_sha or None,
                       'n_from_sets': len(locked_from_sets), 'n_from_file': len(lock_file_ids),
                       'n_locked': len(locked_all), 'n_applied': len(locked_present),
                       'applied_file': 'train_lock_applied.txt', 'applied_sha256': sha256_ids(locked_present)},
        'holdout': {'file': str(holdout_path) if holdout_path else None, 'sha256': holdout_sha or None,
                    'n_ids': len(holdout), 'n_in_val': len(holdout_in_val), 'missing_from_all_batches': holdout_missing},
    }
    (out / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print('\n' + '=' * 64)
    print(f'  {B}COMBINED RELEASE TRAINING SET{NC}   ({manifest["mode"]}, seed {args.seed})')
    print('=' * 64)
    for b in manifest['sets']:
        print(f"  {b['name']:20s} {b['n_files']:>6} slices  {b['n_patients']:>5} patients  {b['classes']}  "
              f"train-locked {b['n_train_locked']}")
    print(f"  patients: {len(written_all)}  (in several batches: {len(multi_batch)})   "
          f"train {len(written['train'])}  val {len(written['val'])}  tune {len(written['tune'])}  "
          f"(holdout in val: {len(holdout_in_val)}/{len(holdout)}; train-locked: {len(locked_present)})")
    print(f'  {B}slices per class:{NC}')
    for c in classes:
        print(f"    {c:14s} train {out_counts.get(('train', c), 0):>5}  val {out_counts.get(('val', c), 0):>5}  "
              f"tune {out_counts.get(('tune', c), 0):>5}")
    if n_dup or n_renamed or n_val_other_batch:
        print(f'  {Y}identical duplicate slices skipped: {n_dup}; name collisions renamed: {n_renamed}; '
              f'val slices of a non-winning batch skipped: {n_val_other_batch}{NC}')
    if cross_conflicts:
        print(f'  {Y}label conflicts between sets: {len(cross_conflicts)} patients (last set wins; a holdout '
              f'patient keeps its first set): {_fmt_ids(c["patient_id"] for c in cross_conflicts)}{NC}')
    if n_conflicting_class:
        print(f'  {Y}{n_conflicting_class} slices of {len(conflicting_class_pids)} patients held under ANOTHER class '
              f'by a non-deciding set were DROPPED (not relabelled): {_fmt_ids(conflicting_class_pids)}{NC}')
    if within_conflicts:
        if dropped_pids:
            print(f'  {Y}WARNING: {len(dropped_pids)} patients held under TWO classes by the deciding set were DROPPED '
                  f'({n_dropped_files} slices): {_fmt_ids(dropped_pids)}{NC}')
        superseded = sorted({c['patient_id'] for c in within_conflicts if c['resolved']})
        if superseded:
            print(f'  {Y}{len(superseded)} patients held under two classes by an older set, decided by a later set: '
                  f'{_fmt_ids(superseded)}{NC}')
    if holdout_missing:
        print(f'  {Y}WARNING: {len(holdout_missing)} holdout ids are in NO batch (cannot be in val): '
              f'{_fmt_ids(holdout_missing)}{NC}')
    tune_per_class = {c: len({patient_id_of(fp) for fp in (out / 'tune' / c).glob('*.png')})
                      for c in classes} if (out / 'tune').is_dir() else {}
    small = {c: n for c, n in tune_per_class.items() if n < args.min_tune_per_class}
    if written['tune'] and small:
        print(f'  {Y}WARNING: tune/ is small ({", ".join(f"{c} {n}" for c, n in sorted(tune_per_class.items()))} '
              f'patients; minimum {args.min_tune_per_class} per class): the trainer picks its epoch on very few '
              f'studies — add new patients or accept a noisy checkpoint choice{NC}')
    if not written['tune']:
        print(f'  {Y}no tune/ split written (tune-frac {args.tune_frac}, too few new train patients) — the trainer '
              f'will fall back to selecting on val (the gate set){NC}')
    print(f'  manifest: {out / "dataset_manifest.json"}')

    failed = False
    for pair, ov in overlaps.items():
        if ov:
            print(f'{R}FAIL: {len(ov)} patients in BOTH {pair}: {_fmt_ids(ov)}{NC}'); failed = True
    if two_classes:
        print(f'{R}FAIL: {len(two_classes)} patients written under TWO classes: {_fmt_ids(two_classes)}{NC}'); failed = True
    if holdout_in_train:
        print(f'{R}FAIL: {len(holdout_in_train)} holdout patients ended up in train/tune: {_fmt_ids(holdout_in_train)}{NC}')
        failed = True
    if locked_in_val:
        print(f'{R}FAIL: {len(locked_in_val)} train-locked patients ended up in val/tune: {_fmt_ids(locked_in_val)}{NC}')
        failed = True
    if holdout & dropped_pids:
        print(f'{R}FAIL: {len(holdout & dropped_pids)} holdout patients were dropped for a within-set class conflict '
              f'(the gate set must keep them): {_fmt_ids(holdout & dropped_pids)}{NC}'); failed = True
    if args.strict_holdout and holdout_missing:
        print(f'{R}FAIL (--strict-holdout): holdout ids missing from every batch{NC}'); failed = True
    if not written['train'] or not written['val']:
        print(f'{R}FAIL: empty split (train {len(written["train"])} / val {len(written["val"])} patients){NC}'); failed = True
    if failed:
        sys.exit(1)
    print(f'\n  {G}✓ no train/val/tune patient overlap, one class per patient{NC}  ->  '
          f'training/finetune_brain_classifier.py --data {out} --classes "{",".join(classes)}"')


if __name__ == '__main__':
    main()
