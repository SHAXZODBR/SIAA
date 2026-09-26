#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BUILD A LOCAL TRAINING SET FROM MATCHED STUDIES + REPORT LABELS
================================================================================
  Release pipeline stage 4 (docs/dev/release_pipeline_contract.md).

  Turns matched DICOM brain studies (match_images_to_reports.py output) into
  labelled 2D slices organised into class folders, SPLIT BY PATIENT (no
  leakage), ready for training/finetune_brain_classifier.py and for
  scripts/combine_training_sets.py (which re-splits across batches).

  For each brain study it extracts a few central axial slices from the best
  sequence — central_slices() is the SAME rule production inference uses
  (src/inference/brain_analysis.py::_triage_mean_probs) — and files them as
  <out>/<split>/<cls>/<patient_id>_<k>.png, plus <out>/studies.csv
  (patient_id, study_folder, cls, split, n_slices, label, label_source).

  Binary class: with --abnormal-col the class comes from that column
  (yes -> abnormal, no -> normal; empty -> fall back to label); without it,
  label == 'normal' -> normal, anything else -> abnormal.

  Resumable: a study whose PNGs already exist in <out> is not re-processed and
  keeps its SPLIT; its CLASS always comes from the current matched.csv. When a
  label changed since the last run (LLM import, rules fix, needs_review
  resolved) the files are MOVED to the new class folder of the same split and
  counted as 'relabelled N'. Which files belong to which study is read from
  the previous studies.csv (written before any slice of a run is built, so an
  interrupted run stays consistent), so a patient whose study list changed
  (a study added/removed shifts '<pid>' / '<pid>_s2' numbering) gets its own
  images back. PNGs of studies that are no longer selected (label now
  unusable, needs_review, not brain) are REMOVED on a full run (no --limit /
  --max-per-class; with those they are only warned about). studies.csv is
  rewritten with the current classes. --fresh wipes <out>.

  NOTE (honest): report labels are STUDY-level; we apply them to slices, which
  is weak supervision (not every slice shows the finding). Good enough to teach
  the local scanner distribution; not a substitute for per-slice annotation.

  Usage:
    python scripts/build_training_set.py --images "<folder>" \
        --matched data/<batch>_matched.csv --out data/sets/<batch> \
        --mode binary --abnormal-col abnormal --skip-needs-review \
        [--index data/<batch>_index.csv] [--max-per-class 0] [--limit 0] [--fresh]
================================================================================
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.brain_mri_preprocessor import (  # noqa: E402
    process_brain_study, central_band_best_slice, robust_normalize_slice,
    CLASSIFIER_SEQUENCE_PREFERENCE, triage_sequence, central_slice_indices)

G, Y, B, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0;31m', '\033[0m'
# keep only findings with enough matched studies to learn (drop tiny classes
# like hemorrhage=6, hydrocephalus=13 that would just add noise this round)
KEEP = {'tumor', 'stroke', 'atrophy', 'ms_wml', 'cyst', 'normal'}
STUDIES_COLUMNS = ('patient_id', 'study_folder', 'cls', 'split', 'n_slices', 'label', 'label_source')
_PNG_RE = re.compile(r'^(?P<prefix>.+)_(?P<k>\d+)$')
STAGING_DIR = '.relabel_staging'      # files in flight while a relabel/rename is applied


def central_slices(study, k=5):
    """Up to k axial central slices of the triage series — the SAME series / slice rule
    production uses (brain_mri_preprocessor.triage_sequence + central_slice_indices, shared
    with brain_analysis._triage_mean_probs). [] when the study has no usable axial series
    (sagittal/coronal-only, or only degenerate e.g. colour-strip volumes)."""
    seq = triage_sequence(study)
    if seq is None:
        return []
    vol = study.sequences[seq]
    if vol.ndim != 3:
        return [robust_normalize_slice(vol)]
    return [robust_normalize_slice(vol[i]) for i in central_slice_indices(vol.shape[0], k)]


def binary_class(label: str, abnormal: str) -> Optional[str]:
    """'abnormal' | 'normal' | None (unlabelled). abnormal column wins when filled."""
    a = (abnormal or '').strip().lower()
    if a in ('yes', 'y', '1', 'true'):
        return 'abnormal'
    if a in ('no', 'n', '0', 'false'):
        return 'normal'
    lbl = (label or '').strip()
    if not lbl or lbl == 'unspecified':
        return None
    return 'normal' if lbl == 'normal' else 'abnormal'


def multiclass_class(label: str) -> Optional[str]:
    lbl = (label or '').strip()
    if not lbl or lbl == 'unspecified':
        return None
    if lbl == 'normal' or lbl in KEEP:
        return lbl
    return None  # drop 'other'/tiny classes for clean multiclass


def select_studies(rows: list[dict], mode: str, abnormal_col: Optional[str], skip_needs_review: bool,
                   index_folders: Optional[dict[str, str]]) -> tuple[list[dict], Counter]:
    """Filter matched rows down to trainable brain studies. Returns (studies, drop reasons)."""
    dropped: Counter = Counter()
    studies: list[dict] = []
    for r in rows:
        if r.get('is_brain') != 'yes':
            dropped['not brain'] += 1; continue
        if r.get('report_matched') != 'yes':
            dropped['no report'] += 1; continue
        if skip_needs_review and (r.get('needs_review') or '').strip().lower() == 'yes':
            dropped['needs_review'] += 1; continue
        if index_folders is not None:
            err = index_folders.get(r['study_folder'])
            if err is None:
                dropped['not in index'] += 1; continue
            if err:
                dropped['index error'] += 1; continue
        if mode == 'binary':
            cls = binary_class(r.get('label') or '', (r.get(abnormal_col) or '') if abnormal_col else '')
        else:
            cls = multiclass_class(r.get('label') or '')
        if cls is None:
            dropped['unlabelled/dropped class'] += 1; continue
        studies.append({'patient_id': (r.get('patient_id') or '').strip(), 'study_folder': r['study_folder'],
                        'cls': cls, 'label': (r.get('label') or '').strip(),
                        'label_source': (r.get('label_source') or '').strip()})
    studies.sort(key=lambda s: (s['patient_id'], s['study_folder']))
    return studies, dropped


def assign_prefixes(studies: list[dict]) -> None:
    """First study of a patient -> '<pid>'; further studies -> '<pid>_s2', '<pid>_s3' …
    so two studies of one patient never overwrite each other's PNGs."""
    seen: Counter = Counter()
    for s in studies:
        seen[s['patient_id']] += 1
        n = seen[s['patient_id']]
        s['prefix'] = s['patient_id'] if n == 1 else f"{s['patient_id']}_s{n}"


def scan_existing(out: Path) -> dict[str, list[tuple[str, str, int]]]:
    """prefix -> [(split, cls, n_png), …] for PNGs already under <out>/<split>/<cls>/.
    More than one location for a prefix means a damaged tree (the caller rebuilds it)."""
    counts: Counter = Counter()
    for split_dir in (d for d in out.iterdir() if d.is_dir() and not d.name.startswith('.')) if out.is_dir() else []:
        for cls_dir in (d for d in split_dir.iterdir() if d.is_dir()):
            for fp in cls_dir.glob('*.png'):
                m = _PNG_RE.match(fp.stem)
                if m:
                    counts[(m['prefix'], split_dir.name, cls_dir.name)] += 1
    found: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for (prefix, split, cls), n in sorted(counts.items()):
        found[prefix].append((split, cls, n))
    return dict(found)


def prefix_files(out: Path, split: str, cls: str, prefix: str) -> list[Path]:
    """The '<prefix>_<k>.png' files of ONE study ('123_*' must not pick up '123_s2_*')."""
    d = out / split / cls
    return sorted((fp for fp in d.glob(f'{prefix}_*.png')
                   if (m := _PNG_RE.match(fp.stem)) and m['prefix'] == prefix),
                  key=lambda fp: int(_PNG_RE.match(fp.stem)['k']))


def read_prefix_map(out: Path) -> Optional[dict[str, str]]:
    """study_folder -> prefix of the files the PREVIOUS run wrote, reconstructed from
    <out>/studies.csv with the same numbering rule (assign_prefixes over (patient_id,
    study_folder) order). None when there is no usable studies.csv (a set built by an
    older version): the caller then trusts the prefix itself."""
    sc = out / 'studies.csv'
    if not sc.is_file():
        return None
    with open(sc, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or 'study_folder' not in rows[0] or 'patient_id' not in rows[0]:
        return None
    old = [{'patient_id': (r.get('patient_id') or '').strip(), 'study_folder': r.get('study_folder') or ''} for r in rows]
    if len({o['study_folder'] for o in old}) != len(old):
        return None                                   # ambiguous: fall back to prefix identity
    old.sort(key=lambda o: (o['patient_id'], o['study_folder']))
    assign_prefixes(old)
    return {o['study_folder']: o['prefix'] for o in old}


def reconcile_existing(out: Path, studies: list[dict], prune: bool) -> dict:
    """Align the PNGs already under <out> with THIS run's plan (studies carry 'prefix' + 'cls').

    A study keeps the split its files are in; when its class (from matched.csv) differs
    from the folder the files sit in, or its prefix shifted, the files are MOVED to
    <split>/<new cls>/<new prefix>_<k>.png. Moves go through a staging dir so renames
    never overwrite each other. Files of prefixes no study claims are stale: removed when
    `prune` (a full run) or when their name is needed by a current study, else reported.

    Returns {'placed': {prefix: (split, n)}, 'relabelled': [(prefix, old_cls, new_cls)],
             'renamed': int, 'stale_removed': [prefix], 'stale_kept': [prefix]}.
    """
    found = scan_existing(out)
    prefix_map = read_prefix_map(out)
    damaged = {p for p, locs in found.items() if len(locs) > 1}
    moves: list[tuple[Path, Path]] = []
    placed: dict[str, tuple[str, int]] = {}
    claimed: set[str] = set()
    relabelled: list[tuple[str, str, str]] = []
    renamed = 0
    for s in studies:
        src = prefix_map.get(s['study_folder']) if prefix_map is not None else s['prefix']
        if src is None or src not in found or src in damaged:
            continue
        (split, old_cls, n), = found[src]
        claimed.add(src)
        if old_cls != s['cls'] or src != s['prefix']:
            files = prefix_files(out, split, old_cls, src)
            for j, fp in enumerate(files):
                moves.append((fp, out / split / s['cls'] / f"{s['prefix']}_{_PNG_RE.match(fp.stem)['k']}.png"))
            n = len(files)
            if old_cls != s['cls']:
                relabelled.append((s['prefix'], old_cls, s['cls']))
            if src != s['prefix']:
                renamed += 1
        placed[s['prefix']] = (split, n)
    current = {s['prefix'] for s in studies}
    stale = sorted(p for p in found if p not in claimed)
    remove = [p for p in stale if prune or p in current or p in damaged]
    kept = [p for p in stale if p not in remove]

    staging = out / STAGING_DIR
    if moves:
        staging.mkdir(parents=True, exist_ok=False)
        for i, (src_fp, _) in enumerate(moves):
            os.replace(src_fp, staging / f'{i}.png')
    for p in remove:
        for split, cls, _ in found[p]:
            for fp in prefix_files(out, split, cls, p):
                fp.unlink()
    if moves:
        for i, (_, dst) in enumerate(moves):
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / f'{i}.png', dst)
        staging.rmdir()
    return {'placed': placed, 'relabelled': relabelled, 'renamed': renamed,
            'stale_removed': [p for p in remove], 'stale_kept': kept}


def write_studies(out: Path, records: list[dict]) -> None:
    tmp = out / 'studies.csv.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(STUDIES_COLUMNS))
        w.writeheader(); w.writerows(records)
    os.replace(tmp, out / 'studies.csv')


def patient_split(pids: list[str], val_frac: float) -> set[str]:
    """Deterministic stride split over the sorted patient list (unchanged rule)."""
    pats = sorted(set(pids))
    n_val = int(len(pats) * val_frac)
    return set(pats[::max(1, len(pats) // max(1, n_val))][:n_val]) if n_val else set()


def write_slices(slices: list[np.ndarray], dest: Path, prefix: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for j, slc in enumerate(slices):
        Image.fromarray((np.clip(slc, 0, 1) * 255).astype(np.uint8)).convert('RGB').save(dest / f'{prefix}_{j}.png')
    return len(slices)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a per-patient-split slice training set from matched studies.')
    ap.add_argument('--images', required=True, help='root folder; studies live at <images>/<study_folder>')
    ap.add_argument('--matched', default='data/matched_dataset.csv')
    ap.add_argument('--out', default='data/local_finetune')
    ap.add_argument('--mode', choices=['multiclass', 'binary'], default='multiclass')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--abnormal-col', default=None,
                    help='binary class from this column (yes/no) when non-empty, else from label')
    ap.add_argument('--skip-needs-review', action='store_true', help='drop rows with needs_review=yes')
    ap.add_argument('--index', default=None,
                    help='index CSV from index_new_studies.py: only studies it recorded (without error) are used')
    ap.add_argument('--max-per-class', type=int, default=0, help='cap studies per class (0 = no cap)')
    ap.add_argument('--limit', type=int, default=0, help='process at most N studies (smoke runs; 0 = all)')
    ap.add_argument('--fresh', action='store_true', help='remove --out first (otherwise resumable)')
    args = ap.parse_args()

    img_root = Path(args.images)
    if not img_root.is_dir():
        print(f'{R}image root not found: {img_root}{NC}'); sys.exit(1)
    matched = Path(args.matched)
    if not matched.is_file():
        print(f'{R}matched CSV not found: {matched}{NC}'); sys.exit(1)
    with open(matched, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = reader.fieldnames or []
    if args.abnormal_col and args.abnormal_col not in columns:
        print(f'{Y}--abnormal-col {args.abnormal_col!r} is not a column of {matched} — falling back to label{NC}')
        args.abnormal_col = None

    index_folders: Optional[dict[str, str]] = None
    if args.index:
        with open(args.index, encoding='utf-8') as fh:
            index_folders = {r['study_folder']: (r.get('error') or '').strip() for r in csv.DictReader(fh)}
        print(f'index: {len(index_folders)} study folders from {args.index}')

    studies, dropped = select_studies(rows, args.mode, args.abnormal_col, args.skip_needs_review, index_folders)
    if args.max_per_class > 0:
        per_cls: Counter = Counter(); capped = []
        for s in studies:
            per_cls[s['cls']] += 1
            if per_cls[s['cls']] <= args.max_per_class:
                capped.append(s)
        dropped['max-per-class cap'] += len(studies) - len(capped)
        studies = capped
    if args.limit > 0 and len(studies) > args.limit:
        dropped['--limit'] += len(studies) - args.limit
        studies = studies[:args.limit]
    assign_prefixes(studies)
    if not studies:
        print(f'{R}no trainable brain studies after filtering: {dict(dropped)}{NC}'); sys.exit(1)

    out = Path(args.out)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / STAGING_DIR).exists():
        print(f'{R}{out / STAGING_DIR} exists: an earlier relabel of this set was interrupted mid-move — '
              f'rebuild with --fresh{NC}'); sys.exit(1)
    full_run = args.limit == 0 and args.max_per_class == 0
    rec_state = reconcile_existing(out, studies, prune=full_run)
    placed = rec_state['placed']
    # patients that already have files keep the split their files are in
    fixed_split = {s['patient_id']: placed[s['prefix']][0] for s in studies if s['prefix'] in placed}
    val_pats = patient_split([s['patient_id'] for s in studies], args.val_frac)
    val_pats = {p for p in val_pats if fixed_split.get(p, 'val') == 'val'} | {p for p, sp in fixed_split.items() if sp == 'val'}

    records: list[dict] = []
    for s in studies:
        split = placed[s['prefix']][0] if s['prefix'] in placed else ('val' if s['patient_id'] in val_pats else 'train')
        records.append({'patient_id': s['patient_id'], 'study_folder': s['study_folder'], 'cls': s['cls'],
                        'split': split, 'n_slices': placed[s['prefix']][1] if s['prefix'] in placed else 0,
                        'label': s['label'], 'label_source': s['label_source']})
    # written BEFORE building: the next run maps files to studies with it, even if this one is interrupted
    write_studies(out, records)

    counts: Counter = Counter()
    n_reused = n_failed = n_built = 0
    for i, (s, rec) in enumerate(zip(studies, records), 1):
        split = rec['split']
        if s['prefix'] in placed:
            counts[(split, s['cls'])] += rec['n_slices']; n_reused += 1
            continue
        study_dir = img_root / s['study_folder']
        files = [f for f in study_dir.rglob('*') if f.suffix.lower() == '.dcm'] if study_dir.is_dir() else []
        try:
            study = process_brain_study(files) if files else None
            slices = central_slices(study) if study is not None else []
        except Exception as e:  # unreadable study: leave it out, it will be retried next run
            print(f'  {Y}skip {s["study_folder"][:40]}…: {type(e).__name__}: {str(e)[:60]}{NC}')
            slices = []
        if not slices:
            n_failed += 1
            continue
        n = write_slices(slices, out / split / s['cls'], s['prefix'])
        rec['n_slices'] = n
        counts[(split, s['cls'])] += n; n_built += 1
        if i % 50 == 0:
            print(f'  …{i}/{len(studies)}  built={n_built} reused={n_reused} failed={n_failed}', flush=True)

    write_studies(out, records)
    n_relabelled = len(rec_state['relabelled'])

    pats = sorted({s['patient_id'] for s in studies})
    n_val = len({s['patient_id'] for s in records if s['split'] == 'val'})
    print('\n' + '=' * 60)
    print(f"  {B}LOCAL TRAINING SET BUILT ({args.mode}){NC}")
    print('=' * 60)
    print(f"  studies: {len(studies)}  built {n_built}  reused {n_reused}  relabelled {n_relabelled}"
          f"  {'failed ' + str(n_failed) if n_failed else ''}")
    if n_relabelled:
        moved = Counter((o, n) for _, o, n in rec_state['relabelled'])
        print(f"  {Y}relabelled {n_relabelled} studies since the last build (files moved, same split): "
              + ', '.join(f'{o}->{n} {c}' for (o, n), c in sorted(moved.items())) + NC)
    if rec_state['renamed']:
        print(f"  {Y}renumbered {rec_state['renamed']} studies whose '<pid>_sN' prefix shifted{NC}")
    if rec_state['stale_removed']:
        print(f"  {Y}removed the PNGs of {len(rec_state['stale_removed'])} studies no longer selected "
              f"(label unusable / needs_review / not in matched.csv){NC}")
    if rec_state['stale_kept']:
        print(f"  {Y}WARNING: {len(rec_state['stale_kept'])} studies on disk are not in this (--limit/--max-per-class) "
              f"selection and were left in place — run without those flags or with --fresh to drop them{NC}")
    print(f"  patients: {len(pats)}  ({n_val} held out for val)")
    if dropped:
        print(f"  dropped rows: " + ', '.join(f'{k}={v}' for k, v in sorted(dropped.items())))
    print(f"  {B}slices per class:{NC}")
    classes = sorted(set(c for (_, c) in counts))
    for cls in classes:
        tr, va = counts.get(('train', cls), 0), counts.get(('val', cls), 0)
        print(f"    {cls:14s} train {tr:>4}  val {va:>4}")
    print(f"  studies.csv: {out / 'studies.csv'}")
    if n_failed and not n_built and not n_reused:
        print(f'{R}every study failed to produce slices{NC}'); sys.exit(1)
    print(f"\n  ✓ Ready for: python training/finetune_brain_classifier.py --data {out} \\")
    print(f"      --classes \"{','.join(classes)}\" --base-model google/vit-base-patch16-224 --balance --device mps")


if __name__ == '__main__':
    main()
