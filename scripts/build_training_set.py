#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BUILD A LOCAL TRAINING SET FROM MATCHED STUDIES + REPORT LABELS
================================================================================
  Turns matched DICOM brain studies (matched_dataset.csv) into labelled 2D
  slices organised into class folders, SPLIT BY PATIENT (no leakage), ready for
  finetune_brain_classifier.py.

  For each brain study it extracts a few central axial slices from the best
  sequence and files them under the report's diagnosis. Multi-finding and
  'unspecified' reports are skipped for clean labels (configurable).

  NOTE (honest): report labels are STUDY-level; we apply them to slices, which
  is weak supervision (not every slice shows the finding). Good enough to teach
  the local scanner distribution; not a substitute for per-slice annotation.

  Usage:
    python scripts/build_training_set.py --images "<folder>" \
        --matched data/matched_dataset.csv --out data/local_finetune \
        --mode multiclass          # or: binary  (normal vs abnormal)
================================================================================
"""
from __future__ import annotations
import argparse, csv, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.brain_mri_preprocessor import (
    process_brain_study, central_band_best_slice, robust_normalize_slice,
    CLASSIFIER_SEQUENCE_PREFERENCE)

G, Y, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0m'
# keep only findings with enough matched studies to learn (drop tiny classes
# like hemorrhage=6, hydrocephalus=13 that would just add noise this round)
KEEP = {'tumor', 'stroke', 'atrophy', 'ms_wml', 'cyst', 'normal'}


def central_slices(study, k=5):
    """Up to k axial slices from the best available sequence."""
    def axial(key): return (study.seq_meta.get(key, {}).get('orientation') or 'axial') == 'axial'
    seq = None
    for pref in CLASSIFIER_SEQUENCE_PREFERENCE:
        if pref in study.sequences and axial(pref): seq = pref; break
    if seq is None:
        seq = next((k for k in study.sequences if axial(k)), None)
    if seq is None: return []
    vol = study.sequences[seq]
    if vol.ndim != 3: return [robust_normalize_slice(vol)]
    d = vol.shape[0]; c = d // 2
    step = max(1, d // 12)
    offs = range(-(k // 2) * step, (k // 2 + 1) * step, step)
    idxs = sorted(set(max(0, min(d - 1, c + off)) for off in offs))[:k]
    return [robust_normalize_slice(vol[i]) for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--matched', default='data/matched_dataset.csv')
    ap.add_argument('--out', default='data/local_finetune')
    ap.add_argument('--mode', choices=['multiclass', 'binary'], default='multiclass')
    ap.add_argument('--val-frac', type=float, default=0.2)
    args = ap.parse_args()

    img_root = Path(args.images)
    rows = [r for r in csv.DictReader(open(args.matched, encoding='utf-8'))
            if r['is_brain'] == 'yes' and r['report_matched'] == 'yes']
    # clean single-label studies
    studies = []
    for r in rows:
        lbl = (r['label'] or '').strip()              # clean LLM labels are single-value
        if not lbl or lbl == 'unspecified':
            continue
        if args.mode == 'binary':
            cls = 'normal' if lbl == 'normal' else 'abnormal'   # keep ALL abnormal incl. 'other'
        else:
            if lbl == 'normal':
                cls = 'normal'
            elif lbl in KEEP:
                cls = lbl
            else:
                continue  # drop 'other'/tiny classes for clean multiclass
        studies.append((r['patient_id'], r['study_folder'], cls))

    # patient-level split (no leakage)
    by_pat = defaultdict(list)
    for pid, sf, cls in studies:
        by_pat[pid].append((sf, cls))
    pats = sorted(by_pat)
    n_val = int(len(pats) * args.val_frac)
    val_pats = set(pats[::max(1, len(pats)//max(1, n_val))][:n_val]) if n_val else set()

    out = Path(args.out)
    import shutil; shutil.rmtree(out, ignore_errors=True)
    counts = Counter()
    for pid in pats:
        split = 'val' if pid in val_pats else 'train'
        for sf, cls in by_pat[pid]:
            files = [f for f in (img_root / sf).rglob('*') if f.suffix.lower() == '.dcm']
            try:
                study = process_brain_study(files)
            except Exception:
                continue
            for j, slc in enumerate(central_slices(study)):
                d = out / split / cls; d.mkdir(parents=True, exist_ok=True)
                Image.fromarray((np.clip(slc, 0, 1)*255).astype(np.uint8)).convert('RGB').save(d / f"{pid}_{j}.png")
                counts[(split, cls)] += 1

    print('\n' + '='*60)
    print(f"  {B}LOCAL TRAINING SET BUILT ({args.mode}){NC}")
    print('='*60)
    print(f"  patients: {len(pats)}  ({len(val_pats)} held out for val)")
    print(f"  {B}slices per class:{NC}")
    classes = sorted(set(c for (_, c) in counts))
    for cls in classes:
        tr, va = counts.get(('train', cls), 0), counts.get(('val', cls), 0)
        print(f"    {cls:14s} train {tr:>4}  val {va:>4}")
    print(f"\n  ✓ Ready for: python training/finetune_brain_classifier.py --data {out} \\")
    print(f"      --classes \"{','.join(classes)}\" --base-model google/vit-base-patch16-224 --balance --device mps")


if __name__ == '__main__':
    main()
