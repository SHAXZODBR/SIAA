#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PREP RSNA INTRACRANIAL HEMORRHAGE → train/val class folders
================================================================================
  RSNA dataset = DICOM slices + stage_2_train.csv (6 rows/image: the 5 subtypes
  + 'any'). This turns it into the class-folder layout our trainer expects:

    data/ft/rsna_hemorrhage/{train,val}/{hemorrhage,no_hemorrhage}/*.png

  Applies a brain window (WL=40, WW=80) to each CT slice and saves PNG.
  Patient-level split (by StudyInstanceUID) to avoid leakage. Use --subtype to
  build a 6-class set instead of binary.

  Run AFTER downloading (see training/TRAINING_GUIDE.md):
    python training/prep_rsna_hemorrhage.py \
        --src /path/to/rsna-intracranial-hemorrhage-detection \
        --limit 60000          # cap for a manageable first model (RSNA is huge)
================================================================================
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    import pydicom
    from PIL import Image
except ImportError:
    print("need pydicom + pillow: pip install pydicom pillow"); sys.exit(1)

SUBTYPES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


def brain_window(ds, wl=40, ww=80) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
    inter = float(getattr(ds, 'RescaleIntercept', 0) or 0)
    hu = arr * slope + inter
    lo, hi = wl - ww / 2, wl + ww / 2
    out = np.clip((hu - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='RSNA download root (has stage_2_train.csv + stage_2_train/)')
    ap.add_argument('--out', default='data/ft/rsna_hemorrhage')
    ap.add_argument('--subtype', action='store_true', help='6-class subtypes instead of binary')
    ap.add_argument('--limit', type=int, default=0, help='cap images (RSNA has ~750k)')
    args = ap.parse_args()

    src = Path(args.src)
    csv_path = next(src.rglob('stage_2_train.csv'), None) or next(src.rglob('*train*.csv'), None)
    img_dir = next((p for p in src.rglob('stage_2_train') if p.is_dir()), None) or \
              next((p for p in src.rglob('*train*') if p.is_dir() and any(p.glob('*.dcm'))), None)
    if not csv_path or not img_dir:
        print(f"Couldn't find CSV/images under {src}"); sys.exit(1)
    print(f"CSV: {csv_path}\nImages: {img_dir}")

    # pivot CSV: image_id -> {subtype: 0/1}
    labels = defaultdict(dict)
    for row in csv.DictReader(open(csv_path)):
        # ID format: ID_<imageid>_<subtype>
        parts = row['ID'].rsplit('_', 1)
        img_id, sub = parts[0], parts[1]
        labels[img_id][sub] = int(row['Label'])
    print(f"images labeled: {len(labels)}")

    out = Path(args.out)
    import shutil; shutil.rmtree(out, ignore_errors=True)
    counts = defaultdict(int)
    processed = 0
    for i, (img_id, subs) in enumerate(labels.items()):
        if args.limit and processed >= args.limit:
            break
        dcm = img_dir / f"{img_id}.dcm"
        if not dcm.exists():
            continue
        try:
            ds = pydicom.dcmread(str(dcm), force=True)
            png = brain_window(ds)
            # patient-level split via StudyInstanceUID hash
            study = str(getattr(ds, 'StudyInstanceUID', img_id))
            split = 'val' if (hash(study) % 5 == 0) else 'train'
            if args.subtype:
                cls = next((s for s in SUBTYPES if subs.get(s)), None) or 'no_hemorrhage'
            else:
                cls = 'hemorrhage' if subs.get('any') else 'no_hemorrhage'
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(png).convert('L').save(d / f"{img_id}.png")
            counts[(split, cls)] += 1
            processed += 1
        except Exception:
            continue
        if (i + 1) % 5000 == 0:
            print(f"  …{processed} images written")

    print("\nRSNA hemorrhage dataset built:")
    for k, c in sorted(counts.items()):
        print(f"  {k[0]}/{k[1]}: {c}")
    print(f"\n✓ Train with:\n  python training/finetune_brain_classifier.py --data {out} "
          f"--classes \"{'hemorrhage,no_hemorrhage' if not args.subtype else ','.join(SUBTYPES)+',no_hemorrhage'}\" "
          f"--base-model google/vit-base-patch16-224 --balance --device cuda --epochs 8 --output models/head_ct_finetuned")


if __name__ == '__main__':
    main()
