#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — DICOM INGESTION ROBUSTNESS TEST
================================================================================

  Walks a folder of real DICOM studies and proves how much of it Sentinel can
  actually decode + load into volumes. Surfaces the format issues that bite in
  the field: compressed transfer syntaxes, multi-frame objects, MONOCHROME1,
  signed pixels, odd bit depths, RGB, missing pixels.

  Reports per-series success/failure with the *reason*, grouped by transfer
  syntax — so you know exactly which scanners/formats (if any) still need work.

  Usage:
    python scripts/dicom_robustness_test.py --src data/hospital_anon/studies
    python scripts/dicom_robustness_test.py --src /path/to/any/dicom --limit 200
================================================================================
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pydicom  # noqa: E402
from pydicom.uid import UID  # noqa: E402
from src.pipeline.brain_mri_preprocessor import (  # noqa: E402
    group_dicom_files_into_series, load_volume_from_series,
)

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'


def ts_name(uid: str) -> str:
    try:
        return UID(uid).name
    except Exception:
        return uid or '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(REPO / 'data' / 'hospital_anon' / 'studies'))
    ap.add_argument('--limit', type=int, default=0, help='cap study folders (0=all)')
    args = ap.parse_args()

    src = Path(args.src)
    # study = src/<patient>/<studyfolder>  OR just leaf dirs of files
    study_dirs = []
    for patient in sorted(src.iterdir()):
        if patient.is_dir():
            subs = [d for d in patient.iterdir() if d.is_dir()]
            study_dirs.extend(subs if subs else [patient])
    if args.limit:
        study_dirs = study_dirs[:args.limit]

    print(f'Scanning {len(study_dirs)} study folders under {src}\n')

    series_total = series_ok = series_fail = non_image = 0
    by_ts = defaultdict(lambda: [0, 0])         # ts -> [ok, fail]  (image series only)
    multiframe = singleframe = 0
    photometrics = Counter()
    bits = Counter()
    fail_reasons = Counter()
    studies_with_zero = []

    def series_has_pixels(se) -> bool:
        """Does ANY file in this series actually carry pixel data? If not, it's a
        non-image object (DICOMDIR/SR/ExamCard/presentation state) — correctly
        skipped, not a load failure."""
        for f in se.files[:8]:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if 'PixelData' in ds or ds.get('Rows'):
                    return True
            except Exception:
                continue
        return False

    for sd in study_dirs:
        files = [f for f in sd.rglob('*') if f.is_file()]
        series_list = group_dicom_files_into_series(files)
        study_loaded = 0
        study_image_series = 0
        for se in series_list:
            is_image = series_has_pixels(se)
            if not is_image:
                non_image += 1
                continue
            study_image_series += 1
            series_total += 1
            ts = '?'
            try:
                ds0 = pydicom.dcmread(str(se.files[0]), stop_before_pixels=True, force=True)
                ts = str(getattr(getattr(ds0, 'file_meta', None), 'TransferSyntaxUID', '?'))
                photometrics[str(ds0.get('PhotometricInterpretation', '?'))] += 1
                bits[int(ds0.get('BitsStored', 0) or 0)] += 1
                nf = int(ds0.get('NumberOfFrames', 1) or 1)
                if nf > 1:
                    multiframe += 1
                else:
                    singleframe += 1
            except Exception:
                pass
            try:
                vol = load_volume_from_series(se)
                if vol is not None and vol.size > 0 and vol.ndim == 3:
                    series_ok += 1
                    by_ts[ts][0] += 1
                    study_loaded += 1
                else:
                    series_fail += 1
                    by_ts[ts][1] += 1
                    fail_reasons['has pixels but volume empty'] += 1
            except Exception as e:
                series_fail += 1
                by_ts[ts][1] += 1
                fail_reasons[f'{type(e).__name__}'] += 1
        # Only a concern if the study HAD image series but none loaded.
        if study_image_series > 0 and study_loaded == 0:
            studies_with_zero.append(f'{sd.parent.name}/{sd.name}')

    pct = (series_ok / series_total * 100) if series_total else 0
    print('=' * 74)
    print(f'  {B}DICOM INGESTION ROBUSTNESS{NC}')
    print('=' * 74)
    color = G if pct >= 99 else (Y if pct >= 90 else R)
    print(f'  Image series loaded OK:  {color}{series_ok}/{series_total}  ({pct:.1f}%){NC}')
    print(f'  Image series failed:     {series_fail}')
    print(f'  Non-image objects skipped (DICOMDIR/SR/ExamCard): {non_image}')
    print(f'  Multi-frame series: {multiframe}   single-frame: {singleframe}')
    print(f'\n  {B}By transfer syntax (ok/fail):{NC}')
    for ts, (ok, fail) in sorted(by_ts.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        c = G if fail == 0 else R
        print(f'    {c}{ok:>4} ok / {fail:<4} fail{NC}  {ts_name(ts)}')
    print(f'\n  {B}Photometric:{NC} {dict(photometrics)}')
    print(f'  {B}BitsStored:{NC}  {dict(bits)}')
    if fail_reasons:
        print(f'\n  {R}Failure reasons:{NC}')
        for r, c in fail_reasons.most_common():
            print(f'    {c:>4}  {r}')
    if studies_with_zero:
        print(f'\n  {Y}Studies that loaded ZERO series ({len(studies_with_zero)}):{NC}')
        for s in studies_with_zero[:15]:
            print(f'    {s}')
    if pct >= 99 and not studies_with_zero:
        print(f'\n  {G}✓ Ingestion robust across this dataset.{NC}')
    else:
        print(f'\n  ⚠ Review the failures above before claiming full format coverage.')


if __name__ == '__main__':
    main()
