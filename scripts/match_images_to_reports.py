#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — MATCH DICOM IMAGE STUDIES TO RADIOLOGIST REPORT LABELS
================================================================================
  Walks a folder of DICOM studies, reads ONE header per study for PatientID +
  modality + body part, and joins to the parsed report labels (by PatientID).
  Output = the labelled IMAGE dataset: which scan has which diagnosis.

  PHI: keeps only patient_id (the join key) + study path; drops names.

  Usage:
    python scripts/match_images_to_reports.py \
        --images "<image folder>" --labels data/hospital_reports_labels.csv \
        --out data/matched_dataset.csv
================================================================================
"""
from __future__ import annotations
import argparse, csv, os
from collections import Counter
from pathlib import Path
import pydicom

G, Y, B, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0;31m', '\033[0m'


def first_dcm(study_dir: Path):
    for root, _, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                return Path(root) / f
    return None


def is_brain(body_part: str, name: str, desc: str) -> bool:
    t = f"{body_part} {name} {desc}".upper()
    return any(k in t for k in ['BRAIN', 'HEAD', 'SKULL', 'CEREBR'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', default='data/hospital_reports_labels.csv')
    ap.add_argument('--out', default='data/matched_dataset.csv')
    args = ap.parse_args()

    # load labels: patient_id -> {body_part, label, ...}
    labels = {}
    with open(args.labels, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            labels[r['patient_id']] = r
    print(f'Loaded {len(labels)} report labels.')

    img_root = Path(args.images)
    study_dirs = [d for d in img_root.iterdir() if d.is_dir()]
    print(f'{len(study_dirs)} image study folders. Reading one header each…')

    rows = []
    matched = brain_matched = no_header = unmatched = 0
    bp_counter, find_counter = Counter(), Counter()
    for i, sd in enumerate(study_dirs):
        f = first_dcm(sd)
        if f is None:
            no_header += 1; continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            no_header += 1; continue
        pid = str(ds.get('PatientID', '')).strip()
        modality = str(ds.get('Modality', ''))
        bp = str(ds.get('BodyPartExamined', ''))
        name = str(ds.get('PatientName', ''))
        desc = str(ds.get('StudyDescription', ''))
        brainy = is_brain(bp, name, desc)
        rep = labels.get(pid)
        row = {'study_folder': sd.name, 'patient_id': pid, 'modality': modality,
               'body_part_dicom': bp or ('brain' if brainy else ''), 'is_brain': 'yes' if brainy else 'no',
               'report_matched': 'yes' if rep else 'no',
               'report_body_part': rep['body_part'] if rep else '',
               'label': rep['label'] if rep else ''}
        rows.append(row)
        if brainy:
            bp_counter['brain'] += 1
        else:
            bp_counter[bp or 'other'] += 1
        if rep:
            matched += 1
            if brainy and rep['label'] not in ('', 'unspecified'):
                brain_matched += 1
                for lbl in rep['label'].split(';'):
                    if lbl and lbl != 'normal':
                        find_counter[lbl] += 1
                    elif lbl == 'normal':
                        find_counter['normal'] += 1
        else:
            unmatched += 1
        if (i + 1) % 300 == 0:
            print(f'  …{i+1}/{len(study_dirs)}  matched={matched}')

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print('\n' + '=' * 70)
    print(f'  {B}MATCHED IMAGE ↔ REPORT DATASET{NC}')
    print('=' * 70)
    print(f'  Image studies:          {len(rows)}')
    print(f'  {G}Matched to a report:    {matched}{NC}   (unmatched: {unmatched}, no header: {no_header})')
    print(f'  Brain image studies:    {bp_counter.get("brain",0)}')
    print(f'  {G}{B}BRAIN studies WITH a diagnosis label: {brain_matched}{NC}')
    print(f'\n  {B}Labelled brain findings (ready to validate/train):{NC}')
    for k, c in find_counter.most_common():
        print(f'    {c:>4}  {k}')
    print(f'\n  ✓ Matched dataset: {out}')


if __name__ == '__main__':
    main()
