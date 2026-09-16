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
from collections import Counter, defaultdict
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
    ap.add_argument('--index', default=None,
                    help='study index CSV from scripts/index_new_studies.py — skips the DICOM walk (fast path)')
    args = ap.parse_args()

    # load labels: patient_id -> {body_part, label, ...}
    labels_by_pid = defaultdict(list)
    with open(args.labels, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            labels_by_pid[r['patient_id']].append(r)
    print(f'Loaded {sum(len(v) for v in labels_by_pid.values())} report labels for {len(labels_by_pid)} patients.')

    def pick_report(pid, study_date, brainy):
        """Best report for this study: same study_date > compatible body part > only/first report."""
        cands = labels_by_pid.get(pid) or []
        if not cands:
            return None, ''
        by_date = [r for r in cands if study_date and (r.get('study_date') or '').replace('-', '') == study_date.replace('-', '')]
        if by_date:
            return by_date[0], 'date'
        if len(cands) == 1:
            return cands[0], 'pid'
        want = 'brain' if brainy else None
        by_bp = [r for r in cands if (r.get('body_part') == 'brain') == (want == 'brain')]
        return (by_bp[0], 'body_part') if by_bp else (cands[0], 'pid-ambiguous')

    img_root = Path(args.images)
    if args.index:
        idx_rows = [r for r in csv.DictReader(open(args.index, encoding='utf-8')) if not r.get('error')]
        print(f'{len(idx_rows)} studies from index {args.index} (no DICOM walk).')
        study_iter = [(r['study_folder'], r['patient_id'], r['modality'], r['body_part'], '', r['study_description'], r['study_date']) for r in idx_rows]
    else:
        study_dirs = [d for d in img_root.iterdir() if d.is_dir()]
        print(f'{len(study_dirs)} image study folders. Reading one header each…')
        study_iter = None

    rows = []
    matched = brain_matched = no_header = unmatched = 0
    bp_counter, find_counter = Counter(), Counter()
    def iter_studies():
        if study_iter is not None:
            for t in study_iter:
                yield t
            return
        for sd in study_dirs:
            f = first_dcm(sd)
            if f is None:
                yield (sd.name, None, '', '', '', '', ''); continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                yield (sd.name, None, '', '', '', '', ''); continue
            yield (sd.name, str(ds.get('PatientID', '')).strip(), str(ds.get('Modality', '')), str(ds.get('BodyPartExamined', '')),
                   str(ds.get('PatientName', '')), str(ds.get('StudyDescription', '')), str(ds.get('StudyDate', '')))

    total = len(study_iter) if study_iter is not None else len(study_dirs)
    for i, (folder, pid, modality, bp, name, desc, study_date) in enumerate(iter_studies()):
        if pid is None:
            no_header += 1; continue
        brainy = is_brain(bp, name, desc)
        rep, quality = pick_report(pid, study_date, brainy)
        row = {'study_folder': folder, 'patient_id': pid, 'modality': modality, 'study_date': study_date, 'match_quality': quality,
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
            print(f'  …{i+1}/{total}  matched={matched}')

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
