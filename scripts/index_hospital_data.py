#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — HOSPITAL DATA INDEXER (read-only, never modifies source)
================================================================================

  Walks a folder of DICOM studies and builds a CSV inventory + a summary of
  what's actually in there: modality/body-part breakdown, patient & study
  counts, how many are brain MRI (usable by Sentinel V1.0) vs other.

  Usage:
    python scripts/index_hospital_data.py --src data/hospital_raw \
        --out data/hospital_anon/inventory.csv
================================================================================
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

try:
    import pydicom
except ImportError:
    print("pydicom required: pip install pydicom"); sys.exit(1)

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

FIELDS = [
    'file_path', 'patient_id', 'patient_name', 'patient_age', 'patient_sex',
    'birth_date', 'modality', 'body_part', 'study_desc', 'series_desc',
    'protocol_name', 'sequence_name', 'study_date', 'study_uid', 'series_uid',
    'manufacturer', 'model', 'institution', 'num_frames', 'rows', 'cols',
]


def classify_sentinel(modality: str, body_part: str, study_desc: str) -> str:
    """Which Sentinel V1.0 model (if any) handles this study."""
    m = (modality or '').upper()
    bp = (body_part or '').upper()
    sd = (study_desc or '').upper()
    text = f"{bp} {sd}"
    if ('BRAIN' in text or 'HEAD' in text) and m in ('MR', 'MRI'):
        return 'brain_tumor_class ✓'
    if ('HEAD' in text or 'BRAIN' in text or 'SKULL' in text) and m == 'CT':
        return 'head_ct ✓'
    if 'CHEST' in text or 'THORAX' in text or 'LUNG' in text:
        return 'chest ✓'
    if 'BREAST' in text or m == 'MG':
        return 'mammography ✓'
    if 'LIVER' in text or 'ABDOMEN' in text:
        return 'liver/abdomen ✗ (no model)'
    if 'SPINE' in text:
        return 'spine ✗ (no model)'
    return f'{m}/{bp or "?"} ✗ (no model)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/hospital_raw')
    p.add_argument('--out', default='data/hospital_anon/inventory.csv')
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'Scanning {src} (read-only)…')
    all_files = [f for f in src.rglob('*') if f.is_file()]
    print(f'  {len(all_files)} files total')

    rows = []
    errors = 0
    sentinel_class = Counter()
    for i, f in enumerate(all_files):
        # Skip junk
        name = f.name
        if name.startswith('.') or 'System Volume' in str(f) or name == 'DO_NOT_SHARE.txt':
            continue
        if f.suffix.lower() in ('.txt', '.xml', '.docx', '.doc', '.pdf'):
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            # Must look like a real image/series object
            if not ds.get('SOPClassUID') and not ds.get('Modality'):
                continue
            modality = str(ds.get('Modality', ''))
            body_part = str(ds.get('BodyPartExamined', ''))
            study_desc = str(ds.get('StudyDescription', ''))
            cls = classify_sentinel(modality, body_part, study_desc)
            sentinel_class[cls] += 1
            rows.append({
                'file_path': str(f.relative_to(src)),
                'patient_id': str(ds.get('PatientID', '')),
                'patient_name': str(ds.get('PatientName', '')),
                'patient_age': str(ds.get('PatientAge', '')),
                'patient_sex': str(ds.get('PatientSex', '')),
                'birth_date': str(ds.get('PatientBirthDate', '')),
                'modality': modality,
                'body_part': body_part,
                'study_desc': study_desc,
                'series_desc': str(ds.get('SeriesDescription', '')),
                'protocol_name': str(ds.get('ProtocolName', '')),
                'sequence_name': str(ds.get('SequenceName', '')),
                'study_date': str(ds.get('StudyDate', '')),
                'study_uid': str(ds.get('StudyInstanceUID', '')),
                'series_uid': str(ds.get('SeriesInstanceUID', '')),
                'manufacturer': str(ds.get('Manufacturer', '')),
                'model': str(ds.get('ManufacturerModelName', '')),
                'institution': str(ds.get('InstitutionName', '')),
                'num_frames': int(ds.get('NumberOfFrames', 1) or 1),
                'rows': int(ds.get('Rows', 0) or 0),
                'cols': int(ds.get('Columns', 0) or 0),
            })
        except Exception:
            errors += 1
        if (i + 1) % 1000 == 0:
            print(f'  …{i+1}/{len(all_files)} scanned, {len(rows)} DICOM so far')

    # Write CSV
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Summary
    print()
    print('=' * 74)
    print(f'  {B}INVENTORY SUMMARY{NC}')
    print('=' * 74)
    print(f'  DICOM files indexed:  {len(rows)}  ({errors} non-DICOM/unreadable skipped)')
    patients = set(r['patient_id'] for r in rows if r['patient_id'])
    studies = set(r['study_uid'] for r in rows if r['study_uid'])
    series = set(r['series_uid'] for r in rows if r['series_uid'])
    print(f'  Unique patients:      {len(patients)}')
    print(f'  Unique studies:       {len(studies)}')
    print(f'  Unique series:        {len(series)}')

    print(f'\n  {B}Body part / modality breakdown:{NC}')
    bp_counter = Counter(f"{r['modality']} / {r['body_part'] or r['study_desc'] or '?'}" for r in rows)
    for k, c in bp_counter.most_common(20):
        print(f'    {c:>5}  {k}')

    print(f'\n  {B}Sentinel V1.0 coverage:{NC}')
    for k, c in sentinel_class.most_common():
        color = G if '✓' in k else Y
        print(f'    {color}{c:>5}  {k}{NC}')

    # Brain study count (by unique study)
    brain_studies = set(r['study_uid'] for r in rows
                        if ('BRAIN' in (r['body_part']+r['study_desc']).upper()
                            or 'HEAD' in (r['body_part']+r['study_desc']).upper())
                        and r['modality'] in ('MR', 'MRI'))
    print(f'\n  {G}{B}BRAIN MRI studies usable by Sentinel: {len(brain_studies)}{NC}')

    # PHI check
    named = sum(1 for r in rows if r['patient_name'].strip())
    print(f'\n  {R}PHI: {named}/{len(rows)} files still have patient names → ANONYMIZE NEXT{NC}')
    print(f'\n  ✓ Full inventory: {out}')


if __name__ == '__main__':
    main()
