#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — INDEX A NEW BATCH OF DICOM STUDIES (before reports exist)
================================================================================
  One row per study folder: identifiers needed to match reports later
  (PatientID + StudyDate), modality / body part / descriptions, scanner, and
  file counts. Reads ONE header per series (stop_before_pixels) — no pixel data.
  PatientName is stored only as a sha256 prefix (no names on disk).
  Resumable: rows already in --out are skipped.

  Usage:
    python scripts/index_new_studies.py --root /Volumes/Transcend/new1 --out data/new1_index.csv
================================================================================
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys, time
from pathlib import Path
import pydicom

FIELDS = ['study_folder', 'study_instance_uid', 'patient_id', 'patient_name_sha16', 'patient_sex', 'patient_age',
          'patient_birth_date', 'study_date', 'study_time', 'accession_number', 'modality', 'body_part',
          'study_description', 'series_descriptions', 'n_series', 'n_dcm', 'n_bmp', 'manufacturer',
          'manufacturer_model', 'institution', 'referring_physician_sha16', 'error']


def walk_counts(study_dir: Path):
    n_dcm = n_bmp = 0
    series_first: dict[str, Path] = {}
    for root, dirs, files in os.walk(study_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith('.dcm'):
                n_dcm += 1
                series_first.setdefault(root, Path(root) / fn)
            elif low.endswith('.bmp'):
                n_bmp += 1
    return n_dcm, n_bmp, series_first


def sha16(v: str) -> str:
    return hashlib.sha256(v.encode('utf-8', 'ignore')).hexdigest()[:16] if v else ''


def index_study(d: Path) -> dict:
    row = {k: '' for k in FIELDS}; row['study_folder'] = d.name
    try:
        n_dcm, n_bmp, series_first = walk_counts(d)
        row.update(n_dcm=n_dcm, n_bmp=n_bmp, n_series=len(series_first))
        if not series_first:
            row['error'] = 'no dcm'; return row
        descs = []; first = None
        for i, (sdir, fp) in enumerate(sorted(series_first.items())):
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            except Exception as e:
                descs.append(f'<unreadable:{type(e).__name__}>'); continue
            descs.append(str(getattr(ds, 'SeriesDescription', '')).strip())
            if first is None: first = ds
        if first is None:
            row['error'] = 'no readable header'; return row
        g = lambda k: str(getattr(first, k, '')).strip()
        row.update(study_instance_uid=g('StudyInstanceUID'), patient_id=g('PatientID'),
                   patient_name_sha16=sha16(g('PatientName')), patient_sex=g('PatientSex'), patient_age=g('PatientAge'),
                   patient_birth_date=g('PatientBirthDate'), study_date=g('StudyDate'), study_time=g('StudyTime')[:6],
                   accession_number=g('AccessionNumber'), modality=g('Modality'), body_part=g('BodyPartExamined'),
                   study_description=g('StudyDescription'), series_descriptions=' | '.join(x for x in descs if x),
                   manufacturer=g('Manufacturer'), manufacturer_model=g('ManufacturerModelName'),
                   institution=g('InstitutionName'), referring_physician_sha16=sha16(g('ReferringPhysicianName')))
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'[:120]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', default='data/new1_index.csv')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r['study_folder'] for r in csv.DictReader(open(out, encoding='utf-8'))}
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(('.', '$', 'System')))
    if a.limit: dirs = dirs[:a.limit]
    todo = [d for d in dirs if d.name not in done]
    print(f"studies: {len(dirs)}  already indexed: {len(done)}  to do: {len(todo)}", flush=True)
    t0 = time.time(); n = 0
    with open(out, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done and fh.tell() == 0: w.writeheader()
        for d in todo:
            w.writerow(index_study(d)); fh.flush(); n += 1
            if n % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"progress {n}/{len(todo)}  {rate*60:.0f} studies/min  eta {((len(todo)-n)/max(rate,1e-6))/60:.0f} min", flush=True)
    print(f"DONE indexed {n} new studies → {out}", flush=True)


if __name__ == '__main__':
    main()
