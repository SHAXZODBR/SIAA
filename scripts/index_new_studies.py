#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — INDEX A NEW BATCH OF DICOM STUDIES (before reports exist)
================================================================================
  One row per study folder: identifiers needed to match reports later
  (PatientID + StudyDate), modality / body part / descriptions, scanner, and
  file counts. Reads ONE header per series directory (stop_before_pixels) — no
  pixel data — plus, when that header is not an image object (a report or
  presentation state exported next to the images in one folder), further
  headers of that directory until an image object turns up (at most
  HEADERS_PER_DIR per directory). A study with no image object is still
  described by its first readable header, with error 'no image object' (or
  'no image object in the first N headers of each folder' when the cap was
  hit): the matcher and build_training_set.py --index leave error rows out.
  PatientName is stored only as a sha256 prefix (no names on disk).
  Resumable: rows already in --out are skipped — except rows that describe a
  study by a non-image object (modality 'SR' / 'PR' / … or none): an older
  version took the study fields from the first series header, so those studies
  are indexed again once (a study that really holds no image now gets its
  error and is not read again).

  Usage:
    python scripts/index_new_studies.py --root /Volumes/Transcend/new1 --out data/new1_index.csv
================================================================================
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys, time
from pathlib import Path
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.dicom_objects import NON_IMAGE_MODALITIES, is_image_object  # noqa: E402

FIELDS = ['study_folder', 'study_instance_uid', 'patient_id', 'patient_name_sha16', 'patient_sex', 'patient_age',
          'patient_birth_date', 'study_date', 'study_time', 'accession_number', 'modality', 'body_part',
          'study_description', 'series_descriptions', 'n_series', 'n_dcm', 'n_bmp', 'manufacturer',
          'manufacturer_model', 'institution', 'referring_physician_sha16', 'error']
HEADERS_PER_DIR = 50      # headers read per directory, at most, while looking for an image object
NO_IMAGE = 'no image object'


def walk_counts(study_dir: Path):
    """(n_dcm, n_bmp, {directory: its first HEADERS_PER_DIR .dcm paths in walk order}). One
    'series' per directory, as before: the first path of each list is its series header."""
    n_dcm = n_bmp = 0
    series_files: dict[str, list[Path]] = {}
    for root, dirs, files in os.walk(study_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith('.dcm'):
                n_dcm += 1
                cand = series_files.setdefault(root, [])
                if len(cand) < HEADERS_PER_DIR:
                    cand.append(Path(root) / fn)
            elif low.endswith('.bmp'):
                n_bmp += 1
    return n_dcm, n_bmp, series_files


def sha16(v: str) -> str:
    return hashlib.sha256(v.encode('utf-8', 'ignore')).hexdigest()[:16] if v else ''


def index_study(d: Path) -> dict:
    row = {k: '' for k in FIELDS}; row['study_folder'] = d.name
    try:
        n_dcm, n_bmp, series_files = walk_counts(d)
        row.update(n_dcm=n_dcm, n_bmp=n_bmp, n_series=len(series_files))
        if not series_files:
            row['error'] = 'no dcm'; return row
        descs = []; first = None; first_image = None; capped = False
        for sdir, paths in sorted(series_files.items()):
            try:
                ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
            except Exception as e:
                descs.append(f'<unreadable:{type(e).__name__}>'); ds = None
            else:
                descs.append(str(getattr(ds, 'SeriesDescription', '')).strip())
                if first is None: first = ds
            if first_image is not None:
                continue
            if ds is not None and is_image_object(ds):
                first_image = ds; continue
            # the directory's first file is a report / presentation state (flat export): look further
            for fp in paths[1:]:
                try:
                    more = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                if first is None: first = more
                if is_image_object(more):
                    first_image = more; break
            else:
                capped = capped or len(paths) == HEADERS_PER_DIR
        if first is None:
            row['error'] = 'no readable header'; return row
        # study-level fields come from an IMAGE header: a structured report / presentation state
        # listed first used to make an MR brain study 'SR' with no body part. No image -> described
        # by the first readable header, with an error (not trainable; not re-read on resume).
        if first_image is not None:
            first = first_image
        else:
            row['error'] = (f'{NO_IMAGE} in the first {HEADERS_PER_DIR} headers of each folder' if capped
                            else NO_IMAGE)
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


def is_stale_row(r: dict) -> bool:
    """A row without an error whose modality is not an acquisition modality ('SR', 'PR', '' …):
    the study was described by a non-image object and must be indexed again."""
    if (r.get('error') or '').strip():
        return False
    return (r.get('modality') or '').strip().upper() in NON_IMAGE_MODALITIES | {''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', default='data/new1_index.csv')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        keep = [r for r in rows if not is_stale_row(r)]
        if len(keep) != len(rows):
            # rewrite without them (atomically), then they are indexed again below like new studies
            tmp = out.with_name(out.name + '.tmp')
            with open(tmp, 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
                w.writeheader(); w.writerows(keep)
            os.replace(tmp, out)
            print(f"re-indexing {len(rows) - len(keep)} studies whose row was described by a non-image object "
                  f"(modality SR / PR / none)", flush=True)
        done = {r['study_folder'] for r in keep}
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
