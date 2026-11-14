#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — MATCH DICOM IMAGE STUDIES TO RADIOLOGIST REPORT LABELS
================================================================================
  Release pipeline stage 3 (docs/dev/release_pipeline_contract.md).

  Walks a folder of DICOM studies (or, fast path, reads the study index that
  scripts/index_new_studies.py wrote) — one header per study for PatientID +
  StudyDate + Modality + body part — and joins to the parsed + labelled
  report rows (scripts/label_reports.py output, or the old
  data/hospital_reports_labels.csv). Output = the labelled IMAGE dataset:
  which scan folder has which diagnosis.

  Join order (best first) — one PDF may hold several studies, so a patient
  can have several report rows:
    1. same patient_id AND same study date                 -> match_quality 'date'
    2. same patient + compatible modality (MR/CT) + body part -> 'modality+body_part'
       same body part, modality unknown/different           -> 'body_part'
       same modality, report body part unknown              -> 'modality'
    3. the patient has exactly one report                   -> 'pid'
    4. several reports, none compatible: first one          -> 'pid-ambiguous'

  Carried through from the labels file when present (old files without them
  simply get ''): abnormal, label_source, label_confidence, needs_review.

  PHI: keeps only patient_id (the join key) + study folder; never names.

  Usage:
    python scripts/match_images_to_reports.py --index data/<batch>_index.csv \
        --labels data/<batch>_labels.csv --out data/<batch>_matched.csv
    (without an index: --images "<image folder>" instead of --index)
================================================================================
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

G, Y, B, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0;31m', '\033[0m'

# report/DICOM modality spellings -> canonical 'MR' | 'CT' | '' (unknown)
_MODALITY_ALIASES = {
    'MR': 'MR', 'MRI': 'MR', 'MRT': 'MR', 'МРТ': 'MR', 'МР': 'MR',
    'CT': 'CT', 'MSCT': 'CT', 'КТ': 'CT', 'МСКТ': 'CT', 'РКТ': 'CT',
}
CARRY_COLUMNS = ('abnormal', 'label_source', 'label_confidence', 'needs_review')
OUTPUT_COLUMNS = ('study_folder', 'patient_id', 'modality', 'study_date', 'match_quality',
                  'body_part_dicom', 'is_brain', 'report_matched', 'report_body_part', 'label',
                  *CARRY_COLUMNS, 'report_modality', 'report_study_date')


def first_dcm(study_dir: Path) -> Optional[Path]:
    for root, _, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                return Path(root) / f
    return None


def is_brain(body_part: str, name: str, desc: str) -> bool:
    t = f"{body_part} {name} {desc}".upper()
    return any(k in t for k in ['BRAIN', 'HEAD', 'SKULL', 'CEREBR'])


def norm_modality(value: str) -> str:
    """'МСКТ' / 'CT' / 'ct ' -> 'CT'; 'МРТ' / 'MR' -> 'MR'; anything else -> ''."""
    v = (value or '').strip().upper()
    if v in _MODALITY_ALIASES:
        return _MODALITY_ALIASES[v]
    for alias, canon in _MODALITY_ALIASES.items():
        if len(alias) >= 2 and alias in v:
            return canon
    return ''


def norm_date(value: str) -> str:
    """'2026-07-21' / '20260721' / '21.07.2026' -> '20260721'; unknown -> ''."""
    v = (value or '').strip()
    if not v:
        return ''
    if '.' in v and len(v.split('.')) == 3 and len(v.split('.')[2]) == 4:
        d, m, y = v.split('.')
        return f'{y}{m.zfill(2)}{d.zfill(2)}'
    digits = ''.join(ch for ch in v if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ''


def pick_report(cands: list[dict], study_date: str, modality: str, brainy: bool) -> tuple[Optional[dict], str]:
    """Best report row for one image study — see module docstring for the order."""
    if not cands:
        return None, ''
    want_bp = 'brain' if brainy else None
    dicom_mod = norm_modality(modality)

    def bp_ok(r: dict) -> Optional[bool]:
        bp = (r.get('body_part') or '').strip().lower()
        if not bp:
            return None
        return (bp == 'brain') == (want_bp == 'brain')

    def mod_ok(r: dict) -> Optional[bool]:
        rm = norm_modality(r.get('modality') or '')
        if not rm or not dicom_mod:
            return None
        return rm == dicom_mod

    def compat_rank(r: dict) -> int:
        """Higher is better; used only to break ties INSIDE a tier."""
        return int(bp_ok(r) is True) * 2 + int(mod_ok(r) is True)

    sd = norm_date(study_date)
    if sd:
        by_date = [r for r in cands if norm_date(r.get('study_date') or '') == sd]
        if by_date:
            return max(by_date, key=compat_rank), 'date'

    both = [r for r in cands if bp_ok(r) is True and mod_ok(r) is True]
    if both:
        return both[0], 'modality+body_part'
    only_bp = [r for r in cands if bp_ok(r) is True]          # right body part, modality unknown or different
    if only_bp:
        return only_bp[0], 'body_part'
    only_mod = [r for r in cands if mod_ok(r) is True and bp_ok(r) is None]
    if only_mod:
        return only_mod[0], 'modality'
    if len(cands) == 1:
        return cands[0], 'pid'
    return cands[0], 'pid-ambiguous'


def load_labels(path: Path) -> dict[str, list[dict]]:
    labels_by_pid: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            pid = (r.get('patient_id') or '').strip()
            if pid:
                labels_by_pid[pid].append(r)
    return labels_by_pid


def make_row(folder: str, pid: str, modality: str, bp: str, study_date: str, brainy: bool,
             rep: Optional[dict], quality: str) -> dict:
    row = {'study_folder': folder, 'patient_id': pid, 'modality': modality, 'study_date': study_date,
           'match_quality': quality,
           'body_part_dicom': bp or ('brain' if brainy else ''), 'is_brain': 'yes' if brainy else 'no',
           'report_matched': 'yes' if rep else 'no',
           'report_body_part': (rep.get('body_part') or '') if rep else '',
           'label': (rep.get('label') or '') if rep else ''}
    for c in CARRY_COLUMNS:
        row[c] = (rep.get(c) or '') if rep else ''
    row['report_modality'] = (rep.get('modality') or '') if rep else ''
    row['report_study_date'] = (rep.get('study_date') or '') if rep else ''
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='Join DICOM image studies to labelled report rows by patient_id.')
    ap.add_argument('--images', default=None,
                    help='root folder with one sub-folder per study (required without --index; unused with it)')
    ap.add_argument('--labels', default='data/hospital_reports_labels.csv',
                    help='labelled report rows (label_reports.py output; old hospital_reports_labels.csv also works)')
    ap.add_argument('--out', default='data/matched_dataset.csv')
    ap.add_argument('--index', default=None,
                    help='study index CSV from scripts/index_new_studies.py — skips the DICOM walk (fast path)')
    args = ap.parse_args()
    if not args.index and not args.images:
        ap.error('--images is required when no --index is given (the study folders are walked for headers)')

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f'{R}labels file not found: {labels_path}{NC}'); sys.exit(1)
    labels_by_pid = load_labels(labels_path)
    n_reports = sum(len(v) for v in labels_by_pid.values())
    print(f'Loaded {n_reports} report labels for {len(labels_by_pid)} patients.')
    if n_reports == 0:
        print(f'{R}no report rows with a patient_id in {labels_path}{NC}'); sys.exit(1)

    study_iter: Optional[list[tuple]] = None
    study_dirs: list[Path] = []
    if args.index:
        with open(args.index, encoding='utf-8') as fh:
            idx_rows = [r for r in csv.DictReader(fh) if not (r.get('error') or '').strip()]
        print(f'{len(idx_rows)} studies from index {args.index} (no DICOM walk).')
        study_iter = [(r['study_folder'], (r.get('patient_id') or '').strip(), r.get('modality') or '',
                       r.get('body_part') or '', '', r.get('study_description') or '', r.get('study_date') or '')
                      for r in idx_rows]
    else:
        img_root = Path(args.images)
        if not img_root.is_dir():
            print(f'{R}image root not found: {img_root}{NC}'); sys.exit(1)
        study_dirs = sorted(d for d in img_root.iterdir() if d.is_dir())
        print(f'{len(study_dirs)} image study folders. Reading one header each…')

    def iter_studies() -> Iterator[tuple]:
        if study_iter is not None:
            yield from study_iter
            return
        import pydicom
        for sd in study_dirs:
            f = first_dcm(sd)
            if f is None:
                yield (sd.name, None, '', '', '', '', ''); continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                yield (sd.name, None, '', '', '', '', ''); continue
            # PatientName is deliberately NOT read: a name containing e.g. "…HEAD…" used to turn a
            # C-spine study into a "brain" row (seen on 3 of 66 rehearsal studies), and it is PHI.
            yield (sd.name, str(ds.get('PatientID', '')).strip(), str(ds.get('Modality', '')), str(ds.get('BodyPartExamined', '')),
                   '', str(ds.get('StudyDescription', '')), str(ds.get('StudyDate', '')))

    rows: list[dict] = []
    matched = brain_matched = no_header = unmatched = brain_unmatched = 0
    bp_counter: Counter = Counter()
    find_counter: Counter = Counter()
    quality_brain: Counter = Counter()
    abnormal_counter: Counter = Counter()
    review_counter: Counter = Counter()
    total = len(study_iter) if study_iter is not None else len(study_dirs)
    for i, (folder, pid, modality, bp, name, desc, study_date) in enumerate(iter_studies()):
        if pid is None or pid == '':
            no_header += 1; continue
        brainy = is_brain(bp, name, desc)
        rep, quality = pick_report(labels_by_pid.get(pid) or [], study_date, modality, brainy)
        rows.append(make_row(folder, pid, modality, bp, study_date, brainy, rep, quality))
        bp_counter['brain' if brainy else (bp or 'other')] += 1
        if rep:
            matched += 1
            if brainy:
                quality_brain[quality] += 1
                lbl = (rep.get('label') or '').strip()
                if lbl not in ('', 'unspecified'):
                    brain_matched += 1
                    for part in lbl.split(';'):
                        if part:
                            find_counter[part] += 1
                abn = (rep.get('abnormal') or '').strip()
                if abn:
                    abnormal_counter[abn] += 1
                if (rep.get('needs_review') or '').strip() == 'yes':
                    review_counter['yes'] += 1
        else:
            unmatched += 1
            if brainy:
                brain_unmatched += 1
        if (i + 1) % 300 == 0:
            print(f'  …{i+1}/{total}  matched={matched}')

    if not rows:
        print(f'{R}no studies with a readable PatientID — nothing to write{NC}'); sys.exit(1)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        w.writeheader(); w.writerows(rows)

    n_brain = bp_counter.get('brain', 0)
    print('\n' + '=' * 70)
    print(f'  {B}MATCHED IMAGE ↔ REPORT DATASET{NC}')
    print('=' * 70)
    print(f'  Image studies:          {len(rows)}')
    print(f'  {G}Matched to a report:    {matched}{NC}   (unmatched: {unmatched}, no header: {no_header})')
    print(f'  Brain image studies:    {n_brain}   (no report: {brain_unmatched})')
    print(f'  {B}Brain match quality:{NC}')
    for q in ('date', 'modality+body_part', 'body_part', 'modality', 'pid', 'pid-ambiguous'):
        if quality_brain.get(q):
            flag = '' if q == 'date' else f'  {Y}(weaker key){NC}'
            print(f'    {quality_brain[q]:>5}  {q}{flag}')
    weak = sum(c for q, c in quality_brain.items() if q != 'date')
    print(f'    by date: {quality_brain.get("date", 0)}   by weaker keys: {weak}')
    print(f'  {G}{B}BRAIN studies WITH a diagnosis label: {brain_matched}{NC}')
    if abnormal_counter:
        print(f'  abnormal column (brain, matched): ' + ', '.join(f'{k}={v}' for k, v in sorted(abnormal_counter.items())))
    else:
        print(f'  {Y}labels file has no abnormal column — build_training_set.py will use label only{NC}')
    if review_counter.get('yes'):
        print(f'  {Y}needs_review=yes on {review_counter["yes"]} matched brain studies '
              f'(build_training_set.py --skip-needs-review drops them){NC}')
    print(f'\n  {B}Labelled brain findings (ready to validate/train):{NC}')
    for k, c in find_counter.most_common():
        print(f'    {c:>4}  {k}')
    print(f'\n  ✓ Matched dataset: {out}')


if __name__ == '__main__':
    main()
