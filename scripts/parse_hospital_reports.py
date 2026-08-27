#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PARSE HOSPITAL PDF REPORTS INTO A LABELLED DATASET
================================================================================
  Reads a folder of radiologist PDF reports (named <patientID>_<Name>.pdf),
  extracts structured labels, and writes a CSV keyed by PATIENT ID — so it
  auto-matches to the DICOM scans (same PatientID in the headers) later.

  Extracts: patient_id · modality (МРТ/МСКТ) · body_part · study_date ·
  conclusion (ЗАКЛЮЧЕНИЕ) · finding flags (tumor/bleed/stroke/…).

  PHI: patient NAMES are dropped from the output; only the numeric ID is kept
  (the matching key). Keep the source PDFs OFF git/cloud.

  Usage: python scripts/parse_hospital_reports.py --src "<folder>" --out labels.csv
================================================================================
"""
from __future__ import annotations
import argparse, csv, re, subprocess, sys
from collections import Counter
from pathlib import Path

G, Y, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0m'

# body-part keywords (Russian) -> canonical
BODY_PARTS = [
    ('brain',  ['головного мозга', 'головной мозг', 'гипофиз', 'головы']),
    ('spine',  ['позвоночник', 'отдел позвоноч', 'пояснич', 'шейн', 'грудн', 'крестц', 'копчик']),
    ('knee',   ['коленн']),
    ('hip',    ['тазобедрен']),
    ('abdomen',['брюшн', 'печен', 'почек', 'забрюшин', 'орган брюш']),
    ('pelvis', ['малого таза', 'таза']),
    ('neck',   ['шеи', 'мягких тканей шеи', 'гортан']),
    ('chest',  ['грудной клетки', 'легк', 'органов грудн']),
    ('joint',  ['сустав', 'плечев', 'локтев', 'голеностоп', 'стоп', 'кист']),
    ('sinus',  ['придаточных пазух', 'пазух носа']),
    ('orbit',  ['орбит', 'глазниц']),
    ('vessel', ['ангиограф', 'сосуд', 'артери']),
]
# finding keywords in the ЗАКЛЮЧЕНИЕ -> flag
FINDINGS = [
    ('tumor',        ['опухол', 'новообразов', 'глиом', 'менингиом', 'аденом', 'mts', 'метастаз', 'объемн']),
    ('hemorrhage',   ['кровоизлия', 'гематом', 'геморраг']),
    ('stroke',       ['инсульт', 'инфаркт', 'ишеми', 'очаг']),
    ('atrophy',      ['атроф']),
    ('hydrocephalus',['гидроцефал', 'расширен желудоч', 'вентрикуломегал']),
    ('disc_hernia',  ['грыж', 'протруз']),
    ('degenerative', ['остеохондроз', 'дегенератив', 'спондил']),
    ('cyst',         ['кист']),
    ('ms_wml',       ['демиелин', 'рассеянн', 'белого вещества', 'лейкоареоз']),
    ('sinusitis',    ['синусит', 'гайморит']),
    ('normal',       ['без патолог', 'патологических изменений не выявлено', 'не изменен', 'норм', 'без особенностей']),
]


def pdf_text(path: Path) -> str:
    try:
        r = subprocess.run(['pdftotext', '-layout', str(path), '-'],
                           capture_output=True, timeout=30)
        return r.stdout.decode('utf-8', 'ignore')
    except Exception:
        return ''


def classify(text_lc: str, table):
    hits = [name for name, kws in table if any(k in text_lc for k in kws)]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='data/hospital_reports_labels.csv')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    pdfs = sorted(Path(args.src).glob('*.pdf'))
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f'Parsing {len(pdfs)} reports…')

    rows, errors = [], 0
    bp_counter, mod_counter, find_counter = Counter(), Counter(), Counter()
    for i, f in enumerate(pdfs):
        pid = re.match(r'(\d+)_', f.name)
        pid = pid.group(1) if pid else ''
        txt = pdf_text(f)
        if not txt.strip():
            errors += 1; continue
        lc = txt.lower()
        modality = 'МСКТ' if ('мскт' in lc or 'компьютерн' in lc) else ('МРТ' if 'мрт' in lc or 'магнитно' in lc else '?')
        bps = classify(lc, BODY_PARTS)
        body_part = bps[0] if bps else 'other'
        # study date: first YYYY-MM-DD or DD.MM.YYYY in a study line
        m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', txt) or re.search(r'\((\d{2}\.\d{2}\.\d{4})\)', txt)
        study_date = m.group(1) if m else ''
        # conclusion
        mc = re.search(r'(ЗАКЛЮЧЕНИЕ|Заключение)[:\s]*(.+)', txt, re.S)
        conclusion = re.sub(r'\s+', ' ', mc.group(2)).strip()[:600] if mc else ''
        finds = classify((conclusion or lc).lower(), FINDINGS)
        # if 'normal' present with others, keep others as the signal
        pos = [x for x in finds if x != 'normal']
        label = 'normal' if (not pos and 'normal' in finds) else (';'.join(pos) if pos else 'unspecified')

        rows.append({'patient_id': pid, 'modality': modality, 'body_part': body_part,
                     'study_date': study_date, 'label': label, 'conclusion': conclusion})
        bp_counter[body_part] += 1; mod_counter[modality] += 1
        for x in pos: find_counter[x] += 1
        if (i+1) % 500 == 0:
            print(f'  …{i+1}/{len(pdfs)}')

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['patient_id','modality','body_part','study_date','label','conclusion'])
        w.writeheader(); w.writerows(rows)

    print('\n' + '='*70)
    print(f'  {B}PARSED {len(rows)} REPORTS  ({errors} unreadable){NC}')
    print('='*70)
    print(f'  Modality: {dict(mod_counter)}')
    print(f'\n  {B}Body part:{NC}')
    for k,c in bp_counter.most_common():
        mark = G if k=='brain' else ''
        print(f'    {mark}{c:>5}  {k}{NC}')
    brain = bp_counter.get('brain',0)
    print(f'\n  {G}{B}BRAIN reports (our focus): {brain}{NC}')
    print(f'\n  {B}Findings flagged in conclusions:{NC}')
    for k,c in find_counter.most_common():
        print(f'    {c:>5}  {k}')
    print(f'\n  ✓ Labels CSV (keyed by patient_id for scan matching): {out}')
    print(f'  ⚠ Names dropped; keep source PDFs off git/cloud (PHI).')


if __name__ == '__main__':
    main()
