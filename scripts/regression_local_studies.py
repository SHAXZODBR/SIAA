#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — REGRESSION RUN OVER ALL LOCAL STUDIES (robustness, not validation)
================================================================================
  Runs POST /analyze/study in-process over every study folder under --studies
  (default: the 66 local brain_studies_keep studies) and reports, per study:
  HTTP status, route, rejected/requires_review, detectors run, the triage flag,
  the tumor flag and latency. Joins report-derived labels from --matched by
  folder name and prints triage agreement.

  IMPORTANT: many of these studies overlap the TRAINING split, so agreement
  here is a smoke/robustness signal only — the honest number is the held-out
  gate in scripts/eval_study_level.py. The run FAILS (exit 1) if any study
  returns a 5xx or raises; 4xx rejections are allowed and counted.

  Usage:
    python scripts/regression_local_studies.py [--studies DIR] [--matched CSV] [--out JSON]
================================================================================
"""
from __future__ import annotations
import argparse, csv, json, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SENTINEL_DEV_INSECURE', '1')
os.environ.setdefault('DEV_BYPASS_LICENSE', '1')
os.environ.setdefault('SENTINEL_DATA_DIR', tempfile.mkdtemp(prefix='sentinel_regress_'))
os.environ.setdefault('LOGURU_LEVEL', 'WARNING')

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--studies', default='/Users/shakhzodbtr/Desktop/Siaa_ai/brain_studies_keep')
    ap.add_argument('--matched', default=str(ROOT / 'data/matched_dataset.csv'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    labels = {}
    if Path(args.matched).exists():
        for r in csv.DictReader(open(args.matched, encoding='utf-8')):
            key = (r.get('study_folder') or '').rstrip('/').split('/')[-1]
            if key: labels[key] = (r.get('label') or '').strip()

    from fastapi.testclient import TestClient
    from src.inference.server import app
    rows, crashes = [], 0
    with TestClient(app) as client:
        h = client.get('/health').json()
        print(f"  health: {h.get('status')}  brain_triage loaded={h.get('models', {}).get('brain_triage', {}).get('loaded')}")
        dirs = sorted(d for d in Path(args.studies).iterdir() if d.is_dir())
        if args.limit: dirs = dirs[:args.limit]
        for d in dirs:
            files = sorted(d.rglob('*.dcm'))
            if not files: continue
            t0 = time.time()
            row = {'study': d.name, 'n_files': len(files), 'label': labels.get(d.name, '')}
            try:
                fh = [('files', (f.name, open(f, 'rb'), 'application/dicom')) for f in files]
                res = client.post('/analyze/study?language=ru', files=fh)
                for _, (_, fobj, _) in fh: fobj.close()
                row['status'] = res.status_code
                body = res.json() if res.headers.get('content-type', '').startswith('application/json') else {}
                if res.status_code == 200:
                    tri = next((f for f in body.get('findings', []) if f.get('detector') == 'triage'), None)
                    tum = next((f for f in body.get('findings', []) if f.get('detector') == 'tumor_class'), None)
                    row.update({'rejected': body.get('rejected'), 'requires_review': body.get('requires_review'),
                                'modality': body.get('modality'), 'body_part': body.get('body_part'),
                                'triage_pos': tri and tri.get('positive'), 'triage_conf': tri and tri.get('confidence'),
                                'tumor_pos': tum and tum.get('positive'), 'tumor_class': tum and tum.get('class_name'),
                                'tumor_conf': tum and tum.get('confidence')})
                else:
                    row['detail'] = str(body.get('detail', ''))[:120]
                    if res.status_code >= 500: crashes += 1
            except Exception as e:
                row.update({'status': 'EXC', 'detail': repr(e)[:160]}); crashes += 1
            row['sec'] = round(time.time() - t0, 1)
            rows.append(row)
            flag = 'ABN' if row.get('triage_pos') else ('---' if row.get('status') == 200 else str(row.get('status')))
            print(f"  {d.name[-22:]:22s} files {row['n_files']:3d}  {flag:4s} triage={row.get('triage_conf') or '-'}  tumor={row.get('tumor_class') or '-'}  label={row['label'][:22]:22s} {row['sec']}s")

    ok = [r for r in rows if r.get('status') == 200]
    rej = [r for r in ok if r.get('rejected')]
    print(f"\n{'='*64}\n  {B}REGRESSION SUMMARY{NC}  studies {len(rows)}  200={len(ok)}  rejected={len(rej)}  non-200={len(rows)-len(ok)}  crashes={crashes}")
    lab = [r for r in ok if r['label'] and r['label'] != 'unspecified' and r.get('triage_pos') is not None]
    if lab:
        tp = sum(1 for r in lab if r['label'] != 'normal' and r['triage_pos'])
        fn = sum(1 for r in lab if r['label'] != 'normal' and not r['triage_pos'])
        tn = sum(1 for r in lab if r['label'] == 'normal' and not r['triage_pos'])
        fp = sum(1 for r in lab if r['label'] == 'normal' and r['triage_pos'])
        print(f"  triage vs report labels (n={len(lab)}, overlaps TRAIN — smoke only): sens {tp/max(tp+fn,1):.2f} ({tp}/{tp+fn})  spec {tn/max(tn+fp,1):.2f} ({tn}/{tn+fp})")
    lat = sorted(r['sec'] for r in ok)
    if lat: print(f"  latency: median {lat[len(lat)//2]}s  max {lat[-1]}s")
    out = Path(args.out) if args.out else ROOT / 'data' / 'regression_local_studies.json'
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False)); print(f"  saved → {out}")
    if crashes:
        print(f"  {R}✗ {crashes} crash(es){NC}"); sys.exit(1)
    print(f"  {G}✓ no crashes{NC}")


if __name__ == '__main__':
    main()
