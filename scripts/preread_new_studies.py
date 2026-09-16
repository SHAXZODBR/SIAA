#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — AI PRE-READ OF A NEW STUDY BATCH (no reports yet)
================================================================================
  Runs POST /analyze/study in-process over studies selected from an index CSV
  (scripts/index_new_studies.py) and stores one JSON line per study:
  route, findings (detector/status/positive/confidence), overall assessment,
  rejection reason, latency. When the radiology reports arrive, these pre-reads
  become a prospective-style comparison set (AI read BEFORE the label was known).
  Resumable; .bmp files are skipped. Never certifies normal.

  Usage:
    python scripts/preread_new_studies.py --root /Volumes/Transcend/new1 \
        --index data/new1_index.csv --out data/new1_prereads.jsonl \
        --modality CT --body-part HEAD [--limit N]
================================================================================
"""
from __future__ import annotations
import argparse, csv, json, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SENTINEL_DEV_INSECURE', '1')
os.environ.setdefault('DEV_BYPASS_LICENSE', '1')
os.environ.setdefault('SENTINEL_DATA_DIR', tempfile.mkdtemp(prefix='sentinel_preread_'))
os.environ.setdefault('LOGURU_LEVEL', 'WARNING')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--index', default='data/new1_index.csv')
    ap.add_argument('--out', default='data/new1_prereads.jsonl')
    ap.add_argument('--modality', default=''); ap.add_argument('--body-part', default='')
    ap.add_argument('--desc-contains', default='', help='also select when study_description contains this (case-insensitive)')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--language', default='ru')
    ap.add_argument('--no-select-series', action='store_true',
                    help='upload every file instead of pre-selecting the axial brain series client-side (CT head only)')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index, encoding='utf-8')))
    sel = []
    for r in rows:
        if r.get('error'): continue
        if a.modality and r['modality'].upper() != a.modality.upper(): continue
        ok_bp = (not a.body_part) or (a.body_part.upper() in r['body_part'].upper())
        ok_desc = bool(a.desc_contains) and a.desc_contains.upper() in r['study_description'].upper()
        if ok_bp or ok_desc: sel.append(r)
    if a.limit: sel = sel[:a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding='utf-8'):
            try: done.add(json.loads(line)['study_folder'])
            except Exception: pass
    todo = [r for r in sel if r['study_folder'] not in done]
    print(f"selected {len(sel)} studies; already done {len(done)}; to do {len(todo)}", flush=True)

    from fastapi.testclient import TestClient
    from src.inference.server import app
    t0 = time.time(); n = 0
    with TestClient(app) as client, open(out, 'a', encoding='utf-8') as fh:
        h = client.get('/health').json(); print("health:", h.get('status'), {k: v['loaded'] for k, v in h.get('models', {}).items()}, flush=True)
        for r in todo:
            d = Path(a.root) / r['study_folder']
            files = sorted(p for p in d.rglob('*.dcm'))
            rec = {'study_folder': r['study_folder'], 'patient_id': r['patient_id'], 'study_date': r['study_date'],
                   'modality': r['modality'], 'body_part': r['body_part'], 'study_description': r['study_description'],
                   'n_files': len(files)}
            # CT head: choose the axial brain series here (same helper the server uses) and upload only
            # that series — the server re-runs the same selection on what it receives. Cuts drive IO a lot.
            if (not a.no_select_series) and r['modality'].upper() == 'CT' and 'HEAD' in r['body_part'].upper():
                try:
                    from src.pipeline.ct_head_selection import select_ct_head_slices
                    sel = select_ct_head_slices(files)
                    if sel is not None and sel.series.slices:
                        files = [sl.path for sl in sel.series.slices]
                        rec['preselected_series'] = sel.series.description
                except Exception as e:
                    rec['preselect_error'] = repr(e)[:120]
            rec['n_files_uploaded'] = len(files)
            ts = time.time()
            try:
                handles = [('files', (f.name, open(f, 'rb'), 'application/dicom')) for f in files]
                res = client.post(f'/analyze/study?language={a.language}', files=handles)
                for _, (_, fo, _) in handles: fo.close()
                rec['status'] = res.status_code
                body = res.json() if res.headers.get('content-type', '').startswith('application/json') else {}
                if res.status_code == 200:
                    rec.update(study_id=body.get('study_id'), route=body.get('route'), rejected=body.get('rejected'),
                               requires_review=body.get('requires_review'), rejection_reason=body.get('rejection_reason'),
                               overall=(body.get('overall_assessment') or {}).get('text') or body.get('overall_impression'),
                               model_identity=[m.get('sha256_12') for m in body.get('model_identity', [])],
                               findings=[{k: f.get(k) for k in ('detector', 'class_name', 'status', 'positive', 'confidence')}
                                         for f in body.get('findings', [])])
                else:
                    rec['detail'] = str(body.get('detail', ''))[:160]
            except Exception as e:
                rec['status'] = 'EXC'; rec['detail'] = repr(e)[:160]
            rec['sec'] = round(time.time() - ts, 1)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush(); n += 1
            if n % 10 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"progress {n}/{len(todo)}  {rate*60:.1f}/min  eta {((len(todo)-n)/max(rate,1e-6))/60:.0f} min  last={rec.get('status')} {rec.get('route','')} {rec['sec']}s", flush=True)
    print(f"DONE {n} pre-reads → {out}", flush=True)


if __name__ == '__main__':
    main()
