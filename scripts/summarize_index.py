#!/usr/bin/env python3
"""Summarize a study index CSV (scripts/index_new_studies.py): modality / body part /
description mix, date range, unique patients, sex & age, series/file counts, errors.
Usage: python scripts/summarize_index.py --index data/new1_index.csv"""
from __future__ import annotations
import argparse, csv, re
from collections import Counter

B, NC = '\033[0;34m', '\033[0m'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--index', default='data/new1_index.csv'); a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index, encoding='utf-8')))
    ok = [r for r in rows if not r.get('error')]
    print(f"\n{B}INDEX SUMMARY{NC}  {a.index}\n  studies {len(rows)}  readable {len(ok)}  errors {len(rows)-len(ok)}")
    for r in [r for r in rows if r.get('error')][:5]: print(f"    err: {r['study_folder'][-20:]} {r['error'][:80]}")
    def top(name, key, n=12):
        c = Counter((key(r) or '(blank)') for r in ok)
        print(f"\n  {B}{name}{NC}"); [print(f"    {v:5d}  {k}") for k, v in c.most_common(n)]
    top('modality', lambda r: r['modality'])
    top('body part', lambda r: r['body_part'])
    top('study description', lambda r: r['study_description'], 20)
    top('modality × body part', lambda r: f"{r['modality']} / {r['body_part'] or '?'}", 15)
    dates = sorted(r['study_date'] for r in ok if r['study_date'])
    pts = {r['patient_id'] for r in ok if r['patient_id']}
    print(f"\n  {B}dates{NC} {dates[0] if dates else '?'} → {dates[-1] if dates else '?'}   unique patients {len(pts)}   institutions {Counter(r['institution'] for r in ok).most_common(2)}")
    sex = Counter(r['patient_sex'] or '?' for r in ok); print(f"  {B}sex{NC} {dict(sex)}")
    ages = []
    for r in ok:
        m = re.match(r'(\d+)Y', r['patient_age'] or '')
        if m: ages.append(int(m.group(1)))
    if ages:
        bins = Counter((min(a, 89) // 10) * 10 for a in ages)
        print(f"  {B}age{NC} n={len(ages)} median {sorted(ages)[len(ages)//2]}  by decade " + ' '.join(f"{k}s:{v}" for k, v in sorted(bins.items())))
    ns = [int(r['n_series'] or 0) for r in ok]; nd = [int(r['n_dcm'] or 0) for r in ok]
    print(f"  {B}per study{NC} series median {sorted(ns)[len(ns)//2] if ns else 0}  dcm median {sorted(nd)[len(nd)//2] if nd else 0}  max {max(nd) if nd else 0}  total dcm {sum(nd):,}")
    head = [r for r in ok if r['modality'] == 'CT' and ('HEAD' in r['body_part'].upper() or re.search(r'HEAD|BCA|BRAIN', r['study_description'], re.I))]
    print(f"\n  {B}brain-relevant (CT head){NC}: {len(head)} studies, {len({r['patient_id'] for r in head})} patients")
    print(f"    descriptions: {Counter(r['study_description'] for r in head).most_common(8)}")
    mr = [r for r in ok if r['modality'] == 'MR']; print(f"  {B}MR studies{NC}: {len(mr)}")


if __name__ == '__main__':
    main()
