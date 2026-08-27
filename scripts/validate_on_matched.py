#!/usr/bin/env python3
"""First real validation: run the brain panel on matched studies, compare its
output to the radiologist's diagnosis. Small N = a first signal, not final."""
from __future__ import annotations
import csv, sys, warnings
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.brain_analysis import analyze_brain_study

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'
IMG = Path("/Users/shakhzodbtr/Desktop/Siaa_ai/МРТ + МСКТ")

rows = [r for r in csv.DictReader(open('data/matched_dataset.csv', encoding='utf-8'))
        if r['is_brain'] == 'yes' and r['label'] not in ('', 'unspecified')]
print(f"Validating on {len(rows)} labelled brain studies…\n")

# tumor detector confusion (report says tumor?  vs  panel flags tumor?)
tp = fp = tn = fn = 0
abn_agree = 0; n_norm = n_abn = 0
done = 0
for r in rows:
    files = [f for f in (IMG / r['study_folder']).rglob('*') if f.suffix.lower() == '.dcm']
    try:
        res = analyze_brain_study(files, device='cpu', include_experimental=False)
    except Exception:
        continue
    if res.get('rejected') or res.get('error'):
        continue
    done += 1
    report_tumor = 'tumor' in r['label']
    panel_tumor = any(f['detector'] == 'tumor_class' and f['positive'] for f in res.get('findings', []))
    if report_tumor and panel_tumor: tp += 1
    elif report_tumor and not panel_tumor: fn += 1
    elif not report_tumor and panel_tumor: fp += 1
    else: tn += 1
    # normal vs abnormal
    report_abn = r['label'] != 'normal'
    panel_abn = res.get('overall_assessment', {}).get('abnormal_flagged', False)
    if report_abn: n_abn += 1
    else: n_norm += 1
    if report_abn == panel_abn: abn_agree += 1

print('=' * 64)
print(f"  {B}FIRST VALIDATION — {done} real labelled brain studies{NC}")
print('=' * 64)
print(f"\n  {B}Tumor detector (the shippable model) vs radiologist:{NC}")
sens = tp/(tp+fn)*100 if tp+fn else 0
spec = tn/(tn+fp)*100 if tn+fp else 0
print(f"    report tumor: {tp+fn}   |   report not-tumor: {tn+fp}")
print(f"    ✓ caught tumor (TP): {tp}    ✗ missed (FN): {fn}")
print(f"    false alarms (FP): {fp}     correct clear (TN): {tn}")
print(f"    sensitivity: {sens:.0f}%   specificity: {spec:.0f}%")
print(f"\n  {B}Normal-vs-abnormal agreement (production panel):{NC}")
print(f"    {abn_agree}/{done} agree   (normal cases: {n_norm}, abnormal: {n_abn})")
print(f"\n  {Y}⚠ N is small (per-finding counts tiny) — a first signal on real{NC}")
print(f"  {Y}  local patients, not a final accuracy. More images → robust.{NC}")
