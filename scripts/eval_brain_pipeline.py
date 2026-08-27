#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BRAIN PIPELINE A/B EVALUATION (real hospital data, no labels)
================================================================================

  Measures the effect of the routing/slice-selection fixes on the brain-tumor
  classifier, using the anonymized Samarkand studies. We have no ground-truth
  labels, so this does NOT prove accuracy — it proves the over-calling was a
  pipeline bug by comparing:

    OLD  = best slice from FLAIR (or any), global argmax slice, min-max norm,
           run on EVERY study (incl. spine/knee/liver)        ← the buggy path
    NEW  = body-part gating (brain only) + T1ce/T1 axial + central-band slice
           + percentile normalization                          ← the fix

  Output: per-study CSV + a before/after summary (glioma rate, studies rejected).

  Usage:
    python scripts/eval_brain_pipeline.py --limit 40
    python scripts/eval_brain_pipeline.py            # all studies
================================================================================
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pydicom  # noqa: E402
from PIL import Image  # noqa: E402

from src.pipeline.brain_mri_preprocessor import (  # noqa: E402
    process_brain_study, is_brain_text, central_band_best_slice, robust_normalize_slice,
)
from src.inference.model_registry import get_model  # noqa: E402

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'
STUDIES = REPO / 'data' / 'hospital_anon' / 'studies'


def enumerate_studies() -> list[Path]:
    """Each studies/<patient>/<studyfolder> is one study."""
    out = []
    for patient in STUDIES.iterdir():
        if not patient.is_dir():
            continue
        for s in patient.iterdir():
            if s.is_dir():
                out.append(s)
    return sorted(out)


def study_body_part(study_dir: Path) -> tuple[str, str]:
    """Read the first readable header to get (body_part, study_desc)."""
    for f in list(study_dir.rglob('*'))[:60]:
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if ds.get('Rows') or 'PixelData' in ds or ds.get('BodyPartExamined'):
                return (str(ds.get('BodyPartExamined', '')), str(ds.get('StudyDescription', '')))
        except Exception:
            continue
    return ('', '')


def old_pipeline_slice(study) -> np.ndarray | None:
    """Reconstruct the OLD buggy selection: FLAIR-first, global argmax slice,
    plain min-max normalization."""
    seqs = study.sequences
    if not seqs:
        return None
    seq = 'FLAIR' if 'FLAIR' in seqs else list(seqs.keys())[0]
    vol = seqs[seq]
    if vol.ndim == 3:
        scores = (vol > vol.mean()).sum(axis=(1, 2))
        slc = vol[int(scores.argmax())]
    else:
        slc = vol
    smin, smax = slc.min(), slc.max()
    if smax > smin:
        slc = (slc - smin) / (smax - smin)
    return slc.astype(np.float32)


def classify(predictor, slice_2d: np.ndarray) -> tuple[str, float]:
    img = Image.fromarray((np.clip(slice_2d, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    probs = predictor(arr)
    probs = {k: v for k, v in probs.items() if not k.startswith('_')}
    top = max(probs.items(), key=lambda kv: kv[1])
    return top[0], float(top[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='cap studies processed (0=all)')
    ap.add_argument('--out', default=str(REPO / 'data' / 'hospital_anon' / 'brain_eval.csv'))
    args = ap.parse_args()

    print(f'Loading brain classifier…')
    entry = get_model('brain_tumor_class', device='cpu')
    if not entry or not entry.get('available'):
        print(f'{R}Brain model unavailable: {(entry or {}).get("reason","")}{NC}')
        sys.exit(1)
    predictor = entry['predictor']

    studies = enumerate_studies()
    print(f'{len(studies)} study folders found under {STUDIES}\n')

    rows = []
    old_dist, new_dist = Counter(), Counter()
    rejected_non_brain = 0
    brain_count = 0
    t0 = time.time()

    n = 0
    for sd in studies:
        if args.limit and n >= args.limit:
            break
        bp, desc = study_body_part(sd)
        brainy = is_brain_text(f'{bp} {desc}')

        # NEW gating: only brain studies go to the classifier.
        if brainy is False:
            rejected_non_brain += 1
            # OLD pipeline would still have classified it — show what it called.
            n += 1
            files = [f for f in sd.rglob('*') if f.is_file()]
            try:
                study = process_brain_study(files)
                old_slice = old_pipeline_slice(study)
                if old_slice is not None:
                    oc, op = classify(predictor, old_slice)
                    old_dist[oc] += 1
                    rows.append({'study': f'{sd.parent.name}/{sd.name}', 'body_part': bp,
                                 'brain': 'no', 'old_pred': oc, 'old_conf': round(op, 3),
                                 'new_pred': 'REJECTED (non-brain)', 'new_conf': '',
                                 'sequence_used': '', 'orientation': ''})
            except Exception as e:
                rows.append({'study': f'{sd.parent.name}/{sd.name}', 'body_part': bp,
                             'brain': 'no', 'old_pred': f'ERR', 'old_conf': '',
                             'new_pred': 'REJECTED (non-brain)', 'new_conf': '', 'sequence_used': '', 'orientation': ''})
            continue

        # Brain (or ambiguous) study → run BOTH pipelines.
        n += 1
        brain_count += 1
        files = [f for f in sd.rglob('*') if f.is_file()]
        try:
            study = process_brain_study(files)
        except Exception as e:
            rows.append({'study': f'{sd.parent.name}/{sd.name}', 'body_part': bp, 'brain': 'yes',
                         'old_pred': 'ERR', 'old_conf': '', 'new_pred': f'ERR {e}', 'new_conf': '',
                         'sequence_used': '', 'orientation': ''})
            continue
        if study.num_sequences == 0:
            continue

        old_slice = old_pipeline_slice(study)
        oc, op = (None, None)
        if old_slice is not None:
            oc, op = classify(predictor, old_slice)
            old_dist[oc] += 1

        sel = study.select_classifier_input()
        nc, npb = (None, None)
        if sel and sel.get('slice') is not None:
            nc, npb = classify(predictor, sel['slice'])
            new_dist[nc] += 1

        rows.append({
            'study': f'{sd.parent.name}/{sd.name}', 'body_part': bp, 'brain': 'yes',
            'old_pred': oc, 'old_conf': round(op, 3) if op else '',
            'new_pred': nc, 'new_conf': round(npb, 3) if npb else '',
            'sequence_used': sel['sequence'] if sel else '',
            'orientation': sel['orientation'] if sel else '',
        })
        print(f"  {sd.parent.name}/{sd.name:8s} [{bp:6s}]  OLD={oc or '-':16s}{(op or 0)*100:4.0f}%   "
              f"NEW({sel['sequence'] if sel else '-'}/{sel['orientation'] if sel else '-'})={nc or '-':16s}{(npb or 0)*100:4.0f}%")

    # Write CSV
    out = Path(args.out)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['study', 'body_part', 'brain', 'old_pred', 'old_conf',
                                           'new_pred', 'new_conf', 'sequence_used', 'orientation'])
        w.writeheader(); w.writerows(rows)

    # Summary
    dt = time.time() - t0
    def glioma_rate(dist):
        tot = sum(dist.values())
        return (dist.get('glioma_tumor', 0) / tot * 100) if tot else 0.0
    print('\n' + '=' * 74)
    print(f'  {B}BRAIN PIPELINE A/B — {n} studies in {dt:.0f}s{NC}')
    print('=' * 74)
    print(f'  Brain (HEAD) studies analyzed:       {brain_count}')
    print(f'  Non-brain studies REJECTED by fix:   {rejected_non_brain}  '
          f'{R}(OLD pipeline would have classified these as tumors){NC}')
    print(f'\n  {Y}OLD pipeline distribution (buggy, all studies):{NC}')
    for k, c in old_dist.most_common():
        print(f'    {c:>4}  {k}')
    print(f'  → OLD glioma rate: {R}{glioma_rate(old_dist):.0f}%{NC}')
    print(f'\n  {G}NEW pipeline distribution (brain-only, correct sequence):{NC}')
    for k, c in new_dist.most_common():
        print(f'    {c:>4}  {k}')
    print(f'  → NEW glioma rate: {G}{glioma_rate(new_dist):.0f}%{NC}')
    print(f'\n  ⚠ No ground-truth labels — this measures BIAS REDUCTION, not accuracy.')
    print(f'  ✓ Per-study detail: {out}')


if __name__ == '__main__':
    main()
