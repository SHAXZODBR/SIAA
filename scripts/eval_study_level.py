#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — STUDY-LEVEL EVALUATION OF A SLICE CLASSIFIER
================================================================================
  Slice-level accuracy understates the clinically-relevant number: a radiologist
  triages a STUDY, not one slice. This script groups val slices by patient
  (filenames are {patient_id}_{slice_idx}.png), aggregates slice probabilities
  per study, and reports study-level sensitivity/specificity/accuracy.

  Aggregations compared:
    mean  — average abnormal-probability across slices
    max   — most-suspicious slice decides ("any slice abnormal → study abnormal")
    vote  — majority vote of slice argmax

  Usage:
    python scripts/eval_study_level.py --model models/brain_triage_local \
        --data data/local_finetune_bin/val --device mps
================================================================================
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

G, Y, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/brain_triage_local')
    ap.add_argument('--data', default='data/local_finetune_bin/val')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--abnormal-class', default='abnormal')
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(args.device).eval()
    id2label = model.config.id2label
    abn_idx = next(i for i, l in id2label.items() if l == args.abnormal_class)

    root = Path(args.data)
    # slice probabilities grouped by (study := patient_id from filename, true label)
    studies: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_slices = 0
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        true_cls = cls_dir.name
        for fp in sorted(cls_dir.glob('*.png')):
            pid = fp.stem.rsplit('_', 1)[0]          # {patient_id}_{slice_idx}
            img = Image.open(fp).convert('RGB')
            with torch.no_grad():
                inputs = processor(images=img, return_tensors='pt').to(args.device)
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
            studies[(pid, true_cls)].append(float(probs[int(abn_idx)]))
            n_slices += 1

    print(f"\n{'='*64}\n  {B}STUDY-LEVEL EVALUATION{NC}  ({len(studies)} studies, {n_slices} slices)\n{'='*64}")

    def score(threshold_fn, name):
        tp = tn = fp_ = fn = 0
        for (pid, true_cls), probs in studies.items():
            pred_abn = threshold_fn(probs)
            is_abn = true_cls == args.abnormal_class
            if is_abn and pred_abn: tp += 1
            elif is_abn and not pred_abn: fn += 1
            elif not is_abn and pred_abn: fp_ += 1
            else: tn += 1
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp_, 1)
        acc = (tp + tn) / max(tp + tn + fp_ + fn, 1)
        bal = (sens + spec) / 2
        print(f"  {name:22s} acc {acc:.3f} | sens {sens:.3f} | spec {spec:.3f} | balanced {bal:.3f}"
              f"   (TP {tp} FN {fn} TN {tn} FP {fp_})")
        return {'name': name, 'acc': acc, 'sens': sens, 'spec': spec, 'balanced': bal,
                'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp_}

    results = []
    results.append(score(lambda ps: sum(ps) / len(ps) >= 0.5, 'mean-prob @0.5'))
    results.append(score(lambda ps: max(ps) >= 0.5, 'max-prob @0.5'))
    results.append(score(lambda ps: sum(p >= 0.5 for p in ps) > len(ps) / 2, 'majority vote'))
    # threshold sweep on mean-prob for the best balanced score
    best = None
    for t in [x / 20 for x in range(2, 19)]:
        r = None
        tp = tn = fp_ = fn = 0
        for (pid, true_cls), probs in studies.items():
            pred_abn = (sum(probs) / len(probs)) >= t
            is_abn = true_cls == args.abnormal_class
            if is_abn and pred_abn: tp += 1
            elif is_abn and not pred_abn: fn += 1
            elif not is_abn and pred_abn: fp_ += 1
            else: tn += 1
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp_, 1)
        bal = (sens + spec) / 2
        if best is None or bal > best[1]:
            best = (t, bal, sens, spec)
    t, bal, sens, spec = best
    print(f"\n  {G}best mean-prob threshold {t:.2f}: balanced {bal:.3f} (sens {sens:.3f} / spec {spec:.3f}){NC}")

    out = Path(args.model) / 'study_level_eval.json'
    out.write_text(json.dumps({'results': results,
                               'best_mean_threshold': {'threshold': t, 'balanced': bal,
                                                       'sens': sens, 'spec': spec}}, indent=2))
    print(f"  saved → {out}\n")


if __name__ == '__main__':
    main()
