#!/usr/bin/env python3
"""Generic classifier evaluator: accuracy + per-class recall + confusion on a
labeled folder (class subdirs). Usage:
  python scripts/eval_classifier.py --model-dir models/X --data data/ft/X/val
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

G, Y, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data', required=True, help='val dir with class subfolders')
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    proc = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir).eval()
    id2label = model.config.id2label
    classes = [id2label[i] for i in range(len(id2label))]

    data = Path(args.data)
    folders = [d for d in data.iterdir() if d.is_dir()]
    conf = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int); cor = defaultdict(int)
    n = correct = 0
    for d in folders:
        true = d.name
        for f in d.iterdir():
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                img = Image.open(f).convert('RGB')
                inp = proc(images=img, return_tensors='pt')
                with torch.no_grad():
                    pred = id2label[int(model(**inp).logits.argmax(-1))]
            except Exception:
                continue
            n += 1; tot[true] += 1; conf[true][pred] += 1
            if pred == true:
                correct += 1; cor[true] += 1
    acc = correct / n * 100 if n else 0
    pred_total = {c: sum(conf[t][c] for t in classes) for c in classes}  # column sums
    f1s = []
    print(f"\n{'='*72}\n  {args.title or args.model_dir} — {n} images\n{'='*72}")
    print(f"  Overall accuracy: {G if acc>=80 else Y}{acc:.1f}%{NC}  ({correct}/{n})")
    print(f"  {'class':20s} {'precision':>10s} {'recall':>9s} {'f1':>7s}  support")
    for c in classes:
        tp = conf[c][c]
        prec = tp / pred_total[c] * 100 if pred_total[c] else 0
        rec = tp / tot[c] * 100 if tot[c] else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s.append(f1)
        col = G if f1 >= 70 else (Y if f1 >= 40 else R)
        print(f"    {c:18s} {prec:9.1f}% {rec:8.1f}% {col}{f1:6.1f}%{NC}  {tot[c]}")
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    print(f"  {'-'*50}")
    print(f"    {'MACRO-F1 (avg)':18s} {'':10s} {'':9s} {G if macro_f1>=70 else (Y if macro_f1>=40 else R)}{macro_f1:6.1f}%{NC}")
    print("  Confusion (row=true → predicted):")
    for tc in classes:
        if tot[tc] == 0: continue
        cells = ", ".join(f"{pc.split('_')[0][:8]}:{conf[tc][pc]}" for pc in classes if conf[tc][pc])
        print(f"    {tc:20s} → {cells}")


if __name__ == '__main__':
    main()
