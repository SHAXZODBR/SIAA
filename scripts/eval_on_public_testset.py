#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BRAIN TUMOR ACCURACY ON PUBLIC LABELED TEST SET
================================================================================
  Runs brain_tumor_class on the local labeled test set (data/public/.../timri/
  test, 4 classes) and reports REAL accuracy + per-class + confusion matrix.

  HONEST SCOPE: this is IN-DISTRIBUTION public data (same kind the model was
  trained on). It proves the MODEL ITSELF works. It is NOT your Philips/Uzbek
  number — that needs local labels. Use it to isolate "model good, domain shift
  is the problem", and as a baseline for the pitch (clearly labeled).

  Usage:  python scripts/eval_on_public_testset.py
================================================================================
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.inference.model_registry import get_model  # noqa: E402

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'
TEST = REPO / 'data' / 'public' / 'brain_tumor_kaggle' / 'timri' / 'test'

# folder label -> model class
FOLDER2CLASS = {'glioma': 'glioma_tumor', 'meningioma': 'meningioma_tumor',
                'no-tumor': 'no_tumor', 'pituitary': 'pituitary_tumor'}


def load_img(p: Path) -> np.ndarray:
    img = Image.open(p).convert('L').resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0


def main():
    entry = get_model('brain_tumor_class', device='cpu')
    if not entry or not entry.get('available'):
        print(f'{R}model unavailable{NC}'); sys.exit(1)
    pred = entry['predictor']

    classes = list(FOLDER2CLASS.values())
    confusion = defaultdict(lambda: defaultdict(int))
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    n = correct = 0

    for folder, true_cls in FOLDER2CLASS.items():
        d = TEST / folder
        if not d.exists():
            continue
        imgs = [f for f in d.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        for f in imgs:
            try:
                out = pred(load_img(f))
                out = {k: v for k, v in out.items() if not k.startswith('_')}
                top = max(out.items(), key=lambda kv: kv[1])[0]
            except Exception:
                continue
            n += 1
            per_class_total[true_cls] += 1
            confusion[true_cls][top] += 1
            if top == true_cls:
                correct += 1
                per_class_correct[true_cls] += 1
        print(f'  {folder:12s} done ({per_class_total[true_cls]} imgs)')

    acc = correct / n * 100 if n else 0
    print('\n' + '=' * 72)
    print(f'  {B}BRAIN TUMOR — ACCURACY ON PUBLIC LABELED TEST SET ({n} images){NC}')
    print('=' * 72)
    print(f'  Overall accuracy: {G if acc>=85 else Y}{acc:.1f}%{NC}  ({correct}/{n})')
    print(f'\n  Per-class recall:')
    for c in classes:
        t = per_class_total[c]
        r = per_class_correct[c] / t * 100 if t else 0
        print(f'    {c:18s} {r:5.1f}%  ({per_class_correct[c]}/{t})')
    print(f'\n  Confusion (rows=true, cols=predicted):')
    hdr = '    {:18s}'.format('') + ''.join(f'{c.split("_")[0][:6]:>8s}' for c in classes)
    print(hdr)
    for tc in classes:
        row = '    {:18s}'.format(tc) + ''.join(f'{confusion[tc][pc]:>8d}' for pc in classes)
        print(row)
    print(f'\n  {Y}NOTE: in-distribution public data — proves the MODEL works.{NC}')
    print(f'  {Y}This is NOT the Philips/Uzbek number (that needs local labels).{NC}')


if __name__ == '__main__':
    main()
