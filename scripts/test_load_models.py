#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PRETRAINED MODEL TEST-LOADER (verification filter)
================================================================================

  Given candidate HuggingFace repo ids, ACTUALLY download + load + run each one
  on a dummy 224x224 image. Only repos that pass here are safe to wire into the
  registry — this rejects dead/hallucinated/gated/uncompatible repos.

  Usage:
    python scripts/test_load_models.py org/repo1 org/repo2 ...
    python scripts/test_load_models.py --file candidates.txt   # one repo per line
================================================================================
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np  # noqa: E402

G, Y, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0m'


def try_classification(repo: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    from PIL import Image
    proc = None
    try:
        proc = AutoImageProcessor.from_pretrained(repo)
    except Exception:
        proc = None
    model = AutoModelForImageClassification.from_pretrained(repo)
    model.eval()
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    if proc is not None:
        inputs = proc(images=img, return_tensors='pt')
    else:
        from transformers import AutoImageProcessor as P
        inputs = P.from_pretrained('google/vit-base-patch16-224')(images=img, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
    n_params = sum(p.numel() for p in model.parameters())
    id2label = getattr(model.config, 'id2label', {}) or {}
    classes = [id2label.get(i, str(i)) for i in range(len(id2label))] if id2label else []
    return {'task': 'image-classification', 'params_m': round(n_params / 1e6, 1),
            'classes': classes, 'logits': list(out.logits.shape)}


def try_segmentation(repo: str):
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    import torch
    from PIL import Image
    model = AutoModelForSemanticSegmentation.from_pretrained(repo)
    model.eval()
    proc = AutoImageProcessor.from_pretrained(repo)
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    inputs = proc(images=img, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
    n_params = sum(p.numel() for p in model.parameters())
    return {'task': 'segmentation', 'params_m': round(n_params / 1e6, 1),
            'classes': list(getattr(model.config, 'id2label', {}).values()),
            'logits': list(out.logits.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repos', nargs='*')
    ap.add_argument('--file')
    args = ap.parse_args()
    repos = list(args.repos)
    if args.file:
        repos += [l.strip() for l in Path(args.file).read_text().splitlines()
                  if l.strip() and not l.startswith('#')]
    if not repos:
        print('No repos given.'); sys.exit(1)

    print(f'Test-loading {len(repos)} candidate model(s)…\n')
    ok, fail = [], []
    for repo in repos:
        print(f'  → {repo}')
        result = None
        for loader in (try_classification, try_segmentation):
            try:
                result = loader(repo)
                break
            except Exception as e:
                last = f'{type(e).__name__}: {str(e)[:120]}'
        if result:
            ok.append((repo, result))
            print(f'    {G}✓ LOADS{NC}  [{result["task"]}, {result["params_m"]}M] '
                  f'classes={result["classes"][:6]}{"…" if len(result["classes"])>6 else ""}')
        else:
            fail.append((repo, last))
            print(f'    {R}✗ FAIL{NC}  {last}')
        print()

    print('=' * 70)
    print(f'  {G}LOADABLE: {len(ok)}{NC}   {R}FAILED: {len(fail)}{NC}')
    print('=' * 70)
    for repo, r in ok:
        print(f'  {G}✓{NC} {repo:55s} {r["task"]:16s} {r["params_m"]}M  {r["classes"][:5]}')
    for repo, e in fail:
        print(f'  {R}✗{NC} {repo:55s} {e}')


if __name__ == '__main__':
    main()
