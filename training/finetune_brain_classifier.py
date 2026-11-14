#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BRAIN TUMOR CLASSIFIER FINE-TUNING
================================================================================

  Fine-tunes a pretrained brain MRI classifier on YOUR clinic's data.

  Why fine-tune?
    Pretrained models (Devarshi, BehradG, andrei-teodor) were trained on
    Kaggle / public datasets dominated by Western patient populations.
    Uzbek MRI scans differ in:
      • scanner make/model (Siemens 1.5T common in UZ)
      • acquisition protocol (slice thickness, contrast volume)
      • patient demographics (age, body habitus)
    Even 200 fine-tuning images can shift accuracy from 85 → 92 %.

  Approach:
    1. Start from a pretrained HuggingFace classifier (already in registry).
    2. Freeze the backbone, train only the classifier head — fast, stable.
    3. Optionally unfreeze the last 2 transformer/conv blocks if you have
       1000+ images.
    4. Save the fine-tuned model to models/brain_finetuned/ — the registry
       will auto-pick it up over the public weights.

  Data layout expected:
    data/brain_finetune/
      train/
        glioma_tumor/    img001.png img002.png ...
        meningioma_tumor/...
        no_tumor/
        pituitary_tumor/
      val/
        (same structure)
      tune/            (optional; written by scripts/combine_training_sets.py)
        (same structure)

  Checkpoint selection (--select-split): the best epoch on the SELECTION split
  is saved. Default: tune/ when <data>/tune has images, else val/ with a loud
  WARNING — in a release set val/ is the GATE set (locked holdout, scored by
  scripts/eval_study_level.py), and picking the best of N epochs on it biases
  the gate numbers upward. training_metrics.json records which split and which
  rule selected the checkpoint (checkpoint_selection).

  What "best" means (--select-metric):
    slice_acc  (default) the highest slice accuracy. Blind to the operating
               point: on a 3:1 abnormal:normal split an epoch that calls almost
               everything abnormal scores as well as a balanced one.
    study      the shipped protocol on the selection split — a study is flagged
               when the mean abnormal probability of its slices is >= 0.50 (the
               study key is the file name without its trailing _<k>, as in
               scripts/eval_study_level.py). Among the epochs whose study
               sensitivity is >= --select-min-sens (default 0.90: a triage tool
               must first not miss) the one with the highest specificity is
               kept; when no epoch reaches it, the most sensitive one. Needs a
               class named by --abnormal-class (default 'abnormal').
  The study-level numbers of every epoch are recorded in training_metrics.json
  whichever rule selects.

  Augmentation (--augment): random horizontal flip, small rotation / shift /
  zoom and brightness / contrast jitter on the TRAINING images only. Without it
  the last blocks memorise a few thousand slices within three epochs (train
  accuracy 1.00 while the selection loss doubles).

  --seed N makes the shuffling, the augmentation and the new head reproducible.

  Run:
    python training/finetune_brain_classifier.py \\
        --data data/brain_finetune \\
        --epochs 15 \\
        --batch-size 16 \\
        --lr 5e-5

  Output:
    models/brain_finetuned/
      pytorch_model.bin
      config.json
      preprocessor_config.json
      training_metrics.json
================================================================================
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Make repo importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ----------------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------------

CLASS_ORDER = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


class BrainImageDataset(Dataset):
    def __init__(self, root: Path, processor, classes: list[str], augment=None):
        self.processor = processor
        self.augment = augment            # PIL image -> PIL image, training split only
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls in classes:
            cls_dir = root / cls
            if not cls_dir.is_dir():
                print(f"[warn] missing class dir: {cls_dir}")
                continue
            for ext in IMAGE_EXTS:
                for fp in sorted(cls_dir.glob(ext)):
                    self.samples.append((fp, self.cls_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        print(f"[data] {root.name}: {len(self.samples)} images across {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._load(path)
        if self.augment is not None:
            img = self.augment(img)
        proc = self.processor(images=img, return_tensors='pt')
        return {
            'pixel_values': proc['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def _load(path: Path):
        from PIL import Image
        if path.suffix.lower() in ('.dcm', '.dicom'):
            import pydicom
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array
            # Window to 0-255
            arr = arr.astype('float32')
            arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255
            img = Image.fromarray(arr.astype('uint8')).convert('RGB')
        else:
            img = Image.open(path).convert('RGB')
        return img


IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.dcm')


def _has_images(root: Path) -> bool:
    return root.is_dir() and any(next(root.glob(f'*/{ext}'), None) is not None for ext in IMAGE_EXTS)


def resolve_select_split(data: Path, requested: Optional[str]) -> tuple[str, Optional[str]]:
    """(split used to pick the best epoch, warning or None).

    Default: 'tune' when <data>/tune holds images, else 'val' + a warning that the gate
    set is being used for checkpoint selection. An explicit 'tune' without images is an
    error (SystemExit); an explicit 'val' still warns.
    """
    if requested == 'tune' and not _has_images(data / 'tune'):
        raise SystemExit(f"--select-split tune: no images under {data / 'tune'}/<class>/ "
                         f"(build the set with scripts/combine_training_sets.py --tune-frac > 0)")
    split = requested or ('tune' if _has_images(data / 'tune') else 'val')
    warning = None
    if split == 'val':
        warning = (f"WARNING: checkpoint selection uses {data / 'val'} — in a release set that is the GATE set "
                   f"(locked holdout). The best-of-epochs pick is then tuned to the patients the gate scores, "
                   f"so its sens/spec are optimistic. Rebuild the set with a tune/ split "
                   f"(scripts/combine_training_sets.py --tune-frac 0.1) or pass --select-split tune.")
    return split, warning


SELECT_METRICS = ('slice_acc', 'study')
STUDY_THRESHOLD = 0.5                 # the shipped protocol: mean abnormal probability >= 0.50 -> flagged


def build_augmentation():
    """Training-time augmentation of one PIL slice: left-right flip, small rotation / shift / zoom,
    brightness / contrast jitter. Deliberately mild — the anatomy stays an axial brain slice."""
    from torchvision import transforms as T
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ])


def study_key(path: Path) -> str:
    """'<study key>_<k>.png' -> study key (the rule of scripts/eval_study_level.py)."""
    return Path(path).stem.rsplit('_', 1)[0]


def _auc(pos: list[float], neg: list[float]) -> Optional[float]:
    """Rank AUC (ties count 1/2). None when a class is empty."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def study_metrics(samples: list[tuple[Path, int]], abnormal_probs: list[float], abnormal_idx: int,
                  threshold: float = STUDY_THRESHOLD) -> dict:
    """Shipped-protocol metrics of the selection split: slices are grouped into studies by file name,
    a study is flagged when the mean abnormal probability of its slices is >= threshold."""
    by_study: dict[tuple[str, int], list[float]] = {}
    for (path, label), prob in zip(samples, abnormal_probs):
        by_study.setdefault((study_key(path), label), []).append(float(prob))
    pos = [sum(v) / len(v) for (_, label), v in by_study.items() if label == abnormal_idx]
    neg = [sum(v) / len(v) for (_, label), v in by_study.items() if label != abnormal_idx]
    tp = sum(1 for m in pos if m >= threshold)
    tn = sum(1 for m in neg if m < threshold)
    sens = tp / len(pos) if pos else None
    spec = tn / len(neg) if neg else None
    return {'n_abnormal': len(pos), 'n_normal': len(neg), 'tp': tp, 'fn': len(pos) - tp,
            'tn': tn, 'fp': len(neg) - tn, 'sensitivity': sens, 'specificity': spec,
            'balanced_accuracy': (sens + spec) / 2 if sens is not None and spec is not None else None,
            'auc': _auc(pos, neg), 'threshold': threshold}


def selection_key(metric: str, sel_acc: float, sel_loss: float, study: Optional[dict], min_sens: float) -> tuple:
    """Sort key of an epoch — the larger, the better.
    slice_acc: the slice accuracy alone (the first of equal epochs wins, as before).
    study    : epochs whose study sensitivity reaches min_sens rank above all others and among
               themselves by specificity, then sensitivity; the rest by sensitivity, then
               specificity; a lower selection loss breaks what is still tied."""
    if metric == 'slice_acc':
        return (sel_acc,)
    sens = (study or {}).get('sensitivity')
    spec = (study or {}).get('specificity')
    sens = -1.0 if sens is None else sens
    spec = -1.0 if spec is None else spec
    if sens >= min_sens:
        return (1, spec, sens, -sel_loss)
    return (0, sens, spec, -sel_loss)


# ----------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, optim, sched, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, batch in enumerate(loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        out = model(pixel_values=pixel_values)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = criterion(logits, labels)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 10 == 0:
            print(f"  ep {epoch}/{total_epochs} step {step}/{len(loader)}  loss {loss.item():.4f}  acc {correct/total:.3f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, probs_out: Optional[list] = None):
    """(loss, slice accuracy, per-class counts). With probs_out (a list) the softmax row of every
    sample is appended in loader order (the loader must not shuffle)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class = {}
    for batch in loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        out = model(pixel_values=pixel_values)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        if probs_out is not None:
            probs_out.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            per_class.setdefault(l, {'correct': 0, 'total': 0})
            per_class[l]['total'] += 1
            if p == l:
                per_class[l]['correct'] += 1
    return total_loss / max(total, 1), correct / max(total, 1), per_class


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True,
                     help='Root with train/ and val/ subdirs')
    p.add_argument('--base-model', default='andrei-teodor/resnet-pretrained-brain-mri',
                     help='HuggingFace repo to start from')
    p.add_argument('--output', type=Path, default=Path('models/brain_finetuned'))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--unfreeze-last-blocks', type=int, default=0,
                     help='How many of the last backbone blocks to unfreeze (0 = head-only)')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--classes', default=None,
                     help='comma-separated class folder names (default = tumor 4-class)')
    p.add_argument('--base-model-fresh', action='store_true',
                     help='reinit classifier head for a NEW task (different class set)')
    p.add_argument('--balance', action='store_true',
                     help='inverse-frequency class weighting (critical for imbalanced data like hemorrhage)')
    p.add_argument('--select-split', choices=['tune', 'val'], default=None,
                     help='split that picks the saved epoch (default: tune when <data>/tune exists, '
                          'else val with a WARNING: val is the gate set of a release)')
    p.add_argument('--select-metric', choices=SELECT_METRICS, default='slice_acc',
                     help="what 'best epoch' means: slice_acc (default) or study = the shipped protocol on the "
                          'selection split: sensitivity >= --select-min-sens first, then the highest specificity')
    p.add_argument('--select-min-sens', type=float, default=0.90,
                     help='--select-metric study: study sensitivity an epoch must reach (default 0.90)')
    p.add_argument('--abnormal-class', default='abnormal',
                     help="class whose probability flags a study (--select-metric study; default 'abnormal')")
    p.add_argument('--augment', action='store_true',
                     help='augment the training images (flip, small rotation / shift / zoom, brightness / contrast)')
    p.add_argument('--seed', type=int, default=None,
                     help='seed for shuffling, augmentation and a new head (default: not seeded)')
    args = p.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    select_split, select_warning = resolve_select_split(args.data, args.select_split)
    if select_warning:
        print('\033[1;33m' + '!' * 78 + '\n' + select_warning + '\n' + '!' * 78 + '\033[0m', flush=True)
    else:
        print(f"[select] checkpoint selection on {args.data / select_split} (the val/ gate set is not used)")

    # Generic: any disease, not just tumor. Folder names must match these.
    class_order = [c.strip() for c in args.classes.split(',')] if args.classes else CLASS_ORDER
    abnormal_idx = class_order.index(args.abnormal_class) if args.abnormal_class in class_order else None
    if args.select_metric == 'study' and abnormal_idx is None:
        raise SystemExit(f"--select-metric study needs the class '{args.abnormal_class}' in --classes "
                         f"(got {class_order}); pass --abnormal-class or use --select-metric slice_acc")
    if not 0.0 <= args.select_min_sens <= 1.0:
        raise SystemExit('--select-min-sens must be between 0 and 1')

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"[device] {device}")

    # Load base model
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    print(f"[model] loading base: {args.base_model}")
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    base_model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(class_order),
        id2label={i: c for i, c in enumerate(class_order)},
        label2id={c: i for i, c in enumerate(class_order)},
        ignore_mismatched_sizes=True,
    )

    # Freeze backbone, leave head trainable
    for name, param in base_model.named_parameters():
        if 'classifier' in name or 'head' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optionally unfreeze last N transformer/conv blocks
    if args.unfreeze_last_blocks > 0:
        all_params = list(base_model.named_parameters())
        # Heuristic: last 25 % of params usually = last few blocks
        unfreeze_count = int(len(all_params) * (args.unfreeze_last_blocks / 12))
        for name, param in all_params[-unfreeze_count:]:
            param.requires_grad = True

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print(f"[params] trainable {trainable:,} / total {total:,} ({100*trainable/total:.1f}%)")

    base_model.to(device)

    # Data
    augment = build_augmentation() if args.augment else None
    if augment is not None:
        print('[augment] training images: horizontal flip, rotation ±10°, shift 5 %, zoom 0.9–1.1, '
              'brightness / contrast ±15 %')
    train_ds = BrainImageDataset(args.data / 'train', processor, class_order, augment=augment)
    sel_ds   = BrainImageDataset(args.data / select_split, processor, class_order)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=False)
    sel_loader   = DataLoader(sel_ds,   batch_size=args.batch_size, shuffle=False)

    # Optim
    optim = AdamW([p for p in base_model.parameters() if p.requires_grad],
                  lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))
    # Class weighting — for imbalanced data (e.g. hemorrhage 14% positive), plain
    # CrossEntropy lets the model "cheat" by predicting the majority class, giving
    # high accuracy but useless recall. Inverse-frequency weights force it to care
    # about the rare (clinically critical) class.
    if args.balance:
        from collections import Counter
        counts = Counter(lbl for _, lbl in train_ds.samples)
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (len(class_order) * max(counts.get(i, 1), 1)) for i in range(len(class_order))],
            dtype=torch.float32, device=device)
        print(f"[balance] class weights: {[round(w,2) for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {'epochs': []}
    best_sel_acc = -1.0          # slice accuracy of the saved epoch
    best_key = None              # the first epoch is always saved
    best_epoch = 0
    best_study = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            base_model, train_loader, optim, sched, criterion,
            device, epoch, args.epochs,
        )
        probs: Optional[list] = [] if abnormal_idx is not None else None
        sel_loss, sel_acc, per_cls = evaluate(base_model, sel_loader, criterion, device, probs_out=probs)
        study = (study_metrics(sel_ds.samples, [row[abnormal_idx] for row in probs], abnormal_idx)
                 if probs is not None else None)
        elapsed = (time.time() - start) / 60
        print(f"  >> epoch {epoch}: train acc {tr_acc:.3f} | {select_split} acc {sel_acc:.3f} | "
              f"{select_split} loss {sel_loss:.4f} | {elapsed:.1f}min elapsed")
        if study is not None and study['sensitivity'] is not None and study['specificity'] is not None:
            auc = study['auc']
            print(f"     {select_split} studies ({study['n_abnormal']} abnormal / {study['n_normal']} normal): "
                  f"sensitivity {study['sensitivity']:.3f} | specificity {study['specificity']:.3f} | "
                  f"auc {'—' if auc is None else format(auc, '.3f')}")

        ep = {
            'epoch': epoch,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'select_split': select_split, 'select_loss': sel_loss, 'select_acc': sel_acc,
            'per_class': {class_order[k]: v for k, v in per_cls.items()},
        }
        if study is not None:
            ep['study'] = study
        if select_split == 'val':            # legacy keys: only when val really was evaluated
            ep.update(val_loss=sel_loss, val_acc=sel_acc)
        metrics['epochs'].append(ep)

        key = selection_key(args.select_metric, sel_acc, sel_loss, study, args.select_min_sens)
        if best_key is None or key > best_key:
            best_key = key
            best_sel_acc = sel_acc
            best_epoch = epoch
            best_study = study
            base_model.save_pretrained(args.output)
            processor.save_pretrained(args.output)
            why = (f"{select_split}_acc {sel_acc:.3f}" if args.select_metric == 'slice_acc' else
                   f"{select_split} study sensitivity {study['sensitivity']:.3f} / specificity "
                   f"{study['specificity']:.3f}" if study and study['sensitivity'] is not None
                   and study['specificity'] is not None else f"{select_split}_acc {sel_acc:.3f}")
            print(f"     ✓ saved best to {args.output} ({why})")

    metrics['checkpoint_selection'] = {
        'split': select_split,
        'metric': ('slice accuracy' if args.select_metric == 'slice_acc' else
                   f'study protocol (mean {args.abnormal_class} probability >= {STUDY_THRESHOLD:.2f}): '
                   f'sensitivity >= {args.select_min_sens:.2f} first, then specificity'),
        'select_metric': args.select_metric,
        'best_epoch': best_epoch, 'best_acc': best_sel_acc,
        'data': str(args.data / select_split), 'gate_set_used_for_selection': select_split == 'val',
    }
    if args.select_metric == 'study':
        metrics['checkpoint_selection'].update(min_sensitivity=args.select_min_sens, best_study=best_study)
    metrics['augment'] = bool(args.augment)
    metrics['seed'] = args.seed
    if select_split == 'val':
        metrics['best_val_acc'] = best_sel_acc
    metrics['base_model'] = args.base_model
    metrics['training_minutes'] = (time.time() - start) / 60
    (args.output / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ DONE. best {select_split} acc {best_sel_acc:.3f} (epoch {best_epoch}) | saved to {args.output}/")
    if select_warning:
        print('\033[1;33m' + select_warning + '\033[0m')


if __name__ == '__main__':
    main()
