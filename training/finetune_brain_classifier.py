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
    def __init__(self, root: Path, processor, classes: list[str]):
        self.processor = processor
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls in classes:
            cls_dir = root / cls
            if not cls_dir.is_dir():
                print(f"[warn] missing class dir: {cls_dir}")
                continue
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.dcm'):
                for fp in cls_dir.glob(ext):
                    self.samples.append((fp, self.cls_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        print(f"[data] {root.name}: {len(self.samples)} images across {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._load(path)
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
def evaluate(model, loader, criterion, device):
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
    args = p.parse_args()

    # Generic: any disease, not just tumor. Folder names must match these.
    class_order = [c.strip() for c in args.classes.split(',')] if args.classes else CLASS_ORDER

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
    train_ds = BrainImageDataset(args.data / 'train', processor, class_order)
    val_ds   = BrainImageDataset(args.data / 'val',   processor, class_order)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

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
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            base_model, train_loader, optim, sched, criterion,
            device, epoch, args.epochs,
        )
        val_loss, val_acc, per_cls = evaluate(base_model, val_loader, criterion, device)
        elapsed = (time.time() - start) / 60
        print(f"  >> epoch {epoch}: train acc {tr_acc:.3f} | val acc {val_acc:.3f} | val loss {val_loss:.4f} | {elapsed:.1f}min elapsed")

        metrics['epochs'].append({
            'epoch': epoch,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'per_class': {class_order[k]: v for k, v in per_cls.items()},
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            base_model.save_pretrained(args.output)
            processor.save_pretrained(args.output)
            print(f"     ✓ saved best to {args.output} (val_acc {val_acc:.3f})")

    metrics['best_val_acc'] = best_val_acc
    metrics['base_model'] = args.base_model
    metrics['training_minutes'] = (time.time() - start) / 60
    (args.output / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ DONE. best val acc {best_val_acc:.3f} | saved to {args.output}/")


if __name__ == '__main__':
    main()
