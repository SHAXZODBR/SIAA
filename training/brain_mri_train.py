"""
================================================================================
  SENTINEL MEDICAL AI — BRAIN MRI TRAINING TEMPLATE (PHASE 4)
================================================================================
  Brain MRI is 3D (volumes) not 2D (images), so we need different architecture.

  APPROACH OPTIONS:
  1. Slice-by-slice 2D — Process axial slices with DenseNet121 (simpler)
  2. Full 3D CNN — DenseNet3D for entire volume (better accuracy)
  3. Transformer — ViT3D (state of the art, needs more data)

  THIS SCRIPT: Uses MONAI's DenseNet3D for 3D brain volume classification.

  DATASETS SUPPORTED:
  - BraTS (Brain Tumor Segmentation Challenge)
  - IXI (healthy brain MRIs)
  - ADNI (Alzheimer's — requires application)

  CONDITIONS DETECTED:
  - Normal
  - Glioblastoma (high-grade glioma)
  - Meningioma
  - Metastasis
  - Stroke (acute/chronic)
  - Hemorrhage
  - Multiple sclerosis lesions
  - Alzheimer's atrophy
  - Hydrocephalus

  REQUIREMENTS:
  - MONAI >= 1.3
  - GPU with 16GB+ VRAM (T4 x2 on Kaggle works)
  - 3D nifti files (.nii.gz format)

  USAGE:
    # On Kaggle:
    # 1. Add BraTS dataset
    # 2. Paste this script + run
================================================================================
"""

import os
import sys
import json
import time
import math
import copy
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings('ignore')

# MONAI — medical imaging framework (required for 3D)
try:
    from monai.networks.nets import DenseNet121 as DenseNet3D
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityd, RandFlipd, RandRotated, RandScaleIntensityd,
        RandShiftIntensityd, RandAdjustContrastd, Resized, EnsureTyped,
    )
    from monai.data import Dataset as MonaiDataset, DataLoader as MonaiLoader
    from monai.metrics import ROCAUCMetric
    HAS_MONAI = True
except ImportError:
    print("Install MONAI: pip install monai[nibabel]")
    HAS_MONAI = False


# ===================================================================
# CONFIG
# ===================================================================

CONFIG = {
    # Classes — Brain pathologies
    'class_names': [
        'Normal',
        'Glioblastoma',           # GBM — aggressive brain tumor
        'Meningioma',             # Benign brain tumor
        'Metastasis',             # Cancer spread to brain
        'Stroke_Acute',           # Recent stroke
        'Stroke_Chronic',         # Old stroke
        'Hemorrhage',             # Brain bleeding
        'Multiple_Sclerosis',     # MS lesions
        'Alzheimer_AD',           # AD atrophy pattern
        'Hydrocephalus',          # Ventricle enlargement
    ],

    # Paths
    'data_dir': '/kaggle/input/brain-mri',
    'output_dir': '/kaggle/working/sentinel_brain',

    # Model — MONAI DenseNet121 for 3D
    'spatial_dims': 3,
    'in_channels': 1,                 # Single-channel (T1 or T2)
    'image_size': [128, 128, 128],    # 3D volume size

    # Training
    'epochs': 50,
    'batch_size': 2,                  # 3D uses more memory
    'gradient_accumulation': 4,       # Effective batch 8
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,

    # Augmentation — medical-specific for 3D
    'augmentation': {
        'rand_flip_prob': 0.5,
        'rand_rotate_range': [0.2, 0.2, 0.2],
        'rand_scale_intensity': 0.1,
    },

    # Hardware
    'mixed_precision': True,
    'num_workers': 4,
    'seed': 42,

    # Targets
    'target_recall': 0.80,          # Slightly lower than chest (3D is harder)
    'target_auc': 0.85,
}


# ===================================================================
# DATA LOADING
# ===================================================================

def build_data_list(data_dir):
    """Build list of {image, label} dicts for MONAI pipeline."""
    data_dir = Path(data_dir)
    data_list = []

    # Scan for nifti files
    for class_idx, class_name in enumerate(CONFIG['class_names']):
        class_dir = data_dir / class_name.lower()
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob('*.nii.gz'):
            data_list.append({
                'image': str(img_path),
                'label': class_idx,
                'class_name': class_name,
            })

    print(f"[DATA] Found {len(data_list)} MRI volumes")
    counts = {}
    for d in data_list:
        counts[d['class_name']] = counts.get(d['class_name'], 0) + 1
    for cls, count in sorted(counts.items()):
        print(f"  {cls:25s}: {count}")

    return data_list


def get_transforms(is_train=True):
    """3D augmentation pipeline with MONAI."""
    if is_train:
        return Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
            ScaleIntensityd(keys=['image']),
            Resized(keys=['image'], spatial_size=CONFIG['image_size']),
            RandFlipd(keys=['image'], prob=CONFIG['augmentation']['rand_flip_prob'], spatial_axis=0),
            RandFlipd(keys=['image'], prob=CONFIG['augmentation']['rand_flip_prob'], spatial_axis=1),
            RandRotated(
                keys=['image'],
                range_x=CONFIG['augmentation']['rand_rotate_range'][0],
                range_y=CONFIG['augmentation']['rand_rotate_range'][1],
                range_z=CONFIG['augmentation']['rand_rotate_range'][2],
                prob=0.5,
            ),
            RandScaleIntensityd(keys=['image'], factors=CONFIG['augmentation']['rand_scale_intensity'], prob=0.5),
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
            EnsureTyped(keys=['image']),
        ])
    else:
        return Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
            ScaleIntensityd(keys=['image']),
            Resized(keys=['image'], spatial_size=CONFIG['image_size']),
            EnsureTyped(keys=['image']),
        ])


# ===================================================================
# MODEL — 3D DenseNet121 via MONAI
# ===================================================================

def build_model():
    """Build 3D DenseNet121 for brain MRI classification."""
    model = DenseNet3D(
        spatial_dims=CONFIG['spatial_dims'],
        in_channels=CONFIG['in_channels'],
        out_channels=len(CONFIG['class_names']),
        dropout_prob=0.3,
    )
    return model


# ===================================================================
# TRAINING LOOP
# ===================================================================

def train_brain_model():
    """Main training function for brain MRI classification."""
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("="*70)
    print("  SENTINEL BRAIN MRI TRAINING")
    print("  Classes: 10 brain pathologies (3D DenseNet121)")
    print("="*70)

    # Load data
    data_list = build_data_list(CONFIG['data_dir'])
    if len(data_list) == 0:
        print("No data found! Check CONFIG['data_dir']")
        return

    # Stratified split
    labels = [d['label'] for d in data_list]
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_list, labels, train_size=0.8, random_state=42, stratify=labels,
    )
    train_data, val_data = train_test_split(
        train_data, train_size=0.9, random_state=42,
        stratify=[d['label'] for d in train_data],
    )
    print(f"\n  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # MONAI datasets
    train_ds = MonaiDataset(data=train_data, transform=get_transforms(is_train=True))
    val_ds = MonaiDataset(data=val_data, transform=get_transforms(is_train=False))
    test_ds = MonaiDataset(data=test_data, transform=get_transforms(is_train=False))

    train_loader = MonaiLoader(train_ds, batch_size=CONFIG['batch_size'],
                                shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = MonaiLoader(val_ds, batch_size=CONFIG['batch_size'],
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = MonaiLoader(test_ds, batch_size=CONFIG['batch_size'],
                               num_workers=CONFIG['num_workers'], pin_memory=True)

    # Model
    print("\n[MODEL] Building 3D DenseNet121...")
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Loss — weighted cross-entropy for class imbalance
    class_counts = np.bincount([d['label'] for d in train_data], minlength=len(CONFIG['class_names']))
    class_weights = torch.tensor(
        [1.0 / max(c, 1) for c in class_counts], dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"  Class weights: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
    )

    # LR scheduler
    total_steps = CONFIG['epochs'] * len(train_loader)
    warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    # Training loop
    best_auc = 0
    best_recall = 0
    history = []
    patience = 7
    patience_counter = 0

    print(f"\n[TRAIN] Starting {CONFIG['epochs']} epochs...")

    for epoch in range(1, CONFIG['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'E{epoch} Train')

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with autocast(enabled=CONFIG['mixed_precision']):
                outputs = model(images)
                loss = criterion(outputs, labels) / CONFIG['gradient_accumulation']

            scaler.scale(loss).backward()

            if (step + 1) % CONFIG['gradient_accumulation'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * CONFIG['gradient_accumulation']
            pbar.set_postfix({'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}'})

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val', leave=False):
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                with autocast(enabled=CONFIG['mixed_precision']):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        preds = np.concatenate(all_preds)
        labs = np.concatenate(all_labels)
        pred_classes = preds.argmax(axis=1)

        recall = recall_score(labs, pred_classes, average='macro', zero_division=0)
        prec = precision_score(labs, pred_classes, average='macro', zero_division=0)
        f1 = f1_score(labs, pred_classes, average='macro', zero_division=0)

        # Multi-class AUC
        try:
            auc = roc_auc_score(
                np.eye(len(CONFIG['class_names']))[labs],
                preds, average='macro', multi_class='ovr',
            )
        except ValueError:
            auc = 0.0

        lr_current = optimizer.param_groups[0]['lr']
        print(f"  E{epoch:>2d} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"recall={recall:.3f} prec={prec:.3f} auc={auc:.3f} | lr={lr_current:.5f}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'recall': recall, 'precision': prec, 'f1': f1, 'auc': auc,
        })

        # Save best
        improved = auc > best_auc or recall > best_recall
        if auc > best_auc:
            best_auc = auc
        if recall > best_recall:
            best_recall = recall

        if improved:
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CONFIG['class_names'],
                'recall': recall, 'auc': auc,
                'config': CONFIG,
            }, f"{CONFIG['output_dir']}/best_brain_model.pt")
            print(f"    → New best! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop")
                break

    # Test evaluation
    print("\n[TEST] Final evaluation...")
    ckpt = torch.load(f"{CONFIG['output_dir']}/best_brain_model.pt", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            with autocast(enabled=CONFIG['mixed_precision']):
                outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labs = np.concatenate(all_labels)
    pred_classes = preds.argmax(axis=1)

    test_recall = recall_score(labs, pred_classes, average='macro', zero_division=0)
    test_prec = precision_score(labs, pred_classes, average='macro', zero_division=0)
    test_f1 = f1_score(labs, pred_classes, average='macro', zero_division=0)

    try:
        test_auc = roc_auc_score(
            np.eye(len(CONFIG['class_names']))[labs],
            preds, average='macro', multi_class='ovr',
        )
    except ValueError:
        test_auc = 0.0

    # Per-class
    print("\n" + "="*70)
    print("  FINAL BRAIN MRI RESULTS")
    print("="*70)
    print(f"  Macro Recall:    {test_recall:.4f}")
    print(f"  Macro Precision: {test_prec:.4f}")
    print(f"  Macro F1:        {test_f1:.4f}")
    print(f"  Macro AUC-ROC:   {test_auc:.4f}")
    print("\n  Per-class recall:")

    per_class_recalls = {}
    for i, name in enumerate(CONFIG['class_names']):
        class_mask = labs == i
        if class_mask.sum() > 0:
            class_recall = (pred_classes[class_mask] == i).sum() / class_mask.sum()
            per_class_recalls[name] = float(class_recall)
            print(f"    {name:25s}: {class_recall:.3f}")

    # Save final results
    results = {
        'overall': {
            'recall': float(test_recall),
            'precision': float(test_prec),
            'f1': float(test_f1),
            'auc_roc': float(test_auc),
        },
        'per_class': per_class_recalls,
        'history': history,
    }

    with open(f"{CONFIG['output_dir']}/brain_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {CONFIG['output_dir']}/")
    print("="*70)


if __name__ == '__main__':
    if not HAS_MONAI:
        print("Install MONAI first: pip install 'monai[nibabel]'")
        sys.exit(1)

    train_brain_model()
