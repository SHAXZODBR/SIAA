"""
================================================================================
  SENTINEL MEDICAL AI — BRAIN TUMOR CLASSIFICATION (PHASE 1.2)
================================================================================
  Train on Kaggle for FREE — Brain MRI Tumor Detection

  WHAT IT DETECTS:
  - Glioma (aggressive brain tumor)
  - Meningioma (benign brain tumor)
  - Pituitary tumor (gland tumor)
  - No Tumor (healthy brain)

  DATASETS TO ADD ON KAGGLE:
  1. "Brain Tumor Classification (MRI)" by Sartaj Bhuvaji
     Search: brain-tumor-classification-mri
  2. (Optional) "Brain MRI Images for Brain Tumor Detection" by Navoneel
     Search: brain-mri-images-for-brain-tumor-detection

  GPU: T4 x2
  Expected training time: 3-4 hours (small dataset)
  Expected accuracy: 95%+ on tumor detection

  HOW TO USE:
  1. Create new Kaggle notebook
  2. Settings > GPU T4 x2
  3. Add dataset: brain-tumor-classification-mri
  4. Paste this entire script
  5. Save Version > Save & Run All (Commit)
  6. Download best_brain_tumor_model.pt after ~4 hours
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
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as T
import cv2
from PIL import Image
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===================================================================
# CONFIG
# ===================================================================

CONFIG = {
    # Dataset paths (auto-detected)
    'data_root': '/kaggle/input',
    'output_dir': '/kaggle/working',

    # Brain tumor classes (alphabetical for consistency)
    'class_names': [
        'glioma_tumor',
        'meningioma_tumor',
        'no_tumor',
        'pituitary_tumor',
    ],

    # Training
    'image_size': 224,              # Smaller than chest — brain MRI is small
    'batch_size': 64,                # Can fit more, dataset small
    'epochs': 30,                    # Small dataset needs more epochs
    'learning_rate': 1e-4,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'label_smoothing': 0.1,          # Higher for small dataset
    'dropout': 0.4,                  # More regularization

    # EMA
    'ema_decay': 0.999,

    # Hardware
    'mixed_precision': True,
    'num_workers': 4,
    'seed': 42,

    # Targets
    'target_recall': 0.90,           # Brain tumor = higher stakes
    'target_auc': 0.95,
}


# ===================================================================
# UTILS
# ===================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpus = torch.cuda.device_count()
        print(f"GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} — {props.total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU! Enable T4 x2 in Settings.")
    return device


# ===================================================================
# DATASET DISCOVERY — Auto-detect Kaggle dataset structure
# ===================================================================

def discover_dataset():
    """Find brain tumor dataset folders on Kaggle."""
    print("[DATASET] Searching for brain tumor dataset...")

    # Common structures for Kaggle brain tumor datasets:
    # Training/glioma_tumor/*.jpg
    # Training/meningioma_tumor/*.jpg
    # etc.

    patterns = [
        '/kaggle/input/**/Training/glioma_tumor',
        '/kaggle/input/**/Training/glioma',
        '/kaggle/input/**/train/glioma_tumor',
        '/kaggle/input/**/train/glioma',
        '/kaggle/input/**/glioma_tumor',
        '/kaggle/input/**/glioma',
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            glioma_dir = matches[0]
            # Get the parent (contains all classes)
            classes_dir = os.path.dirname(glioma_dir)
            print(f"  Found dataset root: {classes_dir}")
            return classes_dir

    # If not found, list what IS available
    print("  [WARN] Brain tumor dataset structure not found automatically.")
    print("  Available in /kaggle/input/:")
    for item in os.listdir('/kaggle/input'):
        print(f"    - {item}")
    return None


def build_file_list(classes_dir, class_names):
    """Build (image_path, label_idx) list from class folders."""
    entries = []

    # Also check for Testing/ folder as separate data
    parent_dir = os.path.dirname(classes_dir)
    dataset_dirs = [classes_dir]

    # Look for alternate split (Testing/ typically)
    for sub in ['Testing', 'testing', 'test', 'Test']:
        testing = os.path.join(parent_dir, sub)
        if os.path.isdir(testing):
            dataset_dirs.append(testing)
            print(f"  Also including: {testing}")

    for data_dir in dataset_dirs:
        for class_idx, class_name in enumerate(class_names):
            # Try various name formats
            candidates = [
                class_name,                        # glioma_tumor
                class_name.replace('_tumor', ''),  # glioma
                class_name.replace('_', ''),       # gliomatumor
                class_name.title(),                # Glioma_Tumor
            ]
            for candidate in candidates:
                class_dir = os.path.join(data_dir, candidate)
                if os.path.isdir(class_dir):
                    imgs = [f for f in os.listdir(class_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    for img in imgs:
                        entries.append((os.path.join(class_dir, img), class_idx))
                    break

    return entries


# ===================================================================
# DATASET
# ===================================================================

class BrainMRIDataset(Dataset):
    """Brain MRI dataset — handles jpg/png/bmp images."""

    def __init__(self, entries, is_train=True, image_size=224):
        self.entries = entries
        self.is_train = is_train
        self.image_size = image_size

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                # No vertical flip — brain has top-bottom orientation
                T.RandomRotation(15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label_idx = self.entries[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            img = torch.zeros(3, self.image_size, self.image_size)

        label = torch.tensor(label_idx, dtype=torch.long)
        return img, label


# ===================================================================
# MODEL — DenseNet121 adapted for 4-class
# ===================================================================

class BrainTumorNet(nn.Module):
    """DenseNet121-based brain tumor classifier."""

    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        self.backbone = models.densenet121(weights=weights)

        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = F.relu(feat, inplace=True)
        return self.head(feat)


# ===================================================================
# TRAIN / EVAL
# ===================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'E{epoch} Train')
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            loss = criterion(out, labels)

        total_loss += loss.item()
        probs = F.softmax(out, dim=1).cpu().numpy()
        _, preds = torch.max(out, 1)

        all_probs.append(probs)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = (all_preds == all_labels).mean()
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(
            np.eye(len(CONFIG['class_names']))[all_labels],
            all_probs, average='macro', multi_class='ovr',
        )
    except ValueError:
        auc = 0.0

    return total_loss / len(loader), all_preds, all_labels, all_probs, accuracy, recall, precision, f1, auc


# ===================================================================
# MAIN
# ===================================================================

def main():
    seed_everything(CONFIG['seed'])
    device = get_device()

    print("="*70)
    print("  SENTINEL BRAIN TUMOR CLASSIFICATION")
    print("  4 classes: Glioma, Meningioma, Pituitary, No Tumor")
    print("="*70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 1. Find dataset
    classes_dir = discover_dataset()
    if classes_dir is None:
        print("\n[FATAL] Dataset not found!")
        print("\nTo fix: Click + Add Data in right sidebar and search:")
        print("  'brain-tumor-classification-mri'")
        return

    # 2. Build entries
    print(f"\n[1/5] Loading images from {classes_dir}...")
    entries = build_file_list(classes_dir, CONFIG['class_names'])

    if len(entries) == 0:
        print("[FATAL] No images found in class folders!")
        return

    # Class distribution
    counts = {}
    for _, label_idx in entries:
        counts[CONFIG['class_names'][label_idx]] = counts.get(CONFIG['class_names'][label_idx], 0) + 1

    print(f"\n  Total images: {len(entries)}")
    print(f"  Class distribution:")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:25s}: {cnt}")

    # 3. Split 80/10/10 stratified
    print(f"\n[2/5] Splitting 80/10/10...")
    labels_only = [e[1] for e in entries]
    train_entries, temp = train_test_split(
        entries, train_size=0.8, random_state=42, stratify=labels_only,
    )
    temp_labels = [e[1] for e in temp]
    val_entries, test_entries = train_test_split(
        temp, train_size=0.5, random_state=42, stratify=temp_labels,
    )
    print(f"  Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")

    # 4. DataLoaders
    print(f"\n[3/5] Creating DataLoaders...")
    train_ds = BrainMRIDataset(train_entries, is_train=True, image_size=CONFIG['image_size'])
    val_ds = BrainMRIDataset(val_entries, is_train=False, image_size=CONFIG['image_size'])
    test_ds = BrainMRIDataset(test_entries, is_train=False, image_size=CONFIG['image_size'])

    # Weighted sampler for class balance
    class_counts = np.zeros(len(CONFIG['class_names']))
    for _, label_idx in train_entries:
        class_counts[label_idx] += 1
    sample_weights = [1.0 / class_counts[label_idx] for _, label_idx in train_entries]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                               num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)

    # 5. Model
    print(f"\n[4/5] Building model...")
    model = BrainTumorNet(num_classes=len(CONFIG['class_names']), dropout=CONFIG['dropout']).to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BrainTumorNet: {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])

    total_steps = CONFIG['epochs'] * len(train_loader)
    warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(CONFIG['min_lr'] / CONFIG['learning_rate'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    # Training
    print(f"\n[5/5] Training {CONFIG['epochs']} epochs...")
    print("="*70)

    best_auc = 0
    best_recall = 0
    history = []
    t_start = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        e_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)

        for _ in range(len(train_loader)):
            scheduler.step()

        val_loss, val_preds, val_labs, val_probs, val_acc, val_rec, val_prec, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - e_start
        lr = optimizer.param_groups[0]['lr']

        print(f"  E{epoch:>2d} | loss={train_loss:.3f} acc={train_acc:.3f} | "
              f"val_acc={val_acc:.3f} rec={val_rec:.3f} f1={val_f1:.3f} auc={val_auc:.3f} | "
              f"{elapsed:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_recall': val_rec,
            'val_f1': val_f1, 'val_auc': val_auc, 'lr': lr,
        })

        improved = val_auc > best_auc or val_rec > best_recall
        if val_auc > best_auc:
            best_auc = val_auc
        if val_rec > best_recall:
            best_recall = val_rec

        if improved:
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'class_names': CONFIG['class_names'],
                'recall': val_rec, 'auc': val_auc, 'accuracy': val_acc,
                'config': CONFIG,
            }, f"{CONFIG['output_dir']}/best_brain_tumor_model.pt")
            print(f"    → New best! Saved.")

    total_time = time.time() - t_start
    print(f"\n  Training time: {total_time/60:.1f} minutes")

    # Test evaluation
    print(f"\n[TEST] Final evaluation...")
    ckpt = torch.load(f"{CONFIG['output_dir']}/best_brain_tumor_model.pt",
                      weights_only=False, map_location=device)
    base_model = BrainTumorNet(num_classes=len(CONFIG['class_names']), dropout=0).to(device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model.eval()

    test_loss, test_preds, test_labs, test_probs, test_acc, test_rec, test_prec, test_f1, test_auc = evaluate(
        base_model, test_loader, criterion, device
    )

    # Per-class metrics
    print("\n" + "="*70)
    print("  FINAL BRAIN TUMOR RESULTS")
    print("="*70)
    print(f"  Accuracy:        {test_acc:.4f}")
    print(f"  Macro Recall:    {test_rec:.4f}  {'PASS' if test_rec >= 0.90 else 'CHECK'}")
    print(f"  Macro Precision: {test_prec:.4f}")
    print(f"  Macro F1:        {test_f1:.4f}")
    print(f"  Macro AUC-ROC:   {test_auc:.4f}")
    print("\n  Per-class results:")

    per_class = {}
    for i, name in enumerate(CONFIG['class_names']):
        mask = test_labs == i
        if mask.sum() > 0:
            class_rec = (test_preds[mask] == i).sum() / mask.sum()
            per_class[name] = float(class_rec)
            print(f"    {name:25s}: recall={class_rec:.3f}  ({int(mask.sum())} samples)")

    # Confusion matrix
    try:
        cm = confusion_matrix(test_labs, test_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CONFIG['class_names'], yticklabels=CONFIG['class_names'], ax=ax)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_title('Brain Tumor Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/brain_confusion_matrix.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Training curves
    try:
        import pandas as pd
        df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df['train_loss'], label='train'); axes[0].plot(df['val_loss'], label='val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(df['val_acc'], color='blue', label='acc'); axes[1].plot(df['val_recall'], color='red', label='recall')
        axes[1].axhline(0.90, color='gray', linestyle='--')
        axes[1].set_title('Accuracy & Recall'); axes[1].legend()
        axes[2].plot(df['val_auc'], color='green'); axes[2].set_title('AUC-ROC')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/brain_training_curves.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  Curves plot failed: {e}")

    # Save final results
    results = {
        'overall': {
            'accuracy': float(test_acc),
            'recall': float(test_rec),
            'precision': float(test_prec),
            'f1': float(test_f1),
            'auc_roc': float(test_auc),
        },
        'per_class': per_class,
        'training_time_min': total_time / 60,
        'total_samples': len(entries),
    }
    with open(f"{CONFIG['output_dir']}/brain_tumor_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved:")
    print(f"    - best_brain_tumor_model.pt  (download this)")
    print(f"    - brain_tumor_results.json")
    print(f"    - brain_confusion_matrix.png")
    print(f"    - brain_training_curves.png")
    print("="*70)


if __name__ == '__main__':
    main()
