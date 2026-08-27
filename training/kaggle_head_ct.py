"""
================================================================================
  SENTINEL MEDICAL AI — HEAD CT HEMORRHAGE DETECTION (PHASE 1.3)
================================================================================
  CT Scan classification for brain hemorrhage — LIFE-SAVING critical AI.

  WHAT IT DETECTS (6 hemorrhage types):
  - Epidural hemorrhage      (between skull and dura mater)
  - Subdural hemorrhage      (under dura mater)
  - Intraventricular         (in ventricles)
  - Intraparenchymal         (in brain tissue)
  - Subarachnoid             (in subarachnoid space)
  - Any hemorrhage           (binary — yes/no)

  DATASET TO ADD ON KAGGLE:
  - "RSNA Intracranial Hemorrhage Detection" (RSNA competition)
  - OR "head-ct-hemorrhage" datasets (various mirrors)

  NOTE: RSNA Intracranial Hemorrhage is 470 GB (huge!).
  For first training, we use a subset. After, download full for production.

  ALTERNATIVE EASIER DATASETS:
  - "Head CT - hemorrhage" by Felipe Kitamura (200 images — fast test)
  - "CT Brain Stroke Classification" (various)

  GPU: T4 x2 required
  Expected training time: 6-8 hours (with subset)
  Expected accuracy: 90%+ AUC on binary hemorrhage detection
================================================================================
"""

import os
import sys
import json
import time
import math
import random
import warnings
import glob
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


CONFIG = {
    'output_dir': '/kaggle/working',

    # 6 hemorrhage types (multi-label — can have multiple at once)
    'class_names': [
        'epidural',
        'intraparenchymal',
        'intraventricular',
        'subarachnoid',
        'subdural',
        'any',  # binary — any hemorrhage
    ],

    'image_size': 256,
    'batch_size': 64,
    'epochs': 15,
    'learning_rate': 2e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 2,
    'label_smoothing': 0.05,
    'dropout': 0.3,

    # Focal Loss (severe class imbalance in hemorrhage data)
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,  # Most samples are negative

    # Sample limit (RSNA is 470 GB — use subset for first training)
    'max_samples': 100000,  # ~100K slices should give 90%+ AUC

    'mixed_precision': True,
    'num_workers': 4,
    'seed': 42,

    'target_auc': 0.90,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
    else:
        device = torch.device('cpu')
    return device


# ===================================================================
# DATASET DISCOVERY
# ===================================================================

def discover_dataset():
    """Find RSNA intracranial hemorrhage dataset."""
    print("[DATASET] Searching for head CT dataset...")

    # Try RSNA intracranial hemorrhage format
    csv_patterns = [
        '/kaggle/input/**/stage_2_train.csv',
        '/kaggle/input/**/stage_1_train.csv',
        '/kaggle/input/**/train.csv',
        '/kaggle/input/**/labels.csv',
    ]

    csv_path = None
    for pattern in csv_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            csv_path = matches[0]
            print(f"  Found CSV: {csv_path}")
            break

    # Find images (DICOM or PNG)
    img_patterns = [
        '/kaggle/input/**/stage_2_train_images',
        '/kaggle/input/**/stage_1_train_images',
        '/kaggle/input/**/train_images',
        '/kaggle/input/**/images',
        '/kaggle/input/**/*.dcm',  # direct DICOM files
        '/kaggle/input/**/*.png',  # direct PNG files
    ]

    image_dir = None
    for pattern in img_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            if os.path.isdir(matches[0]):
                image_dir = matches[0]
            else:
                image_dir = os.path.dirname(matches[0])
            print(f"  Found images: {image_dir}")
            break

    return csv_path, image_dir


def load_rsna_hemorrhage_labels(csv_path):
    """Load RSNA intracranial hemorrhage labels.

    RSNA CSV format: ID = {image_id}_{type}, Label = 0/1
    e.g. ID_000012eaf_subdural, 1
    """
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from CSV")

    # Parse ID into image_id and hemorrhage type
    df['ImageID'] = df['ID'].str.rsplit('_', n=1).str[0]
    df['Type'] = df['ID'].str.rsplit('_', n=1).str[1]

    # Pivot: one row per image, columns per type
    labels_df = df.pivot_table(
        index='ImageID', columns='Type', values='Label', aggfunc='first',
    ).fillna(0).astype(int)

    # Rename columns to match CONFIG
    labels_df = labels_df.reindex(columns=CONFIG['class_names'], fill_value=0)

    # Build entries list
    entries = []
    for img_id, row in labels_df.iterrows():
        label_vec = row.tolist()
        entries.append((img_id, label_vec))

    print(f"  Unique images: {len(entries)}")
    return entries


# ===================================================================
# DATASET
# ===================================================================

class HeadCTDataset(Dataset):
    """Head CT dataset — supports DICOM and PNG formats."""

    def __init__(self, entries, image_dir, is_train=True, image_size=256):
        self.entries = entries
        self.image_dir = image_dir
        self.is_train = is_train
        self.image_size = image_size

        # Detect file format
        self.file_ext = None
        for f in os.listdir(image_dir)[:5]:
            if f.endswith('.dcm'):
                self.file_ext = '.dcm'
                break
            elif f.endswith('.png'):
                self.file_ext = '.png'
                break
            elif f.endswith('.jpg') or f.endswith('.jpeg'):
                self.file_ext = '.jpg'
                break
        self.file_ext = self.file_ext or '.dcm'

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.entries)

    def _load_dicom(self, path):
        """Load DICOM with CT windowing for brain/bleed visualization."""
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        img = ds.pixel_array.astype(np.float32)

        # Apply rescale
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = img * slope + intercept

        # Brain window (center=40, width=80) + Blood window (center=80, width=200)
        # Combine into RGB channels for multi-window approach
        brain = self._window(img, 40, 80)
        blood = self._window(img, 80, 200)
        bone = self._window(img, 600, 2800)

        rgb = np.stack([brain, blood, bone], axis=-1)
        return Image.fromarray((rgb * 255).astype(np.uint8))

    def _window(self, img, center, width):
        """CT windowing."""
        lo = center - width / 2
        hi = center + width / 2
        windowed = np.clip(img, lo, hi)
        return (windowed - lo) / (hi - lo)

    def __getitem__(self, idx):
        img_id, label_vec = self.entries[idx]
        label = torch.tensor(label_vec, dtype=torch.float32)

        try:
            path = os.path.join(self.image_dir, f'{img_id}{self.file_ext}')
            if self.file_ext == '.dcm':
                img = self._load_dicom(path)
            else:
                img = Image.open(path).convert('RGB')

            img = self.transform(img)
            return img, label
        except Exception as e:
            return torch.zeros(3, self.image_size, self.image_size), label


# ===================================================================
# MODEL
# ===================================================================

class HeadCTNet(nn.Module):
    """Head CT classifier — DenseNet121 backbone with multi-label output."""

    def __init__(self, num_classes=6, dropout=0.3):
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
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = F.relu(feat, inplace=True)
        return self.head(feat)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ===================================================================
# TRAIN / EVAL
# ===================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'E{epoch} Train')
    for images, labels in pbar:
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
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            loss = criterion(out, labels)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labs = np.concatenate(all_labels)
    binary = (preds >= 0.5).astype(int)

    recall = recall_score(labs, binary, average='macro', zero_division=0)
    precision = precision_score(labs, binary, average='macro', zero_division=0)
    f1 = f1_score(labs, binary, average='macro', zero_division=0)

    aucs = []
    for i in range(labs.shape[1]):
        if len(np.unique(labs[:, i])) > 1:
            aucs.append(roc_auc_score(labs[:, i], preds[:, i]))
    auc = np.mean(aucs) if aucs else 0.0

    return total_loss / len(loader), preds, labs, recall, precision, f1, auc


# ===================================================================
# MAIN
# ===================================================================

def main():
    seed_everything(CONFIG['seed'])
    device = get_device()

    print("="*70)
    print("  SENTINEL HEAD CT HEMORRHAGE DETECTION")
    print("  6 classes (multi-label): 5 hemorrhage types + any")
    print("="*70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    csv_path, image_dir = discover_dataset()
    if not csv_path or not image_dir:
        print("\n[FATAL] Dataset not found!")
        print("\nAdd this dataset in Kaggle sidebar:")
        print("  'rsna-intracranial-hemorrhage-detection'")
        print("\nOr any similar head CT hemorrhage dataset")
        return

    print(f"\n[1/5] Loading labels...")
    entries = load_rsna_hemorrhage_labels(csv_path)

    # Sample to max_samples if dataset too big
    if len(entries) > CONFIG['max_samples']:
        print(f"  Limiting to {CONFIG['max_samples']:,} samples (from {len(entries):,})")
        random.shuffle(entries)
        entries = entries[:CONFIG['max_samples']]

    # Class distribution
    print(f"\n  Total entries: {len(entries)}")
    class_counts = np.zeros(len(CONFIG['class_names']))
    for _, vec in entries:
        class_counts += np.array(vec)
    print(f"  Class distribution:")
    for i, name in enumerate(CONFIG['class_names']):
        pct = class_counts[i] / len(entries) * 100
        print(f"    {name:20s}: {int(class_counts[i]):>6d} ({pct:.1f}%)")

    # Split
    print(f"\n[2/5] Splitting 80/10/10...")
    # Stratify by 'any' hemorrhage (last class)
    stratify = [e[1][-1] for e in entries]
    train_entries, temp = train_test_split(entries, train_size=0.8, random_state=42, stratify=stratify)
    temp_strat = [e[1][-1] for e in temp]
    val_entries, test_entries = train_test_split(temp, train_size=0.5, random_state=42, stratify=temp_strat)
    print(f"  Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")

    # DataLoaders
    print(f"\n[3/5] Creating DataLoaders...")
    train_ds = HeadCTDataset(train_entries, image_dir, is_train=True, image_size=CONFIG['image_size'])
    val_ds = HeadCTDataset(val_entries, image_dir, is_train=False, image_size=CONFIG['image_size'])
    test_ds = HeadCTDataset(test_entries, image_dir, is_train=False, image_size=CONFIG['image_size'])

    # Weighted sampling for rare positive cases
    weights = [5.0 if e[1][-1] == 1 else 1.0 for e in train_entries]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                               num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)

    # Model
    print(f"\n[4/5] Building model...")
    model = HeadCTNet(num_classes=len(CONFIG['class_names']), dropout=CONFIG['dropout']).to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  HeadCTNet: {n_params:,} parameters")

    criterion = FocalLoss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha'],
                          label_smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])

    total_steps = CONFIG['epochs'] * len(train_loader)
    warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(CONFIG['min_lr']/CONFIG['learning_rate'], 0.5*(1+math.cos(math.pi*progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    print(f"\n[5/5] Training {CONFIG['epochs']} epochs...")
    print("="*70)

    best_auc = 0
    history = []
    t_start = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        e_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        for _ in range(len(train_loader)):
            scheduler.step()

        val_loss, _, _, recall, prec, f1, auc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - e_start

        print(f"  E{epoch:>2d} | loss={train_loss:.3f} vl={val_loss:.3f} | "
              f"rec={recall:.3f} prec={prec:.3f} f1={f1:.3f} auc={auc:.3f} | {elapsed:.0f}s")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'recall': recall, 'f1': f1, 'auc': auc})

        if auc > best_auc:
            best_auc = auc
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'class_names': CONFIG['class_names'],
                'auc': auc, 'recall': recall,
                'config': CONFIG,
            }, f"{CONFIG['output_dir']}/best_head_ct_model.pt")
            print(f"    → New best! Saved.")

    total_time = time.time() - t_start

    # Final test
    print(f"\n[TEST] Final evaluation...")
    ckpt = torch.load(f"{CONFIG['output_dir']}/best_head_ct_model.pt",
                      weights_only=False, map_location=device)
    base_model = HeadCTNet(num_classes=len(CONFIG['class_names']), dropout=0).to(device)
    base_model.load_state_dict(ckpt['model_state_dict'])

    _, test_preds, test_labs, test_rec, test_prec, test_f1, test_auc = evaluate(
        base_model, test_loader, criterion, device
    )

    print("\n" + "="*70)
    print("  FINAL HEAD CT RESULTS")
    print("="*70)
    print(f"  Macro Recall:    {test_rec:.4f}")
    print(f"  Macro Precision: {test_prec:.4f}")
    print(f"  Macro F1:        {test_f1:.4f}")
    print(f"  Macro AUC-ROC:   {test_auc:.4f}  {'PASS' if test_auc >= 0.90 else 'NEEDS MORE'}")
    print(f"\n  Per-class AUC:")
    for i, name in enumerate(CONFIG['class_names']):
        if len(np.unique(test_labs[:, i])) > 1:
            auc_i = roc_auc_score(test_labs[:, i], test_preds[:, i])
            print(f"    {name:20s}: AUC={auc_i:.4f}")

    results = {
        'overall': {
            'recall': float(test_rec), 'precision': float(test_prec),
            'f1': float(test_f1), 'auc_roc': float(test_auc),
        },
        'training_time_min': total_time / 60,
        'total_samples': len(entries),
    }
    with open(f"{CONFIG['output_dir']}/head_ct_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved:")
    print(f"    - best_head_ct_model.pt")
    print(f"    - head_ct_results.json")
    print("="*70)


if __name__ == '__main__':
    main()
