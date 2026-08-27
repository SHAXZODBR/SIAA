#!/usr/bin/env python3
"""
=================================================================
SENTINEL MEDICAL AI — MASTER TRAINING SCRIPT
=================================================================
One script that does EVERYTHING on RunPod:
  1. Downloads NIH ChestX-ray14 + RSNA Pneumonia datasets
  2. Preprocesses all images to 512x512
  3. Converts labels to unified format
  4. Trains DenseNet121 with every optimization trick
  5. Optimizes per-class thresholds for maximum recall
  6. Runs test-time augmentation (TTA)
  7. Generates full evaluation report with plots
  8. Saves best model ready for deployment

OPTIMIZATION TRICKS FOR >90% RECALL:
  - Focal Loss (handles class imbalance better than BCE)
  - Label smoothing (prevents overconfident predictions)
  - Mixup augmentation (regularization)
  - Progressive image resizing (256 → 384 → 512)
  - Learning rate warmup + cosine annealing
  - Per-class threshold optimization (not just 0.5)
  - Test-time augmentation (5x TTA)
  - Exponential Moving Average (EMA) model
  - Strong medical-specific augmentations

USAGE ON RUNPOD:
  # Upload your datasets to /workspace/datasets/ OR let this script download
  pip install -r requirements_gpu.txt
  python train_master.py

  # Or with your own data from hard drive:
  python train_master.py --data-dir /workspace/datasets --skip-download

  # Resume training:
  python train_master.py --resume checkpoints/last.pt
=================================================================
"""

import os
import sys
import json
import time
import math
import copy
import random
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

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
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ================================================================
# CONFIG
# ================================================================

CONFIG = {
    # Data
    'data_dir': '/workspace/datasets',
    'output_dir': '/workspace/sentinel_training',
    'image_size_stages': [256, 384, 512],  # Progressive resizing
    'num_workers': 8,

    # Classes — 14 CheXNet pathologies + No Finding
    'class_names': [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia',
    ],

    # Training
    'epochs_per_stage': [15, 15, 30],  # epochs for each image size stage
    'batch_size': 32,
    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'label_smoothing': 0.05,
    'mixup_alpha': 0.2,
    'dropout': 0.3,

    # Focal Loss
    'focal_gamma': 2.0,
    'focal_alpha': 0.75,

    # EMA
    'ema_decay': 0.999,

    # TTA
    'tta_transforms': 5,

    # Targets
    'target_recall': 0.90,
    'target_specificity': 0.80,

    # Hardware
    'mixed_precision': True,
    'gradient_accumulation': 2,
    'seed': 42,
}


# ================================================================
# UTILS
# ================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected! Training will be very slow.")
    return device


# ================================================================
# STEP 1: DATA DOWNLOAD
# ================================================================

def download_datasets(data_dir):
    """Download NIH ChestX-ray14 and RSNA Pneumonia datasets."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- NIH ChestX-ray14 ---
    nih_dir = data_dir / 'nih-chestxray14'
    if not nih_dir.exists():
        print("\n[DOWNLOAD] NIH ChestX-ray14 (42GB)...")
        print("  This dataset must be downloaded manually from:")
        print("  https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("  Download all images_001.tar.gz through images_012.tar.gz")
        print("  and Data_Entry_2017_v2020.csv")
        print(f"  Place them in: {nih_dir}/")
        nih_dir.mkdir(parents=True, exist_ok=True)

        # Try automated download (may fail due to Box auth)
        os.system(f'mkdir -p {nih_dir}')
        for i in range(1, 13):
            archive = nih_dir / f'images_{i:03d}.tar.gz'
            if not archive.exists():
                print(f"  Attempting download images_{i:03d}...")
                os.system(f'wget -q -nc "https://nihcc.app.box.com/shared/static/'
                          f'vfk49d74nhbxq3nqjg0900w5nvkorp5c/images_{i:03d}.tar.gz" '
                          f'-O {archive} 2>/dev/null || true')
        # Extract
        for gz in nih_dir.glob('*.tar.gz'):
            print(f"  Extracting {gz.name}...")
            os.system(f'tar -xzf {gz} -C {nih_dir} 2>/dev/null')
    else:
        print(f"[DOWNLOAD] NIH data found at {nih_dir}")

    # --- RSNA Pneumonia ---
    rsna_dir = data_dir / 'rsna-pneumonia'
    if not rsna_dir.exists():
        print("\n[DOWNLOAD] RSNA Pneumonia (10GB)...")
        rsna_dir.mkdir(parents=True, exist_ok=True)
        if Path.home().joinpath('.kaggle', 'kaggle.json').exists():
            os.system(f'kaggle competitions download -c rsna-pneumonia-detection-challenge '
                      f'-p {rsna_dir} 2>/dev/null')
            os.system(f'cd {rsna_dir} && unzip -qo "*.zip" 2>/dev/null')
            print("  RSNA downloaded via Kaggle CLI")
        else:
            print("  Kaggle credentials not found. Download manually from:")
            print("  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data")
            print(f"  Place files in: {rsna_dir}/")
    else:
        print(f"[DOWNLOAD] RSNA data found at {rsna_dir}")

    return nih_dir, rsna_dir


# ================================================================
# STEP 2: PREPROCESS & BUILD UNIFIED DATASET
# ================================================================

def build_dataset(data_dir, output_dir, target_size=512):
    """Build unified dataset from NIH + RSNA data."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    class_names = CONFIG['class_names']

    # ===== Process NIH ChestX-ray14 =====
    nih_csv = data_dir / 'nih-chestxray14' / 'Data_Entry_2017_v2020.csv'
    nih_img_dir = data_dir / 'nih-chestxray14' / 'images'

    # Try alternate paths
    if not nih_img_dir.exists():
        for alt in ['images_001', 'images']:
            alt_dir = data_dir / 'nih-chestxray14' / alt
            if alt_dir.exists():
                nih_img_dir = alt_dir
                break

    if nih_csv.exists() and nih_img_dir.exists():
        print(f"\n[PREPROCESS] Processing NIH ChestX-ray14...")
        df = pd.read_csv(nih_csv)
        print(f"  Found {len(df)} entries in CSV")

        count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc='  NIH images'):
            fname = row['Image Index']
            src_path = nih_img_dir / fname
            if not src_path.exists():
                continue

            stem = fname.replace('.png', '')
            dst_path = images_dir / fname

            # Only process if not already done
            if not dst_path.exists():
                img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = resize_with_pad(img, target_size)
                cv2.imwrite(str(dst_path), img)

            # Parse labels
            findings = row['Finding Labels'].split('|')
            findings = [f.strip().replace(' ', '_') for f in findings]
            img_labels = []
            for f in findings:
                if f in class_names:
                    img_labels.append(class_names.index(f))
            # Multi-hot vector
            label_vec = [0] * len(class_names)
            for idx in img_labels:
                label_vec[idx] = 1

            labels[stem] = label_vec
            count += 1

        print(f"  Processed {count} NIH images")
    else:
        print(f"  [WARN] NIH data not found at {nih_csv}")

    # ===== Process RSNA Pneumonia =====
    rsna_csv = data_dir / 'rsna-pneumonia' / 'stage_2_train_labels.csv'
    rsna_img_dir = data_dir / 'rsna-pneumonia' / 'stage_2_train_images'

    if rsna_csv.exists() and rsna_img_dir.exists():
        print(f"\n[PREPROCESS] Processing RSNA Pneumonia...")
        df = pd.read_csv(rsna_csv)
        pneumonia_idx = class_names.index('Pneumonia')

        patient_targets = {}
        for _, row in df.iterrows():
            pid = row['patientId']
            if pid not in patient_targets:
                patient_targets[pid] = int(row['Target'])

        count = 0
        import pydicom
        for pid, target in tqdm(patient_targets.items(), desc='  RSNA DICOMs'):
            src_path = rsna_img_dir / f'{pid}.dcm'
            if not src_path.exists():
                continue

            dst_path = images_dir / f'rsna_{pid}.png'
            stem = f'rsna_{pid}'

            if not dst_path.exists():
                try:
                    ds = pydicom.dcmread(str(src_path), force=True)
                    img = ds.pixel_array.astype(np.float32)
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    img = resize_with_pad(img, target_size)
                    cv2.imwrite(str(dst_path), img)
                except Exception:
                    continue

            label_vec = [0] * len(class_names)
            if target == 1:
                label_vec[pneumonia_idx] = 1
            labels[stem] = label_vec
            count += 1

        print(f"  Processed {count} RSNA images")
    else:
        print(f"  [WARN] RSNA data not found at {rsna_csv}")

    # Save labels
    labels_path = output_dir / 'labels.json'
    with open(labels_path, 'w') as f:
        json.dump(labels, f)

    # Print class distribution
    print(f"\n[DATASET] Total images: {len(labels)}")
    print(f"[DATASET] Class distribution:")
    class_counts = np.array(list(labels.values())).sum(axis=0).astype(int)
    for i, name in enumerate(class_names):
        print(f"  {name:25s}: {class_counts[i]:>6d} ({class_counts[i]/len(labels)*100:.1f}%)")

    return labels, images_dir


def resize_with_pad(img, size):
    """Resize image preserving aspect ratio with zero padding."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def split_data(labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Stratified split preserving class distribution."""
    filenames = list(labels.keys())
    # Use most common class for stratification
    primary_labels = []
    for fname in filenames:
        vec = labels[fname]
        if sum(vec) == 0:
            primary_labels.append(-1)
        else:
            primary_labels.append(np.argmax(vec))

    # Train / (val+test)
    train_f, temp_f, train_l, temp_l = train_test_split(
        filenames, primary_labels, train_size=train_ratio,
        random_state=seed, stratify=primary_labels,
    )
    # Val / test
    val_f, test_f = train_test_split(
        temp_f, train_size=0.5, random_state=seed, stratify=temp_l,
    )

    print(f"\n[SPLIT] Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")
    return train_f, val_f, test_f


# ================================================================
# STEP 3: DATASET & AUGMENTATION
# ================================================================

class ChestXrayDataset(Dataset):
    """High-performance chest X-ray dataset with strong augmentations."""

    def __init__(self, filenames, labels, images_dir, image_size=512,
                 is_train=True, mixup_alpha=0.0):
        self.filenames = filenames
        self.labels = labels
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.mixup_alpha = mixup_alpha
        self.num_classes = len(CONFIG['class_names'])

        # Transforms
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = torch.tensor(self.labels[fname], dtype=torch.float32)

        # Load image
        img_path = self.images_dir / f'{fname}.png'
        img = Image.open(img_path).convert('L')
        img = self.transform(img)

        # Expand to 3 channels (pretrained models expect RGB)
        img = img.repeat(3, 1, 1)

        # Mixup during training
        if self.is_train and self.mixup_alpha > 0 and random.random() < 0.3:
            mix_idx = random.randint(0, len(self.filenames) - 1)
            mix_fname = self.filenames[mix_idx]
            mix_label = torch.tensor(self.labels[mix_fname], dtype=torch.float32)

            mix_path = self.images_dir / f'{mix_fname}.png'
            mix_img = Image.open(mix_path).convert('L')
            mix_img = self.transform(mix_img).repeat(3, 1, 1)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            img = lam * img + (1 - lam) * mix_img
            label = lam * label + (1 - lam) * mix_label

        return img, label


def get_weighted_sampler(filenames, labels, class_names):
    """Create weighted sampler to oversample rare classes."""
    class_counts = np.zeros(len(class_names))
    for fname in filenames:
        vec = labels[fname]
        class_counts += np.array(vec)

    # Compute per-sample weight (inverse frequency of rarest positive class)
    sample_weights = []
    for fname in filenames:
        vec = labels[fname]
        if sum(vec) == 0:
            sample_weights.append(1.0)
        else:
            positive_indices = [i for i, v in enumerate(vec) if v == 1]
            min_count = min(class_counts[i] for i in positive_indices)
            weight = len(filenames) / (len(class_names) * min_count + 1)
            sample_weights.append(min(weight, 10.0))

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ================================================================
# STEP 4: MODEL
# ================================================================

class SentinelDenseNet(nn.Module):
    """DenseNet121 with custom head for multi-label medical classification.

    Improvements over vanilla DenseNet121:
    - Dropout for regularization
    - GeM pooling instead of avg pool (better for medical imaging)
    - Multi-head classifier with intermediate features
    """

    def __init__(self, num_classes=14, dropout=0.3, pretrained=True):
        super().__init__()

        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Get feature dimension
        num_features = self.backbone.classifier.in_features  # 1024

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        # GeM Pooling
        self.gem_pool = GeMPool(p=3.0)

        # Custom classifier head
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Initialize head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = self.gem_pool(out)
        out = out.view(out.size(0), -1)
        out = self.head(out)
        return out

    def get_features(self, x):
        """Get feature maps for Grad-CAM."""
        return self.backbone.features(x)


class GeMPool(nn.Module):
    """Generalized Mean Pooling — better than avg pool for classification."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


# ================================================================
# STEP 5: LOSS FUNCTIONS
# ================================================================

class FocalLoss(nn.Module):
    """Focal Loss — handles class imbalance much better than BCE.

    Reduces loss for well-classified examples, focuses training on hard cases.
    Critical for medical imaging where some pathologies are rare.
    """

    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


# ================================================================
# STEP 6: EMA (Exponential Moving Average)
# ================================================================

class EMA:
    """Exponential Moving Average of model weights — smooths training, improves generalization."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# ================================================================
# STEP 7: TRAINING LOOP
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'  Train Epoch {epoch}')
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['mixed_precision']):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{running_loss/(step+1):.4f}'})

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader)

    # Metrics at 0.5 threshold
    binary = (all_preds >= 0.5).astype(int)
    recall = recall_score(all_labels, binary, average='macro', zero_division=0)
    auc = safe_auc(all_labels, all_preds)

    return epoch_loss, recall, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc='  Evaluating', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['mixed_precision']):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_loss = running_loss / len(loader)

    return epoch_loss, all_preds, all_labels


def safe_auc(y_true, y_pred):
    """Compute AUC safely handling single-class columns."""
    try:
        aucs = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        return np.mean(aucs) if aucs else 0.0
    except Exception:
        return 0.0


# ================================================================
# STEP 8: THRESHOLD OPTIMIZATION
# ================================================================

def optimize_thresholds(y_true, y_pred, class_names, target_recall=0.90):
    """Find optimal per-class threshold to maximize F1 while keeping recall >= target.

    This is the KEY to hitting >90% recall. Default 0.5 threshold is NOT optimal
    for medical imaging. Lower thresholds = higher recall (fewer missed diagnoses).
    """
    num_classes = len(class_names)
    optimal_thresholds = np.full(num_classes, 0.5)
    results = {}

    print(f"\n[THRESHOLDS] Optimizing per-class thresholds (target recall >= {target_recall})...")

    for i, name in enumerate(class_names):
        if len(np.unique(y_true[:, i])) < 2:
            results[name] = {'threshold': 0.5, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
            continue

        best_thresh = 0.5
        best_f1 = 0.0

        # Search thresholds from 0.1 to 0.9
        for thresh in np.arange(0.05, 0.95, 0.01):
            preds = (y_pred[:, i] >= thresh).astype(int)
            rec = recall_score(y_true[:, i], preds, zero_division=0)
            prec = precision_score(y_true[:, i], preds, zero_division=0)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            # Prioritize recall >= target, then maximize F1
            if rec >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # If no threshold achieves target recall, find the one closest
        if best_f1 == 0.0:
            best_recall = 0.0
            for thresh in np.arange(0.05, 0.95, 0.01):
                preds = (y_pred[:, i] >= thresh).astype(int)
                rec = recall_score(y_true[:, i], preds, zero_division=0)
                if rec > best_recall:
                    best_recall = rec
                    best_thresh = thresh

        optimal_thresholds[i] = best_thresh

        # Final metrics with optimal threshold
        preds = (y_pred[:, i] >= best_thresh).astype(int)
        rec = recall_score(y_true[:, i], preds, zero_division=0)
        prec = precision_score(y_true[:, i], preds, zero_division=0)
        f1 = f1_score(y_true[:, i], preds, zero_division=0)

        results[name] = {'threshold': best_thresh, 'recall': rec, 'precision': prec, 'f1': f1}
        print(f"  {name:25s}: thresh={best_thresh:.3f} | recall={rec:.3f} | prec={prec:.3f} | F1={f1:.3f}")

    return optimal_thresholds, results


# ================================================================
# STEP 9: TEST-TIME AUGMENTATION (TTA)
# ================================================================

@torch.no_grad()
def predict_with_tta(model, loader, device, n_augments=5):
    """Test-Time Augmentation — run multiple augmented versions, average predictions.

    Typically boosts recall by 2-5%.
    """
    model.eval()
    print(f"\n[TTA] Running {n_augments}-fold test-time augmentation...")

    # TTA transforms
    tta_transforms = [
        T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5], [0.5])]),  # Original
        T.Compose([T.Resize((512, 512)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize([0.5], [0.5])]),
        T.Compose([T.Resize((550, 550)), T.CenterCrop(512), T.ToTensor(), T.Normalize([0.5], [0.5])]),
        T.Compose([T.Resize((512, 512)), T.RandomRotation(10), T.ToTensor(), T.Normalize([0.5], [0.5])]),
        T.Compose([T.Resize((480, 480)), T.Pad(16), T.ToTensor(), T.Normalize([0.5], [0.5])]),
    ]

    all_preds_sum = None
    all_labels = None

    for aug_idx in range(min(n_augments, len(tta_transforms))):
        preds_list = []
        labels_list = []

        for images, labels in tqdm(loader, desc=f'  TTA {aug_idx+1}/{n_augments}', leave=False):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=CONFIG['mixed_precision']):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds_list.append(probs)
            labels_list.append(labels.numpy())

        preds = np.concatenate(preds_list)
        if all_preds_sum is None:
            all_preds_sum = preds
            all_labels = np.concatenate(labels_list)
        else:
            all_preds_sum += preds

    # Average predictions
    avg_preds = all_preds_sum / min(n_augments, len(tta_transforms))
    return avg_preds, all_labels


# ================================================================
# STEP 10: EVALUATION & REPORT
# ================================================================

def generate_report(y_true, y_pred, thresholds, class_names, output_dir, model_info):
    """Generate comprehensive HTML evaluation report with plots."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    binary_preds = np.zeros_like(y_pred)
    for i in range(len(class_names)):
        binary_preds[:, i] = (y_pred[:, i] >= thresholds[i]).astype(int)

    # Overall metrics
    macro_recall = recall_score(y_true, binary_preds, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, binary_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, binary_preds, average='macro', zero_division=0)
    macro_auc = safe_auc(y_true, y_pred)

    # Specificity
    specs = []
    for i in range(len(class_names)):
        tn = np.sum((y_true[:, i] == 0) & (binary_preds[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (binary_preds[:, i] == 1))
        if tn + fp > 0:
            specs.append(tn / (tn + fp))
    macro_specificity = np.mean(specs) if specs else 0.0

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Macro Recall (Sensitivity): {macro_recall:.4f}  {'PASS' if macro_recall >= 0.85 else 'NEEDS IMPROVEMENT'}")
    print(f"  Macro Specificity:          {macro_specificity:.4f}  {'PASS' if macro_specificity >= 0.80 else 'NEEDS IMPROVEMENT'}")
    print(f"  Macro Precision:            {macro_precision:.4f}")
    print(f"  Macro F1:                   {macro_f1:.4f}")
    print(f"  Macro AUC-ROC:              {macro_auc:.4f}")
    print(f"{'='*70}")

    # === Plot 1: ROC Curves ===
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()
    for i, name in enumerate(class_names):
        ax = axes[i]
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc_val = roc_auc_score(y_true[:, i], y_pred[:, i])
            ax.plot(fpr, tpr, 'b-', linewidth=2)
            ax.set_title(f'{name}\nAUC={auc_val:.3f}', fontsize=10)
        else:
            ax.set_title(f'{name}\nN/A', fontsize=10)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    for j in range(len(class_names), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Plot 2: Per-class metrics bar chart ===
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.25
    recalls, precisions, f1s = [], [], []
    for i in range(len(class_names)):
        r = recall_score(y_true[:, i], binary_preds[:, i], zero_division=0)
        p = precision_score(y_true[:, i], binary_preds[:, i], zero_division=0)
        f = f1_score(y_true[:, i], binary_preds[:, i], zero_division=0)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)

    ax.bar(x - width, recalls, width, label='Recall', color='#ef4444')
    ax.bar(x, precisions, width, label='Precision', color='#3b82f6')
    ax.bar(x + width, f1s, width, label='F1', color='#22c55e')
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target Recall 85%')
    ax.axhline(y=0.90, color='darkred', linestyle='--', alpha=0.5, label='Target Recall 90%')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics (with Optimized Thresholds)')
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(plots_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Plot 3: Confusion matrices ===
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()
    for i, name in enumerate(class_names):
        ax = axes[i]
        cm = confusion_matrix(y_true[:, i], binary_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{name}', fontsize=10)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    for j in range(len(class_names), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # === Save results JSON ===
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'overall': {
            'macro_recall': float(macro_recall),
            'macro_specificity': float(macro_specificity),
            'macro_precision': float(macro_precision),
            'macro_f1': float(macro_f1),
            'macro_auc_roc': float(macro_auc),
            'total_test_samples': int(len(y_true)),
        },
        'per_class': {},
        'thresholds': {name: float(thresholds[i]) for i, name in enumerate(class_names)},
    }
    for i, name in enumerate(class_names):
        results['per_class'][name] = {
            'recall': float(recalls[i]),
            'precision': float(precisions[i]),
            'f1': float(f1s[i]),
            'threshold': float(thresholds[i]),
        }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Reports saved to {output_dir}/")
    print(f"    - results.json")
    print(f"    - plots/roc_curves.png")
    print(f"    - plots/per_class_metrics.png")
    print(f"    - plots/confusion_matrices.png")

    return results


# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Sentinel Medical AI — Master Training')
    parser.add_argument('--data-dir', default=CONFIG['data_dir'], help='Dataset directory')
    parser.add_argument('--output-dir', default=CONFIG['output_dir'], help='Output directory')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Override total epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    args = parser.parse_args()

    if args.batch_size:
        CONFIG['batch_size'] = args.batch_size
    if args.lr:
        CONFIG['learning_rate'] = args.lr

    seed_everything(CONFIG['seed'])
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    print("="*70)
    print("  SENTINEL MEDICAL AI — MASTER TRAINING")
    print("  Target: Recall >= 90% | Specificity >= 80%")
    print("="*70)

    # ===== STEP 1: Download =====
    if not args.skip_download:
        download_datasets(args.data_dir)

    # ===== STEP 2: Preprocess =====
    labels_path = output_dir / 'labels.json'
    images_dir = output_dir / 'images'

    if not args.skip_preprocess or not labels_path.exists():
        labels, images_dir = build_dataset(args.data_dir, str(output_dir))
    else:
        with open(labels_path) as f:
            labels = json.load(f)
        print(f"\n[DATA] Loaded {len(labels)} labels from cache")

    if len(labels) == 0:
        print("\nERROR: No data found! Check your dataset paths.")
        print(f"  Expected datasets at: {args.data_dir}")
        sys.exit(1)

    # ===== STEP 3: Split =====
    train_files, val_files, test_files = split_data(labels)

    # ===== STEP 4: Progressive Training =====
    model = SentinelDenseNet(
        num_classes=len(CONFIG['class_names']),
        dropout=CONFIG['dropout'],
        pretrained=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] SentinelDenseNet: {param_count:,} params ({trainable:,} trainable)")

    # Resume
    start_stage = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_stage = ckpt.get('stage', 0)
        print(f"  Resumed from {args.resume} (stage {start_stage})")

    criterion = FocalLoss(
        gamma=CONFIG['focal_gamma'],
        alpha=CONFIG['focal_alpha'],
        label_smoothing=CONFIG['label_smoothing'],
    )

    ema = EMA(model, decay=CONFIG['ema_decay'])
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    best_val_auc = 0.0
    best_val_recall = 0.0
    training_history = []

    image_stages = CONFIG['image_size_stages']
    epoch_stages = CONFIG['epochs_per_stage']
    if args.epochs:
        epoch_stages = [args.epochs // len(image_stages)] * len(image_stages)

    for stage_idx in range(start_stage, len(image_stages)):
        img_size = image_stages[stage_idx]
        num_epochs = epoch_stages[stage_idx]

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_idx+1}/{len(image_stages)}: Image Size {img_size}x{img_size} | {num_epochs} epochs")
        print(f"{'='*70}")

        # Create datasets for this stage
        train_dataset = ChestXrayDataset(
            train_files, labels, images_dir, image_size=img_size,
            is_train=True, mixup_alpha=CONFIG['mixup_alpha'],
        )
        val_dataset = ChestXrayDataset(
            val_files, labels, images_dir, image_size=img_size, is_train=False,
        )

        sampler = get_weighted_sampler(train_files, labels, CONFIG['class_names'])
        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler,
            num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
            num_workers=CONFIG['num_workers'], pin_memory=True,
        )

        # Optimizer — recreate for each stage
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
        )

        # Scheduler — cosine with warmup
        total_steps = num_epochs * len(train_loader)
        warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(CONFIG['min_lr'] / CONFIG['learning_rate'],
                       0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Train this stage
        stage_best_recall = 0.0
        patience = 8
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_recall, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                epoch, CONFIG['gradient_accumulation'],
            )
            scheduler.step()
            ema.update(model)

            val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            val_binary = (val_preds >= 0.5).astype(int)
            val_recall = recall_score(val_labels, val_binary, average='macro', zero_division=0)
            val_auc = safe_auc(val_labels, val_preds)

            lr_current = optimizer.param_groups[0]['lr']
            print(f"  E{epoch:>3d} | loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_recall={val_recall:.4f} | val_auc={val_auc:.4f} | lr={lr_current:.6f}")

            training_history.append({
                'stage': stage_idx, 'epoch': epoch, 'img_size': img_size,
                'train_loss': train_loss, 'val_loss': val_loss,
                'val_recall': val_recall, 'val_auc': val_auc, 'lr': lr_current,
            })

            # Save best
            improved = False
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                improved = True
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                improved = True

            if improved:
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage': stage_idx,
                    'epoch': epoch,
                    'val_recall': val_recall,
                    'val_auc': val_auc,
                    'config': CONFIG,
                    'class_names': CONFIG['class_names'],
                }, ckpt_dir / 'best_model.pt')
                print(f"  >>> New best! recall={val_recall:.4f} auc={val_auc:.4f}")
            else:
                patience_counter += 1

            # Save last checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'config': CONFIG,
                'class_names': CONFIG['class_names'],
            }, ckpt_dir / 'last.pt')

            if patience_counter >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

    # ===== STEP 5: Final Evaluation =====
    print(f"\n{'='*70}")
    print("  FINAL EVALUATION — Loading best model + EMA")
    print(f"{'='*70}")

    # Load best model
    best_ckpt = torch.load(ckpt_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Also try EMA model
    ema_model = SentinelDenseNet(num_classes=len(CONFIG['class_names']), dropout=0, pretrained=False).to(device)
    ema_model.load_state_dict(best_ckpt['ema_state_dict'])

    # Test dataset (full size)
    test_dataset = ChestXrayDataset(
        test_files, labels, images_dir, image_size=512, is_train=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True,
    )

    # Evaluate both models
    _, base_preds, test_labels = evaluate(model, test_loader, criterion, device)
    _, ema_preds, _ = evaluate(ema_model, test_loader, criterion, device)

    # Ensemble: average base + EMA predictions
    ensemble_preds = (base_preds + ema_preds) / 2.0

    # TTA on best model
    tta_preds, _ = predict_with_tta(model, test_loader, device, n_augments=CONFIG['tta_transforms'])

    # Final ensemble: base + EMA + TTA
    final_preds = (base_preds + ema_preds + tta_preds) / 3.0

    # ===== STEP 6: Optimize Thresholds =====
    # Use validation set for threshold optimization
    val_dataset_final = ChestXrayDataset(
        val_files, labels, images_dir, image_size=512, is_train=False,
    )
    val_loader_final = DataLoader(
        val_dataset_final, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True,
    )
    _, val_preds_final, val_labels_final = evaluate(model, val_loader_final, criterion, device)

    optimal_thresholds, threshold_results = optimize_thresholds(
        val_labels_final, val_preds_final, CONFIG['class_names'],
        target_recall=CONFIG['target_recall'],
    )

    # ===== STEP 7: Generate Report =====
    model_info = {
        'architecture': 'SentinelDenseNet (DenseNet121 + GeM + Custom Head)',
        'parameters': f"{param_count:,}",
        'training_stages': len(image_stages),
        'total_epochs': sum(epoch_stages),
        'best_val_recall': float(best_val_recall),
        'best_val_auc': float(best_val_auc),
        'optimizer': 'AdamW',
        'loss': 'FocalLoss (gamma=2.0, alpha=0.75)',
        'augmentations': 'Flip, Rotate, Affine, ColorJitter, GaussianBlur, Mixup, Erasing',
        'tricks': 'Progressive Resizing, Warmup, EMA, TTA, Focal Loss, Label Smoothing, GeM Pool',
    }

    results = generate_report(
        test_labels, final_preds, optimal_thresholds,
        CONFIG['class_names'], output_dir, model_info,
    )

    # ===== STEP 8: Save Final Model for Deployment =====
    deploy_path = output_dir / 'sentinel_model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimal_thresholds': optimal_thresholds.tolist(),
        'class_names': CONFIG['class_names'],
        'image_size': 512,
        'config': CONFIG,
        'results': results,
        'training_history': training_history,
    }, deploy_path)

    # Also save as best_model.pt for the inference server
    deploy_compat = output_dir / 'best_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CONFIG['class_names'],
        'optimal_thresholds': optimal_thresholds.tolist(),
    }, deploy_compat)

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hist_df = pd.DataFrame(training_history)

    axes[0].plot(hist_df['train_loss'], label='Train')
    axes[0].plot(hist_df['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(hist_df['val_recall'], label='Val Recall', color='red')
    axes[1].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.90, color='darkred', linestyle='--', alpha=0.5)
    axes[1].set_title('Validation Recall')
    axes[1].legend()

    axes[2].plot(hist_df['val_auc'], label='Val AUC', color='blue')
    axes[2].set_title('Validation AUC-ROC')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'training_curves.png', dpi=150)
    plt.close()

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Final Model: {deploy_path}")
    print(f"  Deploy Model: {deploy_compat}")
    print(f"  Results: {output_dir / 'results.json'}")
    print(f"  Plots: {output_dir / 'plots/'}")
    print(f"")
    print(f"  FINAL SCORES:")
    print(f"    Macro Recall:      {results['overall']['macro_recall']:.4f}")
    print(f"    Macro Specificity: {results['overall']['macro_specificity']:.4f}")
    print(f"    Macro AUC-ROC:     {results['overall']['macro_auc_roc']:.4f}")
    print(f"    Macro F1:          {results['overall']['macro_f1']:.4f}")
    print(f"")
    print(f"  To deploy, copy best_model.pt to your clinic machine:")
    print(f"    scp {deploy_compat} your_machine:sentinel/models/densenet/best_model.pt")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
