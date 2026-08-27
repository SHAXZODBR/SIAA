"""
===================================================================
  SENTINEL MEDICAL AI — KAGGLE NOTEBOOK TRAINING SCRIPT
===================================================================
  RUN THIS ON KAGGLE (100% FREE, NO CREDIT CARD NEEDED)

  GPU: T4 x2 (16GB VRAM each) — better than Colab free!
  Time: 30 hrs/week free
  Storage: Built-in NIH + RSNA datasets (no upload!)

  HOW TO USE:
  1. Go to https://www.kaggle.com/code
  2. Click "+ New Notebook"
  3. Settings (right sidebar):
     - Accelerator: GPU T4 x2
     - Add Datasets:
       * nih-chest-xrays/data (NIH ChestX-ray14)
       * rsna-pneumonia-detection-challenge (RSNA)
  4. Paste this entire script into a cell
  5. Click "Run All"
  6. Come back in ~8 hours
  7. Download best_model.pt from Output

  DATASETS USED (already on Kaggle, no upload needed):
  - /kaggle/input/data/Data_Entry_2017.csv (NIH labels)
  - /kaggle/input/data/images_001/... through images_012/ (NIH images)
  - /kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/
  - /kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv
===================================================================
"""

import os
import sys
import json
import time
import math
import copy
import random
import warnings
import gc
from pathlib import Path
from datetime import datetime

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

# ===================================================================
# CONFIG — Optimized for Kaggle T4 x2
# ===================================================================

CONFIG = {
    # Kaggle paths (standard) — update these to match YOUR added datasets!
    # Check actual folder names at /kaggle/input/ after adding datasets
    'nih_dir': '/kaggle/input/data',                           # NIH Chest X-rays
    'rsna_dir': '/kaggle/input/rsna-pneumonia-detection-challenge',  # ORIGINAL RSNA with DICOMs
    # Alternative for PNG version (auto-detected below):
    'rsna_png_dir': '/kaggle/input/rsna-pneumonia-detection-challenge-train-pngs',
    'output_dir': '/kaggle/working',               # Writable output
    'cache_dir': '/kaggle/working/cache',          # Preprocessed images
    'use_png_format': True,                        # Use PNG instead of DICOM if available

    # 14 pathology classes (CheXNet)
    'class_names': [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia',
    ],

    # Training — MEMORY SAFE version for Kaggle 30GB RAM limit
    'image_size': 224,              # Smaller — less memory per batch (was 256)
    'batch_size': 32,                # Reduced (was 48) to prevent OOM
    'epochs': 8,                     # Fewer epochs, safer completion (was 10)
    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 2,
    'label_smoothing': 0.05,
    'dropout': 0.3,

    # Focal Loss for class imbalance
    'focal_gamma': 2.0,
    'focal_alpha': 0.75,

    # EMA
    'ema_decay': 0.999,

    # Targets
    'target_recall': 0.85,
    'target_specificity': 0.80,

    # Hardware — MEMORY SAFE
    'mixed_precision': True,
    'num_workers': 2,                # Reduced from 4 — fewer workers = less RAM
    'persistent_workers': False,     # Don't keep workers alive (saves RAM)
    'seed': 42,
    'max_train_samples': 60000,      # CAP at 60K — prevents memory overflow
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
        print("WARNING: No GPU! Enable GPU in Kaggle Settings > Accelerator.")
    return device


def resize_with_pad(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# ===================================================================
# STEP 1: LOAD LABELS
# ===================================================================

def find_file_anywhere(filename_patterns, base='/kaggle/input'):
    """FAST: Use Python glob with recursive matching."""
    import glob
    for pattern in filename_patterns:
        # Recursive glob — handles any nesting depth
        matches = glob.glob(f'{base}/**/{pattern}', recursive=True)
        if matches:
            return matches[0]
        # Case-insensitive fallback
        matches = glob.glob(f'{base}/**/*{pattern.lower()}', recursive=True)
        if matches:
            return matches[0]
        # Try uppercase too
        matches = glob.glob(f'{base}/**/*{pattern}', recursive=True)
        if matches:
            return matches[0]
    return None


def find_nih_image_dirs():
    """FAST: Find NIH image directories using Python glob."""
    import glob
    # NIH structure: images_001/, images_002/, ... images_012/
    # Each may contain images directly or a nested 'images/' folder
    patterns = [
        '/kaggle/input/**/images_*/images',  # Nested: images_001/images/
        '/kaggle/input/**/images_*',          # Direct: images_001/*.png
    ]

    image_dirs = []
    seen = set()

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.isdir(path) or path in seen:
                continue
            # Verify it contains PNG files (check first 3 items)
            try:
                items = os.listdir(path)
                has_png = any(item.endswith('.png') for item in items[:5])
                if has_png:
                    image_dirs.append(path)
                    seen.add(path)
            except Exception:
                pass

    return image_dirs


def find_rsna_png_dir():
    """FAST: Find RSNA image directory using glob."""
    import glob
    # Try common names for RSNA image folders
    patterns = [
        '/kaggle/input/**/stage_2_train_images',
        '/kaggle/input/**/train_images',
        '/kaggle/input/**/*rsna*/**/orig',
        '/kaggle/input/**/orig',  # User's dataset showed "orig" subfolder
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for path in matches:
            if os.path.isdir(path):
                try:
                    items = os.listdir(path)
                    pngs = [f for f in items[:10] if f.endswith('.png')]
                    if pngs:
                        return path
                except Exception:
                    pass
    return None


def load_nih_labels():
    """Load NIH ChestX-ray14 labels — auto-detects path on any Kaggle layout."""
    print(f"  Searching for NIH CSV...")

    # Try common filename patterns
    csv_path = find_file_anywhere([
        'Data_Entry_2017_v2020.csv',
        'Data_Entry_2017.csv',
        'Data_Entry.csv',
    ])

    if csv_path is None:
        print(f"  [WARN] NIH CSV not found. Full /kaggle/input/ structure:")
        import glob as _glob
        # Print all CSV files found
        csvs = _glob.glob('/kaggle/input/**/*.csv', recursive=True)
        print(f"  Found CSV files ({len(csvs)}):")
        for csv in csvs[:20]:
            print(f"    - {csv}")
        # Print first few PNG dirs
        png_dirs = set()
        for png in _glob.glob('/kaggle/input/**/*.png', recursive=True)[:50]:
            png_dirs.add(os.path.dirname(png))
        print(f"\n  Sample image dirs ({len(png_dirs)}):")
        for d in list(png_dirs)[:10]:
            print(f"    - {d}")
        return {}

    print(f"  Found NIH CSV: {csv_path}")
    CONFIG['_nih_csv'] = csv_path
    CONFIG['_nih_base'] = os.path.dirname(csv_path)

    # Find image directories — start with the csv parent
    nih_image_dirs = find_nih_image_dirs()

    # Also check directly near the CSV
    csv_dir = os.path.dirname(csv_path)
    import glob as _glob
    near_csv_patterns = [
        f'{csv_dir}/images_*/images',
        f'{csv_dir}/images_*',
        f'{csv_dir}/images',
        f'{csv_dir}/*/images',
    ]
    for pattern in near_csv_patterns:
        for path in _glob.glob(pattern):
            if os.path.isdir(path):
                try:
                    items = os.listdir(path)
                    if any(f.endswith('.png') for f in items[:5]):
                        if path not in nih_image_dirs:
                            nih_image_dirs.append(path)
                except Exception:
                    pass

    CONFIG['_nih_image_dirs'] = nih_image_dirs
    print(f"  Found {len(nih_image_dirs)} NIH image directories")
    if nih_image_dirs:
        for d in nih_image_dirs[:3]:
            print(f"    - {d}")
        if len(nih_image_dirs) > 3:
            print(f"    ... and {len(nih_image_dirs) - 3} more")

    df = pd.read_csv(csv_path)
    class_names = CONFIG['class_names']
    labels = {}

    print(f"  Found {len(df)} entries in NIH CSV")

    for _, row in df.iterrows():
        fname = row['Image Index']
        stem = fname.replace('.png', '')

        findings = row['Finding Labels'].split('|')
        findings = [f.strip().replace(' ', '_') for f in findings]

        label_vec = [0] * len(class_names)
        for f in findings:
            if f in class_names:
                label_vec[class_names.index(f)] = 1
        labels[stem] = (label_vec, 'nih', fname)

    print(f"  NIH labels: {len(labels)} entries")
    return labels


def load_rsna_labels():
    """Load RSNA Pneumonia labels. Auto-detects PNG vs DICOM format at any nesting."""
    # Search for RSNA CSV recursively
    csv_path = find_file_anywhere([
        'stage_2_train_labels.csv',
        'train_labels.csv',
    ])

    if csv_path is None:
        print(f"  [WARN] RSNA CSV not found — RSNA will be skipped")
        return {}

    print(f"  Found RSNA CSV: {csv_path}")

    # Find RSNA images — search for image folder near the CSV
    rsna_format = None
    rsna_img_dir = None
    csv_parent = os.path.dirname(csv_path)

    # Try obvious locations near the CSV
    candidates = [
        os.path.join(csv_parent, 'stage_2_train_images'),
        os.path.join(csv_parent, 'train_images'),
        os.path.join(csv_parent, 'train'),
        os.path.join(csv_parent, 'images'),
        csv_parent,  # Sometimes files are next to CSV
    ]

    for path in candidates:
        if os.path.isdir(path):
            files = os.listdir(path)
            pngs = [f for f in files if f.endswith('.png')]
            dcms = [f for f in files if f.endswith('.dcm')]
            if len(pngs) > 100:
                rsna_format = 'png'
                rsna_img_dir = path
                break
            elif len(dcms) > 100:
                rsna_format = 'dcm'
                rsna_img_dir = path
                break

    # Still not found? Use our aggressive search
    if rsna_format is None:
        rsna_img_dir = find_rsna_png_dir()
        if rsna_img_dir:
            rsna_format = 'png'

    if rsna_format is None:
        print(f"  [WARN] RSNA images not found near CSV ({csv_parent})")
        return {}

    CONFIG['_rsna_img_dir'] = rsna_img_dir
    print(f"  Found RSNA images: {rsna_img_dir} ({rsna_format.upper()} format)")

    df = pd.read_csv(csv_path)
    class_names = CONFIG['class_names']
    pneumonia_idx = class_names.index('Pneumonia')

    labels = {}
    ext = '.png' if rsna_format == 'png' else '.dcm'
    source_tag = f'rsna_{rsna_format}'

    for _, row in df.iterrows():
        pid = row['patientId']
        if pid in labels:
            continue
        target = int(row['Target'])

        label_vec = [0] * len(class_names)
        if target == 1:
            label_vec[pneumonia_idx] = 1
        labels[f'rsna_{pid}'] = (label_vec, source_tag, f'{pid}{ext}')

    print(f"  RSNA labels: {len(labels)} entries ({rsna_format.upper()} format)")
    return labels


def find_nih_image(filename):
    """NIH images can be in many directories — check cached list first.
    This is called per-image during training, so MUST be fast."""
    cached_dirs = CONFIG.get('_nih_image_dirs', [])

    # Most common case: image is in one of the cached directories
    for img_dir in cached_dirs:
        path = os.path.join(img_dir, filename)
        if os.path.exists(path):
            return path

    return None


# ===================================================================
# STEP 2: DATASET (on-the-fly loading, no preprocessing step)
# ===================================================================

class ChestDataset(Dataset):
    """Load NIH/RSNA images on-the-fly for maximum disk efficiency."""

    def __init__(self, entries, is_train=True, image_size=320):
        """entries: list of (stem, label_vec, source, original_filename)"""
        self.entries = entries
        self.is_train = is_train
        self.image_size = image_size

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(12),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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
        stem, label_vec, source, orig_filename = self.entries[idx]
        label = torch.tensor(label_vec, dtype=torch.float32)

        try:
            if source == 'nih':
                img_path = find_nih_image(orig_filename)
                if img_path is None:
                    raise FileNotFoundError(orig_filename)
                img = Image.open(img_path).convert('RGB')
            elif source == 'rsna_png':
                # PNG format — fast path
                png_dir = CONFIG.get('_rsna_img_dir', CONFIG.get('_rsna_png_dir', ''))
                img_path = os.path.join(png_dir, orig_filename)
                img = Image.open(img_path).convert('RGB')
            elif source == 'rsna_dcm' or source == 'rsna':
                # DICOM fallback
                import pydicom
                dcm_dir = CONFIG.get('_rsna_img_dir', f"{CONFIG['rsna_dir']}/stage_2_train_images")
                dcm_path = os.path.join(dcm_dir, orig_filename)
                ds = pydicom.dcmread(dcm_path, force=True)
                arr = ds.pixel_array.astype(np.float32)
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                img = Image.fromarray(arr).convert('RGB')
            else:
                raise ValueError(f'Unknown source: {source}')

            img = self.transform(img)
            return img, label
        except Exception as e:
            # Return blank image on error
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, label


def get_weighted_sampler(entries, class_names):
    """Oversample rare classes."""
    class_counts = np.zeros(len(class_names))
    for _, label_vec, _, _ in entries:
        class_counts += np.array(label_vec)

    weights = []
    for _, label_vec, _, _ in entries:
        if sum(label_vec) == 0:
            weights.append(1.0)
        else:
            pos = [i for i, v in enumerate(label_vec) if v == 1]
            min_c = min(class_counts[i] for i in pos)
            w = len(entries) / (len(class_names) * min_c + 1)
            weights.append(min(w, 8.0))
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ===================================================================
# STEP 3: MODEL
# ===================================================================

class SentinelNet(nn.Module):
    def __init__(self, num_classes=14, dropout=0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        self.backbone = models.densenet121(weights=weights)

        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.gem = GeMPool(p=3.0)
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = F.relu(feat, inplace=True)
        feat = self.gem(feat).flatten(1)
        return self.head(feat)


class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0):
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
        focal = alpha_t * (1 - p_t) ** self.gamma
        return (focal * bce).mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)


# ===================================================================
# STEP 4: TRAIN / EVAL
# ===================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train one epoch. Saves checkpoint every 500 steps (mid-epoch safety)."""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'E{epoch} Train')

    checkpoint_interval = 500  # Save every 500 batches

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
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Mid-epoch checkpoint — safety net if session dies
        if (step + 1) % checkpoint_interval == 0:
            try:
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'model_state_dict': state,
                    'class_names': CONFIG['class_names'],
                    'epoch': epoch,
                    'step': step + 1,
                    'loss': loss.item(),
                }, f"{CONFIG['output_dir']}/latest_checkpoint.pt")
            except Exception as e:
                print(f"\n  [WARN] Checkpoint save failed: {e}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Memory-efficient eval — uses pre-allocated arrays instead of list.append."""
    model.eval()
    total_loss = 0

    # Pre-allocate arrays (known size)
    total_samples = len(loader.dataset)
    num_classes = len(CONFIG['class_names'])
    all_preds = np.zeros((total_samples, num_classes), dtype=np.float32)
    all_labels = np.zeros((total_samples, num_classes), dtype=np.float32)
    idx = 0

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            loss = criterion(out, labels)

        total_loss += loss.item()
        bs = images.size(0)
        all_preds[idx:idx+bs] = torch.sigmoid(out).cpu().numpy()
        all_labels[idx:idx+bs] = labels.cpu().numpy()
        idx += bs

        # Clear GPU cache periodically
        del images, labels, out, loss

    # Trim to actual size (last batch may be smaller)
    preds = all_preds[:idx]
    labs = all_labels[:idx]
    binary = (preds >= 0.5).astype(np.int8)  # int8 = less memory

    # Metrics
    recall = recall_score(labs, binary, average='macro', zero_division=0)
    prec = precision_score(labs, binary, average='macro', zero_division=0)
    f1 = f1_score(labs, binary, average='macro', zero_division=0)

    # AUC safely
    aucs = []
    for i in range(labs.shape[1]):
        if len(np.unique(labs[:, i])) > 1:
            aucs.append(roc_auc_score(labs[:, i], preds[:, i]))
    auc = np.mean(aucs) if aucs else 0.0

    # Clear memory
    del binary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / len(loader), preds, labs, recall, prec, f1, auc


def optimize_thresholds(y_true, y_pred, class_names, target_recall=0.85):
    """Find optimal threshold per class."""
    thresholds = np.full(len(class_names), 0.5)
    results = {}

    for i, name in enumerate(class_names):
        if len(np.unique(y_true[:, i])) < 2:
            results[name] = {'threshold': 0.5, 'recall': 0.0, 'f1': 0.0}
            continue

        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.02):
            binary = (y_pred[:, i] >= t).astype(int)
            r = recall_score(y_true[:, i], binary, zero_division=0)
            f = f1_score(y_true[:, i], binary, zero_division=0)
            if r >= target_recall and f > best_f1:
                best_f1 = f
                best_thresh = t

        if best_f1 == 0:
            # Fallback: find best recall
            best_r = 0
            for t in np.arange(0.05, 0.95, 0.02):
                binary = (y_pred[:, i] >= t).astype(int)
                r = recall_score(y_true[:, i], binary, zero_division=0)
                if r > best_r:
                    best_r = r
                    best_thresh = t

        thresholds[i] = best_thresh
        binary = (y_pred[:, i] >= best_thresh).astype(int)
        r = recall_score(y_true[:, i], binary, zero_division=0)
        p = precision_score(y_true[:, i], binary, zero_division=0)
        f = f1_score(y_true[:, i], binary, zero_division=0)
        results[name] = {'threshold': float(best_thresh), 'recall': float(r), 'precision': float(p), 'f1': float(f)}

    return thresholds, results


# ===================================================================
# MAIN
# ===================================================================

def main():
    seed_everything(CONFIG['seed'])
    device = get_device()

    print("="*70)
    print("  SENTINEL MEDICAL AI — KAGGLE FREE TRAINING")
    print("  Free GPU: T4 x2 · Dataset: NIH + RSNA (built-in)")
    print("="*70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ===== Load labels =====
    print("\n[1/6] Loading dataset labels...")
    nih_labels = load_nih_labels()
    rsna_labels = load_rsna_labels()

    # Merge into single entries list
    all_entries = []
    for stem, (vec, src, fname) in nih_labels.items():
        all_entries.append((stem, vec, src, fname))
    for stem, (vec, src, fname) in rsna_labels.items():
        all_entries.append((stem, vec, src, fname))

    if CONFIG.get('max_train_samples'):
        random.shuffle(all_entries)
        all_entries = all_entries[:CONFIG['max_train_samples']]

    print(f"  Total entries: {len(all_entries)}")

    if len(all_entries) == 0:
        print("\n  [ERROR] No data found! Cannot proceed.")
        print("  Check that datasets are added in Kaggle sidebar.")
        print("  Expected: NIH Chest X-rays + RSNA Pneumonia Detection Challenge Train PNGs")
        return

    # Class distribution
    print("\n  Class distribution:")
    class_counts = np.zeros(len(CONFIG['class_names']))
    for _, vec, _, _ in all_entries:
        class_counts += np.array(vec)
    for i, name in enumerate(CONFIG['class_names']):
        print(f"    {name:25s}: {int(class_counts[i]):>6d}")

    # ===== Split =====
    print("\n[2/6] Splitting dataset 80/10/10...")
    primary = [np.argmax(vec) if sum(vec) > 0 else -1 for _, vec, _, _ in all_entries]

    train_entries, temp = train_test_split(
        all_entries, train_size=0.8, random_state=42,
        stratify=primary,
    )
    temp_primary = [np.argmax(vec) if sum(vec) > 0 else -1 for _, vec, _, _ in temp]
    val_entries, test_entries = train_test_split(
        temp, train_size=0.5, random_state=42, stratify=temp_primary,
    )
    print(f"  Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")

    # ===== Datasets & Loaders =====
    print("\n[3/6] Creating DataLoaders...")
    train_ds = ChestDataset(train_entries, is_train=True, image_size=CONFIG['image_size'])
    val_ds = ChestDataset(val_entries, is_train=False, image_size=CONFIG['image_size'])
    test_ds = ChestDataset(test_entries, is_train=False, image_size=CONFIG['image_size'])

    sampler = get_weighted_sampler(train_entries, CONFIG['class_names'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                               num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)

    # ===== Model =====
    print("\n[4/6] Building model...")
    model = SentinelNet(num_classes=len(CONFIG['class_names']), dropout=CONFIG['dropout']).to(device)

    # Multi-GPU if available (Kaggle T4 x2)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SentinelNet: {n_params:,} parameters")

    criterion = FocalLoss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha'],
                          label_smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])

    # LR scheduler — warmup + cosine
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
    ema = EMA(model, decay=CONFIG['ema_decay'])

    # ===== Training =====
    print(f"\n[5/6] Training {CONFIG['epochs']} epochs...")
    print("="*70)

    best_recall = 0
    best_auc = 0
    history = []
    patience = 5
    patience_counter = 0

    t_start = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)

        # Step scheduler per batch (simulated)
        for _ in range(len(train_loader)):
            scheduler.step()

        ema.update(model.module if hasattr(model, 'module') else model)

        # Clear memory before eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss, val_preds, val_labs, recall, prec, f1, auc = evaluate(model, val_loader, criterion, device)

        # Free val arrays after use
        del val_preds, val_labs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        print(f"  E{epoch:>2d} | loss={train_loss:.4f} vl={val_loss:.4f} | "
              f"recall={recall:.3f} prec={prec:.3f} f1={f1:.3f} auc={auc:.3f} | "
              f"lr={lr:.5f} | {elapsed:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'recall': recall, 'precision': prec, 'f1': f1, 'auc': auc, 'lr': lr,
        })

        improved = False
        if auc > best_auc:
            best_auc = auc
            improved = True
        if recall > best_recall:
            best_recall = recall
            improved = True

        if improved:
            patience_counter = 0
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'class_names': CONFIG['class_names'],
                'recall': recall, 'auc': auc,
                'config': CONFIG,
            }, f"{CONFIG['output_dir']}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop (no improvement {patience} epochs)")
                break

    total_time = time.time() - t_start
    print(f"\n  Training time: {total_time/60:.1f} minutes")

    # ===== Final evaluation =====
    print(f"\n[6/6] Final evaluation on test set...")
    print("="*70)

    ckpt = torch.load(f"{CONFIG['output_dir']}/best_model.pt", map_location=device, weights_only=False)
    base_model = SentinelNet(num_classes=len(CONFIG['class_names']), dropout=0).to(device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model.eval()

    test_loss, test_preds, test_labs, test_r, test_p, test_f, test_auc = evaluate(
        base_model, test_loader, criterion, device
    )

    # Optimize thresholds on validation
    _, val_preds, val_labs, _, _, _, _ = evaluate(base_model, val_loader, criterion, device)
    optimal_thresholds, thresh_results = optimize_thresholds(
        val_labs, val_preds, CONFIG['class_names'],
        target_recall=CONFIG['target_recall']
    )

    # Final metrics with optimized thresholds
    binary = np.zeros_like(test_preds)
    for i in range(len(CONFIG['class_names'])):
        binary[:, i] = (test_preds[:, i] >= optimal_thresholds[i]).astype(int)
    opt_recall = recall_score(test_labs, binary, average='macro', zero_division=0)
    opt_prec = precision_score(test_labs, binary, average='macro', zero_division=0)
    opt_f1 = f1_score(test_labs, binary, average='macro', zero_division=0)

    print("\n  PER-CLASS RESULTS (optimized thresholds):")
    for name, r in thresh_results.items():
        print(f"    {name:25s} | t={r['threshold']:.2f} | recall={r['recall']:.3f} "
              f"prec={r['precision']:.3f} f1={r['f1']:.3f}")

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Macro Recall (0.5):       {test_r:.4f}")
    print(f"  Macro Recall (optimized): {opt_recall:.4f}  {'PASS' if opt_recall >= 0.85 else 'NEEDS MORE TRAINING'}")
    print(f"  Macro Precision:          {opt_prec:.4f}")
    print(f"  Macro F1:                 {opt_f1:.4f}")
    print(f"  Macro AUC-ROC:            {test_auc:.4f}")
    print(f"{'='*70}")

    # Save final deployable model
    torch.save({
        'model_state_dict': ckpt['model_state_dict'],
        'class_names': CONFIG['class_names'],
        'optimal_thresholds': optimal_thresholds.tolist(),
        'image_size': CONFIG['image_size'],
        'recall': float(opt_recall),
        'auc': float(test_auc),
        'training_time_minutes': total_time / 60,
    }, f"{CONFIG['output_dir']}/best_model.pt")

    # Save results JSON
    results = {
        'overall': {
            'recall_optimized': float(opt_recall),
            'precision_optimized': float(opt_prec),
            'f1_optimized': float(opt_f1),
            'auc_roc': float(test_auc),
            'recall_default_0.5': float(test_r),
        },
        'per_class': thresh_results,
        'history': history,
        'config': {k: v for k, v in CONFIG.items() if not callable(v)},
        'training_time_min': total_time / 60,
    }
    with open(f"{CONFIG['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plots
    try:
        df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df['train_loss'], label='train'); axes[0].plot(df['val_loss'], label='val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(df['recall'], color='red', label='recall'); axes[1].plot(df['auc'], color='blue', label='auc')
        axes[1].axhline(0.85, color='gray', linestyle='--'); axes[1].set_title('Recall & AUC'); axes[1].legend()
        axes[2].plot(df['lr']); axes[2].set_title('Learning Rate')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=120, bbox_inches='tight')
        plt.close()

        # Per-class bars
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(CONFIG['class_names']))
        recalls = [thresh_results[n]['recall'] for n in CONFIG['class_names']]
        precs = [thresh_results[n]['precision'] for n in CONFIG['class_names']]
        ax.bar(x - 0.2, recalls, 0.4, label='Recall', color='#ef4444')
        ax.bar(x + 0.2, precs, 0.4, label='Precision', color='#3b82f6')
        ax.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='Target 85%')
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG['class_names'], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/per_class_metrics.png", dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    print(f"\n  Outputs saved to /kaggle/working/:")
    print(f"    - best_model.pt       (Download this!)")
    print(f"    - results.json")
    print(f"    - training_curves.png")
    print(f"    - per_class_metrics.png")
    print(f"\n  To download: Click 'Output' tab in Kaggle, then click 'Download All'")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
