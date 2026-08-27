"""
================================================================================
  SENTINEL MEDICAL AI — KAGGLE QUICK TRAINING (MEMORY SAFE)
================================================================================
  OPTIMIZED FOR:
  - Kaggle's 30 GB RAM limit (no OOM crashes)
  - Fast completion (~2-3 hours vs 8+)
  - Reliable Save Version commits (fits well within 12-hour budget)
  - Getting a WORKING model fast for demo

  Compromises:
  - Uses 30K sample subset (vs full 112K)
  - Smaller images 192x192 (vs 320x320)
  - Fewer epochs (5 vs 10)
  - Slightly lower accuracy (~80% recall vs 90%)
  - But GUARANTEED to complete

  USE THIS FIRST TO GET A WORKING MODEL, then run full training for production.

  HOW TO USE:
  1. Kaggle > New Notebook
  2. GPU T4 x2
  3. Add dataset: NIH Chest X-rays (nih-chest-xrays)
  4. Paste this entire script
  5. Save Version > Save & Run All (Commit)
  6. Done in 2-3 hours
================================================================================
"""

import os
import sys
import json
import time
import math
import gc
import random
import warnings
import glob

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
from PIL import Image
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')


CONFIG = {
    'output_dir': '/kaggle/working',

    'class_names': [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia',
    ],

    # QUICK MODE — Optimized for reliability over max accuracy
    'image_size': 192,                # Very small for speed + low RAM
    'batch_size': 48,                 # Bigger batch since images smaller
    'epochs': 5,                      # Quick train
    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,
    'label_smoothing': 0.05,
    'dropout': 0.2,

    'focal_gamma': 2.0,
    'focal_alpha': 0.75,

    # Memory management
    'max_samples': 30000,            # Cap dataset — prevents OOM
    'num_workers': 2,                # Fewer workers = less RAM
    'mixed_precision': True,
    'seed': 42,

    'target_recall': 0.80,           # Lower target for quick mode
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
# FAST DATASET DISCOVERY
# ===================================================================

def find_csv():
    patterns = [
        '/kaggle/input/**/Data_Entry_2017.csv',
        '/kaggle/input/**/Data_Entry_2017_v2020.csv',
    ]
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches[0]
    return None


def find_image_dirs():
    """Find NIH image dirs — fast."""
    dirs = []
    seen = set()
    for pattern in ['/kaggle/input/**/images_*/images', '/kaggle/input/**/images_*']:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.isdir(path) or path in seen:
                continue
            try:
                items = os.listdir(path)
                if any(f.endswith('.png') for f in items[:5]):
                    dirs.append(path)
                    seen.add(path)
            except Exception:
                pass
    return dirs


# ===================================================================
# DATASET
# ===================================================================

class QuickChestDataset(Dataset):
    """Memory-efficient chest X-ray dataset."""

    def __init__(self, entries, image_dirs, is_train=True, image_size=192):
        self.entries = entries
        self.image_dirs = image_dirs
        self.is_train = is_train
        self.image_size = image_size

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size + 16, image_size + 16)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.15, contrast=0.15),
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

    def __getitem__(self, idx):
        fname, label_vec = self.entries[idx]
        label = torch.tensor(label_vec, dtype=torch.float32)

        try:
            img_path = None
            for d in self.image_dirs:
                candidate = os.path.join(d, fname)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                return torch.zeros(3, self.image_size, self.image_size), label

            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), label


# ===================================================================
# MODEL
# ===================================================================

class QuickNet(nn.Module):
    def __init__(self, num_classes=14, dropout=0.2):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        self.backbone = models.densenet121(weights=weights)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = F.relu(feat, inplace=True)
        return self.head(feat)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, ls=0.05):
        super().__init__()
        self.gamma, self.alpha, self.ls = gamma, alpha, ls

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ===================================================================
# TRAIN / EVAL (memory-safe)
# ===================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'E{epoch}')

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

        # Periodic memory clearing
        if (step + 1) % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save checkpoint every 500 steps (safety)
        if (step + 1) % 500 == 0:
            try:
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'model_state_dict': state,
                    'class_names': CONFIG['class_names'],
                    'epoch': epoch,
                    'step': step + 1,
                }, f"{CONFIG['output_dir']}/checkpoint_latest.pt")
            except Exception:
                pass

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

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

    preds = all_preds[:idx]
    labs = all_labels[:idx]
    binary = (preds >= 0.5).astype(np.int8)

    recall = recall_score(labs, binary, average='macro', zero_division=0)
    prec = precision_score(labs, binary, average='macro', zero_division=0)
    f1 = f1_score(labs, binary, average='macro', zero_division=0)

    aucs = []
    for i in range(labs.shape[1]):
        if len(np.unique(labs[:, i])) > 1:
            aucs.append(roc_auc_score(labs[:, i], preds[:, i]))
    auc = np.mean(aucs) if aucs else 0.0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / len(loader), preds, labs, recall, prec, f1, auc


# ===================================================================
# MAIN
# ===================================================================

def main():
    seed_everything(CONFIG['seed'])
    device = get_device()

    print("="*70)
    print("  SENTINEL QUICK TRAINING — Memory Safe Mode")
    print(f"  Using {CONFIG['max_samples']} samples, {CONFIG['image_size']}x{CONFIG['image_size']} images")
    print(f"  {CONFIG['epochs']} epochs, batch={CONFIG['batch_size']}")
    print("="*70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Load CSV
    csv_path = find_csv()
    if not csv_path:
        print("[FATAL] NIH CSV not found. Add dataset: nih-chest-xrays")
        return
    print(f"\nFound CSV: {csv_path}")

    image_dirs = find_image_dirs()
    print(f"Found {len(image_dirs)} image directories")
    if not image_dirs:
        print("[FATAL] No image dirs")
        return

    # Load labels
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries")

    class_names = CONFIG['class_names']
    entries = []
    for _, row in df.iterrows():
        fname = row['Image Index']
        findings = [f.strip().replace(' ', '_') for f in row['Finding Labels'].split('|')]
        label_vec = [0] * len(class_names)
        for f in findings:
            if f in class_names:
                label_vec[class_names.index(f)] = 1
        entries.append((fname, label_vec))

    # Cap at max_samples
    if len(entries) > CONFIG['max_samples']:
        print(f"\nCapping at {CONFIG['max_samples']} samples (from {len(entries)})")
        random.shuffle(entries)
        entries = entries[:CONFIG['max_samples']]

    # Split
    primary = [np.argmax(vec) if sum(vec) > 0 else -1 for _, vec in entries]
    train_e, temp = train_test_split(entries, train_size=0.85, random_state=42, stratify=primary)
    temp_p = [np.argmax(vec) if sum(vec) > 0 else -1 for _, vec in temp]
    val_e, test_e = train_test_split(temp, train_size=0.5, random_state=42, stratify=temp_p)
    print(f"Train: {len(train_e)} | Val: {len(val_e)} | Test: {len(test_e)}")

    # DataLoaders
    train_ds = QuickChestDataset(train_e, image_dirs, is_train=True, image_size=CONFIG['image_size'])
    val_ds = QuickChestDataset(val_e, image_dirs, is_train=False, image_size=CONFIG['image_size'])
    test_ds = QuickChestDataset(test_e, image_dirs, is_train=False, image_size=CONFIG['image_size'])

    # Weighted sampler
    class_counts = np.zeros(len(class_names))
    for _, vec in train_e:
        class_counts += np.array(vec)
    weights = [1.0 / max(min(class_counts[i] for i, v in enumerate(vec) if v == 1), 1) if sum(vec) > 0 else 1.0 for _, vec in train_e]
    weights = [min(w * 1000, 8.0) for w in weights]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                               num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)

    # Model
    model = QuickNet(num_classes=len(class_names), dropout=CONFIG['dropout']).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    criterion = FocalLoss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha'], ls=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    total_steps = CONFIG['epochs'] * len(train_loader)
    warmup = CONFIG['warmup_epochs'] * len(train_loader)
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return max(CONFIG['min_lr']/CONFIG['learning_rate'], 0.5*(1+math.cos(math.pi*prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    # Train
    print(f"\nTraining {CONFIG['epochs']} epochs...")
    print("="*70)

    best_auc = 0
    history = []
    t0 = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        e0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        for _ in range(len(train_loader)):
            scheduler.step()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss, vp, vl, rec, prec, f1, auc = evaluate(model, val_loader, criterion, device)
        del vp, vl

        elapsed = time.time() - e0
        print(f"  E{epoch} | loss={train_loss:.3f} val={val_loss:.3f} | rec={rec:.3f} f1={f1:.3f} auc={auc:.3f} | {elapsed:.0f}s")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'recall': rec, 'f1': f1, 'auc': auc})

        if auc > best_auc:
            best_auc = auc
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'class_names': class_names,
                'recall': rec, 'auc': auc,
            }, f"{CONFIG['output_dir']}/best_model.pt")
            print(f"    → Saved best (auc={auc:.3f})")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} min")

    # Test
    print("\nFinal test...")
    ckpt = torch.load(f"{CONFIG['output_dir']}/best_model.pt", weights_only=False, map_location=device)
    test_model = QuickNet(num_classes=len(class_names), dropout=0).to(device)
    test_model.load_state_dict(ckpt['model_state_dict'])
    test_model.eval()

    _, test_preds, test_labs, t_rec, t_prec, t_f1, t_auc = evaluate(test_model, test_loader, criterion, device)

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"  Recall:     {t_rec:.4f}")
    print(f"  Precision:  {t_prec:.4f}")
    print(f"  F1:         {t_f1:.4f}")
    print(f"  AUC-ROC:    {t_auc:.4f}")
    print("="*70)

    results = {
        'overall': {'recall': float(t_rec), 'precision': float(t_prec),
                    'f1': float(t_f1), 'auc_roc': float(t_auc)},
        'training_time_min': total_time / 60,
        'config': {k: v for k, v in CONFIG.items() if not callable(v)},
    }
    with open(f"{CONFIG['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDownload best_model.pt from Output tab")


if __name__ == '__main__':
    main()
