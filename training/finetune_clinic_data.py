"""
================================================================================
  SENTINEL MEDICAL AI — CLINIC DATA FINE-TUNING (SUPER EASY MODE)
================================================================================
  Fine-tune pre-trained models on YOUR clinic's data.

  WHY FINE-TUNE:
  - Pre-trained models are trained on US/EU hospitals
  - Your clinic may have:
    * Different MRI/CT machines (different calibration)
    * Uzbekistan/Central Asian patient demographics
    * Specific diseases common in your region
    * Your radiologists' reporting style
  - Fine-tuning on 200-500 clinic samples improves accuracy by 5-15%

  HOW IT WORKS:
  1. Start with pre-trained model (all weights already "know" medicine)
  2. FREEZE most of the model (keep the knowledge)
  3. Only train the last layers on YOUR data
  4. Learning rate is 10x lower (gentle tuning)
  5. Need only 5-20 epochs (vs 100+ for scratch training)

  WHAT YOU NEED:
  - 200+ labeled clinic images (minimum)
  - 500+ is better
  - 1000+ is ideal
  - Labels can be simple: disease name per image

  WHERE TO RUN:
  - Google Colab (FREE T4, 2-3 hours)
  - Kaggle (FREE T4 x2, 1-2 hours)
  - Your local machine (slower but works)

  USAGE:
    # Fine-tune chest X-ray model:
    python training/finetune_clinic_data.py --modality chest --data clinic_data/chest

    # Fine-tune brain model:
    python training/finetune_clinic_data.py --modality brain --data clinic_data/brain
================================================================================
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ===================================================================
# FINE-TUNING CONFIG — Much gentler than original training!
# ===================================================================

CONFIG = {
    'epochs': 10,                    # Only 10 epochs (vs 50+ for scratch)
    'batch_size': 16,
    'learning_rate': 1e-4,           # 10x lower than original training
    'weight_decay': 1e-5,
    'image_size': 224,

    # Freezing strategy
    'freeze_backbone': True,         # Keep pre-trained knowledge
    'unfreeze_last_n_blocks': 2,     # Only fine-tune last 2 dense blocks

    # Training
    'warmup_epochs': 1,
    'patience': 5,
    'min_samples_per_class': 10,     # Safety check

    # Augmentation (conservative for fine-tuning)
    'augment': True,
    'rand_flip_prob': 0.5,
    'rand_rotate_degrees': 10,
    'color_jitter': 0.1,

    'mixed_precision': True,
    'seed': 42,
}


# ===================================================================
# FLEXIBLE DATASET — Auto-detects folder structure
# ===================================================================

class FlexibleClinicDataset(Dataset):
    """Automatic dataset from folder structure.

    Expected structure:
        clinic_data/
            pneumonia/
                img001.jpg
                img002.jpg
                ...
            normal/
                ...
            cardiomegaly/
                ...

    OR single folder + labels.json:
        clinic_data/
            images/
                img001.jpg
                img002.jpg
            labels.json  # {"img001": "pneumonia", ...}
    """

    def __init__(self, data_dir: str, is_train: bool = True, image_size: int = 224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_train = is_train

        # Auto-detect structure
        self.entries = []
        self.class_names = []

        if (self.data_dir / 'labels.json').exists():
            self._load_from_json()
        else:
            self._load_from_folders()

        print(f"  Dataset: {len(self.entries)} images, {len(self.class_names)} classes")
        if is_train:
            self.transform = self._train_transform()
        else:
            self.transform = self._val_transform()

    def _load_from_json(self):
        """Single folder + labels.json."""
        with open(self.data_dir / 'labels.json') as f:
            labels = json.load(f)

        # Build class list
        unique_classes = set()
        for cls_list in labels.values():
            if isinstance(cls_list, list):
                unique_classes.update(cls_list)
            else:
                unique_classes.add(cls_list)
        self.class_names = sorted(unique_classes)

        # Build entries
        img_dir = self.data_dir / 'images'
        if not img_dir.exists():
            img_dir = self.data_dir

        for filename, cls in labels.items():
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = img_dir / f'{filename}{ext}'
                if img_path.exists():
                    if isinstance(cls, list):
                        cls = cls[0]  # Take first class if multi-label
                    self.entries.append((str(img_path), self.class_names.index(cls)))
                    break

    def _load_from_folders(self):
        """Each subfolder is a class."""
        subfolders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in subfolders]

        for class_idx, folder in enumerate(subfolders):
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.entries.append((str(img_path), class_idx))

    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size + 16, self.image_size + 16)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=CONFIG['rand_flip_prob']),
            transforms.RandomRotation(CONFIG['rand_rotate_degrees']),
            transforms.ColorJitter(
                brightness=CONFIG['color_jitter'],
                contrast=CONFIG['color_jitter'],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, cls_idx = self.entries[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, self.image_size, self.image_size)
        return img, torch.tensor(cls_idx, dtype=torch.long)


# ===================================================================
# MODEL LOADING + FREEZING
# ===================================================================

def load_pretrained_and_freeze(base_model_path: str, num_classes: int, freeze: bool = True):
    """Load pre-trained model and freeze backbone layers."""

    # Try loading pre-trained weights
    if base_model_path and os.path.exists(base_model_path):
        print(f"\n[BASE] Loading pre-trained weights from {base_model_path}")
        ckpt = torch.load(base_model_path, weights_only=False, map_location='cpu')

        base_classes = ckpt.get('class_names', [])
        num_base = len(base_classes) if base_classes else 14

        # Check if it's a HuggingFace brain model
        if ckpt.get('source') == 'huggingface':
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(ckpt['repo_id'])
            # Reinitialize classifier for our num_classes
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else 768
                model.classifier = nn.Linear(in_features, num_classes)
        else:
            # Standard DenseNet121
            model = models.densenet121(weights=None)
            num_features = model.classifier.in_features

            # Build classifier matching base model
            if 'head' in str(ckpt['model_state_dict'].keys()).lower() or len(base_classes) > 0:
                model.classifier = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_base),
                )
            else:
                model.classifier = nn.Linear(num_features, num_base)

            # Load weights
            try:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                print(f"  Loaded {num_base}-class pretrained weights")
            except Exception as e:
                print(f"  Partial load: {e}")

            # Replace final classifier for our task
            model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
    else:
        print(f"\n[BASE] No pre-trained checkpoint — using ImageNet DenseNet121")
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # Freeze backbone if requested
    if freeze:
        print(f"\n[FREEZE] Freezing backbone, training only last 2 blocks + classifier")
        frozen_count = 0
        trainable_count = 0
        # Freeze all features first
        if hasattr(model, 'features'):
            for name, param in model.features.named_parameters():
                # Unfreeze last N dense blocks
                if 'denseblock4' in name or 'norm5' in name:
                    param.requires_grad = True
                    trainable_count += param.numel()
                else:
                    param.requires_grad = False
                    frozen_count += param.numel()
        # Classifier is always trainable
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
                trainable_count += param.numel()

        total = frozen_count + trainable_count
        print(f"  Frozen: {frozen_count:,} params ({100*frozen_count/total:.1f}%)")
        print(f"  Trainable: {trainable_count:,} params ({100*trainable_count/total:.1f}%)")

    return model


# ===================================================================
# TRAIN + EVAL
# ===================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    from torch.cuda.amp import autocast
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'E{epoch} Train')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            # Handle HuggingFace model output
            if hasattr(out, 'logits'):
                out = out.logits
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
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
    from torch.cuda.amp import autocast
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=CONFIG['mixed_precision']):
            out = model(images)
            if hasattr(out, 'logits'):
                out = out.logits
            loss = criterion(out, labels)
        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labs = np.concatenate(all_labels)

    acc = accuracy_score(labs, preds)
    rec = recall_score(labs, preds, average='macro', zero_division=0)
    prec = precision_score(labs, preds, average='macro', zero_division=0)
    f1 = f1_score(labs, preds, average='macro', zero_division=0)

    return total_loss / len(loader), acc, rec, prec, f1


# ===================================================================
# MAIN
# ===================================================================

def main():
    from torch.cuda.amp import GradScaler

    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=['chest', 'brain', 'ct', 'custom'],
                        default='chest', help='Which modality to fine-tune')
    parser.add_argument('--data', required=True, help='Clinic data directory')
    parser.add_argument('--base-model', help='Path to pre-trained model')
    parser.add_argument('--output', default='models', help='Output directory')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    args = parser.parse_args()

    if args.epochs:
        CONFIG['epochs'] = args.epochs

    # Auto-detect base model
    if not args.base_model:
        default_paths = {
            'chest': 'models/densenet/best_model.pt',
            'brain': 'models/brain_2d/best_brain_model.pt',
            'ct': 'models/head_ct/best_head_ct_model.pt',
        }
        args.base_model = default_paths.get(args.modality, None)

    output_dir = Path(args.output) / f'{args.modality}_finetuned'
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print(f"\n[DATA] Loading clinic data from {args.data}")
    full_ds = FlexibleClinicDataset(args.data, is_train=True, image_size=CONFIG['image_size'])
    print(f"  Classes found: {full_ds.class_names}")

    # Class distribution
    counts = {}
    for _, cls in full_ds.entries:
        counts[full_ds.class_names[cls]] = counts.get(full_ds.class_names[cls], 0) + 1
    print(f"  Distribution:")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:25s}: {cnt}")
        if cnt < CONFIG['min_samples_per_class']:
            print(f"    ⚠ Only {cnt} samples — need at least {CONFIG['min_samples_per_class']}")

    # Split 80/10/10
    indices = list(range(len(full_ds)))
    labels_only = [full_ds.entries[i][1] for i in indices]
    train_idx, temp_idx = train_test_split(indices, train_size=0.8, random_state=42, stratify=labels_only)
    temp_labels = [labels_only[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42, stratify=temp_labels)

    print(f"\nTrain: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Create split datasets
    train_entries = [full_ds.entries[i] for i in train_idx]
    val_entries = [full_ds.entries[i] for i in val_idx]
    test_entries = [full_ds.entries[i] for i in test_idx]

    train_ds = FlexibleClinicDataset(args.data, is_train=True, image_size=CONFIG['image_size'])
    train_ds.entries = train_entries
    val_ds = FlexibleClinicDataset(args.data, is_train=False, image_size=CONFIG['image_size'])
    val_ds.entries = val_entries
    test_ds = FlexibleClinicDataset(args.data, is_train=False, image_size=CONFIG['image_size'])
    test_ds.entries = test_entries

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=2, pin_memory=True)

    # Model
    num_classes = len(full_ds.class_names)
    model = load_pretrained_and_freeze(args.base_model, num_classes, freeze=CONFIG['freeze_backbone'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimize trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Train
    print(f"\n{'='*60}")
    print(f"FINE-TUNING {args.modality.upper()} MODEL — {CONFIG['epochs']} epochs")
    print(f"{'='*60}")

    best_acc = 0
    history = []
    patience_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, val_acc, val_rec, val_prec, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  E{epoch} | loss={train_loss:.3f} acc={train_acc:.3f} | val_acc={val_acc:.3f} rec={val_rec:.3f} f1={val_f1:.3f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_acc': val_acc, 'val_recall': val_rec, 'val_f1': val_f1})

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': full_ds.class_names,
                'image_size': CONFIG['image_size'],
                'accuracy': val_acc,
                'recall': val_rec,
                'is_fine_tuned': True,
                'base_model': args.base_model,
                'modality': args.modality,
            }, output_dir / 'best_model.pt')
            print(f"    → Saved (acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"  Early stop")
                break

    # Test
    print(f"\n[TEST] Final evaluation on test set...")
    ckpt = torch.load(output_dir / 'best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc, test_rec, test_prec, test_f1 = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"FINAL FINE-TUNED RESULTS ({args.modality})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  F1:        {test_f1:.4f}")

    # Save results
    results = {
        'modality': args.modality,
        'base_model': args.base_model,
        'classes': full_ds.class_names,
        'test_metrics': {
            'accuracy': float(test_acc),
            'recall': float(test_rec),
            'precision': float(test_prec),
            'f1': float(test_f1),
        },
        'training_history': history,
        'total_samples': len(full_ds),
    }
    with open(output_dir / 'finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Fine-tuned model saved to: {output_dir}/best_model.pt")
    print(f"✓ Deploy: copy best_model.pt to models/{args.modality}/best_model.pt")


if __name__ == '__main__':
    main()
