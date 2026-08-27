"""PyTorch Dataset and DataLoader for Sentinel Medical AI.

Handles dataset splitting, label loading, and efficient data loading
for DenseNet121 classification training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from loguru import logger
import cv2
import json

from src.pipeline.augmentation import get_train_transforms, get_val_transforms


class SentinelDataset(Dataset):
    """PyTorch Dataset for medical image classification.

    Loads preprocessed PNG images and their corresponding labels
    for DenseNet121 training.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_file: str | Path,
        class_names: list[str],
        transforms=None,
        image_size: int = 512,
    ):
        """
        Args:
            image_dir: Directory containing preprocessed PNG images.
            label_file: Path to JSON label file mapping filename → list of classes.
            class_names: List of all class names.
            transforms: MONAI transforms to apply.
            image_size: Expected image size.
        """
        self.image_dir = Path(image_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transforms = transforms
        self.image_size = image_size

        # Load labels
        label_path = Path(label_file)
        if label_path.exists():
            with open(label_path) as f:
                self.labels = json.load(f)
        else:
            # If no label file, create empty labels (for inference/demo)
            self.labels = {}
            logger.warning(f"No label file found at {label_path}. Using empty labels.")

        # Get image file list
        self.image_files = sorted(
            [f for f in self.image_dir.glob("*.png") if f.stem in self.labels]
            if self.labels
            else list(self.image_dir.glob("*.png"))
        )

        logger.info(
            f"Dataset initialized: {len(self.image_files)} images, "
            f"{self.num_classes} classes from {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single sample.

        Returns:
            Dict with:
                - "image": tensor of shape (1, H, W)
                - "label": multi-hot tensor of shape (num_classes,)
                - "filename": original filename
        """
        img_path = self.image_files[idx]
        filename = img_path.stem

        # Load image as grayscale float32
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            # Return zeros as fallback
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            img = img.astype(np.float32) / 255.0

        # Add channel dim: (H, W) → (1, H, W)
        img = img[np.newaxis, ...]

        # Build multi-hot label vector
        label = np.zeros(self.num_classes, dtype=np.float32)
        if filename in self.labels:
            for class_name in self.labels[filename]:
                if class_name in self.class_names:
                    class_idx = self.class_names.index(class_name)
                    label[class_idx] = 1.0

        data = {"image": img}

        # Apply MONAI transforms
        if self.transforms is not None:
            data = self.transforms(data)
            img = data["image"]
        else:
            img = torch.from_numpy(img)

        return {
            "image": img,
            "label": torch.from_numpy(label),
            "filename": filename,
        }


def compute_class_weights(label_file: str | Path, class_names: list[str]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced dataset.

    Args:
        label_file: Path to JSON label file.
        class_names: List of class names.

    Returns:
        Tensor of class weights, shape (num_classes,).
    """
    with open(label_file) as f:
        labels = json.load(f)

    num_classes = len(class_names)
    class_counts = np.zeros(num_classes)

    for filename, classes in labels.items():
        for cls in classes:
            if cls in class_names:
                idx = class_names.index(cls)
                class_counts[idx] += 1

    total = len(labels)
    # Inverse frequency weighting, clamped to prevent extreme values
    weights = np.where(class_counts > 0, total / (num_classes * class_counts), 1.0)
    weights = np.clip(weights, 0.5, 10.0)

    logger.info(f"Class weights computed: {dict(zip(class_names, weights.round(3)))}")
    return torch.from_numpy(weights).float()


def split_dataset(
    image_dir: str | Path,
    label_file: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """Split dataset into train/val/test sets with stratification.

    Copies images and labels into organized directory structure.

    Args:
        image_dir: Source directory with all preprocessed images.
        label_file: JSON file mapping filenames to labels.
        output_dir: Destination directory for split dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with split statistics.
    """
    import shutil

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    with open(label_file) as f:
        labels = json.load(f)

    filenames = list(labels.keys())
    # Use primary (first) class for stratification
    primary_classes = [labels[f][0] if labels[f] else "unknown" for f in filenames]

    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filenames,
        primary_classes,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=primary_classes,
    )

    # Second split: val vs test
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=relative_val,
        random_state=random_seed,
        stratify=temp_labels,
    )

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split_name, files in splits.items():
        split_dir = output_dir / split_name / "images"
        split_dir.mkdir(parents=True, exist_ok=True)

        split_labels = {}
        for f in files:
            src = image_dir / f"{f}.png"
            if src.exists():
                shutil.copy2(src, split_dir / f"{f}.png")
            split_labels[f] = labels[f]

        # Save split-specific label file
        label_out = output_dir / split_name / "labels.json"
        with open(label_out, "w") as lf:
            json.dump(split_labels, lf, indent=2)

    stats = {
        "total": len(filenames),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }
    logger.info(f"Dataset split: {stats}")
    return stats


def create_dataloaders(
    processed_dir: str | Path,
    class_names: list[str],
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
) -> dict:
    """Create train/val/test DataLoaders from split dataset.

    Args:
        processed_dir: Root directory containing train/val/test subdirs.
        class_names: List of class names.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        image_size: Image size.

    Returns:
        Dict with "train", "val", "test" DataLoader objects.
    """
    processed_dir = Path(processed_dir)

    dataloaders = {}
    for split in ["train", "val", "test"]:
        img_dir = processed_dir / split / "images"
        label_file = processed_dir / split / "labels.json"

        if not img_dir.exists():
            logger.warning(f"Split directory not found: {img_dir}")
            continue

        transforms = get_train_transforms() if split == "train" else get_val_transforms()

        dataset = SentinelDataset(
            image_dir=img_dir,
            label_file=label_file,
            class_names=class_names,
            transforms=transforms,
            image_size=image_size,
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

        logger.info(f"DataLoader [{split}]: {len(dataset)} samples, batch_size={batch_size}")

    return dataloaders
