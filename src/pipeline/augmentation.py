"""MONAI-based medical image augmentation pipeline for Sentinel Medical AI.

Uses medical-specific augmentations: random flips, rotations, intensity shifts,
zoom, and elastic deformation — designed for radiology images.
"""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
    ToTensord,
)
from typing import Optional
from loguru import logger


def get_train_transforms(
    image_size: int = 512,
    flip_prob: float = 0.5,
    rotate_range: float = 0.2618,
    scale_intensity: float = 0.1,
    shift_intensity: float = 0.1,
    zoom_range: tuple = (0.9, 1.1),
) -> Compose:
    """Get MONAI augmentation transforms for training.

    Args:
        image_size: Input image size (assumes square).
        flip_prob: Probability of horizontal flip.
        rotate_range: Max rotation in radians (±15° = 0.2618).
        scale_intensity: Scale factor range for intensity augmentation.
        shift_intensity: Shift offset range for intensity augmentation.
        zoom_range: Min/max zoom factors.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = Compose(
        [
            # Random horizontal flip
            RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),
            # Random rotation ±15 degrees
            RandRotated(
                keys=["image"],
                range_x=rotate_range,
                prob=0.5,
                mode="bilinear",
                padding_mode="zeros",
            ),
            # Random intensity scaling (brightness variation)
            RandScaleIntensityd(keys=["image"], factors=scale_intensity, prob=0.5),
            # Random intensity shift (contrast variation)
            RandShiftIntensityd(keys=["image"], offsets=shift_intensity, prob=0.5),
            # Random zoom
            RandZoomd(
                keys=["image"],
                min_zoom=zoom_range[0],
                max_zoom=zoom_range[1],
                prob=0.3,
                mode="bilinear",
                padding_mode="constant",
            ),
            # Random Gaussian noise
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
            # Random contrast adjustment
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
            # Ensure output is tensor
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )

    logger.debug("Created training augmentation pipeline")
    return transforms


def get_val_transforms() -> Compose:
    """Get transforms for validation/test — no augmentation, just convert to tensor."""
    transforms = Compose(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    return transforms


def apply_augmentation(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    is_training: bool = True,
    config: Optional[dict] = None,
) -> dict:
    """Apply augmentation to a single image.

    Args:
        image: Preprocessed image array (H, W), float32, [0, 1].
        label: Optional label array (for classification: int, for segmentation: mask).
        is_training: If True, apply augmentation. If False, only convert to tensor.
        config: Optional augmentation config dict.

    Returns:
        Dict with "image" tensor and optionally "label".
    """
    # Add channel dimension: (H, W) → (1, H, W) for MONAI
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    data = {"image": image}
    if label is not None:
        data["label"] = label

    if is_training:
        if config:
            transforms = get_train_transforms(
                flip_prob=config.get("rand_flip_prob", 0.5),
                rotate_range=config.get("rand_rotate_range", 0.2618),
                scale_intensity=config.get("rand_scale_intensity_factors", [0.1])[0],
                shift_intensity=config.get("rand_shift_intensity_offsets", [0.1])[0],
                zoom_range=tuple(config.get("rand_zoom_range", [0.9, 1.1])),
            )
        else:
            transforms = get_train_transforms()
    else:
        transforms = get_val_transforms()

    augmented = transforms(data)
    return augmented
