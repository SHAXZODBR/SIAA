"""Fine-tune the pre-trained DenseNet121 model on your own clinic DICOM images.

This is STEP 5 — after training on public datasets, fine-tune on local
Uzbekistan clinic data to adapt the model to local equipment, protocols,
and patient population.

Usage:
    python run_finetune.py \
        --base-model models/densenet/best_model.pt \
        --data data/clinic_processed \
        --epochs 30 \
        --lr 0.00001

Why fine-tuning matters:
    - Public datasets are mostly from US/European hospitals
    - Your clinic's MRI/CT machines have different calibration
    - Patient demographics differ (age, conditions common in Central Asia)
    - Fine-tuning on even 500-1000 local images significantly improves accuracy
"""

import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.dicom_loader import load_dicom_folder
from src.pipeline.preprocessor import preprocess_study, save_preprocessed
from src.pipeline.dataset import create_dataloaders, split_dataset, compute_class_weights
from src.training.densenet_trainer import (
    DenseNet121Classifier,
    train_one_epoch,
    evaluate,
    get_device,
)
from src.utils.logger import setup_logger
from src.utils.config import get_config
from tqdm import tqdm


def prepare_clinic_data(
    dicom_dir: str,
    labels_file: str,
    output_dir: str,
    target_size: int = 512,
):
    """Preprocess clinic DICOM images and split into train/val/test.

    Args:
        dicom_dir: Directory containing clinic DICOM files.
        labels_file: JSON file with labels: {"filename": ["Pneumonia"], ...}
        output_dir: Output directory for processed images.
        target_size: Image resize target.

    Returns:
        Stats dict with counts.
    """
    logger = setup_logger()
    logger.info(f"Preparing clinic data from {dicom_dir}...")

    # Load and preprocess DICOM files
    studies = load_dicom_folder(dicom_dir)
    if not studies:
        logger.error("No DICOM files found!")
        return None

    images_dir = Path(output_dir) / "all" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for study in tqdm(studies, desc="Preprocessing clinic DICOMs"):
        try:
            processed = preprocess_study(study, target_size=target_size)
            filename = Path(study.file_path).stem
            save_preprocessed(processed, images_dir / f"{filename}.png")
            success += 1
        except Exception as e:
            logger.error(f"Failed: {study.file_path}: {e}")

    logger.info(f"Preprocessed {success}/{len(studies)} clinic images")

    # Split dataset
    stats = split_dataset(
        image_dir=str(images_dir),
        label_file=labels_file,
        output_dir=output_dir,
    )
    return stats


def finetune(
    base_model_path: str,
    dataloaders: dict,
    class_names: list[str],
    epochs: int = 30,
    learning_rate: float = 1e-5,
    class_weights: torch.Tensor = None,
) -> dict:
    """Fine-tune a pre-trained DenseNet121 on clinic data.

    Uses a much lower learning rate than initial training to avoid
    catastrophic forgetting of knowledge from the large public dataset.

    Args:
        base_model_path: Path to pre-trained model checkpoint.
        dataloaders: Dict with "train" and "val" DataLoaders.
        class_names: List of class names.
        epochs: Number of fine-tuning epochs (typically 20-50).
        learning_rate: Learning rate (typically 10x lower than initial training).
        class_weights: Optional class weights for imbalanced data.

    Returns:
        Dict with best model path and metrics.
    """
    logger = setup_logger()
    device = get_device()

    # Load base model
    logger.info(f"Loading base model from {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)

    base_class_names = checkpoint.get("class_names", class_names)
    num_base_classes = len(base_class_names)
    num_new_classes = len(class_names)

    model = DenseNet121Classifier(num_classes=num_base_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # If class count changed, replace the classifier head
    if num_new_classes != num_base_classes:
        logger.info(f"Adapting classifier: {num_base_classes} → {num_new_classes} classes")
        num_features = model.densenet.classifier[0].in_features
        model.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_new_classes),
        )

    model.to(device)

    # Freeze early layers — only fine-tune later layers + classifier
    # This prevents catastrophic forgetting
    freeze_until = "densenet.features.denseblock3"
    frozen_count = 0
    for name, param in model.named_parameters():
        if freeze_until in name:
            break
        param.requires_grad = False
        frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen: {frozen_count} layers | Trainable: {trainable:,}/{total:,} params")

    # Loss & optimizer
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_recall = 0.0
    patience = 10
    patience_counter = 0
    checkpoint_dir = Path("models/densenet_finetuned")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting fine-tuning: {epochs} epochs, lr={learning_rate}")
    logger.info("=" * 60)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, dataloaders["val"], criterion, device, class_names)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val Spec: {val_metrics['specificity']:.4f}"
        )
        
        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall": best_recall,
                "val_metrics": val_metrics,
                "class_names": class_names,
                "base_model": base_model_path,
                "fine_tuned": True,
            }, checkpoint_dir / "best_model_finetuned.pt")
            logger.info(f"  ✓ New best model (Recall: {best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            this is the way to wgo euw

    # Test evaluation
    if "test" in dataloaders:
        test_metrics = evaluate(model, dataloaders["test"], criterion, device, class_names)
        logger.info(f"FINE-TUNED TEST: Recall={test_metrics['recall']:.4f} | "
                     f"Specificity={test_metrics['specificity']:.4f}")
    else:
        test_metrics = None

    return {
        "best_model_path": str(checkpoint_dir / "best_model_finetuned.pt"),
        "best_recall": best_recall,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Sentinel — Fine-tune on Clinic Data")
    parser.add_argument("--base-model", required=True, help="Path to pre-trained model (.pt)")
    parser.add_argument("--dicom-dir", help="Raw clinic DICOM directory (will preprocess)")
    parser.add_argument("--data", help="Already-processed clinic data directory")
    parser.add_argument("--labels", help="Labels JSON file for clinic data")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 0.00001)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    logger = setup_logger()
    config = get_config()
    class_names = config["classification"]["class_names"]

    logger.info("=" * 60)
    logger.info("Sentinel — Fine-Tuning on Clinic Data")
    logger.info("=" * 60)

    # Step 1: Preprocess raw DICOM if provided
    if args.dicom_dir:
        if not args.labels:
            logger.error("--labels is required when using --dicom-dir")
            logger.info("Create a labels.json file:")
            logger.info('  {"scan_001": ["Pneumonia"], "scan_002": ["Normal"], ...}')
            return
        prepare_clinic_data(args.dicom_dir, args.labels, "data/clinic_processed")
        data_dir = "data/clinic_processed"
    elif args.data:
        data_dir = args.data
    else:
        logger.error("Provide either --dicom-dir (raw DICOM) or --data (processed)")
        return

    # Step 2: Create DataLoaders
    dataloaders = create_dataloaders(
        processed_dir=data_dir,
        class_names=class_names,
        batch_size=args.batch_size,
    )

    if "train" not in dataloaders:
        logger.error("No training data found!")
        return

    # Step 3: Compute class weights
    train_labels = Path(data_dir) / "train" / "labels.json"
    class_weights = None
    if train_labels.exists():
        class_weights = compute_class_weights(train_labels, class_names)

    # Step 4: Fine-tune
    results = finetune(
        base_model_path=args.base_model,
        dataloaders=dataloaders,
        class_names=class_names,
        epochs=args.epochs,
        learning_rate=args.lr,
        class_weights=class_weights,
    )

    logger.info("=" * 60)
    logger.info("Fine-Tuning Complete!")
    logger.info(f"Model: {results['best_model_path']}")
    logger.info(f"Best Recall: {results['best_recall']:.4f}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next: Copy the fine-tuned model to your clinic machine:")
    logger.info(f"  cp {results['best_model_path']} models/densenet/best_model.pt")
    logger.info("  python run_server.py")


if __name__ == "__main__":
    main()
