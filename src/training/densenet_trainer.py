"""DenseNet121 (CheXNet) classification trainer for Sentinel Medical AI.

Trains a DenseNet121 model for multi-label chest pathology classification.
Stanford-validated architecture achieving 85-92% recall on medical imaging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import numpy as np
import time
import json
try:
    import mlflow
except ImportError:
    mlflow = None  # MLflow is optional — only needed for training
from pathlib import Path
from typing import Optional
try:
    from sklearn.metrics import (
        recall_score,
        precision_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    # sklearn optional — only needed for training
    recall_score = precision_score = f1_score = roc_auc_score = None
    confusion_matrix = classification_report = None
from loguru import logger
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # fallback if tqdm not installed
        return x


class DenseNet121Classifier(nn.Module):
    """DenseNet121-based medical image classifier (CheXNet architecture).

    Replaces the final classifier with a custom head for multi-label
    pathology classification.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        # Load pretrained DenseNet121
        self.densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )

        # Modify first conv layer to accept 1-channel (grayscale) input
        # Original: Conv2d(3, 64, 7, 2, 3)
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # If pretrained, average the RGB weights to initialize grayscale weights
        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.densenet.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Replace classifier for multi-label classification
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.densenet(x)


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train model for one epoch.

    Returns:
        Dict with loss and metric values.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Binary predictions at 0.5 threshold
    binary_preds = (all_preds >= 0.5).astype(int)

    epoch_loss = running_loss / len(dataloader.dataset)
    recall = recall_score(all_labels, binary_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, binary_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except ValueError:
        auc = 0.0

    return {
        "loss": epoch_loss,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc_roc": auc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate model on a dataset.

    Returns:
        Dict with loss, metrics, and per-class results.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)

    epoch_loss = running_loss / len(dataloader.dataset)
    recall = recall_score(all_labels, binary_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, binary_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)
    specificity = compute_specificity(all_labels, binary_preds)

    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except ValueError:
        auc = 0.0

    results = {
        "loss": epoch_loss,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc_roc": auc,
        "specificity": specificity,
    }

    # Per-class metrics
    if class_names:
        per_class = {}
        for i, name in enumerate(class_names):
            cls_recall = recall_score(
                all_labels[:, i], binary_preds[:, i], zero_division=0
            )
            cls_precision = precision_score(
                all_labels[:, i], binary_preds[:, i], zero_division=0
            )
            per_class[name] = {"recall": cls_recall, "precision": cls_precision}
        results["per_class"] = per_class

    return results


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged specificity (True Negative Rate)."""
    specificities = []
    for i in range(y_true.shape[1]):
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        if tn + fp > 0:
            specificities.append(tn / (tn + fp))
    return np.mean(specificities) if specificities else 0.0


def train(
    dataloaders: dict,
    config: dict,
    class_names: list[str],
    class_weights: Optional[torch.Tensor] = None,
) -> dict:
    """Full training loop for DenseNet121 classification.

    Args:
        dataloaders: Dict with "train" and "val" DataLoaders.
        config: Classification config dict.
        class_names: List of class names.
        class_weights: Optional tensor of class weights for imbalanced data.

    Returns:
        Dict with best model path and final metrics.
    """
    device = get_device()

    # Create model
    model = DenseNet121Classifier(
        num_classes=len(class_names),
        pretrained=config.get("pretrained", True),
        in_channels=config.get("input_channels", 1),
    ).to(device)

    logger.info(f"DenseNet121 created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function — BCEWithLogitsLoss for multi-label classification
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer — AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Cosine annealing scheduler
    epochs = config.get("epochs", 100)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Early stopping
    patience = config.get("early_stopping_patience", 15)
    best_recall = 0.0
    patience_counter = 0
    checkpoint_dir = Path(config.get("checkpoint_dir", "models/densenet"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # MLflow tracking
    mlflow.set_tracking_uri(config.get("mlflow_uri", "mlruns"))
    mlflow.set_experiment("sentinel_densenet121")

    with mlflow.start_run(run_name="densenet121_training"):
        mlflow.log_params({
            "num_classes": len(class_names),
            "epochs": epochs,
            "batch_size": config.get("batch_size", 16),
            "learning_rate": config.get("learning_rate", 1e-4),
            "weight_decay": config.get("weight_decay", 1e-5),
            "image_size": config.get("image_size", 512),
            "patience": patience,
        })

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            start_time = time.time()

            # Train
            train_metrics = train_one_epoch(
                model, dataloaders["train"], criterion, optimizer, device
            )

            # Validate
            val_metrics = evaluate(
                model, dataloaders["val"], criterion, device, class_names
            )

            scheduler.step()

            epoch_time = time.time() - start_time

            # Log metrics
            logger.info(
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f} | "
                f"Val Specificity: {val_metrics['specificity']:.4f} | "
                f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
                "val_auc_roc": val_metrics["auc_roc"],
                "val_f1": val_metrics["f1"],
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            # Early stopping on validation recall (medical priority)
            if val_metrics["recall"] > best_recall:
                best_recall = val_metrics["recall"]
                patience_counter = 0

                # Save best model
                best_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_recall": best_recall,
                    "val_metrics": val_metrics,
                    "class_names": class_names,
                    "config": config,
                }, best_path)
                logger.info(f"  ✓ New best model saved (Recall: {best_recall:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    break

        # Final test evaluation
        if "test" in dataloaders:
            logger.info("Running final evaluation on test set...")
            test_metrics = evaluate(
                model, dataloaders["test"], criterion, device, class_names
            )
            logger.info(
                f"TEST RESULTS: Recall={test_metrics['recall']:.4f} | "
                f"Specificity={test_metrics['specificity']:.4f} | "
                f"AUC={test_metrics['auc_roc']:.4f}"
            )
            mlflow.log_metrics({
                "test_recall": test_metrics["recall"],
                "test_specificity": test_metrics["specificity"],
                "test_auc_roc": test_metrics["auc_roc"],
                "test_f1": test_metrics["f1"],
            })

            # Check targets
            if test_metrics["recall"] < 0.85:
                logger.warning(
                    f"⚠ Recall {test_metrics['recall']:.4f} is BELOW target 0.85!\n"
                    f"Recommendations:\n"
                    f"  1. Increase augmentation intensity\n"
                    f"  2. Adjust class weights (increase for rare classes)\n"
                    f"  3. Lower classification threshold from 0.5\n"
                    f"  4. Add more training data (especially for low-recall classes)\n"
                    f"  5. Try longer training (increase patience)\n"
                )

    return {
        "best_model_path": str(checkpoint_dir / "best_model.pt"),
        "best_recall": best_recall,
        "test_metrics": test_metrics if "test" in dataloaders else None,
    }
