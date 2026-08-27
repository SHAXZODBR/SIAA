"""Main entry point for DenseNet121 model training.

Usage:
    python run_training.py --data data/processed --epochs 100
    python run_training.py --data data/processed --resume models/densenet/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.dataset import create_dataloaders, compute_class_weights
from src.training.densenet_trainer import train
from src.utils.logger import setup_logger
from src.utils.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Sentinel — DenseNet121 Training")
    parser.add_argument("--data", "-d", default="data/processed", help="Processed dataset directory")
    parser.add_argument("--epochs", "-e", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    logger = setup_logger()
    config = get_config()
    cls_config = config["classification"]

    # Apply CLI overrides
    if args.epochs:
        cls_config["epochs"] = args.epochs
    if args.batch_size:
        cls_config["batch_size"] = args.batch_size
    if args.lr:
        cls_config["learning_rate"] = args.lr

    class_names = cls_config["class_names"]

    logger.info("="*60)
    logger.info("Sentinel Medical AI — DenseNet121 Training")
    logger.info("="*60)
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Classes: {len(class_names)}")
    logger.info(f"Epochs: {cls_config['epochs']}")
    logger.info(f"Batch size: {cls_config['batch_size']}")
    logger.info(f"Learning rate: {cls_config['learning_rate']}")

    # Create DataLoaders
    dataloaders = create_dataloaders(
        processed_dir=args.data,
        class_names=class_names,
        batch_size=cls_config["batch_size"],
        image_size=cls_config["image_size"],
    )

    if "train" not in dataloaders:
        logger.error("No training data found! Run the pipeline first:")
        logger.error("  python run_pipeline.py preprocess -i data/raw_dicom -o data/processed/all/images")
        logger.error("  python run_pipeline.py split -i data/processed/all/images -o data/processed -l data/labels.json")
        return

    # Compute class weights for imbalanced dataset
    train_label_file = Path(args.data) / "train" / "labels.json"
    class_weights = None
    if train_label_file.exists():
        class_weights = compute_class_weights(train_label_file, class_names)
        logger.info(f"Using computed class weights for imbalanced data")

    # Train
    results = train(
        dataloaders=dataloaders,
        config=cls_config,
        class_names=class_names,
        class_weights=class_weights,
    )

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info(f"Best model: {results['best_model_path']}")
    logger.info(f"Best recall: {results['best_recall']:.4f}")
    if results.get("test_metrics"):
        logger.info(f"Test recall: {results['test_metrics']['recall']:.4f}")
        logger.info(f"Test specificity: {results['test_metrics']['specificity']:.4f}")
        logger.info(f"Test AUC-ROC: {results['test_metrics']['auc_roc']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
