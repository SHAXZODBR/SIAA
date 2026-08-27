"""Main entry point for Sentinel Medical AI data pipeline.

Usage:
    python run_pipeline.py preprocess --input data/raw_dicom --output data/processed
    python run_pipeline.py anonymize --input data/raw_dicom --output data/anonymized
    python run_pipeline.py split --input data/processed --output data/processed
    python run_pipeline.py stats --input data/processed
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.dicom_loader import load_dicom, load_dicom_folder
from src.pipeline.anonymizer import anonymize_folder
from src.pipeline.preprocessor import preprocess_study, save_preprocessed
from src.pipeline.dataset import split_dataset
from src.utils.logger import setup_logger
from src.utils.config import get_config


def cmd_preprocess(args):
    """Preprocess all DICOM files: load → normalize → resize → save as PNG."""
    logger = setup_logger()
    config = get_config()
    target_size = config["data"]["image_size"]

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing DICOM files from {input_dir}...")

    studies = load_dicom_folder(input_dir)
    if not studies:
        logger.error("No DICOM files found!")
        return

    success = 0
    failed = 0
    for study in tqdm(studies, desc="Preprocessing"):
        try:
            processed = preprocess_study(study, target_size=target_size)
            filename = Path(study.file_path).stem
            save_preprocessed(processed, output_dir / f"{filename}.png")
            success += 1
        except Exception as e:
            logger.error(f"Failed to preprocess {study.file_path}: {e}")
            failed += 1

    logger.info(f"Preprocessing complete: {success} succeeded, {failed} failed")

    # Generate stats
    print(f"\n{'='*50}")
    print(f"Preprocessing Summary")
    print(f"{'='*50}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files:      {len(studies)}")
    print(f"Succeeded:        {success}")
    print(f"Failed:           {failed}")
    print(f"Image size:       {target_size}x{target_size}")
    print(f"{'='*50}")


def cmd_anonymize(args):
    """Anonymize all DICOM files — remove patient PII."""
    logger = setup_logger()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    logger.info(f"Anonymizing DICOM files from {input_dir}...")
    stats = anonymize_folder(input_dir, output_dir)

    print(f"\nAnonymization: {stats['success']}/{stats['total']} succeeded")


def cmd_split(args):
    """Split preprocessed dataset into train/val/test."""
    logger = setup_logger()
    config = get_config()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    label_file = Path(args.labels) if args.labels else input_dir / "labels.json"

    if not label_file.exists():
        logger.error(f"Label file not found: {label_file}")
        logger.info("Create a labels.json file mapping filenames to class lists:")
        logger.info('  {"filename1": ["Pneumonia"], "filename2": ["Normal"], ...}')
        return

    stats = split_dataset(
        image_dir=input_dir,
        label_file=label_file,
        output_dir=output_dir,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        random_seed=config["split"]["random_seed"],
    )

    print(f"\nDataset split: Train={stats['train']} | Val={stats['val']} | Test={stats['test']}")


def cmd_stats(args):
    """Show dataset statistics."""
    input_dir = Path(args.input)
    images = list(input_dir.rglob("*.png"))
    label_files = list(input_dir.rglob("labels.json"))

    print(f"\n{'='*50}")
    print(f"Dataset Statistics: {input_dir}")
    print(f"{'='*50}")
    print(f"Total images: {len(images)}")

    for split in ["train", "val", "test"]:
        split_dir = input_dir / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*.png")))
            print(f"  {split}: {count} images")

    if label_files:
        for lf in label_files:
            with open(lf) as f:
                labels = json.load(f)
            # Count classes
            class_counts = {}
            for filename, classes in labels.items():
                for c in classes:
                    class_counts[c] = class_counts.get(c, 0) + 1
            print(f"\nClass distribution ({lf.parent.name}):")
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Sentinel Medical AI — Data Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Preprocess
    p_pre = subparsers.add_parser("preprocess", help="Preprocess DICOM files to PNG")
    p_pre.add_argument("--input", "-i", required=True, help="Input DICOM directory")
    p_pre.add_argument("--output", "-o", default="data/processed/all/images", help="Output directory")

    # Anonymize
    p_anon = subparsers.add_parser("anonymize", help="Anonymize DICOM files")
    p_anon.add_argument("--input", "-i", required=True, help="Input DICOM directory")
    p_anon.add_argument("--output", "-o", required=True, help="Output directory")

    # Split
    p_split = subparsers.add_parser("split", help="Split dataset into train/val/test")
    p_split.add_argument("--input", "-i", required=True, help="Preprocessed images directory")
    p_split.add_argument("--output", "-o", required=True, help="Output split directory")
    p_split.add_argument("--labels", "-l", help="Path to labels.json file")

    # Stats
    p_stats = subparsers.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--input", "-i", required=True, help="Dataset directory")

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "anonymize":
        cmd_anonymize(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
