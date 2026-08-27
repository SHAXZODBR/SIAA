"""Interactive label creator for your own clinic DICOM images.

Run this tool to create labels.json for your local DICOM dataset.
A radiologist reviews each image and assigns pathology labels.

Usage:
    python create_labels.py --dicom-dir data/raw_dicom --output data/labels.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.dicom_loader import load_dicom_folder

PATHOLOGY_CLASSES = [
    "Normal",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def main():
    parser = argparse.ArgumentParser(description="Create labels for clinic DICOM images")
    parser.add_argument("--dicom-dir", "-d", required=True, help="Directory with DICOM files")
    parser.add_argument("--output", "-o", default="data/clinic_labels.json", help="Output labels JSON")
    parser.add_argument("--resume", action="store_true", help="Resume labeling (load existing file)")
    args = parser.parse_args()

    # Load existing labels if resuming
    output_path = Path(args.output)
    labels = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            labels = json.load(f)
        print(f"Resuming: {len(labels)} images already labeled")

    # Load DICOM files
    studies = load_dicom_folder(args.dicom_dir)
    remaining = [s for s in studies if Path(s.file_path).stem not in labels]

    print(f"\n{'='*60}")
    print(f"  Sentinel — DICOM Label Creator")
    print(f"{'='*60}")
    print(f"  Total DICOM files: {len(studies)}")
    print(f"  Already labeled:   {len(labels)}")
    print(f"  Remaining:         {len(remaining)}")
    print(f"{'='*60}")
    print(f"\nAvailable classes:")
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        print(f"  {i:2d}. {cls}")
    print(f"\nInstructions:")
    print(f"  - Enter class numbers separated by commas (e.g., '7' for Pneumonia)")
    print(f"  - Enter '0' for Normal (no pathology)")
    print(f"  - Enter 's' to skip")
    print(f"  - Enter 'q' to save and quit")
    print(f"  - Enter multiple: '3,7' for Effusion + Pneumonia")
    print()

    for i, study in enumerate(remaining):
        filename = Path(study.file_path).stem
        print(f"\n[{i+1}/{len(remaining)}] {filename}")
        print(f"  Modality: {study.modality} | Size: {study.rows}x{study.cols} | "
              f"Bits: {study.bits_stored} | Body: {study.body_part}")

        while True:
            answer = input(f"  Labels (numbers/s/q): ").strip().lower()

            if answer == 'q':
                # Save and quit
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(labels, f, indent=2)
                print(f"\nSaved {len(labels)} labels to {output_path}")
                return

            if answer == 's':
                break

            try:
                indices = [int(x.strip()) for x in answer.split(",")]
                selected_classes = []
                for idx in indices:
                    if 0 <= idx < len(PATHOLOGY_CLASSES):
                        selected_classes.append(PATHOLOGY_CLASSES[idx])
                    else:
                        print(f"  Invalid class number: {idx}")
                        continue

                if selected_classes:
                    labels[filename] = selected_classes
                    print(f"  → {', '.join(selected_classes)}")
                    break
            except ValueError:
                print("  Invalid input. Enter numbers separated by commas.")

    # Save final
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nDone! Saved {len(labels)} labels to {output_path}")
    print(f"\nNext step: fine-tune the model:")
    print(f"  python run_finetune.py --base-model models/densenet/best_model.pt "
          f"--dicom-dir {args.dicom_dir} --labels {output_path}")


if __name__ == "__main__":
    main()
