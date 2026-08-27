#!/bin/bash
# ================================================================
# Sentinel Medical AI — RunPod Complete Setup Script
# ================================================================
# Run this ONCE on your RunPod pod after starting it.
#
# Prerequisites:
#   - RunPod pod: RTX 4090, PyTorch template, 500GB disk
#   - Datasets already on your hard drive (or download here)
#
# Usage:
#   Option A: Upload datasets from your hard drive first, then:
#     bash runpod_setup.sh --skip-download
#
#   Option B: Download datasets directly on RunPod:
#     bash runpod_setup.sh
# ================================================================

set -e  # Exit on error

echo "============================================================"
echo "  Sentinel Medical AI — RunPod Setup"
echo "============================================================"

WORKSPACE="/workspace"
DATA_DIR="$WORKSPACE/datasets"
PROJECT_DIR="$WORKSPACE/sentinel"

# ======================== STEP 1: System Setup ========================
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 unzip wget > /dev/null 2>&1
echo "  ✓ System dependencies installed"

# ======================== STEP 2: Clone Project ========================
echo ""
echo "[2/6] Setting up project..."
if [ ! -d "$PROJECT_DIR" ]; then
    # If you have a GitHub repo, uncomment:
    # git clone https://github.com/YOUR_USERNAME/sentinel.git $PROJECT_DIR

    # For now, create the directory structure
    mkdir -p $PROJECT_DIR
    echo "  ⚠ Copy your sentinel/ project files to $PROJECT_DIR"
    echo "    Use: scp -r sentinel/ runpod:$PROJECT_DIR/"
fi

cd $PROJECT_DIR

# Install Python dependencies
echo "  Installing Python packages..."
pip install -q torch torchvision monai pydicom numpy opencv-python-headless \
    Pillow scikit-learn scipy mlflow tqdm matplotlib seaborn \
    fastapi uvicorn python-multipart pydantic loguru pyyaml \
    python-dotenv kaggle

echo "  ✓ Python dependencies installed"

# ======================== STEP 3: Download Datasets ========================
if [ "$1" != "--skip-download" ]; then
    echo ""
    echo "[3/6] Downloading datasets..."
    mkdir -p $DATA_DIR

    # --- NIH ChestX-ray14 ---
    echo ""
    echo "  Downloading NIH ChestX-ray14 (42GB)..."
    mkdir -p $DATA_DIR/nih-chestxray14
    cd $DATA_DIR/nih-chestxray14

    # Labels (small files — download first)
    if [ ! -f "Data_Entry_2017_v2020.csv" ]; then
        wget -q "https://nihcc.app.box.com/shared/static/yha43k3hg9pqitgt3frn0fua34k2ys2c/Data_Entry_2017_v2020.csv" 2>/dev/null || echo "  ⚠ Could not auto-download NIH labels. Download manually from: https://nihcc.app.box.com/v/ChestXray-NIHCC"
    fi

    if [ ! -f "BBox_List_2017.csv" ]; then
        wget -q "https://nihcc.app.box.com/shared/static/hgvk6o2kl1h0yw3sgpajknbox3fn8dqp/BBox_List_2017.csv" 2>/dev/null || true
    fi

    # Image archives
    for i in $(seq -w 1 12); do
        if [ ! -f "images_${i}.tar.gz" ] && [ ! -d "images" ]; then
            echo "    Downloading images_${i}.tar.gz..."
            wget -q "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c/images_0${i}.tar.gz" -O "images_${i}.tar.gz" 2>/dev/null || echo "    ⚠ Auto-download may fail for Box links. Use browser download."
        fi
    done

    # Extract
    for f in images_*.tar.gz; do
        [ -f "$f" ] && tar -xzf "$f" && echo "    Extracted: $f"
    done
    echo "  ✓ NIH ChestX-ray14 ready"

    # --- RSNA Pneumonia ---
    echo ""
    echo "  Downloading RSNA Pneumonia (10GB)..."
    mkdir -p $DATA_DIR/rsna-pneumonia
    cd $DATA_DIR/rsna-pneumonia

    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
        kaggle competitions download -c rsna-pneumonia-detection-challenge -p . 2>/dev/null
        unzip -qo "*.zip" 2>/dev/null || true
        echo "  ✓ RSNA Pneumonia ready"
    else
        echo "  ⚠ Kaggle credentials not found!"
        echo "    1. Go to kaggle.com/settings → API → Create New Token"
        echo "    2. Upload kaggle.json:"
        echo "       mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        echo "    3. Then re-run this script"
    fi

    cd $PROJECT_DIR
else
    echo ""
    echo "[3/6] Skipping download (--skip-download flag set)"
    echo "  Make sure datasets are at: $DATA_DIR/"
fi

# ======================== STEP 4: Convert Labels ========================
echo ""
echo "[4/6] Converting dataset labels to Sentinel format..."

cd $PROJECT_DIR

python3 << 'PYTHON_SCRIPT'
import csv
import json
import os
from pathlib import Path

DATA_DIR = "/workspace/datasets"
OUTPUT_DIR = "/workspace/sentinel/data"

os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

# ===== NIH ChestX-ray14 Labels =====
nih_csv = f"{DATA_DIR}/nih-chestxray14/Data_Entry_2017_v2020.csv"
if os.path.exists(nih_csv):
    print("  Converting NIH ChestX-ray14 labels...")
    labels = {}
    with open(nih_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["Image Index"].replace(".png", "")
            findings = row["Finding Labels"].split("|")
            findings = [f.strip() for f in findings if f.strip() != "No Finding"]
            if findings:
                labels[filename] = findings
            else:
                labels[filename] = ["Normal"]

    with open(f"{OUTPUT_DIR}/nih_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    # Count classes
    class_counts = {}
    for fname, classes in labels.items():
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1

    print(f"    {len(labels)} images, {len(class_counts)} classes")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {cls}: {count}")
    print(f"  ✓ Saved to {OUTPUT_DIR}/nih_labels.json")

# ===== RSNA Pneumonia Labels =====
rsna_csv = f"{DATA_DIR}/rsna-pneumonia/stage_2_train_labels.csv"
if os.path.exists(rsna_csv):
    print("\n  Converting RSNA Pneumonia labels...")
    labels = {}
    with open(rsna_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["patientId"]
            target = int(row["Target"])
            if patient_id not in labels:
                labels[patient_id] = ["Pneumonia"] if target == 1 else ["Normal"]

    with open(f"{OUTPUT_DIR}/rsna_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    pneumonia_count = sum(1 for v in labels.values() if "Pneumonia" in v)
    normal_count = sum(1 for v in labels.values() if "Normal" in v)
    print(f"    {len(labels)} images: {pneumonia_count} pneumonia, {normal_count} normal")
    print(f"  ✓ Saved to {OUTPUT_DIR}/rsna_labels.json")

# ===== CheXpert Labels =====
chexpert_csv = f"{DATA_DIR}/chexpert/CheXpert-v1.0/train.csv"
if os.path.exists(chexpert_csv):
    print("\n  Converting CheXpert labels...")
    chexpert_classes = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    labels = {}
    with open(chexpert_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["Path"]
            filename = Path(path).stem
            findings = []
            for cls in chexpert_classes:
                val = row.get(cls, "0")
                if val == "1.0" or val == "1":
                    findings.append(cls)
            if not findings:
                findings = ["Normal"]
            labels[filename] = findings

    with open(f"{OUTPUT_DIR}/chexpert_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"    {len(labels)} images")
    print(f"  ✓ Saved to {OUTPUT_DIR}/chexpert_labels.json")

print("\n  ✓ All label conversions complete")
PYTHON_SCRIPT

echo "  ✓ Labels converted"

# ======================== STEP 5: Preprocess ========================
echo ""
echo "[5/6] Preprocessing images..."

cd $PROJECT_DIR

python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, "/workspace/sentinel")

import os
import csv
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "/workspace/datasets"
OUTPUT_DIR = "/workspace/sentinel/data/processed/all/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = 512
processed_count = 0

# ===== Process NIH images (already PNG) =====
nih_dir = f"{DATA_DIR}/nih-chestxray14/images"
if os.path.isdir(nih_dir):
    print("Processing NIH ChestX-ray14 images...")
    files = [f for f in os.listdir(nih_dir) if f.endswith(".png")]
    for fname in tqdm(files, desc="NIH"):
        src = os.path.join(nih_dir, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(dst):
            continue
        try:
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Resize with padding
            h, w = img.shape
            scale = TARGET_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (new_w, new_h))
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            y_off = (TARGET_SIZE - new_h) // 2
            x_off = (TARGET_SIZE - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            cv2.imwrite(dst, canvas)
            processed_count += 1
        except Exception as e:
            print(f"  Error: {fname}: {e}")
    print(f"  ✓ NIH: {processed_count} images processed")

# ===== Process RSNA images (DICOM) =====
rsna_dir = f"{DATA_DIR}/rsna-pneumonia/stage_2_train_images"
if os.path.isdir(rsna_dir):
    print("\nProcessing RSNA Pneumonia images (DICOM → PNG)...")
    import pydicom
    files = [f for f in os.listdir(rsna_dir) if f.endswith(".dcm")]
    rsna_count = 0
    for fname in tqdm(files, desc="RSNA"):
        stem = fname.replace(".dcm", "")
        dst = os.path.join(OUTPUT_DIR, f"{stem}.png")
        if os.path.exists(dst):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(rsna_dir, fname), force=True)
            img = ds.pixel_array.astype(np.float32)
            # Normalize
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            # Resize with padding
            h, w = img.shape[:2]
            scale = TARGET_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (new_w, new_h))
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            y_off = (TARGET_SIZE - new_h) // 2
            x_off = (TARGET_SIZE - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            cv2.imwrite(dst, canvas)
            rsna_count += 1
        except Exception as e:
            pass
    print(f"  ✓ RSNA: {rsna_count} images processed")
    processed_count += rsna_count

print(f"\n✓ Total processed: {processed_count} images")
print(f"  Output: {OUTPUT_DIR}")
PYTHON_SCRIPT

echo "  ✓ Preprocessing complete"

# ======================== STEP 6: Merge Labels & Split ========================
echo ""
echo "[6/6] Merging labels and splitting dataset..."

cd $PROJECT_DIR

python3 << 'PYTHON_SCRIPT'
import json
import os
import sys
sys.path.insert(0, "/workspace/sentinel")

DATA_DIR = "/workspace/sentinel/data"
PROCESSED_DIR = f"{DATA_DIR}/processed/all/images"

# Merge all label files into one
merged = {}
for label_file in ["nih_labels.json", "rsna_labels.json", "chexpert_labels.json"]:
    path = os.path.join(DATA_DIR, label_file)
    if os.path.exists(path):
        with open(path) as f:
            labels = json.load(f)
        # Only keep labels for images we actually processed
        for filename, classes in labels.items():
            img_path = os.path.join(PROCESSED_DIR, f"{filename}.png")
            if os.path.exists(img_path):
                merged[filename] = classes

print(f"Merged labels: {len(merged)} images with labels")

# Save merged labels
merged_path = os.path.join(DATA_DIR, "labels.json")
with open(merged_path, "w") as f:
    json.dump(merged, f, indent=2)

# Class distribution
class_counts = {}
for fname, classes in merged.items():
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1

print("\nClass distribution:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")

# Split dataset
from src.pipeline.dataset import split_dataset
from src.utils.config import get_config

config = get_config()
stats = split_dataset(
    image_dir=PROCESSED_DIR,
    label_file=merged_path,
    output_dir=f"{DATA_DIR}/processed",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)

print(f"\nDataset split complete:")
print(f"  Train: {stats['train']}")
print(f"  Val:   {stats['val']}")
print(f"  Test:  {stats['test']}")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "  ✓ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "  Now run training:"
echo "    cd $PROJECT_DIR"
echo "    python run_training.py --data data/processed --epochs 100"
echo ""
echo "  Training will take 8-16 hours on RTX 4090."
echo "  You can close the browser — training continues on RunPod."
echo ""
echo "  After training, download the model:"
echo "    scp runpod:/workspace/sentinel/models/densenet/best_model.pt ./models/"
echo ""
echo "============================================================"
