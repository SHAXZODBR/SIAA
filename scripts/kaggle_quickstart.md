# Kaggle Quickstart — Get Real Brain MRI Data Fast

When you need real (not synthetic) brain MRI data for validation or fine-tuning, Kaggle is the fastest source. Takes ~10 minutes start to finish.

## One-time setup (5 minutes)

```bash
# 1. Install the Kaggle CLI
source ~/venv/bin/activate
pip install -q kaggle

# 2. Get your API token from https://www.kaggle.com/settings → "Create New Token"
#    Downloads kaggle.json — move it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Recommended datasets (with real disk costs)

```bash
cd /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel

# Brain Tumor MRI 4-class — 7,023 images (~150 MB, ~30 sec to download)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset \
    -p data/public/brain_tumor_kaggle --unzip

# Brain Tumor with bounding boxes — for segmentation (~50 MB)
kaggle datasets download -d ahmedhamada0/brain-tumor-detection \
    -p data/public/brain_tumor_bbox --unzip

# RSNA Intracranial Hemorrhage Detection — head CT (~2 GB, ~3 min)
kaggle competitions download -c rsna-intracranial-hemorrhage-detection \
    -p data/public/rsna_ich

# Breast Ultrasound Images — for mammography model (~150 MB)
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset \
    -p data/public/breast_ultrasound --unzip

# Brain MRI for Brain Tumor Segmentation — BraTS-derived (~1.5 GB)
kaggle datasets download -d awsaf49/brats20-dataset-training-validation \
    -p data/public/brats20 --unzip
```

## After download — restructure for fine-tuning

The Brain Tumor MRI dataset comes in this layout:
```
data/public/brain_tumor_kaggle/
├── Training/
│   ├── glioma/      ~1300 images
│   ├── meningioma/  ~1300 images
│   ├── notumor/     ~1500 images
│   └── pituitary/   ~1450 images
└── Testing/
    ├── glioma/      ~300 images
    ├── meningioma/  ~300 images
    ├── notumor/     ~400 images
    └── pituitary/   ~300 images
```

**Re-organize for our fine-tuner** (it expects `train/<class>/` + `val/<class>/` with our class names):

```bash
cd data/public/brain_tumor_kaggle

# Map Kaggle class names → our class names + restructure
mkdir -p ready/train ready/val
for kaggle_cls in glioma meningioma notumor pituitary; do
    case "$kaggle_cls" in
        glioma)     our="glioma_tumor" ;;
        meningioma) our="meningioma_tumor" ;;
        notumor)    our="no_tumor" ;;
        pituitary)  our="pituitary_tumor" ;;
    esac
    mkdir -p "ready/train/$our" "ready/val/$our"
    cp Training/$kaggle_cls/*.jpg "ready/train/$our/" 2>/dev/null
    cp Testing/$kaggle_cls/*.jpg  "ready/val/$our/" 2>/dev/null
done

cd ../../..  # back to repo root
```

## Now validate or fine-tune

```bash
# Just check accuracy on the 1300 test images
python scripts/validate_accuracy.py \
    --data data/public/brain_tumor_kaggle/ready/val \
    --model brain_tumor_class

# Or fine-tune for 15 epochs on YOUR machine (M-series Mac handles this in ~30 min)
python training/finetune_brain_classifier.py \
    --data data/public/brain_tumor_kaggle/ready \
    --epochs 15 \
    --batch-size 16
```

## Total disk after all 5 datasets

| Dataset | Size |
|---|---|
| Brain Tumor MRI 4-class | 150 MB |
| Brain Tumor with bbox | 50 MB |
| RSNA ICH | 2 GB |
| Breast Ultrasound | 150 MB |
| BraTS20 | 1.5 GB |
| **Total** | **~3.85 GB** |

Plenty of room on your 794 GB free.
