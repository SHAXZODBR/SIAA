# Sentinel Medical AI — Dataset Download Guide
## Complete Step-by-Step Instructions

---

## STEP 1: Create Accounts (Do This First — Some Need Approval)

### 1.1 Kaggle Account (for RSNA dataset)
1. Go to https://www.kaggle.com
2. Sign up / sign in
3. Go to https://www.kaggle.com/settings → API → "Create New Token"
4. This downloads `kaggle.json` — save it somewhere safe
5. Go to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
6. Click "I Understand and Accept" (required to download)

### 1.2 PhysioNet Account (for MIMIC-CXR)
1. Go to https://physionet.org/register/
2. Create account
3. Complete the CITI training course (required for medical data access)
   - Go to https://about.citiprogram.org/
   - Complete "Data or Specimens Only Research" course
   - Upload certificate to PhysioNet
4. Go to https://physionet.org/content/mimic-cxr-jpg/2.1.0/
5. Click "Request Access" — approval takes 1-3 days
6. **Note:** MIMIC-CXR is the largest (400GB+). You can skip this for Phase 1.

### 1.3 Stanford AIMI Account (for CheXpert)
1. Go to https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
2. Register with your email
3. Accept data use agreement
4. You'll get a download link via email

---

## STEP 2: Download All Datasets to Your Hard Drive

### Recommended: Use a PC with fast internet + external hard drive (1TB minimum)

### 2.1 NIH ChestX-ray14 (112,120 images — 42GB)

**Option A: Direct download (browser)**
1. Go to https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download all files:
   - `images_001.tar.gz` through `images_012.tar.gz` (12 files)
   - `Data_Entry_2017_v2020.csv` (labels file — IMPORTANT)
   - `BBox_List_2017.csv` (bounding boxes)
3. Save to: `YOUR_DRIVE/datasets/nih-chestxray14/`

**Option B: Command line (faster)**
```bash
mkdir -p /Volumes/YOUR_DRIVE/datasets/nih-chestxray14
cd /Volumes/YOUR_DRIVE/datasets/nih-chestxray14

# Download all 12 image archives
for i in $(seq -w 1 12); do
    wget "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c/images_0${i}.tar.gz" -O images_${i}.tar.gz
done

# Download labels
wget "https://nihcc.app.box.com/shared/static/yha43k3hg9pqitgt3frn0fua34k2ys2c/Data_Entry_2017_v2020.csv"
wget "https://nihcc.app.box.com/shared/static/hgvk6o2kl1h0yw3sgpajknbox3fn8dqp/BBox_List_2017.csv"

# Extract all archives
for f in images_*.tar.gz; do
    tar -xzf "$f"
    echo "Extracted: $f"
done
```

### 2.2 RSNA Pneumonia Detection (30,000 images — 10GB)

**Using Kaggle CLI:**
```bash
# Install kaggle CLI
pip install kaggle

# Set up credentials (copy kaggle.json you downloaded earlier)
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download
mkdir -p /Volumes/YOUR_DRIVE/datasets/rsna-pneumonia
cd /Volumes/YOUR_DRIVE/datasets/rsna-pneumonia
kaggle competitions download -c rsna-pneumonia-detection-challenge

# Extract
unzip rsna-pneumonia-detection-challenge.zip
```

**Using browser (if CLI doesn't work):**
1. Go to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
2. Click "Download All"
3. Save to: `YOUR_DRIVE/datasets/rsna-pneumonia/`
4. Unzip the file

### 2.3 CheXpert (224,316 images — 11GB download, 439GB extracted)

```bash
mkdir -p /Volumes/YOUR_DRIVE/datasets/chexpert
cd /Volumes/YOUR_DRIVE/datasets/chexpert

# After receiving the download link via email from Stanford:
wget "YOUR_DOWNLOAD_LINK_FROM_EMAIL" -O CheXpert-v1.0.zip

# Extract (this will take a while — 439GB)
unzip CheXpert-v1.0.zip
```

**Note:** CheXpert has its own `train.csv` and `valid.csv` label files included.

### 2.4 MIMIC-CXR (227,827 images — 300GB download)

**Only after PhysioNet approval:**
```bash
mkdir -p /Volumes/YOUR_DRIVE/datasets/mimic-cxr
cd /Volumes/YOUR_DRIVE/datasets/mimic-cxr

# Download using wget with PhysioNet credentials
wget -r -N -c -np \
    --user YOUR_PHYSIONET_USERNAME \
    --ask-password \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

**Note:** This is 300GB+ download. Skip for Phase 1 if storage is tight.

---

## STEP 3: Verify Downloads

After downloading, your hard drive should look like this:

```
YOUR_DRIVE/datasets/
├── nih-chestxray14/           (42GB)
│   ├── images/                ← 112,120 PNG files
│   ├── Data_Entry_2017_v2020.csv  ← labels
│   └── BBox_List_2017.csv     ← bounding boxes
│
├── rsna-pneumonia/            (10GB)
│   ├── stage_2_train_images/  ← 26,684 DICOM files
│   ├── stage_2_test_images/   ← 3,000 DICOM files
│   ├── stage_2_train_labels.csv ← labels
│   └── stage_2_detailed_class_info.csv
│
├── chexpert/                  (439GB)
│   ├── CheXpert-v1.0/
│   │   ├── train/             ← 224,316 images
│   │   ├── valid/             ← 234 images
│   │   ├── train.csv          ← labels
│   │   └── valid.csv
│
└── mimic-cxr/                 (400GB+) — OPTIONAL Phase 1
    ├── files/
    ├── mimic-cxr-2.0.0-chexpert.csv
    └── mimic-cxr-2.0.0-metadata.csv
```

Run this to verify file counts:
```bash
echo "=== NIH ChestX-ray14 ==="
ls datasets/nih-chestxray14/images/ | wc -l

echo "=== RSNA Pneumonia ==="
ls datasets/rsna-pneumonia/stage_2_train_images/ | wc -l

echo "=== CheXpert ==="
find datasets/chexpert/ -name "*.jpg" | wc -l

echo "=== MIMIC-CXR ==="
find datasets/mimic-cxr/ -name "*.jpg" | wc -l
```

---

## STEP 4: Upload to RunPod & Train

See `runpod_setup.sh` script — it automates everything on RunPod.

---

## STEP 5: Fine-Tune with Your Own Clinic DICOM Images

After training on public data, you fine-tune on your local labeled DICOMs
to adapt the model to Uzbekistan clinic equipment and patient population.

See `run_finetune.py` script for this step.

---

## Storage Requirements Summary

| Dataset | Download | Extracted | Priority |
|---------|----------|-----------|----------|
| NIH ChestX-ray14 | 42 GB | 45 GB | **Must have** |
| RSNA Pneumonia | 10 GB | 12 GB | **Must have** |
| CheXpert | 11 GB | 439 GB | Nice to have |
| MIMIC-CXR | 300 GB | 400 GB+ | Phase 2 |
| **Minimum needed** | **52 GB** | **57 GB** | NIH + RSNA only |
| **Full download** | **363 GB** | **896 GB** | All 4 datasets |

**Recommendation:** Start with NIH + RSNA only (57GB total). That's 142,000 images — more than enough for a strong model.
