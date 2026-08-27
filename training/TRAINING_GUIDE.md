# Sentinel — Training Guide (cloud-ready)

How to download datasets, prep them, and train brain-disease detectors on
Kaggle/Colab/RunPod GPU. The whole pipeline is ready — drop in a dataset, run
two commands, get a model + metrics.

---

## 0. Data folder convention

Put raw downloads anywhere, then prep them into this layout (the trainer needs it):

```
data/ft/<disease>/
  train/<classA>/*.png|jpg
  train/<classB>/...
  val/<classA>/...
  val/<classB>/...
```

Already prepped on this machine: `data/ft/alzheimer`, `data/ft/hemorrhage` (CT-ICH).

---

## 1. One-time: Kaggle API (for any Kaggle download)

```bash
pip install kaggle
# kaggle.com → Account → "Create New API Token" → downloads kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## 2. Download the big datasets

### 🩸 RSNA Intracranial Hemorrhage (the big hemorrhage set, ~180 GB DICOM)
1. **First accept the rules** (required) at
   `https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/rules`
2. Download:
   ```bash
   kaggle competitions download -c rsna-intracranial-hemorrhage-detection
   unzip rsna-intracranial-hemorrhage-detection.zip -d data/raw/rsna
   ```
   ⚠️ Huge. **Smaller option**: search Kaggle for a PNG re-upload, e.g.
   `kaggle datasets download -d <user>/rsna-ich-png` (a few GB).
3. Prep (handles DICOM windowing + patient split):
   ```bash
   python training/prep_rsna_hemorrhage.py --src data/raw/rsna --limit 60000
   ```

### 🧠 ISLES 2022 (ischemic stroke, MRI + masks — segmentation)
1. Register/join at `https://isles22.grand-challenge.org/`
2. Data is on Zenodo:
   ```bash
   pip install zenodo_get
   zenodo_get 10.5281/zenodo.7153326      # ISLES'22 release
   ```
   → NIfTI volumes (DWI/ADC/FLAIR) + lesion masks. Segmentation → train on GPU.

### ⚪ WMH Segmentation Challenge 2017 (white-matter lesions — segmentation)
1. Register at `https://wmh.isi.uu.nl/` (free) → download the training set.
   → FLAIR/T1 NIfTI + WMH masks. Segmentation → GPU.

> Classification sets (RSNA) train on your Mac OR cloud. **Segmentation sets
> (ISLES, WMH, BraTS, LGG) realistically need a GPU** — use Kaggle (free) or
> RunPod (~$0.40/hr).

---

## 3. Train on cloud GPU (Kaggle/Colab/RunPod)

Upload this repo (or just `training/` + `scripts/` + your `data/ft/<disease>`)
to a Kaggle notebook with **GPU T4 ×2** enabled, then:

```bash
pip install -r requirements.txt
# classification (tumor, hemorrhage, alzheimer, …)
python training/finetune_brain_classifier.py \
    --data data/ft/<disease> \
    --classes "classA,classB,..." \
    --base-model google/vit-base-patch16-224 \
    --balance \                 # ALWAYS for imbalanced data (hemorrhage etc.)
    --device cuda \             # cuda on Kaggle/RunPod; mps on Mac
    --epochs 10 --batch-size 32 \
    --output models/<disease>_finetuned
```

Then full metrics (accuracy + precision + recall + F1 + confusion):

```bash
python scripts/eval_classifier.py \
    --model-dir models/<disease>_finetuned \
    --data data/ft/<disease>/val --title "<disease>"
```

Download `models/<disease>_finetuned/` back — the app's registry auto-loads
`models/<modality>_finetuned/` and serves it immediately.

---

## 4. When your REAL clinic data arrives (the important one)

1. Organize labeled scans into `data/ft/clinic_<finding>/{train,val}/<class>/`
   (or one-click labels → I'll script the conversion).
2. Run the **same** train command with `--base-model` pointed at the public
   model you already trained (so it fine-tunes *on top* of it).
3. Eval → that's your **real, local, validated number** — the one that matters.

Use a **medical-pretrained backbone** (RadImageNet, `github.com/BMEII-AI/RadImageNet`)
as `--base-model` to need fewer local labels.

---

## Status of detectors

| Disease | Data | Trained? | Notes |
|---|---|---|---|
| Tumor (4-class) | local timri | ✅ 92% | live |
| Hemorrhage (binary) | CT-ICH | ✅ 45% recall | needs RSNA for better |
| Dementia (4-class) | OASIS-derived | ✅ 68% | hard; patient-split needed |
| Stroke | ISLES (download) | ⏳ | segmentation, GPU |
| White-matter/MS | WMH (download) | ⏳ | segmentation, GPU |
