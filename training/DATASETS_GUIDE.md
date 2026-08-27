# 🆓 Complete Free Medical Datasets Guide

All datasets below are **100% free**, work on **Kaggle free GPU**, no credit card needed.

---

## 🫁 PHASE 1.1: CHEST X-RAY (DONE ✅)

| Dataset | Kaggle Search | Status |
|---------|---------------|--------|
| NIH Chest X-rays | `nih-chest-xrays` | ✅ Training now |
| RSNA Pneumonia | `rsna-pneumonia-detection` | ✅ Ready |

**Your model:** 14 pathologies, target recall 85%+

---

## 🧠 PHASE 1.2: BRAIN MRI — TUMOR DETECTION

### Recommended (START HERE):

**Dataset:** "Brain Tumor Classification (MRI)" by Sartaj Bhuvaji
**Kaggle search:** `brain-tumor-classification-mri`
**Size:** ~100 MB (super fast download)
**Classes:** Glioma, Meningioma, Pituitary, No Tumor
**Samples:** ~7,000 images
**Training time:** 3-4 hours on T4 x2

**Script:** `training/kaggle_brain_tumor.py` ← **READY**

### Alternative brain MRI datasets:

| Dataset | Kaggle Search | Purpose |
|---------|---------------|---------|
| **Br35H** | `ahmedhamada0/brain-tumor-detection` | Tumor yes/no (simpler) |
| **Figshare Brain Tumor** | `denizkavi1/brain-tumor` | 3,064 MRIs, 3 classes |
| **Brain MRI Detection** | `navoneel/brain-mri-images-for-brain-tumor-detection` | Very small test set |
| **BraTS 2020** | `awsaf49/brats20-dataset-training-validation` | Full challenge dataset (big — 8GB) |
| **Brain Tumor Segmentation** | `awsaf49/brats2021-task1` | Segmentation masks |

---

## 💀 PHASE 1.3: HEAD CT — HEMORRHAGE DETECTION

### Recommended (LIFE-SAVING AI):

**Dataset:** "RSNA Intracranial Hemorrhage Detection"
**Kaggle search:** `rsna-intracranial-hemorrhage-detection`
**Size:** 470 GB (huge — use subset)
**Classes:** 5 hemorrhage types + binary any
**Samples:** 750,000+ CT slices
**Training time:** 6-8 hours on T4 x2 (with 100K subset)

**Script:** `training/kaggle_head_ct.py` ← **READY**

### Alternative head CT datasets (smaller):

| Dataset | Kaggle Search | Purpose |
|---------|---------------|---------|
| **Head CT - hemorrhage** | `felipekitamura/head-ct-hemorrhage` | 200 images, binary |
| **Computed Tomography CT Images** | `kmader/ct-brain-stroke-dataset` | Stroke detection |
| **CT Medical Images** | `kmader/siim-medical-images` | Various |

---

## 🫁 PHASE 1.4: CHEST CT — LUNG CANCER

### Recommended:

**Dataset:** "LUNA16" (Lung Nodule Analysis 2016)
**Kaggle search:** `kmader/luna16-low-resolution-hdf5`
**Size:** ~30 GB
**Classes:** Lung nodule detection
**Samples:** 888 3D CT scans
**Training time:** 8-10 hours (3D processing)

### Alternative chest CT:

| Dataset | Kaggle Search |
|---------|---------------|
| **Chest CT Scan Images** | `mohamedhanyyy/chest-ctscan-images` |
| **COVID-19 CT scans** | `plameneduardo/sarscov2-ctscan-dataset` |
| **Lung Cancer CT** | `mohamedhanyyy/chest-ctscan-images` |

---

## 🫀 FUTURE PHASE: CARDIOLOGY (ECG + Echo)

**NOT chest images — different approach entirely.** Here for reference:

| Dataset | Purpose |
|---------|---------|
| **MIT-BIH Arrhythmia** | ECG classification |
| **PTB-XL ECG Database** | 21,837 ECGs with diagnoses |
| **EchoNet-Dynamic** | Echocardiography videos |

Will build in Phase 2-3.

---

## 🦴 BONUS: Bone X-ray (Fractures)

Easy win, huge market:

| Dataset | Kaggle Search |
|---------|---------------|
| **MURA Bone X-ray** | `cjinny/mura-v11` — 40,000 bone X-rays |
| **Pediatric wrist fractures** | `shashwatwork/bone-break-classification` |

Could be your Phase 1.5.

---

## 🍼 FUTURE: Neonatology (Sepsis Screening)

Not imaging — uses tabular lab data:

| Dataset | Purpose |
|---------|---------|
| **MIMIC-III** | ICU data (requires CITI certification) |
| **PhysioNet Challenges** | Various ICU datasets |

Different tech stack (time-series + tabular). Phase 3.

---

## 📅 SUGGESTED TRAINING SCHEDULE

### Week 1: Chest X-ray (DONE)
- ✅ Training on Kaggle now

### Week 2: Brain Tumor MRI
```
1. Kaggle → New Notebook
2. Add: brain-tumor-classification-mri
3. GPU: T4 x2
4. Paste: kaggle_brain_tumor.py
5. Save Version → Save & Run All
6. Download: best_brain_tumor_model.pt (4 hours)
```

### Week 3: Head CT Hemorrhage
```
1. Kaggle → New Notebook
2. Add: rsna-intracranial-hemorrhage-detection
3. GPU: T4 x2
4. Paste: kaggle_head_ct.py
5. Save Version → Save & Run All
6. Download: best_head_ct_model.pt (8 hours)
```

### Week 4: Test + Deploy All 3 Models
```
1. Copy all 3 .pt files to your Mac
2. Update desktop app modality selector
3. Test with real DICOMs
4. Pitch to first clinic
```

---

## 💰 Total Cost Summary

| Phase | Data Cost | GPU Cost | Total |
|-------|-----------|----------|-------|
| Chest X-ray | $0 | $0 (Kaggle) | $0 |
| Brain MRI | $0 | $0 (Kaggle) | $0 |
| Head CT | $0 | $0 (Kaggle) | $0 |
| Chest CT | $0 | $0 (Kaggle) | $0 |
| Bone X-ray | $0 | $0 (Kaggle) | $0 |
| **All 5 models** | **$0** | **$0** | **$0** |

Uses ~15-20 hours of your 30 hrs/week Kaggle quota.

---

## 🎯 After Training — Deployment

After all 3 models are trained and downloaded to your Mac:

```
sentinel/models/
├── densenet/best_model.pt                 ← Chest X-ray (Phase 1.1)
├── brain_tumor/best_brain_tumor_model.pt  ← Brain MRI (Phase 1.2)
└── head_ct/best_head_ct_model.pt          ← Head CT (Phase 1.3)
```

The integrated pipeline auto-detects modality from DICOM and routes to the correct model.

---

## 🏆 Your Product After Phase 1.1 + 1.2 + 1.3

You'll be able to demo:

- ✅ Upload chest X-ray → detects 14 pathologies
- ✅ Upload brain MRI → detects 4 tumor types
- ✅ Upload head CT → detects 5 hemorrhage types + any

**This is more comprehensive than 95% of medical AI startups in Central Asia.**

---

## 🚀 Start Phase 1.2 RIGHT NOW

While chest X-ray is still training:

1. Open another Kaggle notebook (you have 30 hrs/week quota)
2. Follow the brain tumor guide above
3. Run both trainings in parallel (two separate notebooks)
4. By tomorrow you'll have BOTH models

Kaggle lets you run multiple notebooks — use it!
