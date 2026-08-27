# Public Medical Imaging Datasets — Sentinel Catalog
**Last verified: May 2026**

This is the complete inventory of free public datasets you can use to:
- **Validate** Sentinel's pretrained models (measure real accuracy)
- **Fine-tune** the brain classifier on a known-correct corpus
- **Demo** to clinics with non-patient data

Datasets are tiered by **size** so you can pick what fits on your MacBook.
Use [`scripts/download_public_datasets.py`](scripts/download_public_datasets.py) to fetch them.

---

## 🟢 TIER 1 — STARTER (Total: ~500 MB)

Small, fast, perfect for first-day validation. **Download all of these first.**

| Dataset | Size | Source | What's inside | Best for |
|---|---|---|---|---|
| **Brain Tumor MRI (Kaggle 4-class)** | 150 MB | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 brain MRIs labeled glioma/meningioma/pituitary/no-tumor | Validate `brain_tumor_class` model |
| **MedMNIST 2D (BrainMNIST)** | 30 MB | [medmnist.com](https://medmnist.com) | 28×28 brain MRI thumbnails, 4-class | Quick smoke test |
| **MedMNIST 2D (BreastMNIST)** | 8 MB | medmnist.com | 28×28 breast ultrasound, benign/malignant | Validate mammography |
| **MedMNIST 2D (ChestMNIST)** | 35 MB | medmnist.com | 14-class chest X-ray | Validate chest model |
| **TCIA NSCLC Sample** | 80 MB | [cancerimagingarchive.net](https://www.cancerimagingarchive.net) | ~10 chest CT studies, full DICOMs | Demo with real DICOM |
| **OpenNeuro IXI Sample** | 200 MB | [openneuro.org](https://openneuro.org) | 10 brain MRI volumes (T1, T2) — healthy | Demo brain workflow |

**License:** All Tier 1 are CC-BY or research-friendly. Commercial deployment OK with attribution.

---

## 🟡 TIER 2 — VALIDATION SET (Total: ~5 GB)

Enough data to run a proper accuracy validation. Download these once you have your demo working.

| Dataset | Size | Source | What's inside | Best for |
|---|---|---|---|---|
| **Brain Tumor Segmentation Sample (BraTS-Lite)** | 1.5 GB | [HuggingFace BraTS subset](https://huggingface.co/datasets/yasserh/brain-tumor-segmentation) | 50 cases × 4 sequences (T1/T1ce/T2/FLAIR) + masks | Validate 3D segmentation |
| **RSNA Intracranial Hemorrhage Sample** | 800 MB | [Kaggle](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) | 2,000 head CT slices labeled by 5 hemorrhage subtypes | Validate `head_ct` model |
| **Calgary-Campinas Brain MRI** | 1.5 GB | [calgarybcmrigroup.ca](https://sites.google.com/view/calgary-campinas-dataset) | 359 T1 brain MRIs from 6 scanner vendors | Test scanner-generalization |
| **CBIS-DDSM Sample** | 1.0 GB | TCIA | Mammography subset (CC + MLO views, BI-RADS labels) | Validate mammography |
| **CheXpert Small (downsampled)** | 11 GB ⚠ | [stanfordmlgroup](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224K chest X-rays, 14 pathologies | Full chest validation |

**Skip CheXpert** unless you specifically want chest as a focus — 11 GB is a lot.

---

## 🔴 TIER 3 — FULL TRAINING (Total: ~80–500 GB) ⚠ ONLY IF NEEDED

These are the corpora used to TRAIN the models you're already using. Don't download unless you're doing serious from-scratch training (you almost certainly aren't).

| Dataset | Size | Source | When you need it |
|---|---|---|---|
| **BraTS 2021 Full** | 50 GB | [Synapse / RACAI](https://www.synapse.org/brats2021) | Re-training SwinUNETR from scratch |
| **NIH ChestX-ray14** | 42 GB | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Re-training chest model |
| **MIMIC-CXR** | 470 GB | [physionet.org/mimic-cxr](https://physionet.org/content/mimic-cxr/) | Has free-text radiology reports — fine-tune MedGemma |
| **ADNI Brain MRI** | 100 GB | [adni.loni.usc.edu](https://adni.loni.usc.edu) | Dementia model fine-tuning |
| **OASIS-3** | 100 GB | [oasis-brains.org](https://www.oasis-brains.org) | Alzheimer's longitudinal data |
| **TCIA Cancer Imaging Archive** | varies | [cancerimagingarchive.net](https://www.cancerimagingarchive.net) | Many cancer-specific subsets |

**Authentication needed:** ADNI, OASIS, MIMIC, MIMIC-CXR all require free registration and a research-use signed agreement. CheXpert just requires an email. Kaggle datasets need a Kaggle account.

---

## 🩻 BY MODALITY — Quick Reference

### Brain MRI
- **Tumor classification:** Brain Tumor MRI Kaggle (Tier 1) → matches `brain_tumor_class` 4-class output
- **Tumor segmentation:** BraTS-Lite (Tier 2) → matches `brain_tumor_seg_3d`
- **Healthy normals:** IXI Sample, Calgary-Campinas (Tier 1/2)
- **Dementia:** OASIS-3, ADNI (Tier 3 — both registration-gated)

### Head CT
- **Hemorrhage:** RSNA ICH Kaggle (Tier 2) → matches `head_ct` model
- **Stroke:** RSNA Stroke 2024 (~10 GB, registration)

### Chest
- **Multi-pathology:** CheXpert (Tier 2), NIH ChestX-ray14 (Tier 3)
- **TB-specific (UZ priority):** [TB Portals](https://tbportals.niaid.nih.gov) — needs registration
- **COVID:** COVID-19 Chest X-ray Database (Kaggle, ~3 GB)

### Mammography
- **Screening:** CBIS-DDSM (Tier 2), RSNA Screening Mammography 2023 (Kaggle, ~370 GB Tier 3)
- **Ultrasound:** Breast Ultrasound Images Kaggle (~150 MB, Tier 1)

### Spine MRI
- **Lumbar:** [SpineWeb](http://spineweb.digitalimaginggroup.ca) (Tier 2)
- **No good public segmentation dataset** — needs custom training

---

## ⚖ LICENSE OVERVIEW

| Dataset | License | Commercial OK? |
|---|---|---|
| Brain Tumor MRI Kaggle | CC0 | ✅ Yes |
| MedMNIST | CC-BY-4.0 | ✅ Yes (with attribution) |
| BraTS | CC-BY-NC | ⚠ Non-commercial only |
| RSNA datasets | Research use | ⚠ Non-commercial only |
| CheXpert | Research use | ⚠ Non-commercial only |
| MIMIC-CXR | PhysioNet Credentialed | ⚠ Research only |
| TCIA | Mostly CC-BY | ✅ Yes (per-collection) |
| ADNI / OASIS | Research use | ⚠ Non-commercial only |
| IXI | CC-BY-SA | ✅ Yes |

**Rule of thumb:** For a paid commercial product (Sentinel = paid), you can:
- ✅ Use any Apache/MIT/CC0/CC-BY model weights freely
- ✅ Use CC-BY datasets if you give attribution in the EULA
- ❌ Re-distribute any dataset you downloaded (only the trained model)
- ❌ Use BraTS / RSNA / MIMIC weights in a paid product unless you re-train your own model on commercial-OK data

The pretrained models in your registry are ALL Apache-2.0 or MIT — safe for paid deployment.

---

## 🎯 RECOMMENDED FIRST DOWNLOAD

Tonight, before the deadline, download Tier 1 only (~500 MB):
```bash
python scripts/download_public_datasets.py tier1
```

This gives you:
- Sample brain MRIs to demo with
- Validation data for the brain classifier
- Sample DICOMs for the desktop app

Skip Tier 2 unless you specifically need accuracy numbers for the sales sheet.
Skip Tier 3 entirely unless you're re-training models (you're not).

---

*Catalog by SIA Medical AI · Updated May 2026*
