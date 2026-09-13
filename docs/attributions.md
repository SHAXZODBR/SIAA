# SIAA — Sentinel Medical AI · Third-Party Data, Models and Software Attributions

Version 1.0.0 · Document revision 2026-09 · English, with a Russian summary at the end

This document lists the third-party datasets, model weights and major software components used in Sentinel Medical AI, with their licence names as recorded in the source code (`src/inference/model_registry.py`, `scripts/import_public_normals.py`, `requirements.txt`, `desktop-app/package.json`). Licence texts are those published by the respective projects; before redistributing any component, verify the current licence text at the source.

## 1. Datasets

| Dataset | Use in Sentinel | Licence | Attribution |
|---|---|---|---|
| **OpenNeuro ds000221 — MPI-Leipzig Mind-Brain-Body (LEMON)** | Healthy-control **T2-weighted** brain MRI volumes, converted to central axial slices by `scripts/import_public_normals.py` and used as **additional "normal" training data** for the local triage classifier. Attribution is kept in the output `MANIFEST.json` of that script. | **CC0 1.0** (public domain dedication) | Babayan A. et al., "A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults", *Scientific Data* 6, 180308 (2019). OpenNeuro accession ds000221. |
| **Local clinical data (Tashkent)** | Brain MRI studies and radiology reports used to fine-tune **and validate** the triage detector. | Not public — used under the clinic data agreement between SIAA Medical AI and the hospital; patient data never leaves the clinic. | — |

## 2. Model weights

| Model | Role in Sentinel | Validation status | Licence (as recorded) | Source / citation |
|---|---|---|---|---|
| **Study triage — normal vs abnormal (local)** | The only validated detector: study-level sensitivity 0.90 / specificity 0.47 on 178 held-out local patients. `models/brain_triage_finetuned/`, pinned in `MANIFEST.json` (SHA-256 `0d559766ce58…`). | validated | **Proprietary — SIAA Medical AI.** Trained on local Tashkent data under the clinic data agreement, plus CC0 LEMON normals. Warm-started from a ViT-B image-classification backbone (Apache-2.0 lineage). | SIAA fine-tune (ViT-B) on matched Tashkent hospital studies |
| **Brain tumor classifier (2D, 4-class)** | Public Kaggle-derived classifier (glioma / meningioma / pituitary / no tumor). Runs in the panel with status **pending** — over-calls on local scanners. | pending | Apache-2.0 / MIT (per repository) | Hugging Face `andrei-teodor/resnet-pretrained-brain-mri` (primary) and fallbacks `BehradG/resnet-18-MRI-Brain`, `dwiedarioo/vit-base-patch16-224-in21k-brainmri2.0`, `Devarshi/Brain_Tumor_Classification`, others; trained on the Kaggle "Brain Tumor MRI Dataset" (check the Kaggle listing for the dataset's own terms). |
| **TorchXRayVision chest model** (DenseNet121, `densenet121-res224-all`) | Chest X-ray classifier for non-brain studies when installed; not part of the brain pilot. | pending | Apache-2.0 | Cohen J.P. et al., "TorchXRayVision: A library of chest X-ray datasets and models", MIDL 2022. |
| **MONAI `brats_mri_segmentation` bundle** (SegResNet, MONAI Model Zoo, v0.5.4) | 3D brain-tumor segmentation (BraTS 2018 training); status pending, optional. `models/monai_bundles/brats_mri_segmentation/`. | pending | Apache-2.0 (LICENSE file inside the bundle) | MONAI Consortium; BraTS challenge data (Menze et al., Bakas et al.). Alternative SwinUNETR weights: `anhaltai/swinunetrv2_BraTS2021_mini`, Apache-2.0 (Hatamizadeh et al., Swin UNETR, 2022). |
| **Google Gemma 3** (e.g. `gemma3:4b`) via **Ollama** | Local language model for report wording, translation and the *Ask AI* tab. Runs on localhost; optional (template fallback). | n/a (text only) | **Gemma Terms of Use** (Google); Ollama runtime: MIT | Google DeepMind, Gemma 3 model card; Ollama (ollama.com). |
| Ischemic stroke (DWI) — `BTX24/beit-finetuned-stroke-diff-mri` | Experimental; not run in whole-study analysis. | experimental | Apache-2.0 | Community model, BEiT fine-tune |
| Dementia / atrophy — `dhritic9/vit-base-brain-mri-dementia-detection` | Experimental; not run in whole-study analysis. | experimental | Apache-2.0 | ViT-B fine-tune on ADNI-derived images |
| Head CT hemorrhage — `DifeiT/rsna-intracranial-hemorrhage-detection` | Optional CT route; pending. | pending | MIT (as recorded) | RSNA Intracranial Hemorrhage Detection challenge (2019) |
| MedSAM (`wanglab/medsam-vit-base`), MedGemma BraTS reporter, mammography / TB / pneumonia / COVID classifiers | Present in the model registry; not part of the brain pilot deployment. | pending / research | Apache-2.0 (per registry) | See `model_registry.py` cards |

Validation statuses are the ones the software itself reports with every result (`validated` / `pending` / `experimental`).

## 3. Major software components

| Component | Licence | Use |
|---|---|---|
| PyTorch, torchvision | BSD-3-Clause | Inference |
| MONAI | Apache-2.0 | Medical imaging transforms and bundles |
| Hugging Face transformers, safetensors, huggingface_hub | Apache-2.0 | Model loading |
| pydicom, pylibjpeg (+ libjpeg / openjpeg / rle plugins), python-gdcm | MIT / BSD-style | DICOM reading and decompression |
| SimpleITK | Apache-2.0 | N4 bias-field correction |
| HD-BET (DKFZ) | Apache-2.0 | Skull stripping (optional) |
| nibabel | MIT | NIfTI reading (data preparation) |
| FastAPI, uvicorn, pydantic | MIT / BSD | AI server |
| loguru, PyYAML, requests | MIT / Apache-2.0 | Logging, config, HTTP |
| cryptography, bcrypt, python-jose | Apache-2.0 / BSD / Apache-2.0 / MIT | Licensing, password hashing, tokens |
| Electron, React, Vite, TypeScript, Tailwind CSS, zustand, axios | MIT | Desktop application |
| Cornerstone.js (core, tools, DICOM image loader), dicom-parser | MIT | DICOM viewer |
| jsPDF, date-fns, lucide-react | MIT | Desktop utilities |
| SQLite | Public domain | Local database |

## 4. Statement on the validated model

The validated triage detector shipped with Sentinel Medical AI was trained on brain MRI studies and radiology reports from a partner hospital in Tashkent, Uzbekistan, **under the clinic data agreement**, supplemented with CC0 healthy-control data from OpenNeuro ds000221 (LEMON). Patient data remained on premises; the resulting weights are the property of SIAA Medical AI and are licensed to clinics under the Sentinel end-user licence agreement. No other clinical dataset was used for training this model.

---

## Краткое изложение на русском языке

В Sentinel Medical AI используются следующие сторонние данные и модели:

- **OpenNeuro ds000221 (LEMON, MPI Leipzig)** — Т2-взвешенные МРТ головного мозга здоровых добровольцев, лицензия **CC0**; использованы как дополнительные обучающие примеры класса «норма» для локального триаж-классификатора (скрипт `scripts/import_public_normals.py`). Ссылка: Babayan A. et al., *Scientific Data* 6, 180308 (2019).
- **Классификатор опухолей мозга (2D, 4 класса)** — публичная модель с Hugging Face (`andrei-teodor/resnet-pretrained-brain-mri` и резервные репозитории), обучена на Kaggle-наборе «Brain Tumor MRI Dataset»; лицензии Apache-2.0 / MIT согласно репозиториям. Статус в продукте — **валидация не завершена**; на локальных сканерах даёт избыточные срабатывания.
- **TorchXRayVision** (DenseNet121, рентген грудной клетки) — Apache-2.0, Cohen et al., MIDL 2022; не входит в пилот по головному мозгу.
- **MONAI `brats_mri_segmentation`** (SegResNet, BraTS) — Apache-2.0; сегментация опухоли, статус «валидация не завершена», необязательный компонент.
- **Google Gemma 3** через **Ollama** — локальная языковая модель для текста отчётов, перевода и вкладки «Спросить ИИ»; условия использования Gemma (Google), Ollama — MIT. Работает только на локальном компьютере.
- Экспериментальные детекторы (инсульт на DWI, атрофия) и прочие модели реестра — Apache-2.0 / MIT согласно `model_registry.py`; в анализе целого исследования не запускаются либо не входят в пилот.

**Валидированная триаж-модель** (`models/brain_triage_finetuned`, SHA-256 `0d559766ce58…`) обучена и проверена на локальных данных ташкентской клиники **в рамках соглашения о данных с клиникой** (чувствительность 0,90 / специфичность 0,47 на 178 отложенных пациентах); данные пациентов не покидали клинику. Веса модели — собственность SIAA Medical AI.
