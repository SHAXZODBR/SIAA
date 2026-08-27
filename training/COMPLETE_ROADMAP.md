# 🗺️ Sentinel Medical AI — Complete Training Roadmap

## From Zero to Production — All 4 Phases

---

## 📊 The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PHASE 1: CHEST X-RAY BASELINE  ←  YOU ARE HERE                │
│   (DenseNet121 on public data)                                  │
│   Cost: $0 (Kaggle free)   Time: 6-8 hrs                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PHASE 2: CLINIC FINE-TUNING                                   │
│   (DenseNet121 + your Uzbekistan clinic data)                   │
│   Cost: $0 (Colab free)   Time: 2-3 hrs                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PHASE 3: GEMMA REPORT GENERATION                              │
│   (Gemma 3 VL fine-tuned on radiology reports)                  │
│   Cost: $0 (Kaggle free)   Time: 4-6 hrs                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PHASE 4: BRAIN MRI EXPANSION                                  │
│   (New DenseNet/ViT model for brain imaging)                    │
│   Cost: $0-10   Time: 8-12 hrs                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

  TOTAL: 3-4 weeks elapsed | 20-30 hrs active GPU work | $0-10 total
```

---

## PHASE 1: CHEST X-RAY BASELINE

### 🎯 Goal
Train a production-grade chest X-ray classifier that detects 14 pathologies with 85%+ recall.

### 📦 What You Need
- Kaggle account (free)
- Phone-verified account (unlocks GPU)
- No data upload — NIH + RSNA already on Kaggle

### 🚀 Steps

**1.1** Sign up for Kaggle → verify phone
**1.2** Go to https://www.kaggle.com/code/new
**1.3** Settings → Accelerator: **GPU T4 x2**
**1.4** Add Data: `nih-chest-xrays` + `rsna-pneumonia-detection-challenge`
**1.5** Paste `training/kaggle_train.py` into notebook → Run All
**1.6** Wait 6-8 hours (notebook runs while you sleep)
**1.7** Download `best_model.pt` from Output tab
**1.8** Place in `sentinel/models/densenet/best_model.pt`

### ✅ Success Criteria
- Macro Recall ≥ 85%
- AUC-ROC ≥ 0.85
- Inference < 5 sec on GTX 1650
- Heatmaps visualized correctly

### 📚 Files Built
- `training/kaggle_train.py` — Master training script
- `training/FREE_TRAINING_GUIDE.md` — Step-by-step guide

---

## PHASE 2: CLINIC FINE-TUNING

### 🎯 Goal
Adapt the public-data model to YOUR clinic's specific:
- Imaging equipment calibration
- Patient demographics (Central Asian population)
- Common pathologies in Uzbekistan
- Scan protocols used locally

**Why this matters:** Public datasets are mostly from US/EU hospitals. Local fine-tuning
can boost accuracy 2-5% for your specific clinic.

### 📦 What You Need
- 500-1000 labeled DICOM images from clinic pilot
- Phase 1 trained model (`best_model.pt`)
- Kaggle or Google Colab (free)

### 🚀 Steps

**2.1 Collect Clinic Data**
```bash
# 1. Partner with a clinic in Tashkent
# 2. Get permission to access their DICOMs (anonymized)
# 3. Transfer to your machine
```

**2.2 Label Data Using Our Tool**
```bash
cd sentinel
python training/clinic_data_tool.py label \
    --dicom-dir /path/to/clinic/dicoms \
    --data-dir ./clinic_data \
    --language ru
```

The tool walks through each image and asks the radiologist to:
- Select pathologies (Pneumonia, Effusion, etc.)
- Write a brief radiology report
- Saves everything for later training

**2.3 Check Data Stats**
```bash
python training/clinic_data_tool.py stats --data-dir ./clinic_data
```

Should show class distribution. Aim for:
- 500+ total images minimum
- At least 30 samples per major class
- Balance between normal and pathological

**2.4 Export for Fine-tuning**
```bash
python training/clinic_data_tool.py export \
    --format densenet \
    --output ./clinic_data/densenet_labels.json
```

**2.5 Upload to Colab/Kaggle and Fine-tune**
```bash
# Upload: clinic_data/ folder + best_model.pt

# Run fine-tuning:
python run_finetune.py \
    --base-model models/densenet/best_model.pt \
    --dicom-dir clinic_data/images \
    --labels clinic_data/densenet_labels.json \
    --epochs 30 \
    --lr 0.00001
```

### ✅ Success Criteria
- Fine-tuned recall ≥ 88% on clinic test set
- No degradation on public test set
- Model generalizes to other Uzbekistan clinics

### 📚 Files Built
- `training/clinic_data_tool.py` — Data collection tool (DONE)
- `run_finetune.py` — Fine-tuning script (DONE)

---

## PHASE 3: GEMMA 3 REPORT GENERATION

### 🎯 Goal
Generate natural-language Russian/Uzbek radiology reports from:
- X-ray image
- DenseNet121 findings

Instead of rigid templates, radiologists get human-quality draft text that adapts to image details.

### 📦 What You Need
- 500-1000 (image, report) pairs from clinic
- Gemma 3 base model (auto-downloaded)
- Kaggle T4 x2 GPU (free, 16GB VRAM each)

### 🚀 Steps

**3.1 Collect Real Radiology Reports**
While collecting Phase 2 data, ALSO save the radiologist's written reports.
Our `clinic_data_tool.py` does both simultaneously:

```bash
# When labeling, answer 'y' to "Write report?"
# Tool saves (image_path, findings_text, report_text) tuples
```

**3.2 Export for Gemma**
```bash
python training/clinic_data_tool.py export \
    --format gemma \
    --output ./clinic_data/gemma_dataset.jsonl
```

Output format:
```json
{"image_path": "clinic_data/images/P-001.png",
 "findings_text": "DenseNet findings: Pneumonia 87%",
 "report": "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ: ...",
 "language": "ru"}
```

**3.3 Fine-tune Gemma 3 on Kaggle**

Upload your dataset + `training/gemma_vl_finetune.py` to Kaggle, then:

```python
# In Kaggle notebook:
!pip install -q transformers peft bitsandbytes accelerate trl datasets

# Paste gemma_vl_finetune.py content
# Update CONFIG paths:
CONFIG['dataset_path'] = '/kaggle/input/your-dataset/gemma_dataset.jsonl'

# Run training
train()
```

Training:
- Uses **QLoRA** (4-bit quantization) — fits Gemma 3 4B on T4
- 3 epochs ~ 4-6 hours
- Saves LoRA adapter (~200MB)

**3.4 Integrate Gemma Into Main Pipeline**

Back on your Mac/Victus:
```bash
# Copy Gemma adapter to:
sentinel/models/gemma_medical/

# Update integrated pipeline:
```

```python
# In server.py or app:
from src.inference.integrated_pipeline import IntegratedPipeline

pipeline = IntegratedPipeline(
    densenet_path='models/densenet/best_model.pt',
    gemma_path='models/gemma_medical',  # NEW
)

result = pipeline.analyze('chest.dcm', language='ru')
print(result.natural_report)  # Generated by Gemma!
```

### ✅ Success Criteria
- Generated reports are medically accurate
- Russian/Uzbek grammar is correct
- Reports follow standard radiology structure
- Radiologists approve 80%+ without heavy editing

### 📚 Files Built
- `training/gemma_vl_finetune.py` — Gemma fine-tuning (DONE)
- `src/inference/integrated_pipeline.py` — DenseNet + Gemma integration (DONE)

---

## PHASE 4: BRAIN MRI EXPANSION

### 🎯 Goal
Expand beyond chest X-rays to brain MRI analysis.
Detect: stroke, tumors, multiple sclerosis, hemorrhage, atrophy.

### 📦 What You Need
- Brain MRI dataset (public or clinic)
- New training pipeline for 3D volumes
- More GPU memory (CT/MRI are 3D)

### 🗄️ Public Brain MRI Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| **BraTS** (Brain Tumor Segmentation) | 2000+ MRIs | Tumor detection |
| **IXI** | 600 MRIs | Healthy brain templates |
| **ADNI** | 1000+ MRIs | Alzheimer's (needs application) |
| **ATLAS** | 300 MRIs | Stroke lesions |
| **OASIS** | 2000+ MRIs | Aging brain |

All free, all available on Kaggle or direct download.

### 🚀 Steps

**4.1 Download Brain MRI Data**
```bash
# Option A: From Kaggle
# - Search "brain mri" → add relevant datasets

# Option B: Download BraTS directly
# https://www.synapse.org/brats

# Save to clinic_data/brain_mri/
```

**4.2 Adapt Model Architecture for 3D**

Brain MRI is 3D (volumes), not 2D like chest X-ray. We need:

```python
# Option 1: Slice-by-slice (2D, simpler)
#   Process each axial slice as 2D image with DenseNet121
#   Aggregate predictions across slices
#   Pros: Use existing code
#   Cons: Loses 3D context

# Option 2: 3D CNN (better accuracy)
#   Use MONAI's DenseNet3D or UNet3D
#   Takes full volume (128x128x128) as input
#   Pros: Full 3D context
#   Cons: Needs more VRAM (A40/A100 preferred)

# Option 3: Transformer-based (Vision Transformer 3D)
#   State-of-the-art for brain MRI
#   Needs large dataset (1000+ volumes)
```

**4.3 Train Brain Classifier**

Use `training/brain_train.py` (I'll build this next). Structure:

```python
CONFIG_BRAIN = {
    'classes': [
        'Normal', 'Glioblastoma', 'Meningioma', 'Metastasis',
        'Stroke_Acute', 'Stroke_Chronic', 'Hemorrhage',
        'Multiple_Sclerosis', 'Alzheimer_AD', 'Hydrocephalus',
    ],
    'image_size': [128, 128, 128],  # 3D
    'batch_size': 2,                 # 3D is memory-heavy
    'architecture': 'densenet3d_121',
}
```

**4.4 Add to Desktop App**
Extend Sentinel app to handle multiple modalities:
- CHEST ← DenseNet121 2D
- BRAIN ← DenseNet121 3D
- ABDOMEN ← Phase 5
- SPINE ← Phase 5

### ✅ Success Criteria
- Brain MRI detection recall ≥ 80%
- Sub-30 second inference on clinic PC
- Integrated into existing UI

---

## 💰 COMPLETE BUDGET

| Phase | Cost | Platform | Time |
|-------|------|----------|------|
| 1. Chest X-ray baseline | $0 | Kaggle free | 6-8 hrs |
| 2. Clinic fine-tuning | $0 | Colab free | 2-3 hrs |
| 3. Gemma report generation | $0 | Kaggle free | 4-6 hrs |
| 4. Brain MRI | $0-10 | Kaggle/RunPod | 8-12 hrs |
| **TOTAL** | **$0-10** | | **20-30 hrs GPU time** |

Your time: ~30 hours active work spread over 3-4 weeks.

---

## 📅 REALISTIC TIMELINE

### Week 1: Phase 1 (Chest baseline)
- **Day 1-2:** Create Kaggle, setup notebook, start training
- **Day 3:** Training completes, download, test locally
- **Day 4:** Integrate with desktop app, show boss

### Week 2-3: Phase 2 (Clinic fine-tuning)
- **Day 5-10:** Collect clinic data (500 images)
- **Day 11-12:** Label with radiologist
- **Day 13:** Fine-tune on Colab
- **Day 14:** Test, compare before/after accuracy

### Week 4: Phase 3 (Gemma reports)
- **Day 15-18:** Collect 500+ radiology reports (during Phase 2)
- **Day 19-20:** Format + upload to Kaggle
- **Day 21:** Fine-tune Gemma
- **Day 22:** Integrate + test

### Month 2-3: Phase 4 (Brain MRI)
- Expand to second modality
- Repeat Phases 1-3 for brain data
- Add to app as new modality option

---

## 🎓 KEY DECISIONS YOU'LL MAKE

### Decision 1: Data Collection Strategy
**Option A:** Work with ONE clinic intensively (500-1000 images)
**Option B:** Spread across 3-5 clinics (100-200 each)

**My recommendation:** Start with Option A. Easier to coordinate, better data quality.

### Decision 2: Report Language Priority
**Option A:** Russian only (99% of Uzbek clinicians read Russian)
**Option B:** Russian + Uzbek (for patient-friendly copies)
**Option C:** All three (RU + UZ + EN)

**My recommendation:** Start with A, add B when you have 1000+ reports.

### Decision 3: Gemma Model Size
**Option A:** Gemma 3 4B (fits on free T4) — faster, cheaper
**Option B:** Gemma 3 12B (needs paid GPU) — better quality
**Option C:** Gemma 3 27B (needs A100) — best quality

**My recommendation:** Start with A. Upgrade only if reports quality is insufficient.

### Decision 4: Brain MRI Approach
**Option A:** Slice-by-slice 2D (reuse chest code)
**Option B:** True 3D CNN (more accurate)
**Option C:** Segmentation + classification combo

**My recommendation:** Start with A for speed. Move to B after first brain clinic.

---

## 🏆 END STATE

After all 4 phases:

### Your Product Has:
- ✅ Chest X-ray AI (90%+ recall on 14 pathologies)
- ✅ Brain MRI AI (80%+ recall on 10 conditions)
- ✅ Auto-generated natural language reports (RU/UZ/EN)
- ✅ Grad-CAM explainability
- ✅ Fine-tuned to Uzbekistan population
- ✅ Runs on clinic PC (no internet)

### Market Position:
- First multi-modal medical AI in Uzbekistan
- 10x cheaper than US/EU competitors ($500/mo vs $5,000/mo)
- Local language + local data advantage
- Government-friendly (data sovereignty)

### Revenue Path:
- Month 1: Free pilot at 1 clinic
- Month 3: 2 paying clinics ($1,000 MRR)
- Month 6: 10 paying clinics ($5,000 MRR)
- Month 12: 30 paying clinics ($15,000 MRR)
- Month 18: Expand to Kyrgyzstan, Kazakhstan

---

## 🎬 What To Do RIGHT NOW

**Step 1 (this week):** Kaggle Phase 1 training
→ Use `training/FREE_TRAINING_GUIDE.md`

**Step 2 (week 2):** Partner with a clinic
→ Sign LOI for data access

**Step 3 (week 2-3):** Collect + label 500 images
→ Use `training/clinic_data_tool.py`

**Step 4 (week 3-4):** Fine-tune both DenseNet + Gemma
→ Use `run_finetune.py` + `training/gemma_vl_finetune.py`

**Step 5 (month 2):** Deploy at pilot clinic
→ Use `install_clinic.sh`

---

## 🆘 Still Overwhelmed?

Start with JUST ONE thing this week:

```
→ Create Kaggle account
→ Run kaggle_train.py
→ Wait 8 hours
→ See if model works
```

That's it. One action. Everything else follows.

---

**You have everything you need. The system is built. Now execute.**
