# Sentinel Medical AI — Brain-First Startup Roadmap
**SIA Medical AI · Tashkent, Uzbekistan**
**Supersedes: `Sentinel_AI_NotebookLM (1).md` (the original chest-first plan)**

---

## Why brain instead of chest

The original NotebookLM plan led with chest X-ray because chest is the most-imaged modality globally and has the largest open dataset (CheXpert, NIH ChestX-ray14, MIMIC-CXR — combined ~600 K images). It's a defensible first product.

But for a **startup with one founder building toward a clinic pilot in Uzbekistan**, brain is the better wedge:

| Dimension | Chest | Brain |
|---|---|---|
| Differentiation in UZ market | Some competitors offer chest tools | **Zero competitors** offering brain MRI AI in CIS |
| Average revenue per case | $2–3 | **$15–25** (MRI scans cost more, willing to pay more for AI) |
| Pretrained model quality (2026) | Mature (TorchXRayVision, 0.81 AUC avg) | **Mature** (BraTS-trained SwinUNETR v2, 0.85 Dice on tumors) |
| Fine-tuning data needed | 1000+ images for meaningful gain | **200 images** because brain pathologies are less population-variable |
| Clinical urgency | Moderate (most chest findings non-urgent) | **High** (hemorrhages and tumors are time-critical) |
| Doctor's pain level | "Another chest review" | **"I dread brain MRI day"** (hardest reads) |
| Local TB confounder | High — Uzbek TB rates skew chest models | None for brain |

The Uzbekistan winning bet is: **be the brain AI nobody else offers**, then expand to chest/CT/mammo once the brand is established.

---

## What's already built (status as of today)

The Sentinel codebase has a working scaffold:

### ✅ Running today
- **Multi-modality model registry** with 4 brain models registered: 2D tumor classifier, 3D BraTS segmentation (SwinUNETR), MedSAM (prompt-based), Dementia/Alzheimer detection ([model_registry.py](src/inference/model_registry.py))
- **MedGemma 4B integration** for vision-language radiology reports — Google's medical-domain Gemma fine-tune ([model_registry.py:160-181](src/inference/model_registry.py))
- **3D multi-sequence preprocessor** that takes T1/T1ce/T2/FLAIR DICOM series and produces the (4,128,128,128) tensor BraTS models expect ([brain_mri_preprocessor.py](src/pipeline/brain_mri_preprocessor.py))
- **Auto-routing endpoint** (`POST /analyze/auto`) that picks the right model from DICOM tags
- **Brain-specific report templates** in RU/UZ/EN covering tumor + hemorrhage + dementia ([brain_report_templates.py](src/inference/brain_report_templates.py))
- **Fine-tuning pipelines** for both the classifier (head-only LoRA-style) and MedGemma (full LoRA) ([training/](training/))
- **Cyrillic-correct PDF export** via Electron's native printToPDF
- **Working language switcher** that calls backend Gemma to translate (not just template substitution)
- **Multi-language PDF export** (current language OR all 3 in one document)
- **License system** — RSA-signed, machine-bound, demo-mode rate limiter for anti-piracy
- **Clinic deployment script** (`install_clinic.sh`)

### ⚠ Built but not yet validated
- Brain model accuracy on real clinic data — need 50-100 known cases to measure
- 3D segmentation pipeline — code complete but needs end-to-end test with a real BraTS-format study
- MedGemma fine-tune — script ready, needs ~200 doctor-written reports to train

### ❌ Not yet built
- Real-time DICOM watcher pulling from clinic PACS automatically
- DICOM SR (Structured Report) export back to the clinic's PACS
- Multi-patient comparison view
- Tumor volumetry tracking across visits

---

## The brain model lineup — what to use when

| Task | Model | When to use | Tier |
|---|---|---|---|
| Quick tumor screening | `andrei-teodor/resnet-pretrained-brain-mri` | Any single brain MRI slice arrives | Production |
| Tumor type classification | Same — 4-class output | After screening hits | Production |
| Pixel-precise tumor boundary | `anhaltai/swinunetrv2_BraTS2021_mini` | When 4 sequences (T1/T1ce/T2/FLAIR) available | Production |
| Doctor draws box, get exact mask | `wanglab/medsam-vit-base` (MedSAM) | Interactive segmentation, volume measurements | Production |
| Dementia screening | `dhritic9/vit-base-brain-mri-dementia-detection` | Elderly patients, cognitive complaints | Beta |
| AI-drafted radiology text | `Jesteban247/brats_medgemma` (MedGemma + BraTS) | Best-quality natural-language reports | Beta |
| Report fallback (no GPU) | Local templates in `brain_report_templates.py` | When no LLM available | Production |

Every entry has Apache-2.0 or MIT license — commercial deployment is OK.

---

## How to make it actually work end-to-end

### Step 1 — Get the models on disk
```bash
cd /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel
source ~/venv/bin/activate

# Pull all 4 brain models + chest + head_ct + mammo
python scripts/download_all_models.py
```
First run takes 15-30 minutes (downloads ~700 MB total). Cached forever after.

### Step 2 — Set up Gemma 4 / MedGemma for reports
```bash
# Best for clinic — local Ollama + Gemma 4 (offline, fast)
bash scripts/setup_gemma_brain.sh local

# Best for radiology quality — MedGemma 4B (needs GPU)
bash scripts/setup_gemma_brain.sh medgemma

# Check status anytime
bash scripts/setup_gemma_brain.sh status
```

### Step 3 — Run server + app
```bash
# Terminal 1
DEV_BYPASS_LICENSE=1 python run_server.py

# Terminal 2
cd desktop-app && npm run dev
```

### Step 4 — Test brain workflow
1. In the sidebar, pick **"Brain MRI"** in the AI Model selector
2. Upload a brain MRI DICOM (any single slice works for the 2D path)
3. Wait ~3-5 seconds
4. See:
   - Tumor classification (glioma / meningioma / pituitary / no tumor)
   - Confidence scores
   - Russian report drafted by Gemma 4 / MedGemma
5. Click **РУС / O'ZB / ENG** to retranslate
6. Click **"Export PDF (3 lang)"** for the multi-language report

### Step 5 — Fine-tune on clinic data (when you have 200+ reports)
```bash
# Tune the classifier on Uzbek brain MRIs
python training/finetune_brain_classifier.py \
    --data data/clinic_brain \
    --epochs 15

# Tune MedGemma on doctor-written reports (Russian/Uzbek)
python training/finetune_medgemma_reporter.py \
    --data data/clinic_reports.jsonl \
    --epochs 3
```
The fine-tuned model lives in `models/brain_finetuned/` — the registry auto-prefers it over the public weights.

---

## The startup play — what makes this defensible

### 1. The data flywheel is the moat
Every clinic that uses Sentinel produces:
- DICOM scans → can be added to fine-tuning corpus (after anonymization)
- AI-drafted reports → doctors edit them → corrections become training signal
- Model improves on UZ patient population specifically
- Hard for any non-UZ competitor to match without the same clinic relationships

This is wired into the codebase — `src/inference/data_collector.py` records every analysis with anonymized findings, and `/report/save_correction` captures every doctor edit.

### 2. Anti-piracy is built-in
Sentinel's RSA-signed license file (`src/utils/license.py`) binds each install to one machine's hardware fingerprint. Move the install to another PC = license invalidates. A clinic that bought 1 seat cannot resell or duplicate the install.

### 3. Local-first is the law in CIS
Uzbek medical data law forbids patient DICOM leaving the country. Sentinel runs 100% on-premise — Ollama + Gemma 4 + local model registry. No cloud dependency. Competitors offering cloud-based AI are legally blocked from this market.

### 4. Brain has higher willingness-to-pay
Chest is a $5-10/case ceiling. Brain MRI averages $15-25/case at private clinics. Even if Sentinel takes 10% of that as AI fee, brain workflow is 3x more lucrative per case.

### 5. Russian/Uzbek native is a real competitive moat
Every US/EU competitor offers English-only reports. Doctors in UZ want reports in Russian (medical lingua franca) and Uzbek (patient-facing). Gemma 4 + the brain templates handle both natively.

---

## The 12-week brain-first timeline

### Week 1-2 — Brain model integration & testing
- [x] Brain registry built with 4 models
- [x] 3D preprocessor for T1/T1ce/T2/FLAIR
- [x] Auto-routing endpoint
- [ ] Validate each model with `scripts/validate_accuracy.py` on 20 known DICOMs
- [ ] Fix any model-loading edge cases

### Week 3-4 — First clinic pilot prep
- [ ] Pick 1 friendly clinic with a brain-imaging radiologist
- [ ] Sign 1-page Data Use Agreement (template in `CLINIC_DATA_AND_STRATEGY.md`)
- [ ] Install Sentinel on a clinic laptop
- [ ] Train clinic technician on uploading + reviewing
- [ ] Get 50 retrospective brain MRIs (anonymized)

### Week 5-6 — Clinic validation & feedback loop
- [ ] Run 50 known cases through Sentinel
- [ ] Compare AI findings vs original radiologist report
- [ ] Document concordance rate (target: 80%+)
- [ ] Get the radiologist to use Sentinel on 10 fresh cases live
- [ ] Capture every edit they make (the corrections data flywheel starts)

### Week 7-8 — First fine-tune
- [ ] Fine-tune classifier on clinic's 50 cases (`training/finetune_brain_classifier.py`)
- [ ] Re-validate — accuracy should go up 5-10 points
- [ ] If MedGemma deployed: fine-tune on the 50 corrected reports
- [ ] Ship updated model to clinic

### Week 9-10 — Add 2nd & 3rd clinics
- [ ] Use first clinic's results as case study
- [ ] Sign 2 more clinics
- [ ] Each clinic = 50 more training cases
- [ ] Now training corpus = 150 brain MRIs + 150 paired reports

### Week 11-12 — Productize for scale
- [ ] Windows installer (already started — `electron-builder` config in package.json)
- [ ] One-click setup: license activation + model download in installer
- [ ] Auto-update mechanism for model improvements
- [ ] Documentation in Russian for clinic IT
- [ ] First paid contract: $500-1000/month per clinic

### Months 4-6 — Add adjacent modalities
- [ ] Head CT (already in registry — just needs validation)
- [ ] Chest X-ray (production-ready, just needs UI prominence)
- [ ] Mammography (registry-ready)
- [ ] Spine MRI (custom training needed)

### Months 7-12 — Volume + product depth
- [ ] 10+ paying clinics → $5-10K MRR
- [ ] DICOM SR export (writes back to clinic PACS)
- [ ] Multi-visit comparison ("tumor changed from last MRI")
- [ ] Tumor volumetry tracking
- [ ] Apply for Ministry of Health certification

---

## Money math

### Costs to get to first clinic
| Item | Cost |
|---|---|
| Claude Code Max (3 months) | $300 |
| RunPod (5 fine-tune runs) | $50 |
| One pilot laptop (Lenovo + RTX 4060) | $1500 |
| **Total to clinic-ready** | **$1850** |

### Revenue at 1, 3, 10 clinics
| Stage | MRR | ARR |
|---|---|---|
| 1 clinic | $700 | $8.4K |
| 3 clinics | $2100 | $25K |
| 10 clinics | $7000 | $84K |
| 25 clinics | $17500 | $210K |

Break-even on dev costs hits at clinic #1. After that every clinic is profit.

### What 25 clinics looks like
There are roughly 200 MRI scanners in Uzbekistan (Tashkent + regional). 25 clinics = 12% market share = a real local champion. Once you have that, expansion to Kazakhstan + Kyrgyzstan + Tajikistan opens up — same regulatory environment, same Russian-language reports.

---

## What I would NOT promise to clinics yet

Be honest about scope:

✅ **Promise**: AI-drafted brain MRI reports in Russian/Uzbek with high concordance to your radiologist
✅ **Promise**: 3-5 second analysis on a chest X-ray, 10-15 seconds on a 3D brain MRI
✅ **Promise**: Patient data never leaves the clinic
✅ **Promise**: Fine-tuned to your scanner protocols and patient population after 1 month

❌ **Don't promise**: Replacement for radiologist
❌ **Don't promise**: 100% accuracy
❌ **Don't promise**: Pediatric brain (no pediatric pretrained models in registry yet)
❌ **Don't promise**: Stroke detection in <30 minutes (not ER-grade yet)
❌ **Don't promise**: Ministry of Health certification (in progress, takes 6-12 months)

Position Sentinel as **"AI assistant that drafts your reports faster"** — not a diagnostic device. That's defensible scope and it's what radiologists actually want.

---

## The one-sentence pitch

> "Sentinel is the only on-premise AI radiology assistant for Central Asia clinics — it reads brain MRIs, drafts reports in Russian and Uzbek, and gets smarter on your patient data without ever sending it to the cloud."

That's the wedge. Build that. Ship it.

---

*Document by SIA Medical AI / Sentinel Project · April 2026*
*Contact: shakhzodbatirjonov@gmail.com · siaa.uz · Tashkent*
