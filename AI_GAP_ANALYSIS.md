# Sentinel Medical AI — What We DON'T Detect (Honest Gap Analysis)

**Be transparent with clinics about gaps. Trust > false promises.**

For sales conversations: lead with what we DO detect (42 pathologies, see [`AI_DETECTION_INVENTORY.md`](AI_DETECTION_INVENTORY.md)) and proactively disclose what we don't. Doctors respect honesty and stop trusting vendors who oversell.

---

## 🚫 ENTIRE MODALITIES WE DON'T HANDLE

These body parts / exam types have NO Sentinel coverage at all:

| Modality | What's typically scanned | V1.1 priority |
|---|---|---|
| **Spine MRI** | Cervical / thoracic / lumbar — disc herniation, stenosis, spinal cord lesions | 🔥 High (common UZ request) |
| **Abdomen CT/MRI** | Liver, kidney, pancreas, spleen, appendix, bowel | 🔥 High |
| **Pelvis CT/MRI** | Bladder, prostate, gynecologic, rectum | Medium |
| **Cardiac CT / MRI** | Coronary arteries, valves, myocardium, ischemia | Medium |
| **Musculoskeletal MRI** | Knee/shoulder/hip joints, meniscus, ligaments, cartilage | Medium |
| **Extremity X-ray** | Wrist, ankle, hand, foot, long-bone fractures | 🔥 High (ER common) |
| **Cardiac Ultrasound (Echo)** | Cardiac function, valves, ejection fraction | Low (specialized) |
| **Abdominal Ultrasound** | Liver, gallbladder, kidneys | Low (highly operator-dependent) |
| **Thyroid Ultrasound** | Nodules, goiter | Low |
| **Pediatric imaging** (any modality) | Different anatomy + growth plates | Medium |
| **Dental X-ray / CBCT** | Teeth, jaw, sinuses | Low |
| **Mammography 3D (DBT)** | Tomosynthesis — better than 2D | Medium |
| **Breast MRI** | Dense breast tissue, contrast-enhanced | Low |
| **CT/MR Angiography (CTA/MRA)** | Aneurysms, vascular malformations, dissection | Medium |
| **Nuclear medicine** (PET, SPECT) | Metabolic imaging, oncology staging | Low |
| **Fluoroscopy** | Real-time motion imaging | Low |
| **DSA** (digital subtraction angiography) | Interventional radiology | Low |
| **Ophthalmologic** (OCT, fundus) | Retinal imaging | Low |

---

## 🧠 BRAIN MRI — Gaps Within Our Coverage

We detect tumors, dementia, and (via MedSAM) prompt-based segmentation. We DON'T detect:

| Finding | Why missing | Clinical importance |
|---|---|---|
| **Ischemic stroke** (acute) | No DWI/ADC model trained | 🚨 Time-critical |
| **Chronic infarcts** | Not in training data | High |
| **Multiple sclerosis (MS) lesions** | Niche dataset (MSSEG-2), no public model | High |
| **White matter hyperintensities** | Research task, no production model | Medium |
| **Cerebral microbleeds** | Needs SWI sequence, not in our pipeline | Medium |
| **Cerebral aneurysm** | Needs MRA (we only do brain MRI) | 🚨 Time-critical |
| **Arteriovenous malformations (AVM)** | Needs MRA | High |
| **Encephalitis / meningitis** | Subtle imaging, hard task | High |
| **Brain abscess** | Rare, no pretrained model | High |
| **Hydrocephalus** | Volumetric task, not classified | Medium |
| **Traumatic brain injury** (contusion, DAI) | Trauma-specific, not in our model | 🚨 Time-critical |
| **Cysts** (arachnoid, colloid, etc.) | Rare, not in classifier | Low |
| **Demyelinating diseases** (besides MS) | Niche, no public model | Low |
| **Pediatric brain tumors** | Adult-trained models miss pediatric | Medium |
| **Pituitary microadenomas** | Smaller than our detection threshold | Low |
| **Cerebellar lesions** | Often missed by adult tumor model | Medium |
| **Brainstem pathology** | Same — model focuses on supratentorial | Medium |

---

## 🫁 CHEST X-RAY — Gaps Within Our 18 Pathologies

We have 18 classes from TorchXRayVision. Things we **don't explicitly label** (sometimes captured as "Infiltration" / "Consolidation"):

| Finding | Why missing | UZ relevance |
|---|---|---|
| **Tuberculosis (TB) specifically** | Trained on US data (low TB) — calls TB "Infiltration" | 🔥 **Critical for Uzbekistan** (high TB rate) |
| **COVID-19 specific pattern** | Trained pre-COVID | Lower priority now |
| **Lung cancer staging** | We detect Mass/Nodule but not stage T/N/M | High |
| **Pulmonary embolism** | Needs CT pulmonary angiogram | Critical (different scan) |
| **Aortic dissection** | Needs CT angiogram | Critical (different scan) |
| **Sarcoidosis** | Subtle, not in 18 classes | Medium |
| **Bronchiectasis** | Difficult on plain X-ray | Medium |
| **Mediastinal masses** (specific types) | We have "Enlarged Cardiomediastinum" but not subtypes | Medium |
| **Foreign bodies** (esp. pediatric) | Not in training data | High in pediatric ER |
| **Pneumomediastinum** | Specific pattern, not labeled | Medium |
| **Right vs left lung specificity** | Detected but heatmap may miss laterality | High |
| **Pediatric chest pathology** | Different anatomy, different diseases | Medium |

---

## 🩸 HEAD CT — Gaps Within Hemorrhage Detection

We detect 6 hemorrhage types. We DON'T detect:

| Finding | Why missing | Clinical importance |
|---|---|---|
| **Ischemic stroke** (early hypodensity) | Different task — needs CT/MRI fusion | 🚨 Time-critical |
| **Skull fracture** | Specific bone window analysis missing | High |
| **Brain tumor on CT** | We have brain tumor MRI, not CT | Medium |
| **Hydrocephalus** | Volumetric measurement, not classified | High |
| **Midline shift quantification** | Geometric measurement, not in model | 🚨 Critical for trauma triage |
| **Mass effect grading** | Quantification not automated | High |
| **Calcifications** (vascular / basal ganglia) | Not labeled | Low (mostly incidental) |
| **Acute infarct mapping** (DWI-equivalent) | Different sequence | Critical |

---

## 🎀 MAMMOGRAPHY — Gaps

We detect 5 categories (mass, calc, asymmetry, distortion, normal). We DON'T:

| Finding | Why missing |
|---|---|
| **Full BI-RADS auto-assignment** (1-6) | We label findings but don't auto-score BI-RADS |
| **Lesion size measurement** | Need automated geometry |
| **Density category** (A/B/C/D) | Separate task |
| **Architectural distortion grading** | Subtle, hard task |
| **Cyst vs solid mass differentiation** | Needs ultrasound correlation |
| **Lymph node assessment** | Out of frame for mammo |

---

## 🛠 CAPABILITIES / WORKFLOW WE DON'T HAVE

| Missing capability | Why it matters | Effort to build |
|---|---|---|
| **Auto lesion measurement** (volumetry) | Doctors need size for staging | 1 week |
| **Longitudinal comparison** (vs prior scans) | "Has the tumor grown?" | 2 weeks |
| **Triage / urgency scoring** | Helps prioritize worklist | 1 week |
| **DICOM SR export back to PACS** | Clinic integration | 3 days |
| **Bone age estimation** | Pediatric endocrinology | Separate model needed |
| **Coronary calcium scoring** | Cardiac risk | Separate model |
| **Lung-RADS / TI-RADS / PI-RADS scoring** | Standardized scoring | Per-system model |
| **Tumor T/N/M staging** | Oncology workflow | Multi-modal, hard |
| **RECIST treatment response** | Oncology follow-up | Hard |
| **Multi-modal fusion** (CT + MRI same patient) | Better diagnosis | Hard |
| **Quality control** on incoming images | Reject bad scans before AI | 2 days |
| **Image enhancement / denoising** | Improve low-quality scans | 1 week |
| **Auto-routing without Orthanc** | Direct from machine | 1 week per machine model |
| **Real-time PACS auto-pull** (Orthanc-less) | Easier deployment | 1 week |
| **Multi-patient dashboard** | Aggregate stats | 1 week |

---

## ⚠ POPULATION-SPECIFIC GAPS (Uzbekistan Reality)

| Issue | What it means | Fix |
|---|---|---|
| **TB localization** | Chest model was trained on US patients (low TB) — calls TB "Infiltration" generically | Fine-tune on local TB cases |
| **Smaller average body habitus** (Central Asia) | Models trained on US/EU populations may have subtle biases | Fine-tune on local data |
| **Older imaging equipment** | Many UZ clinics use 10+ year old machines — image quality varies | Add denoising preprocessing |
| **No pediatric coverage** | UZ has high birth rate, lots of pediatric imaging demand | Separate pediatric model needed |
| **Cyrillic OCR** for legacy reports | Old paper reports need digitization | Use Tesseract or similar |

---

## 📋 What to TELL clinics honestly

When a radiologist asks "does it detect X", here's the script:

| If they ask about... | Honest answer |
|---|---|
| Brain tumor (glioma/meningioma/pituitary) | ✅ Yes, 99% accuracy on test data |
| Stroke on CT | ⚠ Hemorrhage yes, ischemic stroke no (V1.1 roadmap) |
| Stroke on MRI | ❌ Not in V1.0 |
| Multiple sclerosis | ❌ Not yet — we can add if you give us 100 MS cases |
| Lung pneumonia | ✅ Yes, 78% AUC |
| Lung TB specifically | ⚠ We detect TB-pattern as "Infiltration" but don't label as TB — needs fine-tune on UZ data |
| Cardiomegaly | ✅ Yes, 87% AUC |
| Lung cancer screening | ⚠ Mass + Nodule detection yes, staging no |
| Aortic dissection | ❌ Needs CT angiogram (different scan) |
| Hemorrhage on head CT | ✅ Yes, 6 subtypes, 96% AUC |
| Skull fracture | ❌ Not in V1.0 |
| Spine MRI | ❌ Not in V1.0 — V1.1 candidate |
| Knee MRI | ❌ Not in V1.0 |
| Mammography mass | ✅ Yes, 85% AUC |
| Full BI-RADS scoring | ❌ Findings yes, BI-RADS auto-score no |
| Pediatric anything | ❌ Adult models — accuracy unknown on pediatric |
| Ultrasound (any) | ❌ Not in V1.0 |
| Cardiac (echo/CT) | ❌ Not in V1.0 |

---

## 🎯 V1.1 / V2.0 ROADMAP — What to add (in priority order)

**V1.1 — Add to existing modalities (next 6 months):**
1. 🔥 **TB-specific fine-tune** on UZ chest X-rays (most clinically valuable for UZ)
2. **Ischemic stroke on CT** (early hypodensity model)
3. **Skull fracture detection** on head CT
4. **DICOM SR export** (clinic integration)
5. **Longitudinal comparison** (vs prior studies)
6. **Lesion volumetry** (auto-measurement)

**V2.0 — New modalities (6-12 months):**
1. **Spine MRI** (lumbar disc/stenosis) — high UZ demand
2. **Extremity X-ray** fracture detection — ER use case
3. **Pediatric chest X-ray** — separate model
4. **Abdomen CT** (appendicitis, liver) — ER use case
5. **Cardiac CT** (coronary)
6. **Knee/shoulder MRI**

**V3.0 — Advanced (12+ months):**
1. Multi-modal fusion
2. Triage / urgency scoring
3. Auto-staging (Lung-RADS, BI-RADS scoring)
4. Treatment response (RECIST)

---

## 💡 Strategic Note

**You DON'T need to cover everything to win.** The top 3 clinic requests in Uzbekistan are:
1. **Chest X-ray** (we have ✅ 18 pathologies, 81% avg AUC)
2. **Brain MRI** (we have ✅ 99% tumor classification)
3. **Head CT** for trauma (we have ✅ 96% hemorrhage AUC)

That's enough to demo confidently and sign your first 5 clinics. Add modalities by **customer pull** (when 3 clinics ask for the same thing, build it).

Don't pre-build features nobody asked for.

---

*Generated for SIA Medical AI · May 2026*
