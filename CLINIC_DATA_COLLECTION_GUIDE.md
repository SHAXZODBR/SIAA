# Clinic Data Collection Guide
**SIA Medical AI · Sentinel · How to ask clinics for data and what to do with it**

This is your playbook for the conversation **after** the demo and **before** "give me the contract." Every successful medical AI startup grew on a continuous flow of anonymized clinic data → model improvement → better demo → next clinic.

---

## TL;DR — What You Want From Each Clinic

| What | How much | Why | Priority |
|---|---|---|---|
| Anonymized brain MRIs with original radiologist reports | 50-200 cases | Validate brain models + fine-tune | **P0** |
| Doctor's edits to AI-drafted reports | Every case the system processes | Trains MedGemma on local phrasing | **P0** (auto) |
| Permission to pilot for 30 days free | 1 PC, 1 radiologist | Get foot in door | **P0** |
| Anonymized chest X-rays (Uzbek TB cases) | 100+ TB cases | Localize chest model | **P1** |
| Anonymized head CT trauma | 50 cases | Validate hemorrhage model | **P1** |
| Outcome data (was the AI right?) | When available | Real accuracy measure | **P2** |

---

## 1. THE ASK — Sample Email to a Friendly Radiologist

```
Subject: AI assistant for brain MRI reports — 30-min demo + free 1-month pilot

[Doctor name],

I'm Shaxzod, building Sentinel — an on-premise AI that drafts radiology
reports in Russian and Uzbek for brain MRI, head CT, chest X-ray, and
mammography.

Everything runs locally on your PC. No patient data leaves the clinic.

I'd like to:
1. Show you a 30-minute demo this week
2. If you like it, install it free for 30 days on one of your PCs
3. In return, share 50 anonymized brain MRIs we can use to make the AI
   smarter on your patient population

The 30-day pilot is at zero cost to you. You get faster reports, I get
to learn what actually works in a real clinic.

Available [tomorrow / Friday / next Monday]?

Best,
Shaxzod Batirjonov
SIA Medical AI · siaa.uz · +998 XX XXX XX XX
```

**Don't send this cold.** Use a personal intro through your network. Cold-emailing a radiologist about AI = 2 % response rate.

---

## 2. WHAT TO COLLECT — By Modality

### Brain MRI (your primary product)

**Minimum viable validation set: 50 cases**

For each case:
- Full DICOM series (all sequences — T1, T1ce, T2, FLAIR, DWI if available)
- The original radiologist's signed report (PDF or text, in Russian/Uzbek)
- The final diagnosis if known (from biopsy / surgery / follow-up)

Mix you want:
- 10 normal scans (no pathology)
- 10 confirmed gliomas (any grade)
- 10 confirmed meningiomas
- 5 pituitary adenomas
- 10 strokes (acute or chronic)
- 5 traumatic injuries

**Why this mix:** matches the classes our `brain_tumor_class` and `head_ct` models predict, so we can compute per-class accuracy.

### Chest X-ray (high-value secondary)

**Goal: 200 cases** — Uzbekistan has high TB prevalence, current TorchXRayVision was trained on US data and underperforms on TB. Even 100 local TB cases shifts the chest model's TB recall by 8-12%.

For each case:
- DICOM file (CR or DX modality)
- Original report
- TB confirmation method if positive (sputum, GeneXpert, culture)

Mix you want:
- 30 normal
- 80 tuberculosis (active + latent)
- 30 pneumonia (bacterial / viral / COVID)
- 20 cardiomegaly
- 20 pneumothorax
- 20 mass / nodule (with biopsy result)

### Head CT (emergency department use case)

**Goal: 50 cases**

- 10 normal
- 25 with hemorrhage (mix of all 5 subtypes: epidural, subdural, SAH, IPH, IVH)
- 10 ischemic stroke
- 5 trauma without bleed

### Mammography (women's health)

**Goal: 100 cases with biopsy results** (otherwise impossible to validate)

- 40 normal (BIRADS 1–2)
- 30 benign findings (BIRADS 3 with biopsy)
- 30 malignant (BIRADS 4–5 with biopsy)

Both CC and MLO views per breast = ~400 images.

---

## 3. THE LEGAL WRAPPER

You need TWO documents signed before any data transfer:

### 3a. Data Use Agreement (DUA) — between you and the clinic

**One page. Plain Russian.** Copy-paste from this template, fill in the name + sign:

```
СОГЛАШЕНИЕ О ИСПОЛЬЗОВАНИИ ДАННЫХ

Между:
[Клиника], в лице [Главный врач], ("Клиника")
и
ИП Батыржонов Шахзод (SIA Medical AI), ("Разработчик")

1. ПРЕДМЕТ
   Клиника передаёт Разработчику анонимизированные DICOM-снимки
   и обезличенные радиологические отчёты с целью улучшения
   медицинского ИИ-программного обеспечения "Sentinel".

2. АНОНИМИЗАЦИЯ
   До передачи Разработчику Клиника удаляет из DICOM-метаданных:
   ФИО пациента, дату рождения, адрес, номер истории болезни,
   ФИО лечащего врача, наименование Клиники.
   Разработчик предоставляет программу-анонимизатор для этой цели.

3. ИСПОЛЬЗОВАНИЕ
   Разработчик использует данные ИСКЛЮЧИТЕЛЬНО для:
   а) обучения и валидации алгоритмов ИИ;
   б) измерения точности модели на узбекском населении;
   в) внутренних исследований SIA Medical AI.

   Разработчик НЕ:
   а) передаёт данные третьим лицам;
   б) пытается реидентифицировать пациентов;
   в) продаёт исходные данные.

4. КОМПЕНСАЦИЯ
   В обмен на данные Клиника получает:
   а) бесплатное использование Sentinel в течение 30 дней;
   б) приоритетную скидку 50 % на годовой контракт после пилота.

5. РАСТОРЖЕНИЕ
   Любая сторона может расторгнуть Соглашение с уведомлением
   за 30 дней. При расторжении Разработчик удаляет копии
   полученных данных в течение 14 дней.

6. ПОДСУДНОСТЬ
   Споры разрешаются в Хозяйственном суде г. Ташкента согласно
   законодательству Республики Узбекистан.

[Дата]                                  [Дата]
[Подпись Клиники]                       [Подпись Разработчика]
```

### 3b. Patient Consent Addendum — clinic adds to their intake form

Most clinics already have broad-consent language in their intake form. If not, ask them to add this paragraph (in their existing form):

> *Я даю согласие на использование обезличенных результатов моих
> медицинских исследований (DICOM-изображений и радиологических
> отчётов) для улучшения медицинских информационных систем,
> включая системы искусственного интеллекта. Идентифицирующая
> информация удаляется до любой передачи.*

Once signed by the patient, the clinic is legally cleared to share anonymized DICOMs with you.

**Uzbek-specific legal note:** Under the Law on Personal Data of the Republic of Uzbekistan (2019, amended 2023), once PHI tags are stripped, the data is no longer "personal data" and falls outside the scope of the Act. So the bar to clear is just: ensure proper anonymization happened.

Confirm with a UZ-licensed lawyer (~$200) before your first transfer.

---

## 4. ANONYMIZATION — DO THIS FIRST, ALWAYS

**Never** copy raw DICOM files to your laptop. Even briefly. Once the file leaves the clinic with PHI intact, you've broken the agreement.

### The Sentinel anonymizer (already built)

```bash
# Single file
python -m src.utils.dicom_anonymizer input.dcm --output anon.dcm

# Whole directory (preserves folder structure, links same patient across files)
python -m src.utils.dicom_anonymizer /path/to/clinic_dicoms \
    --output ./data/clinic_anonymous \
    --batch

# Verify it actually worked
python -m src.utils.dicom_anonymizer anon.dcm --output /dev/null --verify
```

### The flow at the clinic

1. **You bring a USB stick with**:
   - Sentinel installed
   - The anonymizer script
   - An empty external drive

2. **At the clinic**:
   - Clinic IT / radiologist exports DICOMs to the empty drive
   - Run: `python -m src.utils.dicom_anonymizer drive_path --output drive_path/anon --batch`
   - **Verify** (run with `--verify` on a sample) — if any PHI remains, abort
   - Take only the `anon/` folder home

3. **At home**:
   - Re-verify on a few random files
   - Move to `data/clinic_<name>/` in your repo
   - Add to git-ignore (NEVER commit clinic data to git)

4. **The Russian-language reports**:
   - Doctor saves reports as PDF or TXT alongside DICOMs
   - These DO contain PHI in the report header (patient name, etc.)
   - **Strip it manually**: open each, replace patient name with `ANONYMIZED`
   - Or better — script it: `scripts/scrub_report_phi.py` (not yet built — write when needed)

---

## 5. WHAT TO DO WITH THE DATA — Once you have it

### Step 5a. Validate the model

```bash
# Put 50 brain MRIs in data/clinic_<name>/brain_test/
# Each in a folder named after the ground truth class:
#   data/clinic_<name>/brain_test/
#     glioma_tumor/        case001.dcm  case002.dcm
#     meningioma_tumor/    ...
#     no_tumor/            ...
#     pituitary_tumor/     ...

python scripts/validate_accuracy.py \
    --data data/clinic_<name>/brain_test \
    --model brain_tumor_class
```

You'll get:
- Overall accuracy (target: >85 %)
- Per-class precision / recall / F1
- Confusion matrix
- AUC-ROC if probabilities available

If accuracy is below 85%, you fine-tune.

### Step 5b. Fine-tune the classifier

```bash
# Re-organize data into train/val splits:
#   data/clinic_<name>/brain_finetune/
#     train/
#       glioma_tumor/   ...
#       meningioma_tumor/  ...
#       ...
#     val/
#       glioma_tumor/   ...
#       ...

python training/finetune_brain_classifier.py \
    --data data/clinic_<name>/brain_finetune \
    --epochs 15 \
    --batch-size 16
```

Output: `models/brain_finetuned/` — the inference server auto-picks this up over the public weights.

### Step 5c. Fine-tune the report writer (MedGemma)

If the clinic shared their original signed reports, you can teach MedGemma their phrasing:

```bash
# Build training JSONL — one line per case:
#   {"image_path": "case001.png", "findings": "glioma 87% left frontal",
#    "report": "<the doctor's actual report text in Russian>"}
python scripts/build_report_jsonl.py \
    --dicoms data/clinic_<name>/brain_finetune \
    --reports data/clinic_<name>/reports/ \
    --output data/clinic_<name>/medgemma_train.jsonl

# LoRA fine-tune (small adapter file, ~50 MB output)
python training/finetune_medgemma_reporter.py \
    --data data/clinic_<name>/medgemma_train.jsonl \
    --epochs 3 \
    --output models/medgemma_<clinic>_lora
```

Now reports drafted by Sentinel sound like the clinic's own radiologist wrote them.

### Step 5d. Ship the improved model back

Build a new installer with the fine-tuned models bundled:

```bash
cd desktop-app
npm run build:win   # or build:mac
# Output: desktop-app/release/Sentinel Medical AI Setup 1.0.1.exe

# Send back to the clinic, install over old version, license carries over
```

Their doctor sees the AI gets noticeably better. That's when you convert pilot → paid.

---

## 6. THE FLYWHEEL — Why this gets stronger with every clinic

```
Clinic 1 ─── 50 brain MRIs ─── Fine-tuned model v1.1 ───┐
                                                         │
Clinic 2 ─── 100 brain MRIs ─── v1.2 (better than v1.1) ─┤
                                                         │
Clinic 3 ─── 200 brain MRIs ─── v1.3 (better than v1.2) ─┤
                                                         │
                                                  Better demo
                                                  Easier sale
                                                  More clinics
                                                         │
                                                         ▼
                                              Compounds forever
```

**Important:** the per-clinic data also stays usable for that clinic specifically. Clinic 1's fine-tune doesn't get overwritten when clinic 2 joins — you maintain a per-clinic adapter so each install is tuned to its own scanner.

---

## 7. WHAT NOT TO DO

❌ **Never** put clinic DICOMs in cloud storage (Dropbox, Google Drive, etc.) — even briefly.

❌ **Never** commit clinic data to a git repo — even private. One mis-clicked "Share" and you've leaked PHI.

❌ **Never** show one clinic's data to another clinic during a demo. Use public data (Tier 1 datasets) for demos.

❌ **Never** promise "100 % accuracy" — even your best fine-tuned model will miss things. The legal protection in your EULA is "advisory tool, not diagnostic device" — don't undermine it with marketing claims.

❌ **Never** delete the original anonymized data after fine-tuning. You may need it again if you re-train. Backup to a NAS or encrypted external drive.

❌ **Never** bring patient data home in plaintext on a USB stick. Always encrypt:
```bash
# On macOS — encrypt with disk image:
hdiutil create -volname clinic_data -size 5g -encryption AES-256 -fs APFS \
    -srcfolder ./data/clinic_<name> ./data/clinic_<name>.dmg
# Asks for password — write it down separately
# Delete the unencrypted folder after creating the .dmg
rm -rf ./data/clinic_<name>
```

---

## 8. WHAT TO ASK FOR FROM THE CLINIC IT TEAM

When you go to install Sentinel on a clinic PC, the IT person needs to:

1. **Allow** Sentinel to listen on `127.0.0.1:8000` (Windows Firewall prompt — click Allow)
2. **Add an exception** in their antivirus (Sentinel runs models locally — looks suspicious to over-eager AV)
3. **Configure their PACS / DICOM router** to forward incoming studies to:
   - Either: a watch folder (`%APPDATA%\Sentinel Medical AI\dicom_inbox\`)
   - Or: a local Orthanc PACS we install (port `4242` for DICOM C-STORE, `8042` for HTTP)
4. **Provide** read access to the imaging machine's DICOM output folder

90% of clinics use one of these:
- **Carestream Vue** — has DICOM router built-in, send target = our IP:4242
- **GE Centricity** — same, AE Title-based forwarding
- **Old standalone scanners** — write DICOM to a network share, we monitor it
- **Cloud PACS** (rare in UZ clinics) — we add a webhook

---

## 9. PRICING THE CONVERSION (after the 30-day pilot)

| Tier | Price | What's included | When to offer |
|---|---|---|---|
| **Trial** | Free (30 days) | All models, 1 PC, 1 user | Demo phase |
| **Starter** | $300/month | Brain MRI + Head CT, 1 PC | Solo radiologist clinic |
| **Pro** | $700/month | All models, 2 PCs, 3 users | Multi-radiologist clinic |
| **Enterprise** | $1500/month | Above + custom fine-tune + priority support + SLA | 10+ studies/day, hospital |

**Discount strategy:**
- 50 % off year 1 if they prepay for 12 months
- 25 % off year 1 if they help you sign 1 more clinic (referral fee)
- Free upgrade to Pro for the first 3 clinics (early adopter bonus)

---

## 10. CHECKLIST — Before your first clinic visit

- [ ] Sentinel installer (.exe) on USB stick
- [ ] Anonymizer script tested
- [ ] DUA template printed in Russian (2 copies)
- [ ] Personal demo data (3-5 anonymized DICOMs, public datasets only)
- [ ] Sales sheet (1 page Russian, 1 page Uzbek)
- [ ] Encrypted external drive (1 TB, empty)
- [ ] Laptop with Sentinel running for live demo
- [ ] Phone ready to record video testimonial after demo
- [ ] Trial license you'll issue — bring blank `license.dat` template

---

*Guide by SIA Medical AI · May 2026*
