# SIAA — Sentinel Medical AI · User Manual for Radiologists

Version 1.0.0 (desktop app and AI server) · Document revision 2026-09

---

## 1. What this product is — and what it is not

**Sentinel Medical AI is a decision-support tool.** It runs on a computer inside your clinic, reads a brain MRI study, flags studies that may need a closer look, and drafts a structured report in Russian, Uzbek or English.

It is **not** a diagnostic device and it is **not** a replacement for a radiologist:

- **A radiologist reads and signs every study.** The AI output is advisory. The signed report is the radiologist's, not the AI's.
- **The AI never certifies a scan as normal.** When no finding is flagged, the screen says *"No finding flagged by AI — NOT a normal read"*. The AI only checks for the finding types listed in the Findings tab; anything else is invisible to it.
- **Only one detector is validated on local patients** (the *study triage* detector). The *brain tumor* detector is a public model whose validation on this clinic's population is still pending — it is known to over-call tumors.
- The software has not been certified as a medical device by the Ministry of Health of the Republic of Uzbekistan or by any other regulator. Use it under the clinic's own clinical governance.

Everything runs offline on the clinic PC. No image, report or patient identifier leaves the machine.

---

## 2. Logging in and changing your password

Accounts are created by your administrator. Roles: **radiologist** (analyse, edit and sign reports), **admin** (same plus user management and audit log), **technician** (upload only; cannot run analysis or sign).

1. Start *Sentinel Medical AI* from the Start menu (Windows) or Applications (macOS). The AI server starts in the background; wait until the status line at the bottom of the login screen shows **AI server: online**.
2. Choose the interface language (РУС / O'ZB / ENG) — this is also the default language of new reports.
3. Enter your username and password and press **Sign in**.
   [Screenshot: login screen with language switcher and AI-server status line]
4. **First login with a temporary password:** the app opens a *Change password* form before anything else. Enter the temporary password, then a new one (at least 8 characters, twice). Until you do this, the workstation stays locked.
5. The session lasts 12 hours. When it expires you are returned to the login screen (*Session expired — please sign in again*). Anything you had not signed or exported stays on the server and reappears in the worklist after you log in.

To change your password later, ask your administrator (the change-password form is shown by the application whenever the server marks your account as *must change password*).

Sign out with **File → Sign Out**. Signing out clears every patient object from the screen.

---

## 3. First-run setup (once per workstation)

On the first start after installation the app opens a five-step **First-run setup** wizard: *Language → Clinic → AI server → License → Support*. Your IT administrator normally completes it; if it appears for you, ask IT. It can be re-opened later from **Preferences → Clinic → Run setup again**. The clinic name entered there is printed on the letterhead of every PDF.

---

## 4. The worklist

The left panel is the **worklist** — the studies known to this AI server, newest first.

[Screenshot: worklist with search box, modality pills and status badges]

- **Search** by patient ID, accession number or modality. **Filter** by modality with the pills under the search box. **Sort** by time, patient, modality or status.
- Each row shows the patient ID, modality and body part, study date, and a status: **Pending → Analyzing → Complete**, or **Error**. A **Needs review** badge means the AI could not honestly analyse the study (see §5.3) and it must be read entirely by the radiologist.
- After you log in the worklist is **restored from the server** (the last 100 studies). Selecting a restored study reloads its AI result, its report drafts and any signed reports.
- Click a row to open the study in the viewer and the right-hand panel.

---

## 5. Uploading a study

### 5.1 Steps

1. In the worklist press **Upload study**, then **Folder** (recommended — the whole study, all series) or **Files**. You can also **drag and drop** a folder or files onto the lower part of the worklist.
   [Screenshot: Upload study buttons and drop zone]
2. Wait for *Reading files…*, then *Uploading N files · x%*, then *Analyzing N files…*. The study appears in the worklist as **Analyzing** and switches to **Complete** when the result arrives.
3. The viewer shows the **actual slice the model analysed** and the right panel opens on the **Findings** tab.

If the status line says *AI server: offline*, the upload buttons are disabled (see §10).

### 5.2 Accepted formats

- **DICOM** files: `.dcm`, `.dicom`, or files without an extension (typical scanner exports). Hidden files and `DICOMDIR` are skipped automatically. Subfolders are included when you upload a folder.
- Compressed DICOM (JPEG 2000, JPEG-LS, RLE) and multi-frame files from Philips, GE and Siemens scanners are supported.
- **Not** accepted: NIfTI, JPEG/PNG screenshots, PDF, ZIP archives (unpack first). A selection with no DICOM files is refused with *No DICOM (.dcm) files in the selection*.
- Upload the **whole brain MRI study**. The panel picks the axial series it needs (T1 with contrast, T1, T2, FLAIR); if only one series is present it still runs, on that series.

### 5.3 Studies the AI will not analyse

- **Non-brain MRI** (spine, knee, abdomen …) is rejected by the brain panel: the study is stored with **Needs review** and *Not analyzed — requires full radiologist review*. Read it as you normally would.
- **Other modalities** (CT, X-ray, mammography) are routed to other detectors only when a matching model is installed on your server; otherwise they also come back as **Needs review**. The validated triage detector covers **brain MRI only**.
- A study that is not readable as DICOM is refused with an error message.

---

## 6. Reading the Findings tab

[Screenshot: Findings tab with disclaimer banner, overall assessment card, finding cards with two badges, and the models list]

### 6.1 The disclaimer banner

Every result starts with the same notice: *"AI triage assistant. It flags only the listed finding types and CANNOT certify a study as normal. Findings marked 'pending' are not validated on this clinic's population; 'experimental' findings are screening hints only. Every study is read and signed by a radiologist."* It is also printed on every PDF.

### 6.2 Overall assessment

- **AI flagged a finding — requires review** (red): at least one detector fired. The study should be prioritised for reading.
- **No finding flagged by AI — NOT a normal read** (grey): no detector fired. This is *not* a normal result — read the study in full.

Below the card: number of flagged findings, inference time and the decision **threshold** (0.50).

### 6.3 The two brain detectors in this build

| Detector | Status badge | What it does | Known limits |
|---|---|---|---|
| **Study triage — normal vs abnormal (local)** | **Validated** | Trained and tested on studies from a Tashkent hospital. Averages the abnormal-probability over 5 central axial slices; flags the study when the mean is ≥ 0.50. | Study-level sensitivity 0.90 and specificity 0.47 on 178 held-out local patients. In other words: about 1 in 10 abnormal studies is **not** flagged, and about half of normal studies **are** flagged. It prioritises; it does not diagnose, localise, or rule out. |
| **Brain tumor (glioma / meningioma / pituitary)** | **Pending validation** | Public 2D classifier trained on a Kaggle brain-tumor dataset; runs on one central slice. | **Over-calls tumors** on local scanners: many "Suspected glioma / meningioma" flags are false positives. Its "No tumor detected" line is likewise unverified. Treat every tumor flag as a prompt to look, never as a finding. |

Detectors marked **Experimental** (ischemic stroke on DWI, cerebral atrophy) exist in the software but are **not run** for whole-study analysis in this build. Hemorrhage, white-matter disease and hydrocephalus detection are not implemented.

### 6.4 Badges on each finding card

Each card carries **two badges**:

1. **Urgency** — derived from whether the detector fired and its validation status, never from the confidence number:
   - **REVIEW** — a *validated* detector flagged the study.
   - **UNVALIDATED FLAG** — a *pending* or *experimental* detector flagged something.
   - **NOT FLAGGED** — the detector did not fire.
2. **Validation status** of the detector — **Validated** (green), **Pending validation** (amber), **Experimental** (grey).

The card also shows the detector, the MRI sequence used, the raw class and the confidence. A confidence of, say, 0.92 from a *pending* detector is still an unvalidated flag.

### 6.5 Models list

At the bottom: the display name, a 12-character fingerprint (SHA-256) and the status of each model that ran, plus the AI-server version. The same identity is printed on every PDF, so a report can always be traced to the exact model release.

The brain panel does **not** draw lesion heat-maps or locations; the viewer shows the analysed slice only.

---

## 7. The Report tab

[Screenshot: Report tab with РУС / O'ZB / ENG buttons, edit pencil, text area and the Sign / Export buttons]

1. When the analysis finishes, a **draft report** is generated in the interface language from the AI findings (using the local language model when available; otherwise a fixed template — the toast *Template report — AI assistant unavailable* tells you which).
2. **Language buttons РУС / O'ZB / ENG** open the report in another language. A version opened in a language other than the original is marked *Auto-translated by AI — requires review* until it is signed. A dot on a button means a draft exists; a tick means that language is signed.
3. **Edit** (pencil icon) unlocks the text. Edit freely: the draft is only a starting point. The label *Draft edited by the doctor* appears, and your edit is stored on the server as a correction (used to improve future models; no patient identifiers are attached).
4. Everything above the signature line is an **AI DRAFT** until you sign. Draft PDFs carry the banner **AI DRAFT — NOT SIGNED** and a `_DRAFT` suffix in the file name.

The **Ask AI** tab lets you ask the local language model questions in the context of the current findings and report. Its answers are informational only and are not a diagnosis; nothing you type leaves the PC.

---

## 8. Signing a report

Signing is performed **by the AI server**, not by the app, so it is recorded even if the PC is later reinstalled.

1. Open the language you want to sign, check the text, and press **✓ Sign report**.
2. The server stores, in one transaction: the **signer** (your user account), the **date and time**, the **SHA-256 hash** of the exact report text, the AI draft it was based on, the AI findings, and the **model fingerprint(s)** that produced them. An audit-log entry is written.
3. The text becomes read-only (*Signed & locked*). The panel shows *Signed by <name> · <date/time>* and the full SHA-256.
   [Screenshot: signed report block with signer, timestamp and SHA-256]

Rules:

- **One signed report per study and language.** A second attempt returns *A report in this language is already signed*. A signed report cannot be edited or re-signed. If a correction is needed, follow your department's addendum procedure outside the application and inform IT.
- You can sign the same study in a second language; that is a separate signed record.
- Only the **radiologist** and **admin** roles can sign.
- You need to be logged in as yourself: the signature is your account.

---

## 9. Exporting a PDF

1. In the Report tab press **Export signed PDF** (or **Export draft PDF** for an unsigned draft). To export several opened languages in one document, switch the export mode to *Opened languages*.
2. Choose where to save (the default is your Downloads folder). The file is then shown in Explorer / Finder.

What the PDF contains: the clinic letterhead (name, address, phone from Settings), patient ID and study date, the report text, the AI findings with their status badges, the model identities with SHA-256 fingerprints and status, the decision threshold, the application version, the disclaimer, and either the **signature block** (signer, date/time, full SHA-256) or the red **AI DRAFT — NOT SIGNED** line. File name: `<patient or study id>_<study date>_<language>[_DRAFT].pdf`.

Cyrillic and Uzbek text render natively.

---

## 10. When the AI server is unavailable

The application polls the AI server every 15 seconds. A red banner at the top tells you the state:

| Banner | Meaning | What to do |
|---|---|---|
| **AI server is not running — contact IT** | The server on this PC is down or unreachable. | Upload and signing are disabled. The desktop shell restarts the server automatically up to 3 times; if it keeps failing a dialog names the log file. Read studies as usual on your normal viewer and contact IT (the support contact is shown in the banner and on the login screen). Nothing already stored is lost. |
| **AI server is degraded: the validated model failed to load — contact IT** | The server is up but the **validated triage model** is missing. Any result produced now comes only from *pending* detectors. | Do not use AI output for prioritisation until IT has fixed it. |
| **AI server reports an error — contact IT** | Internal error. | Contact IT. |
| *Template report — AI assistant unavailable* (toast) | The local language model (Ollama) is off. Findings are unaffected; the report draft is a template and *Ask AI* is unavailable. | Edit the template as needed; tell IT. |

If an upload fails with *Analysis failed — contact IT*, the study is not stored; retry later.

---

## 11. Who to contact

- **Clinic IT / administrator** — first line for login, licence, server and PDF problems. Their contact is shown on the login screen and in every error banner (configured in Preferences → Clinic → Support contact).
- **Vendor** — SIAA Medical AI, Tashkent · www.siaa.uz — through the contact named in your service agreement, for model updates, licences and suspected AI errors.
- **Clinical concerns** (a missed or false AI flag) — record the study ID (Details tab → Study), your assessment and the model fingerprint, and report it to the pilot lead / head of department. This feedback is what allows the models to be validated and improved.
