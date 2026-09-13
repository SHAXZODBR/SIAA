# SIAA — Sentinel Medical AI · Intended Use Statement

Version 1.0.0 · Document revision 2026-09

## Intended use

Sentinel Medical AI is an **on-premise decision-support software** that analyses brain MRI studies and (a) flags studies that may contain an abnormality so that they can be **prioritised for radiologist review**, and (b) drafts a structured radiology report in Russian, Uzbek or English for the radiologist to edit and sign. It is a triage aid. It does not make a diagnosis, does not localise lesions, and does not declare any study normal.

## Intended users

Licensed radiologists and physicians of the clinic in which the software is installed, working under the clinic's clinical governance; hospital IT staff for installation and administration. Not intended for patients or for non-medical users.

## Indications

- **Brain MRI** (adult routine brain protocols; axial T1, T1 with contrast, T2 and/or FLAIR series): prioritisation of studies for reading. This is the only validated indication.
- **Head CT**: the software can route a head CT to a public intracranial-hemorrhage classifier when that model is installed; its output is marked *pending validation* and **no performance claim is made for CT**. Where the model is not installed the study is returned as "requires review" without AI output.
- Other modalities and body regions are outside the intended use; such studies are returned as "requires review".

## Contraindications and limitations

- **Not for stand-alone diagnosis, screening, or exclusion of disease.** A study with no AI flag has *not* been read as normal: about 1 in 10 abnormal studies in local validation was not flagged.
- **Not a normal certificate.** The detector panel only recognises the finding types it lists; any other pathology is invisible to it.
- The **brain tumor classifier** (glioma / meningioma / pituitary) is a public model with *pending* validation on the local population and is known to over-call tumors. Its flags — and its "no tumor" output — must not be used clinically without the radiologist's own reading.
- Ischemic stroke and cerebral atrophy detectors are *experimental* and are not executed in whole-study analysis. Hemorrhage, white-matter disease and hydrocephalus detection are not implemented.
- Performance has **not** been established for: paediatric patients, post-operative or post-treatment brains, non-standard or non-axial protocols, studies with heavy motion or metal artefacts, scanner models absent from the local training data, or DICOM exports with missing header information.
- Not intended for time-critical decisions (e.g., acute stroke or hemorrhage rule-out).
- Report drafts, translations and *Ask AI* answers are produced by a local language model or by templates; they may contain errors and must be reviewed word by word before signing.
- The software has not been certified as a medical device by the Ministry of Health of the Republic of Uzbekistan or by any other regulatory body.

## Performance summary (validated detector only)

| Detector | Status | Metric | Value |
|---|---|---|---|
| Study triage — normal vs abnormal (local) | validated | Study-level sensitivity | **0.90** (118 of 131 abnormal studies flagged) |
| | | Study-level specificity | **0.47** (22 of 47 normal studies not flagged) |
| | | Accuracy | 0.79 |

**How it was measured.** The triage model (a ViT-B image classifier) was fine-tuned on brain MRI studies from a Tashkent hospital, with study-level labels derived from the hospital's own radiology reports, and tested on **178 held-out patients** (131 abnormal, 47 normal) not used in training — 890 axial slices, 5 central slices per study. For each study the abnormal-probabilities of the 5 slices are averaged and the study is flagged when the mean is ≥ 0.50 — the same protocol the software runs in production. Evaluation script: `scripts/eval_study_level.py`, results in `models/brain_triage_finetuned/study_level_eval.json` (evaluated 2026-09-07). A model release is blocked by the same script if it falls below sensitivity 0.85 or specificity 0.40. Note the small number of normal studies (47): the specificity estimate is imprecise (95% CI roughly 0.33–0.61). The labels are report-derived and applied at study level, not per-slice expert annotations. A radiologist-adjudicated pilot test set (see `pilot_evaluation_protocol_en.md`) is required to confirm these figures at each installation.

No performance figures are claimed for the brain tumor classifier, for CT, or for report text quality.

## Responsibility statement

Sentinel Medical AI provides decision support only. **A qualified radiologist reads every study in full and signs every report; the signed report is the radiologist's professional act and responsibility.** AI output must never be the sole basis for a clinical decision. Radiologists are asked to report every suspected false or missed AI flag to the clinic's pilot lead so that the models can be validated and improved.

## Version and model identification

| Item | Identifier |
|---|---|
| Desktop application / AI server | 1.0.0 (shown on the login screen, status bar, `/health`, and every PDF) |
| Validated triage model | `models/brain_triage_finetuned` — `model.safetensors` SHA-256 `0d559766ce58…` (pinned in `MANIFEST.json`; the first 12 characters are shown in the app and printed on reports) |
| Brain tumor classifier (pending) | public Hugging Face model bundle; identity (name + SHA-256 fingerprint + status) shown with each result |
| Decision threshold | 0.50 (printed on every report) |
| Report language model | Gemma 3 via Ollama, local; template fallback |

Every result and every PDF carries the identities and validation status of the models that produced it, so any report can be traced to the exact model release.
