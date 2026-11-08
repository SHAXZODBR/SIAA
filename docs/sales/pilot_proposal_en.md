**Draft for legal review — not a legal document until reviewed by counsel**

# Pilot agreement proposal — Sentinel Medical AI, brain MRI triage

Proposal no. [____] · Date [__.__.____] · Software release 1.1.0 · Text in square brackets is to be completed by the parties.

## 1. Parties

| Role | Legal name | Represented by | Details |
|---|---|---|---|
| Clinic | [clinic legal name] | [full name, position, basis of authority] | [address, TIN, bank details] |
| Vendor | [SIAA Medical AI — legal form and name], Tashkent | [full name, position, basis of authority] | [address, TIN, bank details] |

Referred to below as "the Clinic" and "the Vendor", together "the Parties".

## 2. Purpose

To measure, at the Clinic and with the Clinic's scanners, how well the validated AI triage flag of Sentinel Medical AI agrees with the Clinic's radiologists, and whether the software fits the reading workflow — before any decision on routine use.

The pilot **does not change clinical care**. Every study is read and signed by a radiologist exactly as today; the AI output is compared with the signed report afterwards. The software is decision support within the limits of its intended use statement (Annex A). It has not been certified as a medical device by the Ministry of Health of the Republic of Uzbekistan or by any other regulator; the Clinic uses it under its own clinical governance. [Counsel to confirm the regulatory basis for pilot use.]

## 3. Scope

- **Indication:** adult brain MRI studies, enrolled **consecutively** during the pilot period and read by at least two participating radiologists of the Clinic.
- **Object of evaluation:** the detector "Study triage — normal vs abnormal (local)" (status *validated*). Vendor-validated performance on **178 held-out patients** not used in training (Annex A): **sensitivity 0.90** (118 of 131 abnormal studies flagged) and **specificity 0.47** (22 of 47 normal studies not flagged; 95% CI 0.33–0.61). The pilot measures the same quantities on the Clinic's own patients.
- **Switched off during the pilot:** the public brain tumour classifier and every other detector that is not validated (the default of this release). Switching one on requires the written agreement of both Parties; it is then evaluated descriptively only (protocol §7.2).
- **Out of scope:** CT, non-brain MRI, paediatric and post-operative studies, and any use of AI output to skip or shorten a reading.
- **Duration:** [N] weeks (the protocol recommends 8–12) from the pilot start date, or until the adjudicated set reaches its target (§4, criterion 1), whichever is later.
- **Workstations:** [number] PC(s) at [address, department].
- **Method:** the Clinic's pilot evaluation protocol (Annex B) — one-click adjudication of each AI flag after signing, an adjudication log identified by study ID only.

## 4. Success criteria

The metrics are those of the pilot evaluation protocol §7 (Annex B), computed at study level on the adjudicated set:

- **Primary:** sensitivity = flagged ÷ all abnormal adjudicated; specificity = not flagged ÷ all normal adjudicated; each with a **95% confidence interval (Wilson score method)**; positive and negative predictive value at the observed prevalence; overall flag rate. The results are compared with the vendor's validated figures and with the release-gate floors (0.85 / 0.40), stating whether the **lower bound** of each pilot confidence interval is above the floor.
- **Secondary (descriptive):** studies returned "needs review" or "not analysed by AI" and why, analysis failures, use and editing of report drafts (§7.2); time-to-read, median and interquartile range, baseline vs pilot (§7.3); uptime, restarts and median inference time (§7.4); safety events (§7.5).

The pilot is a **success** when **all** of the following hold (protocol §9):

1. At least 100 adjudicated studies, including at least 40 adjudicated as normal.
2. Triage sensitivity point estimate ≥ 0.85 **and** specificity ≥ 0.40, with the confidence interval reported; the pilot lead judges whether the lower bounds are acceptable for prioritisation use.
3. No safety event attributable to AI output.
4. System uptime of at least 95% during working hours; median analysis wait acceptable to the radiologists.
5. Time-to-read not worse than baseline (median).
6. Radiologists' assessment (short questionnaire at the end): the flag and the draft are useful or neutral, not harmful.

With about 60 abnormal and 40 normal studies the 95% intervals are roughly ±8–10 points for sensitivity and ±15 points for specificity: enough to detect a serious deviation from the validated figures, not enough to prove small differences. If criterion 2 fails, the model is not used for prioritisation at the Clinic until it has been re-validated; the adjudicated set stays at the Clinic as the locked reference for the next model version.

## 5. Responsibilities

| Clinic | Vendor |
|---|---|
| Provides a PC that meets the specification of the installation guide §2 (Windows 10/11 64-bit, 8 GB RAM minimum, 16 GB recommended) and keeps it physically secure, backed up and with disk encryption switched on. | Installs and configures the software together with the Clinic's IT administrator in one visit (about one afternoon). |
| Provides access to the studies: a PACS connection through a local Orthanc node, or a DICOM export folder, plus the IT administrator's time for installation and backups. | Issues a licence bound to the Clinic's PC for the pilot period and delivers the validated model as a bundle encrypted for that PC. |
| Appoints a pilot lead (senior radiologist), at least two participating radiologists, a data steward and an IT contact. | Gives one 60-minute training session for the radiologists and a briefing for the IT administrator. |
| Gives radiologist time for adjudication (one decision per study after signing) and for the time-to-read log; keeps the adjudication log. | Answers support requests **within 1 business day** ([09:00–18:00] Tashkent time, Monday–Friday) through [phone / email]. |
| Joins a weekly check-in of 15–30 minutes; reports suspected false or missed flags and every safety event to the Vendor within [2] business days. | Delivers model and application updates only after they have passed the release gate, and not during the measurement window unless both Parties agree in writing. |
| Produces the pilot report (protocol §10) signed by the pilot lead, the head of department and IT. | Helps compute the metrics and confidence intervals and draft the report; receives aggregated results only. |

## 6. Data handling

- All images, reports, logs, the application database and the adjudication log **stay on the Clinic's premises**. The software runs offline and does not transmit data. The Vendor takes no patient data off site; on-site support is performed in the presence of the Clinic's IT staff, and remote access, if ever needed, only with the Clinic's written permission for each session.
- At the end of the pilot the Clinic gives the Vendor the **aggregated metrics** (§4). The adjudication log (study ID, date, decision, time — no names, patient IDs or birth dates) and any other **de-identified study-level labels** are given to the Vendor **only with the Clinic's written consent** (Annex D). No images and no reports leave the Clinic under this agreement.
- Any later transfer of de-identified data for model training requires a separate written data agreement, an ethics approval where required, and compliance with the Law of the Republic of Uzbekistan "On personal data" (No. ZRU-547 of 2 July 2019), including its **localisation requirement** that personal data of citizens of Uzbekistan be processed and stored on databases physically located in Uzbekistan. [Counsel to confirm citations and current wording.]
- **Ownership:** the Clinic keeps all rights to its data — images, reports, labels and the adjudication log. A model that the Vendor trains or validates with data provided under such a written consent belongs to the Vendor; it contains no images or reports, and the Vendor does not pass the Clinic's data on to third parties.
- Security measures in the software: user accounts with passwords, an audit log with a tamper-evident hash chain, patient identifiers redacted from technical logs, model weights encrypted for the Clinic's PC. The Clinic is responsible for the physical security of the PC, operating-system updates and backups of the data directory.

## 7. Intellectual property and confidentiality

- The software, the models and the documentation remain the Vendor's intellectual property. The Clinic receives a non-exclusive, non-transferable right to use them on the licensed workstations for the pilot period. The Clinic does not copy, decompile, extract the model weights or pass the software to third parties.
- Each Party keeps the other Party's non-public information confidential, including prices and pilot results before an agreed joint release. Patient information is confidential without time limit.
- The Vendor may publish or present pilot results only in aggregated, de-identified form and only with the Clinic's written consent. The Clinic may use the results internally without restriction.
- Confidentiality obligations survive termination for [3] years.

## 8. Duration and termination

- The agreement takes effect on signature and ends [2] weeks after the end of the pilot period (time for the report), on [date] at the latest.
- Either Party may terminate with [14] days' written notice, and immediately in case of a safety event or a material breach.
- On termination or expiry the licence ends; the Vendor uninstalls the software on request; the Clinic's data and signed reports stay with the Clinic (the Vendor helps export signed reports as PDF on request).
- There is no obligation to buy. If the success criteria are met, the Parties may negotiate a commercial licence agreement.
- Responsibility for clinical decisions stays with the Clinic and its physicians; the signed report is the radiologist's professional act. [Counsel to draft the liability clause.]

## 9. Pricing

| Item | Amount | Notes |
|---|---|---|
| Pilot fee | [amount, UZS] | Installation, licence for the pilot period, training session, support during the pilot. |
| Per-workstation licence (after the pilot) | [amount, UZS per workstation per year] | Applies only if a commercial agreement is signed. |
| Support (after the pilot) | [amount, UZS per year / included] | Response within 1 business day; model updates through the release gate. |

[VAT treatment and payment terms to be confirmed by counsel and accounting.]

## 10. Annexes

- **Annex A** — Intended use statement (`docs/intended_use_en.md`, revision in force at signature).
- **Annex B** — Pilot evaluation protocol (`docs/pilot_evaluation_protocol_en.md`).
- **Annex C** — Clinic onboarding checklist (`docs/sales/clinic_onboarding_checklist_en.md`).
- **Annex D** — Written consent for the transfer of de-identified labels [to be drafted by counsel].

## 11. Signatures

| | Clinic | Vendor |
|---|---|---|
| Name, position | [ ] | [ ] |
| Signature | | |
| Date | [__.__.____] | [__.__.____] |
| Seal | | |
