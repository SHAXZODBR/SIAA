# SIAA — Sentinel Medical AI · Pilot Evaluation Protocol (2–3 months)

Version 1.0.0 · Document revision 2026-09

## 1. Purpose

To measure, at this clinic and with this clinic's scanners, how well the AI triage flag agrees with radiologists, and whether the software fits the reading workflow — before any decision on routine use. The pilot **does not change clinical care**: every study is read and signed by a radiologist exactly as today; the AI output is compared afterwards.

## 2. Scope

- **In scope:** adult brain MRI studies acquired during the pilot period and read by participating radiologists.
- **Primary object of evaluation:** the *Study triage — normal vs abnormal (local)* detector (status *validated*).
- **Secondary (descriptive only):** the *Brain tumor* detector (status *pending*), the usefulness of report drafts, and system availability.
- **Out of scope:** CT, non-brain MRI, paediatric studies, and any use of AI output to skip or shorten a reading.
- **Duration:** 8–12 weeks, or until the adjudicated set reaches its target size (§6), whichever is later.

## 3. Roles

| Role | Responsibilities |
|---|---|
| **Pilot lead** (senior radiologist) | Owns the protocol, adjudication log and final report; resolves disagreements; reports safety events. |
| **Participating radiologists** (≥ 2) | Read and sign as usual; adjudicate AI flags; note reading times. |
| **Hospital IT administrator** | Installation, backups, uptime log, de-identified export at the end. |
| **Data steward** (may be the pilot lead) | Keeps the adjudication log; ensures no PHI leaves the clinic. |
| **Vendor contact** | Model/app updates (none during the measurement window unless agreed), technical support, receives the aggregated metrics only. |

## 4. Workflow

1. **Baseline (weeks 1–2, optional but recommended):** radiologists read and sign without opening the AI panel; the pilot lead records time-to-read for ≥ 30 studies (§7.3) to have a comparison.
2. **Pilot reading (each study):**
   1. Upload the whole study to Sentinel (worklist → Upload study → Folder). The analysis runs while the radiologist reads.
   2. The radiologist reads the study **as usual**, writes the report (the AI draft may be used as a starting point, edited freely) and **signs** it in the application.
   3. Only **after signing**, the radiologist opens the **Findings** tab and adjudicates each AI flag (§5). The signed report is the reference standard; the AI never changes it.
3. **Weekly:** the pilot lead reviews the adjudication log, checks counts against the target, and records any safety event or usability issue.
4. **End of pilot:** IT exports the de-identified metrics (§8); the pilot lead writes the report (§10).

A model or application update during the measurement window restarts the count for that model version — the model fingerprint is recorded with every study for this reason.

## 5. One-click adjudication of each AI flag

For every analysed study the radiologist gives **one decision per detector**, after signing:

| Decision | Meaning |
|---|---|
| **Agree** | The AI flag (or "not flagged") matches the signed report: flagged & report describes a relevant abnormality; or not flagged & report is normal / no relevant finding. |
| **Disagree** | The AI flag contradicts the signed report: flagged but the study is normal (false flag), or not flagged but the report describes a relevant abnormality (missed). |
| **Not applicable** | The AI did not analyse the study (rejected / needs review), the study is out of scope, or the report is inconclusive. Excluded from sensitivity/specificity. |

The decision takes seconds and needs no free text. Record it in the **adjudication log** (a spreadsheet kept by the data steward; template below). Identify the study by the **study ID** shown in the Details tab of the application (a server-generated identifier) — **not** by patient name.

```
study_id | date | radiologist | modality | ai_triage_flag (yes/no) | triage_decision (agree/disagree/na) |
tumor_flag (yes/no/none) | tumor_decision (agree/disagree/na) | report_finding_class (normal/tumor/stroke/atrophy/hemorrhage/other) |
draft_used (yes/no) | time_to_read_min | model_fingerprint | comment (optional, no PHI)
```

"Relevant abnormality" is defined in advance by the pilot lead (recommended: any finding that changes management or requires follow-up; incidental normal variants count as normal). Disagreements about a decision are settled by the pilot lead, whose decision is final and recorded.

## 6. The adjudicated test set

- **Target: ≥ 100 adjudicated brain MRI studies, including ≥ 40 studies adjudicated as normal** (no relevant finding in the signed report). Normal studies are the scarce class and drive the precision of the specificity estimate; include every consecutive normal study, do not select.
- Enrol **consecutive** studies during the pilot period; do not pick "interesting" cases.
- One study per patient (if a patient is scanned twice, keep the first).
- The set stays inside the clinic; it is the locked reference for this and future model versions.

With 60 abnormal and 40 normal studies the 95% confidence intervals are roughly ±8–10 points for sensitivity and ±15 points for specificity — enough to detect a serious deviation from the validated figures, not enough to prove small differences.

## 7. Metrics to report

### 7.1 Primary (triage detector, study level)

- **Sensitivity** = flagged ÷ (all abnormal, adjudicated); **specificity** = not flagged ÷ (all normal, adjudicated); each with a **95% confidence interval** (Wilson score method).
- Positive and negative predictive value at the observed prevalence; overall flag rate.
- Comparison against the vendor's validated figures (sensitivity 0.90, specificity 0.47 on 178 local patients) and against the release-gate floors (0.85 / 0.40): report whether the **lower bound** of the pilot CI is above each floor.

### 7.2 Secondary (descriptive)

- Tumor detector: agree / disagree counts; false-flag rate among normal studies.
- Share of studies returned "needs review" and why; analysis failures.
- Report drafts: % of reports where the draft was used as a starting point; % edited (from the server's corrections table).

### 7.3 Time-to-read

Minutes from opening the study to signing the report, recorded per study (from the reading log or the application's timestamps: analysis time and `signed_at`). Report median and interquartile range for the baseline weeks and for the pilot weeks.

### 7.4 System

Uptime during working hours (from IT's log and `/health` checks), number of restarts, median inference time per study.

### 7.5 Safety

Any event in which AI output could have influenced a reading in the wrong direction (e.g., a real finding downgraded after seeing "not flagged"). Target: zero; every such event is described in the report regardless of outcome.

## 8. Data handling

- All studies, reports, logs and the adjudication log stay **on the clinic's premises**. The application itself never transmits data.
- The adjudication log contains study IDs, dates, decisions and times — **no names, patient IDs or birth dates**. Store it on the clinic network with access restricted to the pilot team.
- At the end, IT produces a **de-identified export for the vendor**: the aggregated metrics (§7) and, if the clinic's data agreement allows it, the adjudication log table with study IDs only. No images, no reports, no database file.
- Retain the adjudication log and the metrics report with the pilot documentation for the clinic's records; the signed reports remain in the application's database as usual.
- The vendor may use the aggregated results to re-validate or retrain models only under the existing clinic data agreement.

## 9. Success criteria

The pilot is a success and routine use may be considered when **all** of the following hold:

1. ≥ 100 adjudicated studies including ≥ 40 normal.
2. Triage sensitivity point estimate ≥ 0.85 **and** specificity ≥ 0.40, with the CI reported; the pilot lead judges whether the lower bounds are acceptable for prioritisation use.
3. No safety event attributable to AI output.
4. System uptime ≥ 95% during working hours; median analysis wait acceptable to the radiologists.
5. Time-to-read not worse than baseline (median).
6. Radiologists' assessment (short questionnaire at the end): the flag and the draft are useful or neutral, not harmful.

If criterion 2 fails, the model must not be used for prioritisation at this clinic until re-validated; the adjudicated set is exactly what allows the vendor to fix it.

## 10. Reporting template

```
PILOT EVALUATION REPORT — Sentinel Medical AI
Clinic: ____________   Period: ______ – ______   Pilot lead: ____________
Software version: 1.0.0   Triage model fingerprint: ____________   Tumor model fingerprint: ____________

1. Enrolment: studies analysed ___ ; adjudicated ___ (normal ___ / abnormal ___); not applicable ___ (reasons)
2. Triage detector: sensitivity ___ (95% CI ___–___); specificity ___ (95% CI ___–___); PPV ___; NPV ___; flag rate ___
   Comparison with validated figures / gate floors: ____________
3. Tumor detector (descriptive): agree ___ / disagree ___ ; false flags among normals ___
4. Report drafts: used ___% ; edited ___% ; typical corrections: ____________
5. Time-to-read: baseline median ___ min (IQR ___) ; pilot median ___ min (IQR ___)
6. System: uptime ___% ; restarts ___ ; median inference time ___ s ; incidents: ____________
7. Safety events: ____________ (none / described)
8. Radiologist feedback summary: ____________
9. Conclusion against success criteria (1–6): pass / fail per criterion
10. Recommendation: continue routine use / extend pilot / stop pending model update
Attachments: de-identified adjudication log (study IDs only), IT uptime log
Signatures: pilot lead ______ ; head of department ______ ; IT ______
```
