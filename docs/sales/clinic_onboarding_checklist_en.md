# Clinic onboarding checklist — Sentinel Medical AI pilot

Software release 1.1.0 · For the vendor's project lead, the clinic's pilot lead and the clinic's IT administrator. Tick each box and write the date and initials next to it; keep the completed checklist with the pilot documentation. Never write patient names or patient IDs on this checklist.

Owners: **V** = vendor, **C** = clinic (pilot lead / head of department), **IT** = clinic IT administrator.

## 1. Signed agreement

- [ ] **V + C** — Pilot agreement signed after legal review (`pilot_proposal_en.md`), with annexes A–D.
- [ ] **C** — Pilot team named: pilot lead (senior radiologist), at least two participating radiologists, data steward, IT administrator. **V** — vendor contact named, with phone and email.
- [ ] **C** — "Relevant abnormality" defined in writing by the pilot lead (pilot evaluation protocol §5); recommended: any finding that changes management or needs follow-up.
- [ ] **C** — Decision recorded on Annex D: may the de-identified adjudication log (study IDs only) be given to the vendor at the end? Yes / no.
- [ ] **V + C** — Installation date, training date and pilot start date agreed.

## 2. IT prerequisites

- [ ] **IT** — PC meets the installation guide §2: Windows 10/11 64-bit, 4-core CPU (8 recommended), 8 GB RAM (16 GB if the local language model runs on the same PC), 20 GB free SSD (50 GB recommended), display 1600×1000.
- [ ] **IT** — Operating system updated; disk encryption (BitLocker) switched on; a local administrator account available for installation.
- [ ] **IT** — Hardware, host name and network card **final** — the licence is bound to the machine fingerprint, which changes if any of these change.
- [ ] **IT** — Source of studies chosen: PACS through a local Orthanc node (scanner or PACS sends to it), or a DICOM export folder.
- [ ] **IT** — Optional local language model for report wording installed while the PC still has internet access (`ollama pull gemma3:4b`); without it the software uses fixed report templates.
- [ ] **IT** — Backup target for the data directory available (a second disk or a network share inside the clinic).
- [ ] **IT** — Helpdesk email and phone ready for the "Support" step of the first-run wizard.

## 3. Installation

- [ ] **V + IT** — Run the Windows installer. For per-clinic model delivery this is the installer flavour without the validated model (artifact name ending in `-NO-VALIDATED-MODEL`). Accept the medical disclaimer.
- [ ] **IT** — First start: find the one-time admin password in the log (`<data dir>\logs\`, banner *FIRST-RUN ADMIN ACCOUNT CREATED*), log in, change it at once. Store it in the clinic's password safe.
- [ ] **IT** — First-run wizard: language, clinic name (printed on every PDF), AI server `http://127.0.0.1:8000`, support contact.
- [ ] **IT** — One personal account per radiologist (role *radiologist*); no shared accounts. Each user changes the temporary password at first login.
- [ ] **IT** — Data directory path noted (from `/health` → `data_dir`) and the first backup job configured.

## 4. Machine fingerprint

- [ ] **IT** — Read the fingerprint (64 hex characters) in Preferences → License, or, on the installed PC, run `<install>\resources\backend\sentinel-backend\sentinel-backend.exe --print-fingerprint` (a source install from the repository uses `venv\Scripts\python.exe -m src.utils.license fingerprint` instead).
- [ ] **IT → V** — Send the fingerprint and the clinic's legal name to the vendor through the agreed channel. The fingerprint is not secret and contains no patient data.

## 5. Licence and encrypted model delivery

- [ ] **V** — Produce the clinic package on the vendor workstation with `packaging/provision_clinic.py` (licence with a model key, model bundle encrypted for this fingerprint, admin README in Russian and English); the issue is recorded in the vendor's licence ledger. See `docs/vendor_provisioning_en.md`.
- [ ] **V → IT** — Hand over the package folder: `license.dat`, `brain_triage_finetuned_<identity>_<fp8>.zip` (`<fp8>` = the first 8 characters of the fingerprint), `README_admin_ru.txt`, `README_admin_en.txt`. The application accepts the bundle only on this PC with this licence. Keep the folder confidential: `license.dat` together with the zip is the key to the model weights.
- [ ] **IT** — Preferences → License → **Import license.dat**; restart the application; Preferences → License shows *Licensed*, the customer name and the expiry date.
- [ ] **IT** — Preferences → AI models → **Import model bundle** (admin), choose the zip. The server checks both checksums and test-decrypts before switching models. A toast shows the new model identity: it must equal the identity printed in the vendor README.
- [ ] **IT** — If the import fails with `license_model_key_missing`, the licence was not imported first (or the app not restarted): import `license.dat`, restart, import the bundle again. If it fails with `bundle_not_for_this_machine`, the installed licence is not the one from the same folder, or the bundle was made for another fingerprint: import the licence from the same folder and restart; if it still fails, send the current fingerprint to the vendor again. Any other failure: Preferences → About → **Export diagnostics** and send the zip to the vendor (it contains no patient data).

## 6. Verification

- [ ] **IT** — Open `http://127.0.0.1:8000/health` on the PC: `status` is ok, `version` is 1.1.0, `license.mode` is `licensed`, `offline` is true, `ai_settings.show_pending_detectors` is false, `models.brain_triage.loaded` is true and `models.brain_triage.identity` equals the identity in the vendor README; in Preferences → AI models the *Model registry* list shows the triage model with the status *Validated*.
- [ ] **IT** — Status line of the app shows **AI server: online**; no red banner.
- [ ] **C** — Upload one brain MRI study chosen by the pilot lead. The Findings tab shows the *Study triage* card with the **Validated** badge and no tumour card; a head CT, if tried, shows "not analysed by AI".
- [ ] **C** — Export a **draft** PDF of that study and check: clinic letterhead, application version 1.1.0, the triage model identity (12 characters) equal to the identity in the vendor README, the decision threshold, the disclaimer and the red *AI DRAFT — NOT SIGNED* line. Destroy the test PDF afterwards according to clinic rules.
- [ ] **IT** — Audit log chain verified (`/audit/verify` returns `ok: true`, see installation guide §11).
- [ ] **IT** — First backup of the data directory completed and a restore tried on a copy.

## 7. Radiologist training session (60 minutes)

| Minutes | Topic | Material |
|---|---|---|
| 0–10 | What the software is and is not: decision support, the radiologist signs every study, no normal certificate, tumour classifier switched off | `one_pager_en.md`, intended use statement |
| 10–20 | The validated numbers in plain words: what a flag means, what "not flagged" does not mean | One-pager, user manual §6.3 |
| 20–35 | Hands-on: sign in, change password, upload a study, Findings tab, badges, "needs review" and "not analysed by AI" | User manual §2–6 |
| 35–45 | Report tab: draft, edit, three languages, sign, export the signed PDF | User manual §7–9 |
| 45–55 | Adjudication: read and sign first, then one decision per study (agree / disagree / not applicable), the adjudication log by study ID only | Pilot evaluation protocol §4–5 |
| 55–60 | When the AI server is down, who to call, how to report a suspected false or missed flag or a safety event | User manual §10–11 |

- [ ] **C** — Attendance list signed by every participating radiologist.

## 8. Pilot start

- [ ] **C** — Optional baseline (weeks 1–2): time-to-read recorded for at least 30 studies without opening the AI panel (protocol §4, §7.3).
- [ ] **C** — Adjudication log created on the clinic network, access limited to the pilot team; template columns from protocol §5.
- [ ] **C** — Pilot start date, application version and triage model identity recorded in the pilot file.
- [ ] **V + C** — Measurement window frozen: no model or application update unless both parties agree in writing (an update restarts the count for that model version).

## 9. Weekly check-ins

- [ ] **C + V** — Every week, 15–30 minutes: adjudicated studies so far and normal studies so far (targets: at least 100 and at least 40), disagreements settled by the pilot lead, safety events (target zero), "needs review" reasons, uptime and restarts, open support requests.
- [ ] **V** — Every support request answered within 1 business day; the answer noted in the check-in minutes.
- [ ] **C** — Minutes kept with study IDs only — no names, patient IDs or dates of birth.

## 10. Pilot report

- [ ] **IT** — Aggregated metrics exported; the de-identified adjudication log (study IDs only) given to the vendor only if Annex D allows it. No images, reports or database files leave the clinic.
- [ ] **C + V** — Metrics computed as in protocol §7 (sensitivity and specificity with 95% Wilson confidence intervals, predictive values, flag rate, time-to-read, uptime, safety events).
- [ ] **C** — Report written on the template of protocol §10 and judged against the success criteria of protocol §9; signed by the pilot lead, the head of department and IT.
- [ ] **C + V** — Decision recorded: continue to routine use (commercial agreement), extend the pilot, or stop pending a model update that has passed the release gate.
