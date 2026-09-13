# SIAA — Sentinel Medical AI · Documentation Index

On-premise brain MRI decision-support software for clinics in Uzbekistan. Version 1.0.0 · docs revision 2026-09.

All hospital-facing documents exist in three languages: Russian (`_ru`), Uzbek in Latin script (`_uz`) and English (`_en`). The Russian and Uzbek files each carry a note asking a practising radiologist to review the clinical terminology before use.

| Document | Audience | RU | UZ | EN |
|---|---|---|---|---|
| **User manual** — what the product is and is not, login, worklist, uploading a study, reading the Findings tab (validated / pending / experimental badges), the Report tab, signing, PDF export, what to do when the AI server is down | Radiologists | [user_manual_ru.md](user_manual_ru.md) | [user_manual_uz.md](user_manual_uz.md) | [user_manual_en.md](user_manual_en.md) |
| **Installation & administration guide** — PC specification, components, offline operation, first start and admin password, first-run wizard, licence activation, data directory and backup, logs and PHI redaction, updates (MANIFEST + release gate), environment variables, troubleshooting via `/health`, security notes | Hospital IT | [admin_install_guide_ru.md](admin_install_guide_ru.md) | [admin_install_guide_uz.md](admin_install_guide_uz.md) | [admin_install_guide_en.md](admin_install_guide_en.md) |
| **Intended use statement** — intended use, users, indications, contraindications and limitations, validated performance and how it was measured, responsibility statement, version and model identification | Clinical governance, management | [intended_use_ru.md](intended_use_ru.md) | [intended_use_uz.md](intended_use_uz.md) | [intended_use_en.md](intended_use_en.md) |
| **Pilot evaluation protocol** — how to run a 2–3 month pilot: scope, roles, workflow, one-click adjudication of AI flags, the ≥100-study adjudicated test set, metrics with 95% CI, data handling, success criteria, reporting template | Pilot lead, head of department, IT | [pilot_evaluation_protocol_ru.md](pilot_evaluation_protocol_ru.md) | [pilot_evaluation_protocol_uz.md](pilot_evaluation_protocol_uz.md) | [pilot_evaluation_protocol_en.md](pilot_evaluation_protocol_en.md) |
| **Attributions** — third-party datasets, model weights and software with licence names; statement on the locally trained validated model | Legal, IT, vendor | — | — | [attributions.md](attributions.md) (EN with RU summary) |

## Facts that every document relies on

- **Decision support only.** A radiologist reads and signs every study; the AI never certifies a scan as normal.
- **Detectors and status** (as reported by the software with every result): *Study triage — normal vs abnormal (local)* = **validated**; *Brain tumor* = **pending** (over-calls on local scanners); *stroke* and *atrophy* = **experimental**, not run in whole-study analysis.
- **Validated performance:** study-level sensitivity 0.90 / specificity 0.47 on 178 held-out local patients (mean over 5 central axial slices, threshold 0.50). No other performance figures are claimed.
- **Model identity:** `models/brain_triage_finetuned`, `MANIFEST.json` SHA-256 `0d559766ce58…`; shown in the app and printed on every PDF.
- **Deployment:** everything on one clinic PC, no internet; AI server on `127.0.0.1:8000`, Ollama on `localhost:11434`; authentication on by default; data in the server's data directory (`/health` → `data_dir`).
- The software is not certified as a medical device by any regulator.

Source of truth for these statements: `src/inference/brain_analysis.py`, `src/inference/model_registry.py`, `src/inference/server.py`, `src/inference/auth_routes.py`, `src/utils/auth.py`, `src/utils/database.py` (audit hash chain), `src/utils/license.py`, `src/utils/offline.py`, `src/utils/model_crypto.py`, `src/utils/paths.py`, `scripts/eval_study_level.py`, `scripts/download_all_models.py`, `models/brain_triage_finetuned/MANIFEST.json` and `study_level_eval.json`, `desktop-app/electron/main.js`, `desktop-app/src` (UI strings quoted in the manuals come from `desktop-app/src/i18n/{ru,uz,en}.ts`).

Drift guard: `python -m pytest -q docs/test_docs_consistency.py` checks that every file linked from this index exists, that the three language versions of each document have the same section structure, and that the routes, environment variables, CLI commands, detector statuses and validated figures quoted in the documents still match the code. It is not collected by the default `pytest -q` (whose `testpaths` is `tests/`), so run it explicitly after editing the docs.

Screenshots are marked as placeholders (`[Screenshot: …]`, `[Скриншот: …]`, `[Skrinshot: …]`) and should be captured from the final build before printing.
