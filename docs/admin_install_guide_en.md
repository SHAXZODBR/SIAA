# SIAA — Sentinel Medical AI · Installation and Administration Guide (Hospital IT)

Version 1.0.0 · Document revision 2026-09

This guide is for the IT administrator of the clinic. The radiologist-facing manual is `user_manual_en.md`.

---

## 1. Architecture in one paragraph

Everything runs on **one clinic PC** (the radiologist's workstation or a small server next to it):

- **Desktop app** (Electron + React) — the user interface. In a packaged build it starts the AI server itself.
- **Python AI server** (FastAPI, PyTorch) listening on **127.0.0.1:8000**. Reads DICOM, runs the brain detector panel, generates report drafts, stores studies, signed reports and the audit log in a local SQLite database.
- **Models folder** — read-only weight files shipped with the install.
- **Ollama** with a Gemma 3 language model on **localhost:11434** — used only for report wording, translation and the *Ask AI* tab. Optional: without it the server falls back to fixed report templates.

No internet connection is required or used during operation. Nothing is uploaded anywhere.

```
scanner export / USB / shared folder → Desktop app → 127.0.0.1:8000 AI server → SQLite + logs (data directory)
                                                              └→ localhost:11434 Ollama (report text only)
```

---

## 2. Minimum PC specification

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 64-bit (NSIS installer, x64) or macOS 12+ (DMG, arm64/x64) | Windows 11 Pro (BitLocker available) |
| CPU | 4-core x86-64 / Apple silicon. **CPU-only inference works**; no GPU required. | 8 cores |
| RAM | 8 GB (AI server + desktop app) | **16 GB** when Ollama runs on the same PC (Gemma 3 4B needs ~8 GB by itself) |
| Disk | 20 GB free on SSD: app + Python backend ~3 GB, models ~0.5–2 GB, Ollama model ~3.3 GB, plus the growing data directory | 50 GB free SSD |
| GPU | Optional. CUDA (NVIDIA) or Apple MPS is used automatically when present (`device` in `/health`). | — |
| Network | None required. Ports 8000 and 11434 are bound to localhost only. | — |
| Display | 1600×1000 or larger (the app opens maximised; minimum window 1200×800) | — |

Inference time per brain study on CPU is seconds to tens of seconds; it is shown with every result.

---

## 3. What gets installed

| Component | Location (Windows) | Notes |
|---|---|---|
| Desktop app | Program Files (or per-user) — `Sentinel Medical AI` | Start-menu and desktop shortcuts. The installer shows a bilingual medical disclaimer and must be accepted. |
| Python backend | `<install>\resources\backend\` — contains `run_server.py`, `src\`, its own `venv\` | Spawned by the app on start. Its stdout/stderr go to `<data dir>\logs\backend.log`. |
| Models | `<install>\resources\backend\models\` (`SENTINEL_MODELS_DIR` overrides) | `brain_triage_finetuned\` (validated triage, ~330 MB, with `MANIFEST.json`), `hf\` (public tumor classifier bundle), optional others. |
| Ollama + Gemma 3 | Separate installer (ollama.com); model pulled with `ollama pull gemma3:4b` | Must be installed **before** the PC goes offline. |
| Data directory | See §6 | Created on first start. |

On macOS the equivalent paths are inside `Sentinel Medical AI.app/Contents/Resources/backend`.

---

## 4. Fully offline operation

- The server runs in **offline posture by default** (`SENTINEL_OFFLINE=1` whenever it is not a developer run). Hugging Face and transformers are put into offline mode at import time; **no model is ever downloaded at runtime**.
- Every model must therefore be present in the models folder before first use. A missing model does not crash the server: `/health` reports it as not loaded with the reason *"model missing — run scripts/download_all_models.py --only <key> on a machine with internet and copy models/ here"*.
- Cloud language models are disabled (`SENTINEL_ALLOW_CLOUD_LLM` is unset). Report text comes from Ollama on localhost or from templates.
- If you prepare the models folder on a machine **with** internet, run `python scripts/download_all_models.py` there and copy the resulting `models\` folder to the clinic PC.

---

## 5. First start

### 5.1 The admin account

The very first start creates the `admin` account (role *admin*):

- If the environment variable **`SENTINEL_ADMIN_PASSWORD`** is set when the server starts for the first time, that value is the admin password.
- Otherwise a **random one-time password** is generated and printed **once**: to the server console and, as WARNING lines, to the log file `<data dir>\logs\sentinel_YYYY-MM-DD.log` (in a packaged build also to `<data dir>\logs\backend.log`). Look for the banner *FIRST-RUN ADMIN ACCOUNT CREATED (one-time password)*.
- Either way the account is flagged **must change password**: the first login opens a change-password form and nothing else works until the password is replaced (minimum 8 characters in the app).

Never leave the one-time password in a file other people can read. The log line is the only place it is stored.

### 5.2 The first-run wizard

On first launch the desktop app opens the **First-run setup** wizard (re-openable from *Preferences → Clinic → Run setup again*):

1. **Language** — interface language and default report language (RU / UZ / EN).
2. **Clinic** — clinic name (required; printed on every PDF letterhead), address, phone.
3. **AI server** — the server URL; keep `http://127.0.0.1:8000` when the server runs on this PC. The wizard shows the live server status.
4. **License** — machine fingerprint and `license.dat` import (§5.3).
5. **Support** — the email/phone shown to doctors in every error banner and on the login screen. Put your own helpdesk here.

[Screenshot: first-run wizard, step 3 "AI server" with a green status]

### 5.3 License activation

The server refuses nothing at first start: **without a valid licence it runs in demo mode — 10 analyses per day**, after which analysis requests are rejected (HTTP 402) until the next day. Signing and viewing keep working.

The licence is a file **`license.dat`**, RSA-signed by the vendor and **bound to this machine's fingerprint** (host name, OS, CPU model, primary MAC address, OS machine ID).

1. Get the fingerprint. Either read it in *Preferences → License* (the app asks the AI server), or on the PC run:
   ```
   cd <install>\resources\backend
   venv\Scripts\python.exe -m src.utils.license fingerprint
   ```
2. Send the fingerprint string (64 hex characters) and the clinic name to the vendor.
3. Save the returned `license.dat` into the **data directory** (§6) — via *Preferences → License → Import license.dat*, or by copying the file there manually.
4. Restart the application (which restarts the server). Check `/health` → `license.mode` is `licensed`, or *Preferences → License* shows *Licensed*, customer name and expiry.
5. To verify a file by hand: `venv\Scripts\python.exe -m src.utils.license verify --license-file <path>`.

The licence becomes invalid if the machine fingerprint changes (new network card, renamed host, cloned VM) or after the expiry date — the reason is shown in *Preferences → License* and in the log. Contact the vendor for a re-issue; the software drops to demo mode meanwhile, it does not stop.

---

## 6. The data directory

All mutable state lives in **one folder** — never inside the program folder:

| Platform | Default location |
|---|---|
| Windows | `%APPDATA%\sentinel-medical-ai\` |
| macOS | `~/Library/Application Support/sentinel-medical-ai/` |
| Linux | `~/.local/share/sentinel-medical-ai/` |

The packaged desktop app passes its own user-data folder to the server via `SENTINEL_DATA_DIR`. **The authoritative path is reported by the server**: `GET /health` → `data_dir`, and in *Preferences → Connections → Server data folder*. Use that path, not this table, when in doubt.

Contents:

| Path | What it is | PHI? |
|---|---|---|
| `sentinel.db` (+ `-wal`, `-shm`) | SQLite database: users (bcrypt hashes), studies (DICOM header fields incl. patient ID/name/birth date), AI results, **signed reports** (text, signer, time, SHA-256, model identity), corrections, **audit log** | **Yes** |
| `logs\sentinel_YYYY-MM-DD.log` | Server log, rotated at 10 MB, zipped, kept 30 days. Patient identifiers are redacted (§7). | Redacted |
| `logs\backend.log` | Raw console output of the server captured by the desktop shell | Redacted |
| `doctor_corrections\` | Append-only JSONL of radiologist edits to AI drafts (report text, study ID, user ID) | Report text |
| `dicom_cache\` | Temporary DICOM cache | Yes |
| `license.dat` | The machine-bound licence | No |
| `jwt_secret.key` | Secret that signs login tokens (mode 0600). Deleting it logs everyone out. | No |
| `model_key.bin` | Per-install secret for encrypted model weights (if the release ships encrypted weights) | No |

Uploaded DICOM files are written to a temporary folder with server-chosen names during analysis and deleted immediately afterwards; the server keeps header fields and a small preview image, not the study itself.

The training-data collector additionally writes **anonymised** thumbnails and labels under `data\training_corpus\` in the backend working directory (no names, IDs or birth dates).

### 6.1 Backup

The database is stored **unencrypted**. Run the PC on an encrypted volume (BitLocker / FileVault) and back up as follows:

1. Close the desktop application (this stops the server) or make sure no analysis is running.
2. Copy the **entire data directory** (§6) to an external drive or NAS — at minimum `sentinel.db`, `sentinel.db-wal`, `sentinel.db-shm`, `doctor_corrections\`, `license.dat`, `jwt_secret.key`, `logs\`.
3. Keep three copies (local, external, off-site), encrypt removable media, and **test a restore** monthly on a spare machine: install the same version, copy the folder back, start, log in, open a signed report.

A helper exists (`scripts\backup_restore.py backup --out <dir>`, producing a checksummed `.tar.gz`), but it expects the database under the backend's `data\` folder; verify that it actually captured your data directory before relying on it. The plain folder copy above is always correct.

### 6.2 Uninstalling

The Windows uninstaller asks whether to delete the data folder. Answer **No** unless the clinic has decided to destroy the records; the signed reports and the audit log are the clinic's medical-legal record.

---

## 7. Logs and PHI redaction

- The server log (`logs\sentinel_*.log`) records study IDs (server-generated UUIDs), routes, timings, model load status, login attempts by username and errors. A global filter replaces obvious patient identifiers (`PatientID=…`, `PatientName=…`, birth dates and DICOM `Last^First` person names) with `<redacted>` before any line is written. Client file names are never logged.
- The **audit log** (database table, also `GET /audit/log` for admins, last 200 entries) records who did what and when: `login`, `analyze_study`, `view_study`, `sign_report`, `change_password`, `register`, with the client IP and details such as the report hash.
- The desktop app itself logs only to its own console (developer tools are open only in development builds).

When sending logs to the vendor, send `sentinel_*.log` / `backend.log` only; never the database.

---

## 8. Updating the app and the models

### 8.1 Application update

1. Back up the data directory (§6.1).
2. Run the new installer over the old version (Windows) or replace the app bundle (macOS). The data directory is untouched.
3. Start the app once as admin. The database **migrates itself** (new columns are added; nothing is dropped). Check `/health` → `version` and the login-screen version.
4. The desktop app and the server carry the same version number (1.0.0 at this revision); the version is printed on every PDF.

### 8.2 Model update — the MANIFEST and the release gate

The validated detector is a folder: `models\brain_triage_finetuned\` containing `model.safetensors`, `config.json`, `preprocessor_config.json`, **`MANIFEST.json`** and `study_level_eval.json`.

- `MANIFEST.json` pins the **SHA-256 of every weight file**. The first 12 hex characters of the `model.safetensors` hash are what the app shows as the model fingerprint (Details tab, Findings → Models, every PDF). Current release: `0d559766ce58…`.
- Before the vendor ships a model, it must pass the **release gate**: `scripts\eval_study_level.py` verifies the manifest, evaluates the model study-by-study on the held-out local validation set and refuses the release (non-zero exit) if study-level sensitivity < 0.85 or specificity < 0.40 at the production operating point (mean over 5 central slices, threshold 0.50). The resulting `study_level_eval.json` ships with the model.

To install a model update:

1. Stop the application.
2. Replace the whole `brain_triage_finetuned\` folder with the one supplied by the vendor (keep a copy of the old one).
3. Verify the hashes yourself: Windows `certutil -hashfile model.safetensors SHA256`, macOS `shasum -a 256 model.safetensors`; compare with `MANIFEST.json`.
4. Start the application; check `/health` → `models.brain_triage.loaded = true` and, after one test study, that the fingerprint in the Details tab equals the manifest hash.
5. Record the model fingerprint and date in the clinic's change log; reports signed from now on carry the new identity.

Never edit or rename files inside a model folder: the server would load a model whose identity no longer matches what was validated.

---

## 9. Environment variables

Set these for the server process (in a packaged build the desktop shell already sets `SENTINEL_REQUIRE_AUTH=1` and `SENTINEL_DATA_DIR`).

| Variable | Default | Purpose |
|---|---|---|
| `SENTINEL_DATA_DIR` | per-platform user-data dir (§6) | Where the DB, logs, corrections, licence and JWT secret live |
| `SENTINEL_MODELS_DIR` | `<backend>\models` | Read-only model bundle |
| `SENTINEL_ADMIN_PASSWORD` | unset (random one-time password) | Password of the `admin` account created on first start; flagged must-change either way |
| `SENTINEL_REQUIRE_AUTH` | on by default | `1` forces authentication on even in a developer run |
| `SENTINEL_DEV_INSECURE` | unset | `1` = **developer mode only**: relaxes auth, exposes `/docs`, allows online model fetches. Never on a clinic PC. |
| `DEV_BYPASS_LICENSE` | unset | `1` skips the licence check — honoured **only** together with `SENTINEL_DEV_INSECURE=1` |
| `SENTINEL_HOSPITAL_BUILD` | unset | `1` marks a hospital build: offline posture is forced on |
| `SENTINEL_OFFLINE` | `1` unless developer mode | `0` allows model downloads (developer machines only) |
| `SENTINEL_JWT_SECRET` | generated into `jwt_secret.key` | Override the token-signing secret |
| `SENTINEL_SKIP_WARMUP` | unset | `1` skips loading the brain models at start-up (tests); `/health` then shows them as not loaded until first use |
| `SENTINEL_ORTHANC` / `ORTHANC_URL` | off / `http://localhost:8042` | `1` enables the optional Orthanc PACS watcher |
| `SENTINEL_ALLOW_CLOUD_LLM` | unset | `1` permits a cloud LLM backend. Keep unset in a clinic. |
| `SENTINEL_CORS_ORIGINS` / `SENTINEL_CORS_ALLOW_ALL` | localhost + Electron origins / off | Extra browser origins; `ALLOW_ALL=1` is not for production |
| `SENTINEL_TLS_CERT`, `SENTINEL_TLS_KEY` | unset | Serve HTTPS — required if you ever expose the server beyond localhost |
| `SENTINEL_WORKERS` | `1` | Uvicorn worker processes |
| `SENTINEL_MODEL_KEY_HEX` | derived from `model_key.bin` + machine fingerprint | Raw AES key for encrypted model weights (release builds that ship `.enc` weights) |
| `SENTINEL_MODEL_DIR_OVERRIDE_<KEY>` | unset | Point one detector at another folder (vendor A/B tests only) |
| `SENTINEL_DEFAULT_LANG` | `ru` | Report language for the optional folder watcher |
| `OLLAMA_KEEP_ALIVE` (Ollama's own) | — | e.g. `24h` keeps Gemma loaded in RAM so the first report of the day is not slow |

---

## 10. Troubleshooting

Start with **`GET http://127.0.0.1:8000/health`** (open it in a browser on the PC; no login needed). It returns:

```json
{ "status": "ok" | "degraded" | "error", "version": "1.0.0", "device": "cpu",
  "auth_required": true,
  "models": { "brain_triage": {"loaded": true, "reason": "…"},
              "brain_tumor_class": {"loaded": true, "reason": "…"},
              "chest": {"loaded": false, "reason": "not loaded"} },
  "llm": { "backend": "ollama" | "template" | null, "reachable": true },
  "license": { "mode": "licensed" | "unlicensed" | "demo" | "dev" },
  "data_dir": "…", "uptime_seconds": 123.4 }
```

| Symptom | Where to look | Fix |
|---|---|---|
| Login screen says **AI server: offline**; red banner *AI server is not running* | `/health` does not answer. `<data dir>\logs\backend.log` and the newest `sentinel_*.log` | Check that `resources\backend\run_server.py` and its `venv` exist; that port 8000 is not taken by another program; read the last traceback in the log. The desktop shell restarts the server 3 times (2 s / 4 s / 8 s) then shows a dialog with the log path. Restart the app; if it persists, restart the PC; then contact the vendor with the log. |
| Banner *AI server is degraded: the validated model failed to load* | `/health` → `status: degraded`, `models.brain_triage.loaded: false`, read `reason` | Usually **model missing**: the `brain_triage_finetuned` folder is absent, incomplete or moved. Restore it from the install media / vendor package and restart. Until then the panel produces only pending-status output — tell the radiologists not to rely on it. |
| `models.brain_tumor_class.loaded: false` | `reason` names the folders searched | Copy the model bundle (`models\hf\…`) from the install media. Non-fatal: the triage detector still runs. |
| Status-bar **LLM: unavailable**; toast *Template report — AI assistant unavailable* | `/health` → `llm.reachable: false` | Ollama is not running or has no Gemma model: `ollama list`, `ollama serve`, `ollama pull gemma3:4b`. Findings and signing are unaffected. |
| *Preferences → License* shows **Unlicensed** / **Demo mode**; analyses stop with "Demo limit reached (10 analyses/day)" | `/license/status` → `info.reason` | `no license file at …` → copy `license.dat` into the data directory; `invalid signature` → file corrupted or edited, get a fresh copy; `bound to a different machine` → hardware/hostname changed, send the new fingerprint to the vendor; `license expired on …` → renew. Restart after replacing the file. |
| Doctor cannot sign: *Your role cannot sign reports* | `/auth/me` role | Only *radiologist* and *admin* can sign; technicians cannot. Re-create the user with the right role. |
| *A report in this language is already signed* (HTTP 409) | expected behaviour | One signed report per study and language. Not an error. |
| Study comes back **Needs review** | Findings tab shows the reason | Non-brain MRI, or a modality with no installed model. The study must be read without AI. Not a fault. |
| Upload refused: *No readable DICOM file in the upload* | — | The files are not DICOM (screenshots, ZIP, NIfTI). Export the study from the scanner/PACS as DICOM. |
| Everyone logged out at once | `jwt_secret.key` deleted or `SENTINEL_JWT_SECRET` changed | Expected after such a change; users simply log in again. |
| Forgot admin password | — | There is no reset UI. Stop the server, back up `sentinel.db`, and contact the vendor for the reset procedure (a new one-time admin password is generated only when no admin account exists). |
| PDF export fails | Report tab toast *Could not export the PDF* | Check free disk space and that the chosen folder is writable; try the Downloads folder. |

Pytest smoke tests for the server (`tests\`) can be run by the vendor on request.

---

## 11. User management (admin)

There is no user-management screen in this build; use the API from the clinic PC (PowerShell shown, `curl` works the same):

```powershell
# 1. log in as admin → token
$r = Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/login -ContentType application/json `
     -Body '{"username":"admin","password":"<admin password>"}'
$h = @{ Authorization = "Bearer $($r.access_token)" }

# 2. create a radiologist (roles: radiologist | technician | admin)
Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/register -Headers $h -ContentType application/json `
     -Body '{"username":"ivanova","password":"<temporary password>","fullName":"Dr. A. Ivanova","role":"radiologist"}'

# 3. list users / read the audit log
Invoke-RestMethod http://127.0.0.1:8000/auth/users -Headers $h
Invoke-RestMethod "http://127.0.0.1:8000/audit/log?limit=200" -Headers $h
```

Give each doctor a personal account — the signature on a report is the account that signed it. Passwords are stored as bcrypt hashes; the API cannot show them back.

---

## 12. Security notes

- **Authentication is on by default** and cannot be switched off without `SENTINEL_DEV_INSECURE=1`, which must never be set on a clinic PC. Clinical routes (`/analyze/study`, `/studies`, `/study/{id}`, `/report/sign`, `/report/save_correction`) require a bearer token; roles are enforced server-side. `/health` and `/license/status` are public and contain no patient data.
- **No cloud.** The server never contacts the internet: offline posture for models, cloud LLM disabled, Hugging Face telemetry disabled. The only network peers are the desktop app and Ollama on the same machine.
- **Ports are bound to localhost** (`127.0.0.1:8000`, `localhost:11434`). Do not change the host to `0.0.0.0` or add firewall exceptions. If a multi-PC deployment is ever required, use TLS (`SENTINEL_TLS_*`) and a network design agreed with the vendor.
- The API documentation pages (`/docs`, `/redoc`) are disabled in production builds.
- **Encrypt the disk** (BitLocker / FileVault): the SQLite database and logs are not encrypted by the application.
- Signed reports are tamper-evident (SHA-256 of the text, signer, time, model identity, audit entry) but the database file itself can be edited by anyone with disk access — protect the account and the disk.
- Uploaded files never keep their client file names on disk; temporary files are removed after analysis.
- Give every user a personal account, remove accounts when staff leave, and keep the one-time admin password out of shared documents.
