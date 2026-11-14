# SIAA — Sentinel Medical AI · Installation and Administration Guide (Hospital IT)

Version 1.1.0 · Document revision 2026-11

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
| Python backend | `<install>\resources\backend\sentinel-backend\` — the AI server frozen into one program (`sentinel-backend.exe`, with its own Python runtime; a source install has `run_server.py`, `src\` and `venv\` instead) | Spawned by the app on start. Its stdout/stderr go to `<data dir>\logs\backend.log`. |
| Models | `<install>\resources\backend\models\` (`SENTINEL_MODELS_DIR` overrides) | `brain_triage_finetuned\` (validated triage, ~330 MB, with `MANIFEST.json` — inside the installer only in the *full* flavour, see below), `hf\` (public tumor classifier bundle), optional others. |
| Ollama + Gemma 3 | Separate installer (ollama.com); model pulled with `ollama pull gemma3:4b` | Must be installed **before** the PC goes offline. |
| Data directory | See §6 | Created on first start. |

On macOS the equivalent paths are inside `Sentinel Medical AI.app/Contents/Resources/backend`.

**Two installer flavours.** The vendor builds the same application in two variants; the vendor's delivery note says which one you received (the setup file has the same name in both):

| Flavour | Build output | Validated triage model | Used for |
|---|---|---|---|
| **Full** | `sentinel-windows-x64-full` — no longer a downloadable workflow artifact: built only by a `v*` tag and attached to that tag's *draft* GitHub Release (§13.1) | Inside the installer. Encrypted for this PC automatically at first start (§8.3). | Installations the vendor performs in person, demonstrations. |
| **Clinic** | `sentinel-windows-x64-NO-VALIDATED-MODEL` (workflow artifact, kept 1 day) | **Not** in the installer. Delivered separately on USB as a bundle encrypted for *this PC only*, together with the matching `license.dat` (§5.3, §5.4). | Every clinic roll-out: the installer itself carries no proprietary model and can be copied freely. |

Until the bundle is imported, a clinic-flavour install reports `/health` → `"status": "degraded"` and the banner *the validated model failed to load* — this is expected and disappears after §5.4.

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

1. Get the fingerprint. Either read it in *Preferences → License* (the app asks the AI server), or on the PC run `<install>\resources\backend\sentinel-backend\sentinel-backend.exe --print-fingerprint` (installer build) or, in a source install:
   ```
   cd <install>\resources\backend
   venv\Scripts\python.exe -m src.utils.license fingerprint
   ```
2. Send the fingerprint string (64 hex characters) and the clinic name to the vendor.
3. Save the returned `license.dat` into the **data directory** (§6) — via *Preferences → License → Import license.dat*, or by copying the file there manually.
4. Restart the application (which restarts the server). Check `/health` → `license.mode` is `licensed`, or *Preferences → License* shows *Licensed*, customer name and expiry.
5. To verify a file by hand: `venv\Scripts\python.exe -m src.utils.license verify --license-file <path>`.

The licence becomes invalid if the machine fingerprint changes (new network card, renamed host, cloned VM) or after the expiry date — the reason is shown in *Preferences → License* and in the log. Contact the vendor for a re-issue; the software drops to demo mode meanwhile, it does not stop.

**Clinic flavour: the licence also unlocks the model.** For a clinic-flavour install the vendor issues the licence with `python -m src.utils.license issue … --with-model-key` (the vendor's `packaging/provision_clinic.py` does this). The signed licence then carries the secret from which this PC's model key is derived, and the model bundle of §5.4 is encrypted with exactly that key — the licence and the bundle are a **matched pair** from one delivery. Consequences:

- Import the licence **first**, restart the application, then import the bundle (§5.4).
- If the licence becomes invalid (expired, hardware or host name changed), the validated model can no longer be decrypted: `/health` turns `degraded` and the triage detector stops. **Renew at least 30 days before the expiry date**; after a hardware change send the new fingerprint and import the new licence + bundle the vendor returns.
- Never replace the licence of a clinic-flavour install with one from another delivery: a licence with a different key makes the installed bundle unreadable until the matching bundle is imported.
- A full-flavour install normally receives a licence **without** `--with-model-key` (its model is encrypted with the per-install key at first start, §8.3). If it does receive one, the server re-encrypts the model with the licence key at the next start, and from then on the model depends on the licence exactly as described above.

### 5.4 Installing the encrypted model bundle (clinic flavour)

The vendor delivers, per PC, a folder named `<clinic>_<first 8 characters of the fingerprint>` containing `license.dat`, `brain_triage_finetuned_<identity12>_<fp8>.zip` and the instructions `README_admin_en.txt` / `README_admin_ru.txt`. `<identity12>` is the model identity (the first 12 hex characters of the plaintext SHA-256 pinned in `MANIFEST.json`); `<fp8>` must equal the first 8 characters of the fingerprint shown in *Preferences → License*.

Prerequisites: you are signed in as **admin**; the `license.dat` from the same folder is installed (§5.3), the application was restarted and *Preferences → License* shows *Licensed*.

1. Copy the zip from the USB stick to the PC. Do **not** unpack it.
2. *Preferences → AI models → Import model bundle* → choose the zip. The application copies it into `<data dir>\model_inbox\` and asks the AI server to install it (`POST /models/import`, admin only).
3. The server, without a restart: checks that the file is inside `model_inbox\`; unpacks it to a temporary folder; verifies the `MANIFEST.json` pins of the plaintext (`model.safetensors`) and of the ciphertext (`model.safetensors.enc`) and the `plaintext_sha256` in the sidecar; **test-decrypts the weights with this PC's key**; moves the current `brain_triage_finetuned\` aside as `brain_triage_finetuned_prev_<old identity>`; puts the new folder in the models folder (§3) and reloads the detector. Plaintext weights are never written to disk. The step is recorded in the audit log.
4. A notification shows the new **identity**. It must equal `<identity12>` in the file name and the identity in the vendor's README. Then check `/health` → `"status": "ok"` and `models.brain_triage.loaded: true`, *Preferences → AI models* lists the *Study triage* model with the same identity, and after one test study the PDF footer carries it too. Record identity and date in the clinic's change log.

If anything fails, the previous model stays in place (automatic rollback) and the notification names a reason:

| Reason code | Meaning | What to do |
|---|---|---|
| `bundle_not_for_this_machine` | The bundle was encrypted for another PC, or the installed licence is not the one from the same delivery (the key does not match). | Compare `<fp8>` in the file name with *Preferences → License*; import the licence from the same folder and restart; otherwise send the current fingerprint to the vendor for a new bundle. |
| `license_model_key_missing` | No licence with a model key is active on this PC: `license.dat` from the delivery was not imported yet, or the application was not restarted after importing it. | Import `license.dat` from the same folder (§5.3), restart the application, then import the bundle again. |
| `manifest_mismatch`, `sidecar_mismatch`, `decrypted_hash_mismatch`, `bundle_incomplete`, `bundle_invalid` | The file was damaged on the way (USB, partial copy), altered, or is incomplete. | Copy it again from the USB stick; compare `certutil -hashfile <zip> SHA256` with the value in the vendor's delivery note. |
| `path_outside_inbox`, `bundle_not_zip`, `bundle_contains_plaintext` | The file was not passed through *Import model bundle*, or it is not an encrypted Sentinel model bundle. | Use the button in *Preferences → AI models*; import only the zip from the vendor's folder. |
| `models_dir_not_writable`, `model_dir_overridden` | The program's models folder cannot be written (the app was installed for all users under Program Files), or `SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE` is set. | Reinstall for the current user (the installer's default) or start the app once with *Run as administrator* for the import; remove the override variable. |
| HTTP 403 / `import_in_progress` | The signed-in user is not an administrator / another import is still running. | Sign in as admin / wait and try again. |

Keep the delivered folder (USB stick plus a copy with the clinic's backups, §6.1): **after every application update import the same bundle again** (§8.1) — the update replaces the program folder that holds the imported model.

### 5.5 Unvalidated (pending) detectors — off by default

The public *Brain tumor* classifier has **pending** validation and over-calls tumors on local scanners. Since version 1.1.0 hospital builds therefore **do not run pending detectors**; doctors see only the validated *Study triage* result. Experimental detectors are never run in whole-study analysis.

- Switch: *Preferences → AI models → Show unvalidated (pending) detectors* (administrators only). The change applies to the next analysis without a restart, is stored in `<data dir>\settings.json` and is written to the audit log (who, when, old → new). While it is on, the AI-models page shows a warning badge. API: `GET` / `PUT /settings/ai` with an admin token; `/health` → `ai_settings` shows the current value and its source (`file`, `env` or `default`).
- The initial value can be preset with `SENTINEL_SHOW_PENDING_DETECTORS` (§9); a value saved in the application wins over it.
- Switch it on only with the written agreement of the clinic's clinical lead (for example for a research comparison during the pilot), and tell the radiologists.
- **Head CT** in this release has no validated detector: with the switch off, a CT head study is accepted into the worklist but returned as *Needs review* with a note that no validated AI detector exists for it — the radiologist reads it as usual. With the switch on, the public hemorrhage classifier runs and its output is marked *pending validation*.

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
| `training_corpus\` | **Anonymised** thumbnails and labels written by the training-data collector (no names, IDs or birth dates) | Anonymised |
| `dicom_cache\` | Temporary DICOM cache | Yes |
| `license.dat` | The machine-bound licence. In the clinic flavour it also carries the secret of the model key (§5.3) — treat it as confidential, never send it to anyone but the vendor. | No |
| `jwt_secret.key` | Secret that signs login tokens (mode 0600). Deleting it logs everyone out. | No |
| `model_key.bin` | Per-install secret for the model key when the licence carries none (full flavour, §8.3). Created on first use, mode 0600. Losing it makes the encrypted weights unreadable. | No |
| `settings.json` | AI settings saved by an administrator (*Show unvalidated (pending) detectors*, §5.5) | No |
| `model_inbox\` | Model bundles imported through *Preferences → AI models → Import model bundle* (§5.4) | No |

Uploaded DICOM files are written to a temporary folder with server-chosen names during analysis and deleted immediately afterwards; the server keeps header fields and a small preview image, not the study itself.

The training-data collector writes **anonymised** thumbnails and labels under `training_corpus\` inside the same data directory (`TRAINING_CORPUS_DIR` in `src/utils/paths.py`) — no names, IDs or birth dates.

### 6.1 Backup

The database is stored **unencrypted**. Run the PC on an encrypted volume (BitLocker / FileVault) and back up as follows:

1. Close the desktop application (this stops the server) or make sure no analysis is running.
2. Copy the **entire data directory** (§6) to an external drive or NAS — at minimum `sentinel.db`, `sentinel.db-wal`, `sentinel.db-shm`, `doctor_corrections\`, `license.dat`, `jwt_secret.key`, `model_key.bin`, `settings.json`, `model_inbox\`, `logs\`. Without `license.dat` / `model_key.bin` a restored install cannot decrypt its model.
3. Keep three copies (local, external, off-site), encrypt removable media, and **test a restore** monthly on a spare machine: install the same version, copy the folder back, start, log in, open a signed report.

A helper exists (`scripts\backup_restore.py backup --out <dir>`, producing a checksummed `.tar.gz`), but it expects the database under the backend's `data\` folder; verify that it actually captured your data directory before relying on it. The plain folder copy above is always correct.

### 6.2 Uninstalling

The Windows uninstaller asks whether to delete the data folder. Answer **No** unless the clinic has decided to destroy the records; the signed reports and the audit log are the clinic's medical-legal record.

---

## 7. Logs and PHI redaction

- The server log (`logs\sentinel_*.log`) records study IDs (server-generated UUIDs), routes, timings, model load status, login attempts by username and errors. A global filter replaces obvious patient identifiers (`PatientID=…`, `PatientName=…`, birth dates and DICOM `Last^First` person names) with `<redacted>` before any line is written. Client file names are never logged.
- The **audit log** (database table, also `GET /audit/log` for admins, last 200 entries) records who did what and when: `login`, `analyze_study`, `view_study`, `sign_report`, `change_password`, `register`, with the client IP and details such as the report hash.
- The audit log is a **hash chain**: every row stores the hash of the previous row (`prev_hash`) and its own (`row_hash`), so a deleted or edited row breaks the chain. `GET /audit/verify` (admin only) walks the chain end to end and returns `{ok, rows, checked, head_hash, first_broken}`. Run it after a restore from backup and before handing the database to anyone; if `ok` is `false`, record the row id in `first_broken` and report it to the vendor.
- The desktop app itself logs only to its own console (developer tools are open only in development builds).

When sending logs to the vendor, send `sentinel_*.log` / `backend.log` only; never the database. Prefer *Export diagnostics* (§10): a log file copied by hand may still contain the first-run one-time admin password (§5), the export removes it.

---

## 8. Updating the app and the models

### 8.1 Application update

1. Back up the data directory (§6.1).
2. Run the new installer over the old version (Windows) or replace the app bundle (macOS). The data directory is untouched.
3. Start the app once as admin. The database **migrates itself** (new columns are added; nothing is dropped). Check `/health` → `version` and the login-screen version.
4. The desktop app and the server carry the same version number (1.1.0 at this revision); the version is printed on every PDF.
5. **Clinic flavour:** the update replaces the program folder, including the imported model — import the same model bundle again (§5.4) and check that `/health` returns `"status": "ok"`. A full-flavour installer brings its model with it and encrypts it again at first start (§8.3).

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

### 8.3 Encrypted model weights — automatic at first start

The locally trained detector is the vendor's intellectual property and is kept **encrypted at rest** (AES-256-GCM) on every clinic PC. An encrypted model folder contains `model.safetensors.enc` and `model.safetensors.enc.meta.json` instead of `model.safetensors`. The server decrypts the weights straight into memory at load time; plaintext is never written to the install disk.

- **Automatic encryption (full flavour).** The installed Windows application starts the server with `SENTINEL_ENCRYPT_MODELS_AT_REST=1`. At the first start the server encrypts the plaintext weights of every proprietary `*_finetuned` model in place with this install's key, decrypts the result back and compares its SHA-256 with the plaintext **before** deleting the plaintext, and restores the original on any failure. A model already encrypted with the per-install key is re-encrypted the same way when the key now comes from the licence. A log line in `sentinel_*.log` (*Encryption at rest: OK …* or *ENCRYPTION AT REST FAILED …*) states the outcome either way; on failure the server still starts. Nothing is done when the variable is unset (developer runs, CI smoke tests, a source install), when `SENTINEL_DEV_INSECURE=1`, or on macOS inside the signed `.app` bundle (that DMG keeps its model unencrypted — use Windows for clinic installs). The model identity does not change: it is the plaintext SHA-256.
- **Clinic flavour.** The bundle arrives already encrypted for this PC (§5.4); nothing is encrypted at start.
- **The key** is resolved in this order: `SENTINEL_MODEL_KEY_HEX` (raw key, vendor use only) → the `model_key_secret` inside a **valid** licence (clinic flavour, §5.3) → `model_key.bin` in the data directory (created on first use, mode 0600). The last two are combined with the machine fingerprint. `python -m src.utils.model_crypto keyinfo` prints which source was used. If the licence expires or becomes invalid, `model_key.bin` is lost, or the fingerprint changes, the weights can no longer be decrypted — `/health` reports `degraded`; contact the vendor.
- **Hash verification** differs from step 3 in §8.2: `MANIFEST.json` pins the **plaintext** SHA-256 (entry `model.safetensors` — the fingerprint the app shows) and the **ciphertext** SHA-256 (entry `model.safetensors.enc`). Compare `certutil -hashfile model.safetensors.enc SHA256` (macOS: `shasum -a 256 model.safetensors.enc`) with the `.enc` entry, and `plaintext_sha256` inside `model.safetensors.enc.meta.json` with the `model.safetensors` entry.
- Command-line tool (from the backend folder): `python -m src.utils.model_crypto encrypt <model_dir>` (weights → `.enc`, plaintext removed), `python -m src.utils.model_crypto decrypt <model_dir>` (the reverse) and `python -m src.utils.model_crypto keyinfo` (key source and key fingerprint; never the key itself). `encrypt`/`decrypt` are the vendor's release-preparation steps; never run them on the live `models\` tree, only on a copy. On a clinic PC only `keyinfo` is normally needed — when the vendor asks for it.

---

## 9. Environment variables

Set these for the server process (in a packaged build the desktop shell already sets `SENTINEL_REQUIRE_AUTH=1`, `SENTINEL_DATA_DIR`, `SENTINEL_HOSPITAL_BUILD=1` and `SENTINEL_ENCRYPT_MODELS_AT_REST=1`).

| Variable | Default | Purpose |
|---|---|---|
| `SENTINEL_DATA_DIR` | per-platform user-data dir (§6) | Where the DB, logs, corrections, licence and JWT secret live |
| `SENTINEL_MODELS_DIR` | `<backend>\models` | Read-only model bundle |
| `SENTINEL_ADMIN_PASSWORD` | unset (random one-time password) | Password of the `admin` account created on first start; flagged must-change either way |
| `SENTINEL_REQUIRE_AUTH` | on by default | `1` forces authentication on even in a developer run |
| `SENTINEL_DEV_INSECURE` | unset | `1` = **developer mode only**: relaxes auth, exposes `/docs`, allows online model fetches. Never on a clinic PC. |
| `DEV_BYPASS_LICENSE` | unset | `1` skips the licence check — honoured **only** together with `SENTINEL_DEV_INSECURE=1` |
| `SENTINEL_HOSPITAL_BUILD` | unset | `1` marks a hospital build: offline posture becomes the default even in a developer run (only an explicit `SENTINEL_OFFLINE=0` overrides it) |
| `SENTINEL_OFFLINE` | `1` unless developer mode | `0` allows model downloads (developer machines only) |
| `SENTINEL_JWT_SECRET` | generated into `jwt_secret.key` | Override the token-signing secret |
| `SENTINEL_SKIP_WARMUP` | unset | `1` skips loading the brain models at start-up (tests); `/health` then shows them as not loaded until first use |
| `SENTINEL_ORTHANC` / `ORTHANC_URL` | off / `http://localhost:8042` | `1` enables the optional Orthanc PACS watcher |
| `SENTINEL_ALLOW_CLOUD_LLM` | unset | `1` permits a cloud LLM backend. Keep unset in a clinic. |
| `SENTINEL_CORS_ORIGINS` / `SENTINEL_CORS_ALLOW_ALL` | localhost + Electron origins / off | Extra browser origins; `ALLOW_ALL=1` is not for production |
| `SENTINEL_TLS_CERT`, `SENTINEL_TLS_KEY` | unset | Serve HTTPS — required if you ever expose the server beyond localhost |
| `SENTINEL_WORKERS` | `1` | Uvicorn worker processes |
| `SENTINEL_MODEL_KEY_HEX` | unset: key from the licence's `model_key_secret`, else `model_key.bin` (+ machine fingerprint) | Raw AES key for encrypted model weights; overrides both other sources (vendor use only; §8.3) |
| `SENTINEL_ENCRYPT_MODELS_AT_REST` | `1` in the installed Windows/macOS app (set by the desktop shell), `0` or unset otherwise | `1` = encrypt plaintext proprietary weights in place at start-up (verify before delete, rollback on failure; §8.3). Ignored with `SENTINEL_DEV_INSECURE=1` and inside a macOS `.app` bundle. Do not set by hand. |
| `SENTINEL_SHOW_PENDING_DETECTORS` | `0` in hospital builds (`SENTINEL_HOSPITAL_BUILD=1`), else `1` | Initial value of *Show unvalidated (pending) detectors* (§5.5); `1` runs pending detectors such as the public tumor classifier. A value saved in *Preferences → AI models* (`settings.json`) takes precedence. |
| `SENTINEL_MODEL_DIR_OVERRIDE_<KEY>` | unset | Point one detector at another folder (vendor A/B tests only) |
| `SENTINEL_DEFAULT_LANG` | `ru` | Report language for the optional folder watcher |
| `OLLAMA_KEEP_ALIVE` (Ollama's own) | — | e.g. `24h` keeps Gemma loaded in RAM so the first report of the day is not slow |

---

## 10. Troubleshooting

Start with **`GET http://127.0.0.1:8000/health`** (open it in a browser on the PC; no login needed). It returns:

```json
{ "status": "ok" | "degraded" | "error", "version": "1.1.0", "device": "cpu",
  "auth_required": true, "offline": true,
  "ai_settings": { "show_pending_detectors": false, "source": "default" },
  "models": { "brain_triage": {"loaded": true, "reason": "…", "identity": "<12 hex>"},
              "brain_tumor_class": {"loaded": false, "reason": "not loaded"},
              "chest": {"loaded": false, "reason": "not loaded"} },
  "llm": { "backend": "ollama" | "template" | null, "reachable": true },
  "license": { "mode": "licensed" | "unlicensed" | "demo" | "dev" },
  "data_dir": "…", "uptime_seconds": 123.4 }
```

| Symptom | Where to look | Fix |
|---|---|---|
| Login screen says **AI server: offline**; red banner *AI server is not running* | `/health` does not answer. `<data dir>\logs\backend.log` and the newest `sentinel_*.log` | Check that `resources\backend\run_server.py` and its `venv` exist; that port 8000 is not taken by another program; read the last traceback in the log. The desktop shell restarts the server 3 times (2 s / 4 s / 8 s) then shows a dialog with the log path. Restart the app; if it persists, restart the PC; then contact the vendor with the log. |
| Banner *AI server is degraded: the validated model failed to load* | `/health` → `status: degraded`, `models.brain_triage.loaded: false`, read `reason` | **Clinic flavour, new install or after an update:** import the model bundle (§5.4). **Licence expired / hardware changed** (clinic flavour): the model key comes from the licence — renew or re-provision (§5.3). Otherwise usually **model missing**: the `brain_triage_finetuned` folder is absent, incomplete or moved — restore it from the install media / vendor package and restart. Until then no validated AI output is produced — tell the radiologists not to rely on the AI. |
| `models.brain_tumor_class.loaded: false` | `ai_settings.show_pending_detectors`; `reason` names the folders searched | **Expected** while pending detectors are off (§5.5): the model is not loaded at start and loads with the first study after an administrator enables them. If they are on and it still fails, copy the model bundle (`models\hf\…`) from the install media. Non-fatal: the triage detector still runs. `models.brain_triage.identity` is the 12-hex identity of the loaded triage model — compare it with the vendor's delivery README. |
| Status-bar **LLM: unavailable**; toast *Template report — AI assistant unavailable* | `/health` → `llm.reachable: false` | Ollama is not running or has no Gemma model: `ollama list`, `ollama serve`, `ollama pull gemma3:4b`. Findings and signing are unaffected. |
| *Preferences → License* shows **Unlicensed** / **Demo mode**; analyses stop with "Demo limit reached (10 analyses/day)" | `/license/status` → `info.reason` | `no license file at …` → copy `license.dat` into the data directory; `invalid signature` → file corrupted or edited, get a fresh copy; `bound to a different machine` → hardware/hostname changed, send the new fingerprint to the vendor; `license expired on …` → renew. Restart after replacing the file. |
| Doctor cannot sign: *Your role cannot sign reports* | `/auth/me` role | Only *radiologist* and *admin* can sign; technicians cannot. Re-create the user with the right role. |
| *A report in this language is already signed* (HTTP 409) | expected behaviour | One signed report per study and language. Not an error. |
| Study comes back **Needs review** | Findings tab shows the reason | Non-brain MRI, a modality with no installed model, or — head CT in this release — no *validated* detector while pending detectors are switched off (§5.5; the note says so). The study must be read without AI. Not a fault. |
| Model import refused | Notification after *Import model bundle*; audit log | See the reason table in §5.4 (`bundle_not_for_this_machine`: licence and bundle are not from the same delivery for this PC). The previous model is still in place. |
| Upload refused: *No readable DICOM file in the upload* | — | The files are not DICOM (screenshots, ZIP, NIfTI). Export the study from the scanner/PACS as DICOM. |
| Everyone logged out at once | `jwt_secret.key` deleted or `SENTINEL_JWT_SECRET` changed | Expected after such a change; users simply log in again. |
| Forgot admin password | — | There is no reset UI. Stop the server, back up `sentinel.db`, and contact the vendor for the reset procedure (a new one-time admin password is generated only when no admin account exists). |
| PDF export fails | Report tab toast *Could not export the PDF* | Check free disk space and that the chosen folder is writable; try the Downloads folder. |

Pytest smoke tests for the server (`tests\`) can be run by the vendor on request.

**Export diagnostics.** For a support request use *Preferences → About → Export diagnostics* (administrators only; the application calls `GET /diagnostics` and asks where to save `sentinel-diagnostics-<YYYYMMDD-HHMM>.zip`). Send that zip to the vendor instead of picking log files by hand. It contains:

- `health.json` (the `/health` answer), `license.json` (licence status only — never the licence file), `ai_settings.json`;
- `versions.json` (application, Python, PyTorch, transformers, operating system) and `models.json` (identity card of every model: name, SHA-256 fingerprint, validation status);
- the tail of `logs\backend.log` (last 2 MB) and of the newest `sentinel_*.log` — already PHI-redacted (§7), with passwords (including the first-run one-time admin password), tokens, API / licence keys, `Authorization` headers and user:password parts of URLs replaced by `[REDACTED]`;
- `audit_tail.jsonl` (last 200 audit entries, with patient ID / name fields replaced by `<redacted>`) and `settings_redacted.json`.

It does **not** contain images or previews, the database, `license.dat`, `jwt_secret.key`, `model_key.bin` or any model file. Open the zip and glance through it before sending it; it is designed to be free of patient data.

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

# 4. verify the audit-log hash chain (expect ok: true)
Invoke-RestMethod http://127.0.0.1:8000/audit/verify -Headers $h
```

Give each doctor a personal account — the signature on a report is the account that signed it. Passwords are stored as bcrypt hashes; the API cannot show them back.

---

## 12. Security notes

- **Authentication is on by default** and cannot be switched off without `SENTINEL_DEV_INSECURE=1`, which must never be set on a clinic PC. Clinical routes (`/analyze/study`, `/studies`, `/study/{id}`, `/report/sign`, `/report/save_correction`) require a bearer token; roles are enforced server-side. `/health` and `/license/status` are public and contain no patient data.
- **No cloud.** The server never contacts the internet: offline posture for models, cloud LLM disabled, Hugging Face telemetry disabled. The only network peers are the desktop app and Ollama on the same machine.
- **Ports are bound to localhost** (`127.0.0.1:8000`, `localhost:11434`). Do not change the host to `0.0.0.0` or add firewall exceptions. If a multi-PC deployment is ever required, use TLS (`SENTINEL_TLS_*`) and a network design agreed with the vendor.
- The API documentation pages (`/docs`, `/redoc`) are disabled in production builds.
- **Encrypt the disk** (BitLocker / FileVault): the SQLite database and logs are not encrypted by the application. The proprietary model weights are (§8.3); the application uses a clinic-flavour bundle only on the PC it was made for. Note that `license.dat` together with the bundle zip is the key to the weights on any computer — keep the delivery folder on an encrypted stick or in locked backups, never on a shared drive or in e-mail.
- Unvalidated (pending) detectors are off by default (§5.5); every change of that setting is audit-logged. The diagnostics zip (§10) contains no patient data, no keys and no licence file.
- Signed reports are tamper-evident (SHA-256 of the text, signer, time, model identity, audit entry) and the audit log is a hash chain verifiable with `GET /audit/verify` (§7), but the database file itself can be edited by anyone with disk access — protect the account and the disk.
- Uploaded files never keep their client file names on disk; temporary files are removed after analysis.
- Give every user a personal account, remove accounts when staff leave, and keep the one-time admin password out of shared documents.

---

## 13. Building the Windows installer

The installer described in §3 is produced by the GitHub Actions workflow `.github/workflows/build-windows.yml` (repository `SHAXZODBR/SIAA`). First a `tests` job runs the whole test suite on Linux with CPU-only PyTorch (`pytest -q -m "not slow"`, the documentation drift guard and the desktop app's type-check and unit tests); the installer jobs start only when it passes. Each installer job then freezes the Python server with PyInstaller (`packaging/build_backend.ps1`), downloads the public models, restores the validated triage model from the private models repository (the release `brain-triage-<identity12>` whose identity the code pins), smoke-tests the frozen server against `/health` and then runs electron-builder (`npm run build:win`). The same workflow builds the macOS disk image (Apple Silicon) on a macOS runner.

### 13.1 Triggering the build and collecting the installer

1. GitHub → **Actions** → **Build installers** → **Run workflow** (any branch), or push a tag of the form `v1.1.0` — tag builds additionally attach the installers to a *draft* GitHub Release (it stays a draft until a person publishes it). *Run workflow* has one input, `include_validated_model` (ticked by default): untick it to build the **clinic flavour** (§3) — the step *Restore validated triage model* is skipped and the job summary says so. Tag builds always include the model: on a `v*` tag the restore runs with `--require`, so a missing model fails the build instead of producing a model-less installer.
2. Wait for the *Windows installer (NSIS x64)* job to finish (it is slow: Python dependencies with CPU-only PyTorch 2.11.0 / torchvision 0.26.0, PyInstaller, models, installer compression).
3. Where the installer appears depends on whether it contains the validated model — the repository is public, and a workflow artifact can be downloaded by any signed-in GitHub user. **Full flavour:** only from a `v*` tag build, attached to the tag's *draft* GitHub Release (**Releases** → the draft → `Sentinel Medical AI Setup 1.1.0.exe`; a draft is visible only to people with write access to the repository; once published, a release of this public repository can be downloaded by anyone, so do not publish a draft that still holds the full installer). Re-running the workflow of a tag whose release was already published (or *Run workflow* on that tag) uploads nothing and fails: the full installer only ever goes into a *draft* release — build it under a new `v*` tag. It is never uploaded as a workflow artifact: a *Run workflow* on a branch with `include_validated_model` ticked is a rehearsal — it builds and smoke-tests, uploads nothing and says so in the job summary. **Clinic flavour:** on the run page, **Artifacts** → `sentinel-windows-x64-NO-VALIDATED-MODEL` — a zip containing `Sentinel Medical AI Setup 1.1.0.exe` (electron-builder writes it to `desktop-app/release/`). Artifacts are kept for **1 day** only (`retention-days: 1`): download the installer right after the run and store it with the release records.
4. Without the validated model the artifact is named `sentinel-windows-x64-NO-VALIDATED-MODEL`. That is the **clinic flavour** when you built it on purpose (`include_validated_model` unticked): hand it out only together with the per-PC model bundle and licence (§5.4, `docs/vendor_provisioning_en.md`). If you expected the full flavour from a branch build, the `MODEL_BUNDLE_TOKEN` secret is missing or cannot read the models repository (13.2) — such an install reports `/health` → `"status": "degraded"` until a bundle is imported. A `v*` tag build fails in that case instead.

### 13.2 Secrets: the validated model and code signing

| Secret | Purpose | If absent |
|---|---|---|
| `MODEL_BUNDLE_URL` | Optional fallback: URL of a zip whose top level is the folder `brain_triage_finetuned/` (`MANIFEST.json`, `config.json`, `preprocessor_config.json`, `model.safetensors`) or of its multipart `manifest.json` — e.g. a private GitHub Release asset (`https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>`) or a pre-signed object-store link. `packaging/fetch_model_bundle.py` reads it only when no models repository is configured (e.g. on a build PC); the workflow always finds the model by identity, so this secret **no longer needs editing per model release**. Whatever the source, every file is re-hashed against `MANIFEST.json` after download, and the frozen server must return `"status": "ok"` in the smoke test. | Nothing changes in CI. |
| `MODEL_BUNDLE_TOKEN` | **The one secret that matters for the model:** a token that can read the private models repository (fine-grained PAT, *Contents: read*). The repository is named by the repository variable `MODEL_BUNDLE_REPO` (unset: `SHAXZODBR/SIAA-models`). With it the job finds the release `brain-triage-<identity12>` whose identity the checked-out code pins (`src/inference/validated_metrics.py`) and downloads its `manifest.json` — nothing is edited per model release. | Branch build: loud warning in the job log and summary; the installer is built **without** the validated triage model. `v*` tag build: fails. |
| `WIN_CSC_LINK`, `WIN_CSC_KEY_PASSWORD` | Authenticode certificate (`.pfx`: base64 string or URL) and its password; electron-builder signs the installer and the application executable. | Unsigned build: on first run Windows SmartScreen shows *"Windows protected your PC"* (**More info → Run anyway**); antivirus software may quarantine the file. Sign the build before a clinic roll-out. |
| `CSC_LINK`, `CSC_KEY_PASSWORD` | macOS Developer ID certificate (`.p12`) for the DMG job. | Unsigned, un-notarized DMG (Gatekeeper: right-click → Open). |

The public models (`scripts/download_all_models.py --only brain_tumor_class,chest,chest_checkpoint,generic_processors`) are cached between runs; the validated model is fetched afresh on every run and is never stored in the repository. A model release therefore needs no secret change: publish it with `packaging/publish_model_bundle.py`, commit and push the `scripts/bump_model_release.py` changes, then push the `v*` tag. A `v*` tag build also runs the restore with `--require-newest`: it fails when the models repository holds a newer model release (a `brain-triage-*` release created after the one the code pins) — usually a tag pushed before the `scripts/bump_model_release.py` commit; push that commit and tag it. To rebuild an older model deliberately, set the repository variable `MODEL_BUNDLE_ALLOW_OLDER` to `1` (the build then only warns) and delete the variable afterwards. To reproduce the Windows build by hand, run `packaging\build_backend.ps1` and `packaging\smoke_backend.ps1` on a Windows PC with Python 3.11, then `npm run build:win` in `desktop-app\`.
