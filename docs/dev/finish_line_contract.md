# Finish-line contract — v1.1.0 "clinic-ready" (2026-11-09)

Goal: close every remaining engineering gap so that (a) the product can be installed, licensed and
supported at a clinic without the developer present, (b) the model cannot be lifted from an installer or
a clinic disk, (c) the next data batch is one command from a new validated release, and (d) the founder
has the documents needed to sign a pilot. Repo: `sentinel/` (python `/Users/shakhzodbtr/venv/bin/python`
3.14 / CI 3.11; desktop Electron+React+TS; tests `pytest` from `sentinel/`, baseline 495 passed / 4 skipped).
Companion contract for the training pipeline: `docs/dev/release_pipeline_contract.md`.

House rules (unchanged): argparse + docstring headers for scripts; `sys.exit(non-zero)` on failure; tests that
need real data/models skip when absent; **no new third-party dependencies**; PHI never in any output,
log, zip or test fixture (synthetic ids ≥ 9000000, dates in 2031); never commit data/ or models/.
`docs/test_docs_consistency.py` enforces: every `SENTINEL_*` env var mentioned in an admin guide exists in
code and every one used in code is documented in all three admin guides; i18n key parity en/ru/uz; routes
quoted in docs exist. Keep it green.

## Decisions (made — implement, do not re-litigate)

D1. **Pending detectors are OFF by default in hospital builds.** The public tumor classifier over-calls
    (fires "glioma" at 0.97–1.00 on report-normal studies). A pilot that measures the validated triage flag
    must not show it by default. Admin can switch it on (Preferences → AI models). Experimental detectors stay off
    as today. When pending detectors are hidden and a study has NO validated detector (e.g. CT head → public
    `head_ct`), the study is accepted into the worklist but returned as `requires_review=true` with
    `rejection_reason='no_validated_detector'` and a localized note; the UI already renders that state
    ("not analysed by AI — radiologist reads as usual").
D2. **Proprietary weights are encrypted at rest automatically on first start of an installed app.** Plaintext
    never lingers on a clinic disk. Gated by `SENTINEL_ENCRYPT_MODELS_AT_REST=1`, which ONLY the Electron shell
    sets, and only when `app.isPackaged` (so CI smoke tests and dev runs never encrypt the build tree). Never
    when `SENTINEL_DEV_INSECURE=1`. Never on macOS inside a `.app` bundle. Verify-before-delete, rollback on
    any failure, loud log line either way.
D3. **Per-clinic model delivery.** The clinic installer may ship WITHOUT the validated model
    (`…-NO-VALIDATED-MODEL`); the model is delivered as a bundle encrypted for that PC, produced by the
    vendor after receiving the machine fingerprint, together with `license.dat`. The key is
    `HKDF(secret ‖ machine_id)` where `secret` travels INSIDE the signed license payload
    (`model_key_secret`), so the vendor can encrypt offline and the clinic never sees plaintext weights.
    Existing installs keep working: key resolution order is `SENTINEL_MODEL_KEY_HEX` > license
    `model_key_secret` (valid license only) > `DATA_DIR/model_key.bin` (today's behaviour).
D4. **Report wording tells the truth about what the model did.** Tumor sentences become classification
    statements ("the classifier assigned class X; features to be confirmed by the radiologist"), the technique
    line is built from the sequences actually present (contrast mentioned only when a post-contrast T1 exists),
    the impression uses human wording in each language, never raw class keys.
D5. **Mixed-body-part CT studies route per series**, not by the first series.
D6. **The pipeline can train two candidates** (`warm` from the current release, `fresh` from the public ViT
    base) and promote the better one that passes the gate.
D7. **CI runs the test suite** before building installers; artifacts expire after 1 day (the code repo is
    public); draft releases stay draft. Version becomes **1.1.0**.
D8. **Spine, the vision-language reporter, regulatory filings** are out of scope (need data / legal).

## API contract (server ↔ desktop ↔ tools)

### `/settings/ai` (admin JWT)
- `GET` → `{"show_pending_detectors": bool, "source": "file"|"env"|"default"}`
- `PUT {"show_pending_detectors": bool}` → same shape; persisted at `DATA_DIR/settings.json`; takes effect
  for the next analysis without restart. Default = env `SENTINEL_SHOW_PENDING_DETECTORS` (`"1"`/`"0"`),
  else `0` when `SENTINEL_HOSPITAL_BUILD=1`, else `1`. `/health` gains `"ai_settings": {...same...}`.
- Analysis: brain panel runs `pending` detectors only when enabled (experimental never on the study route);
  `_analyze_study_sync` for non-brain routes whose card `validation_status != 'validated'` returns the
  `requires_review` / `no_validated_detector` result when pending is hidden. Response `findings[].status`
  unchanged. Audit log records setting changes (who, when, old→new).

### `GET /diagnostics` (admin JWT)
Returns `application/zip` (`Content-Disposition: attachment; filename=sentinel-diagnostics-<YYYYMMDD-HHMM>.zip`)
containing: `health.json`, `license.json` (status only — never the file), `ai_settings.json`,
`versions.json` (app, python, torch, transformers, OS), `models.json` (identity cards from
`get_model_identity` for every registry key), `logs/backend.log` tail (last 2 MB) and the newest
`sentinel_*.log` tail, `audit_tail.jsonl` (last 200 entries, with any `patient_id`/`patient_name`
fields redacted), `settings_redacted.json`. Nothing under `previews/`, no DB, no license.dat, no keys.

### `POST /models/import` (admin JWT) — `{"path": "<absolute path to zip>"}`
Zip layout = `brain_triage_finetuned/{MANIFEST.json, config.json, preprocessor_config.json,
model.safetensors.enc, model.safetensors.enc.meta.json, study_level_eval.json, training_metrics.json}`
(zip made from inside the dir also accepted — reuse `packaging/fetch_model_bundle.extract_bundle`).
Steps: path must be under `DATA_DIR/model_inbox/`; extract to a temp dir; `MANIFEST.json` pins the PLAINTEXT
sha (`model.safetensors`) and the ciphertext sha (`model.safetensors.enc`) — verify both; sidecar
`plaintext_sha256` == pin; **test-decrypt with the current key** (→ 400 `bundle_not_for_this_machine` on
`InvalidTag`); move current `MODELS_DIR/brain_triage_finetuned` → `…_prev_<old identity>`; move the new dir
in; evict the registry cache for `brain_triage` and reload; respond
`{"ok": true, "identity": "<12 hex>", "encrypted_at_rest": true, "status": "validated"}`; on any failure
roll back and respond 4xx/5xx with a reason code. Audit-logged.

### `license.dat` payload
Adds optional `model_key_secret` (64 hex chars) inside the signed JSON. `python -m src.utils.license issue
… --with-model-key` generates it (prints nothing secret; writes it only into the file). `verify_license`
unchanged. `src/utils/model_crypto.get_model_key()` resolution order as in D3; `keyinfo` CLI prints which
source was used.

### `packaging/provision_clinic.py` (vendor side, on the Mac)
```
provision_clinic.py --fingerprint <machine_id> --customer "<clinic>" --expires 2027-12-31
                    [--tier pro] [--features ...] [--model models/brain_triage_finetuned]
                    [--private-key vendor_keys/vendor_private_key.pem] --out <dir>
```
Produces in `<dir>/<customer-slug>_<fp8>/`: `license.dat` (with `model_key_secret`),
`brain_triage_finetuned_<identity12>_<fp8>.zip` (the model dir encrypted with `HKDF(secret‖fingerprint)`,
layout above, built via `model_crypto.export_encrypted_bundle(model_dir, key, out_zip)` on a TEMP copy —
the repo's plaintext model is never touched), `README_admin_ru.txt` + `README_admin_en.txt` (import steps),
and appends a row to `data/licenses/ledger.csv` (customer, fp8, identity, issued, expires, tier — no secret).
Refuses if the model's `study_level_eval.json` fails the release checks (reuse `promote_model.eval_verdict`
hard reasons) or `promotion_log.json` says `forced`.

### Electron shell (`desktop-app/electron/main.js`, `preload.js`)
- env: `SENTINEL_ENCRYPT_MODELS_AT_REST: app.isPackaged ? '1' : '0'`.
- IPC `model-bundle:import` → file picker (*.zip) → copies to `DATA_DIR/model_inbox/<basename>` → returns path.
- IPC `diagnostics:save` (bytes, suggested filename) → save dialog → writes → returns path; opens the folder.

### Desktop UI (`desktop-app/src/**`)
- Preferences → **AI models**: toggle "Show unvalidated (pending) detectors" (admin only; GET/PUT
  `/settings/ai`; explanatory text; warning badge when ON). Below the model list: **Import model bundle**
  (admin) → IPC → `POST /models/import` → toast with the new identity → health refresh.
- Preferences → **About**: **Export diagnostics** → `GET /diagnostics` → IPC save → toast with the path.
- Findings panel: when `rejection_reason === 'no_validated_detector'` show the localized note
  `findings.noValidatedDetector`.
- New i18n keys in en/ru/uz (parity test), API client functions in `services/api.ts`, types in `types/index.ts`.

### CI (`.github/workflows/build-windows.yml`)
- New job `tests` (ubuntu-latest, Python 3.11, CPU torch from the same index, `pip install -r requirements.txt`,
  `pytest -q -m "not slow"`; real-data tests skip). `windows` and `macos` get `needs: tests`.
- `actions/upload-artifact` steps get `retention-days: 1`.
- `workflow_dispatch` input `include_validated_model` (boolean, default `true`); when `false` the "Restore
  validated triage model" step is skipped → artifact named `…-NO-VALIDATED-MODEL` (existing naming).
- `tests/test_build_pipeline.py` structural tests updated accordingly.

### Pipeline additions (`scripts/`)
- `release_pipeline.py --candidates warm,fresh` (default `warm`): trains each candidate
  (`warm` = `--base-model models/brain_triage_finetuned`; `fresh` = `--base-model models/hf/google__vit-base-patch16-224`
  if present else the hub id, `--base-model-fresh`), gates each with `--compare-with`, promotes the best
  passing candidate by balanced accuracy (tie → warm); state records per-candidate results.
- `intake_batch.py --src <drive or folder> --clinic shox --batch <name> --dest <SIAA_DATA root>`: copies
  into `raw/<clinic>/<batch>/{images,reports}` (resumable, size+sha256 manifest `CHECKSUMS.sha256`, verifies
  after copy), writes `README.txt` (received date, source, counts, date range from a quick header scan via
  `index_new_studies.py --limit`), appends `raw/INVENTORY.csv`, prints the exact `release_pipeline.py run`
  command for that batch.
- `backup_mirror.sh <A> <B>`: `rsync -a --checksum --delete-after` mirror with a dry-run first, then
  verification of a random sample of 50 files by sha256, summary + exit code; refuses if B holds a newer
  `INVENTORY.csv` than A.
- `parse_hospital_reports.py`: `.docx` tables (`w:tbl` → cell paragraphs in reading order).

### Docs
- Admin guide ×3: §5.3 licence now mentions `--with-model-key`; new §5.4 "Installing the encrypted model
  bundle" (clinic flavour); §8.3 encryption is automatic on first start (`SENTINEL_ENCRYPT_MODELS_AT_REST`);
  §9 env table adds `SENTINEL_SHOW_PENDING_DETECTORS`, `SENTINEL_ENCRYPT_MODELS_AT_REST`; §10 troubleshooting
  adds "Export diagnostics"; §3 "What gets installed" explains the two installer flavours.
- User manual ×3: pending detectors are hidden by default and how an admin enables them; CT head studies show
  "not analysed by AI" in this release unless enabled.
- Intended use ×3: one sentence that unvalidated detectors are disabled by default.
- Release runbook en/ru: `--candidates`, intake + mirror scripts, code-signing purchase note
  (OV certificate or Azure Trusted Signing; secrets `WIN_CSC_LINK`/`WIN_CSC_KEY_PASSWORD` already wired).
- New `docs/vendor_provisioning_en.md` (+ `_ru`): fingerprint → provision → deliver → import, with the ledger.
- New `docs/sales/`: `one_pager_{ru,uz,en}.md`, `pilot_proposal_{ru,en}.md` (scope, duration, success criteria
  = pilot protocol §7, responsibilities, data handling, support, pricing placeholders, "draft — legal review
  required"), `clinic_onboarding_checklist_{ru,en}.md`; linked from `docs/README.md`.
- Version 1.1.0 everywhere the app version is quoted (`desktop-app/package.json`, `server.APP_VERSION`,
  docs that state "1.0.0").

## File ownership for the implementation round (edit ONLY your files; describe anything else in integration_notes)
- **server**: `src/inference/server.py`, new `tests/test_server_settings.py`, `tests/test_diagnostics.py`,
  `tests/test_model_import.py`, `tests/test_ct_series_routing.py`.
- **report**: `src/inference/brain_report_templates.py`, `src/inference/gemma_report_engine.py`,
  new `tests/test_report_templates.py`.
- **desktop**: everything under `desktop-app/` (src, electron, package.json version), `tests/test_desktop_i18n.py`
  / `tests/test_desktop_electron_i18n.py` only if their expectations must change.
- **vendor-tools**: `src/utils/license.py`, `src/utils/model_crypto.py`, new `packaging/provision_clinic.py`,
  `tests/test_license.py`, `tests/test_model_crypto.py`, new `tests/test_provision_clinic.py`, `.gitignore`
  (add `data/licenses/`, `*.enc`, `model_inbox/`).
- **pipeline**: `scripts/release_pipeline.py`, new `scripts/intake_batch.py`, new `scripts/backup_mirror.sh`,
  `scripts/parse_hospital_reports.py`, `tests/test_release_pipeline.py`, `tests/test_report_parsing.py`,
  new `tests/test_intake_backup.py`, `docs/release_runbook_{en,ru}.md`.
- **ci-docs**: `.github/workflows/build-windows.yml`, `tests/test_build_pipeline.py`, `docs/admin_install_guide_*.md`,
  `docs/user_manual_*.md`, `docs/intended_use_*.md`, new `docs/vendor_provisioning_{en,ru}.md`,
  `docs/test_docs_consistency.py` (only if a new rule is needed), `README.md`.
- **sales**: new `docs/sales/*`, `docs/README.md`.

## Addendum 2026-11-10 — the validated model and the public CI (decisions, made)

The code repository `SHAXZODBR/SIAA` is public; the proprietary model lives only in the private models
repository (`SHAXZODBR/SIAA-models`, one release `brain-triage-<identity12>` per model, published by
`packaging/publish_model_bundle.py`). D7 is amended by:

D9. **A model-bearing installer is never a workflow artifact.** Artifacts of a public repository can be
    downloaded by any signed-in GitHub user while retained; draft-release assets are visible to writers only.
    In both installer jobs "Upload installer artifact" runs only when `steps.bundle.outputs.have_bundle != '1'`
    (clinic flavour / no bundle; artifact `…-NO-VALIDATED-MODEL`, `retention-days: 1`). A model-bearing
    installer (full or UNVALIDATED-DEMO) leaves the runner only through "Attach to draft GitHub Release (tag
    builds)", whose condition now also requires `have_bundle == '1'` and `bundle_validated != '0'`. On a branch
    (Run workflow, `include_validated_model = true`) such a build is a rehearsal: it builds and smoke-tests,
    uploads nothing and says so (`::notice` + job summary).
D10. **A release tag never produces a model-less installer.** On a `v*` tag the restore step (when it runs, i.e.
    not the clinic flavour) calls `fetch_model_bundle.py --require`: a missing bundle fails the build. Branch
    builds keep warn-and-continue.
D11. **No secret is edited per model release.** `fetch_model_bundle.py` resolves the bundle by identity: with a
    models repository configured (`--repo` / `MODEL_BUNDLE_REPO`; the workflow passes
    `vars.MODEL_BUNDLE_REPO || 'SHAXZODBR/SIAA-models'`) it reads release `brain-triage-<identity>` (identity =
    `--identity`, else `FALLBACK_IDENTITY` read as text from `src/inference/validated_metrics.py`) with
    `MODEL_BUNDLE_TOKEN`, takes its `manifest.json` asset and restores it as before; the restored identity must
    equal the requested one. Precedence: explicit `--url` > repository > `MODEL_BUNDLE_URL` (now an optional
    fallback). No token → the "absent" case (exit 0, or 1 with `--require`); a token but a failed lookup (404,
    401/403, no `manifest.json`) → hard failure, never a silent fallback. `TAG_PREFIX` / `MANIFEST_ASSET` are
    shared with the publisher (tested). "Verify bundle identity matches the code" compares the restored identity
    with the same pin the lookup used, so on the lookup path it is NOT independent (it still guards the `--url` /
    `MODEL_BUNDLE_URL` fallback). The independent check on `v*` tags is `--require-newest` (passed next to
    `--require`): after the pinned release is found, one GET of `/repos/<repo>/releases?per_page=100`; a
    published `brain-triage-*` release created after the pinned one fails the build ("the code pins <id> but the
    models repository holds a newer model release <tag>: …") — a tag pushed without the bump commit. Drafts and
    other tags are ignored; an unreadable list fails. Deliberate rebuild of an older model: repository variable
    `MODEL_BUNDLE_ALLOW_OLDER=1` (or `--allow-older`) turns it into a loud warning. The token goes only to the
    bundle URL's host; parts of a bundle found by the lookup must be on `https://api.github.com`. The draft-release
    attach step refuses (fails, uploads nothing) when the tag's release is already published (`isDraft` != true).
    Operator duty per model release: keep `MODEL_BUNDLE_TOKEN` able to read the models repository and push the
    bump commit pinning the identity before the tag.
