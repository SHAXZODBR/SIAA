# SIAA — Sentinel Medical AI · Vendor Provisioning: per-clinic licence and model delivery

Version 1.1.0 · Document revision 2026-11 · **Internal — vendor staff only, never give this document to a clinic**

This is the vendor-side procedure that turns a clinic PC's machine fingerprint into a signed licence plus a copy of the validated triage model that only that PC can decrypt. The clinic-side half of the procedure is §5.3–§5.4 of the admin guide (`admin_install_guide_en.md`); the Russian version of this document is `vendor_provisioning_ru.md`.

---

## 1. Overview

```
clinic PC                          vendor Mac                                    clinic PC
fingerprint (64 hex)  ──────────►  packaging/provision_clinic.py  ──── USB ────►  1. import license.dat, restart
(Preferences → License)            • license.dat (with model_key_secret)          2. Import model bundle (POST /models/import)
                                   • brain_triage_finetuned_<id12>_<fp8>.zip      3. verify identity: toast, /health, PDF
                                   • README_admin_ru.txt / README_admin_en.txt
                                   • row in data/licenses/ledger.csv
```

- The clinic receives the **clinic-flavour installer** (`sentinel-windows-x64-NO-VALIDATED-MODEL`), which contains no proprietary weights and can be copied freely.
- The validated model reaches the clinic only as a bundle encrypted with `HKDF(model_key_secret ‖ machine fingerprint)`. The `model_key_secret` travels **inside the signed licence**, so the vendor encrypts offline and the clinic never sees plaintext weights; the bundle is useless on any other PC and without that licence.
- Licence and bundle are therefore a **matched pair**: every provisioning run produces both, and both are imported together.

---

## 2. Prerequisites on the vendor Mac

| Item | Where | Check |
|---|---|---|
| Vendor signing key | `vendor_keys/vendor_private_key.pem` (never in git, never on a clinic PC; back it up offline) | `ls -l vendor_keys/` — the file exists and is readable only by you |
| Validated model release | `models/brain_triage_finetuned/` with `MANIFEST.json`, `study_level_eval.json`, `promotion_log.json` | the release gate passed and the promotion was **not** forced (the tool refuses otherwise, §4) |
| Python environment | the repository venv (`/Users/shakhzodbtr/venv/bin/python` on the build Mac) | `python -m src.utils.model_crypto keyinfo` runs |
| Clinic-flavour installer | GitHub Actions → **Build installers** → **Run workflow** with `include_validated_model` **unticked** → artifact `sentinel-windows-x64-NO-VALIDATED-MODEL` | download it immediately: artifacts expire after 1 day |
| Ledger folder | `data/licenses/` (git-ignored) | created by the tool on first use |

---

## 3. Step 1 — get the fingerprint from the clinic

1. The clinic IT installs the clinic-flavour installer and opens *Preferences → License* (or the first-run wizard, step *License*). The screen shows the **machine fingerprint**: 64 hexadecimal characters.
2. The clinic sends that string and the clinic's legal name. A photo of the screen is not enough — ask for the text (a single wrong character produces a licence for a different machine).
3. Check it: exactly 64 characters `0-9a-f`. Its first 8 characters (`fp8`) name every file of this delivery.

The fingerprint is computed from host name, OS, CPU model, primary MAC address and OS machine ID. Ask the clinic to finish renaming the PC, joining the domain and fitting network cards **before** sending it.

---

## 4. Step 2 — provision on the vendor Mac

```
python packaging/provision_clinic.py --fingerprint <64-hex fingerprint> --customer "<clinic legal name>" \
       --expires 2027-12-31 [--tier pro] [--features ...] \
       [--model models/brain_triage_finetuned] \
       [--private-key vendor_keys/vendor_private_key.pem] --out <delivery dir>
```

What the tool does:

1. **Refuses** (exit code 1, reason code printed, nothing written) on `bad_fingerprint` (not 64 hex characters), `bad_customer` (empty `--customer`), `bad_expiry` (not `YYYY-MM-DD` or not in the future), `model_missing`, `manifest_missing`, `release_check` (the model's `study_level_eval.json` fails the release rules — the same reasons as `scripts/promote_model.py` and the installer bundle check), `forced_promotion` (`promotion_log.json` records a forced promotion), `private_key` (unreadable, or not the key the application trusts), `output_exists` (a delivery folder is never overwritten) and `output_in_repo` (`--out` resolves to a folder inside the repository: a delivery there would not be git-ignored and the repository is public — use a folder outside it, e.g. `~/sentinel_deliveries/`). A clinic never receives a model that did not pass the gate.
2. Issues `license.dat` with a fresh random `model_key_secret` inside the signed payload — the equivalent of `python -m src.utils.license issue … --with-model-key`. Nothing secret is printed.
3. Copies the model to a **temporary** folder and encrypts the copy with the key derived from that secret and the fingerprint (`model_crypto.export_encrypted_bundle`). The repository's plaintext model is never modified.
4. Writes everything to `<delivery dir>/<customer-slug>_<fp8>/`:

| File | Purpose |
|---|---|
| `license.dat` | signed licence bound to the fingerprint, carrying `model_key_secret` — **confidential** |
| `brain_triage_finetuned_<identity12>_<fp8>.zip` | the encrypted model: `MANIFEST.json`, `config.json`, `preprocessor_config.json`, `model.safetensors.enc`, `model.safetensors.enc.meta.json`, `study_level_eval.json`, `training_metrics.json` |
| `README_admin_ru.txt`, `README_admin_en.txt` | import steps for the clinic IT, with the expected identity |

5. **Self-verifies** before the folder appears: the licence is checked with the application's own verification for this fingerprint, the zip is unpacked the way `/models/import` unpacks it, both `MANIFEST.json` pins are checked and the weights are test-decrypted with the key derived from the signed licence. A failure here prints `FAILED [verify_failed]`, exits with code 2 and leaves nothing behind (a key pair that does not match the application's public key is reported as the refusal `private_key`, exit 1).
6. Appends one row to `data/licenses/ledger.csv` (§8). The summary printed at the end shows the key fingerprint (never the key or the secret).

Before you copy anything, confirm that `<identity12>` in the zip name is the identity of the current release (the 12 characters printed by `scripts/bump_model_release.py --check` and in `docs/README.md`), and that `<fp8>` equals the first 8 characters of the fingerprint the clinic sent.

---

## 5. Step 3 — deliver

1. Copy the whole folder `<customer-slug>_<fp8>/` to a USB stick (preferably an encrypted one). Put the SHA-256 of the zip (the `bundle_sha256` column of the ledger, or `shasum -a 256 <zip>`) in your delivery note; the clinic can compare it with `certutil -hashfile <zip> SHA256`.
2. Hand the stick over in person or by courier. Do **not** send `license.dat` by e-mail or messenger: it carries the model-key secret.
3. Give the clinic the clinic-flavour installer separately if it is not installed yet. The installer contains no secret and no proprietary weights.

---

## 6. Step 4 — clinic-side import

The clinic IT follows admin guide §5.3–§5.4 (also summarised in the README files of the delivery):

1. *Preferences → License → Import license.dat* → restart the application → *Preferences → License* shows *Licensed*, the customer name and the expiry date.
2. Copy the zip to the PC (do not unpack). *Preferences → AI models → Import model bundle* → choose the zip. The app copies it into `<data dir>\model_inbox\` and calls `POST /models/import`.
3. The server verifies the `MANIFEST.json` pins of plaintext and ciphertext, test-decrypts with this PC's key, keeps the previous model as `brain_triage_finetuned_prev_<old identity>`, installs the new one, reloads it without a restart and writes an audit entry. On any failure it rolls back and answers with a reason code; `bundle_not_for_this_machine` means the key did not match (§12).

Order matters: the key is derived from the licence, so the bundle cannot be imported before the matching licence is active.

---

## 7. Step 5 — verify the identity

The installation is complete only when all four agree:

1. The notification after the import shows the identity — equal to `<identity12>` in the zip name.
2. `http://127.0.0.1:8000/health` on the clinic PC: `"status": "ok"`, `models.brain_triage.loaded: true`, `license.mode: "licensed"`, `version: "1.1.0"`.
3. *Preferences → AI models* lists *Study triage — normal vs abnormal (local)* with the same 12-character fingerprint and status *Validated*.
4. A test study analysed and exported to PDF carries the same model identity in the footer.

For remote support ask the clinic for *Preferences → About → Export diagnostics*: its `models.json` contains the identity card of every model and `license.json` the licence status (never the file). Record identity, date and the name of the person who verified it in the ledger notes (§8).

---

## 8. The ledger

`data/licenses/ledger.csv` (git-ignored; back it up with the vendor key, offline) has one row per provisioning run:

| Column | Example | Meaning |
|---|---|---|
| `customer` | Clinic name LLC | customer name as written into the licence |
| `fp8` / `identity` | first 8 characters of the fingerprint / 12-character model identity | which PC, which model release |
| `issued` / `expires` / `tier` / `features` | 2026-11-09 / 2027-12-31 / pro / … | licence dates, tier and feature list |
| `key_fingerprint` | 12 hex characters | fingerprint of the derived model key — `python -m src.utils.model_crypto keyinfo` prints the same value on the clinic PC (source install), so both sides can confirm they agree without revealing the key |
| `bundle_sha256` | 64 hex characters | SHA-256 of the delivered zip, for the delivery note and for checking a copy at the clinic |

The ledger never contains the `model_key_secret`, the key or the full licence. It answers three questions: which clinics run which model release, which licences expire in the next 60 days, and which PCs a new release must be re-provisioned for. Check it monthly.

---

## 9. Renewals

- In the clinic flavour the model key comes from the licence: **when the licence expires, the validated model stops** (`/health` becomes `degraded`) together with the licence. Start renewals **60 days** before `expires` and deliver at least 30 days before.
- A renewal is a normal provisioning run (§4) with the same fingerprint and the new expiry date: it produces a new licence with a new `model_key_secret` and a new bundle encrypted for it. Use a new `--out` folder outside the repository (for example one per date, `~/sentinel_deliveries/2027-11-15/`): the tool refuses with `output_exists` rather than overwrite an earlier delivery. Deliver **both**; the clinic imports the licence, restarts, then imports the bundle (§6).
- Never send a licence alone to a clinic-flavour install: the new secret would make the installed bundle unreadable until the matching bundle is imported.
- After every application update the clinic re-imports the same bundle (admin guide §8.1); no new provisioning is needed for that.

---

## 10. When the PC's hardware changes

A new network card or motherboard, a renamed host, a reinstalled OS or a cloned VM changes the fingerprint. Then:

1. The licence becomes invalid (demo mode) and the model cannot be decrypted (`/health` → `degraded`). The clinic's signed reports and audit log are unaffected.
2. The clinic sends the **new** fingerprint (§3).
3. Provision again with the new fingerprint (§4) and deliver (§5); the clinic imports licence and bundle (§6) and verifies (§7).
4. Mark the old row in the ledger as replaced (note: new `fp8`, date, reason). The old bundle is useless on the new hardware; ask the clinic to delete it from `model_inbox\`.

When a clinic replaces the PC entirely, move the data directory first (admin guide §6.1) and then follow the same steps for the new PC.

---

## 11. Rolling out a new model release

After `scripts/release_pipeline.py` promotes a new validated model (release runbook), every clinic needs a new bundle for its PC:

1. Filter the ledger for active rows (expiry in the future).
2. Re-run §4 for each fingerprint with the new model (licence dates as agreed; a new licence is issued each time).
3. Deliver and verify as in §5–§7; record the new identity per clinic in the ledger.

Clinics keep working on the previous release until they import the new bundle; reports always print the identity of the model that produced them.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `provision_clinic.py` refuses with `release_check` / `forced_promotion` | the model in `--model` did not pass the release gate, or was promoted with force | provision only from a release that passed (`scripts/release_pipeline.py`); never deliver a forced promotion |
| `provision_clinic.py` refuses with `bad_fingerprint`, `bad_expiry`, `private_key`, `output_exists` or `output_in_repo` | typo in the fingerprint, expiry not in the future, wrong / unreadable vendor key, folder from an earlier run, `--out` inside the repository | ask the clinic for the fingerprint as text; fix the date; use `vendor_keys/vendor_private_key.pem`; choose a new `--out` folder outside the repository (e.g. `~/sentinel_deliveries/<date>/`) |
| Clinic import fails with `license_model_key_missing` | the licence of the delivery was not imported, or the app was not restarted after importing it | import `license.dat` from the same folder, restart, import the bundle again |
| Clinic import fails with `bundle_not_for_this_machine` | licence and bundle from different runs, wrong fingerprint, or the licence was not imported / the app not restarted | compare `fp8` with *Preferences → License*; import the licence from the same folder and restart; otherwise re-provision with the fingerprint shown on screen |
| Import fails on a hash / manifest check | zip damaged in transit | compare SHA-256 with the delivery note; copy again |
| `/health` `degraded` weeks after a good install | licence expired or fingerprint changed | §9 or §10 |
| Identity in the notification differs from the zip name | wrong file imported | import the zip from the delivery folder; check `model_inbox\` for stray files |

---

## 13. Security rules

- `vendor_keys/vendor_private_key.pem` signs every licence: keep it on the vendor Mac and on an offline backup only. Whoever holds it can issue licences.
- The delivery folder is **sensitive**: `license.dat` carries the model-key secret AND the fingerprint, so `license.dat` together with the bundle zip decrypts the weights on ANY computer (the application only refuses to *use* them elsewhere). Treat the pair like the plaintext model: ship it on an encrypted USB stick, never e-mail both, and never store both on a shared drive. Keep it only until the clinic has verified the installation (§7), then delete the local copy; the ledger row is the permanent record.
- Never put plaintext weights on a USB stick, in e-mail or in a public repository; the only artefacts that leave the vendor are the clinic-flavour installer and the per-PC delivery folder.
- Never commit `data/licenses/`, `*.enc`, `model_inbox/` or any delivery folder (they are git-ignored).
- The code repository is public: CI artifacts expire after 1 day and tag builds create only draft releases — still download installers right after the build and keep them with the release records.
