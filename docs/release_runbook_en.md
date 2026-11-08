# Release runbook — retraining the brain triage model on a new hospital batch

Founder / ML-engineer copy. Russian version: `release_runbook_ru.md`. Contract every stage is built
against: `dev/release_pipeline_contract.md`. Orchestrator: `scripts/release_pipeline.py`.

Goal: a new batch of brain MRI / CT studies **with radiology reports** arrives on a drive → one
resumable command trains a candidate, gates it against the locked held-out patients, promotes it,
rewrites every pinned number in code and docs, and publishes the model bundle → you tag the app,
CI builds the installer, the hospital receives it. Nothing under `data/` or `models/` is ever
committed; patient names never reach any output file.

## 1. What this runbook covers

| | |
|---|---|
| Input | a drive root with one folder per study (DICOM) + the reports for those studies |
| Output | `models/brain_triage_finetuned/` (new identity), bumped docs/code, `data/release/<tag>/` bundle, a GitHub release of the model, then an installer built by CI |
| Who | one person on the training Mac (Apple Silicon, MPS); ~1 working day of wall-clock for 5,000 studies, mostly unattended |
| Never | `git push`, `git tag` or the GitHub secret are **never** touched by the script — those are your manual steps in section 8 |

## 2. Prerequisites

1. **The drive is mounted** and readable, e.g. `/Volumes/Transcend/new1`. One folder per study; the indexer reads one DICOM header per folder. Check: `ls /Volumes/Transcend/new1 | head`.
2. **Reports in one of the three accepted formats** (all parsed by `scripts/parse_hospital_reports.py`):
   - a folder of per-patient PDFs named `<patientID>_<Name>.pdf` (Shox International Hospital layout; one PDF may contain several studies, each headed by `МРТ головного мозга (YYYY-MM-DD)` / `МСКТ …` and a `ЗАКЛЮЧЕНИЕ:` line), **or**
   - the filled-in **request sheet** `.xlsx`/`.csv` in the layout of `data/new1_report_request_Shox_2026-06-27_to_2026-09-15.xlsx` — Russian headers `ID пациента (PatientID)`, `Дата исследования` (DD.MM.YYYY), `Модальность`, `Область`, `Описание исследования`, `Заключение врача (вставить сюда)`; the hospital pastes the conclusion into the last column, **or**
   - a `.docx` export of the same.
   Copy the reports to a local folder (for example `~/Desktop/reports_<batch>`); never into the repo.
3. **Python venv**: `/Users/shakhzodbtr/venv/bin/python` (3.14, torch 2.11, MPS). Check: `/Users/shakhzodbtr/venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.
4. **Disk**: ≥ 20 GB free on the Mac (slices ≈ 5 PNG per study, two model copies, bundle parts). The pipeline refuses to train below that.
5. **The current release is intact**: `models/brain_triage_finetuned/{model.safetensors,MANIFEST.json,study_level_eval.json}` exist and `data/holdout_patients_v1.txt` (178 locked patient ids) is present.
6. **Clean git tree** before you start (`git status`), so the bump stage's diff is reviewable.
7. **Access for the publish step**: the GitHub token that `git push` uses (keychain) or `SIAA_MODELS_TOKEN` in the environment; write access to the private models repo `SHAXZODBR/SIAA-models` and to the environment secret `MODEL_BUNDLE_URL` of the app repo.

### 2a. When a drive arrives — intake + mirror

Never train straight from the hospital's drive: first archive it (checksummed, inventoried), then
mirror the archive to the second drive, then run the release from the archive copy.

1. **Mount** the hospital drive and the archive drive (`SIAA_DATA`, encrypted). Both scripts refuse a
   destination that does not exist, so an unmounted `/Volumes/…` is never filled on the internal disk.
2. **Intake** — copy into `<SIAA_DATA>/raw/<clinic>/<batch>/{images,reports}`:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/intake_batch.py --src /Volumes/HOSPITAL/new2 \
       --reports /Volumes/HOSPITAL/reports_new2 --clinic shox --batch new2 --dest /Volumes/SIAA_DATA
   ```
   - The source is only read. Every file is copied with its sha256 computed on the fly, renamed into
     place, journaled, then **re-read from the archive drive and verified**. `CHECKSUMS.sha256` (and
     `CHECKSUMS.sizes`) list every file in `shasum` format: `cd <batch dir> && shasum -a 256 -c CHECKSUMS.sha256`.
   - `README.txt` records the received date (`--received YYYY-MM-DD`, default today), the sources, file
     counts, and — from a quick header scan of the copied images (`index_new_studies.py --limit 200`) —
     the modalities and the study-date range (marked "sample" when the batch has more studies).
     `raw/INVENTORY.csv` gets one row per batch, written only after a fully verified intake.
   - Interrupted, or reports arrive later → the same command with `--resume` (copied files are skipped,
     a half-copied `.part` is redone; `--resume --reports <folder>` alone adds the reports). An existing
     batch folder is never overwritten without `--resume`.
   - Exit 3 = a copy differs from the source (listed in `.intake/needs_recopy.txt`; the next `--resume`
     copies it again — check the archive drive). Exit 2 = unreadable source files (bad sectors?);
     everything else is copied. `--verify` re-hashes a whole archived batch at any time.
   - The script prints the exact `release_pipeline.py plan` / `run` commands for the batch
     (`--root` = the folder that holds the study folders, `--reports` = the archived reports).
3. **Mirror** the archive to the backup drive (dry run first, then confirm):

   ```sh
   scripts/backup_mirror.sh /Volumes/SIAA_DATA /Volumes/SIAA_BACKUP            # asks "Type yes"
   scripts/backup_mirror.sh --yes /Volumes/SIAA_DATA /Volumes/SIAA_BACKUP      # unattended
   ```
   `rsync -a --checksum --delete-after` after a dry run that prints how many files would be copied and
   **deleted** on the backup; then 50 random files are hashed on both sides (`--sample N`). It refuses
   when the backup's `raw/INVENTORY.csv` is newer than the archive's (drives swapped, or a batch archived
   on the backup only — mirroring would delete it) and when the archive is empty. Exit 0 ok · 1 refused /
   not confirmed · 2 rsync failed · 3 sample verification failed. `--verify-only` re-checks a sample
   without copying (monthly).
4. Unmount and return the hospital drive only after step 3 printed `B is a verified mirror of A`.

## 3. Time estimates (5,000 studies, Apple Silicon)

| Stage | Expected time | Notes |
|---|---|---|
| index | 10–30 min | one DICOM header per study folder; resumable |
| parse | minutes | ~5,000 PDFs; xlsx/csv is seconds |
| label + benchmark | < 1 min | rules only; the LLM pass (`--label-llm`) is optional and slower |
| match | seconds | needs the index |
| build | ~1 study / s → ~1.5 h | resumable, reuses existing PNGs |
| combine | minutes | symlinks + manifest |
| train | ≈ 45 min per 3,500 slices → **≈ 5 h** for 5,000 studies (12 epochs) | MPS; leave the Mac plugged in, lid open |
| gate | ~5 min | candidate + old model on the same val set |
| promote, bump | seconds | |
| bundle | 5–15 min | zip, split, upload, verify restore |

## 4. The one-command path

```sh
cd ~/Desktop/Siaa_ai/sentinel
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py plan --batch new1 \
    --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1          # read it first
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run  --batch new1 \
    --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
```

- `plan` prints copy-pasteable shell (every command, the guard rails as comments) and runs nothing.
- `run` executes the eleven stages in order, logs each to `data/release/<tag>/logs/<stage>.log`, the
  guard-rail checks to `data/release/<tag>/logs/guards.log`, and keeps `data/release/<tag>/state.json`
  (every effective option, per-stage status / inputs / outputs, guard verdicts).
- **The tag is fixed for the whole release.** The first line printed is `tag: <tag> (…)`. The first `run`
  picks the next unused `brain-triage-v<N>` and writes it to `data/release/LATEST`; every later
  invocation without `--tag` continues LATEST (`tag: <tag> (from LATEST)`) until its `bundle` stage is
  done — even after `promote` has created a new `models/brain_triage_finetuned_prev_*` folder. Pass
  `--tag` only to pick a name yourself or to reopen an older release.
- On the first failure it stops, prints the failing command, the log path and the exact
  `… run … --resume` line, which repeats **every** option of the run (`--epochs`, `--device`,
  `--python`, `--base-model`, `--llm-labels`, …). Fix the cause and paste that line. The short form
  `/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --resume` does the
  same: tag from LATEST, options from `state.json` (printed as `options from state.json: …`).
- `--dry-run` prints the commands without running anything; `--from <stage>` / `--only <stage>` (or
  just `release_pipeline.py <stage> …`) narrow the run; `status` shows the state table.
- Older sets are combined automatically: `data/local_finetune_bin` (batch 1) plus every `data/sets/*`
  except the current batch and rehearsal sets (`data/sets/rehearsal_*`), oldest first; the current
  batch goes **last**, because `combine` gives each patient ONE class and the last set on the command
  line wins a label conflict (conflicts are printed and recorded in `dataset_manifest.json`).
  Override with `--extra-sets`.
- Patients the current model was trained on stay in `train`: the legacy set's `train/` always, and
  from the second retrained release on, the release set named in the current model's
  `training_metrics.json` is passed as `--train-lock data/train_release_<previous tag>` automatically
  (`plan` prints it). `--train-lock <ids.txt | release-set dir>` sets it yourself.

### 4a. Two candidates: `--candidates warm,fresh`

By default (`--candidates warm`) one candidate is fine-tuned from the current release
(`--base-model`, default `models/brain_triage_finetuned`) — exactly the commands above. With
`--candidates warm,fresh` the pipeline also trains a **fresh** candidate from the public ImageNet ViT
(`models/hf/google__vit-base-patch16-224` when that offline copy exists, else `google/vit-base-patch16-224`
from the Hugging Face cache, with `--base-model-fresh`) and keeps the better one:

```sh
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new2 \
    --root /Volumes/SIAA_DATA/raw/shox/new2/images --reports /Volumes/SIAA_DATA/raw/shox/new2/reports \
    --candidates warm,fresh
```

- `train` writes `models/brain_triage_candidate_<tag>_warm` and `…_fresh` (≈ 2 × the train time and disk);
  `gate` evaluates **each** with `--compare-with models/brain_triage_finetuned` on the same val set.
- `promote` moves the best candidate that **passed the gate with `promotion.promote = true`**, by
  balanced accuracy; a tie goes to `warm`. A candidate that fails a floor (exit 2) or is not better than
  the current release (exit 4) is never promoted, whatever its other numbers. When neither qualifies the
  run stops before `promote` with the same `GATE FAILED` / `NOT PROMOTED` message as section 7.
- The pre-train guard checks that the fresh base is on this Mac (**nothing is downloaded during a
  release**; stages run with `HF_HUB_OFFLINE=1`): `fresh base not available offline` stops the run
  before any training and prints the download command for a machine with internet — copy the resulting
  `models/hf/google__vit-base-patch16-224/` into the repo.
- `state.json` → `candidates.<warm|fresh>` (path, base, train / gate exit codes, sens / spec / balanced,
  verdict) and `candidate_choice` (the chosen candidate and the ranking with the reason each one was
  rejected); `plan`, `status` and the printed resume line show them. A `--resume` after one candidate
  finished training retrains only the other one.
- The candidate that was not promoted stays in `models/`; delete it once the release is published.

## 5. Stage by stage — what each prints and what "good" looks like

| # | Stage | Script | Prints | Good looks like |
|---|---|---|---|---|
| 0 | `index` | `scripts/index_new_studies.py --root … --out data/<batch>_index.csv` | studies found, MR/CT split, errors | rows ≈ study folders; `error` column mostly empty; brain MR + head CT counts plausible |
| 1 | `parse` | `scripts/parse_hospital_reports.py --src … --out data/<batch>_reports.csv` | files read, sections found, per-modality counts | one row per study **section**; modality from the section header (MR and CT both present, not all МСКТ); `conclusion` never contains a name |
| 2 | `label` | `scripts/label_reports.py --reports … --out data/<batch>_labels.csv [--import-llm-labels <jsonl>]` then `… --benchmark` | 9-class + abnormal distribution, `needs_review` share; benchmark agreement + confusion vs the 894-patient reference | benchmark agreement on `abnormal` ≥ 0.90 (exit non-zero otherwise); `needs_review` < 15 %; `normal` a sizeable share |
| 3 | `match` | `scripts/match_images_to_reports.py --images … --index … --labels … --out data/<batch>_matched.csv` | matched / unmatched brain studies, `match_quality` histogram | most brain studies matched; quality mostly "same id + same date"; a few "only report for patient" |
| 4 | `build` | `scripts/build_training_set.py … --out data/sets/<batch> --mode binary --abnormal-col abnormal --skip-needs-review` | slices written per class, `studies.csv` | ≈ 5 slices per study; `studies.csv` rows = matched brain studies minus `needs_review` |
| 5 | `combine` | `scripts/combine_training_sets.py --sets data/local_finetune_bin … data/sets/<batch> --holdout data/holdout_patients_v1.txt [--train-lock …] --out data/train_release_<tag>` | per-batch counts, class counts per split (train / val / tune), label conflicts (last set wins), train-locked counts, holdout check | **0** train/val/tune patient overlap; all 178 holdout patients in `val`; one class per patient; `tune/` present; `dataset_manifest.json` written |
| 6 | `train` | `training/finetune_brain_classifier.py … --base-model models/brain_triage_finetuned --epochs 12 --device mps --select-split tune --output models/brain_triage_candidate_<tag>` | per-epoch loss + accuracy; best epoch chosen on `tune/`, never on `val` (the gate set) | loss falls then plateaus; no NaN; `training_metrics.json` written with `checkpoint_selection.split = tune` |
| 7 | `gate` | `scripts/eval_study_level.py --model <candidate> --data data/train_release_<tag>/val --write-manifest --min-sens 0.85 --min-spec 0.40 --compare-with models/brain_triage_finetuned` | study-level sens/spec with 95 % CI for candidate and old model, `promotion` verdict, holdout coverage | gate **passed**, `promotion.promote = true`, `protocol` = production (mean of 5 central slices @0.50, registry predictor), `holdout_covered = true`, `MANIFEST.json` written next to the candidate |
| 8 | `promote` | `scripts/promote_model.py --candidate … --dest models/brain_triage_finetuned` | new identity (12 hex), where the old release went | old release in `models/brain_triage_finetuned_prev_<old_id>/`; registry loads the new dir |
| 9 | `bump` | `scripts/bump_model_release.py --model models/brain_triage_finetuned` | list of rewritten files | every pinned number, identity and date updated; `--check` clean |
| 10 | `bundle` | `packaging/publish_model_bundle.py --model … --out data/release/<tag>/ --github SHAXZODBR/SIAA-models --tag <tag>` | parts, sha256, release URL, **manifest asset API URL**, restore verification | `manifest.json` (`validated: true` + an `eval` summary) + parts < 100 MB; restore into a temp dir verified |

Guard rails the orchestrator enforces **before `train`**: label benchmark passed (or
`--skip-benchmark`), matched brain-study and normal counts printed (warning below 300 normals),
`dataset_manifest.json` present with no train/val/tune patient overlap and every holdout id in `val`,
`tune/` present (the trainer runs with `--select-split tune`; a missing `tune/` stops the release),
≥ 20 GB free, drive mounted. **Before `bundle`**: gate passed and promotion true in the promoted
model, plus every rule `promote_model.py` enforces (release floors, production protocol, release val
set / holdout coverage, eval pinned to these weights), `bump_model_release.py --check` clean, full `pytest -q` and
`docs/test_docs_consistency.py` green (this is the one place the whole suite runs). The full guard
output goes to `data/release/<tag>/logs/guards.log`, the verdict to `state.json` (`guards.pre_train`,
`guards.pre_bundle`).

## 6. Go / no-go rules

| Rule | Threshold | Where enforced |
|---|---|---|
| Study-level sensitivity floor | ≥ 0.85 | `eval_study_level.py --min-sens` (exit 2) |
| Study-level specificity floor | ≥ 0.40 | `eval_study_level.py --min-spec` (exit 2) |
| Promotion vs the current release | new passes the gate **and** balanced accuracy ≥ old − 0.01 **and** sensitivity ≥ old sensitivity − 0.03 | `eval_study_level.py --compare-with` (exit 4), `promote_model.py` |
| Locked held-out patients | all 178 ids of `data/holdout_patients_v1.txt` in `val`; none in `train` | `combine_training_sets.py`, orchestrator pre-train guard |
| Minimum normals in the batch | warning below 300 matched normal studies — specificity will not move without them | orchestrator pre-train guard |
| Label quality | rules agree with the 894-patient reference on `abnormal` ≥ 0.90 | `label_reports.py --benchmark` |
| Manifest | sha256 of the weights pinned in `MANIFEST.json`, verified before every eval | `eval_study_level.py` (exit 3) |
| Release protocol | the eval ran with the release floors, the production gate (`gate mean @0.50`, registry predictor preprocessing), on the release val set (its patient sha256 = `dataset_manifest.json` → `splits.val.patients_sha256`; for a legacy val dir every holdout id scored — `eval_study_level.py --holdout`, default `data/holdout_patients_v1.txt`, `none` to skip) and for these weights | `promote_model.py` (refuses, exit 1), bundle guard, `publish_model_bundle.py` |

`promote_model.py --force` overrides **only** `promotion.promote = false` (the old-vs-new comparison;
logged as `forced` in `promotion_log.json`, and `publish_model_bundle.py` then refuses the model
unless `--allow-unvalidated`). It never overrides a failed gate, lowered floors, a non-production
gate or preprocessing, the wrong val set / uncovered holdout, or an eval of other weights — those
print `not overridable` and `[--force ignored]`. For a legacy val dir `promote_model.py --holdout`
names the locked holdout file (default `data/holdout_patients_v1.txt`).

A sensitivity drop is never traded for a specificity gain: if the candidate is more specific but
misses more abnormal studies than the rule allows, it is **not** promoted.

## 7. When the gate fails

The `gate` stage stops the run **before `promote`** in two distinct ways; `models/brain_triage_finetuned`
is untouched in both, and `state.json` records the verdict (`status` shows it):

| Exit | Printed | Meaning |
|---|---|---|
| 2 | `GATE FAILED` (verdict `gate_failed`) | the candidate is below a floor (sens ≥ 0.85 / spec ≥ 0.40) |
| 4 | `NOT PROMOTED` (verdict `not_promoted`) | the candidate passes the floors but is not better than the current release (promotion rule, section 6) |

1. Read `models/brain_triage_candidate_<tag>/study_level_eval.json` and `data/release/<tag>/logs/gate.log`: which floor failed, how wide the CI is, the `promotion.reason`.
2. **Keep the old model** — always a valid outcome. Nothing has been promoted; the hospital keeps the current release. Delete or keep the candidate folder; `data/release/<tag>/` records what was tried.
3. **Specificity too low** (the usual case): the batch has too few normals. Ask the hospital for more *normal* brain studies with reports (the request sheet in section 2), build the new batch, train again. Do not lower the floor.
4. **Sensitivity too low**: check the labels first — relabel the uncertain reports with the LLM workflow (commands below), then look at `results` in the eval JSON for the `mean` aggregation at other thresholds — a threshold change is a **product** change (`brain_analysis.py`), not a knob to pass the gate.
5. **Retrain from the public ViT base** (fallback when fine-tuning the current release keeps failing): the train stage starts from `--base-model` (default `models/brain_triage_finetuned`). Use the offline copy `models/hf/google__vit-base-patch16-224` if it exists, else the hub id `google/vit-base-patch16-224` (already in `~/.cache/huggingface` on this Mac; the gate-failure message prints the right one). The gate still compares against the current release. Paste the line the gate failure printed, or:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --tag <tag> \
       --base-model google/vit-base-patch16-224 --from train
   ```
   `--from train` (without `--resume`) always retrains; with `--resume` it also retrains, because `train` records its `--base-model` and `--epochs` and a new value counts as a changed input.
   To train both and let the gate pick, use `--candidates warm,fresh --from train` instead (section 4a; the gate-failure message prints that line too).
6. Never fine-tune on a handful of extra studies to squeeze past a floor — the numbers must be comparable across releases (same holdout, same protocol).

**Relabel path (LLM labels), step by step.** Run from `~/Desktop/Siaa_ai/sentinel`; `<tag>` is the tag printed by the run.

1. Export the uncertain rows (`needs_review=yes`; add `--all` for every brain row). This also rewrites the rule labels:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels.csv --export-for-llm data/new1_review.jsonl
   ```
2. In Claude Code, in this repo, run the Workflow (it only runs inside Claude Code's Workflow tool, not with `node`):

   ```
   Workflow({ scriptPath: 'training/relabel_workflow.js', args: { file: 'data/new1_review.jsonl' } })
   ```
3. Save the returned `labels` array as JSONL — one `{"patient_id","study_date","modality","section_index","label","abnormal"}` object per line — to `data/new1_llm_labels.jsonl` (never outside `data/`; it holds no names). The import matches on (patient, date, modality, section), falling back to (patient, date, modality) and (patient, date) only when that key names exactly one row, and prints `N applied, N ambiguous, N skipped`; ambiguous labels are not applied.
4. Merge the labels:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels.csv --import-llm-labels data/new1_llm_labels.jsonl
   ```
5. Continue the release from `match`, recording the JSONL in `state.json`:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --from match \
       --tag <tag> --llm-labels data/new1_llm_labels.jsonl --resume
   ```
   `match → build → combine → train → gate` re-run because their inputs changed (new labels → new
   `studies.csv` → new `dataset_manifest.json`). No `--fresh` is needed: `build` moves the slices of
   a relabelled study to its new class folder and prints `relabelled N`. Re-running the `label` stage **without**
   `--import-llm-labels` would regenerate rule-only labels and drop the LLM labels; with
   `--llm-labels` recorded in `state.json` the orchestrator's `label` stage always passes
   `--import-llm-labels <jsonl>`, also on a later `--from label`.

## 8. Publish: secret, CI, installer

After `bundle` the orchestrator prints this list; the script does none of it.

1. **Secret**: copy the *manifest asset API URL* (`https://api.github.com/repos/SHAXZODBR/SIAA-models/releases/assets/<id>`) into the app repo's environment `MODEL_BUNDLE_URL` → secret `MODEL_BUNDLE_URL`; keep `MODEL_BUNDLE_TOKEN` valid (a token that can read the private models repo).
2. **Tests**: already green (the bundle guard ran them); if you edited anything since, re-run `/Users/shakhzodbtr/venv/bin/python -m pytest -q && /Users/shakhzodbtr/venv/bin/python -m pytest -q docs/test_docs_consistency.py`.
3. **Commit the bump**: `git status`, `git diff` — only code/docs files listed by the bump stage should change; `git add` those files (never `data/`, never `models/`), commit `release: <tag> — bump validated numbers`.
4. **App version — every release**: raise `"version"` in `desktop-app/package.json` for **every** release, including a model-only one (minor bump, e.g. `1.0.0 → 1.1.0`), so that the tag `v<version>` is new — `git tag` refuses an existing tag and pushing an old tag starts no CI run. Commit it.
5. **Tag and push**: `git tag v<version> && git push origin main && git push origin v<version>`. The tag starts `.github/workflows/build-windows.yml` ("Build installers"); it restores the model from `MODEL_BUNDLE_URL` and builds the NSIS installer (and the macOS DMG). Its "Verify bundle identity matches the code" step fails the build when the restored model's identity differs from `src/inference/validated_metrics.FALLBACK_IDENTITY` — so commit the bump and update the secret **before** tagging.
6. **Download the installer** from the Actions run artifacts (or the draft GitHub release the workflow creates for `v*` tags).
7. **Laptop test**: run `windows_laptop_test_en.md` (RU: `windows_laptop_test_ru.md`) on a Windows laptop with Wi-Fi off; every box ticked; the About/PDF footer shows the **new** identity.
8. **Publish the draft release** with release notes: new sensitivity / specificity with CI, number of held-out patients, model identity, batch name.
9. Hand the installer + `admin_install_guide_*.md` to hospital IT; keep `data/release/<tag>/` until the install is confirmed.

### 8a. Code signing — one-time purchase

Without a certificate the installer is **unsigned** and Windows SmartScreen warns "Windows protected
your PC" on every clinic PC. The CI already reads the secrets `WIN_CSC_LINK` (the `.pfx`, base64 or
URL) and `WIN_CSC_KEY_PASSWORD`; when they are set, `electron-builder` signs the NSIS installer — no
code change.

- **OV code-signing certificate** (organisation-validated, from a public CA; the company's legal
  documents are checked, allow 1–3 weeks). Since June 2023 CAs issue code-signing keys only on a
  hardware token or a cloud HSM, so ask the CA **before buying** whether you can get a key usable from
  CI (a `.pfx` for `WIN_CSC_LINK`, or a cloud-signing service) — a USB token can only sign on the
  machine it is plugged into.
- **Azure Trusted Signing** (Microsoft's cloud signing; low monthly fee, SmartScreen reputation builds
  quickly). Check the eligibility rules (country and age of the organisation) before you plan on it.
  It is not a `.pfx`: using it needs a CI change (Azure credentials as secrets + `azureSignOptions` in
  `desktop-app/package.json`) — ask for it as a separate change.
- After the first signed build: right-click the `.exe` → Properties → Digital Signatures shows the
  company name; record the certificate's expiry date in the release checklist.

### 8b. Clinic delivery flavour — installer without the model + per-clinic bundle

For a paying clinic the validated model is not inside the installer: build with the workflow input
`include_validated_model = false` (artifact `sentinel-windows-x64-NO-VALIDATED-MODEL`), then, per clinic
PC, provision a licence + model bundle encrypted for that PC's fingerprint with
`packaging/provision_clinic.py` and deliver both. The full vendor procedure (fingerprint → provision →
deliver → import → verify, the ledger, renewals, a new model release for existing clinics) is
[vendor_provisioning_en.md](vendor_provisioning_en.md) (RU: `vendor_provisioning_ru.md`); the clinic
side is §5.3–§5.4 of `admin_install_guide_en.md`. After every model release, re-provision the bundle
for each clinic in `data/licenses/ledger.csv`.

## 9. Resume, status and troubleshooting

- `/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py status --batch <batch>` — table of stages (pending / running / done / archived / failed) with log paths, guard verdicts and the gate verdict; the tag comes from `data/release/LATEST`.
- `--resume` skips a stage recorded as `done` whose outputs exist and whose inputs are unchanged (sha256 of the input files: `studies.csv` of every combined set — for a legacy set without one, its sorted file listing, hidden files such as Finder's `.DS_Store` ignored — the holdout and train-lock files, `dataset_manifest.json`, the candidate weights, the LLM-label JSONL; for `train` also the `--base-model` and `--epochs` values); otherwise the stage re-runs.
- `promote` **moves** the candidate folder into `models/brain_triage_finetuned`. It records `archived_candidate` in `state.json` and marks `train` and `gate` as `archived` (= done), so a resume after promote runs only what is left (`bump`, `bundle`). A stage whose outputs are gone also counts as done when a later stage is done, unless `--from` / `--only` names it. After promote, `combine` / `train` / `gate` / `promote` never run again under the same tag (the gate would compare the new candidate with itself; combine would rewrite the val set the promoted model's eval is pinned to), and a `--resume` treats every stage up to `promote` as frozen even if an upstream file changed since — only `bump` and `bundle` are left. Changes made after promote go into the next release (new tag).
- Train lock: `combine` locks the patients the current release model was trained on (`data/train_release_<previous tag>`, read from its `training_metrics.json`) to `train`. Keep that folder. If it is gone or was rebuilt, the guard before `combine` stops the run; restore the folder or pass `--train-lock <ids.txt>` with the patient ids that model was trained on.
- Checkpoint selection: `combine` writes a `tune/` split (a slice of the new train patients); the trainer runs with `--select-split tune` and selects its best epoch on `tune/`, never on `val` (the gate set). The pre-train guard stops the release when `tune/` is missing (re-run `combine` with `--tune-frac > 0`).
- `bundle` refuses (exit 1) → either the model is not validated (`publish_model_bundle.py` checks the same gate facts; `--allow-unvalidated` is for demos only and writes `validated: false` into `manifest.json`), or the GitHub release for the tag already holds a bundle of a **different** model identity (`--replace-existing` overwrites it deliberately). Re-running with the same bundle reuses the uploaded assets, so the `MODEL_BUNDLE_URL` asset id does not change; `--reupload` forces a fresh upload (new asset id → re-paste the secret). Without `--tag`, `publish_model_bundle.py` tags `brain-triage-<identity12>`; the orchestrator always passes the release tag.
- Drive unmounted mid-run → `index`/`match`/`build` fail; remount, `--resume`.
- Benchmark below 0.90 → the label rules regressed; fix `label_reports.py`, re-run `label` (`--from label --resume`; imported LLM labels are kept when `--llm-labels` is recorded).
- `combine` fails on overlap → a patient id appears in two sets on different sides; it must be the same numeric id everywhere — check `studies.csv` of both sets.
- `promote` refuses → the app registry could not load the candidate dir; check `config.json` / `preprocessor_config.json` are present and `classes = ['abnormal','normal']`.
- `bump --check` not clean → run the `bump` stage again, then inspect `git diff`.
- Everything intermediate is under `data/release/<tag>/`; to abandon a release, run with a new `--tag` (built slices in `data/sets/` are reused).

## 10. Final release checklist

Evidence paths are relative to `data/release/<tag>/` unless they start with `data/`, `models/` or a URL.

| ☐ | Item | Evidence |
|---|---|---|
| ☐ | Drive archived: intake verified, INVENTORY row written, mirror verified | `<SIAA_DATA>/raw/<clinic>/<batch>/README.txt`, `raw/INVENTORY.csv`, `backup_mirror.sh` summary |
| ☐ | Reports parsed with modality from section headers, no names in any CSV | `data/<batch>_reports.csv`, `logs/parse.log` |
| ☐ | Label benchmark ≥ 0.90 on `abnormal` | `logs/label.log`, `state.json` → `guards.label_benchmark` |
| ☐ | ≥ 300 normal studies matched (or a documented reason to proceed) | `state.json` → `guards.matched_counts` / `guards.pre_train`, `logs/guards.log` |
| ☐ | 0 train/val/tune overlap, 178 holdout ids in val | `data/train_release_<tag>/dataset_manifest.json`, `logs/guards.log` |
| ☐ | Gate passed: sens ≥ 0.85, spec ≥ 0.40, promotion true (with two candidates: the chosen one recorded) | `models/brain_triage_finetuned/study_level_eval.json`, `state.json` → `candidate_choice` |
| ☐ | New identity printed and shown by the app | `MANIFEST.json`, `/health`, PDF footer |
| ☐ | Bump applied, `bump_model_release.py --check` clean | `logs/guard_bump_check.log`, `logs/guards.log` |
| ☐ | `pytest -q` and docs consistency green | `logs/guard_pytest*.log`, `state.json` → `guards.pre_bundle` |
| ☐ | Bundle published, restore verified | `manifest.json`, `logs/bundle.log` |
| ☐ | `MODEL_BUNDLE_URL` secret updated | GitHub → Settings → Environments |
| ☐ | App version raised in `desktop-app/package.json`, bump committed, new `v<version>` tagged and pushed | `git log`, Actions run green |
| ☐ | Installer signed (or the unsigned warning accepted for this pilot) | `.exe` → Properties → Digital Signatures |
| ☐ | Installer downloaded and laptop test passed offline | `windows_laptop_test_*.md` ticked |
| ☐ | Draft release published with numbers + identity | GitHub release page |
| ☐ | Old release kept in `models/brain_triage_finetuned_prev_<id>/` for rollback | `ls models/` |
