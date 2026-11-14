# Release runbook — retraining the brain triage model on a new hospital batch

Founder / ML-engineer copy. Russian version: `release_runbook_ru.md`. Contract every stage is built
against: `dev/release_pipeline_contract.md`. Orchestrator: `scripts/release_pipeline.py`.

Goal: a new batch of brain MRI / CT studies **with radiology reports** arrives on a drive → one
resumable command scores the model shipped today on the new patients, trains a candidate (MR studies
only), gates it against the locked held-out patients, promotes it,
rewrites every pinned number in code and docs, and publishes the model bundle → you tag the app,
CI builds the installer, the hospital receives it. Nothing under `data/` or `models/` is ever
committed; patient names never reach any output file.

## 1. What this runbook covers

| | |
|---|---|
| Input | a drive root with one folder per study (DICOM) + the reports for those studies (one or more deliveries) |
| Output | `models/brain_triage_finetuned/` (new identity), bumped docs/code, `data/release/<tag>/` bundle, a GitHub release of the model, then an installer built by CI |
| Who | one person on the training Mac (Apple Silicon, MPS); ~1 working day of wall-clock for 5,000 studies, mostly unattended |
| Never | `git push`, `git tag` or the GitHub secret are **never** touched by the script — those are your manual steps in section 8 |

## 2. Prerequisites

1. **The drive is mounted** and readable, e.g. `/Volumes/Transcend/new1`. One folder per study; the indexer reads one DICOM header per series (a few more in a folder whose first file is a report object). Only `index` and `build` read the drive; no guard rail checks it. Check: `ls /Volumes/Transcend/new1 | head`.
2. **Reports in one of the three accepted formats** (all parsed by `scripts/parse_hospital_reports.py`):
   - a folder of per-patient PDFs named `<patientID>_<Name>.pdf` (Shox International Hospital layout; one PDF may contain several studies, each headed by `МРТ головного мозга (YYYY-MM-DD)` / `МСКТ …` and a `ЗАКЛЮЧЕНИЕ:` line), **or**
   - the filled-in **request sheet** `.xlsx`/`.csv` in the layout of `data/new1_report_request_Shox_2026-06-27_to_2026-09-15.xlsx` — Russian headers `ID пациента (PatientID)`, `Дата исследования` (DD.MM.YYYY), `Модальность`, `Область`, `Описание исследования`, `Заключение врача (вставить сюда)`; the hospital pastes the conclusion into the last column, **or**
   - a `.docx` export of the same.
   Copy the reports to a local folder (for example `~/Desktop/reports_<batch>`); never into the repo.
   When the reports come in **several deliveries** (for example period A, then period B), keep one folder
   per delivery and never merge them by hand: a patient seen in both periods has a cumulative PDF with
   the **same file name** in each. Pass all of them to `--reports`, oldest first (section 4b).
3. **Python venv**: `/Users/shakhzodbtr/venv/bin/python` (3.14, torch 2.11, MPS). Check: `/Users/shakhzodbtr/venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.
4. **Disk**: ≥ 20 GB free on the Mac (slices ≈ 5 PNG per study, two model copies, bundle parts). The pipeline refuses to train below that.
5. **The current release is intact**: `models/brain_triage_finetuned/{model.safetensors,MANIFEST.json,study_level_eval.json}` exist and `data/holdout_patients_v1.txt` (178 locked patient ids) is present.
6. **Clean git tree** before you start (`git status`), so the bump stage's diff is reviewable.
7. **Access for the publish step**: the GitHub token that `git push` uses (keychain) or `SIAA_MODELS_TOKEN` in the environment; write access to the private models repo `SHAXZODBR/SIAA-models`, and a `MODEL_BUNDLE_TOKEN` secret in the app repo that can read the models repo (no secret is edited per release).

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
   - **One reports folder (or file) per run.** Its files are copied into `reports/` under their own
     names. A later report delivery for the same images goes into a batch of its own, reports only:

     ```sh
     /Users/shakhzodbtr/venv/bin/python scripts/intake_batch.py --reports /Volumes/HOSPITAL/reports_new2_B \
         --clinic shox --batch new2_reports_B --dest /Volumes/SIAA_DATA
     ```
     Do not add it to the first batch with `--resume --reports`: a file whose name is already archived
     is refused when its size differs ("a batch is one delivery") and **skipped as already copied**
     when the size is the same, and new names would be mixed into the same `reports/` folder, where
     the parse stage can no longer tell the deliveries apart. The command the script prints names only
     its own `reports/` folder; for the release pass every archived reports folder to `--reports`,
     oldest delivery first (section 4b).
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
5. **Archive disk smaller than the delivery** (or no archive drive yet): never return the drive with
   nothing kept — the images exist only there, and without them a study can never be re-processed
   or used for another body part's model. Index the drive (stage `index`), then archive the most
   valuable part in priority order onto the largest disk you have:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/archive_studies.py --root /Volumes/HOSPITAL/new2 \
       --index data/new2_index.csv --dest ~/SIAA_DATA/raw/shox/new2 \
       --rule "MR:BRAIN" --rule "CT:HEAD:cap=40" --rule "MR:*" --rule "CT:*:cap=12" --min-free-gb 120
   ```
   Rules are tried in order (`<modality>:<body part pattern>[:cap=N]`; the index row of a study
   decides). Without a cap the whole study folder is copied; with `cap=N` at most N evenly spaced
   `.dcm` files of every series folder (a thin-slice CT series has 600+ files). The files are the
   original DICOM, verified by sha256; `ARCHIVE_STUDIES.csv` lists each study with its mode and
   `ARCHIVE_FILES.sha256` every file. Run the same command again to resume; it stops (exit 2) before
   the disk has less than `--min-free-gb` free, and a later run with a fuller rule copies a capped
   study again in full. A capped study is enough to train slice classifiers but is **not** the full
   scan — only the intake of step 2 keeps a delivery completely. Keep `--dest` outside the
   repository and off cloud-synced folders (it holds patient data).

## 3. Time estimates (5,000 studies, Apple Silicon)

| Stage | Expected time | Notes |
|---|---|---|
| index | 10–30 min | one DICOM header per series; resumable |
| parse | minutes | ~5,000 PDFs; xlsx/csv is seconds; several deliveries are parsed in one run |
| label + benchmark | < 1 min | rules only; the LLM pass (`--label-llm`) is optional and slower; the second read (section 7) is a separate, manual step |
| match | seconds | needs the index, not the drive |
| build | ~1 study / s → ~1.5 h | resumable, reuses existing PNGs that were verified as MR; the first build under the MR-only rule of a set made by an older version reads every study again |
| prospective | minutes (not yet timed on a full batch) | one model pass over the batch's slices, like the gate; no drive |
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
  Its header names the report sources in the order parse will read them
  (`# report sources (oldest delivery first): …`) and the build rule (`# modality rule (build): …`).
- `run` executes the twelve stages in order (index → parse → label → match → build → prospective →
  combine → train → gate → promote → bump → bundle), logs each to `data/release/<tag>/logs/<stage>.log`, the
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
  (`plan` prints it), and to `prospective` as an extra `--seen` (section 4d).
  `--train-lock <ids.txt | release-set dir>` sets it yourself.

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

### 4b. Several report deliveries: `--reports A B …`

When the reports of one batch arrive in more than one delivery, give every delivery to `--reports`,
**oldest first** (one flag with several paths, or the flag repeated — both append):

```sh
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 \
    --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1_A ~/Desktop/reports_new1_B
```

- The parse stage passes them to `parse_hospital_reports.py --src A B` in that order and writes one
  table. Each file is parsed as before; then rows are reconciled **across files**: a section that
  repeats one kept from an earlier file (same patient, date, modality, body part, section header and
  conclusion) is dropped as a repeat, and a new section whose key (patient_id, study_date, modality,
  section_index) is already taken by an earlier file is renumbered to the next free `section_index`.
- Repeats are dropped **by count, not by mere presence**: if the earlier files kept a section once, a
  later file loses at most one copy of it, so a second, identical-looking section of the same visit
  survives. A section with a different header (for example brain MRI and MR angiography on the same day
  with the same conclusion) is never a repeat. A section without a date is never dropped as a repeat
  (it is still renumbered when its key is taken). A request-sheet row and a PDF section of the same
  study have different headers, so both are kept: do not pass a sheet and the PDFs of the same studies
  as two deliveries.
- Why oldest first: labels (rule labels, imported LLM labels) are attached by that key, so it must
  name exactly one section and stay stable. Rows of an earlier delivery are never changed, so adding
  delivery B never moves a label made on delivery A.
- The parse summary prints, with several sources (or when anything was reconciled), the files per
  source (`Sources (oldest first): files per source <A> + <B>`) and
  `N rows dropped as repeats across files · M rows renumbered (section_index taken by an earlier file)`.
  Counts only — never a file name or a conclusion.
- If any one source is missing or holds no report files, parse stops (exit 1) and names it
  (`source 2 of 2`); no CSV is written.
- The order is part of the parse fingerprint: adding a source or swapping the order re-runs `parse`
  on `--resume`, and with it every later stage whose inputs change. A release that ran with one source keeps `parse` and `label`
  done as long as that one source is unchanged.

### 4c. MR studies only: `--modality`

The validated triage model is an MR model, but a matched "brain" study can be a head CT. The `build`
stage therefore runs with `--modality MR` by default (the MR-only rule); `--modality any` turns the rule
off. The value is kept in `state.json`; changing it re-runs `build` and the stages after it (not `match`).
A `build` recorded by an older version of the script, before the rule existed, counts as
`--modality any`: a `--resume` before `promote` re-runs it under the rule (and the stages after it).

- **Row level**, before any DICOM is read: a matched row whose `modality` column names another
  acquisition modality (`CT`) is dropped. A row with an empty or non-image value (`SR`, `PR`, `OT` —
  older indexes say `SR` for many MR studies) is kept and decided at build time.
- **Build time** (the ground truth): the DICOM Modality of the very series the slices come from must
  be `MR`. Otherwise the study gets no slices; it keeps its `studies.csv` row with `n_slices` 0 and is
  counted as excluded, not as failed.
- **Summary**: `modality rule: --modality MR`, then `rows dropped before reading DICOM: modality CT=…`
  and `studies excluded at build time (kept in studies.csv, n_slices 0): modality CT (series)=…,
  modality unknown (series)=…`. If the rule leaves no study with slices, `build` exits 1
  (`no study produced slices: …`) and the release stops.
- **`studies.csv`** has a new last column `modality`: the Modality of the series the slices came from
  (for an excluded study the modality found; empty when not verified). Existing PNGs are reused only
  when the previous `studies.csv` recorded `MR` for that study, so a set built by an older version is
  read again once (as slow as a first build); a study already found to be CT is not read again.
- The rule applies to the set this release builds (`data/sets/<batch>`). The batch-1 legacy set
  `data/local_finetune_bin` — and any other set already in `data/sets/` — is combined **as built**;
  it is not rebuilt or re-checked.

### 4d. Prospective check: the shipped model on patients it never saw

Stage `prospective` answers one question: **how does the model the clinic runs today do on patients
it has never seen?** It runs after `build` and **before** `combine` / `train` use the batch.

- It scores every study of `data/sets/<batch>` (MR only under the rule; `needs_review` rows are already
  left out by `build`) with the release shipped today, `models/brain_triage_finetuned` — always that
  directory, whatever `--base-model` the training starts from; its files are checked against its
  `MANIFEST.json` first — with the production protocol (registry predictor, mean of the central
  slices, threshold 0.50). The release-to-be does not exist yet; nothing is retrained first.
- The sets passed as `--seen` are the sets passed to `combine` (including `data/local_finetune_bin`)
  **plus the release set the shipped model was trained on** (`data/train_release_<previous tag>`, named
  in its `training_metrics.json`, or `--train-lock` when that is a folder) — from the second retrained
  release on; `plan` prints it. A shipped model trained on the legacy set only adds nothing.
- If that release set is gone or was rebuilt, the guard before `prospective` stops the run before
  anything is scored (`guard rails before prospective failed`): without it the `new` group would count
  patients the model was trained on. Restore the folder, or pass `--train-lock <release set dir>` naming
  the set it was trained on (an ids file is not enough here: `--seen` takes only folders).
- Patients are split into groups: `new` (in no `--seen` set), `holdout` (the 178 locked patients, or
  `val/` of the release set or the legacy set), `seen_train` (`train/` or `tune/` of those sets, and
  **every** patient of an earlier per-batch set `data/sets/<batch>` that is not in a holdout source:
  such a set's train/val split is only advisory, `combine` re-splits it, so its patients are seen —
  never `new`), `all`, and `new_same_date` (the `new` studies whose report and images are from the same
  day: the cleanest reference label). A patient of an earlier set counts as seen even if the shipped
  model never trained on that set — the `new` group errs on the safe side.
- The stage runs two commands. First `scripts/report_priority.py` reads the matched reports and writes
  `data/<batch>_priority.csv`: for each study, whether the report carries a **priority finding** — a
  finding that should move the read up the list (mass, recent infarct, haemorrhage, demyelination, true
  hydrocephalus, mass effect, vascular lesion, infection, urgent referral or contrast advised). It reads
  the reports of the release's modality (`--modality`, MR by default); with `--modality any` it reads MR
  reports only, so the priority lines then cover MR studies only. Then
  `scripts/eval_prospective.py` also gets `--index data/<batch>_index.csv` (patient ages) and
  `--priority data/<batch>_priority.csv`.
- Output: `data/release/<tag>/prospective_eval.json` (`kind: prospective_eval`), the full table in
  `logs/prospective.log`, and one line after the stage:
  `prospective, never-seen patients: n N studies · sensitivity x · specificity y · age alone AUC a (model m; model − age [lo, hi]), age-matched p · priority findings flagged k/K, others o/O`.
  The AUCs have three decimals (0.779 against 0.776 must not read as a tie); `[lo, hi]` is the 95 %
  interval of the difference model minus age.
- **Why the age lines.** On 14 November 2026, on 518 never-seen studies, the shipped model was matched
  by patient age alone (AUC 0.779 against 0.776); from the age of 40 almost every study was flagged.
  The stage now prints this for every new batch:
  - `age alone AUC` — how well the patient's age by itself separates abnormal from normal reports;
    `model` — the model's AUC on the same studies. `age-matched` — between two patients of similar age
    (within 5 years), how often the model ranks the abnormal study higher (0.5 = chance).
  - In the JSON (`age_new`): the 95 % interval of the difference model minus age, the age rule that
    reaches the model's sensitivity ("age ≥ T") with its specificity, and the flag rates by age band
    (under 40, 40–59, 60+).
  - `priority findings flagged k/K` — how many studies with a priority finding the shipped threshold
    flags; `others o/O` — how many of all other studies it flags too. In the JSON: `priority_new`,
    with counts per finding (`by_reason`).
- **A model that does not beat the age baseline is not evidence of detection.** If the model's AUC is
  not clearly above the age-alone AUC (the interval of the difference includes 0), or the age-matched
  value is close to 0.5, the model has mostly learned age. Say so in the hand-over notes; do not quote
  its sensitivity as detection.
- **The priority rules are provisional** (`rules-v2`): keyword rules on the report conclusion, not checked
  against a radiologist yet. Read the priority counts as an estimate. The version is in the
  `rules_version` column of `data/<batch>_priority.csv` and in `priority_new.rules_version`; do not
  compare counts made with different versions.
- **The index table is required.** Without `data/<batch>_index.csv` (written by `index`) the guard
  before `prospective` stops the stage before anything runs (`… missing (index stage output …)`):
  restore the file, or re-run the index stage (`--from index`, needs the drive). An index that has no
  age columns is not an error: the age lines are then empty and `notes` says why.
- **Read the headline group `new`** (`groups.new` in the JSON): `n_studies` / `n_patients`, `tp fn tn fp`,
  `sensitivity` with `sensitivity_ci95`, `specificity` with `specificity_ci95`, `balanced_accuracy`,
  `ppv`, `npv`, `flag_rate`, `prevalence`, `auc`. Look at the width of the intervals before the point
  values. `threshold_sweep_new` is descriptive only (the shipped threshold stays 0.50), `findings_new`
  shows detection per finding, `notes` lists what to keep in mind.
- **It never blocks a release.** Exit 0 means it ran, whatever the numbers. Exit 2 means an input was
  unusable (model not loadable, set without PNGs …): an ordinary stage failure — fix it and resume.
- **It never runs again after `promote`** under the same tag (by then `models/brain_triage_finetuned`
  holds the new model, trained on these patients). Keep the JSON: it is the only measurement of the
  old model on them. A release promoted by an older version of the script, before this stage existed,
  has no `prospective` record: `run … --resume` finishes it (`bump`, `bundle`) and `run` / `status`
  show `prospective skipped (promoted before this stage existed)`.
- **A prospective result from before the age lines** (its JSON has no `age_new` / `priority_new`) is not
  reused silently: `run … --resume` runs the stage again and prints
  `stage prospective re-run: recorded before the age baseline and the priority-finding counts existed`;
  `status` marks it too. Only the shipped model is scored again — nothing after it re-runs because of
  this. After `promote` it stays frozen as it is.
- **These numbers — not the gate's — describe real-world performance on new patients.** The gate set is
  reused release after release and the candidate is chosen against it; the prospective model was
  frozen before these reports existed. Honest limits: the reference labels come from the routine
  report (parsed, one radiologist, optionally the LLM second read), not a consensus read; it is one
  clinic and whatever batch it sent, not a pre-registered consecutive cohort; borderline reports
  (`needs_review`) and CT studies are not in it. The new release's own performance on new patients is
  measured by the next batch's `prospective` stage.

## 5. Stage by stage — what each prints and what "good" looks like

| # | Stage | Script | Prints | Good looks like |
|---|---|---|---|---|
| 0 | `index` | `scripts/index_new_studies.py --root … --out data/<batch>_index.csv` | studies found, MR/CT split, errors | rows ≈ study folders; `error` column mostly empty (`no image object` = a folder holding only reports or other non-image objects); brain MR + head CT counts plausible; study fields come from the first **image** header, so `SR` / `PR` should not appear as a study modality |
| 1 | `parse` | `scripts/parse_hospital_reports.py --src <A> [<B> …] --out data/<batch>_reports.csv` | files read, sections found, per-modality counts; with several sources files per source, repeats dropped, rows renumbered | one row per study **section**; modality from the section header (MR and CT both present, not all МСКТ); `conclusion` never contains a name |
| 2 | `label` | `scripts/label_reports.py --reports … --out data/<batch>_labels.csv [--import-llm-labels <jsonl>]` then `… --benchmark` | 9-class + abnormal distribution, `needs_review` share; benchmark agreement + confusion vs the 894-patient reference | benchmark agreement on `abnormal` ≥ 0.90 (exit non-zero otherwise); `needs_review` < 15 %; `normal` a sizeable share |
| 3 | `match` | `scripts/match_images_to_reports.py [--images …] --index … --labels … --out data/<batch>_matched.csv` | matched / unmatched brain studies, `match_quality` histogram | most brain studies matched; quality mostly "same id + same date"; a few "only report for patient" |
| 4 | `build` | `scripts/build_training_set.py … --out data/sets/<batch> --mode binary --abnormal-col abnormal --skip-needs-review --modality MR` | slices written per class, the modality rule (rows dropped, studies excluded), `studies.csv` | ≈ 5 slices per study; `studies.csv` rows = matched brain studies minus `needs_review` minus rows dropped as CT; head CT counted as excluded, never as slices |
| 5 | `prospective` | `scripts/report_priority.py --matched data/<batch>_matched.csv --labels data/<batch>_labels.csv --out data/<batch>_priority.csv [--modality MR]`, then `scripts/eval_prospective.py --model models/brain_triage_finetuned --set data/sets/<batch> --seen <earlier sets> [--seen data/train_release_<previous tag>] --holdout data/holdout_patients_v1.txt --matched data/<batch>_matched.csv --index data/<batch>_index.csv --priority data/<batch>_priority.csv --out data/release/<tag>/prospective_eval.json --device mps` | priority-finding counts; `NEW PATIENTS (never seen by this model)` with sensitivity / specificity and 95 % CI, the age-alone AUC and the priority-finding lines, a table per group, a threshold sweep (descriptive) | `data/<batch>_priority.csv` and `prospective_eval.json` written; `new` group not empty; model AUC compared with the age-alone AUC; numbers read and kept (informational, never a gate — section 4d) |
| 6 | `combine` | `scripts/combine_training_sets.py --sets data/local_finetune_bin … data/sets/<batch> --holdout data/holdout_patients_v1.txt [--train-lock …] --out data/train_release_<tag>` | per-batch counts, class counts per split (train / val / tune), label conflicts (last set wins), train-locked counts, holdout check | **0** train/val/tune patient overlap; all 178 holdout patients in `val`; one class per patient; `tune/` present; `dataset_manifest.json` written |
| 7 | `train` | `training/finetune_brain_classifier.py … --base-model models/brain_triage_finetuned --epochs 12 --device mps --select-split tune --output models/brain_triage_candidate_<tag>` | per-epoch loss + accuracy; best epoch chosen on `tune/`, never on `val` (the gate set) | loss falls then plateaus; no NaN; `training_metrics.json` written with `checkpoint_selection.split = tune` |
| 8 | `gate` | `scripts/eval_study_level.py --model <candidate> --data data/train_release_<tag>/val --write-manifest --min-sens 0.85 --min-spec 0.40 --compare-with models/brain_triage_finetuned` | study-level sens/spec with 95 % CI for candidate and old model, `promotion` verdict, holdout coverage | gate **passed**, `promotion.promote = true`, `protocol` = production (mean of 5 central slices @0.50, registry predictor), `holdout_covered = true`, `MANIFEST.json` written next to the candidate |
| 9 | `promote` | `scripts/promote_model.py --candidate … --dest models/brain_triage_finetuned` | new identity (12 hex), where the old release went | old release in `models/brain_triage_finetuned_prev_<old_id>/`; registry loads the new dir |
| 10 | `bump` | `scripts/bump_model_release.py --model models/brain_triage_finetuned` | list of rewritten files | every pinned number, identity and date updated; `--check` clean |
| 11 | `bundle` | `packaging/publish_model_bundle.py --model … --out data/release/<tag>/ --github SHAXZODBR/SIAA-models` (no `--tag`: the release is tagged `brain-triage-<identity12>`, which the installer CI looks up) | parts, sha256, release URL, **manifest asset API URL**, restore verification | `manifest.json` (`validated: true` + an `eval` summary) + parts < 100 MB; restore into a temp dir verified |

Guard rails the orchestrator enforces **before `train`**: label benchmark passed (or
`--skip-benchmark`), matched brain-study and normal counts printed (warning below 300 normals),
`dataset_manifest.json` present with no train/val/tune patient overlap and every holdout id in `val`,
`tune/` present (the trainer runs with `--select-split tune`; a missing `tune/` stops the release),
≥ 20 GB free (the drive is not checked: `train` reads only the local release set). **Before
`prospective`**: the release set the shipped model was trained on (if any) is present and unchanged, so
it is passed as `--seen` — or `--train-lock` names it as a folder (section 4d); a missing
`data/<batch>_index.csv` stops the stage too (the age baseline needs the patient ages). **Before `combine`**: the
same set is present and unchanged (its patients are locked to `train`), or `--train-lock` is given.
**Before `bundle`**: gate passed and promotion true in the promoted
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
4. **Sensitivity too low**: check the labels first — run the two-round LLM second read of the labels (commands below), then look at `results` in the eval JSON for the `mean` aggregation at other thresholds — a threshold change is a **product** change (`brain_analysis.py`), not a knob to pass the gate.
5. **Retrain from the public ViT base** (fallback when fine-tuning the current release keeps failing): the train stage starts from `--base-model` (default `models/brain_triage_finetuned`). Use the offline copy `models/hf/google__vit-base-patch16-224` if it exists, else the hub id `google/vit-base-patch16-224` (already in `~/.cache/huggingface` on this Mac; the gate-failure message prints the right one). The gate still compares against the current release. Paste the line the gate failure printed, or:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --tag <tag> \
       --base-model google/vit-base-patch16-224 --from train
   ```
   `--from train` (without `--resume`) always retrains; with `--resume` it also retrains, because `train` records its `--base-model` and `--epochs` and a new value counts as a changed input.
   To train both and let the gate pick, use `--candidates warm,fresh --from train` instead (section 4a; the gate-failure message prints that line too).
6. Never fine-tune on a handful of extra studies to squeeze past a floor — the numbers must be comparable across releases (same holdout, same protocol).

**Second read of the labels (LLM, two rounds), step by step.** Run from `~/Desktop/Siaa_ai/sentinel`; `<tag>` is the tag printed by the run.
The rules label every brain report in seconds; the second read has an LLM read the reports
independently and keeps a label only where the readers agree. **Round 1** is one LLM read of every
brain row: a row whose LLM label equals the rule label is decided, every other row is disputed.
**Round 2** is three LLM reads of the disputed rows: the majority decides; a row whose three reads do
not agree on normal-vs-abnormal is a borderline report and keeps `needs_review=yes`, so `build` leaves
it out of training. `scripts/collect_llm_labels.py` runs after each round; the file of the last round is
the one passed with `--llm-labels`.

1. Export every brain row (`--all`). The rule labels go to their **own** file,
   `data/new1_labels_rules.csv`: the collector compares the LLM with the rules and refuses (exit 2) a
   labels file that already holds imported labels:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels_rules.csv --export-for-llm data/new1_review_all.jsonl --all
   wc -l data/new1_review_all.jsonl
   ```
2. **Round 1.** Start each round with an empty parts directory: if an earlier run left
   `data/new1_review_all.jsonl.labels/` (round 2: `data/new1_disputed.jsonl.labels/`), delete it first —
   a run overwrites only the part files it needs, and the collector reads every part file it finds.
   In Claude Code, run the Workflow in direct mode (it only runs inside Claude Code's Workflow tool, not
   with `node`). `scriptPath` and `file` are absolute paths: replace `<repo>` with the absolute path of this
   repository (what `pwd` prints in the `sentinel/` folder that holds `training/` and `data/`). The agents run
   `awk` on `file` and write the part files next to it, so an absolute path works whichever folder the session
   was started in; a relative one does not. `n_lines` is the number `wc -l` printed:

   ```
   Workflow({ scriptPath: '<repo>/training/relabel_workflow.js',
              args: { file: '<repo>/data/new1_review_all.jsonl',
                      n_lines: <wc -l>, votes: 1 } })
   ```
   The agents write their answers to `data/new1_review_all.jsonl.labels/`. Then collect round 1:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/collect_llm_labels.py --export data/new1_review_all.jsonl \
       --rules data/new1_labels_rules.csv --out data/new1_llm_round1.jsonl --disputed data/new1_disputed.jsonl
   ```
   It prints the agreement with the rules and how many rows are disputed, and the exact round-2 call.
   If it reports 0 disputed rows, `data/new1_llm_round1.jsonl` is already the final file: skip step 3.
3. **Round 2.** The same Workflow on the disputed rows with three votes (`n_lines` = the number of
   disputed rows the collector printed), then collect again — without `--disputed`, this is the final round:

   ```
   Workflow({ scriptPath: '<repo>/training/relabel_workflow.js',
              args: { file: '<repo>/data/new1_disputed.jsonl',
                      n_lines: <disputed rows>, votes: 3 } })
   ```

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/collect_llm_labels.py --export data/new1_disputed.jsonl \
       --rules data/new1_labels_rules.csv --prior data/new1_llm_round1.jsonl --out data/new1_llm_labels.jsonl
   ```
   `data/new1_llm_labels.jsonl` holds round 1's decided rows plus round 2's (never outside `data/`; it
   holds no names). Exit 3 = more than 2 % of the votes were rejected (an agent mis-aligned its rows):
   nothing is written to `--out` / `--disputed` (only `--summary`) — run that round's Workflow again.
   The collector prints the rejected count and reasons per part file; when it adds
   `⚠ row_out_of_range votes: part files of an earlier run … may be left in the directory`, re-running
   the Workflow alone cannot fix it: delete the parts directory it names, then run the Workflow and the
   collector again.
   Exit 2 = an input problem (missing file, no answers found, a `--rules` file that already holds
   imported labels); nothing is written.
4. Merge the labels (it writes `data/new1_labels.csv`, the file `match` reads; the rule labels stay in
   `data/new1_labels_rules.csv`):

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels.csv --import-llm-labels data/new1_llm_labels.jsonl
   ```
   The import matches on (patient, date, modality, section), falling back to (patient, date, modality)
   and (patient, date) only when that key names exactly one row (an undated report is matched on the
   same keys with an empty date), and prints `N applied, N ambiguous,
   N skipped` plus `N imported label(s) stay in review` for the borderline rows; ambiguous labels are not
   applied.
5. Continue the release from `match`, recording the JSONL in `state.json`:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --from match \
       --tag <tag> --llm-labels data/new1_llm_labels.jsonl --resume
   ```
   `match → build → prospective → combine → train → gate` re-run because their inputs changed (new
   labels → new `studies.csv` → new `dataset_manifest.json`; `prospective` scores the shipped model again
   against the new reference labels). No `--fresh` is needed: `build` moves the slices of
   a relabelled study to its new class folder and prints `relabelled N`. Re-running the `label` stage **without**
   `--import-llm-labels` would regenerate rule-only labels and drop the LLM labels; with
   `--llm-labels` recorded in `state.json` the orchestrator's `label` stage always passes
   `--import-llm-labels <jsonl>`, also on a later `--from label`.

## 8. Publish: secret, CI, installer

After `bundle` the orchestrator prints this list; the script does none of it.

1. **Secret**: nothing to paste. The installer CI finds the release `brain-triage-<identity12>` in `SHAXZODBR/SIAA-models` (repository variable `MODEL_BUNDLE_REPO`) by the identity the committed bump pins; keep `MODEL_BUNDLE_TOKEN` (app repo → Settings → Environments → `MODEL_BUNDLE_URL`) able to read that repository. The `MODEL_BUNDLE_URL` secret is an optional fallback and needs no update.
2. **Tests**: already green (the bundle guard ran them); if you edited anything since, re-run `/Users/shakhzodbtr/venv/bin/python -m pytest -q && /Users/shakhzodbtr/venv/bin/python -m pytest -q docs/test_docs_consistency.py`.
3. **Commit the bump**: `git status`, `git diff` — only code/docs files listed by the bump stage should change; `git add` those files (never `data/`, never `models/`), commit `release: <tag> — bump validated numbers`.
4. **App version — every release**: raise `"version"` in `desktop-app/package.json` for **every** release, including a model-only one (minor bump, e.g. `1.0.0 → 1.1.0`), so that the tag `v<version>` is new — `git tag` refuses an existing tag and pushing an old tag starts no CI run. Commit it.
5. **Tag and push**: `git tag v<version> && git push origin main && git push origin v<version>`. The tag starts `.github/workflows/build-windows.yml` ("Build installers"); it restores the release the committed bump pins (with `--require`: a missing bundle fails the tag build) and builds the NSIS installer (and the macOS DMG). Its "Verify bundle identity matches the code" step fails the build when the restored model's identity differs from `src/inference/validated_metrics.FALLBACK_IDENTITY` — so commit and push the bump **before** tagging. A tag build also fails (`--require-newest`) when the models repository holds a newer model release than the one the code pins (a `brain-triage-*` release created after it) — usually a tag pushed before the bump commit: push the bump commit and tag that. To rebuild an older model deliberately, set the repository variable `MODEL_BUNDLE_ALLOW_OLDER` to `1` (the build then only warns) and delete the variable afterwards. The installers appear only in the tag's draft release, never as workflow artifacts. Re-running the workflow of a tag whose release was already published (or *Run workflow* on that tag) uploads nothing and fails: build under a new `v<version>` tag.
6. **Download the installer** from the draft GitHub release the workflow creates for `v*` tags (an installer that contains the model is never a workflow artifact of the public repository).
7. **Laptop test**: run `windows_laptop_test_en.md` (RU: `windows_laptop_test_ru.md`) on a Windows laptop with Wi-Fi off; every box ticked; the About/PDF footer shows the **new** identity.
8. **Do not publish the draft release** while it holds the full installer: on the public code repository that would make the model downloadable by anyone. Write the release notes (new sensitivity / specificity with CI, number of held-out patients, model identity, batch name) into the hand-over document instead.
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
- `--resume` skips a stage recorded as `done` whose outputs exist and whose inputs are unchanged (sha256 of the input files: `studies.csv` of every combined set — for a legacy set without one, its sorted file listing, hidden files such as Finder's `.DS_Store` ignored — the holdout and train-lock files, `dataset_manifest.json`, the candidate weights, the LLM-label JSONL; for `parse` also the order of the report sources when there are several, for `build` the `--modality` value, for `prospective` the `MANIFEST.json` of the shipped model and the batch's labels and index tables, for `train` the `--base-model` and `--epochs` values); otherwise the stage re-runs.
- `promote` **moves** the candidate folder into `models/brain_triage_finetuned`. It records `archived_candidate` in `state.json` and marks `train` and `gate` as `archived` (= done), so a resume after promote runs only what is left (`bump`, `bundle`). A stage whose outputs are gone also counts as done when a later stage is done, unless `--from` / `--only` names it. After promote, `prospective` / `combine` / `train` / `gate` / `promote` never run again under the same tag (the gate would compare the new candidate with itself; combine would rewrite the val set the promoted model's eval is pinned to; prospective would score the new model on its own training patients), and a `--resume` treats every stage up to `promote` as frozen even if an upstream file changed since — only `bump` and `bundle` are left. Changes made after promote go into the next release (new tag).
- A release promoted by an older version of the script (before the `prospective` stage existed) is finished the same way: `run --batch <batch> --resume` runs `bump` and `bundle`; `prospective` shows as `skipped (promoted before this stage existed)` in `run` and `status` and is never run under that tag.
- A `build` recorded by an older version, before the MR-only rule, counts as `--modality any`: a `--resume` before promote re-runs `build` under the rule (`stage build re-run: inputs changed`) and with it every later stage.
- Train lock: `combine` locks the patients the current release model was trained on (`data/train_release_<previous tag>`, read from its `training_metrics.json`) to `train`, and `prospective` passes the same set as `--seen`. Keep that folder. If it is gone or was rebuilt, the guard before `prospective` (and before `combine`) stops the run; restore the folder or pass `--train-lock <release set dir>` naming the set that model was trained on. `--train-lock <ids.txt>` with the patient ids satisfies `combine` only — `prospective` still refuses, because its `--seen` takes only folders.
- Checkpoint selection: `combine` writes a `tune/` split (a slice of the new train patients); the trainer runs with `--select-split tune` and selects its best epoch on `tune/`, never on `val` (the gate set). The pre-train guard stops the release when `tune/` is missing (re-run `combine` with `--tune-frac > 0`).
- `bundle` refuses (exit 1) → either the model is not validated (`publish_model_bundle.py` checks the same gate facts; `--allow-unvalidated` is for demos only and writes `validated: false` into `manifest.json`), or the GitHub release for the tag already holds a bundle of a **different** model identity (`--replace-existing` overwrites it deliberately). Re-running with the same bundle reuses the uploaded assets, so the manifest asset id does not change; `--reupload` forces a fresh upload (new asset id; CI is unaffected because it resolves the release by its tag). Without `--tag`, `publish_model_bundle.py` tags `brain-triage-<identity12>`; the orchestrator uses that default, and it is the tag the installer CI looks up.
- Drive not mounted → only `index` and `build` need it (`match` reads the index, `prospective` the built set). A `--resume` without the drive runs what it can (`parse`, `label`, `match`) and stops cleanly before `build` with `drive not mounted: <root> (needed by stage build)`. Remount, paste the printed resume line. Once `build` is done, the rest of the release (`prospective` → `bundle`, including `train`) runs without the drive.
- An index written before the image-header fix: a study whose structured report (or presentation state) sorted first was indexed as modality `SR` with an empty body part, so `match` may not see it as brain. `index` is resumable and heals this by itself: a row whose modality is `SR` / `PR` / empty (and has no error) is indexed again on the next run (`re-indexing N studies whose row was described by a non-image object`). In a study exported flat (report object and images in one folder, report first) the indexer reads further headers of that folder, up to 50, until it finds an image. A study with no image object at all is recorded with error `no image object` (`no image object in the first 50 headers of each folder` when it hit the limit) and is not indexed again on every resume; `match` leaves error rows out, so its report counts as unmatched. With the drive mounted, run `/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --from index --resume`. Before promote only — after promote the change belongs to the next release.
- `build` exits 1 with `no study produced slices: … excluded by --modality MR` → every matched study was CT (or unreadable); check the batch and the `match` output. `--modality any` builds without the rule — only for a deliberate, documented exception.
- `build` exits 1 with `.reconcile_pending exists: an earlier run was interrupted while renaming files — the slices on disk no longer match studies.csv; rebuild with --fresh` → an earlier `build` stopped (crash, full disk, Ctrl-C) between renaming or removing slices and rewriting `studies.csv`; reusing those files could give one study another study's slices. It refuses before reading anything. Run the `build` command that `plan` prints with `--fresh` added (it deletes `data/sets/<batch>` and reads every study again), then paste the resume line.
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
| ☐ | Reports parsed with modality from section headers, every delivery passed oldest first, no names in any CSV | `data/<batch>_reports.csv`, `logs/parse.log` |
| ☐ | Training set MR only (`--modality MR`), head CT counted as excluded | `logs/build.log`, `data/sets/<batch>/studies.csv` → `modality` |
| ☐ | Prospective numbers of the shipped model on never-seen patients read and kept (informational) | `prospective_eval.json` → `groups.new`, `logs/prospective.log` |
| ☐ | Model AUC compared with the age-alone AUC and the priority-finding counts read (a model that does not beat age alone is not evidence of detection) | `prospective_eval.json` → `age_new`, `priority_new`; `data/<batch>_priority.csv` |
| ☐ | Label benchmark ≥ 0.90 on `abnormal` | `logs/label.log`, `state.json` → `guards.label_benchmark` |
| ☐ | ≥ 300 normal studies matched (or a documented reason to proceed) | `state.json` → `guards.matched_counts` / `guards.pre_train`, `logs/guards.log` |
| ☐ | 0 train/val/tune overlap, 178 holdout ids in val | `data/train_release_<tag>/dataset_manifest.json`, `logs/guards.log` |
| ☐ | Gate passed: sens ≥ 0.85, spec ≥ 0.40, promotion true (with two candidates: the chosen one recorded) | `models/brain_triage_finetuned/study_level_eval.json`, `state.json` → `candidate_choice` |
| ☐ | New identity printed and shown by the app | `MANIFEST.json`, `/health`, PDF footer |
| ☐ | Bump applied, `bump_model_release.py --check` clean | `logs/guard_bump_check.log`, `logs/guards.log` |
| ☐ | `pytest -q` and docs consistency green | `logs/guard_pytest*.log`, `state.json` → `guards.pre_bundle` |
| ☐ | Bundle published, restore verified | `manifest.json`, `logs/bundle.log` |
| ☐ | `MODEL_BUNDLE_TOKEN` can read `SHAXZODBR/SIAA-models`; bump pushed before the tag; `MODEL_BUNDLE_ALLOW_OLDER` not set (unless this is a deliberate rebuild of an older model) | GitHub → Settings → Environments; Settings → Secrets and variables → Actions → Variables |
| ☐ | App version raised in `desktop-app/package.json`, bump committed, new `v<version>` tagged and pushed | `git log`, Actions run green |
| ☐ | Installer signed (or the unsigned warning accepted for this pilot) | `.exe` → Properties → Digital Signatures |
| ☐ | Installer downloaded and laptop test passed offline | `windows_laptop_test_*.md` ticked |
| ☐ | Draft release NOT published (it holds the full installer); numbers + identity in the hand-over notes | GitHub release page, hand-over document |
| ☐ | Old release kept in `models/brain_triage_finetuned_prev_<id>/` for rollback | `ls models/` |
