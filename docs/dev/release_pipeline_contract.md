# Release pipeline — shared contract (2026-09-25)

Goal: when a new hospital batch (~5,000 brain MRI/CT studies **with radiology reports**) arrives on a
drive, a small, resumable set of commands takes it from raw files to a **validated, gated, bundled,
documented model release** and an installer build. Every stage is a repo script with a fixed CLI and
fixed file contract (below). All intermediate files live under the git-ignored `data/` tree
(`data/` is in `.gitignore`); **nothing under data/ or models/ is ever committed**. Patient
NAMES never reach any output file — only the numeric hospital `patient_id`.

Repo: `sentinel/` (python: `/Users/shakhzodbtr/venv/bin/python`, 3.14, torch 2.11 + MPS; CI uses 3.11).
Existing conventions: scripts print a coloured summary, use argparse, and `sys.exit(non-zero)` on failure.
Tests: `pytest` from `sentinel/` (`tests/` + `docs/test_docs_consistency.py`); tests that need
real data (drive, PDFs, models) must **skip cleanly** when it is absent.

## The current validated model (baseline to beat)
`models/brain_triage_finetuned/` — ViT-B binary classifier, classes `['abnormal','normal']`,
identity `0d559766ce58` (first 12 hex of sha256(model.safetensors), pinned in `MANIFEST.json`).
Held-out gate: 178 patients (131 abnormal / 47 normal), study-level mean-prob over 5 central slices
@0.50 → sens 0.901 / spec 0.468 (TP 118 FN 13 TN 22 FP 25), `study_level_eval.json` evaluated 2026-09-07.
Gate floors: sens ≥ 0.85, spec ≥ 0.40 (`scripts/eval_study_level.py --min-sens/--min-spec` defaults).
Production protocol = `src/inference/brain_analysis.py::_triage_mean_probs` (5 central slices,
step = depth//12, mean of class probs). Training set builder `scripts/build_training_set.py::central_slices`
uses the same slice rule. **Locked holdout patients (v1):** `data/holdout_patients_v1.txt`
(178 ids, one per line) — these patients must stay in `val` of every future combined set so the new
numbers are comparable to the old ones; new held-out patients may be ADDED.

## Label policy (fixed — the validated model embodies it)
Study-level, one finding per study, from the Russian ЗАКЛЮЧЕНИЕ:
`normal | tumor | stroke | atrophy | hemorrhage | ms_wml | hydrocephalus | cyst | other`
+ `abnormal ∈ {yes,no}`: **`no` ONLY when the brain itself is normal**. Sinus/nose (гайморит,
синусит, пазух, носовой), scalp/subcutaneous, orbit and bone findings with a normal brain →
`normal`, `abnormal=no`. Atrophy / микроангиопатия / Fazekas → `atrophy`, abnormal=yes.
Negations ("не выявлено", "без патологии", "не изменены", "отсутствуют") must be handled: a
keyword inside a negated clause is NOT a finding. Reference set to benchmark any labeller:
`data/matched_clean.csv` (894 brain patients labelled by an LLM with exactly this rubric; columns
`patient_id,label,abnormal`) joined to `data/hospital_reports_labels.csv` (`patient_id,conclusion`).

## Stage contracts (CLI + files)

### 0. index (exists) — `scripts/index_new_studies.py --root <drive> --out data/<batch>_index.csv`
One row per study folder. Columns used downstream: `study_folder, patient_id, study_date (YYYYMMDD),
modality (MR|CT|…), body_part, study_description, error`. Resumable.

### 1. parse-reports — `scripts/parse_hospital_reports.py --src <folder-or-file> --out data/<batch>_reports.csv`
Accepts: a folder of per-patient PDFs named `<patientID>_<Name>.pdf` (Shox International Hospital
layout — ONE PDF may hold SEVERAL studies, each section headed by a line like
`МСКТ головного мозга (2026-07-21)` / `МРТ головного мозга (2026-08-03)` under a modality line,
then a `ЗАКЛЮЧЕНИЕ:` line, then boilerplate starting at `Заключение не является диагнозом` /
`Результат подготовил`), **or** an `.xlsx`/`.csv` in the layout of the request sheet
`data/new1_report_request_Shox_2026-06-27_to_2026-09-15.xlsx` (Russian headers:
`ID пациента (PatientID)`, `Дата исследования` (DD.MM.YYYY), `Модальность`, `Область`,
`Описание исследования`, `Заключение врача (вставить сюда)`; any sheet whose header row contains
the ID and conclusion columns), **or** `.docx`. Output CSV, one row per STUDY SECTION:
```
patient_id, modality (MR|CT|?), body_part (brain|spine|chest|abdomen|…|other), study_date (YYYY-MM-DD or ''),
conclusion (ЗАКЛЮЧЕНИЕ text only, boilerplate stripped, no name), description (protocol text, optional),
source_file (basename only, name part hashed → '<pid>_<sha16>.pdf'), section_index
```
`modality` comes from the SECTION header, not from anywhere in the PDF (old bug: every report was
МСКТ because the letterhead says "Результаты МРТ / МСКТ"). Backward compatible: columns
`patient_id,modality,body_part,study_date,label,conclusion` keep existing downstream working
(`label` may be filled by stage 2; the parser writes `''`).

### 2. label — `scripts/label_reports.py --reports data/<batch>_reports.csv --out data/<batch>_labels.csv [--benchmark] [--llm ollama:<model>] [--import-llm-labels <jsonl>]`
Adds `label` (9-class), `abnormal` (yes/no), `label_confidence` (high|medium|low), `label_source`
(rules|llm|import), `needs_review` (yes/no) to each brain row. Non-brain rows: `label=''`,
`abnormal=''`. `--benchmark` scores the rules against the 894-patient reference (prints agreement on
`abnormal` and on `label`, confusion) and exits non-zero below `--min-agreement` (default 0.90 on
abnormal). `--import-llm-labels` merges a JSONL of
`{"patient_id","study_date","modality","section_index","label","abnormal"}` (output of
`training/relabel_workflow.js`, the Claude Code Workflow used for batch 1) with `label_source=import`.
Matching: (patient_id, study_date, modality, section_index), then (patient_id, study_date, modality),
then (patient_id, study_date) — each fallback only when it names exactly ONE row; patient_id alone only
when the label has no date and the patient has exactly one brain row. A label is applied to at most
one row; an ambiguous or conflicting label is not applied (the row goes back to `needs_review`). The
CLI prints `N applied, N ambiguous, N skipped`. `--export-for-llm <jsonl>` writes
`{"patient_id","study_date","modality","section_index","conclusion"}` lines for the rows that
`needs_review=yes` (or all with `--all`). Hedged findings ("нельзя исключить …", "(?)") keep their label
with confidence ≤ medium and `needs_review=yes`.

### 3. match (exists, extend) — `scripts/match_images_to_reports.py --images <root> --index data/<batch>_index.csv --labels data/<batch>_labels.csv --out data/<batch>_matched.csv`
Join order: same `patient_id` **and** same date > same patient + compatible modality/body part > only
report for the patient. Carry through: `abnormal, label_source, label_confidence, needs_review,
study_date, match_quality`. Existing columns unchanged (`study_folder, patient_id, modality,
body_part_dicom, is_brain, report_matched, report_body_part, label`).

### 4. build-set (exists, extend) — `scripts/build_training_set.py --images <root> --matched data/<batch>_matched.csv --out data/sets/<batch> --mode binary [--abnormal-col abnormal] [--skip-needs-review]`
Writes `data/sets/<batch>/{train,val}/{abnormal,normal}/<patient_id>_<k>.png` **plus**
`data/sets/<batch>/studies.csv` (`patient_id, study_folder, cls, split, n_slices, label, label_source`).
When `--abnormal-col` is given the binary class comes from that column (yes→abnormal, no→normal),
not from `label`. `--fresh` removes the out dir; without it, existing files are reused (resumable).

### 5. combine — `scripts/combine_training_sets.py --sets data/sets/<a> data/sets/<b> … --holdout data/holdout_patients_v1.txt --out data/train_release_<tag> [--val-frac 0.2] [--seed 42]`
Merges batches into one `train/` + `tune/` + `val/` (symlinks or copies), **patient-level split across
ALL batches** (a patient present in two batches goes to one side), locked holdout ids forced into `val`,
writes `dataset_manifest.json` (per-batch counts, class counts per split, n_patients, seed,
holdout file sha256, sorted patient-id list sha256 per split = `splits.<split>.patients_sha256`,
label conflicts, train-lock counts) and **fails** on any train/val/tune patient overlap or when a
holdout patient is missing from val (warns which). Additions after review:
- ONE class per patient: the LAST set on the command line wins a label conflict (the orchestrator
  passes older sets first and the current batch last); conflicts are printed and recorded. In `val/`
  only the winning batch's slices are kept, so the gate scores each patient once.
  Round 2: a locked HOLDOUT patient is decided by the FIRST (oldest) set holding it — the study the
  holdout was locked on stays in val, a newer follow-up never replaces it. Slices a non-deciding set
  holds under a DIFFERENT class are dropped (possibly another study), never relabelled
  (`conflicting_class_slices_dropped` in the manifest).
- `--min-tune-per-class` (default 10, capped at a third of the class's new train patients) lifts a tiny
  `--tune-frac`; a smaller tune/ is warned about (combine + the pre-train guard).
- Train-locked patients (never val/tune): legacy sets' `train/`, `train/`+`tune/` of a combined release
  set given as input, and `--train-lock <ids.txt | release-set dir>` (the orchestrator passes the release
  set the current model was trained on, read from its `training_metrics.json`; when that set is gone or
  was rebuilt — its val hash ≠ the release eval's — the orchestrator refuses to combine without an
  explicit `--train-lock`, it never drops the lock silently). A holdout id that is also
  train-locked is an error. A `build_training_set.py` set's `split` column is advisory and does not lock.
- `--tune-frac` (default 0.1): `tune/` = a slice of the new, non-locked, non-holdout train patients; the
  trainer selects its checkpoint on it.

### 6. train (exists) — `training/finetune_brain_classifier.py --data data/train_release_<tag> --classes "abnormal,normal" --base-model models/brain_triage_finetuned --balance --unfreeze-last-blocks 3 --epochs 12 --device mps --output models/brain_triage_candidate_<tag>`
The orchestrator also passes `--select-split tune` (checkpoint selection on `tune/`, never on the val gate
set; a missing `tune/` is an error). Saves `model.safetensors, config.json, preprocessor_config.json,
training_metrics.json` (with `checkpoint_selection {split, best_epoch, best_acc, data,
gate_set_used_for_selection}`).

### 7. gate + compare — `scripts/eval_study_level.py --model <candidate> --data data/train_release_<tag>/val --write-manifest --min-sens 0.85 --min-spec 0.40 [--compare-with models/brain_triage_finetuned] [--out …]`
Existing behaviour unchanged (exit 2 gate failed, exit 3 manifest failed). Additions: Wilson 95% CI
for sens/spec in the JSON (`gate.sens_ci`, `gate.spec_ci`), and with `--compare-with` the old model
is evaluated on the SAME val set and `promotion` is written into the JSON:
`{"old": {...gate numbers...}, "new": {...}, "promote": bool, "reason": str}` — promote when the new
model passes the gate AND its balanced accuracy ≥ old − 0.01 AND sens ≥ old sens − 0.03 (a big
sensitivity drop is never accepted for a specificity gain). Exit 4 when `--compare-with` says no.
Scoring goes through the SAME predictor production uses (`model_registry._make_hf_predictor` on the
native-resolution slice = `brain_analysis._triage_mean_probs`). The JSON also records `protocol
{aggregation, threshold, n_central_slices, preprocessing='registry_predictor_native_res_bilinear_224'}`,
`n_patients`, `val_patients_sha256` (same formula as combine) and `holdout / holdout_covered /
holdout_missing` (`--holdout`, default `data/holdout_patients_v1.txt`, `none` to skip).

### 8. promote — `scripts/promote_model.py --candidate models/brain_triage_candidate_<tag> [--dest models/brain_triage_finetuned]`
Requires the candidate's `study_level_eval.json` with `gate.passed` and (if present)
`promotion.promote` true, AND the release rules: floors ≥ 0.85/0.40 (both the recorded `min_*` and the
numbers), gate `gate mean @0.50` + production `protocol.preprocessing`, the release val set
(`val_patients_sha256` = `dataset_manifest.json` `splits.val.patients_sha256` AND the manifest's
`holdout` block shows the locked holdout was applied (n_ids > 0, all present ids in val, same file sha)
AND `holdout_covered` true; or `holdout_covered` for a legacy val dir, `--holdout`), eval pinned to these
weights. An interrupt during the move / registry probe rolls the move back. `--force` overrides ONLY
`promotion.promote=false` (logged `forced` in `promotion_log.json`); every other refusal is not
overridable. Then it moves the current release to `models/brain_triage_finetuned_prev_<old_id>/`,
moves the candidate into place, re-runs the MANIFEST check (`eval_study_level.py` logic), prints the
new identity. Refuses if the app's registry could not load the dir.

### 9. bump-docs — `scripts/bump_model_release.py [--model models/brain_triage_finetuned] [--dry-run] [--check]`
Single source of truth = `models/brain_triage_finetuned/study_level_eval.json` + `MANIFEST.json`.
Rewrites EVERY pinned number/identity/date in code and docs (list below). `--check` exits non-zero
when any file is out of sync (this is what the docs consistency test calls). Runtime strings
(registry card notes, brain_analysis triage label, brain_report_templates RU/UZ/EN
"чувствительность 90%, специфичность 47%") must be generated from ONE module
`src/inference/validated_metrics.py` (reads the eval JSON next to the model dir at import time,
falls back to the constants of the last release) so the app never shows stale numbers.
Files that pin numbers today (grep result): `src/inference/model_registry.py` (BRAIN_TRIAGE.notes),
`src/inference/brain_analysis.py` (docstring + comment + "sensitivity 90%" label),
`src/inference/brain_report_templates.py` (TRIAGE_DESCRIPTIONS ru/uz/en),
`docs/intended_use_{en,ru,uz}.md` (performance table: 0.90 / 118 of 131 / 0.47 / 22 of 47 / 0.79;
"178 held-out patients (131 abnormal, 47 normal)"; "890 axial slices"; "evaluated 2026-09-07";
CI "0.33–0.61"; identity `0d559766ce58…`), `docs/user_manual_{en,ru,uz}.md`,
`docs/pilot_evaluation_protocol_{en,ru,uz}.md` (0.90/0.47/178 + floors), `docs/admin_install_guide_{en,ru,uz}.md`
(identity), `docs/attributions.md` (numbers + identity, en + ru), `docs/README.md`, `README.md`,
`docs/windows_laptop_test_{en,ru}.md` (identity), `tests/test_build_pipeline.py` (identity string),
`docs/test_docs_consistency.py::test_validated_numbers_match_eval_json_and_manifest` (literal numbers →
must read the JSON instead).

### 10. bundle + publish — `packaging/publish_model_bundle.py --model models/brain_triage_finetuned --out data/release/<tag>/ [--github SHAXZODBR/SIAA-models --tag brain-triage-v<N>] [--part-mb 80]`
Zip in the layout `packaging/fetch_model_bundle.py` documents (`brain_triage_finetuned/…` at the zip
root), sha256, split into parts < 100 MB, write `manifest.json` `{"parts": [<urls>], "sha256": …,
"identity": …, "created_at": …}`. Local mode always; with `--github` create the release (draft off,
private repo) via the REST API (urllib only — `gh` is NOT installed), upload parts + manifest, then
print the **manifest asset API URL** (`https://api.github.com/repos/<o>/<r>/releases/assets/<id>`)
to paste into the `MODEL_BUNDLE_URL` environment secret, and verify by restoring with
`fetch_model_bundle.py` into a temp dir. Token: `--token`, else env `SIAA_MODELS_TOKEN` /
`GITHUB_TOKEN`, else `git credential fill` for github.com (the keychain entry that `git push` uses);
never print it, never log the URL with a token. `--dry-run` shows what would be uploaded.
Additions after review: publish refuses (exit 1, before writing anything) a model that is not validated
(eval missing, gate not passed, promotion not true, floors below 0.85/0.40, eval of other weights,
`promotion_log.json` forced) unless `--allow-unvalidated` (demo only; `validated: false` in the manifest);
a validated `manifest.json` carries `validated: true` + an `eval` summary. Without `--tag` the tag is
`brain-triage-<identity12>`. When the tag's release already holds a `manifest.json`: a different identity
is refused unless `--replace-existing`; the same bundle is reused untouched (the `MODEL_BUNDLE_URL` asset id
does not change; `--reupload` forces a new upload). `--part-mb` must be 20–99 with `--github`; manifests
over 1 MiB are refused (publish and fetch share the limit). CI's "Verify bundle identity matches the code"
step fails when the restored identity ≠ `src/inference/validated_metrics.FALLBACK_IDENTITY`.

### 11. orchestrator — `scripts/release_pipeline.py <stage|all> --batch <name> --root <drive> --reports <folder-or-file> [--tag <tag>] [--extra-sets …] [--dry-run] [--resume]`
Runs stages 0→10 in order with a state file `data/release/<tag>/state.json` (per-stage status,
inputs' sha, outputs, timestamps, log paths under `data/release/<tag>/logs/`), stops at the first
failure with the exact command to re-run, `--resume` skips completed stages, `plan` prints the
commands without running. After stage 10 it prints the manual steps: set the secret, run the test
suite, commit the bump, `git tag v<version> && git push origin v<version>`, download the artifact,
run `docs/windows_laptop_test_*.md`. Runbook: `docs/release_runbook_{en,ru}.md` (+ link from
`docs/README.md`).

## Rehearsal data available on this Mac (for integration tests / dry runs)
- 4,847 real PDFs (batch 1): `/Users/shakhzodbtr/Desktop/Siaa_ai/mrt_mskt_hospital_2026-04-29_2026-07-27 2`
- 66 real brain studies (batch 1 images): `/Users/shakhzodbtr/Desktop/Siaa_ai/brain_studies_keep/<StudyUID>/…dcm`
- batch 1 reference labels: `data/matched_clean.csv`, `data/hospital_reports_labels.csv`, slices `data/local_finetune_bin/{train,val}`
- batch 2 (new1) index without reports: `data/new1_index.csv` (3,498 studies; drive `/Volumes/Transcend/new1` is NOT mounted now)
- current model: `models/brain_triage_finetuned` (loads on MPS/CPU via `src.inference.model_registry.load_hf_classifier_dir`)
