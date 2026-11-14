# Release pipeline — shared contract (2026-09-25, updated 2026-11-14)

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

The orchestrator runs twelve stages in this order:
`index → parse → label → match → build → prospective → combine → train → gate → promote → bump → bundle`.
The section numbers below are the historical contract numbers that the scripts' docstrings cite, so
they are kept: `prospective` is section 4b (row 5 of the runbook stage table), the manual LLM second
read is section 2b (not an orchestrator stage).

"Which DICOM object speaks for a study" has one definition: `src/pipeline/dicom_objects.py`
(`NON_IMAGE_MODALITIES`, `modality_of`, `pixel_count`, `sop_class_of`, `is_secondary_capture`,
`is_image_object`). A structured report (SR), presentation state (PR), key object (KO), scanned form (OT) …
carries its own Modality and no body part; a Secondary Capture (a CT "Dose Report" page, a screenshot)
carries the acquisition's Modality and pixels but is not an acquired slice. Code that took "the first
file" read MR brain studies as `SR` and a head CT as body part `NECK`. Study-level modality / body part
always come from the first IMAGE object (image matrix, Modality not in `NON_IMAGE_MODALITIES`, not a
Secondary Capture SOP class); a study with no such object is described by its first readable object, as
before. The server's `best_path` (single-file fallback, preview) prefers image objects the same way, and
when an upload holds at least one image series, non-image objects and Secondary Captures (a CT Dose
Report) take no part in CT routing or CT head slice selection (they are grouped apart from the images even
under one SeriesInstanceUID); an upload without any image series is routed on all its objects, as before.

### 0. index (exists) — `scripts/index_new_studies.py --root <drive> --out data/<batch>_index.csv`
One row per study folder. Columns used downstream: `study_folder, patient_id, study_date (YYYYMMDD),
modality (MR|CT|…), body_part, study_description, error`. Resumable (rows already in `--out` are kept —
except a row without an error whose modality is a non-image value or empty: an index written before the
image-header rule described such a study by its report object, so it is indexed again). Reads one header per
series directory (`stop_before_pixels`); when that header is not an image object (a report exported next to
the images in one flat folder), further headers of that directory, at most `HEADERS_PER_DIR` (50), until an
image object turns up. The study-level fields come from the first header, in sorted series order, for which
`is_image_object` is true — when no series has one, from the first readable header, as before, with `error`
`no image object` (`no image object in the first 50 headers of each folder` when a directory hit the cap).
Such an error row is not indexed again on resume, and the matcher leaves error rows out (the report counts
as unmatched).

### 1. parse-reports — `scripts/parse_hospital_reports.py --src <src> [<src> …] --out data/<batch>_reports.csv`
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

Several sources (one folder or file each, non-recursive) are report DELIVERIES, listed OLDEST FIRST — the
order is part of the contract. Each file is parsed as before (in parallel, `--workers`); the files are then
fed, source by source, to `reconcile_files`: a row whose repeat key (patient_id, study_date, modality,
body_part, section_header, conclusion) equals a row kept from an EARLIER file is dropped (repeat; the earlier
row keeps its `section_index`) — by COUNT: a later file drops at most as many rows of one key as the earlier
files kept, so a second identical-looking section of the same visit survives; a row with an empty
`study_date` is never dropped as a repeat (it still takes part in renumbering); a request-sheet row (header
built from sheet name + area) and a PDF section of the same study differ in `section_header`, so both are
kept; and a remaining row whose (patient_id, study_date, modality, section_index) is taken by a
row kept from an earlier file is renumbered to 1 + the largest index used for that patient / date /
modality (earlier files and its own). A file is identified by its position in that order, never by
`source_file` (the hashed name is the same for one patient in two deliveries). Repeats or collisions
inside one file are left as before; rows without patient_id are ignored by both steps. Why: labels are
attached by that key, so it must name ONE section, and appending a newer delivery never changes the rows
or keys of the older ones. `--limit N` = the first N files of all sources together. With several sources,
ANY source missing or without report files → exit 1 naming it (`source 2 of 2`), no CSV. The summary adds
(several sources, or anything reconciled) `Sources (oldest first): files per source N + M` and
`N rows dropped as repeats across files · M rows renumbered (section_index taken by an earlier file)` —
counts only. `OUTPUT_COLUMNS`, `--out`, `--workers` and the exit codes are unchanged.

### 2. label — `scripts/label_reports.py --reports data/<batch>_reports.csv --out data/<batch>_labels.csv [--benchmark] [--llm ollama:<model>] [--import-llm-labels <jsonl>]`
Adds `label` (9-class), `abnormal` (yes/no), `label_confidence` (high|medium|low), `label_source`
(rules|llm|import), `needs_review` (yes/no) to each brain row. Non-brain rows: `label=''`,
`abnormal=''`. `--benchmark` scores the rules against the 894-patient reference (prints agreement on
`abnormal` and on `label`, confusion) and exits non-zero below `--min-agreement` (default 0.90 on
abnormal). `--import-llm-labels` merges a JSONL of
`{"patient_id","study_date","modality","section_index","label","abnormal"}` (output of
`training/relabel_workflow.js`, the Claude Code Workflow used for batch 1) with `label_source=import`.
Matching: (patient_id, study_date, modality, section_index), then (patient_id, study_date, modality),
then (patient_id, study_date) — each fallback only when it names exactly ONE row; a line that carries the
`study_date` key is matched on it even when it is `''` (an undated row); patient_id alone only when the
label has no date and the patient has exactly one brain row. A label is applied to at most
one row; an ambiguous or conflicting label is not applied (the row goes back to `needs_review`). The
CLI prints `N applied, N ambiguous, N skipped`. `--export-for-llm <jsonl>` writes
`{"patient_id","study_date","modality","section_index","conclusion"}` lines for the rows that
`needs_review=yes` (or all with `--all`). Hedged findings ("нельзя исключить …", "(?)") keep their label
with confidence ≤ medium and `needs_review=yes`. An import line may carry an optional `needs_review`
(`yes`/`true`/`1`): the label is applied (and counted as applied) but the row STAYS in review with
`label_confidence=low`; the CLI then also prints `N imported label(s) stay in review`. Without the field
(or `no`) the row leaves review as before. (`label_source=llm` is the optional Ollama pass, `--llm`.)

### 2b. second read (manual, not an orchestrator stage) — `training/relabel_workflow.js` + `scripts/collect_llm_labels.py`
Two rounds, each = one Workflow run in DIRECT mode, then the collector:
- export every brain row once: `label_reports.py --reports … --out data/<batch>_labels_rules.csv
  --export-for-llm data/<batch>_review_all.jsonl --all` — the rule labels go to their OWN file: the collector
  compares the LLM with the rules and refuses a `--rules` file whose brain rows carry imported / LLM labels.
- round 1: `Workflow({ scriptPath: 'training/relabel_workflow.js', args: { file: '<abs path of the export>',
  n_lines: <wc -l>, votes: 1 } })` (optional `batch`, default 75; `model`). No reader agent: every
  labelling agent prints its own line range and writes `<file>.labels/v<vote>-b<batch>.tsv`
  (`<row>\t<patient_id>\t<label>`). EACH round starts with an empty (or absent) `<file>.labels/`: the
  collector reads every part file in it and a run overwrites only the batches it needs, so part files of an
  earlier run (a longer export) would be counted as rejected votes. Then `collect_llm_labels.py --export <export> --rules
  data/<batch>_labels_rules.csv --out data/<batch>_llm_round1.jsonl --disputed data/<batch>_disputed.jsonl`: a row
  is DECIDED when the LLM label equals the rule label; every other export line is copied verbatim to
  `--disputed`.
- round 2: the same Workflow on the disputed file with `votes: 3`, then `collect_llm_labels.py --export
  data/<batch>_disputed.jsonl --rules data/<batch>_labels_rules.csv --prior data/<batch>_llm_round1.jsonl --out
  data/<batch>_llm_labels.jsonl` (no `--disputed` = FINAL). Majority on the abnormal bit, then the most
  common finding among the majority (tie → the rule label when tied, else the most severe). A row whose
  reads split on normal-vs-abnormal gets the majority label with `needs_review=yes`; a row still undecided
  keeps its rule label with `needs_review=yes` (no rule label → left out, counted).
- the final file holds `--prior` + this round, one line per row: the 4 import keys copied from the EXPORT
  (never from an agent), `label`, `abnormal`, `needs_review`, `decided_by`, `votes`, `rule`. It is imported
  with `label_reports.py --import-llm-labels` and recorded with `release_pipeline.py … --llm-labels`.
- a vote counts only when its row is inside the export, its patient_id equals the export line's and the
  label is one of the nine; anything else is rejected and counted (it can only send a row to the next
  round). `--result <json>` (the workflow's return value) cross-checks the part files of its `out_dir`, or
  is used directly when that run left no part file. Exit 0 ok · 2 usage / unreadable input / duplicate
  export keys / a `--rules` file that already holds imported labels / a row labelled in both `--prior` and
  this round / no vote lines at all (nothing written) · 3 more than `--max-rejected` (default 2 %) rejected
  votes, including "every vote rejected": only `--summary` is written, `--out` and `--disputed` are left
  untouched (an agent shifted by one row can pass the patient check on a same-patient CT/MR pair, so even
  the accepted votes of such a run are not trusted). Exit 3 prints the rejected count and reasons per part
  file (no patient ids) and, when any `row_out_of_range` rejection is present, a warning naming the parts
  directory: part files of an earlier run may be left there, re-running the workflow alone cannot fix it —
  delete the directory, then re-run the workflow and the collector.

### 3. match (exists, extend) — `scripts/match_images_to_reports.py [--images <root>] --index data/<batch>_index.csv --labels data/<batch>_labels.csv --out data/<batch>_matched.csv`
`--images` is optional when `--index` is given (it is not read then); with neither, argparse exits 2
(`--images is required when no --index is given`). So match does not need the drive.
Join order: same `patient_id` **and** same date > same patient + compatible modality/body part > only
report for the patient. Carry through: `abnormal, label_source, label_confidence, needs_review,
study_date, match_quality`. Existing columns unchanged (`study_folder, patient_id, modality,
body_part_dicom, is_brain, report_matched, report_body_part, label`).

### 4. build-set (exists, extend) — `scripts/build_training_set.py --images <root> --matched data/<batch>_matched.csv --out data/sets/<batch> --mode binary [--abnormal-col abnormal] [--skip-needs-review] [--modality MR]`
Writes `data/sets/<batch>/{train,val}/{abnormal,normal}/<patient_id>_<k>.png` **plus**
`data/sets/<batch>/studies.csv` (`patient_id, study_folder, cls, split, n_slices, label, label_source,
modality` — `modality` appended last). When `--abnormal-col` is given the binary class comes from that
column (yes→abnormal, no→normal), not from `label`. `--fresh` removes the out dir; without it, existing
files are reused (resumable). `--images` stays required.

MR-only rule (`--modality MOD`, upper-cased; a non-image value such as `SR` / `OT` or an empty value is
refused, argparse exit 2; without the flag nothing is filtered). The orchestrator passes `--modality MR`
by default (`release_pipeline.py --modality any` leaves the flag out):
- row level (`select_studies`, before any DICOM is read): a matched row whose `modality` names ANOTHER
  acquisition modality is dropped as `modality CT`; an empty or non-image value (`SR`, `PR`, `OT` …: older
  indexes say `SR` for many MR studies) is kept and decided at build time;
- build time: `triage_input(study, k)` returns the slices AND the Modality of the series
  `triage_sequence` picks (`central_slices` is now a thin wrapper — same series, same rule). That
  Modality must equal the rule exactly; otherwise the study gets no slices, keeps its `studies.csv` row
  with `n_slices` 0 and is counted `modality CT (series)` / `modality unknown (series)`, not as failed. A
  study already recorded as another modality is not read again on later runs with the same rule;
- `studies.csv` `modality` = the Modality of the series the slices came from (excluded study: the
  modality found; `''` = not verified). PNGs are reused without reading DICOM only when the previous
  `studies.csv` recorded `modality == rule`; otherwise the study is re-read and its files rewritten or
  REMOVED. A set built by this version without a rule already records modalities (its MR studies count
  as verified); a set from an older version (no column) is re-read. The pre-loop `studies.csv` carries
  only already-verified modalities, so an interrupted run never leaves unverified files that look verified;
- reconcile marker: before reconcile renames or removes files on disk (and before the `--modality`
  stale-file removal), the builder writes and fsyncs `<out>/.reconcile_pending`; it is removed right after
  the run's first `studies.csv` write. A run that finds it exits 1 before reading anything: `… an earlier
  run was interrupted while renaming files — the slices on disk no longer match studies.csv; rebuild with
  --fresh` (`--fresh` removes the out dir, marker included). A reconcile that changes nothing writes no
  marker; a crash later in the build loop leaves none;
- summary: `modality rule: --modality MR`, `rows dropped before reading DICOM: …`, `studies excluded at
  build time (kept in studies.csv, n_slices 0): …`, plus re-verified / removed counts. When studies were
  excluded and none built or reused: exit 1 `no study produced slices: N failed, M excluded by --modality MR`;
  when the row filter removes everything, the existing "no trainable brain studies" exit 1 fires.
- The rule applies to the set being built. Sets that already exist — the batch-1 legacy set
  `data/local_finetune_bin`, earlier `data/sets/*` — are combined as built and never re-checked.

### 4b. prospective (orchestrator stage after build) — `scripts/eval_prospective.py --model <shipped model> --set data/sets/<batch> [--seen <set>]… [--holdout data/holdout_patients_v1.txt] [--matched data/<batch>_matched.csv] --out data/release/<tag>/prospective_eval.json [--device mps]`
Question: how does the model shipped TODAY do on patients it never saw? Runs BEFORE combine / train use
the batch, with the shipped protocol of `eval_study_level.py` (one `make_predictor`, `collect_study_probs`
on every split of the set, `PROTOCOL_AGGREGATION` @ `PROTOCOL_THRESHOLD`, `score_studies`, Wilson CIs).
The orchestrator passes `--model models/brain_triage_finetuned` (`BASE_MODEL`, the release directory the
gate compares with and promote replaces — NOT `--base-model`, which is only where training starts),
`--seen` = every extra set it combines (incl. `data/local_finetune_bin`) plus, after them, the release set
the shipped model was trained on (`prospective_seen_release`: `--train-lock` when it is a directory, else
the set named in `BASE_MODEL/training_metrics.json`; not added when it is already an extra set, when the
model was trained on the legacy set only, or when it is this tag's own release set), the holdout and the
matched CSV. That set is a stage input (`<set>/dataset_manifest.json`). Guard `pre_prospective`: when the
set is gone or rebuilt (`train_lock_problem`), the stage is refused before it runs — never scored with an
incomplete `--seen` list, which would count trained-on patients as `new`; remedy: restore the set or pass
`--train-lock <release set dir>` (`--seen` takes directories only, so an ids file does not satisfy it).
- `--seen` set kinds: release set (`dataset_manifest.json`; `train/`+`tune/` trained on, `val/` not —
  authoritative), legacy set (no `studies.csv`, no manifest — authoritative the same way), per-batch set
  (`studies.csv`, no manifest: its train/val split is ADVISORY, combine re-splits it; a note says so).
- groups: `seen_train` (a PNG under `train/` or `tune/` of a release / legacy `--seen` set — wins over
  holdout; else any patient of a per-batch `--seen` set that is in no holdout source), `holdout`
  (`--holdout` or `val/` of a release / legacy `--seen` set), `new` (in no `--seen` set at all — THE
  HEADLINE), `all`, and with `--matched` `new_same_date` (`new` studies whose matched rows all have
  `match_quality == 'date'`).
- per group: `n_studies, n_patients, n_abnormal, n_normal, tp, fn, tn, fp, sensitivity(+_ci95),
  specificity(+_ci95), balanced_accuracy, ppv, npv, flag_rate, prevalence, auc` (empty denominator →
  null). For `new` also `threshold_sweep_new` (descriptive only) and `findings_new` (per finding; a label
  that disagrees with its class folder is keyed `<label> (reference <cls>)`).
- JSON: `kind: prospective_eval`, `schema_version: 1`, `evaluated_on`, `model {path, identity}`,
  `protocol`, `inputs`, `groups`, `threshold_sweep_new`, `findings_new`, `notes` — written atomically,
  counts only (no patient ids, no study folders, neither in the JSON nor on stdout). Stdout starts with
  `NEW PATIENTS (never seen by this model): …`.
- exit 0 = it ran, whatever the numbers (never a gate); exit 2 = unusable input (path missing, set
  without PNG, model not loadable / no `abnormal` class, a study key under both classes or both splits).
  The model's files are checked against its `MANIFEST.json` before scoring (`eval_study_level.
  check_manifest`): a mismatch is exit 2; a directory without a manifest is scored with a note and
  `model.manifest_verified: false`.
- orchestrator: inputs `studies.csv`, the matched CSV, the holdout, the extra sets, the trained-on release
  set (when added), the shipped model's `MANIFEST.json`; good =
  the JSON exists, parses, has `kind` `prospective_eval` and a `groups` object (else the stage is failed,
  and an invalid JSON is not done on `--resume`); prints `prospective, never-seen patients: n N studies ·
  sensitivity x · specificity y`. Never re-run after promote (it is in `NO_RERUN_AFTER_PROMOTE` and
  `FROZEN_AFTER_PROMOTE`): the release directory then holds the new model, trained on these patients. A
  state promoted by an older version (no `prospective` record) counts it as done with the reason
  `NOT_APPLICABLE_AFTER_PROMOTE`: `run` prints `stage prospective skipped (promoted before this stage
  existed)`, `status` shows `skipped (…)`, and `--resume` finishes bump + bundle.
- honest scope: these numbers, not the gate's, describe performance on new patients; reference labels are
  report-derived (one reader, parsed), one clinic, not a pre-registered consecutive cohort; `needs_review`
  studies and (under the rule) non-MR studies are not in the set.

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

### 10. bundle + publish — `packaging/publish_model_bundle.py --model models/brain_triage_finetuned --out data/release/<tag>/ [--github SHAXZODBR/SIAA-models] [--tag …] [--part-mb 80]`
Zip in the layout `packaging/fetch_model_bundle.py` documents (`brain_triage_finetuned/…` at the zip
root), sha256, split into parts < 100 MB, write `manifest.json` `{"parts": [<urls>], "sha256": …,
"identity": …, "created_at": …}`. Local mode always; with `--github` create the release (draft off,
private repo) via the REST API (urllib only — `gh` is NOT installed), upload parts + manifest, then
print the **manifest asset API URL** (`https://api.github.com/repos/<o>/<r>/releases/assets/<id>`)
for information (the optional `MODEL_BUNDLE_URL` fallback), and verify by restoring with
`fetch_model_bundle.py` into a temp dir. Token: `--token`, else env `SIAA_MODELS_TOKEN` /
`GITHUB_TOKEN`, else `git credential fill` for github.com (the keychain entry that `git push` uses);
never print it, never log the URL with a token. `--dry-run` shows what would be uploaded.
Additions after review: publish refuses (exit 1, before writing anything) a model that is not validated
(eval missing, gate not passed, promotion not true, floors below 0.85/0.40, eval of other weights,
`promotion_log.json` forced) unless `--allow-unvalidated` (demo only; `validated: false` in the manifest);
a validated `manifest.json` carries `validated: true` + an `eval` summary. Without `--tag` the tag is
`brain-triage-<identity12>` — the orchestrator's bundle stage MUST use that default (it passes no `--tag`):
the installer CI resolves the bundle by the code pin (`packaging/fetch_model_bundle.py --repo`, release
`brain-triage-<FALLBACK_IDENTITY>`, asset `manifest.json`; `--require` on `v*` tags), so no secret is
edited per release and `MODEL_BUNDLE_TOKEN` only needs read access to the models repo. An installer that
contains the model is never a workflow artifact of the public repository: it reaches the operator only
through the tag's DRAFT release, which must not be published. When the tag's release already holds a
`manifest.json`: a different identity is refused unless `--replace-existing`; the same bundle is reused
untouched (the manifest asset id does not change; `--reupload` forces a new upload). `--part-mb` must be 20–99 with `--github`; manifests
over 1 MiB are refused (publish and fetch share the limit). CI's "Verify bundle identity matches the code"
step fails when the restored identity ≠ `src/inference/validated_metrics.FALLBACK_IDENTITY`. On the
lookup path that comparison is not independent (the release was found by that same pin): the independent
check is `fetch_model_bundle.py --require-newest` (also on `v*` tags only), which fails when the models
repository holds a published `brain-triage-*` release created after the pinned one — usually a tag pushed
before the bump commit; the deliberate override is the repository variable `MODEL_BUNDLE_ALLOW_OLDER=1`
(or `--allow-older`: a warning instead), to be deleted after that build. An unreadable release list fails
even with the override. The CI's "Attach to draft GitHub Release" step reads the tag release's `isDraft`
before any upload and fails, uploading nothing, when the release is already published (a re-run or
*Run workflow* on such a tag): a model-bearing installer only ever goes into a DRAFT release.

### 11. orchestrator — `scripts/release_pipeline.py <stage|all> --batch <name> --root <drive> --reports <src> [<src> …] [--modality MR|any|…] [--tag <tag>] [--extra-sets …] [--dry-run] [--resume]`
Runs the twelve stages `index → parse → label → match → build → prospective → combine → train → gate →
promote → bump → bundle` (sections 0–10 above, prospective = 4b) in order with a state file
`data/release/<tag>/state.json` (per-stage status,
inputs' sha, outputs, timestamps, log paths under `data/release/<tag>/logs/`), stops at the first
failure with the exact command to re-run, `--resume` skips completed stages, `plan` prints the
commands without running. Additions after the 2026-11 round:
- `--reports` takes one or more sources (`nargs='+'`, repeating the flag appends), OLDEST delivery first,
  stored as a list in `state.json` (an older state with one string loads as a one-element list). parse gets
  them all after `--src` in that order; each source is a parse input, and with two or more an
  `opt:report_order` input re-runs parse when a source is added or the order changes. With one source the
  parse inputs are exactly as before, so an existing release keeps parse / label done.
- `--modality` (default `MR`; `any` = no rule) is passed to build as `--modality <value>` and fingerprinted
  as `opt:modality`; a change re-runs build and later stages, not match. A build recorded without the key
  (before the MR-only rule) counts as `opt:any` (`LEGACY_OPTION_VALUES`), so a `--resume` re-runs it under
  the current rule; any other `opt:*` key a state did not record still counts as unchanged. `plan` prints `# report sources (oldest delivery first): …` and `# modality rule (build): …`.
- `prospective` (4b) sits between build and combine; like combine it is never re-run after promote. Its
  guard (`guard_pre_prospective`, in `GUARDS` / `GUARD_TEXT`) refuses the stage when the release set the
  shipped model was trained on is gone or rebuilt. After promote, a frozen stage with no record (a state
  promoted before `prospective` existed) is done with the reason `NOT_APPLICABLE_AFTER_PROMOTE`; an explicit
  `prospective` / `combine` / `train` / `gate` / `promote` run after promote is still refused.
- Only `index` and `build` need the drive (`NEEDS_ROOT`); match gets `--images <root>` only when a root is
  configured. `main()` refuses up front only when the first stage that will actually run needs the drive;
  otherwise `cmd_run` runs the stages before it and stops at the first NEEDS_ROOT stage with
  `drive not mounted: <root> (needed by stage <name>)`. No guard rail checks the drive (the pre-train
  guard does not: train reads only the local release set). After stage 10 it prints the manual steps:
  nothing to paste (CI finds the model by the pinned identity; `MODEL_BUNDLE_TOKEN` must read the models
  repo), run the test suite, commit + push the bump, bump the app version,
  `git tag v<version> && git push origin v<version>`, download the installer from the tag's draft release,
  run `docs/windows_laptop_test_*.md`, and do NOT publish the draft release while it holds the full
  installer. Runbook: `docs/release_runbook_{en,ru}.md` (+ link from
`docs/README.md`).

## Rehearsal data available on this Mac (for integration tests / dry runs)
- 4,847 real PDFs (batch 1): `/Users/shakhzodbtr/Desktop/Siaa_ai/mrt_mskt_hospital_2026-04-29_2026-07-27 2`
- 66 real brain studies (batch 1 images): `/Users/shakhzodbtr/Desktop/Siaa_ai/brain_studies_keep/<StudyUID>/…dcm`
- batch 1 reference labels: `data/matched_clean.csv`, `data/hospital_reports_labels.csv`, slices `data/local_finetune_bin/{train,val}`
- batch 2 (new1): index `data/new1_index.csv` (3,500 study folders: 1,953 MR, 1,545 CT; drive `/Volumes/Transcend/new1`), reports received 2026-11-10 (`reports_shox_2026-06-27_2026-09-20`, period A, matches the images; `reports_shox_2026-09-21_2026-11-08`, period B, images not received yet), second-read labels `data/new1_intake/llm_labels_A.jsonl`
- current model: `models/brain_triage_finetuned` (loads on MPS/CPU via `src.inference.model_registry.load_hf_classifier_dir`)
