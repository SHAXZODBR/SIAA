#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PROSPECTIVE EVALUATION OF THE SHIPPED MODEL ON A NEW BATCH
================================================================================
  WHAT
    Scores every study of a freshly reported batch (a build_training_set.py set:
    <set>/{train,val}/<cls>/<prefix>_<k>.png + studies.csv) with the model that
    is shipped TODAY, using exactly the shipped protocol of
    scripts/eval_study_level.py (registry predictor, native-resolution PNG,
    mean abnormal probability over the study's slices >= 0.50 → flagged).
    The train/val split of a per-batch set is irrelevant here: every study is
    scored. A --seen set is one of three kinds:
      release     a combined release set (dataset_manifest.json): train/ and
                  tune/ were trained on, val/ was not (authoritative);
      legacy      no studies.csv and no dataset_manifest.json (e.g.
                  data/local_finetune_bin): train/ (and tune/) trained on,
                  val/ not (authoritative);
      per-batch   a build_training_set.py output (studies.csv, no
                  dataset_manifest.json): its train/val split is ADVISORY —
                  combine_training_sets.py re-splits those patients and
                  trains on most of them — so every patient of it is seen,
                  role unknown.
    Patients are split into groups:
      seen_train  a PNG of the patient is under train/ or tune/ of a release
                  or legacy --seen set; else (not in any holdout source) any
                  PNG of the patient in a per-batch --seen set
      holdout     listed in --holdout or under val/ of a release / legacy
                  --seen set, and not under train/ or tune/ of one — the
                  model was validated, not trained, on them
      new         everyone else (in no --seen set at all) — THE HEADLINE GROUP
      all         every scored study
      new_same_date (with --matched) the 'new' studies whose report and images
                  are from the same day (match_quality == 'date'): the
                  cleanest reference label
    Per group: confusion counts, sensitivity / specificity with Wilson 95%
    intervals, balanced accuracy, PPV, NPV, flag rate, prevalence and a
    rank-based AUC of the study mean probability. For 'new' also a threshold
    sweep (DESCRIPTIVE ONLY — the shipped threshold stays 0.50) and per-finding
    detection from the studies.csv 'label' column.
    With --index (the batch index: patient_age / birth + study date), 'age_new':
    does the model beat AGE ALONE (AUC of the age, patient-level bootstrap CI
    of the difference), its AUC between patients within 5 years of age, the
    age rule 'age >= T' at the model's sensitivity and flags per age band.
    With --priority (report_priority.py output), 'priority_new': how the flag
    treats studies with a priority finding (mass, recent infarct, ...) versus
    the rest. Both describe the 'new' group only; neither is a gate.

  WHY
    When a batch gets its reports, the first question is how the frozen,
    shipped model does on patients nobody had reports for when it was trained.
    That is the closest thing to a prospective validation this product has:
    it goes into the clinic documentation and it tells us whether a retrain is
    needed. It must run BEFORE the batch is used for training.
    On new1 (2026-11-14) the shipped model was matched by patient age alone
    (AUC 0.779 vs 0.776) and flagged 29 of 30 normal studies from the age of
    40: most 'abnormal' reports describe age-typical change. age_new and
    priority_new print that for every batch, so no model looks good by
    learning age.

  HONEST SCOPE
    The model was frozen before these reports existed, so 'new' is a genuine
    out-of-sample test on the clinic's own scanners; but the reference labels
    are derived from the routine radiology report (parsed, one reader), not a
    consensus read, and the batch is whatever the clinic sent — not a
    pre-registered consecutive cohort.

  OUTPUT
    --out JSON (kind 'prospective_eval', schema_version 2), written atomically,
    plus a compact summary on stdout. Counts only: no patient ids and no study
    folders, neither in the JSON nor on stdout.

  USAGE
    python scripts/eval_prospective.py --model models/brain_triage_finetuned \
        --set data/sets/<batch> --seen data/local_finetune_bin \
        --seen data/train_release_<tag> --holdout data/holdout_patients_v1.txt \
        --matched data/<batch>_matched.csv --index data/<batch>_index.csv \
        --priority data/<batch>_priority.csv \
        --out data/release/<tag>/prospective_eval.json --device mps

  EXIT CODES
    0  the evaluation ran (whatever the numbers: this stage is informational
       and never blocks a release)
    2  unusable inputs: model dir missing / not loadable / without the
       abnormal class / with files that do not match its MANIFEST.json, set
       without PNG files, a study key under both classes or in both splits,
       a --seen / --holdout / --matched / --index / --priority path that is
       missing (an existing but unusable --index / --priority is a note only)
================================================================================
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package

import eval_study_level as esl                                 # noqa: E402
from build_training_set import read_prefix_map                 # noqa: E402

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_OK = 0
EXIT_UNUSABLE = 2
SCHEMA_VERSION = 2                             # 2: + age_new, priority_new (null without --index / --priority)

SET_SPLITS = ('train', 'val')                  # a per-batch set: every study is scored
SEEN_TRAIN_SPLITS = ('train', 'tune')          # the shipped model was fitted / epoch-selected on these
SEEN_HOLDOUT_SPLITS = ('val',)                 # ... and validated on these (release / legacy sets only)
SWEEP_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
GROUP_ORDER = ('new', 'new_same_date', 'holdout', 'seen_train', 'all')


class UnusableInput(Exception):
    """An input the evaluation cannot run on (→ exit 2). Messages never carry ids."""


# ------------------------------------------------------------------ patients / groups

def study_key_of(png_stem: str) -> str:
    """'<study key>_<k>' → study key (the same rule as eval_study_level.collect_study_probs)."""
    return png_stem.rsplit('_', 1)[0]


def _png_patients(root: Path, splits) -> set[str]:
    pids: set[str] = set()
    for sp in splits:
        d = root / sp
        if not d.is_dir():
            continue
        for cls_dir in (c for c in d.iterdir() if c.is_dir()):
            for fp in cls_dir.glob('*.png'):
                pids.add(esl.patient_id_of(study_key_of(fp.stem)))
    return pids


def is_per_batch_set(d: Path) -> bool:
    """A build_training_set.py output (studies.csv, no dataset_manifest.json): its split is advisory."""
    return (d / 'studies.csv').is_file() and not (d / 'dataset_manifest.json').is_file()


def load_seen_patients(seen_dirs) -> tuple[set[str], set[str], set[str], list[str]]:
    """(seen_train, seen_holdout, seen_batch, notes) over every --seen set.

    seen_train / seen_holdout: train/+tune/ and val/ of the AUTHORITATIVE sets — a combined release
    set (dataset_manifest.json) or the legacy set (no studies.csv). seen_batch: every patient of a
    per-batch set (studies.csv, no dataset_manifest.json), whose train/val split is only advisory
    (combine re-splits it and trains on most of it): seen, role unknown."""
    train: set[str] = set()
    val: set[str] = set()
    batch: set[str] = set()
    notes: list[str] = []
    for d in seen_dirs:
        d = Path(d)
        if not d.is_dir():
            raise UnusableInput(f'--seen {d}: not a directory')
        if not any((d / sp).is_dir() for sp in SEEN_TRAIN_SPLITS + SEEN_HOLDOUT_SPLITS):
            notes.append(f'--seen {d} has no train/, tune/ or val/ folder: it contributed no patients')
        if is_per_batch_set(d):
            batch |= _png_patients(d, SEEN_TRAIN_SPLITS + SEEN_HOLDOUT_SPLITS)
            notes.append(f'--seen {d} is a per-batch set (studies.csv, no dataset_manifest.json): its train/val '
                         'split is advisory, so its patients count as seen_train unless a release / legacy val/ '
                         'or --holdout lists them')
            continue
        train |= _png_patients(d, SEEN_TRAIN_SPLITS)
        val |= _png_patients(d, SEEN_HOLDOUT_SPLITS)
    return train, val, batch, notes


def assign_groups(pids, seen_train: set[str], seen_val: set[str], holdout_ids: set[str],
                  seen_batch: set[str] = frozenset()) -> dict[str, str]:
    """patient id → 'seen_train' | 'holdout' | 'new'. Precedence: seen_train (authoritative train/tune)
    > holdout (--holdout or an authoritative val/) > seen_batch (a per-batch set: seen_train) > new."""
    groups = {}
    for pid in pids:
        if pid in seen_train:
            groups[pid] = 'seen_train'
        elif pid in holdout_ids or pid in seen_val:
            groups[pid] = 'holdout'
        elif pid in seen_batch:
            groups[pid] = 'seen_train'
        else:
            groups[pid] = 'new'
    return groups


# ------------------------------------------------------------------ scoring

def collect_set(model_dir: Path, set_dir: Path, predictor, abnormal_class: str = 'abnormal'
                ) -> tuple[esl.StudyProbs, int]:
    """Every PNG under <set>/{train,val}/<cls>/ through the shipped protocol
    (eval_study_level.collect_study_probs with ONE predictor). Refuses a study
    key under both classes or in both splits."""
    studies: esl.StudyProbs = {}
    split_of: dict[str, str] = {}
    n_png = 0
    n_both_cls = n_both_split = 0
    for sp in SET_SPLITS:
        root = Path(set_dir) / sp
        if not root.is_dir():
            continue
        try:
            part, n = esl.collect_study_probs(model_dir, root, abnormal_class=abnormal_class, predictor=predictor)
        except SystemExit as e:                    # model without the abnormal class
            raise UnusableInput(str(e.code) if e.code is not None else 'model scoring refused') from None
        except OSError as e:                       # unreadable PNG (the path is not echoed: it holds an id)
            raise UnusableInput(f'unreadable PNG in {sp}/ ({type(e).__name__})') from None
        cls_of: dict[str, set[str]] = defaultdict(set)
        for key, cls in part:
            cls_of[key].add(cls)
        n_both_cls += sum(1 for c in cls_of.values() if len(c) > 1)
        for key in cls_of:
            if key in split_of:
                n_both_split += 1
            split_of[key] = sp
        studies.update(part)
        n_png += n
    if n_both_cls:
        raise UnusableInput(f'{n_both_cls} study key(s) appear under more than one class folder')
    if n_both_split:
        raise UnusableInput(f'{n_both_split} study key(s) appear in both train/ and val/')
    return studies, n_png


def study_mean(probs: list[float]) -> float:
    return sum(probs) / len(probs)


def auc(pos: list[float], neg: list[float]) -> Optional[float]:
    """Rank-based (Mann-Whitney) AUC, ties counted 1/2 via mid-ranks. None when a class is empty."""
    n1, n0 = len(pos), len(neg)
    if not n1 or not n0:
        return None
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda t: t[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        mid = (i + j) / 2 + 1                      # 1-based mid-rank of the tie block
        rank_sum_pos += mid * sum(lab for _, lab in scored[i:j + 1])
        i = j + 1
    return (rank_sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ratio(k: int, n: int) -> Optional[float]:
    return k / n if n else None


def group_metrics(studies: esl.StudyProbs, abnormal_class: str = 'abnormal', name: str = '') -> dict:
    """Shipped-protocol metrics of one group of {(study_key, true_cls): [slice probs]}.
    Empty denominators give None (never a division by zero)."""
    r = esl.score_studies(studies, esl.gate_fn(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD),
                          name, abnormal_class)
    tp, fn, tn, fp = r['tp'], r['fn'], r['tn'], r['fp']
    n_abn, n_norm = tp + fn, tn + fp
    n = n_abn + n_norm
    sens = r['sens'] if n_abn else None
    spec = r['spec'] if n_norm else None
    pos = [study_mean(p) for (_, c), p in studies.items() if c == abnormal_class]
    neg = [study_mean(p) for (_, c), p in studies.items() if c != abnormal_class]
    return {'n_studies': n, 'n_patients': len({esl.patient_id_of(k) for k, _ in studies}),
            'n_abnormal': n_abn, 'n_normal': n_norm,
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': sens, 'sensitivity_ci95': r['sens_ci'] if n_abn else None,
            'specificity': spec, 'specificity_ci95': r['spec_ci'] if n_norm else None,
            'balanced_accuracy': (sens + spec) / 2 if sens is not None and spec is not None else None,
            'ppv': _ratio(tp, tp + fp), 'npv': _ratio(tn, tn + fn),
            'flag_rate': _ratio(tp + fp, n), 'prevalence': _ratio(n_abn, n),
            'auc': auc(pos, neg)}


def threshold_sweep(studies: esl.StudyProbs, abnormal_class: str = 'abnormal') -> list[dict]:
    """Mean-probability threshold sweep — DESCRIPTIVE ONLY, the shipped threshold stays 0.50."""
    n_abn = sum(1 for _, c in studies if c == abnormal_class)
    n_norm = len(studies) - n_abn
    rows = []
    for t in SWEEP_THRESHOLDS:
        r = esl.score_studies(studies, esl.gate_fn('mean', t), f'mean @{t:.2f}', abnormal_class)
        rows.append({'threshold': t,
                     'sensitivity': r['sens'] if n_abn else None,
                     'specificity': r['spec'] if n_norm else None,
                     'flagged_share': _ratio(r['tp'] + r['fp'], len(studies)),
                     'shipped': t == esl.PROTOCOL_THRESHOLD})
    return rows


# ------------------------------------------------------------------ studies.csv / matched.csv

def load_study_table(set_dir: Path) -> tuple[Optional[dict[str, dict]], Optional[str]]:
    """study key (file prefix) → {'study_folder', 'label', 'cls'} from <set>/studies.csv,
    numbered with build_training_set.read_prefix_map. (None, note) when unusable."""
    sc = Path(set_dir) / 'studies.csv'
    if not sc.is_file():
        return None, 'studies.csv not found: per-finding table and new_same_date omitted'
    try:
        prefix_of = read_prefix_map(Path(set_dir))
        with open(sc, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return None, f'studies.csv unreadable ({type(e).__name__}): per-finding table and new_same_date omitted'
    if prefix_of is None:
        return None, ('studies.csv unusable (no rows, missing columns or duplicate study folders): '
                      'per-finding table and new_same_date omitted')
    table: dict[str, dict] = {}
    for r in rows:
        folder = r.get('study_folder') or ''
        table[prefix_of[folder]] = {'study_folder': folder, 'label': (r.get('label') or '').strip(),
                                    'cls': (r.get('cls') or '').strip()}
    return table, None


def findings_table(studies: esl.StudyProbs, table: dict[str, dict], abnormal_class: str = 'abnormal'
                   ) -> tuple[dict, int]:
    """Per-finding detection of the shipped protocol. Abnormal-reference studies count
    detected/total under their label; normal-reference studies count wrongly_flagged/total
    under 'normal' (a label that disagrees with the reference class is keyed
    '<label> (reference <cls>)'). Returns (table, n studies without a studies.csv row)."""
    decide = esl.gate_fn(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD)
    acc: dict[str, dict] = {}
    n_unmapped = 0
    for (key, cls), probs in studies.items():
        row = table.get(key)
        if row is None:
            n_unmapped += 1
            continue
        label = row['label'] or 'unspecified'
        is_abn = cls == abnormal_class
        typical = (label != 'normal') if is_abn else (label == 'normal')
        name = label if typical else f'{label} (reference {cls})'
        e = acc.setdefault(name, {'reference': 'abnormal' if is_abn else 'normal', 'total': 0,
                                  ('detected' if is_abn else 'wrongly_flagged'): 0})
        e['total'] += 1
        if decide(probs):
            e['detected' if is_abn else 'wrongly_flagged'] += 1
    for e in acc.values():
        e['rate'] = e.get('detected', e.get('wrongly_flagged', 0)) / e['total']
    return dict(sorted(acc.items(), key=lambda kv: (kv[1]['reference'] != 'abnormal', -kv[1]['total'], kv[0]))), n_unmapped


def same_date_keys(keys, table: dict[str, dict], matched: Path) -> tuple[Optional[set[str]], Optional[str]]:
    """The study keys whose matched.csv row(s) all have match_quality == 'date'."""
    with open(matched, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        cols = reader.fieldnames or []
    if 'study_folder' not in cols or 'match_quality' not in cols:
        return None, '--matched has no study_folder / match_quality column: new_same_date omitted'
    quality: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        quality[r.get('study_folder') or ''].add((r.get('match_quality') or '').strip())
    out = set()
    for k in keys:
        row = table.get(k)
        if row is not None and quality.get(row['study_folder']) == {'date'}:
            out.add(k)
    return out, None


# ------------------------------------------------------------------ age (--index) and priority findings (--priority)
#
# WHY: on new1 (2026-11-14) the shipped model was matched by PATIENT AGE ALONE (AUC 0.779 vs 0.776) and
# flagged almost every study from the age of 40, because most 'abnormal' reports describe age-typical change.
# These blocks print that comparison for every new batch, so no future model can look good by learning age,
# and they count how the flag treats the ~10 % of studies with a finding that should move the read up the list.

AGE_MATCH_WINDOW_YEARS = 5.0
AGE_BANDS = (('under 40', 0.0, 40.0), ('40-59', 40.0, 60.0), ('60+', 60.0, float('inf')))
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261114                      # fixed: the same batch must print the same interval every run
MAX_PLAUSIBLE_AGE = 130.0
_AGE_CODE_DIVISOR = {'Y': 1.0, 'M': 12.0, 'W': 52.0, 'D': 365.0}


def parse_age(patient_age: str, birth_date: str = '', study_date: str = '') -> Optional[float]:
    """Age in years at the study: DICOM AS 'NNNY' / 'NNNM' (/12) / 'NNNW' (/52) / 'NNND' (/365), else
    birth date + study date (YYYYMMDD). None for anything else, a negative or an implausible (> 130) age."""
    s = (patient_age or '').strip().upper()
    if len(s) >= 2 and s[:-1].isdigit() and s[-1] in _AGE_CODE_DIVISOR:
        years = int(s[:-1]) / _AGE_CODE_DIVISOR[s[-1]]
        return years if years <= MAX_PLAUSIBLE_AGE else None
    try:
        b = datetime.strptime((birth_date or '').strip(), '%Y%m%d').date()
        d = datetime.strptime((study_date or '').strip(), '%Y%m%d').date()
    except ValueError:
        return None
    years = (d - b).days / 365.25
    return years if 0.0 <= years <= MAX_PLAUSIBLE_AGE else None


def load_index_ages(index: Path) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """study_folder → age in years from a batch index csv (the first row of a folder with a usable age
    wins). (None, note) when the file is unreadable or lacks the columns: never a crash."""
    try:
        with open(index, encoding='utf-8-sig', newline='') as fh:      # -sig: a csv re-saved by Excel has a BOM
            reader = csv.DictReader(fh)
            rows = list(reader)
            cols = set(reader.fieldnames or [])
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return None, f'--index unreadable ({type(e).__name__}): age_new omitted'
    if 'study_folder' not in cols or not ({'patient_age'} <= cols or {'patient_birth_date', 'study_date'} <= cols):
        return None, ('--index has no study_folder column or neither patient_age nor patient_birth_date + '
                      'study_date: age_new omitted')
    ages: dict[str, float] = {}
    for r in rows:
        folder = r.get('study_folder') or ''
        if not folder or folder in ages:
            continue
        a = parse_age(r.get('patient_age') or '', r.get('patient_birth_date') or '', r.get('study_date') or '')
        if a is not None:
            ages[folder] = a
    return ages, None


def load_priority(priority: Path) -> tuple[Optional[dict[str, tuple[str, list[str]]]], Optional[str],
                                           Optional[str]]:
    """study_folder → (priority 'yes' | 'no' | '', [reasons]) from a report_priority.py file, plus its
    rules_version. A folder listed twice keeps the strongest answer (yes > no > ''), so a duplicate can never
    hide a priority finding. (None, None, note) when unreadable or without the columns."""
    try:
        with open(priority, encoding='utf-8-sig', newline='') as fh:   # -sig: a csv re-saved by Excel has a BOM
            reader = csv.DictReader(fh)
            rows = list(reader)
            cols = set(reader.fieldnames or [])
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return None, None, f'--priority unreadable ({type(e).__name__}): priority_new omitted'
    if not {'study_folder', 'priority'} <= cols:
        return None, None, '--priority has no study_folder / priority column: priority_new omitted'
    rank = {'yes': 2, 'no': 1, '': 0}
    out: dict[str, tuple[str, list[str]]] = {}
    versions = set()
    for r in rows:
        folder = r.get('study_folder') or ''
        if not folder:
            continue
        p = (r.get('priority') or '').strip().lower()
        p = p if p in rank else ''
        reasons = [x.strip().lower() for x in (r.get('reasons') or '').split(';') if x.strip()]
        if folder not in out or rank[p] > rank[out[folder][0]]:
            out[folder] = (p, reasons)
        v = (r.get('rules_version') or '').strip()
        if v:
            versions.add(v)
    return out, (','.join(sorted(versions)) or None), None


def _auc_np(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """The same mid-rank AUC as auc(), vectorised for the bootstrap (labels: bool, True = positive)."""
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if not n1 or not n0:
        return None
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    mid = np.cumsum(counts) - (counts - 1) / 2.0
    return float((mid[inv][labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def age_matched_auc(pos: list[tuple[float, float]], neg: list[tuple[float, float]],
                    window: float = AGE_MATCH_WINDOW_YEARS) -> tuple[Optional[float], int]:
    """(AUC, n pairs) over abnormal / normal pairs of (score, age) whose ages differ by <= window years:
    how often the model ranks the abnormal study higher BETWEEN PATIENTS OF SIMILAR AGE (ties 1/2)."""
    wins = 0.0
    n = 0
    for sp, ap in pos:
        for sn, an in neg:
            if abs(ap - an) <= window:
                n += 1
                wins += 1.0 if sp > sn else 0.5 if sp == sn else 0.0
    return (wins / n if n else None), n


def auc_difference_ci(rows: list[tuple[str, float, float, bool]], resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> Optional[list[float]]:
    """95 % percentile interval of AUC(model) - AUC(age) over rows (patient, score, age, is_abnormal),
    resampling PATIENTS with replacement (all studies of a drawn patient come along: studies of one patient
    are not independent). Resamples without both classes are skipped. None with < 2 patients per class."""
    by_pid: dict[str, list[int]] = defaultdict(list)
    for i, (pid, _, _, _) in enumerate(rows):
        by_pid[pid].append(i)
    pids = sorted(by_pid)
    if sum(1 for p in pids if any(rows[i][3] for i in by_pid[p])) < 2 or \
            sum(1 for p in pids if any(not rows[i][3] for i in by_pid[p])) < 2:
        return None
    score = np.array([r[1] for r in rows], dtype=float)
    age = np.array([r[2] for r in rows], dtype=float)
    lab = np.array([r[3] for r in rows], dtype=bool)
    groups = [np.array(by_pid[p], dtype=int) for p in pids]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        a_m, a_a = _auc_np(score[idx], lab[idx]), _auc_np(age[idx], lab[idx])
        if a_m is not None and a_a is not None:
            diffs.append(a_m - a_a)
    if not diffs:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [float(lo), float(hi)]


def _median(v: list[float]) -> Optional[float]:
    return float(np.median(v)) if v else None


def age_block(studies: esl.StudyProbs, table: dict[str, dict], ages: dict[str, float],
              abnormal_class: str = 'abnormal') -> dict:
    """The 'age_new' block over one group (the never-seen studies) at the shipped protocol: does the model
    beat age alone, and does it still rank abnormal above normal between patients of similar age?"""
    decide = esl.gate_fn(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD)
    rows: list[tuple[str, float, float, bool, bool]] = []    # (patient, score, age, is_abnormal, flagged)
    n_without = 0
    for (key, cls), probs in studies.items():
        row = table.get(key)
        a = ages.get(row['study_folder']) if row is not None else None
        if a is None:
            n_without += 1
            continue
        rows.append((esl.patient_id_of(key), study_mean(probs), a, cls == abnormal_class, decide(probs)))
    pos = [r for r in rows if r[3]]
    neg = [r for r in rows if not r[3]]
    am, n_pairs = age_matched_auc([(r[1], r[2]) for r in pos], [(r[1], r[2]) for r in neg])

    model_sens = _ratio(sum(1 for r in pos if r[4]), len(pos))
    rule = None
    if model_sens is not None:
        best = 0
        for t in range(0, int(max(r[2] for r in rows)) + 2):
            if sum(1 for r in pos if r[2] >= t) / len(pos) >= model_sens:
                best = t
        rule = {'age_from': best,
                'sensitivity': sum(1 for r in pos if r[2] >= best) / len(pos),
                'specificity': _ratio(sum(1 for r in neg if r[2] < best), len(neg))}

    bands = []
    for name, lo, hi in AGE_BANDS:
        bp = [r for r in pos if lo <= r[2] < hi]
        bn = [r for r in neg if lo <= r[2] < hi]
        abn_flagged = sum(1 for r in bp if r[4])
        norm_flagged = sum(1 for r in bn if r[4])
        bands.append({'band': name, 'n_abnormal': len(bp), 'abnormal_flagged': abn_flagged,
                      'n_normal': len(bn), 'normal_flagged': norm_flagged,
                      'sensitivity': _ratio(abn_flagged, len(bp)),
                      'specificity': _ratio(len(bn) - norm_flagged, len(bn))})
    return {'n_with_age': len(rows), 'n_without_age': n_without,
            'median_age': {'normal': _median([r[2] for r in neg]), 'abnormal': _median([r[2] for r in pos])},
            'auc_model': auc([r[1] for r in pos], [r[1] for r in neg]),
            'auc_age_alone': auc([r[2] for r in pos], [r[2] for r in neg]),
            'auc_difference_ci95': auc_difference_ci([r[:4] for r in rows]),
            'auc_age_matched': am, 'age_matched_pairs': n_pairs,
            'age_match_window_years': int(AGE_MATCH_WINDOW_YEARS),
            'age_rule_at_model_sensitivity': rule,
            'bands': bands,
            'normal_reports_aged_60_plus': sum(1 for r in neg if r[2] >= 60.0)}


def priority_block(studies: esl.StudyProbs, table: dict[str, dict], prio: dict[str, tuple[str, list[str]]],
                   rules_version: Optional[str], ages: Optional[dict[str, float]]) -> dict:
    """The 'priority_new' block: how the shipped flag treats studies whose report carries a priority
    finding ('yes') versus the rest ('no'). Unknown ('' or no row) studies are counted and left out.
    by_reason counts the 'yes' studies under each rule family that fired."""
    decide = esl.gate_fn(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD)
    yes: list[tuple[float, Optional[float], bool]] = []
    no: list[tuple[float, Optional[float], bool]] = []
    n_unknown = 0
    by_reason: dict[str, dict] = {}
    for (key, _), probs in studies.items():
        row = table.get(key)
        p, reasons = prio.get(row['study_folder'], ('', [])) if row is not None else ('', [])
        if p not in ('yes', 'no'):
            n_unknown += 1
            continue
        a = ages.get(row['study_folder']) if ages is not None else None
        flagged = decide(probs)
        (yes if p == 'yes' else no).append((study_mean(probs), a, flagged))
        if p == 'yes':
            for reason in dict.fromkeys(reasons):
                e = by_reason.setdefault(reason, {'n': 0, 'flagged': 0})
                e['n'] += 1
                e['flagged'] += int(flagged)
    age_auc = None
    if ages is not None:
        age_auc = auc([a for _, a, _ in yes if a is not None], [a for _, a, _ in no if a is not None])
    return {'rules_version': rules_version,
            'n_priority': len(yes), 'priority_flagged': sum(1 for r in yes if r[2]),
            'n_other': len(no), 'other_flagged': sum(1 for r in no if r[2]),
            'n_unknown': n_unknown,
            'auc_model': auc([s for s, _, _ in yes], [s for s, _, _ in no]),
            'auc_age_alone': age_auc,
            'by_reason': dict(sorted(by_reason.items(), key=lambda kv: (-kv[1]['n'], kv[0])))}


# ------------------------------------------------------------------ evaluation

def verify_model_pin(model_dir: Path) -> bool:
    """True when <model>/MANIFEST.json exists and every file it pins matches (eval_study_level.
    check_manifest — the same check the release gate runs); False when the directory has no
    MANIFEST.json. Weights that do NOT match their pin are unusable: the result would be quoted
    for an identity that was not the one scored."""
    manifest = Path(model_dir) / 'MANIFEST.json'
    if not manifest.is_file():
        return False
    try:
        esl.check_manifest(Path(model_dir), manifest)
    except SystemExit:
        raise UnusableInput(f'--model {model_dir}: the files do not match MANIFEST.json — these are not '
                            'the weights of the pinned release') from None
    return True


def evaluate(model_dir, set_dir, seen=(), holdout=None, matched=None, device: str = 'cpu',
             abnormal_class: str = 'abnormal', predictor=None, index=None, priority=None) -> dict:
    """The full prospective evaluation → the JSON payload. Raises UnusableInput.
    index / priority (optional) add the 'age_new' / 'priority_new' blocks; an unusable file there is a
    note and a null block (informational, never a reason to refuse the evaluation)."""
    model_dir, set_dir = Path(model_dir), Path(set_dir)
    notes: list[str] = []
    if not set_dir.is_dir():
        raise UnusableInput(f'--set {set_dir}: not a directory')
    if not any((set_dir / sp).is_dir() and any((set_dir / sp).glob('*/*.png')) for sp in SET_SPLITS):
        raise UnusableInput(f'--set {set_dir}: no PNG files under train/<cls>/ or val/<cls>/')
    holdout_ids: set[str] = set()
    if holdout is not None:
        if not Path(holdout).is_file():
            raise UnusableInput(f'--holdout {holdout}: file not found')
        holdout_ids = esl.load_ids(Path(holdout))
    if matched is not None and not Path(matched).is_file():
        raise UnusableInput(f'--matched {matched}: file not found')
    if index is not None and not Path(index).is_file():
        raise UnusableInput(f'--index {index}: file not found')
    if priority is not None and not Path(priority).is_file():
        raise UnusableInput(f'--priority {priority}: file not found')
    seen_train, seen_val, seen_batch, seen_notes = load_seen_patients(seen)
    notes += seen_notes
    if not seen:
        notes.append("no --seen set given: 'new' is NOT verified against the model's training data")

    manifest_verified = False
    if predictor is None:
        if not model_dir.is_dir():
            raise UnusableInput(f'--model {model_dir}: not a directory')
        # the numbers are quoted for ONE identity: the weights on disk must be the pinned release
        manifest_verified = verify_model_pin(model_dir)
        if not manifest_verified:
            notes.append('the model directory has no MANIFEST.json: the identity is the hash of the weights '
                         'file, not a verified release pin')
        try:
            predictor = esl.make_predictor(model_dir, device)
        except (Exception, SystemExit) as e:
            raise UnusableInput(f'--model {model_dir}: cannot be loaded ({type(e).__name__})') from None
    studies, n_png = collect_set(model_dir, set_dir, predictor, abnormal_class)
    if not studies:
        raise UnusableInput(f'--set {set_dir}: no PNG files were scored')

    groups_of = assign_groups({esl.patient_id_of(k) for k, _ in studies}, seen_train, seen_val, holdout_ids,
                              seen_batch=seen_batch)
    by_group: dict[str, esl.StudyProbs] = {'all': dict(studies), 'new': {}, 'seen_train': {}, 'holdout': {}}
    for sk, probs in studies.items():
        by_group[groups_of[esl.patient_id_of(sk[0])]][sk] = probs

    table, note = load_study_table(set_dir)
    if note:
        notes.append(note)
    if table is not None:
        scored_keys = {k for k, _ in studies}
        n_no_png = sum(1 for k in table if k not in scored_keys)
        if n_no_png:
            notes.append(f'{n_no_png} studies.csv row(s) have no PNG files (not built: unreadable, no usable '
                         'axial series, or excluded by the modality rule): not scored')
        n_cls_mismatch = sum(1 for (k, c) in studies if k in table and table[k]['cls'] and table[k]['cls'] != c)
        if n_cls_mismatch:
            notes.append(f'{n_cls_mismatch} study(ies) sit in a class folder that differs from studies.csv cls '
                         '(the folder is used)')
    if matched is not None:
        if table is None:
            notes.append('new_same_date omitted: studies.csv is missing or unusable')
        else:
            keys, note = same_date_keys({k for k, _ in by_group['new']}, table, Path(matched))
            if note:
                notes.append(note)
            else:
                by_group['new_same_date'] = {sk: p for sk, p in by_group['new'].items() if sk[0] in keys}

    groups = {g: group_metrics(s, abnormal_class, g) for g, s in by_group.items()}
    findings = None
    if table is not None:
        findings, n_unmapped = findings_table(by_group['new'], table, abnormal_class)
        if n_unmapped:
            notes.append(f"{n_unmapped} 'new' study(ies) have no studies.csv row: left out of findings_new")
    ages = None
    age_new = None
    if index is None:
        notes.append('no --index given: age_new omitted (the model is not compared with age alone)')
    else:
        ages, note = load_index_ages(Path(index))
        if note:
            notes.append(note)
        elif table is None:
            ages = None
            notes.append('age_new omitted: studies.csv is missing or unusable (no study key → study folder map)')
        else:
            age_new = age_block(by_group['new'], table, ages, abnormal_class)
            if age_new['n_without_age']:
                notes.append(f"{age_new['n_without_age']} 'new' study(ies) have no usable age in --index: "
                             'left out of age_new')
    priority_new = None
    if priority is None:
        notes.append('no --priority given: priority_new omitted')
    else:
        prio, rules_version, note = load_priority(Path(priority))
        if note:
            notes.append(note)
        elif table is None:
            notes.append('priority_new omitted: studies.csv is missing or unusable (no study key → study folder map)')
        else:
            priority_new = priority_block(by_group['new'], table, prio, rules_version, ages)
    if not by_group['new']:
        notes.append("group 'new' is empty: every scored patient was seen by this model or is in the holdout")
    notes.append('threshold_sweep_new is descriptive only: the shipped decision threshold stays '
                 f'{esl.PROTOCOL_THRESHOLD:.2f}')
    notes.append('reference labels are derived from the routine radiology report, not a consensus read')

    return {'kind': 'prospective_eval', 'schema_version': SCHEMA_VERSION,
            'evaluated_on': date.today().isoformat(),
            'model': {'path': str(model_dir), 'identity': esl.model_identity(model_dir),
                      'manifest_verified': manifest_verified},
            'protocol': esl.protocol_block(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD),
            'inputs': {'set': str(set_dir), 'seen': [str(s) for s in seen],
                       'holdout': str(holdout) if holdout is not None else None,
                       'matched': str(matched) if matched is not None else None,
                       'index': str(index) if index is not None else None,
                       'priority': str(priority) if priority is not None else None,
                       'n_png': n_png, 'n_studies': len(studies)},
            'groups': groups,
            'threshold_sweep_new': threshold_sweep(by_group['new'], abnormal_class),
            'findings_new': findings,
            'age_new': age_new,
            'priority_new': priority_new,
            'notes': notes}


def write_json_atomic(out: Path, payload: dict) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(tmp, out)


# ------------------------------------------------------------------ report

def _f(v: Optional[float]) -> str:
    return '  —  ' if v is None else f'{v:.3f}'


def _age(v: Optional[float]) -> str:
    """An age in years for the printed summary: whole years are enough to read."""
    return '—' if v is None else f'{v:.0f}'


def _ci(ci) -> str:
    return '[ —  ]' if ci is None else f'[{ci[0]:.2f}–{ci[1]:.2f}]'


def print_summary(payload: dict) -> None:
    g = payload['groups']
    n = g['new']
    print(f"\n{'='*64}\n  {B}PROSPECTIVE EVALUATION{NC}  model {payload['model']['identity']}  "
          f"({payload['inputs']['n_studies']} studies, {payload['inputs']['n_png']} slices)\n{'='*64}")
    head = G if n['n_studies'] else Y
    print(f"  {head}NEW PATIENTS (never seen by this model):{NC} n={n['n_studies']} studies / {n['n_patients']} patients"
          f"  sensitivity {_f(n['sensitivity'])} {_ci(n['sensitivity_ci95'])}"
          f"  specificity {_f(n['specificity'])} {_ci(n['specificity_ci95'])}")
    print(f"\n  {'group':14s} {'n':>5s} {'abn/norm':>9s} {'sens':>6s} {'95% CI':>11s} {'spec':>6s} {'95% CI':>11s}"
          f" {'bal':>6s} {'auc':>6s} {'flag':>6s}   TP/FN/TN/FP")
    for name in GROUP_ORDER:
        if name not in g:
            continue
        m = g[name]
        print(f"  {name:14s} {m['n_studies']:5d} {str(m['n_abnormal']) + '/' + str(m['n_normal']):>9s}"
              f" {_f(m['sensitivity']):>6s} {_ci(m['sensitivity_ci95']):>11s} {_f(m['specificity']):>6s}"
              f" {_ci(m['specificity_ci95']):>11s} {_f(m['balanced_accuracy']):>6s} {_f(m['auc']):>6s}"
              f" {_f(m['flag_rate']):>6s}   {m['tp']}/{m['fn']}/{m['tn']}/{m['fp']}")
    if n['n_studies']:
        print(f"\n  {B}threshold sweep, new patients{NC} (descriptive only — shipped threshold "
              f"{esl.PROTOCOL_THRESHOLD:.2f})")
        for r in payload['threshold_sweep_new']:
            mark = '  ← shipped' if r['shipped'] else ''
            print(f"    @{r['threshold']:.2f}  sens {_f(r['sensitivity'])}  spec {_f(r['specificity'])}"
                  f"  flagged {_f(r['flagged_share'])}{mark}")
    if payload['findings_new']:
        print(f"\n  {B}per finding, new patients{NC}")
        for name, e in payload['findings_new'].items():
            if e['reference'] == 'abnormal':
                print(f"    {name:28s} detected        {e['detected']:4d} / {e['total']:<4d} ({e['rate']:.2f})")
            else:
                print(f"    {name:28s} wrongly flagged {e['wrongly_flagged']:4d} / {e['total']:<4d} ({e['rate']:.2f})")
    a = payload.get('age_new')
    if a:
        ci = a['auc_difference_ci95']
        ci_s = '[ —  ]' if ci is None else f'[{ci[0]:+.3f} to {ci[1]:+.3f}]'
        print(f"\n  {B}age (never-seen){NC}  {a['n_with_age']} studies with an age, {a['n_without_age']} without;"
              f"  median age normal {_age(a['median_age']['normal'])} / abnormal {_age(a['median_age']['abnormal'])}")
        print(f"    AUC model {_f(a['auc_model'])}   age alone {_f(a['auc_age_alone'])}"
              f"   model minus age 95% CI {ci_s}")
        print(f"    AUC between patients within {a['age_match_window_years']} years of age "
              f"{_f(a['auc_age_matched'])} ({a['age_matched_pairs']} pairs)")
        rule = a['age_rule_at_model_sensitivity']
        if rule:
            print(f"    the rule 'age >= {rule['age_from']}' alone: sensitivity {_f(rule['sensitivity'])}"
                  f"  specificity {_f(rule['specificity'])} (at the model's sensitivity)")
        for b in a['bands']:
            print(f"    {b['band']:9s} abnormal flagged {b['abnormal_flagged']:4d} / {b['n_abnormal']:<4d}"
                  f" normal flagged {b['normal_flagged']:4d} / {b['n_normal']:<4d}"
                  f" sens {_f(b['sensitivity'])}  spec {_f(b['specificity'])}")
        print(f"    normal reports aged 60+: {a['normal_reports_aged_60_plus']}")
    p = payload.get('priority_new')
    if p:
        print(f"\n  {B}priority findings (never-seen){NC}  rules {p['rules_version'] or '—'}")
        print(f"    priority flagged {p['priority_flagged']:4d} / {p['n_priority']:<4d}"
              f"  other flagged {p['other_flagged']:4d} / {p['n_other']:<4d}  unknown {p['n_unknown']}")
        print(f"    AUC priority vs other: model {_f(p['auc_model'])}  age alone {_f(p['auc_age_alone'])}")
        for reason, e in p['by_reason'].items():
            print(f"    {reason:28s} flagged {e['flagged']:4d} / {e['n']}")
    for note in payload['notes']:
        print(f"  {Y}note:{NC} {note}")


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Prospective evaluation of the shipped triage model on a new batch.')
    ap.add_argument('--model', default='models/brain_triage_finetuned')
    ap.add_argument('--set', required=True, help='build_training_set.py output of the new batch')
    ap.add_argument('--seen', action='append', default=[], metavar='DIR',
                    help='a set the shipped model was trained (train/, tune/) or validated (val/) on; a per-batch '
                         'set (studies.csv, no dataset_manifest.json) counts as seen as a whole; repeatable')
    ap.add_argument('--holdout', default=None, help='locked holdout patient ids, one per line')
    ap.add_argument('--matched', default=None, help='matched.csv of the batch (enables new_same_date)')
    ap.add_argument('--index', default=None,
                    help='<batch>_index.csv (patient_age / birth + study date): enables age_new')
    ap.add_argument('--priority', default=None,
                    help='<batch>_priority.csv from report_priority.py: enables priority_new')
    ap.add_argument('--out', required=True, help='prospective_eval.json path')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--abnormal-class', default='abnormal')
    args = ap.parse_args(argv)

    try:
        payload = evaluate(args.model, args.set, seen=args.seen, holdout=args.holdout, matched=args.matched,
                           device=args.device, abnormal_class=args.abnormal_class,
                           index=args.index, priority=args.priority)
    except UnusableInput as e:
        print(f"  {R}✗ {e}{NC}")
        return EXIT_UNUSABLE
    write_json_atomic(Path(args.out), payload)
    print_summary(payload)
    print(f"  saved → {args.out}\n")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
