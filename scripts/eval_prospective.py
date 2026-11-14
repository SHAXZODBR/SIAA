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

  WHY
    When a batch gets its reports, the first question is how the frozen,
    shipped model does on patients nobody had reports for when it was trained.
    That is the closest thing to a prospective validation this product has:
    it goes into the clinic documentation and it tells us whether a retrain is
    needed. It must run BEFORE the batch is used for training.

  HONEST SCOPE
    The model was frozen before these reports existed, so 'new' is a genuine
    out-of-sample test on the clinic's own scanners; but the reference labels
    are derived from the routine radiology report (parsed, one reader), not a
    consensus read, and the batch is whatever the clinic sent — not a
    pre-registered consecutive cohort.

  OUTPUT
    --out JSON (kind 'prospective_eval', schema_version 1), written atomically,
    plus a compact summary on stdout. Counts only: no patient ids and no study
    folders, neither in the JSON nor on stdout.

  USAGE
    python scripts/eval_prospective.py --model models/brain_triage_finetuned \
        --set data/sets/<batch> --seen data/local_finetune_bin \
        --seen data/train_release_<tag> --holdout data/holdout_patients_v1.txt \
        --matched data/<batch>_matched.csv \
        --out data/release/<tag>/prospective_eval.json --device mps

  EXIT CODES
    0  the evaluation ran (whatever the numbers: this stage is informational
       and never blocks a release)
    2  unusable inputs: model dir missing / not loadable / without the
       abnormal class / with files that do not match its MANIFEST.json, set
       without PNG files, a study key under both classes or in both splits,
       a --seen / --holdout / --matched path that is missing
================================================================================
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package

import eval_study_level as esl                                 # noqa: E402
from build_training_set import read_prefix_map                 # noqa: E402

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_OK = 0
EXIT_UNUSABLE = 2
SCHEMA_VERSION = 1

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
             abnormal_class: str = 'abnormal', predictor=None) -> dict:
    """The full prospective evaluation → the JSON payload. Raises UnusableInput."""
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
                       'n_png': n_png, 'n_studies': len(studies)},
            'groups': groups,
            'threshold_sweep_new': threshold_sweep(by_group['new'], abnormal_class),
            'findings_new': findings,
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
    ap.add_argument('--out', required=True, help='prospective_eval.json path')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--abnormal-class', default='abnormal')
    args = ap.parse_args(argv)

    try:
        payload = evaluate(args.model, args.set, seen=args.seen, holdout=args.holdout, matched=args.matched,
                           device=args.device, abnormal_class=args.abnormal_class)
    except UnusableInput as e:
        print(f"  {R}✗ {e}{NC}")
        return EXIT_UNUSABLE
    write_json_atomic(Path(args.out), payload)
    print_summary(payload)
    print(f"  saved → {args.out}\n")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
