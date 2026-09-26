#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — RELEASE PIPELINE ORCHESTRATOR  (stage 11 of the release contract)
================================================================================
  WHAT
    One resumable command that takes a new hospital batch (drive of DICOM
    studies + radiology reports) through every stage of
    docs/dev/release_pipeline_contract.md:

        index → parse → label → match → build → combine → train → gate
              → promote → bump → bundle

    Each stage is an existing repo script with a fixed CLI; this file only
    builds the commands, runs them one after another with a tee'd log, keeps a
    state file, and enforces the guard rails that must hold BEFORE training
    and BEFORE bundling. It never calls `git push` / `git tag` itself — the
    manual checklist printed after `bundle` tells the founder what to do next.

  WHY
    A retrain must be boring and repeatable: the same commands, the same
    locked held-out patients, the same gate floors, the same documentation
    bump. Everything intermediate lives under data/release/<tag>/ so a failed
    run can be resumed with one flag instead of re-typing eleven commands.

  USAGE (from sentinel/; PY=/Users/shakhzodbtr/venv/bin/python):
    $PY scripts/release_pipeline.py plan   --batch new1 --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
    $PY scripts/release_pipeline.py run    --batch new1 --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
    $PY scripts/release_pipeline.py run    --batch new1 --resume        # continue: tag from data/release/LATEST,
                                                                        # options from state.json
    $PY scripts/release_pipeline.py gate   --batch new1 --tag brain-triage-v2              # one stage only
    $PY scripts/release_pipeline.py status --batch new1

  STATE MODEL
    data/release/LATEST                 the tag of the release in progress. Without --tag the
                                        orchestrator continues LATEST while its state is not
                                        complete (bundle not done) for the same batch, else it
                                        picks the next unused brain-triage-v<N>.
    data/release/<tag>/state.json       options (every effective CLI option — a --resume
                                        without flags reuses them), stages → {status, started,
                                        finished, cmd, log, outputs, input_sha256s}, guards.
    data/release/<tag>/logs/<stage>.log tee'd stdout+stderr of every stage command
    data/release/<tag>/logs/guards.log  full output of every guard-rail check

    --resume skips a stage whose record is done, whose outputs exist and whose inputs
    (sha256 of files; for a legacy set without studies.csv, sha256 of its sorted file
    listing) are unchanged. promote MOVES the candidate dir into the release dir, so on
    success it records {'archived_candidate': <dest>} and marks train + gate 'archived':
    their candidate paths then resolve to the release dir and they stay done. More
    generally a stage whose outputs are gone counts as done when a LATER stage is done
    (its outputs were consumed), unless --from/--only names it. After promote,
    combine/train/gate/promote never run again under the same tag (the gate would compare
    the new candidate with itself; combine would rewrite the val set the promoted model's
    eval is pinned to) — start a new tag. A --resume after promote treats every stage up to
    promote as FROZEN (done, even when an upstream input changed since — a Finder .DS_Store,
    a rebuilt set): only bump and bundle are left to finish the release.
    train also fingerprints its --base-model and --epochs, so `--from train --resume` with a
    new --base-model (the retrain-from-the-public-ViT fallback) really retrains.
    The current model's training set (data/train_release_<prev>) is locked to train in the
    next combine; if that set is gone or was rebuilt, combine refuses to run without an
    explicit --train-lock (it never silently drops the lock).

  EXIT CODES
    0 ok · 1 usage or guard rail failed · 2 a stage command failed
================================================================================
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

REPO_DEFAULT = Path(__file__).resolve().parent.parent

STAGE_ORDER = ('index', 'parse', 'label', 'match', 'build', 'combine',
               'train', 'gate', 'promote', 'bump', 'bundle')
SUBCOMMANDS = ('plan', 'run', 'all', 'status') + STAGE_ORDER
CANDIDATE_STAGES = ('train', 'gate', 'promote')      # never re-run after promote under one tag
# combine writes the release set the promoted model's eval is pinned to: never rewritten after
# promote either. Every stage up to promote is frozen (done) on a --resume after promote.
NO_RERUN_AFTER_PROMOTE = ('combine',) + CANDIDATE_STAGES
FROZEN_AFTER_PROMOTE = ('index', 'parse', 'label', 'match', 'build', 'combine') + CANDIDATE_STAGES

EXIT_USAGE = 1
EXIT_STAGE_FAILED = 2

# eval_study_level.py exit codes (contract stage 7)
EVAL_GATE_FAILED = 2
EVAL_MANIFEST_FAILED = 3
EVAL_NOT_PROMOTED = 4

GATE_MIN_SENS = 0.85
GATE_MIN_SPEC = 0.40
MIN_NORMALS_DEFAULT = 300
MIN_TUNE_PATIENTS_WARN = 20          # a smaller tune/ makes checkpoint selection mostly noise
MIN_DISK_GB_DEFAULT = 20.0
GITHUB_MODELS_REPO = 'SHAXZODBR/SIAA-models'
BASE_MODEL = 'models/brain_triage_finetuned'        # the release dir: compare-with, promote dest, bump, bundle
PUBLIC_VIT_BASE = 'models/hf/google__vit-base-patch16-224'   # offline copy of google/vit-base-patch16-224
PUBLIC_VIT_HUB_ID = 'google/vit-base-patch16-224'
DEFAULT_HOLDOUT = 'data/holdout_patients_v1.txt'
LATEST_FILE = 'data/release/LATEST'
VENV_PYTHON = '/Users/shakhzodbtr/venv/bin/python'

DONE_STATUSES = ('done', 'archived')

# Every option that shapes the stage commands. state.json keeps the effective values; a
# --resume / --from / --only / <stage> invocation reuses them for flags not given on the CLI,
# and the printed resume line re-emits ALL of them.
OPTION_DEFAULTS: dict[str, object] = {
    'root': None, 'reports': None, 'extra_sets': None, 'holdout': DEFAULT_HOLDOUT,
    'train_lock': None, 'epochs': 12, 'device': 'mps', 'python': sys.executable,
    'base_model': BASE_MODEL, 'skip_benchmark': False, 'skip_tests': False, 'label_llm': None,
    'llm_labels': None, 'github': GITHUB_MODELS_REPO, 'local_bundle': False,
    'min_normals': MIN_NORMALS_DEFAULT, 'min_disk_gb': MIN_DISK_GB_DEFAULT,
}


# ------------------------------------------------------------------ config ----

@dataclass
class Config:
    repo: Path
    batch: str
    tag: str
    root: Optional[Path] = None
    reports: Optional[Path] = None
    extra_sets: list[str] = field(default_factory=list)
    holdout: str = DEFAULT_HOLDOUT
    train_lock: Optional[str] = None
    epochs: int = 12
    device: str = 'mps'
    python: str = sys.executable
    base_model: str = BASE_MODEL
    skip_benchmark: bool = False
    skip_tests: bool = False
    github: Optional[str] = GITHUB_MODELS_REPO
    label_llm: Optional[str] = None
    llm_labels: Optional[str] = None
    min_normals: int = MIN_NORMALS_DEFAULT
    min_disk_gb: float = MIN_DISK_GB_DEFAULT
    dry_run: bool = False
    resume: bool = False
    tag_source: str = '--tag'
    inherited: dict = field(default_factory=dict)     # options taken from state.json

    # derived paths (relative to repo, kept relative so `plan` is copy-pasteable)
    @property
    def index_csv(self) -> str: return f'data/{self.batch}_index.csv'
    @property
    def reports_csv(self) -> str: return f'data/{self.batch}_reports.csv'
    @property
    def labels_csv(self) -> str: return f'data/{self.batch}_labels.csv'
    @property
    def matched_csv(self) -> str: return f'data/{self.batch}_matched.csv'
    @property
    def set_dir(self) -> str: return f'data/sets/{self.batch}'
    @property
    def release_set(self) -> str: return f'data/train_release_{self.tag}'
    @property
    def candidate(self) -> str: return f'models/brain_triage_candidate_{self.tag}'
    @property
    def release_dir(self) -> str: return f'data/release/{self.tag}'
    @property
    def state_path(self) -> Path: return self.repo / self.release_dir / 'state.json'
    @property
    def logs_dir(self) -> Path: return self.repo / self.release_dir / 'logs'

    def rel(self, p: Path) -> str:
        """Path as it should appear in a command: relative when inside the repo."""
        try:
            return str(p.resolve().relative_to(self.repo.resolve()))
        except ValueError:
            return str(p)

    def options(self) -> dict:
        """The effective options, JSON-serialisable — stored in state.json."""
        return {
            'root': str(self.root) if self.root else None,
            'reports': str(self.reports) if self.reports else None,
            'extra_sets': list(self.extra_sets), 'holdout': self.holdout, 'train_lock': self.train_lock,
            'epochs': self.epochs, 'device': self.device, 'python': self.python,
            'base_model': self.base_model, 'skip_benchmark': self.skip_benchmark,
            'skip_tests': self.skip_tests, 'label_llm': self.label_llm, 'llm_labels': self.llm_labels,
            'github': self.github or GITHUB_MODELS_REPO, 'local_bundle': self.github is None,
            'min_normals': self.min_normals, 'min_disk_gb': self.min_disk_gb,
        }


@dataclass
class Stage:
    name: str
    title: str
    cmds: list[list[str]]          # argv lists, run in order, cwd = repo
    inputs: list[str]              # fingerprinted into state (files: sha256; 'ls:<dir>': listing sha256)
    outputs: list[str]             # must exist after a successful run
    good: str = ''                 # one line: what "good" looks like


# ---------------------------------------------------------------- helpers -----

def default_tag(models_dir: Path) -> str:
    """brain-triage-v<N>: the current release is v(prev+1), the candidate v(prev+2)."""
    prev = [p for p in models_dir.glob('brain_triage_finetuned_prev_*') if p.is_dir()]
    current = 1 if (models_dir / 'brain_triage_finetuned').is_dir() else 0
    return f'brain-triage-v{len(prev) + current + 1}'


def next_new_tag(repo: Path) -> str:
    """default_tag(), skipping any tag that already has a data/release/<tag>/ dir (an
    abandoned attempt keeps its number, so tags — and GitHub release names — stay unique)."""
    m = re.fullmatch(r'brain-triage-v(\d+)', default_tag(repo / 'models'))
    n = int(m.group(1)) if m else 1
    while (repo / 'data' / 'release' / f'brain-triage-v{n}').exists():
        n += 1
    return f'brain-triage-v{n}'


def read_latest(repo: Path) -> Optional[str]:
    try:
        tag = (repo / LATEST_FILE).read_text(encoding='utf-8').strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return tag if re.fullmatch(r'[A-Za-z0-9._-]+', tag or '') else None


def write_latest(repo: Path, tag: str) -> None:
    p = repo / LATEST_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tag + '\n', encoding='utf-8')


def release_complete(state: dict) -> bool:
    return (state.get('stages', {}).get('bundle') or {}).get('status') == 'done'


def resolve_tag(repo: Path, batch: str, explicit: Optional[str]) -> tuple[str, str]:
    """(tag, where it came from). --tag wins; else LATEST while that release is unfinished
    for the same batch; else the next unused brain-triage-v<N>."""
    if explicit:
        return explicit, '--tag'
    latest = read_latest(repo)
    if latest:
        st = _read_json(repo / 'data' / 'release' / latest / 'state.json')
        same_batch = st.get('batch') in (None, batch)
        if same_batch and not release_complete(st):
            return latest, 'from LATEST'
        if not same_batch and not release_complete(st):
            return next_new_tag(repo), (f'new — LATEST {latest} is an unfinished release of batch '
                                        f'{st.get("batch")}; pass --tag {latest} to continue it')
    return next_new_tag(repo), 'new release'


def default_extra_sets(repo: Path, batch: str) -> list[str]:
    """Every already-built set except the current batch (+ the legacy batch-1 set), OLDEST
    first. Rehearsal artefacts (data/sets/rehearsal_*) are never combined into a release.
    Order matters: combine_training_sets.py lets the LAST set win a label conflict, and the
    current batch is appended after these (see build_stages)."""
    sets: list[str] = []
    legacy = repo / 'data' / 'local_finetune_bin'
    if legacy.is_dir():
        sets.append('data/local_finetune_bin')
    sets_root = repo / 'data' / 'sets'
    if sets_root.is_dir():
        for p in sorted(sets_root.iterdir()):
            if p.is_dir() and p.name != batch and not p.name.startswith(('.', 'rehearsal')):
                sets.append(f'data/sets/{p.name}')
    return sets


def _trained_on_set(repo: Path) -> Optional[tuple[str, Path]]:
    """(repo-relative path, absolute path) of the release set the CURRENT release model was
    trained on — its training_metrics.json checkpoint_selection.data = <release set>/tune —
    or None for the legacy v1 model (no checkpoint_selection)."""
    metrics = _read_json(repo / BASE_MODEL / 'training_metrics.json')
    data = (metrics.get('checkpoint_selection') or {}).get('data')
    if not data:
        return None
    set_dir = Path(data).parent
    abs_dir = set_dir if set_dir.is_absolute() else repo / set_dir
    try:
        rel = abs_dir.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        rel = str(abs_dir)
    return rel, abs_dir


def default_train_lock(repo: Path, release_set: str) -> Optional[str]:
    """The release set the CURRENT release model was trained on, so combine locks those
    patients to train (never val/tune) in the next release. None for the legacy v1 model
    (trained on data/local_finetune_bin, whose train/ is locked by combine anyway), when that
    set is the one being built now (resume after promote under the same tag), and when the
    set is gone or was rebuilt — train_lock_problem() then names the problem and the combine
    guard refuses to run WITHOUT a lock (the lock never fails open)."""
    found = _trained_on_set(repo)
    if found is None:
        return None
    rel, _abs = found
    if rel == release_set or train_lock_problem(repo, release_set):
        return None
    return rel


def train_lock_problem(repo: Path, release_set: str) -> Optional[str]:
    """Why the default train lock cannot be derived although the current release model was
    trained on a combined release set (None = fine, or not needed):
      - that set (its dataset_manifest.json) is gone;
      - that set was rebuilt after the model was evaluated on it: its val patient hash no
        longer equals the release eval's val_patients_sha256, so its train/ + tune/ are not
        what the model saw."""
    found = _trained_on_set(repo)
    if found is None:
        return None
    rel, abs_dir = found
    if rel == release_set:
        return None
    manifest = abs_dir / 'dataset_manifest.json'
    if not manifest.is_file():
        return (f'the current release model was trained on {rel} ({BASE_MODEL}/training_metrics.json) '
                f'but {rel}/dataset_manifest.json is missing — its training patients cannot be locked '
                f'to train. Restore that folder, or pass --train-lock <ids.txt> listing the patients it '
                f'was trained on')
    want = ((_read_json(manifest).get('splits') or {}).get('val') or {}).get('patients_sha256')
    have = read_eval(repo / BASE_MODEL).get('val_patients_sha256')
    if want and have and want != have:
        return (f'{rel} was rebuilt after the current release model was evaluated on it (val patient '
                f'hash {want[:12]}… ≠ the release eval\'s {have[:12]}…): its train/ + tune/ are not what '
                f'the model was trained on. Restore the original set, or pass --train-lock <ids.txt> '
                f'listing the patients it was trained on')
    return None


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def listing_sha256(d: Path) -> str:
    """sha256 of the sorted relative file listing of a (small, local) set dir — the change
    marker of a legacy training set that has no studies.csv. Hidden files and dirs (Finder's
    .DS_Store, ._* resource forks) are not part of the set and never count as a change."""
    names: list[str] = []
    for dirpath, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith('.')]
        for fn in files:
            if not fn.startswith('.'):
                names.append((Path(dirpath) / fn).relative_to(d).as_posix())
    return 'ls:' + hashlib.sha256('\n'.join(sorted(names)).encode('utf-8')).hexdigest()


def fingerprint(p: Path) -> str:
    """sha256 for files; a cheap 'dir:<n entries>' marker for directories
    (a drive with a million DICOMs is never walked here); 'missing' otherwise."""
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        try:
            return f'dir:{len(os.listdir(p))}'
        except OSError:
            return 'dir:?'
    return 'missing'


def set_input(repo: Path, set_dir: str) -> str:
    """Input spec of a training set for the combine stage: its studies.csv, or — for a legacy
    set without one — 'ls:<dir>' (fingerprinted by its file listing)."""
    return f'{set_dir}/studies.csv' if (repo / set_dir / 'studies.csv').is_file() or \
        not (repo / set_dir).is_dir() else f'ls:{set_dir}'


def free_gb(path: Path) -> float:
    probe = path if path.exists() else path.parent
    return shutil.disk_usage(probe).free / 1e9


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def app_version(repo: Path) -> str:
    try:
        return json.loads((repo / 'desktop-app' / 'package.json').read_text())['version']
    except Exception:
        return '<version>'


def next_minor_version(ver: str) -> str:
    """1.0.0 → 1.1.0 (a model-only release bumps the minor version so v<version> is a new tag)."""
    m = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(.*)', ver or '')
    return f'{m.group(1)}.{int(m.group(2)) + 1}.0' if m else '<next version>'


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y')


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def count_matched(matched_csv: Path) -> tuple[int, int, int]:
    """(matched brain studies, normal, abnormal) from data/<batch>_matched.csv.
    `abnormal` column (contract stage 3) wins; falls back to label == normal."""
    n = normal = abnormal = 0
    with open(matched_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not _truthy(row.get('is_brain', '')) or not _truthy(row.get('report_matched', '')):
                continue
            n += 1
            ab = (row.get('abnormal') or '').strip().lower()
            if ab in ('yes', 'no'):
                normal += ab == 'no'
                abnormal += ab == 'yes'
            else:
                is_norm = (row.get('label') or '').strip().lower() == 'normal'
                normal += is_norm
                abnormal += not is_norm
    return n, normal, abnormal


def _split_patients(split_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not split_dir.is_dir():
        return ids
    for cls in split_dir.iterdir():
        if cls.is_dir():
            for f in cls.iterdir():
                if f.suffix.lower() == '.png':
                    ids.add(f.name.split('_')[0])
    return ids


def check_leakage(release_set: Path, holdout: Path) -> list[str]:
    """Independent re-check of what combine_training_sets.py promises: no patient on two
    sides (train / val / tune), every locked holdout patient in val."""
    errors: list[str] = []
    if not (release_set / 'dataset_manifest.json').is_file():
        errors.append(f'{release_set}/dataset_manifest.json missing — run the combine stage')
        return errors
    train = _split_patients(release_set / 'train')
    val = _split_patients(release_set / 'val')
    tune = _split_patients(release_set / 'tune')
    if not train or not val:
        errors.append(f'{release_set}: train ({len(train)} patients) / val ({len(val)} patients) '
                      'must both be non-empty')
    overlap = train & val
    if overlap:
        errors.append(f'{len(overlap)} patients in BOTH train and val (leakage), e.g. '
                      f'{sorted(overlap)[:5]}')
    for name, other in (('train', train), ('val', val)):
        both = tune & other
        if both:
            errors.append(f'{len(both)} patients in BOTH tune and {name} (leakage), e.g. {sorted(both)[:5]}')
    if holdout.is_file():
        ids = {ln.strip() for ln in holdout.read_text().splitlines() if ln.strip()}
        missing = ids - val
        if missing:
            errors.append(f'{len(missing)} locked holdout patients missing from val, e.g. '
                          f'{sorted(missing)[:5]}')
    else:
        errors.append(f'holdout file {holdout} not found')
    return errors


def promote_verdict(repo: Path, ev: dict, holdout: str = DEFAULT_HOLDOUT) -> tuple[list[str], list[str]]:
    """promote_model.eval_verdict() for the release model under `repo` (hard, soft reasons).
    The eval's relative `data` path is resolved against `repo`, not the cwd."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
    from promote_model import eval_verdict                         # light: no torch at import
    ev = dict(ev)
    data = ev.get('data')
    if data and not Path(str(data)).is_absolute():
        ev['data'] = str(repo / str(data))
    manifest = _read_json(repo / BASE_MODEL / 'MANIFEST.json')
    return eval_verdict(ev, manifest or None, repo / holdout)


def read_eval(model_dir: Path) -> dict:
    p = model_dir / 'study_level_eval.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ------------------------------------------------------------------ stages ----

def build_stages(cfg: Config) -> list[Stage]:
    py = cfg.python
    root = cfg.rel(cfg.root) if cfg.root else '<drive-root>'
    reports = cfg.rel(cfg.reports) if cfg.reports else '<reports-folder-or-file>'
    manifest = f'{cfg.release_set}/dataset_manifest.json'

    label_cmds = [[py, 'scripts/label_reports.py', '--reports', cfg.reports_csv, '--out', cfg.labels_csv]]
    label_inputs = [cfg.reports_csv]
    if cfg.llm_labels:
        # imported LLM labels (training/relabel_workflow.js) are merged on EVERY label run — a
        # re-run of the stage never silently regenerates rule-only labels
        label_cmds[0] += ['--import-llm-labels', cfg.llm_labels]
        label_inputs.append(cfg.llm_labels)
    if cfg.label_llm:
        label_cmds[0] += ['--llm', cfg.label_llm]
    if not cfg.skip_benchmark:
        label_cmds.append([py, 'scripts/label_reports.py', '--reports', cfg.reports_csv,
                           '--out', f'{cfg.release_dir}/label_benchmark.csv', '--benchmark'])

    # the current batch goes LAST: combine_training_sets.py resolves a patient labelled in two
    # sets with the class from the LAST set on the command line (the newest labels win)
    combine_cmd = [py, 'scripts/combine_training_sets.py', '--sets', *cfg.extra_sets, cfg.set_dir,
                   '--holdout', cfg.holdout, '--out', cfg.release_set, '--val-frac', '0.2', '--seed', '42']
    if cfg.train_lock:
        combine_cmd[-4:-4] = ['--train-lock', cfg.train_lock]
    # combine_training_sets.py refuses a non-empty --out; on a re-run of the stage (--only/--from
    # combine, or a failed first attempt) the release set already exists, so rebuild it (cheap: symlinks).
    release_set_dir = cfg.repo / cfg.release_set
    if release_set_dir.is_dir() and any(release_set_dir.iterdir()):
        combine_cmd.append('--fresh')
    # a rebuilt set (new labels → new studies.csv), a changed legacy set, holdout or train-lock
    # all re-run combine (and, through dataset_manifest.json, train and gate)
    combine_inputs = [*(set_input(cfg.repo, s) for s in cfg.extra_sets), f'{cfg.set_dir}/studies.csv',
                      cfg.holdout, *([cfg.train_lock] if cfg.train_lock else [])]

    bundle_cmd = [py, 'packaging/publish_model_bundle.py', '--model', BASE_MODEL,
                  '--out', cfg.release_dir + '/']
    if cfg.github:
        bundle_cmd += ['--github', cfg.github, '--tag', cfg.tag]

    return [
        Stage('index', 'Index the drive (one row per study)',
              [[py, 'scripts/index_new_studies.py', '--root', root, '--out', cfg.index_csv]],
              inputs=[root], outputs=[cfg.index_csv],
              good='n studies ≈ number of study folders, error column mostly empty, MR/CT split printed'),
        Stage('parse', 'Parse the hospital reports (PDF folder / xlsx / csv / docx)',
              [[py, 'scripts/parse_hospital_reports.py', '--src', reports, '--out', cfg.reports_csv]],
              inputs=[reports], outputs=[cfg.reports_csv],
              good='one row per study SECTION, modality from the section header, no patient names'),
        Stage('label', 'Rule-based labels (+ imported LLM labels) + benchmark against the 894-patient reference',
              label_cmds, inputs=label_inputs,
              outputs=[cfg.labels_csv],
              good='agreement on abnormal ≥ 0.90; needs_review share small (< 15%)'),
        Stage('match', 'Join images ↔ reports (patient id + date > modality > only report)',
              [[py, 'scripts/match_images_to_reports.py', '--images', root, '--index', cfg.index_csv,
                '--labels', cfg.labels_csv, '--out', cfg.matched_csv]],
              inputs=[cfg.index_csv, cfg.labels_csv], outputs=[cfg.matched_csv],
              good='most brain studies matched; match_quality mostly "id+date"'),
        Stage('build', 'Export central slices to data/sets/<batch> (binary, resumable)',
              [[py, 'scripts/build_training_set.py', '--images', root, '--matched', cfg.matched_csv,
                '--out', cfg.set_dir, '--mode', 'binary', '--abnormal-col', 'abnormal', '--skip-needs-review']],
              inputs=[cfg.matched_csv], outputs=[cfg.set_dir, f'{cfg.set_dir}/studies.csv'],
              good='≈ 5 slices per study, studies.csv row count = matched brain studies minus needs_review'),
        Stage('combine', 'Merge all sets, patient-level split (train / val / tune), locked holdout → val',
              [combine_cmd], inputs=combine_inputs,
              outputs=[cfg.release_set, manifest],
              good='0 train/val/tune overlap, all 178 holdout patients in val, tune/ written, per-batch counts printed'),
        Stage('train', f'Fine-tune the candidate from {cfg.base_model}',
              [[py, 'training/finetune_brain_classifier.py', '--data', cfg.release_set,
                '--classes', 'abnormal,normal', '--base-model', cfg.base_model, '--balance',
                '--unfreeze-last-blocks', '3', '--epochs', str(cfg.epochs), '--device', cfg.device,
                '--select-split', 'tune', '--output', cfg.candidate]],
              # opt:* = option VALUES that shape the weights: a new --base-model / --epochs (the
              # retrain-from-the-public-ViT fallback) re-runs train on --resume
              inputs=[manifest, 'opt:base_model', 'opt:epochs'],
              outputs=[cfg.candidate, f'{cfg.candidate}/model.safetensors'],
              good='val loss falls then plateaus; checkpoint selected on tune/ (not on the gate set); '
                   'training_metrics.json written'),
        Stage('gate', 'Study-level gate + comparison with the current release',
              [[py, 'scripts/eval_study_level.py', '--model', cfg.candidate, '--data', f'{cfg.release_set}/val',
                '--write-manifest', '--min-sens', f'{GATE_MIN_SENS:.2f}', '--min-spec', f'{GATE_MIN_SPEC:.2f}',
                '--compare-with', BASE_MODEL, '--device', cfg.device]],
              inputs=[f'{cfg.candidate}/model.safetensors', manifest],
              outputs=[f'{cfg.candidate}/study_level_eval.json', f'{cfg.candidate}/MANIFEST.json'],
              good=f'sens ≥ {GATE_MIN_SENS}, spec ≥ {GATE_MIN_SPEC}, promotion.promote = true'),
        Stage('promote', 'Move the candidate into models/brain_triage_finetuned (old → _prev_<id>)',
              [[py, 'scripts/promote_model.py', '--candidate', cfg.candidate, '--dest', BASE_MODEL]],
              inputs=[f'{cfg.candidate}/study_level_eval.json'],
              outputs=[f'{BASE_MODEL}/study_level_eval.json', f'{BASE_MODEL}/MANIFEST.json'],
              good='new identity printed, registry loads the dir, old release kept as _prev_'),
        Stage('bump', 'Rewrite every pinned number / identity / date in code and docs',
              [[py, 'scripts/bump_model_release.py', '--model', BASE_MODEL]],
              inputs=[f'{BASE_MODEL}/study_level_eval.json'], outputs=[],
              good='list of rewritten files; `--check` clean afterwards'),
        Stage('bundle', 'Zip + split + sha256 the release, publish to the private models repo',
              [bundle_cmd], inputs=[f'{BASE_MODEL}/MANIFEST.json'],
              outputs=[f'{cfg.release_dir}/manifest.json'],
              good='manifest asset API URL printed; restore with fetch_model_bundle.py verified'),
    ]


def stage_names(stages: list[Stage]) -> list[str]:
    return [s.name for s in stages]


# ------------------------------------------------------------------ state -----

def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            print(f'{Y}⚠ state file {path} unreadable — starting fresh{NC}')
    return {'stages': {}, 'guards': {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)


def archived_candidate(state: dict) -> Optional[str]:
    """Where promote moved the candidate dir (None before a successful promote)."""
    rec = state.get('stages', {}).get('promote') or {}
    return rec.get('archived_candidate') if rec.get('status') == 'done' else None


def resolve_path(cfg: Config, state: dict, p: str) -> str:
    """A candidate path after promote moved the candidate: models/brain_triage_candidate_<tag>/x
    → <archived dest>/x. Everything else unchanged."""
    dest = archived_candidate(state)
    if dest and (p == cfg.candidate or p.startswith(cfg.candidate + '/')):
        return dest + p[len(cfg.candidate):]
    return p


def fingerprint_input(cfg: Config, state: dict, spec: str) -> str:
    if spec.startswith('opt:'):
        return 'opt:' + str(getattr(cfg, spec[4:]))
    if spec.startswith('ls:'):
        d = cfg.repo / spec[3:]
        return listing_sha256(d) if d.is_dir() else 'missing'
    return fingerprint(cfg.repo / resolve_path(cfg, state, spec))


def input_fingerprints(cfg: Config, stage: Stage, state: dict) -> dict[str, str]:
    return {i: fingerprint_input(cfg, state, i) for i in stage.inputs}


# ------------------------------------------------------------ run command -----

def stage_env() -> dict[str, str]:
    """Environment of every stage / guard command.
    PYTHONUNBUFFERED: a child python block-buffers stdout when it is a pipe, so a 5-hour train
    stage would show (and log) nothing for minutes at a time. LOGURU_LEVEL=INFO: the repo's
    loguru logging stays readable in the tee'd logs (no DEBUG flood, no hidden INFO)."""
    return {**os.environ, 'PYTHONUNBUFFERED': '1', 'LOGURU_LEVEL': 'INFO'}


def run_command(argv: list[str], log_path: Path, cwd: Path, dry_run: bool = False) -> int:
    """Run one command, tee stdout+stderr to the terminal and to log_path."""
    print(f'{B}$ {shlex.join(argv)}{NC}')
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'\n===== {now_iso()}  {shlex.join(argv)}\n')
        log.flush()
        proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace', bufsize=1, env=stage_env())
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
        log.write(f'===== exit {rc}\n')
    return rc


def args_line(cfg: Config) -> str:
    """Every effective option, so a pasted resume line reproduces the run exactly."""
    o = cfg.options()
    parts = ['--batch', cfg.batch, '--tag', cfg.tag]
    if cfg.repo.resolve() != REPO_DEFAULT.resolve():
        parts += ['--repo', str(cfg.repo)]
    if o['root']:
        parts += ['--root', o['root']]
    if o['reports']:
        parts += ['--reports', o['reports']]
    parts += ['--extra-sets', *o['extra_sets']]
    parts += ['--holdout', o['holdout']]
    if o['train_lock']:
        parts += ['--train-lock', o['train_lock']]
    parts += ['--epochs', str(o['epochs']), '--device', o['device'], '--python', o['python'],
              '--base-model', o['base_model']]
    if o['label_llm']:
        parts += ['--label-llm', o['label_llm']]
    if o['llm_labels']:
        parts += ['--llm-labels', o['llm_labels']]
    parts += ['--skip-benchmark' if o['skip_benchmark'] else '--no-skip-benchmark',
              '--skip-tests' if o['skip_tests'] else '--no-skip-tests']
    parts += ['--local-bundle'] if o['local_bundle'] else ['--github', o['github']]
    parts += ['--min-normals', str(o['min_normals']), '--min-disk-gb', f'{o["min_disk_gb"]:g}']
    return shlex.join(parts)


def _resume_hint(cfg: Config) -> str:
    return f'{shlex.quote(cfg.python)} scripts/release_pipeline.py run {args_line(cfg)} --resume'


# ------------------------------------------------------------------ guards ----

def guard_pre_train(cfg: Config, state: dict) -> tuple[list[str], list[str]]:
    """Guard rails that must hold before the (hours-long) train stage."""
    errors: list[str] = []
    warnings: list[str] = []
    repo = cfg.repo

    # 1. drive mounted
    if cfg.root is None:
        warnings.append('no --root given — drive mount not checked')
    elif not cfg.root.is_dir():
        errors.append(f'drive not mounted: {cfg.root}')

    # 2. label benchmark
    bench = state.get('guards', {}).get('label_benchmark')
    if cfg.skip_benchmark:
        warnings.append('label benchmark SKIPPED (--skip-benchmark)')
    elif bench != 'passed':
        errors.append('label benchmark has not passed (run the label stage, or --skip-benchmark)')

    # 3. matched counts
    matched = repo / cfg.matched_csv
    if matched.is_file():
        n, normal, abnormal = count_matched(matched)
        print(f'  matched brain studies: {G}{n}{NC}  (normal {normal} / abnormal {abnormal})')
        state.setdefault('guards', {})['matched_counts'] = {'matched': n, 'normal': normal, 'abnormal': abnormal}
        if normal < cfg.min_normals:
            warnings.append(f'only {normal} normal studies matched in this batch (< {cfg.min_normals}): '
                            'specificity will not improve much — ask the hospital for more normals')
    else:
        errors.append(f'{cfg.matched_csv} missing — run the match stage')

    # 4. leakage / manifest / checkpoint-selection split
    errors += check_leakage(repo / cfg.release_set, repo / cfg.holdout)
    if (repo / cfg.release_set / 'dataset_manifest.json').is_file() and \
            not _split_patients(repo / cfg.release_set / 'tune'):
        # the train stage passes --select-split tune: selecting the checkpoint on val (= the gate
        # set) would make the gate numbers optimistic, so a missing tune/ stops the release here
        errors.append(f'{cfg.release_set}/tune/ is missing or empty: the trainer selects its checkpoint on '
                      'tune/ (never on the val gate set) — re-run combine with --tune-frac > 0')
    tune_dir = repo / cfg.release_set / 'tune'
    if _split_patients(tune_dir):
        n_tune = len(_split_patients(tune_dir))
        tune_classes = {c.name for c in tune_dir.iterdir() if c.is_dir() and any(c.glob('*.png'))}
        if n_tune < MIN_TUNE_PATIENTS_WARN or len(tune_classes) < 2:
            warnings.append(f'tune/ has only {n_tune} patients ({", ".join(sorted(tune_classes))}): the epoch '
                            f'the trainer keeps is chosen on very few studies (≥ {MIN_TUNE_PATIENTS_WARN} '
                            'patients of both classes recommended — combine --min-tune-per-class)')

    # 5. disk
    gb = free_gb(repo)
    print(f'  free disk: {gb:.1f} GB')
    if gb < cfg.min_disk_gb:
        errors.append(f'only {gb:.1f} GB free (need ≥ {cfg.min_disk_gb:.0f} GB for slices + checkpoints)')

    return errors, warnings


def guard_pre_combine(cfg: Config, state: dict) -> tuple[list[str], list[str]]:
    """The current model's training patients must be locked out of val/tune (fail closed)."""
    errors: list[str] = []
    if not cfg.train_lock:
        problem = train_lock_problem(cfg.repo, cfg.release_set)
        if problem:
            errors.append(problem)
    elif not (cfg.repo / cfg.train_lock).exists():
        errors.append(f'--train-lock {cfg.train_lock} not found')
    return errors, []


def guard_pre_bundle(cfg: Config, state: dict) -> tuple[list[str], list[str]]:
    """gate passed + promotion true + bump --check clean + full pytest green."""
    errors: list[str] = []
    warnings: list[str] = []
    repo = cfg.repo
    ev = read_eval(repo / BASE_MODEL)
    gate = ev.get('gate') or {}
    if not gate.get('passed'):
        errors.append(f'{BASE_MODEL}/study_level_eval.json: gate.passed is not true '
                      '(promote stage not run, or the candidate failed the gate)')
    promo = ev.get('promotion')
    if promo is None:
        warnings.append('study_level_eval.json has no "promotion" block (gate run without --compare-with)')
    elif not promo.get('promote'):
        errors.append(f'promotion.promote is false: {promo.get("reason", "")}')
    # the same release rules promote_model.py enforces (floors, production protocol, release
    # val set / holdout coverage, eval pinned to these weights) — never bundle a model that
    # promote would have refused
    if ev:
        hard, _soft = promote_verdict(repo, ev, cfg.holdout)
        for h in hard:
            if not h.startswith('gate.passed is'):          # already reported above
                errors.append(f'{BASE_MODEL}/study_level_eval.json: {h}')

    logs = cfg.logs_dir
    rc = run_command([cfg.python, 'scripts/bump_model_release.py', '--check'],
                     logs / 'guard_bump_check.log', repo, cfg.dry_run)
    if rc != 0:
        errors.append(f'bump_model_release.py --check is not clean (exit {rc}) — run the bump stage')

    if cfg.skip_tests:
        warnings.append('full test suite SKIPPED (--skip-tests)')
    else:
        rc = run_command([cfg.python, '-m', 'pytest', '-q'], logs / 'guard_pytest.log', repo, cfg.dry_run)
        if rc != 0:
            errors.append(f'pytest -q failed (exit {rc}) — see {cfg.rel(logs / "guard_pytest.log")}')
        rc = run_command([cfg.python, '-m', 'pytest', '-q', 'docs/test_docs_consistency.py'],
                         logs / 'guard_pytest_docs.log', repo, cfg.dry_run)
        if rc != 0:
            errors.append(f'docs consistency tests failed (exit {rc}) — see '
                          f'{cfg.rel(logs / "guard_pytest_docs.log")}')
    return errors, warnings


GUARDS: dict[str, Callable[[Config, dict], tuple[list[str], list[str]]]] = {
    'combine': guard_pre_combine,
    'train': guard_pre_train,
    'bundle': guard_pre_bundle,
}
GUARD_TEXT = {
    'combine': ['the set the current model was trained on is present and unchanged (its patients are '
                'locked to train), or --train-lock is given'],
    'train': ['label benchmark passed (or --skip-benchmark)',
              f'matched brain studies + normal count (warn < {MIN_NORMALS_DEFAULT} normals)',
              'dataset_manifest.json present, no train/val/tune patient overlap, holdout ⊂ val, tune/ present',
              f'≥ {MIN_DISK_GB_DEFAULT:.0f} GB free disk', 'drive mounted'],
    'bundle': ['gate.passed true and promotion.promote true in the promoted model',
               'promote_model.eval_verdict: release floors, production protocol, release val set, eval pinned '
               'to these weights',
               'bump_model_release.py --check clean',
               'pytest -q and docs/test_docs_consistency.py green'],
}


class _Tee:
    """stdout that also appends to a log file (colour codes stripped in the log)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.log.write(ANSI_RE.sub('', s))
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def run_guards(cfg: Config, stage_name: str, state: dict) -> bool:
    """Run the guard rails before `stage_name`; full output → logs/guards.log, verdict →
    state.json guards.pre_<stage>."""
    fn = GUARDS.get(stage_name)
    if fn is None:
        return True
    print(f'\n{B}── guard rails before {stage_name} ──{NC}')
    if cfg.dry_run:
        for t in GUARD_TEXT[stage_name]:
            print(f'  # would check: {t}')
        return True
    log_path = cfg.logs_dir / 'guards.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'\n===== {now_iso()}  guard rails before {stage_name}  (tag {cfg.tag}, batch {cfg.batch})\n')
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            errors, warnings = fn(cfg, state)
            for w in warnings:
                print(f'{Y}  ⚠ {w}{NC}')
            for e in errors:
                print(f'{R}  ✗ {e}{NC}')
            if errors:
                print(f'{R}✗ guard rails before {stage_name} failed — nothing was run{NC}')
            else:
                print(f'{G}✓ guard rails before {stage_name} passed{NC}')
    state.setdefault('guards', {})[f'pre_{stage_name}'] = {
        'at': now_iso(), 'passed': not errors, 'errors': errors, 'warnings': warnings,
        'log': cfg.rel(log_path)}
    return not errors


# ----------------------------------------------------------------- running ----

def stage_is_done(cfg: Config, stage: Stage, state: dict,
                  explicit: frozenset[str] = frozenset()) -> tuple[bool, str]:
    """Done = recorded done/archived, outputs present, inputs unchanged.

    Outputs that are gone do not force a re-run when a LATER stage is done (they were
    consumed — promote moves the candidate dir) or the stage is 'archived', unless
    --from/--only names this stage. Changed inputs always force a re-run."""
    stages = state.get('stages', {})
    rec = stages.get(stage.name)
    if not rec or rec.get('status') not in DONE_STATUSES:
        return False, ''
    if stage.name in FROZEN_AFTER_PROMOTE and archived_candidate(state):
        # the release was promoted from exactly these stage results: a changed upstream input
        # (a rebuilt set, a .DS_Store, a new holdout file) belongs to the NEXT release — never
        # rewrite the promoted model's release set or retrain under the same tag
        return True, ''
    missing = [o for o in stage.outputs if not (cfg.repo / resolve_path(cfg, state, o)).exists()]
    if missing:
        later = STAGE_ORDER[STAGE_ORDER.index(stage.name) + 1:] if stage.name in STAGE_ORDER else ()
        later_done = any((stages.get(n) or {}).get('status') in DONE_STATUSES for n in later)
        consumed = rec.get('status') == 'archived' or later_done
        if stage.name in explicit or not consumed:
            return False, f'outputs missing: {missing}'
    recorded = rec.get('input_sha256s')
    if recorded and inputs_changed(recorded, input_fingerprints(cfg, stage, state)):
        return False, 'inputs changed since the last run'
    return True, ''


def inputs_changed(recorded: dict, current: dict) -> bool:
    """A directory input (the drive, the reports folder: 'dir:<n>') that is now absent is
    'cannot tell', not 'changed' — unplugging the drive after build must not re-run index.
    An option input ('opt:…') that a state written before it existed did not record is not a change."""
    for k in set(recorded) | set(current):
        r, c = recorded.get(k), current.get(k)
        if isinstance(r, str) and r.startswith('dir:') and c == 'missing':
            continue
        if k.startswith('opt:') and r is None:
            continue
        if r != c:
            return True
    return False


def _archive_candidate(cfg: Config, state: dict, rec: dict) -> None:
    """After a successful promote: the candidate dir now lives at BASE_MODEL."""
    if (cfg.repo / cfg.candidate).exists() or not (cfg.repo / BASE_MODEL).exists():
        return
    rec['archived_candidate'] = BASE_MODEL
    for name in ('train', 'gate'):
        r = state.get('stages', {}).get(name)
        if r and r.get('status') == 'done':
            r.update(status='archived', archived_to=BASE_MODEL, archived_at=now_iso())


def _gate_verdict_message(cfg: Config, rc: int) -> list[str]:
    ev_path = f'{cfg.candidate}/study_level_eval.json'
    common = [f'the current release {BASE_MODEL} is untouched — promote was NOT run and will not run',
              f'read {ev_path} and {cfg.release_dir}/logs/gate.log (runbook section 7)',
              'options: keep the old model (nothing to do) · add more normal studies and retrain · '
              f'retrain from the public ViT base: {_retrain_hint(cfg)}']
    if rc == EVAL_GATE_FAILED:
        return [f'GATE FAILED: the candidate is below the floors (sens ≥ {GATE_MIN_SENS:.2f}, '
                f'spec ≥ {GATE_MIN_SPEC:.2f}) on {cfg.release_set}/val'] + common
    return ['NOT PROMOTED: the candidate passed the floors but is not better than the current release '
            '(balanced accuracy ≥ old − 0.01 and sensitivity ≥ old − 0.03 required)'] + common


def _retrain_hint(cfg: Config) -> str:
    base = PUBLIC_VIT_BASE if (cfg.repo / PUBLIC_VIT_BASE).is_dir() else PUBLIC_VIT_HUB_ID
    line = args_line(cfg).replace(f'--base-model {shlex.quote(cfg.base_model)}', f'--base-model {base}')
    # no --resume: train must run again even though its record is 'done' (train also
    # fingerprints --base-model, so a --resume with the new base would re-run it as well)
    return f'{shlex.quote(cfg.python)} scripts/release_pipeline.py run {line} --from train'


def run_stage(cfg: Config, stage: Stage, state: dict) -> int:
    print(f'\n{B}═══ stage {stage.name}: {stage.title} ═══{NC}')
    log = cfg.logs_dir / f'{stage.name}.log'
    rec = {
        'status': 'running', 'started': now_iso(), 'finished': None,
        'cmd': ' && '.join(shlex.join(c) for c in stage.cmds),
        'cmds': [shlex.join(c) for c in stage.cmds],
        'log': cfg.rel(log), 'outputs': list(stage.outputs), 'inputs': list(stage.inputs),
        'input_sha256s': {} if cfg.dry_run else input_fingerprints(cfg, stage, state),
    }
    state.setdefault('stages', {})[stage.name] = rec
    if not cfg.dry_run:
        save_state(cfg.state_path, state)

    for k, argv in enumerate(stage.cmds):
        rc = run_command(argv, log, cfg.repo, cfg.dry_run)
        if stage.name == 'label' and k == 1 and not cfg.dry_run:
            state.setdefault('guards', {})['label_benchmark'] = 'passed' if rc == 0 else 'failed'
        if rc != 0:
            rec.update(status='failed', finished=now_iso(), returncode=rc, failed_cmd=shlex.join(argv))
            if stage.name == 'gate' and rc in (EVAL_GATE_FAILED, EVAL_NOT_PROMOTED):
                rec['verdict'] = 'gate_failed' if rc == EVAL_GATE_FAILED else 'not_promoted'
            if not cfg.dry_run:
                save_state(cfg.state_path, state)
            print(f'\n{R}✗ stage {stage.name} FAILED (exit {rc}){NC}')
            if rec.get('verdict'):
                msg = _gate_verdict_message(cfg, rc)
                print(f'{R}  {msg[0]}{NC}')
                for m in msg[1:]:
                    print(f'  {m}')
            elif stage.name == 'gate' and rc == EVAL_MANIFEST_FAILED:
                print(f'{R}  MANIFEST check failed: the candidate weights do not match MANIFEST.json{NC}')
            print(f'  command : {shlex.join(argv)}')
            print(f'  log     : {cfg.rel(log)}')
            return rc

    if not cfg.dry_run:
        missing = [o for o in stage.outputs if not (cfg.repo / o).exists()]
        if missing:
            rec.update(status='failed', finished=now_iso(), returncode=0,
                       error=f'expected outputs missing: {missing}')
            save_state(cfg.state_path, state)
            print(f'{R}✗ stage {stage.name} finished but expected outputs are missing: {missing}{NC}')
            print(f'  log     : {cfg.rel(log)}')
            return 1
    rec.update(status='done', finished=now_iso(), returncode=0)
    if stage.name == 'promote' and not cfg.dry_run:
        _archive_candidate(cfg, state, rec)
    if not cfg.dry_run:
        save_state(cfg.state_path, state)
    print(f'{G}✓ stage {stage.name} done{NC}' + (f'  — good: {stage.good}' if stage.good else ''))
    return 0


def select_stages(stages: list[Stage], only: Optional[str], from_: Optional[str]) -> list[Stage]:
    names = stage_names(stages)
    if only:
        return [s for s in stages if s.name == only]
    if from_:
        return stages[names.index(from_):]
    return list(stages)


def manual_steps(cfg: Config) -> list[str]:
    ver = app_version(cfg.repo)
    nxt = next_minor_version(ver)
    return [
        'Copy the manifest asset API URL printed by the bundle stage into the GitHub environment '
        'secret MODEL_BUNDLE_URL (repo Settings → Environments → MODEL_BUNDLE_URL); keep '
        'MODEL_BUNDLE_TOKEN valid.',
        'Review the bump: git status && git diff — then git add ONLY the files the bump stage listed '
        f'(never data/ or models/) && git commit -m "release: {cfg.tag} — bump validated numbers"',
        f'EVERY release bumps "version" in desktop-app/package.json (currently {ver}; a model-only release '
        f'raises the minor version: {ver} → {nxt}) so the tag v<version> is new; commit it, then '
        f'git tag v{nxt} && git push origin main && git push origin v{nxt}   '
        '(pushing the tag starts the "Build installers" CI workflow; its "Verify bundle identity matches '
        'the code" step fails unless the committed bump and the MODEL_BUNDLE_URL secret name the same model)',
        'Download the installer artifact from the Actions run (or the draft GitHub release).',
        'Run docs/windows_laptop_test_en.md (RU: docs/windows_laptop_test_ru.md) on a Windows laptop, '
        'Wi-Fi off; tick every box.',
        'Publish the draft release (installer + release notes with the new sensitivity / specificity '
        'and the model identity).',
        f'Keep {cfg.release_dir}/ (state.json = stage + guard verdicts, logs/guards.log = guard evidence, '
        'logs/, bundle parts) until the hospital confirms the install.',
    ]


def print_manual_checklist(cfg: Config) -> None:
    print(f'\n{G}═══ pipeline finished — manual steps (the script never pushes or tags) ═══{NC}')
    for i, s in enumerate(manual_steps(cfg), 1):
        print(f'  {i}. {s}')
    print()


# ------------------------------------------------------------- subcommands ----

def cmd_plan(cfg: Config, stages: list[Stage]) -> int:
    print('#!/bin/sh')
    print(f'# release pipeline plan — batch {cfg.batch}, tag {cfg.tag} ({cfg.tag_source}), generated {now_iso()}')
    print(f'# extra sets combined with data/sets/{cfg.batch} (oldest first; the current batch goes last and '
          f'wins label conflicts): {" ".join(cfg.extra_sets) or "(none)"}')
    lock_problem = None if cfg.train_lock else train_lock_problem(cfg.repo, cfg.release_set)
    print(f'# train-locked patients (the current model was trained on them): '
          f'{cfg.train_lock or ("!! NONE — " + lock_problem if lock_problem else "(legacy train/ only)")}')
    print('set -e')
    print(f'cd {shlex.quote(str(cfg.repo))}')
    for i, s in enumerate(stages):
        print(f'\n# --- stage {i}: {s.name} — {s.title}')
        for t in GUARD_TEXT.get(s.name, []):
            print(f'#     guard: {t}')
        for c in s.cmds:
            print(shlex.join(c))
    print('\n# --- manual: set MODEL_BUNDLE_URL secret, commit the bump, bump desktop-app/package.json version '
          '(every release), git tag v<new app version>, git push tag, download installer, '
          'run docs/windows_laptop_test_*.md, publish the release')
    return 0


def cmd_status(cfg: Config, stages: list[Stage]) -> int:
    state = load_state(cfg.state_path)
    print(f'{B}release {cfg.tag} — batch {cfg.batch} — {cfg.rel(cfg.state_path)}{NC}')
    if not state.get('stages'):
        print(f'{Y}  no stage has run yet{NC}')
    col = {'done': G, 'archived': G, 'failed': R, 'running': Y}
    for s in stages:
        rec = state.get('stages', {}).get(s.name)
        if rec is None:
            print(f'  {s.name:<8} {"pending":<8}')
            continue
        st = rec.get('status', '?')
        print(f'  {s.name:<8} {col.get(st, "")}{st:<8}{NC} {rec.get("started") or "":<20} '
              f'{rec.get("finished") or "":<20} {rec.get("log", "")}')
        if st == 'failed':
            print(f'           {R}{rec.get("verdict") or rec.get("failed_cmd") or rec.get("error", "")}{NC}')
    for name, g in (state.get('guards') or {}).items():
        if isinstance(g, dict) and 'passed' in g:
            mark = f'{G}passed{NC}' if g.get('passed') else f'{R}FAILED{NC}'
            print(f'  guard {name:<12} {mark} {g.get("at", "")}')
        elif not isinstance(g, dict):
            print(f'  guard {name:<12} {g}')
    return 0


def cmd_run(cfg: Config, stages: list[Stage], selected: list[Stage],
            explicit: frozenset[str] = frozenset()) -> int:
    state = load_state(cfg.state_path)
    state.update(tag=cfg.tag, batch=cfg.batch, python=cfg.python, updated=now_iso(), options=cfg.options())
    state.setdefault('runs', []).append({'at': now_iso(), 'stages': [s.name for s in selected],
                                        'resume': cfg.resume, 'dry_run': cfg.dry_run,
                                        'explicit': sorted(explicit)})
    if not cfg.dry_run:
        save_state(cfg.state_path, state)
        write_latest(cfg.repo, cfg.tag)
    print(f'{B}release {cfg.tag} — batch {cfg.batch} — stages: {", ".join(s.name for s in selected)}'
          f'{"  [DRY RUN]" if cfg.dry_run else ""}{NC}')
    for s in selected:
        if cfg.resume:
            done, why = stage_is_done(cfg, s, state, explicit)
            if done:
                print(f'{G}↷ stage {s.name} already done — skipped{NC}')
                continue
            if why:
                print(f'{Y}↻ stage {s.name} re-run: {why}{NC}')
        promoted_to = archived_candidate(state)
        if s.name in NO_RERUN_AFTER_PROMOTE and promoted_to:
            print(f'{R}✗ the candidate of {cfg.tag} was already promoted into {promoted_to}: '
                  f'{s.name} cannot run again under the same tag (the gate would compare the new candidate '
                  f'with itself and promote would bury the real baseline under a second _prev_ dir).{NC}')
            print(f'  finish this release with: {_resume_hint(cfg)}')
            print(f'  or start a new release:   {shlex.quote(cfg.python)} scripts/release_pipeline.py run '
                  f'--batch {cfg.batch} --tag {next_new_tag(cfg.repo)} …')
            return EXIT_USAGE
        if not run_guards(cfg, s.name, state):
            if not cfg.dry_run:
                save_state(cfg.state_path, state)
            print(f'  guard output: {cfg.rel(cfg.logs_dir / "guards.log")}')
            print(f'  fix the problem, then: {_resume_hint(cfg)}')
            return EXIT_USAGE
        try:
            rc = run_stage(cfg, s, state)
        except KeyboardInterrupt:
            rec = state.get('stages', {}).get(s.name) or {}
            rec.update(status='failed', finished=now_iso(), returncode=130, error='interrupted (Ctrl-C)')
            if not cfg.dry_run:
                save_state(cfg.state_path, state)
            print(f'\n{R}✗ stage {s.name} INTERRUPTED{NC}' + (
                ' — promote_model.py rolls the move back on an interrupt: check that '
                f'{cfg.candidate} and {BASE_MODEL} are both present' if s.name == 'promote' else ''))
            print(f'  resume with: {_resume_hint(cfg)}')
            return 130
        if rc != 0:
            if (state['stages'].get(s.name) or {}).get('verdict'):
                print(f'  {Y}stopped before promote.{NC} Status: {shlex.quote(cfg.python)} '
                      f'scripts/release_pipeline.py status --batch {cfg.batch} --tag {cfg.tag}')
            else:
                print(f'  resume with: {_resume_hint(cfg)}')
            return EXIT_STAGE_FAILED
    if selected and selected[-1].name == 'bundle':
        print_manual_checklist(cfg)
    else:
        print(f'\n{G}✓ requested stages finished{NC}  (status: {shlex.quote(cfg.python)} '
              f'scripts/release_pipeline.py status --batch {cfg.batch} --tag {cfg.tag})')
    return 0


# --------------------------------------------------------------------- CLI ----

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description='Sentinel release pipeline orchestrator (see docs/release_runbook_en.md).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='stages, in order: ' + ' → '.join(STAGE_ORDER) + '\n\nOptions left out of a --resume / --from / '
               '--only / <stage> invocation are taken from data/release/<tag>/state.json.')
    bool_opt = argparse.BooleanOptionalAction
    ap.add_argument('command', choices=SUBCOMMANDS,
                    help='plan (print copy-pasteable shell) | run/all | status | <stage> (run that stage only)')
    ap.add_argument('--batch', required=True, help='batch name, e.g. new1 → data/new1_*.csv, data/sets/new1')
    ap.add_argument('--root', default=None, help='drive root with the study folders (index/match/build read it)')
    ap.add_argument('--reports', default=None, help='folder of report PDFs, or one .xlsx/.csv/.docx (parse stage)')
    ap.add_argument('--tag', help='release tag (default: data/release/LATEST while that release is unfinished, '
                                  'else the next unused brain-triage-v<N>)')
    ap.add_argument('--extra-sets', nargs='*', default=None,
                    help='already-built data/sets/* dirs to combine (default: every set except the batch '
                         '+ data/local_finetune_bin as the legacy batch-1 set)')
    ap.add_argument('--holdout', default=None, help=f'locked holdout ids (default {DEFAULT_HOLDOUT})')
    ap.add_argument('--train-lock', default=None, help='extra patient ids locked to train (combine --train-lock)')
    ap.add_argument('--epochs', type=int, default=None, help='default 12')
    ap.add_argument('--device', default=None, help='default mps')
    ap.add_argument('--base-model', default=None,
                    help=f'model the train stage starts from (default {BASE_MODEL}; fallback after a failed '
                         f'gate: {PUBLIC_VIT_BASE} or {PUBLIC_VIT_HUB_ID})')
    ap.add_argument('--from', dest='from_', choices=STAGE_ORDER, help='start at this stage')
    ap.add_argument('--only', choices=STAGE_ORDER, help='run this stage only')
    ap.add_argument('--resume', action='store_true', help='skip stages recorded as done in state.json')
    ap.add_argument('--dry-run', action='store_true', help='print the commands, run nothing')
    ap.add_argument('--python', default=None, help='interpreter for the stage scripts (default: this one)')
    ap.add_argument('--skip-benchmark', action=bool_opt, default=None,
                    help='do not run / require the label benchmark')
    ap.add_argument('--skip-tests', action=bool_opt, default=None,
                    help='do not run the full pytest suite before bundle')
    ap.add_argument('--label-llm', default=None, help='pass --llm ollama:<model> to label_reports.py')
    ap.add_argument('--llm-labels', default=None,
                    help='JSONL from training/relabel_workflow.js: the label stage passes it as '
                         '--import-llm-labels on every run (recorded in state.json)')
    ap.add_argument('--github', default=None, help=f'models repo for the bundle publish (default {GITHUB_MODELS_REPO})')
    ap.add_argument('--local-bundle', action=bool_opt, default=None, help='bundle locally only (no GitHub release)')
    ap.add_argument('--min-normals', type=int, default=None, help=f'default {MIN_NORMALS_DEFAULT}')
    ap.add_argument('--min-disk-gb', type=float, default=None, help=f'default {MIN_DISK_GB_DEFAULT:g}')
    ap.add_argument('--repo', default=str(REPO_DEFAULT), help=argparse.SUPPRESS)
    return ap


def _inherits_options(a: argparse.Namespace) -> bool:
    """Continuing a release (--resume, --from, --only, a stage subcommand, status) reuses the
    options stored in state.json; a plain `run` / `plan` starts from the CLI + defaults."""
    if a.command == 'plan':
        return False
    return bool(a.resume or a.from_ or a.only or a.command in STAGE_ORDER or a.command == 'status')


def config_from_args(a: argparse.Namespace) -> Config:
    repo = Path(a.repo).resolve()
    tag, source = resolve_tag(repo, a.batch, a.tag)
    stored: dict = {}
    if _inherits_options(a):
        stored = _read_json(repo / 'data' / 'release' / tag / 'state.json').get('options') or {}
    o: dict = {}
    inherited: dict = {}
    computed_defaults = {**OPTION_DEFAULTS, 'extra_sets': default_extra_sets(repo, a.batch),
                         'train_lock': default_train_lock(repo, f'data/train_release_{tag}')}
    for k, default in computed_defaults.items():
        v = getattr(a, k)
        if v is None and stored.get(k) is not None:
            v = stored[k]
            if v != default:
                inherited[k] = v
        o[k] = default if v is None else v
    llm_labels = o['llm_labels']
    if llm_labels:
        llm_labels = Config(repo=repo, batch=a.batch, tag=tag).rel(Path(llm_labels).expanduser())
    return Config(
        repo=repo, batch=a.batch, tag=tag,
        root=Path(o['root']).expanduser() if o['root'] else None,
        reports=Path(o['reports']).expanduser() if o['reports'] else None,
        extra_sets=list(o['extra_sets']), holdout=o['holdout'], train_lock=o['train_lock'],
        epochs=int(o['epochs']), device=o['device'], python=o['python'], base_model=o['base_model'],
        skip_benchmark=bool(o['skip_benchmark']), skip_tests=bool(o['skip_tests']),
        github=None if o['local_bundle'] else o['github'], label_llm=o['label_llm'], llm_labels=llm_labels,
        min_normals=int(o['min_normals']), min_disk_gb=float(o['min_disk_gb']),
        dry_run=a.dry_run, resume=a.resume, tag_source=source, inherited=inherited,
    )


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    stages = build_stages(cfg)

    if a.command == 'plan':
        return cmd_plan(cfg, stages)
    print(f'tag: {cfg.tag} ({cfg.tag_source})')
    if cfg.inherited:
        print('options from state.json: ' + ', '.join(f'{k}={v}' for k, v in cfg.inherited.items()))
    if a.command == 'status':
        return cmd_status(cfg, stages)

    only = a.only
    if a.command in STAGE_ORDER:
        if only and only != a.command:
            print(f'{R}✗ "{a.command}" and --only {only} disagree{NC}')
            return EXIT_USAGE
        only = a.command
    if only and a.from_:
        print(f'{R}✗ use either --only or --from, not both{NC}')
        return EXIT_USAGE
    selected = select_stages(stages, only, a.from_)
    explicit = frozenset(x for x in (only, a.from_) if x)

    needs_root = {'index', 'match', 'build'}
    needs_reports = {'parse'}
    wanted = {s.name for s in selected}
    if wanted & needs_root and cfg.root is None:
        print(f'{R}✗ --root is required for stages {sorted(wanted & needs_root)}{NC}')
        return EXIT_USAGE
    if wanted & needs_reports and cfg.reports is None:
        print(f'{R}✗ --reports is required for the parse stage{NC}')
        return EXIT_USAGE
    if not cfg.dry_run and wanted & needs_root and not cfg.root.is_dir():
        # a stage that --resume will skip does not need the drive
        state = load_state(cfg.state_path) if cfg.resume else {}
        todo = {s.name for s in selected if s.name in needs_root and
                not (cfg.resume and stage_is_done(cfg, s, state, explicit)[0])}
        if todo:
            print(f'{R}✗ drive not mounted: {cfg.root}{NC}')
            return EXIT_USAGE

    return cmd_run(cfg, stages, selected, explicit)


if __name__ == '__main__':
    sys.exit(main())
