#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — RELEASE PIPELINE ORCHESTRATOR  (stage 11 of the release contract)
================================================================================
  WHAT
    One resumable command that takes a new hospital batch (drive of DICOM
    studies + radiology reports) through every stage of
    docs/dev/release_pipeline_contract.md:

        index → parse → label → match → build → prospective → combine → train
              → gate → promote → bump → bundle

    Each stage is an existing repo script with a fixed CLI; this file only
    builds the commands, runs them one after another with a tee'd log, keeps a
    state file, and enforces the guard rails that must hold BEFORE training
    and BEFORE bundling. It never calls `git push` / `git tag` itself — the
    manual checklist printed after `bundle` tells the founder what to do next.

  WHY
    A retrain must be boring and repeatable: the same commands, the same
    locked held-out patients, the same gate floors, the same documentation
    bump. Everything intermediate lives under data/release/<tag>/ so a failed
    run can be resumed with one flag instead of re-typing twelve commands.

  USAGE (from sentinel/; PY=/Users/shakhzodbtr/venv/bin/python):
    $PY scripts/release_pipeline.py plan   --batch new1 --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
    $PY scripts/release_pipeline.py run    --batch new1 --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
    $PY scripts/release_pipeline.py run    --batch new1 --root /Volumes/Transcend/new1 \
                                           --reports ~/Desktop/reports_A ~/Desktop/reports_B   # oldest delivery first
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
    prospective/combine/train/gate/promote never run again under the same tag (the gate would
    compare the new candidate with itself; combine would rewrite the val set the promoted
    model's eval is pinned to; prospective would score the NEW model, now in the release dir, on
    its own training data) — start a new tag. A --resume after promote treats every stage up to
    promote as FROZEN (done, even when an upstream input changed since — a Finder .DS_Store,
    a rebuilt set): only bump and bundle are left to finish the release.
    train also fingerprints its --base-model and --epochs, so `--from train --resume` with a
    new --base-model (the retrain-from-the-public-ViT fallback) really retrains.
    build fingerprints --modality (default MR: the MR-only rule; 'any' disables it).
    A build recorded before the MR-only rule (no opt:modality) counts as '--modality any' and is
    re-run under the current rule on --resume. A stage that a promoted release has NO record of
    (prospective, after a promote by an older version) is 'skipped (promoted before this stage
    existed)'. --reports takes several sources, OLDEST delivery first: parse gets them in that order and
    a changed order re-runs parse. Only index and build read the drive (--root; no guard rail checks
    it either); match passes --images only when a root is configured.
    The current model's training set (data/train_release_<prev>) is locked to train in the
    next combine; if that set is gone or was rebuilt, combine refuses to run without an
    explicit --train-lock (it never silently drops the lock).

  PROSPECTIVE (stage 5)
    eval_prospective.py scores the model shipped BEFORE this release (the release dir
    models/brain_triage_finetuned — always that one, whatever --base-model the training starts
    from) on the new batch's set, reporting the patients it has never seen (--seen = the extra
    sets + the release set the shipped model was trained on, when it is a pipeline release)
    separately: data/release/<tag>/prospective_eval.json. Informational, never a gate:
    exit 0 = it ran. Its guard rail refuses to run when that trained-on set is gone or rebuilt
    (restore it, or pass --train-lock <release set dir>).
    Two commands: first report_priority.py writes data/<batch>_priority.csv (which matched report
    carries a priority finding — mass, recent infarct, haemorrhage, …; provisional rules, version in
    the file; --modality as for build, MR when --modality any), then
    eval_prospective.py gets --index data/<batch>_index.csv and --priority <that file> and adds
    'age_new' (the AUC of patient age alone, age-matched pairs, age bands) and 'priority_new' (how many
    priority-finding studies vs other studies the shipped threshold flags). WHY: on 2026-11-14 the
    shipped model was matched by age alone on 518 never-seen studies. The headline prints both. A
    missing index table is a guard-rail ERROR (eval_prospective.py refuses a missing --index file, and
    the age baseline must be printed for every batch): restore it or re-run --from index. An index
    without usable age columns is not an error: the script then writes age_new null with a note.
    An eval_prospective.py older than --index / --priority fails the stage (argparse, exit 2).
    A prospective result written before these blocks
    (no age_new / priority_new key) is re-run by --resume, never reused silently — except after
    promote, where the stage stays frozen as it is.

  CANDIDATES (--candidates warm,fresh; default warm — decision D6 of docs/dev/finish_line_contract.md)
    warm   starts from --base-model (default the current release models/brain_triage_finetuned)
    fresh  starts from the public ViT base: models/hf/google__vit-base-patch16-224 when that offline copy
           exists, else the hub id google/vit-base-patch16-224 from the local Hugging Face cache (the
           pre-train guard refuses 'fresh base not available offline' when neither holds it — nothing
           is downloaded; every stage runs with HF_HUB_OFFLINE=1 unless the caller set it) and passes
           --base-model-fresh.
    With the default (warm only) every path and command is exactly the single-candidate pipeline:
    models/brain_triage_candidate_<tag>. With more than warm, each candidate trains into
    models/brain_triage_candidate_<tag>_<cand>, the gate evaluates EACH with --compare-with the current
    release, and promote moves the best candidate that PASSED the gate with promotion.promote = true,
    by balanced accuracy (a tie goes to warm). state.json 'candidates' holds per-candidate paths, base,
    train/gate exit codes and gate numbers; 'candidate_choice' records the choice and the ranking. A
    candidate whose train record is done with unchanged inputs is not retrained on --resume. When no
    candidate qualifies, the gate stops the release before promote (same message as one candidate).

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

STAGE_ORDER = ('index', 'parse', 'label', 'match', 'build', 'prospective', 'combine',
               'train', 'gate', 'promote', 'bump', 'bundle')
SUBCOMMANDS = ('plan', 'run', 'all', 'status') + STAGE_ORDER
CANDIDATE_STAGES = ('train', 'gate', 'promote')      # never re-run after promote under one tag
# combine writes the release set the promoted model's eval is pinned to: never rewritten after
# promote either. prospective scores the model in the release dir (BASE_MODEL), which after promote
# IS the new model (trained on this batch). Every stage up to promote is frozen (done) on a --resume
# after promote.
NO_RERUN_AFTER_PROMOTE = ('prospective', 'combine') + CANDIDATE_STAGES
FROZEN_AFTER_PROMOTE = ('index', 'parse', 'label', 'match', 'build', 'prospective',
                        'combine') + CANDIDATE_STAGES
NEEDS_ROOT = ('index', 'build')                      # stages that read the drive (match: --images optional)
PROSPECTIVE_KIND = 'prospective_eval'

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
HF_WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')
HF_REQUIRED_FILES = ('config.json', 'preprocessor_config.json')
CANDIDATE_KINDS = ('warm', 'fresh')                  # preference order: a balanced-accuracy tie → warm
DEFAULT_CANDIDATES = ['warm']
DEFAULT_MODALITY = 'MR'                              # build --modality MR (MR-only rule); 'any' = no rule
MODALITY_ANY = 'any'
BALANCED_TIE_EPS = 1e-9
DEFAULT_HOLDOUT = 'data/holdout_patients_v1.txt'
LATEST_FILE = 'data/release/LATEST'
VENV_PYTHON = '/Users/shakhzodbtr/venv/bin/python'

DONE_STATUSES = ('done', 'archived')
# stage_is_done's reason for a frozen stage with NO record after promote (status / run print it)
NOT_APPLICABLE_AFTER_PROMOTE = 'promoted before this stage existed'
# an option input missing from a recorded stage means its LEGACY behaviour, not "no change", when
# that differs from the new default: the build before the MR-only rule exported every modality
LEGACY_OPTION_VALUES = {'opt:modality': f'opt:{MODALITY_ANY}'}

# Every option that shapes the stage commands. state.json keeps the effective values; a
# --resume / --from / --only / <stage> invocation reuses them for flags not given on the CLI,
# and the printed resume line re-emits ALL of them.
OPTION_DEFAULTS: dict[str, object] = {
    'root': None, 'reports': [], 'extra_sets': None, 'holdout': DEFAULT_HOLDOUT,
    'train_lock': None, 'epochs': 12, 'device': 'mps', 'python': sys.executable,
    'base_model': BASE_MODEL, 'skip_benchmark': False, 'skip_tests': False, 'label_llm': None,
    'llm_labels': None, 'github': GITHUB_MODELS_REPO, 'local_bundle': False,
    'min_normals': MIN_NORMALS_DEFAULT, 'min_disk_gb': MIN_DISK_GB_DEFAULT,
    'candidates': DEFAULT_CANDIDATES, 'modality': DEFAULT_MODALITY,
}


# ------------------------------------------------------------------ config ----

@dataclass
class Config:
    repo: Path
    batch: str
    tag: str
    root: Optional[Path] = None
    reports: list[Path] = field(default_factory=list)     # report sources, oldest delivery first
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
    candidates: list[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATES))
    modality: str = DEFAULT_MODALITY
    chosen: Optional[str] = None                      # candidate the gate chose (multi-candidate runs)
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
    def priority_csv(self) -> str: return f'data/{self.batch}_priority.csv'
    @property
    def set_dir(self) -> str: return f'data/sets/{self.batch}'
    @property
    def release_set(self) -> str: return f'data/train_release_{self.tag}'
    @property
    def candidate(self) -> str: return f'models/brain_triage_candidate_{self.tag}'
    @property
    def release_dir(self) -> str: return f'data/release/{self.tag}'
    @property
    def prospective_json(self) -> str: return f'{self.release_dir}/prospective_eval.json'
    @property
    def report_sources(self) -> list[str]:
        """--reports as they appear in the parse command (oldest delivery first)."""
        return [self.rel(p) for p in self.reports]
    @property
    def report_order(self) -> str:
        """The parse stage's 'opt:report_order' fingerprint: the sources in order (a swap re-runs parse)."""
        return ' | '.join(self.report_sources)
    @property
    def multi(self) -> bool:
        """True unless the candidate set is the default (warm only): the single-candidate pipeline
        keeps its historical paths and commands byte for byte."""
        return list(self.candidates) != DEFAULT_CANDIDATES

    def candidate_path(self, cand: str) -> str:
        """Output dir of one candidate: the historical path for the default warm-only run, else
        models/brain_triage_candidate_<tag>_<cand>."""
        return self.candidate if not self.multi else f'{self.candidate}_{cand}'

    def candidate_base(self, cand: str) -> str:
        """The model a candidate starts from (warm: --base-model; fresh: the public ViT base)."""
        return self.base_model if cand == 'warm' else fresh_base(self.repo)

    @property
    def promote_candidate(self) -> str:
        """The candidate dir promote moves: the only candidate of a single run, else the one the gate
        chose (a '<chosen>' placeholder before the gate has run — `plan` shows it)."""
        if not self.multi:
            return self.candidate
        return self.candidate_path(self.chosen) if self.chosen else f'{self.candidate}_<chosen>'
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
            'reports': [str(p) for p in self.reports],
            'extra_sets': list(self.extra_sets), 'holdout': self.holdout, 'train_lock': self.train_lock,
            'epochs': self.epochs, 'device': self.device, 'python': self.python,
            'base_model': self.base_model, 'skip_benchmark': self.skip_benchmark,
            'skip_tests': self.skip_tests, 'label_llm': self.label_llm, 'llm_labels': self.llm_labels,
            'github': self.github or GITHUB_MODELS_REPO, 'local_bundle': self.github is None,
            'min_normals': self.min_normals, 'min_disk_gb': self.min_disk_gb,
            'candidates': list(self.candidates), 'modality': self.modality,
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


def prospective_seen_release(cfg: Config) -> tuple[Optional[str], Optional[str]]:
    """(release set the shipped model was trained on — passed to the prospective stage as an extra
    --seen; problem). Both None for a shipped model trained on the legacy set only (the command is
    then exactly the historical one) and after promote under this tag (it then names this tag's
    own set). --train-lock wins when it is a directory; else _trained_on_set(). When that set cannot
    be used (train_lock_problem: gone or rebuilt) the problem says so — the stage's guard refuses to
    run with an incomplete --seen list, which would count trained-on patients as 'new'."""
    found = _trained_on_set(cfg.repo)
    if found is None or found[0] == cfg.release_set:
        return None, None
    if cfg.train_lock and cfg.train_lock != cfg.release_set and (cfg.repo / cfg.train_lock).is_dir():
        return cfg.train_lock, None
    rel, abs_dir = found
    if train_lock_problem(cfg.repo, cfg.release_set) is None:
        return rel, None
    why = (f'{rel}/dataset_manifest.json is missing' if not (abs_dir / 'dataset_manifest.json').is_file()
           else f'{rel} was rebuilt after the model was evaluated on it')
    return None, (f'the shipped model {BASE_MODEL} was trained on {rel} ({BASE_MODEL}/training_metrics.json) but '
                  f'{why}: its patients cannot be passed to the prospective stage as --seen, so the "new" group '
                  f'would count patients the model was trained on. Restore {rel}, or pass --train-lock '
                  f'<release set dir> naming the set it was trained on')


def parse_candidates(value: str) -> list[str]:
    """'warm,fresh' → ['warm', 'fresh'] in preference order (warm first); argparse type."""
    names = [x.strip().lower() for x in str(value).split(',') if x.strip()]
    unknown = sorted(set(names) - set(CANDIDATE_KINDS))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f'--candidates takes a comma-separated subset of {",".join(CANDIDATE_KINDS)}'
            + (f' (unknown: {",".join(unknown)})' if unknown else ''))
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError('--candidates lists a candidate twice')
    return [c for c in CANDIDATE_KINDS if c in names]


def fresh_base(repo: Path) -> str:
    """The 'fresh' candidate's start point: the offline copy when present, else the hub id."""
    return PUBLIC_VIT_BASE if (repo / PUBLIC_VIT_BASE).is_dir() else PUBLIC_VIT_HUB_ID


def hf_hub_cache_dir() -> Path:
    """The Hugging Face hub cache the trainer subprocess will read (same env resolution as
    huggingface_hub: HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub)."""
    for var in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']).expanduser() / 'hub'
    xdg = os.environ.get('XDG_CACHE_HOME')
    return (Path(xdg).expanduser() if xdg else Path.home() / '.cache') / 'huggingface' / 'hub'


def _snapshot_complete(snap: Path) -> bool:
    return all((snap / f).is_file() for f in HF_REQUIRED_FILES) and \
        any((snap / w).is_file() for w in HF_WEIGHT_FILES)


def hf_cache_has_model(hub_id: str, cache_dir: Optional[Path] = None) -> bool:
    """True when the hub cache holds a complete snapshot (config, preprocessor config, weights) of
    `hub_id` at the revision `main` resolves to — what from_pretrained(hub_id) loads offline.
    Uses huggingface_hub.try_to_load_from_cache; scans models--<org>--<name>/snapshots when the
    library is unavailable. Never touches the network."""
    cache = Path(cache_dir) if cache_dir else hf_hub_cache_dir()
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:                                   # library missing / broken: scan the layout
        snaps = cache / ('models--' + hub_id.replace('/', '--')) / 'snapshots'
        return snaps.is_dir() and any(_snapshot_complete(s) for s in snaps.iterdir() if s.is_dir())
    try:
        found = try_to_load_from_cache(hub_id, 'config.json', cache_dir=str(cache))
    except Exception:
        return False
    return isinstance(found, str) and _snapshot_complete(Path(found).parent)


def fresh_base_problem(repo: Path, python: str = VENV_PYTHON) -> Optional[str]:
    """Why the 'fresh' candidate cannot start offline (None = it can). Checked by the pre-train
    guard, so a missing base stops the release BEFORE hours of training of the other candidate."""
    local = repo / PUBLIC_VIT_BASE
    if local.is_dir():
        if _snapshot_complete(local):
            return None
        missing = [f for f in HF_REQUIRED_FILES if not (local / f).is_file()]
        if not any((local / w).is_file() for w in HF_WEIGHT_FILES):
            missing.append(' or '.join(HF_WEIGHT_FILES))
        return (f'fresh base not available offline: {PUBLIC_VIT_BASE} is incomplete (missing '
                f'{", ".join(missing)}) — delete it and copy it again (download hint: {_fresh_download_hint(python)})')
    if hf_cache_has_model(PUBLIC_VIT_HUB_ID):
        return None
    return (f'fresh base not available offline: neither {PUBLIC_VIT_BASE} nor the Hugging Face cache '
            f'({hf_hub_cache_dir()}) holds {PUBLIC_VIT_HUB_ID}. On a machine with internet run: '
            f'{_fresh_download_hint(python)} — then copy {PUBLIC_VIT_BASE}/ into this repo '
            '(or drop "fresh" from --candidates)')


def _fresh_download_hint(python: str) -> str:
    code = (f"from huggingface_hub import snapshot_download; snapshot_download('{PUBLIC_VIT_HUB_ID}', "
            f"local_dir='{PUBLIC_VIT_BASE}', allow_patterns=['*.json', 'model.safetensors'])")
    return f'{shlex.quote(python)} -c {shlex.quote(code)}'


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


def prospective_problem(p: Path) -> Optional[str]:
    """Why eval_prospective.py's JSON is not usable (None = good): it must exist, parse, have
    kind 'prospective_eval' and a 'groups' object."""
    if not p.is_file():
        return f'{p.name} missing'
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return f'{p.name} is not valid JSON'
    if not isinstance(d, dict) or d.get('kind') != PROSPECTIVE_KIND:
        return f'{p.name}: kind is not "{PROSPECTIVE_KIND}"'
    if not isinstance(d.get('groups'), dict):
        return f'{p.name}: no "groups" object'
    return None


def prospective_headline(p: Path) -> Optional[str]:
    """One line from the 'new' group (patients no earlier set contains), or None."""
    groups = _read_json(p).get('groups')
    new = groups.get('new') if isinstance(groups, dict) else None
    if not isinstance(new, dict):
        return None
    line = (f'prospective, never-seen patients: n {new.get("n_studies", "—")} studies · '
            f'sensitivity {_fmt(new.get("sensitivity"))} · specificity {_fmt(new.get("specificity"))}')
    return line + _age_priority_suffix(_read_json(p))


def _fmt3(x) -> str:
    return f'{x:.3f}' if isinstance(x, (int, float)) and not isinstance(x, bool) else '—'


def _count(x) -> str:
    return str(x) if isinstance(x, int) and not isinstance(x, bool) else '—'


def _age_priority_suffix(d: dict) -> str:
    """' · age alone AUC … · priority findings flagged …' from the age_new / priority_new blocks.
    WHY on the headline: on 2026-11-14 the shipped model was matched by patient age alone on never-seen
    studies; a release reader must see the age baseline next to the model's numbers every time.
    Three decimals (not two): on new1 age alone scored 0.779 and the model 0.776 — two decimals would print a
    tie and hide that age alone was higher. The interval of the difference (model − age) is what the runbook
    says to judge by, so it is on the line too when present.
    Empty when the blocks are null or absent (an unusable index / priority file, or a JSON written before
    the blocks existed)."""
    out = ''
    age = d.get('age_new')
    if isinstance(age, dict):
        ci = age.get('auc_difference_ci95')
        ci_txt = (f'; model − age [{_fmt3(ci[0])}, {_fmt3(ci[1])}]'
                  if isinstance(ci, (list, tuple)) and len(ci) == 2 else '')
        out += (f' · age alone AUC {_fmt3(age.get("auc_age_alone"))} (model {_fmt3(age.get("auc_model"))}'
                f'{ci_txt}), age-matched {_fmt3(age.get("auc_age_matched"))}')
    pri = d.get('priority_new')
    if isinstance(pri, dict):
        out += (f' · priority findings flagged {_count(pri.get("priority_flagged"))}/{_count(pri.get("n_priority"))}, '
                f'others {_count(pri.get("other_flagged"))}/{_count(pri.get("n_other"))}')
    return out


AGE_PRIORITY_KEYS = ('age_new', 'priority_new')


def prospective_predates_age_priority(p: Path) -> bool:
    """True when a (valid) prospective_eval.json was written before the age baseline and the
    priority-finding counts existed: it has no 'age_new' / 'priority_new' key at all. A key that is
    null means the new script ran without a usable index / priority file — that is current, not stale."""
    d = _read_json(p)
    return bool(d) and any(k not in d for k in AGE_PRIORITY_KEYS)


PREDATES_AGE_PRIORITY = ('recorded before the age baseline and the priority-finding counts existed '
                         '(prospective_eval.json has no age_new / priority_new)')


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
    # every report source, oldest delivery first (parse_hospital_reports.py --src order matters).
    # One source keeps its historical input spec (its path) so an older state.json stays 'done';
    # with several, opt:report_order also records their ORDER.
    reports = cfg.report_sources or ['<reports-folder-or-file>']
    parse_inputs = list(reports) + (['opt:report_order'] if len(cfg.reports) > 1 else [])
    manifest = f'{cfg.release_set}/dataset_manifest.json'

    match_cmd = [py, 'scripts/match_images_to_reports.py']
    if cfg.root:                                       # --images is optional when --index is given
        match_cmd += ['--images', root]
    match_cmd += ['--index', cfg.index_csv, '--labels', cfg.labels_csv, '--out', cfg.matched_csv]

    build_cmd = [py, 'scripts/build_training_set.py', '--images', root, '--matched', cfg.matched_csv,
                 '--out', cfg.set_dir, '--mode', 'binary', '--abnormal-col', 'abnormal', '--skip-needs-review']
    if cfg.modality != MODALITY_ANY:
        build_cmd += ['--modality', cfg.modality]

    # prospective: the model shipped TODAY on the new batch — the release dir BASE_MODEL (what the gate
    # compares with and promote replaces), NOT --base-model: that is only where training starts and is
    # the public ViT in the retrain-from-scratch fallback. The extra sets are what the shipped model
    # (or its predecessors) may have seen, so the 'new' group counts only never-seen patients. After a
    # pipeline release the shipped model was trained on a combined release set, which may hold this very
    # batch (new reports for it): that set is a --seen set too (none for a legacy-trained model).
    seen_release, _problem = prospective_seen_release(cfg)
    seen_release = seen_release if seen_release not in cfg.extra_sets else None
    prospective_cmd = [py, 'scripts/eval_prospective.py', '--model', BASE_MODEL, '--set', cfg.set_dir]
    for s in cfg.extra_sets + ([seen_release] if seen_release else []):
        prospective_cmd += ['--seen', s]
    # --index / --priority: the age baseline and the priority-finding counts (review 2026-11-14: on
    # never-seen studies the shipped model was matched by patient age alone). The priority table comes
    # from the reports (report_priority.py, the stage's first command); the ages from the index stage's
    # table, always data/<batch>_index.csv — eval_prospective.py refuses a missing --index file (exit 2),
    # so the pre-prospective guard stops a run without it before anything is scored.
    # report_priority.py reads the reports of the same modality as the set (build's --modality); with
    # --modality any it keeps its default, MR (it has no 'any'): the priority lines then cover MR only.
    # No opt:modality input is needed: a modality change rebuilds the set, whose studies.csv is an input.
    priority_cmd = [py, 'scripts/report_priority.py', '--matched', cfg.matched_csv,
                    '--labels', cfg.labels_csv, '--out', cfg.priority_csv]
    if cfg.modality != MODALITY_ANY:
        priority_cmd += ['--modality', cfg.modality]
    prospective_cmd += ['--holdout', cfg.holdout, '--matched', cfg.matched_csv,
                        '--index', cfg.index_csv, '--priority', cfg.priority_csv,
                        '--out', cfg.prospective_json, '--device', cfg.device]
    seen_release_input = []
    if seen_release:
        seen_release_input = [f'{seen_release}/dataset_manifest.json'
                              if (cfg.repo / seen_release / 'dataset_manifest.json').is_file()
                              else set_input(cfg.repo, seen_release)]
    prospective_inputs = [f'{cfg.set_dir}/studies.csv', cfg.matched_csv, cfg.labels_csv, cfg.index_csv,
                          cfg.holdout, *(set_input(cfg.repo, s) for s in cfg.extra_sets), *seen_release_input,
                          f'{BASE_MODEL}/MANIFEST.json']

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
        # no --tag: the publisher's default tag brain-triage-<identity12> is what the installer CI
        # looks up by the identity the code pins (packaging/fetch_model_bundle.py --repo)
        bundle_cmd += ['--github', cfg.github]

    # train / gate: one command per candidate (the default warm-only run has exactly one, at the
    # historical candidate path). promote: the candidate the gate chose (cfg.promote_candidate).
    train_cmds = [train_command(cfg, c) for c in cfg.candidates]
    gate_cmds = [[py, 'scripts/eval_study_level.py', '--model', cfg.candidate_path(c),
                  '--data', f'{cfg.release_set}/val', '--write-manifest',
                  '--min-sens', f'{GATE_MIN_SENS:.2f}', '--min-spec', f'{GATE_MIN_SPEC:.2f}',
                  '--compare-with', BASE_MODEL, '--device', cfg.device] for c in cfg.candidates]
    cand_paths = [cfg.candidate_path(c) for c in cfg.candidates]
    train_title = (f'Fine-tune the candidate from {cfg.base_model}' if not cfg.multi else
                   'Fine-tune the candidates: ' + ', '.join(f'{c} from {cfg.candidate_base(c)}'
                                                            for c in cfg.candidates))
    gate_title = ('Study-level gate + comparison with the current release' if not cfg.multi else
                  'Study-level gate + comparison with the current release, per candidate; '
                  'choose the best that passed (balanced accuracy, tie → warm)')
    promote_title = ('Move the candidate into models/brain_triage_finetuned (old → _prev_<id>)' if not cfg.multi
                     else 'Move the chosen candidate into models/brain_triage_finetuned (old → _prev_<id>)')

    return [
        Stage('index', 'Index the drive (one row per study)',
              [[py, 'scripts/index_new_studies.py', '--root', root, '--out', cfg.index_csv]],
              inputs=[root], outputs=[cfg.index_csv],
              good='n studies ≈ number of study folders, error column mostly empty, MR/CT split printed'),
        Stage('parse', 'Parse the hospital reports (PDF folder / xlsx / csv / docx)',
              [[py, 'scripts/parse_hospital_reports.py', '--src', *reports, '--out', cfg.reports_csv]],
              inputs=parse_inputs, outputs=[cfg.reports_csv],
              good='one row per study SECTION, modality from the section header, no patient names'),
        Stage('label', 'Rule-based labels (+ imported LLM labels) + benchmark against the 894-patient reference',
              label_cmds, inputs=label_inputs,
              outputs=[cfg.labels_csv],
              good='agreement on abnormal ≥ 0.90; needs_review share small (< 15%)'),
        Stage('match', 'Join images ↔ reports (patient id + date > modality > only report)',
              [match_cmd],
              inputs=[cfg.index_csv, cfg.labels_csv], outputs=[cfg.matched_csv],
              good='most brain studies matched; match_quality mostly "id+date"'),
        Stage('build', 'Export central slices to data/sets/<batch> (binary, resumable)',
              [build_cmd],
              inputs=[cfg.matched_csv, 'opt:modality'], outputs=[cfg.set_dir, f'{cfg.set_dir}/studies.csv'],
              good='≈ 5 slices per study, studies.csv row count = matched brain studies minus needs_review'),
        Stage('prospective', 'Score the model shipped today on the new batch (never-seen patients reported apart; '
                             'age baseline + priority findings)',
              [priority_cmd, prospective_cmd], inputs=prospective_inputs,
              outputs=[cfg.prospective_json, cfg.priority_csv],
              good=f'{cfg.priority_csv} (priority findings from the reports, provisional rules) and {cfg.prospective_json} '
                   f'with kind "{PROSPECTIVE_KIND}"; the "new" group = patients no earlier set contains; the age-alone '
                   'AUC and the priority-finding counts printed (informational, not a gate: a model that does not '
                   'beat age alone is not evidence of detection)'),
        Stage('combine', 'Merge all sets, patient-level split (train / val / tune), locked holdout → val',
              [combine_cmd], inputs=combine_inputs,
              outputs=[cfg.release_set, manifest],
              good='0 train/val/tune overlap, all 178 holdout patients in val, tune/ written, per-batch counts printed'),
        Stage('train', train_title, train_cmds,
              # opt:* = option VALUES that shape the weights: a new --base-model / --epochs (the
              # retrain-from-the-public-ViT fallback) re-runs train on --resume
              inputs=[manifest, 'opt:base_model', 'opt:epochs', *(['opt:candidates'] if cfg.multi else [])],
              outputs=[o for p in cand_paths for o in (p, f'{p}/model.safetensors')],
              good='val loss falls then plateaus; checkpoint selected on tune/ (not on the gate set); '
                   'training_metrics.json written'),
        Stage('gate', gate_title, gate_cmds,
              inputs=[*(f'{p}/model.safetensors' for p in cand_paths), manifest],
              outputs=[o for p in cand_paths for o in (f'{p}/study_level_eval.json', f'{p}/MANIFEST.json')],
              good=f'sens ≥ {GATE_MIN_SENS}, spec ≥ {GATE_MIN_SPEC}, promotion.promote = true'
                   + ('' if not cfg.multi else ' for at least one candidate')),
        Stage('promote', promote_title,
              [[py, 'scripts/promote_model.py', '--candidate', cfg.promote_candidate, '--dest', BASE_MODEL]],
              inputs=[f'{cfg.promote_candidate}/study_level_eval.json'],
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


def train_command(cfg: Config, cand: str) -> list[str]:
    """finetune_brain_classifier.py argv of one candidate. warm = the historical command; fresh adds
    --base-model-fresh (new 2-class head on the ImageNet ViT)."""
    argv = [cfg.python, 'training/finetune_brain_classifier.py', '--data', cfg.release_set,
            '--classes', 'abnormal,normal', '--base-model', cfg.candidate_base(cand)]
    if cand == 'fresh':
        argv.append('--base-model-fresh')
    return argv + ['--balance', '--unfreeze-last-blocks', '3', '--epochs', str(cfg.epochs),
                   '--device', cfg.device, '--select-split', 'tune', '--output', cfg.candidate_path(cand)]


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


def promoted_candidate(cfg: Config, state: dict) -> str:
    """The candidate dir promote moved (recorded by _archive_candidate; the historical single
    candidate path for a state written before multi-candidate runs existed)."""
    rec = state.get('stages', {}).get('promote') or {}
    return rec.get('promoted_candidate') or cfg.candidate


def resolve_path(cfg: Config, state: dict, p: str) -> str:
    """A candidate path after promote moved the candidate: models/brain_triage_candidate_<tag>[_<cand>]/x
    → <archived dest>/x. Everything else (incl. a candidate that was NOT chosen) unchanged."""
    dest = archived_candidate(state)
    moved = promoted_candidate(cfg, state)
    if dest and (p == moved or p.startswith(moved + '/')):
        return dest + p[len(moved):]
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
    loguru logging stays readable in the tee'd logs (no DEBUG flood, no hidden INFO).
    HF_HUB_OFFLINE=1 (unless the caller set it): a hub id (the fresh candidate's
    google/vit-base-patch16-224) loads from the local cache the pre-train guard checked — the
    pipeline never downloads weights in the middle of a release."""
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'LOGURU_LEVEL': 'INFO'}
    env.setdefault('HF_HUB_OFFLINE', '1')
    return env


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
        parts += ['--reports', *o['reports']]                 # every source, oldest delivery first
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
    parts += ['--min-normals', str(o['min_normals']), '--min-disk-gb', f'{o["min_disk_gb"]:g}',
              '--modality', o['modality']]
    if cfg.multi:
        parts += ['--candidates', ','.join(o['candidates'])]
    return shlex.join(parts)


def _resume_hint(cfg: Config) -> str:
    return f'{shlex.quote(cfg.python)} scripts/release_pipeline.py run {args_line(cfg)} --resume'


# ------------------------------------------------------------------ guards ----

def guard_pre_train(cfg: Config, state: dict) -> tuple[list[str], list[str]]:
    """Guard rails that must hold before the (hours-long) train stage. The drive is not one of them:
    train reads only the local release set (slices + symlinks to data/sets)."""
    errors: list[str] = []
    warnings: list[str] = []
    repo = cfg.repo

    # 1. (no drive check: only index and build read the drive — NEEDS_ROOT)

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

    # 5. fresh candidate: the public ViT base must be on this machine (never downloaded mid-release)
    if 'fresh' in cfg.candidates:
        problem = fresh_base_problem(repo, cfg.python)
        if problem:
            errors.append(problem)
        else:
            print(f'  fresh base: {G}{fresh_base(repo)}{NC} (offline)')

    # 6. disk
    gb = free_gb(repo)
    print(f'  free disk: {gb:.1f} GB')
    if gb < cfg.min_disk_gb:
        errors.append(f'only {gb:.1f} GB free (need ≥ {cfg.min_disk_gb:.0f} GB for slices + checkpoints)')

    return errors, warnings


def guard_pre_prospective(cfg: Config, state: dict) -> tuple[list[str], list[str]]:
    """The set the shipped model was trained on must be a --seen set, or 'new' is not never-seen
    (fail closed: never score with an incomplete --seen list). The batch index table (patient ages) must
    be present too: eval_prospective.py refuses a missing --index file (exit 2), and the age baseline is
    the one line that shows whether a model learned age (review 2026-11-14) — so it is an error here,
    stated before anything runs, not a stage failure after the priority table was already rewritten."""
    _seen, problem = prospective_seen_release(cfg)
    errors = [problem] if problem else []
    if not (cfg.repo / cfg.index_csv).is_file():
        errors.append(f'{cfg.index_csv} missing (index stage output; the prospective stage needs the patient '
                      f'ages for the age baseline) — restore it, or re-run the index stage (--from index)')
    return errors, []


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
    'prospective': guard_pre_prospective,
    'combine': guard_pre_combine,
    'train': guard_pre_train,
    'bundle': guard_pre_bundle,
}
GUARD_TEXT = {
    'prospective': ['the release set the shipped model was trained on (if any) is present and unchanged, '
                    'so it is passed as --seen, or --train-lock names it as a directory',
                    'the batch index table (patient ages) is present'],
    'combine': ['the set the current model was trained on is present and unchanged (its patients are '
                'locked to train), or --train-lock is given'],
    'train': ['label benchmark passed (or --skip-benchmark)',
              f'matched brain studies + normal count (warn < {MIN_NORMALS_DEFAULT} normals)',
              'dataset_manifest.json present, no train/val/tune patient overlap, holdout ⊂ val, tune/ present',
              f'≥ {MIN_DISK_GB_DEFAULT:.0f} GB free disk'],
    'bundle': ['gate.passed true and promotion.promote true in the promoted model',
               'promote_model.eval_verdict: release floors, production protocol, release val set, eval pinned '
               'to these weights',
               'bump_model_release.py --check clean',
               'pytest -q and docs/test_docs_consistency.py green'],
}


def guard_text(cfg: Config, stage_name: str) -> list[str]:
    """GUARD_TEXT of a stage + the checks only a multi-candidate run makes."""
    lines = list(GUARD_TEXT.get(stage_name, []))
    if stage_name == 'train' and 'fresh' in cfg.candidates:
        lines.append(f'fresh base available offline ({PUBLIC_VIT_BASE} or {PUBLIC_VIT_HUB_ID} in the '
                     'Hugging Face cache)')
    return lines


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
        for t in guard_text(cfg, stage_name):
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
    if rec is None and stage.name in FROZEN_AFTER_PROMOTE and archived_candidate(state):
        # promoted by a version that did not have this stage (prospective): not applicable — the
        # release can only be finished (bump, bundle), never re-run up to promote under this tag
        return True, NOT_APPLICABLE_AFTER_PROMOTE
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
    if stage.name == 'prospective' and (cfg.repo / cfg.prospective_json).is_file():
        problem = prospective_problem(cfg.repo / cfg.prospective_json)
        if problem:
            return False, f'outputs invalid: {problem}'
        # a result from before the age / priority lines is never reused silently (a promoted release
        # returned above: frozen as it is). A missing priority csv forgiven by the outputs-consumed
        # rule (a later stage is done) does not hide it.
        if prospective_predates_age_priority(cfg.repo / cfg.prospective_json):
            return False, PREDATES_AGE_PRIORITY
    recorded = rec.get('input_sha256s')
    if recorded and inputs_changed(recorded, input_fingerprints(cfg, stage, state)):
        return False, 'inputs changed since the last run'
    return True, ''


def inputs_changed(recorded: dict, current: dict) -> bool:
    """A directory input (the drive, the reports folder: 'dir:<n>') that is now absent is
    'cannot tell', not 'changed' — unplugging the drive after build must not re-run index.
    An option input ('opt:…') that a state written before it existed did not record is not a change —
    unless the old behaviour differs from the new default (LEGACY_OPTION_VALUES: a build recorded
    before the MR-only rule ran with modality 'any' and is re-run under the current rule)."""
    for k in set(recorded) | set(current):
        r, c = recorded.get(k), current.get(k)
        if isinstance(r, str) and r.startswith('dir:') and c == 'missing':
            continue
        if k.startswith('opt:') and r is None and k in LEGACY_OPTION_VALUES and k in current:
            r = LEGACY_OPTION_VALUES[k]
        if k.startswith('opt:') and r is None:
            continue
        if r != c:
            return True
    return False


def _archive_candidate(cfg: Config, state: dict, rec: dict) -> None:
    """After a successful promote: the (chosen) candidate dir now lives at BASE_MODEL."""
    moved = cfg.promote_candidate
    if (cfg.repo / moved).exists() or not (cfg.repo / BASE_MODEL).exists():
        return
    rec['archived_candidate'] = BASE_MODEL
    rec['promoted_candidate'] = moved
    if cfg.multi and cfg.chosen:
        rec['chosen_candidate'] = cfg.chosen
    for name in ('train', 'gate'):
        r = state.get('stages', {}).get(name)
        if r and r.get('status') == 'done':
            r.update(status='archived', archived_to=BASE_MODEL, archived_at=now_iso())


def _gate_verdict_message(cfg: Config, rc: int) -> list[str]:
    if cfg.multi:
        return _gate_verdict_message_multi(cfg, rc)
    ev_path = f'{cfg.candidate}/study_level_eval.json'
    common = [f'the current release {BASE_MODEL} is untouched — promote was NOT run and will not run',
              f'read {ev_path} and {cfg.release_dir}/logs/gate.log (runbook section 7)',
              'options: keep the old model (nothing to do) · add more normal studies and retrain · '
              f'retrain from the public ViT base: {_retrain_hint(cfg)}',
              f'or train both candidates and promote the better one: {_both_candidates_hint(cfg)}']
    if rc == EVAL_GATE_FAILED:
        return [f'GATE FAILED: the candidate is below the floors (sens ≥ {GATE_MIN_SENS:.2f}, '
                f'spec ≥ {GATE_MIN_SPEC:.2f}) on {cfg.release_set}/val'] + common
    return ['NOT PROMOTED: the candidate passed the floors but is not better than the current release '
            '(balanced accuracy ≥ old − 0.01 and sensitivity ≥ old − 0.03 required)'] + common


def _gate_verdict_message_multi(cfg: Config, rc: int) -> list[str]:
    """Same headline and options as one candidate, naming every candidate's eval."""
    evals = ', '.join(f'{cfg.candidate_path(c)}/study_level_eval.json' for c in cfg.candidates)
    names = ' and '.join(cfg.candidates)
    # a multi-candidate run always includes 'fresh' (the candidate set is a non-default subset of
    # warm,fresh), so the public-ViT fallback has been tried already: more data is the way forward
    common = [f'the current release {BASE_MODEL} is untouched — promote was NOT run and will not run',
              f'read {evals} and {cfg.release_dir}/logs/gate.log (runbook section 7)',
              'options: keep the old model (nothing to do) · add more normal studies and retrain']
    if rc == EVAL_GATE_FAILED:
        return [f'GATE FAILED: no candidate ({names}) reaches the floors (sens ≥ {GATE_MIN_SENS:.2f}, '
                f'spec ≥ {GATE_MIN_SPEC:.2f}) on {cfg.release_set}/val'] + common
    return [f'NOT PROMOTED: no candidate ({names}) passed the floors AND is better than the current release '
            '(balanced accuracy ≥ old − 0.01 and sensitivity ≥ old − 0.03 required)'] + common


def _both_candidates_hint(cfg: Config) -> str:
    """Re-run train → gate with warm AND fresh (D6); --from train, so train really runs again."""
    return (f'{shlex.quote(cfg.python)} scripts/release_pipeline.py run {args_line(cfg)} '
            '--candidates warm,fresh --from train')


def _retrain_hint(cfg: Config) -> str:
    base = PUBLIC_VIT_BASE if (cfg.repo / PUBLIC_VIT_BASE).is_dir() else PUBLIC_VIT_HUB_ID
    line = args_line(cfg).replace(f'--base-model {shlex.quote(cfg.base_model)}', f'--base-model {base}')
    # no --resume: train must run again even though its record is 'done' (train also
    # fingerprints --base-model, so a --resume with the new base would re-run it as well)
    return f'{shlex.quote(cfg.python)} scripts/release_pipeline.py run {line} --from train'


# ------------------------------------------------------- multi-candidate ----

def _candidate_train_fingerprint(cfg: Config, cand: str) -> dict:
    """What shapes one candidate's weights: the release set manifest, its base, the epochs."""
    manifest = cfg.repo / cfg.release_set / 'dataset_manifest.json'
    return {'manifest': fingerprint(manifest), 'base_model': cfg.candidate_base(cand), 'epochs': cfg.epochs}


def _gate_numbers(ev: dict) -> dict:
    """Gate numbers of one candidate's study_level_eval.json (balanced = (sens+spec)/2 when the
    eval predates the 'balanced' key)."""
    gate = ev.get('gate') or {}
    promo = ev.get('promotion') or {}
    sens, spec = gate.get('sens'), gate.get('spec')
    bal = gate.get('balanced')
    if bal is None and isinstance(sens, (int, float)) and isinstance(spec, (int, float)):
        bal = (sens + spec) / 2
    return {'passed': bool(gate.get('passed')), 'promote': bool(promo.get('promote')) if promo else False,
            'has_promotion': bool(promo), 'sens': sens, 'spec': spec, 'balanced': bal,
            'reason': promo.get('reason', '')}


def choose_candidate(results: dict[str, dict]) -> tuple[Optional[str], list[dict]]:
    """(chosen candidate or None, ranking). Qualified = eval exit 0 AND gate.passed AND
    promotion.promote. Best balanced accuracy wins; a tie (|Δ| ≤ 1e-9) goes to the candidate earlier
    in CANDIDATE_KINDS (warm)."""
    ranking: list[dict] = []
    best: Optional[str] = None
    for cand in CANDIDATE_KINDS:
        r = results.get(cand)
        if r is None:
            continue
        why = ''
        if r.get('returncode') != 0:
            why = {EVAL_GATE_FAILED: 'gate failed (below the floors)',
                   EVAL_NOT_PROMOTED: 'not better than the current release'}.get(r.get('returncode'),
                                                                                  f'eval exit {r.get("returncode")}')
        elif not r.get('passed'):
            why = 'gate.passed is not true'
        elif not r.get('has_promotion'):
            why = 'no promotion block (gate ran without --compare-with)'
        elif not r.get('promote'):
            why = 'promotion.promote is false'
        elif not isinstance(r.get('balanced'), (int, float)):
            why = 'no balanced accuracy in the eval'
        qualified = not why
        ranking.append({'candidate': cand, 'qualified': qualified, 'balanced': r.get('balanced'),
                        'sens': r.get('sens'), 'spec': r.get('spec'), 'why_not': why})
        if qualified and (best is None or
                          r['balanced'] > results[best]['balanced'] + BALANCED_TIE_EPS):
            best = cand
    return best, ranking


def _fmt(x) -> str:
    return f'{x:.3f}' if isinstance(x, (int, float)) else '—'


def _print_candidate_table(cfg: Config, state: dict) -> None:
    cands = state.get('candidates') or {}
    choice = state.get('candidate_choice') or {}
    print(f'  candidates (promote = best balanced accuracy among those that PASSED the gate with '
          f'promotion.promote; tie → warm):')
    for cand in cfg.candidates:
        c = cands.get(cand) or {}
        tr = (c.get('train') or {})
        gt = (c.get('gate') or {})
        gate_s = (f'{gt.get("verdict", "?"):<13} sens {_fmt(gt.get("sens"))} spec {_fmt(gt.get("spec"))} '
                  f'balanced {_fmt(gt.get("balanced"))}' if gt else 'gate pending')
        print(f'    {cand:<6} base {c.get("base_model") or cfg.candidate_base(cand):<42} '
              f'train {tr.get("status", "pending"):<8} {gate_s}  → {c.get("path") or cfg.candidate_path(cand)}')
    if choice:
        if choice.get('chosen'):
            print(f'  {G}chosen: {choice["chosen"]}{NC} ({choice.get("path")})')
        else:
            print(f'  {R}chosen: none — no candidate qualified{NC}')


def run_candidates_stage(cfg: Config, stage: Stage, state: dict) -> int:
    """train / gate of a multi-candidate run: one command per candidate, per-candidate records in
    state['candidates'], and (gate) the choice in state['candidate_choice'].

    train: a candidate whose record is done with unchanged inputs and whose weights exist is
    skipped on --resume; any non-zero exit stops the stage (fix and resume).
    gate : eval exit 0 / 2 (gate failed) / 4 (not promoted) is a VERDICT, recorded, and the next
    candidate is gated; any other exit (1 crash, 3 manifest) stops the stage. Afterwards the best
    qualifying candidate is chosen; none → the stage fails with the gate verdict (stop before promote)."""
    print(f'\n{B}═══ stage {stage.name}: {stage.title} ═══{NC}')
    log = cfg.logs_dir / f'{stage.name}.log'
    rec = {
        'status': 'running', 'started': now_iso(), 'finished': None,
        'cmd': ' && '.join(shlex.join(c) for c in stage.cmds),
        'cmds': [shlex.join(c) for c in stage.cmds],
        'log': cfg.rel(log), 'outputs': list(stage.outputs), 'inputs': list(stage.inputs),
        'input_sha256s': {} if cfg.dry_run else input_fingerprints(cfg, stage, state),
        'candidates': list(cfg.candidates),
    }
    state.setdefault('stages', {})[stage.name] = rec
    cands = state.setdefault('candidates', {})
    if stage.name == 'gate':
        state.pop('candidate_choice', None)
    if not cfg.dry_run:
        save_state(cfg.state_path, state)

    def fail(rc: int, argv: list[str], **extra) -> int:
        rec.update(status='failed', finished=now_iso(), returncode=rc, failed_cmd=shlex.join(argv), **extra)
        if not cfg.dry_run:
            save_state(cfg.state_path, state)
        print(f'\n{R}✗ stage {stage.name} FAILED (exit {rc}){NC}')
        print(f'  command : {shlex.join(argv)}')
        print(f'  log     : {cfg.rel(log)}')
        return rc

    results: dict[str, dict] = {}
    for cand, argv in zip(cfg.candidates, stage.cmds):
        path = cfg.candidate_path(cand)
        crec = cands.setdefault(cand, {})
        crec.update(path=path, base_model=cfg.candidate_base(cand), fresh_head=cand == 'fresh')
        if stage.name == 'train':
            fp = {} if cfg.dry_run else _candidate_train_fingerprint(cfg, cand)
            old = crec.get('train') or {}
            if (cfg.resume and not cfg.dry_run and old.get('status') == 'done'
                    and old.get('input_sha256s') == fp and (cfg.repo / path / 'model.safetensors').is_file()):
                print(f'{G}↷ candidate {cand} already trained ({path}) — skipped{NC}')
                continue
            print(f'{B}── candidate {cand}: train from {crec["base_model"]} → {path}{NC}')
            crec['train'] = {'status': 'running', 'started': now_iso(), 'cmd': shlex.join(argv),
                             'input_sha256s': fp}
            crec.pop('gate', None)                     # new weights: an older gate result no longer applies
            if not cfg.dry_run:
                save_state(cfg.state_path, state)
            rc = run_command(argv, log, cfg.repo, cfg.dry_run)
            crec['train'].update(status='done' if rc == 0 else 'failed', returncode=rc, finished=now_iso())
            if rc != 0:
                return fail(rc, argv, failed_candidate=cand)
        else:                                          # gate
            print(f'{B}── candidate {cand}: gate {path}{NC}')
            rc = run_command(argv, log, cfg.repo, cfg.dry_run)
            if cfg.dry_run:
                continue
            if rc not in (0, EVAL_GATE_FAILED, EVAL_NOT_PROMOTED):
                crec['gate'] = {'returncode': rc, 'verdict': 'error', 'finished': now_iso()}
                if rc == EVAL_MANIFEST_FAILED:
                    print(f'{R}  MANIFEST check failed: the {cand} candidate weights do not match MANIFEST.json{NC}')
                return fail(rc, argv, failed_candidate=cand)
            ev_path = f'{path}/study_level_eval.json'
            ev = read_eval(cfg.repo / path)
            if not ev:
                crec['gate'] = {'returncode': rc, 'verdict': 'error', 'finished': now_iso(),
                                'error': f'{ev_path} missing or unreadable'}
                return fail(rc or 1, argv, failed_candidate=cand, error=f'{ev_path} missing or unreadable')
            nums = _gate_numbers(ev)
            verdict = {0: 'passed', EVAL_GATE_FAILED: 'gate_failed', EVAL_NOT_PROMOTED: 'not_promoted'}[rc]
            crec['gate'] = {'returncode': rc, 'verdict': verdict, 'eval': ev_path, 'finished': now_iso(),
                            **{k: nums[k] for k in ('passed', 'promote', 'sens', 'spec', 'balanced', 'reason')}}
            results[cand] = {'returncode': rc, **nums}
            print(f'  {cand}: {verdict} — sens {_fmt(nums["sens"])} spec {_fmt(nums["spec"])} '
                  f'balanced {_fmt(nums["balanced"])}')
        if not cfg.dry_run:
            save_state(cfg.state_path, state)

    if not cfg.dry_run:
        missing = [o for o in stage.outputs if not (cfg.repo / o).exists()]
        if missing:
            rec.update(status='failed', finished=now_iso(), returncode=0,
                       error=f'expected outputs missing: {missing}')
            save_state(cfg.state_path, state)
            print(f'{R}✗ stage {stage.name} finished but expected outputs are missing: {missing}{NC}')
            print(f'  log     : {cfg.rel(log)}')
            return 1

    if stage.name == 'gate' and not cfg.dry_run:
        chosen, ranking = choose_candidate(results)
        state['candidate_choice'] = {
            'chosen': chosen, 'path': cfg.candidate_path(chosen) if chosen else None, 'at': now_iso(),
            'rule': 'best balanced accuracy among candidates with eval exit 0, gate.passed and '
                    'promotion.promote; tie → warm',
            'ranking': ranking}
        cfg.chosen = chosen
        _print_candidate_table(cfg, state)
        if chosen is None:
            all_below_floor = all(r['returncode'] == EVAL_GATE_FAILED for r in results.values())
            rc = EVAL_GATE_FAILED if all_below_floor else EVAL_NOT_PROMOTED
            rec.update(status='failed', finished=now_iso(), returncode=rc,
                       verdict='gate_failed' if rc == EVAL_GATE_FAILED else 'not_promoted',
                       failed_cmd=rec['cmds'][-1])
            save_state(cfg.state_path, state)
            print(f'\n{R}✗ stage gate FAILED (exit {rc}){NC}')
            msg = _gate_verdict_message(cfg, rc)
            print(f'{R}  {msg[0]}{NC}')
            for m in msg[1:]:
                print(f'  {m}')
            print(f'  log     : {cfg.rel(log)}')
            return rc

    rec.update(status='done', finished=now_iso(), returncode=0)
    if not cfg.dry_run:
        save_state(cfg.state_path, state)
    print(f'{G}✓ stage {stage.name} done{NC}' + (f'  — good: {stage.good}' if stage.good else ''))
    return 0


def run_stage(cfg: Config, stage: Stage, state: dict) -> int:
    if cfg.multi and stage.name in ('train', 'gate'):
        return run_candidates_stage(cfg, stage, state)
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
        problem = prospective_problem(cfg.repo / cfg.prospective_json) if stage.name == 'prospective' else None
        if problem:
            rec.update(status='failed', finished=now_iso(), returncode=0, error=problem)
            save_state(cfg.state_path, state)
            print(f'{R}✗ stage {stage.name} finished but its output is not usable: {problem}{NC}')
            print(f'  log     : {cfg.rel(log)}')
            return 1
    rec.update(status='done', finished=now_iso(), returncode=0)
    if stage.name == 'promote' and not cfg.dry_run:
        _archive_candidate(cfg, state, rec)
    if not cfg.dry_run:
        save_state(cfg.state_path, state)
    print(f'{G}✓ stage {stage.name} done{NC}' + (f'  — good: {stage.good}' if stage.good else ''))
    if stage.name == 'prospective' and not cfg.dry_run:
        headline = prospective_headline(cfg.repo / cfg.prospective_json)
        if headline:
            print(f'  {headline}')
        if prospective_predates_age_priority(cfg.repo / cfg.prospective_json):
            # defensive only: a real eval_prospective.py older than --index / --priority fails the stage
            # (argparse exit 2). A JSON without the keys (e.g. a hand-copied old file) is not a failure
            # (nothing here is a gate), but never silent: a --resume runs it again
            print(f'{Y}  ⚠ {cfg.prospective_json} has no age_new / priority_new: it was not written by the '
                  f'current scripts/eval_prospective.py — the age baseline is missing from this release{NC}')
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
        'Nothing to paste: the installer CI finds the release brain-triage-<identity12> in the models '
        f'repository ({cfg.github or GITHUB_MODELS_REPO}; repository variable MODEL_BUNDLE_REPO) by the identity '
        'the committed bump pins. Keep the MODEL_BUNDLE_TOKEN secret (repo Settings → Environments → '
        'MODEL_BUNDLE_URL) able to read that repository; the MODEL_BUNDLE_URL secret is only a fallback.',
        'Review the bump: git status && git diff — then git add ONLY the files the bump stage listed '
        f'(never data/ or models/) && git commit -m "release: {cfg.tag} — bump validated numbers"',
        f'EVERY release bumps "version" in desktop-app/package.json (currently {ver}; a model-only release '
        f'raises the minor version: {ver} → {nxt}) so the tag v<version> is new. Bump the SAME version in '
        f'src/inference/server.py APP_VERSION and in the 12 hospital docs '
        f'docs/{{user_manual,admin_install_guide,intended_use,pilot_evaluation_protocol}}_{{en,ru,uz}}.md (the "Version / Версия / '
        f'Versiya" line, plus the /health example and the "Setup {nxt}.exe" name in the admin guides) — '
        f'docs/test_docs_consistency.py::test_app_version_quoted_in_docs_matches_the_code fails otherwise; '
        f'commit it, then '
        f'git tag v{nxt} && git push origin main && git push origin v{nxt}   '
        '(push the bump commit BEFORE the tag: pushing the tag starts the "Build installers" CI workflow, '
        'which restores the model release that commit pins, fails when it is missing, and re-checks it in '
        '"Verify bundle identity matches the code")',
        "Download the installers from the tag's DRAFT GitHub release (an installer that contains the model "
        'is never a workflow artifact of the public repository).',
        'Run docs/windows_laptop_test_en.md (RU: docs/windows_laptop_test_ru.md) on a Windows laptop, '
        'Wi-Fi off; tick every box.',
        'Do NOT publish the draft release while it holds the full installer: on the public code '
        'repository that would make the model downloadable by anyone. Hand the installer to the clinic '
        'directly, with the new sensitivity / specificity and the model identity in the hand-over notes.',
        f'Read the prospective lines before you describe the model ({cfg.prospective_json}: age_new, '
        f'priority_new; {cfg.priority_csv}): a model that does not beat the age-alone AUC is not evidence of '
        'detection — say so in the hand-over notes. The priority rules are provisional '
        f'(their version is in the rules_version column of {cfg.priority_csv}).',
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
    print(f'# report sources (oldest delivery first): {" ".join(cfg.report_sources) or "(none)"}')
    print('# modality rule (build): ' + (f'none (--modality {MODALITY_ANY})' if cfg.modality == MODALITY_ANY
                                         else f'{cfg.modality} only (--modality {cfg.modality})'))
    print(f'# prospective: scores the model shipped today ({BASE_MODEL}) before this release replaces it; '
          f'writes {cfg.priority_csv} (priority findings from the reports) and prints the age-alone AUC and '
          'the priority-finding counts for never-seen patients (informational, not a gate); priority from '
          + (f'{cfg.modality} reports' if cfg.modality != MODALITY_ANY else 'MR reports only (--modality any)'))
    if cfg.multi:
        print('# candidates: ' + '; '.join(
            f'{c} from {cfg.candidate_base(c)}{" (--base-model-fresh)" if c == "fresh" else ""} → '
            f'{cfg.candidate_path(c)}' for c in cfg.candidates))
        print('# promote moves the best candidate that PASSED the gate with promotion.promote = true '
              '(balanced accuracy, tie → warm); the runner substitutes it for <chosen>. As a shell script '
              'this plan stops at the first gate verdict (exit 2/4).')
    print('set -e')
    print(f'cd {shlex.quote(str(cfg.repo))}')
    for i, s in enumerate(stages):
        print(f'\n# --- stage {i}: {s.name} — {s.title}')
        for t in guard_text(cfg, s.name):
            print(f'#     guard: {t}')
        for c in s.cmds:
            print(shlex.join(c))
    print('\n# --- manual: commit + push the bump (no secret to edit: CI finds the model by the pinned identity; '
          'MODEL_BUNDLE_TOKEN must read the models repo), bump desktop-app/package.json version '
          '(every release) + server APP_VERSION + the version lines of the 12 hospital docs, '
          'git tag v<new app version>, git push tag, download the installer from the DRAFT release, '
          'run docs/windows_laptop_test_*.md, never publish the draft release (public repository)')
    return 0


def cmd_status(cfg: Config, stages: list[Stage]) -> int:
    state = load_state(cfg.state_path)
    print(f'{B}release {cfg.tag} — batch {cfg.batch} — {cfg.rel(cfg.state_path)}{NC}')
    if not state.get('stages'):
        print(f'{Y}  no stage has run yet{NC}')
    col = {'done': G, 'archived': G, 'failed': R, 'running': Y}
    for s in stages:
        rec = state.get('stages', {}).get(s.name)
        if rec is None and s.name in FROZEN_AFTER_PROMOTE and archived_candidate(state):
            print(f'  {s.name:<8} skipped ({NOT_APPLICABLE_AFTER_PROMOTE})')
            continue
        if rec is None:
            print(f'  {s.name:<8} {"pending":<8}')
            continue
        st = rec.get('status', '?')
        print(f'  {s.name:<8} {col.get(st, "")}{st:<8}{NC} {rec.get("started") or "":<20} '
              f'{rec.get("finished") or "":<20} {rec.get("log", "")}')
        if st == 'failed':
            print(f'           {R}{rec.get("verdict") or rec.get("failed_cmd") or rec.get("error", "")}{NC}')
        if s.name == 'prospective' and st == 'done' and \
                prospective_predates_age_priority(cfg.repo / cfg.prospective_json):
            note = ('frozen after promote' if archived_candidate(state)
                    else 'a --resume runs it again to add them')
            print(f'           {Y}{PREDATES_AGE_PRIORITY} — {note}{NC}')
    if cfg.multi or state.get('candidates'):
        _print_candidate_table(cfg, state)
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
    if cfg.multi:
        cfg.chosen = (state.get('candidate_choice') or {}).get('chosen')
        if cfg.candidates:
            print('candidates: ' + ', '.join(f'{c} ({cfg.candidate_base(c)})' for c in cfg.candidates)
                  + (f' — chosen by the last gate: {cfg.chosen}' if cfg.chosen else ''))
    for s in selected:
        if s.name == 'promote' and cfg.multi:
            # the gate (possibly in this very run) chose the candidate: rebuild promote for it
            cfg.chosen = (state.get('candidate_choice') or {}).get('chosen')
            if not cfg.chosen and not archived_candidate(state) and not cfg.dry_run:
                print(f'{R}✗ promote: the gate has not chosen a candidate (state.json candidate_choice) — '
                      f'run the gate stage first{NC}')
                print(f'  {_resume_hint(cfg)}')
                return EXIT_USAGE
            if archived_candidate(state) and not cfg.chosen:
                cfg.chosen = (state['stages']['promote'].get('chosen_candidate'))
            s = next(x for x in build_stages(cfg) if x.name == 'promote')
        if cfg.resume:
            done, why = stage_is_done(cfg, s, state, explicit)
            if done:
                print(f'{G}↷ stage {s.name} skipped ({why}){NC}' if why else
                      f'{G}↷ stage {s.name} already done — skipped{NC}')
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
        if s.name in NEEDS_ROOT and not cfg.dry_run and cfg.root is not None and not cfg.root.is_dir():
            # the stages before it that need no drive (parse, label, match) have run: stop cleanly here
            print(f'{R}✗ drive not mounted: {cfg.root} (needed by stage {s.name}){NC}')
            print(f'  plug in the drive, then: {_resume_hint(cfg)}')
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
                # a verdict is not a crash: re-running unchanged inputs gives the same verdict, so the
                # resume line is for AFTER new data / labels — it re-runs only the stages whose inputs
                # changed (or that are not done), then gates again; nothing is promoted on a failed gate
                print(f'  after adding data or relabelling, continue with (re-runs only changed stages, '
                      f'then the gate): {_resume_hint(cfg)}')
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
    ap.add_argument('--root', default=None,
                    help='drive root with the study folders (index/build read it; match gets --images when given)')
    ap.add_argument('--reports', nargs='+', action='extend', default=None, metavar='SRC',
                    help='report sources for the parse stage, OLDEST delivery first: folders of report PDFs '
                         'and/or .xlsx/.csv/.docx files (repeating the flag appends; the order is part of '
                         'the parse fingerprint)')
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
    ap.add_argument('--candidates', type=parse_candidates, default=None, metavar='warm[,fresh]',
                    help='models to train + gate (default warm). warm starts from --base-model; fresh from '
                         f'{PUBLIC_VIT_BASE} (else {PUBLIC_VIT_HUB_ID} from the HF cache) with a new head. '
                         'promote takes the best that passed the gate by balanced accuracy (tie → warm)')
    ap.add_argument('--modality', default=None,
                    help=f'build only studies of this modality (default {DEFAULT_MODALITY}: the MR-only rule; '
                         f'"{MODALITY_ANY}" disables the rule)')
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
    # a state.json written before --reports took several sources stores ONE string
    reports = [o['reports']] if isinstance(o['reports'], str) else list(o['reports'] or [])
    if 'reports' in inherited:
        inherited['reports'] = reports
    llm_labels = o['llm_labels']
    if llm_labels:
        llm_labels = Config(repo=repo, batch=a.batch, tag=tag).rel(Path(llm_labels).expanduser())
    return Config(
        repo=repo, batch=a.batch, tag=tag,
        root=Path(o['root']).expanduser() if o['root'] else None,
        reports=[Path(r).expanduser() for r in reports],
        extra_sets=list(o['extra_sets']), holdout=o['holdout'], train_lock=o['train_lock'],
        epochs=int(o['epochs']), device=o['device'], python=o['python'], base_model=o['base_model'],
        skip_benchmark=bool(o['skip_benchmark']), skip_tests=bool(o['skip_tests']),
        github=None if o['local_bundle'] else o['github'], label_llm=o['label_llm'], llm_labels=llm_labels,
        min_normals=int(o['min_normals']), min_disk_gb=float(o['min_disk_gb']),
        candidates=[c for c in CANDIDATE_KINDS if c in (o['candidates'] or DEFAULT_CANDIDATES)],
        modality=str(o['modality']),
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

    needs_root = set(NEEDS_ROOT)
    needs_reports = {'parse'}
    wanted = {s.name for s in selected}
    if wanted & needs_root and cfg.root is None:
        print(f'{R}✗ --root is required for stages {sorted(wanted & needs_root)}{NC}')
        return EXIT_USAGE
    if wanted & needs_reports and not cfg.reports:
        print(f'{R}✗ --reports is required for the parse stage{NC}')
        return EXIT_USAGE
    if not cfg.dry_run and wanted & needs_root and not cfg.root.is_dir():
        # a stage that --resume will skip does not need the drive. Refuse up front only when the
        # first stage to run needs it; otherwise the stages before it run and cmd_run stops at it.
        state = load_state(cfg.state_path) if cfg.resume else {}
        first = next((s for s in selected if not (cfg.resume and stage_is_done(cfg, s, state, explicit)[0])), None)
        if first is not None and first.name in needs_root:
            print(f'{R}✗ drive not mounted: {cfg.root} (needed by stage {first.name}){NC}')
            return EXIT_USAGE

    return cmd_run(cfg, stages, selected, explicit)


if __name__ == '__main__':
    sys.exit(main())
