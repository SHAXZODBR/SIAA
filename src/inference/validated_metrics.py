"""
================================================================================
  SENTINEL — VALIDATED TRIAGE METRICS (single source of truth for the numbers)
================================================================================
  WHAT
    Every number the product quotes about the validated brain-triage model —
    sensitivity / specificity / accuracy, the confusion counts, how many
    held-out patients and slices, the evaluation date, the MANIFEST identity,
    the release-gate floors and the operating threshold — comes from ONE
    place: this module. It reads `study_level_eval.json` + `MANIFEST.json`
    next to the model dir at import time and exposes them as constants plus a
    few formatting helpers (`pct`, `dec`, `ci`) and the localized sentence
    fragments the registry card, brain_analysis and the report templates use.

  WHY
    Before this module the same figures were typed by hand into the registry
    notes, the report templates (RU/UZ/EN), brain_analysis labels, twelve
    markdown documents and two tests. After a re-train they drifted. Now the
    runtime strings are generated from the JSON and `scripts/bump_model_release.py`
    rewrites the docs (and the FALLBACK block below) from the same values, so
    `--check` in the docs consistency test catches any stale copy.

  MODEL DIR RESOLUTION (mirrors src/inference/model_registry.py::_local_dirs)
    1. env SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE
    2. <MODELS_DIR>/brain_triage_finetuned, where MODELS_DIR is
       env SENTINEL_MODELS_DIR or <repo>/models (src/utils/paths.py).

  FALLBACK
    A frozen installer without the model dir (or a checkout without the
    weights) must still import: when either JSON file is missing or malformed
    the constants of the LAST RELEASE (the FALLBACK block below — kept in sync
    by scripts/bump_model_release.py) are used and SOURCE == 'fallback'.
    Importing never raises, imports no torch/transformers and does no I/O
    beyond reading the two small JSON files.

  Usage:
    from src.inference import validated_metrics as vm
    vm.SENS, vm.SPEC, vm.N_STUDIES, vm.IDENTITY, vm.EVALUATED_ON
    vm.pct(vm.SENS)          -> '90%'
    vm.dec(vm.SPEC, 'ru')    -> '0,47'
    vm.triage_note('uz')     -> "klinika ma'lumotlarida validatsiya qilingan: ..."
    vm.load(Path('models/brain_triage_candidate_x'))  -> Metrics for another dir
================================================================================
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- paths

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
MODEL_DIR_NAME = 'brain_triage_finetuned'
EVAL_JSON = 'study_level_eval.json'
MANIFEST_JSON = 'MANIFEST.json'


def models_dir() -> Path:
    """Same convention as src/utils/paths.MODELS_DIR (env SENTINEL_MODELS_DIR,
    else <repo>/models) — computed here so this module has no src imports."""
    return Path(os.environ.get('SENTINEL_MODELS_DIR') or REPO_ROOT / 'models').expanduser()


def resolve_model_dir() -> Path:
    """Where the validated triage release lives (env override first, like the
    registry's SENTINEL_MODEL_DIR_OVERRIDE_<KEY> lookup)."""
    override = os.environ.get('SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE', '').strip()
    if override:
        return Path(override).expanduser()
    return models_dir() / MODEL_DIR_NAME


# --------------------------------------------------------------------------- data

@dataclass(frozen=True)
class Metrics:
    sens: float
    spec: float
    acc: float
    tp: int
    fn: int
    tn: int
    fp: int
    n_studies: int
    n_slices: int
    evaluated_on: str          # YYYY-MM-DD
    identity: str              # first 12 hex of sha256(model.safetensors)
    min_sens: float
    min_spec: float
    threshold: float
    sens_ci: tuple[float, float]
    spec_ci: tuple[float, float]
    source: str                # 'json' | 'fallback'
    model_dir: str
    # False when the eval JSON itself says the model did NOT earn release: gate.passed is not
    # true, or promotion.promote is false (review round 2). The app then downgrades the triage
    # detector from 'validated' (brain_analysis.analyze_brain_study); strict loads refuse.
    validated: bool = True
    not_validated_reason: str = ''

    @property
    def n_abnormal(self) -> int:
        return self.tp + self.fn

    @property
    def n_normal(self) -> int:
        return self.tn + self.fp

    def as_dict(self) -> dict:
        d = asdict(self)
        d['n_abnormal'] = self.n_abnormal
        d['n_normal'] = self.n_normal
        return d


# ---- FALLBACK (last release) — rewritten by scripts/bump_model_release.py ----
# bump:fallback-begin
FALLBACK_SENS = 0.9007633587786259
FALLBACK_SPEC = 0.46808510638297873
FALLBACK_ACC = 0.7865168539325843
FALLBACK_TP, FALLBACK_FN, FALLBACK_TN, FALLBACK_FP = 118, 13, 22, 25
FALLBACK_N_STUDIES = 178
FALLBACK_N_SLICES = 890
FALLBACK_EVALUATED_ON = '2026-09-07'
FALLBACK_IDENTITY = '0d559766ce58'
FALLBACK_MIN_SENS = 0.85
FALLBACK_MIN_SPEC = 0.4
FALLBACK_THRESHOLD = 0.5
# bump:fallback-end


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials (0,0 → (0,0))."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _fallback(model_dir: Path) -> Metrics:
    return Metrics(
        sens=FALLBACK_SENS, spec=FALLBACK_SPEC, acc=FALLBACK_ACC,
        tp=FALLBACK_TP, fn=FALLBACK_FN, tn=FALLBACK_TN, fp=FALLBACK_FP,
        n_studies=FALLBACK_N_STUDIES, n_slices=FALLBACK_N_SLICES,
        evaluated_on=FALLBACK_EVALUATED_ON, identity=FALLBACK_IDENTITY,
        min_sens=FALLBACK_MIN_SENS, min_spec=FALLBACK_MIN_SPEC, threshold=FALLBACK_THRESHOLD,
        sens_ci=wilson_ci(FALLBACK_TP, FALLBACK_TP + FALLBACK_FN),
        spec_ci=wilson_ci(FALLBACK_TN, FALLBACK_TN + FALLBACK_FP),
        source='fallback', model_dir=str(model_dir),
    )


def _ci_from(gate: dict, key: str, k: int, n: int) -> tuple[float, float]:
    ci = gate.get(key)
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        try:
            return (float(ci[0]), float(ci[1]))
        except (TypeError, ValueError):
            pass
    return wilson_ci(k, n)


def _threshold_from(ev: dict, gate: dict) -> float:
    for key in ('threshold', 'thr'):
        if isinstance(gate.get(key), (int, float)):
            return float(gate[key])
    name = str(gate.get('name', ''))
    if '@' in name:
        try:
            return float(name.split('@', 1)[1].split()[0])
        except ValueError:
            pass
    return FALLBACK_THRESHOLD


def _identity_from(model_dir: Path, ev: dict) -> Optional[str]:
    """First 12 hex of sha256(model.safetensors): MANIFEST.json first, else the
    manifest block the eval JSON carries."""
    files: dict = {}
    man_path = model_dir / MANIFEST_JSON
    if man_path.is_file():
        try:
            files = json.loads(man_path.read_text(encoding='utf-8')).get('files', {}) or {}
        except (OSError, ValueError):
            files = {}
    if not files:
        files = (ev.get('manifest') or {}).get('files', {}) or {}
    for name in ('model.safetensors', 'pytorch_model.bin', 'model.pt'):
        h = files.get(name)
        if isinstance(h, str) and len(h) >= 12:
            return h[:12].lower()
    return None


def _not_validated_reason(ev: dict) -> str:
    """'' when the eval JSON records a passed gate (and, if compared, a promotion); else why not."""
    gate = ev.get('gate') or {}
    if gate.get('passed') is not True:
        return f"gate.passed is {gate.get('passed')!r}"
    promo = ev.get('promotion')
    if isinstance(promo, dict) and promo.get('promote') is False:
        return f"promotion.promote is false ({promo.get('reason', '')})"
    return ''


def load(model_dir: Optional[Path] = None, *, strict: bool = False) -> Metrics:
    """Metrics of the release in `model_dir` (default: resolve_model_dir()).

    strict=False (import-time behaviour): any missing/malformed file → fallback; an eval whose
    gate did not pass (or whose comparison refused promotion) loads with validated=False.
    strict=True: raise FileNotFoundError / ValueError instead — also for a not-validated eval
    (scripts/bump_model_release.py uses it, so docs never call a failed model 'validated')."""
    model_dir = Path(model_dir) if model_dir is not None else resolve_model_dir()
    ev_path = model_dir / EVAL_JSON
    try:
        if not ev_path.is_file():
            raise FileNotFoundError(f'{ev_path} not found')
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
        gate = ev['gate']
        tp, fn, tn, fp = (int(gate[k]) for k in ('tp', 'fn', 'tn', 'fp'))
        identity = _identity_from(model_dir, ev)
        if identity is None:
            raise ValueError(f'no model.safetensors hash in {model_dir / MANIFEST_JSON} or {ev_path}')
        evaluated_on = str(ev.get('evaluated_at', ''))[:10]
        if len(evaluated_on) != 10:
            raise ValueError(f'{ev_path}: evaluated_at missing or not ISO')
        why = _not_validated_reason(ev)
        if why and strict:
            raise ValueError(f'{ev_path}: not a validated release — {why}')
        return Metrics(
            sens=float(gate['sens']), spec=float(gate['spec']), acc=float(gate['acc']),
            tp=tp, fn=fn, tn=tn, fp=fp,
            n_studies=int(ev.get('n_studies', tp + fn + tn + fp)),
            n_slices=int(ev.get('n_slices', 0)),
            evaluated_on=evaluated_on, identity=identity,
            min_sens=float(gate.get('min_sens', FALLBACK_MIN_SENS)),
            min_spec=float(gate.get('min_spec', FALLBACK_MIN_SPEC)),
            threshold=_threshold_from(ev, gate),
            sens_ci=_ci_from(gate, 'sens_ci', tp, tp + fn),
            spec_ci=_ci_from(gate, 'spec_ci', tn, tn + fp),
            source='json', model_dir=str(model_dir),
            validated=not why, not_validated_reason=why,
        )
    except Exception:
        if strict:
            raise
        return _fallback(model_dir)


# --------------------------------------------------------------------------- module constants

METRICS: Metrics = load()
MODEL_DIR: Path = Path(METRICS.model_dir)
SOURCE: str = METRICS.source

SENS: float = METRICS.sens
SPEC: float = METRICS.spec
ACC: float = METRICS.acc
TP: int = METRICS.tp
FN: int = METRICS.fn
TN: int = METRICS.tn
FP: int = METRICS.fp
N_STUDIES: int = METRICS.n_studies
N_ABNORMAL: int = METRICS.n_abnormal
N_NORMAL: int = METRICS.n_normal
N_SLICES: int = METRICS.n_slices
EVALUATED_ON: str = METRICS.evaluated_on
IDENTITY: str = METRICS.identity
MIN_SENS: float = METRICS.min_sens
MIN_SPEC: float = METRICS.min_spec
THRESHOLD: float = METRICS.threshold
SENS_CI: tuple[float, float] = METRICS.sens_ci
SPEC_CI: tuple[float, float] = METRICS.spec_ci


# --------------------------------------------------------------------------- formatting

def pct(x: float) -> str:
    """0.9007 → '90%'."""
    return f'{int(round(x * 100)):d}%'


def dec(x: float, lang: str = 'en', digits: int = 2) -> str:
    """0.9007 → '0.90' (en) / '0,90' (ru, uz — decimal comma)."""
    s = f'{x:.{digits}f}'
    return s.replace('.', ',') if lang in ('ru', 'uz') else s


def ci(interval: tuple[float, float], lang: str = 'en', digits: int = 2) -> str:
    """(0.333, 0.608) → '0.33–0.61' (en dash, decimal comma for ru/uz)."""
    return f'{dec(interval[0], lang, digits)}–{dec(interval[1], lang, digits)}'


def miss_one_in(m: Metrics = None) -> int:
    """'about 1 in N abnormal studies is not flagged' — N = round(1 / (1 − sens))."""
    sens = (m or METRICS).sens
    return int(round(1.0 / (1.0 - sens))) if sens < 1.0 else 0


# --------------------------------------------------------------------------- localized fragments

_VALIDATED_FRAGMENT = {
    'ru': 'валидирована на данных клиники: чувствительность {sens}, специфичность {spec}',
    'uz': "klinika ma'lumotlarida validatsiya qilingan: sezuvchanlik {sens}, spetsifiklik {spec}",
    'en': "validated on this clinic's data: sensitivity {sens}, specificity {spec}",
}
_SENS_FRAGMENT = {
    'ru': 'чувствительность {sens}',
    'uz': 'sezuvchanlik {sens}',
    'en': 'sensitivity {sens}',
}


def triage_note(lang: str = 'en', m: Metrics = None) -> str:
    """The parenthetical the report templates / registry quote, e.g. (en)
    "validated on this clinic's data: sensitivity 90%, specificity 47%"."""
    m = m or METRICS
    tpl = _VALIDATED_FRAGMENT.get(lang, _VALIDATED_FRAGMENT['en'])
    return tpl.format(sens=pct(m.sens), spec=pct(m.spec))


def sensitivity_note(lang: str = 'en', m: Metrics = None) -> str:
    """'sensitivity 90%' / 'чувствительность 90%' / 'sezuvchanlik 90%'."""
    m = m or METRICS
    tpl = _SENS_FRAGMENT.get(lang, _SENS_FRAGMENT['en'])
    return tpl.format(sens=pct(m.sens))


def registry_notes(m: Metrics = None) -> str:
    """BRAIN_TRIAGE ModelCard.notes (English, shown in the app's model list)."""
    m = m or METRICS
    return (f'Trained AND validated on local patient studies: study-level sensitivity {dec(m.sens)} / '
            f'specificity {dec(m.spec)} at mean-prob {m.threshold:g} over {m.n_studies} held-out local '
            f'patients. Triage aid — prioritizes studies for radiologist review; never certifies a scan '
            f'as normal.')


def summary_line(m: Metrics = None) -> str:
    """One-line human summary (scripts print it)."""
    m = m or METRICS
    return (f'sens {dec(m.sens)} / spec {dec(m.spec)} / acc {dec(m.acc)} on {m.n_studies} studies '
            f'({m.n_abnormal} abnormal, {m.n_normal} normal; TP {m.tp} FN {m.fn} TN {m.tn} FP {m.fp}), '
            f'{m.n_slices} slices, evaluated {m.evaluated_on}, identity {m.identity} [{m.source}]')


__all__ = [
    'Metrics', 'METRICS', 'MODEL_DIR', 'SOURCE', 'load', 'resolve_model_dir', 'models_dir', 'wilson_ci',
    'SENS', 'SPEC', 'ACC', 'TP', 'FN', 'TN', 'FP', 'N_STUDIES', 'N_ABNORMAL', 'N_NORMAL', 'N_SLICES',
    'EVALUATED_ON', 'IDENTITY', 'MIN_SENS', 'MIN_SPEC', 'THRESHOLD', 'SENS_CI', 'SPEC_CI',
    'pct', 'dec', 'ci', 'miss_one_in', 'triage_note', 'sensitivity_note', 'registry_notes', 'summary_line',
]
