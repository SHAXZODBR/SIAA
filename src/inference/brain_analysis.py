"""
================================================================================
  SENTINEL — UNIFIED BRAIN MRI ANALYSIS (multi-finding detector panel)
================================================================================

  A brain MRI is not one question ("is there a glioma?"). A radiologist checks
  for MANY things: tumor, atrophy, stroke, white-matter disease, hemorrhage,
  hydrocephalus. So Sentinel runs a PANEL of detectors over one study and
  aggregates their findings — instead of a single 4-class tumor call.

  Honesty by design:
    Each detector declares a VALIDATION STATUS. We never present an
    unvalidated model's output as a confirmed diagnosis:

      • 'validated'     — checked against ground-truth labels on local data
      • 'pending'       — pretrained, calibrated-pending, plausible but UNVERIFIED
                          on this clinic's population (radiologist confirms)
      • 'experimental'  — pretrained on a different task/population; a screening
                          hint only, high false-positive risk

  Adding a new pathology = add one Detector entry. As the clinic collects
  labels (tumor/no-tumor, stroke y/n, …), detectors graduate pending→validated.

  This module deliberately ROUTES the right sequence to the right detector
  (T1ce/T1 axial for tumor, T1 for atrophy) and GATES non-brain studies out —
  reusing the fixes in brain_mri_preprocessor.py.
================================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from loguru import logger

from src.pipeline.brain_mri_preprocessor import (
    process_brain_study, BrainStudy, central_band_best_slice, robust_normalize_slice,
)


# ----------------------------------------------------------------------------
# FINDING CATALOG — what the brain panel is DESIGNED to cover, with status.
# This is the single source of truth for "what can Sentinel detect in brain".
# ----------------------------------------------------------------------------
BRAIN_FINDING_CATALOG = [
    # key              human label                       status         model wired?
    ('tumor_class',    'Brain tumor (glioma/meningioma/pituitary)', 'pending',      True),
    ('tumor_seg_3d',   'Tumor segmentation + volume (3D BraTS)',    'pending',      True),
    ('stroke',         'Ischemic stroke staging (DWI)',             'experimental', True),
    ('atrophy',        'Cerebral atrophy / dementia pattern',       'experimental', True),
    # --- roadmap: a loadable model exists but isn't wired yet (see
    #     scripts/download_brain_models.sh) or needs custom load / labels ---
    ('hemorrhage',     'Intracranial hemorrhage (CT via head_ct)',  'roadmap',      False),
    ('white_matter',   'White-matter disease / MS plaques (FLAIR)', 'roadmap',      False),
    ('hydrocephalus',  'Ventricular enlargement / hydrocephalus',   'roadmap',      False),
]


@dataclass
class Detector:
    key: str
    label: str
    status: str                 # validated | pending | experimental
    model_key: str              # registry key
    sequence_pref: tuple        # which sequences this detector wants (in order)
    # maps raw model class -> (is_positive, human finding text). None text = skip.
    interpret: Callable


def _slice_for(study: BrainStudy, prefs: tuple) -> Optional[dict]:
    """Pick a normalized 224-ready axial slice for the first available preferred
    sequence (axial only). Returns {sequence, slice} or None."""
    def is_axial(k):
        return (study.seq_meta.get(k, {}).get('orientation') or 'axial') == 'axial'
    for pref in prefs:
        if pref in study.sequences and is_axial(pref):
            vol = study.sequences[pref]
            slc = central_band_best_slice(vol) if vol.ndim == 3 else vol
            return {'sequence': pref, 'slice': robust_normalize_slice(slc)}
    # fall back to the study's own best choice
    sel = study.select_classifier_input()
    if sel and sel.get('slice') is not None:
        return {'sequence': sel['sequence'], 'slice': sel['slice']}
    return None


def _to_model_input(slc: np.ndarray) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((np.clip(slc, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0


# ---- interpreters: turn a model's class probs into findings ----------------

def _interpret_tumor(probs: dict) -> list[dict]:
    probs = {k: v for k, v in probs.items() if not k.startswith('_')}
    if not probs:
        return []
    top, conf = max(probs.items(), key=lambda kv: kv[1])
    if top == 'no_tumor':
        return [{'positive': False, 'label': 'No tumor detected', 'class': top, 'confidence': float(conf)}]
    pretty = top.replace('_', ' ')
    return [{'positive': True, 'label': f'Suspected {pretty}', 'class': top, 'confidence': float(conf)}]


def _interpret_atrophy(probs: dict) -> list[dict]:
    probs = {k: v for k, v in probs.items() if not k.startswith('_')}
    if not probs:
        return []
    top, conf = max(probs.items(), key=lambda kv: kv[1])
    if top == 'Non_Demented':
        return [{'positive': False, 'label': 'No atrophy/dementia pattern', 'class': top, 'confidence': float(conf)}]
    pretty = top.replace('_', ' ')
    return [{'positive': True, 'label': f'Atrophy pattern: {pretty}', 'class': top, 'confidence': float(conf)}]


# Stroke model labels are Turkish / vary by repo. Map any of them to a clean
# acute/subacute/chronic-or-normal verdict so the interpreter is robust to the
# exact class strings the downloaded weights expose.
_STROKE_NEGATIVE = {'normalkronik', 'normal_or_chronic', 'normal', 'inme yok', 'inmeyok', 'no_stroke', 'kronik'}

def _interpret_stroke(probs: dict) -> list[dict]:
    probs = {k: v for k, v in probs.items() if not k.startswith('_')}
    if not probs:
        return []
    top, conf = max(probs.items(), key=lambda kv: kv[1])
    norm = str(top).lower().replace(' ', '').replace('-', '')
    if norm in _STROKE_NEGATIVE:
        return [{'positive': False, 'label': 'No acute/subacute infarct', 'class': top, 'confidence': float(conf)}]
    pretty = {
        'hiperakutakut': 'Acute / hyperacute infarct',
        'acute_infarct': 'Acute / hyperacute infarct',
        'subakut': 'Subacute infarct',
        'subacute_infarct': 'Subacute infarct',
        'inmevar': 'Stroke present',
    }.get(norm, f'Possible infarct: {str(top).replace("_", " ")}')
    return [{'positive': True, 'label': pretty, 'class': top, 'confidence': float(conf)}]


DETECTORS = [
    Detector('tumor_class', 'Brain tumor', 'pending', 'brain_tumor_class',
             ('T1ce', 'T1', 'T2', 'FLAIR'), _interpret_tumor),
    # Stroke wants diffusion (DWI), then ADC/FLAIR. Skipped automatically until
    # the weights are downloaded (get_model returns unavailable -> detector off).
    Detector('stroke', 'Ischemic stroke', 'experimental', 'brain_stroke',
             ('DWI', 'ADC', 'FLAIR', 'T2'), _interpret_stroke),
    Detector('atrophy', 'Cerebral atrophy / dementia', 'experimental', 'brain_dementia',
             ('T1', 'T1ce', 'T2'), _interpret_atrophy),
]


def analyze_brain_study(file_paths: list, device: str = 'cpu',
                        include_experimental: bool = True) -> dict:
    """Run the brain detector panel over one study.

    Returns a dict with: rejected/non_brain, sequences_present, findings (each
    tagged with status + the detector + sequence used), and a roadmap of
    not-yet-wired finding types so the UI can show honest coverage.
    """
    from src.inference.model_registry import get_model

    study = process_brain_study([Path(p) for p in file_paths])
    if study.num_sequences == 0:
        return {'error': 'No readable DICOM image series in study', 'findings': []}

    non_brain = study.non_brain_reason
    if non_brain is not None:
        return {
            'rejected': True, 'requires_review': True, 'non_brain_reason': non_brain,
            'sequences_present': list(study.sequences.keys()), 'findings': [],
            'note': f'Not analyzed by the brain panel: {non_brain}.',
        }

    findings = []
    detectors_run = []
    for det in DETECTORS:
        if det.status == 'experimental' and not include_experimental:
            continue
        entry = get_model(det.model_key, device=device)
        if not entry or not entry.get('available'):
            logger.debug(f"Detector {det.key} unavailable: {(entry or {}).get('reason','')}")
            continue
        picked = _slice_for(study, det.sequence_pref)
        if picked is None:
            continue
        try:
            probs = entry['predictor'](_to_model_input(picked['slice']))
        except Exception as e:
            logger.warning(f"Detector {det.key} failed: {e}")
            continue
        detectors_run.append(det.key)
        for f in det.interpret(probs):
            findings.append({
                'detector': det.key,
                'finding': f['label'],
                'class_name': f.get('class'),
                'confidence': round(f['confidence'], 4),
                'positive': f['positive'],
                'status': det.status,         # pending / experimental / validated
                'sequence_used': picked['sequence'],
            })

    # Surface positives first, then by confidence
    findings.sort(key=lambda f: (not f['positive'], -f['confidence']))

    # OR-ensemble safety net: there is NO "normal" model, so we can NEVER clear a
    # scan as normal — only flag the specific findings the panel detects. If any
    # detector fires, mark abnormal; if none fire, say explicitly that this is
    # NOT a normal read (out-of-scope pathology would be invisible to the panel).
    positives = [f for f in findings if f.get('positive')]
    covered = [d.label for d in DETECTORS if d.key in detectors_run]
    overall_assessment = {
        'abnormal_flagged': bool(positives),
        'flags': [f['finding'] for f in positives],
        'text': (
            'ABNORMAL — flagged for review: ' + '; '.join(f['finding'] for f in positives)
            if positives else
            'No finding flagged by the panel. This is NOT a normal read — the panel only '
            f'detects [{", ".join(covered)}]. Any other pathology requires full radiologist review.'
        ),
    }

    return {
        'rejected': False,
        'sequences_present': list(study.sequences.keys()),
        'detectors_run': detectors_run,
        'findings': findings,
        'overall_assessment': overall_assessment,
        'has_brats_quartet': study.has_brats_quartet,
        'patient_id': study.patient_id,
        'study_uid': study.study_uid,
        'coverage': [
            {'key': k, 'label': lbl, 'status': st, 'wired': wired}
            for (k, lbl, st, wired) in BRAIN_FINDING_CATALOG
        ],
        'disclaimer': (
            'Triage assistant: flags only the listed findings — it CANNOT certify a scan '
            'as normal. Findings marked "pending" are unvalidated on this clinic\'s population; '
            '"experimental" are screening hints. A radiologist reads and signs every study.'
        ),
    }
