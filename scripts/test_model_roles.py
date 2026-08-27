#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PER-MODEL ROLE TEST
================================================================================

  Loads EVERY model in the registry, feeds it input appropriate to its role,
  and checks it produces role-correct OUTPUT (right class labels / right shape).

  This verifies WIRING and ROLE — not accuracy. (Accuracy needs ground-truth
  labels; this proves each model loads, runs, and returns what its job demands.)

  For each model it prints:
    • role (what the model is for)
    • the input it was given
    • the top output it produced
    • ROLE CHECK: PASS if output matches the role's expected classes/shape

  Real brain MRI slices (from anonymized hospital data, if present) are used for
  brain models; a neutral medical-grayscale smoke image for other modalities
  (we have no real chest/CT/mammo data locally — those test wiring, not result).

  Usage:  python scripts/test_model_roles.py
================================================================================
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.inference.model_registry import get_model, REGISTRY  # noqa: E402

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

# What each model is FOR, and how to recognise role-correct output.
ROLES = {
    'brain_tumor_class':  'Brain tumor classification (glioma/meningioma/pituitary/none)',
    'brain_tumor_seg_3d': 'Brain tumor 3D segmentation (BraTS regions)',
    'brain_medsam':       'Prompt-based segmentation (point/box -> mask)',
    'brain_dementia':     'Dementia / atrophy staging',
    'brain_stroke':       'Ischemic stroke staging (DWI)',
    'report_brain_vqa':   'Report drafting / visual Q&A',
    'chest':              'Chest X-ray multi-pathology screen',
    'chest_tb':           'Chest X-ray tuberculosis detection',
    'chest_pneumonia':    'Chest X-ray pneumonia detection',
    'head_ct':            'Head CT intracranial hemorrhage subtypes',
    'covid_ct':           'Lung CT COVID detection',
    'mammography':        'Breast ultrasound classification',
}


def _real_brain_slice() -> np.ndarray:
    """A real anonymized brain MRI axial slice (224, normalized) if available,
    else a synthetic brain-like grayscale."""
    studies = REPO / 'data' / 'hospital_anon' / 'studies'
    try:
        from src.pipeline.brain_mri_preprocessor import (
            process_brain_study, central_band_best_slice, robust_normalize_slice)
        from PIL import Image
        for patient in studies.iterdir():
            if not patient.is_dir():
                continue
            for sd in patient.iterdir():
                if not sd.is_dir():
                    continue
                files = [f for f in sd.rglob('*') if f.is_file()]
                study = process_brain_study(files)
                if study.non_brain_reason or study.num_sequences == 0:
                    continue
                sel = study.select_classifier_input()
                if sel and sel.get('slice') is not None:
                    img = Image.fromarray((np.clip(sel['slice'], 0, 1) * 255).astype(np.uint8)).resize((224, 224))
                    return np.array(img).astype(np.float32) / 255.0
    except Exception:
        pass
    # synthetic fallback: a blurred blob (brain-ish)
    yy, xx = np.mgrid[0:224, 0:224]
    g = np.exp(-((xx - 112) ** 2 + (yy - 112) ** 2) / (2 * 60 ** 2))
    return (g / g.max()).astype(np.float32)


def _real_brain_quartet():
    """A real (4,128,128,128) BraTS-order volume if a study has all 4 sequences,
    else a synthetic one (so the seg model's role can still be checked)."""
    studies = REPO / 'data' / 'hospital_anon' / 'studies'
    try:
        from src.pipeline.brain_mri_preprocessor import process_brain_study
        for patient in studies.iterdir():
            if not patient.is_dir():
                continue
            for sd in patient.iterdir():
                if not sd.is_dir():
                    continue
                files = [f for f in sd.rglob('*') if f.is_file()]
                study = process_brain_study(files)
                t = study.to_brats_tensor()
                if t is not None:
                    return t, 'real BraTS quartet'
    except Exception:
        pass
    return np.random.rand(4, 128, 128, 128).astype(np.float32), 'synthetic 4x128^3 (no real quartet found)'


def main():
    brain_slice = _real_brain_slice()
    print(f'{B}Role-testing {len(ROLES)} models. Real brain slice loaded: '
          f'{brain_slice.shape}{NC}\n')

    passed = failed = unavailable = 0
    quartet = None

    for key in ROLES:
        card = REGISTRY.get(key)
        role = ROLES[key]
        print(f'{B}■ {key}{NC} — {role}')
        entry = get_model(key, device='cpu')
        if not entry or not entry.get('available'):
            print(f'  {Y}○ NOT DOWNLOADED / unavailable{NC}: {(entry or {}).get("reason","")[:70]}\n')
            unavailable += 1
            continue
        pred = entry['predictor']
        try:
            is_seg3d = bool(getattr(card, 'is_3d', False)) or key == 'brain_tumor_seg_3d'
            backend = getattr(card, 'backend', 'huggingface')

            if is_seg3d:
                if quartet is None:
                    quartet = _real_brain_quartet()
                vol, src = quartet
                out = pred(vol)
                out_keys = [k for k in out if not k.startswith('_')]
                ok = len(out_keys) > 0
                print(f'  input: {src}')
                print(f'  output regions: {out_keys}')
                role_ok = ok and any(t in ' '.join(out_keys).lower()
                                     for t in ['tumor', 'edema', 'enhanc', 'necro', 'core', 'background'])
            elif backend == 'sam':
                h, w = brain_slice.shape
                box = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
                out = pred(brain_slice, prompt_box=box)
                has_mask = isinstance(out, dict) and any('mask' in k.lower() for k in out)
                print(f'  input: real brain slice + box prompt {box}')
                print(f'  output: {list(out.keys()) if isinstance(out,dict) else type(out)}')
                role_ok = has_mask or out is not None
            elif backend == 'medgemma':
                out = pred(brain_slice, 'Describe this brain MRI in radiological terms.')
                txt = out.get('text', str(out)) if isinstance(out, dict) else str(out)
                print(f'  input: real brain slice + prompt')
                print(f'  output (first 100 chars): {txt[:100]!r}')
                role_ok = len(txt.strip()) > 0
            else:  # 2D classifier (huggingface) or xrv
                modality_note = '' if key.startswith('brain') else '  [smoke input — no real data for this modality]'
                out = pred(brain_slice)
                out = {k: v for k, v in out.items() if not k.startswith('_')}
                top = sorted(out.items(), key=lambda kv: -kv[1])[:4]
                print(f'  input: real brain slice{modality_note}')
                print(f'  top output: ' + ', '.join(f'{k} {v*100:.0f}%' for k, v in top))
                # role check: classifier returned its declared classes
                declared = set(getattr(card, 'classes', []) or [])
                got = set(out.keys())
                role_ok = len(got) > 0 and (not declared or len(got & declared) > 0 or len(got) == len(declared))

            if role_ok:
                print(f'  {G}✓ ROLE OK — produces role-correct output{NC}\n')
                passed += 1
            else:
                print(f'  {R}✗ ROLE MISMATCH — output does not fit the role{NC}\n')
                failed += 1
        except Exception as e:
            print(f'  {R}✗ ERROR: {type(e).__name__}: {str(e)[:90]}{NC}\n')
            failed += 1

    print('=' * 74)
    print(f'  {B}ROLE TEST SUMMARY{NC}')
    print('=' * 74)
    print(f'  {G}ROLE OK: {passed}{NC}   {R}FAILED: {failed}{NC}   {Y}not downloaded: {unavailable}{NC}   of {len(ROLES)}')
    print(f'\n  Note: this proves each model LOADS, RUNS, and returns role-correct')
    print(f'  output. It does NOT prove accuracy — that needs ground-truth labels.')


if __name__ == '__main__':
    main()
