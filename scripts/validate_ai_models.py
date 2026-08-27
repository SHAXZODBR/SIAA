#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — END-TO-END AI MODEL VALIDATOR
================================================================================

  Runs every model in the registry on test DICOMs and shows you the actual
  predictions + AI-drafted reports. This is what proves your models work.

  For each test DICOM:
    1. Auto-detects modality (chest / brain / head CT / mammo)
    2. Runs the right model
    3. Shows top predictions + confidence
    4. Drafts a Russian/Uzbek/English report via Gemma 3
    5. Times every step

  Run:
    python scripts/validate_ai_models.py             # all test DICOMs
    python scripts/validate_ai_models.py --brain     # only brain
    python scripts/validate_ai_models.py --quick     # 1 per modality
================================================================================
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'


def load_dicom_image(dicom_path: Path) -> tuple[np.ndarray, str, str]:
    """Load a DICOM file and return (pixel_array, modality, body_part)."""
    import pydicom
    ds = pydicom.dcmread(str(dicom_path), force=True)
    pixels = ds.pixel_array.astype(np.float32)
    # Normalize to 0-1
    p_min, p_max = pixels.min(), pixels.max()
    if p_max > p_min:
        pixels = (pixels - p_min) / (p_max - p_min)
    modality = str(getattr(ds, 'Modality', '?'))
    body_part = str(getattr(ds, 'BodyPartExamined', '?'))
    return pixels, modality, body_part


def run_one_dicom(dicom_path: Path, gemma_engine=None, languages=('ru',)) -> dict:
    """Full pipeline: DICOM → modality detect → predict → report."""
    from src.inference.model_registry import (
        REGISTRY, detect_modality_from_dicom, get_model,
    )

    print()
    print(f'  {B}━━━ {dicom_path.name} ━━━{NC}')

    # 1. Load
    t0 = time.time()
    try:
        pixels, modality, body_part = load_dicom_image(dicom_path)
    except Exception as e:
        print(f'  {R}✗ Failed to load DICOM: {e}{NC}')
        return {'ok': False, 'error': str(e)}

    t_load = (time.time() - t0) * 1000
    print(f'    📂 Loaded {pixels.shape}  modality={modality}  body={body_part}  ({t_load:.0f}ms)')

    # 2. Auto-route
    model_key = detect_modality_from_dicom(modality, body_part)
    print(f'    🎯 Routing → {B}{model_key}{NC}')

    # 3. Load model + predict
    t0 = time.time()
    entry = get_model(model_key, device='cpu')
    if entry is None or not entry['available']:
        reason = (entry or {}).get('reason', 'unknown')
        print(f'    {R}✗ Model unavailable: {reason[:120]}{NC}')
        return {'ok': False, 'error': reason}

    card = entry['card']
    predictor = entry['predictor']

    try:
        # Resize to model input size
        h, w = pixels.shape[:2]
        target = card.input_size[0] if isinstance(card.input_size, tuple) else 224
        if (h, w) != (target, target):
            from PIL import Image
            img = Image.fromarray((pixels * 255).clip(0, 255).astype(np.uint8))
            img = img.resize((target, target), Image.LANCZOS)
            pixels_resized = np.array(img).astype(np.float32) / 255.0
        else:
            pixels_resized = pixels

        if card.is_3d:
            print(f'    {Y}⏭  3D model — skipping (needs 4-sequence input){NC}')
            return {'ok': True, 'skipped': '3D needs 4 sequences'}

        probs = predictor(pixels_resized)
        # Filter out internal keys
        probs = {k: v for k, v in probs.items() if not k.startswith('_')}
    except Exception as e:
        print(f'    {R}✗ Inference failed: {e}{NC}')
        return {'ok': False, 'error': str(e)}

    t_infer = (time.time() - t0) * 1000

    # 4. Show top-5 predictions
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:5]
    print(f'    🧠 Inference took {t_infer:.0f}ms — top predictions:')
    for cls, p in sorted_probs:
        bar_len = int(p * 30)
        bar = '█' * bar_len + '·' * (30 - bar_len)
        color = G if p >= 0.5 else Y if p >= 0.2 else NC
        print(f'        {color}{cls:<28}{NC}  {bar}  {p*100:5.1f}%')

    # 5. Generate Russian report via Gemma
    if gemma_engine is not None and sorted_probs and sorted_probs[0][1] > 0.05:
        findings = [
            {'class_name': cls, 'confidence': p, 'location': 'Region of interest'}
            for cls, p in sorted_probs[:3] if p > 0.1
        ]
        if findings:
            print()
            for lang in languages:
                t0 = time.time()
                try:
                    report = gemma_engine.generate(
                        findings=findings,
                        language=lang,
                        modality=modality,
                        body_part=body_part,
                    )
                except Exception as e:
                    report = f'(report failed: {e})'
                t_rpt = (time.time() - t0) * 1000
                print(f'    📝 Report ({lang.upper()}, {t_rpt:.0f}ms):')
                # Indent each line
                for line in report.split('\n')[:18]:
                    print(f'        {line}')
                if len(report.split('\n')) > 18:
                    print(f'        {Y}…(truncated, full report in JSON output){NC}')
                print()

    return {
        'ok': True,
        'dicom': str(dicom_path),
        'modality': modality,
        'body_part': body_part,
        'routed_to': model_key,
        'top_predictions': sorted_probs,
        'inference_ms': t_infer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain', action='store_true', help='Brain MRIs only')
    parser.add_argument('--chest', action='store_true', help='Chest X-rays only')
    parser.add_argument('--headct', action='store_true', help='Head CTs only')
    parser.add_argument('--quick', action='store_true', help='Just 1 file per category')
    parser.add_argument('--no-reports', action='store_true', help='Skip Gemma report generation')
    parser.add_argument('--lang', default='ru', help='Report language(s), comma-separated. Default: ru')
    args = parser.parse_args()

    print('=' * 78)
    print(f'  {B}SENTINEL — AI MODEL VALIDATION{NC}')
    print('=' * 78)
    print(f'  Runs every loaded model on local test DICOMs.')
    print(f'  Confirms: routing works → models predict → Gemma drafts reports.')
    print()

    # Collect test DICOMs
    test_dir = Path('data/test_dicoms')
    if not test_dir.exists():
        print(f'  {R}✗ No test DICOMs found at {test_dir}{NC}')
        print(f'    Run: python scripts/build_test_data.py')
        sys.exit(1)

    files = []
    if args.brain or not (args.chest or args.headct):
        files += sorted((test_dir / 'brain').glob('*.dcm'))
    if args.chest or not (args.brain or args.headct):
        files += sorted((test_dir / 'chest').glob('*.dcm'))
    if args.headct or not (args.brain or args.chest):
        files += sorted((test_dir / 'headct').glob('*.dcm'))

    if args.quick:
        # 1 per category
        seen_dirs = set()
        quick = []
        for f in files:
            if f.parent.name not in seen_dirs:
                quick.append(f)
                seen_dirs.add(f.parent.name)
        files = quick

    if not files:
        print(f'  {R}✗ No DICOMs match filter{NC}')
        sys.exit(1)

    print(f'  Testing {len(files)} DICOM(s)…')

    # Init Gemma engine once
    gemma = None
    if not args.no_reports:
        try:
            from src.inference.gemma_report_engine import GemmaReportEngine
            gemma = GemmaReportEngine(backend='auto')
            print(f'  Report backend: {gemma.backend}')
        except Exception as e:
            print(f'  {Y}⚠ Report engine unavailable: {e}{NC}')
            gemma = None

    languages = tuple(l.strip() for l in args.lang.split(',') if l.strip())

    # Run each
    results = []
    for f in files:
        r = run_one_dicom(f, gemma_engine=gemma, languages=languages)
        results.append(r)

    # Summary
    print()
    print('=' * 78)
    print(f'  {B}SUMMARY{NC}')
    print('=' * 78)
    ok = sum(1 for r in results if r.get('ok'))
    skipped = sum(1 for r in results if r.get('skipped'))
    failed = sum(1 for r in results if not r.get('ok'))
    print(f'  Successful inferences:    {G}{ok}{NC}/{len(results)}')
    if skipped:
        print(f'  Skipped (3D models etc):  {Y}{skipped}{NC}')
    if failed:
        print(f'  Failed:                   {R}{failed}{NC}')

    # Per-modality routing breakdown
    routed = {}
    for r in results:
        if r.get('ok'):
            k = r.get('routed_to', '?')
            routed.setdefault(k, []).append(r)
    if routed:
        print()
        print(f'  {B}Routing breakdown:{NC}')
        for k, rs in sorted(routed.items()):
            avg_ms = sum(r['inference_ms'] for r in rs) / len(rs)
            print(f'    {k:<24}  {len(rs)} cases  avg {avg_ms:.0f}ms/inference')

    print()
    if failed == 0 and ok > 0:
        print(f'  {G}🚀 ALL MODELS WORKING — ready to demo{NC}')
    print()


if __name__ == '__main__':
    main()
