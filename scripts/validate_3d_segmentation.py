#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — 3D BRAIN TUMOR SEGMENTATION VALIDATOR
================================================================================

  Tests the 3D segmentation pipeline on real BraTS data.

  Supports two input formats:
    .npz   — preprocessed (yuuricho/brats_preprocessing): 'images' (4,D,H,W),
              'seg' (D,H,W) ground-truth mask
    .nii   — raw BraTS (4 separate files per case: T1, T1ce, T2, FLAIR + seg)

  For each case:
    1. Load 4-channel image volume + ground-truth mask
    2. Resize to model input shape (128,128,128)
    3. Run SwinUNETR v2 prediction
    4. Compute Dice score per class (predicted vs ground-truth)
    5. Show inference time + accuracy

  Run:
    python scripts/validate_3d_segmentation.py
    python scripts/validate_3d_segmentation.py --case-limit 10
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


def find_brats_npz(root: Path) -> list[Path]:
    """Find every preprocessed BraTS .npz file."""
    return sorted(root.rglob('*.npz'))


def find_msd_brain_cases(root: Path) -> list[dict]:
    """Find Medical Segmentation Decathlon Task01_BrainTumour cases.

    MSD format:
      Task01_BrainTumour/
        imagesTr/   BRATS_001.nii.gz   (4-channel volume)
        labelsTr/   BRATS_001.nii.gz   (segmentation mask)
    """
    cases = []
    base = root / 'Task01_BrainTumour'
    if not base.exists():
        return cases

    images_tr = base / 'imagesTr'
    labels_tr = base / 'labelsTr'
    if not images_tr.exists() or not labels_tr.exists():
        return cases

    for img_file in sorted(images_tr.glob('*.nii.gz')):
        # MSD hides files starting with "._" (macOS metadata)
        if img_file.name.startswith('._'):
            continue
        label_file = labels_tr / img_file.name
        if not label_file.exists():
            continue
        cases.append({
            'case_id': img_file.stem.replace('.nii', ''),
            'image': img_file,
            'label': label_file,
        })
    return cases


def load_nifti_volume(path: Path) -> np.ndarray:
    """Load a NIfTI file."""
    try:
        import nibabel as nib
    except ImportError:
        print(f'{R}✗ nibabel is required for NIfTI loading. Install it:{NC}')
        print('    pip install nibabel')
        sys.exit(1)
    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(np.float32)


def zscore_per_channel(images: np.ndarray) -> np.ndarray:
    """Z-score normalize per channel using non-zero voxels only.

    images: (4, D, H, W) — same as MONAI's NormalizeIntensityd(nonzero=True, channel_wise=True)
    """
    out = np.zeros_like(images, dtype=np.float32)
    for c in range(images.shape[0]):
        ch = images[c]
        nonzero = ch > 0
        if nonzero.sum() == 0:
            out[c] = ch
            continue
        mean = ch[nonzero].mean()
        std = ch[nonzero].std() + 1e-8
        ch_norm = (ch - mean) / std
        ch_norm[~nonzero] = 0  # keep background at 0 (BraTS convention)
        out[c] = ch_norm
    return out


def crop_or_pad_3d(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Center crop/pad — works on either 3D (D,H,W) or 4D (C,D,H,W) volumes."""
    if vol.ndim == 4:
        out = np.zeros((vol.shape[0],) + target, dtype=vol.dtype)
        for c in range(vol.shape[0]):
            out[c] = crop_or_pad_3d(vol[c], target)
        return out

    out = np.zeros(target, dtype=vol.dtype)
    src = vol.shape

    def bounds(s, t):
        if s >= t:
            start = (s - t) // 2
            return slice(start, start + t), slice(0, t)
        start = (t - s) // 2
        return slice(0, s), slice(start, start + s)

    s = list(bounds(src[i], target[i]) for i in range(3))
    out[s[0][1], s[1][1], s[2][1]] = vol[s[0][0], s[1][0], s[2][0]]
    return out


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Sørensen-Dice coefficient for a single class label."""
    p = (pred == label)
    g = (gt == label)
    if p.sum() + g.sum() == 0:
        return 1.0  # both empty = perfect agreement
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


def dice_binary(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Sørensen-Dice between two binary masks."""
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    if p.sum() + g.sum() == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/public/brats_lite',
                          help='Root with BraTS .npz files')
    parser.add_argument('--case-limit', type=int, default=3,
                          help='Max cases to test (default 3)')
    args = parser.parse_args()

    print('=' * 78)
    print(f'  {B}3D BRAIN SEGMENTATION VALIDATOR — REAL BraTS DATA{NC}')
    print('=' * 78)

    data_root = Path(args.data)

    # Auto-detect format: prefer MSD NIfTI (full volumes) over yuuricho slabs
    msd_root = Path('data/public/msd_brain')
    msd_cases = find_msd_brain_cases(msd_root) if msd_root.exists() else []

    if msd_cases:
        print(f'  Using MSD Task01_BrainTumour: {len(msd_cases)} full-volume cases')
        cases = [{'type': 'msd', **c} for c in msd_cases]
    else:
        print(f'  Searching {data_root}/ for .npz cases…')
        npz_cases = find_brats_npz(data_root)
        print(f'  Found {len(npz_cases)} .npz cases')
        cases = [{'type': 'npz', 'path': p} for p in npz_cases]

    if not cases:
        print()
        print(f'  {Y}⚠ No BraTS cases found. Either:{NC}')
        print(f'    • Download MSD: data/public/msd_brain/Task01_BrainTumour/')
        print(f'    • Or yuuricho slabs: data/public/brats_lite/')
        sys.exit(1)

    # Load model once
    print()
    print(f'  Loading 3D segmentation model…')
    from src.inference.model_registry import get_model
    entry = get_model('brain_tumor_seg_3d', device='cpu')
    if entry is None or not entry['available']:
        reason = (entry or {}).get('reason', 'unknown')
        print(f'  {R}✗ 3D model unavailable: {reason}{NC}')
        sys.exit(1)
    print(f'  {G}✓{NC} model ready: {entry["reason"][:80]}')

    predictor = entry['predictor']
    card = entry['card']
    target_shape = card.input_size if isinstance(card.input_size, tuple) else (128, 128, 128)
    if len(target_shape) == 2:
        target_shape = (128,) + target_shape

    cases = cases[:args.case_limit]
    successes = 0
    dice_avgs = []

    for i, case in enumerate(cases, 1):
        case_id = (case.get('case_id') or
                   (case['path'].name if 'path' in case else f'case{i}'))
        print()
        print(f'  {B}━━━ Case {i}/{len(cases)}: {case_id} ━━━{NC}')

        try:
            t0 = time.time()
            if case['type'] == 'msd':
                # MSD NIfTI: image is (H, W, D, 4), label is (H, W, D)
                img = load_nifti_volume(case['image'])
                lbl = load_nifti_volume(case['label'])
                # Reshape: NIfTI is (X, Y, Z, C) → we want (C, D, H, W)
                # MSD convention: image is (240, 240, 155, 4), label is (240, 240, 155)
                if img.ndim == 4 and img.shape[-1] == 4:
                    images = img.transpose(3, 2, 0, 1)   # → (4, D, H, W)
                else:
                    images = img[np.newaxis, ...].transpose(0, 3, 1, 2) if img.ndim == 3 else img
                seg_gt = lbl.transpose(2, 0, 1) if lbl.ndim == 3 else lbl

                # MSD images are NOT pre-normalized — apply z-score per channel
                images = zscore_per_channel(images)
                t_load = (time.time() - t0) * 1000
                print(f'    Loaded MSD case ({t_load:.0f}ms): images {images.shape}  seg {seg_gt.shape}')
            else:
                data = np.load(case['path'])
                images = data['images']
                seg_gt = data.get('seg')
                t_load = (time.time() - t0) * 1000
                print(f'    Loaded npz ({t_load:.0f}ms): images {images.shape}  '
                      f'seg {seg_gt.shape if seg_gt is not None else "n/a"}')

            # MONAI bundle uses sliding-window inference — handles arbitrary
            # input shapes. Don't crop! Cropping destroys spatial relationships
            # the model learned during training.
            print(f'    Using full volume {images.shape} (sliding window will tile it)')

            # Run inference
            print(f'    Running SwinUNETR v2 (~30s on CPU)…', end=' ', flush=True)
            t0 = time.time()
            result = predictor(images)
            t_inf = time.time() - t0
            print(f'{G}✓{NC} {t_inf:.1f}s')

            # Class fractions from prediction
            mask = result.pop('_mask_array', None)
            result.pop('_probs', None)   # MONAI bundle adds this
            # Only keep numeric scalar values for the table
            scalar_results = {k: v for k, v in result.items()
                              if isinstance(v, (int, float)) and not k.startswith('_')}
            print(f'    Predicted region voxel fractions:')
            for cls, frac in sorted(scalar_results.items(), key=lambda x: -x[1]):
                bar = '█' * int(frac * 30) + '·' * (30 - int(frac * 30))
                print(f'      {cls:<22} {bar}  {frac*100:5.2f}%')

            # Compute BraTS-style TC/WT/ET Dice scores
            if seg_gt is not None and mask is not None:
                # MSD Task01 uses labels {0,1,2,3} (ET=3), BraTS 2020+ uses {0,1,2,4} (ET=4).
                # Detect which convention is in use:
                gt_uniques = set(int(x) for x in np.unique(seg_gt) if x > 0)
                ET_LABEL = 3 if 3 in gt_uniques and 4 not in gt_uniques else 4
                # TC = necrotic + enhancing; WT = TC + edema; ET = enhancing only
                gt_TC = ((seg_gt == 1) | (seg_gt == ET_LABEL))
                gt_WT = ((seg_gt == 1) | (seg_gt == 2) | (seg_gt == ET_LABEL))
                gt_ET = (seg_gt == ET_LABEL)

                if mask.ndim == 4 and mask.shape[0] == 3:
                    # MONAI bundle output: 3 binary masks already (TC, WT, ET)
                    pred_TC = mask[0]
                    pred_WT = mask[1]
                    pred_ET = mask[2]
                    print(f'    {B}BraTS Dice scores (predicted vs ground truth):{NC}')
                    case_dices = []
                    for name, p, g in [
                        ('Tumor Core (TC)',     pred_TC, gt_TC),
                        ('Whole Tumor (WT)',    pred_WT, gt_WT),
                        ('Enhancing (ET)',      pred_ET, gt_ET),
                    ]:
                        d = dice_binary(p, g)
                        case_dices.append(d)
                        color = G if d > 0.65 else Y if d > 0.4 else R
                        print(f'      {name:<22}: {color}{d:.3f}{NC}')
                    if case_dices:
                        avg = sum(case_dices) / len(case_dices)
                        dice_avgs.append(avg)
                else:
                    # Multi-class output (older single-channel SwinUNETR path)
                    print(f'    Multi-class Dice scores (per label):')
                    case_dices = []
                    for label, name in [(1, 'necrotic'), (2, 'edema'), (4, 'enhancing')]:
                        d = dice_score(mask, seg_gt, label)
                        case_dices.append(d)
                        color = G if d > 0.5 else Y if d > 0.2 else R
                        print(f'      label {label} ({name:<11}): {color}{d:.3f}{NC}')
                    if case_dices:
                        dice_avgs.append(sum(case_dices) / len(case_dices))
            successes += 1
        except Exception as e:
            print(f'    {R}✗ Failed: {e}{NC}')
            import traceback
            traceback.print_exc()

    print()
    print('=' * 78)
    print(f'  {G if successes else R}{successes}/{len(cases)} cases ran successfully{NC}')
    if dice_avgs:
        avg_all = sum(dice_avgs) / len(dice_avgs)
        print(f'  Average Dice (across cases): {avg_all:.3f}')
        if avg_all > 0.5:
            print(f'  {G}🎯 SwinUNETR is producing meaningful segmentations.{NC}')
        elif avg_all > 0.2:
            print(f'  {Y}⚠ Predictions correlate with ground truth but accuracy is moderate.{NC}')
            print(f'    The model needs the same preprocessing pipeline that BraTS used.')
        else:
            print(f'  {R}✗ Predictions don\'t match ground truth — likely a class-mapping issue.{NC}')
    print('=' * 78)


if __name__ == '__main__':
    main()
