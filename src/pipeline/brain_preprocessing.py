"""
================================================================================
  SENTINEL — BRAIN MRI PREPROCESSING (validated open-source, zero-label)
================================================================================

  Aligns real clinical Philips scans to the distribution our detectors were
  trained on — the single biggest LABEL-FREE lever for cutting false positives
  (the glioma/stroke over-calling we measured). All commercial-safe licenses:

    • Skull-stripping  — HD-BET (Apache-2.0, DKFZ; validated WITH pathology on
      T1/T1ce/T2/FLAIR so it won't clip tumors). nnU-Net based.
    • N4 bias-field    — SimpleITK N4 (Apache-2.0). Removes Philips RF shading.

  Both are OPTIONAL and degrade gracefully: if the package/weights aren't
  present, the function logs and returns the input unchanged, so the pipeline
  never breaks. Skull-strip is a 3D op (volume in, brain-only volume out).
================================================================================
"""
from __future__ import annotations
import os
import tempfile
from typing import Optional
import numpy as np
from loguru import logger

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False

_hdbet_predictor = None
_hdbet_failed = False


def _np_to_sitk(vol: np.ndarray, spacing: tuple):
    """numpy (D,H,W) + spacing (z,y,x) -> SimpleITK image (sitk uses x,y,z)."""
    img = sitk.GetImageFromArray(vol.astype(np.float32))
    img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    return img


def n4_bias_correct(vol: np.ndarray, spacing: tuple = (1, 1, 1), shrink: int = 4) -> np.ndarray:
    """N4ITK bias-field correction (gold standard, deterministic, CPU). Returns
    input unchanged if SimpleITK is unavailable or it errors."""
    if not _HAS_SITK or vol.ndim != 3:
        return vol
    try:
        img = _np_to_sitk(vol, spacing)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([30] * 3)
        small = sitk.Shrink(img, [shrink] * 3)
        small_mask = sitk.Shrink(mask, [shrink] * 3)
        corrector.Execute(small, small_mask)
        log_bias = corrector.GetLogBiasFieldAsImage(img)
        out = img / sitk.Exp(log_bias)
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception as e:
        logger.debug(f"N4 skipped: {e}")
        return vol


def _get_hdbet(device: str = 'cpu'):
    """Lazy-load the HD-BET predictor + weights once. Returns None on failure."""
    global _hdbet_predictor, _hdbet_failed
    if _hdbet_predictor is not None:
        return _hdbet_predictor
    if _hdbet_failed:
        return None
    try:
        import torch
        from HD_BET.entry_point import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
        maybe_download_parameters()  # one-time; pre-bake for air-gapped clinics
        _hdbet_predictor = get_hdbet_predictor(
            use_tta=False, device=torch.device(device), verbose=False)
        logger.info("HD-BET skull-stripping ready")
        return _hdbet_predictor
    except Exception as e:
        logger.warning(f"HD-BET unavailable ({e}); skull-stripping disabled. "
                       f"Install: pip install HD-BET")
        _hdbet_failed = True
        return None


def skull_strip(vol: np.ndarray, spacing: tuple = (1, 1, 1),
                device: str = 'cpu') -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Remove skull/scalp/neck so the brain matches the model training domain.

    Returns (brain_only_volume, brain_mask). On any failure returns
    (vol, None) — the caller proceeds with the un-stripped volume.
    HD-BET is 3D + CPU-slow (~1-3 min/volume without GPU)."""
    if not _HAS_SITK or vol.ndim != 3:
        return vol, None
    predictor = _get_hdbet(device)
    if predictor is None:
        return vol, None
    try:
        from HD_BET.hd_bet_prediction import hdbet_predict
        with tempfile.TemporaryDirectory() as td:
            in_f = os.path.join(td, 'vol.nii.gz')
            out_f = os.path.join(td, 'vol_bet.nii.gz')
            mask_f = os.path.join(td, 'vol_bet_bet.nii.gz')  # HD-BET writes *_bet.nii.gz mask
            sitk.WriteImage(_np_to_sitk(vol, spacing), in_f)
            hdbet_predict(in_f, out_f, predictor,
                          keep_brain_mask=True, compute_brain_extracted_image=True)
            stripped = sitk.GetArrayFromImage(sitk.ReadImage(out_f)).astype(np.float32)
            mask = None
            for cand in (mask_f, out_f.replace('.nii.gz', '_bet.nii.gz')):
                if os.path.exists(cand):
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(cand)).astype(np.float32)
                    break
            return stripped, mask
    except Exception as e:
        logger.warning(f"Skull-strip failed ({e}); using un-stripped volume")
        return vol, None


def is_skull_strip_available() -> bool:
    """True if HD-BET + SimpleITK are importable (weights download on first use)."""
    if not _HAS_SITK:
        return False
    try:
        import HD_BET  # noqa: F401
        import torch    # noqa: F401
        return True
    except Exception:
        return False
