"""
================================================================================
  SENTINEL — BRAIN MRI MULTI-SEQUENCE PREPROCESSOR
================================================================================

  Real brain MRI workflows are multi-sequence (T1, T1-contrast, T2, FLAIR)
  and 3D (a stack of slices, not one image). This module turns a folder of
  DICOM files into the 4-channel 3D volume that BraTS-trained models expect.

  Pipeline (matches the public BraTS preprocessing recipe):

    1. Sort DICOM files into series by SeriesInstanceUID + sequence label.
    2. Detect which sequence each series is (T1, T1ce, T2, FLAIR) using
       DICOM SeriesDescription, SequenceName, ProtocolName tags.
    3. Build a 3D volume per sequence by stacking slices in InstanceNumber
       order.
    4. Skull-strip (rough — full SyN registration is overkill for inference).
    5. Resample to 1mm³ isotropic.
    6. Z-score normalize each sequence independently.
    7. Crop / pad to (128, 128, 128) — the BraTS canonical input shape.
    8. Stack into a (4, 128, 128, 128) tensor in BraTS channel order:
       [T1, T1ce, T2, FLAIR].

  Falls back to single-sequence 2D mode if only one series is present.
================================================================================
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


# ----------------------------------------------------------------------------
# SEQUENCE DETECTION
# ----------------------------------------------------------------------------

# We treat both "_" and whitespace as separators when matching sequences.
# The (?:...) groups use either real word boundaries OR explicit
# punctuation so things like "T1_axial" and "T2_FLAIR" still match.
_SEP = r'(?:[\s_\-/])'  # one separator
_OPT_SEP = r'(?:[\s_\-/]*)'

SEQUENCE_PATTERNS = {
    'T1ce':  [
        rf'(?:^|[\W_])t1{_OPT_SEP}(?:ce|c|gd|gad|\+c|contrast|post)(?:[\W_]|$)',
        rf'(?:^|[\W_])(?:ce|gd|gad){_OPT_SEP}t1(?:[\W_]|$)',
    ],
    'FLAIR': [
        rf'(?:^|[\W_])flair(?:[\W_]|$)',
        rf'(?:^|[\W_])t2{_OPT_SEP}flair(?:[\W_]|$)',
    ],
    'T1':    [
        rf'(?:^|[\W_])t1w?(?:[\W_]|$)',
        rf'(?:^|[\W_])t1{_OPT_SEP}tse(?:[\W_]|$)',
    ],
    'T2':    [
        rf'(?:^|[\W_])t2w?(?:[\W_]|$)',
        rf'(?:^|[\W_])t2{_OPT_SEP}tse(?:[\W_]|$)',
    ],
    'DWI':   [rf'(?:^|[\W_])dwi(?:[\W_]|$)', rf'(?:^|[\W_])diffusion(?:[\W_]|$)'],
    'ADC':   [rf'(?:^|[\W_])adc(?:[\W_]|$)'],
}


def detect_sequence(text: str) -> Optional[str]:
    """Match a DICOM-derived string against the canonical brain sequences.

    Order matters — more specific tokens (T1ce, FLAIR) checked first so
    "T1ce" doesn't get caught by the generic T1 pattern, and "T2_FLAIR"
    becomes FLAIR instead of T2.
    """
    if not text:
        return None
    s = ' ' + text.lower() + ' '   # pad so word-boundary works at edges
    for seq in ['T1ce', 'FLAIR', 'T1', 'T2', 'DWI', 'ADC']:
        for pattern in SEQUENCE_PATTERNS[seq]:
            if re.search(pattern, s):
                return seq
    return None


# ----------------------------------------------------------------------------
# BODY-PART / ORIENTATION GATING
# ----------------------------------------------------------------------------
#
# The 2D brain-tumor classifier was trained on AXIAL BRAIN slices only. Feeding
# it a spine, knee, or sagittal series produces confident nonsense (the glioma
# over-calling we saw on real hospital data). These helpers let us route ONLY
# appropriate series to the classifier and flag everything else for review.

# Tokens that mean "this is NOT a brain study" (spine/MSK/body on the same scanner)
_NON_BRAIN_TOKENS = [
    'spine', 'spinal', 'lspine', 'cspine', 'tspine', 'l-spine', 'c-spine', 't-spine',
    'lumbar', 'cervical', 'thoracic', 'sacr', 'knee', 'shoulder', 'hip', 'ankle',
    'wrist', 'elbow', 'foot', 'hand', 'pelvis', 'abdomen', 'liver', 'breast',
    'prostate', 'neck', 'orbit', 'cardiac', 'heart', 'femur', 'tibia',
]
_BRAIN_TOKENS = ['brain', 'head', 'skull', 'cerebr', 'cranial', 'tumor', 'tumour', 'glioma']


def is_brain_text(text: str) -> Optional[bool]:
    """Best-effort: does this DICOM-derived text describe a BRAIN study?

    Returns True (brain), False (clearly not brain), or None (unknown).
    Checks negative tokens first — 'L-SPINE' must win over a stray match.
    """
    if not text:
        return None
    t = text.lower()
    if any(tok in t for tok in _NON_BRAIN_TOKENS):
        return False
    if any(tok in t for tok in _BRAIN_TOKENS):
        return True
    return None


def orientation_from_iop(iop) -> str:
    """Derive slice orientation ('axial'/'sagittal'/'coronal') from the DICOM
    ImageOrientationPatient (0020,0037) direction cosines. Empty string if
    unavailable. The classifier wants AXIAL; sagittal/coronal slices look
    nothing like its training data."""
    try:
        if iop is None or len(iop) < 6:
            return ''
        row = np.array([float(x) for x in iop[:3]])
        col = np.array([float(x) for x in iop[3:6]])
        normal = np.cross(row, col)
        ax = int(np.argmax(np.abs(normal)))  # 0=Sagittal,1=Coronal,2=Axial plane normal
        return {0: 'sagittal', 1: 'coronal', 2: 'axial'}[ax]
    except Exception:
        return ''


# ----------------------------------------------------------------------------
# SERIES GROUPING
# ----------------------------------------------------------------------------

@dataclass
class BrainSeries:
    sequence: str            # 'T1' / 'T1ce' / 'T2' / 'FLAIR' / 'unknown'
    series_uid: str
    series_description: str
    files: list[Path] = field(default_factory=list)
    pixel_volume: Optional[np.ndarray] = None  # (D, H, W)
    pixel_spacing: tuple = (1.0, 1.0, 1.0)
    patient_id: str = ''
    study_uid: str = ''
    body_part: str = ''          # DICOM BodyPartExamined (BRAIN / LSPINE / KNEE…)
    orientation: str = ''        # 'axial' / 'sagittal' / 'coronal' / '' (from ImageOrientationPatient)


def group_dicom_files_into_series(dicom_paths: list[Path]) -> list[BrainSeries]:
    """Read DICOM headers and group files by SeriesInstanceUID."""
    if not _HAS_PYDICOM:
        raise RuntimeError("pydicom required: pip install pydicom")

    by_series: dict[str, BrainSeries] = {}

    for p in dicom_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception as e:
            logger.debug(f"Skipping {p}: {e}")
            continue

        suid = getattr(ds, 'SeriesInstanceUID', None) or f'series-{p.parent.name}'
        if suid not in by_series:
            descr = (
                getattr(ds, 'SeriesDescription', '') or
                getattr(ds, 'ProtocolName', '') or
                getattr(ds, 'SequenceName', '')
            )
            seq = detect_sequence(descr) or detect_sequence(p.parent.name) or 'unknown'
            body_part = str(getattr(ds, 'BodyPartExamined', '') or '')
            orient = orientation_from_iop(getattr(ds, 'ImageOrientationPatient', None))
            by_series[suid] = BrainSeries(
                sequence=seq,
                series_uid=suid,
                series_description=descr,
                patient_id=getattr(ds, 'PatientID', '') or '',
                study_uid=getattr(ds, 'StudyInstanceUID', '') or '',
                body_part=body_part,
                orientation=orient,
            )
        by_series[suid].files.append(p)

    return list(by_series.values())


def _multiframe_iop(ds) -> str:
    """Orientation for an enhanced/multi-frame DICOM, where
    ImageOrientationPatient lives in the Shared/Per-frame functional groups
    rather than at the top level."""
    for grp_name in ('SharedFunctionalGroupsSequence', 'PerFrameFunctionalGroupsSequence'):
        grp = getattr(ds, grp_name, None)
        if grp and len(grp) > 0:
            pos = getattr(grp[0], 'PlaneOrientationSequence', None)
            if pos and len(pos) > 0:
                iop = getattr(pos[0], 'ImageOrientationPatient', None)
                o = orientation_from_iop(iop)
                if o:
                    return o
    return ''


def _multiframe_spacing(ds, n_frames: int) -> tuple:
    """(slice, row, col) spacing for a multi-frame object from its functional
    groups; falls back to top-level tags."""
    ss = float(getattr(ds, 'SliceThickness', 0) or 0)
    ps = getattr(ds, 'PixelSpacing', None)
    shared = getattr(ds, 'SharedFunctionalGroupsSequence', None)
    if shared and len(shared) > 0:
        pm = getattr(shared[0], 'PixelMeasuresSequence', None)
        if pm and len(pm) > 0:
            ps = getattr(pm[0], 'PixelSpacing', ps) or ps
            ss = float(getattr(pm[0], 'SliceThickness', ss) or ss)
            sbs = getattr(pm[0], 'SpacingBetweenSlices', None)
            if sbs:
                ss = float(sbs)
    if ps and ss:
        return (ss, float(ps[0]), float(ps[1]))
    if ps:
        return (1.0, float(ps[0]), float(ps[1]))
    return (1.0, 1.0, 1.0)


def load_volume_from_series(series: BrainSeries) -> Optional[np.ndarray]:
    """Stack a series' DICOM into a 3D volume (D, H, W).

    Handles BOTH layouts:
      • classic — one 2D image per file (stack by InstanceNumber)
      • enhanced/multi-frame — ONE file holds the whole 3D series as a
        (frames, H, W) pixel_array (Philips Ingenia default). The old code
        np.stack'd these into a 4D array and crashed downstream — that was the
        real reason real hospital studies produced 0 sequences.
    """
    if not series.files:
        return None

    slices = []  # (sort_key, 2D array)
    for fp in series.files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if 'PixelData' not in ds:
                logger.debug(f"Skipping (no pixels) {fp}")
                continue
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception as decode_err:
                # Compressed transfer syntax with no decoder, or corrupt pixels.
                # Isolate the failure to THIS file so the study still loads.
                logger.warning(
                    f"Pixel decode failed for {fp.name} "
                    f"(transfer syntax {getattr(getattr(ds,'file_meta',None),'TransferSyntaxUID','?')}): "
                    f"{decode_err}. Install pylibjpeg/gdcm for compressed DICOM.")
                continue
            slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
            arr = arr * slope + intercept
            # MONOCHROME1 = inverted grayscale (higher value -> darker). Flip it
            # so every volume is MONOCHROME2-style (higher -> brighter), which is
            # what the models and our normalization assume.
            if str(getattr(ds, 'PhotometricInterpretation', '')).strip() == 'MONOCHROME1':
                arr = arr.max() - arr
            inst = int(getattr(ds, 'InstanceNumber', 0) or 0)

            if arr.ndim == 2:
                slices.append((float(inst), arr))
                ps = getattr(ds, 'PixelSpacing', None)
                ssk = getattr(ds, 'SliceThickness', None)
                if ps and ssk:
                    series.pixel_spacing = (float(ssk), float(ps[0]), float(ps[1]))
                if not series.orientation:
                    series.orientation = orientation_from_iop(
                        getattr(ds, 'ImageOrientationPatient', None))
            elif arr.ndim == 3:
                # multi-frame: each frame is a slice; this file IS the volume
                for i, frame in enumerate(arr):
                    slices.append((float(inst) * 100000 + i, frame))
                series.pixel_spacing = _multiframe_spacing(ds, arr.shape[0])
                if not series.orientation:
                    series.orientation = _multiframe_iop(ds)
            elif arr.ndim == 4:
                # (frames, H, W, samples) — e.g. RGB; take luminance of each frame
                for i, frame in enumerate(arr):
                    slices.append((float(inst) * 100000 + i, frame.mean(axis=-1)))
        except Exception as e:
            logger.debug(f"Skipping slice {fp}: {e}")
            continue

    if not slices:
        return None

    # All frames in a series must share H,W to stack; drop odd ones out.
    slices.sort(key=lambda x: x[0])
    shapes = Counter(s[1].shape for s in slices)
    main_shape = shapes.most_common(1)[0][0]
    kept = [s[1] for s in slices if s[1].shape == main_shape]
    if not kept:
        return None
    volume = np.stack(kept, axis=0)  # (D, H, W)
    series.pixel_volume = volume
    return volume


# ----------------------------------------------------------------------------
# NORMALIZATION + RESAMPLING
# ----------------------------------------------------------------------------

def zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalization on non-zero foreground voxels (BraTS standard)."""
    fg = volume > 0 if mask is None else mask > 0
    if fg.sum() == 0:
        return volume.astype(np.float32)
    mean = volume[fg].mean()
    std = volume[fg].std() + 1e-8
    out = (volume - mean) / std
    return out.astype(np.float32)


def crop_or_pad_to(volume: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
    """Center crop/pad a 3D volume to target shape."""
    out = np.zeros(target_shape, dtype=volume.dtype)
    src_d, src_h, src_w = volume.shape
    tgt_d, tgt_h, tgt_w = target_shape

    # Compute slice/pad bounds for each axis
    def bounds(src, tgt):
        if src >= tgt:
            start = (src - tgt) // 2
            return slice(start, start + tgt), slice(0, tgt)
        else:
            start = (tgt - src) // 2
            return slice(0, src), slice(start, start + src)

    src_sl_d, dst_sl_d = bounds(src_d, tgt_d)
    src_sl_h, dst_sl_h = bounds(src_h, tgt_h)
    src_sl_w, dst_sl_w = bounds(src_w, tgt_w)

    out[dst_sl_d, dst_sl_h, dst_sl_w] = volume[src_sl_d, src_sl_h, src_sl_w]
    return out


def resample_to_isotropic(volume: np.ndarray, current_spacing: tuple,
                            target_mm: float = 1.0) -> np.ndarray:
    """Resample to isotropic spacing using simple linear interpolation.

    For production-grade resampling use SimpleITK; this lightweight version
    avoids the SITK dependency for simple cases.
    """
    from scipy.ndimage import zoom
    factors = tuple(s / target_mm for s in current_spacing)
    if all(abs(f - 1.0) < 0.05 for f in factors):
        return volume  # already isotropic-ish
    return zoom(volume, factors, order=1, prefilter=False).astype(np.float32)


# ----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------

# The 2D classifier (andrei-teodor/resnet-pretrained-brain-mri) was trained on
# AXIAL T1-weighted (mostly post-contrast) brain slices. Feed it the matching
# sequence — NOT FLAIR, which was the old default and a big driver of the
# glioma over-calling on real Philips data. Preference order, best first:
CLASSIFIER_SEQUENCE_PREFERENCE = ['T1ce', 'T1', 'T2', 'FLAIR']


def robust_normalize_slice(slc: np.ndarray) -> np.ndarray:
    """Percentile-clip (1–99) then scale to [0,1].

    Plain min-max is hijacked by a single hot voxel (metal artifact, fat,
    scanner spike) and squashes the brain into a narrow band — which shifts the
    image off the model's training distribution. Percentile clipping is the
    cheap, label-free standardization the audit recommended."""
    slc = slc.astype(np.float32)
    fg = slc[slc > 0]
    if fg.size < 10:
        lo, hi = float(slc.min()), float(slc.max())
    else:
        lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    if hi <= lo:
        lo, hi = float(slc.min()), float(slc.max())
    if hi <= lo:
        return np.zeros_like(slc)
    out = np.clip((slc - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def central_band_best_slice(vol: np.ndarray, band: float = 0.6) -> np.ndarray:
    """Pick the most-brain slice from the CENTRAL band of the volume.

    The old code scanned all slices and could land on the first/last slice
    (skull base, empty air, neck) which look nothing like the model's training
    data. Restricting to the central `band` fraction keeps us in mid-brain."""
    if vol.ndim != 3:
        return vol
    d = vol.shape[0]
    if d <= 2:
        return vol[d // 2]
    half = max(1, int(d * band / 2))
    c = d // 2
    lo, hi = max(0, c - half), min(d, c + half + 1)
    sub = vol[lo:hi]
    scores = (sub > sub.mean()).sum(axis=(1, 2))  # most foreground content
    return sub[int(scores.argmax())]


@dataclass
class BrainStudy:
    """A processed brain MRI study with per-sequence volumes ready for inference."""
    patient_id: str = ''
    study_uid: str = ''
    sequences: dict[str, np.ndarray] = field(default_factory=dict)  # 'T1' -> (D, H, W)
    target_shape: tuple = (128, 128, 128)
    seq_meta: dict = field(default_factory=dict)   # seq_key -> {body_part, orientation, description}

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def has_brats_quartet(self) -> bool:
        """True when all 4 BraTS sequences are present."""
        return all(s in self.sequences for s in ['T1', 'T1ce', 'T2', 'FLAIR'])

    @property
    def non_brain_reason(self) -> Optional[str]:
        """If the study looks like it is NOT a brain study, say why (so the
        server can flag-for-review instead of confidently calling a tumor on a
        spine/knee scan). Returns None when it looks like a brain study or is
        ambiguous."""
        verdicts = []
        for key, meta in self.seq_meta.items():
            for txt in (meta.get('body_part', ''), meta.get('description', '')):
                v = is_brain_text(txt)
                if v is not None:
                    verdicts.append(v)
        if verdicts and not any(verdicts):
            # every informative tag said "not brain"
            parts = sorted({m.get('body_part') or m.get('description', '')
                            for m in self.seq_meta.values() if m.get('body_part') or m.get('description')})
            return f"non-brain body part detected: {', '.join(p for p in parts if p)[:80]}"
        return None

    def to_brats_tensor(self) -> Optional[np.ndarray]:
        """Pack into the (4, D, H, W) tensor SwinUNETR expects.

        Channel order: [T1, T1ce, T2, FLAIR]. Sequences are stored full-res
        (so the 2D classifier sees the whole brain); the BraTS-canonical 128³
        crop happens HERE, only for the 3D path.
        Returns None when the quartet isn't complete.
        """
        if not self.has_brats_quartet:
            return None
        chans = []
        for seq in ['T1', 'T1ce', 'T2', 'FLAIR']:
            vol = self.sequences[seq]
            spacing = self.seq_meta.get(seq, {}).get('spacing', (1.0, 1.0, 1.0))
            try:
                vol = resample_to_isotropic(vol, spacing, target_mm=1.0)
            except Exception as e:
                logger.debug(f"Resample skipped for {seq}: {e}")
            vol = crop_or_pad_to(vol, self.target_shape)
            chans.append(vol)
        return np.stack(chans, axis=0)

    def select_classifier_input(self) -> Optional[dict]:
        """Choose the right sequence + slice for the 2D brain-tumor classifier.

        Returns a dict {sequence, orientation, slice, warnings} or None if no
        usable series. This is the fix for the glioma over-calling: prefer
        T1ce/T1 axial (the model's training distribution), avoid sagittal/
        coronal, pick a central slice, and percentile-normalize.
        """
        if not self.sequences:
            return None

        def is_axial(key: str) -> bool:
            return (self.seq_meta.get(key, {}).get('orientation') or 'axial') == 'axial'

        # 1) preferred sequence that is axial   2) any axial   3) anything
        ordered = []
        for pref in CLASSIFIER_SEQUENCE_PREFERENCE:
            if pref in self.sequences and is_axial(pref):
                ordered.append(pref)
        ordered += [k for k in self.sequences if is_axial(k) and k not in ordered]
        ordered += [k for k in self.sequences if k not in ordered]
        if not ordered:
            return None

        chosen = ordered[0]
        meta = self.seq_meta.get(chosen, {})
        warnings = []
        if not is_axial(chosen):
            warnings.append(
                f"no axial series found; using {meta.get('orientation') or 'unknown'} "
                f"'{chosen}' — classifier accuracy is reduced off-axial")
        if chosen not in CLASSIFIER_SEQUENCE_PREFERENCE and not chosen.startswith(('T1', 'T2', 'FLAIR')):
            warnings.append(f"sequence '{chosen}' not in the model's training set (T1/T2/FLAIR)")

        vol = self.sequences[chosen]
        slc = central_band_best_slice(vol) if vol.ndim == 3 else vol
        slc = robust_normalize_slice(slc)
        return {
            'sequence': chosen,
            'orientation': meta.get('orientation') or 'unknown',
            'slice': slc,
            'warnings': warnings,
        }

    def best_2d_slice(self, sequence: Optional[str] = None) -> Optional[np.ndarray]:
        """Back-compat shim. With no sequence given, uses the corrected
        preference-based selection. With an explicit sequence, returns a
        robustly-normalized central slice of that sequence."""
        if sequence is None:
            sel = self.select_classifier_input()
            return sel['slice'] if sel else None
        if sequence not in self.sequences:
            sequence = list(self.sequences.keys())[0] if self.sequences else None
        if sequence is None:
            return None
        vol = self.sequences[sequence]
        slc = central_band_best_slice(vol) if vol.ndim == 3 else vol
        return robust_normalize_slice(slc)


def process_brain_study(dicom_paths: list[Path],
                          target_shape=(128, 128, 128)) -> BrainStudy:
    """Take a list of DICOM file paths from one study and return a BrainStudy.

    Handles:
      - Single-series studies (most clinics' default brain MRI)
      - Multi-series studies with all 4 BraTS sequences
      - Mixed/unknown sequence labelling (best-effort detection)
    """
    series_list = group_dicom_files_into_series([Path(p) for p in dicom_paths])
    if not series_list:
        return BrainStudy()

    study = BrainStudy(target_shape=target_shape)
    study.patient_id = series_list[0].patient_id
    study.study_uid = series_list[0].study_uid

    for series in series_list:
        vol = load_volume_from_series(series)
        if vol is None or vol.size == 0:
            continue
        # Store FULL-RESOLUTION z-scored volume. We deliberately do NOT crop to
        # 128³ here — that would chop a 400×400 brain down to its center and
        # starve the 2D classifier. The 128³ BraTS packing happens later, only
        # for the 3D path (see to_brats_tensor).
        vol = zscore_normalize(vol)

        # Decide which slot this fills
        seq = series.sequence
        if seq == 'unknown':
            # Park unknown series under their own key so the user can manually map
            seq = f'unknown_{len(study.sequences)+1}'
        # Don't overwrite: prefer the first occurrence per sequence
        key = seq if seq not in study.sequences else f'{seq}_dup_{len(study.sequences)}'
        study.sequences[key] = vol
        study.seq_meta[key] = {
            'body_part': series.body_part,
            'orientation': series.orientation,
            'description': series.series_description,
            'spacing': series.pixel_spacing,
        }

    return study
