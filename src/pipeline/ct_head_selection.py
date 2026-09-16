"""CT head series selection for the `head_ct` registry route.

A head-CT upload is a whole study: the scout/topogram localizer, a thin bone
reconstruction, the 5 mm soft-tissue axial series, sagittal/coronal reformats
and screen saves. The hemorrhage classifier was trained on AXIAL BRAIN slices
— it must never see the localizer. Picking "the single largest image" did
exactly that on a GE scanner (the 734×835 scout beats every 512×512 axial
slice), so the route now works per series:

  1. header-only pass, grouped by SeriesInstanceUID
  2. drop localizers (ImageType LOCALIZER, or a scout/topogram description)
     and series with fewer than MIN_SERIES_IMAGES images
  3. prefer an axial brain series (description matching BRAIN_DESC_RE and an
     axial ImageOrientationPatient), largest first; fall back to the series
     with the most images
  4. sort its slices along the table axis (ImagePositionPatient projected on
     the slice normal, else InstanceNumber, else upload order)
  5. take up to DEFAULT_N_SLICES central slices, skipping the lowest/highest
     CENTRAL_SKIP_FRACTION of the stack

Slices are windowed in Hounsfield units (RescaleSlope/Intercept) with the
file's own WindowCenter/Width, or the DICOM brain window (40/80 HU) when the
tags are absent. Nothing here touches the MR brain panel.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from loguru import logger

LOCALIZER_DESC_RE = re.compile(r'scout|localizer|topogram|scano', re.I)
BRAIN_DESC_RE = re.compile(r'brain|head|std|axial|routine', re.I)
# Hemorrhage is read on the soft-tissue kernel; a bone/lung kernel is only a
# last resort even when it has the most slices (thin bone recons usually do).
HARD_KERNEL_DESC_RE = re.compile(r'\bbone\b|\blung\b', re.I)

MIN_SERIES_IMAGES = 5
DEFAULT_N_SLICES = 9
CENTRAL_SKIP_FRACTION = 0.2
BRAIN_WINDOW_CENTER = 40.0
BRAIN_WINDOW_WIDTH = 80.0
# |slice normal · z| ≥ cos(~25°) counts as axial (tilted gantry scans included).
AXIAL_MIN_ABS_COS = 0.9


@dataclass
class SliceHeader:
    path: Path
    order: int                              # position in the upload (last-resort sort key)
    instance_number: Optional[int] = None
    z: Optional[float] = None               # along the series normal (or raw IPP z)
    rows: int = 0
    cols: int = 0

    @property
    def sort_key(self) -> tuple:
        # z first when known, then InstanceNumber, then upload order; the
        # booleans push "unknown" entries after the known ones.
        return (self.z is None, self.z if self.z is not None else 0.0,
                self.instance_number is None,
                self.instance_number if self.instance_number is not None else 0,
                self.order)


@dataclass
class SeriesInfo:
    uid: str
    description: str = ''
    image_type: list[str] = field(default_factory=list)
    orientation: Optional[list[float]] = None   # ImageOrientationPatient (6 floats)
    slices: list[SliceHeader] = field(default_factory=list)
    dropped_reason: Optional[str] = None

    @property
    def n_images(self) -> int:
        return len(self.slices)

    @property
    def is_axial(self) -> bool:
        if self.orientation:
            return is_axial_orientation(self.orientation)
        return any(t.upper() == 'AXIAL' for t in self.image_type)

    @property
    def matches_brain(self) -> bool:
        return bool(BRAIN_DESC_RE.search(self.description or ''))

    @property
    def is_hard_kernel(self) -> bool:
        return bool(HARD_KERNEL_DESC_RE.search(self.description or ''))

    @property
    def is_original(self) -> bool:
        # ORIGINAL/PRIMARY acquisitions beat DERIVED reformats and screen saves.
        types = {t.upper() for t in self.image_type}
        return 'ORIGINAL' in types or not types & {'DERIVED', 'SECONDARY'}

    @property
    def rows_cols(self) -> tuple[int, int]:
        if not self.slices:
            return (0, 0)
        return (self.slices[0].rows, self.slices[0].cols)

    def summary(self) -> dict:
        return {
            'series_uid': self.uid,
            'description': self.description,
            'n_images': self.n_images,
            'image_type': list(self.image_type),
            'axial': self.is_axial,
            'brain_keyword': self.matches_brain,
            'original': self.is_original,
            'rows_cols': list(self.rows_cols),
            'dropped_reason': self.dropped_reason,
        }


@dataclass
class CTHeadSelection:
    series: SeriesInfo
    indices: list[int]                   # into series.slices (already sorted)
    slice_paths: list[Path]
    central_path: Path
    n_series: int
    dropped: list[dict]
    candidates: list[dict]

    @property
    def central_index(self) -> int:
        return self.indices[len(self.indices) // 2]


# ---------------------------------------------------------------------------
# headers
# ---------------------------------------------------------------------------

def _first_float(value) -> Optional[float]:
    """DICOM DS / MultiValue → float of the first element, None when absent."""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
            if len(value) == 0:
                return None
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return None


def _floats(value, n: int) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n else None


def _image_type(ds) -> list[str]:
    it = ds.get('ImageType')
    if it is None:
        return []
    if isinstance(it, str):
        return [t.strip() for t in it.split('\\') if t.strip()]
    try:
        return [str(t).strip() for t in it if str(t).strip()]
    except TypeError:
        return [str(it).strip()]


def slice_normal(iop: list[float]) -> Optional[np.ndarray]:
    """Unit normal of the image plane from ImageOrientationPatient."""
    if not iop or len(iop) != 6:
        return None
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:], dtype=float)
    n = np.cross(row, col)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return n / norm


def is_axial_orientation(iop: Optional[list[float]]) -> bool:
    n = slice_normal(iop) if iop else None
    return n is not None and abs(float(n[2])) >= AXIAL_MIN_ABS_COS


def read_series_headers(file_paths: list[Path]) -> list[SeriesInfo]:
    """Header-only pass (stop_before_pixels) grouping the upload by
    SeriesInstanceUID. Files that do not parse as DICOM are skipped. A file
    without SeriesInstanceUID gets its own pseudo-series."""
    by_uid: dict[str, SeriesInfo] = {}
    for order, p in enumerate(file_paths):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not (ds.get('SOPClassUID') or ds.get('Modality')):
            continue       # force=True parses almost anything — require a DICOM signal
        uid = str(ds.get('SeriesInstanceUID') or '').strip() or f'__nouid__{order}'
        series = by_uid.get(uid)
        if series is None:
            series = SeriesInfo(
                uid=uid,
                description=str(ds.get('SeriesDescription') or '').strip(),
                image_type=_image_type(ds),
                orientation=_floats(ds.get('ImageOrientationPatient'), 6),
            )
            by_uid[uid] = series
        else:
            if not series.description:
                series.description = str(ds.get('SeriesDescription') or '').strip()
            if not series.image_type:
                series.image_type = _image_type(ds)
            if series.orientation is None:
                series.orientation = _floats(ds.get('ImageOrientationPatient'), 6)

        ipp = _floats(ds.get('ImagePositionPatient'), 3)
        z = None
        if ipp is not None:
            n = slice_normal(series.orientation) if series.orientation else None
            z = float(np.dot(ipp, n)) if n is not None else float(ipp[2])
        try:
            inst = int(ds.get('InstanceNumber')) if ds.get('InstanceNumber') is not None else None
        except (TypeError, ValueError):
            inst = None
        series.slices.append(SliceHeader(
            path=Path(p), order=order, instance_number=inst, z=z,
            rows=int(ds.get('Rows') or 0), cols=int(ds.get('Columns') or 0),
        ))
    return list(by_uid.values())


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def localizer_reason(series: SeriesInfo) -> Optional[str]:
    """Why a series must not be analysed, or None when it is a candidate."""
    if any(t.upper() == 'LOCALIZER' for t in series.image_type):
        return 'ImageType LOCALIZER'
    if LOCALIZER_DESC_RE.search(series.description or ''):
        return f'localizer description {series.description!r}'
    if series.n_images < MIN_SERIES_IMAGES:
        return f'fewer than {MIN_SERIES_IMAGES} images ({series.n_images})'
    return None


def sort_slices(series: SeriesInfo) -> None:
    series.slices.sort(key=lambda s: s.sort_key)


def _preference_key(series: SeriesInfo) -> tuple:
    if series.matches_brain and series.is_axial:
        tier = 0
    elif series.is_axial:
        tier = 1
    elif series.matches_brain:
        tier = 2
    else:
        tier = 3
    # smaller is better: tier, soft-tissue kernel, original acquisition, then MOST images
    return (tier, series.is_hard_kernel, not series.is_original, -series.n_images)


def choose_brain_series(candidates: list[SeriesInfo]) -> Optional[SeriesInfo]:
    """Axial brain series (largest) → any axial → brain-named → most images."""
    if not candidates:
        return None
    return min(candidates, key=_preference_key)


def central_indices(n: int, n_slices: int = DEFAULT_N_SLICES,
                    skip_fraction: float = CENTRAL_SKIP_FRACTION) -> list[int]:
    """Up to `n_slices` evenly spread indices inside the central band of an
    n-slice stack (the lowest/highest `skip_fraction` are skipped). Always at
    least one index for n ≥ 1; the whole band when it is short."""
    if n <= 0:
        return []
    lo = int(math.floor(n * skip_fraction))
    hi = int(math.ceil(n * (1.0 - skip_fraction)))
    lo = min(lo, n - 1)
    hi = max(min(hi, n), lo + 1)
    band = list(range(lo, hi))
    if len(band) <= n_slices:
        return band
    picks = np.linspace(lo, hi - 1, n_slices)
    return sorted({int(round(float(p))) for p in picks})


def select_ct_head_slices(file_paths: list[Path],
                          n_slices: int = DEFAULT_N_SLICES) -> Optional[CTHeadSelection]:
    """The axial brain series of a head-CT upload and its central slices, or
    None when no series survives (single-file uploads, localizer-only sets)."""
    series_list = read_series_headers(file_paths)
    dropped, candidates = [], []
    for s in series_list:
        reason = localizer_reason(s)
        if reason:
            s.dropped_reason = reason
            dropped.append(s.summary())
        else:
            candidates.append(s)
    chosen = choose_brain_series(candidates)
    if chosen is None:
        logger.warning(f"ct_head_selection: no analysable series among {len(series_list)} "
                       f"(dropped: {[d['description'] or d['series_uid'] for d in dropped]})")
        return None
    sort_slices(chosen)
    idx = central_indices(chosen.n_images, n_slices=n_slices)
    paths = [chosen.slices[i].path for i in idx]
    sel = CTHeadSelection(
        series=chosen, indices=idx, slice_paths=paths,
        central_path=paths[len(paths) // 2],
        n_series=len(series_list), dropped=dropped,
        candidates=sorted((c.summary() for c in candidates),
                          key=lambda c: (c['series_uid'] != chosen.uid, -c['n_images'])),
    )
    logger.info(
        f"ct_head_selection: {len(series_list)} series → '{chosen.description}' "
        f"({chosen.n_images} images, axial={chosen.is_axial}); {len(idx)} central slices; "
        f"dropped {len(dropped)}"
    )
    return sel


def largest_non_localizer(file_paths: list[Path]) -> Optional[Path]:
    """Single-file fallback: the largest image (Rows×Columns) that is not a
    localizer. None when every readable file is a localizer."""
    best, best_px = None, -1
    for s in read_series_headers(file_paths):
        if any(t.upper() == 'LOCALIZER' for t in s.image_type) or \
                LOCALIZER_DESC_RE.search(s.description or ''):
            continue
        for sl in s.slices:
            px = sl.rows * sl.cols
            if px > best_px:
                best, best_px = sl.path, px
    return best


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------

def window_hu(pixel_array: np.ndarray, *, rescale_slope: float = 1.0,
              rescale_intercept: float = 0.0, window_center: Optional[float] = None,
              window_width: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """Raw CT pixels → [0,1] float32 via Hounsfield rescale + window. The
    brain window (40/80 HU) is used when the window tags are absent/invalid."""
    hu = pixel_array.astype(np.float32) * float(rescale_slope) + float(rescale_intercept)
    used_default = window_center is None or window_width is None or not (window_width > 0)
    if used_default:
        window_center, window_width = BRAIN_WINDOW_CENTER, BRAIN_WINDOW_WIDTH
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    out = np.clip((hu - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    meta = {'window_center': float(window_center), 'window_width': float(window_width),
            'default_brain_window': bool(used_default)}
    return out, meta


def window_dataset(ds) -> tuple[np.ndarray, dict]:
    """window_hu() for a fully-read pydicom dataset (multi-frame → middle
    frame; RGB → gray; MONOCHROME1 inverted before windowing)."""
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2) if arr.shape[-1] == 3 else arr[arr.shape[0] // 2]
    if str(ds.get('PhotometricInterpretation', 'MONOCHROME2')) == 'MONOCHROME1':
        arr = arr.max() - arr
    slope = _first_float(ds.get('RescaleSlope'))
    intercept = _first_float(ds.get('RescaleIntercept'))
    img, meta = window_hu(
        arr,
        rescale_slope=1.0 if slope is None else slope,
        rescale_intercept=0.0 if intercept is None else intercept,
        window_center=_first_float(ds.get('WindowCenter')),
        window_width=_first_float(ds.get('WindowWidth')),
    )
    meta['rows'], meta['cols'] = int(arr.shape[0]), int(arr.shape[1])
    inst = ds.get('InstanceNumber')
    meta['instance_number'] = int(inst) if inst is not None else None
    return img, meta


def load_windowed_slice(path: Path, target_size: Optional[int] = None
                        ) -> Optional[tuple[np.ndarray, dict]]:
    """Windowed [0,1] slice from a DICOM file, optionally letter-boxed to
    target_size×target_size (the classifier input). None when unreadable."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        img, meta = window_dataset(ds)
    except Exception as e:
        logger.warning(f"ct_head_selection: cannot decode {Path(path).name}: {e}")
        return None
    if target_size:
        from src.pipeline.preprocessor import resize_with_padding
        img = resize_with_padding(img, target_size)
    return img, meta


def aggregate_slice_probs(per_slice: list[dict]) -> tuple[dict, dict]:
    """(mean, max) per class over the analysed slices. Classes missing on a
    slice are ignored for that slice."""
    if not per_slice:
        return {}, {}
    classes: list[str] = []
    for probs in per_slice:
        for c in probs:
            if c not in classes:
                classes.append(c)
    mean, mx = {}, {}
    for c in classes:
        vals = [float(p[c]) for p in per_slice if c in p]
        mean[c] = float(np.mean(vals))
        mx[c] = float(np.max(vals))
    return mean, mx
