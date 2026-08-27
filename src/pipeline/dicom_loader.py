"""DICOM file loader and metadata extractor for Sentinel Medical AI.

Handles loading .dcm files, extracting pixel arrays and metadata,
with robust error handling for corrupt/non-standard DICOM files.
"""

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class DicomStudy:
    """Represents a loaded DICOM study with image data and metadata."""

    file_path: str
    pixel_array: np.ndarray  # Raw pixel data
    modality: str = "UNKNOWN"
    body_part: str = "UNKNOWN"
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    study_date: str = ""
    study_description: str = ""
    series_description: str = ""
    institution_name: str = ""
    bits_stored: int = 8
    photometric_interpretation: str = "MONOCHROME2"
    rows: int = 0
    cols: int = 0
    pixel_spacing: list = field(default_factory=list)
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0


def load_dicom(file_path: str | Path) -> Optional[DicomStudy]:
    """Load a single DICOM file and extract pixel array + metadata.

    Args:
        file_path: Path to .dcm file.

    Returns:
        DicomStudy object or None if file is corrupt/invalid.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"DICOM file not found: {file_path}")
        return None

    try:
        ds = pydicom.dcmread(str(file_path), force=True)
    except (InvalidDicomError, Exception) as e:
        logger.error(f"Failed to read DICOM file {file_path}: {e}")
        return None

    # Extract pixel array
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except (AttributeError, TypeError, Exception) as e:
        logger.error(f"No pixel data in DICOM file {file_path}: {e}")
        return None

    # Apply rescale slope/intercept (common in CT)
    rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixel_array = pixel_array * rescale_slope + rescale_intercept

    # Handle photometric interpretation — invert if MONOCHROME1
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Window center/width
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    elif wc is not None:
        wc = float(wc)
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    elif ww is not None:
        ww = float(ww)

    # Pixel spacing
    ps = getattr(ds, "PixelSpacing", [])
    if ps:
        ps = [float(x) for x in ps]

    study = DicomStudy(
        file_path=str(file_path),
        pixel_array=pixel_array,
        modality=str(getattr(ds, "Modality", "UNKNOWN")),
        body_part=str(getattr(ds, "BodyPartExamined", "UNKNOWN")),
        patient_id=str(getattr(ds, "PatientID", "")),
        patient_name=str(getattr(ds, "PatientName", "")),
        patient_age=str(getattr(ds, "PatientAge", "")),
        patient_sex=str(getattr(ds, "PatientSex", "")),
        study_date=str(getattr(ds, "StudyDate", "")),
        study_description=str(getattr(ds, "StudyDescription", "")),
        series_description=str(getattr(ds, "SeriesDescription", "")),
        institution_name=str(getattr(ds, "InstitutionName", "")),
        bits_stored=int(getattr(ds, "BitsStored", 8)),
        photometric_interpretation=photometric,
        rows=int(getattr(ds, "Rows", 0)),
        cols=int(getattr(ds, "Columns", 0)),
        pixel_spacing=ps,
        window_center=wc,
        window_width=ww,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
    )

    logger.debug(
        f"Loaded DICOM: {file_path.name} | {study.modality} | "
        f"{study.rows}x{study.cols} | {study.bits_stored}-bit"
    )
    return study


def load_dicom_folder(folder_path: str | Path) -> list[DicomStudy]:
    """Load all DICOM files from a folder (recursive).

    Args:
        folder_path: Path to folder containing .dcm files.

    Returns:
        List of successfully loaded DicomStudy objects.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return []

    # Find all .dcm files (also try files without extension — common in DICOM)
    dcm_files = list(folder_path.rglob("*.dcm"))
    dcm_files += list(folder_path.rglob("*.DCM"))

    # Also check files without extension that might be DICOM
    for f in folder_path.rglob("*"):
        if f.is_file() and f.suffix == "" and f not in dcm_files:
            dcm_files.append(f)

    logger.info(f"Found {len(dcm_files)} potential DICOM files in {folder_path}")

    studies = []
    failed = 0
    for f in dcm_files:
        study = load_dicom(f)
        if study is not None:
            studies.append(study)
        else:
            failed += 1

    logger.info(f"Successfully loaded {len(studies)} studies, {failed} failed")
    return studies
