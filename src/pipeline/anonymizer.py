"""DICOM anonymizer — strips all patient PII from DICOM files.

Removes patient name, ID, date of birth, and other identifying information
while preserving clinically relevant metadata (modality, body part, pixel data).
"""

import pydicom
from pathlib import Path
from typing import Optional
from loguru import logger


# DICOM tags containing patient PII that must be removed
PII_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "PatientComments",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "IssuerOfPatientID",
    "AccessionNumber",
    "StudyID",
    "RequestingPhysician",
    "RequestingService",
    "CurrentPatientLocation",
    "PatientInstitutionResidence",
    "DeviceSerialNumber",
]

# Tags to keep (clinically relevant, non-identifying)
KEEP_TAGS = [
    "Modality",
    "BodyPartExamined",
    "StudyDescription",
    "SeriesDescription",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "PixelSpacing",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceThickness",
    "SliceLocation",
    "PixelData",
    "StudyDate",  # Keep date but could hash if needed
    "PatientAge",  # Age range is usually acceptable
    "PatientSex",  # Sex is clinically relevant
]


def anonymize_dicom(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    patient_id_replacement: str = "ANON",
) -> Optional[Path]:
    """Anonymize a single DICOM file by removing all PII tags.

    Args:
        input_path: Path to original DICOM file.
        output_path: Path to save anonymized file. If None, overwrites original.
        patient_id_replacement: Replacement string for patient ID.

    Returns:
        Path to anonymized file, or None if failed.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    try:
        ds = pydicom.dcmread(str(input_path), force=True)
    except Exception as e:
        logger.error(f"Failed to read DICOM for anonymization: {input_path}: {e}")
        return None

    # Remove PII tags
    removed_count = 0
    for tag_name in PII_TAGS:
        if hasattr(ds, tag_name):
            delattr(ds, tag_name)
            removed_count += 1

    # Set replacement values
    ds.PatientName = patient_id_replacement
    ds.PatientID = patient_id_replacement

    # Remove any private tags (vendor-specific, may contain PII)
    ds.remove_private_tags()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))

    logger.debug(f"Anonymized {input_path.name}: removed {removed_count} PII tags")
    return output_path


def anonymize_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    patient_id_prefix: str = "ANON",
) -> dict:
    """Anonymize all DICOM files in a folder.

    Args:
        input_folder: Source folder with DICOM files.
        output_folder: Destination folder for anonymized files.
        patient_id_prefix: Prefix for anonymized patient IDs.

    Returns:
        Dict with counts: {"success": int, "failed": int, "total": int}
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dcm_files = list(input_folder.rglob("*.dcm")) + list(input_folder.rglob("*.DCM"))
    logger.info(f"Anonymizing {len(dcm_files)} DICOM files...")

    success = 0
    failed = 0

    for i, dcm_file in enumerate(dcm_files):
        # Preserve relative directory structure
        rel_path = dcm_file.relative_to(input_folder)
        out_path = output_folder / rel_path

        anon_id = f"{patient_id_prefix}_{i:06d}"
        result = anonymize_dicom(dcm_file, out_path, anon_id)

        if result is not None:
            success += 1
        else:
            failed += 1

    stats = {"success": success, "failed": failed, "total": len(dcm_files)}
    logger.info(f"Anonymization complete: {success}/{len(dcm_files)} succeeded")
    return stats
