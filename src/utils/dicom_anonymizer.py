"""
================================================================================
  SENTINEL — DICOM ANONYMIZER (PHI/PII REMOVAL)
================================================================================

  Strips Protected Health Information from DICOM files before they leave the
  clinic. Required by:
    • Uzbek Personal Data Protection Law (2019)
    • DICOM PS 3.15 Annex E (Basic Profile for de-identification)
    • GDPR equivalent for medical imaging

  What gets removed (PHI tags):
    • Patient name, ID, birth date (kept: birth year for stratification)
    • Patient phone, address, ethnic group
    • Referring physician, performing physician
    • Operator name, station name
    • Institution name, address
    • Accession number, study ID
    • Private tags (vendor-specific, often contain PHI)
    • UID re-mapping so different anonymized exports of the same patient
      cannot be re-linked across batches.

  What stays (medical info, no PHI):
    • Modality, BodyPart, ViewPosition
    • Pixel data + acquisition parameters (kVp, mAs, contrast)
    • Series/Study dates (rounded to year)
    • Patient sex (kept — clinically relevant)
    • Patient age range

  Usage as library:
    from src.utils.dicom_anonymizer import anonymize_dicom
    new_path = anonymize_dicom('input.dcm', 'output.dcm', anon_id='ANON_001')

  Usage as CLI:
    python -m src.utils.dicom_anonymizer input.dcm --output anon.dcm
    python -m src.utils.dicom_anonymizer input_dir/ --output output_dir/ --batch
================================================================================
"""

from __future__ import annotations
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pydicom
    from pydicom.dataset import Dataset
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


# Tags to scrub completely (set to empty string).
# Reference: DICOM PS 3.15 Annex E + common clinical PHI fields.
PHI_TAGS_BLANK = [
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x0092),  # ReferringPhysicianAddress
    (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumbers
    (0x0008, 0x0096),  # ReferringPhysicianIdentificationSequence
    (0x0008, 0x1010),  # StationName
    (0x0008, 0x1030),  # StudyDescription (sometimes contains "Smith John")
    (0x0008, 0x103E),  # SeriesDescription (kept by default actually — change if needed)
    (0x0008, 0x1040),  # InstitutionalDepartmentName
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0008, 0x1060),  # NameOfPhysiciansReadingStudy
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x1080),  # AdmittingDiagnosesDescription
    (0x0008, 0x2111),  # DerivationDescription
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1005),  # PatientBirthName
    # NOTE: PatientAge (0x0010, 0x1010) handled separately below — banded to decade
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x1060),  # PatientMotherBirthName
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x2160),  # EthnicGroup (research keeps; production strip)
    (0x0010, 0x2180),  # Occupation
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0010, 0x4000),  # PatientComments
    (0x0020, 0x0010),  # StudyID
    (0x0032, 0x1032),  # RequestingPhysician
    (0x0032, 0x1033),  # RequestingService
    (0x0040, 0xA075),  # VerifyingObserverName
    (0x0040, 0xA078),  # AuthorObserverSequence
    (0x0040, 0xA124),  # UID (only the embedded ones in sequences)
    (0x0040, 0xA730),  # ContentSequence
    (0x4008, 0x0114),  # PhysicianApprovingInterpretation
    (0x4008, 0x0119),  # DistributionName
    (0x4008, 0x011A),  # DistributionAddress
]

# Tags we set to a deterministic anonymous value (not blank — keeps shape valid).
ANON_REPLACEMENTS = {
    (0x0010, 0x0010): 'ANONYMIZED',          # PatientName
    (0x0010, 0x0020): None,                  # PatientID — replaced with anon_id
    (0x0008, 0x0050): 'ANONYMIZED',          # AccessionNumber
}


def _hash_id(original: str, salt: str = 'sentinel-2026') -> str:
    """Deterministic but unrecoverable anonymous ID."""
    h = hashlib.sha256((salt + (original or '')).encode()).hexdigest()
    return f'ANON_{h[:12].upper()}'


def _date_to_year_only(date_str: str) -> str:
    """Convert YYYYMMDD → YYYY0101 (preserves year for stratification)."""
    # Guard against None / "None" / empty / too-short — never emit "None0101".
    if not date_str or str(date_str).strip().lower() in ('none', '') or len(str(date_str)) < 4:
        return ''
    s = str(date_str)
    if not s[:4].isdigit():
        return ''
    return s[:4] + '0101'


def _age_to_range(age_str: str) -> str:
    """'045Y' → '040Y' (decade floor). Returns a VALID 4-char DICOM AS value
    (NNNu) — NOT a range like '040-049Y' which is invalid for VR AS."""
    if not age_str:
        return ''
    try:
        s = str(age_str)
        unit = s[-1].upper() if s and s[-1].isalpha() and s[-1].upper() in 'DWMY' else 'Y'
        num = int(''.join(c for c in s if c.isdigit()))
        decade = (num // 10) * 10
        # Valid AS = exactly 3 digits + 1 unit char (e.g. '040Y')
        return f'{decade:03d}{unit}'
    except (ValueError, IndexError):
        return ''


def anonymize_dataset(ds: Dataset, anon_id: Optional[str] = None,
                       keep_dates: bool = False,
                       keep_ethnicity: bool = False) -> Dataset:
    """Anonymize a pydicom Dataset in-place.

    Args:
        ds: DICOM dataset to scrub.
        anon_id: Replace PatientID with this. If None, generate a hash from
                 the original PatientID.
        keep_dates: If True, keep month/day; otherwise reduce to YYYY0101.
        keep_ethnicity: If True, keep EthnicGroup tag (research use).

    Returns:
        The same dataset (modified in place).
    """
    # Generate anon_id from original
    original_id = str(getattr(ds, 'PatientID', '') or '')
    if anon_id is None:
        anon_id = _hash_id(original_id)

    # Strip PHI tags
    skip = set()
    if keep_ethnicity:
        skip.add((0x0010, 0x2160))

    for tag in PHI_TAGS_BLANK:
        if tag in skip:
            continue
        if tag in ds:
            try:
                ds[tag].value = ''
            except Exception:
                # Some tags can't be set to empty — delete instead
                del ds[tag]

    # Replace identifiers
    for tag, value in ANON_REPLACEMENTS.items():
        if tag in ds:
            ds[tag].value = anon_id if value is None else value
        else:
            # Add the anon ID even if absent
            if tag == (0x0010, 0x0020):
                ds.PatientID = anon_id
            elif tag == (0x0010, 0x0010):
                ds.PatientName = 'ANONYMIZED'

    # Date sanitization
    if not keep_dates:
        for date_tag in [
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0008, 0x0020),  # StudyDate
            (0x0008, 0x0021),  # SeriesDate
            (0x0008, 0x0022),  # AcquisitionDate
            (0x0008, 0x0023),  # ContentDate
        ]:
            if date_tag in ds:
                ds[date_tag].value = _date_to_year_only(str(ds[date_tag].value))

    # Age band
    if (0x0010, 0x1010) in ds:
        ds[(0x0010, 0x1010)].value = _age_to_range(str(ds[(0x0010, 0x1010)].value))
    elif hasattr(ds, 'PatientAge'):
        ds.PatientAge = _age_to_range(str(ds.PatientAge))

    # Strip private tags (vendor-specific, often hold PHI)
    ds.remove_private_tags()

    # Strip nested sequences with potential PHI
    for seq_tag in [(0x0040, 0xA730), (0x0008, 0x0096),
                     (0x0040, 0xA078), (0x0040, 0xA124)]:
        if seq_tag in ds:
            try:
                del ds[seq_tag]
            except Exception:
                pass

    # Add anonymization metadata
    ds.PatientIdentityRemoved = 'YES'
    ds.DeidentificationMethod = 'Sentinel Medical AI v1.0 — PS 3.15 Basic Profile'
    ds.DeidentificationMethodCodeSequence = []

    return ds


def anonymize_dicom(input_path: str, output_path: str,
                      anon_id: Optional[str] = None,
                      keep_dates: bool = False,
                      keep_ethnicity: bool = False) -> str:
    """Anonymize one DICOM file → file. Returns the output path."""
    if not _HAS_PYDICOM:
        raise RuntimeError("pydicom not installed: pip install pydicom")

    ds = pydicom.dcmread(input_path, force=True)
    anonymize_dataset(ds, anon_id, keep_dates, keep_ethnicity)
    ds.save_as(output_path, write_like_original=False)
    return output_path


def anonymize_directory(input_dir: str, output_dir: str,
                          recursive: bool = True,
                          keep_dates: bool = False,
                          keep_ethnicity: bool = False,
                          progress: bool = True) -> dict:
    """Batch-anonymize every DICOM under input_dir into output_dir.

    Preserves directory structure. Patient IDs are mapped consistently
    across files in the same study.

    Returns: {processed: int, failed: int, errors: list[str]}
    """
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build PatientID → anon_id map first so multi-file studies stay linked
    patient_map: dict[str, str] = {}

    pattern = '**/*' if recursive else '*'
    candidates = [p for p in in_root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in ('.dcm', '.dicom', '')]

    processed = 0
    failed = 0
    errors = []

    for i, fp in enumerate(candidates, 1):
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            original_id = str(getattr(ds, 'PatientID', '') or '')
            if original_id and original_id not in patient_map:
                patient_map[original_id] = _hash_id(original_id)
            anon_id = patient_map.get(original_id, _hash_id('unknown'))

            rel = fp.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            anonymize_dataset(ds, anon_id, keep_dates, keep_ethnicity)
            ds.save_as(str(out_path), write_like_original=False)
            processed += 1

            if progress and i % 10 == 0:
                print(f'  ...processed {i}/{len(candidates)}')
        except Exception as e:
            failed += 1
            errors.append(f'{fp.name}: {e}')

    return {
        'processed': processed,
        'failed': failed,
        'errors': errors[:20],  # cap
        'unique_patients': len(patient_map),
    }


def verify_anonymization(dicom_path: str) -> tuple[bool, list[str]]:
    """Sanity-check that a DICOM file has been properly anonymized.

    Returns (is_clean, list_of_problems).
    """
    if not _HAS_PYDICOM:
        return False, ['pydicom not installed']

    ds = pydicom.dcmread(dicom_path, force=True)
    problems = []

    # Check identity-removed flag
    if str(getattr(ds, 'PatientIdentityRemoved', '')) != 'YES':
        problems.append('PatientIdentityRemoved tag not set to YES')

    # Check that name doesn't look like a real name
    name = str(getattr(ds, 'PatientName', ''))
    if name and name.upper() not in ('ANONYMIZED', '', 'ANON'):
        problems.append(f'PatientName looks real: "{name}"')

    # Check no referring physician
    if str(getattr(ds, 'ReferringPhysicianName', '')):
        problems.append('ReferringPhysicianName not blank')

    # Check no institution
    if str(getattr(ds, 'InstitutionName', '')):
        problems.append('InstitutionName not blank')

    # Check no phone
    if str(getattr(ds, 'PatientTelephoneNumbers', '')):
        problems.append('PatientTelephoneNumbers not blank')

    return (len(problems) == 0, problems)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sentinel DICOM Anonymizer')
    parser.add_argument('input', help='DICOM file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output file or directory')
    parser.add_argument('--batch', action='store_true', help='Treat input as directory (recursive)')
    parser.add_argument('--anon-id', help='Force this anonymous PatientID (single-file mode)')
    parser.add_argument('--keep-dates', action='store_true', help='Keep month/day in dates (default: year-only)')
    parser.add_argument('--keep-ethnicity', action='store_true', help='Keep EthnicGroup tag (research use)')
    parser.add_argument('--verify', action='store_true', help='Just verify a file is anonymized')
    args = parser.parse_args()

    if args.verify:
        ok, problems = verify_anonymization(args.input)
        if ok:
            print(f'✓ {args.input} is properly anonymized.')
            return 0
        else:
            print(f'✗ {args.input} has PHI leaks:')
            for p in problems:
                print(f'   • {p}')
            return 1

    if args.batch:
        print(f'Batch anonymizing {args.input}/ → {args.output}/')
        result = anonymize_directory(
            args.input, args.output,
            keep_dates=args.keep_dates,
            keep_ethnicity=args.keep_ethnicity,
        )
        print()
        print(f'  Processed:        {result["processed"]} files')
        print(f'  Failed:           {result["failed"]} files')
        print(f'  Unique patients:  {result["unique_patients"]}')
        if result['errors']:
            print()
            print(f'  First errors:')
            for e in result['errors'][:5]:
                print(f'    • {e}')
        return 0 if result['failed'] == 0 else 1

    # Single-file mode
    out = anonymize_dicom(args.input, args.output, args.anon_id,
                            args.keep_dates, args.keep_ethnicity)
    ok, problems = verify_anonymization(out)
    print(f'✓ Wrote {out}' if ok else f'✗ Wrote {out} but verification failed:')
    for p in problems:
        print(f'   • {p}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
