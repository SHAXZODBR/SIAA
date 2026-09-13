"""Signed report -> DICOM Encapsulated PDF (and push to the local Orthanc PACS).

A hospital PACS only understands DICOM. Once a radiologist has signed a
report in Sentinel, the rendered PDF is wrapped as an Encapsulated PDF
instance (SOP Class 1.2.840.10008.5.1.4.1.1.104.1) that carries the SAME
StudyInstanceUID as the source images, so the PACS files it as a new 'DOC'
series inside the original study - exactly where the clinician will look.

    build_encapsulated_pdf(pdf_bytes, meta, signer_name, signed_at) -> Dataset
    dataset_to_bytes(ds) -> bytes             (Part-10 file, Explicit VR LE)
    extract_pdf(ds) -> bytes                  (inverse, strips the OB pad byte)
    push_to_orthanc(dicom_bytes, url, auth) -> {"orthanc_id", "parent_study", ...}

No patient identifiers are logged here.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

import pydicom
from loguru import logger
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ENCAPSULATED_PDF_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.104.1"
SERIES_DESCRIPTION = "SIAA AI-assisted report (signed)"
# pydicom's public root is fine for generate_uid(); the implementation UID
# below identifies THIS software as the creator of the instance.
SIAA_IMPLEMENTATION_UID = "1.2.826.0.1.3680043.10.1583.1"
SIAA_IMPLEMENTATION_VERSION = "SIAA_1.0.0"
PDF_MAGIC = b"%PDF-"

# meta keys accepted by build_encapsulated_pdf (all optional except
# study_instance_uid, which is what ties the document to the study)
META_KEYS = (
    "patient_id", "patient_name", "patient_sex", "patient_birth_date",
    "study_instance_uid", "accession_number", "study_date", "modality",
    "referring_physician", "study_id", "study_description",
)


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _da(v: Any) -> str:
    """DICOM DA (YYYYMMDD) from 'YYYYMMDD', 'YYYY-MM-DD', an int, or empty."""
    s = _s(v).replace("-", "")
    if s.isdigit() and len(s) == 8:
        return s
    return ""


def _coerce_signed_at(signed_at: Any) -> datetime:
    if isinstance(signed_at, datetime):
        return signed_at
    s = _s(signed_at)
    if s:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now()


def _pn(name: Any) -> str:
    """A DICOM Person Name from free text. 'Last^First' passes through; a plain
    'Dr Firstname Lastname' is kept as a single component (still valid PN)."""
    s = _s(name)
    if not s:
        return ""
    if "^" in s:
        return s
    return s.replace("\\", " ")[:64 * 5]


def is_pdf(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and bytes(data[:5]) == PDF_MAGIC


def build_encapsulated_pdf(
    pdf_bytes: bytes,
    meta: dict,
    signer_name: str,
    signed_at: Any,
    *,
    document_title: Optional[str] = None,
    software_version: str = "1.0.0",
) -> Dataset:
    """Wrap a signed-report PDF as a DICOM Encapsulated PDF instance.

    meta: dict(patient_id, patient_name, patient_sex, patient_birth_date,
               study_instance_uid, accession_number, study_date, modality,
               referring_physician[, study_id, study_description]).
    A NEW SeriesInstanceUID / SOPInstanceUID is generated on every call; the
    StudyInstanceUID is taken from meta so the PACS attaches the document to
    the imaged study (a fresh one is generated only if meta has none).
    """
    if not is_pdf(pdf_bytes):
        raise ValueError("pdf_bytes is not a PDF (missing %PDF- header)")
    meta = dict(meta or {})
    when = _coerce_signed_at(signed_at)
    date_s = when.strftime("%Y%m%d")
    time_s = when.strftime("%H%M%S")
    dt_s = when.strftime("%Y%m%d%H%M%S")
    sop_instance_uid = generate_uid()
    series_uid = generate_uid()
    study_uid = _s(meta.get("study_instance_uid")) or generate_uid()

    ds = Dataset()
    # UTF-8: patient / physician names are frequently Cyrillic here.
    ds.SpecificCharacterSet = "ISO_IR 192"

    # --- SOP Common ---
    ds.SOPClassUID = ENCAPSULATED_PDF_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = datetime.now().strftime("%Y%m%d")
    ds.InstanceCreationTime = datetime.now().strftime("%H%M%S")

    # --- Patient ---
    ds.PatientName = _pn(meta.get("patient_name"))
    ds.PatientID = _s(meta.get("patient_id"))
    ds.PatientBirthDate = _da(meta.get("patient_birth_date"))
    ds.PatientSex = _s(meta.get("patient_sex"))[:1].upper()

    # --- General Study (must match the source study) ---
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = _da(meta.get("study_date")) or date_s
    ds.StudyTime = ""
    ds.AccessionNumber = _s(meta.get("accession_number"))[:16]
    ds.ReferringPhysicianName = _pn(meta.get("referring_physician"))
    ds.StudyID = _s(meta.get("study_id"))[:16]
    if _s(meta.get("study_description")):
        ds.StudyDescription = _s(meta.get("study_description"))[:64]

    # --- Encapsulated Document Series ---
    ds.Modality = "DOC"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 999
    ds.SeriesDate = date_s
    ds.SeriesTime = time_s
    ds.SeriesDescription = SERIES_DESCRIPTION
    # Standard extended: who produced this document instance.
    ds.OperatorsName = _pn(signer_name)

    # --- General / SC Equipment ---
    ds.Manufacturer = "SIAA"
    ds.ManufacturerModelName = "Sentinel Medical AI"
    ds.SoftwareVersions = software_version
    ds.ConversionType = "WSD"

    # --- Encapsulated Document ---
    ds.InstanceNumber = 1
    ds.ContentDate = date_s
    ds.ContentTime = time_s
    ds.AcquisitionDateTime = dt_s
    ds.BurnedInAnnotation = "YES"
    title = document_title or (
        f"{SERIES_DESCRIPTION} - {_s(signer_name) or 'signer'} - {when.isoformat(timespec='seconds')}"
    )
    ds.DocumentTitle = title[:1024]
    ds.VerificationFlag = "VERIFIED"
    ds.ConceptNameCodeSequence = []
    ds.MIMETypeOfEncapsulatedDocument = "application/pdf"
    pdf = bytes(pdf_bytes)
    ds.EncapsulatedDocumentLength = len(pdf)
    if len(pdf) % 2:
        pdf += b"\x00"           # OB values must be even-length (trailing NUL pad)
    ds.EncapsulatedDocument = pdf

    # --- File meta / encoding ---
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = ENCAPSULATED_PDF_SOP_CLASS
    fm.MediaStorageSOPInstanceUID = sop_instance_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = SIAA_IMPLEMENTATION_UID
    fm.ImplementationVersionName = SIAA_IMPLEMENTATION_VERSION
    ds.file_meta = fm
    with warnings.catch_warnings():
        # pydicom 3 deprecates these flags in favour of the transfer syntax in
        # file_meta (which is also set); older readers/writers still consult them.
        warnings.simplefilter("ignore", DeprecationWarning)
        ds.is_implicit_VR = False
        ds.is_little_endian = True
    return ds


def dataset_to_bytes(ds: Dataset) -> bytes:
    """Serialize as a DICOM Part-10 file (preamble + file meta), Explicit VR LE."""
    buf = BytesIO()
    try:
        pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    except TypeError:                      # pydicom < 3
        pydicom.dcmwrite(buf, ds, write_like_original=False)
    return buf.getvalue()


def extract_pdf(ds: Dataset) -> bytes:
    """The embedded PDF, with the OB pad byte removed (uses
    EncapsulatedDocumentLength when present)."""
    raw = bytes(ds.EncapsulatedDocument)
    n = ds.get("EncapsulatedDocumentLength")
    if n is not None and 0 < int(n) <= len(raw):
        return raw[: int(n)]
    if raw.endswith(b"\x00") and not raw[:-1].endswith(b"\x00"):
        return raw[:-1]
    return raw


def push_to_orthanc(
    dicom_bytes: bytes,
    orthanc_url: str,
    auth: Optional[tuple] = None,
    *,
    http=None,
    timeout: float = 60,
) -> dict:
    """POST a DICOM file to Orthanc (REST /instances). Returns
    {"orthanc_id", "parent_study", "parent_series", "status"}.
    Raises RuntimeError on a non-2xx answer or transport failure."""
    if http is None:
        import requests
        http = requests
    url = orthanc_url.rstrip("/") + "/instances"
    try:
        resp = http.post(url, content=dicom_bytes, auth=auth, timeout=timeout,
                         headers={"Content-Type": "application/dicom"}) \
            if _is_httpx_like(http) else \
            http.post(url, data=dicom_bytes, auth=auth, timeout=timeout,
                      headers={"Content-Type": "application/dicom"})
    except Exception as e:                         # connection refused, DNS, TLS...
        raise RuntimeError(f"Orthanc unreachable: {type(e).__name__}") from e
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Orthanc refused the instance: HTTP {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = {}
    # Orthanc answers a list when several instances are stored; we send one.
    if isinstance(body, list):
        body = body[0] if body else {}
    out = {
        "orthanc_id": body.get("ID"),
        "parent_study": body.get("ParentStudy"),
        "parent_series": body.get("ParentSeries"),
        "status": body.get("Status"),
    }
    logger.info(f"PACS push ok: orthanc_id={out['orthanc_id']} status={out['status']}")
    return out


def _is_httpx_like(http) -> bool:
    """httpx clients take `content=`; requests takes `data=`."""
    mod = getattr(type(http), "__module__", "") or ""
    return mod.startswith("httpx") or mod.startswith("starlette") or mod.startswith("fastapi")
