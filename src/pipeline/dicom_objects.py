"""Which DICOM objects can speak for a study.

A study folder (and so an upload, a PACS push, a drive export) holds more than
images: structured reports (SR), presentation states (PR), key-object notes (KO),
encapsulated documents, segmentations. These objects often sort or arrive FIRST,
and they carry their own Modality ('SR', 'PR', …) and no BodyPartExamined. Code
that describes a study by "the first file" then reads an MR brain study as
modality 'SR' — the study is not recognised, not routed, not trained on.

Everything that answers "what modality / body part is this study" must look at
an IMAGE object instead. This module is the one definition of that.
"""
from __future__ import annotations

# Modality values that say nothing about how the patient was imaged. 'OT' (Other)
# does carry pixels — screenshots, scanned request forms — but not an acquisition.
NON_IMAGE_MODALITIES = frozenset({
    'SR', 'PR', 'KO', 'DOC', 'SEG', 'REG', 'FID', 'RWV', 'PLAN',
    'RTSTRUCT', 'RTPLAN', 'RTDOSE', 'RTRECORD', 'OT',
})


# Secondary Capture SOP classes (single-frame and the four multi-frame variants): dose reports,
# screenshots, saved reformats. They carry the acquisition's Modality ('CT', 'MR') and pixels,
# but they are not acquired slices — a CT "Dose Report" tagged NECK, sorted first, used to
# describe a whole head study.
SECONDARY_CAPTURE_SOP_ROOT = '1.2.840.10008.5.1.4.1.1.7'


def modality_of(ds) -> str:
    """Upper-case DICOM Modality of a dataset, '' when absent or unreadable."""
    try:
        v = ds.get('Modality')
    except Exception:
        return ''
    return '' if v is None else str(v).strip().upper()


def pixel_count(ds) -> int:
    """Rows × Columns of a dataset header, 0 when it has no image matrix."""
    try:
        return int(ds.get('Rows') or 0) * int(ds.get('Columns') or 0)
    except Exception:
        return 0


def sop_class_of(ds) -> str:
    """SOP Class UID of a dataset (the data set's own tag, else the file meta), '' when unknown."""
    try:
        v = ds.get('SOPClassUID')
        if not v:
            meta = getattr(ds, 'file_meta', None)
            v = meta.get('MediaStorageSOPClassUID') if meta is not None else None
    except Exception:
        return ''
    return '' if v is None else str(v).strip()


def is_secondary_capture(ds) -> bool:
    """True for the Secondary Capture storage classes (…1.1.7 and …1.1.7.1–4)."""
    sop = sop_class_of(ds)
    return sop == SECONDARY_CAPTURE_SOP_ROOT or sop.startswith(SECONDARY_CAPTURE_SOP_ROOT + '.')


def is_image_object(ds) -> bool:
    """True when the object is an acquired image: it has an image matrix, its
    Modality is not one of NON_IMAGE_MODALITIES and it is not a secondary capture.
    An empty Modality with pixels still counts (anonymisers drop the tag; the
    pixels are what the models read). A study that holds ONLY non-image objects
    or secondary captures has no image object: callers then fall back to the
    first readable object, as they always did."""
    return (pixel_count(ds) > 0 and modality_of(ds) not in NON_IMAGE_MODALITIES
            and not is_secondary_capture(ds))
