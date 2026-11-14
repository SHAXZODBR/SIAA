"""src/pipeline/dicom_objects.py — which objects may describe a study."""
from pydicom.dataset import Dataset

from pydicom.dataset import FileMetaDataset

from src.pipeline.dicom_objects import (
    NON_IMAGE_MODALITIES, SECONDARY_CAPTURE_SOP_ROOT, is_image_object, is_secondary_capture, modality_of,
    pixel_count, sop_class_of)


def _ds(modality=None, rows=None, cols=None) -> Dataset:
    ds = Dataset()
    if modality is not None:
        ds.Modality = modality
    if rows is not None:
        ds.Rows = rows
    if cols is not None:
        ds.Columns = cols
    return ds


def test_acquired_images_count():
    assert is_image_object(_ds('MR', 256, 256))
    assert is_image_object(_ds('CT', 512, 512))
    assert is_image_object(_ds(' mr ', 64, 64))


def test_reports_and_presentation_states_do_not_count():
    for m in ('SR', 'PR', 'KO', 'DOC'):
        assert m in NON_IMAGE_MODALITIES
        assert not is_image_object(_ds(m))               # no image matrix at all
        assert not is_image_object(_ds(m, 512, 512))     # and not even with one


def test_other_modality_with_pixels_does_not_describe_the_study():
    assert not is_image_object(_ds('OT', 800, 600))      # screenshot / scanned form


def test_image_without_a_matrix_does_not_count():
    assert not is_image_object(_ds('MR'))
    assert not is_image_object(_ds('MR', 256, None))
    assert not is_image_object(_ds('MR', 0, 256))


def test_missing_modality_with_pixels_still_counts():
    assert is_image_object(_ds(None, 256, 256))          # anonymised header
    assert modality_of(_ds(None, 1, 1)) == ''


def test_helpers_never_raise_on_junk():
    class Broken:
        def get(self, name):
            raise RuntimeError('corrupt')
    assert modality_of(Broken()) == ''
    assert pixel_count(Broken()) == 0
    assert not is_image_object(Broken())
    assert pixel_count(_ds('MR', 'x', 2)) == 0


CT_IMAGE = '1.2.840.10008.5.1.4.1.1.2'
ENHANCED_MR = '1.2.840.10008.5.1.4.1.1.4.1'


def test_secondary_capture_with_the_acquisition_modality_is_not_an_acquired_image():
    """A CT "Dose Report" is a Secondary Capture with Modality 'CT' and pixels: it must not describe the study."""
    for sop in (SECONDARY_CAPTURE_SOP_ROOT, SECONDARY_CAPTURE_SOP_ROOT + '.2', SECONDARY_CAPTURE_SOP_ROOT + '.4'):
        ds = _ds('CT', 512, 512)
        ds.SOPClassUID = sop
        assert is_secondary_capture(ds) and not is_image_object(ds)
    for sop in (CT_IMAGE, ENHANCED_MR, '1.2.840.10008.5.1.4.1.1.77.1'):     # …1.1.77 is NOT …1.1.7.x
        ds = _ds('CT', 512, 512)
        ds.SOPClassUID = sop
        assert not is_secondary_capture(ds) and is_image_object(ds)


def test_sop_class_falls_back_to_the_file_meta():
    ds = _ds('MR', 64, 64)
    assert sop_class_of(ds) == '' and is_image_object(ds)          # unknown class: judged by pixels + modality
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE_SOP_ROOT
    assert sop_class_of(ds) == SECONDARY_CAPTURE_SOP_ROOT and not is_image_object(ds)
    ds.SOPClassUID = ENHANCED_MR                                    # the data set's own tag wins
    assert sop_class_of(ds) == ENHANCED_MR and is_image_object(ds)


def test_sop_class_helpers_never_raise_on_junk():
    class Broken:
        def get(self, name):
            raise RuntimeError('corrupt')
    assert sop_class_of(Broken()) == '' and not is_secondary_capture(Broken())
