"""Study header comes from the first IMAGE object, not the first file.

A PACS export puts non-image objects in the study folder — structured reports
('SR', Enhanced SR Storage), presentation states ('PR'), secondary captures of
a request form ('OT') — and they often come first. _read_study_headers used to
take modality / body part / patient fields from the first readable file, so an
MR brain study with an SR in front read as modality 'SR', body part '' and was
answered with 422 requires_review instead of being analysed.

These tests build SYNTHETIC pydicom objects (ids >= 9000000, dates in 2031 — no
PHI) under tmp_path and check the invariant: putting non-image objects in front
of (or anywhere among) the files changes neither hdr['modality'],
hdr['body_part'], _is_brain_mr(hdr), nor the route _plan_series_route picks.
An upload with no image object at all still describes itself by its first
readable object, exactly as before.
"""

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian, MRImageStorage, SecondaryCaptureImageStorage, generate_uid,
)

from src.inference import server as srv
from test_ct_series_routing import HEAD_HU, build_series

ENHANCED_SR = "1.2.840.10008.5.1.4.1.1.88.22"
GRAYSCALE_PR = "1.2.840.10008.5.1.4.1.1.11.1"

IMAGE_PATIENT = {"PatientID": "9000881", "StudyDate": "20310214", "AccessionNumber": "A9000881"}
# Non-image objects carry DIFFERENT patient/study values so the tests can tell
# which object the header came from (a real SR shares them with its study).
OBJECT_PATIENT = {"PatientID": "9000999", "StudyDate": "20310301", "AccessionNumber": "A9000999"}


def _base(path: Path, sop_class: str, modality: str, *, study_uid: str, patient: dict,
          series_desc: str = "") -> Dataset:
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientID = patient["PatientID"]
    ds.PatientName = "Synthetic^Objects"
    ds.StudyDate = patient["StudyDate"]
    ds.AccessionNumber = patient["AccessionNumber"]
    ds.StudyInstanceUID = study_uid
    ds.StudyDescription = "Synthetic 2031"
    ds.SeriesInstanceUID = generate_uid()
    if series_desc:
        ds.SeriesDescription = series_desc
    return ds


def _add_pixels(ds: Dataset, size: int, value: int) -> None:
    ds.Rows, ds.Columns = size, size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((size, size), value, dtype=np.uint16).tobytes()


def write_sr(path: Path, *, study_uid: str) -> Path:
    """Enhanced SR: Modality 'SR', no BodyPartExamined, no image matrix."""
    ds = _base(path, ENHANCED_SR, "SR", study_uid=study_uid, patient=OBJECT_PATIENT,
               series_desc="Report")
    ds.save_as(str(path), enforce_file_format=True)
    return path


def write_pr(path: Path, *, study_uid: str) -> Path:
    """Grayscale softcopy presentation state: Modality 'PR', no image matrix."""
    ds = _base(path, GRAYSCALE_PR, "PR", study_uid=study_uid, patient=OBJECT_PATIENT,
               series_desc="Presentation state")
    ds.save_as(str(path), enforce_file_format=True)
    return path


def write_ot(path: Path, *, study_uid: str) -> Path:
    """Secondary capture WITH pixels (a scanned request form), Modality 'OT'."""
    ds = _base(path, SecondaryCaptureImageStorage, "OT", study_uid=study_uid,
               patient=OBJECT_PATIENT, series_desc="Scanned form")
    ds.BodyPartExamined = "CHEST"            # a misleading tag must not leak into the study
    _add_pixels(ds, 64, 200)                 # larger than the MR images
    ds.save_as(str(path), enforce_file_format=True)
    return path


def write_mr_series(root: Path, *, study_uid: str, body_part: str, desc: str, n: int = 4,
                    start_index: int = 100) -> list[Path]:
    series_uid = generate_uid()
    out = []
    for i in range(n):
        p = root / f"{start_index + i:04d}.dcm"
        ds = _base(p, MRImageStorage, "MR", study_uid=study_uid, patient=IMAGE_PATIENT,
                   series_desc=desc)
        ds.SeriesInstanceUID = series_uid
        ds.BodyPartExamined = body_part
        ds.ProtocolName = desc
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 5.0]
        _add_pixels(ds, 32, 500)
        ds.save_as(str(p), enforce_file_format=True)
        out.append(p)
    return out


_OBJECT_WRITERS = {"SR": write_sr, "PR": write_pr, "OT": write_ot}


@pytest.fixture
def mr_brain(tmp_path) -> dict:
    study_uid = generate_uid()
    mr = write_mr_series(tmp_path, study_uid=study_uid, body_part="BRAIN", desc="T2 TSE AX")
    return {"root": tmp_path, "study_uid": study_uid, "mr": mr}


def _object(kind: str, root: Path, study_uid: str, index: int = 0) -> Path:
    return _OBJECT_WRITERS[kind](root / f"{index:04d}.dcm", study_uid=study_uid)


_HEADER_KEYS = [out for _tag, out in srv._HEADER_FIELDS]


# ─── 1-3, 8, 9: a non-image object first, MR brain after ──────────────────

@pytest.mark.parametrize("kind", ["SR", "PR", "OT"])
def test_non_image_object_first_mr_brain_keeps_its_header(mr_brain, kind):
    obj = _object(kind, mr_brain["root"], mr_brain["study_uid"])
    paths = [obj] + mr_brain["mr"]
    hdr = srv._read_study_headers(paths)
    assert hdr["modality"] == "MR"
    assert hdr["body_part"] == "BRAIN"
    assert srv._is_brain_mr(hdr) is True
    assert hdr["readable"] == len(paths)                      # the object still counts
    # patient / study fields come from the image object, not the leading object
    assert hdr["patient_id"] == IMAGE_PATIENT["PatientID"]
    assert hdr["study_date"] == IMAGE_PATIENT["StudyDate"]
    assert hdr["accession_number"] == IMAGE_PATIENT["AccessionNumber"]
    assert hdr["study_instance_uid"] == mr_brain["study_uid"]


def test_several_non_image_objects_first(mr_brain):
    objs = [_object(k, mr_brain["root"], mr_brain["study_uid"], i) for i, k in enumerate(("SR", "PR", "OT"))]
    hdr = srv._read_study_headers(objs + mr_brain["mr"])
    assert (hdr["modality"], hdr["body_part"]) == ("MR", "BRAIN")
    assert srv._is_brain_mr(hdr) is True
    assert hdr["readable"] == 3 + len(mr_brain["mr"])


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_sr_position_does_not_change_header_or_route(mr_brain, position):
    sr = _object("SR", mr_brain["root"], mr_brain["study_uid"])
    mr = mr_brain["mr"]
    order = {"first": [sr] + mr, "middle": mr[:2] + [sr] + mr[2:], "last": mr + [sr]}[position]
    hdr = srv._read_study_headers(order)
    ref = srv._read_study_headers(mr)
    for k in _HEADER_KEYS:
        assert hdr[k] == ref[k], k
    assert srv._is_brain_mr(hdr) is srv._is_brain_mr(ref) is True
    route, ref_route = srv._plan_series_route(hdr, order), srv._plan_series_route(ref, mr)
    assert route["modality_key"] == ref_route["modality_key"]


# ─── 4: image first, SR after — only the all-object fields differ ──────────

def test_image_first_sr_after_differs_only_in_all_object_fields(mr_brain):
    sr = _object("SR", mr_brain["root"], mr_brain["study_uid"])
    image_only = srv._read_study_headers(mr_brain["mr"])
    with_sr = srv._read_study_headers(mr_brain["mr"] + [sr])
    differs = {k for k in set(image_only) | set(with_sr) if image_only.get(k) != with_sr.get(k)}
    assert differs == {"readable", "series_descriptions", "series_groups"}
    assert (image_only["readable"], with_sr["readable"]) == (len(mr_brain["mr"]), len(mr_brain["mr"]) + 1)
    assert with_sr["series_descriptions"] == image_only["series_descriptions"] + ["Report"]
    assert with_sr["series_groups"][:-1] == image_only["series_groups"]
    assert with_sr["series_groups"][-1]["paths"] == [sr]
    assert with_sr["best_path"] == image_only["best_path"]          # SR has no pixels


# ─── 5: SR first, CT head series ───────────────────────────────────────────

def test_sr_first_ct_head_routes_like_image_first(tmp_path):
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=6, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=100)
    sr = write_sr(tmp_path / "0000.dcm", study_uid=study_uid)

    sr_first, image_first = [sr] + head, head + [sr]
    hdr_a = srv._read_study_headers(sr_first)
    hdr_b = srv._read_study_headers(image_first)
    assert hdr_a["modality"] == hdr_b["modality"] == "CT"
    assert hdr_a["body_part"] == hdr_b["body_part"] == "HEAD"
    assert hdr_a["readable"] == hdr_b["readable"] == len(head) + 1

    route_a = srv._plan_series_route(hdr_a, sr_first)
    route_b = srv._plan_series_route(hdr_b, image_first)
    image_only = srv._plan_series_route(srv._read_study_headers(head), head)
    assert route_a["modality_key"] is not None
    assert route_a["modality_key"] == route_b["modality_key"] == image_only["modality_key"]
    assert set(route_a["paths"]) == set(route_b["paths"])
    assert set(head) <= set(route_a["paths"])
    assert route_a["body_part"] == route_b["body_part"] == "HEAD"


# ─── 5b: a secondary capture with the acquisition's own Modality ───────────

def write_dose_report(path: Path, *, study_uid: str, size: int = 96, body_part="NECK") -> Path:
    """What a CT scanner adds to every study: a Secondary Capture "Dose Report" page with
    Modality 'CT', pixels, and a body part that says nothing about the examined region.
    `body_part=None` omits BodyPartExamined."""
    ds = _base(path, SecondaryCaptureImageStorage, "CT", study_uid=study_uid, patient=OBJECT_PATIENT,
               series_desc="Dose Report")
    if body_part is not None:
        ds.BodyPartExamined = body_part
    _add_pixels(ds, size, 900)               # larger than the CT slices of build_series
    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_dose_report_first_does_not_describe_or_preview_a_ct_head_study(tmp_path):
    """Seen on a real head CT: the dose-report page sorted first made the study's body part 'NECK'
    and, being the largest matrix, became the preview / single-file fallback."""
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=6, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=100)
    dose = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid)
    image_only = srv._read_study_headers(head)
    for files in ([dose] + head, head + [dose], head[:3] + [dose] + head[3:]):
        hdr = srv._read_study_headers(files)
        assert hdr["modality"] == "CT" and hdr["body_part"] == "HEAD"
        assert hdr["patient_id"] == image_only["patient_id"] != OBJECT_PATIENT["PatientID"]
        assert hdr["readable"] == len(head) + 1
        assert hdr["best_path"] in head, "the preview must be an acquired slice, not the dose-report page"
        route = srv._plan_series_route(hdr, files)
        assert route["modality_key"] == srv._plan_series_route(image_only, head)["modality_key"] is not None
        assert set(head) <= set(route["paths"])


def test_upload_of_secondary_captures_only_still_describes_and_previews_itself(tmp_path):
    study_uid = generate_uid()
    small = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid, size=32)
    big = write_dose_report(tmp_path / "0001.dcm", study_uid=study_uid, size=96)
    hdr = srv._read_study_headers([small, big])
    assert hdr["modality"] == "CT" and hdr["body_part"] == "NECK"          # first readable object, as before
    assert hdr["best_path"] == big and hdr["readable"] == 2               # largest matrix, as before


def test_screenshot_larger_than_the_mr_images_is_not_the_preview(mr_brain):
    ot = write_ot(mr_brain["root"] / "0000.dcm", study_uid=mr_brain["study_uid"])   # 64 px; the MR images are 32
    hdr = srv._read_study_headers([ot] + mr_brain["mr"])
    assert hdr["modality"] == "MR" and hdr["best_path"] in mr_brain["mr"]


# ─── 6: no image object at all — unchanged behaviour ───────────────────────

def test_sr_only_upload_describes_itself(tmp_path):
    sr = write_sr(tmp_path / "0000.dcm", study_uid=generate_uid())
    hdr = srv._read_study_headers([sr])
    assert hdr["modality"] == "SR"
    assert hdr["body_part"] == ""
    assert hdr["patient_id"] == OBJECT_PATIENT["PatientID"]
    assert hdr["readable"] == 1
    assert srv._is_brain_mr(hdr) is False


def test_non_image_only_upload_uses_first_readable_object(tmp_path):
    study_uid = generate_uid()
    pr = write_pr(tmp_path / "0000.dcm", study_uid=study_uid)
    sr = write_sr(tmp_path / "0001.dcm", study_uid=study_uid)
    hdr = srv._read_study_headers([pr, sr])
    assert hdr["modality"] == "PR"
    assert hdr["readable"] == 2


def test_unreadable_files_are_not_counted(tmp_path, mr_brain):
    junk = tmp_path / "junk.dcm"
    junk.write_bytes(b"not a dicom" * 50)
    sr = _object("SR", mr_brain["root"], mr_brain["study_uid"])
    hdr = srv._read_study_headers([junk, sr] + mr_brain["mr"])
    assert hdr["readable"] == 1 + len(mr_brain["mr"])
    assert hdr["modality"] == "MR"


# ─── 7: SR first, cervical-spine MR ────────────────────────────────────────

def test_sr_first_spine_mr_is_mr_but_not_brain(tmp_path):
    study_uid = generate_uid()
    spine = write_mr_series(tmp_path, study_uid=study_uid, body_part="CSPINE", desc="T2 SAG C-spine")
    sr = write_sr(tmp_path / "0000.dcm", study_uid=study_uid)
    hdr = srv._read_study_headers([sr] + spine)
    assert hdr["modality"] == "MR"
    assert hdr["body_part"] == "CSPINE"
    assert srv._is_brain_mr(hdr) is False


# ─── 10: through POST /analyze/study, panel faked (no weights) ─────────────

def test_sr_first_mr_brain_reaches_brain_panel_through_api(monkeypatch, tmp_path, mr_brain):
    from fastapi.testclient import TestClient
    import src.inference.brain_analysis as ba

    seen = []

    def fake_panel(file_paths, device="cpu", include_experimental=True):
        seen.append(len(file_paths))
        return {
            "rejected": False,
            "sequences_present": ["T2"],
            "detectors_run": ["triage"],
            "findings": [{"detector": "triage", "finding": "Not flagged", "class_name": "normal",
                          "confidence": 0.7, "positive": False, "status": "validated",
                          "sequence_used": "T2"}],
            "overall_assessment": {"abnormal_flagged": False, "flags": [], "text": "No finding flagged."},
            "coverage": [],
            "has_brats_quartet": False,
            "_preview_slice": np.full((16, 16), 0.5, dtype=np.float32),
        }

    monkeypatch.setattr(ba, "analyze_brain_study", fake_panel)
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data")        # isolated settings.json
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    monkeypatch.setattr(srv.state, "gemma_engine", None)
    monkeypatch.setattr(srv.state, "license_valid", True)

    sr = _object("SR", mr_brain["root"], mr_brain["study_uid"])
    paths = [sr] + mr_brain["mr"]
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in paths]
    r = TestClient(srv.app).post("/analyze/study?language=en", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "brain_panel"
    assert body["modality"] == "MR"
    assert body["body_part"] == "BRAIN"
    assert body["detectors_run"] == ["triage"]
    assert body["num_files"] == len(paths)
    assert body["patient_id"] == IMAGE_PATIENT["PatientID"]
    assert seen == [len(paths)]


# ─── 5c: non-image / secondary-capture series take no part in CT routing ───
# A dose report's or an SR's own series must be neither an "informed" series (its
# BodyPartExamined NECK said nothing about the head study) nor an "adopted" one, must
# contribute no route paths and must never be the route's best_path / preview.

from src.pipeline import ct_head_selection as chs   # noqa: E402


def _untagged_head_ct(root: Path, study_uid: str, n: int = 12) -> list[Path]:
    """Axial series with no BodyPartExamined and no region token; only the study says 'CT Head'."""
    return build_series(root, desc="Axial 5mm", n=n, hu_value=HEAD_HU, study_uid=study_uid,
                        series_number=2, start_index=100, study_desc="CT Head")


@pytest.mark.parametrize("hand_made_hdr", [False, True])
def test_case_b_neck_dose_report_does_not_turn_an_untagged_head_ct_into_neck(tmp_path, hand_made_hdr):
    study_uid = generate_uid()
    axial = _untagged_head_ct(tmp_path, study_uid)
    dose = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid)
    ref_hdr = srv._read_study_headers(axial)
    ref = srv._plan_series_route(ref_hdr, axial)
    assert (ref["modality_key"], ref["routing"]["mode"]) == ("head_ct", "first_series_fallback")
    for files in ([dose] + axial, axial + [dose]):
        hdr = srv._read_study_headers(files)
        if hand_made_hdr:
            hdr.pop("series_groups")                 # _group_series path
        plan = srv._plan_series_route(hdr, files)
        assert plan["modality_key"] == "head_ct", plan
        assert plan["routing"]["mode"] == "first_series_fallback"
        assert dose not in plan["paths"]
        assert plan["paths"] == axial
        assert plan["best_path"] in axial


def test_case_a_neck_dose_report_does_not_drop_the_untagged_axial_series(tmp_path):
    study_uid = generate_uid()
    topo = build_series(tmp_path, desc="Topogram", n=2, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=1, start_index=50, study_desc="CT Head")
    axial = _untagged_head_ct(tmp_path, study_uid)
    dose = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid)
    for files in ([dose] + topo + axial, topo + axial + [dose]):
        hdr = srv._read_study_headers(files)
        plan = srv._plan_series_route(hdr, files)
        assert plan["modality_key"] == "head_ct"
        assert set(plan["paths"]) == set(topo) | set(axial), "the untagged axial series must be adopted"
        assert dose not in plan["paths"]
        assert [s["description"] for s in plan["routing"]["series_excluded"]] == []
        sel = chs.select_ct_head_slices(plan["paths"])
        assert sel is not None and sel.series.description == "Axial 5mm"


def test_case_c_dose_report_without_body_part_is_not_adopted_nor_the_preview(tmp_path):
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=12, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=100)
    dose = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid, body_part=None)   # 96 px > 32 px
    for files in ([dose] + head, head + [dose]):
        hdr = srv._read_study_headers(files)
        plan = srv._plan_series_route(hdr, files)
        assert plan["modality_key"] == "head_ct" and plan["body_part"] == "HEAD"
        assert set(plan["paths"]) == set(head)
        assert plan["best_path"] in head
        assert "series_adopted_without_body_part" not in plan["routing"]
        assert srv.largest_non_localizer(plan["paths"]) in head
        # even handed every file, the CT head selection never picks the dose page
        assert chs.largest_non_localizer(files) in head
        sel = chs.select_ct_head_slices(files)
        assert sel is not None and dose not in sel.slice_paths
        assert all(c["description"] != "Dose Report" for c in sel.candidates)


@pytest.mark.parametrize("kind", ["SR", "PR"])
def test_sr_pr_objects_in_a_ct_study_contribute_no_route_paths(tmp_path, kind):
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=6, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=100)
    obj = _object(kind, tmp_path, study_uid)
    for files in ([obj] + head, head + [obj]):
        hdr = srv._read_study_headers(files)
        plan = srv._plan_series_route(hdr, files)
        assert plan["modality_key"] == "head_ct"
        assert set(plan["paths"]) == set(head)
    # first-series fallback (no series carries body-part information): the object is not routed either
    (tmp_path / "untagged").mkdir()
    axial = _untagged_head_ct(tmp_path / "untagged", study_uid)
    files = [obj] + axial
    plan = srv._plan_series_route(srv._read_study_headers(files), files)
    assert plan["modality_key"] == "head_ct" and plan["paths"] == axial


def test_series_groups_say_whether_they_hold_an_image(tmp_path):
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=6, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=100)
    dose = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid)
    sr = write_sr(tmp_path / "0001.dcm", study_uid=study_uid)
    groups = srv._read_study_headers([dose, sr] + head)["series_groups"]
    assert {g["description"]: g["is_image"] for g in groups} == {
        "Dose Report": False, "Report": False, "Head 5mm": True}
    assert srv._group_series([dose, sr] + head) == groups


def test_secondary_captures_only_ct_upload_routes_as_before(tmp_path):
    """No acquired image at all: every group takes part, exactly as before the fix."""
    study_uid = generate_uid()
    small = write_dose_report(tmp_path / "0000.dcm", study_uid=study_uid, size=32)
    big = write_dose_report(tmp_path / "0001.dcm", study_uid=study_uid, size=96)
    files = [small, big]
    hdr = srv._read_study_headers(files)
    plan = srv._plan_series_route(hdr, files)
    assert plan["routing"]["mode"] == "per_series_body_part"
    assert plan["body_part"] == "NECK" and plan["modality_key"] is None
    assert set(plan["paths"]) == set(files)
    # and the CT head helpers still see them
    assert chs.largest_non_localizer(files) == big
    assert sum(s.n_images for s in chs.read_series_headers(files)) == 2
