"""D5 — mixed-body-part CT studies are routed PER SERIES, not by the first file.

A CT study can carry a CHEST series and a HEAD series. The study header comes
from the first file (CHEST here), so the old routing either sent the study to
the chest model or let head_ct pick the larger chest series. These tests build
SYNTHETIC pydicom studies (ids >= 9000000, dates in 2031 — no PHI) and check:

  * the HEAD series is selected for head_ct, and the predictor only ever sees
    head slices (head pixels are 40 HU, chest pixels -600 HU → 0 in the brain
    window), even though the chest series is first, larger and 'axial';
  * body-part info comes from BodyPartExamined, else description tokens
    (HEAD / BRAIN / ГОЛОВ …); a study whose series carry none falls back to the
    study header over all files;
  * a chest-only CT never reaches head_ct;
  * with pending detectors hidden (D1) a head CT is accepted but returned as
    requires_review / no_validated_detector without running any model.
"""

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.inference import server as srv

AXIAL = [1, 0, 0, 0, 1, 0]
HEAD_HU = 40          # brain window centre → 0.5 after windowing
CHEST_HU = -600       # far below the brain window → 0.0


def write_ct(path: Path, *, study_uid: str, series_uid: str, desc: str, instance: int,
             hu_value: int, body_part=None, series_number: int = 1, size: int = 32,
             study_desc: str = "CT synthetic 2031") -> Path:
    """One synthetic CT slice (uniform `hu_value`), stored like a scanner does
    (RescaleIntercept -1024). `body_part=None` omits BodyPartExamined."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.SpecificCharacterSet = "ISO_IR 192"          # UTF-8: Russian descriptions survive
    ds.Modality = "CT"
    if body_part is not None:
        ds.BodyPartExamined = body_part
    ds.PatientID = "9000771"
    ds.PatientName = "Synthetic^Routing"
    ds.StudyDate = "20310115"
    ds.AccessionNumber = "A9000771"
    ds.StudyInstanceUID = study_uid
    ds.StudyDescription = study_desc
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = desc
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImageOrientationPatient = [float(v) for v in AXIAL]
    ds.ImagePositionPatient = [0.0, 0.0, float(instance) * 5.0]
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = size, size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    raw = np.full((size, size), hu_value + 1024, dtype=np.int16)
    ds.PixelData = raw.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def build_series(root: Path, *, desc: str, n: int, hu_value: int, body_part=None,
                 study_uid: str, series_number: int, start_index: int, size: int = 32,
                 study_desc: str = "CT synthetic 2031") -> list[Path]:
    series_uid = generate_uid()
    out = []
    for i in range(n):
        out.append(write_ct(root / f"{start_index + i:04d}.dcm", study_uid=study_uid, series_uid=series_uid,
                            desc=desc, instance=i + 1, hu_value=hu_value, body_part=body_part,
                            series_number=series_number, size=size, study_desc=study_desc))
    return out


@pytest.fixture
def mixed_study(tmp_path) -> dict:
    """CHEST first (40 images, larger, described 'Axial 1.0' — the old selector's
    favourite), HEAD second (12 images)."""
    study_uid = generate_uid()
    chest = build_series(tmp_path, desc="Axial 1.0", n=40, hu_value=CHEST_HU, body_part="CHEST",
                         study_uid=study_uid, series_number=2, start_index=0, size=48)
    head = build_series(tmp_path, desc="Head 5mm", n=12, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=3, start_index=100)
    return {"paths": chest + head, "chest": chest, "head": head}


def _fake_head_ct(monkeypatch, seen: list):
    """Fake loaded head_ct (no weights): records every image it is shown."""
    from src.inference import model_registry as mr

    def fake_predict(image):
        seen.append(np.asarray(image, dtype=np.float32))
        return {"any": 0.1, "subdural": 0.05, "epidural": 0.02, "intraparenchymal": 0.02,
                "intraventricular": 0.01, "subarachnoid": 0.01, "normal": 0.79}

    entry = {"card": mr.REGISTRY["head_ct"], "predictor": fake_predict, "available": True,
             "reason": "loaded from fake", "source_path": None, "multi_label": True,
             "activation": "sigmoid"}
    monkeypatch.setattr(mr, "_loaded_models", {"head_ct": entry})
    monkeypatch.setattr(mr, "_identity_cache", {})
    monkeypatch.setattr(srv.state, "gemma_engine", None)


# ─── series region classification ──────────────────────────────────────────

@pytest.mark.parametrize("body_part,desc,expected", [
    ("HEAD", "", "head"),
    ("BRAIN", "whatever", "head"),
    ("CHEST", "Head 5mm", "other"),          # BodyPartExamined wins over the description
    ("", "Head 5mm", "head"),
    ("", "Голова 5 мм", "head"),
    ("", "Головной мозг", "head"),
    ("", "ОГК 1 мм", "other"),
    ("", "Thorax 1.0", "other"),
    ("", "Axial 5mm", None),
    ("", "", None),
])
def test_series_region(body_part, desc, expected):
    region, _label = srv._series_region(body_part, desc)
    assert region == expected


# ─── the D5 regression: CHEST first, HEAD second ───────────────────────────

def test_headers_group_series_with_their_own_body_part(mixed_study):
    hdr = srv._read_study_headers(mixed_study["paths"])
    assert hdr["body_part"] == "CHEST"                      # first file — the old routing key
    groups = hdr["series_groups"]
    assert [g["body_part"] for g in groups] == ["CHEST", "HEAD"]
    assert [g["region"] for g in groups] == ["other", "head"]
    assert [g["n_images"] for g in groups] == [40, 12]


def test_plan_selects_the_head_series_for_head_ct(mixed_study):
    hdr = srv._read_study_headers(mixed_study["paths"])
    plan = srv._plan_series_route(hdr, mixed_study["paths"])
    assert plan["modality_key"] == "head_ct"
    assert plan["body_part"] == "HEAD"
    assert set(plan["paths"]) == set(mixed_study["head"])
    assert not set(plan["paths"]) & set(mixed_study["chest"])
    assert plan["routing"]["mode"] == "per_series_body_part"
    assert [s["body_part"] for s in plan["routing"]["series_excluded"]] == ["CHEST"]


def test_head_ct_analyses_only_head_slices(monkeypatch, mixed_study):
    seen: list = []
    _fake_head_ct(monkeypatch, seen)
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: True)
    hdr = srv._read_study_headers(mixed_study["paths"])
    resp = srv._analyze_study_sync(mixed_study["paths"], hdr, "en", "test-ct-routing-1")
    assert resp["route"] == "head_ct" and resp["detectors_run"] == ["head_ct"]
    assert resp["series_used"] == "Head 5mm"
    assert resp["body_part"] == "HEAD"
    assert resp["slice_rows_cols"] == [32, 32]               # the head series, not the 48×48 chest
    assert seen, "head_ct predictor never called"
    # every slice the model saw is a head slice (40 HU → ~0.5), never chest (-600 HU → 0)
    assert all(img.max() > 0.3 for img in seen), [float(img.max()) for img in seen]
    assert resp["routing"]["series_excluded"][0]["body_part"] == "CHEST"
    assert resp["requires_review"] is False


def test_without_any_body_part_info_falls_back_to_the_study_header(tmp_path):
    study_uid = generate_uid()
    a = build_series(tmp_path, desc="Axial 5mm", n=6, hu_value=HEAD_HU, study_uid=study_uid,
                     series_number=2, start_index=0, study_desc="HEAD CT 2031")
    b = build_series(tmp_path, desc="Series 3", n=6, hu_value=HEAD_HU, study_uid=study_uid,
                     series_number=3, start_index=50, study_desc="HEAD CT 2031")
    paths = a + b
    hdr = srv._read_study_headers(paths)
    assert all(g["region"] is None for g in hdr["series_groups"])
    plan = srv._plan_series_route(hdr, paths)
    assert plan["modality_key"] == "head_ct"                # from the study description
    assert plan["paths"] == paths                            # all files, as before
    assert plan["routing"]["mode"] == "first_series_fallback"


def test_chest_only_ct_never_reaches_head_ct(tmp_path):
    study_uid = generate_uid()
    paths = build_series(tmp_path, desc="Axial 1.0 brain-window recon", n=8, hu_value=CHEST_HU,
                         body_part="CHEST", study_uid=study_uid, series_number=2, start_index=0)
    hdr = srv._read_study_headers(paths)
    plan = srv._plan_series_route(hdr, paths)
    assert plan["modality_key"] != "head_ct"
    assert plan["body_part"] == "CHEST"


def test_description_tokens_route_when_body_part_is_absent(tmp_path):
    study_uid = generate_uid()
    chest = build_series(tmp_path, desc="ОГК 1 мм", n=10, hu_value=CHEST_HU, study_uid=study_uid,
                         series_number=2, start_index=0)
    head = build_series(tmp_path, desc="Голова 5 мм", n=6, hu_value=HEAD_HU, study_uid=study_uid,
                        series_number=3, start_index=50)
    hdr = srv._read_study_headers(chest + head)
    plan = srv._plan_series_route(hdr, chest + head)
    assert plan["modality_key"] == "head_ct"
    assert set(plan["paths"]) == set(head)


def test_hand_made_hdr_without_series_groups_still_routes_per_series(mixed_study):
    hdr = srv._read_study_headers(mixed_study["paths"])
    hdr.pop("series_groups")
    plan = srv._plan_series_route(hdr, mixed_study["paths"])
    assert plan["modality_key"] == "head_ct" and set(plan["paths"]) == set(mixed_study["head"])


# ─── D1 on the CT route: pending hidden → requires_review, nothing runs ────

def test_hidden_pending_head_ct_returns_no_validated_detector(monkeypatch, mixed_study):
    seen: list = []
    _fake_head_ct(monkeypatch, seen)
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    hdr = srv._read_study_headers(mixed_study["paths"])
    resp = srv._analyze_study_sync(mixed_study["paths"], hdr, "ru", "test-ct-routing-2")
    assert seen == [], "no model may run when pending detectors are hidden"
    assert resp["requires_review"] is True and resp["rejected"] is False
    assert resp["rejection_reason"] == "no_validated_detector"
    assert resp["route"] == "head_ct" and resp["detectors_run"] == [] and resp["findings"] == []
    assert resp["overall_impression"] == srv._NO_VALIDATED_NOTES["ru"]
    assert resp["overall_assessment"]["abnormal_flagged"] is False
    assert resp["normal"] is False
    assert resp["body_part"] == "HEAD"
    assert resp["preview_base64"] and resp["preview_base64"].startswith("data:image/png;base64,")


def test_hidden_pending_head_ct_through_the_api_is_accepted_and_persisted(monkeypatch, tmp_path, mixed_study):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data")        # isolated settings.json
    monkeypatch.delenv("SENTINEL_SHOW_PENDING_DETECTORS", raising=False)
    monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", "1")             # default rule → hidden
    monkeypatch.setattr(srv.state, "license_valid", True)
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in mixed_study["paths"]]
    r = TestClient(srv.app).post("/analyze/study?language=en", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requires_review"] is True
    assert body["rejection_reason"] == "no_validated_detector"
    assert body["overall_impression"] == srv._NO_VALIDATED_NOTES["en"]
    assert body["persisted"] is True
    assert body["show_pending_detectors"] is False
