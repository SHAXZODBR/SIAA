"""Head CT → `head_ct` route: series selection, central-slice aggregation,
decision thresholding and the weights hash in model_identity.

Unit tests build SYNTHETIC pydicom studies in a temp dir (a 1-image scout
with ImageType LOCALIZER, a 30-slice 'Brain Std' axial series, a 30-slice
'Bone' series). The integration test posts the real GE head CT from the
external HDD through POST /analyze/study and is skipped when the disk is not
mounted (SENTINEL_TEST_CT_HEAD_DIR overrides the path).
"""

import base64
import io
import os
import random
import re
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.inference.model_registry import HEAD_CT, REGISTRY, EMPTY_SHA256, ModelCard
from src.pipeline import ct_head_selection as chs

REAL_CT_HEAD_DIR = Path(os.environ.get(
    "SENTINEL_TEST_CT_HEAD_DIR",
    "/Volumes/Transcend/new1/1.2.840.113619.2.25.4.123439150.1785398484.165",
))
EMPTY_SHA12 = EMPTY_SHA256[:12]          # 'e3b0c44298fc'
SCOUT_RE = re.compile(r"scout|localizer|topogram|scano", re.I)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]


# ─── synthetic DICOM writer ────────────────────────────────────────────────

def _write_ct(path: Path, *, study_uid: str, series_uid: str, desc: str, image_type,
              iop, ipp, instance: int, hu: np.ndarray, window=None, series_number=1):
    """One CT slice; `hu` is the image in Hounsfield units (int16), stored with
    RescaleIntercept -1024 like a real scanner. window=(center, width) or None."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.BodyPartExamined = "HEAD"
    ds.PatientID = "SYNTH-CT-001"
    ds.PatientName = "Synthetic^Head"
    ds.StudyInstanceUID = study_uid
    ds.StudyDescription = "HEAD CT synthetic"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = desc
    if image_type is not None:
        ds.ImageType = list(image_type)
    if iop is not None:
        ds.ImageOrientationPatient = [float(v) for v in iop]
    if ipp is not None:
        ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    raw = (hu.astype(np.int32) + 1024).astype(np.int16)
    ds.PixelData = raw.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def _brain_like(rows: int, cols: int, spot_hu: int = 80) -> np.ndarray:
    """Air background (-1000 HU), a disc of brain (40 HU), one bright spot."""
    yy, xx = np.mgrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (min(rows, cols) * 0.4) ** 2
    hu = np.full((rows, cols), -1000, dtype=np.int16)
    hu[disc] = 40
    hu[int(cy), int(cx)] = spot_hu
    return hu


@pytest.fixture(scope="module")
def synthetic_study(tmp_path_factory) -> dict:
    """Scout (1 image, LOCALIZER, the LARGEST image), 'Brain Std' (30 axial
    slices, ascending z) and 'Bone' (30 axial slices). Files are written in a
    shuffled upload order so nothing may rely on it."""
    d = tmp_path_factory.mktemp("synthetic_head_ct")
    study_uid = generate_uid()
    scout_uid, brain_uid, bone_uid = generate_uid(), generate_uid(), generate_uid()
    specs = []
    specs.append(dict(series_uid=scout_uid, desc="Scout", image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"],
                      iop=CORONAL, ipp=[-265, 0, 150], instance=1,
                      hu=_brain_like(96, 128), window=(50, 500), series_number=1))
    for i in range(30):
        specs.append(dict(series_uid=brain_uid, desc="Brain Std", image_type=["ORIGINAL", "PRIMARY", "AXIAL"],
                          iop=AXIAL, ipp=[-160, -150, -96 + 2.5 * i], instance=i + 1,
                          hu=_brain_like(64, 64), window=None, series_number=2))
    for i in range(30):
        specs.append(dict(series_uid=bone_uid, desc="Bone", image_type=["ORIGINAL", "PRIMARY", "AXIAL"],
                          iop=AXIAL, ipp=[-160, -150, -96 + 2.5 * i], instance=i + 1,
                          hu=_brain_like(64, 64), window=(350, 3500), series_number=3))
    random.Random(7).shuffle(specs)
    paths = []
    for n, spec in enumerate(specs):
        p = d / f"{n:04d}.dcm"
        _write_ct(p, study_uid=study_uid, **spec)
        paths.append(p)
    return {"dir": d, "paths": paths, "scout_uid": scout_uid, "brain_uid": brain_uid, "bone_uid": bone_uid}


# ─── (a) series selection on synthetic data ────────────────────────────────

def test_headers_grouped_by_series(synthetic_study):
    series = chs.read_series_headers(synthetic_study["paths"])
    by_uid = {s.uid: s for s in series}
    assert set(by_uid) == {synthetic_study["scout_uid"], synthetic_study["brain_uid"], synthetic_study["bone_uid"]}
    assert by_uid[synthetic_study["scout_uid"]].n_images == 1
    assert by_uid[synthetic_study["brain_uid"]].n_images == 30
    assert by_uid[synthetic_study["brain_uid"]].is_axial is True
    assert by_uid[synthetic_study["scout_uid"]].is_axial is False


def test_scout_dropped_and_brain_series_chosen(synthetic_study):
    sel = chs.select_ct_head_slices(synthetic_study["paths"])
    assert sel is not None
    assert sel.series.uid == synthetic_study["brain_uid"]
    assert sel.series.description == "Brain Std"
    assert sel.n_series == 3
    dropped = {d["description"]: d["dropped_reason"] for d in sel.dropped}
    assert "Scout" in dropped and "LOCALIZER" in dropped["Scout"]
    assert "Bone" not in dropped                       # a candidate, just not preferred
    assert all(not SCOUT_RE.search(p.name) for p in sel.slice_paths)
    # the scout is the LARGEST image — the old "largest Rows×Columns" rule would pick it
    scout = next(s for s in chs.read_series_headers(synthetic_study["paths"]) if s.uid == synthetic_study["scout_uid"])
    assert scout.rows_cols == (96, 128) and sel.series.rows_cols == (64, 64)


def test_nine_central_slices_in_middle_60_percent(synthetic_study):
    sel = chs.select_ct_head_slices(synthetic_study["paths"])
    n = sel.series.n_images
    assert n == 30
    assert len(sel.indices) == 9
    assert sel.indices == sorted(sel.indices)
    lo, hi = int(0.2 * n), int(0.8 * n)               # 6 .. 23
    assert all(lo <= i < hi for i in sel.indices), sel.indices
    assert sel.central_index == sel.indices[4]
    assert sel.central_path == sel.slice_paths[4]
    # helper contract on its own
    assert chs.central_indices(30) == sel.indices
    assert chs.central_indices(5) == [1, 2, 3]         # short band → the whole band
    assert chs.central_indices(201)[4] == 100          # true middle of a 201-slice stack
    assert chs.central_indices(0) == []


def test_slices_sorted_by_position_not_upload_order(synthetic_study):
    sel = chs.select_ct_head_slices(synthetic_study["paths"])
    zs = [s.z for s in sel.series.slices]
    assert zs == sorted(zs)
    orders = [s.order for s in sel.series.slices]
    assert orders != sorted(orders)                    # the upload really was shuffled
    chosen_z = [sel.series.slices[i].z for i in sel.indices]
    assert chosen_z == sorted(chosen_z)
    inst = [pydicom.dcmread(str(p), stop_before_pixels=True).InstanceNumber for p in sel.slice_paths]
    assert inst == sorted(inst) and inst[0] >= 7 and inst[-1] <= 24


def test_brain_window_applied_when_tags_missing(synthetic_study):
    sel = chs.select_ct_head_slices(synthetic_study["paths"])
    img, meta = chs.load_windowed_slice(sel.central_path)
    assert meta["default_brain_window"] is True
    assert (meta["window_center"], meta["window_width"]) == (40.0, 80.0)
    assert img.dtype == np.float32 and img.shape == (64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img[0, 0] == pytest.approx(0.0)             # air (-1000 HU) → black
    assert img[32, 40] == pytest.approx(0.5, abs=1e-3)  # brain (40 HU) → mid-gray
    assert img[32, 32] == pytest.approx(1.0)           # 80 HU spot → top of the window
    # letter-boxed to the classifier input
    small, _ = chs.load_windowed_slice(sel.central_path, target_size=224)
    assert small.shape == (224, 224) and 0.0 <= small.min() and small.max() <= 1.0


def test_file_window_tags_are_used_when_present(synthetic_study):
    bone = next(s for s in chs.read_series_headers(synthetic_study["paths"]) if s.description == "Bone")
    img, meta = chs.load_windowed_slice(bone.slices[0].path)
    assert meta["default_brain_window"] is False
    assert (meta["window_center"], meta["window_width"]) == (350.0, 3500.0)
    assert img[32, 40] == pytest.approx((40 - (350 - 1750)) / 3500, abs=1e-3)


def test_window_hu_helper():
    hu = np.array([[-1000, 0, 40, 80, 500]], dtype=np.float32)
    raw = hu + 1024
    out, meta = chs.window_hu(raw, rescale_slope=1, rescale_intercept=-1024)
    assert meta["default_brain_window"] is True
    assert out.tolist() == [[0.0, 0.0, 0.5, 1.0, 1.0]]
    out2, meta2 = chs.window_hu(raw, rescale_slope=1, rescale_intercept=-1024, window_center=0, window_width=0)
    assert meta2["default_brain_window"] is True       # zero width is invalid → brain window
    assert out2.tolist() == out.tolist()


def test_localizer_by_description_and_tiny_series_dropped(tmp_path):
    study_uid = generate_uid()
    paths = []
    topo_uid, small_uid, big_uid = generate_uid(), generate_uid(), generate_uid()
    for i in range(6):                                 # a topogram WITHOUT ImageType LOCALIZER
        p = tmp_path / f"t{i}.dcm"
        _write_ct(p, study_uid=study_uid, series_uid=topo_uid, desc="Topogram", image_type=["ORIGINAL", "PRIMARY"],
                  iop=CORONAL, ipp=[0, 0, i], instance=i + 1, hu=_brain_like(96, 96))
        paths.append(p)
    for i in range(4):                                 # < 5 images
        p = tmp_path / f"s{i}.dcm"
        _write_ct(p, study_uid=study_uid, series_uid=small_uid, desc="Brain 5mm", image_type=["ORIGINAL", "PRIMARY", "AXIAL"],
                  iop=AXIAL, ipp=[0, 0, i], instance=i + 1, hu=_brain_like(64, 64))
        paths.append(p)
    for i in range(8):                                 # no brain keyword, but axial and big enough
        p = tmp_path / f"b{i}.dcm"
        _write_ct(p, study_uid=study_uid, series_uid=big_uid, desc="S2", image_type=["ORIGINAL", "PRIMARY", "AXIAL"],
                  iop=AXIAL, ipp=[0, 0, i], instance=i + 1, hu=_brain_like(64, 64))
        paths.append(p)
    sel = chs.select_ct_head_slices(paths)
    assert sel.series.uid == big_uid
    reasons = {d["description"]: d["dropped_reason"] for d in sel.dropped}
    assert "localizer description" in reasons["Topogram"]
    assert "fewer than 5" in reasons["Brain 5mm"]
    # central band of an 8-slice stack: floor(1.6)=1 .. ceil(6.4)=7 → indices 1..6
    assert sel.indices == [1, 2, 3, 4, 5, 6]
    assert chs.central_indices(8) == [1, 2, 3, 4, 5, 6]


def test_series_preference_order():
    def mk(desc, iop, n, image_type=("ORIGINAL", "PRIMARY")):
        s = chs.SeriesInfo(uid=desc, description=desc, image_type=list(image_type), orientation=iop)
        s.slices = [chs.SliceHeader(path=Path(f"{desc}_{i}"), order=i) for i in range(n)]
        return s
    brain_small = mk("Brain Std", AXIAL, 20)
    bone_big = mk("0.625 bone", AXIAL, 400)
    axial_5mm = mk("5mm", AXIAL, 50)
    coronal_brain = mk("Brain cor MPR", CORONAL, 300, ("DERIVED", "SECONDARY", "REFORMATTED"))
    assert chs.choose_brain_series([bone_big, axial_5mm, coronal_brain, brain_small]) is brain_small
    assert chs.choose_brain_series([bone_big, axial_5mm]) is axial_5mm        # soft kernel over bone
    assert chs.choose_brain_series([bone_big, coronal_brain]) is bone_big     # axial over reformat
    assert chs.choose_brain_series([coronal_brain, mk("misc", None, 10)]) is coronal_brain
    assert chs.choose_brain_series([]) is None


def test_no_series_survives_returns_none_and_fallback_skips_localizer(tmp_path):
    study_uid = generate_uid()
    scout = tmp_path / "scout.dcm"
    _write_ct(scout, study_uid=study_uid, series_uid=generate_uid(), desc="Scout",
              image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"], iop=CORONAL, ipp=[0, 0, 0], instance=1, hu=_brain_like(96, 128))
    single = tmp_path / "axial.dcm"
    _write_ct(single, study_uid=study_uid, series_uid=generate_uid(), desc="5mm",
              image_type=["ORIGINAL", "PRIMARY", "AXIAL"], iop=AXIAL, ipp=[0, 0, 0], instance=1, hu=_brain_like(64, 64))
    assert chs.select_ct_head_slices([scout, single]) is None
    assert chs.largest_non_localizer([scout, single]) == single     # the scout is larger but never chosen
    assert chs.largest_non_localizer([scout]) is None


def test_aggregate_mean_and_max():
    mean, mx = chs.aggregate_slice_probs([{"a": 0.1, "b": 0.9}, {"a": 0.3, "b": 0.5}, {"a": 0.2}])
    assert mean["a"] == pytest.approx(0.2) and mx["a"] == pytest.approx(0.3)
    assert mean["b"] == pytest.approx(0.7) and mx["b"] == pytest.approx(0.9)
    assert chs.aggregate_slice_probs([]) == ({}, {})


# ─── (b) decision thresholding ─────────────────────────────────────────────

RSNA = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def _fake_card(multi_label: bool, threshold: float = 0.5, classes=RSNA, study_flag="any") -> ModelCard:
    return ModelCard(modality="fake_ct", display_name="fake", classes=list(classes),
                     multi_label=multi_label, decision_threshold=threshold, study_flag_class=study_flag)


def test_multi_label_subthreshold_classes_are_not_positive():
    from src.inference.server import _decide_findings, _study_flag, _summarize_findings
    card = _fake_card(True)
    probs = {"any": 0.3, "subdural": 0.29, "subarachnoid": 0.23, "intraparenchymal": 0.21,
             "epidural": 0.05, "intraventricular": 0.02}
    findings = _decide_findings(card, "fake_ct", probs, "en")
    assert len(findings) == 6                          # all reported …
    assert all(f["positive"] is False for f in findings)   # … none flagged
    assert [f["class_name"] for f in findings][:2] == ["any", "subdural"]
    assert _study_flag(card, probs, findings) == (False, "any")
    impression, overall = _summarize_findings(card, probs, findings)
    assert overall["abnormal_flagged"] is False and overall["flags"] == []
    assert "NOT a normal read" in impression           # never certifies normal


def test_multi_label_above_threshold_is_positive_and_any_is_study_flag():
    from src.inference.server import _decide_findings, _summarize_findings
    card = _fake_card(True)
    probs = {"any": 0.8, "subdural": 0.7, "epidural": 0.1, "intraparenchymal": 0.4}
    findings = _decide_findings(card, "fake_ct", probs, "en")
    pos = {f["class_name"] for f in findings if f["positive"]}
    assert pos == {"any", "subdural"}
    assert next(f for f in findings if f["class_name"] == "intraparenchymal")["positive"] is False
    _, overall = _summarize_findings(card, probs, findings)
    assert overall["abnormal_flagged"] is True
    assert overall["study_flag"] == {"class": "any", "confidence": 0.8}
    # 'any' alone decides the study flag, even when no subtype crosses the line
    probs2 = {"any": 0.55, "subdural": 0.3}
    f2 = _decide_findings(card, "fake_ct", probs2, "en")
    _, o2 = _summarize_findings(card, probs2, f2)
    assert o2["abnormal_flagged"] is True and o2["flags"] == ["any"]


def test_softmax_rule_top1_must_reach_decision_threshold():
    from src.inference.server import _decide_findings, _summarize_findings
    card = _fake_card(False, classes=["NORMAL", "TUBERCULOSIS"], study_flag=None)
    f = _decide_findings(card, "fake_ct", {"NORMAL": 0.6, "TUBERCULOSIS": 0.4}, "en")
    assert {x["class_name"]: x["positive"] for x in f} == {"NORMAL": False, "TUBERCULOSIS": False}
    f = _decide_findings(card, "fake_ct", {"NORMAL": 0.2, "TUBERCULOSIS": 0.8}, "en")
    assert {x["class_name"]: x["positive"] for x in f} == {"TUBERCULOSIS": True, "NORMAL": False}
    _, overall = _summarize_findings(card, {"NORMAL": 0.2, "TUBERCULOSIS": 0.8}, f)
    assert overall["abnormal_flagged"] is True and "study_flag" not in overall
    # softmax reporting floor: classes under 0.15 are not listed (pre-existing top-3 rule)
    f = _decide_findings(card, "fake_ct", {"NORMAL": 0.9, "TUBERCULOSIS": 0.1}, "en")
    assert [x["class_name"] for x in f] == ["NORMAL"]
    # a 'normal'-style class is never positive, however confident
    card6 = _fake_card(False, classes=RSNA[1:] + ["normal"], study_flag="any")
    f = _decide_findings(card6, "fake_ct", {"normal": 0.95, "subdural": 0.05}, "en", report_all=True)
    assert all(x["positive"] is False for x in f) and len(f) == 2


def test_registry_cards_declare_their_decision_rule():
    assert HEAD_CT.decision_threshold == 0.5 and HEAD_CT.study_flag_class == "any"
    assert "normal" in HEAD_CT.classes_localized
    for key in ("chest_tb", "chest_pneumonia", "covid_ct"):
        assert REGISTRY[key].multi_label is False and REGISTRY[key].decision_threshold == 0.5
    assert REGISTRY["chest"].multi_label is True          # TorchXRayVision sigmoids
    # the 2D tumor classifier keeps its pre-existing top-3 ≥ 0.15 behaviour
    assert REGISTRY["brain_tumor_class"].decision_threshold == 0.15
    from src.inference.server import _decide_findings
    f = _decide_findings(REGISTRY["brain_tumor_class"], "brain_2d",
                         {"glioma_tumor": 0.45, "meningioma_tumor": 0.3, "no_tumor": 0.2, "pituitary_tumor": 0.05}, "en")
    assert [(x["class_name"], x["positive"]) for x in f] == [("glioma_tumor", True), ("meningioma_tumor", True), ("no_tumor", False)]


def test_checkpoint_problem_type_overrides_card_flag():
    from types import SimpleNamespace
    from src.inference.model_registry import effective_multi_label
    single = SimpleNamespace(config=SimpleNamespace(problem_type="single_label_classification"))
    multi = SimpleNamespace(config=SimpleNamespace(problem_type="multi_label_classification"))
    silent = SimpleNamespace(config=SimpleNamespace(problem_type=None))
    assert effective_multi_label(single, True) is False
    assert effective_multi_label(multi, False) is True
    assert effective_multi_label(silent, True) is True and effective_multi_label(silent, False) is False


# ─── server wiring with a fake predictor (no weights needed) ───────────────

def test_run_ct_head_series_with_fake_predictor(monkeypatch, synthetic_study):
    from src.inference import model_registry as mr
    from src.inference import server as srv

    seen = []

    def fake_predict(image):
        seen.append(image)
        i = len(seen)
        return {"any": 0.20 + 0.02 * i, "subdural": 0.10 + 0.01 * i, "epidural": 0.05,
                "intraparenchymal": 0.05, "intraventricular": 0.02, "subarachnoid": 0.03}

    card = _fake_card(True)
    entry = {"card": card, "predictor": fake_predict, "available": True, "reason": "loaded from fake",
             "source_path": None, "multi_label": True, "activation": "sigmoid"}
    monkeypatch.setattr(mr, "_loaded_models", {"head_ct": entry})
    monkeypatch.setattr(mr, "_identity_cache", {})
    monkeypatch.setitem(mr.REGISTRY, "head_ct", card)

    run = srv._run_ct_head_series(synthetic_study["paths"], "head_ct", "en", {"modality": "CT", "body_part": "HEAD"})
    assert run is not None
    assert run["series_used"] == "Brain Std" and run["n_slices_analyzed"] == 9
    assert run["aggregation"] == "mean_central_9"
    assert len(seen) == 9 and all(im.shape == (224, 224) and 0.0 <= im.min() and im.max() <= 1.0 for im in seen)
    assert run["per_class_max"]["any"] == pytest.approx(0.20 + 0.02 * 9, abs=1e-4)
    any_mean = np.mean([0.20 + 0.02 * i for i in range(1, 10)])
    f_any = next(f for f in run["findings"] if f["class_name"] == "any")
    assert f_any["confidence"] == pytest.approx(any_mean, abs=1e-3)
    assert all(f["positive"] is False for f in run["findings"])
    assert all(f["sequence_used"] == "Brain Std" and f["location"] == "" for f in run["findings"])
    assert run["overall_assessment"]["abnormal_flagged"] is False
    assert run["overall_assessment"]["protocol"]["window"]["default_brain_window"] is True
    assert run["slice_rows_cols"] == [64, 64]
    assert [d["description"] for d in run["series_dropped"]] == ["Scout"]
    png = base64.b64decode(run["preview_base64"].split(",", 1)[1])
    assert png.startswith(PNG_MAGIC)
    # a fake with no weight file: sha is null and the identity SAYS so — never the empty-string hash
    ident = run["model_identity"][0]
    assert ident["sha256_12"] is None and ident["sha256_12"] != EMPTY_SHA12
    assert "not hashable" in ident["validation_note"]
    assert "[unhashed]" in ident["source"]


def test_weights_hash_never_empty_digest(tmp_path):
    from src.inference.model_registry import _sha256_of, _nonempty_weight_files, weights_sha256_for_source
    empty = tmp_path / "model.safetensors"
    empty.write_bytes(b"")
    assert _sha256_of([empty]) is None
    assert _nonempty_weight_files(tmp_path) == []
    assert weights_sha256_for_source(str(tmp_path)) is None
    real = tmp_path / "pytorch_model.bin"
    real.write_bytes(b"weights" * 100)
    assert _nonempty_weight_files(tmp_path) == [real.resolve()]
    full = weights_sha256_for_source(str(tmp_path))
    assert full and len(full) == 64 and full != EMPTY_SHA256


# ─── (c) integration: the real GE head CT through POST /analyze/study ──────

def _real_ct_paths() -> list[Path]:
    if not REAL_CT_HEAD_DIR.is_dir():
        return []
    return sorted(REAL_CT_HEAD_DIR.rglob("*.dcm"))


@pytest.fixture(scope="module")
def real_ct_result(client) -> dict:
    paths = _real_ct_paths()
    if not paths:
        pytest.skip(f"real head CT not available at {REAL_CT_HEAD_DIR} (set SENTINEL_TEST_CT_HEAD_DIR)")
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in paths]
    r = client.post("/analyze/study", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    print("\n[real head CT] route=%s series_used=%r n_slices=%s aggregation=%s activation=%s" % (
        body.get("route"), body.get("series_used"), body.get("n_slices_analyzed"),
        body.get("aggregation"), body.get("activation")))
    for f in body["findings"]:
        print("  finding %-18s conf=%.4f positive=%s status=%s" % (
            f["class_name"], f["confidence"], f["positive"], f["status"]))
    print("  per_class_max:", body.get("per_class_max"))
    print("  overall:", body["overall_assessment"].get("text"))
    print("  identity:", body["model_identity"])
    return body


def test_real_ct_routes_to_head_ct_on_an_axial_series(real_ct_result):
    body = real_ct_result
    assert body["route"] == "head_ct" and body["detectors_run"] == ["head_ct"]
    assert body["modality"].upper() == "CT"
    assert body["series_used"] and not SCOUT_RE.search(body["series_used"])
    assert 3 <= body["n_slices_analyzed"] <= 9
    assert body["aggregation"] == "mean_central_9"
    assert body["rejected"] is False and body["requires_review"] is False
    assert body["persisted"] is True
    assert any(SCOUT_RE.search(d["description"] or "") for d in body["series_dropped"])
    assert body["slice_rows_cols"] == [512, 512]        # the axial series, not the 734×835 scout


def test_real_ct_findings_are_thresholded(real_ct_result):
    body = real_ct_result
    assert body["findings"], "no findings reported"
    assert body["threshold"] == 0.5
    for f in body["findings"]:
        assert f["detector"] == "head_ct" and f["status"] == "pending"
        assert 0.0 <= f["confidence"] <= 1.0
        if f["confidence"] < 0.5:
            assert f["positive"] is False, f
        if f["class_name"].lower() == "normal":
            assert f["positive"] is False
    assert body["normal"] is False                      # never certifies normal
    per_max = body["per_class_max"]
    for f in body["findings"]:
        assert per_max[f["class_name"]] >= f["confidence"] - 1e-4   # max ≥ mean
    assert len(body["per_slice"]) == body["n_slices_analyzed"]
    flagged = body["overall_assessment"]["abnormal_flagged"]
    assert flagged == any(f["positive"] for f in body["findings"])


def test_real_ct_model_identity_has_real_hash(real_ct_result):
    ids = real_ct_result["model_identity"]
    assert ids and ids[0]["key"] == "head_ct"
    sha = ids[0]["sha256_12"]
    assert isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{12}", sha), sha
    assert sha != EMPTY_SHA12
    assert ids[0]["source_path"] and Path(ids[0]["source_path"]).is_dir()
    assert "[unhashed]" not in (ids[0]["source"] or "")


def test_real_ct_preview_is_an_axial_brain_slice(real_ct_result):
    from PIL import Image
    pb = real_ct_result["preview_base64"]
    assert pb and pb.startswith("data:image/png;base64,")
    png = base64.b64decode(pb.split(",", 1)[1])
    assert png.startswith(PNG_MAGIC)
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32)
    h, w = arr.shape
    assert h == w == 512
    centre = arr[h // 2 - h // 8: h // 2 + h // 8, w // 2 - w // 8: w // 2 + w // 8].mean()
    k = h // 8
    corners = np.mean([arr[:k, :k].mean(), arr[:k, -k:].mean(), arr[-k:, :k].mean(), arr[-k:, -k:].mean()])
    # brain-windowed axial slice: brain parenchyma mid-gray in the middle, air black at the corners
    assert centre > corners + 40, (centre, corners)
    assert corners < 20, corners
