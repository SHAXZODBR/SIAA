"""Production triage protocol == training/gate protocol (review finding: eval_study_level.py:348).

The validated brain-triage model was trained on native-resolution uint8 PNGs
(scripts/build_training_set.py) that the ViTImageProcessor resizes (bilinear)
to 224. The release gate (scripts/eval_study_level.py::collect_study_probs)
and production (src/inference/brain_analysis.py::_triage_mean_probs) must
feed the model EXACTLY that — through the ONE registry predictor
(src.inference.model_registry._make_hf_predictor):

  production : normalized native-resolution slice (float [0,1], 2-D)
               → predictor 2-D branch → uint8 → RGB PIL → processor (bilinear 224)
  gate / eval: native-resolution PNG → np.asarray(convert('RGB'))
               → predictor 3-D branch → processor (bilinear 224)

The old production path pre-resized with PIL bicubic to 224 (_to_model_input)
and flipped 8 of 178 holdout decisions (sens 0.939 / spec 0.404 instead of the
pinned 0.901 / 0.468).

Model-free tests always run (a spy predictor). The model tests skip when
models/brain_triage_finetuned or data/local_finetune_bin/val is absent; the
full-holdout reproduction is marked slow (~40-90 s) but is NOT skipped by default.
No patient names are read or printed — only numeric ids from file names.
"""
from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO

MODEL_DIR = REPO / "models" / "brain_triage_finetuned"
VAL_DIR = REPO / "data" / "local_finetune_bin" / "val"
HOLDOUT = REPO / "data" / "holdout_patients_v1.txt"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_study_level", REPO / "scripts" / "eval_study_level.py")

needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "config.json").exists() and VAL_DIR.is_dir()),
    reason="release model (models/brain_triage_finetuned) or data/local_finetune_bin/val not present")


class _Spy:
    """Predictor stand-in: records every image it is handed."""
    def __init__(self):
        self.seen: list[np.ndarray] = []

    def __call__(self, image):
        self.seen.append(np.array(image, copy=True))
        return {"abnormal": float(np.mean(image)), "normal": 1.0 - float(np.mean(image))}


def _study(vol: np.ndarray):
    from src.pipeline.brain_mri_preprocessor import BrainStudy
    return BrainStudy(patient_id="9000001", study_uid="2.25.9000001", sequences={"T1": vol},
                      seq_meta={"T1": {"orientation": "axial"}})


def _identity_norm(monkeypatch):
    """Neutralise the (shared) percentile normalisation so the test pins only the
    slice → predictor hand-off; build_training_set and brain_analysis import the
    same robust_normalize_slice, so normalisation parity holds by construction."""
    import src.inference.brain_analysis as ba
    monkeypatch.setattr(ba, "robust_normalize_slice", lambda s: np.asarray(s, dtype=np.float32))
    return ba


# ------------------------------------------------------------------ model-free

def test_production_feeds_native_resolution_float_slices(monkeypatch):
    """_triage_mean_probs must NOT pre-resize (the old _to_model_input bicubic
    224 path): the predictor gets the 2-D float [0,1] slice at native size."""
    ba = _identity_norm(monkeypatch)
    rng = np.random.default_rng(0)
    vol = rng.random((5, 300, 257)).astype(np.float32)
    spy = _Spy()
    out = ba._triage_mean_probs(_study(vol), "T1", spy)
    assert len(spy.seen) == 5
    for i, img in enumerate(spy.seen):
        assert img.ndim == 2 and img.shape == (300, 257), img.shape       # native resolution, not 224×224
        assert img.dtype == np.float32 and 0.0 <= img.min() and img.max() <= 1.0
        np.testing.assert_array_equal(img, vol[i])                        # depth 5 → slices 0..4 in order
    assert out["abnormal"] == pytest.approx(float(np.mean([vol[i].mean() for i in range(5)])), abs=1e-6)


@pytest.fixture(scope="module")
def bts():
    return _load("build_training_set", REPO / "scripts" / "build_training_set.py")


@pytest.mark.parametrize("depth", list(range(1, 41)) + [64, 128, 176])
def test_central_slice_rule_matches_training_set_builder(monkeypatch, bts, depth):
    """Production picks the SAME slice indices build_training_set.central_slices wrote."""
    ba = _identity_norm(monkeypatch)
    monkeypatch.setattr(bts, "robust_normalize_slice", lambda s: np.asarray(s, dtype=np.float32))
    vol = np.stack([np.full((16, 16), i, dtype=np.float32) for i in range(depth)])
    spy = _Spy()
    ba._triage_mean_probs(_study(vol), "T1", spy)
    prod_idxs = [int(img[0, 0]) for img in spy.seen]
    train_idxs = [int(s[0, 0]) for s in bts.central_slices(_study(vol), k=5)]
    assert prod_idxs == train_idxs and 1 <= len(prod_idxs) <= 5


def test_eval_scores_pngs_through_registry_predictor(tmp_path):
    """collect_study_probs hands the predictor the native-resolution RGB PNG
    (3-D uint8) and groups by study key; patients_block hashes unique pids."""
    from PIL import Image
    rng = np.random.default_rng(1)
    for cls, pid in (("abnormal", "9000002"), ("normal", "9000003")):
        (tmp_path / cls).mkdir()
        for k in range(2):
            arr = rng.integers(0, 256, (37, 53), dtype=np.uint8)
            Image.fromarray(arr).convert("RGB").save(tmp_path / cls / f"{pid}_{k}.png")
    spy = _Spy()
    studies, n = ev.collect_study_probs(Path("unused"), tmp_path, predictor=spy)
    assert n == 4 and set(studies) == {("9000002", "abnormal"), ("9000003", "normal")}
    assert all(img.shape == (37, 53, 3) and img.dtype == np.uint8 for img in spy.seen)
    pb = ev.patients_block(studies, None)
    assert pb["n_patients"] == 2 and pb["val_patients_sha256"] == ev.sha256_ids({"9000002", "9000003"})
    # combined-set study keys '<pid>_s2_<batch>' collapse to the patient id
    assert ev.patient_id_of("9000002_s2_batch_one") == "9000002"


# ------------------------------------------------------------------ real model

@pytest.fixture(scope="module")
def cpu_predictor():
    return ev.make_predictor(MODEL_DIR, "cpu")


def _val_pngs() -> list[Path]:
    return sorted(VAL_DIR.glob("*/*.png"))


def _decode_production(fp: Path) -> np.ndarray:
    """What production hands the predictor for this slice: the normalized
    float [0,1] 2-D slice (the PNG is that slice quantised to uint8)."""
    from PIL import Image
    with Image.open(fp) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


@needs_model
def test_production_and_eval_branches_give_identical_probs(cpu_predictor):
    pngs = _val_pngs()
    picks = [pngs[i] for i in np.linspace(0, len(pngs) - 1, 12).astype(int)]
    for fp in picks:
        p_eval = ev.score_png(cpu_predictor, fp)["abnormal"]
        p_prod = cpu_predictor(_decode_production(fp))["abnormal"]
        assert abs(p_eval - p_prod) < 1e-6, (fp.name, p_eval, p_prod)


@needs_model
def test_triage_mean_probs_equals_eval_mean_for_one_patient(monkeypatch, cpu_predictor):
    ba = _identity_norm(monkeypatch)
    by_pid: dict[str, list[Path]] = defaultdict(list)
    for fp in (VAL_DIR / "abnormal").glob("*.png"):
        by_pid[fp.stem.rsplit("_", 1)[0]].append(fp)
    pid = sorted(p for p, fs in by_pid.items() if len(fs) == 5)[0]
    files = sorted(by_pid[pid], key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    vol = np.stack([_decode_production(f) for f in files])        # depth 5 → c=2, step 1 → slices 0..4
    prod = ba._triage_mean_probs(_study(vol), "T1", cpu_predictor)["abnormal"]
    eval_mean = float(np.mean([ev.score_png(cpu_predictor, f)["abnormal"] for f in files]))
    assert abs(prod - eval_mean) < 1e-6, (prod, eval_mean)


def _release_eval() -> tuple[dict, Path]:
    """The release model's own study_level_eval.json and the val dir it was scored on. The
    expected numbers come from THAT file (bump_model_release.py's single source of truth), never
    from literals: after a promote the release model and its val set change together
    (review round 2 — hard-coded v1 counts made the pre-bundle pytest guard fail on every
    new release)."""
    import json
    rel = json.loads((MODEL_DIR / "study_level_eval.json").read_text())
    data = rel.get("data") or "data/local_finetune_bin/val"      # the v1 eval predates the 'data' key
    val_dir = Path(data) if Path(data).is_absolute() else REPO / data
    return rel, val_dir


@needs_model
@pytest.mark.slow
def test_release_model_reproduces_pinned_gate_on_full_val():
    """The release numbers in models/brain_triage_finetuned/study_level_eval.json (v1: TP 118
    FN 13 TN 22 FP 25 → sens 0.901 / spec 0.468 on 178 patients) come out of the shared
    predictor path — i.e. they ARE production numbers."""
    import torch
    rel, val_dir = _release_eval()
    if not val_dir.is_dir():
        pytest.skip(f"the release val set {val_dir} is not present")
    pinned = ((rel.get("manifest") or {}).get("files") or {}).get("model.safetensors")
    if pinned:
        assert pinned == ev.sha256_file(MODEL_DIR / "model.safetensors"), "eval JSON is not for these weights"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    studies, n_slices = ev.collect_study_probs(MODEL_DIR, val_dir, device)
    _, _, gate = ev.evaluate_studies(studies, verbose=False)
    want = rel["gate"]
    assert gate["name"] == want["name"] == "gate mean @0.50"
    assert (gate["tp"], gate["fn"], gate["tn"], gate["fp"]) == \
        (want["tp"], want["fn"], want["tn"], want["fp"]), (gate, want)
    assert len(studies) == rel["n_studies"] and n_slices == rel["n_slices"]
    holdout = HOLDOUT if HOLDOUT.is_file() else None
    pb = ev.patients_block(studies, holdout)
    if rel.get("n_patients") is not None:
        assert pb["n_patients"] == rel["n_patients"]
    if rel.get("val_patients_sha256"):
        assert pb["val_patients_sha256"] == rel["val_patients_sha256"]
    if holdout is not None:
        assert pb["holdout_covered"] is True


# ------------------------------------------------------------ review round 2

def _write_rgb_dicom(path: Path, sop_class: str, rows: int = 64, cols: int = 80, instance: int = 1):
    """A synthetic single-frame colour DICOM (SamplesPerPixel 3)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.BodyPartExamined = "HEAD"
    ds.PatientID = "9000042"
    ds.StudyInstanceUID = "2.25.9000042"
    ds.SeriesInstanceUID = "2.25.9000042.1"
    ds.SeriesDescription = "Processed Images"
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    rgb = np.random.default_rng(instance).integers(0, 256, (rows, cols, 3), dtype=np.uint8)
    ds.PixelData = rgb.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return rgb


def test_single_frame_rgb_is_one_luminance_slice_not_a_volume_of_strips(tmp_path):
    """A single-frame RGB DICOM (rows, cols, 3) used to be taken for a multi-frame volume whose
    'slices' were (cols, 3) strips (review round 2, build_training_set.py:84)."""
    from src.pipeline.brain_mri_preprocessor import BrainSeries, load_volume_from_series
    ct_sop = "1.2.840.10008.5.1.4.1.1.2"            # CT Image Storage (colour is rare, but legal)
    rgbs = [_write_rgb_dicom(tmp_path / f"c{i}.dcm", ct_sop, instance=i + 1) for i in range(3)]
    s = BrainSeries(series_uid="2.25.9000042.1", sequence="unknown", series_description="Processed Images", files=sorted(tmp_path.glob("c*.dcm")))
    vol = load_volume_from_series(s)
    assert vol is not None and vol.shape == (3, 64, 80), vol.shape
    np.testing.assert_allclose(vol[0], rgbs[0].astype(np.float32).mean(axis=-1), atol=1e-4)


def test_colour_secondary_capture_is_never_part_of_a_volume(tmp_path):
    from src.pipeline.brain_mri_preprocessor import BrainSeries, load_volume_from_series
    sc_sop = "1.2.840.10008.5.1.4.1.1.7"             # Secondary Capture Image Storage
    for i in range(3):
        _write_rgb_dicom(tmp_path / f"sc{i}.dcm", sc_sop, rows=512, cols=512, instance=i + 1)
    s = BrainSeries(series_uid="2.25.9000042.1", sequence="unknown", series_description="Processed Images", files=sorted(tmp_path.glob("sc*.dcm")))
    assert load_volume_from_series(s) is None


def _study_meta(seqs: dict):
    from src.pipeline.brain_mri_preprocessor import BrainStudy
    return BrainStudy(patient_id="9000001", study_uid="2.25.9000001",
                      sequences={k: v for k, (v, _o) in seqs.items()},
                      seq_meta={k: {"orientation": o} for k, (_v, o) in seqs.items()})


def test_triage_rule_rejects_strip_volumes_and_off_axial_only_studies(bts):
    """Production and the training-set builder share ONE rule: an axial series with usable
    slices; a (N, 512, 3) strip volume or a sagittal/coronal-only study gets NO triage input."""
    from src.pipeline.brain_mri_preprocessor import triage_sequence
    rng = np.random.default_rng(3)
    strip = _study_meta({"unknown_3": (rng.random((40, 512, 3)).astype(np.float32), "")})
    assert triage_sequence(strip) is None and bts.central_slices(strip) == []
    off_axial = _study_meta({"T1": (rng.random((20, 64, 64)).astype(np.float32), "sagittal"),
                             "T2": (rng.random((20, 64, 64)).astype(np.float32), "coronal")})
    assert triage_sequence(off_axial) is None and bts.central_slices(off_axial) == []
    mixed = _study_meta({"unknown_3": (rng.random((40, 512, 3)).astype(np.float32), ""),
                         "T2": (rng.random((20, 64, 64)).astype(np.float32), "axial")})
    assert triage_sequence(mixed) == "T2" and len(bts.central_slices(mixed)) == 5


def _fake_registry(monkeypatch, ba):
    import src.inference.model_registry as mr
    spy = _Spy()

    def get_model(key, device="cpu"):
        return {"available": True, "predictor": spy} if key == "brain_triage" else {"available": False}
    monkeypatch.setattr(mr, "get_model", get_model)
    return spy


def test_production_does_not_run_triage_on_an_off_axial_only_study(monkeypatch):
    ba = _identity_norm(monkeypatch)
    rng = np.random.default_rng(4)
    study = _study_meta({"T1": (rng.random((20, 64, 64)).astype(np.float32), "sagittal"),
                         "T2": (rng.random((20, 64, 64)).astype(np.float32), "coronal")})
    study.seq_meta["T1"]["body_part"] = study.seq_meta["T2"]["body_part"] = "HEAD"
    monkeypatch.setattr(ba, "process_brain_study", lambda paths: study)
    spy = _fake_registry(monkeypatch, ba)
    res = ba.analyze_brain_study([], include_experimental=False)
    assert "triage" not in res.get("detectors_run", []) and not spy.seen, res
    assert not any(f["detector"] == "triage" for f in res["findings"])
    assert any("triage not run" in n for n in res["notes"])


def test_triage_status_follows_the_eval_verdict(monkeypatch):
    """An eval next to the model that did not pass the gate (e.g. an --allow-unvalidated demo
    bundle) must never be shown as 'validated' (review round 2, build-windows.yml:195)."""
    import dataclasses
    import src.inference.validated_metrics as vm
    ba = _identity_norm(monkeypatch)
    rng = np.random.default_rng(5)
    study = _study_meta({"T1": (rng.random((20, 64, 64)).astype(np.float32), "axial")})
    monkeypatch.setattr(ba, "process_brain_study", lambda paths: study)
    _fake_registry(monkeypatch, ba)
    res = ba.analyze_brain_study([], include_experimental=False)
    assert [f["status"] for f in res["findings"] if f["detector"] == "triage"] == ["validated"]
    monkeypatch.setattr(vm, "METRICS", dataclasses.replace(vm.METRICS, validated=False,
                                                           not_validated_reason="gate.passed is False"))
    res = ba.analyze_brain_study([], include_experimental=False)
    assert [f["status"] for f in res["findings"] if f["detector"] == "triage"] == ["pending"]
    assert any("NOT a validated release" in n for n in res["notes"])
