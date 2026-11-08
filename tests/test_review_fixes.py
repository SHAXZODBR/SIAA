"""Regression tests for the confirmed review findings on the v1.1.0 finish-line work.

* An empty brain panel (validated triage not installed / no axial series / failed) is a
  NOT-analysed study (no_validated_detector, or no_detector_ran when pending detectors are
  shown) — never "nothing flagged" with a draft report.
* GET /study/{id} restores the not-analysed reason code (the desktop shows the localized note).
* Report triage sentences honour the finding's status and the CURRENT release metrics.
* Per-series CT routing keeps untagged series when every tagged series agrees on HEAD.
* A bundle imported before its licence says license_model_key_missing, not "wrong machine".
* A licence renewal / expiry never orphans ciphertext encrypted with an earlier licence key.

Synthetic data only (ids >= 9000000, dates 2031); no real model is loaded.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from fastapi.testclient import TestClient

from src.inference import server as srv

# fixtures + helpers of the existing suites (pytest collects fixtures imported into a module)
from test_ct_series_routing import HEAD_HU, _fake_head_ct, build_series, mixed_study  # noqa: F401
from test_model_import import (  # noqa: F401
    KEY_B, _assert_untouched, _stage, api, bearer, env, make_encrypted_bundle,
)


# ─── 1. empty brain panel → not analysed ───────────────────────────────────

class _Predictor:
    def __call__(self, image):
        return {"abnormal": 0.9, "normal": 0.1, "glioma": 0.97, "meningioma": 0.01,
                "pituitary": 0.01, "no_tumor": 0.01}


def _study(orientation: str):
    from src.pipeline.brain_mri_preprocessor import BrainStudy
    vol = np.random.default_rng(9000002).random((20, 64, 64)).astype(np.float32)
    return BrainStudy(patient_id="9000002", study_uid="2.25.9000002",
                      sequences={"T1": vol}, seq_meta={"T1": {"orientation": orientation}})


def _brain_hdr() -> dict:
    return {"modality": "MR", "body_part": "BRAIN", "patient_id": "9000002", "study_date": "20310115",
            "series_descriptions": ["T1"], "protocol_names": [], "best_path": None}


class _EngineMustNotRun:
    def generate(self, *a, **k):
        raise AssertionError("no draft report may be generated for a not-analysed study")


@pytest.fixture
def panel(monkeypatch):
    """The REAL brain_analysis panel with fake models; `unavailable` keys report not loaded."""
    import src.inference.brain_analysis as ba
    import src.inference.model_registry as mr

    cfg = {"unavailable": set(), "orientation": "axial"}

    def get_model(key, device="cpu"):
        if key in cfg["unavailable"]:
            return {"available": False, "reason": f"{key} not installed (synthetic)"}
        return {"available": True, "predictor": _Predictor()}

    monkeypatch.setattr(mr, "get_model", get_model)
    monkeypatch.setattr(ba, "process_brain_study", lambda paths: _study(cfg["orientation"]))
    monkeypatch.setattr(ba, "robust_normalize_slice", lambda s: np.asarray(s, dtype=np.float32))
    monkeypatch.setattr(srv.state, "gemma_engine", _EngineMustNotRun())
    return cfg


def _assert_not_analysed(resp: dict, reason: str, lang: str) -> None:
    assert resp["requires_review"] is True and resp["rejected"] is False
    assert resp["rejection_reason"] == reason
    assert resp["findings"] == [] and resp["detectors_run"] == []
    assert resp["report_text"] is None and resp["route"] == "brain_panel"
    assert resp["overall_assessment"]["abnormal_flagged"] is False
    assert resp["overall_assessment"]["rejection_reason"] == reason      # persisted with the row
    assert "No finding flagged" not in resp["overall_impression"]
    notes = srv._NO_VALIDATED_NOTES if reason == "no_validated_detector" else srv._NO_DETECTOR_RAN_NOTES
    assert resp["overall_impression"] == notes[lang]


def test_triage_model_missing_with_pending_hidden_is_no_validated_detector(panel, monkeypatch):
    """Clinic flavour before the bundle import (brain_triage not installed), pending hidden."""
    panel["unavailable"] = {"brain_triage"}
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    resp = srv._analyze_study_sync([], _brain_hdr(), "ru", "test-review-brain-1")
    _assert_not_analysed(resp, "no_validated_detector", "ru")
    assert resp["detectors_hidden"] == ["tumor_class"]
    assert any("model unavailable" in n and "brain_triage not installed" in n for n in resp["panel_notes"])


def test_sagittal_only_brain_mr_with_pending_hidden_is_no_validated_detector(panel, monkeypatch):
    """A full install, but no axial series: the validated triage is outside its protocol."""
    panel["orientation"] = "sagittal"
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    resp = srv._analyze_study_sync([], _brain_hdr(), "en", "test-review-brain-2")
    _assert_not_analysed(resp, "no_validated_detector", "en")
    assert any("no axial series" in n for n in resp["panel_notes"])


def test_nothing_could_run_with_pending_shown_is_no_detector_ran(panel, monkeypatch):
    panel["unavailable"] = {"brain_triage", "brain_tumor_class"}
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: True)
    resp = srv._analyze_study_sync([], _brain_hdr(), "uz", "test-review-brain-3")
    _assert_not_analysed(resp, "no_detector_ran", "uz")
    assert "detectors_hidden" not in resp


def test_triage_running_is_still_an_analysed_study(panel, monkeypatch):
    """Control: with the triage available the study is analysed as before (report generated)."""
    class _Engine:
        def generate(self, findings, language="ru", **kw):
            return "REPORT"
    monkeypatch.setattr(srv.state, "gemma_engine", _Engine())
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    resp = srv._analyze_study_sync([], _brain_hdr(), "en", "test-review-brain-4")
    assert resp["detectors_run"] == ["triage"] and resp["requires_review"] is False
    assert resp["rejection_reason"] is None and resp["report_text"] == "REPORT"


def test_not_analysed_notes_exist_for_every_language():
    assert set(srv._NO_DETECTOR_RAN_NOTES) == {"en", "ru", "uz"}
    assert srv._no_detector_ran_note("xx") == srv._NO_DETECTOR_RAN_NOTES["en"]


# ─── 14. GET /study restores the reason code ───────────────────────────────

def test_restored_no_validated_detector_study_keeps_its_reason(monkeypatch, tmp_path, mixed_study):  # noqa: F811
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("SENTINEL_SHOW_PENDING_DETECTORS", raising=False)
    monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", "1")             # default rule → pending hidden
    monkeypatch.setattr(srv.state, "license_valid", True)
    client = TestClient(srv.app)
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in mixed_study["paths"]]
    r = client.post("/analyze/study?language=en", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejection_reason"] == "no_validated_detector" and body["persisted"] is True
    g = client.get(f"/study/{body['study_id']}")
    assert g.status_code == 200, g.text
    ai = g.json()["ai_result"]
    assert ai["requires_review"] is True and ai["rejected"] is False
    assert ai["rejection_reason"] == "no_validated_detector"


@pytest.mark.parametrize("rejected,review,overall,impression,expected", [
    (True, True, None, "not a brain study", "not a brain study"),               # unchanged
    (False, False, {"text": "x"}, "x", None),                                   # analysed
    (False, True, {"rejection_reason": "no_detector_ran"}, "y", "no_detector_ran"),
    (False, True, {"rejection_reason": "made_up"}, "y", None),                   # unknown codes dropped
    (False, True, {"text": "legacy"}, srv._NO_VALIDATED_NOTES["ru"], "no_validated_detector"),  # old rows
])
def test_persisted_rejection_reason(rejected, review, overall, impression, expected):
    assert srv._persisted_rejection_reason(rejected, review, overall, impression) == expected


# ─── 2. report triage wording follows status + current metrics ─────────────

@pytest.fixture
def metrics(monkeypatch):
    import src.inference.validated_metrics as vm
    for name in ("METRICS", "MODEL_DIR", "SOURCE", "SENS", "SPEC", "ACC", "TP", "FN", "TN", "FP",
                 "N_STUDIES", "N_ABNORMAL", "N_NORMAL", "N_SLICES", "EVALUATED_ON", "IDENTITY",
                 "MIN_SENS", "MIN_SPEC", "THRESHOLD", "SENS_CI", "SPEC_CI"):
        monkeypatch.setattr(vm, name, getattr(vm, name))
    yield vm


def _triage(cls: str, status: str) -> list[dict]:
    return [{"class_name": cls, "confidence": 0.8, "positive": cls == "abnormal",
             "status": status, "detector": "triage"}]


@pytest.mark.parametrize("lang,validated_word", [("en", "validated on this clinic's data"),
                                                 ("ru", "валидирована на данных клиники"),
                                                 ("uz", "validatsiya qilingan")])
def test_pending_triage_finding_is_never_described_as_validated(metrics, lang, validated_word):
    from src.inference.brain_report_templates import build_brain_report
    for cls in ("abnormal", "normal"):
        desc = build_brain_report(_triage(cls, "pending"), language=lang)["description"]
        assert validated_word not in desc, (lang, cls, desc)
        assert metrics.not_validated_note(lang) in desc
        assert metrics.pct(metrics.METRICS.sens) not in desc          # no borrowed figures
    # a finding without status is not claimed validated either
    desc = build_brain_report([{**_triage("abnormal", "x")[0], "status": None}], language=lang)["description"]
    assert validated_word not in desc


def test_report_figures_follow_a_metrics_reload_without_restart(metrics):
    from src.inference import brain_report_templates as brt
    metrics._publish(dataclasses.replace(metrics.METRICS, sens=0.95, spec=0.60, validated=True))
    desc = brt.build_brain_report(_triage("abnormal", "validated"), language="en")["description"]
    assert "sensitivity 95%, specificity 60%" in desc
    assert brt.TRIAGE_DESCRIPTIONS["abnormal"]["en"] in desc          # the shared dict is refreshed in place
    metrics._publish(dataclasses.replace(metrics.METRICS, validated=False,
                                         not_validated_reason="gate.passed is False"))
    desc = brt.build_brain_report(_triage("abnormal", "validated"), language="en")["description"]
    assert "validated on this clinic's data" not in desc and "NOT a validated release" in desc
    assert "NOT a validated release" in metrics.registry_notes()


# ─── 3. CT routing adopts untagged series of an unambiguous head study ─────

def test_head_tagged_scout_plus_untagged_axial_routes_the_axial_series(monkeypatch, tmp_path):
    from pydicom.uid import generate_uid
    study_uid = generate_uid()
    scout = build_series(tmp_path, desc="Topogram 0.6 T20s", n=1, hu_value=HEAD_HU, body_part="HEAD",
                         study_uid=study_uid, series_number=1, start_index=0)
    axial = build_series(tmp_path, desc="5mm Std", n=12, hu_value=HEAD_HU, study_uid=study_uid,
                         series_number=2, start_index=10)
    paths = scout + axial
    hdr = srv._read_study_headers(paths)
    plan = srv._plan_series_route(hdr, paths)
    assert plan["modality_key"] == "head_ct"
    assert set(plan["paths"]) == set(paths)
    assert plan["routing"]["series_excluded"] == []
    assert [s["description"] for s in plan["routing"]["series_adopted_without_body_part"]] == ["5mm Std"]

    seen: list = []
    _fake_head_ct(monkeypatch, seen)
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: True)
    resp = srv._analyze_study_sync(paths, hdr, "en", "test-review-ct-1")
    assert resp["route"] == "head_ct" and resp["series_used"] == "5mm Std"
    assert resp["n_slices_analyzed"] > 1 and seen


def test_untagged_series_stay_excluded_in_a_mixed_study(tmp_path):
    from pydicom.uid import generate_uid
    from test_ct_series_routing import CHEST_HU
    study_uid = generate_uid()
    head = build_series(tmp_path, desc="Head 5mm", n=8, hu_value=HEAD_HU, body_part="HEAD",
                        study_uid=study_uid, series_number=2, start_index=0)
    chest = build_series(tmp_path, desc="Thorax 1.0", n=8, hu_value=CHEST_HU, body_part="CHEST",
                         study_uid=study_uid, series_number=3, start_index=20)
    loose = build_series(tmp_path, desc="Recon 3", n=8, hu_value=CHEST_HU, study_uid=study_uid,
                         series_number=4, start_index=40)
    paths = head + chest + loose
    plan = srv._plan_series_route(srv._read_study_headers(paths), paths)
    assert set(plan["paths"]) == set(head)
    assert {s["description"] for s in plan["routing"]["series_excluded"]} == {"Thorax 1.0", "Recon 3"}
    assert "series_adopted_without_body_part" not in plan["routing"]


# ─── 4/12. bundle imported before its licence ──────────────────────────────

def test_bundle_import_without_licence_key_says_import_the_licence_first(api, bearer, env, monkeypatch):  # noqa: F811
    from src.utils import model_crypto
    from src.inference.auth_routes import db
    monkeypatch.delenv("SENTINEL_MODEL_KEY_HEX", raising=False)
    monkeypatch.setattr(model_crypto, "_license_secret", lambda: None)   # no licence with a model key
    monkeypatch.setattr(model_crypto, "_key_cache", None)
    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_B, seed=7)          # the vendor's licence-derived key
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 400, r.text
    assert r.json()["reason"] == "license_model_key_missing" and r.json()["ok"] is False
    assert model_crypto.key_source() == "install_secret"
    _assert_untouched(env)
    rows = [x for x in db.get_audit_log(limit=20) if x["action"] == "model_import_failed"]
    assert rows and "license_model_key_missing" in rows[0]["details"]


# ─── 6. licence renewal / expiry never orphans ciphertext ──────────────────

MACHINE = "e" * 64


@pytest.fixture
def key_world(monkeypatch, tmp_path):
    import src.utils.license as lic
    import src.utils.paths as paths
    from src.utils import model_crypto as mc
    state = {"secret": None, "license_path": tmp_path / "license.dat"}
    monkeypatch.delenv("SENTINEL_MODEL_KEY_HEX", raising=False)
    monkeypatch.setattr(paths, "MODEL_KEY_PATH", tmp_path / "data" / "model_key.bin")
    monkeypatch.setattr(lic, "get_machine_id", lambda: MACHINE)
    monkeypatch.setattr(lic, "get_license_path", lambda: state["license_path"])
    monkeypatch.setattr(lic, "license_model_key_secret", lambda *a, **k: state["secret"])
    monkeypatch.setattr(mc, "_key_cache", None)
    yield mc, state, tmp_path
    mc._key_cache = None


def _model_dir(root: Path, name: str, weights: bytes) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_renewal_and_expiry_rekey_from_the_previous_licence_key(key_world):
    mc, state, tmp = key_world
    root = tmp / "models"
    weights = os.urandom(30_000)
    d = _model_dir(root, "brain_finetuned", weights)
    enc = d / "model.safetensors.enc"

    install_key = mc.get_model_key(refresh=True)                       # 1. first start, no licence
    assert mc.encrypt_proprietary_models_at_rest(root)["encrypted"] == ["brain_finetuned"]

    state["secret"] = bytes(range(32))                                   # 2. licence A
    key_a = mc.get_model_key(refresh=True)
    assert mc.encrypt_proprietary_models_at_rest(root)["rekeyed"] == ["brain_finetuned"]
    hist = mc._key_history_path()
    assert hist.is_file() and stat.S_IMODE(hist.stat().st_mode) == 0o600
    assert key_a in mc.key_history()
    assert key_a.hex() not in json.dumps(mc.encrypt_proprietary_models_at_rest(root))

    state["secret"] = bytes(range(100, 132))                             # 3. renewal: licence B
    key_b = mc.get_model_key(refresh=True)
    rep = mc.encrypt_proprietary_models_at_rest(root)
    assert rep["rekeyed"] == ["brain_finetuned"] and rep["failed"] == {}, rep
    assert mc.decrypt_file_to_bytes(enc, key_b) == weights
    with pytest.raises(InvalidTag):
        mc.decrypt_file_to_bytes(enc, key_a)

    state["secret"] = None                                               # 4. licence expired
    assert mc.get_model_key(refresh=True) == install_key
    rep = mc.encrypt_proprietary_models_at_rest(root)
    assert rep["rekeyed"] == ["brain_finetuned"] and rep["failed"] == {}, rep
    assert mc.decrypt_file_to_bytes(enc, install_key) == weights


def test_unknown_key_is_still_reported_and_untouched(key_world):
    mc, state, tmp = key_world
    root = tmp / "models"
    d = _model_dir(root, "brain_finetuned", os.urandom(5_000))
    mc.encrypt_proprietary_models_at_rest(root, bytes([7]) * 32)        # a key this install never had
    before = (d / "model.safetensors.enc").read_bytes()
    mc.get_model_key(refresh=True)
    rep = mc.encrypt_proprietary_models_at_rest(root)
    assert rep["failed"]["brain_finetuned"].startswith("key_mismatch")
    assert (d / "model.safetensors.enc").read_bytes() == before


def test_corrupt_key_history_is_ignored(key_world):
    mc, state, tmp = key_world
    mc._key_history_path().parent.mkdir(parents=True, exist_ok=True)
    mc._key_history_path().write_text("{not json")
    assert mc.key_history() == []
    state["secret"] = bytes(range(32))
    key = mc.get_model_key(refresh=True)                                 # rewrites a valid file
    assert mc.key_history() == [key]
