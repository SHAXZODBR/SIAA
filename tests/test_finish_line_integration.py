"""Cross-package integration of the v1.1.0 finish-line contract.

Regression tests for the integration notes applied after the seven package
agents finished:

* brain_analysis.analyze_brain_study(detector_statuses=…) — a hidden pending
  detector's model is never even requested (D1), and the triage output is only
  'validated' when the eval next to the LOADED weights says so.
* validated_metrics.reload() — refreshes METRICS and the module constants.
* server._log_encryption_report — a failed dict never yields the 'OK' line.
* server._encrypt_previous_release — uses the verified encrypt_dir_at_rest.
* server._maybe_encrypt_models_at_rest — .app detection by path component.
* /report/regenerate forwards `sequences` to the engine (D4).
* the uz no-validated-detector note quotes the desktop's menu label.

Synthetic data only (ids ≥ 9000000, dates 2031); no real model is loaded.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from src.inference import server as srv

REPO = Path(__file__).resolve().parents[1]


# ─── brain_analysis: detector_statuses ─────────────────────────────────────

def _axial_study():
    from src.pipeline.brain_mri_preprocessor import BrainStudy
    rng = np.random.default_rng(9000001)
    vol = rng.random((20, 64, 64)).astype(np.float32)
    return BrainStudy(patient_id="9000001", study_uid="2.25.9000001",
                      sequences={"T1": vol}, seq_meta={"T1": {"orientation": "axial"}})


class _Predictor:
    def __call__(self, image):
        return {"abnormal": 0.2, "normal": 0.8, "glioma": 0.9, "meningioma": 0.05,
                "pituitary": 0.03, "no_tumor": 0.02}


@pytest.fixture
def fake_panel(monkeypatch):
    """Every registry key 'loads'; records which keys the panel asked for."""
    import src.inference.brain_analysis as ba
    import src.inference.model_registry as mr

    requested: list[str] = []

    def get_model(key, device="cpu"):
        requested.append(key)
        return {"available": True, "predictor": _Predictor()}

    monkeypatch.setattr(mr, "get_model", get_model)
    monkeypatch.setattr(ba, "process_brain_study", lambda paths: _axial_study())
    monkeypatch.setattr(ba, "robust_normalize_slice", lambda s: np.asarray(s, dtype=np.float32))
    return ba, requested


def test_hidden_pending_detector_is_never_loaded(fake_panel):
    ba, requested = fake_panel
    res = ba.analyze_brain_study([], include_experimental=False, detector_statuses={"validated"})
    assert "brain_tumor_class" not in requested
    assert "tumor_class" not in res["detectors_run"]
    assert all(f["detector"] != "tumor_class" for f in res["findings"])
    assert "triage" in res["detectors_run"]


def test_detector_statuses_none_keeps_legacy_behaviour(fake_panel):
    ba, requested = fake_panel
    res = ba.analyze_brain_study([], include_experimental=False)
    assert "brain_tumor_class" in requested and "tumor_class" in res["detectors_run"]
    # experimental stays governed by include_experimental, even if the set allows it
    requested.clear()
    ba.analyze_brain_study([], include_experimental=False,
                           detector_statuses={"validated", "pending", "experimental"})
    assert "brain_stroke" not in requested and "brain_dementia" not in requested


def test_server_passes_detector_statuses_to_the_real_panel(fake_panel, monkeypatch):
    ba, requested = fake_panel
    out = srv._run_brain_panel([], include_experimental=False, show_pending=False)
    assert "brain_tumor_class" not in requested
    assert out["detectors_hidden"] == ["tumor_class"] and out["detectors_suppressed"] == []


def _triage_status(res: dict) -> list[str]:
    return [f["status"] for f in res["findings"] if f["detector"] == "triage"]


def test_fallback_metrics_do_not_validate_foreign_weights(fake_panel, monkeypatch):
    """An imported/hand-copied model without study_level_eval.json must not inherit the
    built-in release's 'validated' label."""
    import src.inference.model_registry as mr
    import src.inference.validated_metrics as vm
    ba, _ = fake_panel
    monkeypatch.setattr(vm, "METRICS", dataclasses.replace(
        vm.METRICS, source="fallback", validated=True, identity="aaaaaaaaaaaa"))
    monkeypatch.setattr(mr, "get_model_identity", lambda key: {"sha256_12": "bbbbbbbbbbbb"})
    res = ba.analyze_brain_study([], include_experimental=False, detector_statuses={"validated"})
    assert _triage_status(res) == ["pending"]
    assert any("bbbbbbbbbbbb" in n for n in res["notes"])

    monkeypatch.setattr(mr, "get_model_identity", lambda key: {"sha256_12": "aaaaaaaaaaaa"})
    res = ba.analyze_brain_study([], include_experimental=False, detector_statuses={"validated"})
    assert _triage_status(res) == ["validated"]


def test_json_metrics_validated_flag_is_authoritative(fake_panel, monkeypatch):
    import src.inference.validated_metrics as vm
    ba, _ = fake_panel
    monkeypatch.setattr(vm, "METRICS", dataclasses.replace(vm.METRICS, source="json", validated=True))
    assert _triage_status(ba.analyze_brain_study([], include_experimental=False)) == ["validated"]
    monkeypatch.setattr(vm, "METRICS", dataclasses.replace(
        vm.METRICS, source="json", validated=False, not_validated_reason="gate.passed is False"))
    assert _triage_status(ba.analyze_brain_study([], include_experimental=False)) == ["pending"]


# ─── validated_metrics.reload ──────────────────────────────────────────────

_CONSTANTS = ("METRICS", "MODEL_DIR", "SOURCE", "SENS", "SPEC", "ACC", "TP", "FN", "TN", "FP",
              "N_STUDIES", "N_ABNORMAL", "N_NORMAL", "N_SLICES", "EVALUATED_ON", "IDENTITY",
              "MIN_SENS", "MIN_SPEC", "THRESHOLD", "SENS_CI", "SPEC_CI")


def test_validated_metrics_reload_refreshes_constants(monkeypatch, tmp_path):
    import src.inference.validated_metrics as vm
    for name in _CONSTANTS:
        monkeypatch.setattr(vm, name, getattr(vm, name))
    d = tmp_path / "brain_triage_finetuned"
    d.mkdir()
    (d / "study_level_eval.json").write_text(json.dumps({
        "gate": {"name": "gate mean @0.50", "acc": 0.8, "sens": 0.9, "spec": 0.5, "tp": 9, "fn": 1,
                 "tn": 5, "fp": 5, "min_sens": 0.85, "min_spec": 0.4, "passed": True},
        "n_studies": 20, "n_slices": 100, "evaluated_at": "2031-01-15T10:00:00"}))
    (d / "MANIFEST.json").write_text(json.dumps({"files": {"model.safetensors": "c" * 64}}))
    m = vm.reload(d)
    assert m is vm.METRICS and m.source == "json"
    assert (vm.IDENTITY, vm.SENS, vm.SPEC, vm.N_STUDIES, vm.TP) == ("c" * 12, 0.9, 0.5, 20, 9)
    assert vm.MODEL_DIR == d and vm.EVALUATED_ON == "2031-01-15"
    assert vm.summary_line().endswith("identity cccccccccccc [json]")
    # an unreadable eval falls back, exactly like load()
    assert vm.reload(tmp_path / "missing").source == "fallback" and vm.SOURCE == "fallback"


# ─── encryption-at-rest logging / previous release / .app ──────────────────

@pytest.fixture
def log_lines():
    lines: list[tuple[str, str]] = []
    sink = logger.add(lambda msg: lines.append((msg.record["level"].name, msg.record["message"])),
                      level="DEBUG")
    yield lines
    logger.remove(sink)


def test_failed_dict_is_logged_as_error_and_never_ok(log_lines):
    srv._log_encryption_report({"encrypted": [], "already": [], "skipped": [], "rekeyed": [],
                                "failed": {"brain_triage_finetuned": "verify_mismatch: sha differs"}})
    errors = [m for lvl, m in log_lines if lvl == "ERROR"]
    assert any("brain_triage_finetuned" in m and "verify_mismatch" in m for m in errors)
    assert not any("OK" in m for _, m in log_lines)


def test_empty_failed_dict_reports_ok(log_lines):
    srv._log_encryption_report({"encrypted": ["brain_triage_finetuned"], "already": [], "skipped": [],
                                "rekeyed": [], "failed": {}})
    assert not [m for lvl, m in log_lines if lvl == "ERROR"]
    assert any("OK" in m for _, m in log_lines)


def test_previous_release_uses_verified_encryption(monkeypatch, tmp_path):
    from src.utils import model_crypto
    prev = tmp_path / "brain_triage_finetuned_prev_9000aaaaaaaa"
    prev.mkdir()
    (prev / "model.safetensors").write_bytes(b"\0" * 32)
    calls = []
    monkeypatch.setenv("SENTINEL_ENCRYPT_MODELS_AT_REST", "1")
    monkeypatch.setattr(srv, "DEV_INSECURE", False)
    monkeypatch.setattr(model_crypto, "encrypt_dir_at_rest", lambda d, key=None: calls.append(d) or d)
    monkeypatch.setattr(model_crypto, "encrypt_model_dir",
                        lambda *a, **k: pytest.fail("unverified legacy encrypt_model_dir used"))
    srv._encrypt_previous_release(prev)
    assert calls == [prev]


def test_app_bundle_detected_by_path_component(monkeypatch):
    from src.utils import model_crypto
    monkeypatch.setenv("SENTINEL_ENCRYPT_MODELS_AT_REST", "1")
    monkeypatch.setattr(srv, "DEV_INSECURE", False)
    monkeypatch.setattr(srv.sys, "platform", "darwin")
    monkeypatch.setattr(model_crypto, "encrypt_proprietary_models_at_rest",
                        lambda d: pytest.fail("must not encrypt inside a .app"))
    assert srv._maybe_encrypt_models_at_rest(Path("/Volumes/X/Sentinel.app")) is None


# ─── /report/regenerate forwards sequences ─────────────────────────────────

class _SeqEngine:
    def __init__(self):
        self.calls = []

    def generate(self, findings, language="ru", patient_info=None, modality="CR", body_part="CHEST",
                 study_date="", sequences=None):
        self.calls.append({"sequences": sequences, "modality": modality})
        return "REPORT"


def test_regenerate_forwards_sequences(monkeypatch):
    engine = _SeqEngine()
    monkeypatch.setattr(srv.state, "gemma_engine", engine)
    api = TestClient(srv.app)
    body = {"findings": [], "language": "en", "modality": "MR", "body_part": "BRAIN"}
    r = api.post("/report/regenerate", json={**body, "sequences": ["T1", "T1ce", "FLAIR"]})
    assert r.status_code == 200, r.text
    assert engine.calls[-1]["sequences"] == ["T1", "T1ce", "FLAIR"]
    r = api.post("/report/regenerate", json=body)                  # absent → engine default
    assert r.status_code == 200 and engine.calls[-1]["sequences"] is None


# ─── localized note quotes the desktop label ───────────────────────────────

def test_uz_note_quotes_the_desktop_ai_models_label():
    uz = (REPO / "desktop-app/src/i18n/uz.ts").read_text(encoding="utf-8")
    m = re.search(r"'settings\.ai':\s*'([^']+)'", uz)
    assert m, "settings.ai missing from uz.ts"
    label = m.group(1)
    assert f"Sozlamalar → {label}" in srv._NO_VALIDATED_NOTES["uz"]
    assert "SI modellari" not in srv._NO_VALIDATED_NOTES["uz"]


# ─── /health names the loaded triage weights; hidden detectors are not warmed ──

def test_health_reports_cached_triage_identity(monkeypatch):
    from src.inference import model_registry as mr
    monkeypatch.setattr(mr, "_loaded_models", {"brain_triage": {"available": True, "reason": "loaded from x"}})
    monkeypatch.setattr(mr, "_identity_cache", {"brain_triage": {"sha256_12": "9000abcdef12"}})
    body = TestClient(srv.app).get("/health").json()
    assert body["models"]["brain_triage"]["identity"] == "9000abcdef12"
    # not hashed yet → null, never a 350 MB hash inside /health
    monkeypatch.setattr(mr, "_identity_cache", {})
    monkeypatch.setattr(mr, "get_model_identity", lambda k: pytest.fail("/health must not hash"))
    assert TestClient(srv.app).get("/health").json()["models"]["brain_triage"]["identity"] is None


def test_warm_up_skips_hidden_pending_detectors():
    import inspect
    src = inspect.getsource(srv.lifespan)
    warm = src[src.index("Warm the brain panel"):src.index("Load Gemma 3 report engine")]
    assert "if show_pending_detectors() else ('brain_triage',)" in warm
    assert "_get_identity(_key)" in warm
