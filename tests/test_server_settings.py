"""/settings/ai (D1) — pending detectors are OFF by default in hospital builds.

Covers the resolution rule (settings.json > SENTINEL_SHOW_PENDING_DETECTORS >
hospital-build default), persistence round trip through GET/PUT (admin JWT),
the audit entry, /health.ai_settings, CORS for PUT, and that the analysis
honours the setting (brain panel post-filter, sequences passed to the report
engine, /analyze/auto). Every test points the server at a tmp DATA_DIR so the
session's shared settings.json is never touched. Synthetic data only.
"""

import json
import stat
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.inference import server as srv

ADMIN = {"username": "admin", "password": "Test-Admin-Pass-2026"}


@pytest.fixture
def settings_dir(monkeypatch, tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(srv, "DATA_DIR", d)
    monkeypatch.delenv("SENTINEL_SHOW_PENDING_DETECTORS", raising=False)
    monkeypatch.delenv("SENTINEL_HOSPITAL_BUILD", raising=False)
    return d


@pytest.fixture(scope="module")
def api() -> TestClient:
    return TestClient(srv.app)


@pytest.fixture(scope="module")
def bearer(api) -> dict:
    r = api.post("/auth/login", json=ADMIN)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ─── resolution rule ───────────────────────────────────────────────────────

@pytest.mark.parametrize("env_show,hospital,expected,source", [
    (None, None, True, "default"),
    (None, "1", False, "default"),
    ("1", "1", True, "env"),
    ("0", None, False, "env"),
    ("bogus", "1", False, "default"),
    ("bogus", None, True, "default"),
])
def test_default_rule(monkeypatch, settings_dir, env_show, hospital, expected, source):
    if env_show is not None:
        monkeypatch.setenv("SENTINEL_SHOW_PENDING_DETECTORS", env_show)
    if hospital is not None:
        monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", hospital)
    assert srv.get_ai_settings() == {"show_pending_detectors": expected, "source": source}
    assert srv.show_pending_detectors() is expected


def test_file_overrides_env(monkeypatch, settings_dir):
    monkeypatch.setenv("SENTINEL_SHOW_PENDING_DETECTORS", "1")
    old, new = srv.set_ai_settings(False)
    assert old == {"show_pending_detectors": True, "source": "env"}
    assert new == {"show_pending_detectors": False, "source": "file"}
    data = json.loads((settings_dir / "settings.json").read_text())
    assert data["show_pending_detectors"] is False and "updated_at" in data
    assert stat.S_IMODE((settings_dir / "settings.json").stat().st_mode) == 0o600


def test_unknown_keys_are_preserved_on_write(settings_dir):
    (settings_dir / "settings.json").write_text(json.dumps({"future_option": 7}))
    srv.set_ai_settings(True)
    data = json.loads((settings_dir / "settings.json").read_text())
    assert data["future_option"] == 7 and data["show_pending_detectors"] is True


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", json.dumps({"show_pending_detectors": "yes"})])
def test_corrupt_or_invalid_file_falls_back_to_the_rule(monkeypatch, settings_dir, content):
    monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", "1")
    (settings_dir / "settings.json").write_text(content)
    assert srv.get_ai_settings() == {"show_pending_detectors": False, "source": "default"}


# ─── API ───────────────────────────────────────────────────────────────────

def test_settings_require_admin_token(api, settings_dir):
    assert api.get("/settings/ai").status_code == 401
    assert api.put("/settings/ai", json={"show_pending_detectors": False}).status_code == 401


def test_put_get_round_trip_health_and_audit(api, bearer, monkeypatch, settings_dir):
    from src.inference.auth_routes import db

    monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", "1")
    r = api.get("/settings/ai", headers=bearer)
    assert r.status_code == 200 and r.json() == {"show_pending_detectors": False, "source": "default"}

    r = api.put("/settings/ai", json={"show_pending_detectors": True}, headers=bearer)
    assert r.status_code == 200, r.text
    assert r.json() == {"show_pending_detectors": True, "source": "file"}
    assert api.get("/settings/ai", headers=bearer).json() == {"show_pending_detectors": True, "source": "file"}
    assert api.get("/health").json()["ai_settings"] == {"show_pending_detectors": True, "source": "file"}
    assert srv.show_pending_detectors() is True              # next analysis sees it, no restart

    rows = [r for r in db.get_audit_log(limit=50) if r["action"] == "settings_ai_update"]
    assert rows, "setting change not audit-logged"
    assert rows[0]["details"] == "show_pending_detectors: false (default) -> true (file)"
    assert rows[0]["user_id"] and rows[0]["user_id"] != "anonymous"


def test_put_rejects_non_boolean(api, bearer, settings_dir):
    for bad in ({"show_pending_detectors": "yes"}, {"show_pending_detectors": 1}, {}):
        assert api.put("/settings/ai", json=bad, headers=bearer).status_code == 422
    assert not (settings_dir / "settings.json").exists()


def test_cors_preflight_allows_put(api):
    r = api.options("/settings/ai", headers={"Origin": "null", "Access-Control-Request-Method": "PUT",
                                             "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200
    assert "PUT" in r.headers.get("access-control-allow-methods", "")


def test_app_version_is_1_1_0(api, settings_dir):
    assert srv.APP_VERSION == "1.1.0"
    assert api.get("/health").json()["version"] == "1.1.0"


# ─── analysis honours the setting: brain panel ─────────────────────────────

def _panel_result() -> dict:
    """What brain_analysis.analyze_brain_study returns with triage + tumor run."""
    return {
        "rejected": False,
        "sequences_present": ["T1", "T2", "FLAIR"],
        "detectors_run": ["triage", "tumor_class"],
        "findings": [
            {"detector": "tumor_class", "finding": "Glioma", "class_name": "glioma", "confidence": 0.98,
             "positive": True, "status": "pending", "sequence_used": "T1"},
            {"detector": "triage", "finding": "Not flagged", "class_name": "normal", "confidence": 0.7,
             "positive": False, "status": "validated", "sequence_used": "T2"},
        ],
        "overall_assessment": {"abnormal_flagged": True, "flags": ["Glioma"], "text": "ABNORMAL — Glioma"},
        "coverage": [
            {"key": "tumor_class", "label": "Brain tumor", "status": "pending", "wired": True},
            {"key": "stroke", "label": "Ischemic stroke", "status": "experimental", "wired": True},
        ],
        "has_brats_quartet": False,
        "_preview_slice": np.full((16, 16), 0.5, dtype=np.float32),
    }


def test_visibility_filter_removes_pending_findings_and_rebuilds_overall():
    out = srv._apply_detector_visibility(_panel_result(), srv._hidden_brain_detectors(False))
    assert [f["detector"] for f in out["findings"]] == ["triage"]
    assert out["detectors_run"] == ["triage"]
    assert out["detectors_suppressed"] == ["tumor_class"]
    assert out["overall_assessment"]["abnormal_flagged"] is False
    assert "Glioma" not in json.dumps(out["overall_assessment"])
    assert "NOT a normal read" in out["overall_assessment"]["text"]
    tumor_cov = next(c for c in out["coverage"] if c["key"] == "tumor_class")
    assert tumor_cov["hidden"] is True and tumor_cov["wired"] is False
    assert next(c for c in out["coverage"] if c["key"] == "stroke")["wired"] is True


def test_visibility_filter_is_a_noop_when_pending_shown():
    res = _panel_result()
    out = srv._apply_detector_visibility(res, srv._hidden_brain_detectors(True))
    assert out["detectors_run"] == ["triage", "tumor_class"] and len(out["findings"]) == 2


class _Engine:
    """Report engine stub; `with_sequences` mimics the report agent's new keyword."""

    def __init__(self, with_sequences: bool):
        self.calls = []
        self.with_sequences = with_sequences

    def generate(self, findings, language="ru", patient_info=None, modality="CR", body_part="CHEST",
                 study_date="", **kw):
        if kw and not self.with_sequences:
            raise TypeError(f"unexpected {list(kw)}")
        self.calls.append({"findings": findings, "language": language, "modality": modality, **kw})
        return "REPORT"


class _EngineNoKw:
    def __init__(self):
        self.calls = []

    def generate(self, findings, language="ru", patient_info=None, modality="CR", body_part="CHEST",
                 study_date=""):
        self.calls.append({"findings": findings})
        return "REPORT"


def _brain_hdr() -> dict:
    return {"modality": "MR", "body_part": "BRAIN", "patient_id": "9000555", "study_date": "20310115",
            "series_descriptions": ["T1"], "protocol_names": [], "best_path": None}


@pytest.mark.parametrize("show", [False, True])
def test_brain_study_honours_setting_and_passes_sequences(monkeypatch, show):
    import src.inference.brain_analysis as ba

    calls = []

    def fake_panel(file_paths, device="cpu", include_experimental=True):
        calls.append({"include_experimental": include_experimental})
        return _panel_result()

    engine = _Engine(with_sequences=True)
    monkeypatch.setattr(ba, "analyze_brain_study", fake_panel)
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: show)
    monkeypatch.setattr(srv.state, "gemma_engine", engine)
    resp = srv._analyze_study_sync([], _brain_hdr(), "en", "test-settings-brain")
    assert calls == [{"include_experimental": False}]
    detectors = {f["detector"] for f in resp["findings"]}
    if show:
        assert detectors == {"triage", "tumor_class"}
        assert "detectors_hidden" not in resp
    else:
        assert detectors == {"triage"}
        assert resp["detectors_run"] == ["triage"]
        assert resp["detectors_hidden"] == ["tumor_class"]
        assert resp["overall_assessment"]["abnormal_flagged"] is False
        assert all(m["key"] != "brain_tumor_class" for m in resp["model_identity"])
    assert resp["show_pending_detectors"] is show
    assert resp["sequences_present"] == ["T1", "T2", "FLAIR"]
    assert engine.calls[0]["sequences"] == ["T1", "T2", "FLAIR"]
    assert {f["detector"] for f in engine.calls[0]["findings"]} == detectors


def test_detector_statuses_keyword_is_used_when_supported(monkeypatch):
    import src.inference.brain_analysis as ba

    seen = {}

    def fake_panel(file_paths, device="cpu", include_experimental=True, detector_statuses=None):
        seen["statuses"] = detector_statuses
        return _panel_result()

    monkeypatch.setattr(ba, "analyze_brain_study", fake_panel)
    out = srv._run_brain_panel([], include_experimental=False, show_pending=False)
    assert seen["statuses"] == {"validated"}
    assert [f["detector"] for f in out["findings"]] == ["triage"]       # post-filter still applied
    srv._run_brain_panel([], include_experimental=False, show_pending=True)
    assert seen["statuses"] == {"validated", "pending"}


def test_engine_without_sequences_keyword_still_works(monkeypatch):
    engine = _EngineNoKw()
    monkeypatch.setattr(srv.state, "gemma_engine", engine)
    text, used = srv._generate_report([], "en", modality="MR", body_part="BRAIN", sequences=["T1"])
    assert text == "REPORT" and used is True and engine.calls


def test_only_hidden_detectors_ran_gives_no_validated_detector(monkeypatch):
    import src.inference.brain_analysis as ba

    def fake_panel(file_paths, device="cpu", include_experimental=True):
        res = _panel_result()
        res["detectors_run"] = ["tumor_class"]
        res["findings"] = res["findings"][:1]
        return res

    monkeypatch.setattr(ba, "analyze_brain_study", fake_panel)
    monkeypatch.setattr(srv, "show_pending_detectors", lambda: False)
    monkeypatch.setattr(srv.state, "gemma_engine", None)
    resp = srv._analyze_study_sync([], _brain_hdr(), "uz", "test-settings-brain-2")
    assert resp["requires_review"] is True
    assert resp["rejection_reason"] == "no_validated_detector"
    assert resp["findings"] == [] and resp["detectors_run"] == []
    assert resp["overall_impression"] == srv._NO_VALIDATED_NOTES["uz"]


def test_localized_notes_exist_for_every_language():
    assert set(srv._NO_VALIDATED_NOTES) == {"en", "ru", "uz"}
    assert srv._no_validated_note("xx") == srv._NO_VALIDATED_NOTES["en"]


# ─── /analyze/auto honours the setting ─────────────────────────────────────

def test_analyze_auto_hidden_pending_returns_requires_review(api, monkeypatch, settings_dir, tmp_path):
    from pydicom.uid import generate_uid
    from test_ct_series_routing import write_ct

    from src.inference import model_registry as mr

    def must_not_run(*a, **k):
        raise AssertionError("no model may run when pending detectors are hidden")

    monkeypatch.setenv("SENTINEL_HOSPITAL_BUILD", "1")
    monkeypatch.setattr(srv, "_run_registry_model", must_not_run)
    monkeypatch.setattr(srv.state, "license_valid", True)
    assert mr.REGISTRY["head_ct"].validation_status != "validated"
    f = write_ct(tmp_path / "one.dcm", study_uid=generate_uid(), series_uid=generate_uid(), desc="Head 5mm",
                 instance=1, hu_value=40, body_part="HEAD")
    r = api.post("/analyze/auto?language=ru", files={"file": ("one.dcm", f.read_bytes(), "application/dicom")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requires_review"] is True
    assert body["rejection_reason"] == "no_validated_detector"
    assert body["findings"] == [] and body["normal"] is False
    assert body["overall_impression"] == srv._NO_VALIDATED_NOTES["ru"]
    assert body["app_version"] == "1.1.0"
