"""GET /diagnostics — PHI-safe support zip (admin JWT).

Checks the exact member set, the 2 MB log-tail cap, that nothing under
previews/, no DB, no license.dat and no key ever lands in the zip, that patient
fields in the audit tail and the logs are redacted, that secret-looking env
vars are masked, and that the export itself is audit-logged. Synthetic values
only (ids >= 9000000).
"""

import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.inference import server as srv

ADMIN = {"username": "admin", "password": "Test-Admin-Pass-2026"}
FAKE_PID = "9000123"
FAKE_PID_JSON = "9000124"
FAKE_ACC = "A9000125"
FAKE_PN = "Synthetic^Diagpatient"
FAKE_KEY_HEX = "ab" * 32
BASE_MEMBERS = {"health.json", "license.json", "ai_settings.json", "versions.json", "models.json",
                "audit_tail.jsonl", "settings_redacted.json"}


@pytest.fixture(scope="module")
def api() -> TestClient:
    return TestClient(srv.app)


@pytest.fixture(scope="module")
def bearer(api) -> dict:
    r = api.post("/auth/login", json=ADMIN)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def diag_env(monkeypatch, tmp_path) -> dict:
    """Isolated DATA_DIR / LOG_DIR with an oversized backend.log, two app logs,
    a preview PNG, a fake DB and a license.dat that must never be exported."""
    data = tmp_path / "data"
    logs = data / "logs"
    previews = data / "previews"
    logs.mkdir(parents=True)
    previews.mkdir()
    monkeypatch.setattr(srv, "DATA_DIR", data)
    monkeypatch.setattr(srv, "LOG_DIR", logs)
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", FAKE_KEY_HEX)
    monkeypatch.setenv("ORTHANC_PASSWORD", "orthanc-secret-9000")

    filler = ("x" * 99 + "\n").encode()
    backend = logs / "backend.log"
    with open(backend, "wb") as fh:
        fh.write(b"OLDEST-LINE-MUST-BE-CUT\n")
        fh.write(filler * 25_000)                                   # ~2.5 MB
        fh.write(f"study done patient_id={FAKE_PID} name {FAKE_PN}\nBACKEND-LAST-LINE\n".encode())
    old_log = logs / "sentinel_2031-01-01.log"
    old_log.write_text("OLD-APP-LOG\n")
    new_log = logs / "sentinel_2031-01-02.log"
    new_log.write_text(f"NEW-APP-LOG accession_number={FAKE_ACC}\n")
    past = time.time() - 3600
    os.utime(old_log, (past, past))
    (previews / "study-9000123.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (data / "sentinel.db").write_bytes(b"SQLite format 3\x00fake")
    (data / "license.dat").write_text('{"model_key_secret": "deadbeef"}')
    (data / "settings.json").write_text(json.dumps({"show_pending_detectors": False, "api_token": "tok-9000"}))
    return {"data": data, "logs": logs}


@pytest.fixture
def audit_rows():
    from src.inference.auth_routes import db
    db.log_action("diag-test", "test_phi_entry", "study", "diag-study-1",
                  details=f"patient_id={FAKE_PID}; patient_name={FAKE_PN}; accession_number={FAKE_ACC}")
    db.log_action("diag-test", "test_phi_entry_json", "study", "diag-study-2",
                  details=json.dumps({"patient_id": FAKE_PID_JSON, "accession": FAKE_ACC, "route": "head_ct"}))


def _fetch(api, bearer) -> tuple[zipfile.ZipFile, object]:
    r = api.get("/diagnostics", headers=bearer)
    assert r.status_code == 200, r.text
    return zipfile.ZipFile(io.BytesIO(r.content)), r


def test_requires_admin(api):
    assert api.get("/diagnostics").status_code == 401


def test_zip_members_and_headers(api, bearer, diag_env, audit_rows):
    zf, r = _fetch(api, bearer)
    assert r.headers["content-type"] == "application/zip"
    assert re.fullmatch(r"attachment; filename=sentinel-diagnostics-\d{8}-\d{4}\.zip",
                        r.headers["content-disposition"])
    names = set(zf.namelist())
    assert names == BASE_MEMBERS | {"logs/backend.log", "logs/sentinel_2031-01-02.log"}
    assert not any(n.startswith("previews/") or n.endswith((".png", ".db", ".dat", ".bin", ".key"))
                   for n in names)


def test_contents_are_phi_and_secret_free(api, bearer, diag_env, audit_rows):
    zf, _ = _fetch(api, bearer)
    blob = b"".join(zf.read(n) for n in zf.namelist()).decode("utf-8", errors="replace")
    for needle in (FAKE_PID, FAKE_PID_JSON, FAKE_ACC, FAKE_PN, "Diagpatient", FAKE_KEY_HEX,
                   "orthanc-secret-9000", "tok-9000", "deadbeef", "SQLite format 3"):
        assert needle not in blob, f"{needle!r} leaked into the diagnostics zip"

    audit = [json.loads(line) for line in zf.read("audit_tail.jsonl").decode().splitlines() if line]
    assert 0 < len(audit) <= srv.DIAG_AUDIT_TAIL_ROWS
    phi_rows = [a for a in audit if a["action"] in ("test_phi_entry", "test_phi_entry_json")]
    assert len(phi_rows) >= 2 and all("[REDACTED]" in a["details"] for a in phi_rows)
    json_row = next(a for a in phi_rows if a["action"] == "test_phi_entry_json")
    assert json.loads(json_row["details"])["route"] == "head_ct"           # non-PHI kept

    settings = json.loads(zf.read("settings_redacted.json"))
    assert settings["environment"]["SENTINEL_MODEL_KEY_HEX"] == "[REDACTED]"
    assert settings["environment"]["ORTHANC_PASSWORD"] == "[REDACTED]"
    assert settings["settings_file"]["api_token"] == "[REDACTED]"
    assert settings["settings_file"]["show_pending_detectors"] is False


def test_sections_carry_status_versions_and_identities(api, bearer, diag_env):
    zf, _ = _fetch(api, bearer)
    health = json.loads(zf.read("health.json"))
    assert health["version"] == srv.APP_VERSION and "ai_settings" in health
    lic = json.loads(zf.read("license.json"))
    assert set(lic) == {"valid", "info", "demo_calls_today", "demo_limit"}
    assert json.loads(zf.read("ai_settings.json")) == {"show_pending_detectors": False, "source": "file"}
    versions = json.loads(zf.read("versions.json"))
    for key in ("app", "python", "torch", "transformers", "os"):
        assert key in versions
    assert versions["app"] == "1.1.0"
    models = json.loads(zf.read("models.json"))
    assert {"brain_triage", "brain_tumor_class", "head_ct"} <= set(models)
    assert models["brain_triage"]["status"] == "validated"


def test_log_tails_are_capped_and_redacted(api, bearer, diag_env):
    zf, _ = _fetch(api, bearer)
    backend = zf.read("logs/backend.log")
    assert len(backend) <= srv.DIAG_LOG_TAIL_BYTES
    text = backend.decode()
    assert "OLDEST-LINE-MUST-BE-CUT" not in text and "BACKEND-LAST-LINE" in text
    assert "patient_id=" in text and FAKE_PID not in text
    assert "NEW-APP-LOG" in zf.read("logs/sentinel_2031-01-02.log").decode()


def test_missing_logs_are_skipped_not_fatal(api, bearer, monkeypatch, tmp_path):
    empty = tmp_path / "nologs"
    empty.mkdir()
    monkeypatch.setattr(srv, "LOG_DIR", empty)
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    zf, _ = _fetch(api, bearer)
    assert set(zf.namelist()) == BASE_MEMBERS


def test_export_is_audit_logged(api, bearer, diag_env):
    from src.inference.auth_routes import db
    _fetch(api, bearer)
    rows = [r for r in db.get_audit_log(limit=20) if r["action"] == "export_diagnostics"]
    assert rows and rows[0]["details"].startswith("members=")


def test_redaction_helpers():
    assert srv._redact_text(f"patient_id={FAKE_PID}") == "patient_id=[REDACTED]"
    assert FAKE_PID not in srv._redact_text(f'{{"PatientID": "{FAKE_PID}"}}')
    obj = srv._redact_obj({"patient_name": FAKE_PN, "nested": [{"accession_number": FAKE_ACC}], "ok": 1})
    assert obj == {"patient_name": "[REDACTED]", "nested": [{"accession_number": "[REDACTED]"}], "ok": 1}
    assert srv._redact_secrets({"SENTINEL_TLS_KEY": "/p", "SENTINEL_OFFLINE": "1"}) == \
        {"SENTINEL_TLS_KEY": "[REDACTED]", "SENTINEL_OFFLINE": "1"}
