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


# ---------------------------------------------------------------- secrets in log tails

LOG_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}"   # src/utils/logger.py file sink
SECRET_LINES = {
    "pw-kv-9000Zq": "2031-01-02 10:00:00 | INFO | pacs - ORTHANC_PASSWORD=pw-kv-9000Zq connecting",
    "pw-colon-9000Zq": "smtp password: pw-colon-9000Zq",
    "pw-json-9000Zq": '{"user": "synthetic", "password": "pw-json-9000Zq", "port": 4242}',
    "passwd-9000Zq": "passwd = passwd-9000Zq",
    "tok-api-9000Zq": "settings loaded api_token=tok-api-9000Zq ok",
    "sec-9000Zq": "client_secret: 'sec-9000Zq'",
    "apikey-9000Zq": "X-Api-Key: apikey-9000Zq",
    "apikey2-9000Zq": "api key = apikey2-9000Zq",
    "bearer-9000Zq-abcdef": "GET /health Authorization: Bearer bearer-9000Zq-abcdef",
    "basic-9000Zq": '"authorization": "Basic basic-9000Zq"',
    "raw-bearer-9000Zq-xyz": "retrying with bearer raw-bearer-9000Zq-xyz",
    "c" * 64: f'{{"model_key_secret": "{"c" * 64}", "tier": "pro"}}',
    "lic-9000Zq": "licence_key=lic-9000Zq",
    "url-pw-9000Zq": "PACS at http://pacsuser:url-pw-9000Zq@pacs.synthetic.test:8042/dicom-web",
}
ORDINARY = ["routing study to brain_triage key=brain_triage threshold=0.50",
            "model brain_triage loaded on cpu in 1.2s",
            "Default admin created with SENTINEL_ADMIN_PASSWORD from env — must_change_password is set"]


@pytest.fixture
def first_run_banner(monkeypatch, tmp_path, capsys) -> tuple[str, str]:
    """The REAL first-run banner: AuthManager.create_default_admin on an empty
    synthetic DB, captured from stdout and from a log sink in the app's file format.
    Returns (log text, one-time password)."""
    from loguru import logger
    from src.utils.auth import AuthManager
    from src.utils.database import SentinelDB
    monkeypatch.delenv("SENTINEL_ADMIN_PASSWORD", raising=False)
    sink = tmp_path / "banner.log"
    sink_id = logger.add(str(sink), level="WARNING", format=LOG_FMT)
    capsys.readouterr()
    try:
        AuthManager(SentinelDB(str(tmp_path / "banner.db"))).create_default_admin()
    finally:
        logger.remove(sink_id)
    stdout = capsys.readouterr().out
    m = re.search(r"password: (\S+)", stdout)
    assert m and "FIRST-RUN ADMIN ACCOUNT CREATED" in stdout
    pw = m.group(1)
    text = stdout + sink.read_text(encoding="utf-8")
    assert text.count(pw) >= 2                    # console + log file
    return text, pw


def test_log_tails_drop_admin_banner_password_and_secrets(api, bearer, diag_env, first_run_banner):
    banner, pw = first_run_banner
    for name in ("backend.log", "sentinel_2031-01-02.log"):
        with open(diag_env["logs"] / name, "a", encoding="utf-8") as fh:
            fh.write(banner + "\n" + "\n".join(SECRET_LINES.values()) + "\n" + "\n".join(ORDINARY) + "\n")
    zf, _ = _fetch(api, bearer)
    blob = b"".join(zf.read(n) for n in zf.namelist()).decode("utf-8", errors="replace")
    assert pw not in blob, "first-run admin password leaked into the diagnostics zip"
    for secret in SECRET_LINES:
        assert secret not in blob, f"{secret!r} leaked into the diagnostics zip"
    for log in ("logs/backend.log", "logs/sentinel_2031-01-02.log"):
        text = zf.read(log).decode()
        assert "FIRST-RUN ADMIN ACCOUNT CREATED" in text and "password: [REDACTED]" in text
        assert "pacs.synthetic.test:8042/dicom-web" in text and "pacsuser" not in text
        for line in ORDINARY:
            assert line in text
    assert FAKE_PID not in blob and FAKE_ACC not in blob          # PHI redaction still works


def test_env_url_userinfo_is_stripped(api, bearer, diag_env, monkeypatch):
    monkeypatch.setenv("ORTHANC_URL", "http://pacsuser:url-env-9000Zq@pacs.synthetic.test:8042/")
    monkeypatch.setenv("SENTINEL_UPDATE_URL", "https://updates.synthetic.test/feed")
    zf, _ = _fetch(api, bearer)
    blob = b"".join(zf.read(n) for n in zf.namelist()).decode("utf-8", errors="replace")
    assert "url-env-9000Zq" not in blob and "pacsuser" not in blob
    env = json.loads(zf.read("settings_redacted.json"))["environment"]
    assert env["ORTHANC_URL"] == "http://pacs.synthetic.test:8042/"
    assert env["SENTINEL_UPDATE_URL"] == "https://updates.synthetic.test/feed"


def test_secret_text_helper():
    assert srv._redact_secret_text("  password: Ab-c_9000") == "  password: [REDACTED]"
    assert srv._redact_secret_text('{"api_token": "t 9000"}') == '{"api_token": "[REDACTED]"}'
    assert srv._redact_secret_text("Authorization: Bearer abcdefgh123") == "Authorization: Bearer [REDACTED]"
    assert srv._redact_secret_text("ftp://u:p@h.test/x") == "ftp://h.test/x"
    assert srv._redact_secret_text(ORDINARY[0]) == ORDINARY[0]


def test_url_userinfo_with_raw_at_in_password_is_fully_stripped():
    """Clients split userinfo at the LAST '@': no fragment of the password may survive."""
    assert srv._redact_secret_text("Orthanc reachable at http://admin:p@ss9000@pacs.synthetic.test:8042/x") \
        == "Orthanc reachable at http://pacs.synthetic.test:8042/x"
    assert srv._redact_secrets({"ORTHANC_URL": "http://admin:p@ss9000@pacs.synthetic.test:8042"}) \
        == {"ORTHANC_URL": "http://pacs.synthetic.test:8042"}
    q = "https://pacs.synthetic.test/q?mail=a@b.test"
    assert srv._redact_secret_text(q) == q                      # '@' after the path is not userinfo


@pytest.mark.parametrize("line", ["token_" * 20000, "a_" * 200000, "1.2.840." * 50000,
                                  "x" * 100000 + "://" + "y" * 100000])
def test_secret_redaction_is_linear_on_long_lines(line):
    """One long log line without spaces (UID lists, base64url dumps) must not stall the export."""
    t0 = time.perf_counter()
    srv._redact_secret_text(line)
    assert time.perf_counter() - t0 < 1.0


def test_secret_names_with_prefixes_still_redacted_after_anchoring():
    assert srv._redact_secret_text("cfg.orthanc_password=pw-9000Zq") == "cfg.orthanc_password=[REDACTED]"
    assert srv._redact_secret_text("--api-token=tk-9000Zq") == "--api-token=[REDACTED]"


def test_secret_values_with_spaces_and_escaped_quotes_leave_nothing_behind():
    """An unquoted value runs to the next assignment or the end of the line, and a JSON
    string ends at its real closing quote, not at an escaped one."""
    assert srv._redact_secret_text("password=two words pw9000Zq") == "password=[REDACTED]"
    assert srv._redact_secret_text("password=two words pw9000Zq retries=3") == "password=[REDACTED] retries=3"
    assert srv._redact_secret_text("token: ab cd9000Zq 10:42:07 done") == "token: [REDACTED] 10:42:07 done"
    assert srv._redact_secret_text('{"password": "ab\\"cd-9000Zq", "port": 8042}') \
        == '{"password": "[REDACTED]", "port": 8042}'
    assert srv._redact_secret_text("{'secret': 'x\\'y9000Zq'}") == "{'secret': '[REDACTED]'}"
    assert srv._redact_secret_text("key=brain_triage route=head_ct") == "key=brain_triage route=head_ct"


@pytest.mark.parametrize("line", ["password=" + "a " * 100000, "password=x " + "a" * 200000,
                                  'token: "' + "\\\\" * 100000, "secret=" + "a =" * 50000])
def test_secret_value_matching_is_linear_too(line):
    t0 = time.perf_counter()
    srv._redact_secret_text(line)
    assert time.perf_counter() - t0 < 1.0
