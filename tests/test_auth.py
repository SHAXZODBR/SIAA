"""Authentication is REQUIRED by default.

The flags are import-time constants, so the strict posture is exercised in a
subprocess (tests/_auth_probe.py) with SENTINEL_DEV_INSECURE removed from the
env, a fresh SENTINEL_DATA_DIR and a known SENTINEL_ADMIN_PASSWORD.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPO, STUDY_DIR

PROBE = Path(__file__).with_name("_auth_probe.py")
ADMIN_PW = "Probe-Admin-Pass-2026"


@pytest.fixture(scope="module")
def probe() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("SENTINEL_DEV_INSECURE", "DEV_BYPASS_LICENSE", "SENTINEL_REQUIRE_AUTH")}
    env["SENTINEL_DATA_DIR"] = tempfile.mkdtemp(prefix="sentinel_test_auth_")
    env["SENTINEL_ADMIN_PASSWORD"] = ADMIN_PW
    env["SENTINEL_SKIP_WARMUP"] = "1"          # models load lazily if the study is present
    if STUDY_DIR.is_dir():
        env["SENTINEL_TEST_STUDY_DIR"] = str(STUDY_DIR)
    proc = subprocess.run([sys.executable, str(PROBE)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=600)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert proc.returncode == 0 and lines, (
        f"probe failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")
    return json.loads(lines[-1][len("__RESULT__"):])


def test_health_is_public_and_reports_auth_required(probe):
    assert probe["health"] == {"status": 200, "auth_required": True}


def test_openapi_docs_hidden_when_strict(probe):
    assert probe["docs"] == 404
    assert probe["openapi"] == 404


def test_clinical_routes_401_without_token(probe):
    assert probe["analyze_no_token"]["status"] == 401
    assert probe["analyze_no_token"]["detail"] == "Authentication required"
    assert probe["studies_no_token"] == 401
    assert probe["study_no_token"] == 401
    assert probe["sign_no_token"] == 401
    assert probe["correction_no_token"] == 401
    assert probe["pacs_status_no_token"] == 401
    assert probe["report_pdf_no_token"] == 401


def test_wrong_password_401(probe):
    assert probe["login_wrong"] == 401
    assert probe["login_unknown_user"] == 401


def test_login_contract(probe):
    login = probe["login"]
    assert login["status"] == 200
    assert login["keys"] == ["access_token", "must_change_password", "token_type", "user"]
    assert login["token_type"] == "bearer"
    assert login["user_keys"] == ["full_name", "id", "role", "username"]
    assert login["role"] == "admin"
    assert login["must_change_password"] is True     # env-seeded admin must rotate it


def test_bearer_token_accepted(probe):
    assert probe["me_bearer"] == 200
    assert probe["studies_bearer"] == 200


def test_bad_tokens_rejected(probe):
    assert probe["bogus_token"] == 401
    assert probe["basic_scheme"] == 401


def test_analyze_with_bearer(probe):
    if probe["analyze_bearer"] is None:
        pytest.skip("brain MR study not available for the bearer analyze check")
    assert probe["analyze_bearer"]["status"] == 200
    assert probe["analyze_bearer"]["findings"] >= 1
    assert probe["analyze_bearer"]["persisted"] is True
