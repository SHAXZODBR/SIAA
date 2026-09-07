"""Shared fixtures for the Sentinel API tests.

The in-process suite runs the FastAPI app under fastapi.testclient with
SENTINEL_DEV_INSECURE=1 and a throw-away SENTINEL_DATA_DIR. Both flags are
read at IMPORT time by src.utils.auth / src.utils.paths, so they are pinned
here — before anything under src/ is imported. tests/test_auth.py exercises
the strict (auth-required) default posture in a subprocess with a clean env.

Real-data fixtures (the brain MR study, the chest DICOM) live outside the
repo or under the git-ignored data/ tree; tests that need them skip cleanly
when they are absent instead of failing.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRATCH_DATA_DIR = Path(tempfile.mkdtemp(prefix="sentinel_test_data_"))
os.environ["SENTINEL_DEV_INSECURE"] = "1"
os.environ["SENTINEL_DATA_DIR"] = str(SCRATCH_DATA_DIR)
os.environ["SENTINEL_ADMIN_PASSWORD"] = "Test-Admin-Pass-2026"
# Honoured only together with SENTINEL_DEV_INSECURE=1: the suite makes more
# analysis calls than the unlicensed demo limit (10/day) allows, and must not
# depend on a license.dat being present on the machine that runs it.
os.environ["DEV_BYPASS_LICENSE"] = "1"
os.environ.pop("SENTINEL_REQUIRE_AUTH", None)

STUDY_DIR = Path(os.environ.get(
    "SENTINEL_TEST_STUDY_DIR",
    "/Users/shakhzodbtr/Desktop/Siaa_ai/brain_studies_keep/"
    "1.3.12.2.1107.5.2.53.190296.30000026071806470847700000031",
))
CHEST_FIXTURE = REPO / "data" / "test_dicoms" / "chest" / "chest_000.dcm"


def study_dcm_paths() -> list[Path]:
    if not STUDY_DIR.is_dir():
        return []
    return sorted(STUDY_DIR.rglob("*.dcm"))


@pytest.fixture(scope="session")
def client():
    """App under TestClient with the lifespan run (models + engines loaded)."""
    from fastapi.testclient import TestClient
    from src.inference.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def study_files() -> list[tuple]:
    """Multipart parts for POST /analyze/study — every .dcm of the real study."""
    paths = study_dcm_paths()
    if not paths:
        pytest.skip(f"brain MR study not available at {STUDY_DIR} "
                    "(set SENTINEL_TEST_STUDY_DIR)")
    return [("files", (p.name, p.read_bytes(), "application/dicom")) for p in paths]


@pytest.fixture(scope="session")
def analyzed_study(client, study_files) -> dict:
    """The real brain MR study analyzed once per session (shared by the
    analyze + sign tests)."""
    r = client.post("/analyze/study", files=study_files)
    assert r.status_code == 200, r.text
    return r.json()
