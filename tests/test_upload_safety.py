"""Client filenames never touch the disk — traversal names cannot escape the
per-request temp dir, and the temp dir is removed on every exit path."""

import os
import tempfile
from pathlib import Path

import pytest

from conftest import REPO, SCRATCH_DATA_DIR, study_dcm_paths

TRAVERSAL_NAMES = ["../../pwn.dcm", "..\\..\\pwn.dcm", "/tmp/pwn.dcm", "../pwn.dcm"]
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "release"}


def _find(name: str, roots: list[Path]) -> list[str]:
    hits = []
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if name in filenames:
                hits.append(os.path.join(dirpath, name))
    return hits


def _search_roots() -> list[Path]:
    tmp = Path(tempfile.gettempdir()).resolve()
    return [tmp, tmp.parent, REPO, SCRATCH_DATA_DIR, Path.cwd()]


def _leftover_request_dirs() -> list[str]:
    tmp = Path(tempfile.gettempdir())
    return [str(p) for p in tmp.iterdir()
            if p.is_dir() and (p.name.startswith("sentinel_brain_") or p.name.startswith("sentinel_study_"))]


@pytest.fixture(scope="module")
def brain_dcm() -> bytes:
    paths = study_dcm_paths()
    if not paths:
        pytest.skip("brain MR study not available")
    return paths[0].read_bytes()


@pytest.mark.parametrize("bad_name", TRAVERSAL_NAMES)
def test_traversal_name_on_analyze_brain_writes_nothing_outside_tmp(client, brain_dcm, bad_name):
    before = _leftover_request_dirs()
    r = client.post("/analyze/brain", files=[("files", (bad_name, brain_dcm, "application/dicom"))])
    assert r.status_code in (200, 400, 422), r.text
    assert _find("pwn.dcm", _search_roots()) == []
    assert _leftover_request_dirs() == before          # per-request dir cleaned up


@pytest.mark.parametrize("bad_name", TRAVERSAL_NAMES[:2])
def test_traversal_name_on_analyze_study_writes_nothing_outside_tmp(client, brain_dcm, bad_name):
    before = _leftover_request_dirs()
    r = client.post("/analyze/study", files=[("files", (bad_name, brain_dcm, "application/dicom"))])
    assert r.status_code in (200, 400, 422), r.text
    assert _find("pwn.dcm", _search_roots()) == []
    assert _leftover_request_dirs() == before


def test_garbage_payload_with_traversal_name_is_rejected_and_cleaned(client):
    before = _leftover_request_dirs()
    junk = b"definitely not dicom" * 64
    for route in ("/analyze/study", "/analyze/brain"):
        r = client.post(route, files=[("files", ("../../pwn.dcm", junk, "application/dicom"))])
        assert r.status_code in (400, 422), (route, r.text)
    assert _find("pwn.dcm", _search_roots()) == []
    assert _leftover_request_dirs() == before


def test_legacy_single_file_route_cleans_temp(client, brain_dcm):
    tmp = Path(tempfile.gettempdir())
    before = {p.name for p in tmp.glob("*.dcm")}
    r = client.post("/analyze", files=[("file", ("../../pwn.dcm", brain_dcm, "application/dicom"))])
    assert r.status_code in (200, 400, 422, 503), r.text
    assert {p.name for p in tmp.glob("*.dcm")} == before
    assert _find("pwn.dcm", _search_roots()) == []
