"""Fully offline model loading — the hospital-PC posture.

Runs tests/_offline_probe.py in a subprocess with HF_HUB_OFFLINE=1 +
TRANSFORMERS_OFFLINE=1 + SENTINEL_OFFLINE=1 (+ SENTINEL_DEV_INSECURE=1, scratch
DATA_DIR), the HF cache pointed at an empty directory and every non-loopback
socket refused. The validated brain panel must still work from the local
models/ dirs (brain_triage_finetuned, brain_finetuned) and every model whose
weights are absent must report loaded=false with a reason — never raise, never
hang on a download.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPO, STUDY_DIR

PROBE = Path(__file__).with_name("_offline_probe.py")
TRIAGE_DIR = REPO / "models" / "brain_triage_finetuned"
# HF-only keys (no models/<key>_finetuned on any box): with the HF cache hidden
# their weights are absent, exactly like a hospital PC without the bundle.
ABSENT_KEYS = ["brain_stroke", "head_ct", "chest_tb", "covid_ct", "mammography", "chest_pneumonia"]

pytestmark = pytest.mark.skipif(
    not (TRIAGE_DIR / "config.json").exists(),
    reason=f"local triage model missing: {TRIAGE_DIR}")


@pytest.fixture(scope="module")
def probe() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "TRANSFORMERS_", "SENTINEL_MODEL_DIR_OVERRIDE_"))}
    scratch = Path(tempfile.mkdtemp(prefix="sentinel_test_offline_"))
    (scratch / "empty_hf_cache").mkdir()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SENTINEL_OFFLINE": "1",
        "SENTINEL_DEV_INSECURE": "1",
        "DEV_BYPASS_LICENSE": "1",
        "SENTINEL_DATA_DIR": str(scratch / "data"),
        "SENTINEL_ADMIN_PASSWORD": "Probe-Admin-Pass-2026",
        # A hospital PC has no HuggingFace cache: hide this machine's.
        "HF_HOME": str(scratch / "hf_home"),
        "HF_HUB_CACHE": str(scratch / "empty_hf_cache"),
        "SENTINEL_PROBE_ABSENT_KEYS": ",".join(ABSENT_KEYS),
    })
    env.pop("SENTINEL_REQUIRE_AUTH", None)
    if STUDY_DIR.is_dir():
        env["SENTINEL_TEST_STUDY_DIR"] = str(STUDY_DIR)
    proc = subprocess.run([sys.executable, str(PROBE)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert proc.returncode == 0 and lines, (
        f"offline probe failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
    return json.loads(lines[-1][len("__RESULT__"):])


def test_offline_flag_exported(probe):
    assert probe["offline_flag"] is True
    assert probe["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "SENTINEL_OFFLINE": "1"}


def test_health_200_with_local_models(probe):
    h = probe["health"]
    assert h["http_status"] == 200
    assert h["status"] in ("ok", "degraded")
    assert h["offline"] is True
    assert h["models"]["brain_triage"]["loaded"] is True, h["models"]
    assert h["models"]["brain_tumor_class"]["loaded"] is True, h["models"]


@pytest.mark.skipif(not STUDY_DIR.is_dir(), reason=f"brain MR study not available at {STUDY_DIR}")
def test_analyze_study_offline_returns_validated_triage(probe):
    a = probe["analyze"]
    assert a is not None and a["status"] == 200, a
    assert a["route"] == "brain_panel"
    assert "triage" in a["detectors_run"]
    assert a["triage"], "validated triage detector did not run"
    t = a["triage"][0]
    assert t["status"] == "validated" and t["positive"] is True
    ident = next(m for m in a["model_identity"] if m["key"] == "brain_triage")
    assert ident["status"] == "validated"
    assert ident["sha256_12"] == json.loads((TRIAGE_DIR / "MANIFEST.json").read_text())["files"]["model.safetensors"][:12]
    assert ident["encrypted_at_rest"] is False


def test_absent_models_degrade_with_reason(probe):
    for key in ABSENT_KEYS:
        r = probe["absent"][key]
        assert r["raised"] is None, f"{key} raised: {r['raised']}"
        assert r["available"] is False, f"{key} should not load with the HF cache hidden"
        assert "download_all_models.py" in r["reason"], r["reason"]
        assert f"--only {key}" in r["reason"]
        st = probe["loaded_status"][key]
        assert st["loaded"] is False and st["reason"] == r["reason"]
    assert probe["loaded_status"]["brain_triage"]["loaded"] is True
    assert probe["health_after"]["brain_triage"]["loaded"] is True   # still healthy afterwards


def test_no_network_attempted(probe):
    assert probe["blocked_hosts"] == [], f"offline server tried to reach: {probe['blocked_hosts']}"
