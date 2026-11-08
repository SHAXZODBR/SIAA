"""/health unavailable-reason for a missing model (src/utils/offline.missing_model_reason).

Regression for the 2026-11-09 clinic dress rehearsal: on a clinic-flavour install without the
validated model, /health told the admin to run scripts/download_all_models.py --only brain_triage
"on a machine with internet" — but the triage model is vendor-only and cannot be downloaded; it
arrives as an encrypted per-PC bundle imported via POST /models/import.
"""
from __future__ import annotations

from src.utils.offline import VENDOR_ONLY_MODEL_KEYS, missing_model_reason


def test_triage_reason_points_to_the_bundle_import_not_a_download():
    r = missing_model_reason("brain_triage", ["/x/models/brain_triage_finetuned"])
    assert r.startswith("model missing")
    assert "Import model bundle" in r and "/models/import" in r
    assert "download_all_models" not in r and "internet" not in r
    assert "(looked in: /x/models/brain_triage_finetuned)" in r
    assert "brain_triage" in VENDOR_ONLY_MODEL_KEYS


def test_public_models_keep_the_download_hint():
    r = missing_model_reason("head_ct", ["/x/models/hf/a__b"])
    assert r.startswith("model missing — run scripts/download_all_models.py --only head_ct")
    assert r.endswith("(looked in: /x/models/hf/a__b)")
    assert missing_model_reason("chest").endswith("copy models/ here")
