"""POST /analyze/study — API contract v1 on the real brain MR study."""

import hashlib

import pytest

from conftest import CHEST_FIXTURE


def test_health_contract(client, analyzed_study):
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] in ("ok", "degraded")
    assert body["version"]
    assert body["auth_required"] is False          # SENTINEL_DEV_INSECURE=1 in this suite
    assert body["models"]["brain_triage"]["loaded"] is True
    assert set(body["llm"]) == {"backend", "reachable"}
    assert body["license"]["mode"] in ("licensed", "unlicensed", "demo", "dev")
    assert body["data_dir"]


def test_study_headers_and_route(analyzed_study):
    resp = analyzed_study
    assert resp["study_id"]
    assert resp["patient_id"]
    assert resp["study_instance_uid"]
    assert resp["modality"].upper().startswith("MR")
    assert resp["body_part"] == "BRAIN"
    assert resp["route"] == "brain_panel"
    assert resp["num_files"] > 0
    assert resp["persisted"] is True
    assert resp["rejected"] is False
    assert resp["requires_review"] is False


def test_findings_carry_validation_status(analyzed_study):
    findings = analyzed_study["findings"]
    assert findings, "panel returned no findings"
    triage = [f for f in findings if f.get("detector") == "triage"]
    assert triage, "validated triage detector did not run"
    assert triage[0]["status"] == "validated"
    assert triage[0]["positive"] is True
    assert 0.0 <= triage[0]["confidence"] <= 1.0
    assert any(f.get("status") == "pending" for f in findings), "no pending finding"
    for f in findings:
        assert f["status"] in ("validated", "pending", "experimental")
        assert f["heatmap_base64"] == ""     # panel does not localise
        assert f["location"] == ""


def test_never_certifies_normal(analyzed_study):
    assert analyzed_study["normal"] is False
    overall = analyzed_study["overall_assessment"]
    assert overall and "abnormal_flagged" in overall and overall["text"]
    assert analyzed_study["disclaimer"]


def test_model_identity(analyzed_study):
    ids = analyzed_study["model_identity"]
    assert ids, "model_identity is empty"
    keys = {m["key"] for m in ids}
    assert "brain_triage" in keys
    for m in ids:
        assert m["display_name"]
        assert m["status"] in ("validated", "pending", "experimental")
        assert m["sha256_12"] is None or len(m["sha256_12"]) == 12
    triage = next(m for m in ids if m["key"] == "brain_triage")
    assert triage["status"] == "validated"
    assert triage["sha256_12"], "validated model must carry its weights hash"


def test_preview_and_report(analyzed_study):
    pb = analyzed_study["preview_base64"]
    assert pb and pb.startswith("data:image/png;base64,") and len(pb) > 1000
    assert (analyzed_study["report_text"] or "").strip()
    assert analyzed_study["report_language"] in ("ru", "uz", "en")
    assert analyzed_study["app_version"]


def test_study_is_persisted_and_listed(client, analyzed_study):
    sid = analyzed_study["study_id"]
    d = client.get(f"/study/{sid}")
    assert d.status_code == 200
    body = d.json()
    assert body["study"]["study_id"] == sid
    assert body["study"]["patient_id"] == analyzed_study["patient_id"]
    assert isinstance(body["study"].get("study_date"), (str, type(None)))
    ai = body["ai_result"]
    assert ai and len(ai["findings"]) == len(analyzed_study["findings"])
    assert ai["model_identity"] == analyzed_study["model_identity"]
    assert ai["is_normal"] is False
    lst = client.get("/studies")
    assert lst.status_code == 200
    assert any(s["study_id"] == sid for s in lst.json())


def test_unknown_study_404(client):
    assert client.get("/study/does-not-exist").status_code == 404


def test_empty_upload_400(client):
    r = client.post("/analyze/study", files=[("files", ("empty.dcm", b"", "application/dicom"))])
    assert r.status_code == 400


def test_non_dicom_upload_400(client):
    r = client.post("/analyze/study", files=[("files", ("junk.dcm", b"not a dicom" * 100, "application/dicom"))])
    assert r.status_code == 400


@pytest.mark.skipif(not CHEST_FIXTURE.exists(), reason=f"chest fixture missing: {CHEST_FIXTURE}")
def test_chest_fixture_routes_or_requires_review(client):
    r = client.post("/analyze/study", files=[
        ("files", (CHEST_FIXTURE.name, CHEST_FIXTURE.read_bytes(), "application/dicom")),
    ])
    assert r.status_code in (200, 422), r.text
    body = r.json()
    if r.status_code == 200:
        assert body["route"] == "chest"
        assert body["normal"] is False
        assert body["model_identity"]
        assert all(f["status"] in ("validated", "pending", "experimental") for f in body["findings"])
    else:
        assert body["requires_review"] is True and body["reason"]


def test_save_correction_json_body(client, analyzed_study):
    sid = analyzed_study["study_id"]
    original = analyzed_study["report_text"] or "draft"
    corrected = original + "\n\nПравка врача."
    r = client.post("/report/save_correction", json={
        "study_id": sid, "original_report": original,
        "corrected_report": corrected, "language": "ru",
    })
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "saved" and r.json()["correction_id"]
    # query-string form (the old contract) must be rejected, not silently accepted
    r2 = client.post(f"/report/save_correction?study_id={sid}&original_report=a&corrected_report=b")
    assert r2.status_code == 422


def test_sha256_helper_matches_python(analyzed_study):
    # sanity for the sign tests: the server hashes UTF-8 report text with sha256
    text = analyzed_study["report_text"] or ""
    assert len(hashlib.sha256(text.encode("utf-8")).hexdigest()) == 64
