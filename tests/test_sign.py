"""POST /report/sign — one signed report per (study, language)."""

import hashlib


def test_sign_then_conflict(client, analyzed_study):
    sid = analyzed_study["study_id"]
    text = analyzed_study["report_text"] or "Заключение."
    lang = analyzed_study["report_language"] or "ru"
    body = {"study_id": sid, "report_text": text, "language": lang,
            "ai_draft_text": analyzed_study["report_text"]}

    r = client.post("/report/sign", json=body)
    assert r.status_code == 200, r.text
    signed = r.json()
    assert signed["study_id"] == sid
    assert signed["report_id"]
    assert signed["language"] == lang
    assert signed["signed_at"]
    assert set(signed["signer"]) == {"id", "username", "full_name"}
    assert signed["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert signed["model_identity"] == analyzed_study["model_identity"]

    again = client.post("/report/sign", json=body)
    assert again.status_code == 409, again.text
    assert signed["report_id"] in again.json()["detail"]

    # persisted + restorable
    d = client.get(f"/study/{sid}").json()
    reports = d["reports"]
    assert len(reports) == 1
    assert reports[0]["report_id"] == signed["report_id"]
    assert reports[0]["is_signed"] in (1, True)
    assert reports[0]["sha256"] == signed["sha256"]
    assert reports[0]["language"] == lang
    listed = next(s for s in client.get("/studies").json() if s["study_id"] == sid)
    assert lang in listed["signed_languages"]


def test_second_language_can_be_signed(client, analyzed_study):
    sid = analyzed_study["study_id"]
    r = client.post("/report/sign", json={"study_id": sid, "report_text": "Conclusion.", "language": "en"})
    assert r.status_code == 200, r.text
    assert client.post("/report/sign", json={"study_id": sid, "report_text": "Conclusion.", "language": "en"}).status_code == 409


def test_sign_unknown_study_404(client):
    r = client.post("/report/sign", json={"study_id": "no-such-study", "report_text": "x"})
    assert r.status_code == 404


def test_sign_empty_text_400(client, analyzed_study):
    r = client.post("/report/sign", json={"study_id": analyzed_study["study_id"], "report_text": "   "})
    assert r.status_code == 400
