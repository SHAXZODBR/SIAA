"""PACS integration: Encapsulated PDF export, the Orthanc auto-ingest watcher
(against an in-process stub Orthanc + stub inference server) and
POST /report/{id}/pdf on the real app."""

import base64
import hashlib
import json
import os
import re
import stat
from io import BytesIO
from pathlib import Path

import pydicom
import pytest
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.testclient import TestClient
from pydicom.uid import ExplicitVRLittleEndian

from conftest import REPO, SCRATCH_DATA_DIR
from src.inference.orthanc_watcher import (
    OrthancWatcher, WatcherConfig, WatcherError, load_state, main as watcher_main, read_state,
)
from src.utils import dicom_export
from src.utils.dicom_export import (
    ENCAPSULATED_PDF_SOP_CLASS, SERIES_DESCRIPTION, build_encapsulated_pdf, dataset_to_bytes,
    extract_pdf,
)

MINI_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
)
META = {
    "patient_id": "P-001", "patient_name": "Ivanov^Ivan", "patient_sex": "M",
    "patient_birth_date": "1980-02-03", "study_instance_uid": "1.2.826.0.1.3680043.8.498.1234",
    "accession_number": "ACC42", "study_date": "20260913", "modality": "MR",
    "referring_physician": "Petrov^Petr",
}


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


# ─── (a) Encapsulated PDF round-trip ─────────────────────────────────────────

@pytest.mark.parametrize("pad", [b"", b"x"])       # even and odd PDF lengths
def test_encapsulated_pdf_roundtrip(pad):
    pdf = MINI_PDF + pad
    ds = build_encapsulated_pdf(pdf, META, "Dr Karimova", "2026-09-13T10:11:12")
    raw = dataset_to_bytes(ds)
    assert raw[128:132] == b"DICM"
    back = pydicom.dcmread(BytesIO(raw))
    assert back.SOPClassUID == ENCAPSULATED_PDF_SOP_CLASS
    assert back.file_meta.MediaStorageSOPClassUID == ENCAPSULATED_PDF_SOP_CLASS
    assert back.file_meta.MediaStorageSOPInstanceUID == back.SOPInstanceUID == ds.SOPInstanceUID
    assert back.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert back.PatientID == "P-001"
    assert str(back.PatientName) == "Ivanov^Ivan"
    assert back.PatientBirthDate == "19800203"
    assert back.PatientSex == "M"
    assert back.StudyInstanceUID == META["study_instance_uid"]
    assert back.AccessionNumber == "ACC42"
    assert back.StudyDate == "20260913"
    assert back.Modality == "DOC"
    assert back.SeriesDescription == SERIES_DESCRIPTION
    assert back.MIMETypeOfEncapsulatedDocument == "application/pdf"
    assert "Dr Karimova" in back.DocumentTitle
    assert back.ContentDate == "20260913" and back.ContentTime == "101112"
    assert len(bytes(back.EncapsulatedDocument)) % 2 == 0
    assert extract_pdf(back) == pdf
    assert bytes(back.EncapsulatedDocument)[: len(pdf)] == pdf
    assert back.EncapsulatedDocumentLength == len(pdf)
    assert back.SpecificCharacterSet == "ISO_IR 192"


def test_encapsulated_pdf_new_uids_each_call_and_rejects_non_pdf():
    a = build_encapsulated_pdf(MINI_PDF, META, "s", "2026-09-13T00:00:00")
    b = build_encapsulated_pdf(MINI_PDF, META, "s", "2026-09-13T00:00:00")
    assert a.SOPInstanceUID != b.SOPInstanceUID
    assert a.SeriesInstanceUID != b.SeriesInstanceUID
    assert a.StudyInstanceUID == b.StudyInstanceUID == META["study_instance_uid"]
    assert a.is_little_endian is True and a.is_implicit_VR is False
    with pytest.raises(ValueError):
        build_encapsulated_pdf(b"not a pdf", META, "s", "2026-09-13T00:00:00")
    # Cyrillic names survive the UTF-8 round trip
    ds = build_encapsulated_pdf(MINI_PDF, {**META, "patient_name": "Каримова^Дилноза"}, "Врач", "2026-09-13T00:00:00")
    back = pydicom.dcmread(BytesIO(dataset_to_bytes(ds)))
    assert str(back.PatientName) == "Каримова^Дилноза"


# ─── (b) Watcher against a stubbed Orthanc + inference server ────────────────

ORTHANC_AUTH = "Basic " + base64.b64encode(b"sentinel:pw").decode()


class StubOrthanc:
    """Minimal Orthanc REST surface the watcher touches, with basic-auth."""

    def __init__(self):
        self.app = FastAPI()
        self.changes: list[dict] = []
        self.studies: dict[str, dict] = {}       # orthanc id -> {uid, instances: {iid: bytes}}
        self.calls: dict[str, int] = {"changes": 0, "files": 0}
        app = self.app

        def _auth(request: Request):
            if request.headers.get("authorization") != ORTHANC_AUTH:
                raise HTTPException(status_code=401, detail="unauthorized")

        @app.get("/system")
        def system(request: Request):
            _auth(request)
            return {"Name": "StubPACS", "Version": "1.12"}

        @app.get("/statistics")
        def statistics(request: Request):
            _auth(request)
            return {"CountStudies": len(self.studies)}

        @app.get("/changes")
        def changes(request: Request, since: int = 0, limit: int = 100):
            _auth(request)
            self.calls["changes"] += 1
            items = [c for c in self.changes if c["Seq"] > since][:limit]
            last = items[-1]["Seq"] if items else since
            done = not [c for c in self.changes if c["Seq"] > last]
            return {"Changes": items, "Done": done, "Last": last}

        @app.get("/studies/{sid}")
        def study(sid: str, request: Request):
            _auth(request)
            if sid not in self.studies:
                raise HTTPException(status_code=404)
            s = self.studies[sid]
            return {"ID": sid, "Type": "Study", "MainDicomTags": {"StudyInstanceUID": s["uid"]},
                    "PatientMainDicomTags": {"PatientID": "SHOULD-NOT-BE-LOGGED", "PatientName": "Secret^Name"},
                    "Instances": list(s["instances"].keys())}

        @app.get("/studies/{sid}/instances")
        def instances(sid: str, request: Request):
            _auth(request)
            s = self.studies[sid]
            return [{"ID": iid, "Type": "Instance", "IndexInSeries": i}
                    for i, iid in enumerate(s["instances"])]

        @app.get("/instances/{iid}/file")
        def instance_file(iid: str, request: Request):
            _auth(request)
            self.calls["files"] += 1
            for s in self.studies.values():
                if iid in s["instances"]:
                    return Response(content=s["instances"][iid], media_type="application/dicom")
            raise HTTPException(status_code=404)

    def add_study(self, sid: str, uid: str, n_files: int = 3) -> None:
        self.studies[sid] = {"uid": uid, "instances": {
            f"{sid}-inst-{i}": b"DICM-fake-" + f"{sid}-{i}".encode() * (i + 1) for i in range(n_files)}}

    def emit(self, change_type: str, sid: str) -> int:
        seq = (self.changes[-1]["Seq"] + 1) if self.changes else 1
        self.changes.append({"Seq": seq, "ChangeType": change_type, "ID": sid,
                             "Path": f"/studies/{sid}", "ResourceType": "Study", "Date": "20260913T100000"})
        return seq

    def emit_arrival(self, sid: str, uid: str, n_files: int = 3) -> int:
        """The sequence Orthanc produces for a study arriving from a scanner."""
        self.add_study(sid, uid, n_files)
        self.emit("NewPatient", sid)
        self.emit("NewStudy", sid)
        self.emit("NewSeries", sid)
        self.emit("NewInstance", sid)
        stable = self.emit("StableStudy", sid)
        self.emit("StablePatient", sid)
        return stable


class StubInference:
    def __init__(self):
        self.app = FastAPI()
        self.analyze_calls: list[dict] = []
        self.logins = 0
        self.status = 200                 # what /analyze/study answers next
        self.valid_tokens: set[str] = set()
        self.reject_token: str | None = None
        app = self.app

        @app.post("/auth/login")
        def login(body: dict):
            if body != {"username": "svc", "password": "svc-pw"}:
                raise HTTPException(status_code=401)
            self.logins += 1
            tok = f"tok-{self.logins}"
            self.valid_tokens.add(tok)
            return {"access_token": tok, "token_type": "bearer",
                    "user": {"id": "u1", "username": "svc", "full_name": "svc", "role": "radiologist"},
                    "must_change_password": False}

        @app.post("/analyze/study")
        async def analyze(request: Request, files: list[UploadFile] = File(...), language: str = "ru"):
            auth = request.headers.get("authorization", "")
            tok = auth.removeprefix("Bearer ")
            if tok not in self.valid_tokens or tok == self.reject_token:
                raise HTTPException(status_code=401, detail="Authentication required")
            blobs = [await f.read() for f in files]
            self.analyze_calls.append({"n": len(blobs), "language": language, "token": tok,
                                       "sha": sorted(hashlib.sha256(b).hexdigest() for b in blobs)})
            if self.status == 200:
                return {"study_id": f"study-{len(self.analyze_calls)}", "route": "brain_panel",
                        "findings": [{"class_name": "abnormal", "confidence": 0.9}], "persisted": True}
            if self.status == 422:
                return Response(status_code=422, media_type="application/json",
                                content=json.dumps({"detail": "no model", "requires_review": True,
                                                    "reason": "no model", "study_id": "s-422"}))
            raise HTTPException(status_code=self.status, detail="boom")


@pytest.fixture
def pacs(tmp_path):
    orthanc, inference = StubOrthanc(), StubInference()
    cfg = WatcherConfig(
        orthanc_url="http://orthanc.test:8042", orthanc_user="sentinel", orthanc_password="pw",
        inference_url="http://inference.test:8000", service_user="svc", service_password="svc-pw",
        poll_seconds=1, language="uz", state_path=tmp_path / "state" / "orthanc_watcher_state.json",
    )

    def make_watcher(**over):
        c = WatcherConfig(**{**cfg.__dict__, **over})
        return OrthancWatcher(c, orthanc_http=TestClient(orthanc.app), inference_http=TestClient(inference.app))

    return {"orthanc": orthanc, "inference": inference, "cfg": cfg, "make": make_watcher}


def test_watcher_ingests_stable_study_and_persists_state(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    stable_seq = orthanc.emit_arrival("s1", "1.2.3.4.5.1", n_files=3)
    w = pacs["make"]()
    assert w.check_connection() is True

    assert w.poll_once() == 1
    assert len(inf.analyze_calls) == 1
    call = inf.analyze_calls[0]
    assert call["n"] == 3 and call["language"] == "uz" and call["token"] == "tok-1"
    expected = sorted(hashlib.sha256(b).hexdigest() for b in orthanc.studies["s1"]["instances"].values())
    assert call["sha"] == expected                       # every file, byte-exact
    assert orthanc.calls["files"] == 3

    st_path = pacs["cfg"].state_path
    assert st_path.exists() and _mode(st_path) == 0o600
    st = json.loads(st_path.read_text())
    assert st["last_seq"] == stable_seq + 1              # StablePatient after it was consumed too
    assert st["processed"]["1.2.3.4.5.1"]["status"] == "analyzed"
    assert st["processed"]["1.2.3.4.5.1"]["study_id"] == "study-1"
    assert st["counters"]["studies_processed"] == 1
    assert "SHOULD-NOT-BE-LOGGED" not in st_path.read_text() and "Secret" not in st_path.read_text()
    assert not list(Path(os.environ.get("TMPDIR", "/tmp")).glob("sentinel_pacs_*"))

    # second run: nothing new
    assert w.poll_once() == 0
    assert len(inf.analyze_calls) == 1 and orthanc.calls["files"] == 3
    assert load_state(st_path).last_seq == stable_seq + 1

    # the same study re-stabilises (e.g. late instance) -> skipped by UID, seq advances
    seq2 = orthanc.emit("StableStudy", "s1")
    assert w.poll_once() == 0
    assert len(inf.analyze_calls) == 1
    st = load_state(st_path)
    assert st.last_seq == seq2 and st.counters["studies_skipped"] == 1

    # a new study is picked up; a fresh watcher from the same state file re-runs nothing
    orthanc.emit_arrival("s2", "1.2.3.4.5.2", n_files=2)
    assert w.poll_once() == 1 and len(inf.analyze_calls) == 2 and inf.analyze_calls[1]["n"] == 2
    w2 = pacs["make"]()
    assert w2.poll_once() == 0 and len(inf.analyze_calls) == 2
    assert w2.known_orthanc_ids == {"1.2.3.4.5.1", "1.2.3.4.5.2"}

    status = read_state(st_path)
    assert status["processed_count"] == 2 and "processed" not in status
    assert status["counters"]["studies_processed"] == 2


def test_watcher_only_acts_on_stable_study(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    orthanc.add_study("s1", "1.2.3.4.5.9")
    orthanc.emit("NewStudy", "s1")
    orthanc.emit("NewInstance", "s1")
    w = pacs["make"]()
    assert w.poll_once() == 0 and inf.analyze_calls == []
    assert load_state(pacs["cfg"].state_path).last_seq == 2
    orthanc.emit("StableStudy", "s1")
    assert w.poll_once() == 1 and len(inf.analyze_calls) == 1


def test_watcher_transient_failure_backs_off_then_retries(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    stable_seq = orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    inf.status = 503
    w = pacs["make"]()
    with pytest.raises(WatcherError):
        w.poll_once()
    st = load_state(pacs["cfg"].state_path)
    assert st.last_seq == stable_seq - 1                 # the failing change will be re-read
    assert st.counters["consecutive_errors"] == 1 and st.last_error
    assert st.attempts["1.2.3.4.5.1"]["count"] == 1
    assert "1.2.3.4.5.1" not in st.processed
    assert w.backoff_seconds() > w.config.poll_seconds

    inf.status = 200
    assert w.poll_once() == 1
    st = load_state(pacs["cfg"].state_path)
    assert st.processed["1.2.3.4.5.1"]["status"] == "analyzed" and st.attempts == {}
    assert st.counters["consecutive_errors"] == 0 and st.last_error is None
    assert w.backoff_seconds() == w.config.poll_seconds


def test_watcher_quarantines_a_study_that_keeps_failing(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    orthanc.emit_arrival("s2", "1.2.3.4.5.2")
    inf.status = 500
    w = pacs["make"](max_attempts=2)
    with pytest.raises(WatcherError):                    # s1 attempt 1 -> transient
        w.poll_once()
    assert len(inf.analyze_calls) == 1
    with pytest.raises(WatcherError):                    # s1 attempt 2 -> quarantined; s2 attempt 1 -> transient
        w.poll_once()
    assert len(inf.analyze_calls) == 3
    st = load_state(pacs["cfg"].state_path)
    assert st.processed["1.2.3.4.5.1"]["status"] == "failed"
    assert "1.2.3.4.5.2" not in st.processed and st.attempts["1.2.3.4.5.2"]["count"] == 1
    assert st.counters["studies_failed"] == 1
    inf.status = 200
    assert w.poll_once() == 1                            # s2 recovers; s1 stays quarantined
    assert len(inf.analyze_calls) == 4
    st = load_state(pacs["cfg"].state_path)
    assert st.processed["1.2.3.4.5.2"]["status"] == "analyzed" and st.attempts == {}
    assert w.poll_once() == 0 and len(inf.analyze_calls) == 4


def test_watcher_requires_review_and_rejections_are_terminal(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    inf.status = 422
    w = pacs["make"]()
    assert w.poll_once() == 1
    st = load_state(pacs["cfg"].state_path)
    assert st.processed["1.2.3.4.5.1"]["status"] == "requires_review"
    orthanc.emit_arrival("s2", "1.2.3.4.5.2")
    inf.status = 400
    assert w.poll_once() == 0
    st = load_state(pacs["cfg"].state_path)
    assert st.processed["1.2.3.4.5.2"]["status"] == "rejected"
    assert w.poll_once() == 0 and len(inf.analyze_calls) == 2   # neither is retried


def test_watcher_refreshes_expired_token(pacs):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    w = pacs["make"]()
    w.inference.login()                                  # tok-1
    inf.reject_token = "tok-1"                           # server-side: expired
    assert w.poll_once() == 1
    assert inf.logins == 2 and inf.analyze_calls[-1]["token"] == "tok-2"


def test_watcher_survives_orthanc_outage_and_bad_credentials(pacs, tmp_path):
    orthanc = pacs["orthanc"]
    orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    w = pacs["make"](orthanc_password="wrong")
    assert w.check_connection() is False
    with pytest.raises(WatcherError, match="401"):
        w.poll_once()
    assert w.consecutive_errors == 1
    # run_forever must not propagate: stop after the first back-off wait
    import threading
    stop = threading.Event()
    w.config.poll_seconds = 0.01
    w.config.max_backoff_seconds = 0.02
    t = threading.Thread(target=w.run_forever, args=(stop,), daemon=True)
    t.start()
    import time
    time.sleep(0.15)
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive() and w.consecutive_errors >= 2


def test_watcher_once_flag_entry_point(pacs, monkeypatch):
    orthanc, inf = pacs["orthanc"], pacs["inference"]
    orthanc.emit_arrival("s1", "1.2.3.4.5.1")
    monkeypatch.setenv("ORTHANC_URL", "http://orthanc.test:8042")
    monkeypatch.setenv("ORTHANC_USER", "sentinel")
    monkeypatch.setenv("ORTHANC_PASSWORD", "pw")
    monkeypatch.setenv("INFERENCE_URL", "http://inference.test:8000")
    monkeypatch.setenv("SENTINEL_SERVICE_USER", "svc")
    monkeypatch.setenv("SENTINEL_SERVICE_PASSWORD", "svc-pw")
    monkeypatch.setenv("SENTINEL_DEFAULT_LANGUAGE", "en")
    st_path = pacs["cfg"].state_path
    rc = watcher_main(["--once", "--state", str(st_path)],
                      orthanc_http=TestClient(orthanc.app), inference_http=TestClient(inf.app))
    assert rc == 0
    assert len(inf.analyze_calls) == 1 and inf.analyze_calls[0]["language"] == "en"
    assert load_state(st_path).processed["1.2.3.4.5.1"]["status"] == "analyzed"
    inf.status = 503
    orthanc.emit_arrival("s2", "1.2.3.4.5.2")
    assert watcher_main(["--once", "--state", str(st_path)],
                        orthanc_http=TestClient(orthanc.app), inference_http=TestClient(inf.app)) == 1


def test_watcher_config_from_env_defaults(monkeypatch):
    for k in ("ORTHANC_URL", "ORTHANC_USER", "ORTHANC_PASSWORD", "INFERENCE_URL",
              "SENTINEL_SERVICE_USER", "SENTINEL_SERVICE_PASSWORD", "POLL_SECONDS", "SENTINEL_DEFAULT_LANGUAGE"):
        monkeypatch.delenv(k, raising=False)
    cfg = WatcherConfig.from_env()
    assert cfg.orthanc_url == "http://127.0.0.1:8042" and cfg.inference_url == "http://127.0.0.1:8000"
    assert cfg.poll_seconds == 15 and cfg.language == "ru" and cfg.orthanc_auth is None
    assert cfg.state_path == SCRATCH_DATA_DIR / "orthanc_watcher_state.json"
    monkeypatch.setenv("POLL_SECONDS", "2")
    monkeypatch.setenv("ORTHANC_USER", "u")
    assert WatcherConfig.from_env().poll_seconds == 2
    assert WatcherConfig.from_env().orthanc_auth == ("u", "")


# ─── configs/orthanc.json: no baked-in credentials, localhost-only REST ───────

def test_orthanc_config_has_no_default_credentials():
    text = (REPO / "configs" / "orthanc.json").read_text(encoding="utf-8")
    stripped = re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)      # Orthanc allows // comments
    cfg = json.loads(stripped)
    assert "RegisteredUsers" not in cfg
    assert cfg["AuthenticationEnabled"] is True
    assert cfg["RemoteAccessAllowed"] is False
    assert "sentinel2024" not in text
    assert "ORTHANC_USER" in text and "ORTHANC__REGISTERED_USERS" in text   # documented instead


# ─── (c) POST /report/{id}/pdf on the real app (dev-insecure, scratch DATA_DIR)

@pytest.fixture(scope="module")
def pacs_study(client, study_files) -> dict:
    """This module's own analysed study - the shared `analyzed_study` fixture is
    left untouched because tests/test_sign.py counts its report rows."""
    r = client.post("/analyze/study", files=study_files)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def signed_uz_report(client, pacs_study) -> dict:
    r = client.post("/report/sign", json={
        "study_id": pacs_study["study_id"], "report_text": "Xulosa: o'zgarishlar aniqlandi.",
        "language": "uz"})
    assert r.status_code == 200, r.text
    return r.json()


def test_attach_pdf_to_signed_report(client, pacs_study, signed_uz_report):
    rid = signed_uz_report["report_id"]
    r = client.post(f"/report/{rid}/pdf", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report_id"] == rid and body["study_id"] == pacs_study["study_id"]
    assert body["pdf_sha256"] == hashlib.sha256(MINI_PDF).hexdigest()
    assert body["dicom_sop_instance_uid"] and body["pushed"] is False and body["orthanc_id"] is None

    pdf_path = SCRATCH_DATA_DIR / "reports" / f"{rid}.pdf"
    dcm_path = SCRATCH_DATA_DIR / "reports" / f"{rid}.dcm"
    assert pdf_path.read_bytes() == MINI_PDF and _mode(pdf_path) == 0o600
    assert dcm_path.exists() and _mode(dcm_path) == 0o600
    ds = pydicom.dcmread(str(dcm_path))
    assert ds.SOPClassUID == ENCAPSULATED_PDF_SOP_CLASS
    assert ds.SOPInstanceUID == body["dicom_sop_instance_uid"]
    assert ds.StudyInstanceUID == pacs_study["study_instance_uid"]
    assert ds.PatientID == pacs_study["patient_id"]
    assert ds.Modality == "DOC" and ds.SeriesDescription == SERIES_DESCRIPTION
    assert extract_pdf(ds) == MINI_PDF

    # recorded on the report row
    reports = client.get(f"/study/{pacs_study['study_id']}").json()["reports"]
    row = next(x for x in reports if x["report_id"] == rid)
    assert row["pdf_sha256"] == body["pdf_sha256"]
    assert row["dicom_sop_instance_uid"] == body["dicom_sop_instance_uid"]

    # identical re-upload is idempotent; a different PDF is refused
    again = client.post(f"/report/{rid}/pdf", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert again.status_code == 200 and again.json()["dicom_sop_instance_uid"] == body["dicom_sop_instance_uid"]
    other = client.post(f"/report/{rid}/pdf", files=[("pdf", ("r2.pdf", MINI_PDF + b"\n%new", "application/pdf"))])
    assert other.status_code == 409


def test_attach_pdf_push_without_orthanc_configured(client, signed_uz_report, monkeypatch):
    monkeypatch.delenv("ORTHANC_URL", raising=False)
    monkeypatch.delenv("SENTINEL_ORTHANC", raising=False)
    rid = signed_uz_report["report_id"]
    r = client.post(f"/report/{rid}/pdf?push=1", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r.status_code == 200, r.text
    assert r.json()["pushed"] is False and r.json()["orthanc_id"] is None
    assert "not configured" in r.json()["push_error"]


def test_attach_pdf_push_to_mocked_orthanc(client, signed_uz_report, monkeypatch):
    rid = signed_uz_report["report_id"]
    monkeypatch.setenv("ORTHANC_URL", "http://pacs.local:8042/")
    monkeypatch.setenv("ORTHANC_USER", "sentinel")
    monkeypatch.setenv("ORTHANC_PASSWORD", "pw")
    seen = []

    def fake_push(dicom_bytes, url, auth=None, **kw):
        seen.append((dicom_bytes, url, auth))
        return {"orthanc_id": "abc123", "parent_study": "ps", "parent_series": "se", "status": "Success"}

    monkeypatch.setattr(dicom_export, "push_to_orthanc", fake_push)
    r = client.post(f"/report/{rid}/pdf?push=1", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pushed"] is True and body["orthanc_id"] == "abc123" and body["push_error"] is None
    assert len(seen) == 1
    dicom_bytes, url, auth = seen[0]
    assert url == "http://pacs.local:8042" and auth == ("sentinel", "pw")
    ds = pydicom.dcmread(BytesIO(dicom_bytes))
    assert ds.SOPInstanceUID == body["dicom_sop_instance_uid"] and extract_pdf(ds) == MINI_PDF

    # already pushed: reported as pushed, not sent twice
    r2 = client.post(f"/report/{rid}/pdf?push=1", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r2.status_code == 200 and r2.json()["pushed"] is True and r2.json()["orthanc_id"] == "abc123"
    assert len(seen) == 1

    # a push failure is reported, not a 5xx (the PDF/DICOM are stored; retry later)
    from src.inference.auth_routes import db
    db.record_report_push(rid, "")          # clear so the next push is attempted

    def failing_push(*a, **kw):
        raise RuntimeError("Orthanc unreachable: ConnectionError")

    monkeypatch.setattr(dicom_export, "push_to_orthanc", failing_push)
    r3 = client.post(f"/report/{rid}/pdf?push=1", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r3.status_code == 200 and r3.json()["pushed"] is False
    assert "unreachable" in r3.json()["push_error"]


def test_attach_pdf_unsigned_report_409(client, pacs_study):
    from src.inference.auth_routes import db
    rid = db.create_report(pacs_study["study_id"], doctor_id=None, ai_draft_text="draft", language="uz")
    r = client.post(f"/report/{rid}/pdf", files=[("pdf", ("report.pdf", MINI_PDF, "application/pdf"))])
    assert r.status_code == 409, r.text
    assert not (SCRATCH_DATA_DIR / "reports" / f"{rid}.pdf").exists()


def test_attach_pdf_bad_inputs(client, signed_uz_report):
    rid = signed_uz_report["report_id"]
    assert client.post("/report/no-such-report/pdf",
                       files=[("pdf", ("r.pdf", MINI_PDF, "application/pdf"))]).status_code == 404
    assert client.post(f"/report/{rid}/pdf",
                       files=[("pdf", ("r.pdf", b"<html>not a pdf</html>", "application/pdf"))]).status_code == 400
    assert client.post(f"/report/{rid}/pdf",
                       files=[("pdf", ("r.pdf", b"", "application/pdf"))]).status_code == 400
    assert client.post(f"/report/{rid}/pdf").status_code == 422          # multipart field missing


def test_pacs_status(client, monkeypatch):
    monkeypatch.delenv("ORTHANC_URL", raising=False)
    monkeypatch.delenv("SENTINEL_ORTHANC", raising=False)
    r = client.get("/pacs/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is False and body["in_process_watcher"] is False
    assert body["state_file"] == str(SCRATCH_DATA_DIR / "orthanc_watcher_state.json")
    assert "watcher" in body and "orthanc_auth_set" in body
    assert "ORTHANC_PASSWORD" not in r.text
    monkeypatch.setenv("ORTHANC_URL", "http://pacs.local:8042")
    assert client.get("/pacs/status").json()["configured"] is True
