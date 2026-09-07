"""Auth-ON probe — run in a SUBPROCESS by tests/test_auth.py.

Auth flags (SENTINEL_DEV_INSECURE / SENTINEL_REQUIRE_AUTH) are read when
src.utils.auth is imported, so the strict default posture cannot be tested
in the same interpreter as the dev-insecure suite. This script boots the app
under TestClient with whatever env the caller prepared and prints one JSON
line prefixed with __RESULT__.
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    from fastapi.testclient import TestClient
    from src.inference.server import app

    study_dir = Path(os.environ.get("SENTINEL_TEST_STUDY_DIR", ""))
    parts = [("files", (p.name, p.read_bytes(), "application/dicom"))
             for p in (sorted(study_dir.rglob("*.dcm")) if study_dir.is_dir() else [])]
    dummy = [("files", ("a.dcm", b"\x00" * 16, "application/dicom"))]
    out: dict = {}
    with TestClient(app) as c:
        h = c.get("/health")
        out["health"] = {"status": h.status_code, "auth_required": h.json().get("auth_required")}
        out["docs"] = c.get("/docs").status_code
        out["openapi"] = c.get("/openapi.json").status_code

        r = c.post("/analyze/study", files=dummy)
        out["analyze_no_token"] = {"status": r.status_code, "detail": r.json().get("detail")}
        out["studies_no_token"] = c.get("/studies").status_code
        out["study_no_token"] = c.get("/study/x").status_code
        out["sign_no_token"] = c.post("/report/sign", json={"study_id": "x", "report_text": "t"}).status_code
        out["correction_no_token"] = c.post("/report/save_correction", json={
            "study_id": "x", "original_report": "a", "corrected_report": "b"}).status_code

        out["login_wrong"] = c.post("/auth/login", json={"username": "admin", "password": "wrong-pw"}).status_code
        out["login_unknown_user"] = c.post("/auth/login", json={"username": "nobody", "password": "x"}).status_code
        r = c.post("/auth/login", json={"username": "admin",
                                        "password": os.environ["SENTINEL_ADMIN_PASSWORD"]})
        body = r.json() if r.status_code == 200 else {}
        out["login"] = {
            "status": r.status_code,
            "keys": sorted(body.keys()),
            "token_type": body.get("token_type"),
            "user_keys": sorted((body.get("user") or {}).keys()),
            "role": (body.get("user") or {}).get("role"),
            "must_change_password": body.get("must_change_password"),
        }
        token = body.get("access_token", "")
        bearer = {"Authorization": f"Bearer {token}"}
        out["me_bearer"] = c.get("/auth/me", headers=bearer).status_code
        out["studies_bearer"] = c.get("/studies", headers=bearer).status_code
        out["bogus_token"] = c.get("/studies", headers={"Authorization": "Bearer not.a.jwt"}).status_code
        out["basic_scheme"] = c.get("/studies", headers={"Authorization": "Basic abc"}).status_code
        if parts:
            r = c.post("/analyze/study", files=parts, headers=bearer)
            out["analyze_bearer"] = {"status": r.status_code,
                                     "findings": len(r.json().get("findings", [])) if r.status_code == 200 else None,
                                     "persisted": r.json().get("persisted") if r.status_code == 200 else None}
        else:
            out["analyze_bearer"] = None
    print("__RESULT__" + json.dumps(out))


if __name__ == "__main__":
    main()
