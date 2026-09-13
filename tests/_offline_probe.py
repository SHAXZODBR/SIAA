"""Air-gapped probe — run in a SUBPROCESS by tests/test_offline.py.

SENTINEL_OFFLINE / HF_HUB_OFFLINE are read at import time, and the dev-insecure
suite already holds a live app in-process, so the hospital posture is exercised
here with the env the test prepared: offline flags on, the HF cache pointed at
an EMPTY dir (a hospital PC has none) and every non-loopback socket connect
refused — any attempted download fails loudly instead of hanging.
Prints one JSON line prefixed with __RESULT__.
"""

import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BLOCKED: list[str] = []
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _install_network_guard() -> None:
    orig_connect = socket.socket.connect
    orig_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, addr, *a, **kw):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if str(host) not in _LOOPBACK and not str(host).startswith("/"):
            BLOCKED.append(str(host))
            raise OSError(f"network blocked by offline probe: {host}")
        return orig_connect(self, addr, *a, **kw)

    def guarded_getaddrinfo(host, *a, **kw):
        if host not in _LOOPBACK and host is not None:
            BLOCKED.append(str(host))
            raise socket.gaierror(f"DNS blocked by offline probe: {host}")
        return orig_getaddrinfo(host, *a, **kw)

    socket.socket.connect = guarded_connect
    socket.getaddrinfo = guarded_getaddrinfo


def main() -> None:
    _install_network_guard()
    from fastapi.testclient import TestClient
    from src.inference.server import app
    from src.inference.model_registry import get_model, get_loaded_status, REGISTRY
    from src.utils.offline import OFFLINE

    study_dir = Path(os.environ.get("SENTINEL_TEST_STUDY_DIR", ""))
    parts = [("files", (p.name, p.read_bytes(), "application/dicom"))
             for p in (sorted(study_dir.rglob("*.dcm")) if study_dir.is_dir() else [])]
    out: dict = {"offline_flag": OFFLINE,
                 "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "SENTINEL_OFFLINE")}}
    with TestClient(app) as c:
        h = c.get("/health")
        out["health"] = {"http_status": h.status_code, **{k: h.json().get(k) for k in ("status", "offline", "models")}}

        if parts:
            r = c.post("/analyze/study", files=parts)
            body = r.json() if r.status_code == 200 else {"detail": r.text[:500]}
            out["analyze"] = {
                "status": r.status_code,
                "route": body.get("route"),
                "triage": [f for f in body.get("findings", []) if f.get("detector") == "triage"],
                "detectors_run": body.get("detectors_run"),
                "model_identity": body.get("model_identity"),
            }
        else:
            out["analyze"] = None

        # Keys whose weights are NOT on this box (HF cache hidden, no bundle):
        # must degrade to loaded=False + an actionable reason, never raise.
        absent = {}
        for key in os.environ.get("SENTINEL_PROBE_ABSENT_KEYS", "brain_stroke,head_ct,chest_tb").split(","):
            key = key.strip()
            if not key or key not in REGISTRY:
                continue
            try:
                e = get_model(key, device="cpu")
                absent[key] = {"raised": None, "available": bool(e and e.get("available")),
                               "reason": (e or {}).get("reason", "")}
            except Exception as ex:   # pragma: no cover — this is the failure we test for
                absent[key] = {"raised": f"{type(ex).__name__}: {ex}", "available": None, "reason": ""}
        out["absent"] = absent
        out["loaded_status"] = get_loaded_status(list(absent) + ["brain_triage", "brain_tumor_class"])
        out["health_after"] = c.get("/health").json().get("models")
    out["blocked_hosts"] = sorted(set(BLOCKED))
    print("__RESULT__" + json.dumps(out))


if __name__ == "__main__":
    main()
