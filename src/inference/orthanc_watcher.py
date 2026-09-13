"""Orthanc PACS auto-ingest - the scanner-to-AI bridge.

In a hospital the scanner sends every study to the local Orthanc PACS by
DICOM C-STORE. This watcher tails Orthanc's change log and, for every study
that has become *stable* (no new instances for StableAge seconds), pulls its
files and submits them to the Sentinel inference server exactly as the
desktop app would (POST /analyze/study). The signed report later goes back
into the same study as an Encapsulated PDF (src/utils/dicom_export.py).

Configuration (environment):
    ORTHANC_URL                 default http://127.0.0.1:8042
    ORTHANC_USER / ORTHANC_PASSWORD   HTTP basic auth for the Orthanc REST API
    INFERENCE_URL               default http://127.0.0.1:8000
    SENTINEL_SERVICE_USER / SENTINEL_SERVICE_PASSWORD
                                a Sentinel account (role radiologist/admin) used
                                to obtain a JWT via POST /auth/login. Optional
                                only when the server runs SENTINEL_DEV_INSECURE=1.
    SENTINEL_DEFAULT_LANGUAGE   report language passed to /analyze/study (ru)
    POLL_SECONDS                default 15
    ORTHANC_WATCHER_STATE       state file (default DATA_DIR/orthanc_watcher_state.json)

Semantics:
  * Only ChangeType == 'StableStudy' is acted on (Orthanc raises it once per
    study after the last instance arrived).
  * State - the last processed change sequence number and the set of processed
    StudyInstanceUIDs - is persisted atomically, so restarts neither re-run a
    study nor miss one (at-least-once delivery, idempotent by UID).
  * A transient failure (Orthanc/inference down, 5xx) stops the batch at the
    failing change and backs off exponentially; a definitive answer from the
    inference server (400 unreadable, 422 requires_review, 409 ...) is recorded
    and the study is not retried. A study that keeps failing transiently is
    quarantined after MAX_ATTEMPTS so one bad study can never block the queue.
  * Logs carry Orthanc ids, UIDs and counts - never patient names or ids.

Run as a service:  python -m src.inference.orthanc_watcher [--once]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from src.utils.paths import DATA_DIR

STATE_FILE_NAME = "orthanc_watcher_state.json"
STATE_VERSION = 1
CHANGE_TYPE = "StableStudy"
MAX_ATTEMPTS_DEFAULT = 5


class WatcherError(Exception):
    """A transient failure of one poll pass (already logged); the loop backs off."""


class AuthError(WatcherError):
    """Could not obtain / refresh the service JWT."""


def default_state_path() -> Path:
    return Path(os.environ.get("ORTHANC_WATCHER_STATE") or (DATA_DIR / STATE_FILE_NAME))


def orthanc_configured(env: Optional[dict] = None) -> bool:
    """True when the operator pointed Sentinel at a PACS (ORTHANC_URL set, or
    the in-process watcher enabled with SENTINEL_ORTHANC=1)."""
    env = os.environ if env is None else env
    return bool((env.get("ORTHANC_URL") or "").strip()) or env.get("SENTINEL_ORTHANC") == "1"


def orthanc_auth(env: Optional[dict] = None) -> Optional[tuple]:
    env = os.environ if env is None else env
    user = env.get("ORTHANC_USER")
    if user:
        return (user, env.get("ORTHANC_PASSWORD") or "")
    return None


@dataclass
class WatcherConfig:
    orthanc_url: str = "http://127.0.0.1:8042"
    orthanc_user: Optional[str] = None
    orthanc_password: Optional[str] = None
    inference_url: str = "http://127.0.0.1:8000"
    service_user: Optional[str] = None
    service_password: Optional[str] = None
    poll_seconds: float = 15.0
    language: str = "ru"
    state_path: Path = field(default_factory=default_state_path)
    changes_limit: int = 100
    request_timeout: float = 30.0
    analyze_timeout: float = 600.0
    max_backoff_seconds: float = 300.0
    max_attempts: int = MAX_ATTEMPTS_DEFAULT

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "WatcherConfig":
        env = os.environ if env is None else env

        def _f(key: str, default: float) -> float:
            try:
                return float(env.get(key) or default)
            except ValueError:
                return default

        return cls(
            orthanc_url=(env.get("ORTHANC_URL") or "http://127.0.0.1:8042").rstrip("/"),
            orthanc_user=env.get("ORTHANC_USER") or None,
            orthanc_password=env.get("ORTHANC_PASSWORD") or None,
            inference_url=(env.get("INFERENCE_URL") or "http://127.0.0.1:8000").rstrip("/"),
            service_user=env.get("SENTINEL_SERVICE_USER") or None,
            service_password=env.get("SENTINEL_SERVICE_PASSWORD") or None,
            poll_seconds=max(1.0, _f("POLL_SECONDS", 15.0)),
            language=(env.get("SENTINEL_DEFAULT_LANGUAGE") or "ru").strip().lower() or "ru",
            state_path=Path(env.get("ORTHANC_WATCHER_STATE") or (DATA_DIR / STATE_FILE_NAME)),
            max_attempts=int(_f("ORTHANC_WATCHER_MAX_ATTEMPTS", MAX_ATTEMPTS_DEFAULT)),
        )

    @property
    def orthanc_auth(self) -> Optional[tuple]:
        return (self.orthanc_user, self.orthanc_password or "") if self.orthanc_user else None


# ─── Persistent state ────────────────────────────────────────────────────────

@dataclass
class WatcherState:
    last_seq: int = 0
    processed: dict = field(default_factory=dict)      # StudyInstanceUID -> {status, at, ...}
    attempts: dict = field(default_factory=dict)       # StudyInstanceUID -> {count, last_error}
    counters: dict = field(default_factory=lambda: {
        "polls": 0, "changes_seen": 0, "studies_processed": 0, "studies_skipped": 0,
        "studies_failed": 0, "consecutive_errors": 0,
    })
    last_poll_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
    started_at: Optional[str] = None

    @property
    def processed_uids(self) -> set:
        return set(self.processed.keys())

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "last_seq": self.last_seq,
            "processed": self.processed,
            "attempts": self.attempts,
            "counters": self.counters,
            "last_poll_at": self.last_poll_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WatcherState":
        st = cls()
        st.last_seq = int(d.get("last_seq") or 0)
        proc = d.get("processed") or {}
        if isinstance(proc, list):                    # tolerate a plain list of UIDs
            proc = {u: {"status": "done"} for u in proc}
        st.processed = dict(proc)
        st.attempts = dict(d.get("attempts") or {})
        st.counters.update(d.get("counters") or {})
        st.last_poll_at = d.get("last_poll_at")
        st.last_success_at = d.get("last_success_at")
        st.last_error = d.get("last_error")
        st.started_at = d.get("started_at")
        return st


def load_state(path: Path) -> WatcherState:
    try:
        if Path(path).exists():
            return WatcherState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception as e:
        logger.warning(f"watcher state unreadable ({type(e).__name__}) - starting from seq 0: {path}")
    return WatcherState()


def save_state(path: Path, state: WatcherState) -> None:
    """Atomic write (temp + rename), owner-only permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, ensure_ascii=False, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_state(path: Optional[Path] = None) -> Optional[dict]:
    """Raw state-file contents for GET /pacs/status (None when absent).
    Only ids/counters live there - nothing patient-identifying."""
    p = Path(path) if path else default_state_path()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"state file unreadable: {type(e).__name__}", "path": str(p)}
    d["processed_count"] = len(d.get("processed") or {})
    d.pop("processed", None)          # can grow large; ids are not needed by the UI
    d["path"] = str(p)
    return d


# ─── HTTP clients (requests.Session or an httpx/TestClient - same subset) ─────

class OrthancClient:
    """Thin REST client. `http` is anything with .get/.post(url, ...) that
    returns .status_code/.json()/.content (requests.Session, httpx.Client,
    fastapi TestClient)."""

    def __init__(self, base_url: str, auth: Optional[tuple] = None, http=None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        if http is None:
            import requests
            http = requests.Session()
        self.http = http

    def _get(self, path: str, **kw):
        try:
            r = self.http.get(self.base_url + path, auth=self.auth, timeout=self.timeout, **kw)
        except Exception as e:
            raise WatcherError(f"orthanc GET {path} failed: {type(e).__name__}") from e
        if r.status_code == 401 or r.status_code == 403:
            raise WatcherError(f"orthanc GET {path}: HTTP {r.status_code} (check ORTHANC_USER/ORTHANC_PASSWORD)")
        if r.status_code // 100 != 2:
            raise WatcherError(f"orthanc GET {path}: HTTP {r.status_code}")
        return r

    def check_connection(self) -> bool:
        try:
            return self._get("/system").status_code == 200
        except WatcherError:
            return False

    def statistics(self) -> Optional[dict]:
        try:
            return self._get("/statistics").json()
        except Exception:
            return None

    def changes(self, since: int, limit: int) -> dict:
        body = self._get("/changes", params={"since": since, "limit": limit}).json()
        if not isinstance(body, dict) or "Changes" not in body:
            raise WatcherError("orthanc /changes: unexpected payload")
        return body

    def study(self, orthanc_id: str) -> dict:
        return self._get(f"/studies/{orthanc_id}").json()

    def study_instances(self, orthanc_id: str) -> list:
        body = self._get(f"/studies/{orthanc_id}/instances").json()
        ids = []
        for item in body or []:
            iid = item.get("ID") if isinstance(item, dict) else item
            if iid:
                ids.append(str(iid))
        return ids

    def instance_file(self, instance_id: str) -> bytes:
        return self._get(f"/instances/{instance_id}/file").content


class InferenceClient:
    """Sentinel API client: login for a JWT, submit a study."""

    def __init__(self, base_url: str, user: Optional[str], password: Optional[str],
                 http=None, timeout: float = 30.0, analyze_timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.user, self.password = user, password
        self.timeout, self.analyze_timeout = timeout, analyze_timeout
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self.token: Optional[str] = None
        self._warned_no_creds = False

    def login(self) -> Optional[str]:
        if not self.user:
            if not self._warned_no_creds:
                logger.warning("SENTINEL_SERVICE_USER not set - submitting without a token "
                               "(only works when the server runs SENTINEL_DEV_INSECURE=1)")
                self._warned_no_creds = True
            self.token = None
            return None
        try:
            r = self.http.post(self.base_url + "/auth/login",
                               json={"username": self.user, "password": self.password or ""},
                               timeout=self.timeout)
        except Exception as e:
            raise AuthError(f"inference login failed: {type(e).__name__}") from e
        if r.status_code != 200:
            raise AuthError(f"inference login rejected: HTTP {r.status_code}")
        self.token = (r.json() or {}).get("access_token")
        if not self.token:
            raise AuthError("inference login returned no access_token")
        logger.info("service token obtained")
        return self.token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def analyze_study(self, file_paths: list, language: str):
        """POST /analyze/study with repeated `files`. Re-logs in once on 401."""
        if self.user and not self.token:
            self.login()
        for attempt in (1, 2):
            handles = []
            try:
                files = []
                for p in file_paths:
                    fh = open(p, "rb")
                    handles.append(fh)
                    files.append(("files", (Path(p).name, fh, "application/dicom")))
                try:
                    r = self.http.post(self.base_url + "/analyze/study", params={"language": language},
                                       files=files, headers=self._headers(), timeout=self.analyze_timeout)
                except Exception as e:
                    raise WatcherError(f"inference POST /analyze/study failed: {type(e).__name__}") from e
            finally:
                for fh in handles:
                    try:
                        fh.close()
                    except Exception:
                        pass
            if r.status_code == 401 and attempt == 1 and self.user:
                logger.info("service token expired - re-authenticating")
                self.login()
                continue
            return r
        return r  # pragma: no cover


# ─── The watcher ─────────────────────────────────────────────────────────────

class OrthancWatcher:
    """Tail Orthanc's change log and submit every StableStudy for analysis.

    Backwards-compatible with the server.py lifespan call
    ``OrthancWatcher(orthanc_url=..., poll_interval=..., on_new_study=...)``
    plus ``check_connection() / start() / stop() / get_orthanc_stats() /
    known_orthanc_ids``. `orthanc_http` / `inference_http` let tests inject an
    in-process transport (e.g. a fastapi TestClient around a stub).
    """

    def __init__(
        self,
        config: Optional[WatcherConfig] = None,
        *,
        orthanc_url: Optional[str] = None,
        poll_interval: Optional[float] = None,
        on_new_study: Optional[Callable] = None,
        orthanc_http=None,
        inference_http=None,
        db=None,                       # legacy kwarg, unused
    ):
        cfg = config or WatcherConfig.from_env()
        if orthanc_url:
            cfg.orthanc_url = orthanc_url.rstrip("/")
        if poll_interval:
            cfg.poll_seconds = float(poll_interval)
        self.config = cfg
        self.on_new_study = on_new_study      # optional hook(study_uid, result: dict)
        self.orthanc = OrthancClient(cfg.orthanc_url, cfg.orthanc_auth, http=orthanc_http,
                                     timeout=cfg.request_timeout)
        self.inference = InferenceClient(cfg.inference_url, cfg.service_user, cfg.service_password,
                                         http=inference_http, timeout=cfg.request_timeout,
                                         analyze_timeout=cfg.analyze_timeout)
        self.state = load_state(cfg.state_path)
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat(timespec="seconds")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.consecutive_errors = 0
        logger.info(f"OrthancWatcher: orthanc={cfg.orthanc_url} inference={cfg.inference_url} "
                    f"poll={cfg.poll_seconds:g}s lang={cfg.language} state={cfg.state_path} "
                    f"resume_seq={self.state.last_seq} processed={len(self.state.processed)}")

    # ----- legacy-facing API -----
    @property
    def orthanc_url(self) -> str:
        return self.config.orthanc_url

    @property
    def poll_interval(self) -> float:
        return self.config.poll_seconds

    @property
    def known_orthanc_ids(self) -> set:
        """Processed StudyInstanceUIDs (name kept for /orthanc/status)."""
        return self.state.processed_uids

    def check_connection(self) -> bool:
        return self.orthanc.check_connection()

    def get_orthanc_stats(self) -> Optional[dict]:
        return self.orthanc.statistics()

    # ----- state -----
    def _save(self) -> None:
        save_state(self.config.state_path, self.state)

    def _mark(self, uid: str, status: str, **extra) -> None:
        self.state.processed[uid] = {"status": status, "at": datetime.now().isoformat(timespec="seconds"), **extra}
        self.state.attempts.pop(uid, None)

    def _note_attempt(self, uid: str, error: str) -> int:
        a = self.state.attempts.get(uid) or {"count": 0}
        a["count"] = int(a.get("count", 0)) + 1
        a["last_error"] = error[:200]
        a["at"] = datetime.now().isoformat(timespec="seconds")
        self.state.attempts[uid] = a
        return a["count"]

    # ----- one poll pass -----
    def poll_once(self) -> int:
        """Drain /changes from last_seq. Returns the number of studies submitted.
        Raises WatcherError on a transient failure (state is saved first)."""
        with self._lock:
            return self._poll_once_locked()

    def _poll_once_locked(self) -> int:
        cfg, st = self.config, self.state
        st.counters["polls"] = st.counters.get("polls", 0) + 1
        st.last_poll_at = datetime.now().isoformat(timespec="seconds")
        submitted = 0
        try:
            while True:
                page = self.orthanc.changes(st.last_seq, cfg.changes_limit)
                changes = page.get("Changes") or []
                for ch in changes:
                    seq = int(ch.get("Seq") or st.last_seq)
                    st.counters["changes_seen"] = st.counters.get("changes_seen", 0) + 1
                    if ch.get("ChangeType") == CHANGE_TYPE and ch.get("ID"):
                        try:
                            if self._handle_stable_study(str(ch["ID"])):
                                submitted += 1
                        except WatcherError:
                            # leave last_seq at the previous change so this one is retried
                            self._save()
                            raise
                    st.last_seq = max(st.last_seq, seq)
                last = page.get("Last")
                if isinstance(last, int):
                    st.last_seq = max(st.last_seq, last)
                self._save()
                if page.get("Done", True) or not changes:
                    break
            st.last_success_at = datetime.now().isoformat(timespec="seconds")
            st.last_error = None
            st.counters["consecutive_errors"] = 0
            self.consecutive_errors = 0
            self._save()
            return submitted
        except WatcherError as e:
            self.consecutive_errors += 1
            st.counters["consecutive_errors"] = self.consecutive_errors
            st.last_error = str(e)[:300]
            try:
                self._save()
            except Exception as se:
                logger.error(f"watcher state save failed: {type(se).__name__}")
            logger.warning(f"poll failed (consecutive={self.consecutive_errors}): {e}")
            raise

    def _handle_stable_study(self, orthanc_id: str) -> bool:
        """Download + submit one study. True when submitted; False when skipped.
        Raises WatcherError for transient failures (after recording the attempt)."""
        st = self.state
        info = self.orthanc.study(orthanc_id)
        tags = (info or {}).get("MainDicomTags") or {}
        uid = str(tags.get("StudyInstanceUID") or "") or f"orthanc:{orthanc_id}"
        if uid in st.processed:
            st.counters["studies_skipped"] = st.counters.get("studies_skipped", 0) + 1
            logger.info(f"skip study orthanc_id={orthanc_id} uid={uid} (already processed: "
                        f"{st.processed[uid].get('status')})")
            return False

        tmp_dir = Path(tempfile.mkdtemp(prefix="sentinel_pacs_"))
        t0 = time.time()
        try:
            try:
                instance_ids = self.orthanc.study_instances(orthanc_id)
                if not instance_ids:
                    self._mark(uid, "skipped", reason="no instances", orthanc_id=orthanc_id)
                    logger.warning(f"study orthanc_id={orthanc_id} uid={uid} has no instances - skipped")
                    return False
                paths = []
                for i, iid in enumerate(instance_ids):
                    data = self.orthanc.instance_file(iid)
                    p = tmp_dir / f"{i:05d}.dcm"
                    p.write_bytes(data)
                    paths.append(p)
                logger.info(f"downloaded study orthanc_id={orthanc_id} uid={uid} files={len(paths)} "
                            f"in {time.time() - t0:.1f}s")
                resp = self.inference.analyze_study(paths, self.config.language)
            except WatcherError as e:
                n = self._note_attempt(uid, str(e))
                if n >= self.config.max_attempts:
                    st.counters["studies_failed"] = st.counters.get("studies_failed", 0) + 1
                    self._mark(uid, "failed", reason=f"gave up after {n} attempts: {e}"[:300],
                               orthanc_id=orthanc_id)
                    logger.error(f"study orthanc_id={orthanc_id} uid={uid} quarantined after {n} attempts: {e}")
                    return False
                raise

            code = resp.status_code
            body: Any = {}
            try:
                body = resp.json()
            except Exception:
                body = {}
            if code == 200:
                st.counters["studies_processed"] = st.counters.get("studies_processed", 0) + 1
                self._mark(uid, "analyzed", orthanc_id=orthanc_id, study_id=body.get("study_id"),
                           route=body.get("route"), files=len(paths))
                logger.info(f"analyzed study orthanc_id={orthanc_id} uid={uid} study_id={body.get('study_id')} "
                            f"route={body.get('route')} findings={len(body.get('findings') or [])} "
                            f"total={time.time() - t0:.1f}s")
                if self.on_new_study:
                    try:
                        self.on_new_study(uid, body)
                    except Exception as e:
                        logger.warning(f"on_new_study hook failed: {type(e).__name__}")
                return True
            if code == 422:
                st.counters["studies_processed"] = st.counters.get("studies_processed", 0) + 1
                self._mark(uid, "requires_review", orthanc_id=orthanc_id, study_id=body.get("study_id"),
                           reason=str(body.get("reason") or body.get("detail") or "")[:200], files=len(paths))
                logger.info(f"study orthanc_id={orthanc_id} uid={uid} -> requires_review (422)")
                return True
            if code in (400, 409, 413, 415):
                st.counters["studies_failed"] = st.counters.get("studies_failed", 0) + 1
                self._mark(uid, "rejected", orthanc_id=orthanc_id, http=code,
                           reason=str(body.get("detail") or "")[:200])
                logger.warning(f"study orthanc_id={orthanc_id} uid={uid} rejected by inference: HTTP {code}")
                return False
            if code in (401, 403):
                raise AuthError(f"inference refused the service account: HTTP {code} "
                                f"(check SENTINEL_SERVICE_USER role/permissions)")
            if code == 402:
                raise WatcherError("inference server is demo-limited (HTTP 402) - activate a license")
            # 5xx / anything else: transient
            n = self._note_attempt(uid, f"HTTP {code}")
            if n >= self.config.max_attempts:
                st.counters["studies_failed"] = st.counters.get("studies_failed", 0) + 1
                self._mark(uid, "failed", orthanc_id=orthanc_id, reason=f"HTTP {code} x{n}")
                logger.error(f"study orthanc_id={orthanc_id} uid={uid} quarantined after {n} attempts (HTTP {code})")
                return False
            raise WatcherError(f"inference returned HTTP {code} for study orthanc_id={orthanc_id}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ----- loop / supervisor -----
    def backoff_seconds(self) -> float:
        if self.consecutive_errors <= 0:
            return self.config.poll_seconds
        base = self.config.poll_seconds * (2 ** min(self.consecutive_errors, 10))
        return min(self.config.max_backoff_seconds, base) * (0.8 + 0.4 * random.random())

    def run_forever(self, stop: Optional[threading.Event] = None) -> None:
        """Poll until `stop` is set. Never raises: every failure is logged and
        followed by an exponential (capped, jittered) back-off."""
        stop = stop or self._stop
        while not stop.is_set():
            try:
                self.poll_once()
            except WatcherError:
                pass                              # logged + counted in poll_once
            except Exception as e:                # a bug must not kill the service
                self.consecutive_errors += 1
                self.state.last_error = f"{type(e).__name__}: {e}"[:300]
                logger.exception(f"unexpected watcher error: {type(e).__name__}")
            stop.wait(self.backoff_seconds())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("Orthanc watcher already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever, name="orthanc-watcher", daemon=True)
        self._thread.start()
        logger.info("Orthanc watcher started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Orthanc watcher stopped")


def auto_analyze_callback(study_uid: str, result: dict) -> None:
    """Legacy hook kept for server.py's lifespan import. The watcher now submits
    to /analyze/study itself (which persists the study); nothing to do here."""
    logger.debug(f"auto_analyze_callback: uid={study_uid} study_id={result.get('study_id')}")


# ─── CLI / supervisor entry point ────────────────────────────────────────────

def main(argv: Optional[list] = None, *, orthanc_http=None, inference_http=None) -> int:
    parser = argparse.ArgumentParser(description="Sentinel Orthanc PACS auto-ingest watcher")
    parser.add_argument("--once", action="store_true", help="run a single poll pass and exit")
    parser.add_argument("--state", help="state file path (default DATA_DIR/orthanc_watcher_state.json)")
    args = parser.parse_args(argv)

    from src.utils.logger import setup_logger
    setup_logger()

    cfg = WatcherConfig.from_env()
    if args.state:
        cfg.state_path = Path(args.state)
    watcher = OrthancWatcher(cfg, orthanc_http=orthanc_http, inference_http=inference_http)

    if watcher.check_connection():
        logger.info(f"Orthanc reachable at {cfg.orthanc_url}")
    else:
        logger.warning(f"Orthanc not reachable at {cfg.orthanc_url} - will keep retrying")

    if args.once:
        try:
            n = watcher.poll_once()
        except WatcherError as e:
            logger.error(f"poll failed: {e}")
            return 1
        logger.info(f"poll done: submitted={n} last_seq={watcher.state.last_seq}")
        return 0

    stop = threading.Event()

    def _sig(*_):
        logger.info("stop requested")
        stop.set()

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _sig)
        except (ValueError, OSError):          # not the main thread / unsupported
            pass
    try:
        watcher.run_forever(stop)
    except KeyboardInterrupt:
        pass
    logger.info("watcher exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
