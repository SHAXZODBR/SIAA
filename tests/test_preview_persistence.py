"""Persisted analyzed-slice previews - DATA_DIR/previews/<study_id>.png.

POST /analyze/study writes the preview PNG to disk (0600 in a 0700 dir) and
records only its path + sha256 on the ai_results row, so GET /study/{id} can
show the scan again after a restart; GET /studies stays light (has_preview
only). prune_previews() is the retention guard run at startup
(SENTINEL_PREVIEW_RETENTION_DAYS, default 365).

Uses the session-shared `analyzed_study`. The prune test puts the file it
removes back, so the modules that run after this one find the fixture as
they left it.
"""

import base64
import hashlib
import os
import sqlite3
import stat

from conftest import SCRATCH_DATA_DIR

from src.inference.server import (
    DEFAULT_PREVIEW_RETENTION_DAYS,
    _load_preview,
    _preview_retention_days,
    _write_private,
    prune_previews,
)
from src.utils.paths import PREVIEWS_DIR

DATA_URI = "data:image/png;base64,"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _preview_file(study_id: str):
    return PREVIEWS_DIR / f"{study_id}.png"


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─── write side: /analyze/study persists the PNG; SQLite keeps path + sha only ─

def test_previews_dir_is_under_scratch_data_dir():
    assert PREVIEWS_DIR == SCRATCH_DATA_DIR / "previews"


def test_analyze_writes_private_preview_file(analyzed_study):
    sid = analyzed_study["study_id"]
    p = _preview_file(sid)
    assert p.is_file(), f"preview not written: {p}"
    assert _mode(p) == 0o600
    assert _mode(PREVIEWS_DIR) == 0o700
    png = p.read_bytes()
    assert png.startswith(PNG_MAGIC)
    # byte-for-byte what the analysis response carried
    assert png == base64.b64decode(analyzed_study["preview_base64"][len(DATA_URI):])


def test_db_row_holds_path_and_sha_never_bytes(analyzed_study):
    from src.inference.auth_routes import db

    sid = analyzed_study["study_id"]
    row = db.get_ai_result(sid)
    assert row["preview_path"] == str(_preview_file(sid))
    assert row["preview_sha256"] == _sha(_preview_file(sid).read_bytes())
    assert len(row["preview_sha256"]) == 64
    for k, v in row.items():
        assert not (isinstance(v, str) and v.startswith("data:image")), \
            f"image bytes inlined in ai_results.{k}"


def test_migration_adds_preview_columns_to_pre_preview_db(tmp_path):
    from src.utils.database import SentinelDB

    path = tmp_path / "old.db"
    SentinelDB(str(path))
    with sqlite3.connect(path) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(ai_results)")}
        assert {"preview_path", "preview_sha256"} <= cols
        # roll the schema back to before this feature, then "restart"
        c.execute("ALTER TABLE ai_results DROP COLUMN preview_path")
        c.execute("ALTER TABLE ai_results DROP COLUMN preview_sha256")
    SentinelDB(str(path))
    with sqlite3.connect(path) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(ai_results)")}
    assert {"preview_path", "preview_sha256"} <= cols


# ─── read side: GET /study/{id} inlines the image, GET /studies only flags it ─

def test_get_study_returns_preview_data_uri_matching_file(client, analyzed_study):
    sid = analyzed_study["study_id"]
    ai = client.get(f"/study/{sid}").json()["ai_result"]
    pb = ai["preview_base64"]
    assert pb and pb.startswith(DATA_URI)
    png = _preview_file(sid).read_bytes()
    assert base64.b64decode(pb[len(DATA_URI):]) == png
    assert ai["preview_sha256"] == _sha(png)


def test_fresh_db_handle_restores_preview_like_a_restart(analyzed_study):
    """A restart is a new SentinelDB() on the same DATA_DIR plus the file on
    disk - nothing from the analysis response is held in memory."""
    from src.utils.database import SentinelDB

    sid = analyzed_study["study_id"]
    row = SentinelDB().get_ai_result(sid)
    png = _preview_file(sid).read_bytes()
    assert _load_preview(row["preview_path"], row["preview_sha256"]) == \
        DATA_URI + base64.b64encode(png).decode()


def test_list_studies_flags_preview_without_inlining(client, analyzed_study):
    sid = analyzed_study["study_id"]
    rows = client.get("/studies").json()
    row = next(r for r in rows if r["study_id"] == sid)
    assert row["has_preview"] is True
    assert "preview_base64" not in row and "preview_path" not in row
    for r in rows:
        assert isinstance(r["has_preview"], bool)


# ─── retention guard ─────────────────────────────────────────────────────────

def test_retention_days_env_parsing(monkeypatch):
    monkeypatch.delenv("SENTINEL_PREVIEW_RETENTION_DAYS", raising=False)
    assert _preview_retention_days() == DEFAULT_PREVIEW_RETENTION_DAYS == 365
    monkeypatch.setenv("SENTINEL_PREVIEW_RETENTION_DAYS", "30")
    assert _preview_retention_days() == 30
    monkeypatch.setenv("SENTINEL_PREVIEW_RETENTION_DAYS", "-5")
    assert _preview_retention_days() == 0
    monkeypatch.setenv("SENTINEL_PREVIEW_RETENTION_DAYS", "soon")
    assert _preview_retention_days() == DEFAULT_PREVIEW_RETENTION_DAYS
    monkeypatch.setenv("SENTINEL_PREVIEW_RETENTION_DAYS", "  ")
    assert _preview_retention_days() == DEFAULT_PREVIEW_RETENTION_DAYS


def test_prune_keeps_fresh_preview_at_default_retention(monkeypatch, analyzed_study):
    monkeypatch.delenv("SENTINEL_PREVIEW_RETENTION_DAYS", raising=False)
    sid = analyzed_study["study_id"]
    stats = prune_previews()
    assert stats["retention_days"] == 365
    assert stats["removed"] == 0 and stats["errors"] == 0 and stats["kept"] >= 1
    assert _preview_file(sid).is_file()


def test_load_preview_refuses_sha_mismatch_and_missing_file(analyzed_study):
    sid = analyzed_study["study_id"]
    p = _preview_file(sid)
    assert _load_preview(str(p), "0" * 64) is None                  # swapped / tampered
    assert _load_preview(str(p), None).startswith(DATA_URI)         # no recorded hash
    assert _load_preview(str(PREVIEWS_DIR / "missing.png"), None) is None
    assert _load_preview(None) is None


def test_prune_retention_zero_removes_preview_and_api_reports_null(client, analyzed_study):
    sid = analyzed_study["study_id"]
    p = _preview_file(sid)
    png = p.read_bytes()
    try:
        stats = prune_previews(retention_days=0)
        assert stats["retention_days"] == 0
        assert stats["removed"] >= 1 and stats["errors"] == 0
        assert not p.exists()
        ai = client.get(f"/study/{sid}").json()["ai_result"]
        assert ai["preview_base64"] is None
        assert ai["preview_sha256"] == _sha(png)        # the record of what was shown survives
        row = next(r for r in client.get("/studies").json() if r["study_id"] == sid)
        assert row["has_preview"] is False
    finally:
        _write_private(p, png)      # leave the session fixture as found
    assert _mode(p) == 0o600
    assert client.get(f"/study/{sid}").json()["ai_result"]["preview_base64"] == \
        DATA_URI + base64.b64encode(png).decode()
