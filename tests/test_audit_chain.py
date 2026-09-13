"""Tamper-evident audit log: hash chain + append-only triggers + GET /audit/verify."""

import sqlite3

import pytest

from src.utils.database import SentinelDB, AUDIT_GENESIS, audit_row_hash


@pytest.fixture
def db(tmp_path):
    return SentinelDB(tmp_path / "audit.db")


def test_chain_links_rows(db):
    ids = [db.log_action("u1", f"act{i}", "study", f"e{i}", details='d "q"', ip_address="127.0.0.1") for i in range(3)]
    assert ids == [1, 2, 3]
    rows = db.get_audit_log()[::-1]            # oldest first
    assert rows[0]["prev_hash"] == AUDIT_GENESIS
    for prev, cur in zip(rows, rows[1:]):
        assert cur["prev_hash"] == prev["row_hash"]
    for r in rows:
        assert r["row_hash"] == audit_row_hash(r, r["prev_hash"])
    v = db.verify_audit_chain()
    assert v == {"ok": True, "rows": 3, "checked": 3, "head_hash": rows[-1]["row_hash"], "first_broken": None}


def test_update_and_delete_are_refused(db):
    db.log_action("u1", "a")
    raw = sqlite3.connect(str(db.db_path))
    for sql in ("UPDATE audit_log SET details='x' WHERE id=1", "DELETE FROM audit_log WHERE id=1"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            raw.execute(sql)
    raw.close()
    assert db.verify_audit_chain()["ok"] is True


def test_tamper_is_detected_at_first_broken_row(db):
    for i in range(4):
        db.log_action("u1", f"a{i}")
    raw = sqlite3.connect(str(db.db_path))
    raw.execute("DROP TRIGGER audit_log_no_update")
    raw.execute("UPDATE audit_log SET details='forged' WHERE id=3")
    raw.commit(); raw.close()
    v = db.verify_audit_chain()
    assert v["ok"] is False and v["checked"] == 2
    assert v["first_broken"] == {"id": 3, "reason": "row_hash does not match row contents"}
    # a spliced-in row (valid self-hash, wrong link) is caught too
    raw = sqlite3.connect(str(db.db_path)); raw.row_factory = sqlite3.Row
    raw.execute("DROP TRIGGER audit_log_no_delete")
    raw.execute("DELETE FROM audit_log WHERE id=2"); raw.commit(); raw.close()
    v = db.verify_audit_chain()
    assert v["first_broken"]["id"] == 3 and "prev_hash" in v["first_broken"]["reason"]


def test_legacy_rows_are_backfilled_once(tmp_path):
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, action TEXT NOT NULL, "
                "entity_type TEXT, entity_id TEXT, details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    raw.executemany("INSERT INTO audit_log (user_id, action) VALUES (?, ?)", [("a", "x"), ("b", "y")])
    raw.commit(); raw.close()
    db = SentinelDB(path)                      # migration hashes the 2 legacy rows
    db.log_action("c", "z")
    v = db.verify_audit_chain()
    assert v["ok"] is True and v["rows"] == 3
    assert SentinelDB(path).verify_audit_chain()["head_hash"] == v["head_hash"]   # reopen: stable


def test_log_inside_callers_transaction(db):
    conn = db._connect()
    conn.execute("BEGIN IMMEDIATE")
    db.log_action("u1", "first", conn=conn)
    db.log_action("u1", "second", conn=conn)
    conn.commit(); conn.close()
    assert db.verify_audit_chain() == {**db.verify_audit_chain(), "ok": True, "rows": 2}


def test_audit_verify_endpoint_admin_only(client, analyzed_study):
    assert client.get("/audit/verify").status_code == 401
    login = client.post("/auth/login", json={"username": "admin", "password": "Test-Admin-Pass-2026"})
    assert login.status_code == 200, login.text
    bearer = {"Authorization": f"Bearer {login.json()['access_token']}"}
    r = client.get("/audit/verify", headers=bearer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["first_broken"] is None
    assert body["rows"] >= 1 and body["checked"] == body["rows"]     # the analyze audits are in the chain
    assert len(body["head_hash"]) == 64
