"""SQLite database layer for Sentinel Medical AI.

Handles all local data persistence: studies, AI results, reports, users, audit logs.

SECURITY NOTE: This DB is currently stored UNENCRYPTED on local disk. For
production deployment, run on an OS-level encrypted volume (FileVault / BitLocker /
LUKS) or migrate to SQLCipher. Do NOT claim encryption-at-rest until implemented.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'radiologist', 'technician')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    orthanc_id TEXT,
    patient_id TEXT,
    modality TEXT,
    body_part TEXT,
    study_date DATE,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dicom_path TEXT,
    ai_status TEXT DEFAULT 'pending' CHECK(ai_status IN ('pending', 'processing', 'complete', 'error')),
    ai_analyzed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_results (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES studies(id),
    findings_json TEXT,
    heatmap_paths TEXT,
    inference_time_ms INTEGER,
    model_version TEXT,
    is_normal BOOLEAN DEFAULT FALSE,
    overall_impression TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES studies(id),
    doctor_id TEXT REFERENCES users(id),
    report_text TEXT,
    ai_draft_text TEXT,
    pdf_path TEXT,
    language TEXT DEFAULT 'ru',
    is_signed BOOLEAN DEFAULT FALSE,
    signed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    details TEXT,
    ip_address TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_studies_status ON studies(ai_status);
CREATE INDEX IF NOT EXISTS idx_studies_date ON studies(study_date);
CREATE INDEX IF NOT EXISTS idx_ai_results_study ON ai_results(study_id);
CREATE INDEX IF NOT EXISTS idx_reports_study ON reports(study_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""


class SentinelDB:
    """Local SQLite database manager for Sentinel."""

    def __init__(self, db_path: str = "data/sentinel.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database and create tables."""
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"Database initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection.

        busy_timeout makes concurrent writers wait-and-retry instead of raising
        'database is locked' (which would silently drop audit/report writes).
        WAL allows concurrent readers during a write. This is a stopgap for
        multi-user load; for a clinic chain, migrate to PostgreSQL.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")   # wait up to 30s for a lock
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")   # durable + fast under WAL
        return conn

    # ===== Studies =====

    def create_study(
        self,
        patient_id: str,
        modality: str,
        body_part: str = "",
        dicom_path: str = "",
        orthanc_id: str = "",
        study_date: Optional[str] = None,
    ) -> str:
        """Create a new study record. Returns study ID."""
        study_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO studies (id, orthanc_id, patient_id, modality, body_part,
                   study_date, dicom_path) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (study_id, orthanc_id, patient_id, modality, body_part, study_date, dicom_path),
            )
        logger.debug(f"Created study: {study_id} ({modality})")
        return study_id

    def update_study_status(self, study_id: str, status: str):
        """Update AI analysis status of a study."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE studies SET ai_status=?, ai_analyzed_at=? WHERE id=?",
                (status, datetime.now().isoformat() if status == "complete" else None, study_id),
            )

    def get_pending_studies(self) -> list[dict]:
        """Get all studies pending AI analysis."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies WHERE ai_status='pending' ORDER BY received_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_study(self, study_id: str) -> Optional[dict]:
        """Get a single study by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        return dict(row) if row else None

    def get_all_studies(self, limit: int = 100) -> list[dict]:
        """Get recent studies."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies ORDER BY received_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ===== AI Results =====

    def save_ai_result(
        self,
        study_id: str,
        findings_json: str,
        inference_time_ms: int,
        model_version: str,
        is_normal: bool,
        overall_impression: str,
        heatmap_paths: str = "",
    ) -> str:
        """Save AI analysis results. Returns result ID."""
        result_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ai_results (id, study_id, findings_json, heatmap_paths,
                   inference_time_ms, model_version, is_normal, overall_impression)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, study_id, findings_json, heatmap_paths,
                 inference_time_ms, model_version, is_normal, overall_impression),
            )
        self.update_study_status(study_id, "complete")
        return result_id

    def get_ai_result(self, study_id: str) -> Optional[dict]:
        """Get AI results for a study."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_results WHERE study_id=? ORDER BY created_at DESC LIMIT 1",
                (study_id,),
            ).fetchone()
        return dict(row) if row else None

    # ===== Reports =====

    def create_report(
        self,
        study_id: str,
        doctor_id: str,
        ai_draft_text: str,
        language: str = "ru",
    ) -> str:
        """Create a new report from AI draft. Returns report ID."""
        report_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO reports (id, study_id, doctor_id, report_text,
                   ai_draft_text, language) VALUES (?, ?, ?, ?, ?, ?)""",
                (report_id, study_id, doctor_id, ai_draft_text, ai_draft_text, language),
            )
        return report_id

    def update_report_text(self, report_id: str, report_text: str):
        """Update report text (doctor editing)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET report_text=?, updated_at=? WHERE id=?",
                (report_text, datetime.now().isoformat(), report_id),
            )

    def sign_report(self, report_id: str, doctor_id: str) -> bool:
        """Digitally sign a report (locks it from further editing)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET is_signed=1, signed_at=? WHERE id=? AND is_signed=0",
                (datetime.now().isoformat(), report_id),
            )
            self.log_action(doctor_id, "sign_report", "report", report_id)
        logger.info(f"Report {report_id} signed by {doctor_id}")
        return True

    def get_report(self, report_id: str) -> Optional[dict]:
        """Get a report by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    # ===== Users =====

    def create_user(
        self,
        username: str,
        password_hash: str,
        full_name: str,
        role: str = "radiologist",
    ) -> str:
        """Create a new user. Returns user ID."""
        user_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, full_name, role) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, password_hash, full_name, role),
            )
        logger.info(f"User created: {username} ({role})")
        return user_id

    # ===== Audit Log =====

    def log_action(
        self,
        user_id: str,
        action: str,
        entity_type: str = "",
        entity_id: str = "",
        details: str = "",
        ip_address: str = "",
    ):
        """Record an action in the audit log (now captures ip_address)."""
        with self._connect() as conn:
            # Populate ip_address column if it exists in the schema.
            cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
            if "ip_address" in cols:
                conn.execute(
                    "INSERT INTO audit_log (user_id, action, entity_type, entity_id, details, ip_address) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, action, entity_type, entity_id, details, ip_address),
                )
            else:
                conn.execute(
                    "INSERT INTO audit_log (user_id, action, entity_type, entity_id, details) VALUES (?, ?, ?, ?, ?)",
                    (user_id, action, entity_type, entity_id, details),
                )

    def get_audit_log(self, limit: int = 500) -> list:
        """Return recent audit entries (admin view / compliance export)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ===== Stats =====

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connect() as conn:
            total_studies = conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
            completed = conn.execute("SELECT COUNT(*) FROM studies WHERE ai_status='complete'").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM studies WHERE ai_status='pending'").fetchone()[0]
            total_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
            signed_reports = conn.execute("SELECT COUNT(*) FROM reports WHERE is_signed=1").fetchone()[0]

        return {
            "total_studies": total_studies,
            "completed_analyses": completed,
            "pending_analyses": pending,
            "total_reports": total_reports,
            "signed_reports": signed_reports,
        }
