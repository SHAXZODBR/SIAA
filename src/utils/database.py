"""SQLite database layer for Sentinel Medical AI.

Handles all local data persistence: studies, AI results, reports, users, audit logs.

SECURITY NOTE: This DB is currently stored UNENCRYPTED on local disk. For
production deployment, run on an OS-level encrypted volume (FileVault / BitLocker /
LUKS) or migrate to SQLCipher. Do NOT claim encryption-at-rest until implemented.

Schema evolution: SCHEMA_SQL only ever uses CREATE ... IF NOT EXISTS, and
MIGRATIONS lists columns added after v1.0 as (table, column, declaration).
_migrate() applies ALTER TABLE ADD COLUMN for whichever are missing, so an old
pilot DB upgrades in place on first open.

Tamper-evident audit log: every audit_log row carries
    prev_hash = row_hash of the previous row (GENESIS for the first)
    row_hash  = sha256(canonical JSON of the row's fields + prev_hash)
so editing, reordering or removing an interior row breaks the chain, which
verify_audit_chain() / GET /audit/verify walks and reports. SQLite triggers
refuse UPDATE/DELETE on audit_log (append-only). Legacy rows written before the
chain existed are back-filled once, in id order, at migration time.
"""

import hashlib
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
    last_login TIMESTAMP,
    must_change_password INTEGER DEFAULT 0
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
    ai_analyzed_at TIMESTAMP,
    study_instance_uid TEXT,
    accession_number TEXT,
    patient_name TEXT,
    patient_sex TEXT,
    patient_age TEXT,
    patient_birth_date TEXT,
    study_description TEXT,
    manufacturer TEXT,
    scanner_model TEXT,
    num_files INTEGER,
    series_descriptions TEXT,
    rejected INTEGER DEFAULT 0,
    requires_review INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    created_by TEXT
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_identity_json TEXT,
    overall_json TEXT,
    threshold REAL,
    report_text TEXT,
    report_language TEXT
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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    signer_id TEXT,
    sha256 TEXT,
    findings_json TEXT,
    model_identity_json TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    study_id TEXT,
    doctor_id TEXT,
    language TEXT DEFAULT 'ru',
    original_report TEXT,
    corrected_report TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

# Columns added after the v1.0 schema. ALTER TABLE ADD COLUMN cannot take a
# non-constant default in SQLite, so timestamps here are set explicitly on insert.
MIGRATIONS = [
    ("users",      "must_change_password", "INTEGER DEFAULT 0"),
    ("studies",    "study_instance_uid",   "TEXT"),
    ("studies",    "accession_number",     "TEXT"),
    ("studies",    "patient_name",         "TEXT"),
    ("studies",    "patient_sex",          "TEXT"),
    ("studies",    "patient_age",          "TEXT"),
    ("studies",    "patient_birth_date",   "TEXT"),
    ("studies",    "study_description",    "TEXT"),
    ("studies",    "manufacturer",         "TEXT"),
    ("studies",    "scanner_model",        "TEXT"),
    ("studies",    "num_files",            "INTEGER"),
    ("studies",    "series_descriptions",  "TEXT"),
    ("studies",    "rejected",             "INTEGER DEFAULT 0"),
    ("studies",    "requires_review",      "INTEGER DEFAULT 0"),
    ("studies",    "created_at",           "TIMESTAMP"),
    ("studies",    "created_by",           "TEXT"),
    ("ai_results", "model_identity_json",  "TEXT"),
    ("ai_results", "overall_json",         "TEXT"),
    ("ai_results", "threshold",            "REAL"),
    ("ai_results", "report_text",          "TEXT"),
    ("ai_results", "report_language",      "TEXT"),
    ("ai_results", "app_version",          "TEXT"),
    # Persisted preview of the analyzed slice: DATA_DIR/previews/<study_id>.png
    # and its sha256. The PNG itself is never stored in SQLite.
    ("ai_results", "preview_path",         "TEXT"),
    ("ai_results", "preview_sha256",       "TEXT"),
    ("reports",    "signer_id",            "TEXT"),
    ("reports",    "sha256",               "TEXT"),
    ("reports",    "findings_json",        "TEXT"),
    ("reports",    "model_identity_json",  "TEXT"),
    # Signed-report PDF + its DICOM Encapsulated PDF twin (POST /report/{id}/pdf)
    ("reports",    "pdf_sha256",            "TEXT"),
    ("reports",    "dicom_path",            "TEXT"),
    ("reports",    "dicom_sop_instance_uid", "TEXT"),
    ("reports",    "orthanc_instance_id",   "TEXT"),
    ("reports",    "pdf_attached_at",       "TIMESTAMP"),
    ("reports",    "pushed_at",             "TIMESTAMP"),
    ("audit_log",  "ip_address",           "TEXT"),
    ("audit_log",  "prev_hash",            "TEXT"),
    ("audit_log",  "row_hash",             "TEXT"),
]

AUDIT_GENESIS = "0" * 64
# Fields hashed, in this order (id and timestamp included so position + time
# are pinned; details is free text so it is JSON-escaped, not concatenated).
AUDIT_HASH_FIELDS = ("id", "user_id", "action", "entity_type", "entity_id",
                     "details", "ip_address", "timestamp")

AUDIT_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def audit_row_hash(row: dict, prev_hash: str) -> str:
    """sha256 over the canonical JSON of the hashed fields + the previous hash."""
    canon = {k: ("" if row.get(k) is None else str(row.get(k))) for k in AUDIT_HASH_FIELDS}
    canon["prev_hash"] = prev_hash
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReportAlreadySignedError(Exception):
    """Raised when a signed report already exists for (study_id, language)."""


class SentinelDB:
    """Local SQLite database manager for Sentinel."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from src.utils.paths import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database, create tables, apply column migrations."""
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate(conn)
        logger.info(f"Database initialized at {self.db_path}")

    @staticmethod
    def _migrate(conn: sqlite3.Connection):
        """Add any columns an older DB is missing (idempotent)."""
        cols_cache: dict[str, set] = {}
        for table, column, decl in MIGRATIONS:
            if table not in cols_cache:
                cols_cache[table] = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols_cache[table]:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                cols_cache[table].add(column)
                logger.info(f"DB migration: added {table}.{column}")
        # One signed report per (study, language). Partial index; tolerate an
        # old DB that already violates it rather than refusing to open.
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_signed_unique "
                "ON reports(study_id, language) WHERE is_signed=1"
            )
        except sqlite3.Error as e:
            logger.warning(f"Could not create signed-report unique index: {e}")
        SentinelDB._migrate_audit_chain(conn)

    @staticmethod
    def _migrate_audit_chain(conn: sqlite3.Connection):
        """Back-fill prev_hash/row_hash for rows written before the chain
        existed (in id order, once), then install the append-only triggers."""
        n_unhashed = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE row_hash IS NULL OR row_hash=''"
        ).fetchone()[0]
        if n_unhashed:
            # Triggers (if present from an earlier open) would refuse the UPDATE.
            conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
            conn.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
            prev = AUDIT_GENESIS
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
            for r in rows:
                d = dict(r)
                if d.get("row_hash"):
                    prev = d["row_hash"]
                    continue
                h = audit_row_hash(d, prev)
                conn.execute("UPDATE audit_log SET prev_hash=?, row_hash=? WHERE id=?", (prev, h, d["id"]))
                prev = h
            logger.info(f"DB migration: hashed {n_unhashed} legacy audit_log rows into the chain")
        conn.executescript(AUDIT_TRIGGERS_SQL)

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
        patient_id: str = "",
        modality: str = "",
        body_part: str = "",
        dicom_path: str = "",
        orthanc_id: str = "",
        study_date: Optional[str] = None,
        *,
        study_id: Optional[str] = None,
        created_by: Optional[str] = None,
        study_instance_uid: Optional[str] = None,
        accession_number: Optional[str] = None,
        patient_name: Optional[str] = None,
        patient_sex: Optional[str] = None,
        patient_age: Optional[str] = None,
        patient_birth_date: Optional[str] = None,
        study_description: Optional[str] = None,
        manufacturer: Optional[str] = None,
        scanner_model: Optional[str] = None,
        num_files: Optional[int] = None,
        series_descriptions: Optional[list] = None,
        rejected: bool = False,
        requires_review: bool = False,
    ) -> str:
        """Create a new study record. Returns study ID.

        Positional args keep the watcher call sites working; the keyword-only
        block carries the header fields persisted by /analyze/study."""
        study_id = study_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO studies (id, orthanc_id, patient_id, modality, body_part,
                   study_date, dicom_path, study_instance_uid, accession_number, patient_name,
                   patient_sex, patient_age, patient_birth_date, study_description, manufacturer,
                   scanner_model, num_files, series_descriptions, rejected, requires_review,
                   created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (study_id, orthanc_id, patient_id, modality, body_part, study_date, dicom_path,
                 study_instance_uid, accession_number, patient_name, patient_sex, patient_age,
                 patient_birth_date, study_description, manufacturer, scanner_model, num_files,
                 json.dumps(series_descriptions or [], ensure_ascii=False),
                 int(bool(rejected)), int(bool(requires_review)),
                 datetime.now().isoformat(), created_by),
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
        return self._study_row(row) if row else None

    def get_all_studies(self, limit: int = 100) -> list[dict]:
        """Get recent studies."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies ORDER BY received_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_studies(self, limit: int = 100) -> list[dict]:
        """Newest-first studies with a compact AI summary + signed languages —
        what the desktop worklist needs to restore itself after a restart.
        preview_path is the on-disk PNG reference only (no image bytes), for
        the API layer to turn into has_preview."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies ORDER BY COALESCE(created_at, received_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                study = self._study_row(r)
                ai = conn.execute(
                    "SELECT overall_json, is_normal, inference_time_ms, created_at, preview_path "
                    "FROM ai_results WHERE study_id=? ORDER BY created_at DESC LIMIT 1",
                    (study["id"],)
                ).fetchone()
                overall = None
                if ai and ai["overall_json"]:
                    try:
                        overall = json.loads(ai["overall_json"])
                    except Exception:
                        overall = None
                signed = conn.execute(
                    "SELECT language FROM reports WHERE study_id=? AND is_signed=1", (study["id"],)
                ).fetchall()
                study["overall_assessment"] = overall
                study["signed_languages"] = [s["language"] for s in signed]
                study["preview_path"] = ai["preview_path"] if ai else None
                out.append(study)
        return out

    @staticmethod
    def _study_row(row) -> dict:
        d = dict(row)
        try:
            d["series_descriptions"] = json.loads(d.get("series_descriptions") or "[]")
        except Exception:
            d["series_descriptions"] = []
        d["rejected"] = bool(d.get("rejected"))
        d["requires_review"] = bool(d.get("requires_review"))
        return d

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
        *,
        model_identity_json: Optional[str] = None,
        overall_json: Optional[str] = None,
        threshold: Optional[float] = None,
        report_text: Optional[str] = None,
        report_language: Optional[str] = None,
        app_version: Optional[str] = None,
        preview_path: Optional[str] = None,
        preview_sha256: Optional[str] = None,
    ) -> str:
        """Save AI analysis results. Returns result ID.

        preview_path / preview_sha256 point at the PNG of the analyzed slice on
        disk (see server._save_preview); the image bytes never enter the DB."""
        result_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ai_results (id, study_id, findings_json, heatmap_paths,
                   inference_time_ms, model_version, is_normal, overall_impression,
                   model_identity_json, overall_json, threshold, report_text, report_language,
                   app_version, preview_path, preview_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, study_id, findings_json, heatmap_paths,
                 inference_time_ms, model_version, is_normal, overall_impression,
                 model_identity_json, overall_json, threshold, report_text, report_language,
                 app_version, preview_path, preview_sha256),
            )
        self.update_study_status(study_id, "complete")
        return result_id

    def get_ai_result(self, study_id: str) -> Optional[dict]:
        """Get AI results for a study (JSON columns parsed)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_results WHERE study_id=? ORDER BY created_at DESC LIMIT 1",
                (study_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        for col, key in (("findings_json", "findings"), ("model_identity_json", "model_identity"),
                         ("overall_json", "overall_assessment")):
            try:
                d[key] = json.loads(d.get(col) or "null")
            except Exception:
                d[key] = None
        return d

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
        """Digitally sign an existing report row (locks it from further editing)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET is_signed=1, signed_at=?, signer_id=? WHERE id=? AND is_signed=0",
                (datetime.now().isoformat(), doctor_id, report_id),
            )
            self.log_action(doctor_id, "sign_report", "report", report_id, conn=conn)
        logger.info(f"Report {report_id} signed by {doctor_id}")
        return True

    def create_signed_report(
        self,
        study_id: str,
        signer_id: str,
        report_text: str,
        language: str = "ru",
        ai_draft_text: Optional[str] = None,
        findings_json: Optional[str] = None,
        model_identity_json: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> dict:
        """Persist a signed report for (study, language) in one transaction.

        Raises ReportAlreadySignedError if one is already signed (-> HTTP 409),
        sqlite3.IntegrityError if the study does not exist (-> HTTP 404).

        signer_id is always recorded verbatim (audit trail). doctor_id carries a
        foreign key to users, so it is only set when the signer is a real user
        row — the SENTINEL_DEV_INSECURE anonymous principal has none."""
        report_id = str(uuid.uuid4())[:8]
        signed_at = datetime.now().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            dup = conn.execute(
                "SELECT id FROM reports WHERE study_id=? AND language=? AND is_signed=1",
                (study_id, language),
            ).fetchone()
            if dup:
                conn.rollback()
                raise ReportAlreadySignedError(dup["id"])
            user_row = conn.execute("SELECT id FROM users WHERE id=?", (signer_id,)).fetchone()
            doctor_id = signer_id if user_row else None
            conn.execute(
                """INSERT INTO reports (id, study_id, doctor_id, report_text, ai_draft_text,
                   language, is_signed, signed_at, signer_id, sha256, findings_json,
                   model_identity_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)""",
                (report_id, study_id, doctor_id, report_text, ai_draft_text, language,
                 signed_at, signer_id, sha256, findings_json, model_identity_json,
                 signed_at, signed_at),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "idx_reports_signed_unique" in str(e):
                raise ReportAlreadySignedError(study_id) from e
            raise
        finally:
            conn.close()
        logger.info(f"Report {report_id} signed for study {study_id} ({language}) by {signer_id}")
        return {"report_id": report_id, "study_id": study_id, "signed_at": signed_at,
                "language": language, "sha256": sha256}

    def get_report(self, report_id: str) -> Optional[dict]:
        """Get a report by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    def attach_report_pdf(
        self,
        report_id: str,
        *,
        pdf_path: str,
        pdf_sha256: str,
        dicom_path: str,
        sop_instance_uid: str,
    ) -> bool:
        """Record the stored PDF rendering of a signed report and its DICOM
        Encapsulated PDF twin. Returns False when the report does not exist."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE reports SET pdf_path=?, pdf_sha256=?, dicom_path=?,
                   dicom_sop_instance_uid=?, pdf_attached_at=?, updated_at=? WHERE id=?""",
                (pdf_path, pdf_sha256, dicom_path, sop_instance_uid, now, now, report_id),
            )
            return cur.rowcount > 0

    def record_report_push(self, report_id: str, orthanc_instance_id: str) -> bool:
        """Remember the Orthanc instance id the report's DICOM was stored under."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reports SET orthanc_instance_id=?, pushed_at=?, updated_at=? WHERE id=?",
                (orthanc_instance_id, now, now, report_id),
            )
            return cur.rowcount > 0

    def get_reports_for_study(self, study_id: str) -> list[dict]:
        """All report rows for a study, newest first, with signer info joined."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.*, u.username AS signer_username, u.full_name AS signer_full_name
                   FROM reports r LEFT JOIN users u ON u.id = COALESCE(r.signer_id, r.doctor_id)
                   WHERE r.study_id=? ORDER BY r.created_at DESC""",
                (study_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["is_signed"] = bool(d.get("is_signed"))
            for col, key in (("findings_json", "findings"), ("model_identity_json", "model_identity")):
                try:
                    d[key] = json.loads(d.get(col) or "null")
                except Exception:
                    d[key] = None
            out.append(d)
        return out

    # ===== Corrections =====

    def save_correction(
        self,
        study_id: str,
        doctor_id: str,
        original_report: str,
        corrected_report: str,
        language: str = "ru",
    ) -> str:
        """Persist a radiologist's edit of the AI draft. Returns correction ID."""
        correction_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO corrections (id, study_id, doctor_id, language,
                   original_report, corrected_report) VALUES (?, ?, ?, ?, ?, ?)""",
                (correction_id, study_id, doctor_id, language, original_report, corrected_report),
            )
        return correction_id

    # ===== Users =====

    def create_user(
        self,
        username: str,
        password_hash: str,
        full_name: str,
        role: str = "radiologist",
        must_change_password: bool = False,
    ) -> str:
        """Create a new user. Returns user ID."""
        user_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, full_name, role, must_change_password) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, username, password_hash, full_name, role, int(bool(must_change_password))),
            )
        logger.info(f"User created: {username} ({role})")
        return user_id

    def get_user(self, user_id: str) -> Optional[dict]:
        """Public user fields (no password hash)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, full_name, role, created_at, last_login, must_change_password "
                "FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # ===== Audit Log =====

    def log_action(
        self,
        user_id: str,
        action: str,
        entity_type: str = "",
        entity_id: str = "",
        details: str = "",
        ip_address: str = "",
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Append a row to the tamper-evident audit log. Returns the row id.

        The chain link (prev_hash → row_hash) is computed under the write lock
        (BEGIN IMMEDIATE) so concurrent writers cannot both extend the same
        head. Pass `conn` to append inside a caller's open transaction."""
        own = conn is None
        if own:
            conn = self._connect()
        try:
            if own:
                conn.execute("BEGIN IMMEDIATE")
            head = conn.execute(
                "SELECT id, row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = (head["row_hash"] if head and head["row_hash"] else AUDIT_GENESIS)
            next_id = (head["id"] + 1) if head else 1
            row = {
                "id": next_id, "user_id": user_id, "action": action,
                "entity_type": entity_type, "entity_id": entity_id, "details": details,
                "ip_address": ip_address, "timestamp": datetime.now().isoformat(),
            }
            row_hash = audit_row_hash(row, prev_hash)
            conn.execute(
                "INSERT INTO audit_log (id, user_id, action, entity_type, entity_id, details, "
                "ip_address, timestamp, prev_hash, row_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], user_id, action, entity_type, entity_id, details, ip_address,
                 row["timestamp"], prev_hash, row_hash),
            )
            if own:
                conn.commit()
            return next_id
        except Exception:
            if own:
                conn.rollback()
            raise
        finally:
            if own:
                conn.close()

    def verify_audit_chain(self) -> dict:
        """Walk audit_log in id order and re-derive every hash.

        Returns {ok, rows, checked, head_hash, first_broken: {id, reason}|None}.
        Detects: edited fields, a re-linked/removed interior row, a foreign
        row spliced in. (Silently truncating the newest rows is not detectable
        by the chain alone — the DELETE trigger and DB backups cover that.)"""
        prev = AUDIT_GENESIS
        checked = 0
        first_broken = None
        head_hash = None
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
            for r in rows:
                d = dict(r)
                if not d.get("row_hash"):
                    first_broken = {"id": d["id"], "reason": "row has no hash (written outside log_action)"}
                    break
                if d.get("prev_hash") != prev:
                    first_broken = {"id": d["id"], "reason": "prev_hash does not match previous row"}
                    break
                if audit_row_hash(d, prev) != d["row_hash"]:
                    first_broken = {"id": d["id"], "reason": "row_hash does not match row contents"}
                    break
                prev = d["row_hash"]
                head_hash = prev
                checked += 1
        return {
            "ok": first_broken is None,
            "rows": len(rows),
            "checked": checked,
            "head_hash": head_hash,
            "first_broken": first_broken,
        }

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
