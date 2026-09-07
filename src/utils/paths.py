"""Single source of truth for where Sentinel keeps its mutable state.

Everything that is written at runtime (SQLite DB, rotating logs, doctor
corrections, DICOM cache, training corpus, JWT secret) lives under DATA_DIR —
never inside the repo checkout / app bundle, which may be read-only.

Resolution order:
  1. env SENTINEL_DATA_DIR (the Electron shell passes app.getPath('userData'))
  2. per-platform user-data dir:
       macOS   ~/Library/Application Support/sentinel-medical-ai
       Windows %APPDATA%/sentinel-medical-ai
       Linux   ~/.local/share/sentinel-medical-ai  ($XDG_DATA_HOME honoured)
"""

import os
import sys
from pathlib import Path

APP_DIR_NAME = "sentinel-medical-ai"


def _default_data_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", home)) / APP_DIR_NAME
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / APP_DIR_NAME


DATA_DIR: Path = Path(os.environ.get("SENTINEL_DATA_DIR") or _default_data_dir()).expanduser()

DB_PATH: Path = DATA_DIR / "sentinel.db"
LOG_DIR: Path = DATA_DIR / "logs"
CORRECTIONS_DIR: Path = DATA_DIR / "doctor_corrections"
DICOM_CACHE_DIR: Path = DATA_DIR / "dicom_cache"
TRAINING_CORPUS_DIR: Path = DATA_DIR / "training_corpus"
JWT_SECRET_PATH: Path = DATA_DIR / "jwt_secret.key"
LICENSE_PATH: Path = DATA_DIR / "license.dat"


def ensure_data_dirs() -> Path:
    """Create DATA_DIR and its standard sub-directories. Idempotent."""
    for d in (DATA_DIR, LOG_DIR, CORRECTIONS_DIR, DICOM_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
