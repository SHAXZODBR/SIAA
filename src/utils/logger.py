"""Logging setup for Sentinel Medical AI.

Logs go to DATA_DIR/logs (rotating, compressed) plus the console. A global
loguru patcher redacts obvious patient identifiers so a stray f-string can
never leak PatientID / PatientName into the log file. Callers must still log
the server-side study_id — never client filenames or patient fields.
"""

import re
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# key=value / key: value forms of the usual PHI fields
_PHI_KV = re.compile(
    r"(?i)\b(patient[_ ]?id|patient[_ ]?name|patientname|patientid|patient_birth_date|birth[_ ]?date|pid)"
    r"\s*[=:]\s*['\"]?([^'\",;|\s]+)"
)
# DICOM Person Name (PN) — 'Last^First[^Middle]' (letters/Cyrillic, ≥2 chars each side)
_PHI_PN = re.compile(r"\b[^\W\d_]{2,}\^[^\W\d_]{2,}(?:\^[^\W\d_]*)*")


def redact_phi(message: str) -> str:
    """Replace obvious patient identifiers in a log line."""
    if not message:
        return message
    out = _PHI_KV.sub(lambda m: f"{m.group(1)}=<redacted>", message)
    out = _PHI_PN.sub("<pn-redacted>", out)
    return out


def _redact_patcher(record: dict) -> None:
    try:
        record["message"] = redact_phi(record["message"])
    except Exception:
        pass


_configured_dir: Optional[Path] = None


def setup_logger(log_dir: Optional[str] = None, level: str = "INFO", force: bool = False):
    """Configure loguru with console + rotating file output under `log_dir`
    (defaults to DATA_DIR/logs). Idempotent: repeated calls for the same
    directory are no-ops unless `force=True`."""
    global _configured_dir

    if log_dir is None:
        from src.utils.paths import LOG_DIR
        log_dir = LOG_DIR
    log_path = Path(log_dir)
    if _configured_dir == log_path and not force:
        return logger
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()
    logger.configure(patcher=_redact_patcher)

    # Console output — colorized
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # File output — full detail
    logger.add(
        log_path / "sentinel_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )

    _configured_dir = log_path
    return logger
