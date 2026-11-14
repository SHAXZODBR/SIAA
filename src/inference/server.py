"""FastAPI inference server for Sentinel Medical AI.

Accepts DICOM files via REST API, routes each study to the matching detector
(brain MR panel, head-CT series protocol, single-file registry models) and
returns structured, provenance-stamped analysis results.

Clinic-operations endpoints (v1.1.0, docs/dev/finish_line_contract.md):
  GET/PUT /settings/ai   admin: show/hide unvalidated ('pending') detectors,
                         persisted at DATA_DIR/settings.json (audit-logged)
  GET  /diagnostics      admin: PHI-safe support zip (health, versions, model
                         identities, log tails, redacted audit tail)
  POST /models/import    admin: install a per-clinic encrypted model bundle
                         from DATA_DIR/model_inbox/ (verify → test-decrypt →
                         swap → reload, rollback on any failure, audit-logged)
"""

import io
import os
import re
import sys
import json
import time
import shutil
import sqlite3
import hashlib
import asyncio
import inspect
import platform
import threading
import uuid
import base64
import zipfile
import tempfile
from datetime import datetime
import numpy as np
import torch
import cv2
import pydicom
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, StrictBool
from typing import Any, Optional
from contextlib import asynccontextmanager
from loguru import logger

from src.pipeline.dicom_loader import load_dicom, DicomStudy
from src.pipeline.dicom_objects import is_image_object
from src.pipeline.preprocessor import preprocess_study, resize_with_padding
from src.pipeline.ct_head_selection import (
    select_ct_head_slices, load_windowed_slice, aggregate_slice_probs, largest_non_localizer,
)
from src.training.densenet_trainer import DenseNet121Classifier, get_device
from src.training.gradcam import GradCAM
from src.utils.config import get_config
from src.utils.paths import DATA_DIR, LOG_DIR, CORRECTIONS_DIR, PREVIEWS_DIR, REPO_ROOT, ensure_data_dirs
from src.utils.logger import setup_logger
from src.utils.auth import DEV_INSECURE, REQUIRE_AUTH
from src.utils.offline import OFFLINE

# Server/app version reported in /health and stamped on every analysis
# (desktop-app/package.json carries the same number for the UI build).
APP_VERSION = "1.1.0"

# All mutable state (DB, logs, corrections, JWT secret) lives under DATA_DIR.
# Wire the logger before anything else imports (auth_routes creates the default
# admin at import time and its one-time password must land in the log file).
ensure_data_dirs()
setup_logger(LOG_DIR)


# ==================== Response Models ====================

class Finding(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: str = ''
    location: str = ''
    # API contract v1 — per-finding provenance the desktop panel renders
    finding: Optional[str] = None       # human-readable label (falls back to class_name)
    positive: bool = True               # False = detector explicitly did NOT flag
    status: Optional[str] = None        # validated | pending | experimental
    detector: Optional[str] = None      # registry / panel detector key
    sequence_used: Optional[str] = None


class AnalysisResponse(BaseModel):
    study_id: str
    modality: str
    inference_time_ms: int
    findings: list[Finding]
    overall_impression: str
    normal: bool
    model_version: str
    report_text: Optional[str] = None
    report_language: Optional[str] = None
    gemma_available: bool = False
    preview_base64: Optional[str] = None  # data-URI PNG of the analyzed slice (for the viewer)
    # API contract v1 additions (all optional so the legacy /analyze keeps validating)
    body_part: Optional[str] = None
    overall_assessment: Optional[dict] = None
    disclaimer: str = ''
    model_identity: list[dict] = Field(default_factory=list)
    threshold: Optional[float] = None
    requires_review: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None
    app_version: str = APP_VERSION


class SignRequest(BaseModel):
    """POST /report/sign — the radiologist signs the final report text."""
    study_id: str
    report_text: str
    language: str = 'ru'
    ai_draft_text: Optional[str] = None


class CorrectionRequest(BaseModel):
    """POST /report/save_correction — JSON body (was query params)."""
    study_id: str
    original_report: str
    corrected_report: str
    language: str = 'ru'
    anon_id: Optional[str] = None


class ReportRequest(BaseModel):
    """Request for regenerating report in different language."""
    findings: list[dict]
    language: str = 'ru'
    patient_info: Optional[dict] = None
    modality: str = 'CR'
    body_part: str = 'CHEST'
    # brain MR: the sequence names actually present (the /analyze/study
    # 'sequences_present'); lets the regenerated report keep the same honest
    # technique line as the first draft (D4). None → generic line, no claim.
    sequences: Optional[list[str]] = None


class QuestionRequest(BaseModel):
    """Ask AI a follow-up question about an analysis."""
    question: str
    findings: list[dict]
    report_text: str
    language: str = 'ru'


# ==================== Global State ====================

class ModelState:
    """Holds loaded models and configuration."""
    orthanc_watcher: Optional[object] = None
    model: Optional[DenseNet121Classifier] = None
    gradcam: Optional[GradCAM] = None
    device: Optional[torch.device] = None
    class_names: list[str] = []
    optimal_thresholds: Optional[list] = None  # Per-class optimized thresholds
    is_xrv: bool = False  # True if using TorchXRayVision pre-trained
    legacy_model_reason: str = ''  # why the legacy chest checkpoint is NOT loaded (/health)
    gemma_engine: Optional[object] = None  # Gemma 3 report engine
    data_collector: Optional[object] = None  # Training data collector
    config: dict = {}
    start_time: float = 0.0
    license_valid: bool = True
    license_info: dict = {}
    demo_calls_today: int = 0
    demo_calls_date: str = ''
    # Serializes model forward + Grad-CAM (shared hooks on one model instance).
    # Without this, concurrent requests can cross-wire one patient's heatmap
    # onto another's scan — a patient-safety bug, not just a perf issue.
    inference_lock: object = None


state = ModelState()
state.inference_lock = threading.Lock()


def _audit(user: dict, action: str, entity_type: str, entity_id: str,
           details: str = "", request=None):
    """Write a clinical-action audit entry bound to the authenticated user.
    Captures who/what/when (+ IP) for medical-legal defensibility."""
    try:
        from src.inference.auth_routes import db as _audit_db
        ip = ""
        if request is not None and getattr(request, "client", None):
            ip = request.client.host or ""
        _audit_db.log_action(
            user_id=user.get("sub", "anonymous"),
            action=action, entity_type=entity_type, entity_id=entity_id,
            details=details, ip_address=ip,
        )
    except Exception as e:
        logger.debug(f"audit log failed: {e}")


# ==================== AI settings (D1: pending detectors) ====================
# DATA_DIR/settings.json holds the admin's persisted choices. Only
# 'show_pending_detectors' is defined today; unknown keys are preserved on
# write. Resolution (contract /settings/ai):
#   1. settings.json 'show_pending_detectors' (bool)            source='file'
#   2. env SENTINEL_SHOW_PENDING_DETECTORS = '1' | '0'          source='env'
#   3. '0' when SENTINEL_HOSPITAL_BUILD=1, else '1'             source='default'
# Read on every analysis (a tiny file) so a PUT takes effect for the next
# study without a restart. Env vars are read at call time, not import time.

SETTINGS_FILE_NAME = 'settings.json'
_settings_lock = threading.Lock()
_warned_settings: set = set()


def _settings_path() -> Path:
    return DATA_DIR / SETTINGS_FILE_NAME


def _read_settings_file() -> dict:
    """Parsed DATA_DIR/settings.json, or {} when absent/unreadable/not an object.
    A corrupt file is logged and ignored (the env/default rule then applies)."""
    path = _settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        logger.warning(f"settings: {path.name} unreadable ({type(e).__name__}); using env/default rule")
        return {}
    if not isinstance(data, dict):
        logger.warning(f"settings: {path.name} is not a JSON object; using env/default rule")
        return {}
    return data


def _write_settings_file(data: dict) -> None:
    """Atomic write (tmp + os.replace), owner-only (0600)."""
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    payload = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8')
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _default_show_pending() -> tuple[bool, str]:
    """(value, source) from env/default only — settings.json not consulted."""
    raw = os.environ.get('SENTINEL_SHOW_PENDING_DETECTORS')
    if raw is not None and raw.strip() in ('0', '1'):
        return raw.strip() == '1', 'env'
    if raw is not None and raw.strip() and raw not in _warned_settings:
        _warned_settings.add(raw)       # warn once per value (/health polls this)
        logger.warning(f"SENTINEL_SHOW_PENDING_DETECTORS={raw!r} ignored (expected '1' or '0')")
    hospital = os.environ.get('SENTINEL_HOSPITAL_BUILD') == '1'
    return (not hospital), 'default'


def get_ai_settings() -> dict:
    """{'show_pending_detectors': bool, 'source': 'file'|'env'|'default'}."""
    data = _read_settings_file()
    value = data.get('show_pending_detectors')
    if isinstance(value, bool):
        return {'show_pending_detectors': value, 'source': 'file'}
    if value is not None:
        logger.warning("settings: 'show_pending_detectors' is not a boolean; using env/default rule")
    show, source = _default_show_pending()
    return {'show_pending_detectors': show, 'source': source}


def set_ai_settings(show_pending_detectors: bool) -> tuple[dict, dict]:
    """Persist the setting. Returns (old_settings, new_settings)."""
    with _settings_lock:
        old = get_ai_settings()
        data = _read_settings_file()
        data['show_pending_detectors'] = bool(show_pending_detectors)
        data['updated_at'] = datetime.now().isoformat(timespec='seconds')
        _write_settings_file(data)
        new = get_ai_settings()
    return old, new


def show_pending_detectors() -> bool:
    """Whether unvalidated ('pending') detectors run/are reported right now."""
    try:
        return bool(get_ai_settings()['show_pending_detectors'])
    except Exception as e:      # never let a settings glitch break an analysis
        logger.warning(f"settings: resolution failed ({e}); pending detectors hidden")
        return False


# Localized note for a study that has no validated detector while pending
# detectors are hidden (D1). The desktop renders findings.noValidatedDetector.
NO_VALIDATED_DETECTOR = 'no_validated_detector'
_NO_VALIDATED_NOTES = {
    'en': ('Not analysed by AI: no clinically validated detector is available for this study '
           'type, and unvalidated detectors are switched off (Preferences → AI models). '
           'The radiologist reads the study as usual.'),
    'ru': ('ИИ-анализ не выполнялся: для этого типа исследования нет клинически валидированного '
           'детектора, а невалидированные детекторы отключены (Настройки → Модели ИИ). '
           'Врач-рентгенолог описывает исследование в обычном порядке.'),
    'uz': ('Sun’iy intellekt tahlil qilmadi: ushbu tekshiruv turi uchun klinik tasdiqlangan '
           'detektor yo‘q, tasdiqlanmagan detektorlar esa o‘chirilgan (Sozlamalar → AI modellari). '
           'Rentgenolog tekshiruvni odatdagidek o‘qiydi.'),
}


def _no_validated_note(language: str) -> str:
    return _NO_VALIDATED_NOTES.get(language, _NO_VALIDATED_NOTES['en'])


# A brain MR on which NO panel detector produced an output while pending
# detectors are SHOWN (every model missing / failed, or no usable series for
# any of them). Same not-analysed state as D1, but the D1 note ("unvalidated
# detectors are switched off") would be false here. Desktop: findings.noDetectorRan.
NO_DETECTOR_RAN = 'no_detector_ran'
_NO_DETECTOR_RAN_NOTES = {
    'en': ('Not analysed by AI: no AI detector could run on this study (model not installed or '
           'no suitable series). The radiologist reads the study as usual.'),
    'ru': ('ИИ-анализ не выполнялся: ни один ИИ-детектор не смог обработать это исследование '
           '(модель не установлена или нет подходящей серии). Врач-рентгенолог описывает '
           'исследование в обычном порядке.'),
    'uz': ('Sun’iy intellekt tahlil qilmadi: bu tekshiruvda hech bir AI detektori ishlay olmadi '
           '(model o‘rnatilmagan yoki mos seriya yo‘q). Rentgenolog tekshiruvni odatdagidek o‘qiydi.'),
}

# Reason codes of the "accepted, NOT analysed by AI" state (requires_review,
# rejected=False). Persisted inside overall_assessment so GET /study/{id}
# can return the code after a restart (the studies table has no column for it).
NOT_ANALYSED_REASONS = (NO_VALIDATED_DETECTOR, NO_DETECTOR_RAN)


def _no_detector_ran_note(language: str) -> str:
    return _NO_DETECTOR_RAN_NOTES.get(language, _NO_DETECTOR_RAN_NOTES['en'])


def _persisted_rejection_reason(rejected: bool, requires_review: bool, overall: Any,
                                impression: str) -> Optional[str]:
    """rejection_reason for a persisted study (GET /study/{id}).

    rejected → the persisted impression (the human reason, as before).
    Accepted-but-not-analysed (requires_review, not rejected) → the machine
    code stored in overall_assessment['rejection_reason']; rows written before
    that field existed are recognised by their localized note text.
    Otherwise None."""
    if rejected:
        return impression or None
    if not requires_review:
        return None
    if isinstance(overall, dict):
        code = overall.get('rejection_reason')
        if isinstance(code, str) and code in NOT_ANALYSED_REASONS:
            return code
    text = (impression or '').strip()
    if text and text in _NO_VALIDATED_NOTES.values():
        return NO_VALIDATED_DETECTOR
    if text and text in _NO_DETECTOR_RAN_NOTES.values():
        return NO_DETECTOR_RAN
    return None


# ==================== Encryption at rest (D2) ====================

def _maybe_encrypt_models_at_rest(models_dir: Optional[Path] = None) -> Optional[dict]:
    """Encrypt the proprietary weights under MODELS_DIR on start of an
    installed app. Runs ONLY when SENTINEL_ENCRYPT_MODELS_AT_REST=1 (set by the
    packaged Electron shell), never with SENTINEL_DEV_INSECURE=1 and never on
    macOS inside a .app bundle (signed, read-only). Never raises: startup must
    not die over this — a failure is logged loudly and the plaintext/previous
    state is left to model_crypto's own rollback. Returns the report dict, or
    None when skipped/failed."""
    from src.inference import model_registry as _mr
    models_dir = Path(models_dir) if models_dir is not None else Path(_mr.MODELS_DIR)
    if os.environ.get('SENTINEL_ENCRYPT_MODELS_AT_REST') != '1':
        return None
    if DEV_INSECURE:
        logger.warning("SENTINEL_ENCRYPT_MODELS_AT_REST=1 ignored: SENTINEL_DEV_INSECURE=1")
        return None
    if sys.platform == 'darwin' and any(part.endswith('.app') for part in models_dir.parts):
        logger.info(f"Encryption at rest skipped: models dir is inside a macOS .app bundle ({models_dir})")
        return None
    try:
        from src.utils import model_crypto
        fn = getattr(model_crypto, 'encrypt_proprietary_models_at_rest', None)
        if fn is None:
            logger.error("ENCRYPTION AT REST NOT PERFORMED: model_crypto.encrypt_proprietary_models_at_rest "
                         "is not available in this build")
            return None
        report = fn(models_dir)
    except Exception as e:
        logger.error(f"ENCRYPTION AT REST FAILED ({type(e).__name__}: {e}) — models left as they were; "
                     f"startup continues")
        return None
    _log_encryption_report(report)
    return report if isinstance(report, dict) else {'report': report}


def _log_encryption_report(report: Any) -> None:
    """One loud line per outcome bucket (encrypted / already / skipped /
    rekeyed / failed). model_crypto reports 'failed' as a dict
    {dir_name: '<reason_code>: detail'}; every entry is logged at ERROR and
    suppresses the final 'OK' line (a list-valued failure bucket from an older
    model_crypto is handled the same way)."""
    if not isinstance(report, dict):
        logger.info(f"Encryption at rest: {report!r}")
        return
    failed: list[str] = []
    failed_map = report.get('failed') or {}
    if isinstance(failed_map, dict):
        for name, why in failed_map.items():
            # a key_mismatch dir holds only ciphertext (nothing was changed); anything else
            # was rolled back to its plaintext
            kept = ('ciphertext left untouched' if str(why).startswith('key_mismatch')
                    else 'rolled back, plaintext kept')
            logger.error(f"ENCRYPTION AT REST FAILED — {name}: {why} — {kept}")
            failed.append(str(name))
    for key, value in report.items():
        if key == 'failed' and isinstance(value, dict):
            continue
        items = value if isinstance(value, (list, tuple, set)) else None
        if items is None:
            logger.info(f"Encryption at rest: {key} = {value}")
            continue
        names = [str(getattr(v, 'name', v)) for v in items]
        if any(tok in key.lower() for tok in ('fail', 'error', 'rolled')):
            failed.extend(names)
            if names:
                logger.error(f"ENCRYPTION AT REST — {key}: {len(names)} ({', '.join(names)}) — rolled back, "
                             f"plaintext kept; see model_crypto log lines above")
        else:
            logger.info(f"Encryption at rest — {key}: {len(names)}" + (f" ({', '.join(names)})" if names else ''))
    if not failed:
        logger.info("Encryption at rest: OK — no plaintext proprietary weights left under the models dir")


# ==================== App Lifecycle ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Starting Sentinel Inference Server...")
    state.start_time = time.time()

    # Mutable state root (DB / logs / corrections / JWT secret). Idempotent —
    # also done at import so the auth bootstrap is logged to the same file.
    ensure_data_dirs()
    setup_logger(LOG_DIR)
    logger.info(f"Data dir: {DATA_DIR}")
    # Retention guard for the persisted analyzed-slice previews (PHI on disk).
    try:
        prune_previews()
    except Exception as e:
        logger.warning(f"Preview prune failed: {e}")
    if DEV_INSECURE:
        logger.warning("SENTINEL_DEV_INSECURE=1 — auth relaxed, OpenAPI docs exposed. NOT for a clinic.")
    if OFFLINE:
        logger.info("SENTINEL_OFFLINE=1 — air-gapped: models load from the local bundle only, no downloads")
    else:
        logger.warning("SENTINEL_OFFLINE=0 — downloads allowed (developer mode). NOT for a clinic.")

    # ===== License check =====
    # DEV_BYPASS_LICENSE=1 skips the check, but ONLY together with
    # SENTINEL_DEV_INSECURE=1 — a production box can never be unlocked by a
    # stray env var. Otherwise an invalid license drops to demo-limited mode.
    bypass_license = DEV_INSECURE and os.environ.get('DEV_BYPASS_LICENSE') == '1'
    if os.environ.get('DEV_BYPASS_LICENSE') == '1' and not DEV_INSECURE:
        logger.warning("DEV_BYPASS_LICENSE ignored: requires SENTINEL_DEV_INSECURE=1")
    state.license_valid = True
    state.license_info = {'mode': 'dev', 'reason': 'license check bypassed'}
    if not bypass_license:
        try:
            from src.utils.license import load_and_verify_local_license
            result = load_and_verify_local_license()
            state.license_valid = result.valid
            if result.valid and result.license:
                state.license_info = {
                    'mode': 'licensed',
                    'customer': result.license.customer,
                    'tier': result.license.tier,
                    'expires_at': result.license.expires_at,
                    'days_remaining': result.days_remaining,
                    'features': result.license.features,
                }
                logger.info(
                    f"License OK — {result.license.customer} ({result.license.tier}), "
                    f"expires in {result.days_remaining} days"
                )
            else:
                state.license_info = {'mode': 'unlicensed', 'reason': result.reason}
                logger.warning(f"⚠ License invalid: {result.reason}")
                from src.utils.license import get_license_path
                logger.warning(
                    "  Server will run in DEMO MODE (10 analyses/day). "
                    "To unlock, run: python -m src.utils.license fingerprint, "
                    "send the fingerprint to the vendor, save license.dat to "
                    f"{get_license_path()}"
                )
        except Exception as e:
            # Fail CLOSED: any license-check error drops to demo-limited mode,
            # not full-unlocked. (Fix: previously left license_valid=True.)
            logger.warning(f"License check failed (running in demo mode): {e}")
            state.license_valid = False
            state.license_info = {'mode': 'demo', 'reason': str(e)}

    config = get_config()
    state.config = config
    inference_config = config.get("inference", {})
    classification_config = config.get("classification", {})
    state.class_names = classification_config.get("class_names", [])

    # Device
    state.device = get_device()

    # Load DenseNet121 — supports both custom-trained and pre-trained XRV models
    checkpoint_path = Path(inference_config.get("densenet_checkpoint", "models/densenet/best_model.pt"))
    if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
        checkpoint_path = REPO_ROOT / checkpoint_path      # config path is repo-relative, cwd may differ
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=state.device, weights_only=False)
        num_classes = len(checkpoint.get("class_names", state.class_names))
        state.class_names = checkpoint.get("class_names", state.class_names)

        # Detect if this is a pre-trained TorchXRayVision model
        is_xrv = checkpoint.get("is_pretrained", False) and checkpoint.get("source") == "torchxrayvision"

        if is_xrv:
            # The XRV checkpoint carries only metadata; the weights themselves are
            # torchxrayvision's release file, which xrv fetches from GitHub when it
            # is not on disk. On an air-gapped box that must never be attempted:
            # resolve the file from models/xrv/ (offline bundle) or
            # ~/.torchxrayvision first and degrade with a reason if it is absent.
            try:
                import torchxrayvision as xrv
                from src.inference.model_registry import resolve_xrv_weights
                from src.utils.offline import missing_model_reason
                xrv_weights = checkpoint.get("xrv_weights", "densenet121-res224-all")
                cache_dir, weights_file = resolve_xrv_weights(xrv_weights)
                if weights_file is None and OFFLINE:
                    state.legacy_model_reason = missing_model_reason('chest', [str(Path(cache_dir or 'models/xrv'))])
                    logger.warning(f"Chest model not loaded: {state.legacy_model_reason}")
                    state.model = None
                    state.is_xrv = False
                else:
                    logger.info(f"Loading pre-trained TorchXRayVision: {xrv_weights} ({weights_file or 'download'})")
                    state.model = xrv.models.DenseNet(weights=xrv_weights, cache_dir=cache_dir)
                    state.model.to(state.device)
                    state.model.eval()
                    state.is_xrv = True
            except ImportError:
                # An XRV checkpoint can ONLY be loaded by torchxrayvision — its
                # weight namespace ('features.*') is incompatible with our custom
                # DenseNet121Classifier wrapper ('densenet.features.*'). Trying to
                # force-load would crash with a misleading error, so fail with an
                # actionable message instead.
                state.legacy_model_reason = "torchxrayvision not installed (pip install torchxrayvision)"
                logger.error(
                    "This checkpoint is a TorchXRayVision model but torchxrayvision "
                    "is not installed. Install it (pip install torchxrayvision) and "
                    "restart. Running without the chest model."
                )
                state.model = None
                state.is_xrv = False
            except Exception as e:
                # A failed weight download / corrupt file must degrade, never
                # crash the server lifespan: the brain panel still works.
                state.legacy_model_reason = f"chest weights failed to load: {e}"
                logger.error(f"Chest model not loaded: {e}")
                state.model = None
                state.is_xrv = False
        else:
            # Custom-trained model
            state.model = DenseNet121Classifier(num_classes=num_classes, pretrained=False)
            state.model.load_state_dict(checkpoint["model_state_dict"])
            state.model.to(state.device)
            state.model.eval()
            state.is_xrv = False

        # Initialize Grad-CAM — only if a model actually loaded.
        # XRV models expose 'features.denseblock4'; the custom wrapper exposes
        # 'densenet.features.denseblock4'. Pick the right layer per model type.
        if state.model is not None:
            default_layer = "features.denseblock4" if state.is_xrv else "densenet.features.denseblock4"
            gradcam_layer = config.get("gradcam", {}).get("target_layer", default_layer)
            try:
                state.gradcam = GradCAM(state.model, target_layer_name=gradcam_layer)
            except Exception as e:
                logger.warning(f"Grad-CAM init failed ({e}); heatmaps disabled")
                state.gradcam = None

            # Load optimized per-class thresholds (from train_master.py)
            state.optimal_thresholds = checkpoint.get("optimal_thresholds", None)
            if state.optimal_thresholds:
                logger.info(f"Loaded optimized thresholds for {num_classes} classes")
            else:
                logger.info("No optimized thresholds found, using default 0.5")

            logger.info(f"DenseNet121 loaded from {checkpoint_path} ({num_classes} classes)")
    else:
        state.legacy_model_reason = f"no checkpoint at {checkpoint_path}"
        logger.warning(f"No model checkpoint found at {checkpoint_path}. Server running in demo mode.")

    # D2: encrypt proprietary weights at rest on an installed app BEFORE any
    # model is loaded (after the license check: the key may come from the
    # license's model_key_secret). Never fatal.
    try:
        _maybe_encrypt_models_at_rest()
    except Exception as e:      # belt and braces — the helper already never raises
        logger.error(f"Encryption at rest step crashed: {e}")
    _ai = get_ai_settings()
    logger.info(f"AI settings: show_pending_detectors={_ai['show_pending_detectors']} (source: {_ai['source']})")

    # Warm the brain panel so /health can report real load status (and the
    # first study of the day is not the one that pays the load cost). The
    # local triage model is the product's only validated detector — /health
    # reports 'degraded' without it. SENTINEL_SKIP_WARMUP=1 skips (tests/CLI).
    # A hidden pending detector (D1) is not warmed: it would only hold memory;
    # if an admin switches pending detectors on, it loads on the next study.
    if os.environ.get('SENTINEL_SKIP_WARMUP') != '1':
        from src.inference.model_registry import get_model as _get_model, get_model_identity as _get_identity
        _warm = ('brain_triage', 'brain_tumor_class') if show_pending_detectors() else ('brain_triage',)
        for _key in _warm:
            try:
                _entry = _get_model(_key, device=str(state.device))
                if _entry and _entry.get('available'):
                    logger.info(f"Warmed model '{_key}'")
                    if _key == 'brain_triage':
                        # hash once now so /health can name the weights that will run
                        _ident = _get_identity(_key) or {}
                        logger.info(f"brain_triage identity {_ident.get('sha256_12')} "
                                    f"(encrypted at rest: {bool(_ident.get('encrypted_at_rest'))})")
                else:
                    logger.warning(f"Model '{_key}' not available: {(_entry or {}).get('reason', '')}")
            except Exception as e:
                logger.warning(f"Model '{_key}' warm-up failed: {e}")

    # Load Gemma 3 report engine — auto-detects best available backend
    # (ollama → google_ai → templates)
    try:
        from src.inference.gemma_report_engine import GemmaReportEngine
        state.gemma_engine = GemmaReportEngine(backend='auto')
        logger.info(f"Report engine initialized (backend: {state.gemma_engine.backend})")
    except Exception as e:
        logger.warning(f"Gemma report engine not available: {e}")
        state.gemma_engine = None

    # Initialize training data collector
    try:
        from src.inference.data_collector import TrainingDataCollector
        state.data_collector = TrainingDataCollector()
        logger.info("Training data collector initialized — collecting anonymized data")
    except Exception as e:
        logger.warning(f"Data collector failed: {e}")
        state.data_collector = None

    # ===== Orthanc PACS auto-watcher =====
    # If Orthanc is reachable on localhost, auto-pull new studies and analyze.
    # Disabled by default — set SENTINEL_ORTHANC=1 (or pass --orthanc) to enable.
    state.orthanc_watcher = None
    if os.environ.get('SENTINEL_ORTHANC') == '1':
        try:
            from src.inference.orthanc_watcher import OrthancWatcher, auto_analyze_callback
            orthanc_url = os.environ.get('ORTHANC_URL', 'http://localhost:8042')
            watcher = OrthancWatcher(
                orthanc_url=orthanc_url,
                poll_interval=10,
                on_new_study=auto_analyze_callback,
            )
            if watcher.check_connection():
                watcher.start()
                state.orthanc_watcher = watcher
                logger.info(f"Orthanc watcher started — monitoring {orthanc_url}")
            else:
                logger.warning(
                    f"Orthanc unreachable at {orthanc_url} — watcher not started. "
                    f"To enable later, install Orthanc and restart server."
                )
        except Exception as e:
            logger.warning(f"Orthanc watcher init failed: {e}")

    yield

    # Shutdown
    if state.orthanc_watcher:
        state.orthanc_watcher.stop()
    logger.info("Shutting down Sentinel Inference Server")


# ==================== FastAPI App ====================

# OpenAPI docs are a reconnaissance aid on a clinic network — only exposed in
# SENTINEL_DEV_INSECURE=1 development mode.
app = FastAPI(
    title="Sentinel Medical AI — Inference Server",
    description="Local AI analysis for medical DICOM images",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEV_INSECURE else None,
    redoc_url="/redoc" if DEV_INSECURE else None,
    openapi_url="/openapi.json" if DEV_INSECURE else None,
)

# CORS: locked down by default. The Electron app uses file:// (Origin: null)
# and localhost during dev. Set SENTINEL_CORS_ORIGINS (comma-separated) to add
# more. Wildcard '*' only if SENTINEL_CORS_ALLOW_ALL=1 (NOT for production).
if os.environ.get("SENTINEL_CORS_ALLOW_ALL") == "1":
    _cors_origins = ["*"]
else:
    _cors_origins = [
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev
        "http://localhost:8000", "http://127.0.0.1:8000",
        "null",  # Electron file:// origin
    ] + [o.strip() for o in os.environ.get("SENTINEL_CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],   # PUT: /settings/ai
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Content-Disposition"],            # GET /diagnostics filename
)

# Include auth routes
from src.inference.auth_routes import router as auth_router, clinical_auth, require_admin
from fastapi import Depends, Request
app.include_router(auth_router)


def _probe_llm() -> dict:
    """{backend, reachable} for /health. Ollama is probed live (1 s timeout) so
    the status bar reflects the daemon actually being up, not the startup
    snapshot. 'template' means reports come from deterministic templates —
    no LLM is reachable, which is what we report."""
    engine = state.gemma_engine
    backend = getattr(engine, 'backend', None) if engine is not None else None
    reachable = False
    if backend == 'ollama':
        try:
            import requests
            resp = requests.get(f"{engine.ollama_url}/api/tags", timeout=1)
            reachable = resp.status_code == 200
        except Exception:
            reachable = False
    elif backend == 'google_ai':
        reachable = True
    elif backend == 'transformers':
        reachable = getattr(engine, '_model', None) is not None
    return {'backend': backend, 'reachable': reachable}


def _cached_identity(key: str) -> Optional[str]:
    """sha256_12 of a loaded model if the registry already computed it, else None."""
    try:
        from src.inference import model_registry as _mr
        ident = _mr._identity_cache.get(key) or {}
        return ident.get('sha256_12') or None
    except Exception:
        return None


@app.get("/health")
def health_check():
    """Public health/contract endpoint (API contract v1 — polled by the desktop app).

    status: 'ok' | 'degraded' (validated brain triage model not loaded) | 'error'.
    Sync (threadpool) on purpose: the Ollama probe must not block the event loop.
    """
    from src.inference.model_registry import get_loaded_status

    models = get_loaded_status(['brain_triage', 'brain_tumor_class', 'chest'])
    # The registry 'chest' model loads lazily on the first chest study; the
    # legacy DenseNet/XRV checkpoint (POST /analyze) counts as a loaded chest model.
    if not models['chest']['loaded'] and state.model is not None:
        models['chest'] = {
            'loaded': True,
            'reason': 'legacy checkpoint (' + ('torchxrayvision' if state.is_xrv else 'densenet121') + ')',
        }
    elif not models['chest']['loaded'] and state.legacy_model_reason:
        models['chest'] = {'loaded': False, 'reason': state.legacy_model_reason}
    # identity of the loaded triage weights (first 12 hex of sha256), from the
    # registry cache only — /health is polled and must never hash 350 MB itself.
    if models['brain_triage'].get('loaded'):
        models['brain_triage']['identity'] = _cached_identity('brain_triage')
    status = 'ok' if models['brain_triage']['loaded'] else 'degraded'
    return {
        'status': status,
        'version': APP_VERSION,
        'device': str(state.device) if state.device is not None else None,
        'auth_required': REQUIRE_AUTH,
        'models': models,
        'llm': _probe_llm(),
        'license': {'mode': state.license_info.get('mode')},
        'offline': OFFLINE,
        'data_dir': str(DATA_DIR),
        'ai_settings': get_ai_settings(),
        # legacy fields (scripts/production_smoke_test.py, start.sh)
        'model_loaded': state.model is not None,
        'uptime_seconds': round(time.time() - state.start_time, 1),
    }


@app.get("/orthanc/status")
async def orthanc_status():
    """Status of the Orthanc PACS watcher (if running)."""
    if state.orthanc_watcher is None:
        return {
            'enabled': False,
            'reason': 'Set SENTINEL_ORTHANC=1 env var and restart to enable',
        }
    watcher = state.orthanc_watcher
    return {
        'enabled': True,
        'orthanc_url': watcher.orthanc_url,
        'connected': watcher.check_connection(),
        'known_studies': len(watcher.known_orthanc_ids),
        'orthanc_stats': watcher.get_orthanc_stats(),
    }


@app.get("/audit/verify")
async def audit_verify(admin: dict = Depends(require_admin)):
    """Admin-only: walk the tamper-evident audit hash chain (prev_hash/row_hash)
    and report the first broken row, if any. {ok, rows, checked, head_hash,
    first_broken: {id, reason} | null}."""
    from src.inference.auth_routes import db as _db
    return _db.verify_audit_chain()


@app.get("/audit/log")
async def audit_log_view(limit: int = 200, admin: dict = Depends(require_admin)):
    """Admin-only: recent audit entries for compliance / medical-legal review."""
    from src.inference.auth_routes import db as _db
    return {"entries": _db.get_audit_log(limit=limit)}


def _license_status_dict() -> dict:
    """License STATUS only (never the license file or any secret in it)."""
    return {
        'valid': state.license_valid,
        'info': dict(state.license_info or {}),
        'demo_calls_today': state.demo_calls_today,
        'demo_limit': 10,
    }


@app.get("/license/status")
async def license_status():
    """Return current license info — used by the desktop app to show
    customer name, expiry, and which features are unlocked."""
    return _license_status_dict()


def _ops_error(status_code: int, reason: str, message: str) -> JSONResponse:
    """Uniform error body for the clinic-ops endpoints: a stable reason code
    the desktop maps to a localized message + the English detail for logs."""
    return JSONResponse(status_code=status_code,
                        content={'ok': False, 'reason': reason, 'detail': message})


# ==================== /settings/ai (admin) ====================

class AISettingsUpdate(BaseModel):
    """PUT /settings/ai body. StrictBool: "yes"/1 are rejected (422), not coerced."""
    show_pending_detectors: StrictBool


@app.get("/settings/ai")
def get_settings_ai(admin: dict = Depends(require_admin)):
    """Admin: {'show_pending_detectors': bool, 'source': 'file'|'env'|'default'}."""
    return get_ai_settings()


@app.put("/settings/ai")
def put_settings_ai(req: AISettingsUpdate, request: Request, admin: dict = Depends(require_admin)):
    """Admin: persist show_pending_detectors in DATA_DIR/settings.json. Takes
    effect for the next analysis (no restart). Audit-logged old→new."""
    try:
        old, new = set_ai_settings(req.show_pending_detectors)
    except OSError as e:
        logger.error(f"settings: write of {SETTINGS_FILE_NAME} failed: {e}")
        return _ops_error(500, 'settings_write_failed', f'could not write {SETTINGS_FILE_NAME}: {e}')
    _audit(admin, 'settings_ai_update', 'settings', 'ai',
           details=(f"show_pending_detectors: {str(old['show_pending_detectors']).lower()} "
                    f"({old['source']}) -> {str(new['show_pending_detectors']).lower()} ({new['source']})"),
           request=request)
    logger.info(f"AI settings changed by {admin.get('username') or admin.get('sub')}: "
                f"show_pending_detectors {old['show_pending_detectors']} -> {new['show_pending_detectors']}")
    return new


# ==================== GET /diagnostics (admin) ====================
# A support zip IT can e-mail to the vendor. PHI-safe by construction: only
# status/identity/version data, log tails passed through the PHI and secret
# redactors, an audit tail with patient fields redacted. Never previews, the DB, the
# license file, keys or settings secrets.

DIAG_LOG_TAIL_BYTES = 2 * 1024 * 1024
DIAG_AUDIT_TAIL_ROWS = 200
REDACTED = '[REDACTED]'
_PHI_FIELD_RE = re.compile(
    r'(?i)^(patient[_ ]?(id|name|birth[_ ]?date)|patientid|patientname|patientbirthdate|'
    r'accession([_ ]?(number|no))?|accessionnumber)$')
_PHI_TEXT_RE = re.compile(
    r'(?i)\b(patient[_ ]?id|patient[_ ]?name|patientid|patientname|accession(?:[_ ]?(?:number|no))?|'
    r'accessionnumber)(["\']?\s*[=:]\s*)(["\']?)([^"\',;|\s}]+)')
_SECRET_NAME_RE = re.compile(r'(?i)(secret|passw|token|key|credential|user)')


# Secret-shaped assignments in free text (log tails). The NAME may carry a prefix/
# suffix (ORTHANC_PASSWORD, api_token, model_key_secret, SENTINEL_MODEL_KEY_HEX);
# a bare "key" is NOT a secret name (model registry keys are diagnostic data).
# A match starts only at the start of a name and prefix/suffix are bounded, so one
# long run of name characters (UID lists, base64url) cannot backtrack for minutes.
# Userinfo runs to the LAST '@' before the host (clients split there: u:p@ss@host).
_SECRET_TEXT_NAME = (r'[A-Za-z0-9_.-]{0,40}?(?:pass(?:word|wd|phrase)|pwd|secret|token|api[-_ ]?key|apikey|'
                     r'access[-_]?key|private[-_]?key|licen[cs]e[-_ ]?key|model[-_]?key(?:[-_]?(?:hex|secret))?|'
                     r'credentials?|auth[-_]?key)[A-Za-z0-9_.-]{0,40}')
_SECRET_TEXT_RE = re.compile(
    r'(?i)(?<![A-Za-z0-9_.-])(' + _SECRET_TEXT_NAME + r')(["\']?[ \t]*[=:][ \t]*)'
    r'("(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|'
    # unquoted: the value runs on over spaces until the next "name=" / "name:" or the end of
    # the line ("password=two words" must not leave "words" behind); linear, no nesting
    r'[^\s,;}\]]+(?:[ \t]+(?![A-Za-z0-9_.-]+[ \t]*[=:])[^\s,;}\]]+)*)')
_AUTH_HEADER_RE = re.compile(
    r'(?i)\b((?:proxy-)?authorization["\']?[ \t]*[=:][ \t]*["\']?)(?:(bearer|basic|token|digest)[ \t]+)?[^\s"\',;}]+')
_BEARER_RE = re.compile(r'(?i)\b(bearer)[ \t]+[A-Za-z0-9._~+/=-]{8,}')
_URL_USERINFO_RE = re.compile(r'(?i)(?<![a-z0-9+.-])([a-z][a-z0-9+.-]{0,30}://)[^/\s?#"\']*@')


def _strip_url_userinfo(text: str) -> str:
    """scheme://user:pass@host → scheme://host."""
    return _URL_USERINFO_RE.sub(r'\1', text) if text and '://' in text else text


def _redact_secret_text(text: str) -> str:
    """Secrets in free text → [REDACTED]: password/token/secret/api key/licence and
    model key assignments ("key=value", "key: value", '"key": "value"'), the
    first-run admin banner's "password: …" line, Authorization headers, bearer
    tokens, and userinfo in URLs."""
    if not text:
        return text
    def _assign(m):
        v = m.group(3)
        q = v[0] if v[:1] in ('"', "'") else ''
        return f"{m.group(1)}{m.group(2)}{q}{REDACTED}{q}"
    out = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{(m.group(2) + ' ') if m.group(2) else ''}{REDACTED}", text)
    out = _SECRET_TEXT_RE.sub(_assign, out)
    out = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
    return _strip_url_userinfo(out)


def _redact_text(text: str) -> str:
    """Patient identifiers in free text → [REDACTED] (key=value, key: value,
    "key": "value" and DICOM Person Names)."""
    if not text:
        return text
    from src.utils.logger import redact_phi
    out = redact_phi(text)          # the log patcher's rules (PN names, birth dates, …)
    return _PHI_TEXT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", out)


def _redact_obj(obj: Any) -> Any:
    """Recursively redact PHI fields (by key) and PHI in strings (incl. JSON
    strings such as audit 'details')."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _PHI_FIELD_RE.match(str(k)) and v not in (None, '') else _redact_obj(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_obj(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s[:1] in ('{', '[') :
            try:
                return json.dumps(_redact_obj(json.loads(s)), ensure_ascii=False)
            except ValueError:
                pass
        return _redact_text(obj)
    return obj


def _redact_secrets(obj: Any) -> Any:
    """Values of secret-looking keys (key/secret/token/password/user) → [REDACTED];
    userinfo is stripped from URL-valued strings (ORTHANC_URL=http://u:p@host)."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _SECRET_NAME_RE.search(str(k)) and v not in (None, '') else _redact_secrets(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_secrets(v) for v in obj]
    if isinstance(obj, str):
        return _strip_url_userinfo(obj)
    return obj


def _tail_text(path: Path, max_bytes: int = DIAG_LOG_TAIL_BYTES) -> str:
    """Last `max_bytes` of a text file (cut at a line boundary), PHI- and
    secret-redacted (the first-run admin banner writes the one-time password
    to the log, see src/utils/auth.py)."""
    size = path.stat().st_size
    with open(path, 'rb') as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            data = fh.read(max_bytes)
            nl = data.find(b'\n')
            data = data[nl + 1:] if nl >= 0 else data
        else:
            data = fh.read()
    text = data.decode('utf-8', errors='replace')
    return '\n'.join(_redact_secret_text(_redact_text(line)) for line in text.split('\n'))


def _versions_info() -> dict:
    info = {
        'app': APP_VERSION,
        'python': platform.python_version(),
        'torch': getattr(torch, '__version__', None),
        'os': platform.platform(),
        'machine': platform.machine(),
        'frozen': bool(getattr(sys, 'frozen', False)),
        'device': str(state.device) if state.device is not None else None,
        'offline': OFFLINE,
    }
    for mod in ('transformers', 'pydicom', 'fastapi', 'numpy', 'cryptography'):
        try:
            info[mod] = getattr(__import__(mod), '__version__', None)
        except Exception:
            info[mod] = None
    return info


def _models_info() -> dict:
    from src.inference.model_registry import REGISTRY, get_model_identity
    out = {}
    for key in REGISTRY:
        try:
            out[key] = get_model_identity(key)
        except Exception as e:
            out[key] = {'key': key, 'error': f'{type(e).__name__}: {e}'}
    return out


def _audit_tail(limit: int = DIAG_AUDIT_TAIL_ROWS) -> list[dict]:
    from src.inference.auth_routes import db as _db
    return [_redact_obj(dict(r)) for r in _db.get_audit_log(limit=limit)]


def _settings_redacted() -> dict:
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith(('SENTINEL_', 'ORTHANC_', 'DEV_'))}
    return {'settings_file': _redact_secrets(_read_settings_file()),
            'ai_settings': get_ai_settings(),
            'environment': _redact_secrets(env)}


def build_diagnostics_zip() -> tuple[bytes, list[str]]:
    """(zip bytes, member names). Each section is best-effort: a failing
    section is replaced by {'error': ...} instead of failing the export."""
    def _section(fn):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"diagnostics: section {getattr(fn, '__name__', fn)} failed: {e}")
            return {'error': f'{type(e).__name__}: {e}'}

    members: list[tuple[str, str]] = [
        ('health.json', json.dumps(_section(health_check), indent=2, default=str)),
        ('license.json', json.dumps(_section(_license_status_dict), indent=2, default=str)),
        ('ai_settings.json', json.dumps(_section(get_ai_settings), indent=2)),
        ('versions.json', json.dumps(_section(_versions_info), indent=2, default=str)),
        ('models.json', json.dumps(_section(_models_info), indent=2, default=str)),
    ]
    backend_log = LOG_DIR / 'backend.log'
    if backend_log.is_file():
        try:
            members.append(('logs/backend.log', _tail_text(backend_log)))
        except OSError as e:
            logger.warning(f"diagnostics: backend.log unreadable: {e}")
    app_logs = sorted((p for p in LOG_DIR.glob('sentinel_*.log') if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if app_logs:
        try:
            members.append((f'logs/{app_logs[0].name}', _tail_text(app_logs[0])))
        except OSError as e:
            logger.warning(f"diagnostics: {app_logs[0].name} unreadable: {e}")
    audit = _section(_audit_tail)
    audit_lines = audit if isinstance(audit, list) else [audit]
    members.append(('audit_tail.jsonl',
                    ''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n' for r in audit_lines)))
    members.append(('settings_redacted.json', json.dumps(_section(_settings_redacted), indent=2,
                                                         ensure_ascii=False, default=str)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in members:
            zf.writestr(name, text)
    return buf.getvalue(), [n for n, _ in members]


@app.get("/diagnostics")
def diagnostics(request: Request, admin: dict = Depends(require_admin)):
    """Admin: PHI-safe support zip (see build_diagnostics_zip). Audit-logged."""
    data, names = build_diagnostics_zip()
    filename = f"sentinel-diagnostics-{datetime.now().strftime('%Y%m%d-%H%M')}.zip"
    _audit(admin, 'export_diagnostics', 'system', 'diagnostics',
           details=f"members={len(names)}; bytes={len(data)}", request=request)
    return Response(content=data, media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


# ==================== POST /models/import (admin) ====================
# Per-clinic model delivery (D3): the vendor ships brain_triage_finetuned
# encrypted for THIS machine; the desktop copies the zip into
# DATA_DIR/model_inbox/ and posts its path. Verified, test-decrypted, swapped
# in atomically (previous release kept as …_prev_<identity>), reloaded —
# rolled back on any failure.

MODEL_INBOX_NAME = 'model_inbox'
IMPORT_MODEL_KEY = 'brain_triage'
IMPORT_DIR_NAME = 'brain_triage_finetuned'
MAX_BUNDLE_BYTES = 4 * 1024 ** 3
MAX_BUNDLE_UNCOMPRESSED = 6 * 1024 ** 3
MAX_BUNDLE_MEMBERS = 64
PLAIN_WEIGHTS = 'model.safetensors'
ENC_WEIGHTS = 'model.safetensors.enc'
ENC_META = 'model.safetensors.enc.meta.json'
_REQUIRED_BUNDLE_FILES = ('MANIFEST.json', 'config.json', ENC_WEIGHTS, ENC_META)
_import_lock = threading.Lock()


class ModelImportRequest(BaseModel):
    path: str


class _ModelImportFailure(Exception):
    def __init__(self, status_code: int, reason: str, message: str):
        super().__init__(message)
        self.status_code, self.reason, self.message = status_code, reason, message


def _model_inbox() -> Path:
    return DATA_DIR / MODEL_INBOX_NAME


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _bundle_extractor():
    """packaging/fetch_model_bundle.extract_bundle when that file ships next to
    the code (dev / CI); the frozen backend does not bundle packaging/, so an
    equivalent built-in extractor is used there."""
    src = REPO_ROOT / 'packaging' / 'fetch_model_bundle.py'
    if src.is_file():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('sentinel_fetch_model_bundle', src)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.extract_bundle
        except Exception as e:
            logger.warning(f"model import: packaging/fetch_model_bundle.py not loadable ({e}); "
                           f"using the built-in extractor")
    return _extract_bundle_builtin


def _extract_bundle_builtin(zip_path: Path, dest: Path) -> Path:
    """Same contract as fetch_model_bundle.extract_bundle: dest/<IMPORT_DIR_NAME>
    from either layout (dir at the zip root, or MANIFEST.json at the root).
    Refuses absolute / '..' / drive-letter entries; skips macOS/Windows junk."""
    from pathlib import PurePosixPath
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for info in zf.infolist():
            name = info.filename.replace('\\', '/')
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith('/') or '..' in parts
                    or (len(parts[0]) == 2 and parts[0][1] == ':')):
                raise ValueError(f'refusing unsafe zip entry {info.filename!r}')
            if name.startswith('__MACOSX/') or parts[-1] in ('.DS_Store', 'Thumbs.db') or info.is_dir():
                continue
            members.append(info)
        names = [m.filename.replace('\\', '/') for m in members]
        if any(n.startswith(IMPORT_DIR_NAME + '/') for n in names):
            target = dest
        elif 'MANIFEST.json' in names:
            target = dest / IMPORT_DIR_NAME
        else:
            raise ValueError(f'zip holds neither {IMPORT_DIR_NAME}/ nor a top-level MANIFEST.json')
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target, members=members)
    return dest / IMPORT_DIR_NAME


def _resolve_inbox_bundle(path_str: str) -> Path:
    """The bundle path, validated: absolute, inside DATA_DIR/model_inbox/ (after
    resolving symlinks), an existing .zip within the size cap."""
    if not path_str or not path_str.strip():
        raise _ModelImportFailure(400, 'path_missing', 'no bundle path given')
    p = Path(path_str.strip()).expanduser()
    if not p.is_absolute():
        raise _ModelImportFailure(400, 'path_not_absolute', 'bundle path must be absolute')
    inbox = _model_inbox().resolve()
    try:
        rp = p.resolve()
    except OSError as e:
        raise _ModelImportFailure(400, 'path_invalid', f'cannot resolve bundle path: {e}')
    if rp == inbox or not rp.is_relative_to(inbox):
        raise _ModelImportFailure(400, 'path_outside_inbox',
                                  f'the bundle must be copied into {inbox} first')
    if not rp.is_file():
        raise _ModelImportFailure(404, 'bundle_not_found', f'{rp.name} not found in the model inbox')
    if rp.suffix.lower() != '.zip' or not zipfile.is_zipfile(rp):
        raise _ModelImportFailure(400, 'bundle_not_zip', f'{rp.name} is not a zip archive')
    if rp.stat().st_size > MAX_BUNDLE_BYTES:
        raise _ModelImportFailure(413, 'bundle_too_large', f'{rp.name} exceeds {MAX_BUNDLE_BYTES} bytes')
    return rp


def _check_zip_limits(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_BUNDLE_MEMBERS:
            raise _ModelImportFailure(400, 'bundle_invalid', f'zip holds {len(infos)} entries (max {MAX_BUNDLE_MEMBERS})')
        total = sum(i.file_size for i in infos)
        if total > MAX_BUNDLE_UNCOMPRESSED:
            raise _ModelImportFailure(400, 'bundle_invalid', f'zip expands to {total} bytes (max {MAX_BUNDLE_UNCOMPRESSED})')


def _verify_bundle_dir(model_dir: Path) -> dict:
    """MANIFEST pins (plaintext AND ciphertext sha), sidecar, required files.
    Returns the manifest's files dict."""
    missing = [n for n in _REQUIRED_BUNDLE_FILES if not (model_dir / n).is_file()]
    if missing:
        raise _ModelImportFailure(400, 'bundle_incomplete', f'bundle is missing {", ".join(missing)}')
    from src.utils.model_crypto import plaintext_weight_files, read_meta
    if plaintext_weight_files(model_dir):
        raise _ModelImportFailure(400, 'bundle_contains_plaintext',
                                  'bundle carries plaintext weights; only encrypted bundles are installed')
    try:
        manifest = json.loads((model_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
        files = manifest['files']
        if manifest.get('algorithm', 'sha256') != 'sha256' or not isinstance(files, dict):
            raise ValueError('unsupported manifest')
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise _ModelImportFailure(400, 'manifest_invalid', f'MANIFEST.json unreadable: {e}')
    for pin in (PLAIN_WEIGHTS, ENC_WEIGHTS):
        if not re.fullmatch(r'[0-9a-f]{64}', str(files.get(pin, ''))):
            raise _ModelImportFailure(400, 'manifest_invalid', f'MANIFEST.json does not pin {pin}')
    problems = []
    for name, pinned in files.items():
        if name == PLAIN_WEIGHTS:
            continue                       # plaintext is checked after test-decryption
        rel = Path(str(name))
        if rel.is_absolute() or '..' in rel.parts:
            raise _ModelImportFailure(400, 'manifest_invalid', f'unsafe path in MANIFEST.json: {name!r}')
        fp = model_dir / rel
        if not fp.is_file():
            problems.append(f'{name}: missing')
        elif _sha256_path(fp) != str(pinned):
            problems.append(f'{name}: sha256 does not match the pin')
    if problems:
        raise _ModelImportFailure(400, 'manifest_mismatch', '; '.join(problems))
    try:
        sidecar_sha = read_meta(model_dir / ENC_WEIGHTS).get('plaintext_sha256')
    except (OSError, ValueError) as e:
        raise _ModelImportFailure(400, 'sidecar_invalid', f'{ENC_META} unreadable: {e}')
    if sidecar_sha != files[PLAIN_WEIGHTS]:
        raise _ModelImportFailure(400, 'sidecar_mismatch',
                                  f'{ENC_META} plaintext_sha256 does not match the MANIFEST pin')
    return files


def _test_decrypt(model_dir: Path, plaintext_pin: str) -> None:
    """Decrypt into memory with THIS machine's key and check the plaintext pin."""
    from cryptography.exceptions import InvalidTag
    from src.utils import model_crypto
    try:
        key = model_crypto.get_model_key(refresh=True)
        source = model_crypto.key_source()
    except Exception as e:
        raise _ModelImportFailure(500, 'model_key_unavailable', f'model key unavailable: {e}')
    try:
        plaintext = model_crypto.decrypt_file_to_bytes(model_dir / ENC_WEIGHTS, key=key, verify=True)
    except InvalidTag:
        if source == model_crypto.KEY_SOURCE_INSTALL:
            # No licence with a model_key_secret is active, so the key is this install's
            # local secret — a vendor bundle (HKDF(licence secret ‖ fingerprint), D3)
            # can never decrypt under it. Most likely the licence of the delivery was not
            # imported yet (or the app not restarted): say so instead of "wrong machine".
            raise _ModelImportFailure(400, 'license_model_key_missing',
                                      'no licence with a model key is active on this PC — import the '
                                      'license.dat from the same delivery, restart, then import the '
                                      'bundle again')
        raise _ModelImportFailure(400, 'bundle_not_for_this_machine',
                                  'the bundle was not encrypted for this machine (wrong key), or the '
                                  'installed licence is not the one from the same delivery')
    except ValueError as e:
        raise _ModelImportFailure(400, 'decrypted_hash_mismatch', str(e))
    try:
        if hashlib.sha256(plaintext).hexdigest() != plaintext_pin:
            raise _ModelImportFailure(400, 'decrypted_hash_mismatch',
                                      'decrypted weights do not match the MANIFEST plaintext pin')
    finally:
        del plaintext


def _dir_identity(model_dir: Path) -> str:
    from src.utils.model_crypto import plaintext_sha256_of_dir
    try:
        sha = plaintext_sha256_of_dir(model_dir)
    except Exception:
        sha = None
    return (sha or 'unknown')[:12]


def _evict_registry(key: str) -> None:
    from src.inference import model_registry as _mr
    _mr._loaded_models.pop(key, None)
    _mr._identity_cache.pop(key, None)


def _reload_and_check(models_dir: Path, identity: str) -> dict:
    """Load the installed dir through the registry; it must be THAT dir, the
    pinned identity, encrypted at rest."""
    from src.inference import model_registry as _mr
    _evict_registry(IMPORT_MODEL_KEY)
    entry = _mr.get_model(IMPORT_MODEL_KEY, device=str(state.device or 'cpu'))
    if not entry or not entry.get('available'):
        raise _ModelImportFailure(500, 'reload_failed',
                                  f"model did not load: {(entry or {}).get('reason', 'unknown')}")
    src = Path(entry.get('source_path') or '')
    target = (models_dir / IMPORT_DIR_NAME).resolve()
    if not src or src.resolve() != target:
        raise _ModelImportFailure(500, 'reload_failed', f'registry loaded {src} instead of {target}')
    ident = _mr.get_model_identity(IMPORT_MODEL_KEY) or {}
    if ident.get('sha256_12') != identity:
        raise _ModelImportFailure(500, 'reload_failed',
                                  f"loaded identity {ident.get('sha256_12')} != pinned {identity}")
    if not ident.get('encrypted_at_rest'):
        raise _ModelImportFailure(500, 'reload_failed', 'installed model is not encrypted at rest')
    return ident


def _refresh_validated_metrics(model_dir: Path):
    """Reload the release metrics for the new model: validated_metrics.reload
    refreshes METRICS and every module constant (brain_analysis reads METRICS
    at call time to decide 'validated' vs 'pending'; the report templates read
    it for the quoted figures), and the brain_triage registry card's notes are
    rebuilt so the model list shows the new release's numbers without a restart."""
    from src.inference import validated_metrics as _vm
    from src.inference import model_registry as _mr
    metrics = _vm.reload(model_dir)
    card = _mr.REGISTRY.get(IMPORT_MODEL_KEY)
    if card is not None:
        card.notes = _vm.registry_notes(metrics)
    return metrics


def _encrypt_previous_release(prev_dir: Path) -> None:
    """Plaintext must not linger on a clinic disk (D2): encrypt a kept previous
    release that still has plaintext weights, when encryption at rest is on."""
    if os.environ.get('SENTINEL_ENCRYPT_MODELS_AT_REST') != '1' or DEV_INSECURE:
        return
    try:
        from src.utils import model_crypto
        if model_crypto.plaintext_weight_files(prev_dir):
            # verify-before-delete + rollback, same procedure as the startup pass
            model_crypto.encrypt_dir_at_rest(prev_dir)
            logger.info(f"model import: previous release {prev_dir.name} encrypted at rest")
    except Exception as e:
        logger.error(f"model import: could not encrypt the previous release {prev_dir.name}: {e}")


def import_model_bundle(path_str: str) -> dict:
    """Install a per-clinic encrypted brain_triage bundle (contract POST
    /models/import). Raises _ModelImportFailure(status, reason, message);
    on any failure after the swap the previous release is restored and reloaded."""
    from src.inference import model_registry as _mr

    if os.environ.get(f'SENTINEL_MODEL_DIR_OVERRIDE_{IMPORT_MODEL_KEY.upper()}', '').strip():
        raise _ModelImportFailure(409, 'model_dir_overridden',
                                  f'SENTINEL_MODEL_DIR_OVERRIDE_{IMPORT_MODEL_KEY.upper()} is set; '
                                  f'an imported model would not be used')
    bundle = _resolve_inbox_bundle(path_str)
    models_dir = Path(_mr.MODELS_DIR)
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _ModelImportFailure(500, 'models_dir_not_writable', f'{models_dir}: {e}')
    if not os.access(models_dir, os.W_OK):
        raise _ModelImportFailure(500, 'models_dir_not_writable', f'{models_dir} is not writable')
    _check_zip_limits(bundle)

    # Staging under MODELS_DIR → the final move is a same-filesystem rename.
    staging = models_dir / f'.import_staging_{uuid.uuid4().hex[:12]}'
    current = models_dir / IMPORT_DIR_NAME
    prev: Optional[Path] = None
    swapped = False
    try:
        staging.mkdir(mode=0o700)
        try:
            new_dir = _bundle_extractor()(bundle, staging)
        except (ValueError, zipfile.BadZipFile, OSError) as e:
            raise _ModelImportFailure(400, 'bundle_invalid', f'extract failed: {e}')
        if not new_dir.is_dir():
            raise _ModelImportFailure(400, 'bundle_invalid', f'no {IMPORT_DIR_NAME}/ in the bundle')
        files = _verify_bundle_dir(new_dir)
        identity = str(files[PLAIN_WEIGHTS])[:12]
        _test_decrypt(new_dir, str(files[PLAIN_WEIGHTS]))
        from src.inference import validated_metrics as _vm
        metrics = _vm.load(new_dir)
        status = 'validated' if (metrics.source == 'json' and metrics.validated) else 'pending'

        with state.inference_lock:          # no analysis sees a half-swapped model
            old_identity = _dir_identity(current) if current.exists() else None
            try:
                if current.exists():
                    prev = models_dir / f'{IMPORT_DIR_NAME}_prev_{old_identity}'
                    if prev.exists():
                        prev = models_dir / f"{prev.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    os.replace(current, prev)
                os.replace(new_dir, current)
                swapped = True
                ident = _reload_and_check(models_dir, identity)
            except Exception as e:
                failure = e if isinstance(e, _ModelImportFailure) else \
                    _ModelImportFailure(500, 'install_failed', f'{type(e).__name__}: {e}')
                _rollback_import(models_dir, current, prev, swapped)
                raise failure
        try:
            metrics = _refresh_validated_metrics(current)
        except Exception as e:
            logger.warning(f"model import: release metrics not refreshed: {e}")
        if prev is not None:
            _encrypt_previous_release(prev)
        logger.info(f"MODEL IMPORTED: {IMPORT_DIR_NAME} {old_identity or 'none'} -> {identity} "
                    f"(status {status}, encrypted at rest); previous kept as {prev.name if prev else '-'}")
        return {
            'ok': True,
            'identity': identity,
            'encrypted_at_rest': bool(ident.get('encrypted_at_rest')),
            'status': status,
            'previous_identity': old_identity,
            'previous_dir': prev.name if prev else None,
            'not_validated_reason': None if status == 'validated' else
                (getattr(metrics, 'not_validated_reason', '') or 'no study_level_eval.json in the bundle'),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _rollback_import(models_dir: Path, current: Path, prev: Optional[Path], swapped: bool) -> None:
    """Put the previous release back and reload it (best effort, logged)."""
    try:
        if swapped and current.exists():
            failed = models_dir / f'.import_failed_{uuid.uuid4().hex[:8]}'
            os.replace(current, failed)
            shutil.rmtree(failed, ignore_errors=True)
        if prev is not None and prev.exists() and not current.exists():
            os.replace(prev, current)
    except OSError as e:
        logger.error(f"MODEL IMPORT ROLLBACK INCOMPLETE: {e} — check {models_dir}")
    _evict_registry(IMPORT_MODEL_KEY)
    try:
        from src.inference import model_registry as _mr
        entry = _mr.get_model(IMPORT_MODEL_KEY, device=str(state.device or 'cpu'))
        logger.warning(f"model import rolled back; previous model "
                       f"{'reloaded' if entry and entry.get('available') else 'NOT available'}")
    except Exception as e:
        logger.error(f"model import rollback: reload of the previous model failed: {e}")


@app.post("/models/import")
def models_import(req: ModelImportRequest, request: Request, admin: dict = Depends(require_admin)):
    """Admin: install DATA_DIR/model_inbox/<bundle>.zip as the brain_triage
    model. 200 {ok, identity, encrypted_at_rest, status}; 4xx/5xx {ok: false,
    reason, detail}. Audit-logged either way."""
    if not _import_lock.acquire(blocking=False):
        return _ops_error(409, 'import_in_progress', 'another model import is running')
    try:
        name = Path(req.path or '').name
        try:
            out = import_model_bundle(req.path)
        except _ModelImportFailure as f:
            logger.error(f"model import of {name or '?'} failed: {f.reason}: {f.message}")
            _audit(admin, 'model_import_failed', 'model', IMPORT_MODEL_KEY,
                   details=f"bundle={name}; reason={f.reason}", request=request)
            return _ops_error(f.status_code, f.reason, f.message)
        except Exception as e:
            logger.exception(f"model import of {name or '?'} crashed")
            _audit(admin, 'model_import_failed', 'model', IMPORT_MODEL_KEY,
                   details=f"bundle={name}; reason=internal_error", request=request)
            return _ops_error(500, 'internal_error', f'{type(e).__name__}: {e}')
        _audit(admin, 'model_import', 'model', IMPORT_MODEL_KEY,
               details=(f"bundle={name}; identity {out['previous_identity'] or 'none'} -> {out['identity']}; "
                        f"status={out['status']}; encrypted_at_rest={out['encrypted_at_rest']}"),
               request=request)
        return out
    finally:
        _import_lock.release()


def _enforce_license_or_demo_limit():
    """Block /analyze calls when:
       - the license is invalid AND
       - the user has hit the demo daily limit (10 calls)
    """
    if state.license_valid:
        return
    today = datetime.now().date().isoformat()
    if state.demo_calls_date != today:
        state.demo_calls_date = today
        state.demo_calls_today = 0
    if state.demo_calls_today >= 10:
        raise HTTPException(
            status_code=402,
            detail=(
                'Demo limit reached (10 analyses/day). Activate a license to '
                'continue. Run: python -m src.utils.license fingerprint'
            ),
        )
    state.demo_calls_today += 1


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_dicom(
    file: UploadFile = File(...),
    language: str = "ru",
    user: dict = Depends(clinical_auth("analyze")),
):
    """Analyze a DICOM file and return classification results with heatmaps.

    Accepts a DICOM file upload, runs DenseNet121 classification and Grad-CAM,
    returns findings with confidence scores and heatmap overlays.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check model checkpoint path.")

    _enforce_license_or_demo_limit()

    start_time = time.time()
    study_id = str(uuid.uuid4())[:8]

    # Read uploaded DICOM file
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read DICOM file: {e}")

    # The temp file is removed on EVERY exit path (fix: a load/inference error
    # used to leave a PHI-bearing DICOM behind in the temp dir).
    try:
        return await asyncio.to_thread(_analyze_legacy, tmp_path, study_id, start_time, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _analyze_legacy(tmp_path: str, study_id: str, start_time: float, language: str) -> AnalysisResponse:
    """Body of POST /analyze (legacy single-checkpoint DenseNet/XRV path).
    Blocking — runs in a worker thread so /health keeps answering."""
    # Load DICOM
    study = load_dicom(tmp_path)
    if study is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt DICOM file")

    # Preprocess — XRV uses 224x224, custom uses 512x512
    target_size = 224 if state.is_xrv else 512
    processed_image = preprocess_study(study, target_size=target_size)

    # Convert to tensor: (H, W) → (1, 1, H, W)
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(state.device)

    # XRV expects input in [-1024, 1024] range
    if state.is_xrv:
        # Normalize: our preprocessed image is in [0, 1], scale to XRV range
        input_tensor = input_tensor * 2048 - 1024

    # Run classification + Grad-CAM under a lock — the model + its CAM hooks are
    # shared state; concurrent requests must not interleave or heatmaps get
    # attributed to the wrong patient's scan.
    with state.inference_lock:
        with torch.no_grad():
            logits = state.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        default_threshold = state.config.get("inference", {}).get("confidence_threshold", 0.5)
        findings = []

        for i, (class_name, prob) in enumerate(zip(state.class_names, probs)):
            threshold = state.optimal_thresholds[i] if state.optimal_thresholds else default_threshold
            if prob >= threshold:
                # Generate Grad-CAM heatmap for this class (if available)
                if state.gradcam is not None:
                    heatmap = state.gradcam.generate(input_tensor, target_class=i)
                    heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    _, buffer = cv2.imencode(".png", heatmap_colored)
                    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
                    location = _estimate_location(heatmap)
                else:
                    heatmap_b64 = ""
                    location = "N/A"

                findings.append(Finding(
                    class_name=class_name,
                    confidence=round(float(prob), 4),
                    heatmap_base64=heatmap_b64,
                    location=location,
                ))

    # Sort by confidence descending
    findings.sort(key=lambda f: f.confidence, reverse=True)

    # Overall impression
    is_normal = len(findings) == 0
    if is_normal:
        impression = "No significant pathological findings detected."
    else:
        top_findings = ", ".join(
            f"{f.class_name} ({f.confidence:.0%})" for f in findings[:3]
        )
        impression = f"Findings suggestive of: {top_findings}"

    # Generate Gemma report (if available)
    report_text = None
    gemma_available = False
    if state.gemma_engine is not None:
        try:
            findings_dicts = [
                {
                    'class_name': f.class_name,
                    'confidence': f.confidence,
                    'location': f.location,
                }
                for f in findings
            ]
            report_text = state.gemma_engine.generate(
                findings=findings_dicts,
                language=language,
                patient_info={
                    'id': study.patient_id or 'ANON',
                    'age': study.patient_age or 'N/A',
                    'sex': study.patient_sex or 'N/A',
                },
                modality=study.modality,
                body_part=study.body_part,
                study_date=study.study_date,
            )
            gemma_available = True
        except Exception as e:
            logger.warning(f"Gemma report failed, using template: {e}")
            report_text = None

    inference_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Analysis complete: study={study_id} | "
        f"{len(findings)} findings | {inference_time_ms}ms | "
        f"modality={study.modality} | "
        f"gemma={'yes' if gemma_available else 'no'}"
    )

    # Record in training data corpus (anonymized)
    anon_id = None
    if state.data_collector is not None:
        try:
            anon_id = state.data_collector.record_analysis(
                study_id=study_id,
                dicom_path=tmp_path,
                findings=[{
                    'class_name': f.class_name,
                    'confidence': f.confidence,
                    'location': f.location,
                } for f in findings],
                modality=study.modality,
                body_part=study.body_part,
            )
        except Exception as e:
            logger.warning(f"Training data collection failed: {e}")

    return AnalysisResponse(
        study_id=study_id,
        modality=study.modality,
        inference_time_ms=inference_time_ms,
        findings=findings,
        overall_impression=impression,
        normal=is_normal,
        model_version="densenet121-v1.0",
        report_text=report_text,
        report_language=language,
        gemma_available=gemma_available,
        body_part=study.body_part,
        disclaimer=_localized_disclaimer(language),
    )


# ==================== Gemma Report Endpoints ====================

@app.post("/report/regenerate")
async def regenerate_report(req: ReportRequest, user: dict = Depends(clinical_auth("edit_report"))):
    """Regenerate a report in a different language using Gemma 3."""
    if state.gemma_engine is None:
        raise HTTPException(status_code=503, detail="Gemma engine not available. Install Ollama and pull gemma3:4b")

    try:
        kwargs = dict(
            findings=req.findings,
            language=req.language,
            patient_info=req.patient_info,
            modality=req.modality,
            body_part=req.body_part,
        )
        if req.sequences is not None and _engine_accepts(state.gemma_engine.generate, 'sequences'):
            kwargs['sequences'] = [str(x) for x in req.sequences][:32]
        report = state.gemma_engine.generate(**kwargs)
        return {"report": report, "language": req.language}
    except Exception as e:
        logger.error(f"Report regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report/ask")
async def ask_question(req: QuestionRequest, user: dict = Depends(clinical_auth("view"))):
    """Doctor asks AI a follow-up question about an analysis."""
    if state.gemma_engine is None:
        raise HTTPException(status_code=503, detail="Gemma engine not available")

    try:
        answer = state.gemma_engine.answer_question(
            question=req.question,
            context_findings=req.findings,
            report_text=req.report_text,
            language=req.language,
        )
        return {"answer": answer, "language": req.language}
    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/brain")
async def analyze_brain_panel(
    files: list[UploadFile] = File(...),
    include_experimental: bool = False,
    user: dict = Depends(clinical_auth("analyze")),
):
    """Unified brain MRI analysis — runs the multi-finding detector PANEL
    (tumor, atrophy, …) over one study instead of a single 4-class tumor call.

    - Routes the right sequence to each detector, gates out non-brain studies.
    - Every finding is tagged with a validation status (pending/experimental).
    - `include_experimental=False` (default) hides unvalidated screening hints
      so production output stays trustworthy. Returns a `coverage` roadmap of
      all brain finding types and whether each is wired yet.
    """
    import tempfile, shutil

    _enforce_license_or_demo_limit()
    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided")

    study_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f'sentinel_brain_{study_id}_'))
    try:
        file_paths = []
        for i, f in enumerate(files):
            content = await f.read()
            # Server-chosen name: the client filename is never used on disk
            # (fix: '../../x.dcm' could escape the temp dir).
            fp = tmp_dir / f'{i:04d}.dcm'
            with open(fp, 'wb') as out:
                out.write(content)
            file_paths.append(fp)

        # Pending detectors follow the AI setting (D1) on this route too.
        result = await asyncio.to_thread(_run_brain_panel, file_paths, include_experimental,
                                         show_pending_detectors())
        # Unreadable input is a client error, not a 200 with an 'error' field
        # (same contract as /analyze/study).
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        # The panel hands back the routed slice as a numpy array — encode it
        # for the viewer (it is not JSON-serialisable as-is).
        result['preview_base64'] = _encode_preview(result.pop('_preview_slice', None))
        result['study_id'] = study_id
        result['inference_time_ms'] = int((time.time() - start_time) * 1000)
        _audit(user, 'analyze_brain', 'study', study_id,
               details=f"detectors={result.get('detectors_run')} rejected={result.get('rejected')}")
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/analyze/3d")
async def analyze_3d_brain(
    files: list[UploadFile] = File(...),
    language: str = "ru",
    user: dict = Depends(clinical_auth("analyze")),
):
    """3D brain MRI tumor segmentation.

    Accepts multiple DICOM files from a brain MRI study. Auto-detects which
    sequences are present (T1, T1ce, T2, FLAIR) using DICOM SeriesDescription
    + ProtocolName tags, builds a 4-channel volume, and runs SwinUNETR v2
    (BraTS 2021 architecture) for pixel-level tumor segmentation.

    Returns:
        - per-class voxel counts (enhancing tumor, necrotic core, edema)
        - sequences detected vs needed
        - inference time
    """
    from src.inference.model_registry import get_model
    from src.pipeline.brain_mri_preprocessor import process_brain_study
    import tempfile

    _enforce_license_or_demo_limit()

    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided")

    start_time = time.time()
    study_id = str(uuid.uuid4())[:8]

    # Save uploads to a temp dir
    tmp_dir = Path(tempfile.mkdtemp(prefix=f'sentinel_3d_{study_id}_'))
    file_paths = []
    try:
        for i, f in enumerate(files):
            content = await f.read()
            # Server-chosen name — never the client filename (path traversal).
            fp = tmp_dir / f'{i:04d}.dcm'
            with open(fp, 'wb') as out:
                out.write(content)
            file_paths.append(fp)

        # Build the 3D study (groups by series, detects sequences, normalizes)
        study = process_brain_study(file_paths, target_shape=(128, 128, 128))

        if study.num_sequences == 0:
            raise HTTPException(
                status_code=400,
                detail='No valid DICOM series found in uploaded files',
            )

        sequences_present = list(study.sequences.keys())
        has_quartet = study.has_brats_quartet

        # If we have all 4 BraTS sequences, run 3D inference
        if has_quartet:
            entry = get_model('brain_tumor_seg_3d', device=str(state.device))
            if entry is None or not entry['available']:
                raise HTTPException(
                    status_code=503,
                    detail=f"3D model not available: {(entry or {}).get('reason', '')}",
                )

            volume = study.to_brats_tensor()  # (4, D, H, W)
            predictor = entry['predictor']
            predictions = predictor(volume)
            # Strip ALL internal/non-scalar keys (_mask_array, _probs numpy arrays)
            # before scalar comparison, or `v > 0.001` raises on a numpy array.
            mask_array = predictions.pop('_mask_array', None)
            predictions = {k: v for k, v in predictions.items() if not k.startswith('_')}

            inference_time_ms = int((time.time() - start_time) * 1000)
            findings = [
                {'class_name': k, 'voxel_fraction': round(float(v), 4)}
                for k, v in predictions.items() if v > 0.001 and k != 'background'
            ]
            findings.sort(key=lambda f: -f['voxel_fraction'])

            return {
                'study_id': study_id,
                'modality_path': '3D BraTS segmentation (SwinUNETR v2)',
                'inference_time_ms': inference_time_ms,
                'sequences_present': sequences_present,
                'sequences_required': ['T1', 'T1ce', 'T2', 'FLAIR'],
                'has_quartet': True,
                'findings': findings,
                'volume_shape': list(volume.shape),
                'patient_id': study.patient_id,
                'study_uid': study.study_uid,
            }

        # GATE: if this isn't a brain study (spine/knee/etc on the same scanner),
        # do NOT run the brain-tumor classifier — it would confidently mislabel
        # it (the glioma over-calling we saw on real hospital spine series).
        non_brain = study.non_brain_reason
        if non_brain is not None:
            inference_time_ms = int((time.time() - start_time) * 1000)
            return {
                'study_id': study_id,
                'modality_path': 'rejected — not a brain study',
                'inference_time_ms': inference_time_ms,
                'sequences_present': sequences_present,
                'has_quartet': False,
                'findings': [],
                'requires_review': True,
                'note': (
                    f'This study was not analyzed by the brain-tumor model: {non_brain}. '
                    f'The brain model only applies to brain MRI. Route to the correct module.'
                ),
            }

        # Otherwise fall back to 2D classifier on the CORRECT sequence + slice.
        # Use select_classifier_input() (T1ce/T1 axial, central slice, robust
        # normalize) instead of the old hardcoded FLAIR — the root-cause fix.
        entry = get_model('brain_tumor_class', device=str(state.device))
        if entry is None or not entry['available']:
            raise HTTPException(
                status_code=503,
                detail='2D fallback model not available either',
            )

        sel = study.select_classifier_input()
        if sel is None or sel.get('slice') is None:
            raise HTTPException(status_code=400, detail='No usable axial slice for classification')
        slice_2d = sel['slice']

        from PIL import Image
        # slice_2d is already robustly normalized to [0,1] by the preprocessor
        img = Image.fromarray((np.clip(slice_2d, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0

        probs = entry['predictor'](arr)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:4]
        inference_time_ms = int((time.time() - start_time) * 1000)

        findings = [
            {'class_name': cls, 'confidence': round(float(p), 4)}
            for cls, p in sorted_probs if not cls.startswith('_') and p > 0.05
        ]

        return {
            'study_id': study_id,
            'modality_path': '2D classifier (insufficient sequences for 3D)',
            'inference_time_ms': inference_time_ms,
            'sequences_present': sequences_present,
            'sequences_required': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'has_quartet': False,
            'sequence_used': sel['sequence'],
            'orientation_used': sel['orientation'],
            'warnings': sel['warnings'],
            'findings': findings,
            'note': (
                f"Classified on {sel['sequence']} ({sel['orientation']}). "
                f'For full 3D segmentation, upload all 4: T1, T1ce, T2, FLAIR.'
            ),
        }

    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/report/save_correction")
def save_doctor_correction(
    req: CorrectionRequest,
    request: Request,
    user: dict = Depends(clinical_auth("edit_report")),
):
    """Save the radiologist's edit of the AI draft (JSON body — API contract v1).

    Persisted in the corrections table + an append-only JSONL under DATA_DIR
    (fine-tuning corpus export). The doctor_id comes from the AUTHENTICATED
    user, not a client-supplied string."""
    from src.inference.auth_routes import db

    doctor_id = user.get("sub", "unknown")
    language = _norm_language(req.language)
    correction_id = db.save_correction(
        study_id=req.study_id,
        doctor_id=doctor_id,
        original_report=req.original_report,
        corrected_report=req.corrected_report,
        language=language,
    )

    # Local corrections log (DATA_DIR, never the repo checkout)
    try:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        correction = {
            "correction_id": correction_id,
            "study_id": req.study_id,
            "doctor_id": doctor_id,
            "language": language,
            "original_report": req.original_report,
            "corrected_report": req.corrected_report,
            "timestamp": datetime.now().isoformat(),
        }
        with open(CORRECTIONS_DIR / f"corrections_{language}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(correction, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Corrections log write failed: {e}")

    # Also feed to training corpus
    if state.data_collector is not None and req.anon_id:
        try:
            state.data_collector.record_correction(
                study_id=req.study_id,
                anon_id=req.anon_id,
                ai_report=req.original_report,
                doctor_report=req.corrected_report,
                language=language,
                doctor_id=doctor_id,
            )
        except Exception as e:
            logger.warning(f"Training corpus recording failed: {e}")

    _audit(user, "save_correction", "study", req.study_id,
           details=f"correction={correction_id}; lang={language}", request=request)
    return {"status": "saved", "correction_id": correction_id}


@app.get("/corpus/stats")
async def corpus_stats():
    """Get training data corpus statistics."""
    if state.data_collector is None:
        raise HTTPException(status_code=503, detail="Data collector not available")
    return state.data_collector.get_stats()


# ==================== Multi-Modality Endpoints ====================

@app.get("/models/available")
async def models_available():
    """List every modality the server can analyze and whether it's ready."""
    from src.inference.model_registry import get_availability
    return {"models": get_availability()}


# ==================== Shared routing / inference helpers ====================

_LANGUAGES = ('ru', 'uz', 'en')

# There is NO "normal" model in the product: an empty finding list means the
# panel did not fire, never that the study is normal.
_NOT_NORMAL_IMPRESSION = (
    'No finding flagged by the panel — NOT a normal read; full radiologist review required'
)

# Class names that are a detector's explicit "did not flag" output.
_NEGATIVE_CLASSES = {'no_tumor', 'notumor', 'normal', 'no_finding', 'no finding', 'non_demented'}

_BRAIN_BODY_TOKENS = ('BRAIN', 'HEAD', 'SKULL')

_DISCLAIMERS = {
    'en': (
        'AI triage assistant. It flags only the listed finding types and CANNOT certify a '
        'study as normal. Findings marked "pending" are not validated on this clinic\'s '
        'population; "experimental" findings are screening hints only. Every study is read '
        'and signed by a radiologist.'
    ),
    'ru': (
        'ИИ-ассистент для триажа. Отмечает только перечисленные типы находок и НЕ МОЖЕТ '
        'подтвердить норму. Находки со статусом «pending» не валидированы на популяции '
        'данной клиники; «experimental» — только скрининговые подсказки. Каждое '
        'исследование описывает и подписывает врач-рентгенолог.'
    ),
    'uz': (
        'Sun’iy intellekt triaj yordamchisi. Faqat ro‘yxatdagi topilma turlarini belgilaydi '
        'va tekshiruvni NORMA deb tasdiqlay OLMAYDI. «pending» holatidagi topilmalar ushbu '
        'klinika populyatsiyasida tekshirilmagan; «experimental» — faqat skrining ishorasi. '
        'Har bir tekshiruvni rentgenolog shifokor o‘qiydi va imzolaydi.'
    ),
}


class _NeedsReview(Exception):
    """No honest model match — the study goes to the radiologist without AI (HTTP 422)."""


def _norm_language(language: Optional[str]) -> str:
    return language if language in _LANGUAGES else 'ru'


def _localized_disclaimer(language: str) -> str:
    return _DISCLAIMERS.get(language, _DISCLAIMERS['en'])


def _encode_preview(img: Optional[np.ndarray], max_side: int = 512) -> Optional[str]:
    """Encode a [0,1] float slice as a PNG data-URI for the viewer. This is the
    ACTUAL image the model saw (or the routed classifier slice), never a
    placeholder. Downscaled to `max_side` to keep the JSON payload small."""
    if img is None:
        return None
    try:
        from PIL import Image
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("L")
        if max(pil.size) > max_side:
            pil.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"preview encode failed: {e}")
        return None


def _preview_from_file(path: Optional[Path]) -> Optional[str]:
    """Preview of a single DICOM (fallback when the panel produced no slice)."""
    if path is None:
        return None
    try:
        study = load_dicom(path)
        if study is None:
            return None
        return _encode_preview(preprocess_study(study, target_size=512))
    except Exception as e:
        logger.warning(f"fallback preview failed: {e}")
        return None


def _match_registry_card(modality: str, body_part: str, texts=()) -> Optional[str]:
    """Strict routing for NON-brain single-file models: a registry card matches
    only when BOTH its modality tags and its body-part tags hit (BodyPartExamined,
    or the study/series/protocol description text). Brain cards, 3D models and
    non-classifier backends are excluded. None = no honest match → HTTP 422.
    (detect_modality_from_dicom always falls back to 'chest'; a spine X-ray must
    never be scored by the chest model.)"""
    from src.inference.model_registry import REGISTRY

    modality_u = (modality or '').upper().strip()
    body_u = (body_part or '').upper().strip()
    text_u = ' | '.join(t for t in texts if t).upper()
    best_key, best_score = None, 0
    for key, card in REGISTRY.items():
        if key.startswith('brain_') or card.is_3d or card.backend not in ('xrv', 'huggingface'):
            continue
        if not modality_u or not any(m in modality_u for m in card.modality_dicom_tags):
            continue
        in_body = any(b in body_u for b in card.body_part_dicom_tags)
        in_text = any(b in text_u for b in card.body_part_dicom_tags)
        if not (in_body or in_text):
            continue
        score = (3 if in_body else 2) + (1 if card.tier == 'production' else 0)
        if score > best_score:
            best_key, best_score = key, score
    return best_key


def _route_single_file(modality: str, body_part: str, texts=()) -> Optional[str]:
    """Registry key for one DICOM. Brain MR/CT keep the registry's own routing
    (brain_tumor_class / head_ct); everything else goes through the strict
    matcher and may return None."""
    from src.inference.model_registry import detect_modality_from_dicom

    modality_u = (modality or '').upper()
    body_u = (body_part or '').upper()
    is_brain = any(b in body_u for b in _BRAIN_BODY_TOKENS)
    if is_brain and (any(m in modality_u for m in ('MR', 'MRI')) or 'CT' in modality_u):
        return detect_modality_from_dicom(modality, body_part)
    return _match_registry_card(modality, body_part, texts)


def _engine_accepts(fn, name: str) -> bool:
    """True when callable `fn` takes keyword `name` (or **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _generate_report(findings: list[dict], language: str, modality: str,
                     body_part: str, study_date: str = '',
                     sequences: Optional[list[str]] = None) -> tuple[Optional[str], bool]:
    """(report_text, engine_used). Patient identifiers are never passed to the
    engine (PHI minimisation — the prompt may reach an Ollama host).
    `sequences` (brain MR: the sequence names actually present, e.g.
    T1/T1ce/T2/FLAIR/DWI) lets the engine build an honest technique line (D4);
    it is passed only when the engine's generate() accepts it."""
    if state.gemma_engine is None:
        return None, False
    try:
        kwargs = dict(
            findings=[
                {'class_name': f.get('class_name'), 'confidence': f.get('confidence'),
                 'location': f.get('location', ''), 'positive': f.get('positive'),
                 'status': f.get('status'), 'detector': f.get('detector')}
                for f in findings
            ],
            language=language,
            patient_info=None,
            modality=modality,
            body_part=body_part,
            study_date=study_date or '',
        )
        if sequences is not None and _engine_accepts(state.gemma_engine.generate, 'sequences'):
            kwargs['sequences'] = list(sequences)
        text = state.gemma_engine.generate(**kwargs)
        return (text or None), bool(text)
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return None, False


# Registry keys analysed as a SERIES (central axial slices) rather than one
# file on POST /analyze/study. The RSNA hemorrhage classifier needs axial
# brain slices, and a whole head-CT upload also carries the scout/topogram,
# bone and reformat series — "the largest image" was the scout.
_CT_SERIES_ROUTES = {'head_ct'}
CT_HEAD_N_SLICES = 9
CT_HEAD_AGGREGATION = 'mean_central_9'
# Softmax (single-label) models report their top-3 classes at or above this
# floor (pre-existing behaviour); whether a class is POSITIVE is decided by
# card.decision_threshold, not by being reported.
_SOFTMAX_REPORT_FLOOR = 0.15


def _load_registry_entry(modality_key: str) -> dict:
    """Loaded registry entry for a key, or HTTP 503 with the loader's reason."""
    from src.inference.model_registry import get_model

    model_entry = get_model(modality_key, device=str(state.device))
    if model_entry is None or not model_entry['available']:
        reason = (model_entry or {}).get('reason', 'model not registered')
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{modality_key}' not available: {reason}",
        )
    return model_entry


def _clean_probs(raw) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(raw).items() if not str(k).startswith('_')}


def _decide_findings(card, modality_key: str, probs: dict[str, float], language: str, *,
                     multi_label: Optional[bool] = None, threshold: Optional[float] = None,
                     location: str = '', report_all: bool = False,
                     sequence_used: Optional[str] = None) -> list[dict]:
    """Class probabilities → findings. A class is POSITIVE only when its
    probability ≥ the decision threshold (card.decision_threshold unless
    overridden) and it is not a detector's explicit negative class.

      multi-label (independent sigmoids): every class is reported, each
        judged on its own — sub-threshold classes come back positive=False
        so the UI shows them as "not flagged".
      softmax (single-label): the top-3 classes ≥ _SOFTMAX_REPORT_FLOOR are
        reported (as before); only those ≥ threshold are positive.
    `report_all` reports every class regardless of mode."""
    thr = float(card.decision_threshold if threshold is None else threshold)
    is_multi = bool(card.multi_label if multi_label is None else multi_label)
    localized = card.classes_localized or {}

    def _finding(cls: str, prob: float) -> dict:
        return {
            'class_name': cls,
            'finding': (localized.get(cls) or {}).get(language) or cls,
            'confidence': round(prob, 4),
            'positive': bool(prob >= thr) and cls.lower() not in _NEGATIVE_CLASSES,
            'status': card.validation_status,
            'detector': modality_key,
            'sequence_used': sequence_used,
            'heatmap_base64': '',
            'location': location,
        }

    ranked = sorted(probs.items(), key=lambda x: -x[1])
    if is_multi or report_all:
        chosen = ranked
    else:
        chosen = [(c, p) for c, p in ranked[:3] if p >= _SOFTMAX_REPORT_FLOOR]
    findings = [_finding(c, p) for c, p in chosen]
    findings.sort(key=lambda f: (not f['positive'], -f['confidence']))
    return findings


def _study_flag(card, probs: dict[str, float], findings: list[dict],
                threshold: Optional[float] = None) -> tuple[bool, Optional[str]]:
    """(abnormal_flagged, flag_class). card.study_flag_class (RSNA 'any') is
    the study-level flag when the model outputs it; otherwise any positive
    finding flags the study."""
    thr = float(card.decision_threshold if threshold is None else threshold)
    cls = card.study_flag_class
    if cls and cls in probs:
        return bool(probs[cls] >= thr), cls
    return any(f['positive'] for f in findings), None


def _summarize_findings(card, probs: dict[str, float], findings: list[dict],
                        threshold: Optional[float] = None) -> tuple[str, dict]:
    """(impression, overall_assessment). Never declares a study normal."""
    flagged, flag_cls = _study_flag(card, probs, findings, threshold)
    positives = [f for f in findings if f['positive']]
    if flagged:
        parts = [f"{f['class_name']} ({f['confidence']:.0%})" for f in positives[:3]]
        if flag_cls and flag_cls not in [f['class_name'] for f in positives[:3]]:
            parts.insert(0, f"{flag_cls} ({probs[flag_cls]:.0%})")
        impression = "Findings suggestive of: " + ", ".join(parts)
    else:
        impression = _NOT_NORMAL_IMPRESSION
    overall_assessment = {
        'abnormal_flagged': bool(flagged),
        'flags': [f['finding'] for f in positives],
        'text': impression,
    }
    if flag_cls:
        overall_assessment['study_flag'] = {'class': flag_cls, 'confidence': round(probs[flag_cls], 4)}
    return impression, overall_assessment


def _run_registry_model(study: DicomStudy, modality_key: str, language: str) -> dict:
    """Core of the single-file registry path — shared by /analyze/auto and the
    non-brain branch of /analyze/study (chest X-ray etc.; head CT goes through
    _run_ct_head_series first). Blocking: call from a worker thread. Never
    declares a study normal."""
    from src.inference.model_registry import get_model_identity

    model_entry = _load_registry_entry(modality_key)
    card = model_entry['card']
    predictor = model_entry['predictor']
    is_multi = bool(model_entry.get('multi_label', card.multi_label))

    # Preprocess to the model's input size; the preview is exactly what it saw.
    processed_image = preprocess_study(study, target_size=card.input_size[0])
    preview_base64 = _encode_preview(processed_image)

    with state.inference_lock:
        raw_probs = _clean_probs(predictor(processed_image))

    threshold = float(card.decision_threshold)
    findings = _decide_findings(
        card, modality_key, raw_probs, language, multi_label=is_multi,
        location='Region of interest' if is_multi else 'Central region',
    )
    impression, overall_assessment = _summarize_findings(card, raw_probs, findings)

    report_text, engine_used = _generate_report(
        findings, language, modality=study.modality,
        body_part=study.body_part or card.display_name, study_date=study.study_date,
    )
    identity = get_model_identity(modality_key)
    return {
        'findings': findings,
        'impression': impression,
        'overall_assessment': overall_assessment,
        'preview_base64': preview_base64,
        'report_text': report_text,
        'gemma_available': engine_used,
        'card': card,
        'threshold': threshold,
        'model_identity': [identity] if identity else [],
        'activation': model_entry.get('activation') or ('sigmoid' if is_multi else 'softmax'),
        'aggregation': 'single_image',
        'n_slices_analyzed': 1,
        'series_used': None,
    }


def _run_ct_head_series(file_paths: list[Path], modality_key: str, language: str,
                        hdr: Optional[dict] = None,
                        n_slices: int = CT_HEAD_N_SLICES) -> Optional[dict]:
    """Head-CT series protocol for the head_ct route (blocking; worker thread).

    Picks the axial brain series (never the scout/localizer), scores up to
    `n_slices` central slices in the brain window, aggregates per class by
    MEAN (max reported in per_class_max) and previews the central slice.
    Returns None when the upload holds no analysable series (single-file
    uploads) — the caller then falls back to the single-image path."""
    from src.inference.model_registry import get_model_identity

    sel = select_ct_head_slices(file_paths, n_slices=n_slices)
    if sel is None:
        return None
    model_entry = _load_registry_entry(modality_key)
    card = model_entry['card']
    predictor = model_entry['predictor']
    is_multi = bool(model_entry.get('multi_label', card.multi_label))
    size = int(card.input_size[0])

    # Decode each slice once: windowed at native resolution (the central one
    # doubles as the preview), letter-boxed to the classifier input for scoring.
    loaded = []
    for idx, path in zip(sel.indices, sel.slice_paths):
        got = load_windowed_slice(path)
        if got is None:
            continue
        img, meta = got
        loaded.append((idx, path, img, meta))
    if not loaded:
        logger.warning(f"head_ct: none of the {len(sel.slice_paths)} selected slices decoded")
        return None

    per_slice = []
    with state.inference_lock:
        for idx, path, img, meta in loaded:
            probs = _clean_probs(predictor(resize_with_padding(img, size)))
            per_slice.append({
                'index': int(idx),
                'instance_number': meta.get('instance_number'),
                'probs': {k: round(v, 4) for k, v in probs.items()},
            })
    mean_probs, max_probs = aggregate_slice_probs([
        {k: float(v) for k, v in s['probs'].items()} for s in per_slice
    ])

    threshold = float(card.decision_threshold)
    findings = _decide_findings(
        card, modality_key, mean_probs, language, multi_label=is_multi,
        report_all=True, location='', sequence_used=sel.series.description or None,
    )
    impression, overall_assessment = _summarize_findings(card, mean_probs, findings)

    # Preview: the central slice, same window the model saw, at viewer resolution.
    central_img, central = next(((im, m) for i, p, im, m in loaded if p == sel.central_path),
                                loaded[len(loaded) // 2][2:4])
    preview_base64 = _encode_preview(resize_with_padding(central_img, 512))

    protocol = {
        'series_used': sel.series.description,
        'series_uid': sel.series.uid,
        'n_slices_analyzed': len(per_slice),
        'n_images_in_series': sel.series.n_images,
        'aggregation': CT_HEAD_AGGREGATION,
        'window': {k: central.get(k) for k in ('window_center', 'window_width', 'default_brain_window')},
        'slice_rows_cols': [central.get('rows'), central.get('cols')],
    }
    overall_assessment['protocol'] = protocol
    overall_assessment['per_class_max'] = {k: round(v, 4) for k, v in max_probs.items()}

    report_text, engine_used = _generate_report(
        findings, language, modality=(hdr or {}).get('modality') or 'CT',
        body_part=(hdr or {}).get('body_part') or 'HEAD',
        study_date=(hdr or {}).get('study_date') or '',
    )
    identity = get_model_identity(modality_key)
    logger.info(
        f"head_ct series protocol: '{sel.series.description}' {len(per_slice)} slices "
        f"(of {sel.series.n_images}); mean={ {k: round(v, 3) for k, v in mean_probs.items()} }"
    )
    return {
        'findings': findings,
        'impression': impression,
        'overall_assessment': overall_assessment,
        'preview_base64': preview_base64,
        'report_text': report_text,
        'gemma_available': engine_used,
        'card': card,
        'threshold': threshold,
        'model_identity': [identity] if identity else [],
        'activation': model_entry.get('activation') or ('sigmoid' if is_multi else 'softmax'),
        **protocol,
        'per_class_max': overall_assessment['per_class_max'],
        'per_slice': per_slice,
        'series_dropped': sel.dropped,
        'series_candidates': sel.candidates,
    }


@app.post("/analyze/auto", response_model=AnalysisResponse)
async def analyze_auto(
    file: UploadFile = File(...),
    language: str = "ru",
    force_modality: Optional[str] = None,
    request: Request = None,
    user: dict = Depends(clinical_auth("analyze")),
):
    """Single-file multi-modality analysis. Routes by DICOM tags (brain MR/CT,
    chest, head_ct, mammography, …). Pass `force_modality=brain_2d` etc. to
    override. Returns 422 {requires_review} when no registered model honestly
    matches the modality + body part — the study then goes to the radiologist
    without an AI read. Never returns normal=true.
    """
    from src.inference.model_registry import REGISTRY

    _enforce_license_or_demo_limit()
    language = _norm_language(language)

    start_time = time.time()
    study_id = str(uuid.uuid4())[:8]

    # Read uploaded file
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        # Load DICOM (handles non-DICOM gracefully)
        study = load_dicom(tmp_path)
        if study is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupt DICOM file")

        # Decide which model to use
        if force_modality and force_modality in REGISTRY:
            modality_key = force_modality
        else:
            modality_key = _route_single_file(
                study.modality, study.body_part,
                (study.study_description, study.series_description),
            )
        if modality_key is None:
            reason = (
                f"No registered model for modality '{study.modality}' / body part "
                f"'{study.body_part}' — study routed to radiologist review without AI analysis"
            )
            _audit(user, "analyze", "study", study_id,
                   details=f"model=none; requires_review; lang={language}", request=request)
            return JSONResponse(status_code=422, content={
                'detail': reason, 'requires_review': True, 'reason': reason,
                'study_id': study_id, 'modality': study.modality, 'body_part': study.body_part,
            })

        logger.info(
            f"Auto-routing study {study_id}: modality={study.modality}, "
            f"body_part={study.body_part} → model='{modality_key}'"
        )
        card = REGISTRY[modality_key]
        if card.validation_status != 'validated' and not show_pending_detectors():
            # D1: the routed model is not validated and pending detectors are hidden.
            note = _no_validated_note(language)
            _audit(user, "analyze", "study", study_id,
                   details=f"model={modality_key}; requires_review; {NO_VALIDATED_DETECTOR}; lang={language}",
                   request=request)
            return AnalysisResponse(
                study_id=study_id, modality=f"{study.modality}/{modality_key}",
                inference_time_ms=int((time.time() - start_time) * 1000),
                findings=[], overall_impression=note, normal=False,
                model_version=card.display_name, report_language=language,
                preview_base64=_preview_from_file(Path(tmp_path)),
                body_part=study.body_part,
                overall_assessment={'abnormal_flagged': False, 'flags': [], 'text': note},
                disclaimer=_localized_disclaimer(language),
                requires_review=True, rejection_reason=NO_VALIDATED_DETECTOR,
            )
        run = await asyncio.to_thread(_run_registry_model, study, modality_key, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    inference_time_ms = int((time.time() - start_time) * 1000)

    # Audit: who ran which model on which study, and the top finding.
    top = run['findings'][0]['class_name'] if run['findings'] else "none"
    _audit(user, "analyze", "study", study_id,
           details=f"model={modality_key}; top={top}; lang={language}",
           request=request)

    return AnalysisResponse(
        study_id=study_id,
        modality=f"{study.modality}/{modality_key}",
        inference_time_ms=inference_time_ms,
        findings=[Finding(**f) for f in run['findings']],
        overall_impression=run['impression'],
        normal=False,
        model_version=f"{run['card'].display_name}",
        report_text=run['report_text'],
        report_language=language,
        gemma_available=run['gemma_available'],
        preview_base64=run['preview_base64'],
        body_part=study.body_part,
        overall_assessment=run['overall_assessment'],
        disclaimer=_localized_disclaimer(language),
        model_identity=run['model_identity'],
        threshold=run['threshold'],
    )


# ==================== Whole-study analysis (API contract v1) ====================

_HEADER_FIELDS = (
    ('PatientID', 'patient_id'),
    ('PatientName', 'patient_name'),
    ('PatientSex', 'patient_sex'),
    ('PatientAge', 'patient_age'),
    ('PatientBirthDate', 'patient_birth_date'),
    ('StudyDate', 'study_date'),
    ('StudyDescription', 'study_description'),
    ('AccessionNumber', 'accession_number'),
    ('StudyInstanceUID', 'study_instance_uid'),
    ('Modality', 'modality'),
    ('BodyPartExamined', 'body_part'),
    ('Manufacturer', 'manufacturer'),
    ('ManufacturerModelName', 'scanner_model'),
)

_STUDY_HEADER_KEYS = (
    'study_instance_uid', 'accession_number', 'patient_id', 'patient_name', 'patient_sex',
    'patient_age', 'patient_birth_date', 'study_date', 'study_description',
    'manufacturer', 'scanner_model',
)


def _tag(ds, name: str) -> str:
    try:
        v = ds.get(name)
    except Exception:
        return ''
    return '' if v is None else str(v).strip()


def _read_study_headers(file_paths: list[Path]) -> Optional[dict]:
    """Header-only pass (stop_before_pixels) over the uploaded files.

    Patient/study fields (incl. modality / body part) come from the first IMAGE
    object in upload order (is_image_object) — a PACS export often puts an SR /
    PR / key-object note first, whose Modality 'SR' and empty body part would
    leave a brain MR unrouted. Only when the upload holds no image object does
    the first readable DICOM decide. 'readable' counts every readable object.
    SeriesDescription and ProtocolName are collected unique across all files (upload order).
    'best_path' is the largest image (Rows×Columns) among the image objects — the
    single-file fallback and preview; a dose-report / screenshot secondary capture
    wins only when the upload holds no acquired image at all.
    'series_groups' groups the files by SeriesInstanceUID with each series'
    own BodyPartExamined / description (D5: mixed-body-part CT studies are
    routed per series, see _plan_series_route); 'is_image' marks the groups of
    image objects (non-image objects / secondary captures get groups of their own).
    Returns None when nothing parses as DICOM."""
    hdr = None
    hdr_from_image = False
    readable = 0
    series, protocols = [], []
    best_path, best_px = None, -1
    best_image_path, best_image_px = None, -1
    groups: dict[str, dict] = {}
    for order, p in enumerate(file_paths):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        # force=True parses almost anything — require a real DICOM signal
        if not (_tag(ds, 'SOPClassUID') or _tag(ds, 'Modality')):
            continue
        readable += 1
        is_image = is_image_object(ds)
        # First image object wins; a non-image object only stands in until one appears.
        if not hdr_from_image and (hdr is None or is_image):
            hdr = {out: _tag(ds, tag) for tag, out in _HEADER_FIELDS}
            hdr_from_image = is_image
        sd = _tag(ds, 'SeriesDescription')
        if sd and sd not in series:
            series.append(sd)
        pn = _tag(ds, 'ProtocolName')
        if pn and pn not in protocols:
            protocols.append(pn)
        try:
            px = int(ds.get('Rows') or 0) * int(ds.get('Columns') or 0)
        except Exception:
            px = 0
        if px > best_px:
            best_path, best_px = p, px
        if is_image and px > best_image_px:
            best_image_path, best_image_px = p, px
        _add_to_series_group(groups, ds, Path(p), order, px, is_image)
    if hdr is None:
        return None
    hdr['readable'] = readable
    hdr['series_descriptions'] = series
    hdr['protocol_names'] = protocols
    hdr['best_path'] = best_image_path if best_image_path is not None else best_path
    hdr['series_groups'] = _finish_series_groups(groups)
    return hdr


# ---- D5: per-series body-part routing (mixed CT studies) ----
# A CT study can hold several body parts (e.g. HEAD + CHEST). The study-level
# header comes from the FIRST file, so routing on it sent a head+chest study
# wherever the first series pointed and let head_ct pick a chest series. Each
# series is classified by its own BodyPartExamined, else by tokens in its
# SeriesDescription / ProtocolName.
_HEAD_REGION_TOKENS = ('HEAD', 'BRAIN', 'SKULL', 'CRANI', 'ГОЛОВ', 'ЧЕРЕП')
_NON_HEAD_REGION_TOKENS = ('CHEST', 'THORAX', 'LUNG', 'ABDOM', 'PELVI', 'SPINE', 'NECK',
                           'ГРУД', 'ЛЕГК', 'ОГК', 'БРЮШ', 'ПОЗВОН', 'ТАЗ')


def _series_region(body_part: str, description: str, protocol: str = '') -> tuple[Optional[str], str]:
    """('head'|'other'|None, label) for one series. BodyPartExamined wins when
    present; otherwise head tokens, then non-head tokens, in the series'
    description/protocol. None = the series carries no body-part information."""
    bp = (body_part or '').upper().strip()
    if bp:
        if any(t in bp for t in _HEAD_REGION_TOKENS):
            return 'head', 'HEAD'
        return 'other', bp
    text = f'{description or ""} | {protocol or ""}'.upper()
    if any(t in text for t in _HEAD_REGION_TOKENS):
        return 'head', 'HEAD'
    for t in _NON_HEAD_REGION_TOKENS:
        if t in text:
            return 'other', t
    return None, ''


def _add_to_series_group(groups: dict, ds, path: Path, order: int, px: int,
                         is_image: Optional[bool] = None) -> None:
    """Add one object to its series group. Image objects and non-image objects /
    secondary captures (is_image_object) never share a group, even under one
    SeriesInstanceUID: each group's 'is_image' says which kind it holds."""
    if is_image is None:
        is_image = is_image_object(ds)
    uid = _tag(ds, 'SeriesInstanceUID') or f'__nouid__{order}'
    key = uid if is_image else f'__non_image__{uid}'
    g = groups.get(key)
    if g is None:
        g = groups[key] = {
            'uid': uid, 'order': order, 'description': '', 'protocol': '', 'body_part': '',
            'paths': [], 'best_path': None, 'best_px': -1, 'is_image': bool(is_image),
        }
    for key, tag in (('description', 'SeriesDescription'), ('protocol', 'ProtocolName'),
                     ('body_part', 'BodyPartExamined')):
        if not g[key]:
            g[key] = _tag(ds, tag)
    g['paths'].append(path)
    if px > g['best_px']:
        g['best_path'], g['best_px'] = path, px


def _finish_series_groups(groups: dict) -> list[dict]:
    out = []
    for g in sorted(groups.values(), key=lambda x: x['order']):
        region, label = _series_region(g['body_part'], g['description'], g['protocol'])
        out.append({**g, 'region': region, 'region_label': label, 'n_images': len(g['paths'])})
    return out


def _group_series(file_paths: list[Path]) -> list[dict]:
    """Series groups for callers holding a hand-made hdr without 'series_groups'."""
    groups: dict[str, dict] = {}
    for order, p in enumerate(file_paths):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not (_tag(ds, 'SOPClassUID') or _tag(ds, 'Modality')):
            continue
        try:
            px = int(ds.get('Rows') or 0) * int(ds.get('Columns') or 0)
        except Exception:
            px = 0
        _add_to_series_group(groups, ds, Path(p), order, px, is_image_object(ds))
    return _finish_series_groups(groups)


def _series_brief(g: dict) -> dict:
    return {'description': g.get('description') or '', 'body_part': g.get('body_part') or '',
            'region': g.get('region'), 'n_images': g.get('n_images', len(g.get('paths') or []))}


def _plan_series_route(hdr: dict, file_paths: list[Path]) -> dict:
    """Which registry model analyses a NON-brain-MR study, and on which files.

    CT studies whose series carry body-part information are routed per series
    (D5): any head series → head_ct on the head series ONLY; otherwise the
    first informed series' body part picks the model, analysed on the series
    of that same body part. Series with NO body-part information are added to
    the route when every informed series agrees on that body part (an
    unambiguous study; e.g. a HEAD-tagged scout + an untagged axial reformat),
    and excluded in a mixed study. Only when no series carries body-part
    information does routing fall back to the study header (first image object) over
    all files. Series informed as another body part never reach head_ct.
    Series groups holding no image object (SR / PR / KO, a secondary-capture
    "Dose Report" tagged NECK) take no part: they are never informed nor adopted
    and contribute no paths, so they cannot turn a head CT into NECK, make it
    look mixed, or become its preview. Only an upload with no image series at
    all is routed on every group, as before.

    Returns {'modality_key', 'paths', 'best_path', 'body_part', 'routing'}."""
    modality = hdr.get('modality') or ''
    texts = [hdr.get('study_description') or ''] + list(hdr.get('series_descriptions') or []) \
        + list(hdr.get('protocol_names') or [])
    fallback = {
        'modality_key': _match_registry_card(modality, hdr.get('body_part'), texts),
        'paths': list(file_paths),
        'best_path': hdr.get('best_path'),
        'body_part': hdr.get('body_part') or '',
        'routing': {'mode': 'study_header'},
    }
    if 'CT' not in modality.upper():
        return fallback
    groups = hdr.get('series_groups')
    if groups is None:
        groups = _group_series(file_paths)
    # A hand-made group without 'is_image' counts as an image series (as before).
    image_groups = [g for g in groups if g.get('is_image', True)]
    if image_groups and len(image_groups) < len(groups):
        ignored = {Path(p) for g in groups if not g.get('is_image', True) for p in g['paths']}
        groups = image_groups
        fallback['paths'] = [p for p in file_paths if Path(p) not in ignored]
    informed = [g for g in groups if g.get('region')]
    if not informed:
        if fallback['modality_key'] in _CT_SERIES_ROUTES:
            fallback['routing'] = {'mode': 'first_series_fallback',
                                   'reason': 'no series carries body-part information'}
        return fallback

    head = [g for g in informed if g['region'] == 'head']
    if head:
        chosen, body_part = head, 'HEAD'
        key = _match_registry_card(modality, 'HEAD', [g.get('description') or '' for g in head])
    else:
        label = informed[0]['region_label']
        chosen = [g for g in informed if g['region'] == 'other' and g['region_label'] == label]
        body_part = informed[0].get('body_part') or label
        key = _match_registry_card(modality, body_part,
                                   [g.get('description') or '' for g in chosen]
                                   + [g.get('protocol') or '' for g in chosen])
        if key in _CT_SERIES_ROUTES:       # defensive: a non-head series never reaches head_ct
            key = None
    # Series WITHOUT body-part information (no BodyPartExamined, no region token —
    # typical of workstation reformats) join the route only when every informed
    # series agrees on the chosen body part: then the study is unambiguously that
    # body part and dropping them would leave e.g. only a HEAD-tagged scout for
    # head_ct. In a genuinely mixed study they stay excluded (could be either).
    adopted: list[dict] = []
    if len(chosen) == len(informed):
        adopted = [g for g in groups if not g.get('region')]
    routed = chosen + adopted
    paths = [p for g in routed for p in g['paths']]
    best = max(routed, key=lambda g: g.get('best_px', -1))
    excluded = [g for g in groups if not any(g is c for c in routed)]
    routing = {
        'mode': 'per_series_body_part',
        'series_routed': [_series_brief(g) for g in routed],
        'series_excluded': [_series_brief(g) for g in excluded],
    }
    if adopted:
        routing['series_adopted_without_body_part'] = [_series_brief(g) for g in adopted]
        logger.info(f"series routing: {len(adopted)} series without body-part information "
                    f"adopted into the {body_part} route (every informed series is {body_part})")
    if excluded:
        logger.info(f"series routing: {len(routed)} {body_part} series → '{key}'; "
                    f"{len(excluded)} other series excluded "
                    f"({', '.join((g.get('body_part') or g.get('region_label') or '?') for g in excluded)})")
    return {'modality_key': key, 'paths': paths, 'best_path': best.get('best_path'),
            'body_part': body_part, 'routing': routing}


def _is_brain_mr(hdr: dict) -> bool:
    """MR + brain by BodyPartExamined, else by brain/non-brain tokens in the
    study, series and protocol descriptions (the panel gates again per series)."""
    from src.pipeline.brain_mri_preprocessor import is_brain_text

    modality_u = (hdr.get('modality') or '').upper()
    if not any(m in modality_u for m in ('MR', 'MRI')):
        return False
    if any(b in (hdr.get('body_part') or '').upper() for b in _BRAIN_BODY_TOKENS):
        return True
    study_verdict = is_brain_text(hdr.get('study_description') or '')
    if study_verdict is True:
        return True
    if study_verdict is False:
        return False
    texts = list(hdr.get('series_descriptions') or []) + list(hdr.get('protocol_names') or [])
    return any(is_brain_text(t) is True for t in texts)


def _hidden_brain_detectors(show_pending: bool) -> set[str]:
    """Panel detector keys switched off by the AI setting (D1): every detector
    whose STATIC status is 'pending' when pending detectors are hidden.
    (A triage model downgraded to 'pending' at run time because its eval did
    not pass the gate keeps running — its findings already say 'pending'.)"""
    if show_pending:
        return set()
    from src.inference.brain_analysis import DETECTORS
    return {d.key for d in DETECTORS if d.status == 'pending'}


def _run_brain_panel(file_paths: list[Path], include_experimental: bool,
                     show_pending: bool) -> dict:
    """analyze_brain_study under the inference lock, honouring the pending-
    detector setting. When brain_analysis accepts `detector_statuses` the
    hidden detectors are not run at all; until then they still execute
    (~0.2 s, and the panel's preview slice may be the one the hidden tumor
    classifier saw) and are removed afterwards by _apply_detector_visibility."""
    from src.inference.brain_analysis import analyze_brain_study

    kwargs: dict = {'device': str(state.device or 'cpu'), 'include_experimental': include_experimental}
    try:
        params = inspect.signature(analyze_brain_study).parameters
    except (TypeError, ValueError):
        params = {}
    if 'detector_statuses' in params:
        allowed = {'validated', 'pending'} if show_pending else {'validated'}
        if include_experimental:
            allowed.add('experimental')
        kwargs['detector_statuses'] = allowed
    with state.inference_lock:
        result = analyze_brain_study(file_paths, **kwargs)
    return _apply_detector_visibility(result, _hidden_brain_detectors(show_pending))


def _apply_detector_visibility(result: dict, hidden: set[str]) -> dict:
    """Remove hidden detectors' findings / detectors_run entries, mark hidden
    coverage rows, and rebuild overall_assessment from what remains (same
    wording as brain_analysis). Idempotent; a no-op when nothing is hidden."""
    if not hidden or result.get('error') or result.get('rejected'):
        if hidden:
            result['detectors_hidden'] = sorted(hidden)
        return result
    from src.inference.brain_analysis import DETECTORS

    ran_before = list(result.get('detectors_run') or [])
    findings = [f for f in (result.get('findings') or []) if f.get('detector') not in hidden]
    detectors_run = [d for d in ran_before if d not in hidden]
    removed = [d for d in ran_before if d in hidden]
    positives = [f for f in findings if f.get('positive')]
    covered = [d.label for d in DETECTORS if d.key in detectors_run]
    result['findings'] = findings
    result['detectors_run'] = detectors_run
    result['detectors_hidden'] = sorted(hidden)
    result['detectors_suppressed'] = removed
    result['overall_assessment'] = {
        'abnormal_flagged': bool(positives),
        'flags': [f.get('finding') for f in positives],
        'text': (
            'ABNORMAL — flagged for review: ' + '; '.join(str(f.get('finding')) for f in positives)
            if positives else
            'No finding flagged by the panel. This is NOT a normal read — the panel only '
            f'detects [{", ".join(covered)}]. Any other pathology requires full radiologist review.'
        ),
    }
    if result.get('coverage'):
        result['coverage'] = [
            {**c, 'wired': False, 'hidden': True} if (c.get('key') in hidden or c.get('status') == 'pending')
            else c
            for c in result['coverage']
        ]
    return result


def _not_analysed_resp(resp: dict, route: str, preview_path: Optional[Path], *,
                       reason: str, note: str, hidden: list[str]) -> dict:
    """Accepted into the worklist, NOT analysed by AI (requires_review, not
    rejected, no findings, no draft report). `reason` is one of
    NOT_ANALYSED_REASONS; it is also stored inside overall_assessment so the
    persisted row keeps it (GET /study/{id} after a restart)."""
    resp.update({
        'route': route,
        'requires_review': True,
        'rejected': False,
        'rejection_reason': reason,
        'note': note,
        'findings': [],
        'detectors_run': [],
        'model_identity': [],
        'overall_assessment': {'abnormal_flagged': False, 'flags': [], 'text': note,
                               'rejection_reason': reason},
        'overall_impression': note,
        'preview_base64': _preview_from_file(preview_path),
        'report_text': None,
        'gemma_available': False,
    })
    if hidden:
        resp['detectors_hidden'] = sorted(set(hidden))
    return resp


def _no_validated_detector_resp(resp: dict, route: str, language: str,
                                preview_path: Optional[Path], hidden: list[str]) -> dict:
    """D1: accepted into the worklist, not analysed by AI (requires_review)."""
    _not_analysed_resp(resp, route, preview_path, reason=NO_VALIDATED_DETECTOR,
                       note=_no_validated_note(language), hidden=hidden)
    resp['detectors_hidden'] = sorted(set(hidden))
    logger.info(f"study routed to '{route}' — no validated detector and pending detectors hidden: "
                f"requires_review ({NO_VALIDATED_DETECTOR})")
    return resp


def _no_detector_ran_resp(resp: dict, route: str, language: str,
                          preview_path: Optional[Path]) -> dict:
    """Pending detectors shown, yet no panel detector produced an output."""
    _not_analysed_resp(resp, route, preview_path, reason=NO_DETECTOR_RAN,
                       note=_no_detector_ran_note(language), hidden=[])
    logger.warning(f"study routed to '{route}' — no detector could run: requires_review ({NO_DETECTOR_RAN})")
    return resp


def _analyze_study_sync(file_paths: list[Path], hdr: dict, language: str,
                        study_id: str) -> dict:
    """Blocking core of POST /analyze/study (runs in a worker thread).

    Brain MR → the detector panel (validated triage; pending tumor class only
    when the AI setting shows pending detectors). Anything else → per-series
    routing (_plan_series_route) to a registry model, or _NeedsReview. A route
    whose model is not validated while pending detectors are hidden returns
    requires_review / rejection_reason='no_validated_detector' (D1)."""
    from src.inference.brain_analysis import DETECTORS
    from src.inference.model_registry import get_model_identity, REGISTRY

    show_pending = show_pending_detectors()
    resp = {
        'study_id': study_id,
        **{k: (hdr.get(k) or None) for k in _STUDY_HEADER_KEYS},
        'modality': hdr.get('modality') or 'UNKNOWN',
        'body_part': hdr.get('body_part') or '',
        'num_files': len(file_paths),
        'series_descriptions': list(hdr.get('series_descriptions') or []),
        'rejected': False,
        'requires_review': False,
        'rejection_reason': None,
        'findings': [],
        'detectors_run': [],
        'model_identity': [],
        'overall_assessment': None,
        'overall_impression': '',
        'normal': False,                       # the product never certifies normal
        'disclaimer': _localized_disclaimer(language),
        'threshold': 0.5,
        'preview_base64': None,
        'report_text': None,
        'report_language': language,
        'gemma_available': False,
        'app_version': APP_VERSION,
        'show_pending_detectors': show_pending,
    }

    if _is_brain_mr(hdr):
        result = _run_brain_panel(file_paths, include_experimental=False, show_pending=show_pending)
        preview_slice = result.pop('_preview_slice', None)
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])

        resp['route'] = 'brain_panel'
        resp['body_part'] = 'BRAIN'
        sequences = list(result.get('sequences_present', []) or [])
        resp['sequences_present'] = sequences
        if result.get('rejected'):
            reason = result.get('non_brain_reason') or 'not a brain study'
            note = result.get('note') or f'Not analyzed by the brain panel: {reason}.'
            resp.update({
                'rejected': True,
                'requires_review': True,
                'rejection_reason': reason,
                'overall_assessment': {'abnormal_flagged': False, 'flags': [], 'text': note},
                'overall_impression': note,
                'preview_base64': _preview_from_file(hdr.get('best_path')),
            })
            return resp

        if not result.get('detectors_run'):
            # No detector produced an output — an empty panel is NOT an analysed study, so
            # never a "nothing flagged" impression or a draft report. Causes: only hidden
            # (pending) detectors could run; the validated triage model is not installed
            # (clinic flavour before the bundle import) or failed to load/predict; or the
            # study has no axial series (outside the validated protocol). The panel's notes
            # say which, and go into the response for support.
            hidden = result.get('detectors_hidden') or []
            if hidden:
                out = _no_validated_detector_resp(resp, 'brain_panel', language,
                                                  hdr.get('best_path'), hidden)
            else:
                out = _no_detector_ran_resp(resp, 'brain_panel', language, hdr.get('best_path'))
            out['panel_notes'] = [str(n) for n in (result.get('notes') or [])]
            for n in out['panel_notes']:
                logger.info(f"brain panel ({study_id}): {n}")
            return out

        model_keys = {d.key: d.model_key for d in DETECTORS}
        # Panel findings already carry status/detector/positive/sequence_used.
        # The panel does not localise a lesion — location stays empty rather
        # than a made-up region.
        findings = [{**f, 'heatmap_base64': '', 'location': ''} for f in result.get('findings', [])]
        detectors_run = list(result.get('detectors_run', []))
        identities = [get_model_identity(model_keys[d]) for d in detectors_run if d in model_keys]
        overall = result.get('overall_assessment') or {
            'abnormal_flagged': False, 'flags': [], 'text': _NOT_NORMAL_IMPRESSION,
        }
        preview = _encode_preview(preview_slice) if preview_slice is not None \
            else _preview_from_file(hdr.get('best_path'))
        report_text, engine_used = _generate_report(
            findings, language, modality='MR', body_part='BRAIN',
            study_date=hdr.get('study_date') or '', sequences=sequences,
        )
        resp.update({
            'findings': findings,
            'detectors_run': detectors_run,
            'model_identity': [i for i in identities if i],
            'overall_assessment': overall,
            'overall_impression': overall.get('text', ''),
            'preview_base64': preview,
            'report_text': report_text,
            'gemma_available': engine_used,
            'has_brats_quartet': bool(result.get('has_brats_quartet')),
            'coverage': result.get('coverage', []),
        })
        if result.get('detectors_hidden'):
            resp['detectors_hidden'] = result['detectors_hidden']
        return resp

    # ---- non-brain: per-series routing, then the series protocol / single file ----
    plan = _plan_series_route(hdr, file_paths)
    modality_key = plan['modality_key']
    resp['routing'] = plan['routing']
    if modality_key is None:
        raise _NeedsReview(
            f"No registered model for modality '{hdr.get('modality') or '?'}' / body part "
            f"'{plan['body_part'] or hdr.get('body_part') or '?'}' — study routed to radiologist "
            f"review without AI analysis"
        )
    route_paths = plan['paths'] or list(file_paths)
    route_hdr = {**hdr, 'body_part': plan['body_part'] or hdr.get('body_part') or ''}
    card = REGISTRY.get(modality_key)
    if card is not None and card.validation_status != 'validated' and not show_pending:
        preview_path = largest_non_localizer(route_paths) or plan['best_path'] or hdr.get('best_path')
        resp['body_part'] = route_hdr['body_part'] or resp['body_part']
        return _no_validated_detector_resp(resp, modality_key, language, preview_path, [modality_key])

    run = None
    best_path = plan['best_path'] or hdr.get('best_path')
    if modality_key in _CT_SERIES_ROUTES:
        # Whole head-CT study: axial brain series, central slices, never the scout
        # and never a series of another body part (D5).
        run = _run_ct_head_series(route_paths, modality_key, language, route_hdr)
        if run is None:
            # No series with enough images (single-file upload) — single-image
            # path, but still never a localizer when anything else decodes.
            best_path = largest_non_localizer(route_paths) or best_path
            logger.warning(f"head_ct: no analysable series in {len(route_paths)} file(s); "
                           f"falling back to single image {getattr(best_path, 'name', best_path)}")
    if run is None:
        study = load_dicom(best_path) if best_path else None
        if study is None:
            raise HTTPException(status_code=400, detail='Could not decode pixel data from the uploaded DICOM files')
        run = _run_registry_model(study, modality_key, language)

    card = run['card']
    resp.update({
        'route': modality_key,
        'body_part': route_hdr['body_part'] or (card.body_part_dicom_tags[0] if card.body_part_dicom_tags else ''),
        'findings': run['findings'],
        'detectors_run': [modality_key],
        'model_identity': run['model_identity'],
        'overall_assessment': run['overall_assessment'],
        'overall_impression': run['impression'],
        'preview_base64': run['preview_base64'],
        'report_text': run['report_text'],
        'gemma_available': run['gemma_available'],
        'threshold': run['threshold'],
        'activation': run.get('activation'),
        # Series protocol provenance (single-image runs: series_used=None, 1 slice).
        'series_used': run.get('series_used'),
        'n_slices_analyzed': run.get('n_slices_analyzed', 1),
        'aggregation': run.get('aggregation', 'single_image'),
    })
    for extra in ('series_uid', 'n_images_in_series', 'per_class_max', 'per_slice',
                  'series_dropped', 'series_candidates', 'window', 'slice_rows_cols'):
        if extra in run:
            resp[extra] = run[extra]
    return resp


# ==================== Preview persistence ====================
# The analyzed slice (preview_base64) is what the doctor sees next to the
# findings. It is written to DATA_DIR/previews/<study_id>.png (0600, dir 0700)
# so GET /study/{id} can show the scan after a restart; only the path + sha256
# land in SQLite. PHI on disk -> pruned after SENTINEL_PREVIEW_RETENTION_DAYS.

DEFAULT_PREVIEW_RETENTION_DAYS = 365
_DATA_URI_PNG = "data:image/png;base64,"


def _ensure_private_dir(path: Path) -> Path:
    """mkdir -p `path` and keep it owner-only (0700): report PDFs / previews carry PHI."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _write_private(path: Path, data: bytes) -> None:
    """Create/overwrite `path` with mode 0600 (report PDFs and previews carry PHI)."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(data)
    os.chmod(path, 0o600)


def _preview_file(study_id: str) -> Path:
    return PREVIEWS_DIR / f"{study_id}.png"


def _save_preview(study_id: str, data_uri: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Persist a preview data-URI as DATA_DIR/previews/<study_id>.png.
    Returns (path, sha256), or (None, None) when there is nothing to save or
    the write failed — the analysis is still persisted, just without its image."""
    if not data_uri or not data_uri.startswith(_DATA_URI_PNG):
        return None, None
    try:
        png = base64.b64decode(data_uri[len(_DATA_URI_PNG):], validate=True)
        if not png:
            return None, None
        _ensure_private_dir(PREVIEWS_DIR)
        path = _preview_file(study_id)
        _write_private(path, png)
        return str(path), hashlib.sha256(png).hexdigest()
    except Exception as e:
        logger.warning(f"Preview for study {study_id} not persisted: {e}")
        return None, None


def _load_preview(path: Optional[str], sha256: Optional[str] = None) -> Optional[str]:
    """Read a persisted preview back as a data-URI. None when it is gone
    (pruned / never written) or no longer matches its recorded sha256 — a
    wrong image next to a finding is worse than 'image unavailable'."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        png = p.read_bytes()
    except OSError as e:
        logger.warning(f"Preview {p.name} unreadable: {e}")
        return None
    if sha256 and hashlib.sha256(png).hexdigest() != sha256:
        logger.warning(f"Preview {p.name} does not match its recorded sha256 — not served")
        return None
    return _DATA_URI_PNG + base64.b64encode(png).decode()


def _preview_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).is_file()


def _preview_retention_days() -> int:
    raw = os.environ.get('SENTINEL_PREVIEW_RETENTION_DAYS')
    if raw is None or not raw.strip():
        return DEFAULT_PREVIEW_RETENTION_DAYS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"SENTINEL_PREVIEW_RETENTION_DAYS={raw!r} is not an integer; "
                       f"using {DEFAULT_PREVIEW_RETENTION_DAYS}")
        return DEFAULT_PREVIEW_RETENTION_DAYS


def prune_previews(retention_days: Optional[int] = None) -> dict:
    """Delete previews older than `retention_days` (default: env
    SENTINEL_PREVIEW_RETENTION_DAYS, else 365) by file mtime; 0 removes all.
    Called at startup. The ai_results row keeps preview_path/sha256 as the
    record of what was shown — GET /study/{id} then returns preview_base64=null.
    Returns {'retention_days', 'removed', 'kept', 'errors'}."""
    days = _preview_retention_days() if retention_days is None else max(0, int(retention_days))
    stats = {'retention_days': days, 'removed': 0, 'kept': 0, 'errors': 0}
    if not PREVIEWS_DIR.is_dir():
        logger.info(f"Preview retention ({days} days): no previews dir yet at {PREVIEWS_DIR}")
        return stats
    cutoff = time.time() - days * 86400
    for p in PREVIEWS_DIR.glob('*.png'):
        try:
            if p.stat().st_mtime <= cutoff:
                p.unlink()
                stats['removed'] += 1
            else:
                stats['kept'] += 1
        except OSError as e:
            stats['errors'] += 1
            logger.warning(f"Preview prune: could not process {p.name}: {e}")
    logger.info(f"Preview retention ({days} days): removed {stats['removed']}, "
                f"kept {stats['kept']}, errors {stats['errors']} in {PREVIEWS_DIR}")
    return stats


def _persist_analysis(resp: dict, hdr: dict, user: dict, inference_time_ms: int) -> bool:
    """studies + ai_results rows for the worklist / GET /study. The preview
    PNG goes to DATA_DIR/previews (path + sha256 on the row); heatmaps are not
    persisted (findings carry heatmap_base64='')."""
    from src.inference.auth_routes import db

    try:
        db.create_study(
            patient_id=hdr.get('patient_id') or '',
            modality=resp['modality'],
            body_part=resp.get('body_part') or '',
            dicom_path='',
            study_date=hdr.get('study_date') or None,
            study_id=resp['study_id'],
            created_by=user.get('sub'),
            study_instance_uid=hdr.get('study_instance_uid') or None,
            accession_number=hdr.get('accession_number') or None,
            patient_name=hdr.get('patient_name') or None,
            patient_sex=hdr.get('patient_sex') or None,
            patient_age=hdr.get('patient_age') or None,
            patient_birth_date=hdr.get('patient_birth_date') or None,
            study_description=hdr.get('study_description') or None,
            manufacturer=hdr.get('manufacturer') or None,
            scanner_model=hdr.get('scanner_model') or None,
            num_files=resp.get('num_files'),
            series_descriptions=resp.get('series_descriptions') or [],
            rejected=resp.get('rejected', False),
            requires_review=resp.get('requires_review', False),
        )
        model_version = ', '.join(
            m.get('display_name') or m.get('key') or '' for m in resp.get('model_identity') or []
        ) or str(resp.get('route') or '')
        preview_path, preview_sha = _save_preview(resp['study_id'], resp.get('preview_base64'))
        db.save_ai_result(
            study_id=resp['study_id'],
            findings_json=json.dumps(resp.get('findings') or [], ensure_ascii=False),
            inference_time_ms=inference_time_ms,
            model_version=model_version,
            is_normal=False,
            overall_impression=resp.get('overall_impression') or '',
            model_identity_json=json.dumps(resp.get('model_identity') or [], ensure_ascii=False),
            overall_json=json.dumps(resp.get('overall_assessment'), ensure_ascii=False),
            threshold=resp.get('threshold'),
            report_text=resp.get('report_text'),
            report_language=resp.get('report_language'),
            app_version=APP_VERSION,
            preview_path=preview_path,
            preview_sha256=preview_sha,
        )
        return True
    except Exception as e:
        logger.error(f"Persisting study {resp['study_id']} failed: {e}")
        return False


# Real MRI studies from GE/Siemens scanners often contain 1,500–5,000 single-frame
# files. Starlette's default multipart parser refuses >1000 parts, so the study
# routes parse the form themselves with a clinical-scale limit.
MAX_STUDY_FILES = int(os.environ.get('SENTINEL_MAX_STUDY_FILES', '20000'))


async def _read_study_uploads(request: Request, field: str = 'files') -> list[UploadFile]:
    """Parse multipart with a study-scale part limit and return the UploadFile parts."""
    form = await request.form(max_files=MAX_STUDY_FILES, max_fields=MAX_STUDY_FILES)
    # Starlette hands back its own UploadFile class (not FastAPI's subclass), so duck-type.
    uploads = [v for k, v in form.multi_items()
               if k == field and hasattr(v, 'filename') and hasattr(v, 'read')]
    if not uploads:
        raise HTTPException(status_code=400, detail=f"No files uploaded (multipart field '{field}')")
    return uploads


@app.post("/analyze/study")
async def analyze_study(
    request: Request,
    language: str = "ru",
    user: dict = Depends(clinical_auth("analyze")),
):
    """Whole-study analysis — API contract v1 (POST multipart, repeated `files`).

    The server reads the DICOM headers and routes: brain MR → the detector
    panel (local validated triage + pending tumor classifier); head CT → the
    head_ct series protocol (axial brain series, up to 9 central slices in
    the brain window, per-class mean — never the scout/localizer; see
    series_used / n_slices_analyzed / aggregation / per_class_max); other
    modalities → the matching single-file registry model, or 422
    {requires_review} when no model honestly applies. The response carries the
    real header fields, per-finding validation status (a class is positive
    only at or above the card's decision threshold), model provenance
    (model_identity), a localized disclaimer and the actual analyzed slice.
    Persisted to the studies/ai_results tables. `normal` is ALWAYS false.

    v1.1.0: CT studies are routed per series by each series' body part (D5 —
    a HEAD+CHEST study runs head_ct on the head series only, see `routing`);
    unvalidated ('pending') detectors follow GET/PUT /settings/ai (D1) — when
    hidden, a study with no validated detector is accepted with
    requires_review=true / rejection_reason='no_validated_detector'.
    """
    files = await _read_study_uploads(request)
    _enforce_license_or_demo_limit()
    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided")
    language = _norm_language(language)

    start_time = time.time()
    study_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix='sentinel_study_'))
    try:
        file_paths = []
        for i, f in enumerate(files):
            content = await f.read()
            if not content:
                continue
            # Server-chosen name — the client filename never touches the disk.
            fp = tmp_dir / f'{i:04d}.dcm'
            fp.write_bytes(content)
            file_paths.append(fp)
        if not file_paths:
            raise HTTPException(status_code=400, detail="Uploaded files are empty")

        hdr = _read_study_headers(file_paths)
        if hdr is None:
            raise HTTPException(status_code=400, detail="No readable DICOM file in the upload")

        try:
            resp = await asyncio.to_thread(_analyze_study_sync, file_paths, hdr, language, study_id)
        except _NeedsReview as nr:
            _audit(user, 'analyze_study', 'study', study_id,
                   details=(f"route=none; requires_review; files={len(file_paths)}; "
                            f"modality={hdr.get('modality')}; body_part={hdr.get('body_part')}"),
                   request=request)
            return JSONResponse(status_code=422, content={
                'detail': str(nr), 'requires_review': True, 'reason': str(nr),
                'study_id': study_id, 'modality': hdr.get('modality'),
                'body_part': hdr.get('body_part'), 'num_files': len(file_paths),
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    inference_time_ms = int((time.time() - start_time) * 1000)
    resp['inference_time_ms'] = inference_time_ms
    resp['persisted'] = _persist_analysis(resp, hdr, user, inference_time_ms)

    _audit(user, 'analyze_study', 'study', study_id,
           details=(f"route={resp.get('route')}; detectors={resp.get('detectors_run')}; "
                    f"rejected={resp['rejected']}; requires_review={resp.get('requires_review')}; "
                    f"reason={resp.get('rejection_reason') if not resp['rejected'] else 'rejected'}; "
                    f"files={resp['num_files']}; "
                    f"lang={language}; {inference_time_ms}ms"),
           request=request)
    logger.info(
        f"Study analysis complete: study={study_id} | route={resp.get('route')} | "
        f"{len(resp['findings'])} findings | rejected={resp['rejected']} | {inference_time_ms}ms"
    )
    return resp


# ==================== Signed reports + worklist (API contract v1) ====================

def _study_out(row: dict) -> dict:
    """DB study row → API row (adds the study_id alias the desktop reads).
    SQLite's DATE affinity turns a DICOM 'YYYYMMDD' into an integer — keep the
    contract type-stable (string, DICOM DA form)."""
    out = dict(row)
    out['study_id'] = row.get('id')
    if isinstance(out.get('study_date'), (int, float)):
        out['study_date'] = str(int(out['study_date']))
    return out


@app.post("/report/sign")
def sign_report(
    req: SignRequest,
    request: Request,
    user: dict = Depends(clinical_auth("sign_report")),
):
    """Sign the final report for (study, language). The signer is the
    authenticated user; the report text is hashed (sha256) and stored together
    with the findings + model provenance it was based on. One signed report per
    study+language — a second attempt returns 409."""
    from src.inference.auth_routes import db
    from src.utils.database import ReportAlreadySignedError

    if not (req.report_text or '').strip():
        raise HTTPException(status_code=400, detail="Report text is empty")
    language = _norm_language(req.language)

    study = db.get_study(req.study_id)
    if study is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown study_id — analyze the study on this server before signing",
        )
    ai = db.get_ai_result(req.study_id) or {}
    model_identity = ai.get('model_identity') or []
    sha = hashlib.sha256(req.report_text.encode('utf-8')).hexdigest()

    try:
        row = db.create_signed_report(
            study_id=req.study_id,
            signer_id=user.get('sub'),
            report_text=req.report_text,
            language=language,
            ai_draft_text=req.ai_draft_text,
            findings_json=ai.get('findings_json'),
            model_identity_json=json.dumps(model_identity, ensure_ascii=False),
            sha256=sha,
        )
    except ReportAlreadySignedError as e:
        raise HTTPException(
            status_code=409,
            detail=f"A signed {language.upper()} report already exists for this study (report {e})",
        )
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=404, detail=f"Cannot sign: {e}")

    signer_row = db.get_user(user.get('sub')) or {}
    signer = {
        'id': user.get('sub'),
        'username': signer_row.get('username') or user.get('username') or '',
        'full_name': signer_row.get('full_name') or user.get('username') or '',
    }
    _audit(user, 'sign_report', 'report', row['report_id'],
           details=f"study={req.study_id}; lang={language}; sha256={sha[:12]}", request=request)
    return {
        'study_id': req.study_id,
        'report_id': row['report_id'],
        'language': language,
        'signed_at': row['signed_at'],
        'signer': signer,
        'sha256': sha,
        'model_identity': model_identity,
    }


@app.get("/study/{study_id}")
def get_study_detail(
    study_id: str,
    request: Request,
    user: dict = Depends(clinical_auth("view")),
):
    """{study, ai_result, reports[]} for one persisted study (worklist restore)."""
    from src.inference.auth_routes import db

    study = db.get_study(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")

    ai_row = db.get_ai_result(study_id)
    ai_result = None
    if ai_row:
        overall = ai_row.get('overall_assessment')
        ai_result = {
            'result_id': ai_row.get('id'),
            'study_id': study_id,
            'findings': ai_row.get('findings') or [],
            'model_identity': ai_row.get('model_identity') or [],
            'overall': overall,
            'overall_assessment': overall,
            'overall_impression': ai_row.get('overall_impression') or '',
            'inference_time_ms': ai_row.get('inference_time_ms'),
            'model_version': ai_row.get('model_version'),
            'is_normal': False,
            'threshold': ai_row.get('threshold'),
            'report_text': ai_row.get('report_text'),
            'report_language': ai_row.get('report_language'),
            'disclaimer': _localized_disclaimer(ai_row.get('report_language') or 'ru'),
            'requires_review': bool(study.get('requires_review')),
            'rejected': bool(study.get('rejected')),
            # rejected → the human reason; accepted-but-not-analysed (D1) → its machine
            # code, so the desktop shows the same localized note after a restart.
            'rejection_reason': _persisted_rejection_reason(
                bool(study.get('rejected')), bool(study.get('requires_review')), overall,
                ai_row.get('overall_impression') or ''),
            'app_version': ai_row.get('app_version'),
            'created_at': ai_row.get('created_at'),
            # The persisted analyzed slice; null once pruned or never written.
            'preview_base64': _load_preview(ai_row.get('preview_path'), ai_row.get('preview_sha256')),
            'preview_sha256': ai_row.get('preview_sha256'),
        }

    reports = []
    for r in db.get_reports_for_study(study_id):
        r.pop('findings_json', None)
        r.pop('model_identity_json', None)
        r['report_id'] = r.get('id')
        signer_id = r.get('signer_id') or r.get('doctor_id')
        r['signer'] = {
            'id': signer_id,
            'username': r.get('signer_username') or '',
            'full_name': r.get('signer_full_name') or r.get('signer_username') or signer_id or '',
        } if signer_id else None
        reports.append(r)

    _audit(user, 'view_study', 'study', study_id, request=request)
    return {'study': _study_out(study), 'ai_result': ai_result, 'reports': reports}


@app.get("/studies")
def list_studies(limit: int = 100, user: dict = Depends(clinical_auth("view"))):
    """Newest-first persisted studies with a compact AI summary + signed
    languages — what the desktop worklist restores after a restart."""
    from src.inference.auth_routes import db

    limit = max(1, min(int(limit), 500))
    out = []
    for s in db.list_studies(limit=limit):
        row = _study_out(s)
        # Kept light: no image bytes here, only whether GET /study/{id} can serve one.
        row['has_preview'] = _preview_exists(row.pop('preview_path', None))
        out.append(row)
    return out


# ==================== PACS integration (Orthanc) ====================
# Studies arrive from the scanner through the local Orthanc PACS
# (src/inference/orthanc_watcher.py submits them to /analyze/study); the signed
# report goes back into the same study as a DICOM Encapsulated PDF.

REPORTS_DIR = DATA_DIR / "reports"
MAX_REPORT_PDF_BYTES = 25 * 1024 * 1024


@app.get("/pacs/status")
def pacs_status(user: dict = Depends(clinical_auth("view"))):
    """Orthanc integration status: whether a PACS is configured, whether the
    in-process watcher thread is running, and the auto-ingest watcher's
    persisted state/counters (read from DATA_DIR/orthanc_watcher_state.json,
    which the standalone `python -m src.inference.orthanc_watcher` service
    also writes). Ids and counters only - never patient fields."""
    from src.inference.orthanc_watcher import orthanc_configured, default_state_path, read_state

    watcher = state.orthanc_watcher
    return {
        'configured': orthanc_configured(),
        'orthanc_url': (os.environ.get('ORTHANC_URL') or 'http://127.0.0.1:8042').rstrip('/'),
        'orthanc_auth_set': bool(os.environ.get('ORTHANC_USER')),
        'in_process_watcher': watcher is not None,
        'in_process_connected': bool(watcher.check_connection()) if watcher is not None else None,
        'state_file': str(default_state_path()),
        'watcher': read_state(),
    }


@app.post("/report/{report_id}/pdf")
async def attach_report_pdf(
    report_id: str,
    request: Request,
    pdf: UploadFile = File(...),
    push: int = 0,
    user: dict = Depends(clinical_auth("view")),
):
    """Attach the rendered PDF of a SIGNED report (multipart field `pdf`).

    Stores the PDF under DATA_DIR/reports/<report_id>.pdf (0600), records its
    sha256 on the report row, wraps it as a DICOM Encapsulated PDF instance
    (same StudyInstanceUID as the images, new series 'DOC') stored alongside,
    and with ?push=1 sends that instance to the configured Orthanc PACS.
    Allowed for users who may sign reports, or for the report's own signer.
    Unsigned report -> 409. Re-uploading the identical PDF is idempotent (same
    SOP Instance UID); a different PDF for an already-attached report -> 409.
    """
    from src.inference.auth_routes import db
    from src.utils.auth import check_permission
    from src.utils import dicom_export
    from src.inference.orthanc_watcher import orthanc_configured, orthanc_auth

    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    is_owner = bool(report.get('signer_id')) and report.get('signer_id') == user.get('sub')
    if not (check_permission(user.get('role', ''), 'sign_report') or is_owner):
        raise HTTPException(status_code=403, detail="Only the signer or a signing role may attach the report PDF")
    if not report.get('is_signed'):
        raise HTTPException(status_code=409, detail="Report is not signed - sign it before exporting to PACS")

    content = await pdf.read()
    if not content or not dicom_export.is_pdf(content):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")
    if len(content) > MAX_REPORT_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large")
    sha = hashlib.sha256(content).hexdigest()
    if report.get('pdf_sha256') and report['pdf_sha256'] != sha:
        raise HTTPException(status_code=409,
                            detail="A different PDF is already attached to this signed report")

    _ensure_private_dir(REPORTS_DIR)
    pdf_path = REPORTS_DIR / f"{report_id}.pdf"
    dcm_path = REPORTS_DIR / f"{report_id}.dcm"
    sop_uid = report.get('dicom_sop_instance_uid')

    if report.get('pdf_sha256') == sha and sop_uid and pdf_path.exists() and dcm_path.exists():
        dicom_bytes = dcm_path.read_bytes()          # idempotent re-upload
    else:
        study = db.get_study(report['study_id']) or {}
        signer_row = db.get_user(report.get('signer_id') or '') or {}
        signer_name = (signer_row.get('full_name') or signer_row.get('username')
                       or user.get('username') or 'signer')
        meta = {
            'patient_id': study.get('patient_id'),
            'patient_name': study.get('patient_name'),
            'patient_sex': study.get('patient_sex'),
            'patient_birth_date': study.get('patient_birth_date'),
            'study_instance_uid': study.get('study_instance_uid'),
            'accession_number': study.get('accession_number'),
            'study_date': study.get('study_date'),
            'modality': study.get('modality'),
            'referring_physician': '',
            'study_description': study.get('study_description'),
        }
        ds = dicom_export.build_encapsulated_pdf(content, meta, signer_name, report.get('signed_at'),
                                                 software_version=APP_VERSION)
        dicom_bytes = dicom_export.dataset_to_bytes(ds)
        sop_uid = str(ds.SOPInstanceUID)
        _write_private(pdf_path, content)
        _write_private(dcm_path, dicom_bytes)
        db.attach_report_pdf(report_id, pdf_path=str(pdf_path), pdf_sha256=sha,
                             dicom_path=str(dcm_path), sop_instance_uid=sop_uid)

    pushed = False
    orthanc_id = report.get('orthanc_instance_id')
    push_error = None
    if push:
        if orthanc_id:
            pushed = True                                # already in the PACS
        elif not orthanc_configured():
            push_error = "Orthanc is not configured (set ORTHANC_URL)"
        else:
            url = (os.environ.get('ORTHANC_URL') or 'http://127.0.0.1:8042').rstrip('/')
            try:
                res = await asyncio.to_thread(dicom_export.push_to_orthanc, dicom_bytes, url, orthanc_auth())
                orthanc_id = res.get('orthanc_id')
                pushed = bool(orthanc_id)
                if pushed:
                    db.record_report_push(report_id, orthanc_id)
                else:
                    push_error = "Orthanc accepted the instance but returned no ID"
            except RuntimeError as e:
                push_error = str(e)
                logger.warning(f"PACS push failed for report {report_id}: {e}")

    _audit(user, 'attach_report_pdf', 'report', report_id,
           details=f"study={report['study_id']}; sha256={sha[:12]}; sop={sop_uid}; "
                   f"push={int(bool(push))}; pushed={pushed}; orthanc_id={orthanc_id or ''}",
           request=request)
    return {
        'report_id': report_id,
        'study_id': report['study_id'],
        'pdf_sha256': sha,
        'pdf_bytes': len(content),
        'dicom_sop_instance_uid': sop_uid,
        'pushed': pushed,
        'orthanc_id': orthanc_id,
        'push_error': push_error,
    }


def _estimate_location(heatmap: np.ndarray) -> str:
    """Estimate anatomical location from heatmap peak position.

    Simple quadrant-based estimation for chest X-rays.
    """
    h, w = heatmap.shape
    # Find peak location
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)

    # Determine quadrant
    is_left = peak_x < w / 2  # Note: radiological convention — left in image = patient's right
    is_upper = peak_y < h / 2

    side = "Right" if is_left else "Left"  # Radiological convention flip
    vertical = "upper" if is_upper else "lower"

    # Check if central
    center_margin = 0.3
    if center_margin < peak_x / w < (1 - center_margin):
        if center_margin < peak_y / h < (1 - center_margin):
            return "Central / mediastinal region"

    return f"{side} {vertical} zone"


# ==================== Run ====================

def start_server():
    """Start the inference server.

    Set SENTINEL_TLS_CERT + SENTINEL_TLS_KEY to serve HTTPS (required when the
    server is reachable beyond localhost). SENTINEL_WORKERS controls process
    count (default 1; raise it once inference is offloaded to a threadpool).
    """
    import uvicorn

    config = get_config()
    inference_config = config.get("inference", {})

    tls_cert = os.environ.get("SENTINEL_TLS_CERT")
    tls_key = os.environ.get("SENTINEL_TLS_KEY")
    ssl_kwargs = {}
    if tls_cert and tls_key and os.path.exists(tls_cert) and os.path.exists(tls_key):
        ssl_kwargs = {"ssl_certfile": tls_cert, "ssl_keyfile": tls_key}
        logger.info("TLS enabled (HTTPS)")

    workers = int(os.environ.get("SENTINEL_WORKERS", "1"))

    uvicorn.run(
        "src.inference.server:app",
        host=inference_config.get("host", "127.0.0.1"),
        port=inference_config.get("port", 8000),
        reload=False,
        log_level="info",
        workers=workers if workers > 1 else None,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    start_server()
