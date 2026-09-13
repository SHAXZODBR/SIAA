"""FastAPI inference server for Sentinel Medical AI.

Accepts DICOM files via REST API, runs DenseNet121 classification + Grad-CAM,
and returns structured analysis results in under 10 seconds.
"""

import io
import os
import json
import time
import shutil
import sqlite3
import hashlib
import asyncio
import threading
import uuid
import base64
import tempfile
from datetime import datetime
import numpy as np
import torch
import cv2
import pydicom
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from src.pipeline.dicom_loader import load_dicom, DicomStudy
from src.pipeline.preprocessor import preprocess_study
from src.training.densenet_trainer import DenseNet121Classifier, get_device
from src.training.gradcam import GradCAM
from src.utils.config import get_config
from src.utils.paths import DATA_DIR, LOG_DIR, CORRECTIONS_DIR, PREVIEWS_DIR, REPO_ROOT, ensure_data_dirs
from src.utils.logger import setup_logger
from src.utils.auth import DEV_INSECURE, REQUIRE_AUTH
from src.utils.offline import OFFLINE

# Server/app version reported in /health and stamped on every analysis
# (desktop-app/package.json carries the same number for the UI build).
APP_VERSION = "1.0.0"

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

    # Warm the brain panel so /health can report real load status (and the
    # first study of the day is not the one that pays the load cost). The
    # local triage model is the product's only validated detector — /health
    # reports 'degraded' without it. SENTINEL_SKIP_WARMUP=1 skips (tests/CLI).
    if os.environ.get('SENTINEL_SKIP_WARMUP') != '1':
        from src.inference.model_registry import get_model as _get_model
        for _key in ('brain_triage', 'brain_tumor_class'):
            try:
                _entry = _get_model(_key, device=str(state.device))
                if _entry and _entry.get('available'):
                    logger.info(f"Warmed model '{_key}'")
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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


@app.get("/license/status")
async def license_status():
    """Return current license info — used by the desktop app to show
    customer name, expiry, and which features are unlocked."""
    return {
        'valid': state.license_valid,
        'info': state.license_info,
        'demo_calls_today': state.demo_calls_today,
        'demo_limit': 10,
    }


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
        report = state.gemma_engine.generate(
            findings=req.findings,
            language=req.language,
            patient_info=req.patient_info,
            modality=req.modality,
            body_part=req.body_part,
        )
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
    from src.inference.brain_analysis import analyze_brain_study
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

        with state.inference_lock:
            result = analyze_brain_study(file_paths, device=str(state.device),
                                         include_experimental=include_experimental)
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


def _generate_report(findings: list[dict], language: str, modality: str,
                     body_part: str, study_date: str = '') -> tuple[Optional[str], bool]:
    """(report_text, engine_used). Patient identifiers are never passed to the
    engine (PHI minimisation — the prompt may reach an Ollama host)."""
    if state.gemma_engine is None:
        return None, False
    try:
        text = state.gemma_engine.generate(
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
        return (text or None), bool(text)
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return None, False


def _run_registry_model(study: DicomStudy, modality_key: str, language: str) -> dict:
    """Core of the single-file registry path — shared by /analyze/auto and the
    non-brain branch of /analyze/study. Blocking: call from a worker thread.
    Never declares a study normal."""
    from src.inference.model_registry import get_model, get_model_identity

    model_entry = get_model(modality_key, device=str(state.device))
    if model_entry is None or not model_entry['available']:
        reason = (model_entry or {}).get('reason', 'model not registered')
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{modality_key}' not available: {reason}",
        )
    card = model_entry['card']
    predictor = model_entry['predictor']

    # Preprocess to the model's input size; the preview is exactly what it saw.
    processed_image = preprocess_study(study, target_size=card.input_size[0])
    preview_base64 = _encode_preview(processed_image)

    with state.inference_lock:
        raw_probs = predictor(processed_image)
    raw_probs = {str(k): float(v) for k, v in raw_probs.items() if not str(k).startswith('_')}

    threshold = float(state.config.get("inference", {}).get("confidence_threshold", 0.5))
    # Softmax (HF single-label) models: top-3 above 0.15; multi-label: threshold.
    is_softmax = abs(sum(raw_probs.values()) - 1.0) < 0.05
    localized = card.classes_localized or {}

    def _finding(cls: str, prob: float, location: str) -> dict:
        return {
            'class_name': cls,
            'finding': (localized.get(cls) or {}).get(language) or cls,
            'confidence': round(prob, 4),
            'positive': cls.lower() not in _NEGATIVE_CLASSES,
            'status': card.validation_status,
            'detector': modality_key,
            'sequence_used': None,
            'heatmap_base64': '',
            'location': location,
        }

    findings = []
    if is_softmax:
        for cls, prob in sorted(raw_probs.items(), key=lambda x: -x[1])[:3]:
            if prob >= 0.15:
                findings.append(_finding(cls, prob, 'Central region'))
    else:
        for cls, prob in raw_probs.items():
            if prob >= threshold:
                findings.append(_finding(cls, prob, 'Region of interest'))
    findings.sort(key=lambda f: (not f['positive'], -f['confidence']))

    positives = [f for f in findings if f['positive']]
    if positives:
        impression = "Findings suggestive of: " + ", ".join(
            f"{f['class_name']} ({f['confidence']:.0%})" for f in positives[:3]
        )
    else:
        impression = _NOT_NORMAL_IMPRESSION
    overall_assessment = {
        'abnormal_flagged': bool(positives),
        'flags': [f['finding'] for f in positives],
        'text': impression,
    }

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

    Patient/study fields come from the first readable DICOM; SeriesDescription
    and ProtocolName are collected unique across all files (upload order).
    'best_path' is the largest image (Rows×Columns) — the single-file fallback.
    Returns None when nothing parses as DICOM."""
    hdr = None
    series, protocols = [], []
    best_path, best_px = None, -1
    for p in file_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        # force=True parses almost anything — require a real DICOM signal
        if not (_tag(ds, 'SOPClassUID') or _tag(ds, 'Modality')):
            continue
        if hdr is None:
            hdr = {out: _tag(ds, tag) for tag, out in _HEADER_FIELDS}
            hdr['readable'] = 0
        hdr['readable'] += 1
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
    if hdr is None:
        return None
    hdr['series_descriptions'] = series
    hdr['protocol_names'] = protocols
    hdr['best_path'] = best_path
    return hdr


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


def _analyze_study_sync(file_paths: list[Path], hdr: dict, language: str,
                        study_id: str) -> dict:
    """Blocking core of POST /analyze/study (runs in a worker thread).

    Brain MR → the detector panel (validated triage + pending tumor class).
    Anything else → strict single-file registry routing, or _NeedsReview."""
    from src.inference.brain_analysis import analyze_brain_study, DETECTORS
    from src.inference.model_registry import get_model_identity

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
    }

    if _is_brain_mr(hdr):
        with state.inference_lock:
            result = analyze_brain_study(file_paths, device=str(state.device),
                                         include_experimental=False)
        preview_slice = result.pop('_preview_slice', None)
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])

        resp['route'] = 'brain_panel'
        resp['body_part'] = 'BRAIN'
        resp['sequences_present'] = result.get('sequences_present', [])
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
            study_date=hdr.get('study_date') or '',
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
        return resp

    # ---- non-brain: strict single-file routing on the best file ----
    texts = [hdr.get('study_description') or ''] + list(hdr.get('series_descriptions') or []) \
        + list(hdr.get('protocol_names') or [])
    modality_key = _match_registry_card(hdr.get('modality'), hdr.get('body_part'), texts)
    if modality_key is None:
        raise _NeedsReview(
            f"No registered model for modality '{hdr.get('modality') or '?'}' / body part "
            f"'{hdr.get('body_part') or '?'}' — study routed to radiologist review without AI analysis"
        )
    study = load_dicom(hdr['best_path']) if hdr.get('best_path') else None
    if study is None:
        raise HTTPException(status_code=400, detail='Could not decode pixel data from the uploaded DICOM files')

    run = _run_registry_model(study, modality_key, language)
    card = run['card']
    resp.update({
        'route': modality_key,
        'body_part': hdr.get('body_part') or (card.body_part_dicom_tags[0] if card.body_part_dicom_tags else ''),
        'findings': run['findings'],
        'detectors_run': [modality_key],
        'model_identity': run['model_identity'],
        'overall_assessment': run['overall_assessment'],
        'overall_impression': run['impression'],
        'preview_base64': run['preview_base64'],
        'report_text': run['report_text'],
        'gemma_available': run['gemma_available'],
        'threshold': run['threshold'],
    })
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
    panel (local validated triage + pending tumor classifier); other
    modalities → the matching single-file registry model, or 422
    {requires_review} when no model honestly applies. The response carries the
    real header fields, per-finding validation status, model provenance
    (model_identity), a localized disclaimer and the actual analyzed slice.
    Persisted to the studies/ai_results tables. `normal` is ALWAYS false.
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
                    f"rejected={resp['rejected']}; files={resp['num_files']}; "
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
            'rejection_reason': (ai_row.get('overall_impression') or None) if study.get('rejected') else None,
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
