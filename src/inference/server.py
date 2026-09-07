"""FastAPI inference server for Sentinel Medical AI.

Accepts DICOM files via REST API, runs DenseNet121 classification + Grad-CAM,
and returns structured analysis results in under 10 seconds.
"""

import io
import os
import time
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
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from src.pipeline.dicom_loader import load_dicom
from src.pipeline.preprocessor import preprocess_study
from src.training.densenet_trainer import DenseNet121Classifier, get_device
from src.training.gradcam import GradCAM
from src.utils.config import get_config


# ==================== Response Models ====================

class Finding(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: str
    location: str


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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


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

    # ===== License check =====
    # In dev mode (env DEV_BYPASS_LICENSE=1) we skip this.
    # In production, an invalid license blocks all /analyze endpoints.
    state.license_valid = True
    state.license_info = {'mode': 'dev', 'reason': 'license check bypassed'}
    if os.environ.get('DEV_BYPASS_LICENSE') != '1':
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
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=state.device, weights_only=False)
        num_classes = len(checkpoint.get("class_names", state.class_names))
        state.class_names = checkpoint.get("class_names", state.class_names)

        # Detect if this is a pre-trained TorchXRayVision model
        is_xrv = checkpoint.get("is_pretrained", False) and checkpoint.get("source") == "torchxrayvision"

        if is_xrv:
            try:
                import torchxrayvision as xrv
                xrv_weights = checkpoint.get("xrv_weights", "densenet121-res224-all")
                logger.info(f"Loading pre-trained TorchXRayVision: {xrv_weights}")
                state.model = xrv.models.DenseNet(weights=xrv_weights)
                state.model.to(state.device)
                state.model.eval()
                state.is_xrv = True
            except ImportError:
                # An XRV checkpoint can ONLY be loaded by torchxrayvision — its
                # weight namespace ('features.*') is incompatible with our custom
                # DenseNet121Classifier wrapper ('densenet.features.*'). Trying to
                # force-load would crash with a misleading error, so fail with an
                # actionable message instead.
                logger.error(
                    "This checkpoint is a TorchXRayVision model but torchxrayvision "
                    "is not installed. Install it (pip install torchxrayvision) and "
                    "restart. Running without the chest model."
                )
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
        logger.warning(f"No model checkpoint found at {checkpoint_path}. Server running in demo mode.")

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

app = FastAPI(
    title="Sentinel Medical AI — Inference Server",
    description="Local AI analysis for medical DICOM images",
    version="1.0.0",
    lifespan=lifespan,
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=str(state.device),
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


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

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

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
        for f in files:
            content = await f.read()
            fp = tmp_dir / (f.filename or f"slice_{len(file_paths)}.dcm")
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, 'wb') as out:
                out.write(content)
            file_paths.append(fp)

        with state.inference_lock:
            result = analyze_brain_study(file_paths, device=str(state.device),
                                         include_experimental=include_experimental)
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
        for f in files:
            content = await f.read()
            fp = tmp_dir / f.filename
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
async def save_doctor_correction(
    study_id: str,
    original_report: str,
    corrected_report: str,
    language: str = "ru",
    anon_id: Optional[str] = None,
    user: dict = Depends(clinical_auth("edit_report")),
):
    """Save doctor's correction — feeds into training data collector + local log.
    The doctor_id now comes from the AUTHENTICATED user, not a client-supplied
    string (fix: was non-attributable self-asserted doctor_id)."""
    doctor_id = user.get("sub", "unknown")
    from pathlib import Path
    import json

    # Save to local corrections log
    corrections_dir = Path("data/doctor_corrections")
    corrections_dir.mkdir(parents=True, exist_ok=True)
    correction = {
        "study_id": study_id,
        "doctor_id": doctor_id,
        "language": language,
        "original_report": original_report,
        "corrected_report": corrected_report,
        "timestamp": datetime.now().isoformat(),
    }
    with open(corrections_dir / f"corrections_{language}.jsonl", "a") as f:
        f.write(json.dumps(correction, ensure_ascii=False) + "\n")

    # Also feed to training corpus
    if state.data_collector is not None and anon_id:
        try:
            state.data_collector.record_correction(
                study_id=study_id,
                anon_id=anon_id,
                ai_report=original_report,
                doctor_report=corrected_report,
                language=language,
                doctor_id=doctor_id,
            )
        except Exception as e:
            logger.warning(f"Training corpus recording failed: {e}")

    return {"status": "saved", "correction_id": study_id}


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


@app.post("/analyze/auto", response_model=AnalysisResponse)
async def analyze_auto(
    file: UploadFile = File(...),
    language: str = "ru",
    force_modality: Optional[str] = None,
    request: Request = None,
    user: dict = Depends(clinical_auth("analyze")),
):
    """Multi-modality analysis. Auto-detects modality from DICOM tags
    (chest, brain_2d, head_ct, mammography) and routes to the right model.

    Pass `force_modality=brain_2d` etc. to override auto-detection.
    """
    from src.inference.model_registry import (
        detect_modality_from_dicom, get_model, REGISTRY,
    )

    _enforce_license_or_demo_limit()

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

    # Load DICOM (handles non-DICOM gracefully)
    study = load_dicom(tmp_path)
    if study is None:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid or corrupt DICOM file")

    # Decide which model to use
    if force_modality and force_modality in REGISTRY:
        modality_key = force_modality
    else:
        modality_key = detect_modality_from_dicom(study.modality, study.body_part)

    logger.info(
        f"Auto-routing study {study_id}: modality={study.modality}, "
        f"body_part={study.body_part} → model='{modality_key}'"
    )

    model_entry = get_model(modality_key, device=str(state.device))
    if model_entry is None or not model_entry['available']:
        Path(tmp_path).unlink(missing_ok=True)
        reason = (model_entry or {}).get('reason', 'model not registered')
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{modality_key}' not available: {reason}",
        )

    card = model_entry['card']
    predictor = model_entry['predictor']

    # Preprocess to model's input size
    target_size = card.input_size[0]
    processed_image = preprocess_study(study, target_size=target_size)

    # Encode the ACTUAL analyzed slice as a PNG data-URI so the viewer shows
    # the real scan (not a placeholder). This is exactly the image the model saw.
    preview_base64 = None
    try:
        import io as _io, base64 as _b64
        from PIL import Image as _Image
        _img8 = (np.clip(processed_image, 0, 1) * 255).astype(np.uint8)
        _buf = _io.BytesIO()
        _Image.fromarray(_img8).convert("L").save(_buf, format="PNG")
        preview_base64 = "data:image/png;base64," + _b64.b64encode(_buf.getvalue()).decode()
    except Exception as _e:
        logger.warning(f"preview encode failed: {_e}")

    # Run prediction (returns dict[class_name -> prob])
    raw_probs = predictor(processed_image)

    # Build findings — keep only above-threshold (using card-specific threshold)
    threshold = state.config.get("inference", {}).get("confidence_threshold", 0.5)
    # For softmax-based models (HF), keep top-3 instead of thresholding
    is_softmax = abs(sum(raw_probs.values()) - 1.0) < 0.05

    findings = []
    if is_softmax:
        # Single-class prediction — top class with conf >0.4
        sorted_items = sorted(raw_probs.items(), key=lambda x: -x[1])
        for cls, prob in sorted_items[:3]:
            if prob >= 0.15:
                findings.append(Finding(
                    class_name=cls,
                    confidence=round(float(prob), 4),
                    heatmap_base64='',
                    location='Central region',
                ))
    else:
        # Multi-label
        for cls, prob in raw_probs.items():
            if prob >= threshold:
                findings.append(Finding(
                    class_name=cls,
                    confidence=round(float(prob), 4),
                    heatmap_base64='',
                    location='Region of interest',
                ))

    findings.sort(key=lambda f: f.confidence, reverse=True)

    # Overall impression
    is_normal = (len(findings) == 0) or any(
        f.class_name.lower() in ('no_tumor', 'normal', 'no_finding') for f in findings[:1]
    )
    if is_normal:
        impression = "No significant pathological findings detected."
    else:
        top = ", ".join(f"{f.class_name} ({f.confidence:.0%})" for f in findings[:3])
        impression = f"Findings suggestive of: {top}"

    # Generate report via Gemma
    report_text = None
    gemma_available = False
    if state.gemma_engine is not None:
        try:
            findings_dicts = [
                {'class_name': f.class_name, 'confidence': f.confidence, 'location': f.location}
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
                body_part=study.body_part or card.display_name,
                study_date=study.study_date,
            )
            gemma_available = True
        except Exception as e:
            logger.warning(f"Gemma report failed: {e}")

    inference_time_ms = int((time.time() - start_time) * 1000)

    Path(tmp_path).unlink(missing_ok=True)

    # Audit: who ran which model on which study, and the top finding.
    top = findings[0].class_name if findings else "none"
    _audit(user, "analyze", "study", study_id,
           details=f"model={modality_key}; top={top}; lang={language}",
           request=request)

    return AnalysisResponse(
        study_id=study_id,
        modality=f"{study.modality}/{modality_key}",
        inference_time_ms=inference_time_ms,
        findings=findings,
        overall_impression=impression,
        normal=is_normal,
        model_version=f"{card.display_name}",
        report_text=report_text,
        report_language=language,
        gemma_available=gemma_available,
        preview_base64=preview_base64,
    )


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
