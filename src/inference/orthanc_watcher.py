"""Orthanc PACS Watcher — monitors for new incoming DICOM studies.

Polls the Orthanc PACS server every N seconds. When a new study arrives
(from an MRI/CT/X-ray machine via C-STORE), automatically triggers AI analysis.

This runs as a background service alongside the main application.
"""

import time
import threading
import requests
import tempfile
from pathlib import Path
from typing import Callable, Optional
from loguru import logger
from datetime import datetime

from src.utils.database import SentinelDB
from src.utils.config import get_config


class OrthancWatcher:
    """Watches Orthanc PACS for new studies and triggers analysis.

    Flow:
    1. Poll Orthanc /studies every N seconds
    2. Compare against known studies in local DB
    3. For each new study:
       a. Download DICOM instances
       b. Store in local database
       c. Call AI analysis callback
       d. Update study status
    """

    def __init__(
        self,
        orthanc_url: str = "http://localhost:8042",
        poll_interval: int = 10,
        db: Optional[SentinelDB] = None,
        on_new_study: Optional[Callable] = None,
    ):
        self.orthanc_url = orthanc_url.rstrip("/")
        self.poll_interval = poll_interval
        self.db = db or SentinelDB()
        self.on_new_study = on_new_study

        self.known_orthanc_ids: set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(f"OrthancWatcher initialized: {self.orthanc_url} (poll every {self.poll_interval}s)")

    def start(self):
        """Start watching in a background thread."""
        if self._running:
            logger.warning("Watcher already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Orthanc watcher started")

    def stop(self):
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Orthanc watcher stopped")

    def _watch_loop(self):
        """Main polling loop."""
        # Initial sync — load known studies
        existing = self.db.get_all_studies(limit=10000)
        for s in existing:
            if s.get("orthanc_id"):
                self.known_orthanc_ids.add(s["orthanc_id"])
        logger.info(f"Loaded {len(self.known_orthanc_ids)} known studies from DB")

        while self._running:
            try:
                self._poll_orthanc()
            except Exception as e:
                logger.error(f"Watcher poll error: {e}")

            time.sleep(self.poll_interval)

    def _poll_orthanc(self):
        """Poll Orthanc for new studies."""
        try:
            resp = requests.get(f"{self.orthanc_url}/studies", timeout=5)
            resp.raise_for_status()
            orthanc_study_ids = resp.json()
        except requests.ConnectionError:
            return  # Orthanc not available — silently retry
        except Exception as e:
            logger.debug(f"Orthanc poll failed: {e}")
            return

        for orthanc_id in orthanc_study_ids:
            if orthanc_id not in self.known_orthanc_ids:
                self._handle_new_study(orthanc_id)

    def _handle_new_study(self, orthanc_id: str):
        """Process a newly discovered study from Orthanc."""
        logger.info(f"New study detected: {orthanc_id}")
        self.known_orthanc_ids.add(orthanc_id)

        try:
            # Get study details from Orthanc
            resp = requests.get(f"{self.orthanc_url}/studies/{orthanc_id}", timeout=10)
            resp.raise_for_status()
            details = resp.json()

            patient_id = details.get("PatientMainDicomTags", {}).get("PatientID", "UNKNOWN")
            modality = details.get("MainDicomTags", {}).get("ModalitiesInStudy", "UNKNOWN")
            study_date = details.get("MainDicomTags", {}).get("StudyDate", "")
            description = details.get("MainDicomTags", {}).get("StudyDescription", "")

            # Store in local DB
            study_id = self.db.create_study(
                patient_id=patient_id,
                modality=modality,
                body_part=description or "UNKNOWN",
                orthanc_id=orthanc_id,
                study_date=study_date,
            )

            logger.info(f"Study registered: {study_id} | {patient_id} | {modality}")

            # Download first instance for analysis
            instances = details.get("Instances", [])
            if instances:
                dicom_path = self._download_instance(instances[0], study_id)
                if dicom_path:
                    # Trigger AI analysis callback
                    if self.on_new_study:
                        self.db.update_study_status(study_id, "processing")
                        self.on_new_study(study_id, dicom_path, details)

        except Exception as e:
            logger.error(f"Failed to process study {orthanc_id}: {e}")

    def _download_instance(self, instance_id: str, study_id: str) -> Optional[str]:
        """Download a DICOM instance from Orthanc."""
        try:
            resp = requests.get(
                f"{self.orthanc_url}/instances/{instance_id}/file",
                timeout=30,
            )
            resp.raise_for_status()

            # Save to local storage
            storage_dir = Path("data/dicom") / study_id
            storage_dir.mkdir(parents=True, exist_ok=True)
            dicom_path = storage_dir / f"{instance_id}.dcm"

            with open(dicom_path, "wb") as f:
                f.write(resp.content)

            logger.debug(f"Downloaded instance: {instance_id} → {dicom_path}")
            return str(dicom_path)

        except Exception as e:
            logger.error(f"Failed to download instance {instance_id}: {e}")
            return None

    def check_connection(self) -> bool:
        """Check if Orthanc is reachable."""
        try:
            resp = requests.get(f"{self.orthanc_url}/system", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def get_orthanc_stats(self) -> Optional[dict]:
        """Get Orthanc server statistics."""
        try:
            resp = requests.get(f"{self.orthanc_url}/statistics", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None


def auto_analyze_callback(study_id: str, dicom_path: str, details: dict):
    """Callback triggered when a new study is detected.

    Sends the DICOM file to the FastAPI inference server for analysis.
    """
    import requests as req

    logger.info(f"Auto-analyzing study {study_id}...")

    try:
        with open(dicom_path, "rb") as f:
            resp = req.post(
                "http://127.0.0.1:8000/analyze",
                files={"file": (Path(dicom_path).name, f, "application/dicom")},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

        # Save to database
        db = SentinelDB()
        import json
        db.save_ai_result(
            study_id=study_id,
            findings_json=json.dumps(result.get("findings", [])),
            inference_time_ms=result.get("inference_time_ms", 0),
            model_version=result.get("model_version", "unknown"),
            is_normal=result.get("normal", True),
            overall_impression=result.get("overall_impression", ""),
        )

        logger.info(
            f"Analysis complete for {study_id}: "
            f"{len(result.get('findings', []))} findings | "
            f"{result.get('inference_time_ms', 0)}ms"
        )

    except Exception as e:
        logger.error(f"Auto-analysis failed for {study_id}: {e}")
        db = SentinelDB()
        db.update_study_status(study_id, "error")


# ===== Standalone runner =====

def main():
    """Run the Orthanc watcher as a standalone service."""
    from src.utils.logger import setup_logger
    setup_logger()

    config = get_config()
    orthanc_url = config.get("settings", {}).get("orthanc_url", "http://localhost:8042")

    watcher = OrthancWatcher(
        orthanc_url=orthanc_url,
        poll_interval=10,
        on_new_study=auto_analyze_callback,
    )

    if watcher.check_connection():
        logger.info(f"Orthanc connected at {orthanc_url}")
        stats = watcher.get_orthanc_stats()
        if stats:
            logger.info(f"Orthanc stats: {stats}")
    else:
        logger.warning(f"Orthanc not available at {orthanc_url} — will retry")

    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        logger.info("Watcher stopped by user")


if __name__ == "__main__":
    main()
