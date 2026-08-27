"""
================================================================================
  SENTINEL MEDICAL AI — LOCAL FOLDER WATCHER
================================================================================
  Watches a local folder where the MRI/CT machine saves DICOM files.
  When a new file appears → automatically analyzes it → sends to app.

  TYPICAL CLINIC SETUP:
    MRI/CT machine → saves to shared network folder "\\clinic-server\dicom"
    OR
    MRI/CT machine → saves to local folder "C:\DICOM\Studies"
    OR
    Radiology tech exports DICOM → drops in folder "C:\Sentinel\Inbox"

  Sentinel watches any of these folders and auto-processes new files.

  ALTERNATIVE: Orthanc PACS (DICOM standard protocol)
  - Better for hospitals with proper DICOM networking
  - Uses C-STORE protocol (standard)
  - Orthanc service runs on clinic PC
  - MRI machine is configured to send to Orthanc
  - Sentinel watches Orthanc instead of folder
  - See: src/inference/orthanc_watcher.py

  FOLDER WATCHER WORKFLOW:
    1. User configures folder in settings (default: C:\Sentinel\Inbox)
    2. Watcher polls folder every 3 seconds
    3. New DICOM file detected
    4. Check it's complete (not still being written)
    5. Extract patient info, modality, body part
    6. Run AI analysis (chest/brain/CT)
    7. Store results in local DB
    8. Send notification to Electron app
    9. Move processed file to /processed/

  PERFORMANCE:
    - Polling every 3 sec = near real-time detection
    - 3-second GTX 1650 analysis for chest X-ray
    - 15-second for brain CT (larger image)
    - < 30 seconds total from scan-save to result-on-screen
================================================================================
"""

import os
import time
import shutil
import threading
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Set
from loguru import logger

from src.utils.database import SentinelDB
from src.pipeline.dicom_loader import load_dicom


class FolderWatcher:
    """Watches a local folder for new DICOM files and triggers analysis."""

    def __init__(
        self,
        watch_dir: str,
        processed_dir: Optional[str] = None,
        poll_interval: float = 3.0,
        on_new_file: Optional[Callable] = None,
        db: Optional[SentinelDB] = None,
        file_extensions: tuple = ('.dcm', '.dicom', '.DCM'),
        skip_incomplete: bool = True,
    ):
        """
        Args:
            watch_dir: Folder to monitor (e.g. "C:/Sentinel/Inbox")
            processed_dir: Where to move files after processing (default: watch_dir/processed)
            poll_interval: How often to check (seconds)
            on_new_file: Callback(file_path, study_id) when new file arrives
            db: SentinelDB instance (creates new if None)
            file_extensions: Which extensions to watch
            skip_incomplete: Skip files that are still being written
        """
        self.watch_dir = Path(watch_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else self.watch_dir / 'processed'
        self.poll_interval = poll_interval
        self.on_new_file = on_new_file
        self.db = db or SentinelDB()
        self.file_extensions = tuple(ext.lower() for ext in file_extensions)
        self.skip_incomplete = skip_incomplete

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._known_hashes: Set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(f"FolderWatcher: {self.watch_dir} (poll every {poll_interval}s)")

    def start(self):
        """Start watching in background."""
        if self._running:
            logger.warning("Already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Folder watcher started: {self.watch_dir}")

    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Folder watcher stopped")

    def _watch_loop(self):
        """Main polling loop."""
        # Load known hashes from DB (prevents re-processing on restart)
        existing = self.db.get_all_studies(limit=10000)
        for s in existing:
            dp = s.get('dicom_path', '')
            if dp:
                self._known_hashes.add(self._get_file_hash(dp))
        logger.info(f"Loaded {len(self._known_hashes)} known file hashes from DB")

        while self._running:
            try:
                self._scan_folder()
            except Exception as e:
                logger.error(f"Watcher error: {e}")
            time.sleep(self.poll_interval)

    def _scan_folder(self):
        """Scan folder for new DICOM files."""
        for path in self.watch_dir.iterdir():
            if not path.is_file():
                continue
            if not path.name.lower().endswith(self.file_extensions):
                continue

            # Check file is complete (not still being written)
            if self.skip_incomplete and self._is_file_being_written(path):
                continue

            # Check if already processed
            file_hash = self._get_file_hash(str(path))
            if file_hash in self._known_hashes:
                continue

            self._known_hashes.add(file_hash)

            # Process this new file
            self._process_new_file(path)

    def _is_file_being_written(self, path: Path) -> bool:
        """Check if file is still being written by MRI/CT machine."""
        try:
            size1 = path.stat().st_size
            time.sleep(0.5)
            size2 = path.stat().st_size
            return size1 != size2  # Still changing = still writing
        except Exception:
            return True

    def _get_file_hash(self, path: str) -> str:
        """Quick hash based on path + size + mtime."""
        try:
            stat = os.stat(path)
            key = f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
            return hashlib.md5(key.encode()).hexdigest()
        except Exception:
            return hashlib.md5(path.encode()).hexdigest()

    def _process_new_file(self, path: Path):
        """Process a newly detected DICOM file."""
        logger.info(f"New DICOM detected: {path.name}")

        try:
            # Load DICOM to extract metadata
            study = load_dicom(path)
            if study is None:
                logger.error(f"Could not load DICOM: {path.name}")
                self._move_to_failed(path)
                return

            # Register in DB
            study_id = self.db.create_study(
                patient_id=study.patient_id or 'UNKNOWN',
                modality=study.modality or 'UNKNOWN',
                body_part=study.body_part or 'UNKNOWN',
                dicom_path=str(path),
                study_date=study.study_date,
            )

            # Update status to processing
            self.db.update_study_status(study_id, 'processing')

            logger.info(f"Registered study {study_id}: {study.patient_id} | {study.modality}")

            # Call user's callback (usually triggers AI analysis)
            if self.on_new_file:
                try:
                    self.on_new_file(str(path), study_id, study)
                except Exception as e:
                    logger.error(f"Callback failed for {study_id}: {e}")
                    self.db.update_study_status(study_id, 'error')

            # Move to processed
            self._move_to_processed(path, study_id)

        except Exception as e:
            logger.error(f"Failed to process {path.name}: {e}")
            self._move_to_failed(path)

    def _move_to_processed(self, path: Path, study_id: str):
        """Move file to processed folder with study_id in name."""
        try:
            dest = self.processed_dir / f"{study_id}_{path.name}"
            shutil.move(str(path), str(dest))
        except Exception as e:
            logger.error(f"Failed to move file: {e}")

    def _move_to_failed(self, path: Path):
        """Move file to failed folder."""
        failed_dir = self.watch_dir / 'failed'
        failed_dir.mkdir(exist_ok=True)
        try:
            shutil.move(str(path), str(failed_dir / path.name))
        except Exception:
            pass


# ==============================================================================
# AUTO-ANALYSIS CALLBACK
# ==============================================================================

def auto_analyze_callback(file_path: str, study_id: str, study):
    """Called when new DICOM is detected — sends to inference server.

    This runs the full AI pipeline:
    1. Send DICOM to /analyze endpoint
    2. Get findings + heatmap + Gemma report
    3. Save everything to database
    4. Notify desktop app via WebSocket (if connected)
    """
    import requests
    import json

    logger.info(f"Auto-analyzing {study_id} ({study.modality})")

    try:
        # Determine language (clinic default — configurable)
        language = os.environ.get('SENTINEL_DEFAULT_LANG', 'ru')

        with open(file_path, 'rb') as f:
            resp = requests.post(
                f"http://127.0.0.1:8000/analyze?language={language}",
                files={'file': (Path(file_path).name, f, 'application/dicom')},
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()

        # Save results to DB
        db = SentinelDB()
        db.save_ai_result(
            study_id=study_id,
            findings_json=json.dumps(result.get('findings', [])),
            inference_time_ms=result.get('inference_time_ms', 0),
            model_version=result.get('model_version', 'unknown'),
            is_normal=result.get('normal', True),
            overall_impression=result.get('overall_impression', ''),
        )

        # Save Gemma report if generated
        if result.get('report_text'):
            db.create_report(
                study_id=study_id,
                doctor_id='ai',
                ai_draft_text=result['report_text'],
                language=result.get('report_language', 'ru'),
            )

        logger.info(
            f"✓ Analysis complete for {study_id}: "
            f"{len(result.get('findings', []))} findings | "
            f"{result.get('inference_time_ms', 0)}ms | "
            f"Gemma: {'yes' if result.get('gemma_available') else 'no'}"
        )

    except Exception as e:
        logger.error(f"Auto-analysis failed for {study_id}: {e}")
        db = SentinelDB()
        db.update_study_status(study_id, 'error')


# ==============================================================================
# STANDALONE RUNNER
# ==============================================================================

def main():
    """Run the folder watcher as a standalone service."""
    import argparse
    from src.utils.logger import setup_logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', required=True, help='Folder to watch for new DICOMs')
    parser.add_argument('--processed', help='Folder to move processed files')
    parser.add_argument('--interval', type=float, default=3.0, help='Poll interval (sec)')
    parser.add_argument('--language', default='ru', help='Report language (ru/uz/en)')
    args = parser.parse_args()

    setup_logger()
    os.environ['SENTINEL_DEFAULT_LANG'] = args.language

    watcher = FolderWatcher(
        watch_dir=args.watch,
        processed_dir=args.processed,
        poll_interval=args.interval,
        on_new_file=auto_analyze_callback,
    )

    print(f"\n{'='*70}")
    print("  SENTINEL FOLDER WATCHER ACTIVE")
    print(f"{'='*70}")
    print(f"  Watching: {args.watch}")
    print(f"  Report language: {args.language}")
    print(f"  Poll interval: {args.interval}s")
    print(f"\n  Drop DICOM files into the watched folder to auto-analyze.")
    print(f"  Press Ctrl+C to stop.\n")

    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("\nStopped.")


if __name__ == '__main__':
    main()
