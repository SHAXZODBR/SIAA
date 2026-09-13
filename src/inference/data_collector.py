"""
================================================================================
  SENTINEL MEDICAL AI — CENTRALIZED TRAINING DATA COLLECTOR
================================================================================
  Automatically collects anonymized data from every analysis + doctor correction,
  creating a growing training dataset that improves Sentinel over time.

  WHAT IT COLLECTS (per analysis):
    - Anonymized DICOM image
    - AI findings (what the model predicted)
    - Radiologist's final report (after their corrections)
    - Modality, body part, demographics (age range, sex)
    - Confidence scores
    - Correction deltas (what doctor changed)

  WHAT IT DOESN'T COLLECT (privacy):
    - Patient name
    - Patient ID
    - Birth date
    - Clinic-identifying info
    - Any PII

  STORAGE STRUCTURE:
    data/training_corpus/
      ├── images/              ← Anonymized PNG thumbnails (512x512)
      │   ├── S-001.png
      │   ├── S-002.png
      │   └── ...
      ├── labels.jsonl         ← One entry per study
      ├── reports.jsonl        ← Doctor-verified reports
      ├── corrections.jsonl    ← AI vs final comparisons
      └── metadata.json        ← Collection stats

  USAGE:
    # Automatic — called by server after each analysis + correction
    collector.record_analysis(study_id, findings, dicom_path)
    collector.record_correction(study_id, original, corrected)

    # Periodic export for training
    python -m src.inference.data_collector export --output export.jsonl
================================================================================
"""

import os
import json
import hashlib
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

from src.pipeline.dicom_loader import load_dicom
from src.pipeline.preprocessor import preprocess_study


class TrainingDataCollector:
    """Collects anonymized training data from every analysis."""

    def __init__(self, base_dir: Optional[str] = None):
        # Mutable state lives under DATA_DIR (src/utils/paths.py), never inside
        # the repo checkout / app bundle — the old CWD-relative default wrote
        # into the source tree whenever the server ran from the repo root.
        if base_dir is None:
            from src.utils.paths import TRAINING_CORPUS_DIR
            base_dir = TRAINING_CORPUS_DIR
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_path = self.base_dir / "labels.jsonl"
        self.reports_path = self.base_dir / "reports.jsonl"
        self.corrections_path = self.base_dir / "corrections.jsonl"
        self.metadata_path = self.base_dir / "metadata.json"

        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'created': datetime.now().isoformat(),
                'total_analyses': 0,
                'total_corrections': 0,
                'total_signed_reports': 0,
                'clinic_name': 'SIA Medical AI',
                'version': '1.0',
            }
        self._save_metadata()

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _anonymize_id(self, original_id: str) -> str:
        """Create reproducible anonymous ID from patient ID."""
        h = hashlib.sha256(f"SENTINEL_SALT_{original_id}".encode()).hexdigest()
        return f"S-{h[:10]}"

    def _age_to_range(self, age) -> str:
        """Convert exact age to range (privacy). Handles DICOM AS formats
        like '045Y', '012M', '003W', '030D' and plain ints."""
        try:
            s = str(age).strip().upper()
            unit = s[-1] if s and s[-1] in 'YMWD' else 'Y'
            num = int(''.join(c for c in s if c.isdigit()))
            # Normalize weeks/days/months to years for banding
            if unit == 'M':
                num = num // 12
            elif unit in ('W', 'D'):
                num = 0
            age_num = num
            if age_num < 18:
                return 'pediatric'
            elif age_num < 30:
                return '18-29'
            elif age_num < 45:
                return '30-44'
            elif age_num < 60:
                return '45-59'
            elif age_num < 75:
                return '60-74'
            else:
                return '75+'
        except Exception:
            return 'unknown'

    # ============================================================
    # RECORDING
    # ============================================================

    def record_analysis(
        self,
        study_id: str,
        dicom_path: str,
        findings: list[dict],
        modality: str = '',
        body_part: str = '',
    ) -> Optional[str]:
        """Record a new analysis for training data collection.

        Returns the anonymous study ID used in the corpus.
        """
        try:
            # Load DICOM to get metadata + create anonymized image
            study = load_dicom(dicom_path)
            if study is None:
                return None

            anon_id = self._anonymize_id(study.patient_id or study_id)

            # Save anonymized thumbnail image (512x512 PNG)
            processed = preprocess_study(study, target_size=512)
            img_uint8 = (processed * 255).clip(0, 255).astype(np.uint8)
            img_path = self.images_dir / f"{anon_id}.png"
            cv2.imwrite(str(img_path), img_uint8)

            # Build label entry (anonymized)
            entry = {
                'anon_id': anon_id,
                'image_path': f"images/{anon_id}.png",
                'modality': modality or study.modality,
                'body_part': body_part or study.body_part,
                'age_range': self._age_to_range(study.patient_age),
                'sex': study.patient_sex or 'unknown',
                'findings': findings,
                'ai_classes': [f['class_name'] for f in findings],
                'ai_confidences': [round(f['confidence'], 3) for f in findings],
                'recorded_at': datetime.now().isoformat(),
                'source': 'sentinel_ai_analysis',
            }

            # Append to labels.jsonl
            with open(self.labels_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.metadata['total_analyses'] += 1
            self._save_metadata()

            logger.debug(f"Training data: recorded {anon_id}")
            return anon_id

        except Exception as e:
            logger.error(f"Failed to record training data: {e}")
            return None

    def record_correction(
        self,
        study_id: str,
        anon_id: str,
        ai_report: str,
        doctor_report: str,
        language: str = 'ru',
        doctor_id: str = 'anon',
    ):
        """Record doctor's correction of AI report.

        These corrections become training data for Gemma fine-tuning.
        """
        try:
            # Compute correction magnitude
            ai_words = set(ai_report.lower().split())
            doc_words = set(doctor_report.lower().split())
            jaccard = len(ai_words & doc_words) / max(len(ai_words | doc_words), 1)
            change_pct = round((1 - jaccard) * 100, 1)

            entry = {
                'anon_id': anon_id,
                'language': language,
                'ai_report': ai_report,
                'doctor_report': doctor_report,
                'change_percentage': change_pct,
                'doctor_id_hash': hashlib.sha256(doctor_id.encode()).hexdigest()[:8],
                'recorded_at': datetime.now().isoformat(),
            }

            with open(self.corrections_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.metadata['total_corrections'] += 1
            self._save_metadata()

            logger.info(f"Correction recorded: {anon_id} ({language}, {change_pct}% change)")

        except Exception as e:
            logger.error(f"Failed to record correction: {e}")

    def record_signed_report(
        self,
        study_id: str,
        anon_id: str,
        final_report: str,
        language: str = 'ru',
    ):
        """Record a signed (final) report — highest-quality training data."""
        try:
            entry = {
                'anon_id': anon_id,
                'language': language,
                'final_report': final_report,
                'signed_at': datetime.now().isoformat(),
                'is_gold_standard': True,  # Doctor-signed = gold standard
            }

            with open(self.reports_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.metadata['total_signed_reports'] += 1
            self._save_metadata()

        except Exception as e:
            logger.error(f"Failed to record report: {e}")

    # ============================================================
    # STATISTICS
    # ============================================================

    def get_stats(self) -> dict:
        """Get collection statistics."""
        # Count entries
        def count_lines(path):
            if not Path(path).exists():
                return 0
            with open(path) as f:
                return sum(1 for _ in f)

        # Class distribution from labels
        class_counts = {}
        modality_counts = {}
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        for cls in entry.get('ai_classes', []):
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        mod = entry.get('modality', 'unknown')
                        modality_counts[mod] = modality_counts.get(mod, 0) + 1
                    except Exception:
                        pass

        # Correction quality
        correction_avg_change = 0
        if self.corrections_path.exists():
            changes = []
            with open(self.corrections_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        changes.append(entry.get('change_percentage', 0))
                    except Exception:
                        pass
            correction_avg_change = round(sum(changes) / len(changes), 1) if changes else 0

        return {
            'total_analyses': count_lines(self.labels_path),
            'total_corrections': count_lines(self.corrections_path),
            'total_signed_reports': count_lines(self.reports_path),
            'images_saved': len(list(self.images_dir.glob('*.png'))),
            'modality_distribution': modality_counts,
            'class_distribution': class_counts,
            'avg_correction_percentage': correction_avg_change,
            'ready_for_finetuning': {
                'densenet': count_lines(self.labels_path) >= 500,
                'gemma_ru': sum(1 for c in self._get_corrections_by_language('ru')) >= 500,
                'gemma_uz': sum(1 for c in self._get_corrections_by_language('uz')) >= 200,
                'gemma_en': sum(1 for c in self._get_corrections_by_language('en')) >= 100,
            }
        }

    def _get_corrections_by_language(self, language: str):
        """Generator: corrections filtered by language."""
        if not self.corrections_path.exists():
            return
        with open(self.corrections_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('language') == language:
                        yield entry
                except Exception:
                    pass

    # ============================================================
    # EXPORT FOR TRAINING
    # ============================================================

    def export_for_densenet_finetuning(self, output_path: str, min_confidence: float = 0.5):
        """Export training data for DenseNet fine-tuning.

        Format: list of (image_path, multi-label vector)
        """
        entries = []
        if not self.labels_path.exists():
            return entries

        with open(self.labels_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Only include high-confidence findings
                    high_conf = [
                        f for f in entry.get('findings', [])
                        if f.get('confidence', 0) >= min_confidence
                    ]
                    if not high_conf and len(entry.get('findings', [])) > 0:
                        continue  # Skip low-confidence cases

                    entries.append({
                        'image_path': entry['image_path'],
                        'classes': [f['class_name'] for f in high_conf],
                        'modality': entry.get('modality'),
                        'anon_id': entry['anon_id'],
                    })
                except Exception:
                    pass

        # Save as JSONL
        with open(output_path, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')

        logger.info(f"Exported {len(entries)} entries for DenseNet fine-tuning → {output_path}")
        return entries

    def export_for_gemma_finetuning(self, output_path: str, language: str = 'ru'):
        """Export doctor corrections as training pairs for Gemma fine-tuning.

        Format: [{"instruction": ..., "input": ..., "output": doctor_report}]
        """
        entries = []

        # Get corrections
        corrections = {}
        if self.corrections_path.exists():
            with open(self.corrections_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('language') == language:
                            corrections[entry['anon_id']] = entry['doctor_report']
                    except Exception:
                        pass

        # Get findings from labels
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        anon_id = entry['anon_id']
                        if anon_id not in corrections:
                            continue

                        findings_text = ', '.join([
                            f"{f['class_name']} ({f['confidence']*100:.0f}%)"
                            for f in entry.get('findings', [])
                        ]) or 'No pathology detected'

                        entries.append({
                            'anon_id': anon_id,
                            'image_path': entry['image_path'],
                            'findings_text': findings_text,
                            'report': corrections[anon_id],
                            'language': language,
                            'modality': entry.get('modality'),
                            'body_part': entry.get('body_part'),
                        })
                    except Exception:
                        pass

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')

        logger.info(f"Exported {len(entries)} {language} examples for Gemma fine-tuning → {output_path}")
        return entries


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['stats', 'export_densenet', 'export_gemma'])
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--language', default='ru', help='Language for Gemma export')
    parser.add_argument('--data-dir', default='data/training_corpus', help='Collector data dir')
    args = parser.parse_args()

    collector = TrainingDataCollector(args.data_dir)

    if args.command == 'stats':
        stats = collector.get_stats()
        print("\n" + "="*60)
        print("SENTINEL TRAINING DATA CORPUS")
        print("="*60)
        print(f"Total analyses:      {stats['total_analyses']:,}")
        print(f"Doctor corrections:  {stats['total_corrections']:,}")
        print(f"Signed reports:      {stats['total_signed_reports']:,}")
        print(f"Images saved:        {stats['images_saved']:,}")
        print(f"Avg correction %:    {stats['avg_correction_percentage']}%")
        print(f"\nModalities:")
        for mod, cnt in sorted(stats['modality_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {mod:20s}: {cnt}")
        print(f"\nTop pathologies:")
        for cls, cnt in sorted(stats['class_distribution'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:25s}: {cnt}")
        print(f"\nReady for fine-tuning:")
        for task, ready in stats['ready_for_finetuning'].items():
            icon = '✓' if ready else '✗'
            print(f"  {icon} {task}")
        print("="*60)

    elif args.command == 'export_densenet':
        out = args.output or 'export_densenet.jsonl'
        collector.export_for_densenet_finetuning(out)

    elif args.command == 'export_gemma':
        out = args.output or f'export_gemma_{args.language}.jsonl'
        collector.export_for_gemma_finetuning(out, language=args.language)
