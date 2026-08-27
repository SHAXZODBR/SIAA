"""
================================================================================
  SENTINEL MEDICAL AI — CLINIC DATA COLLECTION & LABELING TOOL
================================================================================
  Tool for collecting labeled data from Uzbekistan clinics for fine-tuning.

  WHAT IT DOES:
  1. Walk through clinic DICOMs
  2. Display each image
  3. Radiologist adds labels + writes report
  4. Saves everything in format ready for fine-tuning
     - DenseNet121 (pathology labels)
     - Gemma 3 VL (report text + image pairs)

  OUTPUT FORMAT:
    /clinic_data/
      ├── images/           ← preprocessed PNG images
      ├── labels.json       ← multi-label classifications
      ├── reports.jsonl     ← image + findings + report tuples for Gemma
      └── metadata.json     ← clinic info, collection stats

  USAGE:
    # Label clinic DICOMs:
    python clinic_data_tool.py label --dicom-dir /path/to/clinic/dicoms

    # View statistics:
    python clinic_data_tool.py stats --data-dir ./clinic_data

    # Export for fine-tuning:
    python clinic_data_tool.py export --format gemma --output gemma_dataset.jsonl
    python clinic_data_tool.py export --format densenet --output densenet_labels.json
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pathology classes
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia', 'Normal'
]

# Report section prompts (Russian)
REPORT_SECTIONS_RU = {
    'indication': 'Клиническое показание (зачем назначили обследование):',
    'technique': 'Методика (что сделали — вид снимка):',
    'description': 'Описание (подробно что видно на снимке):',
    'conclusion': 'Заключение (диагностический вывод):',
    'recommendation': 'Рекомендации (что делать дальше):',
}

REPORT_SECTIONS_UZ = {
    'indication': "Klinik ko'rsatma (nima uchun tekshiruv tayinlangan):",
    'technique': "Metodika (nima qilingan — surat turi):",
    'description': "Tavsif (surat nima ko'rsatayapti):",
    'conclusion': "Xulosa (tashxis):",
    'recommendation': "Tavsiyalar (keyingi harakatlar):",
}


class ClinicDataTool:
    """Manages clinic data collection and labeling."""

    def __init__(self, data_dir: str = "./clinic_data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_path = self.data_dir / "labels.json"
        self.reports_path = self.data_dir / "reports.jsonl"
        self.metadata_path = self.data_dir / "metadata.json"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.labels = self._load_json(self.labels_path) or {}
        self.metadata = self._load_json(self.metadata_path) or {
            'created': datetime.now().isoformat(),
            'clinic_name': 'Unknown Clinic',
            'clinic_city': 'Tashkent',
            'total_collected': 0,
            'collection_sessions': [],
        }

    def _load_json(self, path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ===========================================================
    # LABELING WORKFLOW
    # ===========================================================

    def label_dicoms(self, dicom_dir: str, language: str = 'ru', resume: bool = True):
        """Interactive labeling: show image → radiologist labels → save."""
        from src.pipeline.dicom_loader import load_dicom_folder
        from src.pipeline.preprocessor import preprocess_study, save_preprocessed

        dicom_dir = Path(dicom_dir)
        print("\n" + "="*70)
        print(f"  SENTINEL CLINIC DATA COLLECTION")
        print(f"  Loading DICOMs from {dicom_dir}")
        print("="*70)

        # Load DICOM files
        studies = load_dicom_folder(str(dicom_dir))
        print(f"\nFound {len(studies)} DICOM files")

        # Filter out already labeled (if resuming)
        if resume:
            to_label = [s for s in studies if Path(s.file_path).stem not in self.labels]
            print(f"Resuming: {len(self.labels)} already labeled, {len(to_label)} remaining")
        else:
            to_label = studies

        if not to_label:
            print("\nAll images already labeled!")
            return

        session_count = 0

        for i, study in enumerate(to_label):
            stem = Path(study.file_path).stem

            print(f"\n{'='*70}")
            print(f"  IMAGE {i+1}/{len(to_label)}: {stem}")
            print(f"{'='*70}")
            print(f"  Modality: {study.modality}")
            print(f"  Body part: {study.body_part}")
            print(f"  Size: {study.rows}x{study.cols}")
            print(f"  Acquisition: {study.study_date}")

            # Preprocess + save image
            try:
                processed = preprocess_study(study, target_size=512)
                save_preprocessed(processed, self.images_dir / f"{stem}.png")
            except Exception as e:
                print(f"  ERROR processing: {e}")
                continue

            # Show available classes
            print(f"\n  PATHOLOGY CLASSES:")
            for idx, cls in enumerate(CLASSES):
                print(f"    {idx:2d}. {cls}")
            print(f"\n  Commands:")
            print(f"    numbers (comma-sep): e.g. '7,3' → Pneumonia + Effusion")
            print(f"    0 or '14': Normal")
            print(f"    'r': Also write report")
            print(f"    's': Skip this image")
            print(f"    'q': Save and quit")
            print(f"    'u': Undo last entry")

            while True:
                answer = input(f"\n  Labels: ").strip().lower()

                if answer == 'q':
                    print(f"\n  Saving {len(self.labels)} labels...")
                    self._save_json(self.labels_path, self.labels)
                    self._update_metadata(session_count)
                    print(f"  Done! Resume later with: python clinic_data_tool.py label --resume")
                    return

                if answer == 's':
                    break

                if answer == 'u' and self.labels:
                    last_key = list(self.labels.keys())[-1]
                    del self.labels[last_key]
                    print(f"  Undone: {last_key}")
                    self._save_json(self.labels_path, self.labels)
                    continue

                try:
                    indices = [int(x.strip()) for x in answer.split(',')]
                    selected = []
                    for idx in indices:
                        if 0 <= idx < len(CLASSES):
                            selected.append(CLASSES[idx])
                    if not selected:
                        print("  Invalid classes")
                        continue

                    # Save label
                    self.labels[stem] = selected
                    session_count += 1

                    # Ask for report
                    if input("  Write report? (y/n): ").strip().lower() == 'y':
                        report = self._collect_report(stem, selected, language)
                        if report:
                            self._save_report(stem, selected, report, language)

                    print(f"  ✓ Saved: {', '.join(selected)}")

                    # Auto-save every 10 images
                    if session_count % 10 == 0:
                        self._save_json(self.labels_path, self.labels)
                        print(f"  [auto-saved — {len(self.labels)} total]")

                    break
                except ValueError:
                    print("  Invalid input. Enter comma-separated numbers.")

        # Final save
        self._save_json(self.labels_path, self.labels)
        self._update_metadata(session_count)
        print(f"\n  Session complete! Labeled {session_count} images this session.")
        print(f"  Total labels: {len(self.labels)}")

    def _collect_report(self, image_stem: str, classes: list, language: str) -> str:
        """Interactive radiology report collection."""
        sections = REPORT_SECTIONS_RU if language == 'ru' else REPORT_SECTIONS_UZ
        print(f"\n  === WRITING REPORT for {image_stem} ===")
        print(f"  Classes: {', '.join(classes)}")

        report_parts = []
        for section_key, prompt in sections.items():
            print(f"\n  {prompt}")
            text = input("  > ").strip()
            if text:
                report_parts.append(f"{section_key.upper()}:\n{text}")

        full_report = '\n\n'.join(report_parts)
        return full_report

    def _save_report(self, image_stem: str, classes: list, report: str, language: str):
        """Append a report to the JSONL dataset."""
        findings_text = f"DenseNet findings: {', '.join(classes)}"

        entry = {
            'image_path': str(self.images_dir / f"{image_stem}.png"),
            'findings_text': findings_text,
            'report': report,
            'language': language,
            'classes': classes,
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.reports_path, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _update_metadata(self, session_count: int):
        self.metadata['total_collected'] = len(self.labels)
        self.metadata['last_session'] = datetime.now().isoformat()
        self.metadata['collection_sessions'].append({
            'date': datetime.now().isoformat(),
            'count': session_count,
        })
        self._save_json(self.metadata_path, self.metadata)

    # ===========================================================
    # STATISTICS
    # ===========================================================

    def show_stats(self):
        """Show collection statistics."""
        print("\n" + "="*70)
        print(f"  CLINIC DATA STATISTICS — {self.data_dir}")
        print("="*70)

        total = len(self.labels)
        print(f"\n  Total labeled: {total}")
        print(f"  Clinic: {self.metadata.get('clinic_name', 'Unknown')}")
        print(f"  Location: {self.metadata.get('clinic_city', 'Unknown')}")

        # Class distribution
        class_counts = {}
        for stem, classes in self.labels.items():
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1

        print(f"\n  CLASS DISTRIBUTION:")
        for cls in CLASSES:
            count = class_counts.get(cls, 0)
            bar = '█' * (count * 30 // max(max(class_counts.values(), default=1), 1))
            print(f"    {cls:25s} {count:>5d} {bar}")

        # Report stats
        if self.reports_path.exists():
            with open(self.reports_path) as f:
                report_lines = sum(1 for _ in f)
            print(f"\n  Reports written: {report_lines}")

        # Recommendations
        print(f"\n  RECOMMENDATIONS:")
        if total < 100:
            print(f"    ⚠ Need more data! Collect at least 500 images for fine-tuning.")
        elif total < 500:
            print(f"    → Continue collecting. Current: {total}. Aim for 500+.")
        elif total < 2000:
            print(f"    ✓ Good progress. {total} images ready for fine-tuning.")
        else:
            print(f"    ✓ Excellent! {total} images is plenty for production fine-tuning.")

        # Check class balance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if max_count > 10 * min_count:
                print(f"    ⚠ Class imbalance detected. Some classes have 10x more data.")
                print(f"      This is normal for medical data, but use class weights during training.")

        print("="*70)

    # ===========================================================
    # EXPORT
    # ===========================================================

    def export_for_densenet(self, output_path: str):
        """Export labels in format expected by DenseNet training."""
        self._save_json(Path(output_path), self.labels)
        print(f"✓ Exported {len(self.labels)} labels to {output_path}")
        print(f"  Use for fine-tuning:")
        print(f"    python run_finetune.py --base-model models/densenet/best_model.pt \\")
        print(f"        --dicom-dir {self.images_dir} --labels {output_path}")

    def export_for_gemma(self, output_path: str):
        """Export (image, findings, report) tuples for Gemma VL fine-tuning."""
        if not self.reports_path.exists():
            print(f"No reports collected yet.")
            return

        count = 0
        with open(self.reports_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                f_out.write(line)
                count += 1

        print(f"✓ Exported {count} report examples to {output_path}")
        print(f"  Use for Gemma fine-tuning:")
        print(f"    python gemma_vl_finetune.py --mode train")
        print(f"    (set dataset_path={output_path} in the script config)")


# ===========================================================
# MAIN
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="Sentinel Clinic Data Tool")
    sub = parser.add_subparsers(dest='command', required=True)

    p_label = sub.add_parser('label', help='Label DICOM images')
    p_label.add_argument('--dicom-dir', required=True, help='Directory with DICOM files')
    p_label.add_argument('--data-dir', default='./clinic_data', help='Output data dir')
    p_label.add_argument('--language', default='ru', choices=['ru', 'uz', 'en'])
    p_label.add_argument('--fresh', action='store_true', help='Start fresh (don\'t resume)')

    p_stats = sub.add_parser('stats', help='Show collection statistics')
    p_stats.add_argument('--data-dir', default='./clinic_data')

    p_export = sub.add_parser('export', help='Export data for training')
    p_export.add_argument('--format', choices=['densenet', 'gemma'], required=True)
    p_export.add_argument('--output', required=True, help='Output file path')
    p_export.add_argument('--data-dir', default='./clinic_data')

    args = parser.parse_args()
    tool = ClinicDataTool(args.data_dir)

    if args.command == 'label':
        tool.label_dicoms(args.dicom_dir, args.language, resume=not args.fresh)
    elif args.command == 'stats':
        tool.show_stats()
    elif args.command == 'export':
        if args.format == 'densenet':
            tool.export_for_densenet(args.output)
        else:
            tool.export_for_gemma(args.output)


if __name__ == '__main__':
    main()
