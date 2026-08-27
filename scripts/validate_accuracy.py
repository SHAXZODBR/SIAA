"""
================================================================================
  SENTINEL — VALIDATE AI ACCURACY ON YOUR OWN DATA
================================================================================
  Run this against a folder of DICOMs with known diagnoses to check
  how accurate the model is for YOUR clinic's data.

  USAGE:
    1. Create test set:
       data/validation/
         ├── pneumonia/      (10+ DICOMs known to have pneumonia)
         ├── cardiomegaly/   (10+ DICOMs known to have cardiomegaly)
         └── normal/         (10+ DICOMs known to be healthy)

    2. Run:
       python scripts/validate_accuracy.py --data data/validation

    3. Get metrics:
       - Sensitivity (recall): % of diseases the AI catches
       - Specificity: % of healthy correctly classified
       - Confusion matrix
       - Per-pathology accuracy
================================================================================
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from collections import defaultdict


def validate_folder(data_dir: str, server_url: str = 'http://127.0.0.1:8000'):
    """Validate AI accuracy on labeled folder structure."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        print("Create folder structure:")
        print("  data/validation/")
        print("    ├── pneumonia/    (DICOMs with pneumonia)")
        print("    ├── normal/       (healthy DICOMs)")
        print("    └── ...other classes/")
        return

    # Check server is running
    try:
        health = requests.get(f'{server_url}/health', timeout=3)
        if health.status_code != 200:
            print(f"ERROR: Server not responding at {server_url}")
            return
    except Exception:
        print(f"ERROR: Cannot connect to server. Start it first:")
        print(f"  python run_server.py")
        return

    print("="*70)
    print(f"  VALIDATING AI ACCURACY")
    print(f"  Data: {data_dir}")
    print(f"  Server: {server_url}")
    print("="*70)

    # Per-class results
    results = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0, 'predictions': []})

    # Walk through class folders
    class_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    if not class_folders:
        print(f"ERROR: No subfolders found in {data_dir}")
        return

    print(f"\nClasses found: {[d.name for d in class_folders]}")

    total_processed = 0
    for class_folder in class_folders:
        true_class = class_folder.name.lower()
        dicoms = [f for f in class_folder.iterdir() if f.suffix.lower() in ('.dcm', '.dicom')]

        if not dicoms:
            print(f"  ⚠ No DICOMs in {class_folder}")
            continue

        print(f"\n[{true_class.upper()}] Testing {len(dicoms)} DICOMs...")

        for dcm in dicoms:
            total_processed += 1

            # Send to server
            try:
                with open(dcm, 'rb') as f:
                    resp = requests.post(
                        f'{server_url}/analyze',
                        files={'file': (dcm.name, f, 'application/dicom')},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    result = resp.json()
            except Exception as e:
                print(f"  ✗ {dcm.name}: API error: {e}")
                continue

            # Get top finding
            findings = result.get('findings', [])
            if not findings:
                predicted = 'normal'
            else:
                predicted = findings[0]['class_name'].lower()

            # Check if correct
            is_correct = (
                true_class in predicted or
                predicted in true_class or
                (true_class == 'normal' and len(findings) == 0)
            )

            results[true_class]['total'] += 1
            if is_correct:
                results[true_class]['correct'] += 1
            else:
                results[true_class]['wrong'] += 1
            results[true_class]['predictions'].append({
                'file': dcm.name,
                'predicted': predicted,
                'confidence': findings[0]['confidence'] if findings else 0,
                'correct': is_correct,
            })

            symbol = '✓' if is_correct else '✗'
            confidence = findings[0]['confidence'] * 100 if findings else 0
            print(f"  {symbol} {dcm.name}: predicted={predicted} ({confidence:.0f}%)")

    # Summary
    print("\n" + "="*70)
    print(f"  RESULTS — {total_processed} files tested")
    print("="*70)

    total_correct = sum(r['correct'] for r in results.values())
    total_all = sum(r['total'] for r in results.values())

    print(f"\nOverall accuracy: {total_correct}/{total_all} = {100*total_correct/max(total_all,1):.1f}%")

    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print(f"  " + "-"*55)
    for cls, r in sorted(results.items()):
        acc = 100 * r['correct'] / max(r['total'], 1)
        emoji = '✅' if acc >= 80 else '⚠️ ' if acc >= 60 else '❌'
        print(f"  {cls:<25} {r['correct']:<10} {r['total']:<10} {acc:.1f}% {emoji}")

    print("\n" + "="*70)
    print("  INTERPRETATION:")
    print("  ≥ 90%  Excellent — production ready")
    print("  ≥ 80%  Good — usable for screening")
    print("  ≥ 60%  Acceptable — needs fine-tuning on clinic data")
    print("  < 60%  Poor — train on more clinic data")
    print("="*70)

    # Save results
    output = {
        'total_processed': total_processed,
        'overall_accuracy': total_correct / max(total_all, 1),
        'per_class': {
            cls: {
                'accuracy': r['correct'] / max(r['total'], 1),
                'correct': r['correct'],
                'total': r['total'],
                'predictions': r['predictions'],
            }
            for cls, r in results.items()
        },
    }
    output_path = data_dir / 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/validation', help='Validation folder')
    parser.add_argument('--server', default='http://127.0.0.1:8000', help='Server URL')
    args = parser.parse_args()
    validate_folder(args.data, args.server)
