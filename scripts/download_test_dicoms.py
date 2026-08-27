"""
================================================================================
  SENTINEL — DOWNLOAD TEST DICOM SAMPLES
================================================================================
  Downloads sample DICOM files from public sources for testing Sentinel
  without needing a real clinic.

  SOURCES (all FREE, no registration):
  - SIIM Sample Images (free chest X-rays)
  - PyDicom test files (built-in samples)
  - GitHub public DICOM repos
  - NIH demo DICOMs

  USAGE:
    python scripts/download_test_dicoms.py
    # Downloads to: data/test_dicoms/
================================================================================
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

OUTPUT_DIR = Path('data/test_dicoms')


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress."""
    try:
        print(f"  Downloading {dest.name}...", end=' ', flush=True)
        urllib.request.urlretrieve(url, str(dest))
        size_kb = dest.stat().st_size / 1024
        print(f"✓ ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def get_pydicom_samples():
    """Copy DICOM samples bundled with pydicom library."""
    print("\n[SOURCE 1] PyDicom built-in test files")
    try:
        import pydicom
        from pydicom.data import get_testdata_files

        # Get all DICOM test files
        all_files = get_testdata_files('*.dcm')
        if not all_files:
            print("  No pydicom test files found")
            return 0

        copied = 0
        for src_path in all_files[:10]:  # Limit to 10
            try:
                src = Path(src_path)
                # Filter for chest/CT/MRI files
                name = src.name.lower()
                if any(kw in name for kw in ['ct', 'mr', 'cr', 'chest', 'rg', 'sc']):
                    dest = OUTPUT_DIR / f"pydicom_{src.name}"
                    if not dest.exists():
                        shutil.copy(src, dest)
                        copied += 1
                        print(f"  ✓ {src.name}")
            except Exception:
                continue

        # If no medical files found, copy a few generic ones
        if copied == 0:
            for src_path in all_files[:5]:
                src = Path(src_path)
                dest = OUTPUT_DIR / f"pydicom_{src.name}"
                if not dest.exists():
                    shutil.copy(src, dest)
                    copied += 1
                    print(f"  ✓ {src.name}")

        return copied
    except ImportError:
        print("  pydicom not installed — pip install pydicom")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def download_orthanc_samples():
    """Download sample DICOMs from Orthanc public test data."""
    print("\n[SOURCE 2] Orthanc test DICOMs (free, public)")

    # Orthanc has free test DICOM samples on GitHub
    base_urls = [
        # Chest X-ray (CR)
        ("https://github.com/jodogne/Orthanc-Setup-Samples/raw/master/Samples/Knix/Loc/IM-0001-0001.dcm", "knee_localizer.dcm"),
        # Phenix samples (free CT)
        ("https://github.com/pydicom/pydicom-data/raw/master/data_store/data/CT_small.dcm", "ct_small_sample.dcm"),
        ("https://github.com/pydicom/pydicom-data/raw/master/data_store/data/MR_small.dcm", "mr_small_sample.dcm"),
    ]

    count = 0
    for url, name in base_urls:
        dest = OUTPUT_DIR / name
        if dest.exists():
            print(f"  ✓ {name} (already exists)")
            count += 1
            continue
        if download_file(url, dest):
            count += 1
    return count


def download_chest_xray_samples():
    """Download chest X-ray samples specifically."""
    print("\n[SOURCE 3] Chest X-ray samples (NIH public)")

    # NIH released some sample chest X-rays in public domain
    # These are JPGs; we'll convert to DICOM
    urls = [
        # NIH sample chest X-rays from their press kit
        ("https://nihcc.app.box.com/v/ChestXray-NIHCC", "nih_info.txt"),
    ]

    count = 0
    info_file = OUTPUT_DIR / "DOWNLOAD_MORE.md"
    info_file.write_text("""# More Test DICOMs — Free Sources

## For Real Chest X-rays:
Visit https://nihcc.app.box.com/v/ChestXray-NIHCC
Download images_001.tar.gz (~1GB) for 10000+ chest X-rays.

## Other Free Sources:

1. **The Cancer Imaging Archive (TCIA)**
   https://www.cancerimagingarchive.net
   Free DICOM datasets after registration

2. **Orthanc Test Data**
   https://orthanc-server.com/static.php?page=download
   Sample DICOMs for testing

3. **DICOM Library**
   https://www.dicomlibrary.com
   Anonymous DICOM image sharing

4. **OsiriX Sample Datasets**
   https://www.osirix-viewer.com/resources/dicom-image-library/
   Free educational DICOMs

5. **EuroRAD Cases**
   https://www.eurorad.org
   Real clinical cases with DICOMs

## How To Use Downloaded DICOMs:

1. Drop files into: data/test_dicoms/
2. Open Sentinel app: npm run dev
3. Click "Upload DICOM" in sidebar
4. Select your DICOM file
5. AI analyzes automatically
""")
    print(f"  ✓ Created info file: {info_file}")
    count += 1
    return count


def generate_synthetic_dicom():
    """Generate a synthetic DICOM file from a chest X-ray-like image."""
    print("\n[SOURCE 4] Generating synthetic test DICOM")

    try:
        import numpy as np
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        import tempfile
        from datetime import datetime

        # Create synthetic chest X-ray image (512x512)
        img = np.zeros((512, 512), dtype=np.uint16)

        # Background gradient (chest cavity)
        for y in range(512):
            for x in range(512):
                cx, cy = 256, 256
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                # Lung fields
                if 50 < dist < 200:
                    img[y, x] = 500 + np.random.randint(0, 200)
                else:
                    img[y, x] = 1500 + np.random.randint(0, 500)

        # Heart shadow (center-left)
        for y in range(180, 380):
            for x in range(190, 320):
                dist = np.sqrt((x - 250) ** 2 + (y - 280) ** 2)
                if dist < 80:
                    img[y, x] = 2000

        # Build DICOM
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # CR
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        dest = OUTPUT_DIR / "synthetic_chest_xray.dcm"
        ds = FileDataset(str(dest), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient info
        ds.PatientName = "TEST^SYNTHETIC"
        ds.PatientID = "TEST001"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        ds.PatientAge = "045Y"

        # Study info
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.1'
        ds.AccessionNumber = "TEST001"
        ds.StudyDescription = "TEST CHEST X-RAY"
        ds.SeriesDescription = "PA View"

        # Modality
        ds.Modality = "CR"
        ds.BodyPartExamined = "CHEST"
        ds.ViewPosition = "PA"

        # Image attributes
        ds.Rows = 512
        ds.Columns = 512
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelData = img.tobytes()

        # Save
        ds.save_as(str(dest), enforce_file_format=True)
        print(f"  ✓ Generated: {dest.name}")
        return 1

    except Exception as e:
        print(f"  ✗ Synthetic generation failed: {e}")
        return 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SENTINEL — TEST DICOM DOWNLOADER")
    print(f"  Output: {OUTPUT_DIR.absolute()}")
    print("=" * 70)

    total = 0
    total += get_pydicom_samples()
    total += download_orthanc_samples()
    total += download_chest_xray_samples()
    total += generate_synthetic_dicom()

    # Summary
    dicom_files = list(OUTPUT_DIR.glob("*.dcm"))

    print(f"\n{'=' * 70}")
    print(f"  DONE — {len(dicom_files)} DICOM files available")
    print(f"{'=' * 70}")

    for f in dicom_files:
        size_kb = f.stat().st_size / 1024
        print(f"  📁 {f.name} ({size_kb:.0f} KB)")

    print(f"\n  📍 Location: {OUTPUT_DIR.absolute()}")
    print(f"\n  TO TEST:")
    print(f"  1. Open Sentinel desktop app")
    print(f"  2. Click 'Upload DICOM' in sidebar")
    print(f"  3. Navigate to: {OUTPUT_DIR.absolute()}")
    print(f"  4. Select any .dcm file")
    print(f"\n  📚 For real chest X-rays:")
    print(f"     Read: {OUTPUT_DIR / 'DOWNLOAD_MORE.md'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
