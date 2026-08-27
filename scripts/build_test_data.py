#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BUILD LOCAL TEST DATA (no internet bottlenecks)
================================================================================

  When HuggingFace dataset downloads are too slow / flaky, this script
  produces a working test dataset locally in ~30 seconds:

    1. Copies pydicom's bundled test DICOMs (CT_small, MR_small, JPEG_LS)
    2. Generates synthetic chest X-ray DICOMs from gradients
    3. Generates synthetic brain MRI slices
    4. Builds a directory layout that matches what the desktop app
       expects for demos.

  Output:
    data/test_dicoms/
      chest/      (synthetic CR-style)
      brain/      (synthetic MR-style)
      headct/     (synthetic CT-style)
      pydicom_samples/  (real pydicom test files)
      INDEX.md    (one-line summary of each file)

  No internet required. Total size: ~5 MB.
================================================================================
"""

from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime

OUT = Path('data/test_dicoms')


def make_dicom(modality: str, body_part: str, pixel_array: np.ndarray,
                 patient_id: str, description: str) -> Dataset:
    """Build a minimal valid DICOM dataset with the given pixel array."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.3.4'

    ds = FileDataset('', {}, file_meta=file_meta, preamble=b'\0' * 128)
    ds.PatientName = 'TEST^SUBJECT'
    ds.PatientID = patient_id
    ds.PatientBirthDate = '19800101'
    ds.PatientSex = 'O'
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.StudyDescription = description
    ds.SeriesDescription = description
    today = datetime.date.today()
    ds.StudyDate = today.strftime('%Y%m%d')
    ds.StudyTime = '120000'
    ds.AccessionNumber = 'ACC' + patient_id
    ds.Manufacturer = 'Sentinel Synthetic Generator'
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Rows, ds.Columns = pixel_array.shape
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def synth_chest_xray(seed: int = 0) -> np.ndarray:
    """Generate a synthetic 512×512 chest X-ray-ish image."""
    rng = np.random.RandomState(seed)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2

    # Lung fields — two darker ovals
    img = np.full((h, w), 8000.0)
    for sign in (-1, 1):
        rx, ry = 110, 170
        cx_l = cx + sign * 90
        cy_l = cy
        mask = ((xx - cx_l) / rx) ** 2 + ((yy - cy_l) / ry) ** 2 < 1
        img[mask] -= 4000
    # Spine — vertical bright line
    img[:, cx-8:cx+8] += 6000
    # Ribs
    for r in range(80, 440, 35):
        img[r:r+3, 50:462] += 1500
    # Soft tissue gradient
    img += rng.randn(h, w) * 300
    img = np.clip(img, 0, 65535)
    return img.astype(np.uint16)


def synth_brain_mri(seed: int = 0, with_lesion: bool = False) -> np.ndarray:
    """Generate a synthetic 256×256 brain MRI slice."""
    rng = np.random.RandomState(seed)
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2

    # Skull — bright outer ring
    skull_outer = ((xx - cx) ** 2 + (yy - cy) ** 2) < (110 ** 2)
    skull_inner = ((xx - cx) ** 2 + (yy - cy) ** 2) < (95 ** 2)
    brain = ((xx - cx) ** 2 + (yy - cy) ** 2) < (90 ** 2)

    img = np.zeros((h, w))
    img[skull_outer & ~skull_inner] = 30000  # skull
    img[brain] = 18000 + rng.randn(int(brain.sum())) * 1500  # gray matter

    # Ventricles — two dark spots
    for sx in (-15, 15):
        v = ((xx - cx - sx) / 8) ** 2 + ((yy - cy + 5) / 18) ** 2 < 1
        img[v] = 4000

    # Optional lesion (bright spot for tumor-like demo)
    if with_lesion:
        lx = ((xx - cx + 35) ** 2 + (yy - cy - 20) ** 2) < (15 ** 2)
        img[lx] = 35000

    img += rng.randn(h, w) * 400
    img = np.clip(img, 0, 65535)
    return img.astype(np.uint16)


def synth_head_ct(seed: int = 0, with_bleed: bool = False) -> np.ndarray:
    """Synthetic 512×512 head CT slice."""
    rng = np.random.RandomState(seed)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2

    skull_outer = ((xx - cx) ** 2 + (yy - cy) ** 2) < (220 ** 2)
    skull_inner = ((xx - cx) ** 2 + (yy - cy) ** 2) < (200 ** 2)
    brain = ((xx - cx) ** 2 + (yy - cy) ** 2) < (195 ** 2)

    img = np.zeros((h, w))
    img[skull_outer & ~skull_inner] = 50000  # high-density skull
    img[brain] = 15000 + rng.randn(int(brain.sum())) * 1200

    if with_bleed:
        # Bright extra-axial blob (epidural-style)
        bleed = (((xx - 200) / 50) ** 2 + ((yy - cy) / 25) ** 2) < 1
        img[bleed & brain] = 38000

    img += rng.randn(h, w) * 200
    img = np.clip(img, 0, 65535)
    return img.astype(np.uint16)


def copy_pydicom_samples():
    """Copy pydicom's bundled test files."""
    try:
        from pydicom.data import get_testdata_files
    except Exception:
        return []

    out_dir = OUT / 'pydicom_samples'
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_names = ['CT_small.dcm', 'MR_small.dcm', 'rtplan.dcm']
    copied = []
    for name in sample_names:
        try:
            srcs = get_testdata_files(name)
            if srcs:
                src = Path(srcs[0])
                dst = out_dir / src.name
                shutil.copy(src, dst)
                copied.append(dst)
        except Exception as e:
            print(f'  ! could not copy {name}: {e}')
    return copied


def build_index(files):
    """Write a markdown index of every test file."""
    index = OUT / 'INDEX.md'
    with open(index, 'w') as f:
        f.write('# Sentinel Test DICOMs\n\n')
        f.write('Generated locally — no internet needed.\n\n')
        f.write('| File | Modality | Body | Description |\n')
        f.write('|---|---|---|---|\n')
        for fp, modality, body, desc in files:
            rel = fp.relative_to(OUT.parent)
            f.write(f'| `{rel}` | {modality} | {body} | {desc} |\n')
    print(f'  ✓ wrote {index}')


def main():
    if not OUT.parent.exists():
        OUT.parent.mkdir(parents=True)
    print(f'Building test DICOMs into {OUT}/')
    OUT.mkdir(exist_ok=True)
    (OUT / 'chest').mkdir(exist_ok=True)
    (OUT / 'brain').mkdir(exist_ok=True)
    (OUT / 'headct').mkdir(exist_ok=True)

    files = []

    # Chest synthetic
    print('  Generating synthetic chest X-rays…')
    for i in range(3):
        arr = synth_chest_xray(seed=i)
        ds = make_dicom('CR', 'CHEST', arr,
                          f'CHEST{i:03d}', f'Synthetic chest X-ray #{i}')
        fp = OUT / 'chest' / f'chest_{i:03d}.dcm'
        ds.save_as(str(fp), write_like_original=False)
        files.append((fp, 'CR', 'CHEST', f'Synthetic chest #{i}'))

    # Brain MRI — 5 normal + 2 with synthetic "lesion"
    print('  Generating synthetic brain MRI slices…')
    for i in range(5):
        arr = synth_brain_mri(seed=i, with_lesion=False)
        ds = make_dicom('MR', 'BRAIN', arr,
                          f'BRAIN{i:03d}', f'Synthetic brain MRI normal #{i}')
        fp = OUT / 'brain' / f'brain_normal_{i:03d}.dcm'
        ds.save_as(str(fp), write_like_original=False)
        files.append((fp, 'MR', 'BRAIN', f'Synthetic brain MRI normal #{i}'))

    for i in range(2):
        arr = synth_brain_mri(seed=100 + i, with_lesion=True)
        ds = make_dicom('MR', 'BRAIN', arr,
                          f'BRTUM{i:03d}', f'Synthetic brain MRI w/lesion #{i}')
        fp = OUT / 'brain' / f'brain_lesion_{i:03d}.dcm'
        ds.save_as(str(fp), write_like_original=False)
        files.append((fp, 'MR', 'BRAIN', f'Synthetic brain MRI w/lesion #{i}'))

    # Head CT — 2 normal + 1 with bleed
    print('  Generating synthetic head CT slices…')
    for i in range(2):
        arr = synth_head_ct(seed=i, with_bleed=False)
        ds = make_dicom('CT', 'HEAD', arr,
                          f'HEADCT{i:03d}', f'Synthetic head CT normal #{i}')
        fp = OUT / 'headct' / f'headct_normal_{i:03d}.dcm'
        ds.save_as(str(fp), write_like_original=False)
        files.append((fp, 'CT', 'HEAD', f'Synthetic head CT normal #{i}'))

    arr = synth_head_ct(seed=200, with_bleed=True)
    ds = make_dicom('CT', 'HEAD', arr,
                      'HEADCT_BLEED', 'Synthetic head CT with hemorrhage')
    fp = OUT / 'headct' / 'headct_bleed_001.dcm'
    ds.save_as(str(fp), write_like_original=False)
    files.append((fp, 'CT', 'HEAD', 'Synthetic head CT with hemorrhage'))

    # pydicom bundled samples
    print('  Copying pydicom bundled samples…')
    copied = copy_pydicom_samples()
    for c in copied:
        files.append((c, '?', '?', 'pydicom built-in test file'))
        print(f'    + {c.name}')

    # Index
    print('  Writing INDEX.md…')
    build_index(files)

    # Summary
    total_mb = sum(f.stat().st_size for f, *_ in files) / (1024 * 1024)
    print()
    print(f'  ✓ {len(files)} test DICOMs built — {total_mb:.1f} MB total')
    print(f'    Demo with: data/test_dicoms/brain/brain_lesion_000.dcm')
    print(f'    Or:        data/test_dicoms/headct/headct_bleed_001.dcm')


if __name__ == '__main__':
    main()
