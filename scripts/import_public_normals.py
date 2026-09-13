#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — IMPORT PUBLIC HEALTHY-BRAIN MRI (NIfTI) AS 'normal' TRAINING SLICES
================================================================================
  Converts NIfTI volumes of healthy subjects into central axial slices in the
  SAME format as scripts/build_training_set.py (5 central axial slices,
  robust_normalize_slice, 8-bit RGB PNG) so they can be added to the 'normal'
  class of the binary triage set.

  Source used (2026-09): OpenNeuro ds000221 MPI-Leipzig Mind-Brain-Body (LEMON),
  license CC0, healthy adults, raw T2w. Attribution kept in the output manifest.

  Orientation: volumes are reoriented to RAS (nibabel as_closest_canonical),
  axial slices taken along the superior axis, rendered anterior-up in
  radiological convention (patient left on image right) to match DICOM display.

  Usage:
    python scripts/import_public_normals.py --nii-dir <dir with *.nii.gz> \
        --out data/public_normals/lemon --source "OpenNeuro ds000221 (CC0)" [--limit N] [--montage m.png]
================================================================================
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.brain_mri_preprocessor import robust_normalize_slice


def central_axial_slices(vol: np.ndarray, zooms: tuple, k: int = 5) -> list[np.ndarray]:
    """vol is RAS (x=L->R, y=P->A, z=I->S). Unlike DICOM brain stacks (which cover
    just the brain), whole-head research volumes put the volume centre at orbit /
    cerebellum level — so centre on the axial level of MAXIMAL brain cross-section
    (≈ ventricle / basal-ganglia level, matching the local training slices) and
    step by 1/12 of the brain's own superior-inferior extent, as build_training_set
    does over a brain-only stack. In-plane aspect is corrected with voxel spacing."""
    # Anatomical anchor: the local DICOM training slices sit around the lateral
    # ventricles, ~60 mm below the vertex. Whole-head research volumes reach far
    # below the brain, so centre on (vertex - 60 mm) rather than the volume centre,
    # and step ~9 mm (build_training_set steps d//12 ≈ 10 mm over a brain stack).
    zz = float(zooms[2])
    thr = np.percentile(vol[vol > 0], 40) if np.any(vol > 0) else 0
    area = (vol > thr).reshape(-1, vol.shape[2]).sum(axis=0)
    zs = np.where(area > 0.25 * area.max())[0]
    z_top = int(zs.max()) if len(zs) else vol.shape[2] - 1
    c = int(round(z_top - 60.0 / zz))
    step = max(1, int(round(9.0 / zz)))
    offs = range(-(k // 2) * step, (k // 2 + 1) * step, step)
    idxs = sorted(set(max(0, min(vol.shape[2] - 1, c + off)) for off in offs))[:k]
    sx, sy = float(zooms[0]), float(zooms[1])
    out = []
    for i in idxs:
        sl = vol[:, :, i]                 # (X, Y)
        img = np.rot90(sl)                # (Y, X), anterior at top
        img = np.fliplr(img)              # radiological: patient left on image right
        norm = robust_normalize_slice(img.astype(np.float32))
        h, w = norm.shape                 # rows ~ y spacing, cols ~ x spacing
        tgt_w, tgt_h = int(round(w * sx)), int(round(h * sy))   # -> 1 mm isotropic
        if (tgt_w, tgt_h) != (w, h):
            norm = np.asarray(Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8))
                              .resize((tgt_w, tgt_h), Image.BILINEAR)).astype(np.float32) / 255.0
        out.append(norm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii-dir', required=True)
    ap.add_argument('--out', default='data/public_normals/lemon')
    ap.add_argument('--source', default='OpenNeuro ds000221 LEMON (CC0)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--montage', default=None)
    args = ap.parse_args()

    files = sorted(Path(args.nii_dir).glob('*.nii*'))
    if args.limit: files = files[:args.limit]
    out = Path(args.out) / 'normal'; out.mkdir(parents=True, exist_ok=True)
    manifest = {'source': args.source, 'created_at': datetime.now().isoformat(timespec='seconds'),
                'k_slices': args.k, 'subjects': []}
    tiles = []
    for fp in files:
        try:
            img = nib.as_closest_canonical(nib.load(str(fp)))
            vol = np.asanyarray(img.dataobj).astype(np.float32)
            if vol.ndim == 4: vol = vol[..., 0]
        except Exception as e:
            print(f"  skip {fp.name}: {e}"); continue
        sid = fp.name.split('_')[0]
        n = 0
        for j, slc in enumerate(central_axial_slices(vol, img.header.get_zooms()[:3], args.k)):
            arr = (np.clip(slc, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).convert('RGB').save(out / f"{sid}_{j}.png")
            n += 1
            if args.montage and len(tiles) < 12 and j == 2: tiles.append(arr)
        manifest['subjects'].append({'id': sid, 'file': fp.name, 'slices': n, 'shape': list(vol.shape)})
    (Path(args.out) / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    tot = sum(s['slices'] for s in manifest['subjects'])
    print(f"  ✓ {len(manifest['subjects'])} subjects → {tot} normal slices in {out}")
    if args.montage and tiles:
        W = H = 200
        m = Image.new('L', (4 * W, ((len(tiles) + 3) // 4) * H))
        for i, t in enumerate(tiles):
            m.paste(Image.fromarray(t).resize((W, H)), ((i % 4) * W, (i // 4) * H))
        m.save(args.montage); print(f"  montage → {args.montage}")


if __name__ == '__main__':
    main()
