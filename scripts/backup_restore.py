#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BACKUP & RESTORE
================================================================================

  Backs up everything a clinic cannot afford to lose:
    • SQLite database (studies, reports, users, audit log)
    • Signed report PDFs
    • Doctor corrections (training data)
    • license.dat + jwt_secret.key

  Produces a single timestamped, checksummed .tar.gz. Run on a schedule
  (cron / Task Scheduler) to an external drive or NAS.

  Usage:
    python scripts/backup_restore.py backup  --out /path/to/backups
    python scripts/backup_restore.py restore --archive /path/to/backups/sentinel_clinic_YYYYMMDD.tar.gz
    python scripts/backup_restore.py verify  --archive <file>

  Hospital note: keep 3 copies (local + external + offsite), test restore monthly.
================================================================================
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).parent.parent

# What to back up (relative to repo root) — created if missing, skipped if absent.
BACKUP_PATHS = [
    'data/sentinel.db',
    'data/sentinel.db-wal',
    'data/sentinel.db-shm',
    'data/reports',
    'data/doctor_corrections',
    'data/training_corpus',
    'vendor_keys/sentinel_dev_license.dat',
]


def _user_data_dir() -> Path:
    if os.name == 'nt':
        return Path(os.environ.get('APPDATA', Path.home())) / 'sentinel-medical-ai'
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'sentinel-medical-ai'
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'sentinel-medical-ai'


def do_backup(out_dir: Path, stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f'sentinel_clinic_{stamp}.tar.gz'

    added = 0
    with tarfile.open(archive, 'w:gz') as tar:
        for rel in BACKUP_PATHS:
            p = REPO / rel
            if p.exists():
                tar.add(str(p), arcname=rel)
                added += 1
        # license + jwt secret live in the user-data dir
        udir = _user_data_dir()
        for name in ('license.dat', 'jwt_secret.key'):
            f = udir / name
            if f.exists():
                tar.add(str(f), arcname=f'userdata/{name}')
                added += 1

    # Checksum
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out_dir / f'sentinel_clinic_{stamp}.sha256').write_text(f'{sha}  {archive.name}\n')

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f'✓ Backup: {archive}')
    print(f'  {added} items, {size_mb:.1f} MB')
    print(f'  SHA-256: {sha}')
    print(f'  → Copy this to an external drive AND an offsite location.')
    return archive


def do_verify(archive: Path) -> bool:
    sha_file = archive.with_suffix('').with_suffix('.sha256')
    if not sha_file.exists():
        sha_file = Path(str(archive).replace('.tar.gz', '.sha256'))
    if not sha_file.exists():
        print(f'⚠ No .sha256 next to archive — cannot verify integrity.')
        return False
    expected = sha_file.read_text().split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    ok = expected == actual
    print(f'{"✓ INTEGRITY OK" if ok else "✗ CORRUPT — checksum mismatch"}')
    print(f'  expected: {expected}')
    print(f'  actual:   {actual}')
    return ok


def do_restore(archive: Path):
    if not archive.exists():
        print(f'✗ Archive not found: {archive}')
        sys.exit(1)
    if not do_verify(archive):
        print('✗ Refusing to restore a corrupt/unverified archive.')
        sys.exit(1)
    print(f'Restoring from {archive} …')
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.startswith('userdata/'):
                dest = _user_data_dir() / member.name.replace('userdata/', '')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src:
                    dest.write_bytes(src.read())
            else:
                tar.extract(member, path=str(REPO))
    print('✓ Restore complete. Restart the Sentinel server.')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pb = sub.add_parser('backup'); pb.add_argument('--out', default=str(Path.home() / 'sentinel_backups'))
    pr = sub.add_parser('restore'); pr.add_argument('--archive', required=True)
    pv = sub.add_parser('verify'); pv.add_argument('--archive', required=True)
    args = p.parse_args()

    if args.cmd == 'backup':
        # Stamp passed via env to keep it deterministic if scheduled; else now.
        stamp = os.environ.get('BACKUP_STAMP') or time.strftime('%Y%m%d_%H%M%S')
        do_backup(Path(args.out), stamp)
    elif args.cmd == 'verify':
        do_verify(Path(args.archive))
    elif args.cmd == 'restore':
        do_restore(Path(args.archive))


if __name__ == '__main__':
    main()
