#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — ARCHIVE A PRIORITISED PART OF A BATCH OFF THE CLINIC DRIVE
================================================================================
  WHAT
    Copies study folders from the delivery drive to a local archive, in PRIORITY
    order, as the ORIGINAL DICOM files (nothing is converted or anonymised).
    Each --rule is "<modality>:<body part pattern>[:cap=N]"; a study is archived
    by the FIRST rule that matches its row in the index (index_new_studies.py):
      raw     (no cap) every file of the study folder
      cap=N   at most N .dcm files per series directory, evenly spaced over the
              sorted file names, first and last included. A thin-slice CT series
              has 600+ files; a spaced sample keeps every series and the whole
              anatomical range at a fraction of the size. Other files (.bmp …)
              are skipped in this mode.
    Every copy is verified: sha256 of the bytes read from the drive must equal
    sha256 of the file re-read from the archive.

  WHY
    The clinic drive is wiped and reused for the next delivery. The reports and
    the built training slices live on this Mac, but the images themselves exist
    only on that drive: without them a study can never be re-processed (other
    slices, other sequences, another body part's model). The full, lossless way
    is scripts/intake_batch.py onto a dedicated archive disk; this script is for
    when that disk is smaller than the delivery — the most valuable studies go
    first, completely, the bulky rest as a spaced sample, until a free-space
    floor is reached.

  OUTPUT (<dest>/)
    images/<study_folder>/…     the copied files, same relative paths as on the drive
    ARCHIVE_STUDIES.csv         one row per archived study: rule, mode, files on the
                                drive, files copied, bytes, status (appended; the
                                LAST row of a study is its state)
    ARCHIVE_FILES.sha256        "<sha256>  <bytes>  <relative path>" per copied file
    README.txt                  what this archive is and is not

  USAGE (from sentinel/; PY=/Users/shakhzodbtr/venv/bin/python)
    $PY scripts/archive_studies.py --root /Volumes/Transcend/new1 --index data/new1_index.csv \\
        --dest ~/SIAA_DATA/raw/shox/new1 \\
        --rule "MR:BRAIN" --rule "CT:HEAD:cap=40" --rule "MR:*" --rule "CT:*:cap=12" \\
        [--min-free-gb 120] [--limit N] [--dry-run]

  RESUME
    Run the same command again. A study whose last row says "ok" with the same or
    a fuller mode is skipped; a failed or interrupted study is copied again. A
    study archived with a cap is copied again in full when a later run gives it
    a raw rule (or a larger cap).

  EXIT CODES
    0 every selected study is archived and verified · 1 usage / missing input ·
    2 stopped early (free-space floor reached, or studies failed to read) ·
    3 a copy did not verify (the study is marked failed; run again)

  PHI: the archive holds original DICOM (names in the headers). Keep --dest
  outside the git repository and off cloud-synced folders. Prints counts only.
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

STUDY_COLUMNS = ('study_folder', 'modality', 'body_part', 'rule', 'mode', 'files_on_drive',
                 'files_copied', 'bytes', 'status', 'finished')
STUDIES_CSV = 'ARCHIVE_STUDIES.csv'
FILES_SHA = 'ARCHIVE_FILES.sha256'
README = 'README.txt'
IMAGES_DIR = 'images'
CHUNK = 4 * 1024 * 1024
RAW = 'raw'

EXIT_OK, EXIT_USAGE, EXIT_STOPPED, EXIT_VERIFY = 0, 1, 2, 3


class VerifyError(Exception):
    """The archived copy does not hash like the bytes read from the drive."""


@dataclass(frozen=True)
class Rule:
    text: str
    modality: str            # upper-case, '*' allowed
    body: str                # upper-case fnmatch pattern
    cap: Optional[int]       # None = raw

    @property
    def mode(self) -> str:
        return RAW if self.cap is None else f'cap={self.cap}'

    def matches(self, modality: str, body_part: str) -> bool:
        return (fnmatch.fnmatchcase((modality or '').strip().upper(), self.modality)
                and fnmatch.fnmatchcase((body_part or '').strip().upper(), self.body))


def parse_rule(text: str) -> Rule:
    parts = [p.strip() for p in text.split(':')]
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise ValueError(f'rule {text!r}: expected "<modality>:<body part pattern>[:cap=N]"')
    cap = None
    if len(parts) == 3:
        if not parts[2].lower().startswith('cap='):
            raise ValueError(f'rule {text!r}: the third part must be cap=N')
        try:
            cap = int(parts[2][4:])
        except ValueError:
            raise ValueError(f'rule {text!r}: cap must be a whole number') from None
        if cap < 1:
            raise ValueError(f'rule {text!r}: cap must be at least 1')
    return Rule(text, parts[0].upper(), parts[1].upper(), cap)


def mode_rank(mode: str) -> float:
    """raw outranks every cap; a larger cap outranks a smaller one."""
    if mode == RAW:
        return float('inf')
    try:
        return float(mode.split('=', 1)[1])
    except (IndexError, ValueError):
        return -1.0


def spaced_indices(n: int, cap: int) -> list[int]:
    """Up to `cap` indices of range(n), evenly spaced, first and last included."""
    if n <= cap:
        return list(range(n))
    if cap == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})


def select_files(study_dir: Path, cap: Optional[int]) -> tuple[list[Path], int]:
    """(files to copy, number of files on the drive). raw: every regular file. cap: per directory,
    a spaced sample of its .dcm files (sorted by name); other files are left out."""
    chosen: list[Path] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(study_dir):
        dirnames.sort()
        names = sorted(f for f in filenames if not f.startswith('._'))
        total += len(names)
        d = Path(dirpath)
        if cap is None:
            chosen.extend(d / f for f in names)
        else:
            dcm = [f for f in names if f.lower().endswith('.dcm')]
            chosen.extend(d / dcm[i] for i in spaced_indices(len(dcm), cap))
    return chosen, total


def copy_verified(src: Path, dst: Path) -> tuple[str, int]:
    """Copy src → dst (through dst.part) and verify: (sha256, bytes). Raises VerifyError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    h = hashlib.sha256()
    n = 0
    with open(src, 'rb') as fi, open(part, 'wb') as fo:
        while True:
            chunk = fi.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            fo.write(chunk)
            n += len(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    h2 = hashlib.sha256()
    with open(part, 'rb') as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b''):
            h2.update(chunk)
    if h2.hexdigest() != h.hexdigest() or part.stat().st_size != n:
        part.unlink(missing_ok=True)
        raise VerifyError(f'{dst.name}: the archived copy differs from the bytes read from the drive')
    os.replace(part, dst)
    return h.hexdigest(), n


def read_state(dest: Path) -> dict[str, dict]:
    """study_folder → its LAST row in ARCHIVE_STUDIES.csv."""
    p = dest / STUDIES_CSV
    state: dict[str, dict] = {}
    if p.is_file():
        with open(p, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                state[r['study_folder']] = r
    return state


def append_row(dest: Path, row: dict) -> None:
    p = dest / STUDIES_CSV
    new = not p.is_file()
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(STUDY_COLUMNS))
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def write_readme(dest: Path, root: Path, rules: list[Rule]) -> None:
    p = dest / README
    if p.is_file():
        return
    p.write_text(
        'SENTINEL — partial archive of a clinic delivery (scripts/archive_studies.py)\n\n'
        f'source drive folder : {root}\n'
        'rules (first match wins): ' + ', '.join(r.text for r in rules) + '\n\n'
        'images/<study_folder>/ holds ORIGINAL DICOM files with their original relative paths.\n'
        'Studies archived in mode "raw" are complete. Studies archived in mode "cap=N" hold at most N\n'
        'evenly spaced files per series directory: enough to train slice classifiers, NOT the full scan.\n'
        f'{STUDIES_CSV}: one row per archived study (the last row of a study is its state).\n'
        f'{FILES_SHA}: sha256, size and path of every archived file.\n\n'
        'This folder contains patient data (names in the DICOM headers). Do not copy it into the\n'
        'git repository or a cloud-synced folder.\n', encoding='utf-8')


def plan_studies(index_rows: list[dict], rules: list[Rule], state: dict[str, dict]
                 ) -> tuple[list[tuple[dict, Rule]], dict]:
    """Studies to copy, in rule order then index order, and counts for the summary."""
    todo: list[tuple[dict, Rule]] = []
    counts = {'no_rule': 0, 'index_error': 0, 'done': 0}
    per_rule: dict[str, list[dict]] = {r.text: [] for r in rules}
    for row in index_rows:
        if (row.get('error') or '').strip():
            counts['index_error'] += 1
            continue
        rule = next((r for r in rules if r.matches(row.get('modality', ''), row.get('body_part', ''))), None)
        if rule is None:
            counts['no_rule'] += 1
            continue
        prev = state.get(row['study_folder'])
        if prev and prev.get('status') == 'ok' and mode_rank(prev.get('mode', '')) >= mode_rank(rule.mode):
            counts['done'] += 1
            continue
        per_rule[rule.text].append(row)
    for r in rules:
        todo.extend((row, r) for row in per_rule[r.text])
    counts['per_rule'] = {k: len(v) for k, v in per_rule.items()}
    return todo, counts


def archive_study(root: Path, dest: Path, row: dict, rule: Rule, sha_fh) -> dict:
    """Copy one study. Returns its ARCHIVE_STUDIES row (status ok / failed: …). Raises VerifyError."""
    folder = row['study_folder']
    src_dir = root / folder
    out = {'study_folder': folder, 'modality': row.get('modality', ''), 'body_part': row.get('body_part', ''),
           'rule': rule.text, 'mode': rule.mode, 'files_on_drive': 0, 'files_copied': 0, 'bytes': 0,
           'status': '', 'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if not src_dir.is_dir():
        out['status'] = 'failed: not on the drive'
        return out
    files, total = select_files(src_dir, rule.cap)
    out['files_on_drive'] = total
    if not files:
        out['status'] = 'failed: no files selected'
        return out
    dst_dir = dest / IMAGES_DIR / folder
    lines = []
    for f in files:
        rel = f.relative_to(src_dir)
        try:
            digest, n = copy_verified(f, dst_dir / rel)
        except OSError as e:
            out['status'] = f'failed: {type(e).__name__}'
            return out
        out['files_copied'] += 1
        out['bytes'] += n
        lines.append(f'{digest}  {n}  {IMAGES_DIR}/{folder}/{rel.as_posix()}\n')
    sha_fh.writelines(lines)
    sha_fh.flush()
    os.fsync(sha_fh.fileno())
    out['status'] = 'ok'
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Archive a prioritised part of a batch off the clinic drive.')
    ap.add_argument('--root', required=True, help='drive folder with one sub-folder per study')
    ap.add_argument('--index', required=True, help='index CSV of the batch (scripts/index_new_studies.py)')
    ap.add_argument('--dest', required=True, help='archive folder (created if missing); NOT inside the repository')
    ap.add_argument('--rule', action='append', required=True, metavar='MOD:BODY[:cap=N]',
                    help='priority rule, repeatable; the first rule that matches a study decides its mode')
    ap.add_argument('--min-free-gb', type=float, default=120.0,
                    help='stop before the destination volume has less free space than this (default 120)')
    ap.add_argument('--limit', type=int, default=0, help='archive at most N studies in this run (0 = all)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, copy nothing')
    a = ap.parse_args(argv)

    try:
        rules = [parse_rule(t) for t in a.rule]
    except ValueError as e:
        print(f'{R}✗ {e}{NC}')
        return EXIT_USAGE
    root, dest, index = Path(a.root).expanduser(), Path(a.dest).expanduser(), Path(a.index).expanduser()
    if not index.is_file():
        print(f'{R}✗ index not found: {index}{NC}')
        return EXIT_USAGE
    if not root.is_dir():
        print(f'{R}✗ drive folder not found (drive not mounted?): {root}{NC}')
        return EXIT_USAGE
    repo = Path(__file__).resolve().parent.parent
    if repo == dest.resolve() or repo in dest.resolve().parents:
        print(f'{R}✗ --dest is inside the repository ({repo}): the archive holds patient data{NC}')
        return EXIT_USAGE
    with open(index, encoding='utf-8') as fh:
        index_rows = list(csv.DictReader(fh))

    state = read_state(dest)
    todo, counts = plan_studies(index_rows, rules, state)
    if a.limit > 0:
        todo = todo[:a.limit]
    print(f'{B}ARCHIVE PLAN{NC}  {len(index_rows)} studies in the index → {dest}')
    for r in rules:
        print(f'  {r.text:24s} mode {r.mode:8s} to copy: {counts["per_rule"][r.text]}')
    print(f'  already archived: {counts["done"]}   no rule: {counts["no_rule"]}   index errors: {counts["index_error"]}')
    if a.dry_run or not todo:
        print('  (dry run — nothing copied)' if a.dry_run else f'{G}✓ nothing left to copy{NC}')
        return EXIT_OK

    dest.mkdir(parents=True, exist_ok=True)
    write_readme(dest, root, rules)
    floor = a.min_free_gb * 1e9
    n_ok = n_failed = 0
    copied_bytes = 0
    verify_failed = False
    stopped = ''
    t0 = time.time()
    with open(dest / FILES_SHA, 'a', encoding='utf-8') as sha_fh:
        for i, (row, rule) in enumerate(todo, 1):
            free = shutil.disk_usage(dest).free
            if free < floor:
                stopped = (f'free space on the archive volume is {free / 1e9:.0f} GB, below the floor of '
                           f'{a.min_free_gb:g} GB')
                break
            if not root.is_dir():
                stopped = 'the drive is no longer mounted'
                break
            try:
                out = archive_study(root, dest, row, rule, sha_fh)
            except VerifyError:
                verify_failed = True
                out = {'study_folder': row['study_folder'], 'modality': row.get('modality', ''),
                       'body_part': row.get('body_part', ''), 'rule': rule.text, 'mode': rule.mode,
                       'files_on_drive': '', 'files_copied': '', 'bytes': '', 'status': 'failed: verification',
                       'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
            append_row(dest, out)
            if out['status'] == 'ok':
                n_ok += 1
                copied_bytes += int(out['bytes'])
            else:
                n_failed += 1
            if i % 25 == 0 or i == len(todo):
                dt = max(time.time() - t0, 1e-6)
                rate = copied_bytes / dt
                print(f'  …{i}/{len(todo)}  ok {n_ok}  failed {n_failed}  {copied_bytes / 1e9:.1f} GB  '
                      f'{rate / 1e6:.0f} MB/s  free {shutil.disk_usage(dest).free / 1e9:.0f} GB', flush=True)

    left = len(todo) - n_ok - n_failed
    print(f'\n{B}ARCHIVED{NC} {n_ok} studies, {copied_bytes / 1e9:.1f} GB   failed {n_failed}   not reached {left}')
    if stopped:
        print(f'{Y}stopped: {stopped}. Free space (or remount the drive) and run the same command again.{NC}')
    if verify_failed:
        print(f'{R}✗ at least one copy did not verify — run the same command again{NC}')
        return EXIT_VERIFY
    if stopped or n_failed:
        return EXIT_STOPPED
    print(f'{G}✓ every selected study is archived and verified{NC}')
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
