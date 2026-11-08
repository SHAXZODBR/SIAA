#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — INTAKE A NEW HOSPITAL BATCH FROM A DRIVE INTO THE DATA ARCHIVE
================================================================================
  WHAT
    "When a drive arrives" step 1 (docs/release_runbook_en.md §2a). Copies a
    hospital delivery (DICOM study folders + radiology reports) from the drive
    into the archive layout

        <dest>/raw/<clinic>/<batch>/images/      ← everything under --src
        <dest>/raw/<clinic>/<batch>/reports/     ← --reports (a folder or one file)
        <dest>/raw/<clinic>/<batch>/CHECKSUMS.sha256   sha256 of every copied file
        <dest>/raw/<clinic>/<batch>/CHECKSUMS.sizes    byte size of every copied file
        <dest>/raw/<clinic>/<batch>/README.txt         received date, source, counts, date range
        <dest>/raw/<clinic>/<batch>/.intake/           journal, header-scan sample, logs
        <dest>/raw/INVENTORY.csv                       one row per archived batch

    then prints the exact `release_pipeline.py run` command for the batch.

    Both manifests use the `shasum` line format ("<value>  <relative path>",
    paths relative to the batch dir), so a third party can check a copy with
    `cd <batch dir> && shasum -a 256 -c CHECKSUMS.sha256`.

  HOW (and why it is safe)
    * The source is only ever OPENED FOR READING — nothing is written, renamed or
      deleted on the drive. Hidden files / OS junk (.DS_Store, ._*, .Trashes,
      System Volume Information, $RECYCLE.BIN …) and symlinks are skipped.
    * Each file is streamed to "<name>.part" while its sha256 is computed, fsynced,
      renamed into place, and recorded in .intake/journal.tsv (size, sha256, path).
      An interrupted run leaves at most one .part file; --resume deletes stale
      .part files and skips every journaled file whose source size is unchanged
      and whose copy is present with that size.
    * Post-copy verification re-reads every file copied in this run FROM THE
      DESTINATION and compares size + sha256 with the source hash. A mismatch is
      listed in .intake/needs_recopy.txt (the next --resume copies it again) and the
      run exits 3. `--verify` re-hashes the WHOLE batch against CHECKSUMS.sha256.
    * The batch dir is never overwritten: an existing non-empty batch dir is
      refused unless --resume. --dest must already exist (an unmounted
      /Volumes/<drive> would otherwise be created on the internal disk).
    * The disk-space check runs before any byte is copied.
    * INVENTORY.csv gets the batch row only after a fully verified intake (a
      --resume re-run replaces the batch's row; other rows are kept).
    * PHI: stdout, README.txt and INVENTORY.csv carry counts and dates only —
      never a report file name (those are <patientID>_<Name>.pdf) or a header
      value other than modality / body part / study date. The header-scan sample
      CSV (.intake/header_scan.csv) is data, like data/<batch>_index.csv, and
      stays inside the batch dir.

  COUNTS / DATE RANGE
    When images/ is a DICOM tree (.dcm files, a DICOMDIR or 'DICM' preamble files),
    a quick header scan runs `scripts/index_new_studies.py --root <study root>
    --limit <--scan-limit, default 200>` on the COPIED images (one header per
    series, no pixels). The study root is the first directory, going down through
    single-subfolder levels, that holds several study folders. Its modality and
    StudyDate range go into README.txt / INVENTORY.csv (marked as a sample when
    the batch has more studies than the limit). A failed scan is a warning, not a
    failed intake.

  USAGE (from sentinel/; PY=/Users/shakhzodbtr/venv/bin/python)
    $PY scripts/intake_batch.py --src /Volumes/HOSPITAL/new2 --reports /Volumes/HOSPITAL/reports_new2 \\
        --clinic shox --batch new2 --dest /Volumes/SIAA_DATA
    $PY scripts/intake_batch.py ... --resume                    # continue / add reports later
    $PY scripts/intake_batch.py --clinic shox --batch new2 --dest /Volumes/SIAA_DATA --verify

  EXIT CODES
    0 ok · 1 usage / refused (nothing copied) · 2 some source files could not be
    read (the rest is copied + verified; re-run with --resume) · 3 verification failed
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = '/Users/shakhzodbtr/venv/bin/python'

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_READ_ERRORS = 2
EXIT_VERIFY_FAILED = 3

NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}')
CHUNK = 1 << 20
SCAN_LIMIT_DEFAULT = 200
SPACE_MARGIN = 1.02                       # 2 % head-room on the destination volume
JUNK_NAMES = {'.DS_Store', 'Thumbs.db', 'desktop.ini', '.Spotlight-V100', '.Trashes', '.fseventsd',
              '.TemporaryItems', '.DocumentRevisions-V100', 'System Volume Information', '$RECYCLE.BIN'}
REPORT_SUFFIXES = ('.pdf', '.docx', '.xlsx', '.csv')
INVENTORY_FIELDS = ['received_date', 'clinic', 'batch', 'source', 'reports_source', 'n_image_files',
                    'n_report_files', 'total_bytes', 'n_study_folders', 'n_studies_scanned', 'modalities',
                    'study_date_min', 'study_date_max', 'date_range_is_sample', 'checksums_sha256',
                    'intake_finished_at', 'operator']


class IntakeError(Exception):
    """A refusal before anything is copied (exit 1)."""


# ------------------------------------------------------------------ model ----

@dataclass
class Entry:
    """One source file to copy: absolute source path, path relative to the batch dir, size."""
    src: Path
    rel: str
    size: int


@dataclass
class Batch:
    dest_root: Path
    clinic: str
    batch: str

    @property
    def dir(self) -> Path: return self.dest_root / 'raw' / self.clinic / self.batch
    @property
    def images(self) -> Path: return self.dir / 'images'
    @property
    def reports(self) -> Path: return self.dir / 'reports'
    @property
    def meta(self) -> Path: return self.dir / '.intake'
    @property
    def journal(self) -> Path: return self.meta / 'journal.tsv'
    @property
    def needs_recopy(self) -> Path: return self.meta / 'needs_recopy.txt'
    @property
    def checksums(self) -> Path: return self.dir / 'CHECKSUMS.sha256'
    @property
    def sizes(self) -> Path: return self.dir / 'CHECKSUMS.sizes'
    @property
    def readme(self) -> Path: return self.dir / 'README.txt'
    @property
    def inventory(self) -> Path: return self.dest_root / 'raw' / 'INVENTORY.csv'
    @property
    def log(self) -> Path: return self.meta / 'intake.log'


@dataclass
class RunResult:
    copied: list[str] = field(default_factory=list)
    skipped: int = 0
    read_errors: list[tuple[str, str]] = field(default_factory=list)
    bad: list[tuple[str, str]] = field(default_factory=list)          # (rel, why) verification failures
    junk_skipped: int = 0
    symlinks_skipped: int = 0


# ---------------------------------------------------------------- helpers ----

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def is_junk(name: str) -> bool:
    return name in JUNK_NAMES or name.startswith('._') or name.startswith('.')


def _hashed_name(filename: str) -> str:
    """parse_hospital_reports.hashed_source_name: '<pid>_<Name>.pdf' → '<pid>_<sha16>.pdf'."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from parse_hospital_reports import hashed_source_name          # stdlib-only module
    return hashed_source_name(filename)


def display_rel(rel: str) -> str:
    """A batch-relative path for messages: report file names (<patientID>_<Name>.pdf) are hashed,
    so a patient name is never printed."""
    p = Path(rel)
    if p.parts and p.parts[0] == 'reports' and p.suffix.lower() in REPORT_SUFFIXES:
        return str(p.parent / _hashed_name(p.name))
    return rel


def display_source(p: Optional[Path]) -> str:
    """A source path for README / INVENTORY / stdout: a single report FILE is shown with its name
    hashed (it may be <patientID>_<Name>.pdf); folders as they are."""
    if p is None:
        return ''
    if p.suffix.lower() in REPORT_SUFFIXES and not p.is_dir():
        return str(p.parent / _hashed_name(p.name))
    return str(p)


def _inside(a: Path, b: Path) -> bool:
    """True when a is b or lies under b (resolved)."""
    try:
        a.resolve().relative_to(b.resolve())
        return True
    except ValueError:
        return False


def walk_source(src: Path, prefix: str, res: RunResult) -> list[Entry]:
    """Every regular file under src (or src itself when it is a file), relative path prefixed with
    `prefix` ('images' / 'reports'). Hidden/junk entries and symlinks are skipped and counted."""
    entries: list[Entry] = []
    if src.is_file():
        if src.is_symlink():
            res.symlinks_skipped += 1
            return entries
        return [Entry(src, f'{prefix}/{src.name}', src.stat().st_size)]
    for dirpath, dirs, files in os.walk(src, followlinks=False):
        keep = []
        for d in dirs:
            full = Path(dirpath) / d
            if is_junk(d):
                res.junk_skipped += 1
            elif full.is_symlink():
                res.symlinks_skipped += 1
            else:
                keep.append(d)
        dirs[:] = sorted(keep)
        for fn in sorted(files):
            full = Path(dirpath) / fn
            if is_junk(fn):
                res.junk_skipped += 1
                continue
            if full.is_symlink():
                res.symlinks_skipped += 1
                continue
            rel = full.relative_to(src).as_posix()
            if '\n' in rel or '\r' in rel or '\t' in rel:
                res.read_errors.append((f'{prefix}/<name with control characters>', 'unsupported file name'))
                continue
            try:
                size = full.stat().st_size
            except OSError as e:
                res.read_errors.append((f'{prefix}/{rel}', f'{type(e).__name__}: {e.strerror or e}'))
                continue
            entries.append(Entry(full, f'{prefix}/{rel}', size))
    return entries


# ---------------------------------------------------------------- journal ----

def read_journal(path: Path) -> dict[str, tuple[int, str]]:
    """{rel: (size, sha256)} — the last line for a path wins; torn last lines are ignored."""
    out: dict[str, tuple[int, str]] = {}
    if not path.is_file():
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 2)
            if len(parts) != 3 or not parts[0].isdigit() or not re.fullmatch(r'[0-9a-f]{64}', parts[1]):
                continue
            out[parts[2]] = (int(parts[0]), parts[1])
    return out


def append_journal(path: Path, rel: str, size: int, digest: str) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{size}\t{digest}\t{rel}\n')
        f.flush()
        os.fsync(f.fileno())


def rewrite_journal(path: Path, journal: dict[str, tuple[int, str]]) -> None:
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        for rel in sorted(journal):
            size, digest = journal[rel]
            f.write(f'{size}\t{digest}\t{rel}\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_lines(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {ln.rstrip('\n') for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()}


def write_manifests(b: Batch, journal: dict[str, tuple[int, str]]) -> None:
    """CHECKSUMS.sha256 + CHECKSUMS.sizes in `shasum` line format, sorted by path."""
    for path, idx in ((b.checksums, 1), (b.sizes, 0)):
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            for rel in sorted(journal):
                f.write(f'{journal[rel][idx]}  {rel}\n')
        tmp.replace(path)


def read_manifests(b: Batch) -> dict[str, tuple[int, str]]:
    """{rel: (size, sha256)} from CHECKSUMS.sha256 (+ CHECKSUMS.sizes; size -1 when absent)."""
    sums: dict[str, str] = {}
    for line in b.checksums.read_text(encoding='utf-8').splitlines():
        m = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        if m:
            sums[m.group(2)] = m.group(1)
    sizes: dict[str, int] = {}
    if b.sizes.is_file():
        for line in b.sizes.read_text(encoding='utf-8').splitlines():
            m = re.fullmatch(r'(\d+)  (.+)', line)
            if m:
                sizes[m.group(2)] = int(m.group(1))
    return {rel: (sizes.get(rel, -1), digest) for rel, digest in sums.items()}


# ------------------------------------------------------------------- copy ----

def copy_one(src: Path, dst: Path) -> tuple[int, str]:
    """Stream src → dst.part (hashing the bytes read from the source), fsync, rename into place,
    keep the source mtime. Returns (bytes, sha256 of the source bytes). The source is opened
    read-only and never modified."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    h = hashlib.sha256()
    n = 0
    with open(src, 'rb') as fin, open(part, 'wb') as fout:
        for chunk in iter(lambda: fin.read(CHUNK), b''):
            h.update(chunk)
            fout.write(chunk)
            n += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    st = src.stat()
    os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns))
    part.replace(dst)
    return n, h.hexdigest()


def remove_stale_parts(b: Batch) -> int:
    n = 0
    for root in (b.images, b.reports):
        if root.is_dir():
            for p in root.rglob('*.part'):
                p.unlink()
                n += 1
    return n


def verify_files(b: Batch, expected: dict[str, tuple[int, str]], rels: Iterable[str]) -> list[tuple[str, str]]:
    """Re-read each copy from the destination; (rel, why) for every size / sha256 mismatch."""
    bad: list[tuple[str, str]] = []
    for rel in sorted(rels):
        size, digest = expected[rel]
        fp = b.dir / rel
        if not fp.is_file():
            bad.append((rel, 'missing'))
            continue
        actual = fp.stat().st_size
        if size >= 0 and actual != size:
            bad.append((rel, f'size {actual} ≠ {size}'))
            continue
        if sha256_file(fp) != digest:
            bad.append((rel, 'sha256 mismatch'))
    return bad


# ------------------------------------------------------------ header scan ----

def _is_dicom_file(p: Path) -> bool:
    if p.suffix.lower() == '.dcm' or p.name.upper() == 'DICOMDIR':
        return True
    try:
        with open(p, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except OSError:
        return False


def is_dicom_tree(root: Path, probe: int = 400) -> bool:
    """True when one of the first `probe` files under root looks like DICOM."""
    if not root.is_dir():
        return False
    n = 0
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for fn in sorted(files):
            if _is_dicom_file(Path(dirpath) / fn):
                return True
            n += 1
            if n >= probe:
                return False
    return False


def _study_dirs(d: Path) -> list[Path]:
    return [x for x in d.iterdir() if x.is_dir() and not x.name.startswith(('.', '$', 'System'))]


def study_root(images: Path, max_depth: int = 4) -> Path:
    """The dir whose immediate subfolders are the studies (index_new_studies.py --root): go down
    while a level holds exactly one folder and no DICOM files of its own."""
    cur = images
    for _ in range(max_depth):
        subs = _study_dirs(cur)
        own_files = [f for f in cur.iterdir() if f.is_file() and not is_junk(f.name) and f.name.upper() != 'DICOMDIR']
        if len(subs) == 1 and not own_files:
            cur = subs[0]
            continue
        break
    return cur


def header_scan(b: Batch, python: str, limit: int) -> dict:
    """Run index_new_studies.py --limit on the copied images; summary for README / INVENTORY."""
    info: dict = {'dicom': False, 'study_root': None, 'n_study_folders': 0, 'n_scanned': 0,
                  'modalities': {}, 'date_min': '', 'date_max': '', 'sample': False, 'error': ''}
    if not is_dicom_tree(b.images):
        info['error'] = 'not a DICOM tree — no header scan'
        return info
    info['dicom'] = True
    root = study_root(b.images)
    info['study_root'] = root
    info['n_study_folders'] = len(_study_dirs(root))
    out = b.meta / 'header_scan.csv'
    if out.exists():
        out.unlink()                                  # index_new_studies.py appends; rescan from scratch
    argv = [python, str(REPO / 'scripts' / 'index_new_studies.py'), '--root', str(root), '--out', str(out),
            '--limit', str(limit)]
    log = b.meta / 'header_scan.log'
    try:
        with open(log, 'w', encoding='utf-8') as lf:
            rc = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO), timeout=3600).returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        info['error'] = f'header scan could not run: {type(e).__name__}'
        return info
    if rc != 0 or not out.is_file():
        info['error'] = f'header scan failed (exit {rc}) — see {log.relative_to(b.dir)}'
        return info
    mods: Counter = Counter()
    dates: list[str] = []
    with open(out, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            info['n_scanned'] += 1
            mods[(row.get('modality') or '?').strip() or '?'] += 1
            d = (row.get('study_date') or '').strip()
            if re.fullmatch(r'\d{8}', d):
                dates.append(d)
    info['modalities'] = dict(mods.most_common())
    if dates:
        fmt = lambda d: f'{d[:4]}-{d[4:6]}-{d[6:]}'
        info['date_min'], info['date_max'] = fmt(min(dates)), fmt(max(dates))
    info['sample'] = info['n_study_folders'] > info['n_scanned']
    return info


# ---------------------------------------------------- README / INVENTORY ----

def batch_counts(journal: dict[str, tuple[int, str]]) -> dict:
    n_img = sum(1 for r in journal if r.startswith('images/'))
    n_rep = sum(1 for r in journal if r.startswith('reports/'))
    rep_types = Counter(Path(r).suffix.lower() or '(none)' for r in journal if r.startswith('reports/'))
    return {'n_image_files': n_img, 'n_report_files': n_rep, 'total_bytes': sum(s for s, _ in journal.values()),
            'report_types': dict(rep_types)}


def write_readme(b: Batch, *, received: str, source: str, reports_source: str, counts: dict, scan: dict,
                 verified: bool, problems: list[str]) -> None:
    rel_root = scan['study_root'].relative_to(b.dir).as_posix() if scan.get('study_root') else '—'
    mods = ', '.join(f'{k} {v}' for k, v in scan['modalities'].items()) or '—'
    sample_note = (f' (sample: first {scan["n_scanned"]} of {scan["n_study_folders"]} study folders)'
                   if scan.get('sample') else '')
    lines = [
        f'SIAA / Sentinel data archive — batch {b.clinic}/{b.batch}',
        '=' * 64,
        f'received        : {received}',
        f'intake finished : {now_iso()}  (operator {getpass.getuser()})',
        f'source (images) : {source or "—"}',
        f'source (reports): {reports_source or "— (not received yet: add with intake_batch.py --resume --reports …)"}',
        '',
        f'image files     : {counts["n_image_files"]}',
        f'report files    : {counts["n_report_files"]}'
        + (f'  ({", ".join(f"{k} {v}" for k, v in sorted(counts["report_types"].items()))})'
           if counts['report_types'] else ''),
        f'total size      : {counts["total_bytes"]:,} bytes ({counts["total_bytes"] / 1e9:.2f} GB)',
        f'study folders   : {scan.get("n_study_folders") or "—"}  (study root: {rel_root})',
        f'header scan     : {scan["n_scanned"]} studies scanned{sample_note}'
        + (f' — {scan["error"]}' if scan.get('error') else ''),
        f'modalities      : {mods}',
        f'study dates     : {scan["date_min"] or "—"} … {scan["date_max"] or "—"}{" (sample)" if scan.get("sample") else ""}',
        '',
        f'verification    : {"PASSED — every copied file re-read from this disk matches the source sha256" if verified else "FAILED / INCOMPLETE — see below"}',
        *(f'  ! {p}' for p in problems),
        '',
        'Integrity: CHECKSUMS.sha256 (sha256) and CHECKSUMS.sizes (bytes) list every file under images/',
        'and reports/. Check a copy with:  cd <this folder> && shasum -a 256 -c CHECKSUMS.sha256',
        'or: scripts/intake_batch.py --clinic <clinic> --batch <batch> --dest <root> --verify',
        'Patient data: this folder holds identifiable medical data. Keep it on the encrypted archive',
        'drive and its mirror (scripts/backup_mirror.sh) only — never on a laptop, cloud or git.',
    ]
    b.readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def update_inventory(b: Batch, row: dict) -> None:
    """Append the batch row to raw/INVENTORY.csv (a re-run replaces this batch's row; other rows
    and unknown columns are kept). Written via a temp file + rename."""
    inv = b.inventory
    inv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fields = list(INVENTORY_FIELDS)
    if inv.is_file():
        with open(inv, newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for extra in rd.fieldnames or []:
                if extra not in fields:
                    fields.append(extra)
            rows = [r for r in rd if not (r.get('clinic') == b.clinic and r.get('batch') == b.batch)]
    rows.append(row)
    tmp = inv.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(inv)


def pipeline_command(b: Batch, scan: dict, python: str, has_reports: bool) -> list[str]:
    """The release_pipeline.py commands for this batch (plan first, then run)."""
    root = scan.get('study_root') or b.images
    common = ['--batch', b.batch, '--root', str(root), '--reports', str(b.reports)]
    py = shlex.quote(python)
    lines = [f'cd {shlex.quote(str(REPO))}',
             f'{py} scripts/release_pipeline.py plan {shlex.join(common)}',
             f'{py} scripts/release_pipeline.py run {shlex.join(common)}']
    if not has_reports:
        lines.insert(0, '# reports/ is empty: the parse stage needs them — first run intake_batch.py '
                        '--resume --reports <folder-or-xlsx> for this batch')
    return lines


# ------------------------------------------------------------------- main ----

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description='Copy a hospital delivery into <dest>/raw/<clinic>/<batch>/ with sha256 verification.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Exit codes: 0 ok · 1 refused · 2 unreadable source files · 3 verification failed.')
    ap.add_argument('--src', help='drive or folder with the DICOM study folders (copied to images/)')
    ap.add_argument('--reports', help='folder (or one .xlsx/.csv/.docx/.pdf) with the radiology reports '
                                      '(copied to reports/)')
    ap.add_argument('--clinic', required=True, help='clinic short name, e.g. shox')
    ap.add_argument('--batch', required=True, help='batch name, e.g. new2 (= release_pipeline.py --batch)')
    ap.add_argument('--dest', required=True, help='archive root (SIAA_DATA); must already exist')
    ap.add_argument('--received', default=None, help='date the drive was received (YYYY-MM-DD, default today)')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted intake / add reports to an existing batch')
    ap.add_argument('--verify', action='store_true',
                    help='re-hash the whole existing batch against CHECKSUMS.sha256 (copies nothing)')
    ap.add_argument('--scan-limit', type=int, default=SCAN_LIMIT_DEFAULT,
                    help=f'studies read by the header scan (index_new_studies.py --limit, default {SCAN_LIMIT_DEFAULT})')
    ap.add_argument('--no-scan', action='store_true', help='skip the DICOM header scan')
    ap.add_argument('--python', default=sys.executable, help='interpreter for the header scan / printed command')
    return ap


def _validate(a: argparse.Namespace) -> Batch:
    for name, v in (('--clinic', a.clinic), ('--batch', a.batch)):
        if not NAME_RE.fullmatch(v or '') or '..' in v:
            raise IntakeError(f'{name} {v!r}: use letters, digits, ".", "_" or "-" (max 64, no path separators)')
    dest = Path(a.dest).expanduser()
    if not dest.is_dir():
        raise IntakeError(f'--dest {dest} does not exist (is the archive drive mounted?) — it is never created '
                          'automatically')
    b = Batch(dest.resolve(), a.clinic, a.batch)
    if a.received:
        try:
            date.fromisoformat(a.received)
        except ValueError:
            raise IntakeError(f'--received {a.received!r} is not YYYY-MM-DD')
    if a.verify:
        return b
    if not a.src and not a.reports:
        raise IntakeError('--src (images) and/or --reports is required')
    for flag, p in (('--src', a.src), ('--reports', a.reports)):
        if p is None:
            continue
        sp = Path(p).expanduser()
        if not sp.exists():
            raise IntakeError(f'{flag} {display_source(sp)} does not exist (is the hospital drive mounted?)')
        if flag == '--src' and not sp.is_dir():
            raise IntakeError('--src must be a folder (the drive or the folder holding the study folders)')
        if _inside(b.dir, sp) or (sp.is_dir() and _inside(sp, b.dir)):
            raise IntakeError(f'{flag} {display_source(sp)} and the batch dir {b.dir} overlap — copy to another place')
    if b.dir.exists() and any(b.dir.iterdir()) and not a.resume:
        raise IntakeError(f'{b.dir} already exists and is not empty — refusing to overwrite a batch. Use --resume '
                          'to continue it / add reports, --verify to check it, or choose another --batch')
    return b


def cmd_verify(b: Batch) -> int:
    if not b.checksums.is_file():
        print(f'{R}✗ {b.checksums} not found — nothing to verify{NC}')
        return EXIT_USAGE
    expected = read_manifests(b)
    print(f'{B}verifying {len(expected)} files of {b.clinic}/{b.batch} against CHECKSUMS.sha256 …{NC}')
    bad = verify_files(b, expected, expected)
    listed = set(expected)
    strays = [p.relative_to(b.dir).as_posix() for root in (b.images, b.reports) if root.is_dir()
              for p in root.rglob('*') if p.is_file() and not is_junk(p.name)
              and p.relative_to(b.dir).as_posix() not in listed]
    if strays:
        print(f'{Y}⚠ {len(strays)} files are not in CHECKSUMS.sha256 (added after the intake?), e.g. '
              f'{", ".join(display_rel(s) for s in strays[:3])}{NC}')
    if bad:
        b.meta.mkdir(parents=True, exist_ok=True)
        with open(b.needs_recopy, 'a', encoding='utf-8') as f:
            for rel, _why in bad:
                f.write(rel + '\n')
        for rel, why in bad[:20]:
            print(f'{R}  ✗ {display_rel(rel)}: {why}{NC}')
        print(f'{R}✗ VERIFICATION FAILED: {len(bad)} of {len(expected)} files differ from the source checksums{NC}')
        print(f'  listed in {b.needs_recopy} — reconnect the hospital drive and re-run the intake with --resume '
              '(the same --src/--reports) to copy them again')
        return EXIT_VERIFY_FAILED
    print(f'{G}✓ all {len(expected)} files match CHECKSUMS.sha256{NC}')
    return EXIT_OK


def run_intake(a: argparse.Namespace, b: Batch) -> int:
    res = RunResult()
    src = Path(a.src).expanduser().resolve() if a.src else None
    reports = Path(a.reports).expanduser().resolve() if a.reports else None
    entries: list[Entry] = []
    if src:
        entries += walk_source(src, 'images', res)
    if reports:
        entries += walk_source(reports, 'reports', res)
    if not entries and not res.read_errors:
        raise IntakeError('the source holds no files to copy')

    b.dir.mkdir(parents=True, exist_ok=True)
    b.images.mkdir(exist_ok=True)
    b.reports.mkdir(exist_ok=True)
    b.meta.mkdir(exist_ok=True)
    stale = remove_stale_parts(b)
    journal = read_journal(b.journal)
    recopy = read_lines(b.needs_recopy)

    todo: list[Entry] = []
    conflicts: list[str] = []
    for e in entries:
        j = journal.get(e.rel)
        dst = b.dir / e.rel
        if j and e.rel not in recopy:
            if j[0] != e.size:
                conflicts.append(e.rel)
                continue
            if dst.is_file() and dst.stat().st_size == e.size:
                res.skipped += 1
                continue
        todo.append(e)
    if conflicts:
        for rel in conflicts[:10]:
            print(f'{R}  ✗ {display_rel(rel)}: the source size differs from the copy made earlier{NC}')
        raise IntakeError(f'{len(conflicts)} source files changed since the first intake of this batch — a batch is '
                          'one delivery: archive the new delivery under a new --batch')

    need = sum(e.size for e in todo)
    free = shutil.disk_usage(b.dir).free
    if need * SPACE_MARGIN > free:
        raise IntakeError(f'not enough space on {b.dest_root}: need {need / 1e9:.1f} GB (+2 %), free {free / 1e9:.1f} GB')

    print(f'{B}intake {b.clinic}/{b.batch} → {b.dir}{NC}')
    print(f'  files: {len(entries)} in the source · {res.skipped} already copied · {len(todo)} to copy '
          f'({need / 1e9:.2f} GB)' + (f' · {stale} stale .part removed' if stale else ''))
    if res.junk_skipped or res.symlinks_skipped:
        print(f'  skipped: {res.junk_skipped} hidden/OS files, {res.symlinks_skipped} symlinks')

    expected: dict[str, tuple[int, str]] = {}
    done_bytes = 0
    for k, e in enumerate(todo, 1):
        try:
            n, digest = copy_one(e.src, b.dir / e.rel)
        except OSError as err:
            res.read_errors.append((e.rel, f'{type(err).__name__}: {err.strerror or err}'))
            continue
        if n != e.size:
            res.read_errors.append((e.rel, f'read {n} bytes, expected {e.size} (file changing on the drive?)'))
            continue
        append_journal(b.journal, e.rel, n, digest)
        journal[e.rel] = (n, digest)
        expected[e.rel] = (n, digest)
        res.copied.append(e.rel)
        done_bytes += n
        if k % 200 == 0 or k == len(todo):
            print(f'  copied {k}/{len(todo)}  ({done_bytes / 1e9:.2f} GB)', flush=True)

    # post-copy verification: every file copied in THIS run, re-read from the destination
    print(f'{B}verifying {len(res.copied)} copied files (re-read from {b.dest_root}) …{NC}')
    res.bad = verify_files(b, expected, res.copied)
    still_bad = {rel for rel, _ in res.bad}
    remaining = (recopy - set(res.copied)) | still_bad
    if remaining:
        b.needs_recopy.write_text(''.join(r + '\n' for r in sorted(remaining)), encoding='utf-8')
    elif b.needs_recopy.exists():
        b.needs_recopy.unlink()
    rewrite_journal(b.journal, journal)
    write_manifests(b, journal)

    counts = batch_counts(journal)
    scan = ({'dicom': False, 'study_root': None, 'n_study_folders': 0, 'n_scanned': 0, 'modalities': {},
             'date_min': '', 'date_max': '', 'sample': False, 'error': 'header scan skipped (--no-scan)'}
            if a.no_scan else header_scan(b, a.python, a.scan_limit))
    problems = [f'{display_rel(r)}: {w}' for r, w in res.bad[:20]] + \
               [f'unreadable: {display_rel(r)}: {w}' for r, w in res.read_errors[:20]]
    verified = not res.bad and not res.read_errors and not remaining
    received = a.received or date.today().isoformat()
    prev_sources = _readme_sources(b)
    source = display_source(src) if src else prev_sources[0]
    reports_source = display_source(reports) if reports else prev_sources[1]
    write_readme(b, received=received, source=source, reports_source=reports_source, counts=counts, scan=scan,
                 verified=verified, problems=problems)
    with open(b.log, 'a', encoding='utf-8') as lf:
        lf.write(f'{now_iso()}\tcopied={len(res.copied)}\tskipped={res.skipped}\tbad={len(res.bad)}\t'
                 f'read_errors={len(res.read_errors)}\tverified={verified}\n')

    if scan.get('error'):
        print(f'{Y}⚠ {scan["error"]}{NC}')
    elif scan.get('dicom'):
        print(f'  header scan: {scan["n_scanned"]} studies · {", ".join(f"{k} {v}" for k, v in scan["modalities"].items())}'
              f' · dates {scan["date_min"]} … {scan["date_max"]}{" (sample)" if scan["sample"] else ""}')
    print(f'  batch: {counts["n_image_files"]} image files, {counts["n_report_files"]} report files, '
          f'{counts["total_bytes"] / 1e9:.2f} GB · README.txt, CHECKSUMS.sha256, CHECKSUMS.sizes written')

    if res.bad:
        for rel, why in res.bad[:20]:
            print(f'{R}  ✗ {display_rel(rel)}: {why}{NC}')
        print(f'{R}✗ VERIFICATION FAILED: {len(res.bad)} copies differ from the source — listed in '
              f'{b.needs_recopy}; re-run with --resume to copy them again (check the archive drive){NC}')
        return EXIT_VERIFY_FAILED
    if res.read_errors or remaining:
        for rel, why in res.read_errors[:20]:
            print(f'{R}  ✗ {display_rel(rel)}: {why}{NC}')
        print(f'{R}✗ {len(res.read_errors)} source files could not be copied — everything else is copied and '
              f'verified; re-run the same command with --resume{NC}')
        return EXIT_READ_ERRORS

    checksums_sha = sha256_file(b.checksums)
    update_inventory(b, {
        'received_date': received, 'clinic': b.clinic, 'batch': b.batch, 'source': source,
        'reports_source': reports_source, 'n_image_files': counts['n_image_files'],
        'n_report_files': counts['n_report_files'], 'total_bytes': counts['total_bytes'],
        'n_study_folders': scan.get('n_study_folders') or '', 'n_studies_scanned': scan.get('n_scanned') or '',
        'modalities': ';'.join(f'{k}={v}' for k, v in scan['modalities'].items()),
        'study_date_min': scan['date_min'], 'study_date_max': scan['date_max'],
        'date_range_is_sample': 'yes' if scan.get('sample') else 'no', 'checksums_sha256': checksums_sha,
        'intake_finished_at': now_iso(), 'operator': getpass.getuser()})
    print(f'{G}✓ intake verified — {len(res.copied)} copied, {res.skipped} already present; '
          f'row written to {b.inventory}{NC}')
    print(f'\n{B}next (dry run first, then the release):{NC}')
    for line in pipeline_command(b, scan, a.python, counts['n_report_files'] > 0):
        print(f'  {line}')
    print('  # add --candidates warm,fresh to also train from the public ViT base (docs/release_runbook_en.md)')
    print(f'  # then mirror the archive: scripts/backup_mirror.sh {shlex.quote(str(b.dest_root))} <backup drive root>')
    return EXIT_OK


def _readme_sources(b: Batch) -> tuple[str, str]:
    """(images source, reports source) recorded by an earlier run of this batch ('' if none) — a
    --resume that only adds reports keeps the images source in README / INVENTORY."""
    src = rep = ''
    if b.readme.is_file():
        for line in b.readme.read_text(encoding='utf-8').splitlines():
            if line.startswith('source (images) :'):
                v = line.split(':', 1)[1].strip()
                src = '' if v == '—' else v
            elif line.startswith('source (reports):'):
                v = line.split(':', 1)[1].strip()
                rep = '' if v.startswith('—') else v
    return src, rep


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        b = _validate(a)
        if a.verify:
            return cmd_verify(b)
        return run_intake(a, b)
    except IntakeError as e:
        print(f'{R}✗ {e}{NC}')
        return EXIT_USAGE
    except KeyboardInterrupt:
        print(f'\n{R}✗ interrupted — re-run the same command with --resume (copied files are kept){NC}')
        return 130


if __name__ == '__main__':
    sys.exit(main())
