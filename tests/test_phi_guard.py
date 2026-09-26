"""Repo-wide PHI / secret guard — nothing that can be committed may carry a real
patient id, a letterhead name or a real-looking GitHub token.

The origin repo is PUBLIC; data/ and models/ are git-ignored, but tests/,
scripts/, docs/, packaging/, training/ and src/ are not — a real hospital row
pasted into a fixture would be published permanently in git history (it
happened once: tests/test_report_parsing.py had three real request-sheet rows).

Scanned: every text file git would commit (tracked + untracked-not-ignored)
under the directories below plus the repo root's own files; without git, a
walk of the same directories that skips build/dist/__pycache__/node_modules.

Checks
  1. digit tokens (6-8 digits, also when glued to '_' / letters as in
     '<pid>_<Name>.pdf', '<pid>_<batch>_<k>.png', 'pid_<id>') that are also a
     real patient_id in the local data files — EVERY batch's
     data/<batch>_{index,reports,labels,matched}.csv, the reference files,
     data/rehearsal/*.csv, data/sets/*/studies.csv and the holdout lists —
     skipped when none of those files is present (CI, fresh clone).
  2. "Ф.И.О." / "ФИО" followed by a literal Cyrillic name in code (not in
     docs: the runbooks legitimately describe the letterhead field).
  3. real-looking GitHub tokens (ghp_ + 36 alnum, github_pat_…).

A failure names file:line only — the matched id is masked, never printed.
Synthetic fixtures use ids >= 9000000 and dates in 2031.
"""

import csv
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("tests", "scripts", "docs", "packaging", "training", "src")
SKIP_DIR_NAMES = {"build", "dist", "__pycache__", "node_modules", ".git", ".pytest_cache",
                  "data", "models", "logs"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".js", ".ts", ".tsx", ".jsx", ".json", ".csv", ".tsv",
                 ".yml", ".yaml", ".toml", ".ini", ".cfg", ".sh", ".ps1", ".bat", ".spec",
                 ".html", ".css", ".xml", ".sql", ".j2", ".env", ".example", ""}
MAX_BYTES = 4 * 1024 * 1024
DOC_SUFFIXES = {".md", ".txt", ".html", ".rst"}

# local files that hold REAL patient ids (all under the git-ignored data/) — globs, so every
# new batch the release pipeline writes (data/<batch>_index.csv, …) is guarded automatically
ID_SOURCES = [
    ("data/*_index.csv", "patient_id"),
    ("data/*_reports.csv", "patient_id"),
    ("data/*_labels.csv", "patient_id"),
    ("data/*_matched.csv", "patient_id"),
    ("data/hospital_reports_labels.csv", "patient_id"),
    ("data/matched_clean.csv", "patient_id"),
    ("data/rehearsal/*.csv", "patient_id"),
    ("data/sets/*/studies.csv", "patient_id"),
    ("data/holdout_patients_*.txt", None),  # one id per line
]
# A 6-8 digit run not glued to other DIGITS (or to hex letters / a '.', which would make it part of
# a hash or a version / decimal). '_' and other letters do NOT shield an id: '1234567_Name.pdf',
# '1234567_new2_0.png' and 'pid_1234567' are exactly the forms a pasted fixture carries.
ID_TOKEN_RE = re.compile(r"(?<![0-9A-Fa-f.])(\d{6,8})(?![0-9A-Fa-f])")

FIO_NAME_RE = re.compile(
    r"(?:Ф\.\s?И\.\s?О\.?|\bФИО\b)\s*[:\-–]?\s*[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]+|\s+[А-ЯЁ]\.)")
# invented surnames used on purpose by the synthetic fixtures
SYNTHETIC_NAME_STEMS = ("Тестов", "Синтетов", "Докторов", "Лаборантов")

GH_TOKEN_RE = re.compile(r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}")
# the obviously synthetic token of tests/test_publish_bundle.py (exact match)
TOKEN_ALLOWLIST = {"ghp_SECRET_token_value_0123456789"}

THIS_FILE = Path(__file__).resolve()


def _git_files() -> list[Path] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=60, check=True).stdout
    except Exception:
        return None
    files = []
    for rel in out.decode("utf-8", "replace").split("\0"):
        if not rel:
            continue
        parts = Path(rel).parts
        if len(parts) == 1 or parts[0] in SCAN_DIRS:
            if not SKIP_DIR_NAMES.intersection(parts[:-1]):
                files.append(REPO / rel)
    return files


def _walk_files() -> list[Path]:
    files = [p for p in REPO.iterdir() if p.is_file()]
    for d in SCAN_DIRS:
        for root, dirs, names in os.walk(REPO / d):
            dirs[:] = [x for x in dirs if x not in SKIP_DIR_NAMES and not x.startswith(".")]
            files.extend(Path(root) / n for n in names)
    return files


def _text_files() -> list[tuple[Path, str]]:
    files = _git_files()
    if files is None:
        files = _walk_files()
    out = []
    for p in sorted(set(files)):
        if p.resolve() == THIS_FILE or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:8192]:
            continue
        out.append((p, raw.decode("utf-8", "replace")))
    return out


@pytest.fixture(scope="module")
def text_files():
    files = _text_files()
    assert files, "no files to scan — the guard would be vacuous"
    return files


def _id_source_files() -> list[tuple[Path, str | None]]:
    out = []
    for pattern, col in ID_SOURCES:
        out.extend((p, col) for p in sorted(REPO.glob(pattern)) if p.is_file())
    return out


def _real_patient_ids() -> set[str]:
    ids: set[str] = set()
    for p, col in _id_source_files():
        try:
            with open(p, encoding="utf-8-sig", newline="") as fh:
                if col is None:
                    vals = (line.strip() for line in fh)
                else:
                    vals = (row.get(col, "") for row in csv.DictReader(fh))
                for v in vals:
                    v = re.sub(r"\.0$", "", (v or "").strip())
                    if v.isdigit() and 6 <= len(v) <= 8:
                        ids.add(v)
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
    return ids


def _mask(token: str) -> str:
    return token[:1] + "*" * (len(token) - 2) + token[-1:]


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def test_no_real_patient_id_in_committable_files(text_files):
    real = _real_patient_ids()
    if not real:
        pytest.skip("no local patient-id sources under data/ (fresh clone / CI)")
    hits = []
    for p, text in text_files:
        for n, line in enumerate(text.splitlines(), start=1):
            for m in ID_TOKEN_RE.finditer(line):
                if m.group(1) in real:
                    hits.append(f"{_rel(p)}:{n} ({_mask(m.group(1))})")
    assert not hits, ("real patient ids in committable files — replace with synthetic ids "
                      ">= 9000000:\n  " + "\n  ".join(hits[:50]))


def test_no_letterhead_name_in_code(text_files):
    hits = []
    for p, text in text_files:
        if p.suffix.lower() in DOC_SUFFIXES or _rel(p).startswith("docs/"):
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            for m in FIO_NAME_RE.finditer(line):
                if any(stem in m.group(0) for stem in SYNTHETIC_NAME_STEMS):
                    continue
                hits.append(f"{_rel(p)}:{n}")
    assert not hits, "'Ф.И.О.' + a literal name in code:\n  " + "\n  ".join(hits[:50])


def test_no_real_looking_github_token(text_files):
    hits = []
    for p, text in text_files:
        for n, line in enumerate(text.splitlines(), start=1):
            for m in GH_TOKEN_RE.finditer(line):
                if m.group(0) in TOKEN_ALLOWLIST:
                    continue
                hits.append(f"{_rel(p)}:{n} ({m.group(0)[:8]}…)")
    assert not hits, "real-looking GitHub tokens:\n  " + "\n  ".join(hits[:50])


# ------------------------------------------------ the guard's own detectors

def test_detectors_catch_what_they_should():
    assert ID_TOKEN_RE.findall("pid = '9000123'; 12345; 123456789; v1.123456") == ["9000123"]
    # the forms real ids take in file names (review round 2): '_' / letters do not shield an id
    for s in ("'9000123_Surname Name.pdf'", "pid_9000123", "'9000123_new2_0.png'",
              "data/sets/b/train/normal/9000123_3.png", "id=9000123", "ПИД9000123"):
        assert ID_TOKEN_RE.findall(s) == ["9000123"], s
    # ...but a digit run inside a hex digest / decimal is not an id
    assert ID_TOKEN_RE.findall("sha 3fa9000123bc 0.9000123") == []
    assert FIO_NAME_RE.search("Ф.И.О. Иванов Иван")
    assert FIO_NAME_RE.search("ФИО: Петрова А.")
    assert not FIO_NAME_RE.search("_FIO = r'(?:[Фф]\\.\\s?[Ии]\\.\\s?[Оо]\\.?|ФИО)'")
    assert not FIO_NAME_RE.search('f"Ф.И.О. {FAKE_PATIENT}"')
    assert GH_TOKEN_RE.search("ghp_" + "a1" * 18)
    assert GH_TOKEN_RE.search("github_pat_" + "A" * 30)
    assert not GH_TOKEN_RE.search("ghp_SECRET_token_value_0123456789")
    assert _mask("1234567") == "1*****7"


def test_id_sources_cover_every_batch_file_pattern(tmp_path, monkeypatch):
    """A new batch (data/new2_index.csv, …) is guarded without editing this file."""
    import sys
    mod = sys.modules[__name__]
    for rel in ("data/new2_index.csv", "data/new2_reports.csv", "data/new2_labels.csv",
                "data/new2_matched.csv", "data/sets/new2/studies.csv"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("patient_id,x\n9000777,a\n", encoding="utf-8")
    (tmp_path / "data" / "holdout_patients_v2.txt").write_text("9000778\n", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    assert _real_patient_ids() == {"9000777", "9000778"}
    assert len(_id_source_files()) == 6
