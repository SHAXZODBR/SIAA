"""scripts/intake_batch.py + scripts/backup_mirror.sh — "When a drive arrives" (runbook §2a).

Everything runs on synthetic trees under tmp_path: a fake hospital drive with two tiny DICOM
studies (written with pydicom, ids >= 9000000, dates in 2031, no pixel data) plus fake report
files named like the hospital's <patientID>_<Name>.pdf, and a fake archive root. Nothing real is
read; the mirror tests need rsync + shasum (both ship with macOS; skipped when absent).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INTAKE = REPO / "scripts" / "intake_batch.py"
MIRROR = REPO / "scripts" / "backup_mirror.sh"
SYN_NAME = "Синтетов Синтет"


@pytest.fixture(scope="module")
def ib():
    spec = importlib.util.spec_from_file_location("intake_batch", INTAKE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dicom(path: Path, pid: str, study_date: str, modality: str) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = pid
    ds.PatientName = "SYNTHETIC^PATIENT"
    ds.StudyDate = study_date
    ds.Modality = modality
    ds.BodyPartExamined = "HEAD"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesDescription = "T2 AX"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    assert pydicom.dcmread(str(path)).PatientID == pid


@pytest.fixture
def drive(tmp_path: Path) -> dict:
    """A fake hospital drive: drive/new9/<2 study folders>, drive/reports/<pdf, xlsx>, OS junk."""
    root = tmp_path / "HOSPITAL_DRIVE"
    images = root / "new9"
    _dicom(images / "STUDY_A" / "SER1" / "IMG0001.dcm", "9000501", "20310105", "MR")
    _dicom(images / "STUDY_A" / "SER2" / "IMG0001.dcm", "9000501", "20310105", "MR")
    _dicom(images / "STUDY_B" / "SER1" / "IMG0001.dcm", "9000502", "20310220", "CT")
    (images / ".DS_Store").write_bytes(b"\0junk")
    (images / "STUDY_A" / "._IMG0001.dcm").write_bytes(b"resource fork")
    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / f"9000501_{SYN_NAME}.pdf").write_bytes(b"%PDF-1.4 synthetic report 9000501")
    (reports / "request_sheet_2031.xlsx").write_bytes(b"PK synthetic sheet")
    dest = tmp_path / "SIAA_DATA"
    dest.mkdir()
    return {"root": root, "images": images, "reports": reports, "dest": dest}


def _args(d: dict, *extra: str) -> list[str]:
    return ["--src", str(d["images"]), "--reports", str(d["reports"]), "--clinic", "shox", "--batch", "new9",
            "--dest", str(d["dest"]), "--received", "2031-03-01", *extra]


def _batch(d: dict) -> Path:
    return d["dest"] / "raw" / "shox" / "new9"


def _tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------- intake ---

def test_intake_copies_verifies_and_writes_manifest_readme_inventory(ib, drive, capsys):
    before = _tree_digest(drive["root"])
    mtimes = {p: p.stat().st_mtime_ns for p in drive["root"].rglob("*")}
    assert ib.main(_args(drive)) == ib.EXIT_OK
    out = capsys.readouterr().out
    b = _batch(drive)
    # layout + junk skipped
    assert (b / "images" / "STUDY_A" / "SER1" / "IMG0001.dcm").is_file()
    assert (b / "reports" / f"9000501_{SYN_NAME}.pdf").is_file()
    assert not (b / "images" / ".DS_Store").exists() and not (b / "images" / "STUDY_A" / "._IMG0001.dcm").exists()
    assert not list(b.rglob("*.part"))
    # the source is untouched (content AND timestamps)
    assert _tree_digest(drive["root"]) == before
    assert {p: p.stat().st_mtime_ns for p in drive["root"].rglob("*")} == mtimes
    # CHECKSUMS.sha256 in `shasum` format, every copied file, correct hashes; sizes next to it
    lines = (b / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((b / rel).read_bytes()).hexdigest() == digest
    sizes = dict(reversed(ln.split("  ", 1)) for ln in (b / "CHECKSUMS.sizes").read_text(encoding="utf-8").splitlines())
    assert all(int(sizes[rel]) == (b / rel).stat().st_size for rel in sizes) and len(sizes) == 5
    if shutil.which("shasum"):
        assert subprocess.run(["shasum", "-a", "256", "-c", "CHECKSUMS.sha256"], cwd=b,
                              capture_output=True).returncode == 0
    # README: received date, source, counts, header-scan date range + modalities
    readme = (b / "README.txt").read_text(encoding="utf-8")
    assert "received        : 2031-03-01" in readme and str(drive["images"]) in readme
    assert "image files     : 3" in readme and "report files    : 2" in readme
    assert "study folders   : 2" in readme and "header scan     : 2 studies scanned" in readme
    assert "study dates     : 2031-01-05 … 2031-02-20" in readme and "MR 1" in readme and "CT 1" in readme
    assert "verification    : PASSED" in readme
    # INVENTORY.csv row
    rows = list(csv.DictReader(open(drive["dest"] / "raw" / "INVENTORY.csv", encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert (r["clinic"], r["batch"], r["received_date"], r["n_image_files"], r["n_report_files"]) == \
        ("shox", "new9", "2031-03-01", "3", "2")
    assert (r["study_date_min"], r["study_date_max"], r["date_range_is_sample"]) == ("2031-01-05", "2031-02-20", "no")
    assert r["checksums_sha256"] == hashlib.sha256((b / "CHECKSUMS.sha256").read_bytes()).hexdigest()
    # the exact pipeline command: --root is the folder holding the study folders
    assert f"scripts/release_pipeline.py run --batch new9 --root {b / 'images'} --reports {b / 'reports'}" in out
    assert "release_pipeline.py plan" in out and "intake verified" in out
    # PHI: no report file name, no patient id from the headers on stdout / README / INVENTORY
    inv_text = (drive["dest"] / "raw" / "INVENTORY.csv").read_text(encoding="utf-8")
    for blob in (out, readme, inv_text):
        assert SYN_NAME not in blob and "Синтетов" not in blob and "9000502" not in blob


def test_study_root_descends_through_single_folders(ib, tmp_path):
    images = tmp_path / "images"
    for s in ("S1", "S2"):
        (images / "DICOM" / "new9" / s).mkdir(parents=True)
    assert ib.study_root(images) == images / "DICOM" / "new9"
    (images / "DICOM" / "new9" / "S1" / "x.dcm").write_bytes(b"x")
    assert ib.study_root(images) == images / "DICOM" / "new9"
    (images / "README.TXT").write_text("x")
    assert ib.study_root(images) == images


def test_refuses_existing_batch_without_resume(ib, drive, capsys):
    assert ib.main(_args(drive)) == ib.EXIT_OK
    before = _tree_digest(_batch(drive))
    capsys.readouterr()
    assert ib.main(_args(drive)) == ib.EXIT_USAGE
    out = capsys.readouterr().out
    assert "already exists" in out and "--resume" in out
    assert _tree_digest(_batch(drive)) == before


def test_refuses_missing_dest_and_overlapping_paths(ib, drive, tmp_path, capsys):
    a = _args(drive)
    a[a.index("--dest") + 1] = str(tmp_path / "not_mounted")
    assert ib.main(a) == ib.EXIT_USAGE and not (tmp_path / "not_mounted").exists()
    assert "never created" in capsys.readouterr().out
    # archive root inside the source drive → the batch would be copied into itself
    inside = drive["images"] / "archive"
    inside.mkdir()
    a = _args(drive)
    a[a.index("--dest") + 1] = str(inside)
    assert ib.main(a) == ib.EXIT_USAGE and "overlap" in capsys.readouterr().out
    shutil.rmtree(inside)
    a = _args(drive)
    a[a.index("--batch") + 1] = "../escape"
    assert ib.main(a) == ib.EXIT_USAGE


def test_resume_skips_copied_files_and_finishes_an_interrupted_intake(ib, drive, monkeypatch, capsys):
    real_copy = ib.copy_one
    calls: list[str] = []

    def copy_then_crash(src, dst):
        calls.append(src.name)
        if len(calls) == 3:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.with_name(dst.name + ".part").write_bytes(b"half")       # torn copy
            raise KeyboardInterrupt
        return real_copy(src, dst)

    monkeypatch.setattr(ib, "copy_one", copy_then_crash)
    assert ib.main(_args(drive)) == 130
    b = _batch(drive)
    assert len(ib.read_journal(b / ".intake" / "journal.tsv")) == 2
    assert list(b.rglob("*.part")) and not (drive["dest"] / "raw" / "INVENTORY.csv").exists()
    capsys.readouterr()
    copied: list[str] = []

    def counting_copy(src, dst):
        copied.append(src.name)
        return real_copy(src, dst)

    monkeypatch.setattr(ib, "copy_one", counting_copy)
    assert ib.main(_args(drive, "--resume")) == ib.EXIT_OK
    out = capsys.readouterr().out
    assert "2 already copied" in out and "3 to copy" in out and "1 stale .part removed" in out
    assert len(copied) == 3 and not list(b.rglob("*.part"))
    assert len((b / "CHECKSUMS.sha256").read_text().splitlines()) == 5
    # a second resume has nothing to copy
    copied.clear()
    assert ib.main(_args(drive, "--resume")) == ib.EXIT_OK
    assert copied == [] and "5 already copied" in capsys.readouterr().out
    assert len(list(csv.DictReader(open(drive["dest"] / "raw" / "INVENTORY.csv", encoding="utf-8")))) == 1


def test_post_copy_verification_catches_a_flipped_byte(ib, drive, monkeypatch, capsys):
    real_copy = ib.copy_one

    def flipping_copy(src, dst):
        n, digest = real_copy(src, dst)
        if src.name.endswith(".pdf"):                 # the disk "flips" one bit of this copy
            data = bytearray(dst.read_bytes())
            data[5] ^= 0x01
            dst.write_bytes(bytes(data))
        return n, digest

    monkeypatch.setattr(ib, "copy_one", flipping_copy)
    assert ib.main(_args(drive)) == ib.EXIT_VERIFY_FAILED
    out = capsys.readouterr().out
    b = _batch(drive)
    assert "VERIFICATION FAILED" in out and "sha256 mismatch" in out
    assert SYN_NAME not in out and "9000501_" in out                    # the name part is hashed
    recopy = (b / ".intake" / "needs_recopy.txt").read_text(encoding="utf-8").splitlines()
    assert len(recopy) == 1 and recopy[0].startswith("reports/9000501_")
    assert "verification    : FAILED" in (b / "README.txt").read_text(encoding="utf-8")
    assert not (drive["dest"] / "raw" / "INVENTORY.csv").exists()       # never inventoried unverified
    # --resume copies exactly the bad file again (sizes are equal, so only the list says so)
    monkeypatch.setattr(ib, "copy_one", real_copy)
    assert ib.main(_args(drive, "--resume")) == ib.EXIT_OK
    assert "1 to copy" in capsys.readouterr().out
    assert not (b / ".intake" / "needs_recopy.txt").exists()
    assert (drive["dest"] / "raw" / "INVENTORY.csv").is_file()


def test_verify_mode_rehashes_the_whole_batch(ib, drive, capsys):
    assert ib.main(_args(drive)) == ib.EXIT_OK
    b = _batch(drive)
    verify = ["--clinic", "shox", "--batch", "new9", "--dest", str(drive["dest"]), "--verify"]
    assert ib.main(verify) == ib.EXIT_OK
    capsys.readouterr()
    f = b / "images" / "STUDY_B" / "SER1" / "IMG0001.dcm"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))
    assert ib.main(verify) == ib.EXIT_VERIFY_FAILED
    out = capsys.readouterr().out
    assert "images/STUDY_B/SER1/IMG0001.dcm: sha256 mismatch" in out and "1 of 5" in out
    assert "images/STUDY_B/SER1/IMG0001.dcm" in (b / ".intake" / "needs_recopy.txt").read_text()


def test_source_change_between_runs_is_refused(ib, drive, capsys):
    assert ib.main(_args(drive)) == ib.EXIT_OK
    (drive["reports"] / "request_sheet_2031.xlsx").write_bytes(b"PK a different, longer synthetic sheet")
    capsys.readouterr()
    assert ib.main(_args(drive, "--resume")) == ib.EXIT_USAGE
    assert "changed since the first intake" in capsys.readouterr().out


def test_reports_added_later_with_resume(ib, drive, capsys):
    a = [x for x in _args(drive)]
    i = a.index("--reports")
    del a[i:i + 2]
    assert ib.main(a) == ib.EXIT_OK
    out = capsys.readouterr().out
    assert "reports/ is empty" in out
    b = _batch(drive)
    assert "not received yet" in (b / "README.txt").read_text(encoding="utf-8")
    later = ["--reports", str(drive["reports"]), "--clinic", "shox", "--batch", "new9",
             "--dest", str(drive["dest"]), "--resume"]
    assert ib.main(later) == ib.EXIT_OK
    readme = (b / "README.txt").read_text(encoding="utf-8")
    assert str(drive["images"]) in readme and str(drive["reports"]) in readme   # both sources kept
    assert len((b / "CHECKSUMS.sha256").read_text().splitlines()) == 5
    rows = list(csv.DictReader(open(drive["dest"] / "raw" / "INVENTORY.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["n_report_files"] == "2"                  # row replaced, not duplicated


def test_inventory_keeps_other_batches(ib, drive):
    inv = drive["dest"] / "raw" / "INVENTORY.csv"
    inv.parent.mkdir(parents=True)
    inv.write_text("received_date,clinic,batch,legacy_note\n2031-01-01,shox,new8,kept\n", encoding="utf-8")
    assert ib.main(_args(drive)) == ib.EXIT_OK
    rows = list(csv.DictReader(open(inv, encoding="utf-8")))
    assert [r["batch"] for r in rows] == ["new8", "new9"] and rows[0]["legacy_note"] == "kept"


def test_single_report_file_source_name_is_hashed(ib, drive):
    single = drive["reports"] / f"9000501_{SYN_NAME}.pdf"
    a = ["--src", str(drive["images"]), "--reports", str(single), "--clinic", "shox", "--batch", "new9",
         "--dest", str(drive["dest"]), "--no-scan"]
    assert ib.main(a) == ib.EXIT_OK
    readme = (_batch(drive) / "README.txt").read_text(encoding="utf-8")
    inv = (drive["dest"] / "raw" / "INVENTORY.csv").read_text(encoding="utf-8")
    assert SYN_NAME not in readme and SYN_NAME not in inv and "9000501_" in readme
    assert "header scan skipped" in readme


def test_header_scan_failure_is_a_warning(ib, drive, capsys):
    a = _args(drive, "--python", "/nonexistent/python")
    assert ib.main(a) == ib.EXIT_OK
    out = capsys.readouterr().out
    assert "header scan could not run" in out
    assert "header scan could not run" in (_batch(drive) / "README.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------- mirror ---

needs_rsync = pytest.mark.skipif(not (shutil.which("rsync") and shutil.which("shasum") and shutil.which("bash")),
                                 reason="rsync / shasum / bash not available")


def _mirror(*args: str, stdin=subprocess.DEVNULL) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(MIRROR), *args], capture_output=True, text=True, stdin=stdin, timeout=120)


@pytest.fixture
def archives(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "SIAA DATA"                         # a space in the path on purpose
    (a / "raw" / "shox" / "new9" / "images" / "S1").mkdir(parents=True)
    (a / "raw" / "shox" / "new9" / "reports").mkdir(parents=True)
    for i in range(7):
        (a / "raw" / "shox" / "new9" / "images" / "S1" / f"IMG{i:04d}.dcm").write_bytes(os.urandom(256))
    (a / "raw" / "shox" / "new9" / "reports" / f"9000501_{SYN_NAME}.pdf").write_bytes(b"%PDF synthetic")
    (a / "raw" / "INVENTORY.csv").write_text("received_date,clinic,batch\n2031-03-01,shox,new9\n")
    (a / ".DS_Store").write_bytes(b"junk")
    b = tmp_path / "SIAA_BACKUP"
    b.mkdir()
    (b / "stale_from_last_year.txt").write_text("old")
    old = time.time() - 3600
    os.utime(a / "raw" / "INVENTORY.csv", (old, old))
    return a, b


@needs_rsync
def test_mirror_dry_run_changes_nothing(archives):
    a, b = archives
    r = _mirror("--dry-run", str(a), str(b))
    assert r.returncode == 0, r.stderr
    assert "files to copy / update on B: 9" in r.stdout and "files/folders to DELETE on B: 1" in r.stdout
    assert "stale_from_last_year.txt" in r.stdout and "nothing was changed" in r.stdout
    assert sorted(p.name for p in b.iterdir()) == ["stale_from_last_year.txt"]


@needs_rsync
def test_mirror_without_yes_and_without_terminal_is_refused(archives):
    a, b = archives
    r = _mirror(str(a), str(b))
    assert r.returncode == 1 and "not confirmed" in r.stderr
    assert sorted(p.name for p in b.iterdir()) == ["stale_from_last_year.txt"]


@needs_rsync
def test_mirror_with_yes_copies_deletes_and_verifies(archives):
    a, b = archives
    r = _mirror("--yes", str(a), str(b))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (b / "stale_from_last_year.txt").exists() and not (b / ".DS_Store").exists()
    src = {k: v for k, v in _tree_digest(a).items() if k != ".DS_Store"}
    assert _tree_digest(b) == src
    assert "sample verified  : 9 identical, 0 differing" in r.stdout and "verified mirror" in r.stdout
    assert SYN_NAME not in r.stdout
    # a second run is a no-op and still verifies
    r = _mirror("--yes", "--sample", "3", str(a), str(b))
    assert r.returncode == 0 and "files to copy / update on B: 0" in r.stdout and "3 identical" in r.stdout


@needs_rsync
def test_mirror_sample_verification_catches_a_corrupt_backup(archives):
    a, b = archives
    assert _mirror("--yes", str(a), str(b)).returncode == 0
    victim = b / "raw" / "shox" / "new9" / "reports" / f"9000501_{SYN_NAME}.pdf"
    victim.write_bytes(b"%PDF synthetiX")                       # same size, one byte differs
    r = _mirror("--verify-only", "--sample", "100", str(a), str(b))
    assert r.returncode == 3, r.stdout
    assert "sha256 differs" in r.stdout and "SAMPLE VERIFICATION FAILED" in r.stdout
    assert SYN_NAME not in r.stdout and "reports/" in r.stdout                  # the name is hashed
    victim.unlink()
    r = _mirror("--verify-only", "--sample", "100", str(a), str(b))
    assert r.returncode == 3 and "missing on B" in r.stdout


@needs_rsync
def test_mirror_refuses_when_b_inventory_is_newer(archives):
    a, b = archives
    assert _mirror("--yes", str(a), str(b)).returncode == 0
    b_inv = b / "raw" / "INVENTORY.csv"
    b_inv.write_text("received_date,clinic,batch\n2031-03-01,shox,new9\n2031-04-01,shox,new10\n")
    (b / "raw" / "shox" / "new10").mkdir()
    (b / "raw" / "shox" / "new10" / "only_on_b.dcm").write_bytes(b"x")
    r = _mirror("--yes", str(a), str(b))
    assert r.returncode == 1 and "NEWER" in r.stderr
    assert (b / "raw" / "shox" / "new10" / "only_on_b.dcm").is_file()          # nothing deleted
    # B with an inventory, A without one → refused as well
    (a / "raw" / "INVENTORY.csv").unlink()
    r = _mirror("--yes", str(a), str(b))
    assert r.returncode == 1 and "B holds raw/INVENTORY.csv but A does not" in r.stderr


@needs_rsync
def test_mirror_refuses_empty_or_missing_sources(tmp_path):
    a = tmp_path / "A"
    b = tmp_path / "B"
    a.mkdir()
    b.mkdir()
    (b / "keep.txt").write_text("x")
    r = _mirror("--yes", str(a), str(b))
    assert r.returncode == 1 and "empty" in r.stderr and (b / "keep.txt").is_file()
    r = _mirror("--yes", str(a), str(tmp_path / "not_mounted"))
    assert r.returncode == 1 and "never created" in r.stderr and not (tmp_path / "not_mounted").exists()
    r = _mirror("--yes", str(a))
    assert r.returncode == 1
    r = _mirror("--yes", str(b), str(b))
    assert r.returncode == 1 and "same directory" in r.stderr


def test_missing_single_report_file_error_hides_the_name(ib, drive, capsys):
    a = ["--src", str(drive["images"]), "--reports", str(drive["reports"] / f"9000599_{SYN_NAME}.pdf"),
         "--clinic", "shox", "--batch", "new9", "--dest", str(drive["dest"])]
    assert ib.main(a) == ib.EXIT_USAGE
    out = capsys.readouterr().out
    assert "does not exist" in out and "9000599_" in out and SYN_NAME not in out
