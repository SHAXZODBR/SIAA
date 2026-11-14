"""scripts/archive_studies.py — prioritised, verified archive of a batch off the clinic drive.

Synthetic trees only (tmp_path): the "DICOM" files are random bytes named *.dcm, study folders and
ids are made up. Nothing here reads a real study.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "archive_studies.py"


def _load():
    spec = importlib.util.spec_from_file_location("archive_studies", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


A = _load()


def _study(root: Path, folder: str, series: dict[str, int], extra: tuple[str, ...] = ()) -> None:
    for sname, n in series.items():
        d = root / folder / sname
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i:04d}.dcm").write_bytes(os.urandom(64) + f"{folder}/{sname}/{i}".encode())
    for name in extra:
        (root / folder / name).write_bytes(b"not dicom " + name.encode())


@pytest.fixture
def batch(tmp_path: Path) -> dict:
    root = tmp_path / "drive"
    _study(root, "S_brain", {"t1": 3, "t2": 2}, extra=("shot.bmp",))
    _study(root, "S_head", {"thin": 50, "thick": 7}, extra=("shot.bmp",))
    _study(root, "S_knee", {"pd": 4})
    _study(root, "S_chest", {"lung": 30})
    index = tmp_path / "index.csv"
    rows = [("S_brain", "MR", "BRAIN", ""), ("S_head", "CT", "HEAD", ""), ("S_knee", "MR", "KNEE", ""),
            ("S_chest", "CT", "CHEST", ""), ("S_broken", "", "", "no dcm"), ("S_us", "US", "ABDOMEN", "")]
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["study_folder", "modality", "body_part", "error"])
        w.writerows(rows)
    dest = tmp_path / "archive"
    base = ["--root", str(root), "--index", str(index), "--dest", str(dest), "--min-free-gb", "0"]
    return {"root": root, "index": index, "dest": dest, "base": base}


RULES = ["--rule", "MR:BRAIN", "--rule", "CT:HEAD:cap=5", "--rule", "MR:*", "--rule", "CT:*:cap=3"]


def _state(dest: Path) -> dict[str, dict]:
    return A.read_state(dest)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ pure helpers

def test_rule_parsing_and_matching():
    r = A.parse_rule("ct:head:cap=40")
    assert (r.modality, r.body, r.cap, r.mode) == ("CT", "HEAD", 40, "cap=40")
    assert r.matches("CT", "head") and not r.matches("MR", "HEAD") and not r.matches("CT", "CHEST")
    star = A.parse_rule("MR:*SPINE")
    assert star.mode == "raw" and star.matches("mr", "LSPINE") and star.matches("MR", "CSPINE")
    assert not star.matches("MR", "KNEE") and A.parse_rule("MR:*").matches("MR", "")
    for bad in ("MR", "MR:", ":BRAIN", "CT:HEAD:max=3", "CT:HEAD:cap=x", "CT:HEAD:cap=0", "a:b:c:d"):
        with pytest.raises(ValueError):
            A.parse_rule(bad)


@pytest.mark.parametrize("n,cap,expected", [
    (3, 5, [0, 1, 2]), (5, 5, [0, 1, 2, 3, 4]), (50, 5, [0, 12, 24, 37, 49]), (10, 1, [5]), (10, 2, [0, 9]),
    (0, 4, []),
])
def test_spaced_indices(n, cap, expected):
    got = A.spaced_indices(n, cap)
    assert got == expected and len(got) == min(n, cap) and len(set(got)) == len(got)


def test_mode_rank_orders_raw_above_caps():
    assert A.mode_rank("raw") > A.mode_rank("cap=40") > A.mode_rank("cap=12") > A.mode_rank("nonsense")


# ------------------------------------------------------------------ end to end

def test_archive_copies_in_rule_order_raw_and_capped_and_verifies(batch, capsys):
    assert A.main([*batch["base"], *RULES]) == A.EXIT_OK
    out = capsys.readouterr().out
    assert "every selected study is archived and verified" in out and "S_brain" not in out
    dest, root = batch["dest"], batch["root"]
    st = _state(dest)
    assert {k: (v["mode"], v["status"]) for k, v in st.items()} == {
        "S_brain": ("raw", "ok"), "S_head": ("cap=5", "ok"), "S_knee": ("raw", "ok"), "S_chest": ("cap=3", "ok")}
    assert "S_us" not in st and "S_broken" not in st                      # no rule / index error
    # raw: every file, byte for byte, the non-DICOM one too
    for rel in ("t1/img_0000.dcm", "t1/img_0002.dcm", "t2/img_0001.dcm", "shot.bmp"):
        assert (dest / "images" / "S_brain" / rel).read_bytes() == (root / "S_brain" / rel).read_bytes()
    assert int(st["S_brain"]["files_copied"]) == 6 == int(st["S_brain"]["files_on_drive"])
    # cap: per series directory, spaced, first and last kept, non-DICOM skipped
    thin = sorted(p.name for p in (dest / "images" / "S_head" / "thin").iterdir())
    assert thin == [f"img_{i:04d}.dcm" for i in (0, 12, 24, 37, 49)]
    assert len(list((dest / "images" / "S_head" / "thick").iterdir())) == 5
    assert not (dest / "images" / "S_head" / "shot.bmp").exists()
    assert int(st["S_head"]["files_copied"]) == 10 and int(st["S_head"]["files_on_drive"]) == 58
    assert len(list((dest / "images" / "S_chest" / "lung").iterdir())) == 3
    # order: the first rule's studies are copied first
    order = [r["study_folder"] for r in csv.DictReader(open(dest / A.STUDIES_CSV, encoding="utf-8"))]
    assert order == ["S_brain", "S_head", "S_knee", "S_chest"]
    # manifest: one line per copied file, hashes and sizes are those of the archived files
    lines = (dest / A.FILES_SHA).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 + 10 + 4 + 3
    for line in lines:
        digest, size, rel = line.split("  ", 2)
        p = dest / rel
        assert p.is_file() and _sha(p) == digest and p.stat().st_size == int(size)
    assert not list(dest.rglob("*.part")) and (dest / A.README).is_file()


def test_resume_skips_done_and_a_fuller_rule_recopies(batch, capsys):
    assert A.main([*batch["base"], "--rule", "CT:*:cap=3"]) == A.EXIT_OK
    assert len(list((batch["dest"] / "images" / "S_head" / "thin").iterdir())) == 3
    capsys.readouterr()
    # the same run again: nothing to do
    assert A.main([*batch["base"], "--rule", "CT:*:cap=3"]) == A.EXIT_OK
    assert "nothing left to copy" in capsys.readouterr().out
    # a smaller cap never replaces a fuller archive
    assert A.main([*batch["base"], "--rule", "CT:*:cap=2"]) == A.EXIT_OK
    assert len(list((batch["dest"] / "images" / "S_head" / "thin").iterdir())) == 3
    # raw upgrades it
    assert A.main([*batch["base"], "--rule", "CT:HEAD"]) == A.EXIT_OK
    assert len(list((batch["dest"] / "images" / "S_head" / "thin").iterdir())) == 50
    assert _state(batch["dest"])["S_head"]["mode"] == "raw"
    assert (batch["dest"] / "images" / "S_head" / "shot.bmp").is_file()


def test_free_space_floor_stops_before_copying_and_resume_continues(batch, monkeypatch, capsys):
    real = A.shutil.disk_usage
    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        u = real(path)
        return type(u)(u.total, u.used, 5 * 10**9 if calls["n"] > 1 else 500 * 10**9)   # floor hit before study 2
    monkeypatch.setattr(A.shutil, "disk_usage", fake)
    args = [a for a in batch["base"] if a not in ("--min-free-gb", "0")] + ["--min-free-gb", "100", *RULES]
    assert A.main(args) == A.EXIT_STOPPED
    out = capsys.readouterr().out
    assert "below the floor" in out and "run the same command again" in out
    st = _state(batch["dest"])
    assert list(st) == ["S_brain"] and st["S_brain"]["status"] == "ok"
    monkeypatch.setattr(A.shutil, "disk_usage", real)
    assert A.main([*batch["base"], *RULES]) == A.EXIT_OK
    assert set(_state(batch["dest"])) == {"S_brain", "S_head", "S_knee", "S_chest"}


def test_a_copy_that_does_not_verify_fails_the_study_and_is_retried(batch, monkeypatch, capsys):
    real_open = open
    state = {"corrupt": True}

    def corrupting_copy(src, dst):
        digest, n = A_copy(src, dst)
        return digest, n
    A_copy = A.copy_verified

    def bad_copy(src, dst):
        if state["corrupt"] and src.name == "img_0001.dcm" and "S_knee" in str(src):
            raise A.VerifyError("simulated bit flip")
        return A_copy(src, dst)
    monkeypatch.setattr(A, "copy_verified", bad_copy)
    assert A.main([*batch["base"], *RULES]) == A.EXIT_VERIFY
    st = _state(batch["dest"])
    assert st["S_knee"]["status"] == "failed: verification" and st["S_brain"]["status"] == "ok"
    assert st["S_chest"]["status"] == "ok", "one bad study must not stop the others"
    state["corrupt"] = False
    assert A.main([*batch["base"], *RULES]) == A.EXIT_OK
    assert _state(batch["dest"])["S_knee"]["status"] == "ok"
    assert len(list((batch["dest"] / "images" / "S_knee" / "pd").iterdir())) == 4


def test_copy_verified_detects_a_changed_copy(tmp_path, monkeypatch):
    src = tmp_path / "a.dcm"
    src.write_bytes(os.urandom(1000))
    digest, n = A.copy_verified(src, tmp_path / "out" / "a.dcm")
    assert n == 1000 and digest == _sha(src) == _sha(tmp_path / "out" / "a.dcm")
    # the archive volume returns different bytes on re-read
    real_sha = hashlib.sha256
    seen = {"n": 0}

    class Flaky:
        def __init__(self):
            seen["n"] += 1
            self._h = real_sha()
            self._second = seen["n"] % 2 == 0

        def update(self, b):
            self._h.update(b + (b"x" if self._second else b""))

        def hexdigest(self):
            return self._h.hexdigest()
    monkeypatch.setattr(A.hashlib, "sha256", Flaky)
    with pytest.raises(A.VerifyError):
        A.copy_verified(src, tmp_path / "out" / "b.dcm")
    assert not (tmp_path / "out" / "b.dcm").exists() and not (tmp_path / "out" / "b.dcm.part").exists()


def test_missing_study_and_unreadable_file_are_failures_not_crashes(batch, monkeypatch):
    import shutil as _sh
    _sh.rmtree(batch["root"] / "S_knee")
    assert A.main([*batch["base"], *RULES]) == A.EXIT_STOPPED
    st = _state(batch["dest"])
    assert st["S_knee"]["status"] == "failed: not on the drive" and st["S_chest"]["status"] == "ok"


def test_dry_run_and_usage_errors(batch, capsys, tmp_path):
    assert A.main([*batch["base"], *RULES, "--dry-run"]) == A.EXIT_OK
    out = capsys.readouterr().out
    assert "dry run" in out and "to copy: 1" in out and not batch["dest"].exists()
    assert A.main([*batch["base"], "--rule", "nonsense"]) == A.EXIT_USAGE
    assert A.main(["--root", str(tmp_path / "nope"), "--index", str(batch["index"]), "--dest", str(batch["dest"]),
                   "--rule", "MR:*"]) == A.EXIT_USAGE
    assert A.main(["--root", str(batch["root"]), "--index", str(batch["index"]),
                   "--dest", str(REPO / "data" / "x"), "--rule", "MR:*"]) == A.EXIT_USAGE
    assert "inside the repository" in capsys.readouterr().out


def test_limit_archives_only_the_first_studies(batch):
    assert A.main([*batch["base"], *RULES, "--limit", "2"]) == A.EXIT_OK
    assert list(_state(batch["dest"])) == ["S_brain", "S_head"]
