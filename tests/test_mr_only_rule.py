"""MR-only rule for the training set (build_training_set.py --modality), the series modality the
preprocessor records, the image-first study fields of index_new_studies.py and the optional --images
of match_images_to_reports.py.

Everything is synthetic: tiny DICOM series written with pydicom under tmp_path (patient ids >= 9000000,
dates in 2031), CSVs written next to them. No real study, report or patient name is touched.
"""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO

SCRIPTS = REPO / "scripts"
BUILD = SCRIPTS / "build_training_set.py"
MATCH = SCRIPTS / "match_images_to_reports.py"
INDEX = SCRIPTS / "index_new_studies.py"

MR_SOP = "1.2.840.10008.5.1.4.1.1.4"          # MR Image Storage
CT_SOP = "1.2.840.10008.5.1.4.1.1.2"          # CT Image Storage
SR_SOP = "1.2.840.10008.5.1.4.1.1.88.11"      # Basic Text SR


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ synthetic DICOM

def _meta(sop_class: str):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    return ds


def _write_series(folder: Path, modality: str | None, pid: str, n: int = 8, size: int = 32,
                  desc: str = "T1 AX", body_part: str = "BRAIN", tag: str = "img", seed: int = 0) -> Path:
    """n axial single-frame slices of one series; modality None writes no Modality tag."""
    from pydicom.uid import generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    suid = generate_uid()
    rng = np.random.default_rng(seed)
    for i in range(n):
        ds = _meta(CT_SOP if modality == "CT" else MR_SOP)
        if modality is not None:
            ds.Modality = modality
        ds.PatientID = pid
        ds.StudyInstanceUID = f"2.25.{pid}"
        ds.SeriesInstanceUID = suid
        ds.StudyDate = "20310105"
        ds.SeriesDescription = desc
        ds.BodyPartExamined = body_part
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.SliceThickness = 5
        ds.Rows = ds.Columns = size
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = (rng.integers(50, 1000, (size, size)) + 10 * i).astype(np.uint16).tobytes()
        ds.save_as(str(folder / f"{tag}_{i:03d}.dcm"), enforce_file_format=True)
    return folder


def _write_sr(folder: Path, pid: str, name: str = "sr_000.dcm") -> Path:
    """A structured report object: Modality SR, no image matrix, no body part."""
    from pydicom.uid import generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    ds = _meta(SR_SOP)
    ds.Modality = "SR"
    ds.PatientID = pid
    ds.StudyInstanceUID = f"2.25.{pid}"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = "20310105"
    ds.SeriesDescription = "Dose Report"
    ds.InstanceNumber = 1
    ds.save_as(str(folder / name), enforce_file_format=True)
    return folder / name


# ------------------------------------------------------------------ preprocessor

@pytest.mark.parametrize("modality,want", [("MR", "MR"), ("CT", "CT"), (None, "")])
def test_series_modality_lands_on_brain_series_and_seq_meta(tmp_path: Path, modality, want):
    from src.pipeline.brain_mri_preprocessor import group_dicom_files_into_series, process_brain_study
    d = _write_series(tmp_path / "s", modality, "9000001")
    files = sorted(d.glob("*.dcm"))
    series = group_dicom_files_into_series(files)
    assert len(series) == 1 and series[0].modality == want
    study = process_brain_study(files)
    assert study.seq_meta["T1"]["modality"] == want


def test_sr_object_does_not_become_a_volume_and_keeps_its_own_modality(tmp_path: Path):
    from src.pipeline.brain_mri_preprocessor import group_dicom_files_into_series, process_brain_study
    d = _write_series(tmp_path / "s", "MR", "9000002")
    _write_sr(d, "9000002")
    files = sorted(d.glob("*.dcm"))
    assert sorted(s.modality for s in group_dicom_files_into_series(files)) == ["MR", "SR"]
    study = process_brain_study(files)
    assert list(study.sequences) == ["T1"] and study.seq_meta["T1"]["modality"] == "MR"


def test_triage_input_reports_the_modality_of_the_series_its_slices_come_from(tmp_path: Path):
    """The CT series is described T1 (the preferred triage sequence), the MR one is 'unknown': the
    modality must follow the series triage_sequence picks, not the first or any other series."""
    b = _import(BUILD)
    from src.pipeline.brain_mri_preprocessor import process_brain_study, triage_sequence
    d = tmp_path / "mixed"
    _write_series(d / "a", "MR", "9000003", desc="localizer AX", tag="a", seed=1)
    _write_series(d / "b", "CT", "9000003", desc="T1 AX", tag="b", seed=2)
    study = process_brain_study(sorted(d.rglob("*.dcm")))
    assert triage_sequence(study) == "T1"
    slices, modality = b.triage_input(study)
    assert modality == "CT" and len(slices) == 5
    np.testing.assert_array_equal(np.stack(slices), np.stack(b.central_slices(study)))


# ------------------------------------------------------------------ build-set fixture

MATCHED_COLS = ["study_folder", "patient_id", "modality", "study_date", "match_quality", "body_part_dicom",
                "is_brain", "report_matched", "report_body_part", "label", "abnormal", "label_source",
                "label_confidence", "needs_review"]

# folder -> (patient id, matched-row modality, real series modality, abnormal)
STUDIES = {
    "A_mr": ("9000101", "MR", "MR", "yes"),
    "B_ct": ("9000102", "CT", "CT", "no"),        # dropped at row level under --modality MR
    "C_srmr": ("9000103", "SR", "MR", "no"),      # index said SR, really MR -> built
    "D_srct": ("9000104", "SR", "CT", "yes"),     # index said SR, really CT -> excluded at build time
    "E1_ct": ("9000105", "SR", "CT", "no"),       # one patient: a CT study (SR-labelled, sorts first) …
    "E2_mr": ("9000105", "MR", "MR", "yes"),      # … and an MR study
    "F1_ct": ("9000106", "CT", "CT", "no"),       # one patient: a CT study (CT-labelled, sorts first) …
    "F2_mr": ("9000106", "", "MR", "no"),         # … and an MR study whose row has no modality
}


@pytest.fixture
def fixture_set(tmp_path: Path) -> dict:
    images = tmp_path / "images"
    for i, (folder, (pid, _row_mod, real, _ab)) in enumerate(STUDIES.items()):
        _write_series(images / folder / "series1", real, pid, desc="T1 AX" if real == "MR" else "HEAD AX",
                      seed=i)
        if folder == "C_srmr":
            _write_sr(images / folder / "series0", pid)       # an SR object next to the MR images
    rows = [{"study_folder": f, "patient_id": pid, "modality": row_mod, "study_date": "20310105",
             "is_brain": "yes", "report_matched": "yes", "report_body_part": "brain",
             "label": "tumor" if ab == "yes" else "normal", "abnormal": ab, "label_source": "rules",
             "label_confidence": "high", "needs_review": "no"}
            for f, (pid, row_mod, _real, ab) in STUDIES.items()]
    matched = _write_csv(tmp_path / "matched.csv", MATCHED_COLS, rows)
    out = tmp_path / "set"
    base = ["--images", str(images), "--matched", str(matched), "--out", str(out), "--mode", "binary",
            "--abnormal-col", "abnormal", "--val-frac", "0"]
    return {"images": images, "matched": matched, "out": out, "base": base, "tmp": tmp_path}


def _main(b, monkeypatch, capsys, argv: list[str]) -> str:
    monkeypatch.setattr(b.sys, "argv", ["build_training_set.py", *argv])
    b.main()
    return capsys.readouterr().out


class _DicomRead(BaseException):
    """Raised by a process_brain_study stand-in: a BaseException, so the builder's per-study
    `except Exception` cannot swallow it and the test sees every DICOM read."""


def _no_dicom(files, *_a, **_k):
    raise _DicomRead(Path(files[0]).parent.parent.name)


def _pngs(out: Path) -> dict[str, list[str]]:
    """prefix -> sorted file names, over the whole set."""
    found: dict[str, list[str]] = {}
    for fp in out.glob("*/*/*.png"):
        found.setdefault(fp.stem.rsplit("_", 1)[0], []).append(fp.name)
    return {k: sorted(v) for k, v in found.items()}


def _snapshot(out: Path) -> dict[str, int]:
    return {str(p.relative_to(out)): p.stat().st_mtime_ns for p in out.glob("*/*/*.png")}


def _rows(out: Path) -> dict[str, dict]:
    return {r["study_folder"]: r for r in _read_csv(out / "studies.csv")}


def _ct_folders() -> set[str]:
    return {f for f, v in STUDIES.items() if v[2] == "CT"}


def _assert_only_mr_pngs(b, out: Path, images: Path):
    """Every PNG of the set belongs to a study whose series is MR (read back through studies.csv)."""
    rows = _read_csv(out / "studies.csv")
    prefix_of = b.read_prefix_map(out)
    folder_of = {p: f for f, p in prefix_of.items()}
    for prefix in _pngs(out):
        folder = folder_of[prefix]
        assert STUDIES[folder][2] == "MR", f"{folder} ({STUDIES[folder][2]}) has PNGs in an MR-only set"
    assert all(r["modality"] == "MR" for r in rows if int(r["n_slices"]) > 0)


# ------------------------------------------------------------------ build without a rule

def test_build_without_rule_is_unchanged_and_adds_the_modality_column(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    o = _main(b, monkeypatch, capsys, t["base"])
    assert "built 8" in o and "modality rule" not in o, o
    # the same prefixes as before the rule existed: every selected study numbered, CT included
    assert set(_pngs(t["out"])) == {"9000101", "9000102", "9000103", "9000104", "9000105", "9000105_s2",
                                    "9000106", "9000106_s2"}
    rows = _read_csv(t["out"] / "studies.csv")
    assert list(rows[0].keys()) == ["patient_id", "study_folder", "cls", "split", "n_slices", "label",
                                    "label_source", "modality"]
    by = {r["study_folder"]: r for r in rows}
    assert {f: r["modality"] for f, r in by.items()} == {f: v[2] for f, v in STUDIES.items()}
    # PNG content = the unchanged central_slices rule on the study (nothing numeric moved)
    from PIL import Image
    from src.pipeline.brain_mri_preprocessor import process_brain_study
    prefix_of = b.read_prefix_map(t["out"])
    for folder in ("A_mr", "B_ct"):
        study = process_brain_study(sorted((t["images"] / folder).rglob("*.dcm")))
        want = [(np.clip(s, 0, 1) * 255).astype(np.uint8) for s in b.central_slices(study)]
        r = by[folder]
        got = [np.asarray(Image.open(t["out"] / r["split"] / r["cls"] / f"{prefix_of[folder]}_{j}.png").convert("L"))
               for j in range(len(want))]
        assert int(r["n_slices"]) == len(want) == 5
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)


# ------------------------------------------------------------------ build with --modality MR

def test_build_with_mr_rule_row_level_and_build_time_exclusion(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "mr"])
    out = t["out"]
    assert "modality rule: --modality MR" in o, o
    assert "modality CT=2" in o, "B_ct and F1_ct are dropped from their rows, before any DICOM is read"
    assert "modality CT (series)=2" in o, "D_srct and E1_ct are excluded at build time"
    assert "built 4" in o and "excluded 2" in o and "failed" not in o, o

    rows = _rows(out)
    assert set(rows) == set(STUDIES) - {"B_ct", "F1_ct"}, "row-level drops take no '<pid>_sN' number"
    for f in ("A_mr", "C_srmr", "E2_mr", "F2_mr"):
        assert rows[f]["modality"] == "MR" and int(rows[f]["n_slices"]) == 5
    for f in ("D_srct", "E1_ct"):
        assert rows[f]["n_slices"] == "0" and rows[f]["modality"] == "CT"
    assert set(_pngs(out)) == {"9000101", "9000103", "9000105_s2", "9000106"}
    assert not list(out.glob("*/*/9000104_*.png")) and not list(out.glob("*/*/9000105_[0-9].png"))
    _assert_only_mr_pngs(b, out, t["images"])

    # second run: pure reuse, no DICOM read (the excluded CT is remembered), nothing renamed or rebuilt
    snap, csv_before = _snapshot(out), (out / "studies.csv").read_text()
    monkeypatch.setattr(b, "process_brain_study", _no_dicom)
    o2 = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert "built 0" in o2 and "reused 4" in o2 and "renumbered" not in o2 and "relabelled 0" in o2, o2
    assert "modality CT (series)=2" in o2 and "modality CT=2" in o2
    assert _snapshot(out) == snap
    assert (out / "studies.csv").read_text() == csv_before, "prefixes and records stable between runs"


def test_rule_rejects_non_acquisition_values(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    for bad in ("SR", " ", "ot"):
        with pytest.raises(SystemExit) as e:
            _main(b, monkeypatch, capsys, [*fixture_set["base"], "--modality", bad])
        assert e.value.code == 2


def test_rule_that_excludes_everything_exits_1(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    # only the SR-labelled CT study is selectable: excluded at build time -> empty set -> exit 1
    _write_csv(t["matched"], MATCHED_COLS, [r for r in _read_csv(t["matched"]) if r["study_folder"] == "D_srct"])
    with pytest.raises(SystemExit) as e:
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert e.value.code == 1
    assert "0 failed, 1 excluded by --modality MR" in capsys.readouterr().out
    # only CT-labelled rows: nothing survives the row filter -> the existing 'no trainable' exit
    _write_csv(t["matched"], MATCHED_COLS, [{**r, "study_folder": "B_ct", "patient_id": "9000102", "modality": "CT"}
                                           for r in _read_csv(t["matched"])])
    with pytest.raises(SystemExit) as e:
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert e.value.code == 1 and "'modality CT': 1" in capsys.readouterr().out


# ------------------------------------------------------------------ reuse rule

def test_set_built_without_rule_is_verified_when_the_rule_is_added(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    _main(b, monkeypatch, capsys, t["base"])
    assert len(_pngs(out)) == 8                       # CT built too
    # an older version's studies.csv has no modality column at all: nothing counts as verified
    rows = _read_csv(out / "studies.csv")
    _write_csv(out / "studies.csv", [c for c in rows[0] if c != "modality"], rows)

    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert "re-verified 4" in o, o
    assert "removed the PNGs of 2 studies no longer selected" in o, "row-level CT drops are stale on a full run"
    _assert_only_mr_pngs(b, out, t["images"])
    rows = _rows(out)
    assert {f for f, r in rows.items() if r["modality"] == "MR"} == {"A_mr", "C_srmr", "E2_mr", "F2_mr"}
    for f in ("A_mr", "C_srmr", "E2_mr", "F2_mr"):
        assert len(list(out.glob(f"*/*/{b.read_prefix_map(out)[f]}_*.png"))) == 5 == int(rows[f]["n_slices"])
    # prefix of patient 9000106's MR study changed ('_s2' -> '') because its CT row is now dropped:
    # the files followed their study, no extra PNG of any prefix is left over
    assert set(_pngs(out)) == {"9000101", "9000103", "9000105_s2", "9000106"}

    # now verified: the next run reuses without reading DICOM
    monkeypatch.setattr(b, "process_brain_study", _no_dicom)
    o2 = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert "reused 4" in o2 and "built 0" in o2, o2


def test_limited_run_with_rule_removes_unverified_files_outside_the_selection(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    _main(b, monkeypatch, capsys, t["base"])          # no rule: 8 studies, CT included, nothing verified
    rows = _read_csv(out / "studies.csv")
    _write_csv(out / "studies.csv", [c for c in rows[0] if c != "modality"], rows)
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR", "--limit", "1"])
    assert "outside this selection" in o, o
    _assert_only_mr_pngs(b, out, t["images"])
    assert set(_pngs(out)) == {"9000101"}, "only the verified study of this run may keep files"


def test_no_rule_run_keeps_the_verified_modality_of_reused_studies(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    monkeypatch.setattr(b, "process_brain_study", _no_dicom)
    # without a rule the row-dropped CT studies come back: the first DICOM read is one of them
    with pytest.raises(_DicomRead) as e:
        _main(b, monkeypatch, capsys, t["base"])
    assert str(e.value) == "B_ct"
    monkeypatch.undo()
    b = _import(BUILD)
    o = _main(b, monkeypatch, capsys, t["base"])
    assert "reused 4" in o, o
    rows = _rows(t["out"])
    assert {f: r["modality"] for f, r in rows.items()} == {f: v[2] for f, v in STUDIES.items()}


def test_set_built_without_rule_by_this_version_reuses_its_recorded_mr_studies(fixture_set, monkeypatch, capsys):
    """A rule-less build of THIS version already recorded the modality of the series it decoded:
    MR studies are reused as verified, the CT ones are re-read and their files removed."""
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    _main(b, monkeypatch, capsys, t["base"])
    reads: list = []
    real = b.process_brain_study
    monkeypatch.setattr(b, "process_brain_study", lambda files: reads.append(Path(files[0]).parent.parent.name)
                        or real(files))
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert sorted(reads) == ["D_srct", "E1_ct"], reads
    assert "reused 4" in o and "modality CT (series)=2" in o and "re-verified" not in o, o
    _assert_only_mr_pngs(b, out, t["images"])
    assert set(_pngs(out)) == {"9000101", "9000103", "9000105_s2", "9000106"}


# ------------------------------------------------------------------ interrupted run

class _Crash(BaseException):
    """Not an Exception: the builder's per-study `except Exception` must not swallow it (a kill / Ctrl-C)."""


def _crash_after(b, n_ok: int, seen: list):
    from src.pipeline.brain_mri_preprocessor import process_brain_study as real

    def proc(files):
        seen.append(Path(files[0]).parent.parent.name)
        if len(seen) > n_ok:
            raise _Crash()
        return real(files)
    return proc


def test_interrupted_verification_run_never_leaves_unverified_files_reusable(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    _main(b, monkeypatch, capsys, t["base"])
    rows = _read_csv(out / "studies.csv")             # as an older version wrote it: nothing verified
    _write_csv(out / "studies.csv", [c for c in rows[0] if c != "modality"], rows)
    seen: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 2, seen))
    with pytest.raises(_Crash):
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    capsys.readouterr()
    rows = _rows(out)
    # written ahead of the loop: nothing is recorded as verified, so nothing can be reused as verified
    assert all(r["modality"] == "" for r in rows.values()), rows

    monkeypatch.undo()
    b = _import(BUILD)
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    _assert_only_mr_pngs(b, out, t["images"])
    assert "re-verified" in o and set(_pngs(out)) == {"9000101", "9000103", "9000105_s2", "9000106"}


def test_interrupted_fresh_rule_build_rereads_the_half_built_study(fixture_set, monkeypatch, capsys):
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    seen: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 1, seen))
    with pytest.raises(_Crash):
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    capsys.readouterr()
    assert seen[0] == "A_mr" and set(_pngs(out)) == {"9000101"}, "the first study was written before the crash"
    assert _rows(out)["A_mr"]["modality"] == "", "…but not yet recorded as verified"

    seen2: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 99, seen2))
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert "A_mr" in seen2, "its files were not verified: the next run must read it again"
    assert "built 4" in o and "reused 0" in o, o
    _assert_only_mr_pngs(b, out, t["images"])
    assert _rows(out)["A_mr"]["modality"] == "MR"


def test_checkpoint_lets_an_interrupted_rule_build_resume_without_rereading(fixture_set, monkeypatch, capsys):
    """Every CHECKPOINT_EVERY studies read from DICOM, studies.csv is rewritten with the modalities
    verified so far: a build interrupted after an hour resumes instead of re-reading every study."""
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    monkeypatch.setattr(b, "CHECKPOINT_EVERY", 1)
    seen: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 2, seen))
    with pytest.raises(_Crash):
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    capsys.readouterr()
    rows = _rows(out)
    done, crashed = seen[:2], seen[2]
    # the study before the checkpoint is recorded as verified; the one written after it is not yet
    assert rows[done[0]]["modality"] == "MR" and int(rows[done[0]]["n_slices"]) > 0, rows[done[0]]
    assert rows[crashed]["modality"] == "" and int(rows[crashed]["n_slices"]) == 0

    seen2: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 99, seen2))
    o = _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert done[0] not in seen2, "a checkpointed study must be reused without reading DICOM again"
    assert crashed in seen2
    assert "reused" in o
    _assert_only_mr_pngs(b, out, t["images"])
    assert all(r["modality"] == "MR" for r in _rows(out).values() if int(r["n_slices"]) > 0)


def test_checkpoint_never_records_an_unverified_study(fixture_set, monkeypatch, capsys):
    """A checkpoint written while unverified files of an OLDER set are still waiting must not mark
    them as verified (they are re-read and, when they are CT, removed)."""
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    _main(b, monkeypatch, capsys, t["base"])                       # no rule: CT studies get PNGs too
    rows = _read_csv(out / "studies.csv")                          # as an older version wrote it
    _write_csv(out / "studies.csv", [c for c in rows[0] if c != "modality"], rows)
    monkeypatch.setattr(b, "CHECKPOINT_EVERY", 1)
    seen: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 2, seen))
    with pytest.raises(_Crash):
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    capsys.readouterr()
    rows = _rows(out)
    for folder, r in rows.items():
        if folder not in seen[:2]:
            assert r["modality"] == "", f"{folder} was never read under the rule but is recorded as {r['modality']!r}"

    monkeypatch.undo()
    b = _import(BUILD)
    _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    _assert_only_mr_pngs(b, out, t["images"])


def test_crash_between_reconcile_and_first_studies_csv_write_refuses_the_next_run(tmp_path, monkeypatch, capsys):
    """reconcile_existing renames files to THIS run's prefixes before studies.csv records them. A run that
    dies in between (here: write_studies fails with ENOSPC) must not let the next run hand the renamed CT
    files to the MR study that owned the name before: the next run refuses, --fresh recovers MR-only."""
    import errno
    b = _import(BUILD)
    images = tmp_path / "images"
    _write_series(images / "A_mr" / "series1", "MR", "9000201", desc="T1 AX", seed=1)
    _write_series(images / "B_ct" / "series1", "CT", "9000201", desc="HEAD AX", seed=2)
    rows = [{"study_folder": "A_mr", "patient_id": "9000201", "modality": "MR", "is_brain": "yes",
             "report_matched": "yes", "label": "normal", "abnormal": "no", "label_source": "rules",
             "needs_review": "no"},
            {"study_folder": "B_ct", "patient_id": "9000201", "modality": "SR", "is_brain": "yes",
             "report_matched": "yes", "label": "tumor", "abnormal": "yes", "label_source": "rules",
             "needs_review": "no"}]
    matched = _write_csv(tmp_path / "matched.csv", MATCHED_COLS, rows)
    out = tmp_path / "set"
    base = ["--images", str(images), "--matched", str(matched), "--out", str(out), "--mode", "binary",
            "--abnormal-col", "abnormal", "--val-frac", "0"]
    _main(b, monkeypatch, capsys, base)                       # run 1, no rule: A -> '9000201', B -> '..._s2'
    assert set(_pngs(out)) == {"9000201", "9000201_s2"}

    # run 2, --modality MR, A dropped at row level: B is renumbered to '9000201' and its CT files are
    # renamed; then the first studies.csv write fails
    _write_csv(matched, MATCHED_COLS, [{**rows[0], "is_brain": "no"}, rows[1]])
    real_write = b.write_studies

    def enospc(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(b, "write_studies", enospc)
    with pytest.raises(OSError):
        _main(b, monkeypatch, capsys, [*base, "--modality", "MR"])
    capsys.readouterr()
    monkeypatch.setattr(b, "write_studies", real_write)
    assert _rows(out)["A_mr"]["modality"] == "MR", "studies.csv still holds run 1's map"
    assert set(_pngs(out)) == {"9000201"}, "…while the disk already carries B's CT files under A's old name"

    # run 3, A selected again: must refuse instead of reusing B's CT files as verified MR
    _write_csv(matched, MATCHED_COLS, rows)
    reads: list = []
    real_proc = b.process_brain_study
    monkeypatch.setattr(b, "process_brain_study", lambda files: reads.append(Path(files[0]).parent.parent.name)
                        or real_proc(files))
    with pytest.raises(SystemExit) as e:
        _main(b, monkeypatch, capsys, [*base, "--modality", "MR"])
    assert e.value.code == 1
    o = capsys.readouterr().out
    assert "interrupted while renaming files" in o and "--fresh" in o, o
    assert reads == [] and set(_pngs(out)) == {"9000201"}, "a refused run touches nothing"

    # --fresh recovers: only MR slices, A's own
    o = _main(b, monkeypatch, capsys, [*base, "--modality", "MR", "--fresh"])
    assert "built 1" in o and "modality CT (series)=1" in o, o
    assert set(_pngs(out)) == {"9000201"} and _rows(out)["A_mr"]["modality"] == "MR"
    assert _rows(out)["B_ct"]["n_slices"] == "0"
    assert len(list((out / "train" / "normal").glob("9000201_*.png"))) == 5
    assert not list(out.glob("*/abnormal/*.png"))
    assert not (out / b.RECONCILE_MARKER).exists()
    # and the run after that is a plain resume
    o = _main(b, monkeypatch, capsys, [*base, "--modality", "MR"])
    assert "reused 1" in o, o


def test_interrupted_build_loop_leaves_no_reconcile_marker(fixture_set, monkeypatch, capsys):
    """The marker covers only the reconcile window: a run interrupted later (in the build loop) resumes,
    whether its reconcile changed the disk or not."""
    b = _import(BUILD)
    t = fixture_set
    out = t["out"]
    seen: list = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 1, seen))
    with pytest.raises(_Crash):                       # fresh set: reconcile changes nothing
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    assert not (out / b.RECONCILE_MARKER).exists()
    monkeypatch.undo()
    b = _import(BUILD)
    _main(b, monkeypatch, capsys, t["base"])          # no rule: CT files too
    seen = []
    monkeypatch.setattr(b, "process_brain_study", _crash_after(b, 0, seen))
    with pytest.raises(_Crash):                       # reconcile removes the row-dropped CT files, then a crash
        _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    capsys.readouterr()
    assert seen and not (out / b.RECONCILE_MARKER).exists()
    monkeypatch.undo()
    b = _import(BUILD)
    _main(b, monkeypatch, capsys, [*t["base"], "--modality", "MR"])
    _assert_only_mr_pngs(b, out, t["images"])


# ------------------------------------------------------------------ match: --images optional with --index

def _run(script: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, str(script), *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == expect, f"exit {r.returncode} (wanted {expect})\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    return r


def test_match_with_index_needs_no_images_and_errors_without_either(tmp_path: Path):
    index = _write_csv(tmp_path / "index.csv", ["study_folder", "patient_id", "study_date", "modality", "body_part",
                                                "study_description", "error"],
                       [{"study_folder": "S1", "patient_id": "9000201", "study_date": "20310105", "modality": "MR",
                         "body_part": "BRAIN"}])
    labels = _write_csv(tmp_path / "labels.csv", ["patient_id", "modality", "body_part", "study_date", "label"],
                        [{"patient_id": "9000201", "modality": "MR", "body_part": "brain",
                          "study_date": "2031-01-05", "label": "normal"}])
    out = tmp_path / "matched.csv"
    _run(MATCH, "--index", str(index), "--labels", str(labels), "--out", str(out))
    rows = _read_csv(out)
    assert len(rows) == 1 and rows[0]["match_quality"] == "date" and rows[0]["label"] == "normal"
    r = _run(MATCH, "--labels", str(labels), "--out", str(tmp_path / "x.csv"), expect=2)
    assert "--images is required when no --index is given" in r.stderr
    # without --index the walk still works and still checks the folder
    r = _run(MATCH, "--images", str(tmp_path / "missing"), "--labels", str(labels), "--out", str(tmp_path / "y.csv"),
             expect=1)
    assert "image root not found" in r.stdout


# ------------------------------------------------------------------ index: study fields from an image header

def test_index_takes_study_fields_from_the_first_image_series(tmp_path: Path):
    ix = _import(INDEX)
    d = tmp_path / "STUDY1"
    _write_sr(d / "a_sr", "9000301")                                      # sorts first
    _write_series(d / "b_mr", "MR", "9000301", n=3, desc="T2 AX", body_part="BRAIN")
    row = ix.index_study(d)
    assert row["modality"] == "MR" and row["body_part"] == "BRAIN", row
    assert row["patient_id"] == "9000301" and row["study_date"] == "20310105"
    assert (row["n_series"], row["n_dcm"], row["n_bmp"]) == (2, 4, 0)
    assert row["series_descriptions"] == "Dose Report | T2 AX" and row["error"] == ""

    sr_only = tmp_path / "STUDY2"
    _write_sr(sr_only / "a_sr", "9000302")
    row = ix.index_study(sr_only)
    # no image object at all: described by the report object as before, but flagged so that a resume
    # does not read it again and the matcher / builder leave it out
    assert row["modality"] == "SR" and row["body_part"] == "" and row["n_series"] == 1
    assert row["error"] == "no image object" and not ix.is_stale_row(row)


def test_index_resume_reindexes_rows_described_by_a_non_image_object(tmp_path: Path, monkeypatch, capsys):
    """An index written by the older version says 'SR' for an MR study whose report series sorts first.
    A resumed run must heal exactly those rows and leave every other row (and error rows) alone."""
    ix = _import(INDEX)
    root = tmp_path / "drive"
    _write_sr(root / "S_srmr" / "a_sr", "9000401")
    _write_series(root / "S_srmr" / "b_mr", "MR", "9000401", n=3, desc="T2 AX", body_part="BRAIN")
    _write_series(root / "S_mr" / "a_mr", "MR", "9000402", n=3, desc="T1 AX", body_part="BRAIN")
    _write_sr(root / "S_sronly" / "a_sr", "9000403")
    (root / "S_empty").mkdir()
    out = tmp_path / "index.csv"
    marker = "UNTOUCHED"
    old = [{"study_folder": "S_srmr", "patient_id": "9000401", "modality": "SR", "body_part": ""},
           {"study_folder": "S_mr", "patient_id": "9000402", "modality": "MR", "body_part": "BRAIN",
            "study_description": marker},
           {"study_folder": "S_sronly", "patient_id": "9000403", "modality": "SR"},
           {"study_folder": "S_empty", "modality": "", "error": "no dcm", "study_description": marker}]
    _write_csv(out, ix.FIELDS, old)
    seen: list[str] = []
    real = ix.index_study
    monkeypatch.setattr(ix, "index_study", lambda d: (seen.append(d.name), real(d))[1])
    monkeypatch.setattr(ix.sys, "argv", ["index_new_studies.py", "--root", str(root), "--out", str(out)])
    ix.main()
    printed = capsys.readouterr().out
    assert "re-indexing 2 studies" in printed
    assert sorted(seen) == ["S_srmr", "S_sronly"], "only the rows described by a non-image object are read again"
    rows = {r["study_folder"]: r for r in _read_csv(out)}
    assert len(rows) == 4 == len(_read_csv(out))
    assert rows["S_srmr"]["modality"] == "MR" and rows["S_srmr"]["body_part"] == "BRAIN"
    assert rows["S_sronly"]["modality"] == "SR"                       # really no image: same answer …
    assert rows["S_sronly"]["error"] == "no image object"             # … now recorded as such
    assert rows["S_mr"]["study_description"] == marker and rows["S_empty"]["study_description"] == marker
    assert rows["S_empty"]["error"] == "no dcm"

    # a second run reads nothing: the study without an image object carries an error now
    seen.clear()
    ix.main()
    assert seen == []
    assert {r["study_folder"]: r["modality"] for r in _read_csv(out)} == {
        "S_srmr": "MR", "S_mr": "MR", "S_sronly": "SR", "S_empty": ""}


def _walk_by_name(real_walk):
    """os.walk with files in name order (exFAT / FAT list them in creation order; tests pin the order)."""
    def walk(top, *a, **k):
        for root, dirs, files in real_walk(top, *a, **k):
            dirs.sort()
            yield root, dirs, sorted(files)
    return walk


def test_index_flat_study_with_the_report_object_first_is_described_by_an_image(tmp_path: Path, monkeypatch):
    """Series exported flat (report objects next to the images in ONE folder, the report listed first):
    the study fields must still come from an image header, not from the report."""
    ix = _import(INDEX)
    monkeypatch.setattr(ix.os, "walk", _walk_by_name(ix.os.walk))
    d = tmp_path / "FLAT"
    _write_sr(d, "9000501", name="000_sr.dcm")
    _write_sr(d, "9000501", name="001_sr.dcm")
    _write_series(d, "MR", "9000501", n=8, desc="T2 AX", body_part="BRAIN")
    reads: list[str] = []
    real = ix.pydicom.dcmread
    monkeypatch.setattr(ix.pydicom, "dcmread", lambda fp, *a, **k: (reads.append(Path(fp).name), real(fp, *a, **k))[1])
    row = ix.index_study(d)
    assert row["modality"] == "MR" and row["body_part"] == "BRAIN" and row["error"] == "", row
    assert (row["n_series"], row["n_dcm"]) == (1, 10)
    assert row["series_descriptions"] == "Dose Report", "one description per directory: its first file's"
    assert reads == ["000_sr.dcm", "001_sr.dcm", "img_000.dcm"], "stops at the first image header"
    assert not ix.is_stale_row(row)


def test_index_header_scan_per_directory_is_capped(tmp_path: Path, monkeypatch):
    ix = _import(INDEX)
    monkeypatch.setattr(ix.os, "walk", _walk_by_name(ix.os.walk))
    monkeypatch.setattr(ix, "HEADERS_PER_DIR", 3)
    d = tmp_path / "MANY_SR"
    for i in range(6):
        _write_sr(d, "9000502", name=f"{i:03d}_sr.dcm")
    _write_series(d, "MR", "9000502", n=2, desc="T2 AX", body_part="BRAIN", tag="z_img")
    reads: list[str] = []
    real = ix.pydicom.dcmread
    monkeypatch.setattr(ix.pydicom, "dcmread", lambda fp, *a, **k: (reads.append(Path(fp).name), real(fp, *a, **k))[1])
    row = ix.index_study(d)
    assert len(reads) == 3, reads
    assert row["n_dcm"] == 8 and row["modality"] == "SR"
    assert row["error"] and row["error"] != "no image object" and "3" in row["error"], row["error"]
    assert not ix.is_stale_row(row), "not read again on every resume"
    monkeypatch.setattr(ix, "HEADERS_PER_DIR", 50)
    reads.clear()
    row = ix.index_study(d)
    assert row["modality"] == "MR" and row["error"] == "" and len(reads) == 7


def test_index_report_only_flat_study_is_not_reindexed_on_resume(tmp_path: Path, monkeypatch, capsys):
    ix = _import(INDEX)
    monkeypatch.setattr(ix.os, "walk", _walk_by_name(ix.os.walk))
    root = tmp_path / "drive"
    _write_sr(root / "S_rep", "9000503", name="000_sr.dcm")
    _write_sr(root / "S_rep", "9000503", name="001_sr.dcm")
    _write_sr(root / "S_flat", "9000504", name="000_sr.dcm")
    _write_series(root / "S_flat", "MR", "9000504", n=3, desc="T1 AX", body_part="BRAIN")
    out = tmp_path / "index.csv"
    seen: list[str] = []
    real = ix.index_study
    monkeypatch.setattr(ix, "index_study", lambda d: (seen.append(d.name), real(d))[1])
    monkeypatch.setattr(ix.sys, "argv", ["index_new_studies.py", "--root", str(root), "--out", str(out)])
    ix.main()
    rows = {r["study_folder"]: r for r in _read_csv(out)}
    assert rows["S_flat"]["modality"] == "MR" and rows["S_flat"]["error"] == ""
    assert rows["S_rep"]["error"] == "no image object"
    seen.clear()
    ix.main()
    assert seen == [] and "re-indexing" not in capsys.readouterr().out
