"""Release pipeline stages 3-5: match_images_to_reports.py, build_training_set.py,
combine_training_sets.py (docs/dev/release_pipeline_contract.md).

Everything here is synthetic: index/labels CSVs written into tmp dirs, tiny PNGs named
<pid>_<k>.png in class folders emulating two batches with an overlapping patient, a holdout
file. No hospital PDFs, DICOMs or patient names are touched. The one build-set test that
needs a real brain study uses tests/conftest.py::STUDY_DIR with a SYNTHETIC patient id and
skips when the study folder is absent.
"""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO, STUDY_DIR

SCRIPTS = REPO / "scripts"
MATCH = SCRIPTS / "match_images_to_reports.py"
BUILD = SCRIPTS / "build_training_set.py"
COMBINE = SCRIPTS / "combine_training_sets.py"


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(script: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, str(script), *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == expect, f"exit {r.returncode} (wanted {expect})\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    return r


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


# ----------------------------------------------------------------- stage 3: match

INDEX_COLS = ["study_folder", "patient_id", "study_date", "modality", "body_part", "study_description", "error"]
LABEL_COLS = ["patient_id", "modality", "body_part", "study_date", "label", "conclusion",
              "abnormal", "label_confidence", "label_source", "needs_review"]


def _index_rows() -> list[dict]:
    return [
        # patient 1001: two reports (brain MR 07-01, spine MR 07-02); this study is the brain one by DATE
        {"study_folder": "S1", "patient_id": "1001", "study_date": "20260701", "modality": "MR", "body_part": "BRAIN"},
        # patient 1001 again, a different date -> no date match -> modality+body_part picks the brain MR report
        {"study_folder": "S2", "patient_id": "1001", "study_date": "20260715", "modality": "MR", "body_part": "HEAD"},
        # patient 1002: one CT brain report, DICOM is MR brain, dates differ -> body part only -> 'body_part'
        {"study_folder": "S3", "patient_id": "1002", "study_date": "20260801", "modality": "MR", "body_part": "BRAIN"},
        # patient 1003: CT brain + MR brain reports, no date match, DICOM CT -> modality tie-break -> CT report
        {"study_folder": "S4", "patient_id": "1003", "study_date": "20260820", "modality": "CT", "body_part": "HEAD",
         "study_description": "Head CT"},
        # patient 1004: no report at all
        {"study_folder": "S5", "patient_id": "1004", "study_date": "20260901", "modality": "MR", "body_part": "BRAIN"},
        # non-brain study of patient 1001 -> the spine report by body part
        {"study_folder": "S6", "patient_id": "1001", "study_date": "20260720", "modality": "MR", "body_part": "LSPINE"},
        # indexing error rows are skipped
        {"study_folder": "S7", "patient_id": "1005", "study_date": "20260901", "modality": "MR", "body_part": "BRAIN",
         "error": "InvalidDicomError: boom"},
    ]


def _label_rows() -> list[dict]:
    return [
        {"patient_id": "1001", "modality": "MR", "body_part": "brain", "study_date": "2026-07-01", "label": "tumor",
         "abnormal": "yes", "label_confidence": "high", "label_source": "rules", "needs_review": "no"},
        {"patient_id": "1001", "modality": "MR", "body_part": "spine", "study_date": "2026-07-02", "label": "",
         "abnormal": "", "label_confidence": "", "label_source": "rules", "needs_review": "no"},
        {"patient_id": "1002", "modality": "CT", "body_part": "brain", "study_date": "2026-06-10", "label": "normal",
         "abnormal": "no", "label_confidence": "medium", "label_source": "llm", "needs_review": "yes"},
        {"patient_id": "1003", "modality": "CT", "body_part": "brain", "study_date": "2026-05-05", "label": "stroke",
         "abnormal": "yes", "label_confidence": "high", "label_source": "rules", "needs_review": "no"},
        {"patient_id": "1003", "modality": "MR", "body_part": "brain", "study_date": "2026-05-06", "label": "normal",
         "abnormal": "no", "label_confidence": "high", "label_source": "rules", "needs_review": "no"},
    ]


@pytest.fixture
def match_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    index = _write_csv(tmp_path / "index.csv", INDEX_COLS, _index_rows())
    labels = _write_csv(tmp_path / "labels.csv", LABEL_COLS, _label_rows())
    images = tmp_path / "images"
    images.mkdir()
    return index, labels, images


def test_match_helpers_normalise_modality_and_date():
    m = _import(MATCH)
    assert m.norm_modality("МСКТ") == "CT" and m.norm_modality("ct") == "CT" and m.norm_modality("МРТ") == "MR"
    assert m.norm_modality("MR") == "MR" and m.norm_modality("?") == "" and m.norm_modality("") == ""
    assert m.norm_date("2026-07-21") == "20260721" == m.norm_date("20260721") == m.norm_date("21.07.2026")
    assert m.norm_date("") == "" and m.norm_date("n/a") == ""


def test_match_join_order_and_carry_through(match_inputs):
    index, labels, images = match_inputs
    out = index.parent / "matched.csv"
    r = _run(MATCH, "--images", str(images), "--index", str(index), "--labels", str(labels), "--out", str(out))
    rows = {row["study_folder"]: row for row in _read_csv(out)}
    assert set(rows) == {"S1", "S2", "S3", "S4", "S5", "S6"}, "error rows of the index must be skipped"

    s1 = rows["S1"]
    assert s1["match_quality"] == "date" and s1["label"] == "tumor" and s1["is_brain"] == "yes"
    assert (s1["abnormal"], s1["label_source"], s1["label_confidence"], s1["needs_review"]) == ("yes", "rules", "high", "no")
    assert s1["report_study_date"] == "2026-07-01"

    s2 = rows["S2"]
    assert s2["match_quality"] == "modality+body_part" and s2["label"] == "tumor" and s2["report_body_part"] == "brain"

    s3 = rows["S3"]
    assert s3["match_quality"] == "body_part" and s3["label"] == "normal" and s3["needs_review"] == "yes"
    assert s3["label_source"] == "llm"

    s4 = rows["S4"]
    assert s4["match_quality"] == "modality+body_part" and s4["label"] == "stroke", "CT study must take the CT report"
    assert s4["report_modality"] == "CT"

    s5 = rows["S5"]
    assert s5["report_matched"] == "no" and s5["label"] == "" and s5["abnormal"] == "" and s5["match_quality"] == ""

    s6 = rows["S6"]
    assert s6["is_brain"] == "no" and s6["report_body_part"] == "spine" and s6["match_quality"] == "modality+body_part"

    # existing columns unchanged and in the old order
    head = list(rows["S1"].keys())
    assert head[:10] == ["study_folder", "patient_id", "modality", "study_date", "match_quality", "body_part_dicom",
                         "is_brain", "report_matched", "report_body_part", "label"]
    assert "by date: 1" in r.stdout and "by weaker keys: 3" in r.stdout, r.stdout


def test_match_accepts_old_labels_without_abnormal_column(match_inputs):
    index, _, images = match_inputs
    old_cols = ["patient_id", "modality", "body_part", "study_date", "label", "conclusion"]
    old_rows = [{"patient_id": "1001", "modality": "МСКТ", "body_part": "brain", "study_date": "2026-07-01",
                 "label": "normal", "conclusion": "x"},
                {"patient_id": "1002", "modality": "МСКТ", "body_part": "brain", "study_date": "", "label": "unspecified",
                 "conclusion": "x"}]
    labels = _write_csv(index.parent / "old_labels.csv", old_cols, old_rows)
    out = index.parent / "matched_old.csv"
    r = _run(MATCH, "--images", str(images), "--index", str(index), "--labels", str(labels), "--out", str(out))
    rows = {row["study_folder"]: row for row in _read_csv(out)}
    assert rows["S1"]["label"] == "normal" and rows["S1"]["match_quality"] == "date"
    assert rows["S1"]["abnormal"] == "" and rows["S1"]["needs_review"] == "" and rows["S1"]["label_source"] == ""
    # 1002: old report says МСКТ (CT) brain, the DICOM is MR brain, no date -> body part -> 'body_part'
    assert rows["S3"]["match_quality"] == "body_part" and rows["S3"]["label"] == "unspecified"
    assert "no abnormal column" in r.stdout


def test_match_pick_report_prefers_compatible_row_on_same_date():
    m = _import(MATCH)
    cands = [{"patient_id": "1", "modality": "MR", "body_part": "spine", "study_date": "2026-01-01", "label": ""},
             {"patient_id": "1", "modality": "MR", "body_part": "brain", "study_date": "2026-01-01", "label": "cyst"}]
    rep, q = m.pick_report(cands, "20260101", "MR", brainy=True)
    assert q == "date" and rep["label"] == "cyst"
    rep, q = m.pick_report(cands, "20260202", "CT", brainy=True)
    assert q == "body_part" and rep["label"] == "cyst", "modality incompatible, body part decides"
    rep, q = m.pick_report(cands, "", "MR", brainy=False)
    assert q == "modality+body_part" and rep["body_part"] == "spine"
    assert m.pick_report([], "20260101", "MR", True) == (None, "")
    # body part unknown on the report side: only the modality can decide, else the only report -> 'pid'
    bare = [{"patient_id": "1", "modality": "", "body_part": "", "study_date": "", "label": "normal"}]
    assert m.pick_report(bare, "20260101", "MR", True)[1] == "pid"
    assert m.pick_report([{**bare[0], "modality": "МРТ"}], "", "MR", True)[1] == "modality"
    assert m.pick_report(bare * 2, "", "MR", True)[1] == "pid-ambiguous"


# -------------------------------------------------------------- stage 4: build-set

def test_build_set_class_rules():
    b = _import(BUILD)
    assert b.binary_class("tumor", "") == "abnormal" and b.binary_class("normal", "") == "normal"
    assert b.binary_class("", "") is None and b.binary_class("unspecified", "") is None
    assert b.binary_class("tumor", "no") == "normal", "abnormal column wins over label"
    assert b.binary_class("", "yes") == "abnormal"
    assert b.multiclass_class("other") is None and b.multiclass_class("stroke") == "stroke"
    assert b.patient_split(["a", "b", "c", "d", "e"], 0.2) == {"a"}
    assert b.patient_split(["a"], 0.2) == set()


def test_build_set_select_studies_filters_and_prefixes():
    b = _import(BUILD)
    rows = [
        {"study_folder": "A1", "patient_id": "1", "is_brain": "yes", "report_matched": "yes", "label": "tumor",
         "abnormal": "yes", "needs_review": "no", "label_source": "rules"},
        {"study_folder": "A2", "patient_id": "1", "is_brain": "yes", "report_matched": "yes", "label": "normal",
         "abnormal": "no", "needs_review": "no", "label_source": "rules"},
        {"study_folder": "B1", "patient_id": "2", "is_brain": "yes", "report_matched": "yes", "label": "atrophy",
         "abnormal": "yes", "needs_review": "yes", "label_source": "llm"},
        {"study_folder": "C1", "patient_id": "3", "is_brain": "no", "report_matched": "yes", "label": "normal",
         "abnormal": "no", "needs_review": "no", "label_source": "rules"},
        {"study_folder": "D1", "patient_id": "4", "is_brain": "yes", "report_matched": "no", "label": "",
         "abnormal": "", "needs_review": "", "label_source": ""},
        {"study_folder": "E1", "patient_id": "5", "is_brain": "yes", "report_matched": "yes", "label": "",
         "abnormal": "", "needs_review": "no", "label_source": "rules"},
    ]
    studies, dropped = b.select_studies(rows, "binary", "abnormal", True, None)
    assert [s["study_folder"] for s in studies] == ["A1", "A2"]
    assert dropped["needs_review"] == 1 and dropped["not brain"] == 1 and dropped["no report"] == 1
    assert dropped["unlabelled/dropped class"] == 1
    b.assign_prefixes(studies)
    assert [s["prefix"] for s in studies] == ["1", "1_s2"], "second study of a patient must not overwrite the first"

    studies, dropped = b.select_studies(rows, "binary", "abnormal", False, {"A1": "", "B1": "Err"})
    assert [s["study_folder"] for s in studies] == ["A1"]
    assert dropped["index error"] == 1 and dropped["not in index"] >= 1


@pytest.mark.skipif(not STUDY_DIR.is_dir(), reason=f"brain MR study not available at {STUDY_DIR}")
def test_build_set_real_study_writes_pngs_and_studies_csv(tmp_path: Path):
    pid = "9900001"  # synthetic id — the matched CSV, not the DICOM header, decides the patient id
    matched = _write_csv(tmp_path / "matched.csv",
                         ["study_folder", "patient_id", "modality", "study_date", "match_quality", "body_part_dicom",
                          "is_brain", "report_matched", "report_body_part", "label", "abnormal", "label_source",
                          "label_confidence", "needs_review"],
                         [{"study_folder": STUDY_DIR.name, "patient_id": pid, "modality": "MR", "is_brain": "yes",
                           "report_matched": "yes", "report_body_part": "brain", "label": "tumor", "abnormal": "yes",
                           "label_source": "import", "label_confidence": "high", "needs_review": "no"}])
    out = tmp_path / "set"
    args = ["--images", str(STUDY_DIR.parent), "--matched", str(matched), "--out", str(out), "--mode", "binary",
            "--abnormal-col", "abnormal", "--skip-needs-review", "--limit", "1"]
    r = _run(BUILD, *args)
    pngs = sorted(out.glob("*/abnormal/*.png"))
    assert pngs and all(p.stem.startswith(f"{pid}_") for p in pngs), [p.name for p in pngs]
    assert 1 <= len(pngs) <= 5
    rows = _read_csv(out / "studies.csv")
    assert list(rows[0].keys()) == ["patient_id", "study_folder", "cls", "split", "n_slices", "label", "label_source"]
    assert rows[0]["patient_id"] == pid and rows[0]["cls"] == "abnormal" and int(rows[0]["n_slices"]) == len(pngs)
    assert rows[0]["label"] == "tumor" and rows[0]["label_source"] == "import" and rows[0]["split"] in ("train", "val")
    assert "built 1" in r.stdout
    # resumable: second run reuses the PNGs instead of re-processing the study
    mtimes = {p: p.stat().st_mtime_ns for p in pngs}
    r2 = _run(BUILD, *args)
    assert "reused 1" in r2.stdout and "built 0" in r2.stdout
    assert {p: p.stat().st_mtime_ns for p in pngs} == mtimes
    # --fresh rebuilds
    r3 = _run(BUILD, *args, "--fresh")
    assert "built 1" in r3.stdout


MATCHED_COLS = ["study_folder", "patient_id", "modality", "study_date", "match_quality", "body_part_dicom",
                "is_brain", "report_matched", "report_body_part", "label", "abnormal", "label_source",
                "label_confidence", "needs_review"]


def _matched_row(folder: str, pid: str, abnormal: str, label: str = "", needs_review: str = "no") -> dict:
    return {"study_folder": folder, "patient_id": pid, "modality": "MR", "study_date": "20310105", "is_brain": "yes",
            "report_matched": "yes", "report_body_part": "brain", "label": label or ("tumor" if abnormal == "yes" else "normal"),
            "abnormal": abnormal, "label_source": "rules", "label_confidence": "high", "needs_review": needs_review}


class _FakeStudies:
    """Stands in for DICOM decoding: every study folder yields 3 slices whose grey value encodes the folder,
    so a test can tell which study a PNG came from. Counts how many studies were really processed."""

    def __init__(self, values: dict[str, int]):
        self.values = values
        self.processed: list[str] = []

    def process(self, files):
        folder = Path(files[0]).parent.name
        self.processed.append(folder)
        return folder

    def slices(self, study, k=5):
        import numpy as np
        return [np.full((8, 8), self.values[study] / 255.0, dtype=np.float32) for _ in range(3)]


def _build_in_process(b, monkeypatch, fake: _FakeStudies, argv: list[str], capsys) -> str:
    monkeypatch.setattr(b, "process_brain_study", fake.process)
    monkeypatch.setattr(b, "central_slices", fake.slices)
    monkeypatch.setattr(b.sys, "argv", ["build_training_set.py", *argv])
    b.main()
    return capsys.readouterr().out


def _grey(png: Path) -> int:
    from PIL import Image
    return Image.open(png).convert("L").getpixel((0, 0))


def _set_tree(out: Path) -> dict[str, tuple[str, str]]:
    """'<prefix>_<k>.png' tree -> {prefix: (split, cls)} (asserts one location per prefix)."""
    loc: dict[str, set] = {}
    for fp in out.glob("*/*/*.png"):
        loc.setdefault(fp.stem.rsplit("_", 1)[0], set()).add((fp.parent.parent.name, fp.parent.name))
    assert all(len(v) == 1 for v in loc.values()), loc
    return {k: next(iter(v)) for k, v in loc.items()}


def test_build_set_resume_moves_relabelled_study_and_updates_studies_csv(tmp_path: Path, monkeypatch, capsys):
    """build -> flip `abnormal` in matched.csv -> build again (no --fresh): the PNGs move to the new class
    folder (same split, not re-decoded), studies.csv carries the new class, 'relabelled 1' is printed."""
    b = _import(BUILD)
    images = tmp_path / "images"
    folders = {"F1": ("9000001", "yes"), "F2": ("9000002", "no"), "F3": ("9000003", "yes"),
               "F4": ("9000004", "no"), "F5": ("9000005", "yes")}
    for f in folders:
        (images / f).mkdir(parents=True)
        (images / f / "x.dcm").write_bytes(b"synthetic")
    fake = _FakeStudies({f: 10 * (i + 1) for i, f in enumerate(folders)})
    matched = tmp_path / "matched.csv"
    _write_csv(matched, MATCHED_COLS, [_matched_row(f, pid, ab) for f, (pid, ab) in folders.items()])
    out = tmp_path / "set"
    argv = ["--images", str(images), "--matched", str(matched), "--out", str(out), "--mode", "binary",
            "--abnormal-col", "abnormal", "--skip-needs-review"]

    out1 = _build_in_process(b, monkeypatch, fake, argv, capsys)
    assert "built 5" in out1 and "relabelled 0" in out1
    before = _set_tree(out)
    split1, cls1 = before["9000001"]
    assert cls1 == "abnormal"
    grey1 = _grey(out / split1 / cls1 / "9000001_0.png")

    # the LLM relabel flips 9000001 abnormal -> normal
    _write_csv(matched, MATCHED_COLS, [_matched_row(f, pid, "no" if pid == "9000001" else ab)
                                       for f, (pid, ab) in folders.items()])
    fake.processed.clear()
    out2 = _build_in_process(b, monkeypatch, fake, argv, capsys)
    assert "relabelled 1" in out2 and "reused 5" in out2 and "built 0" in out2, out2
    assert fake.processed == [], "a relabel must move the files, not re-decode the study"
    after = _set_tree(out)
    assert after["9000001"] == (split1, "normal"), "same split, new class folder"
    assert not list((out / split1 / "abnormal").glob("9000001_*.png")), "nothing may stay under the old class"
    assert len(list((out / split1 / "normal").glob("9000001_*.png"))) == 3
    assert _grey(out / split1 / "normal" / "9000001_0.png") == grey1
    assert {k: v for k, v in after.items() if k != "9000001"} == {k: v for k, v in before.items() if k != "9000001"}
    rows = {r["patient_id"]: r for r in _read_csv(out / "studies.csv")}
    assert rows["9000001"]["cls"] == "normal" and rows["9000001"]["split"] == split1
    assert rows["9000001"]["n_slices"] == "3"
    assert not (out / b.STAGING_DIR).exists()

    # a third run with nothing changed is a pure reuse
    out3 = _build_in_process(b, monkeypatch, fake, argv, capsys)
    assert "relabelled 0" in out3 and "reused 5" in out3


def test_build_set_resume_follows_study_folder_when_prefixes_shift_and_drops_stale(tmp_path: Path, monkeypatch, capsys):
    """A new study sorting BEFORE an existing one shifts '<pid>' / '<pid>_s2': each study must keep ITS images.
    A study that is no longer selected (needs_review now) loses its PNGs on a full run."""
    b = _import(BUILD)
    images = tmp_path / "images"
    for f in ("B", "C", "Z", "A"):
        (images / f).mkdir(parents=True)
        (images / f / "x.dcm").write_bytes(b"synthetic")
    fake = _FakeStudies({"A": 40, "B": 80, "C": 120, "Z": 200})
    matched = tmp_path / "matched.csv"
    out = tmp_path / "set"
    argv = ["--images", str(images), "--matched", str(matched), "--out", str(out), "--mode", "binary",
            "--abnormal-col", "abnormal", "--skip-needs-review"]
    _write_csv(matched, MATCHED_COLS, [_matched_row("B", "9000010", "yes"), _matched_row("C", "9000010", "no"),
                                       _matched_row("Z", "9000011", "no")])
    _build_in_process(b, monkeypatch, fake, argv, capsys)
    t1 = _set_tree(out)
    assert set(t1) == {"9000010", "9000010_s2", "9000011"}
    # run 2: study A (sorts first) appears for 9000010, study C flips to abnormal, 9000011 becomes needs_review
    _write_csv(matched, MATCHED_COLS, [_matched_row("A", "9000010", "no"), _matched_row("B", "9000010", "yes"),
                                       _matched_row("C", "9000010", "yes"),
                                       _matched_row("Z", "9000011", "no", needs_review="yes")])
    fake.processed.clear()
    o = _build_in_process(b, monkeypatch, fake, argv, capsys)
    assert fake.processed == ["A"], "only the new study is decoded"
    assert "relabelled 1" in o and "renumbered 2" in o and "removed the PNGs of 1" in o, o
    t2 = _set_tree(out)
    assert set(t2) == {"9000010", "9000010_s2", "9000010_s3"}, "stale 9000011 files must be gone"
    greys = {p: _grey(next(out.glob(f"*/*/{p}_0.png"))) for p in t2}
    assert greys == {"9000010": 40, "9000010_s2": 80, "9000010_s3": 120}, "each prefix must hold its own study"
    assert t2["9000010_s3"][1] == "abnormal" and t2["9000010"][1] == "normal"
    assert len({sp for sp, _ in t2.values()}) == 1, "one patient, one split"
    rows = _read_csv(out / "studies.csv")
    assert [(r["study_folder"], r["cls"]) for r in rows] == [("A", "normal"), ("B", "abnormal"), ("C", "abnormal")]


def test_build_set_refuses_interrupted_relabel(tmp_path: Path, monkeypatch, capsys):
    b = _import(BUILD)
    images = tmp_path / "images"; (images / "F").mkdir(parents=True); (images / "F" / "x.dcm").write_bytes(b"s")
    matched = _write_csv(tmp_path / "m.csv", MATCHED_COLS, [_matched_row("F", "9000020", "yes")])
    out = tmp_path / "set"; (out / b.STAGING_DIR).mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        _build_in_process(b, monkeypatch, _FakeStudies({"F": 1}),
                          ["--images", str(images), "--matched", str(matched), "--out", str(out), "--mode", "binary"],
                          capsys)
    assert e.value.code == 1


# ---------------------------------------------------------------- stage 5: combine

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8cfc0f01f00050001"
    "0234b1e3a30000000049454e44ae426082")
STUDY_COLS = ["patient_id", "study_folder", "cls", "split", "n_slices", "label", "label_source"]


def _make_batch(root: Path, name: str, layout: dict[str, dict[str, list[str]]], k: int = 2,
                studies_csv: bool = True, content: str | None = None) -> Path:
    """layout: {split: {cls: [pid, …]}} -> <root>/<name>/<split>/<cls>/<pid>_<j>.png (+ studies.csv).
    Slice bytes depend on (content or name, pid, j): two batches with the same `content` tag hold
    byte-identical slices; without studies_csv the set has the LEGACY layout (data/local_finetune_bin)."""
    base = root / name
    tag = (content or name).encode()
    studies = []
    for split, classes in layout.items():
        for cls, pids in classes.items():
            d = base / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for pid in pids:
                for j in range(k):
                    (d / f"{pid}_{j}.png").write_bytes(PNG_1x1 + tag + pid.encode() + bytes([j]))
                studies.append({"patient_id": pid, "study_folder": f"F{pid}", "cls": cls, "split": split,
                                "n_slices": k, "label": cls, "label_source": "rules"})
    if studies_csv:
        _write_csv(base / "studies.csv", STUDY_COLS, studies)
    return base


def P(n: int) -> str:
    """Synthetic patient id (>= 9000000, never a real one)."""
    return str(9000000 + n)


@pytest.fixture
def two_batches(tmp_path: Path) -> dict:
    a = _make_batch(tmp_path, "batch_a", {
        "train": {"abnormal": [P(101), P(102), P(103), P(104)], "normal": [P(201), P(202), P(203)]},
        "val": {"abnormal": [P(105), P(106)], "normal": [P(204)]},
    })
    b = _make_batch(tmp_path, "batch_b", {
        "train": {"abnormal": [P(301), P(302), P(303), P(105)], "normal": [P(401), P(402), P(403), P(404)]},  # 105 overlaps
        "val": {"abnormal": [P(304)], "normal": [P(405)]},
    })
    holdout = tmp_path / "holdout.txt"
    holdout.write_text(f"{P(102)}\n{P(301)}\n{P(404)}\n{P(999999)}\n", encoding="utf-8")  # the last is in no batch
    return {"a": a, "b": b, "holdout": holdout, "out": tmp_path / "release"}


def _pids(split_dir: Path) -> set[str]:
    return {p.stem.split("_", 1)[0] for p in split_dir.glob("*/*.png")}


def _classes_of(out: Path) -> dict[str, set[str]]:
    cls: dict[str, set[str]] = {}
    for fp in out.glob("*/*/*.png"):
        cls.setdefault(fp.stem.split("_", 1)[0], set()).add(fp.parent.name)
    return cls


def _manifest(out: Path) -> dict:
    return json.loads((out / "dataset_manifest.json").read_text())


ALL_AB = {P(i) for i in (101, 102, 103, 104, 105, 106, 201, 202, 203, 204,
                          301, 302, 303, 304, 401, 402, 403, 404, 405)}


def test_combine_patient_level_split_holdout_and_manifest(two_batches):
    t = two_batches
    r = _run(COMBINE, "--sets", str(t["a"]), str(t["b"]), "--holdout", str(t["holdout"]), "--out", str(t["out"]),
             "--val-frac", "0.25", "--tune-frac", "0", "--seed", "7")
    out = t["out"]
    train, val = _pids(out / "train"), _pids(out / "val")
    assert not (train & val), "patient leakage between train and val"
    assert train | val == ALL_AB and not (out / "tune").exists()
    assert {P(102), P(301), P(404)} <= val, "holdout ids must be forced into val"
    assert P(999999) in r.stdout and "WARNING" in r.stdout, "holdout id absent from every batch must be listed"

    # symlinks resolve to the batch files
    links = list((out / "val").glob("*/*.png")) + list((out / "train").glob("*/*.png"))
    assert links and all(p.is_symlink() and p.resolve().is_file() for p in links)
    assert all(os.path.isabs(os.readlink(p)) for p in links)

    m = _manifest(out)
    # patient 105 has DIFFERENT slices in both batches: in train both are kept (the second batch's renamed),
    # in val (the gate set) only the winning (last) batch's export
    in_val_105 = P(105) in val
    n_files = 2 * 19 + (0 if in_val_105 else 2)
    assert m["n_files"] == n_files and m["n_patients"] == 19 and m["n_studies"] == 19 and m["seed"] == 7
    assert m["n_patients_in_several_batches"] == 1 and m["duplicate_slices_skipped"] == 0
    split_105 = "val" if in_val_105 else "train"
    names_105 = sorted(p.name for p in (out / split_105 / "abnormal").glob(f"{P(105)}_*.png"))
    if in_val_105:
        assert names_105 == [f"{P(105)}_0.png", f"{P(105)}_1.png"]
        assert all("batch_b" in str(p.resolve()) for p in (out / "val" / "abnormal").glob(f"{P(105)}_*.png"))
        assert m["val_slices_from_non_winning_batch_skipped"] == 2 and m["slices_renamed_on_collision"] == 0
    else:
        assert names_105 == [f"{P(105)}_0.png", f"{P(105)}_1.png", f"{P(105)}_batch_b_0.png", f"{P(105)}_batch_b_1.png"]
        assert m["slices_renamed_on_collision"] == 2
    assert m["holdout"]["missing_from_all_batches"] == [P(999999)] and m["holdout"]["n_in_val"] == 3
    assert m["holdout"]["sha256"] and len(m["holdout"]["sha256"]) == 64
    assert {s["name"]: s["n_files"] for s in m["sets"]} == {"batch_a": 20, "batch_b": 20}
    assert {s["name"]: s["kind"] for s in m["sets"]} == {"batch_a": "build set", "batch_b": "build set"}
    assert m["train_lock"]["n_locked"] == 0, "a build set's own split is advisory: it locks nothing"
    assert m["splits"]["train"]["n_patients"] == len(train) and m["splits"]["val"]["n_patients"] == len(val)
    assert m["splits"]["val"]["n_study_keys"] == len(val), "eval_study_level must see one study per val patient"
    assert sum(m["splits"]["train"]["classes"].values()) + sum(m["splits"]["val"]["classes"].values()) == n_files
    # val fraction applies to NEW patients only; holdout added on top
    new_val = val - {P(102), P(301), P(404)}
    assert 2 <= len(new_val) <= 6, new_val

    assert set((out / "train_patients.txt").read_text().split()) == train
    assert set((out / "val_patients.txt").read_text().split()) == val
    studies = _read_csv(out / "studies.csv")
    assert {s["batch"] for s in studies} == {"batch_a", "batch_b"}
    assert all((s["split"] == "val") == (s["patient_id"] in val) for s in studies)


def test_combine_is_deterministic_for_a_seed(two_batches, tmp_path: Path):
    t = two_batches
    outs = []
    for i, seed in enumerate(("3", "3")):
        out = tmp_path / f"rel{i}"
        _run(COMBINE, "--sets", str(t["a"]), str(t["b"]), "--holdout", str(t["holdout"]), "--out", str(out),
             "--seed", seed, "--copy")
        outs.append(out)
    same = [tuple(_pids(o / sp) for sp in ("train", "val", "tune")) for o in outs]
    assert same[0] == same[1]
    assert _manifest(outs[0])["splits"] == _manifest(outs[1])["splits"]
    # --copy gives real files, not links
    files = list((outs[0] / "train").glob("*/*.png"))
    assert files and not any(p.is_symlink() for p in files)


def test_combine_refuses_non_empty_out_and_bad_inputs(two_batches, tmp_path: Path):
    t = two_batches
    out = t["out"]
    _run(COMBINE, "--sets", str(t["a"]), "--out", str(out))
    _run(COMBINE, "--sets", str(t["a"]), "--out", str(out), expect=1)                 # not empty, no --fresh
    _run(COMBINE, "--sets", str(t["a"]), "--out", str(out), "--fresh")               # ok again
    _run(COMBINE, "--sets", str(tmp_path / "nope"), "--out", str(tmp_path / "x"), expect=1)
    _run(COMBINE, "--sets", str(t["a"]), "--holdout", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "y"), expect=1)
    _run(COMBINE, "--sets", str(t["a"]), "--train-lock", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "w"),
         expect=1)
    r = _run(COMBINE, "--sets", str(t["a"]), str(t["b"]), "--holdout", str(t["holdout"]), "--out", str(tmp_path / "z"),
             "--strict-holdout", expect=1)
    assert "strict-holdout" in r.stdout


def test_combine_identical_duplicate_slices_are_skipped(two_batches, tmp_path: Path):
    """The same batch exported twice under two names: every slice is byte-identical -> kept once, no rename."""
    import shutil
    twin = tmp_path / "batch_a_again"
    shutil.copytree(two_batches["a"], twin)
    out = tmp_path / "dup"
    _run(COMBINE, "--sets", str(two_batches["a"]), str(twin), "--out", str(out), "--copy")
    m = _manifest(out)
    assert m["duplicate_slices_skipped"] + m["val_slices_from_non_winning_batch_skipped"] == 20
    assert m["slices_renamed_on_collision"] == 0 and m["n_files"] == 20
    assert m["n_patients"] == m["n_studies"] == 10 and m["label_conflicts"]["n"] == 0


def test_combine_detects_leakage_in_written_tree(two_batches, tmp_path: Path, monkeypatch, capsys):
    """The overlap check reads the WRITTEN tree back, so a bug in placement cannot slip through."""
    c = _import(COMBINE)
    real_place = c.place

    def leaky_place(src, dst, copy):            # every file also lands in the other split
        real_place(src, dst, copy)
        other = "val" if dst.parent.parent.name == "train" else "train"
        real_place(src, dst.parent.parent.parent / other / dst.parent.name / dst.name, copy)

    monkeypatch.setattr(c, "place", leaky_place)
    monkeypatch.setattr(c.sys, "argv", ["combine", "--sets", str(two_batches["a"]), "--out", str(tmp_path / "leak")])
    with pytest.raises(SystemExit) as e:
        c.main()
    assert e.value.code == 1
    assert "in BOTH train and val" in capsys.readouterr().out


# ---- train-lock: the previous release's training patients never reach val -----

@pytest.fixture
def legacy_and_new(tmp_path: Path) -> dict:
    """Legacy set (no studies.csv, like data/local_finetune_bin): train/ = what the current model was trained on,
    val/ = the locked holdout. New batch: some legacy-train patients again, 2 holdout patients (one under the
    batch's own train/), 30 brand-new patients."""
    legacy_train_abn = [P(i) for i in range(1, 21)]
    legacy_train_nor = [P(i) for i in range(21, 36)]
    hold_abn, hold_nor = [P(i) for i in range(50, 56)], [P(i) for i in range(56, 60)]
    legacy = _make_batch(tmp_path, "legacy_bin", {"train": {"abnormal": legacy_train_abn, "normal": legacy_train_nor},
                                                  "val": {"abnormal": hold_abn, "normal": hold_nor}},
                         studies_csv=False)
    new_abn, new_nor = [P(i) for i in range(100, 118)], [P(i) for i in range(200, 212)]
    new = _make_batch(tmp_path, "new1", {
        "train": {"abnormal": new_abn[:14] + [P(1), P(2), P(50)], "normal": new_nor[:10] + [P(21)]},
        "val": {"abnormal": new_abn[14:] + [P(3)], "normal": new_nor[10:] + [P(56)]},
    })
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("\n".join(hold_abn + hold_nor) + "\n", encoding="utf-8")
    return {"legacy": legacy, "new": new, "holdout": holdout, "legacy_train": set(legacy_train_abn + legacy_train_nor),
            "holdout_ids": set(hold_abn + hold_nor), "new_ids": set(new_abn + new_nor), "tmp": tmp_path}


@pytest.mark.parametrize("seed", ["1", "42", "1234"])
def test_combine_legacy_train_patients_are_locked_to_train(legacy_and_new, seed):
    t = legacy_and_new
    out = t["tmp"] / f"rel_{seed}"
    r = _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]), "--out", str(out),
             "--val-frac", "0.5", "--seed", seed)
    train, val, tune = _pids(out / "train"), _pids(out / "val"), _pids(out / "tune")
    assert not (t["legacy_train"] & (val | tune)), "the previous release's training patients must never be scored"
    assert t["legacy_train"] <= train
    assert val - t["holdout_ids"] <= t["new_ids"], "val = holdout + NEW patients only"
    assert t["holdout_ids"] <= val, "holdout forced into val even when the new batch put one under train/"
    assert tune <= t["new_ids"] and not (tune & train) and not (tune & val)
    assert len(val & t["new_ids"]) >= 10, "val-frac 0.5 of 30 new patients"
    applied = set((out / "train_lock_applied.txt").read_text().split())
    assert applied == t["legacy_train"]
    m = _manifest(out)
    assert m["train_lock"]["n_from_sets"] == 35 and m["train_lock"]["n_applied"] == 35
    assert m["train_lock"]["applied_sha256"] and m["train_lock"]["file"] is None
    assert {s["name"]: s["n_train_locked"] for s in m["sets"]} == {"legacy_bin": 35, "new1": 0}
    assert m["splits"]["val"]["n_patients"] == len(val) and m["holdout"]["n_in_val"] == 10
    assert "train-locked: 35" in r.stdout


def test_combine_holdout_that_is_train_locked_is_an_error(legacy_and_new, tmp_path: Path):
    t = legacy_and_new
    bad = tmp_path / "holdout_bad.txt"
    bad.write_text(t["holdout"].read_text() + f"{P(7)}\n{P(22)}\n", encoding="utf-8")   # both in legacy train/
    r = _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(bad), "--out", str(tmp_path / "x"),
             expect=1)
    assert "BOTH in the holdout file and train-locked" in r.stdout and P(7) in r.stdout and P(22) in r.stdout
    # the same through an explicit --train-lock file
    lock = tmp_path / "lock.txt"
    lock.write_text(f"# ids the previous model saw\n{P(50)}\n", encoding="utf-8")
    r = _run(COMBINE, "--sets", str(t["new"]), "--holdout", str(t["holdout"]), "--train-lock", str(lock),
             "--out", str(tmp_path / "y"), expect=1)
    assert P(50) in r.stdout


def test_combine_train_lock_file_and_release_set_dir(legacy_and_new, tmp_path: Path):
    t = legacy_and_new
    new_ids = sorted(t["new_ids"])
    lock = tmp_path / "lock.txt"
    lock.write_text("\n".join(new_ids[:12]) + "\n", encoding="utf-8")
    out = tmp_path / "rel_lockfile"
    _run(COMBINE, "--sets", str(t["new"]), "--holdout", str(t["holdout"]), "--train-lock", str(lock),
         "--out", str(out), "--val-frac", "0.9")
    assert set(new_ids[:12]) <= _pids(out / "train")
    m = _manifest(out)
    assert m["train_lock"]["n_from_file"] == 12 and m["train_lock"]["sha256"]

    # release N (combined set) -> release N+1 with --train-lock <release N dir>: its train+tune stay locked
    rel1 = tmp_path / "train_release_v2"
    _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]), "--out", str(rel1),
         "--seed", "5")
    trained_on = _pids(rel1 / "train") | _pids(rel1 / "tune")
    batch2 = _make_batch(tmp_path, "new2", {"train": {"abnormal": [P(300), P(301)], "normal": [P(400)]},
                                            "val": {"abnormal": [P(302)], "normal": [P(401)]}})
    rel2 = tmp_path / "train_release_v3"
    _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), str(batch2), "--holdout", str(t["holdout"]),
         "--train-lock", str(rel1), "--out", str(rel2), "--val-frac", "0.9", "--seed", "6")
    assert not (trained_on & (_pids(rel2 / "val") | _pids(rel2 / "tune")))
    # a release set given as an INPUT locks its own train/ + tune/ (it has dataset_manifest.json)
    rel3 = tmp_path / "train_release_v3b"
    _run(COMBINE, "--sets", str(rel1), str(batch2), "--holdout", str(t["holdout"]), "--out", str(rel3),
         "--val-frac", "0.9", "--seed", "6")
    assert not (trained_on & (_pids(rel3 / "val") | _pids(rel3 / "tune")))
    assert {s["name"]: s["kind"] for s in _manifest(rel3)["sets"]}["train_release_v2"] == "release set"


# ---- cross-set dedupe: ONE class per patient, last set wins -------------------

def test_combine_label_conflict_last_set_wins_and_one_class_per_patient(tmp_path: Path):
    old = _make_batch(tmp_path, "batch_old", {"train": {"abnormal": [P(1), P(2), P(3)], "normal": [P(4), P(5)]}})
    new = _make_batch(tmp_path, "batch_new", {"train": {"normal": [P(1), P(6)], "abnormal": [P(2), P(7)]}})
    hold = tmp_path / "h.txt"; hold.write_text(f"{P(5)}\n", encoding="utf-8")     # val must not be empty
    for order, want in (((old, new), "normal"), ((new, old), "abnormal")):
        out = tmp_path / f"rel_{order[-1].name}"
        r = _run(COMBINE, "--sets", *map(str, order), "--holdout", str(hold), "--out", str(out),
                 "--val-frac", "0", "--tune-frac", "0")
        cls = _classes_of(out)
        assert all(len(v) == 1 for v in cls.values()), "a patient may never sit under two classes"
        assert cls[P(1)] == {want}, f"the last set ({order[-1].name}) decides"
        # only the DECIDING set's slices are kept: the other set's slices carry the other label and may be
        # another study (follow-up scan), so they are dropped, never relabelled (review round 2)
        names = sorted(p.name for p in (out / "train" / want).glob(f"{P(1)}_*.png"))
        assert names == [f"{P(1)}_0.png", f"{P(1)}_1.png"], names
        assert all(order[-1].name in str(p.resolve()) for p in (out / "train" / want).glob(f"{P(1)}_*.png"))
        m = _manifest(out)
        assert m["conflicting_class_slices_dropped"]["n_files"] == 2
        assert m["conflicting_class_slices_dropped"]["patient_ids"] == [P(1)]
        assert "DROPPED (not relabelled)" in r.stdout
        assert m["label_conflicts"]["n"] == 1 and m["label_conflicts"]["patient_ids"] == [P(1)]
        assert m["label_conflicts"]["details"][0]["winner"] == want
        assert m["label_conflicts"]["details"][0]["winner_set"] == order[-1].name
        assert "label conflicts between sets: 1" in r.stdout and P(1) in r.stdout
        assert m["n_patients"] == m["n_studies"] == 7, "unique patients, not per-batch copies"
        rows = [x for x in _read_csv(out / "studies.csv") if x["patient_id"] == P(1)]
        assert {x["cls"] for x in rows} == {want} and {x["cls_in_batch"] for x in rows} == {"abnormal", "normal"}


def test_combine_holdout_patient_in_two_batches_is_one_val_study(tmp_path: Path):
    """The gate set must stay one study per held-out patient: eval_study_level keys studies by the file-name
    prefix, so a second export ('<pid>_<batch>_<k>') would be scored as a second study."""
    legacy = _make_batch(tmp_path, "legacy_bin", {"train": {"abnormal": [P(1)], "normal": [P(2)]},
                                                  "val": {"abnormal": [P(50)], "normal": [P(51)]}}, studies_csv=False)
    new = _make_batch(tmp_path, "new1", {"train": {"abnormal": [P(50), P(10)], "normal": [P(11)]},
                                         "val": {"normal": [P(12)]}}, k=3)
    holdout = tmp_path / "h.txt"; holdout.write_text(f"{P(50)}\n{P(51)}\n", encoding="utf-8")
    out = tmp_path / "rel"
    _run(COMBINE, "--sets", str(legacy), str(new), "--holdout", str(holdout), "--out", str(out), "--tune-frac", "0")
    v = sorted(p.name for p in (out / "val" / "abnormal").glob(f"{P(50)}_*.png"))
    # a HOLDOUT patient keeps the study the holdout was locked on (the FIRST set), never a newer follow-up
    assert v == [f"{P(50)}_0.png", f"{P(50)}_1.png"], "only the locked (first) batch's export"
    assert all("legacy_bin" in str(p.resolve()) for p in (out / "val" / "abnormal").glob(f"{P(50)}_*.png"))
    m = _manifest(out)
    assert m["splits"]["val"]["n_study_keys"] == m["splits"]["val"]["n_patients"]
    assert m["val_slices_from_non_winning_batch_skipped"] == 3


def test_combine_within_set_two_classes(tmp_path: Path):
    """A set holding one patient under two classes: superseded when a later set decides, else the patient is
    dropped (and a dropped HOLDOUT patient is fatal)."""
    a = _make_batch(tmp_path, "a", {"train": {"abnormal": [P(1), P(2), P(3)], "normal": [P(1), P(4), P(5)]}})
    b = _make_batch(tmp_path, "b", {"train": {"normal": [P(1), P(6)], "abnormal": [P(7)]}})
    h5 = tmp_path / "h5.txt"; h5.write_text(f"{P(5)}\n", encoding="utf-8")
    out = tmp_path / "r1"
    _run(COMBINE, "--sets", str(a), str(b), "--holdout", str(h5), "--out", str(out), "--val-frac", "0",
         "--tune-frac", "0")
    assert _classes_of(out)[P(1)] == {"normal"}
    assert _manifest(out)["patients_dropped_within_set_conflict"]["n"] == 0
    out2 = tmp_path / "r2"
    r = _run(COMBINE, "--sets", str(b), str(a), "--holdout", str(h5), "--out", str(out2), "--val-frac", "0",
             "--tune-frac", "0")
    assert P(1) not in _classes_of(out2) and "DROPPED" in r.stdout
    assert _manifest(out2)["patients_dropped_within_set_conflict"]["patient_ids"] == [P(1)]
    hold = tmp_path / "h.txt"; hold.write_text(f"{P(1)}\n{P(5)}\n", encoding="utf-8")
    # a holdout patient is decided by the first set that holds it with ONE class (b), so it stays in val
    out3 = tmp_path / "r3"
    _run(COMBINE, "--sets", str(b), str(a), "--holdout", str(hold), "--out", str(out3))
    assert _classes_of(out3)[P(1)] == {"normal"} and P(1) in _pids(out3 / "val")
    # ...but when no set holds it with one class it is dropped, and a dropped holdout patient is fatal
    c = _make_batch(tmp_path, "c", {"train": {"abnormal": [P(1), P(8)], "normal": [P(1), P(9)]}})
    r = _run(COMBINE, "--sets", str(a), str(c), "--holdout", str(hold), "--out", str(tmp_path / "r4"), expect=1)
    assert "holdout patients were dropped" in r.stdout


# ---- tune/: checkpoint-selection split, disjoint from train and val -----------

def test_combine_tune_split_is_new_patients_only_and_disjoint(legacy_and_new):
    t = legacy_and_new
    out = t["tmp"] / "rel_tune"
    _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]), "--out", str(out),
         "--val-frac", "0.2", "--tune-frac", "0.25", "--seed", "42")
    train, val, tune = _pids(out / "train"), _pids(out / "val"), _pids(out / "tune")
    assert tune and not (tune & train) and not (tune & val) and not (train & val)
    assert tune <= t["new_ids"], "tune holds NEW patients only: never holdout, never train-locked"
    new_train_side = t["new_ids"] - val
    assert abs(len(tune) - 0.25 * len(new_train_side)) <= 2
    assert set((out / "tune_patients.txt").read_text().split()) == tune
    m = _manifest(out)
    assert m["splits"]["tune"]["n_patients"] == len(tune) and m["tune_frac_new_train_patients"] == 0.25
    assert set(m["splits"]["tune"]["classes"]) == {"abnormal", "normal"}
    # default 0.1 also writes a tune/; --tune-frac 0 writes none
    out0 = t["tmp"] / "rel_notune"
    r = _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]), "--out", str(out0),
             "--tune-frac", "0")
    assert not (out0 / "tune").exists() and "no tune/ split" in r.stdout


def test_split_patients_unit():
    c = _import(COMBINE)
    pids = {P(i) for i in range(40)}
    cls = {p: ("abnormal" if int(p) % 2 else "normal") for p in pids}
    locked = {P(i) for i in range(10)}
    hold = {P(10), P(11)}
    tr, va, tu = c.split_patients(pids, hold, cls, 0.25, 42, locked=locked, tune_frac=0.2)
    assert tr | va | tu == pids and not (tr & va) and not (tr & tu) and not (va & tu)
    assert locked <= tr and hold <= va
    assert len(va - hold) == 8 and len(tu) == 4          # per class: 14 new -> round(3.5)=4 val, round(2.0)=2 tune
    assert c.split_patients(pids, hold, cls, 0.25, 42, locked=locked, tune_frac=0.2) == (tr, va, tu)


# ---------------------------------------------------------------- stage 6: trainer checkpoint selection

TRAINER = REPO / "training" / "finetune_brain_classifier.py"


def _trainer():
    pytest.importorskip("torch")
    return _import(TRAINER)


def test_trainer_select_split_defaults(tmp_path: Path):
    tr = _trainer()
    data = tmp_path / "set"
    (data / "val" / "abnormal").mkdir(parents=True)
    (data / "val" / "abnormal" / f"{P(1)}_0.png").write_bytes(PNG_1x1)
    split, warn = tr.resolve_select_split(data, None)
    assert split == "val" and warn and "GATE set" in warn
    with pytest.raises(SystemExit):
        tr.resolve_select_split(data, "tune")
    (data / "tune" / "normal").mkdir(parents=True)
    split, warn = tr.resolve_select_split(data, None)
    assert split == "val" and warn, "an EMPTY tune/ is not a selection split"
    (data / "tune" / "normal" / f"{P(2)}_0.png").write_bytes(PNG_1x1)
    assert tr.resolve_select_split(data, None) == ("tune", None)
    split, warn = tr.resolve_select_split(data, "val")
    assert split == "val" and warn, "an explicit val still warns"


def _tiny_vit(dst: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ViTConfig(image_size=32, patch_size=16, hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=16, num_labels=2,
                                 id2label={0: "abnormal", 1: "normal"}, label2id={"abnormal": 0, "normal": 1})
    transformers.ViTForImageClassification(cfg).save_pretrained(dst)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(dst)
    return dst


def _png_set(root: Path, splits: tuple[str, ...]) -> Path:
    from PIL import Image
    for sp in splits:
        for ci, cls in enumerate(("abnormal", "normal")):
            d = root / sp / cls
            d.mkdir(parents=True)
            for j in range(3):
                Image.new("RGB", (32, 32), (60 + 120 * ci, 10 * j, 0)).save(d / f"{P(10 * ci + j)}_{j}.png")
    return root


@pytest.mark.parametrize("with_tune", [True, False])
def test_trainer_records_which_split_selected_the_checkpoint(tmp_path: Path, with_tune: bool):
    _trainer()
    base = _tiny_vit(tmp_path / "base")
    data = _png_set(tmp_path / "set", ("train", "val", "tune") if with_tune else ("train", "val"))
    out = tmp_path / "cand"
    r = subprocess.run([sys.executable, str(TRAINER), "--data", str(data), "--classes", "abnormal,normal",
                        "--base-model", str(base), "--epochs", "1", "--batch-size", "4", "--device", "cpu",
                        "--balance", "--output", str(out)], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads((out / "training_metrics.json").read_text())
    sel = m["checkpoint_selection"]
    assert (out / "model.safetensors").is_file(), "the first epoch is always saved"
    if with_tune:
        assert sel["split"] == "tune" and sel["gate_set_used_for_selection"] is False
        assert "WARNING" not in r.stdout and "best_val_acc" not in m and "val_acc" not in m["epochs"][0]
    else:
        assert sel["split"] == "val" and sel["gate_set_used_for_selection"] is True
        assert "WARNING: checkpoint selection uses" in r.stdout and m["best_val_acc"] == sel["best_acc"]
    assert sel["best_epoch"] == 1 and m["epochs"][0]["select_split"] == sel["split"]


# ---- review round 2: follow-up studies, holdout pinning, minimum tune size ----

def test_combine_never_relabels_an_older_study_and_keeps_the_locked_holdout_study(tmp_path: Path):
    """A patient scanned normal in an older batch and abnormal in a newer one (a follow-up study): the
    older slices must not be trained as abnormal; a locked holdout patient keeps its locked study in
    val even when a newer batch holds a follow-up with another label (review round 2, combine:222)."""
    a = _make_batch(tmp_path, "batchA", {"train": {"normal": [P(5001), P(5002)], "abnormal": [P(5003)]},
                                         "val": {"normal": [P(7001)], "abnormal": [P(7002)]}}, studies_csv=False)
    b = _make_batch(tmp_path, "batchB", {"train": {"abnormal": [P(5001), P(7001), P(5004)],
                                                   "normal": [P(5005)]}}, k=3)
    hold = tmp_path / "h.txt"; hold.write_text(f"{P(7001)}\n{P(7002)}\n", encoding="utf-8")
    out = tmp_path / "rel"
    r = _run(COMBINE, "--sets", str(a), str(b), "--holdout", str(hold), "--out", str(out),
             "--val-frac", "0", "--tune-frac", "0")
    # 5001: batchB (the newer label) decides; batchA's NORMAL study is dropped, not moved into abnormal/
    f5001 = list(out.glob(f"*/*/{P(5001)}_*.png"))
    assert {p.parent.name for p in f5001} == {"abnormal"} and len(f5001) == 3
    assert all("batchB" in str(p.resolve()) for p in f5001), [str(p.resolve()) for p in f5001]
    # 7001 (holdout): the locked batchA study, label normal, in val — the follow-up is not used
    f7001 = list(out.glob(f"*/*/{P(7001)}_*.png"))
    assert {(p.parent.parent.name, p.parent.name) for p in f7001} == {("val", "normal")} and len(f7001) == 2
    assert all("batchA" in str(p.resolve()) for p in f7001)
    m = _manifest(out)
    assert m["conflicting_class_slices_dropped"]["patient_ids"] == [P(5001), P(7001)]
    assert m["conflicting_class_slices_dropped"]["n_files"] == 2 + 3
    det = {d["patient_id"]: d for d in m["label_conflicts"]["details"]}
    assert det[P(7001)]["winner"] == "normal" and det[P(7001)]["winner_set"] == "batchA"
    assert det[P(5001)]["winner"] == "abnormal" and det[P(5001)]["winner_set"] == "batchB"
    assert "one class per patient" in r.stdout


def test_combine_tune_split_has_a_minimum_per_class(legacy_and_new):
    """tune/ is where the trainer picks its epoch: a 2-patient tune set makes that pick noise (review
    round 2, combine:254). --min-tune-per-class lifts tiny fractions, capped at a third of the rest."""
    t = legacy_and_new
    out = t["tmp"] / "rel_min_tune"
    _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]), "--out", str(out),
         "--val-frac", "0.2", "--tune-frac", "0.01", "--min-tune-per-class", "3", "--seed", "42")
    m = _manifest(out)
    tune = _pids(out / "tune")
    assert not (tune & _pids(out / "train")) and not (tune & _pids(out / "val"))
    per_cls = {c: len({p.stem.split("_", 1)[0] for p in (out / "tune" / c).glob("*.png")})
               for c in ("abnormal", "normal")}
    assert per_cls == {"abnormal": 3, "normal": 3}, per_cls      # round(0.01 × n) = 0 → lifted to 3
    assert m["min_tune_per_class"] == 3
    c = _import(COMBINE)
    pids = {P(i) for i in range(12)}
    cls = {p: "normal" for p in pids}
    _tr, _va, tu = c.split_patients(pids, set(), cls, 0.0, 1, tune_frac=0.05, min_tune_per_class=10)
    assert len(tu) == 4, "capped at a third of the class's new train patients"
    r = _run(COMBINE, "--sets", str(t["legacy"]), str(t["new"]), "--holdout", str(t["holdout"]),
             "--out", str(t["tmp"] / "rel_small_tune"), "--tune-frac", "0.01", "--min-tune-per-class", "50")
    assert "tune/ is small" in r.stdout
