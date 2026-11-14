"""Prospective evaluation of the shipped model on a new batch (scripts/eval_prospective.py).

No model and no patient data: tiny constant-grey PNG trees are built under
tmp_path and scored by a stub predictor (abnormal probability = mean pixel /
255) that replaces eval_study_level.make_predictor, so every study decision
is known in advance. Ids are synthetic (>= 9000000), studies.csv / matched.csv
are synthetic.

Covered:
  * grouping precedence (seen_train beats holdout and --seen val; '<pid>' and
    '<pid>_s2' stay together; 9000010 vs 90000101 are not confused); the
    three kinds of --seen set (release / legacy / per-batch, whose val/ is
    advisory) and their precedence
  * group metrics on a hand-computed confusion table, Wilson CIs equal
    eval_study_level.wilson_ci, empty groups give zeros / None
  * rank AUC against hand-computed values (ties, empty class) and brute force
  * threshold sweep monotonicity; per-finding table; new_same_date join
    through studies.csv incl. a '_s2' study
  * missing studies.csv, empty 'new' group, a study key under both classes /
    in both splits, unusable model / set / --seen / --holdout → exit 2
  * no ids or study folders on stdout or in the JSON; the JSON keys
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "eval_prospective.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ep = _load("eval_prospective", SCRIPT)
esl = ep.esl

STUDIES_HEADER = ["patient_id", "study_folder", "cls", "split", "n_slices", "label", "label_source", "modality"]


# ------------------------------------------------------------------ fixtures

def _stub(image):
    p = float(np.mean(image)) / 255.0
    return {"abnormal": p, "normal": 1.0 - p}


PREDICTOR_CALLS: list[tuple[Path, str]] = []


@pytest.fixture
def stub_model(monkeypatch, tmp_path):
    """An existing (empty) model dir whose 'predictor' is the stub
    (every make_predictor call is logged in PREDICTOR_CALLS)."""
    PREDICTOR_CALLS.clear()

    def fake_make_predictor(model_dir, device="cpu"):
        PREDICTOR_CALLS.append((Path(model_dir), device))
        return _stub
    monkeypatch.setattr(ep.esl, "make_predictor", fake_make_predictor)
    d = tmp_path / "model"
    d.mkdir()
    return d


def _png(root: Path, split: str, cls: str, key: str, prob: float, n: int = 2) -> None:
    from PIL import Image
    d = root / split / cls
    d.mkdir(parents=True, exist_ok=True)
    v = int(round(prob * 255))
    for k in range(n):
        Image.fromarray(np.full((12, 10), v, dtype=np.uint8)).convert("RGB").save(d / f"{key}_{k}.png")


def _csv(fp: Path, header: list[str], rows: list[list]) -> None:
    with open(fp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _batch(tmp_path: Path, with_studies_csv: bool = True) -> dict:
    """A per-batch set + one seen set + holdout + matched. Expected groups:
      9000001  seen_train (also in holdout file: seen_train wins)   normal  0.2 → TN
      9000002  holdout                                           abnormal  0.8 → TP
      9000010  new: '9000010' tumor 0.9 → TP (match 'id'), '9000010_s2' stroke 0.3 → FN (match 'date')
      90000101 new  normal 0.6 → FP (match 'date')
      9000020  new  normal 0.1 → TN (val split, match 'date')
      9000030  in studies.csv with n_slices 0 (no PNG)"""
    s = tmp_path / "sets" / "batch_x"
    _png(s, "train", "normal", "9000001", 0.2)
    _png(s, "val", "abnormal", "9000002", 0.8)
    _png(s, "train", "abnormal", "9000010", 0.9)
    _png(s, "train", "abnormal", "9000010_s2", 0.3, n=3)
    _png(s, "train", "normal", "90000101", 0.6)
    _png(s, "val", "normal", "9000020", 0.1)
    if with_studies_csv:
        _csv(s / "studies.csv", STUDIES_HEADER, [
            # deliberately NOT in (patient_id, study_folder) order: read_prefix_map sorts
            ["9000010", "STUDY_K", "abnormal", "train", 3, "stroke", "rules", "MR"],
            ["9000010", "STUDY_J", "abnormal", "train", 2, "tumor", "rules", "MR"],
            ["9000001", "STUDY_A", "normal", "train", 2, "normal", "rules", "MR"],
            ["9000002", "STUDY_B", "abnormal", "val", 2, "tumor", "llm", "MR"],
            ["90000101", "STUDY_C", "normal", "train", 2, "normal", "rules", "MR"],
            ["9000020", "STUDY_D", "normal", "val", 2, "normal", "rules", "MR"],
            ["9000030", "STUDY_E", "abnormal", "train", 0, "cyst", "rules", "MR"],
        ])
    seen = tmp_path / "seen_release"
    _png(seen, "train", "normal", "9000001_old", 0.5, n=1)
    _png(seen, "val", "abnormal", "9000040", 0.5, n=1)
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("# locked\n9000002\n9000001\n", encoding="utf-8")
    matched = tmp_path / "batch_x_matched.csv"
    _csv(matched, ["study_folder", "patient_id", "modality", "study_date", "match_quality"], [
        ["STUDY_J", "9000010", "MR", "20310105", "id"],
        ["STUDY_K", "9000010", "MR", "20310301", "date"],
        ["STUDY_C", "90000101", "MR", "20310110", "date"],
        ["STUDY_D", "9000020", "MR", "20310111", "date"],
        ["STUDY_A", "9000001", "MR", "20310112", "date"],
        ["STUDY_B", "9000002", "MR", "20310113", "date"],
    ])
    return {"set": s, "seen": seen, "holdout": holdout, "matched": matched,
            "ids": ["9000001", "9000002", "9000010", "90000101", "9000020", "9000030", "9000040"],
            "folders": ["STUDY_A", "STUDY_B", "STUDY_C", "STUDY_D", "STUDY_E", "STUDY_J", "STUDY_K"]}


def _argv(b: dict, model: Path, out: Path, *, seen=True, holdout=True, matched=True) -> list[str]:
    argv = ["--model", str(model), "--set", str(b["set"]), "--out", str(out), "--device", "cpu"]
    if seen:
        argv += ["--seen", str(b["seen"])]
    if holdout:
        argv += ["--holdout", str(b["holdout"])]
    if matched:
        argv += ["--matched", str(b["matched"])]
    return argv


def _assert_no_ids(text: str, b: dict) -> None:
    for pid in b["ids"]:
        assert not re.search(rf"(?<!\d){pid}(?!\d)", text), f"patient id leaked: {pid}"
    for folder in b["folders"]:
        assert folder not in text, f"study folder leaked: {folder}"


# ------------------------------------------------------------------ grouping

def test_grouping_precedence_and_prefix_ids(tmp_path):
    seen = tmp_path / "seen"
    _png(seen, "train", "normal", "9000001", 0.1, n=1)
    _png(seen, "tune", "abnormal", "9000005_s2_batch_one", 0.9, n=1)
    _png(seen, "val", "normal", "90000011", 0.1, n=1)
    _png(seen, "val", "normal", "9000001_s2", 0.1, n=1)      # also val: train still wins
    train, val, batch, notes = ep.load_seen_patients([seen])
    assert train == {"9000001", "9000005"} and val == {"90000011", "9000001"} and notes == []
    assert batch == set()
    groups = ep.assign_groups({"9000001", "90000011", "9000002", "9000003", "9000005", "900000"},
                              train, val, holdout_ids={"9000001", "9000002"})
    assert groups == {"9000001": "seen_train", "9000005": "seen_train", "90000011": "holdout",
                      "9000002": "holdout", "9000003": "new", "900000": "new"}


def _release_set(root: Path) -> Path:
    """A combined release set (combine_training_sets.py output): dataset_manifest.json present."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    _csv(root / "studies.csv", STUDIES_HEADER, [])
    return root


def _per_batch_set(root: Path) -> Path:
    """A build_training_set.py output: studies.csv, no dataset_manifest.json (its split is advisory)."""
    root.mkdir(parents=True, exist_ok=True)
    _csv(root / "studies.csv", STUDIES_HEADER, [])
    return root


def test_seen_set_kinds_release_legacy_and_per_batch(tmp_path):
    rel = _release_set(tmp_path / "train_release_vX")
    _png(rel, "train", "normal", "9000601_batch_a", 0.1, n=1)
    _png(rel, "tune", "abnormal", "9000602_s2_batch_a", 0.9, n=1)
    _png(rel, "val", "normal", "9000603", 0.1, n=1)
    train, holdout, batch, notes = ep.load_seen_patients([rel])
    assert (train, holdout, batch, notes) == ({"9000601", "9000602"}, {"9000603"}, set(), [])

    legacy = tmp_path / "local_finetune_bin"
    _png(legacy, "train", "normal", "9000611", 0.1, n=1)
    _png(legacy, "val", "abnormal", "9000612", 0.9, n=1)
    train, holdout, batch, notes = ep.load_seen_patients([legacy])
    assert (train, holdout, batch, notes) == ({"9000611"}, {"9000612"}, set(), [])

    per_batch = _per_batch_set(tmp_path / "sets" / "new1")
    _png(per_batch, "train", "normal", "9000621", 0.1, n=1)
    _png(per_batch, "val", "abnormal", "9000622", 0.9, n=1)
    _png(per_batch, "val", "abnormal", "9000622_s2", 0.9, n=1)
    train, holdout, batch, notes = ep.load_seen_patients([per_batch])
    assert train == set() and holdout == set() and batch == {"9000621", "9000622"}, \
        "the val/ of a per-batch set is advisory: combine re-splits and trains on most of it"
    assert len(notes) == 1 and "advisory" in notes[0]


def test_per_batch_set_precedence(tmp_path):
    """seen_train (authoritative train/tune) > holdout (release val/, legacy val/, --holdout) >
    per-batch set (seen, role unknown -> seen_train) > new."""
    per_batch = _per_batch_set(tmp_path / "sets" / "new1")
    for pid in ("9000701", "9000702", "9000703", "9000704", "9000706"):
        _png(per_batch, "val", "normal", pid, 0.1, n=1)
    _png(per_batch, "train", "normal", "9000705", 0.1, n=1)
    rel = _release_set(tmp_path / "train_release_vX")
    _png(rel, "val", "normal", "9000701_new1", 0.1, n=1)      # per-batch val + release val -> holdout
    _png(rel, "train", "normal", "9000704_new1", 0.1, n=1)    # per-batch + release train -> seen_train
    _png(rel, "train", "normal", "9000708_new1", 0.1, n=1)    # release train + --holdout -> seen_train
    legacy = tmp_path / "local_finetune_bin"
    _png(legacy, "val", "normal", "9000706", 0.1, n=1)        # per-batch + legacy val -> holdout
    train, holdout, batch, _ = ep.load_seen_patients([legacy, per_batch, rel])
    groups = ep.assign_groups({"9000701", "9000702", "9000703", "9000704", "9000705", "9000706", "9000707",
                               "9000708"}, train, holdout, holdout_ids={"9000703", "9000708"}, seen_batch=batch)
    assert groups == {"9000701": "holdout",      # release val/ is authoritative
                      "9000702": "seen_train",   # only in a per-batch set (its val/ is advisory)
                      "9000703": "holdout",      # per-batch + --holdout file
                      "9000704": "seen_train",
                      "9000705": "seen_train",
                      "9000706": "holdout",
                      "9000707": "new",
                      "9000708": "seen_train"}


def test_evaluate_counts_a_per_batch_val_patient_as_seen_not_holdout(tmp_path, stub_model):
    s = tmp_path / "sets" / "new2"
    _png(s, "train", "abnormal", "9000801", 0.9)
    _png(s, "train", "normal", "9000802", 0.2)
    _png(s, "train", "normal", "9000803", 0.2)
    prev = _per_batch_set(tmp_path / "sets" / "new1")
    _png(prev, "val", "abnormal", "9000801", 0.9, n=1)        # follow-up of a per-batch val patient
    _png(prev, "train", "normal", "9000802", 0.2, n=1)
    payload = ep.evaluate(stub_model, s, seen=[str(prev)])
    g = payload["groups"]
    assert (g["seen_train"]["n_studies"], g["holdout"]["n_studies"], g["new"]["n_studies"]) == (2, 0, 1)
    assert any("advisory" in n for n in payload["notes"])


def test_seen_dir_without_splits_is_noted_and_missing_dir_refused(tmp_path):
    (tmp_path / "flat").mkdir()
    _, _, _, notes = ep.load_seen_patients([tmp_path / "flat"])
    assert len(notes) == 1 and "no train/" in notes[0]
    with pytest.raises(ep.UnusableInput):
        ep.load_seen_patients([tmp_path / "absent"])


# ------------------------------------------------------------------ metrics

def _studies(abn: list[float], norm: list[float]) -> dict:
    st = {(f"{9000100 + i}", "abnormal"): [p] for i, p in enumerate(abn)}
    st.update({(f"{9000200 + i}", "normal"): [p] for i, p in enumerate(norm)})
    return st


def test_group_metrics_hand_computed():
    st = _studies([0.9, 0.7, 0.6, 0.2], [0.1, 0.3, 0.4, 0.55, 0.8, 0.2])
    st[("9000100_s2", "abnormal")] = [0.4, 0.8]            # mean 0.6 → TP, same patient as 9000100
    m = ep.group_metrics(st)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (4, 1, 4, 2)
    assert m["n_studies"] == 11 and m["n_patients"] == 10 and m["n_abnormal"] == 5 and m["n_normal"] == 6
    assert m["sensitivity"] == pytest.approx(4 / 5) and m["specificity"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx((4 / 5 + 4 / 6) / 2)
    assert m["ppv"] == pytest.approx(4 / 6) and m["npv"] == pytest.approx(4 / 5)
    assert m["flag_rate"] == pytest.approx(6 / 11) and m["prevalence"] == pytest.approx(5 / 11)
    assert m["sensitivity_ci95"] == esl.wilson_ci(4, 5)
    assert m["specificity_ci95"] == esl.wilson_ci(4, 6)
    # AUC by hand over study means: abn [0.9,0.7,0.6,0.2,0.6] vs norm [0.1,0.3,0.4,0.55,0.8,0.2]
    wins = sum(1.0 if a > n else 0.5 if a == n else 0.0
               for a in [0.9, 0.7, 0.6, 0.2, 0.6] for n in [0.1, 0.3, 0.4, 0.55, 0.8, 0.2])
    assert m["auc"] == pytest.approx(wins / 30)


def test_group_metrics_boundary_is_shipped_protocol():
    """mean >= 0.5 is flagged (eval_study_level.gate_fn('mean', 0.5))."""
    m = ep.group_metrics({("9000300", "abnormal"): [0.5], ("9000301", "normal"): [0.25, 0.75]})
    assert (m["tp"], m["fp"]) == (1, 1)


def test_empty_group_is_zeros_and_none():
    m = ep.group_metrics({})
    assert all(m[k] == 0 for k in ("n_studies", "n_patients", "n_abnormal", "n_normal", "tp", "fn", "tn", "fp"))
    assert all(m[k] is None for k in ("sensitivity", "sensitivity_ci95", "specificity", "specificity_ci95",
                                      "balanced_accuracy", "ppv", "npv", "flag_rate", "prevalence", "auc"))
    only_abn = ep.group_metrics(_studies([0.9, 0.1], []))
    assert only_abn["sensitivity"] == 0.5 and only_abn["specificity"] is None and only_abn["auc"] is None
    assert only_abn["npv"] == 0.0 and only_abn["balanced_accuracy"] is None


def test_auc_hand_computed_ties_and_empty():
    assert ep.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ep.auc([0.1], [0.9]) == 0.0
    assert ep.auc([0.5], [0.5]) == 0.5
    assert ep.auc([0.3, 0.7], [0.5, 0.7]) == pytest.approx(1.5 / 4)
    assert ep.auc([0.5, 0.5, 0.5], [0.5, 0.2]) == pytest.approx((3 * 0.5 + 3 * 1) / 6)
    assert ep.auc([], [0.1]) is None and ep.auc([0.1], []) is None and ep.auc([], []) is None
    rng = np.random.default_rng(3)
    pos = list(np.round(rng.random(40), 1))
    neg = list(np.round(rng.random(30), 1))                 # rounding forces many ties
    brute = sum(1.0 if a > n else 0.5 if a == n else 0.0 for a, n in product(pos, neg)) / (40 * 30)
    assert ep.auc(pos, neg) == pytest.approx(brute)


def test_threshold_sweep_monotone_and_marks_shipped():
    rng = np.random.default_rng(7)
    st = _studies(list(rng.random(25)), list(rng.random(30)))
    sweep = ep.threshold_sweep(st)
    assert [r["threshold"] for r in sweep] == [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    assert [r["shipped"] for r in sweep] == [t == 0.5 for t in ep.SWEEP_THRESHOLDS]
    for a, b in zip(sweep, sweep[1:]):
        assert b["sensitivity"] <= a["sensitivity"]
        assert b["specificity"] >= a["specificity"]
        assert b["flagged_share"] <= a["flagged_share"]
    at05 = next(r for r in sweep if r["shipped"])
    m = ep.group_metrics(st)
    assert at05["sensitivity"] == m["sensitivity"] and at05["specificity"] == m["specificity"]
    assert at05["flagged_share"] == m["flag_rate"]
    empty = ep.threshold_sweep({})
    assert all(r["sensitivity"] is None and r["specificity"] is None and r["flagged_share"] is None
               for r in empty)


def test_findings_table():
    table = {"9000400": {"study_folder": "F1", "label": "tumor", "cls": "abnormal"},
             "9000401": {"study_folder": "F2", "label": "tumor", "cls": "abnormal"},
             "9000402": {"study_folder": "F3", "label": "stroke", "cls": "abnormal"},
             "9000403": {"study_folder": "F4", "label": "normal", "cls": "normal"},
             "9000404": {"study_folder": "F5", "label": "normal", "cls": "normal"},
             "9000405": {"study_folder": "F6", "label": "atrophy", "cls": "normal"}}
    st = {("9000400", "abnormal"): [0.9], ("9000401", "abnormal"): [0.2], ("9000402", "abnormal"): [0.6],
          ("9000403", "normal"): [0.7], ("9000404", "normal"): [0.1], ("9000405", "normal"): [0.8],
          ("9000406", "normal"): [0.1]}
    f, n_unmapped = ep.findings_table(st, table)
    assert n_unmapped == 1
    assert f["tumor"] == {"reference": "abnormal", "total": 2, "detected": 1, "rate": 0.5}
    assert f["stroke"] == {"reference": "abnormal", "total": 1, "detected": 1, "rate": 1.0}
    assert f["normal"] == {"reference": "normal", "total": 2, "wrongly_flagged": 1, "rate": 0.5}
    assert f["atrophy (reference normal)"]["wrongly_flagged"] == 1
    assert list(f)[:2] == ["tumor", "stroke"]                 # abnormal findings first


# ------------------------------------------------------------------ end to end

def test_end_to_end_groups_findings_same_date_and_json(stub_model, tmp_path, capsys):
    b = _batch(tmp_path)
    out = tmp_path / "release" / "v9" / "prospective_eval.json"
    assert ep.main(_argv(b, stub_model, out)) == 0
    assert PREDICTOR_CALLS == [(stub_model, "cpu")]            # ONE predictor for the whole set
    stdout = capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert not (out.parent / (out.name + ".tmp")).exists()

    assert set(payload) == {"kind", "schema_version", "evaluated_on", "model", "protocol", "inputs", "groups",
                            "threshold_sweep_new", "findings_new", "notes"}
    assert payload["kind"] == "prospective_eval" and payload["schema_version"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["evaluated_on"])
    assert payload["model"] == {"path": str(stub_model), "identity": esl.model_identity(stub_model),
                                "manifest_verified": False}          # the stub model dir has no MANIFEST.json
    assert payload["protocol"] == esl.protocol_block(esl.PROTOCOL_AGGREGATION, esl.PROTOCOL_THRESHOLD)
    assert payload["inputs"]["n_png"] == 13 and payload["inputs"]["n_studies"] == 6
    assert payload["inputs"]["seen"] == [str(b["seen"])]

    g = payload["groups"]
    assert set(g) == {"all", "new", "seen_train", "holdout", "new_same_date"}
    new = g["new"]
    assert (new["n_studies"], new["n_patients"]) == (4, 3)     # '9000010' + '9000010_s2' = one patient
    assert (new["tp"], new["fn"], new["tn"], new["fp"]) == (1, 1, 1, 1)
    assert new["sensitivity_ci95"] == esl.wilson_ci(1, 2)
    assert (g["seen_train"]["n_studies"], g["seen_train"]["tn"]) == (1, 1)
    assert (g["holdout"]["n_studies"], g["holdout"]["tp"]) == (1, 1)
    assert g["all"]["n_studies"] == 6 and g["all"]["n_patients"] == 5
    sd = g["new_same_date"]                                    # '_s2' (date) in, first study (id) out
    assert (sd["n_studies"], sd["tp"], sd["fn"], sd["tn"], sd["fp"]) == (3, 0, 1, 1, 1)

    f = payload["findings_new"]
    assert f["tumor"]["detected"] == 1 and f["tumor"]["total"] == 1
    assert f["stroke"]["detected"] == 0 and f["stroke"]["total"] == 1
    assert f["normal"]["wrongly_flagged"] == 1 and f["normal"]["total"] == 2
    assert "cyst" not in f                                      # n_slices 0: never scored
    assert any("no PNG files" in n for n in payload["notes"])
    assert any("descriptive only" in n for n in payload["notes"])
    assert len(payload["threshold_sweep_new"]) == 8

    assert "NEW PATIENTS (never seen by this model)" in stdout and "n=4" in stdout
    _assert_no_ids(stdout, b)
    _assert_no_ids(text, b)


def test_all_group_equals_eval_study_level_scoring(stub_model, tmp_path):
    """Same protocol as the release gate: 'all' == score_studies(gate mean @0.5) over the set."""
    b = _batch(tmp_path)
    payload = ep.evaluate(stub_model, b["set"], device="cpu")
    studies = {}
    for sp in ("train", "val"):
        part, _ = esl.collect_study_probs(stub_model, b["set"] / sp, predictor=_stub)
        studies.update(part)
    r = esl.score_studies(studies, esl.gate_fn("mean", 0.5))
    a = payload["groups"]["all"]
    assert (a["tp"], a["fn"], a["tn"], a["fp"]) == (r["tp"], r["fn"], r["tn"], r["fp"])
    assert a["sensitivity_ci95"] == r["sens_ci"] and a["specificity_ci95"] == r["spec_ci"]
    assert any("NOT verified" in n for n in payload["notes"])  # no --seen given


def test_missing_studies_csv_omits_findings_and_same_date(stub_model, tmp_path, capsys):
    b = _batch(tmp_path, with_studies_csv=False)
    out = tmp_path / "pe.json"
    assert ep.main(_argv(b, stub_model, out)) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "new_same_date" not in payload["groups"] and payload["findings_new"] is None
    assert any("studies.csv not found" in n for n in payload["notes"])
    assert payload["groups"]["new"]["n_studies"] == 4
    assert "studies.csv not found" in capsys.readouterr().out


def test_unusable_studies_csv_is_a_note(stub_model, tmp_path):
    b = _batch(tmp_path)
    _csv(b["set"] / "studies.csv", ["patient_id", "cls"], [["9000010", "abnormal"]])
    payload = ep.evaluate(stub_model, b["set"], matched=b["matched"], device="cpu")
    assert "new_same_date" not in payload["groups"] and payload["findings_new"] is None
    assert any("unusable" in n for n in payload["notes"])


def test_empty_new_group(stub_model, tmp_path, capsys):
    s = tmp_path / "set"
    _png(s, "train", "abnormal", "9000501", 0.9)
    _png(s, "val", "normal", "9000502", 0.2)
    seen = tmp_path / "seen"
    _png(seen, "train", "abnormal", "9000501", 0.9, n=1)
    _png(seen, "val", "normal", "9000502", 0.2, n=1)
    out = tmp_path / "pe.json"
    assert ep.main(["--model", str(stub_model), "--set", str(s), "--seen", str(seen),
                    "--out", str(out), "--device", "cpu"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    new = payload["groups"]["new"]
    assert new["n_studies"] == 0 and new["sensitivity"] is None and new["auc"] is None
    assert all(r["sensitivity"] is None for r in payload["threshold_sweep_new"])
    assert payload["groups"]["seen_train"]["n_studies"] == 1 and payload["groups"]["holdout"]["n_studies"] == 1
    assert any("'new' is empty" in n for n in payload["notes"])
    assert "n=0" in capsys.readouterr().out


# ------------------------------------------------------------------ refusals (exit 2)

def test_study_key_under_both_classes_exits_2(stub_model, tmp_path, capsys):
    s = tmp_path / "set"
    _png(s, "train", "abnormal", "9000601", 0.9)
    _png(s, "train", "normal", "9000601", 0.2)
    out = tmp_path / "pe.json"
    assert ep.main(["--model", str(stub_model), "--set", str(s), "--out", str(out), "--device", "cpu"]) == 2
    assert not out.exists()
    stdout = capsys.readouterr().out
    assert "more than one class" in stdout and "9000601" not in stdout


def test_study_key_in_both_splits_exits_2(stub_model, tmp_path, capsys):
    s = tmp_path / "set"
    _png(s, "train", "abnormal", "9000602", 0.9)
    _png(s, "val", "abnormal", "9000602", 0.9)
    assert ep.main(["--model", str(stub_model), "--set", str(s), "--out", str(tmp_path / "pe.json")]) == 2
    stdout = capsys.readouterr().out
    assert "both train/ and val/" in stdout and "9000602" not in stdout


def test_unusable_inputs_exit_2(stub_model, tmp_path, monkeypatch):
    b = _batch(tmp_path)
    out = tmp_path / "pe.json"
    base = ["--set", str(b["set"]), "--out", str(out), "--device", "cpu"]
    assert ep.main(["--model", str(tmp_path / "no_model"), *base]) == 2
    empty = tmp_path / "empty_set"
    (empty / "train" / "normal").mkdir(parents=True)
    assert ep.main(["--model", str(stub_model), "--set", str(empty), "--out", str(out)]) == 2
    assert ep.main(["--model", str(stub_model), "--set", str(tmp_path / "absent"), "--out", str(out)]) == 2
    assert ep.main(["--model", str(stub_model), *base, "--seen", str(tmp_path / "absent")]) == 2
    assert ep.main(["--model", str(stub_model), *base, "--holdout", str(tmp_path / "absent.txt")]) == 2
    assert ep.main(["--model", str(stub_model), *base, "--matched", str(tmp_path / "absent.csv")]) == 2
    # a model without the abnormal class (collect_study_probs refuses)
    monkeypatch.setattr(ep.esl, "make_predictor", lambda d, device="cpu": (lambda img: {"tumor": 0.5}))
    assert ep.main(["--model", str(stub_model), *base]) == 2
    # a model dir that cannot be loaded

    def broken(d, device="cpu"):
        raise OSError("no weights")
    monkeypatch.setattr(ep.esl, "make_predictor", broken)
    assert ep.main(["--model", str(stub_model), *base]) == 2
    assert not out.exists()


def test_runs_as_script_from_repo_root(tmp_path):
    env = {"PYTHONPATH": "", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "scripts/eval_prospective.py", "--help"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--seen" in r.stdout
    r = subprocess.run([sys.executable, "scripts/eval_prospective.py", "--set", str(tmp_path / "absent"),
                        "--out", str(tmp_path / "pe.json")], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and not (tmp_path / "pe.json").exists()


# ------------------------------------------------------------------ the model pin

def _pin(model_dir: Path, content: bytes = b"weights-v1") -> None:
    """A model dir with one pinned file and a MANIFEST.json written by the release gate's own writer."""
    (model_dir / "model.safetensors").write_bytes(content)
    esl.write_manifest(model_dir, model_dir / "MANIFEST.json")


def test_model_pin_is_verified_and_recorded(tmp_path, stub_model, capsys):
    b = _batch(tmp_path)
    _pin(stub_model)
    out = tmp_path / "out.json"
    assert ep.main(["--model", str(stub_model), "--set", str(b["set"]), "--out", str(out), "--device", "cpu"]) == 0
    payload = json.loads(out.read_text())
    assert payload["model"]["manifest_verified"] is True
    assert payload["model"]["identity"] == esl.model_identity(stub_model) != "unknown"
    assert not any("MANIFEST.json" in n for n in payload["notes"])


def test_weights_that_do_not_match_the_pin_are_refused(tmp_path, stub_model, capsys):
    b = _batch(tmp_path)
    _pin(stub_model)
    (stub_model / "model.safetensors").write_bytes(b"other weights")       # swapped after the pin
    out = tmp_path / "out.json"
    rc = ep.main(["--model", str(stub_model), "--set", str(b["set"]), "--out", str(out), "--device", "cpu"])
    assert rc == ep.EXIT_UNUSABLE and not out.exists()
    assert "do not match MANIFEST.json" in capsys.readouterr().out
    assert PREDICTOR_CALLS == [], "a model that fails its pin must not even be loaded"


def test_model_without_a_manifest_is_scored_with_a_note(tmp_path, stub_model, capsys):
    b = _batch(tmp_path)
    out = tmp_path / "out.json"
    assert ep.main(["--model", str(stub_model), "--set", str(b["set"]), "--out", str(out), "--device", "cpu"]) == 0
    payload = json.loads(out.read_text())
    assert payload["model"]["manifest_verified"] is False
    assert any("no MANIFEST.json" in n for n in payload["notes"])
