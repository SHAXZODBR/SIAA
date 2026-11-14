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
  * --index (age_new): age parsing incl. month / week / day codes and the
    birth-date fallback; a score that IS the age (difference 0, age-matched
    AUC near 0.5) and one independent of age; missing ages; an index without
    the columns; the deterministic patient-level bootstrap; bands and the
    age rule on a hand-computed example
  * --priority (priority_new): counts with unknown rows and reasons; the JSON
    without the new arguments only gains two null keys and notes
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
                            "threshold_sweep_new", "findings_new", "age_new", "priority_new", "notes"}
    assert payload["kind"] == "prospective_eval" and payload["schema_version"] == 2
    assert payload["age_new"] is None and payload["priority_new"] is None      # no --index / --priority
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


# ------------------------------------------------------------------ age (--index) and priority (--priority)

INDEX_HEADER = ["study_folder", "patient_id", "patient_age", "patient_birth_date", "study_date", "modality"]
PRIORITY_HEADER = ["study_folder", "patient_id", "report_study_date", "priority", "reasons", "wm_grade", "focal",
                   "rules_version"]


def test_parse_age_codes_and_birth_date_fallback():
    assert ep.parse_age("045Y") == 45.0
    assert ep.parse_age("010M") == pytest.approx(10 / 12)
    assert ep.parse_age("003W") == pytest.approx(3 / 52)
    assert ep.parse_age("020D") == pytest.approx(20 / 365)
    assert ep.parse_age(" 006m ") == pytest.approx(0.5)
    assert ep.parse_age("", "20010315", "20310315") == pytest.approx(30.0, abs=0.01)
    assert ep.parse_age("45", "20010315", "20310315") == pytest.approx(30.0, abs=0.01)   # bad code → dates
    assert ep.parse_age("045Y", "20010315", "20310315") == 45.0                          # the code wins
    assert ep.parse_age("") is None and ep.parse_age("abcY") is None and ep.parse_age("Y") is None
    assert ep.parse_age("", "20310316", "20310315") is None                              # born after the study
    assert ep.parse_age("", "2001-03-15", "20310315") is None
    assert ep.parse_age("150Y") is None


def _age_set(ages_abn, ages_norm, scores_abn, scores_norm, pid_base=9001000):
    """(studies, table, ages): one patient per study, study folder 'F<pid>'."""
    studies, table, ages = {}, {}, {}
    for cls, ag, sc, off in (("abnormal", ages_abn, scores_abn, 0), ("normal", ages_norm, scores_norm, 5000)):
        for i, (a, s) in enumerate(zip(ag, sc)):
            key = str(pid_base + off + i)
            studies[(key, cls)] = [s]
            table[key] = {"study_folder": f"F{key}", "label": "", "cls": cls}
            if a is not None:
                ages[f"F{key}"] = a
    return studies, table, ages


def test_age_block_hand_computed_bands_rule_and_matched_auc():
    abn_age, abn_sc = [30, 45, 50, 65], [0.7, 0.9, 0.3, 0.6]        # flagged T T F T → model sens 0.75
    nrm_age, nrm_sc = [25, 35, 42, 70], [0.1, 0.55, 0.95, 0.2]      # flagged F T T F
    st, table, ages = _age_set(abn_age, nrm_age, abn_sc, nrm_sc)
    a = ep.age_block(st, table, ages)
    assert (a["n_with_age"], a["n_without_age"]) == (8, 0)
    assert a["median_age"] == {"normal": 38.5, "abnormal": 47.5}
    assert a["auc_model"] == pytest.approx(ep.auc(abn_sc, nrm_sc))
    assert a["auc_age_alone"] == pytest.approx(ep.auc(abn_age, nrm_age))
    # pairs within 5 years: 30-25 win, 30-35 win, 45-42 loss (0.9 < 0.95), 65-70 win
    assert a["age_matched_pairs"] == 4 and a["auc_age_matched"] == pytest.approx(0.75)
    assert a["age_match_window_years"] == 5
    # 'age >= T' sensitivity: 1.0 up to 30, 0.75 for 31..45, 0.5 for 46..50 → oldest T keeping 0.75 is 45
    assert a["age_rule_at_model_sensitivity"] == {"age_from": 45, "sensitivity": 0.75, "specificity": 0.75}
    assert a["bands"] == [
        {"band": "under 40", "n_abnormal": 1, "abnormal_flagged": 1, "n_normal": 2, "normal_flagged": 1,
         "sensitivity": 1.0, "specificity": 0.5},
        {"band": "40-59", "n_abnormal": 2, "abnormal_flagged": 1, "n_normal": 1, "normal_flagged": 1,
         "sensitivity": 0.5, "specificity": 0.0},
        {"band": "60+", "n_abnormal": 1, "abnormal_flagged": 1, "n_normal": 1, "normal_flagged": 0,
         "sensitivity": 1.0, "specificity": 1.0}]
    assert a["normal_reports_aged_60_plus"] == 1
    assert a["auc_difference_ci95"] is not None                     # 4 patients per class


def test_age_bands_with_empty_denominators_and_band_edges():
    st, table, ages = _age_set([39.99, 40.0, 59.99], [60.0], [0.9, 0.9, 0.9], [0.9])
    a = ep.age_block(st, table, ages)
    bands = {b["band"]: b for b in a["bands"]}
    assert (bands["under 40"]["n_abnormal"], bands["40-59"]["n_abnormal"], bands["60+"]["n_abnormal"]) == (1, 2, 0)
    assert bands["under 40"]["specificity"] is None and bands["60+"]["sensitivity"] is None
    assert bands["60+"]["specificity"] == 0.0 and a["normal_reports_aged_60_plus"] == 1
    assert a["auc_difference_ci95"] is None                         # < 2 patients per class
    only_normal = ep.age_block(*_age_set([], [30.0, 50.0], [], [0.2, 0.7]))
    assert only_normal["age_rule_at_model_sensitivity"] is None and only_normal["auc_model"] is None


def test_score_that_is_the_age_matches_age_alone():
    """A 'model' that only learned age: AUC difference exactly 0, and between patients of similar age it
    is close to a coin flip."""
    rng = np.random.default_rng(11)
    ages = rng.uniform(20, 80, 400)
    is_abn = rng.random(400) < 1 / (1 + np.exp(-(ages - 50) / 8))   # older → more often 'abnormal'
    abn_age = [float(a) for a, y in zip(ages, is_abn) if y]
    nrm_age = [float(a) for a, y in zip(ages, is_abn) if not y]
    st, table, amap = _age_set(abn_age, nrm_age, [a / 100 for a in abn_age], [a / 100 for a in nrm_age])
    a = ep.age_block(st, table, amap)
    assert a["auc_age_alone"] > 0.8
    assert a["auc_model"] == pytest.approx(a["auc_age_alone"])
    assert a["auc_difference_ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert 0.4 < a["auc_age_matched"] < 0.65 and a["age_matched_pairs"] > 1000
    brute = sum(1 for x in abn_age for y in nrm_age if abs(x - y) <= 5.0)
    assert a["age_matched_pairs"] == brute


def test_score_independent_of_age_beats_age_alone():
    rng = np.random.default_rng(12)
    abn_age, nrm_age = list(rng.uniform(20, 80, 150)), list(rng.uniform(20, 80, 150))
    abn_sc = list(np.clip(rng.normal(0.7, 0.15, 150), 0, 1))
    nrm_sc = list(np.clip(rng.normal(0.3, 0.15, 150), 0, 1))
    a = ep.age_block(*_age_set(abn_age, nrm_age, abn_sc, nrm_sc))
    assert 0.38 < a["auc_age_alone"] < 0.62 and a["auc_model"] > 0.9
    assert a["auc_age_matched"] == pytest.approx(a["auc_model"], abs=0.05)
    lo, hi = a["auc_difference_ci95"]
    assert 0 < lo < a["auc_model"] - a["auc_age_alone"] < hi


def test_month_coded_infants_and_missing_ages(tmp_path):
    idx = tmp_path / "idx.csv"
    _csv(idx, INDEX_HEADER, [["F9001000", "9001000", "006M", "", "20310101", "MR"],
                             ["F9001001", "9001001", "003W", "", "20310101", "MR"],
                             ["F9001001", "9001001", "040Y", "", "20310101", "MR"],    # first usable row wins
                             ["F9006000", "9006000", "020D", "", "20310101", "MR"],
                             ["F9006001", "9006001", "", "", "", "MR"],                  # no age at all
                             ["F9006002", "9006002", "", "19910101", "20310101", "MR"]])
    ages, note = ep.load_index_ages(idx)
    assert note is None and set(ages) == {"F9001000", "F9001001", "F9006000", "F9006002"}
    assert ages["F9001000"] == pytest.approx(0.5) and ages["F9001001"] == pytest.approx(3 / 52)
    st, table, _ = _age_set([None, None], [None, None, None], [0.9, 0.2], [0.6, 0.1, 0.3])
    st[("9006003", "normal")] = [0.3]                                # a 'new' study without a studies.csv row
    a = ep.age_block(st, table, ages)
    assert (a["n_with_age"], a["n_without_age"]) == (4, 2)
    assert a["bands"][0]["n_abnormal"] == 2 and a["bands"][0]["n_normal"] == 1
    assert a["bands"][1]["n_normal"] == 1                            # 40 years from the birth date
    assert a["median_age"]["abnormal"] == pytest.approx((0.5 + 3 / 52) / 2)


def test_index_without_columns_or_unreadable_is_a_note(tmp_path):
    bad = tmp_path / "idx.csv"
    _csv(bad, ["study_folder", "modality"], [["F1", "MR"]])
    ages, note = ep.load_index_ages(bad)
    assert ages is None and "age_new omitted" in note
    _csv(bad, ["patient_age", "study_date"], [["045Y", "20310101"]])
    assert ep.load_index_ages(bad)[0] is None
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"\xff\xfe\x00bad")
    ages, note = ep.load_index_ages(raw)
    assert ages is None and "unreadable" in note
    bad_p = tmp_path / "p.csv"
    _csv(bad_p, ["study_folder", "reasons"], [["F1", "mass"]])
    prio, version, note = ep.load_priority(bad_p)
    assert prio is None and "priority_new omitted" in note


def test_bootstrap_is_deterministic_and_patient_level():
    rows = [(f"{9002000 + i // 2}", float(s), float(a), bool(y)) for i, (s, a, y) in
            enumerate(zip(np.linspace(0, 1, 40), np.linspace(20, 80, 40)[::-1], [i % 3 == 0 for i in range(40)]))]
    ci1, ci2 = ep.auc_difference_ci(rows), ep.auc_difference_ci(rows)
    assert ci1 == ci2 and ci1[0] <= ci1[1]
    assert ep.auc_difference_ci(rows, seed=ep.BOOTSTRAP_SEED + 1) != ci1
    # two patients with abnormal studies but only ONE normal patient → no interval
    few = [("9002100", 0.9, 50.0, True), ("9002101", 0.8, 40.0, True),
           ("9002102", 0.1, 30.0, False), ("9002102_s2", 0.2, 31.0, False)]
    few = [(ep.esl.patient_id_of(p), s, a, y) for p, s, a, y in few]
    assert ep.auc_difference_ci(few) is None


def test_auc_np_equals_auc():
    rng = np.random.default_rng(5)
    s = np.round(rng.random(60), 1)
    y = rng.random(60) < 0.4
    assert ep._auc_np(s, y) == pytest.approx(ep.auc(list(s[y]), list(s[~y])))
    assert ep._auc_np(s, np.zeros(60, dtype=bool)) is None


def test_priority_block_counts_unknown_and_reasons(tmp_path):
    pf = tmp_path / "prio.csv"
    _csv(pf, PRIORITY_HEADER, [
        ["F9003000", "9003000", "20310101", "yes", "mass;recent_infarct", "1", "yes", "rules-v1"],
        ["F9003001", "9003001", "20310101", "yes", "mass", "0", "yes", "rules-v1"],
        ["F9003002", "9003002", "20310101", "no", "", "2", "no", "rules-v1"],
        ["F9008000", "9008000", "20310101", "no", "", "0", "no", "rules-v1"],
        ["F9008001", "9008001", "20310101", "", "", "", "", "rules-v1"],
        ["F9008002", "9008002", "20310101", "no", "", "0", "no", "rules-v1"],
        ["F9008002", "9008002", "20310101", "yes", "haemorrhage", "0", "yes", "rules-v1"],   # yes wins
    ])
    prio, version, note = ep.load_priority(pf)
    assert note is None and version == "rules-v1" and prio["F9008002"] == ("yes", ["haemorrhage"])
    st, table, ages = _age_set([60, 30, 70], [20, 25, 50, 33], [0.9, 0.3, 0.8], [0.6, 0.1, 0.7, 0.2],
                               pid_base=9003000)
    p = ep.priority_block(st, table, prio, version, None)
    # yes: F9003000 (0.9 flagged), F9003001 (0.3), F9008002 (0.7 flagged); no: F9003002 (0.8 flagged), F9008000
    # (0.6 flagged); unknown: F9008001 ('') and F9008003 (no row)
    assert (p["n_priority"], p["priority_flagged"], p["n_other"], p["other_flagged"], p["n_unknown"]) == \
        (3, 2, 2, 2, 2)
    assert p["by_reason"] == {"mass": {"n": 2, "flagged": 1}, "haemorrhage": {"n": 1, "flagged": 1},
                              "recent_infarct": {"n": 1, "flagged": 1}}
    assert p["auc_model"] == pytest.approx(ep.auc([0.9, 0.3, 0.7], [0.8, 0.6]))
    assert p["auc_age_alone"] is None and p["rules_version"] == "rules-v1"
    p2 = ep.priority_block(st, table, prio, version, ages)
    assert p2["auc_age_alone"] == pytest.approx(ep.auc([60, 30, 50], [70, 20]))


def _index_and_priority(tmp_path: Path) -> tuple[Path, Path]:
    """For _batch: new studies J (abn 0.9, 45 y), K (abn 0.3, ~45.15 y from dates), C (normal 0.6, 6 months),
    D (normal 0.1, no index row)."""
    idx = tmp_path / "batch_x_index.csv"
    _csv(idx, INDEX_HEADER, [["STUDY_J", "9000010", "045Y", "", "20310105", "MR"],
                             ["STUDY_K", "9000010", "", "19860105", "20310301", "MR"],
                             ["STUDY_C", "90000101", "006M", "", "20310110", "MR"],
                             ["STUDY_A", "9000001", "070Y", "", "20310112", "MR"]])
    pf = tmp_path / "batch_x_priority.csv"
    _csv(pf, PRIORITY_HEADER, [["STUDY_J", "9000010", "20310105", "yes", "mass", "1", "yes", "rules-v1"],
                               ["STUDY_K", "9000010", "20310301", "no", "", "1", "no", "rules-v1"],
                               ["STUDY_C", "90000101", "20310110", "", "", "", "", "rules-v1"],
                               ["STUDY_D", "9000020", "20310111", "yes", "haemorrhage;mass", "0", "yes", "rules-v1"],
                               ["STUDY_A", "9000001", "20310112", "yes", "mass", "0", "yes", "rules-v1"]])
    return idx, pf


def test_end_to_end_with_index_and_priority(stub_model, tmp_path, capsys):
    b = _batch(tmp_path)
    idx, pf = _index_and_priority(tmp_path)
    out = tmp_path / "pe.json"
    assert ep.main(_argv(b, stub_model, out) + ["--index", str(idx), "--priority", str(pf)]) == 0
    stdout = capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    a = payload["age_new"]
    assert (a["n_with_age"], a["n_without_age"]) == (3, 1)
    assert a["auc_model"] == 0.5 and a["auc_age_alone"] == 1.0
    assert a["age_rule_at_model_sensitivity"] == {"age_from": 45, "sensitivity": 1.0, "specificity": 1.0}
    assert a["bands"][0]["normal_flagged"] == 1 and a["bands"][0]["specificity"] == 0.0
    assert a["bands"][1]["abnormal_flagged"] == 1 and a["bands"][1]["n_abnormal"] == 2
    assert a["normal_reports_aged_60_plus"] == 0                    # the 70-year-old is seen, not new
    assert any("no usable age" in n for n in payload["notes"])
    p = payload["priority_new"]
    assert (p["n_priority"], p["priority_flagged"], p["n_other"], p["other_flagged"], p["n_unknown"]) == \
        (2, 1, 1, 0, 1)
    assert p["by_reason"] == {"mass": {"n": 2, "flagged": 1}, "haemorrhage": {"n": 1, "flagged": 0}}
    assert p["auc_model"] == 0.5 and p["auc_age_alone"] == 0.0 and p["rules_version"] == "rules-v1"
    assert "age (never-seen)" in stdout and "priority findings (never-seen)" in stdout
    assert "age >= 45" in stdout
    _assert_no_ids(stdout, b)
    _assert_no_ids(text, b)


def test_new_arguments_change_no_existing_number(stub_model, tmp_path):
    """Without --index / --priority the payload only gains two null keys and two notes; with them, every
    existing key keeps its value."""
    b = _batch(tmp_path)
    idx, pf = _index_and_priority(tmp_path)
    plain = ep.evaluate(stub_model, b["set"], seen=[str(b["seen"])], holdout=b["holdout"], matched=b["matched"])
    full = ep.evaluate(stub_model, b["set"], seen=[str(b["seen"])], holdout=b["holdout"], matched=b["matched"],
                       index=idx, priority=pf)
    assert plain["age_new"] is None and plain["priority_new"] is None
    assert sum(1 for n in plain["notes"] if "no --index" in n or "no --priority" in n) == 2
    assert (plain["inputs"]["index"], plain["inputs"]["priority"]) == (None, None)
    assert (full["inputs"]["index"], full["inputs"]["priority"]) == (str(idx), str(pf))
    strip = lambda d: {k: ({kk: vv for kk, vv in v.items() if kk not in ("index", "priority")} if k == "inputs" else v)
                       for k, v in d.items() if k not in ("age_new", "priority_new", "notes")}
    assert strip(plain) == strip(full)
    assert [n for n in plain["notes"] if "--index" not in n and "--priority" not in n] == \
        [n for n in full["notes"] if "--index" not in n and "--priority" not in n and "no usable age" not in n]


def test_bom_in_index_and_priority_is_accepted(stub_model, tmp_path):
    """A csv re-saved by Excel starts with a UTF-8 BOM: the study_folder column must still be found,
    or the age baseline silently disappears behind a misleading 'no study_folder column' note."""
    b = _batch(tmp_path)
    idx, pf = _index_and_priority(tmp_path)
    plain = ep.evaluate(stub_model, b["set"], seen=[str(b["seen"])], holdout=b["holdout"], matched=b["matched"],
                        index=idx, priority=pf)
    for fp in (idx, pf):
        fp.write_bytes(b"\xef\xbb\xbf" + fp.read_bytes())
    bom = ep.evaluate(stub_model, b["set"], seen=[str(b["seen"])], holdout=b["holdout"], matched=b["matched"],
                      index=idx, priority=pf)
    assert bom["age_new"] is not None and bom["priority_new"] is not None
    assert bom["age_new"] == plain["age_new"] and bom["priority_new"] == plain["priority_new"]
    assert not any("omitted" in n for n in bom["notes"])


def test_index_and_priority_failures(stub_model, tmp_path):
    b = _batch(tmp_path)
    out = tmp_path / "pe.json"
    assert ep.main(_argv(b, stub_model, out) + ["--index", str(tmp_path / "absent.csv")]) == 2
    assert ep.main(_argv(b, stub_model, out) + ["--priority", str(tmp_path / "absent.csv")]) == 2
    assert not out.exists()
    bad = tmp_path / "bad.csv"
    _csv(bad, ["modality"], [["MR"]])
    assert ep.main(_argv(b, stub_model, out) + ["--index", str(bad), "--priority", str(bad)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["age_new"] is None and payload["priority_new"] is None
    assert any("age_new omitted" in n for n in payload["notes"])
    assert any("priority_new omitted" in n for n in payload["notes"])
    # no studies.csv → no study key → folder map: a note, not a crash
    b2 = _batch(tmp_path / "b2", with_studies_csv=False)
    idx, pf = _index_and_priority(tmp_path)
    payload = ep.evaluate(stub_model, b2["set"], index=idx, priority=pf)
    assert payload["age_new"] is None and payload["priority_new"] is None
    assert sum(1 for n in payload["notes"] if "no study key" in n) == 2
