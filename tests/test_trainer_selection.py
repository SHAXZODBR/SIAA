"""training/finetune_brain_classifier.py — what "best epoch" means, augmentation and the seed.

Model-free unit tests for the study-level selection rule, plus tiny end-to-end runs (a 1-layer
ViT made on the spot, 32-px synthetic PNGs, CPU). No patient data.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

TRAINER = REPO / "training" / "finetune_brain_classifier.py"


def _trainer():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("finetune_brain_classifier", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ study-level metrics

def test_study_key_is_the_file_name_without_the_slice_index():
    tr = _trainer()
    assert tr.study_key(Path("x/abnormal/9000001_3.png")) == "9000001"
    assert tr.study_key(Path("9000001_s2_0.png")) == "9000001_s2"
    assert tr.study_key(Path("9000001_new1_4.png")) == "9000001_new1"


def test_study_metrics_group_slices_and_apply_the_shipped_threshold():
    tr = _trainer()
    # class index 0 = abnormal. Four studies:
    #   A (abnormal) slices .9 .7      mean .80  -> flagged  TP
    #   B (abnormal) slices .6 .2      mean .40  -> missed   FN
    #   C (normal)   slices .5 .5      mean .50  -> flagged  FP   (>= threshold flags)
    #   D (normal)   slices .1 .3 .2   mean .20  -> clear    TN
    samples = [(Path("a/9000001_0.png"), 0), (Path("a/9000001_1.png"), 0),
               (Path("a/9000002_0.png"), 0), (Path("a/9000002_1.png"), 0),
               (Path("n/9000003_0.png"), 1), (Path("n/9000003_1.png"), 1),
               (Path("n/9000004_0.png"), 1), (Path("n/9000004_1.png"), 1), (Path("n/9000004_2.png"), 1)]
    probs = [.9, .7, .6, .2, .5, .5, .1, .3, .2]
    m = tr.study_metrics(samples, probs, abnormal_idx=0)
    assert (m["n_abnormal"], m["n_normal"], m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 2, 1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5 and m["balanced_accuracy"] == 0.5
    # study means: abnormal .80 .40 ; normal .50 .20 -> pairs won 1 + 0 + 1 + 1 of 4 ... (.40 < .50)
    assert m["auc"] == pytest.approx(0.75) and m["threshold"] == 0.5
    # the same study key under the two classes is two studies (it cannot happen in a release set)
    empty = tr.study_metrics([(Path("a/9000001_0.png"), 0)], [0.9], abnormal_idx=0)
    assert empty["specificity"] is None and empty["auc"] is None and empty["balanced_accuracy"] is None


def test_auc_counts_ties_as_half():
    tr = _trainer()
    assert tr._auc([0.5], [0.5]) == 0.5 and tr._auc([0.9, 0.5], [0.5, 0.1]) == pytest.approx(0.875)
    assert tr._auc([], [0.1]) is None


# ------------------------------------------------------------------ the selection rule

def _key(tr, sens, spec, loss=1.0, acc=0.5, min_sens=0.90):
    return tr.selection_key("study", acc, loss, {"sensitivity": sens, "specificity": spec}, min_sens)


def test_study_rule_puts_sensitivity_first_then_specificity():
    tr = _trainer()
    # among epochs that reach the sensitivity floor, the most specific wins — not the most sensitive
    assert _key(tr, 0.91, 0.60) > _key(tr, 0.98, 0.40) > _key(tr, 0.90, 0.39)
    # an epoch below the floor never beats one that reaches it, whatever its specificity
    assert _key(tr, 0.90, 0.10) > _key(tr, 0.89, 0.95)
    # below the floor: the most sensitive, then the most specific
    assert _key(tr, 0.85, 0.20) > _key(tr, 0.80, 0.90) and _key(tr, 0.85, 0.30) > _key(tr, 0.85, 0.20)
    # everything else equal: the lower selection loss
    assert _key(tr, 0.92, 0.50, loss=0.4) > _key(tr, 0.92, 0.50, loss=0.9)
    # a split without normal (or abnormal) studies cannot outrank a measurable epoch
    assert tr.selection_key("study", 0.9, 0.1, {"sensitivity": None, "specificity": None}, 0.9) < _key(tr, 0.5, 0.5)
    assert tr.selection_key("study", 0.9, 0.1, None, 0.9) < _key(tr, 0.5, 0.5)


def test_slice_accuracy_rule_is_the_accuracy_alone():
    tr = _trainer()
    k = tr.selection_key("slice_acc", 0.75, 9.9, {"sensitivity": 0.1, "specificity": 0.1}, 0.9)
    assert k == (0.75,) and tr.selection_key("slice_acc", 0.76, 0.0, None, 0.9) > k


# ------------------------------------------------------------------ dataset / augmentation

def test_augmentation_is_applied_to_the_training_dataset_only_and_samples_are_sorted(tmp_path: Path):
    tr = _trainer()
    from PIL import Image
    for name in ("9000002_0.png", "9000001_1.png", "9000001_0.png"):
        d = tmp_path / "train" / "abnormal"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (10, 20, 30)).save(d / name)
    seen = []

    class Proc:
        def __call__(self, images, return_tensors):
            import torch
            seen.append(images.size)
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}
    marked = []
    ds = tr.BrainImageDataset(tmp_path / "train", Proc(), ["abnormal", "normal"],
                              augment=lambda im: (marked.append(1), im.resize((8, 8)))[1])
    assert [p.name for p, _ in ds.samples] == ["9000001_0.png", "9000001_1.png", "9000002_0.png"]
    ds[0]
    assert marked == [1] and seen[-1] == (8, 8)
    plain = tr.BrainImageDataset(tmp_path / "train", Proc(), ["abnormal", "normal"])
    plain[0]
    assert marked == [1] and seen[-1] == (40, 30)
    aug = tr.build_augmentation()
    out = aug(Image.new("RGB", (40, 30), (120, 120, 120)))
    assert out.size == (40, 30) and out.mode == "RGB"


# ------------------------------------------------------------------ end to end (tiny ViT, CPU)

def _tiny_vit(dst: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ViTConfig(image_size=32, patch_size=16, hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=16, num_labels=2,
                                 id2label={0: "abnormal", 1: "normal"}, label2id={"abnormal": 0, "normal": 1})
    transformers.ViTForImageClassification(cfg).save_pretrained(dst)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(dst)
    return dst


def _png_set(root: Path) -> Path:
    from PIL import Image
    for sp in ("train", "tune"):
        for ci, cls in enumerate(("abnormal", "normal")):
            d = root / sp / cls
            d.mkdir(parents=True)
            for study in range(3):
                for k in range(2):
                    Image.new("RGB", (32, 32), (40 + 150 * ci, 20 * study, 10 * k)).save(
                        d / f"{9000100 + 10 * ci + study}_{k}.png")
    return root


def _run(tmp_path: Path, out: str, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    base = tmp_path / "base"
    if not base.exists():
        _tiny_vit(base)
    data = tmp_path / "set"
    if not data.exists():
        _png_set(data)
    cand = tmp_path / out
    r = subprocess.run([sys.executable, str(TRAINER), "--data", str(data), "--classes", "abnormal,normal",
                        "--base-model", str(base), "--epochs", "2", "--batch-size", "4", "--device", "cpu",
                        "--balance", "--output", str(cand), *extra],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    m = json.loads((cand / "training_metrics.json").read_text()) if (cand / "training_metrics.json").is_file() else {}
    return r, m


def test_default_run_still_selects_by_slice_accuracy_and_records_the_study_numbers(tmp_path: Path):
    _trainer()
    r, m = _run(tmp_path, "cand_default")
    assert r.returncode == 0, r.stdout + r.stderr
    sel = m["checkpoint_selection"]
    assert sel["metric"] == "slice accuracy" and sel["select_metric"] == "slice_acc" and "best_study" not in sel
    assert m["augment"] is False and m["seed"] is None
    for ep in m["epochs"]:
        s = ep["study"]
        assert (s["n_abnormal"], s["n_normal"]) == (3, 3) and 0.0 <= s["sensitivity"] <= 1.0
    best = max(m["epochs"], key=lambda e: (e["select_acc"], -e["epoch"]))
    assert sel["best_epoch"] == best["epoch"] and sel["best_acc"] == best["select_acc"]


def test_study_rule_run_records_the_rule_and_saves_the_epoch_it_names(tmp_path: Path):
    tr = _trainer()
    r, m = _run(tmp_path, "cand_study", "--select-metric", "study", "--select-min-sens", "0.9",
                "--augment", "--seed", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    sel = m["checkpoint_selection"]
    assert sel["select_metric"] == "study" and sel["min_sensitivity"] == 0.9
    assert "sensitivity >= 0.90 first, then specificity" in sel["metric"]
    assert m["augment"] is True and m["seed"] == 3 and "[augment]" in r.stdout
    keys = [tr.selection_key("study", e["select_acc"], e["select_loss"], e["study"], 0.9) for e in m["epochs"]]
    expected = keys.index(max(keys)) + 1                       # the first of equal epochs is kept
    assert sel["best_epoch"] == expected and sel["best_study"] == m["epochs"][expected - 1]["study"]
    assert (tmp_path / "cand_study" / "model.safetensors").is_file()


def test_the_seed_makes_a_run_repeatable(tmp_path: Path):
    _trainer()
    r1, m1 = _run(tmp_path, "cand_a", "--augment", "--seed", "11")
    r2, m2 = _run(tmp_path, "cand_b", "--augment", "--seed", "11")
    assert r1.returncode == 0 and r2.returncode == 0, r1.stdout + r1.stderr + r2.stdout + r2.stderr
    strip = lambda m: [(e["epoch"], round(e["train_loss"], 5), round(e["select_loss"], 5)) for e in m["epochs"]]
    assert strip(m1) == strip(m2)


def test_study_rule_needs_the_abnormal_class(tmp_path: Path):
    _trainer()
    base = _tiny_vit(tmp_path / "base")
    data = _png_set(tmp_path / "set")
    r = subprocess.run([sys.executable, str(TRAINER), "--data", str(data), "--classes", "abnormal,normal",
                        "--base-model", str(base), "--epochs", "1", "--device", "cpu", "--output",
                        str(tmp_path / "cand"), "--select-metric", "study", "--abnormal-class", "tumor"],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "--select-metric study needs the class 'tumor'" in (r.stdout + r.stderr)
