"""Release gate + compare + promote (stages 7-8 of docs/dev/release_pipeline_contract.md).

Nothing here touches the real 343 MB model or any patient data: a TINY ViT
(hidden 32, one layer, 32 px) is trained for ~60 Adam steps on synthetic
bright-vs-dark slices (bright = abnormal), which makes the study-level gate
outcome deterministic. The eval CLI (scripts/eval_study_level.py) and the
promote CLI (scripts/promote_model.py) are driven in subprocesses exactly the
way the runbook does, and the pure scoring helpers are unit-tested directly.

Covered:
  * Wilson 95% CI against known values (118/131, 22/47 == the docs' 0.33–0.61)
  * score_studies / evaluate_studies / decide_promotion without a model
  * manifest write → verify → tamper → exit 3
  * gate fail → exit 2
  * --compare-with: same model as old → promotion computed, exit 0;
    worse candidate → promotion.promote false, exit 4
  * eval JSON records the protocol (mean @0.50, 5 central slices,
    registry-predictor native-res preprocessing), n_patients,
    val_patients_sha256 and holdout coverage
  * promote_model: happy path (backup + move + registry probe + log),
    refusal (gate failed / promote false / no eval / floors lowered /
    non-production gate or preprocessing / wrong val set / holdout not
    covered), --force overrides ONLY promotion.promote false (never a failed
    gate or lowered floors), combined-set val sha check via
    dataset_manifest.json, rollback when the registry cannot load a
    corrupted candidate.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO

EVAL = REPO / "scripts" / "eval_study_level.py"
PROMOTE = REPO / "scripts" / "promote_model.py"
PY = sys.executable


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_study_level", EVAL)


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_MODEL_DIR_OVERRIDE_")}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LOGURU_LEVEL": "WARNING",
                "PYTHONPATH": str(REPO)})
    return env


def _run(*argv: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *map(str, argv)], cwd=REPO, env=_env(),
                          capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------ fixtures

def _synthetic_slice(rng: np.random.Generator, bright: bool) -> np.ndarray:
    base = 190 if bright else 60
    return np.clip(rng.normal(base, 25, (32, 32, 3)), 0, 255).astype(np.uint8)


def _processor():
    """The real ViT processor from the offline HF cache when present (resized to
    32 px), else a ViTImageProcessor built directly — never a network call."""
    from transformers import AutoImageProcessor, ViTImageProcessor
    try:
        proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", local_files_only=True)
        proc.size = {"height": 32, "width": 32}
        return proc
    except Exception:
        return ViTImageProcessor(size={"height": 32, "width": 32})


@pytest.fixture(scope="session")
def tiny_model_src(tmp_path_factory) -> Path:
    """Tiny ViT trained (seconds, CPU) so bright slices → 'abnormal'."""
    import torch
    from transformers import ViTConfig, ViTForImageClassification
    torch.manual_seed(0)
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                    image_size=32, patch_size=8, num_labels=2,
                    id2label={0: "abnormal", 1: "normal"}, label2id={"abnormal": 0, "normal": 1})
    model = ViTForImageClassification(cfg)
    proc = _processor()
    rng = np.random.default_rng(0)
    imgs = [_synthetic_slice(rng, i % 2 == 0) for i in range(64)]
    px = proc(images=imgs, return_tensors="pt")["pixel_values"]
    y = torch.tensor([0 if i % 2 == 0 else 1 for i in range(64)])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(60):
        loss = model(pixel_values=px, labels=y).loss
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(pixel_values=px).logits.argmax(-1) == y).float().mean().item()
    assert acc >= 0.95, f"tiny model did not learn bright-vs-dark (acc {acc})"
    d = tmp_path_factory.mktemp("tiny_model") / "brain_triage_tiny"
    model.save_pretrained(d)
    proc.save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def degraded_model_src(tmp_path_factory, tiny_model_src) -> Path:
    """Same architecture, classifier bias forced to 'abnormal' for everything:
    sens 1.0 / spec 0.0 — a different identity that is clearly WORSE."""
    import torch
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(str(tiny_model_src), local_files_only=True)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    d = tmp_path_factory.mktemp("degraded_model") / "brain_triage_degraded"
    model.save_pretrained(d)
    shutil.copy(tiny_model_src / "preprocessor_config.json", d / "preprocessor_config.json")
    return d


@pytest.fixture(scope="session")
def val_set(tmp_path_factory) -> Path:
    """<val>/{abnormal,normal}/<pid>_<k>.png — 8 abnormal (one dark → FN),
    6 normal (one bright → FP) patients, 3 slices each."""
    from PIL import Image
    root = tmp_path_factory.mktemp("val_set") / "val"
    rng = np.random.default_rng(1)
    plan = {"abnormal": [(f"A{i:02d}", i != 7) for i in range(8)],      # A07 is dark → missed
            "normal": [(f"N{i:02d}", i == 5) for i in range(6)]}         # N05 is bright → false alarm
    for cls, pids in plan.items():
        (root / cls).mkdir(parents=True)
        for pid, bright in pids:
            for k in range(3):
                Image.fromarray(_synthetic_slice(rng, bright)).save(root / cls / f"{pid}_{k}.png")
    return root


@pytest.fixture(scope="session")
def holdout_file(tmp_path_factory) -> Path:
    """A synthetic locked holdout: three patients that ARE in val_set."""
    fp = tmp_path_factory.mktemp("holdout") / "holdout_patients.txt"
    fp.write_text("# synthetic holdout\nA00\nA07\nN05\n")
    return fp


def _val_sha(val_root: Path) -> str:
    import hashlib
    pids = sorted({f.stem.split("_", 1)[0] for f in val_root.glob("*/*.png")})
    return hashlib.sha256("\n".join(pids).encode("utf-8")).hexdigest()


def _fresh_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ------------------------------------------------------------------ pure helpers

def test_wilson_ci_known_values():
    lo, hi = ev.wilson_ci(118, 131)                       # the validated model's sensitivity
    assert abs(lo - 0.84) < 0.01 and abs(hi - 0.94) < 0.01, (lo, hi)
    lo, hi = ev.wilson_ci(22, 47)                         # specificity → docs quote 0.33–0.61
    assert round(lo, 2) == 0.33 and round(hi, 2) == 0.61, (lo, hi)
    assert ev.wilson_ci(10, 10) == pytest.approx([0.7225, 1.0], abs=1e-3)
    assert ev.wilson_ci(0, 10)[0] == 0.0
    assert ev.wilson_ci(0, 0) == [0.0, 1.0]
    lo, hi = ev.wilson_ci(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9


def _studies() -> dict:
    return {("p1", "abnormal"): [0.9, 0.8, 0.7], ("p2", "abnormal"): [0.2, 0.3, 0.9],   # p2: mean .47 miss, max hit
            ("p3", "abnormal"): [0.6, 0.6, 0.6],
            ("n1", "normal"): [0.1, 0.2, 0.1], ("n2", "normal"): [0.7, 0.6, 0.55],    # n2: false alarm
            ("n3", "normal"): [0.4, 0.45, 0.3]}


def test_score_studies_without_a_model():
    r = ev.score_studies(_studies(), ev.gate_fn("mean", 0.5), "mean @0.5")
    assert (r["tp"], r["fn"], r["tn"], r["fp"]) == (2, 1, 2, 1)
    assert r["sens"] == pytest.approx(2 / 3) and r["spec"] == pytest.approx(2 / 3)
    assert r["balanced"] == pytest.approx(2 / 3) and r["acc"] == pytest.approx(4 / 6)
    assert r["sens_ci"] == ev.wilson_ci(2, 3) and r["spec_ci"] == ev.wilson_ci(2, 3)
    r_max = ev.score_studies(_studies(), ev.gate_fn("max", 0.5))
    assert (r_max["tp"], r_max["fn"]) == (3, 0)
    r_vote = ev.score_studies(_studies(), ev.gate_fn("vote", 0.5))
    assert (r_vote["tp"], r_vote["fn"], r_vote["tn"], r_vote["fp"]) == (2, 1, 2, 1)


def test_evaluate_studies_gate_and_floors(capsys):
    results, best, gate = ev.evaluate_studies(_studies(), min_sens=0.5, min_spec=0.5, verbose=False)
    assert [r["name"] for r in results] == ["mean-prob @0.5", "max-prob @0.5", "majority vote"]
    assert gate["passed"] is True and gate["min_sens"] == 0.5 and "sens_ci" in gate
    assert 0.1 <= best["threshold"] <= 0.9 and best["balanced"] >= gate["balanced"]
    _, _, gate2 = ev.evaluate_studies(_studies(), min_sens=0.85, min_spec=0.40, verbose=True)
    assert gate2["passed"] is False
    assert "GATE FAILED" in capsys.readouterr().out


def test_decide_promotion_rule():
    old = {"sens": 0.90, "spec": 0.47, "balanced": 0.685, "passed": True}
    ok = ev.decide_promotion(old, {**old, "sens": 0.88, "spec": 0.55, "balanced": 0.715, "passed": True})
    assert ok["promote"] is True and ok["delta_sens"] == pytest.approx(-0.02)
    big_sens_drop = ev.decide_promotion(old, {"sens": 0.86, "spec": 0.70, "balanced": 0.78, "passed": True})
    assert big_sens_drop["promote"] is False and "sens" in big_sens_drop["reason"]
    worse_bal = ev.decide_promotion(old, {"sens": 0.90, "spec": 0.44, "balanced": 0.67, "passed": True})
    assert worse_bal["promote"] is False and "balanced" in worse_bal["reason"]
    gate_fail = ev.decide_promotion(old, {"sens": 0.99, "spec": 0.30, "balanced": 0.645, "passed": False,
                                          "min_sens": 0.85, "min_spec": 0.40})
    assert gate_fail["promote"] is False and "gate failed" in gate_fail["reason"]
    tie = ev.decide_promotion(old, {**old, "balanced": 0.68})       # −0.005 within tolerance
    assert tie["promote"] is True


def test_exit_codes_are_pinned():
    assert (ev.EXIT_GATE_FAILED, ev.EXIT_MANIFEST_FAILED, ev.EXIT_NOT_PROMOTED) == (2, 3, 4)


# ------------------------------------------------------------------ eval CLI

def test_eval_cli_manifest_write_verify_tamper_and_gate(tmp_path, tiny_model_src, val_set, holdout_file):
    model = _fresh_copy(tiny_model_src, tmp_path / "cand")
    out = tmp_path / "eval.json"
    # write manifest + evaluate → gate passes (7/8 sens, 5/6 spec), exit 0
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--write-manifest", "--out", out,
             "--holdout", holdout_file)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (model / "MANIFEST.json").exists()
    j = json.loads(out.read_text())
    g = j["gate"]
    assert g["passed"] is True and (g["tp"], g["fn"], g["tn"], g["fp"]) == (7, 1, 5, 1)
    assert g["sens_ci"] == ev.wilson_ci(7, 8) and g["spec_ci"] == ev.wilson_ci(5, 6)
    assert all("sens_ci" in r and "spec_ci" in r for r in j["results"])
    assert j["n_studies"] == 14 and j["n_slices"] == 42 and "promotion" not in j
    assert j["manifest"]["files"]["model.safetensors"] == ev.sha256_file(model / "model.safetensors")
    assert j["identity"] == j["manifest"]["files"]["model.safetensors"][:12]
    assert "GATE PASSED" in p.stdout and "wrote manifest" in p.stdout
    # protocol + which patients were scored (what promote_model.py checks)
    assert j["protocol"] == {"aggregation": "mean", "threshold": 0.5, "n_central_slices": 5,
                             "preprocessing": "registry_predictor_native_res_bilinear_224"}
    assert j["gate"]["name"] == "gate mean @0.50"
    assert j["n_patients"] == 14 and j["val_patients_sha256"] == _val_sha(val_set)
    assert j["holdout_covered"] is True and j["holdout_missing"] == []
    assert j["holdout"]["n_ids"] == 3 and j["holdout"]["sha256"] == ev.sha256_file(holdout_file)

    # a holdout id that is not in val → holdout_covered false (the gate itself still passes)
    h2 = tmp_path / "holdout2.txt"
    h2.write_text("A00\n9000001\n")
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--out", out, "--holdout", h2)
    assert p.returncode == 0, p.stdout + p.stderr
    j2 = json.loads(out.read_text())
    assert j2["holdout_covered"] is False and j2["holdout_missing"] == ["9000001"]
    assert "holdout NOT covered" in p.stdout
    # --holdout none → coverage not assessed
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--out", out, "--holdout", "none")
    assert p.returncode == 0 and json.loads(out.read_text())["holdout_covered"] is None

    # verify: same files, manifest present → OK, exit 0
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--out", out)
    assert p.returncode == 0 and "manifest OK" in p.stdout, p.stdout + p.stderr

    # gate fail: impossible specificity floor → exit 2, JSON still written with passed false
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--out", out, "--min-spec", "0.95")
    assert p.returncode == ev.EXIT_GATE_FAILED and "GATE FAILED" in p.stdout
    assert json.loads(out.read_text())["gate"]["passed"] is False

    # tamper: one byte of config.json → exit 3 BEFORE any evaluation
    cfg = model / "config.json"
    cfg.write_text(cfg.read_text() + "\n")
    out.unlink()
    p = _run(EVAL, "--model", model, "--data", val_set, "--device", "cpu", "--out", out)
    assert p.returncode == ev.EXIT_MANIFEST_FAILED and "MANIFEST CHECK FAILED" in p.stdout
    assert not out.exists()


def test_eval_cli_compare_with_same_model_promotes(tmp_path, tiny_model_src, val_set):
    cand = _fresh_copy(tiny_model_src, tmp_path / "cand")
    old = _fresh_copy(tiny_model_src, tmp_path / "old")
    _run(EVAL, "--model", old, "--data", val_set, "--device", "cpu", "--write-manifest")    # pin the old one too
    p = _run(EVAL, "--model", cand, "--data", val_set, "--device", "cpu", "--write-manifest",
             "--compare-with", old)
    assert p.returncode == 0, p.stdout + p.stderr
    j = json.loads((cand / "study_level_eval.json").read_text())
    promo = j["promotion"]
    assert promo["promote"] is True
    assert promo["delta_sens"] == 0 and promo["delta_spec"] == 0 and promo["delta_balanced"] == 0
    assert promo["old"]["identity"] == promo["new"]["identity"] == j["identity"]
    assert promo["old"]["model"] == str(old) and promo["new"]["model"] == str(cand)
    assert promo["old"]["sens_ci"] == promo["new"]["sens_ci"]
    assert j["compare_with"]["gate"]["tp"] == j["gate"]["tp"] == 7
    assert "OLD vs NEW" in p.stdout and "PROMOTE" in p.stdout


def test_eval_cli_compare_with_worse_candidate_exits_4(tmp_path, tiny_model_src, degraded_model_src, val_set):
    cand = _fresh_copy(degraded_model_src, tmp_path / "cand")
    old = _fresh_copy(tiny_model_src, tmp_path / "old")
    # floors at 0 so the (bad) candidate passes the gate — the comparison alone must say no
    p = _run(EVAL, "--model", cand, "--data", val_set, "--device", "cpu", "--write-manifest",
             "--compare-with", old, "--min-sens", "0", "--min-spec", "0")
    assert p.returncode == ev.EXIT_NOT_PROMOTED, p.stdout + p.stderr
    j = json.loads((cand / "study_level_eval.json").read_text())
    assert j["gate"]["passed"] is True and j["gate"]["spec"] == 0.0
    promo = j["promotion"]
    assert promo["promote"] is False and "balanced" in promo["reason"]
    assert promo["delta_balanced"] < -0.01 and promo["old"]["identity"] != promo["new"]["identity"]
    assert "NOT PROMOTED" in p.stdout
    # a gate failure is the more specific verdict: default floors → exit 2, promotion still recorded
    p = _run(EVAL, "--model", cand, "--data", val_set, "--device", "cpu", "--compare-with", old)
    assert p.returncode == ev.EXIT_GATE_FAILED
    assert json.loads((cand / "study_level_eval.json").read_text())["promotion"]["promote"] is False


# ------------------------------------------------------------------ promote CLI

def _gated_candidate(tmp_path: Path, src: Path, val_set: Path, holdout: Path, name: str = "candidate",
                     *extra: str, expect: int = 0) -> Path:
    cand = _fresh_copy(src, tmp_path / name)
    p = _run(EVAL, "--model", cand, "--data", val_set, "--device", "cpu", "--write-manifest",
             "--holdout", holdout, *extra)
    assert p.returncode == expect, p.stdout + p.stderr
    return cand


def _promote(cand: Path, dest: Path, holdout: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run(PROMOTE, "--candidate", cand, "--dest", dest, "--holdout", holdout, *extra)


def _release_slot(tmp_path: Path, src: Path) -> Path:
    dest = _fresh_copy(src, tmp_path / "models" / "brain_triage_finetuned")
    ev.write_manifest(dest, dest / "MANIFEST.json")
    return dest


def test_promote_happy_path(tmp_path, tiny_model_src, degraded_model_src, val_set, holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = _release_slot(tmp_path, degraded_model_src)          # current release = the worse model
    old_id, new_id = ev.model_identity(dest), ev.model_identity(cand)
    assert old_id != new_id
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"{old_id} → {new_id}" in p.stdout
    assert not cand.exists()
    backup = dest.parent / f"brain_triage_finetuned_prev_{old_id}"
    assert backup.is_dir() and ev.model_identity(backup) == old_id
    assert ev.model_identity(dest) == new_id
    ev.check_manifest(dest, dest / "MANIFEST.json")               # pin still verifies at the new path
    log = json.loads((dest / "promotion_log.json").read_text())
    assert log["old_identity"] == old_id and log["new_identity"] == new_id and log["forced"] is False
    assert log["backup"] == str(backup) and log["eval"]["gate"]["passed"] is True
    assert "registry loads it" in p.stdout


def test_promote_first_release_without_existing_dest(tmp_path, tiny_model_src, val_set, holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = tmp_path / "models" / "brain_triage_finetuned"
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 0, p.stdout + p.stderr
    assert dest.is_dir() and not cand.exists() and "(none) →" in p.stdout
    assert json.loads((dest / "promotion_log.json").read_text())["backup"] is None


def test_promote_refuses_and_force_overrides(tmp_path, tiny_model_src, degraded_model_src, val_set, holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = _release_slot(tmp_path, degraded_model_src)
    before = sorted(x.name for x in dest.parent.iterdir())
    ev_path = cand / "study_level_eval.json"
    j = json.loads(ev_path.read_text())

    # gate failed
    j["gate"]["passed"] = False
    ev_path.write_text(json.dumps(j))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "REFUSED" in p.stdout and "gate.passed" in p.stdout
    assert cand.is_dir() and sorted(x.name for x in dest.parent.iterdir()) == before
    # ... and --force can NOT override a failed gate
    p = _promote(cand, dest, holdout_file, "--force")
    assert p.returncode == 1 and "not overridable" in p.stdout and "--force ignored" in p.stdout
    assert cand.is_dir() and sorted(x.name for x in dest.parent.iterdir()) == before

    # comparison said no
    j["gate"]["passed"] = True
    j["promotion"] = {"promote": False, "reason": "balanced 0.50 < old 0.85 − 0.01"}
    ev_path.write_text(json.dumps(j))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "promotion.promote" in p.stdout
    assert cand.is_dir() and sorted(x.name for x in dest.parent.iterdir()) == before

    # eval belongs to other weights
    j["promotion"]["promote"] = True
    j["manifest"]["files"]["model.safetensors"] = "f" * 64
    ev_path.write_text(json.dumps(j))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "re-run the eval" in p.stdout

    # no eval at all
    ev_path.unlink()
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "study_level_eval.json missing" in p.stdout

    # --force: loud warning, still promoted, recorded in the log
    j["manifest"]["files"]["model.safetensors"] = ev.sha256_file(cand / "model.safetensors")
    j["promotion"]["promote"] = False
    ev_path.write_text(json.dumps(j))
    p = _promote(cand, dest, holdout_file, "--force")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "--force" in p.stdout and "PROMOTED" in p.stdout
    log = json.loads((dest / "promotion_log.json").read_text())
    assert log["forced"] is True and any("promotion.promote" in r for r in log["refusal_reasons_overridden"])


def test_promote_rolls_back_when_registry_cannot_load(tmp_path, tiny_model_src, degraded_model_src, val_set, holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = _release_slot(tmp_path, degraded_model_src)
    old_id = ev.model_identity(dest)
    old_pin = json.loads((dest / "MANIFEST.json").read_text())["files"]["model.safetensors"]
    # corrupt the weights, then re-pin so the MANIFEST check passes and only the load can catch it
    weights = cand / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[: weights.stat().st_size // 2])
    ev.write_manifest(cand, cand / "MANIFEST.json")
    j = json.loads((cand / "study_level_eval.json").read_text())
    j["manifest"]["files"]["model.safetensors"] = ev.sha256_file(weights)
    (cand / "study_level_eval.json").write_text(json.dumps(j))

    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "registry load failed" in p.stdout and "rolling back" in p.stdout
    assert cand.is_dir() and (cand / "model.safetensors").exists()          # candidate put back
    assert dest.is_dir() and ev.model_identity(dest) == old_id               # release restored
    assert json.loads((dest / "MANIFEST.json").read_text())["files"]["model.safetensors"] == old_pin
    assert not (dest.parent / f"brain_triage_finetuned_prev_{old_id}").exists()
    assert not (dest / "promotion_log.json").exists()


def test_promote_refuses_tampered_candidate_before_moving(tmp_path, tiny_model_src, degraded_model_src, val_set, holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = _release_slot(tmp_path, degraded_model_src)
    (cand / "config.json").write_text((cand / "config.json").read_text() + "\n")
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == ev.EXIT_MANIFEST_FAILED and "MANIFEST CHECK FAILED" in p.stdout
    assert cand.is_dir() and not (dest / "promotion_log.json").exists()
    assert not any(x.name.startswith("brain_triage_finetuned_prev_") for x in dest.parent.iterdir())


def test_promote_refuses_lowered_floors_even_with_force(tmp_path, degraded_model_src, val_set, holdout_file):
    """A candidate that 'passed' only because the eval ran with --min-sens 0 --min-spec 0
    (the degraded model: sens 1.0 / spec 0.0) must never be promoted, --force or not."""
    cand = _gated_candidate(tmp_path, degraded_model_src, val_set, holdout_file, "candidate",
                            "--min-sens", "0", "--min-spec", "0")
    j = json.loads((cand / "study_level_eval.json").read_text())
    assert j["gate"]["passed"] is True and j["gate"]["spec"] == 0.0
    dest = tmp_path / "models" / "brain_triage_finetuned"
    for extra in ((), ("--force",)):
        p = _promote(cand, dest, holdout_file, *extra)
        assert p.returncode == 1, p.stdout + p.stderr
        assert "gate.min_sens" in p.stdout and "gate.min_spec" in p.stdout and "release floor" in p.stdout
        assert "not overridable" in p.stdout
        assert cand.is_dir() and not dest.exists()


def test_promote_refuses_non_production_protocol_or_wrong_patients(tmp_path, tiny_model_src, val_set,
                                                                   holdout_file):
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = tmp_path / "models" / "brain_triage_finetuned"
    ev_path = cand / "study_level_eval.json"
    good = json.loads(ev_path.read_text())

    def refused_with(mutate, needle: str) -> None:
        j = json.loads(json.dumps(good))
        mutate(j)
        ev_path.write_text(json.dumps(j))
        for extra in ((), ("--force",)):
            p = _promote(cand, dest, holdout_file, *extra)
            assert p.returncode == 1 and needle in p.stdout, (needle, p.stdout + p.stderr)
            assert cand.is_dir() and not dest.exists()

    # gate scored on another aggregation / threshold (e.g. --gate-agg max --gate-threshold 0.2)
    refused_with(lambda j: j["gate"].update(name="gate max @0.20"), "not the production protocol")
    refused_with(lambda j: j["protocol"].update(aggregation="max", threshold=0.2), "protocol aggregation")
    # the old eval path (processor on the PNG without the production predictor) / pre-protocol JSON
    refused_with(lambda j: j["protocol"].update(preprocessing="processor_on_png"), "protocol.preprocessing")
    refused_with(lambda j: j.pop("protocol"), "protocol.preprocessing")
    # patients scored are not recorded
    refused_with(lambda j: j.pop("val_patients_sha256"), "no val_patients_sha256")
    # legacy val dir: holdout not covered / not assessed / checked against another holdout file
    refused_with(lambda j: j.update(holdout_covered=False, holdout_missing=["9000001"]), "holdout_covered is False")
    refused_with(lambda j: j.update(holdout_covered=None), "holdout_covered is None")
    refused_with(lambda j: j["holdout"].update(sha256="0" * 64), "different file")

    # the untouched eval still promotes (proves the checks above were the reason)
    ev_path.write_text(json.dumps(good))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 0, p.stdout + p.stderr
    log = json.loads((dest / "promotion_log.json").read_text())
    assert log["eval"]["protocol"]["preprocessing"] == "registry_predictor_native_res_bilinear_224"
    assert log["eval"]["val_patients_sha256"] == _val_sha(val_set) and log["forced"] is False


def test_promote_checks_combined_set_val_sha(tmp_path, tiny_model_src, val_set, holdout_file):
    """Combined release set (stage 5): <data>/../dataset_manifest.json pins the val
    patients; the eval must have scored exactly those."""
    rel = tmp_path / "train_release_t"
    shutil.copytree(val_set, rel / "val")
    manifest = rel / "dataset_manifest.json"
    manifest.write_text(json.dumps({"splits": {"val": {"patients_sha256": _val_sha(val_set)}}}))
    cand = _gated_candidate(tmp_path, tiny_model_src, rel / "val", holdout_file)
    dest = tmp_path / "models" / "brain_triage_finetuned"
    # wrong val set pinned → refused, --force cannot help
    manifest.write_text(json.dumps({"splits": {"val": {"patients_sha256": "1" * 64}}}))
    for extra in ((), ("--force",)):
        p = _promote(cand, dest, holdout_file, *extra)
        assert p.returncode == 1 and "different patient set" in p.stdout, p.stdout + p.stderr
    import hashlib
    split = {"val": {"patients_sha256": _val_sha(val_set)}}

    def hblock(fp: Path, n_in_val: int = 3) -> dict:
        return {"file": str(fp), "sha256": hashlib.sha256(fp.read_bytes()).hexdigest(), "n_ids": 3,
                "n_in_val": n_in_val, "missing_from_all_batches": []}
    # matching pin but the set was combined WITHOUT the locked holdout → refused (review round 2)
    manifest.write_text(json.dumps({"splits": split}))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "records no locked holdout" in p.stdout, p.stdout + p.stderr
    # combined with a different holdout file, or with holdout patients outside val → refused
    other_holdout = tmp_path / "other_holdout.txt"
    other_holdout.write_text("A01\n")
    manifest.write_text(json.dumps({"splits": split, "holdout": hblock(other_holdout)}))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "different holdout file" in p.stdout, p.stdout + p.stderr
    manifest.write_text(json.dumps({"splits": split, "holdout": hblock(holdout_file, n_in_val=2)}))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "only 2 of 3 locked holdout" in p.stdout, p.stdout + p.stderr
    # an eval that skipped the holdout check (--holdout none) → refused
    ev_path = cand / "study_level_eval.json"
    ev = json.loads(ev_path.read_text())
    manifest.write_text(json.dumps({"splits": split, "holdout": hblock(holdout_file)}))
    ev_path.write_text(json.dumps({**ev, "holdout_covered": None}))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 1 and "did not check the locked holdout" in p.stdout, p.stdout + p.stderr
    # matching pin + combined with the locked holdout + coverage checked → promoted
    ev_path.write_text(json.dumps(ev))
    p = _promote(cand, dest, holdout_file)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PROMOTED" in p.stdout


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_promote_interrupted_during_registry_probe_rolls_back(tmp_path, tiny_model_src, degraded_model_src,
                                                              val_set, holdout_file, sig):
    """Ctrl-C / SIGTERM while the (slow) registry probe runs must put the candidate and the previous
    release back (review round 2, promote_model.py:343) — never leave an unverified model in place."""
    import signal
    import time
    cand = _gated_candidate(tmp_path, tiny_model_src, val_set, holdout_file)
    dest = _release_slot(tmp_path, degraded_model_src)
    old_id = ev.model_identity(dest)
    started = tmp_path / "probe_started"
    slow = tmp_path / "slow_python.sh"
    slow.write_text(f"#!/bin/sh\ntouch {started}\nsleep 60\n")
    slow.chmod(0o755)
    proc = subprocess.Popen([sys.executable, str(PROMOTE), "--candidate", str(cand), "--dest", str(dest),
                             "--holdout", str(holdout_file), "--python", str(slow)],
                            cwd=str(REPO), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while not started.exists() and time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert started.exists(), proc.communicate()[0]
    proc.send_signal(getattr(signal, sig))
    out = proc.communicate(timeout=60)[0]
    assert proc.returncode != 0, out
    assert "rolling back" in out and "promote interrupted" in out, out
    assert cand.is_dir() and (cand / "model.safetensors").exists()          # candidate put back
    assert dest.is_dir() and ev.model_identity(dest) == old_id               # release restored
    assert not (dest / "promotion_log.json").exists()
    assert not list(dest.parent.glob("brain_triage_finetuned_prev_*"))
