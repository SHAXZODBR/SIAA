"""scripts/release_pipeline.py — the stage-11 orchestrator, tested without running any real stage.

Everything runs against a throw-away synthetic repo under tmp_path (`--repo`): a fake drive, a
fake reports folder, a fake current model, a fake legacy training set. Stage commands are
replaced by a fake runner (monkeypatched `run_command` / `subprocess.Popen`) that writes the
outputs the contract promises, so the graph, the plan output, the state/resume logic and the
guard rails are exercised end-to-end in well under a second. No real patient data anywhere.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release_pipeline.py"
DOCS = REPO / "docs"

EXPECTED_ORDER = ("index", "parse", "label", "match", "build", "combine",
                  "train", "gate", "promote", "bump", "bundle")
CONTRACT_SCRIPTS = {
    "index": "scripts/index_new_studies.py",
    "parse": "scripts/parse_hospital_reports.py",
    "label": "scripts/label_reports.py",
    "match": "scripts/match_images_to_reports.py",
    "build": "scripts/build_training_set.py",
    "combine": "scripts/combine_training_sets.py",
    "train": "training/finetune_brain_classifier.py",
    "gate": "scripts/eval_study_level.py",
    "promote": "scripts/promote_model.py",
    "bump": "scripts/bump_model_release.py",
    "bundle": "packaging/publish_model_bundle.py",
}
HOLDOUT_IDS = ["9001", "9002", "9003"]


def _load_module():
    spec = importlib.util.spec_from_file_location("release_pipeline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod          # dataclasses resolve postponed annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rp():
    return _load_module()


# ------------------------------------------------------------ synthetic repo --

def _sha_ids(ids) -> str:
    """eval_study_level.sha256_ids / combine_training_sets splits.<split>.patients_sha256."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


def _write_eval(model_dir: Path, passed: bool = True, promote: bool = True, weights: bool = True,
                data: str = "data/local_finetune_bin/val", **overrides) -> None:
    """A study_level_eval.json in the shape the real eval_study_level.py writes: production
    protocol, release floors, scored-patient sha + holdout coverage, manifest pin."""
    model_dir.mkdir(parents=True, exist_ok=True)
    repo = model_dir.parent.parent
    if weights:                  # the real eval_study_level.py never touches the weights it scores
        (model_dir / "model.safetensors").write_bytes(b"\0" * 64)
    pin = {"files": {"model.safetensors": "00" * 32}}
    (model_dir / "MANIFEST.json").write_text(json.dumps(pin))
    holdout = repo / "data" / "holdout_patients_v1.txt"
    ds_manifest = repo / data / ".." / "dataset_manifest.json"
    val_sha = (json.loads(ds_manifest.read_text())["splits"]["val"]["patients_sha256"]
               if ds_manifest.is_file() else _sha_ids(HOLDOUT_IDS))
    ev = {"gate": {"name": "gate mean @0.50", "passed": passed, "sens": 0.9, "spec": 0.5,
                   "min_sens": 0.85, "min_spec": 0.40},
          "protocol": {"aggregation": "mean", "threshold": 0.5, "n_central_slices": 5,
                       "preprocessing": "registry_predictor_native_res_bilinear_224"},
          "data": data, "val_patients_sha256": val_sha, "holdout_covered": True, "holdout_missing": [],
          "holdout": {"file": str(holdout),
                      "sha256": hashlib.sha256(holdout.read_bytes()).hexdigest() if holdout.is_file() else None},
          "manifest": pin,
          "promotion": {"promote": promote, "reason": "synthetic"}}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(ev.get(k), dict):
            ev[k] = {**ev[k], **v}
        else:
            ev[k] = v
    (model_dir / "study_level_eval.json").write_text(json.dumps(ev))


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sentinel"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "holdout_patients_v1.txt").write_text("\n".join(HOLDOUT_IDS) + "\n")
    for split in ("train", "val"):
        for cls in ("abnormal", "normal"):
            (repo / "data" / "local_finetune_bin" / split / cls).mkdir(parents=True)
    _write_eval(repo / "models" / "brain_triage_finetuned")
    (repo / "desktop-app").mkdir()
    (repo / "desktop-app" / "package.json").write_text(json.dumps({"version": "1.0.0"}))
    (repo / "scripts").mkdir()
    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "STUDY_1").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "1_x.pdf").write_bytes(b"%PDF-fake")
    return repo


def _base_args(tmp_path: Path, repo: Path, *extra: str) -> list[str]:
    return ["--batch", "fakebatch", "--repo", str(repo), "--root", str(tmp_path / "drive"),
            "--reports", str(tmp_path / "reports"), "--tag", "brain-triage-vTEST", *extra]


class FakeRunner:
    """Stands in for release_pipeline.run_command: writes the contract outputs."""

    def __init__(self, repo: Path, n_normal: int = 400, fail_on: str | None = None,
                 leak: bool = False, eval_rc: int = 0, no_tune: bool = False):
        self.repo = repo
        self.n_normal = n_normal
        self.fail_on = fail_on
        self.leak = leak
        self.eval_rc = eval_rc          # eval_study_level.py exit code (2 gate failed, 4 not promoted)
        self.no_tune = no_tune          # simulate an old combine that wrote no tune/
        self.n_combine = 0
        self.calls: list[list[str]] = []
        self.dry_calls: list[list[str]] = []     # what a --dry-run would have executed

    def _arg(self, argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def __call__(self, argv: list[str], log_path: Path, cwd: Path, dry_run: bool = False) -> int:
        if dry_run:
            self.dry_calls.append(list(argv))
            return 0
        self.calls.append(list(argv))
        script = argv[1] if len(argv) > 1 else ""
        if self.fail_on and self.fail_on in script:
            self.fail_on = None          # fail once, succeed on the retry
            return 1
        repo = self.repo
        if script.endswith("index_new_studies.py") or script.endswith("parse_hospital_reports.py") \
                or (script.endswith("label_reports.py") and "--benchmark" not in argv):
            out = repo / self._arg(argv, "--out")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("patient_id\n1\n")
        elif script.endswith("label_reports.py"):
            return 0
        elif script.endswith("match_images_to_reports.py"):
            out = repo / self._arg(argv, "--out")
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["study_folder", "patient_id", "is_brain", "report_matched", "abnormal", "label"])
                for i in range(self.n_normal):
                    w.writerow([f"S{i}", str(1000 + i), "1", "1", "no", "normal"])
                for i in range(50):
                    w.writerow([f"A{i}", str(5000 + i), "1", "1", "yes", "tumor"])
                w.writerow(["SPINE", "7", "0", "1", "", ""])
        elif script.endswith("build_training_set.py"):
            out = repo / self._arg(argv, "--out")
            (out / "train" / "normal").mkdir(parents=True, exist_ok=True)
            (out / "studies.csv").write_text("patient_id,cls\n")
        elif script.endswith("combine_training_sets.py"):
            out = repo / self._arg(argv, "--out")
            splits = (("train", "abnormal", ["10", "11"]), ("train", "normal", ["12"]),
                      ("val", "abnormal", HOLDOUT_IDS[:2]), ("val", "normal", HOLDOUT_IDS[2:] + ["13"]))
            if not self.no_tune:         # combine_training_sets.py writes tune/ by default (--tune-frac 0.1)
                splits += (("tune", "normal", ["14"]),)
            for split, cls, pids in splits:
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for pid in pids:
                    (d / f"{pid}_0.png").write_bytes(b"png")
            if self.leak:
                (out / "train" / "normal" / f"{HOLDOUT_IDS[0]}_0.png").write_bytes(b"png")
            self.n_combine += 1          # a real manifest changes when its inputs change
            val_ids = [pid for sp, _c, pids in splits if sp == "val" for pid in pids]
            hold = repo / self._arg(argv, "--holdout")
            (out / "dataset_manifest.json").write_text(json.dumps(
                {"seed": 42, "run": self.n_combine,
                 "splits": {"val": {"n_patients": len(val_ids), "patients_sha256": _sha_ids(val_ids)}},
                 # what the real combine records for --holdout (promote / the bundle guard check it)
                 "holdout": {"file": str(hold), "sha256": hashlib.sha256(hold.read_bytes()).hexdigest(),
                             "n_ids": len(HOLDOUT_IDS), "n_in_val": len(HOLDOUT_IDS),
                             "missing_from_all_batches": []}}))
        elif script.endswith("finetune_brain_classifier.py"):
            out = repo / self._arg(argv, "--output")
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.safetensors").write_bytes(b"\1" * 64)
        elif script.endswith("eval_study_level.py"):
            _write_eval(repo / self._arg(argv, "--model"), passed=self.eval_rc != 2,
                        promote=self.eval_rc == 0, weights=False, data=self._arg(argv, "--data"))
            return self.eval_rc
        elif script.endswith("promote_model.py"):
            cand = repo / self._arg(argv, "--candidate")
            dest = repo / self._arg(argv, "--dest")
            prev = dest.parent / "brain_triage_finetuned_prev_synthetic"
            if dest.exists():
                shutil.move(str(dest), str(prev))
            shutil.move(str(cand), str(dest))
        elif script.endswith("bump_model_release.py"):
            return 0
        elif script.endswith("publish_model_bundle.py"):
            out = repo / self._arg(argv, "--out")
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(json.dumps({"parts": [], "sha256": "x"}))
        elif argv[1:3] == ["-m", "pytest"]:
            return 0
        return 0


@pytest.fixture
def plenty_of_disk(rp, monkeypatch):
    monkeypatch.setattr(rp, "free_gb", lambda p: 500.0)


# ------------------------------------------------------------- stage graph ----

def test_stage_order_is_fixed(rp, fake_repo, tmp_path):
    assert rp.STAGE_ORDER == EXPECTED_ORDER
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo)]))
    stages = rp.build_stages(cfg)
    assert rp.stage_names(stages) == list(EXPECTED_ORDER)
    for s in stages:
        assert s.cmds, s.name
        assert any(CONTRACT_SCRIPTS[s.name] in c for c in s.cmds[0]), (s.name, s.cmds[0])
    # every stage-name subcommand is accepted by the parser
    for name in EXPECTED_ORDER:
        assert name in rp.SUBCOMMANDS


def test_stage_commands_follow_the_contract(rp, fake_repo, tmp_path):
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo)]))
    by = {s.name: s for s in rp.build_stages(cfg)}
    joined = {k: " ".join(v.cmds[0]) for k, v in by.items()}
    assert "--out data/fakebatch_index.csv" in joined["index"]
    assert "--out data/fakebatch_reports.csv" in joined["parse"]
    assert "--reports data/fakebatch_reports.csv --out data/fakebatch_labels.csv" in joined["label"]
    assert "--benchmark" in " ".join(by["label"].cmds[1])
    assert "--index data/fakebatch_index.csv --labels data/fakebatch_labels.csv --out data/fakebatch_matched.csv" in joined["match"]
    assert "--out data/sets/fakebatch --mode binary --abnormal-col abnormal --skip-needs-review" in joined["build"]
    # the current batch goes LAST: combine lets the last set on the command line win a label conflict
    assert "--sets data/local_finetune_bin data/sets/fakebatch --holdout data/holdout_patients_v1.txt" in joined["combine"]
    assert "--out data/train_release_brain-triage-vTEST" in joined["combine"]
    assert "--classes abnormal,normal --base-model models/brain_triage_finetuned --balance --unfreeze-last-blocks 3 --epochs 12 --device mps" in joined["train"]
    assert "--select-split tune --output models/brain_triage_candidate_brain-triage-vTEST" in joined["train"]
    assert "--train-lock" not in joined["combine"]        # v1 model: trained on the legacy set only
    assert "--write-manifest --min-sens 0.85 --min-spec 0.40 --compare-with models/brain_triage_finetuned" in joined["gate"]
    assert "--candidate models/brain_triage_candidate_brain-triage-vTEST --dest models/brain_triage_finetuned" in joined["promote"]
    assert joined["bump"].endswith("scripts/bump_model_release.py --model models/brain_triage_finetuned")
    assert "--out data/release/brain-triage-vTEST/ --github SHAXZODBR/SIAA-models --tag brain-triage-vTEST" in joined["bundle"]
    assert "--fresh" not in joined["combine"]          # first run: the release set does not exist yet


def test_combine_rerun_adds_fresh_when_release_set_exists(rp, fake_repo, tmp_path):
    # combine_training_sets.py refuses a non-empty --out, so a re-run of the stage must pass --fresh
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo)]))
    release_set = fake_repo / cfg.release_set
    (release_set / "train" / "normal").mkdir(parents=True)
    (release_set / "train" / "normal" / "1_0.png").write_bytes(b"png")
    combine = " ".join({s.name: s for s in rp.build_stages(cfg)}["combine"].cmds[0])
    assert combine.endswith("--seed 42 --fresh")


def test_default_tag_counts_previous_releases(rp, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    assert rp.default_tag(models) == "brain-triage-v1"
    (models / "brain_triage_finetuned").mkdir()
    assert rp.default_tag(models) == "brain-triage-v2"
    (models / "brain_triage_finetuned_prev_0d559766ce58").mkdir()
    assert rp.default_tag(models) == "brain-triage-v3"


def test_default_extra_sets_excludes_current_batch_and_adds_legacy(rp, fake_repo):
    (fake_repo / "data" / "sets" / "older").mkdir(parents=True)
    (fake_repo / "data" / "sets" / "fakebatch").mkdir(parents=True)
    (fake_repo / "data" / "sets" / "rehearsal_keep66").mkdir(parents=True)     # never part of a release
    # legacy (oldest) first, so the newer batches win label conflicts in combine
    assert rp.default_extra_sets(fake_repo, "fakebatch") == ["data/local_finetune_bin", "data/sets/older"]
    shutil.rmtree(fake_repo / "data" / "local_finetune_bin")
    assert rp.default_extra_sets(fake_repo, "fakebatch") == ["data/sets/older"]


def test_train_lock_defaults_to_the_set_the_current_model_was_trained_on(rp, fake_repo, tmp_path):
    prev = fake_repo / "data" / "train_release_brain-triage-v2"
    (prev / "tune").mkdir(parents=True)
    (prev / "dataset_manifest.json").write_text("{}")
    (fake_repo / "models" / "brain_triage_finetuned" / "training_metrics.json").write_text(json.dumps(
        {"checkpoint_selection": {"split": "tune", "data": "data/train_release_brain-triage-v2/tune"}}))
    args = [a if a != "brain-triage-vTEST" else "brain-triage-v3" for a in _base_args(tmp_path, fake_repo)]
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *args]))
    assert cfg.train_lock == "data/train_release_brain-triage-v2"
    combine = " ".join({s.name: s for s in rp.build_stages(cfg)}["combine"].cmds[0])
    assert "--out data/train_release_brain-triage-v3 --train-lock data/train_release_brain-triage-v2 " in combine
    # the set being built under this very tag is never its own lock (resume after promote)
    args2 = [a if a != "brain-triage-vTEST" else "brain-triage-v2" for a in _base_args(tmp_path, fake_repo)]
    assert rp.config_from_args(rp.build_parser().parse_args(["plan", *args2])).train_lock is None
    # an explicit --train-lock wins
    cfg3 = rp.config_from_args(rp.build_parser().parse_args(["plan", *args, "--train-lock", "data/x.txt"]))
    assert cfg3.train_lock == "data/x.txt"


# ---------------------------------------------------------------------- plan --

def test_plan_prints_shell_and_runs_nothing(rp, fake_repo, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("plan must not execute anything")
    monkeypatch.setattr(rp, "run_command", boom)
    monkeypatch.setattr(rp.subprocess, "Popen", boom)
    rc = rp.main(["plan", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.splitlines()
    assert lines[0] == "#!/bin/sh"
    assert "set -e" in lines
    assert f"cd {shlex.quote(str(fake_repo))}" in lines
    for name in EXPECTED_ORDER:
        assert f"# --- stage {EXPECTED_ORDER.index(name)}: {name}" in out
        assert CONTRACT_SCRIPTS[name] in out
    # every non-comment line must be a parseable shell command
    for ln in lines:
        if ln and not ln.startswith("#"):
            assert shlex.split(ln)
    assert "guard:" in out and "manual:" in out
    assert not (fake_repo / "data" / "release").exists()


def test_dry_run_prints_commands_and_writes_no_state(rp, fake_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp.subprocess, "Popen", lambda *a, **k: pytest.fail("dry-run spawned a process"))
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out and "would check" in out
    assert "manual steps" in out
    assert not (fake_repo / "data" / "release").exists()


# ----------------------------------------------------------------- run/state --

def test_full_run_with_fake_runner_records_state_and_prints_checklist(
        rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == 0, out
    state = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    # promote MOVED the candidate: train + gate are recorded 'archived' (= done), promote knows where
    assert [s for s in EXPECTED_ORDER if state["stages"][s]["status"] in ("done", "archived")] == list(EXPECTED_ORDER)
    assert state["stages"]["train"]["status"] == state["stages"]["gate"]["status"] == "archived"
    assert state["stages"]["promote"]["archived_candidate"] == "models/brain_triage_finetuned"
    for s in EXPECTED_ORDER:
        rec = state["stages"][s]
        assert rec["started"] and rec["finished"] and rec["cmd"] and rec["log"].endswith(f"logs/{s}.log")
        assert "input_sha256s" in rec and "outputs" in rec
    assert state["guards"]["label_benchmark"] == "passed"
    assert state["guards"]["pre_train"]["passed"] is True
    assert state["guards"]["pre_bundle"]["passed"] is True
    # guard rails ran the checks the contract names
    scripts_run = [c[1] for c in runner.calls]
    assert "scripts/bump_model_release.py" in scripts_run
    assert any(c[1:4] == ["-m", "pytest", "-q"] for c in runner.calls)
    assert any("docs/test_docs_consistency.py" in c for c in runner.calls)
    assert "matched brain studies" in out
    # manual checklist, never a git push/tag by the script itself
    assert "MODEL_BUNDLE_URL" in out and "git tag v" in out and "windows_laptop_test" in out
    assert "never pushes or tags" in out
    assert not any(c[0] == "git" for c in runner.calls)


def test_failure_stops_with_command_log_and_resume_hint_then_resume_skips_done(
        rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, fail_on="match_images_to_reports.py")
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_STAGE_FAILED
    assert "stage match FAILED" in out
    assert "scripts/match_images_to_reports.py" in out
    assert "data/release/brain-triage-vTEST/logs/match.log" in out
    assert "--resume" in out
    state_path = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    state = json.loads(state_path.read_text())
    assert state["stages"]["match"]["status"] == "failed"
    assert state["stages"]["label"]["status"] == "done"
    assert "train" not in state["stages"]

    n_before = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"])
    out = capsys.readouterr().out
    assert rc == 0, out
    for s in ("index", "parse", "label"):
        assert f"stage {s} already done — skipped" in out
    new_calls = runner.calls[n_before:]
    assert not any("index_new_studies.py" in c[1] or "parse_hospital_reports.py" in c[1] for c in new_calls)
    assert any("match_images_to_reports.py" in c[1] for c in new_calls)
    state = json.loads(state_path.read_text())
    assert all(state["stages"][s]["status"] in ("done", "archived") for s in EXPECTED_ORDER)


def test_resume_reruns_a_stage_whose_inputs_changed(rp, fake_repo, tmp_path, monkeypatch,
                                                    plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--only", "index"]) == 0
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--only", "parse"]) == 0
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--only", "label"]) == 0
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--only", "match"]) == 0
    capsys.readouterr()
    (fake_repo / "data" / "fakebatch_labels.csv").write_text("patient_id\n1\n2\n")   # new labels
    n = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--only", "match"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "inputs changed" in out
    assert len(runner.calls) == n + 1

    n = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--only", "match"])
    assert rc == 0 and len(runner.calls) == n            # unchanged → skipped


def test_stage_name_subcommand_equals_only(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo)]) == 0
    assert [c[1] for c in runner.calls] == ["scripts/index_new_studies.py"]
    assert rp.main(["index", *_base_args(tmp_path, fake_repo), "--only", "parse"]) == rp.EXIT_USAGE
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--only", "parse", "--from", "index"]) == rp.EXIT_USAGE


def test_status_lists_every_stage(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    rp.main(["index", *_base_args(tmp_path, fake_repo)])
    capsys.readouterr()
    assert rp.main(["status", *_base_args(tmp_path, fake_repo)]) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)     # strip colours
    assert "index    done" in out
    for s in EXPECTED_ORDER[1:]:
        assert f"{s:<8} pending" in out


def test_run_command_tees_output_to_log(rp, tmp_path):
    log = tmp_path / "logs" / "x.log"
    rc = rp.run_command([sys.executable, "-c", "print('hello-tee'); import sys; sys.exit(3)"],
                        log, tmp_path)
    assert rc == 3
    text = log.read_text()
    assert "hello-tee" in text and "exit 3" in text


# -------------------------------------------------------------- guard rails --

def test_guard_missing_drive_blocks_train(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    for s in ("index", "parse", "label", "match", "build", "combine"):
        assert rp.main([s, *_base_args(tmp_path, fake_repo)]) == 0
    capsys.readouterr()
    gone = tmp_path / "unmounted_drive"
    args = ["train", "--batch", "fakebatch", "--repo", str(fake_repo), "--root", str(gone),
            "--tag", "brain-triage-vTEST"]
    n = len(runner.calls)
    rc = rp.main(args)
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert "drive not mounted" in out
    assert "nothing was run" in out
    assert len(runner.calls) == n                        # train never started
    state = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert state["guards"]["pre_train"]["passed"] is False
    assert any("drive not mounted" in e for e in state["guards"]["pre_train"]["errors"])


def test_guard_low_normals_warns_but_runs(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, n_normal=12)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--from", "index"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "only 12 normal studies" in out and "< 300" in out
    assert "matched brain studies" in out
    state = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert state["guards"]["pre_train"]["passed"] is True
    assert any("normal" in w for w in state["guards"]["pre_train"]["warnings"])


def test_guard_leakage_blocks_train(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, leak=True)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert "leakage" in out
    assert not any("finetune_brain_classifier.py" in c[1] for c in runner.calls)


def test_guard_low_disk_blocks_train(rp, fake_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "free_gb", lambda p: 3.5)
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert "3.5 GB free" in out


def test_guard_benchmark_required_unless_skipped(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    for s in ("index", "parse", "match", "build", "combine"):       # label stage deliberately skipped
        assert rp.main([s, *_base_args(tmp_path, fake_repo)]) == 0
    (fake_repo / "data" / "fakebatch_labels.csv").write_text("patient_id\n")
    capsys.readouterr()
    assert rp.main(["train", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    assert "label benchmark has not passed" in capsys.readouterr().out
    assert rp.main(["train", *_base_args(tmp_path, fake_repo), "--skip-benchmark"]) == 0
    assert "SKIPPED" in capsys.readouterr().out


def test_guard_pre_bundle_needs_gate_promotion_check_and_tests(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    _write_eval(fake_repo / "models" / "brain_triage_finetuned", passed=False, promote=False)
    rc = rp.main(["bundle", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert "gate.passed is not true" in out and "promotion.promote is false" in out
    assert not any("publish_model_bundle.py" in c[1] for c in runner.calls)

    _write_eval(fake_repo / "models" / "brain_triage_finetuned", passed=True, promote=True)

    def failing_tests(argv, log_path, cwd, dry_run=False):
        runner.calls.append(list(argv))
        return 1 if argv[1:3] == ["-m", "pytest"] else runner(argv, log_path, cwd, dry_run)
    monkeypatch.setattr(rp, "run_command", failing_tests)
    rc = rp.main(["bundle", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE and "pytest -q failed" in out

    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["bundle", *_base_args(tmp_path, fake_repo), "--skip-tests"]) == 0
    out = capsys.readouterr().out
    assert "test suite SKIPPED" in out and "manual steps" in out


def test_count_matched_prefers_abnormal_column(rp, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("study_folder,patient_id,is_brain,report_matched,abnormal,label\n"
                 "a,1,1,1,no,normal\nb,2,1,1,yes,normal\nc,3,0,1,no,normal\nd,4,1,0,no,normal\n"
                 "e,5,1,1,,normal\nf,6,1,1,,tumor\n")
    assert rp.count_matched(p) == (4, 2, 2)


def test_check_leakage_reports_overlap_and_missing_holdout(rp, tmp_path):
    rel = tmp_path / "rel"
    hold = tmp_path / "hold.txt"
    hold.write_text("9001\n9002\n")
    assert any("dataset_manifest.json missing" in e for e in rp.check_leakage(rel, hold))
    for split, pid in (("train", "1"), ("train", "9001"), ("val", "9001")):
        d = rel / split / "abnormal"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pid}_0.png").write_bytes(b"x")
    (rel / "dataset_manifest.json").write_text("{}")
    errs = rp.check_leakage(rel, hold)
    assert any("BOTH train and val" in e for e in errs)
    assert any("holdout patients missing from val" in e and "9002" in e for e in errs)


# ----------------------------------------------------------------- runbooks --

RUNBOOKS = ("release_runbook_en.md", "release_runbook_ru.md")
STAGE_ROW = re.compile(r"^\|\s*\d+\s*\|\s*`([a-z]+)`\s*\|", re.M)


@pytest.mark.parametrize("name", RUNBOOKS)
def test_runbook_stage_names_exist_in_script(rp, name):
    text = (DOCS / name).read_text(encoding="utf-8")
    names = STAGE_ROW.findall(text)
    assert names, f"{name}: no stage table rows like '| 0 | `index` |'"
    unknown = sorted(set(names) - set(rp.STAGE_ORDER))
    assert not unknown, f"{name} names stages the script does not have: {unknown}"
    assert names == list(rp.STAGE_ORDER), f"{name}: stage table order differs from the script"
    for sub in ("plan", "run", "status", "--resume", "--dry-run", "--from", "--only"):
        assert sub in text, f"{name} does not mention {sub}"


def test_runbooks_share_structure():
    counts = {}
    for name in RUNBOOKS:
        text = (DOCS / name).read_text(encoding="utf-8")
        counts[name] = (len(re.findall(r"^## \d+\.", text, re.M)),
                        len(re.findall(r"^### ", text, re.M)),
                        len(STAGE_ROW.findall(text)))
        assert counts[name][0] >= 8, f"{name}: runbook must be numbered with ≥ 8 top-level sections"
    assert len(set(counts.values())) == 1, counts


def test_docs_index_and_readme_link_the_runbook():
    docs_index = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "release_runbook_en.md" in docs_index and "release_runbook_ru.md" in docs_index
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    gates = [ln for ln in readme.splitlines() if "**Quality gates**" in ln]
    assert gates and "release_pipeline.py" in gates[0]


# ------------------------------------------------- state model after promote --

def _scripts(calls: list[list[str]]) -> list[str]:
    return [c[1] for c in calls]


def test_resume_after_promote_runs_only_bundle(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    """Reviewer scenario: stages 0-9 succeed, bundle fails (no token / GitHub 5xx). promote has
    MOVED the candidate dir. The printed resume line must run bundle only — never train/gate/promote."""
    runner = FakeRunner(fake_repo, fail_on="publish_model_bundle.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stage bundle FAILED" in out
    assert not (fake_repo / "models" / "brain_triage_candidate_brain-triage-vTEST").exists()   # moved
    hint = [ln for ln in out.splitlines() if "resume with:" in ln][-1].split("resume with:", 1)[1].strip()
    resume_argv = shlex.split(hint)[3:]           # drop "<python> scripts/release_pipeline.py run"

    # dry run of the resume line: nothing but bundle (and its guard) would execute
    n_calls = len(runner.calls)
    rc = rp.main(["run", *resume_argv, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    for s in EXPECTED_ORDER[:-1]:
        assert f"stage {s} already done — skipped" in out, s
    assert len(runner.calls) == n_calls                          # dry run executed nothing
    assert _scripts(runner.dry_calls) == ["packaging/publish_model_bundle.py"], runner.dry_calls

    # the real resume: only the bundle stage + its guard commands run — even with the drive unplugged
    (tmp_path / "drive").rename(tmp_path / "drive_unplugged")
    n = len(runner.calls)
    rc = rp.main(["run", *resume_argv])
    out = capsys.readouterr().out
    assert rc == 0, out
    new = _scripts(runner.calls[n:])
    assert set(new) == {"scripts/bump_model_release.py", "-m", "packaging/publish_model_bundle.py"}, new
    assert new.count("packaging/publish_model_bundle.py") == 1
    assert not (fake_repo / "models" / "brain_triage_finetuned_prev_synthetic_2").exists()
    state = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert state["stages"]["bundle"]["status"] == "done"


def test_train_gate_promote_refused_after_promote_under_same_tag(rp, fake_repo, tmp_path, monkeypatch,
                                                                 plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == 0
    capsys.readouterr()
    for stage in ("train", "gate", "promote"):
        n = len(runner.calls)
        assert rp.main([stage, *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
        out = capsys.readouterr().out
        assert "already promoted" in out and "new release" in out
        assert len(runner.calls) == n, stage
    # --from gate --resume: gate/promote resolve to the release dir → skipped, not refused
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--from", "gate", "--resume", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "stage gate already done — skipped" in out and "stage promote already done — skipped" in out


def test_later_stage_done_forgives_missing_outputs_unless_named(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--from", "index"][:-2] + ["--only", "index"]) == 0
    assert rp.main(["parse", *_base_args(tmp_path, fake_repo)]) == 0
    cfg = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo), "--resume"]))
    index = {s.name: s for s in rp.build_stages(cfg)}["index"]
    (fake_repo / cfg.index_csv).unlink()                       # consumed / cleaned up
    state = rp.load_state(cfg.state_path)
    assert rp.stage_is_done(cfg, index, state) == (True, "")   # parse (later) is done
    done, why = rp.stage_is_done(cfg, index, state, frozenset({"index"}))
    assert not done and "outputs missing" in why
    state["stages"].pop("parse")
    done, why = rp.stage_is_done(cfg, index, state)
    assert not done and "outputs missing" in why


# -------------------------------------------------------------- stage inputs --

def test_combine_train_gate_rerun_when_a_set_is_rebuilt(rp, fake_repo, tmp_path, monkeypatch,
                                                        plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, fail_on="promote_model.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    state_path = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    rec = json.loads(state_path.read_text())["stages"]
    assert set(rec["combine"]["input_sha256s"]) == {"data/sets/fakebatch/studies.csv", "ls:data/local_finetune_bin",
                                                    "data/holdout_patients_v1.txt"}
    assert set(rec["train"]["input_sha256s"]) == {"data/train_release_brain-triage-vTEST/dataset_manifest.json",
                                                  "opt:base_model", "opt:epochs"}
    assert set(rec["gate"]["input_sha256s"]) == {
        "models/brain_triage_candidate_brain-triage-vTEST/model.safetensors",
        "data/train_release_brain-triage-vTEST/dataset_manifest.json"}
    capsys.readouterr()

    # 1. build rebuilt the batch set (e.g. after a relabel): studies.csv changed
    (fake_repo / "data" / "sets" / "fakebatch" / "studies.csv").write_text("patient_id,cls\n9000001,normal\n")
    runner.fail_on = "promote_model.py"
    n = len(runner.calls)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stage combine re-run: inputs changed" in out
    new = _scripts(runner.calls[n:])
    for script in ("scripts/combine_training_sets.py", "training/finetune_brain_classifier.py",
                   "scripts/eval_study_level.py"):
        assert script in new, script
    assert "scripts/build_training_set.py" not in new and "scripts/match_images_to_reports.py" not in new

    # 2. a legacy set (no studies.csv) gained a file: its listing changed → combine re-runs
    (fake_repo / "data" / "local_finetune_bin" / "train" / "normal" / "9000002_0.png").write_bytes(b"png")
    n = len(runner.calls)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"]) == 0
    out = capsys.readouterr().out
    assert "stage combine re-run: inputs changed" in out
    assert "scripts/combine_training_sets.py" in _scripts(runner.calls[n:])


def test_train_lock_is_passed_to_combine_and_fingerprinted(rp, fake_repo, tmp_path):
    (fake_repo / "data" / "train_lock.txt").write_text("9000003\n")
    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["plan", *_base_args(tmp_path, fake_repo), "--train-lock", "data/train_lock.txt"]))
    combine = {s.name: s for s in rp.build_stages(cfg)}["combine"]
    assert "--train-lock data/train_lock.txt" in " ".join(combine.cmds[0])
    assert "data/train_lock.txt" in combine.inputs


# ------------------------------------------------------------ tag persistence --

def _args_no_tag(tmp_path: Path, repo: Path, *extra: str) -> list[str]:
    return ["--batch", "fakebatch", "--repo", str(repo), "--root", str(tmp_path / "drive"),
            "--reports", str(tmp_path / "reports"), *extra]


def test_default_tag_follows_latest_until_the_release_is_complete(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_args_no_tag(tmp_path, fake_repo)]) == 0
    out = capsys.readouterr().out
    assert "tag: brain-triage-v2 (new release)" in out
    assert (fake_repo / "data" / "release" / "LATEST").read_text().strip() == "brain-triage-v2"

    # promote creates a _prev_ dir → default_tag() alone would now say v3; LATEST keeps v2
    (fake_repo / "models" / "brain_triage_finetuned_prev_0d559766ce58").mkdir()
    assert rp.default_tag(fake_repo / "models") == "brain-triage-v3"
    assert rp.main(["parse", *_args_no_tag(tmp_path, fake_repo)]) == 0
    assert "tag: brain-triage-v2 (from LATEST)" in capsys.readouterr().out
    assert rp.main(["status", "--batch", "fakebatch", "--repo", str(fake_repo)]) == 0
    out = capsys.readouterr().out
    assert "tag: brain-triage-v2 (from LATEST)" in out and "release brain-triage-v2" in out

    # another batch does not hijack the unfinished release: new tag, v2 dir skipped
    tag, src = rp.resolve_tag(fake_repo, "otherbatch", None)
    assert tag == "brain-triage-v3" and "--tag brain-triage-v2" in src

    # release complete (bundle done) → the next unused tag
    sp = fake_repo / "data" / "release" / "brain-triage-v2" / "state.json"
    st = json.loads(sp.read_text())
    st["stages"]["bundle"] = {"status": "done"}
    sp.write_text(json.dumps(st))
    assert rp.resolve_tag(fake_repo, "fakebatch", None) == ("brain-triage-v3", "new release")
    (fake_repo / "data" / "release" / "brain-triage-v3").mkdir()          # an abandoned attempt
    assert rp.resolve_tag(fake_repo, "fakebatch", None)[0] == "brain-triage-v4"
    assert rp.resolve_tag(fake_repo, "fakebatch", "mytag") == ("mytag", "--tag")


# ----------------------------------------------------------- option persistence --

def test_resume_line_reemits_every_option_and_resume_inherits_them(rp, fake_repo, tmp_path, monkeypatch,
                                                                    plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, fail_on="match_images_to_reports.py")
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--device", "cpu", "--epochs", "2",
                  "--label-llm", "ollama:x", "--min-normals", "7", "--local-bundle", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_STAGE_FAILED
    hint = [ln for ln in out.splitlines() if "resume with:" in ln][-1]
    for tok in ("--device cpu", "--epochs 2", "--label-llm ollama:x", "--min-normals 7", "--local-bundle",
                "--skip-tests", "--holdout data/holdout_patients_v1.txt", "--python", "--base-model",
                "--min-disk-gb 20", "--extra-sets data/local_finetune_bin", "--tag brain-triage-vTEST",
                "--root", "--reports", "--resume"):
        assert tok in hint, tok
    st = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert st["options"]["device"] == "cpu" and st["options"]["epochs"] == 2 and st["options"]["local_bundle"]

    # a bare --resume (no --tag, no options) continues the same release with the same options
    n = len(runner.calls)
    rc = rp.main(["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--resume"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "tag: brain-triage-vTEST (from LATEST)" in out and "options from state.json" in out
    train = [c for c in runner.calls[n:] if c[1] == "training/finetune_brain_classifier.py"][0]
    assert train[train.index("--device") + 1] == "cpu" and train[train.index("--epochs") + 1] == "2"
    bundle = [c for c in runner.calls[n:] if c[1] == "packaging/publish_model_bundle.py"][0]
    assert "--github" not in bundle
    # an explicit flag still wins over state.json
    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST", "--resume",
         "--device", "mps", "--no-skip-tests"]))
    assert cfg.device == "mps" and cfg.skip_tests is False and cfg.epochs == 2


def test_plain_run_does_not_inherit_options(rp, fake_repo, tmp_path, monkeypatch, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo), "--device", "cpu"]) == 0
    cfg = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo)]))
    assert cfg.device == "mps"
    cfg = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo), "--resume"]))
    assert cfg.device == "cpu"


# --------------------------------------------------------------------- guards --

def test_guard_output_goes_to_guards_log_and_verdict_to_state(rp, fake_repo, tmp_path, monkeypatch,
                                                              plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, n_normal=12)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == 0
    rel = fake_repo / "data" / "release" / "brain-triage-vTEST"
    log = (rel / "logs" / "guards.log").read_text(encoding="utf-8")
    assert "guard rails before train" in log and "guard rails before bundle" in log
    assert "matched brain studies: 62" in log and "only 12 normal studies" in log
    assert "passed" in log and "\x1b[" not in log                       # colour codes stripped
    st = json.loads((rel / "state.json").read_text())
    assert st["guards"]["pre_train"]["log"].endswith("logs/guards.log")
    assert st["guards"]["matched_counts"] == {"matched": 62, "normal": 12, "abnormal": 50}
    assert not st["guards"]["pre_train"]["errors"]


def test_missing_tune_split_stops_before_train(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    # the trainer is run with --select-split tune: without tune/ it would have to pick its
    # checkpoint on the val gate set, so the guard stops the release before training
    runner = FakeRunner(fake_repo, no_tune=True)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE and "tune/ is missing or empty" in out
    assert not any("finetune_brain_classifier.py" in c[1] for c in runner.calls)


@pytest.mark.parametrize("override,needle", [
    ({"gate": {"min_spec": 0.30}}, "gate.min_spec"),
    ({"gate": {"name": "gate max @0.50"}}, "not the production protocol"),
    ({"protocol": {"preprocessing": "pil_bicubic_224"}}, "protocol.preprocessing"),
    ({"holdout_covered": False}, "holdout_covered"),
    ({"manifest": {"files": {"model.safetensors": "11" * 32}}}, "MANIFEST.json pins"),
])
def test_guard_pre_bundle_applies_promote_release_rules(rp, fake_repo, tmp_path, monkeypatch, capsys,
                                                        override, needle):
    # gate.passed + promotion.promote alone are not enough: the bundle guard refuses whatever
    # promote_model.py would have refused (floors, protocol, val set, stale eval)
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    _write_eval(fake_repo / "models" / "brain_triage_finetuned", **override)
    rc = rp.main(["bundle", *_base_args(tmp_path, fake_repo), "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE and needle in out, out
    assert not any("publish_model_bundle.py" in c[1] for c in runner.calls)


def test_stage_env_has_loguru_info_and_unbuffered(rp, tmp_path):
    log = tmp_path / "env.log"
    code = "import os; print('ENV', os.environ.get('LOGURU_LEVEL'), os.environ.get('PYTHONUNBUFFERED'))"
    assert rp.run_command([sys.executable, "-c", code], log, tmp_path) == 0
    assert "ENV INFO 1" in log.read_text()


def test_inputs_changed_tolerates_an_unmounted_dir_but_not_a_changed_file(rp):
    assert not rp.inputs_changed({"/Volumes/x": "dir:12"}, {"/Volumes/x": "missing"})
    assert rp.inputs_changed({"/Volumes/x": "dir:12"}, {"/Volumes/x": "dir:13"})
    assert rp.inputs_changed({"a.csv": "aa"}, {"a.csv": "missing"})
    assert rp.inputs_changed({"a.csv": "aa"}, {"a.csv": "aa", "b.jsonl": "bb"})


def test_check_leakage_covers_the_tune_split(rp, tmp_path):
    rel = tmp_path / "rel"
    for split, pid in (("train", "1"), ("val", "9001"), ("tune", "1"), ("tune", "9001")):
        d = rel / split / "normal"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pid}_0.png").write_bytes(b"x")
    (rel / "dataset_manifest.json").write_text("{}")
    hold = tmp_path / "hold.txt"
    hold.write_text("9001\n")
    errs = rp.check_leakage(rel, hold)
    assert any("BOTH tune and train" in e for e in errs) and any("BOTH tune and val" in e for e in errs)


@pytest.mark.parametrize("eval_rc,headline,verdict", [(2, "GATE FAILED", "gate_failed"),
                                                      (4, "NOT PROMOTED", "not_promoted")])
def test_gate_exit_codes_stop_before_promote(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys,
                                             eval_rc, headline, verdict):
    runner = FakeRunner(fake_repo, eval_rc=eval_rc)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo)])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_STAGE_FAILED
    assert headline in out and "promote was NOT run" in out and "stopped before promote" in out
    assert "keep the old model" in out
    assert "--base-model google/vit-base-patch16-224" in out and "--from train" in out   # no offline copy here
    assert "scripts/promote_model.py" not in _scripts(runner.calls)
    st = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert st["stages"]["gate"]["status"] == "failed" and st["stages"]["gate"]["verdict"] == verdict
    assert "promote" not in st["stages"]
    # the offline ViT copy is preferred when present
    (fake_repo / "models" / "hf" / "google__vit-base-patch16-224").mkdir(parents=True)
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo)]))
    assert "--base-model models/hf/google__vit-base-patch16-224" in rp._retrain_hint(cfg)


def test_base_model_option_changes_only_the_train_start_point(rp, fake_repo, tmp_path):
    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["plan", *_base_args(tmp_path, fake_repo), "--base-model", "models/hf/google__vit-base-patch16-224"]))
    by = {s.name: " ".join(s.cmds[0]) for s in rp.build_stages(cfg)}
    assert "--base-model models/hf/google__vit-base-patch16-224" in by["train"]
    assert "--compare-with models/brain_triage_finetuned" in by["gate"]
    assert "--dest models/brain_triage_finetuned" in by["promote"]


# ---------------------------------------------------------------- LLM labels --

def test_llm_labels_are_imported_on_every_label_run(rp, fake_repo, tmp_path, monkeypatch, capsys):
    jsonl = fake_repo / "data" / "fakebatch_llm_labels.jsonl"
    jsonl.write_text('{"patient_id": "9000001", "study_date": "2031-01-02", "label": "normal", "abnormal": "no"}\n')
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo)]) == 0
    assert rp.main(["parse", *_base_args(tmp_path, fake_repo)]) == 0
    assert rp.main(["label", *_base_args(tmp_path, fake_repo), "--llm-labels", str(jsonl)]) == 0
    label = [c for c in runner.calls if c[1] == "scripts/label_reports.py" and "--benchmark" not in c][-1]
    assert label[label.index("--import-llm-labels") + 1] == "data/fakebatch_llm_labels.jsonl"   # repo-relative
    st = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert st["options"]["llm_labels"] == "data/fakebatch_llm_labels.jsonl"
    assert "data/fakebatch_llm_labels.jsonl" in st["stages"]["label"]["input_sha256s"]
    capsys.readouterr()
    # re-running the label stage later WITHOUT the flag keeps importing (never rule-only by accident)
    assert rp.main(["label", *_base_args(tmp_path, fake_repo)]) == 0
    assert "llm_labels=data/fakebatch_llm_labels.jsonl" in capsys.readouterr().out
    label = [c for c in runner.calls if c[1] == "scripts/label_reports.py" and "--benchmark" not in c][-1]
    assert "--import-llm-labels" in label
    # a new JSONL re-runs the stage on --resume
    jsonl.write_text(jsonl.read_text() + '{"patient_id": "9000002", "study_date": "2031-01-03", '
                                         '"label": "tumor", "abnormal": "yes"}\n')
    n = len(runner.calls)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--only", "label"]) == 0
    assert "inputs changed" in capsys.readouterr().out and len(runner.calls) > n


def test_manual_checklist_states_the_version_rule_and_evidence(rp, fake_repo, tmp_path):
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo)]))
    text = "\n".join(rp.manual_steps(cfg))
    assert "EVERY release bumps" in text and "desktop-app/package.json" in text
    assert "1.0.0 → 1.1.0" in text and "git tag v1.1.0" in text and "If this is a new app build" not in text
    assert "state.json" in text and "logs/guards.log" in text
    assert rp.next_minor_version("1.4.2") == "1.5.0" and rp.next_minor_version("x") == "<next version>"


# ------------------------------------------------------------ runbook commands --

BARE_PYTHON = re.compile(r"(?<![\w/.-])python3? (?:-m |scripts/|training/|packaging/)")


@pytest.mark.parametrize("name", RUNBOOKS)
def test_runbook_commands_are_runnable(name):
    text = (DOCS / name).read_text(encoding="utf-8")
    assert not BARE_PYTHON.findall(text), f"{name}: bare `python` (not on PATH) — use /Users/shakhzodbtr/venv/bin/python"
    # relabel path: every step is a command, not a description
    joined = re.sub(r"\\\n\s*", " ", text)                       # shell line continuations
    export = [ln for ln in joined.splitlines() if "--export-for-llm" in ln and "label_reports.py" in ln]
    assert export and all("--reports" in ln and "--out" in ln for ln in export), export
    imp = [ln for ln in joined.splitlines() if "--import-llm-labels" in ln and "label_reports.py" in ln]
    assert imp and all("--reports" in ln and "--out" in ln for ln in imp), imp
    assert "Workflow({" in text and "training/relabel_workflow.js" in text
    assert "--import-llm-labels" in text and "--llm-labels" in text
    assert re.search(r"venv/bin/python scripts/release_pipeline\.py run .*--from match .*--tag .*--resume", joined)
    # state model, tag persistence, evidence, version rule, fallback base
    for tok in ("data/release/LATEST", "logs/guards.log", "state.json", "desktop-app/package.json",
                "1.0.0 → 1.1.0", "--base-model", "google__vit-base-patch16-224", "tune/", "archived"):
        assert tok in text, (name, tok)
    assert "pre-train guard output in `logs/train.log`" not in text


# ------------------------------------------------------------ review round 2 --

def _hint_argv(out: str, marker: str) -> list[str]:
    line = [ln for ln in out.splitlines() if marker in ln][-1]
    cmd = line.split(marker, 1)[1].strip()
    return shlex.split(cmd)[3:]              # drop "<python> scripts/release_pipeline.py run"


def test_printed_retrain_hint_really_retrains_from_the_public_base(rp, fake_repo, tmp_path, monkeypatch,
                                                                  plenty_of_disk, capsys):
    """Gate fails → the printed 'retrain from the public ViT base' line must run train again
    (it used to skip train as 'already done' and re-gate the same candidate)."""
    runner = FakeRunner(fake_repo, eval_rc=2)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    argv = _hint_argv(capsys.readouterr().out, "retrain from the public ViT base:")
    assert "--resume" not in argv
    runner.eval_rc = 0
    n = len(runner.calls)
    assert rp.main(["run", *argv, "--skip-tests"]) == 0
    trains = [c for c in runner.calls[n:] if c[1] == "training/finetune_brain_classifier.py"]
    assert len(trains) == 1 and trains[0][trains[0].index("--base-model") + 1] == "google/vit-base-patch16-224"


def test_resume_with_a_new_base_model_reruns_train(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, eval_rc=2)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    capsys.readouterr()
    runner.eval_rc = 0
    n = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--base-model", "google/vit-base-patch16-224",
                  "--from", "train", "--resume", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "stage train re-run: inputs changed" in out
    assert "training/finetune_brain_classifier.py" in _scripts(runner.calls[n:])


def test_resume_after_promote_ignores_ds_store_and_changed_upstream_inputs(rp, fake_repo, tmp_path, monkeypatch,
                                                                          plenty_of_disk, capsys):
    """Bundle fails after promote; meanwhile Finder drops a .DS_Store into the legacy set and a set is rebuilt.
    The printed resume line must still finish the release (bundle only) and never rewrite the release set."""
    runner = FakeRunner(fake_repo, fail_on="publish_model_bundle.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    argv = _hint_argv(capsys.readouterr().out, "resume with:")
    (fake_repo / "data" / "local_finetune_bin" / ".DS_Store").write_bytes(b"\0")
    (fake_repo / "data" / "local_finetune_bin" / "train" / ".DS_Store").write_bytes(b"\0")
    (fake_repo / "data" / "sets" / "fakebatch" / "studies.csv").write_text("patient_id,cls\n9000001,normal\n")
    manifest = fake_repo / "data" / "train_release_brain-triage-vTEST" / "dataset_manifest.json"
    before = manifest.read_text()
    n = len(runner.calls)
    rc = rp.main(["run", *argv])
    out = capsys.readouterr().out
    assert rc == 0, out
    new = _scripts(runner.calls[n:])
    assert "scripts/combine_training_sets.py" not in new and "training/finetune_brain_classifier.py" not in new
    assert "packaging/publish_model_bundle.py" in new
    assert manifest.read_text() == before
    # combine explicitly requested under the promoted tag is refused (it would rewrite the pinned val set)
    n = len(runner.calls)
    assert rp.main(["combine", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    assert "already promoted" in capsys.readouterr().out and len(runner.calls) == n


def test_listing_sha_ignores_hidden_files(rp, tmp_path):
    d = tmp_path / "set"
    (d / "train" / "normal").mkdir(parents=True)
    (d / "train" / "normal" / "9000001_0.png").write_bytes(b"png")
    before = rp.listing_sha256(d)
    (d / ".DS_Store").write_bytes(b"x")
    (d / "train" / "._9000001_0.png").write_bytes(b"x")
    (d / ".hidden").mkdir()
    (d / ".hidden" / "a.png").write_bytes(b"x")
    assert rp.listing_sha256(d) == before
    (d / "train" / "normal" / "9000002_0.png").write_bytes(b"png")
    assert rp.listing_sha256(d) != before


def _trained_on_v2(repo: Path) -> Path:
    prev = repo / "data" / "train_release_brain-triage-v2"
    (prev / "tune" / "normal").mkdir(parents=True)
    (prev / "dataset_manifest.json").write_text(json.dumps(
        {"splits": {"val": {"patients_sha256": _sha_ids(HOLDOUT_IDS)}}}))
    (repo / "models" / "brain_triage_finetuned" / "training_metrics.json").write_text(json.dumps(
        {"checkpoint_selection": {"split": "tune", "data": "data/train_release_brain-triage-v2/tune"}}))
    return prev


def test_train_lock_fails_closed_when_the_trained_on_set_is_gone_or_rebuilt(rp, fake_repo, tmp_path, monkeypatch,
                                                                           plenty_of_disk, capsys):
    prev = _trained_on_v2(fake_repo)
    args = [a if a != "brain-triage-vTEST" else "brain-triage-v3" for a in _base_args(tmp_path, fake_repo)]
    assert rp.train_lock_problem(fake_repo, "data/train_release_brain-triage-v3") is None
    # rebuilt: the set's val hash no longer matches the release eval
    (prev / "dataset_manifest.json").write_text(json.dumps({"splits": {"val": {"patients_sha256": "ab" * 32}}}))
    assert "rebuilt" in rp.train_lock_problem(fake_repo, "data/train_release_brain-triage-v3")
    # gone
    (prev / "dataset_manifest.json").unlink()
    problem = rp.train_lock_problem(fake_repo, "data/train_release_brain-triage-v3")
    assert problem and "missing" in problem
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *args]))
    assert cfg.train_lock is None
    rp.cmd_plan(cfg, rp.build_stages(cfg))
    assert "!! NONE" in capsys.readouterr().out
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *args]) == rp.EXIT_USAGE
    out = capsys.readouterr().out
    assert "guard rails before combine failed" in out and "--train-lock" in out
    assert "scripts/combine_training_sets.py" not in _scripts(runner.calls)
    # an explicit --train-lock file lets the release go on
    (fake_repo / "data" / "v2_trained_ids.txt").write_text("10\n11\n12\n")
    assert rp.main(["run", *args, "--train-lock", "data/v2_trained_ids.txt", "--skip-tests"]) == 0


def test_interrupted_stage_is_recorded_failed_with_a_resume_line(rp, fake_repo, tmp_path, monkeypatch, capsys):
    def interrupted(argv, log_path, cwd, dry_run=False):
        raise KeyboardInterrupt
    monkeypatch.setattr(rp, "run_command", interrupted)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo)]) == 130
    out = capsys.readouterr().out
    assert "INTERRUPTED" in out and "resume with:" in out
    st = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert st["stages"]["index"]["status"] == "failed"
