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

EXPECTED_ORDER = ("index", "parse", "label", "match", "build", "prospective", "combine",
                  "train", "gate", "promote", "bump", "bundle")
CONTRACT_SCRIPTS = {
    "index": "scripts/index_new_studies.py",
    "parse": "scripts/parse_hospital_reports.py",
    "label": "scripts/label_reports.py",
    "match": "scripts/match_images_to_reports.py",
    "build": "scripts/build_training_set.py",
    "prospective": "scripts/eval_prospective.py",
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
                 leak: bool = False, eval_rc: int = 0, no_tune: bool = False,
                 cand_eval: dict | None = None):
        self.repo = repo
        # multi-candidate runs: {'warm'|'fresh': (eval exit code, balanced accuracy)}, keyed by the
        # candidate dir suffix models/brain_triage_candidate_<tag>_<cand>
        self.cand_eval = cand_eval or {}
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
        elif script.endswith("eval_prospective.py"):
            # the shape eval_prospective.py writes; exit 0 = it ran (never a gate)
            out = repo / self._arg(argv, "--out")
            out.parent.mkdir(parents=True, exist_ok=True)
            group = {"n_studies": 412, "sensitivity": 0.92, "specificity": 0.27}
            out.write_text(json.dumps({"kind": "prospective_eval", "model": self._arg(argv, "--model"),
                                       "groups": {"all": group, "new": group}}))
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
            model = self._arg(argv, "--model")
            cand = model.rsplit("_", 1)[-1]
            if cand in self.cand_eval:
                rc, bal = self.cand_eval[cand]
                _write_eval(repo / model, passed=rc != 2, promote=rc == 0, weights=False,
                            data=self._arg(argv, "--data"), gate={"balanced": bal})
                return rc
            _write_eval(repo / model, passed=self.eval_rc != 2,
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
    assert joined["bundle"].endswith("--out data/release/brain-triage-vTEST/ --github SHAXZODBR/SIAA-models")
    # no --tag: the publisher's default brain-triage-<identity12> is the tag the installer CI looks up
    assert "--tag" not in joined["bundle"]
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

def test_guard_missing_drive_does_not_block_train(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    """train reads only local slices: an unmounted --root is no longer a train guard error (review 2)."""
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
    assert rc == 0, out
    assert "drive not mounted" not in out
    assert _scripts(runner.calls[n:]) == ["training/finetune_brain_classifier.py"]
    state = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert state["guards"]["pre_train"]["passed"] is True
    assert not any("drive" in e for e in state["guards"]["pre_train"]["errors"] +
                   state["guards"]["pre_train"]["warnings"])


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
    # the app version is quoted in code + 12 hospital docs; the checklist names all of them
    assert "APP_VERSION" in text and "test_app_version_quoted_in_docs_matches_the_code" in text
    assert "docs/{user_manual,admin_install_guide,intended_use,pilot_evaluation_protocol}_{en,ru,uz}.md" in text
    assert "Setup 1.1.0.exe" in text
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
    # the prospective stage (before combine) already refuses: it would need that set as --seen
    assert "guard rails before prospective failed" in out and "--train-lock" in out
    assert rp.main(["combine", *args]) == rp.EXIT_USAGE
    out = capsys.readouterr().out
    assert "guard rails before combine failed" in out and "--train-lock" in out
    assert "scripts/combine_training_sets.py" not in _scripts(runner.calls)
    # an explicit --train-lock file satisfies the combine lock
    (fake_repo / "data" / "v2_trained_ids.txt").write_text("10\n11\n12\n")
    assert rp.main(["combine", *args, "--train-lock", "data/v2_trained_ids.txt"]) == 0
    # the whole release goes on with the set restored by hand as --train-lock <dir> (also the --seen set)
    (fake_repo / "data" / "v2_restored" / "train" / "normal").mkdir(parents=True)
    assert rp.main(["run", *args, "--train-lock", "data/v2_restored", "--skip-tests"]) == 0


def test_interrupted_stage_is_recorded_failed_with_a_resume_line(rp, fake_repo, tmp_path, monkeypatch, capsys):
    def interrupted(argv, log_path, cwd, dry_run=False):
        raise KeyboardInterrupt
    monkeypatch.setattr(rp, "run_command", interrupted)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo)]) == 130
    out = capsys.readouterr().out
    assert "INTERRUPTED" in out and "resume with:" in out
    st = json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())
    assert st["stages"]["index"]["status"] == "failed"


# ------------------------------------------- D6: --candidates warm,fresh ----

VIT_LOCAL = "models/hf/google__vit-base-patch16-224"
CAND = "models/brain_triage_candidate_brain-triage-vTEST"


def _offline_vit(repo: Path) -> Path:
    """A (tiny, fake) complete offline copy of the public ViT base."""
    d = repo / VIT_LOCAL
    d.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "preprocessor_config.json"):
        (d / f).write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    return d


def _fake_hf_cache(root: Path, complete: bool = True) -> Path:
    """The huggingface_hub cache layout: models--org--name/{refs/main, snapshots/<rev>/…}."""
    m = root / "models--google--vit-base-patch16-224"
    (m / "refs").mkdir(parents=True)
    (m / "refs" / "main").write_text("0123abcd")
    snap = m / "snapshots" / "0123abcd"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "preprocessor_config.json").write_text("{}")
    if complete:
        (snap / "model.safetensors").write_bytes(b"\0" * 16)
    return root


def _state(repo: Path) -> dict:
    return json.loads((repo / "data" / "release" / "brain-triage-vTEST" / "state.json").read_text())


def _cfg(rp, tmp_path, repo, *extra):
    return rp.config_from_args(rp.build_parser().parse_args(["plan", *_base_args(tmp_path, repo), *extra]))


def test_candidates_default_and_explicit_warm_are_the_single_candidate_pipeline(rp, fake_repo, tmp_path):
    default = _cfg(rp, tmp_path, fake_repo)
    explicit = _cfg(rp, tmp_path, fake_repo, "--candidates", "warm")
    assert default.candidates == explicit.candidates == ["warm"] and not default.multi
    assert [s.cmds for s in rp.build_stages(default)] == [s.cmds for s in rp.build_stages(explicit)]
    by = {s.name: s for s in rp.build_stages(default)}
    assert len(by["train"].cmds) == len(by["gate"].cmds) == 1
    assert by["train"].cmds[0][-1] == CAND and "--base-model-fresh" not in by["train"].cmds[0]
    assert "--candidates" not in rp.args_line(default)
    assert default.options()["candidates"] == ["warm"]


@pytest.mark.parametrize("bad", ["", "hot", "warm,warm", "warm,fresh,cold"])
def test_candidates_option_rejects_unknown_or_duplicate(rp, fake_repo, tmp_path, bad):
    with pytest.raises(SystemExit):
        rp.build_parser().parse_args(["plan", *_base_args(tmp_path, fake_repo), "--candidates", bad])


def test_two_candidates_train_and_gate_separately_and_plan_shows_them(rp, fake_repo, tmp_path, capsys):
    _offline_vit(fake_repo)
    cfg = _cfg(rp, tmp_path, fake_repo, "--candidates", "fresh,warm")       # order normalised: warm first
    assert cfg.candidates == ["warm", "fresh"] and cfg.multi
    by = {s.name: s for s in rp.build_stages(cfg)}
    warm, fresh = by["train"].cmds
    assert warm[warm.index("--base-model") + 1] == "models/brain_triage_finetuned"
    assert warm[-1] == f"{CAND}_warm" and "--base-model-fresh" not in warm
    assert fresh[fresh.index("--base-model") + 1] == VIT_LOCAL and "--base-model-fresh" in fresh
    assert fresh[-1] == f"{CAND}_fresh" and "--select-split" in fresh
    gates = [" ".join(c) for c in by["gate"].cmds]
    assert len(gates) == 2 and all("--compare-with models/brain_triage_finetuned" in g for g in gates)
    assert f"--model {CAND}_warm " in gates[0] and f"--model {CAND}_fresh " in gates[1]
    assert f"{CAND}_<chosen>" in by["promote"].cmds[0]
    assert "opt:candidates" in by["train"].inputs
    assert "--candidates warm,fresh" in rp.args_line(cfg)
    rp.cmd_plan(cfg, rp.build_stages(cfg))
    out = capsys.readouterr().out
    assert "# candidates: warm from models/brain_triage_finetuned" in out
    assert f"fresh from {VIT_LOCAL} (--base-model-fresh) → {CAND}_fresh" in out
    assert "fresh base available offline" in out          # the extra pre-train guard is listed
    for ln in out.splitlines():
        if ln and not ln.startswith("#"):
            shlex.split(ln)


@pytest.mark.parametrize("cand_eval,chosen", [
    ({"warm": (2, 0.80), "fresh": (0, 0.70)}, "fresh"),      # warm fails the gate, fresh passes
    ({"warm": (0, 0.70), "fresh": (0, 0.75)}, "fresh"),      # both pass: higher balanced accuracy wins
    ({"warm": (0, 0.78), "fresh": (0, 0.74)}, "warm"),
    ({"warm": (0, 0.72), "fresh": (0, 0.72)}, "warm"),       # tie → warm
    ({"warm": (4, 0.90), "fresh": (0, 0.71)}, "fresh"),      # not promoted never wins, whatever its numbers
])
def test_promote_takes_the_best_candidate_that_passed(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk,
                                                      capsys, cand_eval, chosen):
    _offline_vit(fake_repo)
    runner = FakeRunner(fake_repo, cand_eval=cand_eval)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == 0, out
    trains = [c for c in runner.calls if c[1] == "training/finetune_brain_classifier.py"]
    assert [c[-1] for c in trains] == [f"{CAND}_warm", f"{CAND}_fresh"]
    promote = [c for c in runner.calls if c[1] == "scripts/promote_model.py"]
    assert len(promote) == 1 and promote[0][promote[0].index("--candidate") + 1] == f"{CAND}_{chosen}"
    other = "warm" if chosen == "fresh" else "fresh"
    assert (fake_repo / f"{CAND}_{other}").is_dir() and not (fake_repo / f"{CAND}_{chosen}").exists()
    st = _state(fake_repo)
    choice = st["candidate_choice"]
    assert choice["chosen"] == chosen and choice["path"] == f"{CAND}_{chosen}" and "tie → warm" in choice["rule"]
    assert {r["candidate"]: r["qualified"] for r in choice["ranking"]} == \
        {c: rc_bal[0] == 0 for c, rc_bal in cand_eval.items()}
    for cand, (erc, bal) in cand_eval.items():
        rec = st["candidates"][cand]
        assert rec["path"] == f"{CAND}_{cand}" and rec["train"]["status"] == "done"
        assert rec["train"]["returncode"] == 0 and rec["gate"]["returncode"] == erc
        assert rec["gate"]["balanced"] == pytest.approx(bal)
        assert rec["gate"]["verdict"] == {0: "passed", 2: "gate_failed", 4: "not_promoted"}[erc]
    assert st["candidates"]["fresh"]["base_model"] == VIT_LOCAL and st["candidates"]["fresh"]["fresh_head"]
    assert st["stages"]["promote"]["promoted_candidate"] == f"{CAND}_{chosen}"
    assert st["stages"]["promote"]["chosen_candidate"] == chosen
    assert st["stages"]["train"]["status"] == st["stages"]["gate"]["status"] == "archived"
    assert st["options"]["candidates"] == ["warm", "fresh"]
    assert f"chosen: {chosen}" in out
    # status shows the per-candidate results and the choice
    assert rp.main(["status", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST"]) == 0
    status = capsys.readouterr().out
    assert "candidates (promote = best balanced accuracy" in status and f"chosen: {chosen}" in status
    assert f"{CAND}_warm" in status and f"{CAND}_fresh" in status


@pytest.mark.parametrize("cand_eval,headline,verdict,rc_want", [
    ({"warm": (2, 0.6), "fresh": (2, 0.6)}, "GATE FAILED", "gate_failed", 2),
    ({"warm": (4, 0.7), "fresh": (2, 0.6)}, "NOT PROMOTED", "not_promoted", 4),
])
def test_no_candidate_qualifies_stops_before_promote(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk,
                                                     capsys, cand_eval, headline, verdict, rc_want):
    _offline_vit(fake_repo)
    runner = FakeRunner(fake_repo, cand_eval=cand_eval)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh"])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_STAGE_FAILED
    assert headline in out and "promote was NOT run" in out and "stopped before promote" in out
    assert "keep the old model" in out and "chosen: none" in out
    assert f"{CAND}_warm/study_level_eval.json" in out and f"{CAND}_fresh/study_level_eval.json" in out
    assert "scripts/promote_model.py" not in _scripts(runner.calls)
    assert len([c for c in runner.calls if c[1] == "scripts/eval_study_level.py"]) == 2   # both were gated
    st = _state(fake_repo)
    assert st["stages"]["gate"]["status"] == "failed" and st["stages"]["gate"]["verdict"] == verdict
    assert st["stages"]["gate"]["returncode"] == rc_want
    assert st["candidate_choice"]["chosen"] is None and "promote" not in st["stages"]
    assert (fake_repo / "models" / "brain_triage_finetuned" / "study_level_eval.json").is_file()   # untouched


def test_gate_crash_of_one_candidate_stops_the_stage(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    _offline_vit(fake_repo)
    runner = FakeRunner(fake_repo, cand_eval={"warm": (3, 0.8), "fresh": (0, 0.8)})
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh"]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "MANIFEST check failed" in out and "resume with:" in out and "--candidates warm,fresh" in out
    st = _state(fake_repo)
    assert st["stages"]["gate"]["failed_candidate"] == "warm" and st["candidates"]["warm"]["gate"]["verdict"] == "error"
    assert len([c for c in runner.calls if c[1] == "scripts/eval_study_level.py"]) == 1


def test_resume_does_not_retrain_a_finished_candidate(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    _offline_vit(fake_repo)

    class FailFreshOnce(FakeRunner):
        failed = False

        def __call__(self, argv, log_path, cwd, dry_run=False):
            if "--base-model-fresh" in argv and not self.failed and not dry_run:
                self.failed = True
                self.calls.append(list(argv))
                return 1
            return super().__call__(argv, log_path, cwd, dry_run)

    runner = FailFreshOnce(fake_repo, cand_eval={"warm": (0, 0.7), "fresh": (0, 0.8)})
    monkeypatch.setattr(rp, "run_command", runner)
    args = [*_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh", "--skip-tests"]
    assert rp.main(["run", *args]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stage train FAILED" in out
    st = _state(fake_repo)
    assert st["candidates"]["warm"]["train"]["status"] == "done"
    assert st["candidates"]["fresh"]["train"]["status"] == "failed" and st["stages"]["train"]["failed_candidate"] == "fresh"
    # the printed resume line (it carries --candidates) retrains fresh only
    argv = _hint_argv(out, "resume with:")
    assert "--candidates" in argv and argv[argv.index("--candidates") + 1] == "warm,fresh"
    n = len(runner.calls)
    assert rp.main(["run", *argv]) == 0
    out = capsys.readouterr().out
    assert "candidate warm already trained" in out
    trains = [c for c in runner.calls[n:] if c[1] == "training/finetune_brain_classifier.py"]
    assert [c[-1] for c in trains] == [f"{CAND}_fresh"]
    assert _state(fake_repo)["candidate_choice"]["chosen"] == "fresh"
    # a bare --resume inherits the candidates from state.json
    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST", "--resume"]))
    assert cfg.candidates == ["warm", "fresh"] and cfg.inherited["candidates"] == ["warm", "fresh"]


# ---------------------------------------------------- fresh base offline guard --

def test_fresh_base_resolution_prefers_the_offline_copy(rp, fake_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty_cache"))
    assert rp.fresh_base(fake_repo) == "google/vit-base-patch16-224"
    problem = rp.fresh_base_problem(fake_repo, "/x/python")
    assert problem.startswith("fresh base not available offline") and "snapshot_download" in problem
    assert str(tmp_path / "empty_cache") in problem
    # an incomplete offline copy is named as such
    d = fake_repo / VIT_LOCAL
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert rp.fresh_base(fake_repo) == VIT_LOCAL
    problem = rp.fresh_base_problem(fake_repo)
    assert "incomplete" in problem and "preprocessor_config.json" in problem and "model.safetensors" in problem
    _offline_vit(fake_repo)
    assert rp.fresh_base_problem(fake_repo) is None


def test_fresh_base_from_the_hf_cache(rp, fake_repo, tmp_path, monkeypatch):
    cache = _fake_hf_cache(tmp_path / "hf_cache", complete=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert rp.hf_hub_cache_dir() == cache
    assert not rp.hf_cache_has_model("google/vit-base-patch16-224")           # no weights yet
    (cache / "models--google--vit-base-patch16-224" / "snapshots" / "0123abcd" / "model.safetensors").write_bytes(b"\0")
    assert rp.hf_cache_has_model("google/vit-base-patch16-224")
    assert rp.fresh_base_problem(fake_repo) is None
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert rp.hf_hub_cache_dir() == tmp_path / "home" / "hub"


def test_fresh_base_missing_stops_before_any_training(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty_cache"))
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh"])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert "fresh base not available offline" in out and "guard rails before train failed" in out
    assert "training/finetune_brain_classifier.py" not in _scripts(runner.calls)       # not even warm
    st = _state(fake_repo)
    assert any("fresh base not available offline" in e for e in st["guards"]["pre_train"]["errors"])
    # once the cache holds it, the same resume trains both — fresh from the hub id, offline
    shutil.rmtree(tmp_path / "empty_cache", ignore_errors=True)
    _fake_hf_cache(tmp_path / "empty_cache")
    runner.cand_eval = {"warm": (0, 0.7), "fresh": (0, 0.7)}
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh", "--resume",
                    "--skip-tests"]) == 0
    fresh = [c for c in runner.calls if "--base-model-fresh" in c][0]
    assert fresh[fresh.index("--base-model") + 1] == "google/vit-base-patch16-224"
    assert _state(fake_repo)["candidate_choice"]["chosen"] == "warm"


def test_warm_only_never_checks_the_fresh_base(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk):
    monkeypatch.setattr(rp, "fresh_base_problem", lambda *a, **k: pytest.fail("checked the fresh base"))
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--skip-tests"]) == 0


def test_stage_env_is_offline_for_the_hub_unless_overridden(rp, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert rp.stage_env()["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert rp.stage_env()["HF_HUB_OFFLINE"] == "0"


def test_single_candidate_gate_failure_also_offers_both_candidates(rp, fake_repo, tmp_path, monkeypatch,
                                                                   plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo, eval_rc=2)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    argv = _hint_argv(capsys.readouterr().out, "train both candidates and promote the better one:")
    assert argv[argv.index("--candidates") + 1] == "warm,fresh" and "--from" in argv and "--resume" not in argv


@pytest.mark.parametrize("name", RUNBOOKS)
def test_runbook_covers_candidates_intake_mirror_signing_and_clinic_delivery(rp, name):
    text = (DOCS / name).read_text(encoding="utf-8")
    for tok in ("--candidates warm,fresh", "--base-model-fresh", "fresh base not available offline",
                "candidate_choice", "scripts/intake_batch.py", "scripts/backup_mirror.sh", "CHECKSUMS.sha256",
                "raw/INVENTORY.csv", "--resume", "--verify", "WIN_CSC_LINK", "WIN_CSC_KEY_PASSWORD",
                "Azure Trusted Signing", "OV", "NO-VALIDATED-MODEL", "packaging/provision_clinic.py"):
        assert tok in text, (name, tok)
    lang = name.rsplit("_", 1)[1][:2]
    assert f"vendor_provisioning_{lang}.md" in text
    # every intake / mirror flag the runbook quotes exists in the scripts
    intake_src = (REPO / "scripts" / "intake_batch.py").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n\s*", " ", text)
    for ln in joined.splitlines():
        if "intake_batch.py" in ln:
            for flag in re.findall(r"(--[a-z][a-z-]+)", ln):
                assert f"'{flag}'" in intake_src, (name, flag)
    mirror_src = (REPO / "scripts" / "backup_mirror.sh").read_text(encoding="utf-8")
    for flag in ("--yes", "--sample", "--verify-only", "--dry-run"):
        assert flag in mirror_src
    assert "--candidates" in " ".join(rp.build_parser().format_help().split())


def test_verdict_stop_prints_a_resume_line_that_reruns_only_the_gate(rp, fake_repo, tmp_path, monkeypatch,
                                                                     plenty_of_disk, capsys):
    """Dress rehearsal 2026-11-09: a gate verdict (no candidate qualifies) used to print only a status
    line. It must also print the resume line, and that line must skip every done stage (index → train)
    and re-run just the gate — never promote while no candidate qualifies."""
    _offline_vit(fake_repo)
    runner = FakeRunner(fake_repo, cand_eval={"warm": (4, 0.66), "fresh": (2, 0.66)})
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--candidates", "warm,fresh"]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stopped before promote" in out
    argv = _hint_argv(out, "then the gate):")
    assert "--resume" in argv and argv[argv.index("--candidates") + 1] == "warm,fresh"
    assert argv[argv.index("--tag") + 1] == "brain-triage-vTEST"
    runner.calls.clear()
    assert rp.main(["run", *argv]) == rp.EXIT_STAGE_FAILED          # same weights → same verdict
    out = capsys.readouterr().out
    for stage in ("index", "parse", "label", "match", "build", "combine", "train"):
        assert f"stage {stage} already done" in out, stage
    assert _scripts(runner.calls) == ["scripts/eval_study_level.py", "scripts/eval_study_level.py"]
    st = _state(fake_repo)
    assert st["stages"]["gate"]["verdict"] == "not_promoted" and "promote" not in st["stages"]


# ------------------------------- several report sources, MR-only rule, prospective --

def _args_no_reports(tmp_path: Path, repo: Path, *extra: str) -> list[str]:
    return ["--batch", "fakebatch", "--repo", str(repo), "--root", str(tmp_path / "drive"),
            "--tag", "brain-triage-vTEST", *extra]


def _second_reports(tmp_path: Path) -> Path:
    d = tmp_path / "reports_b"
    d.mkdir(exist_ok=True)
    (d / "sheet_2031.xlsx").write_bytes(b"PK synthetic")
    return d


def _stage(rp, cfg, name: str):
    return {s.name: s for s in rp.build_stages(cfg)}[name]


def test_twelve_stages_and_plan_lists_prospective_between_build_and_combine(rp, fake_repo, tmp_path, capsys):
    assert len(rp.STAGE_ORDER) == 12 and "prospective" in rp.SUBCOMMANDS
    assert rp.STAGE_ORDER.index("prospective") == rp.STAGE_ORDER.index("build") + 1
    assert rp.STAGE_ORDER.index("combine") == rp.STAGE_ORDER.index("prospective") + 1
    assert rp.main(["plan", *_base_args(tmp_path, fake_repo)]) == 0
    out = capsys.readouterr().out
    i_build, i_pro, i_comb = (out.index(f"# --- stage {k}: {n}") for k, n in ((4, "build"), (5, "prospective"),
                                                                              (6, "combine")))
    assert i_build < i_pro < i_comb
    assert "# modality rule (build): MR only (--modality MR)" in out
    assert f"# report sources (oldest delivery first): {tmp_path / 'reports'}" in out
    assert "eleven" not in rp.__doc__ and "eleven" not in rp.build_parser().format_help()
    assert "twelve" in rp.__doc__


def test_reports_one_several_and_repeated_flag(rp, fake_repo, tmp_path):
    a, b = tmp_path / "reports", _second_reports(tmp_path)
    one = _cfg(rp, tmp_path, fake_repo)
    assert one.reports == [a]
    several = rp.config_from_args(rp.build_parser().parse_args(
        ["plan", *_args_no_reports(tmp_path, fake_repo), "--reports", str(a), str(b)]))
    repeated = rp.config_from_args(rp.build_parser().parse_args(
        ["plan", *_args_no_reports(tmp_path, fake_repo), "--reports", str(a), "--reports", str(b)]))
    assert several.reports == repeated.reports == [a, b]
    parse = _stage(rp, several, "parse").cmds[0]
    i = parse.index("--src")
    assert parse[i + 1:i + 3] == [str(a), str(b)] and parse[i + 3] == "--out"
    assert several.options()["reports"] == [str(a), str(b)]
    # no --reports at all: an empty list, and the parse stage refuses
    none = rp.config_from_args(rp.build_parser().parse_args(["plan", *_args_no_reports(tmp_path, fake_repo)]))
    assert none.reports == [] and none.options()["reports"] == []
    assert rp.main(["parse", *_args_no_reports(tmp_path, fake_repo)]) == rp.EXIT_USAGE


def test_reports_state_round_trip_and_args_line(rp, fake_repo, tmp_path, monkeypatch, capsys):
    a, b = tmp_path / "reports", _second_reports(tmp_path)
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["parse", *_base_args(tmp_path, fake_repo), "--reports", str(b), "--modality", "any"]) == 0
    st = _state(fake_repo)
    assert st["options"]["reports"] == [str(a), str(b)] and st["options"]["modality"] == "any"
    parse = [c for c in runner.calls if c[1] == "scripts/parse_hospital_reports.py"][-1]
    assert parse[parse.index("--src") + 1:parse.index("--out")] == [str(a), str(b)]
    # a bare --resume inherits both sources (in order) and the modality
    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST", "--resume"]))
    assert cfg.reports == [a, b] and cfg.modality == "any"
    assert cfg.inherited["reports"] == [str(a), str(b)] and cfg.inherited["modality"] == "any"
    # the printed resume line reproduces the same options through the parser
    line = rp.args_line(cfg)
    assert f"--reports {shlex.quote(str(a))} {shlex.quote(str(b))}" in line and "--modality any" in line
    again = rp.config_from_args(rp.build_parser().parse_args(["run", *shlex.split(line)]))
    assert again.reports == cfg.reports and again.modality == "any" and again.options() == cfg.options()
    default = _cfg(rp, tmp_path, fake_repo)
    again = rp.config_from_args(rp.build_parser().parse_args(["run", *shlex.split(rp.args_line(default))]))
    assert again.options() == default.options() and again.modality == "MR"


def test_legacy_string_reports_state_keeps_parse_and_label_done(rp, fake_repo, tmp_path, monkeypatch, capsys):
    """The release in progress on the real machine: state.json written by the previous version stores
    "reports": "<folder>" and parse + label are done. Both must stay done (no re-parse)."""
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["parse", *_base_args(tmp_path, fake_repo)]) == 0
    assert rp.main(["label", *_base_args(tmp_path, fake_repo)]) == 0
    sp = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    st = json.loads(sp.read_text())
    folder = str(tmp_path / "reports")
    # exactly what the previous version wrote: ONE string, no modality, parse fingerprinted as the folder
    st["options"]["reports"] = folder
    st["options"].pop("modality")
    st["stages"]["parse"]["input_sha256s"] = {folder: "dir:1"}
    sp.write_text(json.dumps(st))
    capsys.readouterr()

    cfg = rp.config_from_args(rp.build_parser().parse_args(
        ["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST", "--resume"]))
    assert cfg.reports == [Path(folder)] and cfg.options()["reports"] == [folder]
    assert cfg.modality == "MR" and "modality" not in cfg.inherited
    # one source: the parse fingerprint is the one the old string form produced
    assert rp.input_fingerprints(cfg, _stage(rp, cfg, "parse"), st) == {folder: "dir:1"}
    for name in ("parse", "label"):
        assert rp.stage_is_done(cfg, _stage(rp, cfg, name), st) == (True, ""), name

    n = len(runner.calls)
    assert rp.main(["run", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST",
                    "--resume", "--only", "parse"]) == 0
    assert "stage parse already done — skipped" in capsys.readouterr().out and len(runner.calls) == n
    assert rp.main(["status", "--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vTEST"]) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "parse    done" in out and "label    done" in out and "prospective pending" in out


def test_parse_fingerprint_follows_sources_and_their_order(rp, fake_repo, tmp_path):
    a, b = tmp_path / "reports", _second_reports(tmp_path)

    def fp(*sources):
        args = [x for s in sources for x in ("--reports", str(s))]
        cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_args_no_reports(tmp_path, fake_repo),
                                                                *args]))
        return rp.input_fingerprints(cfg, _stage(rp, cfg, "parse"), {})

    assert fp(a) == {str(a): "dir:1"}                               # one source: no order marker
    assert rp.inputs_changed(fp(a), fp(a, b))                       # a source added
    assert rp.inputs_changed(fp(a, b), fp(b, a))                    # order swapped
    assert not rp.inputs_changed(fp(a, b), fp(a, b))
    assert fp(a, b)["opt:report_order"] == f"opt:{a} | {b}"


def test_modality_default_any_inherited_and_legacy(rp, fake_repo, tmp_path, monkeypatch):
    build = _stage(rp, _cfg(rp, tmp_path, fake_repo), "build").cmds[0]
    assert build[build.index("--modality") + 1] == "MR"
    assert "opt:modality" in _stage(rp, _cfg(rp, tmp_path, fake_repo), "build").inputs
    assert "--modality" not in _stage(rp, _cfg(rp, tmp_path, fake_repo, "--modality", "any"), "build").cmds[0]
    ct = _stage(rp, _cfg(rp, tmp_path, fake_repo, "--modality", "CT"), "build").cmds[0]
    assert ct[ct.index("--modality") + 1] == "CT"
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo), "--modality", "any"]) == 0
    sub = rp.config_from_args(rp.build_parser().parse_args(["build", *_base_args(tmp_path, fake_repo)]))
    assert sub.modality == "any"                                    # inherited by a stage subcommand
    plain = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo)]))
    assert plain.modality == "MR"                                   # a plain run does not inherit
    sp = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    st = json.loads(sp.read_text())
    st["options"].pop("modality")                                   # a state.json from before the option
    sp.write_text(json.dumps(st))
    assert rp.config_from_args(rp.build_parser().parse_args(["build", *_base_args(tmp_path, fake_repo)])).modality == "MR"


def test_changing_modality_reruns_build_and_what_follows_but_not_match(rp, fake_repo, tmp_path, monkeypatch,
                                                                     plenty_of_disk, capsys):
    class ModalityRunner(FakeRunner):
        """build writes a studies.csv that depends on the modality rule (as the real one does)."""
        def __call__(self, argv, log_path, cwd, dry_run=False):
            rc = super().__call__(argv, log_path, cwd, dry_run)
            if not dry_run and argv[1] == "scripts/build_training_set.py":
                rule = argv[argv.index("--modality") + 1] if "--modality" in argv else "any"
                (self.repo / self._arg(argv, "--out") / "studies.csv").write_text(f"patient_id,cls\n# {rule}\n")
            return rc

    runner = ModalityRunner(fake_repo, fail_on="combine_training_sets.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    capsys.readouterr()
    st = _state(fake_repo)
    assert st["stages"]["build"]["input_sha256s"]["opt:modality"] == "opt:MR"
    cfg = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo), "--resume",
                                                            "--modality", "any"]))
    assert rp.stage_is_done(cfg, _stage(rp, cfg, "match"), st) == (True, "")
    done, why = rp.stage_is_done(cfg, _stage(rp, cfg, "build"), st)
    assert not done and "inputs changed" in why

    n = len(runner.calls)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--modality", "any", "--skip-tests"]) == 0
    out = capsys.readouterr().out
    assert "stage match already done — skipped" in out
    assert "stage build re-run: inputs changed" in out and "stage prospective re-run: inputs changed" in out
    new = _scripts(runner.calls[n:])
    assert "scripts/match_images_to_reports.py" not in new
    for script in ("scripts/build_training_set.py", "scripts/eval_prospective.py",
                   "scripts/combine_training_sets.py", "training/finetune_brain_classifier.py"):
        assert script in new, script
    build = [c for c in runner.calls[n:] if c[1] == "scripts/build_training_set.py"][0]
    assert "--modality" not in build and _state(fake_repo)["options"]["modality"] == "any"


def test_prospective_command_line(rp, fake_repo, tmp_path):
    (fake_repo / "data" / "sets" / "older").mkdir(parents=True)
    (fake_repo / "data" / "sets" / "older" / "studies.csv").write_text("patient_id,cls\n")
    cfg = _cfg(rp, tmp_path, fake_repo, "--device", "cpu")
    s = _stage(rp, cfg, "prospective")
    assert len(s.cmds) == 1
    assert s.cmds[0] == [cfg.python, "scripts/eval_prospective.py", "--model", "models/brain_triage_finetuned",
                         "--set", "data/sets/fakebatch", "--seen", "data/local_finetune_bin",
                         "--seen", "data/sets/older", "--holdout", "data/holdout_patients_v1.txt",
                         "--matched", "data/fakebatch_matched.csv",
                         "--out", "data/release/brain-triage-vTEST/prospective_eval.json", "--device", "cpu"]
    assert s.inputs == ["data/sets/fakebatch/studies.csv", "data/fakebatch_matched.csv",
                        "data/holdout_patients_v1.txt", "ls:data/local_finetune_bin",
                        "data/sets/older/studies.csv", "models/brain_triage_finetuned/MANIFEST.json"]
    assert s.outputs == ["data/release/brain-triage-vTEST/prospective_eval.json"]
    # its only guard rail checks the trained-on release set (none for this legacy-trained model)
    assert rp.GUARDS["prospective"](cfg, {}) == ([], []) and "prospective" not in rp.NEEDS_ROOT


def test_prospective_scores_the_shipped_model_whatever_base_model_training_starts_from(rp, fake_repo, tmp_path):
    """--base-model is where TRAINING starts (the public ViT in the retrain-from-scratch fallback).
    The prospective stage must still score the model that is shipped today — the release dir the
    gate compares with and promote replaces."""
    cfg = _cfg(rp, tmp_path, fake_repo, "--base-model", "google/vit-base-patch16-224")
    assert cfg.base_model == "google/vit-base-patch16-224"
    s = _stage(rp, cfg, "prospective")
    cmd = s.cmds[0]
    assert cmd[cmd.index("--model") + 1] == rp.BASE_MODEL == "models/brain_triage_finetuned"
    assert "google/vit-base-patch16-224" not in cmd
    assert "opt:base_model" not in s.inputs and f"{rp.BASE_MODEL}/MANIFEST.json" in s.inputs
    gate = _stage(rp, cfg, "gate")
    assert all(c[c.index("--compare-with") + 1] == cmd[cmd.index("--model") + 1] for c in gate.cmds), \
        "prospective and the gate's --compare-with must look at the same (shipped) model"


def test_prospective_good_check(rp, tmp_path):
    p = tmp_path / "prospective_eval.json"
    assert "missing" in rp.prospective_problem(p)
    p.write_text("{not json")
    assert "not valid JSON" in rp.prospective_problem(p)
    p.write_text(json.dumps({"kind": "study_level_eval", "groups": {}}))
    assert "kind" in rp.prospective_problem(p)
    p.write_text(json.dumps({"kind": "prospective_eval", "groups": []}))
    assert "groups" in rp.prospective_problem(p)
    p.write_text(json.dumps({"kind": "prospective_eval", "groups": {"all": {"n_studies": 3}}}))
    assert rp.prospective_problem(p) is None
    assert rp.prospective_headline(p) is None                       # no 'new' group → no headline
    p.write_text(json.dumps({"kind": "prospective_eval",
                             "groups": {"new": {"n_studies": 7, "sensitivity": 0.5, "specificity": None}}}))
    assert rp.prospective_headline(p) == ("prospective, never-seen patients: n 7 studies · "
                                          "sensitivity 0.500 · specificity —")


def test_prospective_runs_prints_headline_and_a_bad_json_fails_the_stage(rp, fake_repo, tmp_path, monkeypatch,
                                                                         plenty_of_disk, capsys):
    class BadKind(FakeRunner):
        rc = 0

        def __call__(self, argv, log_path, cwd, dry_run=False):
            if argv[1] == "scripts/eval_prospective.py" and not dry_run and self.rc is not None:
                self.calls.append(list(argv))
                out = self.repo / self._arg(argv, "--out")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"kind": "something_else", "groups": {}}))
                return self.rc
            return super().__call__(argv, log_path, cwd, dry_run)

    runner = BadKind(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stage prospective finished but its output is not usable" in out and "something_else" not in out
    assert _state(fake_repo)["stages"]["prospective"]["status"] == "failed"
    assert "scripts/combine_training_sets.py" not in _scripts(runner.calls)
    # exit 2 (unusable inputs) is a failed stage, never a gate verdict
    runner.rc = 2
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"]) == rp.EXIT_STAGE_FAILED
    out = capsys.readouterr().out
    assert "stage prospective FAILED (exit 2)" in out and "resume with:" in out
    assert "stopped before promote" not in out and "verdict" not in _state(fake_repo)["stages"]["prospective"]
    # the real script: the resume runs the stage again and prints the headline from the 'new' group
    runner.rc = None
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "prospective, never-seen patients: n 412 studies · sensitivity 0.920 · specificity 0.270" in out
    st = _state(fake_repo)
    assert st["stages"]["prospective"]["status"] == "done"
    # it scored the model shipped BEFORE this release (promote ran after it)
    assert json.loads((fake_repo / "data" / "release" / "brain-triage-vTEST" / "prospective_eval.json")
                      .read_text())["model"] == "models/brain_triage_finetuned"
    assert _scripts(runner.calls).index("scripts/eval_prospective.py") < \
        _scripts(runner.calls).index("scripts/promote_model.py")


@pytest.mark.parametrize("stage", ["prospective", "combine"])
def test_prospective_never_reruns_after_promote_exactly_like_combine(rp, fake_repo, tmp_path, monkeypatch,
                                                                    plenty_of_disk, capsys, stage):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--skip-tests"]) == 0
    capsys.readouterr()
    assert stage in rp.NO_RERUN_AFTER_PROMOTE and stage in rp.FROZEN_AFTER_PROMOTE
    report = fake_repo / "data" / "release" / "brain-triage-vTEST" / "prospective_eval.json"
    before = report.read_text()
    # the stage subcommand (= --only) is refused under the promoted tag
    n = len(runner.calls)
    assert rp.main([stage, *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    out = capsys.readouterr().out
    assert "already promoted" in out and f"{stage} cannot run again" in out and len(runner.calls) == n
    # a rebuilt set after promote: --resume, --from <stage> --resume, --only <stage> --resume all skip it
    (fake_repo / "data" / "sets" / "fakebatch" / "studies.csv").write_text("patient_id,cls\n9000001,normal\n")
    for extra in (["--resume"], ["--from", stage, "--resume"], ["--only", stage, "--resume"]):
        rc = rp.main(["run", *_base_args(tmp_path, fake_repo), *extra, "--skip-tests"])
        out = capsys.readouterr().out
        assert rc == 0, (extra, out)
        assert f"stage {stage} already done — skipped" in out, extra
    assert "scripts/eval_prospective.py" not in _scripts(runner.calls[n:])
    assert "scripts/combine_training_sets.py" not in _scripts(runner.calls[n:])
    assert report.read_text() == before
    assert rp.main(["status", *_base_args(tmp_path, fake_repo)]) == 0
    assert re.search(rf"\b{stage}\s+\S*done", re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out))


def test_prospective_and_match_run_without_the_drive_build_stops(rp, fake_repo, tmp_path, monkeypatch,
                                                                 plenty_of_disk, capsys):
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["index", *_base_args(tmp_path, fake_repo)]) == 0
    (tmp_path / "drive").rename(tmp_path / "drive_unplugged")
    capsys.readouterr()
    n = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"])
    out = capsys.readouterr().out
    assert rc == rp.EXIT_USAGE
    assert f"drive not mounted: {tmp_path / 'drive'} (needed by stage build)" in out and "resume" in out
    assert _scripts(runner.calls[n:]) == ["scripts/parse_hospital_reports.py", "scripts/label_reports.py",
                                          "scripts/label_reports.py", "scripts/match_images_to_reports.py"]
    st = _state(fake_repo)
    assert st["stages"]["match"]["status"] == "done" and "build" not in st["stages"]
    # build alone without the drive is refused up front, nothing runs
    n = len(runner.calls)
    assert rp.main(["build", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    assert "drive not mounted" in capsys.readouterr().out and len(runner.calls) == n
    # a fresh run whose first stage is index is refused up front too (no state written for it)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)[:-1], "brain-triage-vNEW"]) == rp.EXIT_USAGE
    assert "needed by stage index" in capsys.readouterr().out
    assert not (fake_repo / "data" / "release" / "brain-triage-vNEW").exists()
    # match and prospective run with the drive gone (match gets --images only with a configured root)
    (tmp_path / "drive_unplugged").rename(tmp_path / "drive")
    assert rp.main(["build", *_base_args(tmp_path, fake_repo)]) == 0
    (tmp_path / "drive").rename(tmp_path / "drive_unplugged")
    assert rp.main(["prospective", *_base_args(tmp_path, fake_repo)]) == 0
    assert "scripts/eval_prospective.py" in _scripts(runner.calls)
    match = [c for c in runner.calls if c[1] == "scripts/match_images_to_reports.py"][-1]
    assert match[match.index("--images") + 1] == str(tmp_path / "drive")


def test_match_without_root_has_no_images_flag(rp, fake_repo, tmp_path, monkeypatch, capsys):
    args = ["--batch", "fakebatch", "--repo", str(fake_repo), "--tag", "brain-triage-vNOROOT"]
    cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *args]))
    match = _stage(rp, cfg, "match").cmds[0]
    assert "--images" not in match and match[2:4] == ["--index", "data/fakebatch_index.csv"]
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["match", *args]) == 0
    assert rp.main(["build", *args]) == rp.EXIT_USAGE
    assert "--root is required for stages ['build']" in capsys.readouterr().out
    assert rp.main(["run", *args]) == rp.EXIT_USAGE
    assert "--root is required for stages ['build', 'index']" in capsys.readouterr().out


# ------------------------------------------- review 2026-11-11 regressions ----

V3 = "brain-triage-v3"
V2_SET = "data/train_release_brain-triage-v2"


def _v3_args(tmp_path: Path, repo: Path, *extra: str) -> list[str]:
    return [a if a != "brain-triage-vTEST" else V3 for a in _base_args(tmp_path, repo)] + list(extra)


def test_rerelease_of_the_same_batch_passes_the_trained_on_release_set_as_seen(rp, fake_repo, tmp_path,
                                                                               monkeypatch, plenty_of_disk, capsys):
    """Review 4 (blocker): after a pipeline release the shipped model was trained on a combined release
    set. Re-evaluating a batch that set contains (new reports for the same batch, or a hand-given
    --extra-sets) must count those patients as seen, never as 'new'."""
    _trained_on_v2(fake_repo)
    for extra in ((), ("--extra-sets", "data/local_finetune_bin"),
                  ("--train-lock", "data/v2_trained_ids.txt")):
        (fake_repo / "data" / "v2_trained_ids.txt").write_text("10\n11\n")
        cfg = rp.config_from_args(rp.build_parser().parse_args(["plan", *_v3_args(tmp_path, fake_repo, *extra)]))
        s = _stage(rp, cfg, "prospective")
        cmd = s.cmds[0]
        seen = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--seen"]
        assert seen == ["data/local_finetune_bin", V2_SET], (extra, cmd)
        assert f"{V2_SET}/dataset_manifest.json" in s.inputs, extra
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_v3_args(tmp_path, fake_repo), "--skip-tests"]) == 0, capsys.readouterr().out
    pro = [c for c in runner.calls if c[1] == "scripts/eval_prospective.py"][0]
    assert pro[pro.index("--seen", pro.index("--seen") + 1) + 1] == V2_SET
    st = json.loads((fake_repo / "data" / "release" / V3 / "state.json").read_text())
    assert f"{V2_SET}/dataset_manifest.json" in st["stages"]["prospective"]["input_sha256s"]


def test_missing_trained_on_set_fails_the_prospective_stage_before_it_runs(rp, fake_repo, tmp_path, monkeypatch,
                                                                         plenty_of_disk, capsys):
    prev = _trained_on_v2(fake_repo)
    (prev / "dataset_manifest.json").unlink()
    runner = FakeRunner(fake_repo)
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_v3_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    out = capsys.readouterr().out
    assert "guard rails before prospective failed" in out and "nothing was run" in out
    assert V2_SET in out and "--train-lock" in out and "Restore" in out
    assert "scripts/eval_prospective.py" not in _scripts(runner.calls)
    st = json.loads((fake_repo / "data" / "release" / V3 / "state.json").read_text())
    assert st["guards"]["pre_prospective"]["passed"] is False
    # an ids file does not name the patients' set for --seen: still refused
    (fake_repo / "data" / "v2_trained_ids.txt").write_text("10\n11\n")
    assert rp.main(["run", *_v3_args(tmp_path, fake_repo, "--train-lock", "data/v2_trained_ids.txt"),
                    "--resume"]) == rp.EXIT_USAGE
    assert "guard rails before prospective failed" in capsys.readouterr().out
    # the restored set given by hand as --train-lock <dir> lets the release go on and is the --seen set
    restored = fake_repo / "data" / "v2_restored"
    (restored / "train" / "normal").mkdir(parents=True)
    assert rp.main(["run", *_v3_args(tmp_path, fake_repo, "--train-lock", "data/v2_restored"),
                    "--resume", "--skip-tests"]) == 0, capsys.readouterr().out
    pro = [c for c in runner.calls if c[1] == "scripts/eval_prospective.py"][-1]
    assert [pro[i + 1] for i, x in enumerate(pro) if x == "--seen"] == ["data/local_finetune_bin", "data/v2_restored"]


def test_legacy_trained_model_keeps_the_exact_prospective_command(rp, fake_repo, tmp_path):
    """The shipped model trained on data/local_finetune_bin only (no release set): the command and the
    inputs stay byte for byte what the release in progress recorded — even with a --train-lock dir."""
    lock = fake_repo / "data" / "some_set"
    (lock / "train" / "normal").mkdir(parents=True)
    for extra in ((), ("--train-lock", "data/some_set")):
        cfg = _cfg(rp, tmp_path, fake_repo, "--device", "cpu", *extra)
        s = _stage(rp, cfg, "prospective")
        assert s.cmds[0] == [cfg.python, "scripts/eval_prospective.py", "--model", "models/brain_triage_finetuned",
                             "--set", "data/sets/fakebatch", "--seen", "data/local_finetune_bin",
                             "--holdout", "data/holdout_patients_v1.txt", "--matched", "data/fakebatch_matched.csv",
                             "--out", "data/release/brain-triage-vTEST/prospective_eval.json", "--device", "cpu"]
        assert s.inputs == ["data/sets/fakebatch/studies.csv", "data/fakebatch_matched.csv",
                            "data/holdout_patients_v1.txt", "ls:data/local_finetune_bin",
                            "models/brain_triage_finetuned/MANIFEST.json"]
        assert rp.guard_pre_prospective(cfg, {}) == ([], [])


def test_legacy_promoted_state_without_prospective_finishes_bump_and_bundle(rp, fake_repo, tmp_path, monkeypatch,
                                                                          plenty_of_disk, capsys):
    """Review 2: a release promoted by the previous version has no 'prospective' record. --resume must
    finish bump/bundle; status shows the stage as skipped; an explicit prospective is still refused."""
    runner = FakeRunner(fake_repo, fail_on="publish_model_bundle.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    sp = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    st = json.loads(sp.read_text())
    st["stages"].pop("prospective")
    sp.write_text(json.dumps(st))
    capsys.readouterr()
    n = len(runner.calls)
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "cannot run again" not in out
    assert "stage prospective skipped (promoted before this stage existed)" in out
    assert "packaging/publish_model_bundle.py" in _scripts(runner.calls[n:])
    assert "scripts/eval_prospective.py" not in _scripts(runner.calls[n:])
    assert _state(fake_repo)["stages"]["bundle"]["status"] == "done"
    assert "prospective" not in _state(fake_repo)["stages"]
    assert rp.main(["status", *_base_args(tmp_path, fake_repo)]) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert re.search(r"prospective\s+skipped \(promoted before this stage existed\)", out), out
    # an explicit prospective run after promote stays refused
    n = len(runner.calls)
    assert rp.main(["prospective", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_USAGE
    out = capsys.readouterr().out
    assert "already promoted" in out and "prospective cannot run again" in out and len(runner.calls) == n


def test_legacy_build_record_without_modality_reruns_build_under_mr(rp, fake_repo, tmp_path, monkeypatch,
                                                                    plenty_of_disk, capsys):
    """Review 2: a build recorded by the previous version (no MR-only rule = 'any') is re-run under the
    MR rule; a build recorded by the current version (opt:modality stored) stays done."""
    runner = FakeRunner(fake_repo, fail_on="combine_training_sets.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    capsys.readouterr()
    cfg = rp.config_from_args(rp.build_parser().parse_args(["run", *_base_args(tmp_path, fake_repo), "--resume"]))
    st = _state(fake_repo)
    assert rp.stage_is_done(cfg, _stage(rp, cfg, "build"), st) == (True, "")        # current record
    sp = fake_repo / "data" / "release" / "brain-triage-vTEST" / "state.json"
    st["stages"]["build"]["input_sha256s"].pop("opt:modality")                      # the previous version's
    st["options"].pop("modality")
    sp.write_text(json.dumps(st))
    done, why = rp.stage_is_done(cfg, _stage(rp, cfg, "build"), st)
    assert not done and "inputs changed" in why
    assert not rp.inputs_changed({"opt:epochs": None}, {"opt:epochs": "opt:12"})    # other options: tolerant
    n = len(runner.calls)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--skip-tests"]) == 0
    out = capsys.readouterr().out
    assert "stage match already done — skipped" in out and "stage build re-run: inputs changed" in out
    build = [c for c in runner.calls[n:] if c[1] == "scripts/build_training_set.py"]
    assert len(build) == 1 and build[0][build[0].index("--modality") + 1] == "MR"
    assert _state(fake_repo)["stages"]["build"]["input_sha256s"]["opt:modality"] == "opt:MR"


def test_resume_without_the_drive_passes_the_train_guard(rp, fake_repo, tmp_path, monkeypatch, plenty_of_disk,
                                                         capsys):
    """Review 2: train reads only local slices — an unplugged drive after build must not stop it."""
    runner = FakeRunner(fake_repo, fail_on="eval_prospective.py")
    monkeypatch.setattr(rp, "run_command", runner)
    assert rp.main(["run", *_base_args(tmp_path, fake_repo)]) == rp.EXIT_STAGE_FAILED
    (tmp_path / "drive").rename(tmp_path / "drive_unplugged")
    capsys.readouterr()
    rc = rp.main(["run", *_base_args(tmp_path, fake_repo), "--resume", "--skip-tests"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "drive not mounted" not in out and "guard rails before train passed" in out
    assert "training/finetune_brain_classifier.py" in _scripts(runner.calls)
    assert "drive mounted" not in " ".join(rp.GUARD_TEXT["train"])
