"""Stage 9 "bump-docs": scripts/bump_model_release.py + src/inference/validated_metrics.py.

Everything runs on a COPY of the docs/READMEs/laptop checklists/validated_metrics.py in tmp_path
against a synthetic model dir (fake study_level_eval.json + MANIFEST.json with different numbers).
No real patient data, no torch, no network. The real checkout is only read, never written:
its own sync is asserted by docs/test_docs_consistency.py (bump --check).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bump_model_release.py"
VM_PATH = REPO / "src" / "inference" / "validated_metrics.py"

# ---- the synthetic "new release" -------------------------------------------------------
NEW_TP, NEW_FN, NEW_TN, NEW_FP = 892, 74, 137, 131          # sens 0.923, spec 0.511, n 1234
NEW_N = NEW_TP + NEW_FN + NEW_TN + NEW_FP
NEW_SLICES = NEW_N * 5
NEW_DATE = "2027-01-15"
NEW_IDENTITY = "abcdef012345"
NEW_SENS = NEW_TP / (NEW_TP + NEW_FN)
NEW_SPEC = NEW_TN / (NEW_TN + NEW_FP)

# ---- tokens of the CURRENT release that must disappear from every doc after the bump ----
OLD_TOKENS_EVERYWHERE = ["0d559766ce58", "2026-09-07", "0.33–0.61", "0,33–0,61"]
OLD_NUMBER_RE = re.compile(r"(?<![\d.,])(?:0[.,]90|0[.,]47|178|118|131|890)(?![\d])")


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve `from __future__` annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    """The script as a module (for its file table); it imports validated_metrics torch-free."""
    return _load_by_path("bump_model_release_under_test", SCRIPT)


def _write_fake_model(d: Path, *, tp=NEW_TP, fn=NEW_FN, tn=NEW_TN, fp=NEW_FP, identity=NEW_IDENTITY,
                      date=NEW_DATE, extra_gate: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    n = tp + fn + tn + fp
    gate = {"name": "gate mean @0.50", "acc": (tp + tn) / n, "sens": tp / (tp + fn), "spec": tn / (tn + fp),
            "balanced": 0.0, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "min_sens": 0.85, "min_spec": 0.4,
            "passed": True}
    gate.update(extra_gate or {})
    (d / "study_level_eval.json").write_text(json.dumps({
        "results": [], "gate": gate, "n_studies": n, "n_slices": n * 5,
        "model": str(d), "evaluated_at": f"{date}T10:00:00"}), encoding="utf-8")
    (d / "MANIFEST.json").write_text(json.dumps({
        "model_dir": str(d), "algorithm": "sha256",
        "files": {"config.json": "11" * 32, "model.safetensors": identity + "0" * 52}}), encoding="utf-8")
    return d


@pytest.fixture
def tree(tmp_path, bump):
    """Copy of every file the bump table touches, at the same relative paths."""
    root = tmp_path / "tree"
    for rel in bump.FILES:
        src = REPO / rel
        assert src.is_file(), rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return root


@pytest.fixture
def fake_model(tmp_path):
    return _write_fake_model(tmp_path / "fake_model")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI; --model is always explicit, so the env override must not leak in."""
    env = {k: v for k, v in os.environ.items() if k != "SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, capture_output=True, text=True, env=env)


def _snapshot(root: Path, files) -> dict[str, str]:
    return {rel: (root / rel).read_text(encoding="utf-8") for rel in files}


def _md_files(bump) -> list[str]:
    return [f for f in bump.FILES if f.endswith(".md")]


# ============================================================================ validated_metrics

def test_validated_metrics_reads_json_and_manifest(fake_model):
    vm = _load_by_path("vm_json", VM_PATH)
    m = vm.load(fake_model, strict=True)
    assert m.source == "json"
    assert (m.tp, m.fn, m.tn, m.fp) == (NEW_TP, NEW_FN, NEW_TN, NEW_FP)
    assert (m.n_studies, m.n_abnormal, m.n_normal, m.n_slices) == (NEW_N, NEW_TP + NEW_FN, NEW_TN + NEW_FP, NEW_SLICES)
    assert m.sens == pytest.approx(NEW_SENS) and m.spec == pytest.approx(NEW_SPEC)
    assert (m.identity, m.evaluated_on, m.threshold, m.min_sens, m.min_spec) == (NEW_IDENTITY, NEW_DATE, 0.5, 0.85, 0.4)
    # Wilson CI computed here when the JSON has none
    lo, hi = m.spec_ci
    assert lo < NEW_SPEC < hi and hi - lo < 0.15
    assert m.spec_ci == vm.wilson_ci(NEW_TN, NEW_TN + NEW_FP)
    assert vm.dec(m.sens) == "0.92" and vm.dec(m.sens, "ru") == "0,92" and vm.dec(m.spec, "uz") == "0,51"
    assert vm.pct(m.sens) == "92%" and vm.pct(m.spec) == "51%"
    assert vm.miss_one_in(m) == 13
    assert vm.ci(m.spec_ci, "ru") == vm.ci(m.spec_ci).replace(".", ",")


def test_validated_metrics_prefers_ci_from_json(tmp_path):
    d = _write_fake_model(tmp_path / "m", extra_gate={"spec_ci": [0.41, 0.61], "sens_ci": [0.90, 0.94]})
    vm = _load_by_path("vm_ci", VM_PATH)
    m = vm.load(d, strict=True)
    assert m.spec_ci == (0.41, 0.61) and m.sens_ci == (0.90, 0.94)


def test_wilson_ci_matches_the_documented_interval():
    vm = _load_by_path("vm_wilson", VM_PATH)
    lo, hi = vm.wilson_ci(22, 47)          # the current release's specificity, docs say 0.33–0.61
    assert (round(lo, 2), round(hi, 2)) == (0.33, 0.61)
    assert vm.wilson_ci(0, 0) == (0.0, 0.0)


def test_validated_metrics_fallback_when_files_missing(tmp_path):
    vm = _load_by_path("vm_fb", VM_PATH)
    empty = tmp_path / "no_model_here"
    m = vm.load(empty)
    assert m.source == "fallback" and m.model_dir == str(empty)
    assert (m.tp, m.fn, m.tn, m.fp) == (vm.FALLBACK_TP, vm.FALLBACK_FN, vm.FALLBACK_TN, vm.FALLBACK_FP)
    assert (m.identity, m.evaluated_on, m.n_studies) == (vm.FALLBACK_IDENTITY, vm.FALLBACK_EVALUATED_ON, vm.FALLBACK_N_STUDIES)
    assert m.n_studies == m.tp + m.fn + m.tn + m.fp
    with pytest.raises(FileNotFoundError):
        vm.load(empty, strict=True)
    # malformed JSON → fallback too (never raises at import)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "study_level_eval.json").write_text("{not json", encoding="utf-8")
    assert vm.load(bad).source == "fallback"
    with pytest.raises(ValueError):
        vm.load(bad, strict=True)


def test_validated_metrics_import_honours_env_override_and_never_raises(tmp_path, fake_model):
    code = ("from src.inference import validated_metrics as vm; "
            "print(vm.SOURCE, vm.IDENTITY, vm.N_STUDIES, vm.EVALUATED_ON, vm.pct(vm.SENS), vm.pct(vm.SPEC))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE": str(fake_model)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["json", NEW_IDENTITY, str(NEW_N), NEW_DATE, "92%", "51%"]
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE": str(tmp_path / "missing")})
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("fallback ")


def test_runtime_strings_follow_the_model_dir(fake_model):
    """Report templates (ru/uz/en) and the registry card notes are generated from the JSON."""
    code = ("from src.inference.brain_report_templates import TRIAGE_DESCRIPTIONS as T; "
            "from src.inference.model_registry import BRAIN_TRIAGE as C; "
            "import json; print(json.dumps({'T': T, 'notes': C.notes}, ensure_ascii=False))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE": str(fake_model)})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for lang in ("ru", "uz", "en"):
        abn, nrm = out["T"]["abnormal"][lang], out["T"]["normal"][lang]
        assert "92%" in abn and "51%" in abn and "90%" not in abn and "47%" not in abn, (lang, abn)
        assert "92%" in nrm and "90%" not in nrm, (lang, nrm)
    # wording around the numbers is unchanged (the manuals quote it)
    assert out["T"]["normal"]["en"] == "The local triage model did not flag this study (sensitivity 92%) — this is NOT a normal read."
    assert out["T"]["normal"]["uz"].endswith("bu norma xulosasi EMAS.")
    assert out["T"]["abnormal"]["ru"].startswith("Локальная модель триажа (валидирована на данных клиники: чувствительность 92%, специфичность 51%)")
    assert "sensitivity 0.92 / specificity 0.51" in out["notes"] and f"over {NEW_N} held-out" in out["notes"]


# ============================================================================ bump script

def test_check_reports_drift_and_writes_nothing(tree, fake_model, bump):
    before = _snapshot(tree, bump.FILES)
    r = _run("--model", str(fake_model), "--root", str(tree), "--check")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "out of sync" in r.stdout
    for rel in _md_files(bump):
        assert rel in r.stdout, rel            # every stale file is named
    assert _snapshot(tree, bump.FILES) == before


def test_dry_run_prints_unified_diff_and_writes_nothing(tree, fake_model, bump):
    before = _snapshot(tree, bump.FILES)
    r = _run("--model", str(fake_model), "--root", str(tree), "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "+++ b/docs/intended_use_ru.md" in r.stdout and "--- a/docs/intended_use_ru.md" in r.stdout
    assert f"+FALLBACK_IDENTITY = '{NEW_IDENTITY}'" in r.stdout
    assert "dry run" in r.stdout and "nothing written" in r.stdout
    assert _snapshot(tree, bump.FILES) == before


def test_bump_rewrites_every_language_and_is_idempotent(tree, fake_model, bump):
    r = _run("--model", str(fake_model), "--root", str(tree))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rewrote" in r.stdout

    en_new = ["0.92", "0.51", str(NEW_N), str(NEW_TP), str(NEW_TP + NEW_FN), str(NEW_TN), str(NEW_TN + NEW_FP),
              str(NEW_SLICES), NEW_DATE, NEW_IDENTITY, "1 in 13"]
    ru_new = ["0,92", "0,51", str(NEW_N), str(NEW_TP), str(NEW_TP + NEW_FN), str(NEW_TN), str(NEW_TN + NEW_FP),
              str(NEW_SLICES), NEW_DATE, NEW_IDENTITY, "1 из 13", "патологических исследований"]
    uz_new = ["0,92", "0,51", str(NEW_N), str(NEW_TP), str(NEW_TP + NEW_FN), str(NEW_TN), str(NEW_TN + NEW_FP),
              str(NEW_SLICES), NEW_DATE, NEW_IDENTITY, "har 13 ta"]
    for lang, toks in (("en", en_new), ("ru", ru_new), ("uz", uz_new)):
        text = (tree / f"docs/intended_use_{lang}.md").read_text(encoding="utf-8")
        for tok in toks:
            assert tok in text, (lang, tok)
        for stem in ("user_manual", "pilot_evaluation_protocol"):
            text = (tree / f"docs/{stem}_{lang}.md").read_text(encoding="utf-8")
            assert toks[0] in text and toks[1] in text and str(NEW_N) in text, (stem, lang)
        assert f"`{NEW_IDENTITY}…`" in (tree / f"docs/admin_install_guide_{lang}.md").read_text(encoding="utf-8"), lang
    for rel in ("README.md", "docs/README.md", "docs/attributions.md"):
        text = (tree / rel).read_text(encoding="utf-8")
        assert "0.92" in text and "0.51" in text and str(NEW_N) in text, rel
    assert NEW_IDENTITY in (tree / "docs/attributions.md").read_text(encoding="utf-8")
    assert "0,92" in (tree / "docs/attributions.md").read_text(encoding="utf-8")   # RU summary paragraph
    for rel in ("docs/windows_laptop_test_en.md", "docs/windows_laptop_test_ru.md"):
        assert f"`{NEW_IDENTITY}`" in (tree / rel).read_text(encoding="utf-8"), rel

    # not a single old token survives: identity / date / CI nowhere in any document, and no old
    # count on any line the bump rewrote (a second "178" in the same sentence would survive a
    # too-narrow anchor). Other lines may legitimately carry e.g. "178 locked holdout ids".
    for rel in _md_files(bump):
        text = (tree / rel).read_text(encoding="utf-8")
        for tok in OLD_TOKENS_EVERYWHERE:
            assert tok not in text, (rel, tok)
        old_lines = set((REPO / rel).read_text(encoding="utf-8").splitlines())
        rewritten = [l for l in text.splitlines() if l not in old_lines]
        assert rewritten, rel
        hits = [(l[:60], OLD_NUMBER_RE.findall(l)) for l in rewritten if OLD_NUMBER_RE.search(l)]
        assert not hits, (rel, hits)

    # the FALLBACK block of validated_metrics.py was bumped too → a frozen installer
    # without the model dir shows the NEW numbers
    vm_copy = _load_by_path("vm_bumped", tree / "src/inference/validated_metrics.py")
    m = vm_copy.load(tree / "definitely_missing")
    assert m.source == "fallback"
    assert (m.tp, m.fn, m.tn, m.fp, m.n_studies, m.n_slices) == (NEW_TP, NEW_FN, NEW_TN, NEW_FP, NEW_N, NEW_SLICES)
    assert (m.identity, m.evaluated_on) == (NEW_IDENTITY, NEW_DATE)
    assert m.sens == pytest.approx(NEW_SENS) and m.spec == pytest.approx(NEW_SPEC)

    # idempotent: a second run changes nothing, and --check is now green
    after_first = _snapshot(tree, bump.FILES)
    r2 = _run("--model", str(fake_model), "--root", str(tree))
    assert r2.returncode == 0 and "already in sync" in r2.stdout, r2.stdout + r2.stderr
    assert _snapshot(tree, bump.FILES) == after_first
    r3 = _run("--model", str(fake_model), "--root", str(tree), "--check")
    assert r3.returncode == 0, r3.stdout + r3.stderr


def test_bump_back_to_the_current_release_restores_the_originals(tree, bump, tmp_path):
    """Round trip: new numbers → current numbers reproduces today's docs byte for byte, which
    proves no anchor swallowed neighbouring text."""
    original = _snapshot(tree, bump.FILES)
    fake_new = _write_fake_model(tmp_path / "new")
    assert _run("--model", str(fake_new), "--root", str(tree)).returncode == 0
    assert _snapshot(tree, bump.FILES) != original
    real = REPO / "models" / "brain_triage_finetuned"
    if not (real / "study_level_eval.json").exists():
        pytest.skip("validated model not present in this checkout")
    assert _run("--model", str(real), "--root", str(tree)).returncode == 0
    assert _snapshot(tree, bump.FILES) == original


def test_reworded_document_fails_loudly(tree, fake_model, bump):
    p = tree / "docs/README.md"
    p.write_text(p.read_text(encoding="utf-8").replace("held-out local patients", "held out patients"), encoding="utf-8")
    before = _snapshot(tree, bump.FILES)
    r = _run("--model", str(fake_model), "--root", str(tree))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "docs/README.md" in r.stdout and "matched 0×" in r.stdout and "Anchor errors" in r.stdout
    assert _snapshot(tree, bump.FILES) == before      # nothing partially written


def test_duplicated_anchor_is_ambiguous_and_refused(tree, fake_model, bump):
    p = tree / "docs/windows_laptop_test_en.md"
    text = p.read_text(encoding="utf-8")
    p.write_text(text + "\nextra line with the model identity `0d559766ce58`.\n", encoding="utf-8")
    r = _run("--model", str(fake_model), "--root", str(tree))
    assert r.returncode == 1 and "windows_laptop_test_en.md" in r.stdout and "matched 2×" in r.stdout


def test_missing_model_dir_is_exit_2(tree, tmp_path):
    r = _run("--model", str(tmp_path / "nowhere"), "--root", str(tree), "--check")
    assert r.returncode == 2 and "cannot read validated metrics" in r.stdout


def test_every_table_entry_matches_exactly_once_in_the_real_checkout(bump):
    """The anchors are phrases that exist in today's documents; if someone rewords a doc
    this points at the file + pattern before the docs test's --check does."""
    vm = _load_by_path("vm_real", VM_PATH)
    results, errors = bump.compute(REPO, vm.METRICS)
    assert not errors, "\n".join(errors)
    assert set(results) == set(bump.FILES)


# ============================================================================ derived prose
# Words that follow from the numbers ("about half of normal studies are flagged", "the small
# number of normal studies … imprecise") must change with them, in all three languages.

def test_normals_flagged_in_10_rounding(bump):
    assert bump.normals_flagged_in_10(0.468) == 5          # today's release → "about half"
    assert bump.normals_flagged_in_10(0.7208) == 3
    assert bump.normals_flagged_in_10(0.40) == 6           # the specificity floor
    assert bump.normals_flagged_in_10(0.55) == 5           # 4.5 rounds half-up
    assert bump.normals_flagged_in_10(0.99) == 1           # clamped, never "0 in 10"
    assert bump.normals_flagged_in_10(0.0) == 10


def test_derived_prose_rewritten_for_a_high_specificity_large_normal_release(tree, bump, tmp_path):
    """spec 0.72 on 480 normals: no document may still say 'about half' or 'small number'."""
    original = _snapshot(tree, bump.FILES)
    m = _write_fake_model(tmp_path / "spec72", tp=900, fn=60, tn=346, fp=134, identity="72" * 6)
    r = _run("--model", str(m), "--root", str(tree))
    assert r.returncode == 0, r.stdout + r.stderr

    um = {lang: (tree / f"docs/user_manual_{lang}.md").read_text(encoding="utf-8") for lang in ("en", "ru", "uz")}
    assert "about 3 in 10 normal studies **are** flagged" in um["en"] and "about half" not in um["en"]
    assert "около 3 из 10 нормальных **отмечаются**" in um["ru"] and "половины" not in um["ru"]
    assert "har 10 ta normal tekshiruvdan esa taxminan 3 tasi **belgilanadi**" in um["uz"]
    assert "yarmi" not in um["uz"]

    iu = {lang: (tree / f"docs/intended_use_{lang}.md").read_text(encoding="utf-8") for lang in ("en", "ru", "uz")}
    assert "With 480 normal studies the specificity estimate has a 95% CI of 0.68–0.76." in iu["en"]
    assert "small number" not in iu["en"] and "imprecise" not in iu["en"]
    assert "При 480 нормальных исследованиях 95 % ДИ специфичности составляет 0,68–0,76." in iu["ru"]
    assert "малое число" not in iu["ru"] and "неточна" not in iu["ru"]
    assert "Normal tekshiruvlar soni 480 ta: spetsifiklikning 95 % ishonch oralig‘i 0,68–0,76." in iu["uz"]
    assert "kamligiga" not in iu["uz"] and "noaniq" not in iu["uz"]

    # idempotent + --check green on the new wording
    after = _snapshot(tree, bump.FILES)
    r2 = _run("--model", str(m), "--root", str(tree))
    assert r2.returncode == 0 and "already in sync" in r2.stdout, r2.stdout
    assert _snapshot(tree, bump.FILES) == after
    assert _run("--model", str(m), "--root", str(tree), "--check").returncode == 0

    # the anchors accept either variant: small-N release next ("small number" comes back) …
    small = _write_fake_model(tmp_path / "small", tp=90, fn=10, tn=40, fp=20, identity="60" * 6)
    assert _run("--model", str(small), "--root", str(tree)).returncode == 0
    iu_en = (tree / "docs/intended_use_en.md").read_text(encoding="utf-8")
    assert "Note the small number of normal studies (60): the specificity estimate is imprecise" in iu_en
    assert "about 3 in 10 normal studies" in (tree / "docs/user_manual_en.md").read_text(encoding="utf-8")
    # … and back to spec 0.72 reproduces the first rewrite byte for byte
    assert _run("--model", str(m), "--root", str(tree)).returncode == 0
    assert _snapshot(tree, bump.FILES) == after

    # round trip to today's release restores the originals ("about half", "small number (47)")
    real = REPO / "models" / "brain_triage_finetuned"
    if not (real / "study_level_eval.json").exists():
        pytest.skip("validated model not present in this checkout")
    assert _run("--model", str(real), "--root", str(tree)).returncode == 0
    assert _snapshot(tree, bump.FILES) == original


def test_derived_prose_russian_singular_forms(bump, tmp_path):
    vm = _load_by_path("vm_prose", VM_PATH)
    m = vm.load(_write_fake_model(tmp_path / "ru1", tp=900, fn=60, tn=95, fp=6), strict=True)   # spec 0.94, 101 normals
    prose = bump.derived_prose(m)
    assert prose["normals_flagged_ru"] == "около 1 из 10 нормальных **отмечается**"
    assert prose["spec_note_ru"].startswith("При 101 нормальном исследовании 95 % ДИ")
    assert prose["normals_flagged_en"] == "about 1 in 10"


# ------------------------------------------------------------ review round 2

def test_bump_refuses_a_model_that_did_not_earn_release(tmp_path, bump, capsys):
    """An --allow-unvalidated / gate-failed / --force-promoted model must never get 'validated'
    docs and a pinned identity (review round 2, build-windows.yml:195)."""
    vm = _load_by_path("vm_round2", VM_PATH)
    failed = _write_fake_model(tmp_path / "failed", extra_gate={"passed": False})
    with pytest.raises(ValueError, match="not a validated release"):
        vm.load(failed, strict=True)
    m = vm.load(failed)                       # the app still loads it — but flagged
    assert m.source == "json" and m.validated is False and "gate.passed" in m.not_validated_reason
    assert bump.main(["--model", str(failed), "--dry-run"]) == bump.EXIT_NO_METRICS
    assert "cannot read validated metrics" in capsys.readouterr().out

    ok = _write_fake_model(tmp_path / "ok")
    assert vm.load(ok, strict=True).validated is True
    (ok / "promotion_log.json").write_text(json.dumps({"forced": True}))
    assert bump.main(["--model", str(ok), "--dry-run", "--root", str(tmp_path / "empty_root")]) == bump.EXIT_NO_METRICS
    assert "--force promotion" in capsys.readouterr().out

    refused = _write_fake_model(tmp_path / "refused")
    ev = json.loads((refused / "study_level_eval.json").read_text())
    ev["promotion"] = {"promote": False, "reason": "sens dropped"}
    (refused / "study_level_eval.json").write_text(json.dumps(ev))
    assert vm.load(refused).validated is False
    assert bump.main(["--model", str(refused), "--dry-run"]) == bump.EXIT_NO_METRICS
