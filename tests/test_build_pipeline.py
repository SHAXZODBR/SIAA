"""Installer build pipeline — structural checks of .github/workflows/build-windows.yml and the
packaging/ scripts it runs, plus unit tests of packaging/fetch_model_bundle.py.

Nothing here needs Windows, Node, PyInstaller or the network: the workflow is validated as
data (it parses, every repo file it references exists, every `npm run` script exists, model
keys are registry keys, artifact paths match electron-builder's output dir, the versions it
pins are the ones the admin guides quote, the `tests` job gates both installer jobs, artifacts
expire after one day, the clinic-flavour dispatch input skips only the model restore) and the
bundle restorer is exercised against
synthetic zips through file:// URLs. The PowerShell scripts can only be checked by
inspection on this machine — they are covered for the invariants that must stay in sync
with their bash twins.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "build-windows.yml"
PACKAGING = REPO / "packaging"
FETCH = PACKAGING / "fetch_model_bundle.py"
DOCS = REPO / "docs"
LANGS = ("ru", "uz", "en")
SECRETS = ("MODEL_BUNDLE_URL", "MODEL_BUNDLE_TOKEN", "WIN_CSC_LINK", "WIN_CSC_KEY_PASSWORD",
           "CSC_LINK", "CSC_KEY_PASSWORD")


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(job: str) -> list[dict]:
    return _workflow()["jobs"][job]["steps"]


def _package_json() -> dict:
    return json.loads(_text("desktop-app/package.json"))


def _guide(lang: str) -> str:
    return (DOCS / f"admin_install_guide_{lang}.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- workflow

def test_workflow_parses_with_expected_triggers_and_runners():
    wf = _workflow()
    on = wf.get("on") or wf.get(True)          # PyYAML reads a bare `on:` key as boolean True
    assert "workflow_dispatch" in on
    assert on["push"]["tags"] == ["v*"]
    assert wf["jobs"]["windows"]["runs-on"] == "windows-latest"
    assert wf["jobs"]["macos"]["runs-on"] == "macos-latest"
    assert wf["jobs"]["tests"]["runs-on"] == "ubuntu-latest"
    assert wf["jobs"]["windows"]["defaults"]["run"]["shell"] == "pwsh"
    assert wf["jobs"]["macos"]["defaults"]["run"]["shell"] == "bash"
    assert wf["permissions"] == {"contents": "write"}


# Build outputs the workflow names (PyInstaller dist/work dirs): git-ignored, absent on a fresh checkout.
BUILD_OUTPUT_DIRS = {"packaging/dist", "packaging/build"}


def test_workflow_references_only_existing_repo_files():
    # steps only — the comment header abbreviates twins as build_backend.{ps1,sh}
    text = "\n".join(l for l in WORKFLOW.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
    refs = set(re.findall(r"(?:packaging|scripts)[\\/][A-Za-z0-9_.]+", text))
    assert refs, "workflow references no packaging/ or scripts/ files"
    # PyInstaller's output dirs are produced BY the build (git-ignored), so a fresh CI
    # checkout never has them — they must not count as missing repo files.
    norm = {r.replace("\\", "/") for r in refs}
    outputs = norm & BUILD_OUTPUT_DIRS
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    for out in outputs:
        assert f"{out}/" in gitignore, f"{out} is referenced as build output but not git-ignored"
    missing = sorted(r for r in norm - BUILD_OUTPUT_DIRS if not (REPO / r).exists())
    assert not missing, missing
    # both freeze twins and both smoke twins are wired in
    for name in ("build_backend.ps1", "build_backend.sh", "smoke_backend.ps1", "smoke_backend.sh",
                 "fetch_model_bundle.py"):
        assert any(r.endswith(name) for r in refs), name


def test_workflow_refs_exist_on_a_fresh_checkout():
    """The CI `tests` job runs on a FRESH checkout (no git-ignored files): every packaging/ or
    scripts/ path the workflow names, other than the build's own output dirs, must be tracked
    (or at least not ignored) — a path that only exists on the developer Mac breaks CI."""
    try:
        r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"git unavailable: {e}")
    if r.returncode != 0:
        pytest.skip(f"not a git checkout: {r.stderr.strip()[:200]}")
    files = set(r.stdout.splitlines())
    dirs = {str(Path(f).parent.as_posix()) for f in files}
    dirs |= {p for d in list(dirs) for p in (str(x.as_posix()) for x in Path(d).parents)}
    text = "\n".join(l for l in WORKFLOW.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
    refs = {x.replace("\\", "/") for x in re.findall(r"(?:packaging|scripts)[\\/][A-Za-z0-9_.]+", text)}
    absent = sorted(x for x in refs - BUILD_OUTPUT_DIRS if x not in files and x not in dirs)
    assert not absent, f"not in a fresh checkout: {absent}"


def test_workflow_npm_scripts_exist_in_package_json():
    scripts = _package_json()["scripts"]
    names = set(re.findall(r"npm run ([A-Za-z0-9:_-]+)", WORKFLOW.read_text(encoding="utf-8")))
    assert {"build:win", "build:mac:arm64", "type-check"} <= names
    assert not (names - set(scripts)), names - set(scripts)
    assert scripts["build:mac:arm64"] == "vite build && electron-builder --mac --arm64"
    assert scripts["build:mac:x64"] == "vite build && electron-builder --mac --x64"


def test_artifact_paths_match_electron_builder_output_dir():
    out = _package_json()["build"]["directories"]["output"]
    uploads = [s for job in ("windows", "macos") for s in _steps(job)
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert len(uploads) == 2
    for s in uploads:
        assert s["with"]["path"].startswith(f"desktop-app/{out}/"), s["with"]["path"]
        assert s["with"]["if-no-files-found"] == "error"
    assert uploads[0]["with"]["path"].endswith("*.exe")
    assert uploads[1]["with"]["path"].endswith("*.dmg")
    # the artifact name says loudly when the validated model is absent
    for s in uploads:
        assert "NO-VALIDATED-MODEL" in s["with"]["name"] and "steps.bundle.outputs.have_bundle" in s["with"]["name"]


def test_public_model_keys_are_registry_or_extra_keys():
    keys = _workflow()["env"]["PUBLIC_MODEL_KEYS"].split(",")
    spec = importlib.util.spec_from_file_location("download_all_models", REPO / "scripts" / "download_all_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from src.inference.model_registry import REGISTRY
    unknown = [k for k in keys if k not in REGISTRY and k not in mod.EXTRA_KEYS]
    assert not unknown, unknown
    assert "brain_tumor_class" in keys and "chest" in keys and "generic_processors" in keys
    assert "brain_triage" not in keys, "the validated model is not downloadable — it comes from MODEL_BUNDLE_URL"


def test_torch_is_cpu_pinned_and_versions_match_the_docs():
    env = _workflow()["env"]
    assert env["TORCH_INDEX_URL"] == "https://download.pytorch.org/whl/cpu"
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["TORCH_VERSION"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["TORCHVISION_VERSION"])
    assert env["PYTHON_VERSION"] == "3.11" and env["NODE_VERSION"] == "20"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("--index-url $env:TORCH_INDEX_URL") == 1 and text.count('--index-url "$TORCH_INDEX_URL"') == 2
    # one CPU-torch install per job, every one from the pinned index with the pinned versions
    for job, frag in (("windows", "--index-url $env:TORCH_INDEX_URL"), ("macos", '--index-url "$TORCH_INDEX_URL"'),
                      ("tests", '--index-url "$TORCH_INDEX_URL"')):
        runs = "\n".join(str(s.get("run", "")) for s in _steps(job))
        assert runs.count(frag) == 1, job
        assert "TORCH_VERSION" in runs and "TORCHVISION_VERSION" in runs, job
    for lang in LANGS:
        g = _guide(lang)
        assert f"PyTorch {env['TORCH_VERSION']}" in g and f"torchvision {env['TORCHVISION_VERSION']}" in g, lang
        assert "Python 3.11" in g, lang


def test_smoke_steps_run_after_freeze_and_before_electron_build():
    for job in ("windows", "macos"):
        names = [s.get("name", s.get("uses", "")) for s in _steps(job)]
        idx = {k: next(i for i, n in enumerate(names) if k in n)
               for k in ("Freeze backend", "Restore validated", "Smoke-test", "npm ci", "Upload installer")}
        assert idx["Freeze backend"] < idx["Restore validated"] < idx["Smoke-test"] < idx["npm ci"] < idx["Upload installer"], (job, names)
        smoke = next(s for s in _steps(job) if "Smoke-test" in s.get("name", ""))
        assert smoke["env"]["REQUIRE_VALIDATED"] == "${{ steps.bundle.outputs.have_bundle }}"
        assert smoke["env"]["SENTINEL_MODELS_DIR"].endswith("models")


IDENTITY_STEP = "Verify bundle identity matches the code"
PIN_SNIPPET = "import src.inference.validated_metrics as m; print(m.FALLBACK_IDENTITY)"


def test_bundle_identity_is_checked_against_the_code_pin_in_both_jobs():
    """The model CI restores comes from the hand-set MODEL_BUNDLE_URL secret; the numbers and
    identity the app/docs quote come from the committed bump (validated_metrics.FALLBACK_IDENTITY).
    Both jobs must fail when they differ, right after the restore, before anything is built."""
    from src.inference import validated_metrics as vm
    assert re.fullmatch(r"[0-9a-f]{12}", vm.FALLBACK_IDENTITY), vm.FALLBACK_IDENTITY
    fetch_src = FETCH.read_text(encoding="utf-8")
    assert "_gh_output('model_identity', identity)" in fetch_src      # the output the step compares
    for job, fail in (("windows", 'throw "bundle identity'), ("macos", "exit 1")):
        steps = _steps(job)
        names = [s.get("name", s.get("uses", "")) for s in steps]
        restore = next(i for i, n in enumerate(names) if n.startswith("Restore validated triage model"))
        assert steps[restore]["id"] == "bundle"
        step = steps[restore + 1]
        assert step.get("name") == IDENTITY_STEP, (job, names)
        assert names.count(IDENTITY_STEP) == 1, (job, names)
        assert restore + 1 < next(i for i, n in enumerate(names) if "Smoke-test" in n)
        assert step["if"] == "steps.bundle.outputs.have_bundle == '1'", job
        assert step["env"] == {"BUNDLE_IDENTITY": "${{ steps.bundle.outputs.model_identity }}"}, job
        run = step["run"]
        assert f'-c "{PIN_SNIPPET}"' in run, (job, run)
        # both values are printed, a mismatch is annotated and fails the job
        assert run.count("BUNDLE_IDENTITY") >= 3 and "pinned" in run, (job, run)
        assert "::error title=Bundle identity mismatch::" in run and fail in run, job
    assert "set -euo pipefail" in _steps("macos")[[s.get("name") for s in _steps("macos")].index(IDENTITY_STEP)]["run"]


def test_bundle_identity_snippet_prints_the_pin_from_the_repo_root():
    """What the CI step executes, run the same way (cwd = checkout root, plain `python -c`)."""
    from src.inference import validated_metrics as vm
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE")}
    r = subprocess.run([sys.executable, "-c", PIN_SNIPPET], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().splitlines()[-1] == vm.FALLBACK_IDENTITY


def test_secrets_are_consumed_by_the_workflow_and_documented_everywhere():
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in SECRETS:
        assert f"secrets.{name}" in text, name
        for lang in LANGS:
            assert f"`{name}`" in _guide(lang), (lang, name)
    # signing secrets are only exported when present (an empty CSC_LINK confuses electron-builder)
    assert "if ($env:WIN_CSC_LINK_IN)" in text and 'if [ -n "$CSC_LINK_IN" ]' in text
    assert "CSC_IDENTITY_AUTO_DISCOVERY=false" in text
    assert "--publish never" in text


# ---------------------------------------------------------------- docs section

def test_windows_installer_section_present_with_identical_structure():
    counts = {}
    for lang in LANGS:
        g = _guide(lang)
        assert g.count("\n## 13. ") == 1, lang
        assert ".github/workflows/build-windows.yml" in g and "sentinel-windows-x64-full" in g, lang
        assert "sentinel-windows-x64-NO-VALIDATED-MODEL" in g and "SmartScreen" in g, lang
        assert "packaging\\build_backend.ps1" in g and "packaging\\smoke_backend.ps1" in g, lang
        assert "brain_triage_finetuned/" in g and "MANIFEST.json" in g, lang
        tail = g[g.index("\n## 13. "):]
        counts[lang] = (len(re.findall(r"^## ", tail, re.M)), len(re.findall(r"^### ", tail, re.M)),
                        len(re.findall(r"^\|", tail, re.M)))
    assert len(set(counts.values())) == 1, counts
    assert counts["en"] == (1, 2, 6)


# ---------------------------------------------------------------- packaging scripts

def test_build_backend_ps1_mirrors_the_sh_twin():
    ps1 = _text("packaging/build_backend.ps1")
    sh = _text("packaging/build_backend.sh")
    for frag in ("--noconfirm --clean", "packaging/sentinel_backend.spec", "--distpath packaging/dist",
                 "--workpath packaging/build"):
        assert frag in ps1 and frag in sh, frag
    assert "sentinel-backend.exe" in ps1 and "$env:PYTHON" in ps1
    assert "$ErrorActionPreference = 'Stop'" in ps1 and "$LASTEXITCODE" in ps1
    assert "2>&1" not in ps1 and "*>" not in ps1, "stderr redirection breaks under Windows PowerShell 5.1 + Stop"
    assert "test -x packaging/dist/sentinel-backend/sentinel-backend" in sh


def test_spec_is_windows_aware_and_matches_electron_launcher():
    spec = _text("packaging/sentinel_backend.spec")
    assert "name='sentinel-backend'" in spec and "console=True" in spec
    assert "sys.platform == 'win32'" in spec and "icon.ico" in spec
    assert (REPO / "desktop-app" / "build" / "icon.ico").exists()
    for pkg in ("'pylibjpeg_rle'", "'gdcm'", "'pylibjpeg_openjpeg'"):
        assert pkg in spec, pkg
    main_js = _text("desktop-app/electron/main.js")
    assert "'sentinel-backend.exe'" in main_js and "'sentinel-backend'" in main_js
    assert "'backend', 'sentinel-backend'" in main_js   # resources/backend/sentinel-backend/<exe>
    extra = _package_json()["build"]["extraResources"]
    assert any(e["from"] == "../packaging/dist/sentinel-backend" and e["to"] == "backend/sentinel-backend" for e in extra)


def test_spec_ships_the_bundle_extractor_the_server_loads():
    """POST /models/import uses packaging/fetch_model_bundle.extract_bundle via
    REPO_ROOT/packaging/… — the frozen backend must carry that file."""
    spec = _text("packaging/sentinel_backend.spec")
    assert "(os.path.join(ROOT, 'packaging', 'fetch_model_bundle.py'), 'packaging')" in spec
    server = _text("src/inference/server.py")
    assert "REPO_ROOT / 'packaging' / 'fetch_model_bundle.py'" in server


def test_smoke_scripts_share_the_health_contract():
    sh = _text("packaging/smoke_backend.sh")
    ps1 = _text("packaging/smoke_backend.ps1")
    for frag in ("SENTINEL_DEV_INSECURE", "SENTINEL_OFFLINE", "SENTINEL_DATA_DIR", "SENTINEL_MODELS_DIR",
                 "/health", "8765", "REQUIRE_VALIDATED", "SMOKE OK", "'degraded'", "--host"):
        assert frag in sh and frag in ps1, frag
    assert "sentinel-backend.exe" in ps1 and "sentinel-backend.exe" not in sh
    assert "Stop-Process" in ps1 and "trap cleanup EXIT" in sh
    assert os.access(PACKAGING / "smoke_backend.sh", os.X_OK) and os.access(PACKAGING / "build_backend.sh", os.X_OK)


# ---------------------------------------------------------------- fetch_model_bundle.py

def _make_bundle(root: Path, tamper: bool = False) -> Path:
    d = root / "brain_triage_finetuned"
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"architectures": ["ResNetForImageClassification"]}')
    (d / "preprocessor_config.json").write_text('{"size": 224}')
    (d / "model.safetensors").write_bytes(os.urandom(4096))
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
             for n in ("config.json", "preprocessor_config.json", "model.safetensors")}
    if tamper:
        files["model.safetensors"] = "0" * 64
    (d / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))
    return d


def _zip_dir(model_dir: Path, zip_path: Path, prefix: str) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(model_dir.iterdir()):
            zf.write(p, prefix + p.name)
    return zip_path


def _run_fetch(tmp: Path, *args: str, env_extra: dict | None = None) -> tuple[int, str, dict]:
    out = tmp / "gh_output.txt"
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_BUNDLE_URL", "MODEL_BUNDLE_TOKEN", "GITHUB_TOKEN")}
    env.update({"GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(tmp / "summary.md"), **(env_extra or {})})
    r = subprocess.run([sys.executable, str(FETCH), *args], cwd=str(tmp), env=env, capture_output=True, text=True)
    outputs = dict(l.split("=", 1) for l in out.read_text().splitlines() if "=" in l) if out.exists() else {}
    return r.returncode, r.stdout + r.stderr, outputs


def test_fetch_restores_documented_layout_and_verifies_manifest(tmp_path):
    model_dir = _make_bundle(tmp_path / "src")
    zip_path = _zip_dir(model_dir, tmp_path / "bundle.zip", "brain_triage_finetuned/")
    dest = tmp_path / "models"
    rc, log, outputs = _run_fetch(tmp_path, "--dest", str(dest), "--url", zip_path.as_uri())
    assert rc == 0, log
    assert (dest / "brain_triage_finetuned" / "model.safetensors").read_bytes() == (model_dir / "model.safetensors").read_bytes()
    assert outputs["have_bundle"] == "1" and len(outputs["model_identity"]) == 12
    assert "MANIFEST.json verified" in log
    assert "bundle.zip" not in log.split("downloading")[1].split("\n")[0], "the URL must not be echoed (pre-signed links are secrets)"


def test_fetch_accepts_zip_made_from_inside_the_model_dir(tmp_path):
    model_dir = _make_bundle(tmp_path / "src")
    zip_path = _zip_dir(model_dir, tmp_path / "flat.zip", "")
    dest = tmp_path / "models"
    rc, log, outputs = _run_fetch(tmp_path, "--dest", str(dest), "--url", zip_path.as_uri())
    assert rc == 0, log
    assert (dest / "brain_triage_finetuned" / "MANIFEST.json").exists()
    assert outputs["have_bundle"] == "1"


def test_fetch_fails_on_manifest_mismatch(tmp_path):
    model_dir = _make_bundle(tmp_path / "src", tamper=True)
    zip_path = _zip_dir(model_dir, tmp_path / "bad.zip", "brain_triage_finetuned/")
    rc, log, outputs = _run_fetch(tmp_path, "--dest", str(tmp_path / "models"), "--url", zip_path.as_uri())
    assert rc == 1 and "verification failed" in log and "model.safetensors" in log, log
    assert outputs["have_bundle"] == "0"


def test_fetch_rejects_zip_without_a_model_and_path_traversal(tmp_path):
    with zipfile.ZipFile(tmp_path / "junk.zip", "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    rc, log, _ = _run_fetch(tmp_path, "--dest", str(tmp_path / "m1"), "--url", (tmp_path / "junk.zip").as_uri())
    assert rc == 1 and "extract failed" in log, log
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as zf:
        zf.writestr("brain_triage_finetuned/MANIFEST.json", "{}")
        zf.writestr("../escape.txt", "x")
    rc, log, _ = _run_fetch(tmp_path, "--dest", str(tmp_path / "m2"), "--url", (tmp_path / "evil.zip").as_uri())
    assert rc == 1 and "unsafe zip entry" in log, log
    assert not (tmp_path / "escape.txt").exists()


def test_fetch_without_url_warns_and_exits_zero_unless_required(tmp_path):
    rc, log, outputs = _run_fetch(tmp_path, "--dest", str(tmp_path / "models"))
    assert rc == 0 and "WITHOUT" in log and "MODEL_BUNDLE_URL is not set" in log, log
    assert outputs["have_bundle"] == "0"
    assert "MISSING" in (tmp_path / "summary.md").read_text()
    rc, log, _ = _run_fetch(tmp_path, "--dest", str(tmp_path / "models"), "--require")
    assert rc == 1, log
    # the GitHub annotation is only emitted inside Actions
    rc, log, _ = _run_fetch(tmp_path, "--dest", str(tmp_path / "models"), env_extra={"GITHUB_ACTIONS": "true"})
    assert rc == 0 and "::warning title=Validated model missing::" in log


def test_fetch_uses_env_url_and_github_token_fallback_only_for_api_host(tmp_path):
    model_dir = _make_bundle(tmp_path / "src")
    zip_path = _zip_dir(model_dir, tmp_path / "bundle.zip", "brain_triage_finetuned/")
    rc, log, outputs = _run_fetch(tmp_path, "--dest", str(tmp_path / "models"),
                                  env_extra={"MODEL_BUNDLE_URL": zip_path.as_uri(), "GITHUB_TOKEN": "ghs_x"})
    assert rc == 0 and outputs["have_bundle"] == "1" and "(auth: none)" in log, log


def test_cross_host_redirect_drops_bearer_token():
    spec = importlib.util.spec_from_file_location("fetch_model_bundle", FETCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    handler = mod._CrossHostRedirect()
    req = urllib.request.Request("https://api.github.com/repos/o/r/releases/assets/1",
                                 headers={"Authorization": "Bearer t", "Accept": "application/octet-stream"})
    cross = handler.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/x?X-Amz-Signature=s")
    assert cross.get_header("Authorization") is None
    same = handler.redirect_request(req, None, 302, "Found", {}, "https://api.github.com/other")
    assert same.get_header("Authorization") == "Bearer t"


def test_verify_only_passes_on_the_real_validated_model_if_present():
    real = REPO / "models" / "brain_triage_finetuned"
    if not (real / "MANIFEST.json").exists():
        pytest.skip("validated model not present in this checkout")
    r = subprocess.run([sys.executable, str(FETCH), "--dest", str(real.parent), "--verify-only"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    from src.inference import validated_metrics as vm      # the MANIFEST pin every document quotes
    assert f"identity {vm.load(real, strict=True).identity}" in r.stdout


def test_fetch_model_bundle_multipart_manifest(tmp_path):
    """MODEL_BUNDLE_URL may be a JSON manifest of <100 MB parts (GitHub times out on big
    single uploads): parts are concatenated in order and the whole-zip sha256 verified."""
    import hashlib, json, subprocess, sys, zipfile
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    model_dir = tmp_path / 'src' / 'brain_triage_finetuned'; model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{"architectures":["ViTForImageClassification"]}')
    (model_dir / 'model.safetensors').write_bytes(b'\x00' * 5000)
    files = {n: hashlib.sha256((model_dir / n).read_bytes()).hexdigest() for n in ('config.json', 'model.safetensors')}
    (model_dir / 'MANIFEST.json').write_text(json.dumps({'algorithm': 'sha256', 'files': files}))
    zpath = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        for f in model_dir.iterdir(): zf.write(f, f'brain_triage_finetuned/{f.name}')
    data = zpath.read_bytes(); n = 3; step = (len(data) + n - 1) // n
    parts = []
    for i in range(n):
        pp = tmp_path / f'p{i}'; pp.write_bytes(data[i * step:(i + 1) * step]); parts.append(pp.as_uri())
    man = tmp_path / 'm.json'; man.write_text(json.dumps({'parts': parts, 'sha256': hashlib.sha256(data).hexdigest()}))
    dest = tmp_path / 'dest'
    r = subprocess.run([sys.executable, str(root / 'packaging' / 'fetch_model_bundle.py'), '--dest', str(dest),
                        '--url', man.as_uri()], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / 'brain_triage_finetuned' / 'model.safetensors').read_bytes() == b'\x00' * 5000
    # tampered hash → refused
    man.write_text(json.dumps({'parts': parts, 'sha256': '0' * 64}))
    r = subprocess.run([sys.executable, str(root / 'packaging' / 'fetch_model_bundle.py'), '--dest', str(tmp_path / 'dest2'),
                        '--url', man.as_uri()], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and 'sha256' in (r.stdout + r.stderr)


# ---------------------------------------------------------------- v1.1.0 CI gate / retention / clinic flavour

BUILDERS = ("windows", "macos")
RESTORE_IF = "github.event_name != 'workflow_dispatch' || inputs.include_validated_model"
SKIP_IF = "github.event_name == 'workflow_dispatch' && !inputs.include_validated_model"


def test_tests_job_runs_the_suite_and_gates_both_installer_jobs():
    """D7: the suite runs before any installer is built; a red suite produces no installer."""
    wf = _workflow()
    job = wf["jobs"]["tests"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job.get("permissions") == {"contents": "read"}, "the test job needs no write token"
    assert "environment" not in job, "the test job must not see the MODEL_BUNDLE_URL environment secrets"
    assert job["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, "the CI test run must be hermetic"
    for name in BUILDERS:
        assert wf["jobs"][name]["needs"] in ("tests", ["tests"]), name
    steps = job["steps"]
    setup = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"
    assert setup["with"]["cache"] == "pip"
    assert "requirements.txt" in setup["with"]["cache-dependency-path"]
    runs = [str(s.get("run", "")) for s in steps]
    install = next(r for r in runs if "pip install" in r)
    assert '"torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"' in install
    assert "-r requirements.txt -r requirements-dev.txt --extra-index-url \"$TORCH_INDEX_URL\"" in install
    assert 'expected $TORCH_VERSION' in install and "set -euo pipefail" in install
    assert '"$PYTHON" -m pytest -q -m "not slow"' in runs
    assert '"$PYTHON" -m pytest -q docs/test_docs_consistency.py docs/sales/test_sales_docs.py' in runs
    # the desktop half of the suite runs in the same gate (node --test + tsc), without the Electron binary
    desktop = next(s for s in steps if s.get("working-directory") == "desktop-app")
    assert all(cmd in desktop["run"] for cmd in ("npm ci", "npm run type-check", "npm test")), desktop["run"]
    assert desktop["env"] == {"ELECTRON_SKIP_BINARY_DOWNLOAD": "1"}
    assert _package_json()["scripts"]["test"].startswith("node --test")
    node = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-node"))
    assert node["with"]["node-version"] == "${{ env.NODE_VERSION }}" and node["with"]["cache"] == "npm"
    # no secrets and no artifacts from the test job
    job_text = yaml.safe_dump(job)
    assert "secrets." not in job_text
    assert not any(str(s.get("uses", "")).startswith("actions/upload-artifact") for s in steps)
    # requirements-dev.txt really carries pytest (what the job relies on)
    assert re.search(r"^pytest\b", _text("requirements-dev.txt"), re.M)
    assert "slow:" in _text("pytest.ini"), "the -m 'not slow' filter needs the registered marker"


def test_installer_artifacts_expire_after_one_day():
    """D7: the code repository is public — installers must not stay downloadable."""
    uploads = [s for job in BUILDERS for s in _steps(job)
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert len(uploads) == 2
    for s in uploads:
        assert s["with"]["retention-days"] == 1, s["with"]
    # tag builds still only create DRAFT releases
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("gh release create") == 2 and text.count("--draft") == 2
    assert "gh release edit" not in text and "--draft=false" not in text


def test_dispatch_input_include_validated_model_skips_only_the_restore_step():
    """D3/D7: Run workflow with include_validated_model=false builds the clinic flavour
    (...-NO-VALIDATED-MODEL); tag pushes keep restoring the model as before. Besides the
    restore + notice, only the draft-release attach depends on the input (full flavour only)."""
    wf = _workflow()
    on = wf.get("on") or wf.get(True)
    inp = on["workflow_dispatch"]["inputs"]["include_validated_model"]
    assert inp["type"] == "boolean" and inp["default"] is True
    assert on["push"] == {"tags": ["v*"]}, "tag pushes are unchanged"
    for job in BUILDERS:
        steps = _steps(job)
        names = [s.get("name", s.get("uses", "")) for s in steps]
        restore = next(i for i, n in enumerate(names) if n.startswith("Restore validated triage model"))
        assert steps[restore]["if"] == RESTORE_IF, job
        # a visible notice replaces it in the clinic flavour, and nothing else is conditional on the input
        note = next(i for i, n in enumerate(names) if n.startswith("Validated model excluded"))
        assert steps[note]["if"] == SKIP_IF and note < restore, job
        assert "provision_clinic.py" in steps[note]["run"] and "GITHUB_STEP_SUMMARY" in steps[note]["run"], job
        conditional = [n for s, n in zip(steps, names) if "include_validated_model" in str(s.get("if", ""))]
        attach = next(i for i, n in enumerate(names) if n.startswith("Attach to draft GitHub Release"))
        assert sorted(conditional) == sorted([names[restore], names[note], names[attach]]), (job, conditional)
        # with the restore skipped, have_bundle is empty -> the artifact falls back to NO-VALIDATED-MODEL,
        # the identity check is skipped and the smoke test does not demand 'ok'
        upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        assert upload["with"]["name"].endswith("|| 'NO-VALIDATED-MODEL' }}"), upload["with"]["name"]
        ident = next(s for s in steps if s.get("name") == IDENTITY_STEP)
        assert ident["if"] == "steps.bundle.outputs.have_bundle == '1'"
    assert "${REQUIRE_VALIDATED:-0}" in _text("packaging/smoke_backend.sh")
    assert "$env:REQUIRE_VALIDATED -eq '1'" in _text("packaging/smoke_backend.ps1")


def _eval_if(expr: str, *, event: str, ref: str, include: bool) -> bool:
    """Evaluate the small subset of GitHub expressions the attach/restore guards use."""
    py = (expr.replace("&&", " and ").replace("||", " or ").replace("!inputs.", " not inputs.")
          .replace("github.event_name", repr(event)).replace("inputs.include_validated_model", repr(include))
          .replace("startsWith(github.ref, 'refs/tags/v')", repr(ref.startswith("refs/tags/v"))))
    return bool(eval(py, {"__builtins__": {}}, {}))                       # noqa: S307 - test-only, fixed input


@pytest.mark.parametrize("event,ref,include,attach", [
    ("push", "refs/tags/v1.1.0", None, True),                 # tag push: full flavour attached
    ("workflow_dispatch", "refs/tags/v1.1.0", True, True),    # full flavour dispatched on a tag
    ("workflow_dispatch", "refs/tags/v1.1.0", False, False),  # clinic flavour on a tag: never clobbers
    ("workflow_dispatch", "refs/heads/main", True, False),    # branch build: no release
])
def test_clinic_flavour_dispatched_on_a_tag_never_replaces_the_release_installer(event, ref, include, attach):
    for job in BUILDERS:
        step = next(s for s in _steps(job) if s.get("name", "").startswith("Attach to draft GitHub Release"))
        assert _eval_if(step["if"], event=event, ref=ref, include=include) is attach, (job, event, ref, include)
        assert "--clobber" in step["run"]              # same file names in both flavours → the guard is what protects it
    group = _workflow()["concurrency"]["group"]
    assert "github.event_name" in group and "inputs.include_validated_model" in group, group


def _load_checker():
    spec = importlib.util.spec_from_file_location("_check_bundle_release", PACKAGING / "check_bundle_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _release_dir(tmp_path: Path, *, evaluated_at: str, identity: str | None = None) -> Path:
    from src.inference import validated_metrics as vm
    d = tmp_path / "brain_triage_finetuned"
    d.mkdir(parents=True, exist_ok=True)
    (d / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256", "files": {
        "model.safetensors": (identity or vm.FALLBACK_IDENTITY) + "0" * 52}}))
    (d / "study_level_eval.json").write_text(json.dumps({
        "gate": {"name": "gate mean @0.50", "sens": vm.FALLBACK_SENS, "spec": vm.FALLBACK_SPEC,
                 "acc": vm.FALLBACK_ACC, "tp": vm.FALLBACK_TP, "fn": vm.FALLBACK_FN, "tn": vm.FALLBACK_TN,
                 "fp": vm.FALLBACK_FP, "min_sens": vm.FALLBACK_MIN_SENS, "min_spec": vm.FALLBACK_MIN_SPEC,
                 "passed": True},
        "n_studies": vm.FALLBACK_N_STUDIES, "n_slices": vm.FALLBACK_N_SLICES, "evaluated_at": evaluated_at}))
    return d


def test_bundle_release_record_check_catches_a_re_dated_eval(tmp_path, capsys):
    """Same weights (identity) re-evaluated: the bundle's eval date must equal the bumped pin."""
    from src.inference import validated_metrics as vm
    chk = _load_checker()
    ok = _release_dir(tmp_path / "ok", evaluated_at=f"{vm.FALLBACK_EVALUATED_ON}T03:02:43")
    assert chk.main(["--model-dir", str(ok)]) == 0
    stale = _release_dir(tmp_path / "stale", evaluated_at="2031-01-15T10:00:00")
    assert chk.main(["--model-dir", str(stale)]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH: evaluated_on" in out and "::error title=Bundle release mismatch::" in out
    other = _release_dir(tmp_path / "other", evaluated_at=f"{vm.FALLBACK_EVALUATED_ON}T00:00:00",
                         identity="abcdefabcdef")
    assert chk.compare(other)[0] == 1
    assert chk.main(["--model-dir", str(tmp_path / "missing")]) == 2


def test_both_jobs_check_the_release_record_after_the_identity():
    for job in BUILDERS:
        steps = _steps(job)
        names = [s.get("name", "") for s in steps]
        i = names.index(IDENTITY_STEP)
        rec = names.index("Verify bundle release record matches the code")
        assert rec == i + 1, (job, names)
        assert steps[rec]["if"] == "steps.bundle.outputs.have_bundle == '1'"
        assert "check_bundle_release.py --model-dir models" in steps[rec]["run"]
        assert rec < next(k for k, n in enumerate(names) if "Smoke-test" in n)


def test_admin_guides_describe_ci_gate_retention_and_flavours():
    """The §13 build instructions follow the workflow: tests gate, 1-day artifacts, the input."""
    for lang in LANGS:
        g = _guide(lang)
        tail = g[g.index("\n## 13. "):]
        assert "include_validated_model" in tail, lang
        assert "retention-days: 1" in tail, lang
        assert "pytest -q -m \"not slow\"" in tail, lang
        assert not re.search(r"\b14 (days|дней|kun)\b", tail), f"{lang}: stale 14-day retention in §13"
        assert "NO-VALIDATED-MODEL" in g[:g.index("\n## 4. ")], f"{lang}: §3 must name the clinic flavour"

