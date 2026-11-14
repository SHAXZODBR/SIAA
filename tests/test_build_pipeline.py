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


UPLOAD_IF = "steps.bundle.outputs.have_bundle != '1'"


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
    # only a model-less installer is ever an artifact (public repo), and its name says so loudly
    assert uploads[0]["with"]["name"] == "sentinel-windows-x64-NO-VALIDATED-MODEL"
    assert uploads[1]["with"]["name"] == "sentinel-macos-arm64-NO-VALIDATED-MODEL"
    for s in uploads:
        assert s["if"] == UPLOAD_IF, s


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
    """The model CI restores is the release the code pin names (or the MODEL_BUNDLE_URL fallback); the
    numbers and identity the app/docs quote come from the committed bump (validated_metrics.FALLBACK_IDENTITY).
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

# every variable fetch_model_bundle.py reads — cleared so the developer's shell cannot steer a test
FETCH_ENV = ("MODEL_BUNDLE_URL", "MODEL_BUNDLE_TOKEN", "MODEL_BUNDLE_REPO", "GITHUB_TOKEN", "GITHUB_REPOSITORY")


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
    env = {k: v for k, v in os.environ.items() if k not in FETCH_ENV}
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
        # with the restore skipped, have_bundle is empty -> the NO-VALIDATED-MODEL artifact is uploaded,
        # the identity check is skipped and the smoke test does not demand 'ok'
        upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        assert upload["with"]["name"].endswith("-NO-VALIDATED-MODEL"), upload["with"]["name"]
        assert _eval_if(upload["if"], event="workflow_dispatch", ref="refs/heads/main", include=False, have_bundle="")
        ident = next(s for s in steps if s.get("name") == IDENTITY_STEP)
        assert ident["if"] == "steps.bundle.outputs.have_bundle == '1'"
    assert "${REQUIRE_VALIDATED:-0}" in _text("packaging/smoke_backend.sh")
    assert "$env:REQUIRE_VALIDATED -eq '1'" in _text("packaging/smoke_backend.ps1")


def _eval_expr(expr: str, *, event: str, ref: str, include, have_bundle: str = "1", validated: str = "1"):
    """Evaluate the small subset of GitHub expressions the restore/upload/attach guards use. A skipped
    restore step leaves its outputs empty (have_bundle=''); bundle_validated is only written for
    multipart manifests, so it may be '' on a model-bearing build too."""
    py = expr
    for name, value in (("startsWith(github.ref, 'refs/tags/v')", ref.startswith("refs/tags/v")),
                        ("steps.bundle.outputs.have_bundle", have_bundle),
                        ("steps.bundle.outputs.bundle_validated", validated),
                        ("github.event_name", event), ("inputs.include_validated_model", include)):
        py = py.replace(name, repr(value))
    py = (py.replace("&&", " and ").replace("||", " or ").replace("!=", "\0").replace("!", " not ")
          .replace("\0", "!="))
    return eval(py, {"__builtins__": {}}, {})                             # noqa: S307 - test-only, fixed input


def _eval_if(expr: str, *, event: str, ref: str, include, have_bundle: str = "1", validated: str = "1") -> bool:
    return bool(_eval_expr(expr, event=event, ref=ref, include=include, have_bundle=have_bundle, validated=validated))


def _expr_body(value: str) -> str:
    """'${{ a && b }}' -> 'a && b'."""
    m = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", value.strip())
    assert m, value
    return m.group(1)


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



# ---------------------------------------------------------------- 2026-11-10: public repo, tag gate, bundle by identity

NOTICE_STEP = "Model-bearing installer not uploaded (branch build)"
NOTICE_TEXT = ("installer contains the validated model and is not uploaded from a branch build of a public "
               "repository; push a v* tag to receive it in the tag")
ATTACH_STEP = "Attach to draft GitHub Release (tag builds)"
MODELS_REPO_EXPR = "${{ vars.MODEL_BUNDLE_REPO || 'SHAXZODBR/SIAA-models' }}"


def _step(job: str, prefix: str) -> dict:
    return next(s for s in _steps(job) if s.get("name", "").startswith(prefix))


def test_model_bearing_installer_is_never_a_workflow_artifact():
    """(A) Artifacts of a public repository are downloadable by any signed-in user: an installer that
    holds the validated model leaves the runner only through the tag's draft release."""
    for job, ext in (("windows", "*.exe"), ("macos", "*.dmg")):
        steps = _steps(job)
        names = [s.get("name", s.get("uses", "")) for s in steps]
        uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
        assert len(uploads) == 1, job
        up = uploads[0]
        assert up["if"] == UPLOAD_IF, (job, up)
        assert "${{" not in up["with"]["name"] and up["with"]["name"].endswith("-NO-VALIDATED-MODEL"), up["with"]
        assert up["with"]["retention-days"] == 1 and up["with"]["path"].endswith(ext)
        # the rehearsal notice: model-bearing build outside a v* tag
        notice = _step(job, NOTICE_STEP)
        assert notice["if"] == "steps.bundle.outputs.have_bundle == '1' && !startsWith(github.ref, 'refs/tags/v')", job
        assert "::notice title=" in notice["run"] and "GITHUB_STEP_SUMMARY" in notice["run"], job
        assert notice["run"].count(NOTICE_TEXT) == 2, (job, notice["run"])   # annotation + summary line
        assert "secrets." not in yaml.safe_dump(notice)
        # draft release: tag, not the clinic flavour, a model inside, never an UNVALIDATED-DEMO bundle
        attach = _step(job, ATTACH_STEP)
        cond = attach["if"]
        for frag in ("startsWith(github.ref, 'refs/tags/v')", RESTORE_IF, "steps.bundle.outputs.have_bundle == '1'",
                     "steps.bundle.outputs.bundle_validated != '0'"):
            assert frag in cond, (job, frag, cond)
        assert "gh release create" in attach["run"] and "--draft" in attach["run"]
        build = next(i for i, n in enumerate(names) if n.startswith("Build "))
        refuse = names.index("Refuse an unvalidated model bundle on a release tag")
        assert refuse < build < names.index("Upload installer artifact") < names.index(NOTICE_STEP) < names.index(ATTACH_STEP), (job, names)
    header = WORKFLOW.read_text(encoding="utf-8").split("\nname:", 1)[0]
    for frag in ("PUBLIC", "NEVER a workflow artifact", "DRAFT GitHub Release", "rehearsal", "retention-days: 1",
                 "--require", "MODEL_BUNDLE_REPO", "brain-triage-<identity12>"):
        assert frag in header, frag
    # header secrets section: the token is what matters, the URL is an optional fallback
    assert "the one secret that matters" in header and "optional fallback, no longer edited per model release" in header


# (event, ref, include_validated_model, have_bundle, bundle_validated) -> (artifact?, draft release?, notice?)
# have_bundle '' = restore step skipped (clinic flavour); bundle_validated '' = a plain-zip bundle.
@pytest.mark.parametrize("event,ref,include,have,validated,artifact,attach,notice", [
    ("push", "refs/tags/v1.1.0", None, "1", "1", False, True, False),                  # release: draft only
    ("push", "refs/tags/v1.1.0", None, "1", "", False, True, False),                   # zip bundle (no flag)
    ("workflow_dispatch", "refs/tags/v1.1.0", True, "1", "1", False, True, False),     # full, dispatched on a tag
    ("workflow_dispatch", "refs/tags/v1.1.0", False, "", "", True, False, False),      # clinic on a tag: artifact only
    ("workflow_dispatch", "refs/heads/main", True, "1", "1", False, False, True),      # branch full: rehearsal
    ("workflow_dispatch", "refs/heads/main", True, "1", "0", False, False, True),      # branch UNVALIDATED-DEMO
    ("workflow_dispatch", "refs/heads/main", True, "0", "", True, False, False),       # branch, no token: model-less
    ("workflow_dispatch", "refs/heads/main", False, "", "", True, False, False),       # clinic flavour on a branch
])
def test_installer_route_truth_table(event, ref, include, have, validated, artifact, attach, notice):
    for job in BUILDERS:
        kw = dict(event=event, ref=ref, include=include, have_bundle=have, validated=validated)
        upload = next(s for s in _steps(job) if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        assert _eval_if(upload["if"], **kw) is artifact, (job, kw)
        assert _eval_if(_step(job, ATTACH_STEP)["if"], **kw) is attach, (job, kw)
        assert _eval_if(_step(job, NOTICE_STEP)["if"], **kw) is notice, (job, kw)
        # a model-bearing installer always takes at most one route, and never the public one
        if have == "1":
            assert not artifact and not (attach and notice)


@pytest.mark.parametrize("event,ref,include", [
    ("push", "refs/tags/v1.1.0", None),
    ("workflow_dispatch", "refs/tags/v1.1.0", True),
])
def test_tag_build_without_a_bundle_fails_instead_of_attaching(event, ref, include):
    """(B) the rows "tag + have_bundle != '1'" are unreachable: the restore runs with --require and fails."""
    for job in BUILDERS:
        restore = _step(job, "Restore validated triage model")
        assert _eval_if(restore["if"], event=event, ref=ref, include=include)
        assert _eval_expr(_expr_body(restore["env"]["BUNDLE_REQUIRED"]), event=event, ref=ref, include=include) == "1"


def test_restore_passes_require_on_tags_only():
    """(B) a v* tag ref: --require (a missing bundle fails); a branch: warn and continue."""
    for job in BUILDERS:
        restore = _step(job, "Restore validated triage model")
        expr = _expr_body(restore["env"]["BUNDLE_REQUIRED"])
        assert _eval_expr(expr, event="push", ref="refs/tags/v1.1.0", include=None) == "1"
        assert _eval_expr(expr, event="workflow_dispatch", ref="refs/tags/v2.0.0", include=True) == "1"
        assert _eval_expr(expr, event="workflow_dispatch", ref="refs/heads/main", include=True) == "0"
        assert _eval_expr(expr, event="workflow_dispatch", ref="refs/heads/v-feature", include=True) == "0"
        assert len(re.findall(r"--require(?![-\w])", restore["run"])) == 1, (job, restore["run"])
    win = _step("windows", "Restore validated triage model")["run"]
    assert "if ($env:BUNDLE_REQUIRED -eq '1') { $fetchArgs += @('--require', '--require-newest') }" in win
    assert "packaging\\fetch_model_bundle.py @fetchArgs" in win and "$LASTEXITCODE -ne 0" in win


@pytest.mark.parametrize("required,expect_require", [("1", True), ("0", False)])
def test_macos_restore_script_passes_require_only_when_required(tmp_path, required, expect_require):
    """Run the macOS step's bash exactly as written, with a fake $PYTHON that records its argv."""
    run = _step("macos", "Restore validated triage model")["run"]
    rec = tmp_path / "argv.txt"
    fake = tmp_path / "fake_python"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{rec}"\n')
    fake.chmod(0o755)
    r = subprocess.run(["bash", "-c", run], cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
                       env={"PATH": os.environ.get("PATH", ""), "PYTHON": str(fake), "BUNDLE_REQUIRED": required})
    assert r.returncode == 0, r.stdout + r.stderr
    argv = rec.read_text().splitlines()
    assert argv[:3] == ["packaging/fetch_model_bundle.py", "--dest", "models"], argv
    assert ("--require" in argv) is expect_require, argv


def test_restore_step_resolves_the_bundle_by_identity_from_the_models_repo():
    """(C) both jobs pass MODEL_BUNDLE_REPO (repository variable with a default); MODEL_BUNDLE_URL stays
    wired as the fallback; the identity check stays an independent step."""
    for job in BUILDERS:
        restore = _step(job, "Restore validated triage model")
        env = restore["env"]
        assert env["MODEL_BUNDLE_REPO"] == MODELS_REPO_EXPR, job
        assert env["MODEL_BUNDLE_TOKEN"] == "${{ secrets.MODEL_BUNDLE_TOKEN }}"
        assert env["MODEL_BUNDLE_URL"] == "${{ secrets.MODEL_BUNDLE_URL }}"
        assert env["GITHUB_TOKEN"] == "${{ github.token }}"
        names = [s.get("name", "") for s in _steps(job)]
        assert names[names.index(restore["name"]) + 1] == IDENTITY_STEP
    # no other step sees the model token
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("secrets.MODEL_BUNDLE_TOKEN") == 2 and text.count("secrets.MODEL_BUNDLE_URL") == 2


# ---- fetch_model_bundle.py: resolution by identity (in-process, fake GitHub API, no network)

API_TOKEN = "ghp_FAKE_fetch_token_0123456789"
MODELS_REPO = "o/models"
ASSET_URL = "https://api.github.com/repos/o/models/releases/assets/4242"


def _load_fetch():
    spec = importlib.util.spec_from_file_location("fetch_model_bundle_under_test", FETCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _identity_of(model_dir: Path) -> str:
    return json.loads((model_dir / "MANIFEST.json").read_text())["files"]["model.safetensors"][:12]


class _FakeApi:
    """Stands in for fetch_model_bundle._http_get (the lookup) and .download (the asset bytes)."""

    def __init__(self, mod, release: dict | None = None, error: BaseException | None = None,
                 assets: dict | None = None):
        self.mod, self.release, self.error = mod, release, error
        self.assets = assets or {}          # asset API URL -> local file standing in for its bytes
        self.gets: list[tuple[str, dict]] = []
        self.downloads: list[tuple[str, str]] = []
        self._download = mod.download

    def http_get(self, url, headers):
        self.gets.append((url, dict(headers)))
        if self.error is not None:
            raise self.error
        return json.dumps(self.release).encode()

    def download(self, url, out, token=""):
        self.downloads.append((url, token))
        if url in self.assets:
            url = self.assets[url].as_uri()
        return self._download(url, out, token)


@pytest.fixture
def fetch_env(tmp_path, monkeypatch):
    for var in FETCH_ENV + ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY"):
        monkeypatch.delenv(var, raising=False)
    out = tmp_path / "gh_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    mod = _load_fetch()
    model_dir = _make_bundle(tmp_path / "src")
    zip_path = _zip_dir(model_dir, tmp_path / "bundle.zip", "brain_triage_finetuned/")
    identity = _identity_of(model_dir)
    release = {"tag_name": f"brain-triage-{identity}", "assets": [
        {"name": f"brain_triage_finetuned_{identity}.zip.part-00",
         "url": "https://api.github.com/repos/o/models/releases/assets/4241"},
        {"name": "manifest.json", "url": ASSET_URL}]}
    api = _FakeApi(mod, release=release, assets={ASSET_URL: zip_path})
    monkeypatch.setattr(mod, "_http_get", api.http_get)
    monkeypatch.setattr(mod, "download", api.download)

    def outputs() -> dict:
        return dict(l.split("=", 1) for l in out.read_text().splitlines() if "=" in l) if out.exists() else {}

    return mod, api, identity, zip_path, outputs


def test_fetch_resolves_release_by_identity_and_restores_its_manifest_asset(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    monkeypatch.setenv("MODEL_BUNDLE_REPO", MODELS_REPO)
    monkeypatch.setenv("MODEL_BUNDLE_TOKEN", API_TOKEN)
    dest = tmp_path / "models"
    rc = mod.main(["--dest", str(dest), "--identity", identity])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    # exactly one lookup: the release named after the identity, with the bearer token in the header only
    assert len(api.gets) == 1
    url, headers = api.gets[0]
    assert url == f"https://api.github.com/repos/{MODELS_REPO}/releases/tags/brain-triage-{identity}"
    assert headers["Authorization"] == f"Bearer {API_TOKEN}" and API_TOKEN not in url
    assert headers["Accept"] == "application/vnd.github+json"
    # the manifest asset's API URL is then used exactly like a --url (same token)
    assert api.downloads == [(ASSET_URL, API_TOKEN)]
    assert (dest / "brain_triage_finetuned" / "model.safetensors").is_file()
    assert outputs() == {"have_bundle": "1", "model_identity": identity}
    assert f"brain-triage-{identity}" in cap.out and ASSET_URL not in cap.out
    assert API_TOKEN not in cap.out + cap.err


def test_fetch_identity_defaults_to_the_code_pin(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    pin = tmp_path / "validated_metrics.py"
    pin.write_text(f"FALLBACK_EVALUATED_ON = '2031-01-02'\nFALLBACK_IDENTITY = '{identity}'\n")
    monkeypatch.setattr(mod, "PIN_FILE", pin)
    rc = mod.main(["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN])
    assert rc == 0, capsys.readouterr()
    assert api.gets[0][0].endswith(f"/releases/tags/brain-triage-{identity}")
    assert outputs()["model_identity"] == identity


@pytest.mark.parametrize("flag_url,repo,env_url,expect", [
    (True, True, True, "flag"),          # an operator's explicit --url always wins
    (True, False, True, "flag"),
    (False, True, True, "repo"),         # a configured repository beats the environment URL
    (False, True, False, "repo"),
    (False, False, True, "env"),         # the fallback, exactly as before
    (False, False, False, "absent"),
])
def test_fetch_source_precedence(tmp_path, fetch_env, monkeypatch, capsys, flag_url, repo, env_url, expect):
    mod, api, identity, zip_path, outputs = fetch_env
    flag_zip = tmp_path / "flag.zip"
    env_zip = tmp_path / "env.zip"
    flag_zip.write_bytes(zip_path.read_bytes())
    env_zip.write_bytes(zip_path.read_bytes())
    monkeypatch.setenv("MODEL_BUNDLE_TOKEN", API_TOKEN)
    if repo:
        monkeypatch.setenv("MODEL_BUNDLE_REPO", MODELS_REPO)
    if env_url:
        monkeypatch.setenv("MODEL_BUNDLE_URL", env_zip.as_uri())
    argv = ["--dest", str(tmp_path / "models"), "--identity", identity]
    if flag_url:
        argv += ["--url", flag_zip.as_uri()]
    rc = mod.main(argv)
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    first = api.downloads[0][0] if api.downloads else None
    assert first == {"flag": flag_zip.as_uri(), "repo": ASSET_URL, "env": env_zip.as_uri(), "absent": None}[expect]
    assert bool(api.gets) is (expect == "repo")
    assert outputs()["have_bundle"] == ("0" if expect == "absent" else "1")


def _http_error(code: int):
    import io
    import urllib.error
    body = io.BytesIO(json.dumps({"message": f"Bad credentials {API_TOKEN}"}).encode())
    return urllib.error.HTTPError("https://api.github.com/x", code, f"err {API_TOKEN}", {}, body)


def _url_error():
    import urllib.error
    return urllib.error.URLError(f"connection refused (token {API_TOKEN})")


@pytest.mark.parametrize("case,error,release,expect", [
    ("not_found", _http_error(404), None,
     "no release brain-triage-{id} in o/models, or the token cannot read that repository (expired?)"),
    ("unauthorized", _http_error(401), None, "token rejected (HTTP 401)"),
    ("forbidden", _http_error(403), None, "token rejected (HTTP 403)"),
    ("server_error", _http_error(502), None, "GitHub API answered HTTP 502"),
    ("unreachable", _url_error(), None, "cannot reach the GitHub API"),
    ("no_manifest", None, {"assets": [{"name": "other.zip", "url": ASSET_URL}]},
     "release brain-triage-{id} exists in o/models but holds no manifest.json: re-run packaging/publish_model_bundle.py"),
    ("foreign_asset_host", None, {"assets": [{"name": "manifest.json", "url": "https://evil.example/m.json"}]},
     "has no GitHub API URL"),
    ("not_json", None, "<html>", "is not JSON"),
])
def test_fetch_lookup_failures_are_hard_and_never_fall_back(tmp_path, fetch_env, monkeypatch, capsys,
                                                            case, error, release, expect):
    mod, api, identity, zip_path, outputs = fetch_env
    api.error, api.release = error, release
    if release == "<html>":
        monkeypatch.setattr(mod, "_http_get", lambda url, headers: b"<html>")
    monkeypatch.setenv("MODEL_BUNDLE_REPO", MODELS_REPO)
    monkeypatch.setenv("MODEL_BUNDLE_TOKEN", API_TOKEN)
    monkeypatch.setenv("MODEL_BUNDLE_URL", zip_path.as_uri())      # a perfectly good fallback — must NOT be used
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc = mod.main(["--dest", str(tmp_path / "models"), "--identity", identity])
    cap = capsys.readouterr()
    assert rc == 1, (case, cap.out + cap.err)
    assert expect.format(id=identity) in cap.err, (case, cap.err)
    assert "::error title=Validated model bundle::" in cap.out
    assert API_TOKEN not in cap.out + cap.err, case
    assert api.downloads == [], case                                 # no silent fallback to MODEL_BUNDLE_URL
    assert outputs() == {"have_bundle": "0"}
    assert not (tmp_path / "models" / "brain_triage_finetuned").exists()


def test_fetch_repo_without_token_is_the_absent_case(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, zip_path, outputs = fetch_env
    monkeypatch.setenv("MODEL_BUNDLE_REPO", MODELS_REPO)
    monkeypatch.setenv("MODEL_BUNDLE_URL", zip_path.as_uri())
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_job_token")              # the job token of ANOTHER repository
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/code")
    rc = mod.main(["--dest", str(tmp_path / "models"), "--identity", identity])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert "MODEL_BUNDLE_TOKEN is not set" in cap.out and "MODEL_BUNDLE_URL is not set" not in cap.out
    assert "WITHOUT" in cap.out and outputs() == {"have_bundle": "0"}
    assert api.gets == [] and api.downloads == []
    rc = mod.main(["--dest", str(tmp_path / "models"), "--identity", identity, "--require"])
    assert rc == 1 and "MODEL_BUNDLE_TOKEN is not set" in capsys.readouterr().out
    assert api.gets == [] and api.downloads == []


def test_fetch_uses_the_job_token_only_for_the_building_repository(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    monkeypatch.setenv("MODEL_BUNDLE_REPO", MODELS_REPO)
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_job_token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "O/Models")              # case-insensitive, like GitHub
    rc = mod.main(["--dest", str(tmp_path / "models"), "--identity", identity])
    assert rc == 0, capsys.readouterr()
    assert api.gets[0][1]["Authorization"] == "Bearer ghs_job_token"
    assert api.downloads == [(ASSET_URL, "ghs_job_token")]


def test_fetch_fails_when_the_restored_identity_is_not_the_requested_one(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    other = "abcdefabcdef"
    assert other != identity
    api.release["tag_name"] = f"brain-triage-{other}"                # a release holding the wrong model
    rc = mod.main(["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN,
                   "--identity", other])
    cap = capsys.readouterr()
    assert rc == 1, cap.out + cap.err
    assert f"restored identity {identity} != requested identity {other}" in cap.err
    assert outputs().get("have_bundle") == "0" and "model_identity" not in outputs()
    assert API_TOKEN not in cap.out + cap.err


@pytest.mark.parametrize("argv,expect", [
    (["--repo", "not-a-repo"], "must be OWNER/NAME"),
    (["--repo", "o/.."], "must be OWNER/NAME"),
    (["--repo", MODELS_REPO, "--identity", "XYZ"], "--identity must be 12 lowercase hex digits"),
])
def test_fetch_rejects_malformed_repo_or_identity(tmp_path, fetch_env, capsys, argv, expect):
    mod, api, *_ = fetch_env
    rc = mod.main(["--dest", str(tmp_path / "models"), "--token", API_TOKEN, *argv])
    assert rc == 1 and expect in capsys.readouterr().err
    assert api.gets == [] and api.downloads == []


def test_pin_reader_reads_the_real_code_pin_without_importing_it():
    from src.inference import validated_metrics as vm
    mod = _load_fetch()
    assert mod.PIN_FILE == REPO / "src" / "inference" / "validated_metrics.py"
    assert mod.read_pinned_identity() == vm.FALLBACK_IDENTITY
    assert mod.read_pinned_identity(mod.PIN_FILE) == vm.FALLBACK_IDENTITY


@pytest.mark.parametrize("text,expect", [
    (None, "cannot read the code-pinned identity"),                                 # file missing
    ("FALLBACK_IDENTITY = '0d559766ce58'\ndef broken(:\n", "cannot read the code-pinned identity"),
    ("FALLBACK_SENS = 0.9\n", "no FALLBACK_IDENTITY"),
    ("FALLBACK_IDENTITY = compute()\n", "not a 12-hex-digit string literal"),
    ("FALLBACK_IDENTITY = 'NOT-HEX-AT-ALL'\n", "not a 12-hex-digit string literal"),
])
def test_pin_reader_refuses_a_broken_pin_file(tmp_path, fetch_env, monkeypatch, capsys, text, expect):
    mod, api, *_ = fetch_env
    pin = tmp_path / "validated_metrics.py"
    if text is not None:
        pin.write_text(text)
    with pytest.raises(ValueError, match=re.escape(expect)):
        mod.read_pinned_identity(pin)
    # through main: a clear error before any network call (a token is available)
    monkeypatch.setattr(mod, "PIN_FILE", pin)
    rc = mod.main(["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN])
    assert rc == 1 and expect in capsys.readouterr().err
    assert api.gets == [] and api.downloads == []


def test_fetch_module_import_has_no_side_effects_and_no_pin_read(tmp_path, monkeypatch):
    """The frozen backend execs this file for POST /models/import: importing it must not read the pin."""
    calls = []
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: calls.append(self) or "")
    _load_fetch()
    assert not any(p.name == "validated_metrics.py" for p in calls)


def test_admin_guides_describe_installer_routes_tag_gate_and_no_per_release_secret():
    """(A)(B)(C) in §13 of every guide: model-bearing installers only in the tag's draft release, the clinic
    flavour as a 1-day artifact, --require on tags, and the model found by identity (no secret edit)."""
    markers = {
        "en": ("It is never uploaded as a workflow artifact", "no longer needs editing per model release",
               "The one secret that matters for the model"),
        "ru": ("Он никогда не загружается как артефакт рабочего процесса",
               "больше не нужно менять при каждом выпуске модели", "Единственный секрет, от которого зависит модель"),
        "uz": ("U hech qachon ish jarayoni artefakti sifatida yuklanmaydi",
               "har bir model relizida o‘zgartirish endi shart emas", "Model uchun yagona muhim sir"),
    }
    for lang in LANGS:
        g = _guide(lang)
        tail = g[g.index("\n## 13. "):]
        for frag in ("MODEL_BUNDLE_REPO", "brain-triage-<identity12>", "--require", "`v*`",
                     "**Artifacts** → `sentinel-windows-x64-NO-VALIDATED-MODEL`", "**Releases**",
                     "packaging/publish_model_bundle.py", "scripts/bump_model_release.py", *markers[lang]):
            assert frag in tail, (lang, frag)
        assert "**Artifacts** → `sentinel-windows-x64-full`" not in g, f"{lang}: the full flavour is no longer an artifact"


# ---------------------------------------------------------------- 2026-11-11 review: published release, part hosts, newest pin

def _attach_bash_run(tmp_path: Path, *, exists: bool, is_draft: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the macOS attach step's bash exactly as written, with a fake `gh` on PATH that logs its argv."""
    run = _step("macos", ATTACH_STEP)["run"]
    work = tmp_path / "ws"
    (work / "desktop-app" / "release").mkdir(parents=True)
    (work / "desktop-app" / "release" / "Sentinel Medical AI-9.9.9-arm64.dmg").write_bytes(b"dmg")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "gh.log"
    gh = bindir / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'case "$*" in\n'
        '  "release view "*"--json isDraft"*) printf "%s\\n" "$FAKE_IS_DRAFT"; exit 0;;\n'
        '  "release view "*) [ "$FAKE_EXISTS" = "1" ] && exit 0; exit 1;;\n'
        'esac\n'
        'exit 0\n')
    gh.chmod(0o755)
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", "GITHUB_REF_NAME": "v9.9.9",
           "GITHUB_RUN_ID": "1", "FAKE_EXISTS": "1" if exists else "0", "FAKE_IS_DRAFT": is_draft}
    r = subprocess.run(["bash", "-c", run], cwd=str(work), capture_output=True, text=True, timeout=60, env=env)
    return r, (log.read_text().splitlines() if log.exists() else [])


@pytest.mark.parametrize("exists,is_draft", [(True, "false"), (True, ""), (True, "False")])
def test_attach_refuses_a_published_release_and_uploads_nothing(tmp_path, exists, is_draft):
    """A re-run of the tag's workflow (or a dispatch on the tag ref) after the release was published must
    not upload the model-bearing installer into that public release."""
    r, calls = _attach_bash_run(tmp_path, exists=exists, is_draft=is_draft)
    assert r.returncode != 0, (r.stdout, r.stderr, calls)
    assert not any(c.startswith("release upload") for c in calls), calls
    assert "::error" in r.stdout and "published" in r.stdout and "nothing was uploaded" in r.stdout, r.stdout


@pytest.mark.parametrize("exists", [True, False])
def test_attach_uploads_into_a_draft_release(tmp_path, exists):
    r, calls = _attach_bash_run(tmp_path, exists=exists, is_draft="true")
    assert r.returncode == 0, (r.stdout, r.stderr, calls)
    uploads = [c for c in calls if c.startswith("release upload")]
    assert uploads == ["release upload v9.9.9 desktop-app/release/Sentinel Medical AI-9.9.9-arm64.dmg --clobber"], calls
    assert ("release create v9.9.9 --draft" in "\n".join(calls)) is (not exists), calls
    draft_check = next(i for i, c in enumerate(calls) if "--json isDraft" in c)
    assert draft_check < calls.index(uploads[0])


def test_both_attach_scripts_check_is_draft_before_the_first_upload():
    for job in BUILDERS:
        run = _step(job, ATTACH_STEP)["run"]
        check = run.find("--json isDraft -q .isDraft")
        assert check != -1, (job, run)
        assert check < run.index("gh release upload"), job
        assert check > run.index("gh release create"), job                # after view / create
        assert "::error title=" in run and "nothing was uploaded" in run, job
    win = _step("windows", ATTACH_STEP)["run"]
    assert "-cne 'true'" in win and "throw" in win[win.index("isDraft"):win.index("gh release upload")]
    mac = _step("macos", ATTACH_STEP)["run"]
    assert '!= "true"' in mac and "exit 1" in mac[mac.index("isDraft"):mac.index("gh release upload")]


# ---- token only for the bundle's host; lookup-resolved parts only on api.github.com

PART_SAME = "https://api.github.com/repos/o/models/releases/assets/4241"
PART_FOREIGN = "https://evil.example.net/part1"


def _multipart_assets(tmp_path: Path, zip_path: Path, parts: list[str]) -> dict:
    data = zip_path.read_bytes()
    half = len(data) // 2
    chunks = [data[:half], data[half:]] if len(parts) == 2 else [data]
    assets = {}
    for i, (purl, chunk) in enumerate(zip(parts, chunks)):
        p = tmp_path / f"part{i}"
        p.write_bytes(chunk)
        assets[purl] = p
    man = tmp_path / "multipart_manifest.json"
    man.write_text(json.dumps({"parts": parts, "sha256": hashlib.sha256(data).hexdigest()}))
    assets[ASSET_URL] = man
    return assets


def test_lookup_resolved_bundle_refuses_a_part_outside_the_github_api(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, zip_path, outputs = fetch_env
    api.assets = _multipart_assets(tmp_path, zip_path, [PART_SAME, PART_FOREIGN])
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc = mod.main(["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN,
                   "--identity", identity])
    cap = capsys.readouterr()
    assert rc == 1, cap.out + cap.err
    assert not any(u == PART_FOREIGN for u, _ in api.downloads), api.downloads     # no request to that host
    assert not any(u == PART_SAME for u, _ in api.downloads), api.downloads        # refused before any part
    assert "https://api.github.com" in cap.err and "evil.example.net" in cap.err, cap.err
    assert API_TOKEN not in cap.out + cap.err
    assert outputs() == {"have_bundle": "0"}


def test_explicit_url_sends_the_token_only_to_the_bundle_host(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, zip_path, outputs = fetch_env
    api.assets = _multipart_assets(tmp_path, zip_path, [PART_SAME, PART_FOREIGN])
    rc = mod.main(["--dest", str(tmp_path / "models"), "--url", ASSET_URL, "--token", API_TOKEN])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert api.downloads == [(ASSET_URL, API_TOKEN), (ASSET_URL, API_TOKEN), (PART_SAME, API_TOKEN),
                             (PART_FOREIGN, "")], api.downloads
    assert outputs()["have_bundle"] == "1"
    assert API_TOKEN not in cap.out + cap.err


def test_lookup_resolved_bundle_with_all_parts_on_the_api_restores(tmp_path, fetch_env, capsys):
    mod, api, identity, zip_path, outputs = fetch_env
    other = "https://api.github.com/repos/o/models/releases/assets/4243"
    api.assets = _multipart_assets(tmp_path, zip_path, [PART_SAME, other])
    rc = mod.main(["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN,
                   "--identity", identity])
    assert rc == 0, capsys.readouterr()
    assert [t for _, t in api.downloads] == [API_TOKEN] * 4


# ---- --require-newest: the pinned release must be the newest model release (independent of the pin)

NEWEST_MSG = ("the code pins {id} but the models repository holds a newer model release {tag}: commit and push "
              "the scripts/bump_model_release.py changes before tagging")
LIST_URL = f"https://api.github.com/repos/{MODELS_REPO}/releases?per_page=100"


def _listing_api(api, identity: str, listing):
    api.release["created_at"] = "2031-03-01T10:00:00Z"
    pinned = {"tag_name": f"brain-triage-{identity}", "created_at": "2031-03-01T10:00:00Z", "draft": False}
    body = listing(pinned) if callable(listing) else listing
    tag_body = json.dumps(api.release).encode()

    def http_get(url, headers):
        api.gets.append((url, dict(headers)))
        if url == LIST_URL:
            if isinstance(body, BaseException):
                raise body
            return body if isinstance(body, bytes) else json.dumps(body).encode()
        return tag_body
    return http_get


def _newest_argv(tmp_path, identity, *extra):
    return ["--dest", str(tmp_path / "models"), "--repo", MODELS_REPO, "--token", API_TOKEN,
            "--identity", identity, "--require", "--require-newest", *extra]


def test_require_newest_fails_when_the_models_repo_holds_a_newer_model(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    newer = "brain-triage-aaaaaaaaaaaa"
    monkeypatch.setattr(mod, "_http_get", _listing_api(api, identity, lambda pinned: [
        {"tag_name": newer, "created_at": "2031-04-01T09:00:00Z", "draft": False}, pinned]))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc = mod.main(_newest_argv(tmp_path, identity))
    cap = capsys.readouterr()
    assert rc == 1, cap.out + cap.err
    assert NEWEST_MSG.format(id=identity, tag=newer) in cap.err, cap.err
    assert "::error title=" in cap.out
    assert [u for u, _ in api.gets] == [f"https://api.github.com/repos/{MODELS_REPO}/releases/tags/brain-triage-{identity}",
                                        LIST_URL]
    assert api.gets[1][1]["Authorization"] == f"Bearer {API_TOKEN}" and API_TOKEN not in api.gets[1][0]
    assert api.downloads == [] and outputs() == {"have_bundle": "0"}
    assert API_TOKEN not in cap.out + cap.err


def test_require_newest_passes_when_the_pinned_release_is_the_newest(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    monkeypatch.setattr(mod, "_http_get", _listing_api(api, identity, lambda pinned: [
        pinned, {"tag_name": "brain-triage-bbbbbbbbbbbb", "created_at": "2031-01-01T00:00:00Z", "draft": False}]))
    rc = mod.main(_newest_argv(tmp_path, identity))
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert len(api.gets) == 2 and outputs()["have_bundle"] == "1"


@pytest.mark.parametrize("how", ["env", "flag"])
def test_require_newest_override_warns_and_continues(tmp_path, fetch_env, monkeypatch, capsys, how):
    mod, api, identity, _, outputs = fetch_env
    newer = "brain-triage-aaaaaaaaaaaa"
    monkeypatch.setattr(mod, "_http_get", _listing_api(api, identity, lambda pinned: [
        {"tag_name": newer, "created_at": "2031-04-01T09:00:00Z", "draft": False}, pinned]))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    extra = ()
    if how == "env":
        monkeypatch.setenv("MODEL_BUNDLE_ALLOW_OLDER", "1")
    else:
        extra = ("--allow-older",)
    rc = mod.main(_newest_argv(tmp_path, identity, *extra))
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert "WARNING" in cap.out and newer in cap.out and "::warning title=" in cap.out, cap.out
    assert outputs()["have_bundle"] == "1"
    assert API_TOKEN not in cap.out + cap.err


def test_require_newest_ignores_drafts_and_unrelated_tags(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    monkeypatch.setattr(mod, "_http_get", _listing_api(api, identity, lambda pinned: [
        {"tag_name": "brain-triage-cccccccccccc", "created_at": "2031-05-01T00:00:00Z", "draft": True},
        {"tag_name": "v2.0.0", "created_at": "2031-05-02T00:00:00Z", "draft": False},
        {"tag_name": "chest-triage-dddddddddddd", "created_at": "2031-05-03T00:00:00Z", "draft": False},
        pinned]))
    rc = mod.main(_newest_argv(tmp_path, identity))
    assert rc == 0, capsys.readouterr()


@pytest.mark.parametrize("case,body", [
    ("not_json", b"<html>"),
    ("not_a_list", {"message": "Not Found"}),
    ("entry_without_date", lambda pinned: [{"tag_name": "brain-triage-aaaaaaaaaaaa", "draft": False}, pinned]),
    ("bad_date", lambda pinned: [{"tag_name": "brain-triage-aaaaaaaaaaaa", "created_at": "yesterday",
                                  "draft": False}, pinned]),
    ("http_error", _http_error(403)),
    ("unreachable", _url_error()),
])
def test_require_newest_malformed_listing_is_a_hard_failure(tmp_path, fetch_env, monkeypatch, capsys, case, body):
    mod, api, identity, _, outputs = fetch_env
    monkeypatch.setattr(mod, "_http_get", _listing_api(api, identity, body))
    monkeypatch.setenv("MODEL_BUNDLE_ALLOW_OLDER", "1")          # the override never excuses an unreadable answer
    rc = mod.main(_newest_argv(tmp_path, identity))
    cap = capsys.readouterr()
    assert rc == 1, (case, cap.out + cap.err)
    assert "releases of o/models" in cap.err, (case, cap.err)
    assert API_TOKEN not in cap.out + cap.err, case
    assert api.downloads == [] and outputs() == {"have_bundle": "0"}


def test_require_newest_pinned_release_without_date_is_a_hard_failure(tmp_path, fetch_env, monkeypatch, capsys):
    mod, api, identity, _, outputs = fetch_env
    get = _listing_api(api, identity, lambda pinned: [pinned])
    api.release.pop("created_at")
    monkeypatch.setattr(mod, "_http_get", lambda url, headers: (json.dumps(api.release).encode()
                                                                 if "/tags/" in url else get(url, headers)))
    rc = mod.main(_newest_argv(tmp_path, identity))
    assert rc == 1 and "created_at" in capsys.readouterr().err


def test_require_newest_does_nothing_without_a_repository_lookup(tmp_path, fetch_env, capsys):
    mod, api, identity, zip_path, outputs = fetch_env
    rc = mod.main(["--dest", str(tmp_path / "models"), "--url", zip_path.as_uri(), "--require", "--require-newest"])
    assert rc == 0, capsys.readouterr()
    assert api.gets == []


def test_restore_passes_require_newest_on_tags_only_and_forwards_the_override():
    for job in BUILDERS:
        restore = _step(job, "Restore validated triage model")
        assert restore["env"]["MODEL_BUNDLE_ALLOW_OLDER"] == "${{ vars.MODEL_BUNDLE_ALLOW_OLDER }}", job
        assert restore["run"].count("--require-newest") == 1, job
    win = _step("windows", "Restore validated triage model")["run"]
    assert "if ($env:BUNDLE_REQUIRED -eq '1') { $fetchArgs += @('--require', '--require-newest') }" in win
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Independent re-check" not in text
    assert text.count("vars.MODEL_BUNDLE_ALLOW_OLDER") == 2


@pytest.mark.parametrize("required,expect", [("1", True), ("0", False)])
def test_macos_restore_script_passes_require_newest_only_on_tags(tmp_path, required, expect):
    run = _step("macos", "Restore validated triage model")["run"]
    rec = tmp_path / "argv.txt"
    fake = tmp_path / "fake_python"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{rec}"\n')
    fake.chmod(0o755)
    r = subprocess.run(["bash", "-c", run], cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
                       env={"PATH": os.environ.get("PATH", ""), "PYTHON": str(fake), "BUNDLE_REQUIRED": required})
    assert r.returncode == 0, r.stdout + r.stderr
    argv = rec.read_text().splitlines()
    assert ("--require-newest" in argv) is expect and ("--require" in argv) is expect, argv


def test_identity_recheck_is_documented_as_what_it_is():
    """D11 and the workflow comment must not call the pin comparison independent; the newest-release check is."""
    d11 = _text("docs/dev/finish_line_contract.md")
    d11 = d11[d11.index("D11."):]
    d11 = d11[:d11.index("\nD12.")] if "\nD12." in d11 else d11
    assert "stays as an independent check" not in d11
    assert "--require-newest" in d11 and "MODEL_BUNDLE_ALLOW_OLDER" in d11
    assert "MODEL_BUNDLE_URL secret" not in _text("packaging/check_bundle_release.py")


def test_admin_guides_describe_published_release_refusal_and_newest_model_gate():
    markers = {
        "en": ("uploads nothing and fails", "newer model release"),
        "ru": ("ничего не загружает и завершается ошибкой", "более новый выпуск модели"),
        "uz": ("hech narsa yuklamaydi va xato bilan tugaydi", "yangiroq model relizi"),
    }
    for lang in LANGS:
        tail = _guide(lang)
        tail = tail[tail.index("\n## 13. "):]
        for frag in (*markers[lang], "MODEL_BUNDLE_ALLOW_OLDER", "--require-newest"):
            assert frag in tail, (lang, frag)
