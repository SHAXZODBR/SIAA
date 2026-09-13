"""Installer build pipeline — structural checks of .github/workflows/build-windows.yml and the
packaging/ scripts it runs, plus unit tests of packaging/fetch_model_bundle.py.

Nothing here needs Windows, Node, PyInstaller or the network: the workflow is validated as
data (it parses, every repo file it references exists, every `npm run` script exists, model
keys are registry keys, artifact paths match electron-builder's output dir, the versions it
pins are the ones the admin guides quote) and the bundle restorer is exercised against
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
    assert wf["jobs"]["windows"]["defaults"]["run"]["shell"] == "pwsh"
    assert wf["jobs"]["macos"]["defaults"]["run"]["shell"] == "bash"
    assert wf["permissions"] == {"contents": "write"}


def test_workflow_references_only_existing_repo_files():
    # steps only — the comment header abbreviates twins as build_backend.{ps1,sh}
    text = "\n".join(l for l in WORKFLOW.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
    refs = set(re.findall(r"(?:packaging|scripts)[\\/][A-Za-z0-9_.]+", text))
    assert refs, "workflow references no packaging/ or scripts/ files"
    missing = [r for r in refs if not (REPO / r.replace("\\", "/")).exists()]
    assert not missing, missing
    # both freeze twins and both smoke twins are wired in
    for name in ("build_backend.ps1", "build_backend.sh", "smoke_backend.ps1", "smoke_backend.sh",
                 "fetch_model_bundle.py"):
        assert any(r.endswith(name) for r in refs), name


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
    assert text.count("--index-url $env:TORCH_INDEX_URL") == 1 and text.count('--index-url "$TORCH_INDEX_URL"') == 1
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
    assert "identity 0d559766ce58" in r.stdout    # the MANIFEST pin every document quotes
