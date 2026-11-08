"""POST /models/import (D3) and encryption at rest on startup (D2).

A TINY synthetic ViT (a few KB, random weights) stands in for the triage
model: it is encrypted with model_crypto under a test key
(SENTINEL_MODEL_KEY_HEX), wrapped with a MANIFEST.json that pins BOTH the
plaintext and the ciphertext sha256, zipped in the contract layout and dropped
into DATA_DIR/model_inbox/. The registry's MODELS_DIR, its caches, DATA_DIR
and validated_metrics.METRICS are monkeypatched per test, so the real
models/ tree and the session's loaded models are never touched.

Covers: success (swap, previous release kept as …_prev_<identity>, registry
reload, identity, audit), bundle_not_for_this_machine under another key with
nothing changed, inbox confinement, manifest/sidecar/plaintext checks,
rollback when the reload fails, the built-in extractor's parity with
packaging/fetch_model_bundle.extract_bundle, and the startup encryption gate.
"""

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.inference import server as srv

ADMIN = {"username": "admin", "password": "Test-Admin-Pass-2026"}
KEY_A = "11" * 32          # "this machine"
KEY_B = "22" * 32          # another clinic's PC


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_tiny_model(dest: Path, seed: int) -> Path:
    """config.json + preprocessor_config.json + model.safetensors (plaintext)."""
    import torch
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    torch.manual_seed(seed)
    cfg = ViTConfig(image_size=32, patch_size=16, hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32, num_labels=2,
                    id2label={0: "abnormal", 1: "normal"}, label2id={"abnormal": 0, "normal": 1})
    ViTForImageClassification(cfg).save_pretrained(str(dest), safe_serialization=True)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(str(dest))
    return dest


def write_eval(model_dir: Path, passed: bool = True) -> None:
    (model_dir / "study_level_eval.json").write_text(json.dumps({
        "gate": {"name": "gate mean @0.50", "acc": 0.8, "sens": 0.9, "spec": 0.5, "tp": 9, "fn": 1,
                 "tn": 5, "fp": 5, "min_sens": 0.85, "min_spec": 0.4, "passed": passed},
        "n_studies": 20, "n_slices": 100, "evaluated_at": "2031-01-15T10:00:00",
    }))


def make_encrypted_bundle(root: Path, key_hex: str, seed: int = 1, *, layout: str = "dir",
                          tamper=None, pin_enc: bool = True, keep_plaintext: bool = False) -> tuple[Path, str]:
    """Encrypted model dir + MANIFEST (plaintext AND ciphertext pins) → zip.
    Returns (zip path, plaintext sha256)."""
    from src.utils import model_crypto

    src = make_tiny_model(root / f"src_{seed}_{key_hex[:2]}" / "brain_triage_finetuned", seed)
    write_eval(src)
    plain = src / "model.safetensors"
    plain_sha = _sha(plain)
    if keep_plaintext:
        model_crypto.encrypt_file(plain, bytes.fromhex(key_hex), remove_plaintext=False)
    else:
        model_crypto.encrypt_file(plain, bytes.fromhex(key_hex))
    files = {n: _sha(src / n) for n in ("config.json", "preprocessor_config.json", "study_level_eval.json")}
    files["model.safetensors"] = plain_sha
    if pin_enc:
        files["model.safetensors.enc"] = _sha(src / "model.safetensors.enc")
    (src / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))
    if tamper:
        tamper(src)
    out = root / f"bundle_{seed}_{key_hex[:2]}_{layout}.zip"
    with zipfile.ZipFile(out, "w") as zf:
        for f in sorted(src.iterdir()):
            arc = f"brain_triage_finetuned/{f.name}" if layout == "dir" else f.name
            zf.write(f, arc)
    return out, plain_sha


@pytest.fixture(scope="module")
def api() -> TestClient:
    return TestClient(srv.app)


@pytest.fixture(scope="module")
def bearer(api) -> dict:
    r = api.post("/auth/login", json=ADMIN)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def env(monkeypatch, tmp_path) -> dict:
    """Isolated MODELS_DIR (with a plaintext 'current' release), DATA_DIR,
    registry caches, metrics and model key."""
    from src.inference import model_registry as mr
    from src.inference import validated_metrics as vm
    from src.utils import model_crypto

    models = tmp_path / "models"
    data = tmp_path / "data"
    inbox = data / "model_inbox"
    inbox.mkdir(parents=True)
    current = make_tiny_model(models / "brain_triage_finetuned", seed=99)
    write_eval(current)
    old_sha = _sha(current / "model.safetensors")
    (current / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256",
                                                        "files": {"model.safetensors": old_sha}}))
    monkeypatch.setattr(mr, "MODELS_DIR", models)
    monkeypatch.setattr(mr, "_loaded_models", {})
    monkeypatch.setattr(mr, "_identity_cache", {})
    monkeypatch.setattr(srv, "DATA_DIR", data)
    # validated_metrics.reload() rewrites METRICS + every module constant and the
    # registry card's notes after an import: snapshot them all, restored after the test
    for name in ("METRICS", "MODEL_DIR", "SOURCE", "SENS", "SPEC", "ACC", "TP", "FN", "TN", "FP",
                 "N_STUDIES", "N_ABNORMAL", "N_NORMAL", "N_SLICES", "EVALUATED_ON", "IDENTITY",
                 "MIN_SENS", "MIN_SPEC", "THRESHOLD", "SENS_CI", "SPEC_CI"):
        monkeypatch.setattr(vm, name, getattr(vm, name))
    monkeypatch.setattr(mr.REGISTRY["brain_triage"], "notes", mr.REGISTRY["brain_triage"].notes)
    monkeypatch.setattr(model_crypto, "_key_cache", None)
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", KEY_A)
    monkeypatch.delenv("SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE", raising=False)
    monkeypatch.delenv("SENTINEL_ENCRYPT_MODELS_AT_REST", raising=False)
    return {"models": models, "data": data, "inbox": inbox, "tmp": tmp_path, "old_sha": old_sha}


def _stage(env, bundle: Path) -> Path:
    dest = env["inbox"] / bundle.name
    dest.write_bytes(bundle.read_bytes())
    return dest


def _no_leftovers(models: Path) -> bool:
    return not [p for p in models.iterdir() if p.name.startswith((".import_staging_", ".import_failed_"))]


# ─── success ───────────────────────────────────────────────────────────────

def test_import_succeeds_with_the_right_key(api, bearer, env):
    from src.inference import model_registry as mr
    from src.inference import validated_metrics as vm
    from src.inference.auth_routes import db

    bundle, plain_sha = make_encrypted_bundle(env["tmp"], KEY_A)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["identity"] == plain_sha[:12]
    assert body["encrypted_at_rest"] is True and body["status"] == "validated"
    assert body["previous_identity"] == env["old_sha"][:12]

    installed = env["models"] / "brain_triage_finetuned"
    assert (installed / "model.safetensors.enc").is_file() and not (installed / "model.safetensors").exists()
    prev = env["models"] / f"brain_triage_finetuned_prev_{env['old_sha'][:12]}"
    assert prev.is_dir() and _sha(prev / "model.safetensors") == env["old_sha"]
    assert _no_leftovers(env["models"])

    entry = mr._loaded_models["brain_triage"]
    assert entry["available"] and Path(entry["source_path"]).resolve() == installed.resolve()
    ident = mr.get_model_identity("brain_triage")
    assert ident["sha256_12"] == plain_sha[:12] and ident["encrypted_at_rest"] is True
    assert vm.METRICS.identity == plain_sha[:12] and vm.METRICS.validated
    # reload() refreshes the module constants and the model-list card without a restart
    assert vm.IDENTITY == plain_sha[:12] and vm.SENS == 0.9 and vm.N_STUDIES == 20
    assert "over 20 held-out" in mr.REGISTRY["brain_triage"].notes
    rows = [x for x in db.get_audit_log(limit=20) if x["action"] == "model_import"]
    assert rows and plain_sha[:12] in rows[0]["details"]


def test_flat_zip_layout_is_accepted_and_unvalidated_eval_reported(api, bearer, env):
    bundle, plain_sha = make_encrypted_bundle(env["tmp"], KEY_A, seed=3, layout="flat",
                                              tamper=lambda d: _rewrite_eval(d, passed=False))
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 200, r.text
    assert r.json()["identity"] == plain_sha[:12]
    assert r.json()["status"] == "pending" and r.json()["not_validated_reason"]


def _rewrite_eval(d: Path, passed: bool) -> None:
    write_eval(d, passed=passed)
    man = json.loads((d / "MANIFEST.json").read_text())
    man["files"]["study_level_eval.json"] = _sha(d / "study_level_eval.json")
    (d / "MANIFEST.json").write_text(json.dumps(man))


# ─── refusals leave the installed model untouched ──────────────────────────

def _assert_untouched(env):
    cur = env["models"] / "brain_triage_finetuned"
    assert _sha(cur / "model.safetensors") == env["old_sha"]
    assert not list(env["models"].glob("brain_triage_finetuned_prev_*"))
    assert _no_leftovers(env["models"])


def test_wrong_machine_key_is_refused(api, bearer, env):
    from src.inference.auth_routes import db

    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_B, seed=4)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 400, r.text
    assert r.json()["reason"] == "bundle_not_for_this_machine" and r.json()["ok"] is False
    _assert_untouched(env)
    rows = [x for x in db.get_audit_log(limit=20) if x["action"] == "model_import_failed"]
    assert rows and "bundle_not_for_this_machine" in rows[0]["details"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"tamper": lambda d: (d / "config.json").write_text('{"tampered": true}')}, "manifest_mismatch"),
    ({"pin_enc": False}, "manifest_invalid"),
    ({"keep_plaintext": True}, "bundle_contains_plaintext"),
    ({"tamper": lambda d: (d / "model.safetensors.enc").write_bytes(
        (d / "model.safetensors.enc").read_bytes()[:-1] + b"\x00")}, "manifest_mismatch"),
    ({"tamper": lambda d: (d / "model.safetensors.enc.meta.json").unlink()}, "bundle_incomplete"),
])
def test_bundle_integrity_checks(api, bearer, env, kwargs, reason):
    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_A, seed=5, **kwargs)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 400, r.text
    assert r.json()["reason"] == reason
    _assert_untouched(env)


def test_sidecar_pin_mismatch(env):
    """Sidecar not pinned in MANIFEST but disagreeing with the plaintext pin."""
    def tamper(d: Path):
        (d / "model.safetensors.enc.meta.json").write_text(json.dumps({"plaintext_sha256": "0" * 64}))
        man = json.loads((d / "MANIFEST.json").read_text())
        man["files"].pop("model.safetensors.enc.meta.json", None)
        (d / "MANIFEST.json").write_text(json.dumps(man))

    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_A, seed=6, tamper=tamper)
    with pytest.raises(srv._ModelImportFailure) as ei:
        srv.import_model_bundle(str(_stage(env, bundle)))
    assert ei.value.reason == "sidecar_mismatch"
    _assert_untouched(env)


def test_path_must_be_inside_the_inbox(api, bearer, env):
    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_A, seed=7)
    r = api.post("/models/import", json={"path": str(bundle)}, headers=bearer)        # outside
    assert r.status_code == 400 and r.json()["reason"] == "path_outside_inbox"
    link = env["inbox"] / "link.zip"
    os.symlink(bundle, link)                                                         # escapes via symlink
    r = api.post("/models/import", json={"path": str(link)}, headers=bearer)
    assert r.status_code == 400 and r.json()["reason"] == "path_outside_inbox"
    r = api.post("/models/import", json={"path": "model_inbox/x.zip"}, headers=bearer)
    assert r.status_code == 400 and r.json()["reason"] == "path_not_absolute"
    r = api.post("/models/import", json={"path": str(env["inbox"] / "missing.zip")}, headers=bearer)
    assert r.status_code == 404 and r.json()["reason"] == "bundle_not_found"
    (env["inbox"] / "fake.zip").write_text("not a zip")
    r = api.post("/models/import", json={"path": str(env["inbox"] / "fake.zip")}, headers=bearer)
    assert r.status_code == 400 and r.json()["reason"] == "bundle_not_zip"
    _assert_untouched(env)


def test_requires_admin(api, env):
    assert api.post("/models/import", json={"path": "/x.zip"}).status_code == 401


def test_override_env_blocks_import(api, bearer, env, monkeypatch):
    monkeypatch.setenv("SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE", str(env["tmp"]))
    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_A, seed=8)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 409 and r.json()["reason"] == "model_dir_overridden"


def test_reload_failure_rolls_back(api, bearer, env, monkeypatch):
    from src.inference import model_registry as mr

    def failing_reload(models_dir, identity):
        raise srv._ModelImportFailure(500, "reload_failed", "simulated load failure")

    monkeypatch.setattr(srv, "_reload_and_check", failing_reload)
    bundle, _ = make_encrypted_bundle(env["tmp"], KEY_A, seed=9)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 500 and r.json()["reason"] == "reload_failed"
    _assert_untouched(env)
    entry = mr._loaded_models.get("brain_triage")                  # previous model reloaded
    assert entry and entry["available"]
    assert Path(entry["source_path"]).resolve() == (env["models"] / "brain_triage_finetuned").resolve()


def test_import_without_a_current_model(api, bearer, env):
    import shutil
    shutil.rmtree(env["models"] / "brain_triage_finetuned")
    bundle, plain_sha = make_encrypted_bundle(env["tmp"], KEY_A, seed=10)
    r = api.post("/models/import", json={"path": str(_stage(env, bundle))}, headers=bearer)
    assert r.status_code == 200, r.text
    assert r.json()["previous_identity"] is None and r.json()["previous_dir"] is None


# ─── extractor parity with packaging/fetch_model_bundle ───────────────────

@pytest.mark.parametrize("layout", ["dir", "flat"])
def test_builtin_extractor_matches_packaging(tmp_path, layout):
    bundle, _ = make_encrypted_bundle(tmp_path, KEY_A, seed=11, layout=layout)
    ref = srv._bundle_extractor()
    assert ref is not srv._extract_bundle_builtin, "packaging/fetch_model_bundle.py should load in the repo"
    a = ref(bundle, tmp_path / "a")
    b = srv._extract_bundle_builtin(bundle, tmp_path / "b")
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())
    assert all(_sha(a / p.name) == _sha(b / p.name) for p in a.iterdir())


def test_builtin_extractor_refuses_unsafe_entries(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        srv._extract_bundle_builtin(bad, tmp_path / "out")


# ─── D2: encryption at rest on startup ────────────────────────────────────

@pytest.fixture
def fake_encryptor(monkeypatch):
    from src.utils import model_crypto
    calls = []

    def fake(models_dir):
        calls.append(Path(models_dir))
        return {"encrypted": ["brain_triage_finetuned"], "already_encrypted": [], "skipped": [],
                "failed": []}

    monkeypatch.setattr(model_crypto, "encrypt_proprietary_models_at_rest", fake, raising=False)
    return calls


def test_encrypt_at_rest_runs_only_when_gated(monkeypatch, fake_encryptor, tmp_path):
    monkeypatch.delenv("SENTINEL_ENCRYPT_MODELS_AT_REST", raising=False)
    monkeypatch.setattr(srv, "DEV_INSECURE", False)
    assert srv._maybe_encrypt_models_at_rest(tmp_path) is None and fake_encryptor == []

    monkeypatch.setenv("SENTINEL_ENCRYPT_MODELS_AT_REST", "1")
    report = srv._maybe_encrypt_models_at_rest(tmp_path)
    assert report["encrypted"] == ["brain_triage_finetuned"] and fake_encryptor == [tmp_path]

    monkeypatch.setattr(srv, "DEV_INSECURE", True)                  # never in dev-insecure mode
    assert srv._maybe_encrypt_models_at_rest(tmp_path) is None and len(fake_encryptor) == 1


def test_encrypt_at_rest_skips_macos_app_bundle(monkeypatch, fake_encryptor):
    monkeypatch.setenv("SENTINEL_ENCRYPT_MODELS_AT_REST", "1")
    monkeypatch.setattr(srv, "DEV_INSECURE", False)
    monkeypatch.setattr(srv.sys, "platform", "darwin")
    assert srv._maybe_encrypt_models_at_rest(Path("/Applications/Sentinel.app/Contents/Resources/models")) is None
    assert fake_encryptor == []


def test_encrypt_at_rest_never_raises(monkeypatch, tmp_path):
    from src.utils import model_crypto

    def boom(models_dir):
        raise RuntimeError("disk full")

    monkeypatch.setenv("SENTINEL_ENCRYPT_MODELS_AT_REST", "1")
    monkeypatch.setattr(srv, "DEV_INSECURE", False)
    monkeypatch.setattr(model_crypto, "encrypt_proprietary_models_at_rest", boom, raising=False)
    assert srv._maybe_encrypt_models_at_rest(tmp_path) is None
    monkeypatch.delattr(model_crypto, "encrypt_proprietary_models_at_rest", raising=False)
    assert srv._maybe_encrypt_models_at_rest(tmp_path) is None


def test_lifespan_encrypts_before_warm_up():
    import inspect
    src = inspect.getsource(srv.lifespan)
    assert src.index("_maybe_encrypt_models_at_rest()") < src.index("Warm the brain panel")
    assert src.index("load_and_verify_local_license") < src.index("_maybe_encrypt_models_at_rest()")
