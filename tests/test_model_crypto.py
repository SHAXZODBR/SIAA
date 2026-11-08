"""Encrypted model weights at rest (src/utils/model_crypto.py).

Round-trip + wrong-key + tamper at the byte/file level, then the real thing: a
COPY of models/brain_triage_finetuned is encrypted in place with the CLI
(plaintext removed), the registry is pointed at it via
SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE, and the same image through the
plaintext and the encrypted model must give identical probabilities. Model
identity must report the PLAINTEXT sha (== the MANIFEST pin) either way, and
scripts/eval_study_level.py's manifest check must accept the encrypted dir.

Set SENTINEL_TEST_SCRATCH to choose where the 343 MB copy is made.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from conftest import REPO
from src.utils import model_crypto as mc

KEY_A = bytes(range(32))
KEY_B = bytes(range(1, 33))
KEY_A_HEX = KEY_A.hex()
PLAIN_DIR = REPO / "models" / "brain_triage_finetuned"
NEEDS_MODEL = pytest.mark.skipif(not (PLAIN_DIR / "model.safetensors").exists(),
                                 reason=f"plaintext triage model missing: {PLAIN_DIR}")


# ---------------------------------------------------------------- primitives

def test_bytes_round_trip_and_format():
    blob = mc.encrypt_bytes(b"weights" * 1000, KEY_A, "model.safetensors")
    assert len(blob) == mc.NONCE_BYTES + 7000 + 16          # nonce || ct || tag
    assert mc.decrypt_bytes(blob, KEY_A, "model.safetensors") == b"weights" * 1000
    assert mc.encrypt_bytes(b"x", KEY_A) != mc.encrypt_bytes(b"x", KEY_A)   # fresh nonce every time


def test_wrong_key_name_or_flipped_byte_fails():
    blob = mc.encrypt_bytes(b"secret", KEY_A, "model.safetensors")
    with pytest.raises(InvalidTag):
        mc.decrypt_bytes(blob, KEY_B, "model.safetensors")
    with pytest.raises(InvalidTag):
        mc.decrypt_bytes(blob, KEY_A, "pytorch_model.bin")     # AAD binds the file name
    tampered = bytearray(blob)
    tampered[mc.NONCE_BYTES + 2] ^= 0x01
    with pytest.raises(InvalidTag):
        mc.decrypt_bytes(bytes(tampered), KEY_A, "model.safetensors")


def test_file_round_trip_removes_plaintext_and_writes_sidecar(tmp_path):
    fp = tmp_path / "model.safetensors"
    data = os.urandom(50_000)
    fp.write_bytes(data)
    enc = mc.encrypt_file(fp, KEY_A)
    assert enc == tmp_path / "model.safetensors.enc" and enc.exists()
    assert not fp.exists(), "plaintext must be removed"
    meta = mc.read_meta(enc)
    assert meta["alg"] == "AES-256-GCM" and meta["size"] == len(data)
    assert meta["plaintext_sha256"] == hashlib.sha256(data).hexdigest()
    assert meta["created_at"]
    assert mc.is_encrypted_dir(tmp_path)
    assert mc.plaintext_sha256_of_dir(tmp_path) == meta["plaintext_sha256"]
    assert mc.decrypt_file_to_bytes(enc, KEY_A) == data
    with pytest.raises(InvalidTag):
        mc.decrypt_file_to_bytes(enc, KEY_B)
    # a doctored sidecar is caught too
    m = json.loads(mc.meta_path_for(enc).read_text()); m["plaintext_sha256"] = "0" * 64
    mc.meta_path_for(enc).write_text(json.dumps(m))
    with pytest.raises(ValueError):
        mc.decrypt_file_to_bytes(enc, KEY_A)
    mc.meta_path_for(enc).write_text(json.dumps(meta))
    out = mc.decrypt_file(enc, KEY_A)
    assert out == fp and fp.read_bytes() == data and not enc.exists()
    assert not mc.is_encrypted_dir(tmp_path)


def test_key_sources(monkeypatch):
    # a license.dat on the machine running the suite must not change the source under test
    monkeypatch.setattr("src.utils.license.license_model_key_secret", lambda *a, **k: None)
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", KEY_A_HEX)
    assert mc.get_model_key(refresh=True) == KEY_A
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", "abcd")
    with pytest.raises(ValueError):
        mc.get_model_key(refresh=True)
    monkeypatch.delenv("SENTINEL_MODEL_KEY_HEX")
    k1 = mc.get_model_key(refresh=True)                    # install secret + machine id
    k2 = mc.get_model_key(refresh=True)
    assert k1 == k2 and len(k1) == 32 and k1 != KEY_A
    from src.utils.paths import MODEL_KEY_PATH
    assert MODEL_KEY_PATH.exists() and (MODEL_KEY_PATH.stat().st_mode & 0o777) == 0o600
    assert mc.derive_key(b"s", "m1") != mc.derive_key(b"s", "m2")   # machine-bound
    mc._key_cache = None


# ---------------------------------------------------------------- real model

@pytest.fixture(scope="module")
def enc_dir():
    """Encrypted COPY of models/brain_triage_finetuned (never the real dir)."""
    if not (PLAIN_DIR / "model.safetensors").exists():
        pytest.skip(f"plaintext triage model missing: {PLAIN_DIR}")
    root = Path(tempfile.mkdtemp(prefix="sentinel_enc_", dir=os.environ.get("SENTINEL_TEST_SCRATCH") or None))
    d = root / "brain_triage_encrypted"
    d.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "MANIFEST.json"):
        if (PLAIN_DIR / name).exists():
            shutil.copy2(PLAIN_DIR / name, d / name)
    env = dict(os.environ, SENTINEL_MODEL_KEY_HEX=KEY_A_HEX)
    proc = subprocess.run([sys.executable, "-m", "src.utils.model_crypto", "encrypt", str(d)],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (d / "model.safetensors").exists()
    assert (d / "model.safetensors.enc").exists() and (d / "model.safetensors.enc.meta.json").exists()
    yield d
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def key_a(monkeypatch):
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", KEY_A_HEX)
    mc.get_model_key(refresh=True)
    yield KEY_A
    mc._key_cache = None


def _test_image() -> np.ndarray:
    return np.random.RandomState(0).rand(224, 224).astype(np.float32)


@NEEDS_MODEL
def test_identity_reports_plaintext_sha_for_encrypted_dir(enc_dir):
    from src.inference import model_registry as mr
    pinned = json.loads((PLAIN_DIR / "MANIFEST.json").read_text())["files"]["model.safetensors"]
    assert mr.weights_sha256_for_source(str(PLAIN_DIR)) == pinned
    assert mr.weights_sha256_for_source(str(enc_dir)) == pinned      # via the sidecar, no key needed
    assert mr.is_source_encrypted(str(enc_dir)) is True
    assert mr.is_source_encrypted(str(PLAIN_DIR)) is False


@NEEDS_MODEL
def test_encrypted_and_plaintext_models_give_identical_probabilities(enc_dir, key_a):
    from src.inference import model_registry as mr
    img = _test_image()
    plain = mr.build_predictor_from_dir(PLAIN_DIR, "cpu")(img)
    enc = mr.build_predictor_from_dir(enc_dir, "cpu")(img)
    assert set(plain) == {"abnormal", "normal"} == set(enc)
    for k in plain:
        assert enc[k] == pytest.approx(plain[k], abs=1e-6), (k, plain, enc)
    assert not (enc_dir / "model.safetensors").exists(), "plaintext must never be written to disk"


@NEEDS_MODEL
def test_registry_override_loads_encrypted_dir(enc_dir, key_a, monkeypatch):
    from src.inference import model_registry as mr
    card = mr.REGISTRY["brain_triage"]
    monkeypatch.setenv("SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE", str(enc_dir))
    assert mr.local_model_dirs(card) == [enc_dir]                  # override is the ONLY candidate
    predictor, available, reason = mr._build_hf_predictor(card, "cpu")
    assert available is True, reason
    assert reason == f"loaded from {enc_dir}" and predictor.weights_source == str(enc_dir)
    img = _test_image()
    ref = mr.build_predictor_from_dir(PLAIN_DIR, "cpu")(img)
    got = predictor(img)
    for k in ref:
        assert got[k] == pytest.approx(ref[k], abs=1e-6)
    monkeypatch.setenv("SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE", str(enc_dir / "does-not-exist"))
    assert mr.local_model_dirs(card) == []                          # a bad override never falls back silently


@NEEDS_MODEL
def test_wrong_key_cannot_load_encrypted_model(enc_dir, monkeypatch):
    from src.inference import model_registry as mr
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", KEY_B.hex())
    mc.get_model_key(refresh=True)
    try:
        with pytest.raises(InvalidTag):
            mr.build_predictor_from_dir(enc_dir, "cpu")
        card = mr.REGISTRY["brain_triage"]
        monkeypatch.setenv("SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE", str(enc_dir))
        _, available, reason = mr._build_hf_predictor(card, "cpu")   # degrades, never raises
        assert available is False and reason
    finally:
        mc._key_cache = None


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_study_level", REPO / "scripts" / "eval_study_level.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@NEEDS_MODEL
def test_manifest_check_accepts_encrypted_dir_and_still_plaintext(enc_dir, tmp_path):
    ev = _eval_module()
    pinned = json.loads((PLAIN_DIR / "MANIFEST.json").read_text())
    # existing plaintext behaviour untouched
    assert ev.check_manifest(PLAIN_DIR, PLAIN_DIR / "MANIFEST.json")["files"] == pinned["files"]
    # the SAME manifest verifies the encrypted copy (plaintext sha via sidecar)
    assert ev.check_manifest(enc_dir, enc_dir / "MANIFEST.json")["files"] == pinned["files"]
    # a manifest written from the encrypted dir pins plaintext AND ciphertext
    written = ev.write_manifest(enc_dir, tmp_path / "MANIFEST.json")["files"]
    assert written["model.safetensors"] == pinned["files"]["model.safetensors"]
    assert written["model.safetensors.enc"] == ev.sha256_file(enc_dir / "model.safetensors.enc")
    # a doctored sidecar fails the gate (exit 3)
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for f in enc_dir.iterdir():
        if f.name != "model.safetensors.enc":
            shutil.copy2(f, tampered / f.name)
    (tampered / "model.safetensors.enc").write_bytes(b"")
    m = json.loads((tampered / "model.safetensors.enc.meta.json").read_text()); m["plaintext_sha256"] = "f" * 64
    (tampered / "model.safetensors.enc.meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit) as ex:
        ev.check_manifest(tampered, tampered / "MANIFEST.json")
    assert ex.value.code == ev.EXIT_MANIFEST_FAILED


# ---------------------------------------------------------------- key resolution (contract D3)

LICENSE_SECRET = bytes(range(100, 132))           # synthetic 32-byte model_key_secret
MACHINE = "f" * 64


@pytest.fixture
def key_env(monkeypatch, tmp_path):
    """Isolated key world: install secret in tmp, fixed machine id, a controllable
    license secret and license file. Yields a dict the test mutates."""
    import src.utils.license as lic
    import src.utils.paths as paths
    state = {"secret": None, "license_path": tmp_path / "license.dat"}
    monkeypatch.delenv("SENTINEL_MODEL_KEY_HEX", raising=False)
    monkeypatch.setattr(paths, "MODEL_KEY_PATH", tmp_path / "model_key.bin")
    monkeypatch.setattr(lic, "get_machine_id", lambda: MACHINE)
    monkeypatch.setattr(lic, "get_license_path", lambda: state["license_path"])
    monkeypatch.setattr(lic, "license_model_key_secret", lambda *a, **k: state["secret"])
    mc._key_cache = None
    yield state
    mc._key_cache = None


def test_key_resolution_order_env_license_install(key_env, monkeypatch, tmp_path):
    import src.utils.paths as paths
    key_env["secret"] = LICENSE_SECRET
    monkeypatch.setenv("SENTINEL_MODEL_KEY_HEX", KEY_A_HEX)
    assert mc.get_model_key(refresh=True) == KEY_A and mc.key_source() == "env"
    monkeypatch.delenv("SENTINEL_MODEL_KEY_HEX")
    assert mc.get_model_key(refresh=True) == mc.derive_key(LICENSE_SECRET, MACHINE)
    assert mc.key_source() == "license"
    assert not paths.MODEL_KEY_PATH.exists(), "a license key must not create an install secret"
    key_env["secret"] = None
    k = mc.get_model_key(refresh=True)
    assert mc.key_source() == "install_secret" and paths.MODEL_KEY_PATH.exists()
    assert k == mc.derive_key(paths.MODEL_KEY_PATH.read_bytes(), MACHINE)
    assert k != mc.derive_key(LICENSE_SECRET, MACHINE)


def test_clinic_key_from_secret_hex_matches_derive_key():
    assert mc.clinic_key_from_secret_hex(LICENSE_SECRET.hex(), MACHINE) == mc.derive_key(LICENSE_SECRET, MACHINE)


def test_key_cache_follows_a_license_imported_at_runtime(key_env):
    """No restart needed: the key is re-resolved when license.dat appears/changes."""
    install_key = mc.get_model_key()
    assert mc.key_source() == "install_secret"
    key_env["secret"] = LICENSE_SECRET
    assert mc.get_model_key() == install_key                  # nothing changed on disk yet → cached
    key_env["license_path"].write_text("signed-license-bytes")    # the import writes the file
    assert mc.get_model_key() == mc.derive_key(LICENSE_SECRET, MACHINE)
    assert mc.key_source() == "license"


def test_keyinfo_prints_source_never_key(key_env, capsys):
    key_env["secret"] = LICENSE_SECRET
    assert mc.main(["keyinfo"]) == 0
    out = capsys.readouterr().out
    key = mc.derive_key(LICENSE_SECRET, MACHINE)
    assert "key source:      license" in out and mc.key_fingerprint(key) in out
    assert key.hex() not in out and LICENSE_SECRET.hex() not in out


# ---------------------------------------------------------------- at-rest encryption (contract D2)

def _hf_dir(root: Path, name: str, weights: bytes, weight_name: str = "model.safetensors") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"architectures":["ViTForImageClassification"]}')
    (d / "preprocessor_config.json").write_text('{"size":224}')
    (d / weight_name).write_bytes(weights)
    return d


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def models_tree(tmp_path):
    root = tmp_path / "models"
    w = {n: os.urandom(30_000 + i) for i, n in enumerate(
        ("triage", "dementia", "prev", "cand", "hf", "bin"))}
    _hf_dir(root, "brain_triage_finetuned", w["triage"])
    d = _hf_dir(root, "brain_dementia_finetuned", w["dementia"])
    mc.encrypt_file(d / "model.safetensors", KEY_A)                   # already encrypted
    _hf_dir(root, "brain_triage_finetuned_prev_0123456789ab", w["prev"])
    _hf_dir(root, "brain_triage_candidate_v2", w["cand"])
    _hf_dir(root / "hf", "google__vit-base-patch16-224", w["hf"])
    _hf_dir(root, "legacy_finetuned", w["bin"], weight_name="pytorch_model.bin")
    return root, w


def test_encrypt_at_rest_synthetic_tree(models_tree):
    root, w = models_tree
    untouched = {n: _snapshot(root / n) for n in ("brain_triage_finetuned_prev_0123456789ab",
                                                  "brain_triage_candidate_v2", "hf", "legacy_finetuned")}
    rep = mc.encrypt_proprietary_models_at_rest(root, KEY_A)
    assert rep["encrypted"] == ["brain_triage_finetuned"]
    assert rep["already"] == ["brain_dementia_finetuned"]
    assert sorted(rep["skipped"]) == ["brain_triage_candidate_v2", "brain_triage_finetuned_prev_0123456789ab",
                                      "legacy_finetuned"]
    assert rep["failed"] == {} and rep["rekeyed"] == []
    d = root / "brain_triage_finetuned"
    assert not (d / "model.safetensors").exists() and mc.is_encrypted_dir(d)
    assert mc.decrypt_file_to_bytes(d / "model.safetensors.enc", KEY_A) == w["triage"]
    assert mc.read_meta(d / "model.safetensors.enc")["plaintext_sha256"] == hashlib.sha256(w["triage"]).hexdigest()
    assert not [p for p in root.rglob("*") if ".at-rest-tmp" in p.name], "temp files left behind"
    for n, snap in untouched.items():
        assert _snapshot(root / n) == snap, f"{n} must not be touched"
    again = mc.encrypt_proprietary_models_at_rest(root, KEY_A)           # idempotent
    assert again["encrypted"] == [] and sorted(again["already"]) == ["brain_dementia_finetuned",
                                                                        "brain_triage_finetuned"]


def test_encrypt_at_rest_rolls_back_when_verify_fails(models_tree, monkeypatch):
    root, w = models_tree
    d = root / "brain_triage_finetuned"
    before = _snapshot(d)

    def boom(*a, **k):
        raise ValueError("verify_mismatch: forced by test")
    monkeypatch.setattr(mc, "_verify_encrypted_file", boom)
    rep = mc.encrypt_proprietary_models_at_rest(root, KEY_A)            # never raises
    assert rep["failed"]["brain_triage_finetuned"].startswith("verify_mismatch")
    assert "brain_triage_finetuned" not in rep["encrypted"]
    assert _snapshot(d) == before and (d / "model.safetensors").read_bytes() == w["triage"]
    assert not (d / "model.safetensors.enc").exists() and not (d / "model.safetensors.enc.meta.json").exists()


def test_encrypt_at_rest_rolls_back_after_rename(models_tree, monkeypatch):
    """A failure AFTER the ciphertext was renamed into place removes it again and keeps the plaintext."""
    root, w = models_tree
    d = root / "brain_triage_finetuned"
    before = _snapshot(d)
    monkeypatch.setattr(mc, "read_meta", lambda p: {"plaintext_sha256": "0" * 64})
    rep = mc.encrypt_proprietary_models_at_rest(root, KEY_A)
    assert rep["failed"]["brain_triage_finetuned"].startswith("verify_mismatch")
    assert _snapshot(d) == before


def test_encrypt_at_rest_never_raises_without_key(models_tree, monkeypatch):
    root, _ = models_tree
    before = _snapshot(root)

    def no_key(*a, **k):
        raise ValueError("SENTINEL_MODEL_KEY_HEX is not valid hex")
    monkeypatch.setattr(mc, "get_model_key", no_key)
    rep = mc.encrypt_proprietary_models_at_rest(root)
    assert list(rep["failed"].values())[0].startswith("key_unavailable")
    assert _snapshot(root) == before
    assert mc.encrypt_proprietary_models_at_rest(root / "missing", KEY_A)["failed"] == {}


def test_encrypt_at_rest_reports_foreign_ciphertext(models_tree):
    """Ciphertext made with another key is reported (key_mismatch) and left untouched."""
    root, _ = models_tree
    before = _snapshot(root / "brain_dementia_finetuned")
    rep = mc.encrypt_proprietary_models_at_rest(root, KEY_B)
    assert rep["failed"]["brain_dementia_finetuned"].startswith("key_mismatch")
    assert _snapshot(root / "brain_dementia_finetuned") == before


def test_encrypt_at_rest_rekeys_install_secret_ciphertext_to_license_key(key_env, tmp_path):
    """An install encrypted before its license (with model_key_secret) arrived keeps loading."""
    root = tmp_path / "m"
    w = os.urandom(20_000)
    d = _hf_dir(root, "brain_triage_finetuned", w)
    install_key = mc.get_model_key(refresh=True)
    assert mc.key_source() == "install_secret"
    assert mc.encrypt_proprietary_models_at_rest(root)["encrypted"] == ["brain_triage_finetuned"]
    key_env["secret"] = LICENSE_SECRET
    license_key = mc.get_model_key(refresh=True)
    assert mc.key_source() == "license" and license_key != install_key
    rep = mc.encrypt_proprietary_models_at_rest(root)
    assert rep["rekeyed"] == ["brain_triage_finetuned"] and rep["failed"] == {}
    assert mc.decrypt_file_to_bytes(d / "model.safetensors.enc", license_key) == w
    with pytest.raises(InvalidTag):
        mc.decrypt_file_to_bytes(d / "model.safetensors.enc", install_key)
    assert mc.encrypt_proprietary_models_at_rest(root)["rekeyed"] == []       # once only


@pytest.mark.parametrize("env,platform,models_dir,expected", [
    ({}, "win32", "C:/Sentinel/resources/models", (False, "not_requested")),
    ({"SENTINEL_ENCRYPT_MODELS_AT_REST": "1"}, "win32", "C:/Sentinel/resources/models", (True, "enabled")),
    ({"SENTINEL_ENCRYPT_MODELS_AT_REST": "1", "SENTINEL_DEV_INSECURE": "1"}, "win32", "x", (False, "dev_insecure")),
    ({"SENTINEL_ENCRYPT_MODELS_AT_REST": "1"}, "darwin", "/Applications/Sentinel.app/Contents/Resources/models",
     (False, "macos_app_bundle")),
])
def test_at_rest_policy(monkeypatch, env, platform, models_dir, expected):
    monkeypatch.delenv("SENTINEL_ENCRYPT_MODELS_AT_REST", raising=False)
    monkeypatch.delenv("SENTINEL_DEV_INSECURE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mc.sys, "platform", platform)
    assert mc.at_rest_encryption_policy(Path(models_dir)) == expected


# ---------------------------------------------------------------- per-clinic bundle (contract D3)

def _release_dir(root: Path, weights: bytes) -> Path:
    d = _hf_dir(root, "brain_triage_finetuned", weights)
    (d / "training_metrics.json").write_text('{"epochs": 3}')
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
             for n in ("config.json", "preprocessor_config.json", "model.safetensors")}
    (d / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))
    (d / "study_level_eval.json").write_text(json.dumps({"gate": {"passed": True},
                                                         "manifest": {"files": files}}))
    (d / ".DS_Store").write_bytes(b"junk")
    return d


def _snapshot_with_mtime(d: Path) -> dict:
    return {str(p.relative_to(d)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(d.rglob("*"))}


def test_export_encrypted_bundle_round_trip(tmp_path):
    import zipfile
    weights = os.urandom(40_000)
    src = _release_dir(tmp_path / "src", weights)
    before = _snapshot_with_mtime(src)
    secret, fp = LICENSE_SECRET, "0123abcd" + "e" * 56
    key = mc.derive_key(secret, fp)
    out = mc.export_encrypted_bundle(src, key, tmp_path / "out" / "bundle.zip", fp[:8])
    assert out == tmp_path / "out" / "bundle.zip" and out.is_file()
    assert _snapshot_with_mtime(src) == before, "the source dir must be byte-identical (never touched)"
    assert sorted(p.name for p in out.parent.iterdir()) == ["bundle.zip"]     # no temp zip left
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
        zf.extractall(tmp_path / "x")
    assert names == sorted(f"brain_triage_finetuned/{n}" for n in (
        "MANIFEST.json", "config.json", "preprocessor_config.json", "model.safetensors.enc",
        "model.safetensors.enc.meta.json", "study_level_eval.json", "training_metrics.json"))
    d = tmp_path / "x" / "brain_triage_finetuned"
    assert mc.decrypt_file_to_bytes(d / "model.safetensors.enc", mc.derive_key(secret, fp)) == weights
    with pytest.raises(InvalidTag):
        mc.decrypt_file_to_bytes(d / "model.safetensors.enc", mc.derive_key(secret, "1" * 64))
    man = json.loads((d / "MANIFEST.json").read_text())
    assert man["files"]["model.safetensors"] == hashlib.sha256(weights).hexdigest()
    assert man["files"]["model.safetensors.enc"] == hashlib.sha256((d / "model.safetensors.enc").read_bytes()).hexdigest()
    assert man["encrypted_for_machine_fp8"] == fp[:8] and str(tmp_path) not in json.dumps(man)
    meta = mc.read_meta(d / "model.safetensors.enc")
    assert meta["kdf"] == "HKDF-SHA256" and meta["key_fingerprint"] == mc.key_fingerprint(key)
    assert meta["bound_machine_fp8"] == fp[:8]
    # parity with the release gate's manifest writer + checker
    ev = _eval_module()
    assert mc.MANIFEST_FILES == ev.MANIFEST_FILES
    assert ev.write_manifest(d, tmp_path / "parity.json")["files"] == man["files"]
    assert ev.check_manifest(d, d / "MANIFEST.json")["files"] == man["files"]


def test_export_refuses_corrupted_or_encrypted_source(tmp_path):
    src = _release_dir(tmp_path / "src", os.urandom(10_000))
    (src / "model.safetensors").write_bytes(os.urandom(10_000))          # weights != MANIFEST pin
    with pytest.raises(mc.BundleExportError) as ex:
        mc.export_encrypted_bundle(src, KEY_A, tmp_path / "o.zip", "0123abcd")
    assert ex.value.code == "manifest_mismatch" and not (tmp_path / "o.zip").exists()
    enc_src = _release_dir(tmp_path / "enc", os.urandom(10_000))
    mc.encrypt_file(enc_src / "model.safetensors", KEY_A)
    with pytest.raises(mc.BundleExportError) as ex:
        mc.export_encrypted_bundle(enc_src, KEY_A, tmp_path / "o.zip", "0123abcd")
    assert ex.value.code == "source_not_plaintext"
    good = _release_dir(tmp_path / "good", os.urandom(10_000))
    for fp8, key, code in (("0123ABCD", KEY_A, "bad_fingerprint"), ("0123abc", KEY_A, "bad_fingerprint"),
                           ("0123abcd", b"short", "bad_key")):
        with pytest.raises(mc.BundleExportError) as ex:
            mc.export_encrypted_bundle(good, key, tmp_path / "o.zip", fp8)
        assert ex.value.code == code
    assert not (tmp_path / "o.zip").exists()


def test_encrypt_dir_at_rest_single_dir(tmp_path, monkeypatch):
    """The public one-dir helper (server: previous release after an import) verifies before delete."""
    w = os.urandom(12_000)
    d = _hf_dir(tmp_path, "brain_triage_finetuned_prev_0123456789ab", w)
    before = _snapshot(d)
    real_verify = mc._verify_encrypted_file

    def boom(*a, **k):
        raise ValueError("verify_mismatch: forced by test")
    monkeypatch.setattr(mc, "_verify_encrypted_file", boom)
    with pytest.raises(ValueError, match="^verify_mismatch"):
        mc.encrypt_dir_at_rest(d, KEY_A)
    assert _snapshot(d) == before
    monkeypatch.setattr(mc, "_verify_encrypted_file", real_verify)
    enc = mc.encrypt_dir_at_rest(d, KEY_A)
    assert enc == d / "model.safetensors.enc" and not (d / "model.safetensors").exists()
    assert mc.decrypt_file_to_bytes(enc, KEY_A) == w
    with pytest.raises(ValueError, match="^not_plaintext"):
        mc.encrypt_dir_at_rest(d, KEY_A)
