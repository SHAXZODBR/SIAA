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
