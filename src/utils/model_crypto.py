"""
================================================================================
  SENTINEL MEDICAL AI — ENCRYPTED MODEL WEIGHTS AT REST (IP protection)
================================================================================

  The locally fine-tuned detectors (models/*_finetuned) are the product's IP.
  On a clinic PC they sit on a disk anyone with a screwdriver can image, so the
  weight files are stored encrypted and decrypted straight into memory at load
  time — plaintext never touches the disk of the install.

  Cipher       AES-256-GCM (authenticated: a flipped byte fails to decrypt)
  Key          HKDF-SHA256 over  <secret> || <machine fingerprint>
               get_model_key() resolves, in this order (contract D3):
                 1. SENTINEL_MODEL_KEY_HEX=<64 hex>  raw key        source 'env'
                 2. license.dat model_key_secret     (valid license
                    for THIS machine only) + fingerprint             source 'license'
                 3. DATA_DIR/model_key.bin (32 random bytes, 0600,
                    created on first use) + fingerprint              source 'install_secret'
               machine fingerprint = src.utils.license.get_machine_id().
               The vendor encrypts a clinic's bundle offline with
               derive_key(secret, fingerprint) (packaging/provision_clinic.py →
               export_encrypted_bundle), so the clinic never sees plaintext.
               The cached key is re-resolved automatically when the env var or
               the license file changes (a license imported at runtime takes
               effect for the next decrypt without a restart).
  File format  <name>.enc            = 12-byte nonce || ciphertext || 16-byte GCM tag
               <name>.enc.meta.json  = {plaintext_sha256, size, created_at, alg, ...}
               The sidecar carries the PLAINTEXT sha256 so model identity
               (model_identity.sha256_12) and MANIFEST.json pins stay stable
               across installs, whatever key encrypted the bytes.

  At rest       encrypt_proprietary_models_at_rest(models_dir) encrypts every
               models/*_finetuned dir on first start of an installed app (D2):
               verify-before-delete, rollback on any failure, never raises.
               at_rest_encryption_policy() says whether this process may do it.
  Delivery      export_encrypted_bundle(model_dir, key, out_zip, fp8) builds the
               per-clinic zip from a TEMP copy (the source dir is never touched).

  CLI (never run it on the real models/ tree — work on a copy):
    python -m src.utils.model_crypto encrypt <model_dir>   # weights → .enc, plaintext REMOVED
    python -m src.utils.model_crypto decrypt <model_dir>   # reverse
    python -m src.utils.model_crypto keyinfo               # key fingerprint (not the key) + source
================================================================================
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

from cryptography.exceptions import InvalidTag  # noqa: F401  (re-exported for callers)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ALG = "AES-256-GCM"
KDF = "HKDF-SHA256"
NONCE_BYTES = 12
KEY_BYTES = 32
ENC_SUFFIX = ".enc"
META_SUFFIX = ".enc.meta.json"
# Weight files the CLI encrypts / the loaders know how to decrypt.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.pt")

_HKDF_SALT = b"sentinel-model-key-v1"
_HKDF_INFO = b"sentinel model weights AES-256-GCM"
_AAD_PREFIX = b"sentinel-model-v1:"

KEY_ENV = "SENTINEL_MODEL_KEY_HEX"
KEY_SOURCE_ENV = "env"
KEY_SOURCE_LICENSE = "license"
KEY_SOURCE_INSTALL = "install_secret"
_KEY_SOURCE_LABELS = {
    KEY_SOURCE_ENV: "SENTINEL_MODEL_KEY_HEX (raw key)",
    KEY_SOURCE_LICENSE: "license.dat model_key_secret + machine fingerprint",
    KEY_SOURCE_INSTALL: "install secret (DATA_DIR/model_key.bin) + machine fingerprint",
}

logger = logging.getLogger(__name__)


# ============================================================================
# KEY
# ============================================================================

def _read_or_create_install_secret(path: Optional[Path] = None) -> bytes:
    if path is None:
        from src.utils.paths import MODEL_KEY_PATH
        path = MODEL_KEY_PATH
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 16:
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = os.urandom(KEY_BYTES)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(secret)
    finally:
        try:
            os.chmod(str(path), 0o600)
        except OSError:
            pass
    return secret


def derive_key(install_secret: bytes, machine_id: str) -> bytes:
    """HKDF-SHA256(secret || machine_id) → 32-byte AES key. Deterministic, so the
    vendor can reproduce a clinic's key from the secret it provisioned + the
    fingerprint the clinic sent (same one the license is bound to)."""
    ikm = install_secret + b"||" + machine_id.encode("utf-8")
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=_HKDF_SALT, info=_HKDF_INFO).derive(ikm)


def clinic_key_from_secret_hex(secret_hex: str, machine_id: str) -> bytes:
    """The key a clinic's app will resolve from a license carrying
    model_key_secret=<secret_hex> on the machine with fingerprint <machine_id>
    (vendor side: packaging/provision_clinic.py encrypts with exactly this)."""
    return derive_key(bytes.fromhex(secret_hex), machine_id)


_key_cache: Optional[bytes] = None
_key_source_cache: Optional[str] = None
_key_cache_token: Optional[tuple] = None


def _key_state_token() -> tuple:
    """What the cached key depends on: the env override and the license file
    (path, mtime, size). A change in either re-resolves the key."""
    env = os.environ.get(KEY_ENV, "").strip()
    lic: Optional[tuple] = None
    try:
        from src.utils.license import get_license_path
        lp = get_license_path()
        st = lp.stat()
        lic = (str(lp), st.st_mtime_ns, st.st_size)
    except Exception:
        lic = None
    return (env, lic)


def _env_key(hex_key: str) -> bytes:
    try:
        key = bytes.fromhex(hex_key)
    except ValueError as e:
        raise ValueError("SENTINEL_MODEL_KEY_HEX is not valid hex") from e
    if len(key) != KEY_BYTES:
        raise ValueError(f"SENTINEL_MODEL_KEY_HEX must be {KEY_BYTES} bytes ({KEY_BYTES * 2} hex chars)")
    return key


def _license_secret() -> Optional[bytes]:
    """model_key_secret of a license valid for THIS machine, else None (never raises)."""
    try:
        from src.utils.license import license_model_key_secret
        return license_model_key_secret()
    except Exception as e:                                    # pragma: no cover - defensive
        logger.warning("model key: license lookup failed (%s: %s) — using the install secret",
                       type(e).__name__, e)
        return None


def _resolve_key() -> tuple[bytes, str]:
    """(key, source) by the D3 order: env > license > install secret."""
    hex_key = os.environ.get(KEY_ENV, "").strip()
    if hex_key:
        return _env_key(hex_key), KEY_SOURCE_ENV
    from src.utils.license import get_machine_id
    secret = _license_secret()
    if secret is not None:
        return derive_key(secret, get_machine_id()), KEY_SOURCE_LICENSE
    return derive_key(_read_or_create_install_secret(), get_machine_id()), KEY_SOURCE_INSTALL


def get_model_key(refresh: bool = False) -> bytes:
    """The AES key for this install, resolved SENTINEL_MODEL_KEY_HEX > license
    model_key_secret (valid license only) > DATA_DIR/model_key.bin, each HKDF'd
    with the machine fingerprint (the env key is used raw). Cached per process;
    re-resolved on refresh=True or when the env var / license file changed.
    Raises ValueError for a malformed SENTINEL_MODEL_KEY_HEX."""
    global _key_cache, _key_source_cache, _key_cache_token
    token = _key_state_token()
    if _key_cache is not None and not refresh and token == _key_cache_token and _key_source_cache:
        return _key_cache
    key, source = _resolve_key()
    if source != _key_source_cache or key != _key_cache:
        logger.info("model key: source=%s fingerprint=%s", source, key_fingerprint(key))
        if source == KEY_SOURCE_LICENSE:
            _remember_key(key, source)
    _key_cache, _key_source_cache, _key_cache_token = key, source, token
    return key


def key_source(refresh: bool = False) -> str:
    """Where the current model key comes from: 'env' | 'license' | 'install_secret'."""
    get_model_key(refresh=refresh)
    assert _key_source_cache is not None
    return _key_source_cache


def _install_secret_key_if_present() -> Optional[bytes]:
    """The install-secret key WITHOUT creating DATA_DIR/model_key.bin (used to
    re-key weights encrypted before a license with model_key_secret arrived)."""
    try:
        from src.utils.paths import MODEL_KEY_PATH
        if not MODEL_KEY_PATH.exists():
            return None
        data = MODEL_KEY_PATH.read_bytes()
        if len(data) < 16:
            return None
        from src.utils.license import get_machine_id
        return derive_key(data, get_machine_id())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Key history — keys of EARLIER licences of this install.
#
# A licence renewal (a new provision_clinic run) or an expiry changes the key
# this install resolves. Ciphertext in MODELS_DIR (every *_finetuned dir the
# at-rest pass encrypted or re-keyed, e.g. brain_finetuned) would otherwise be
# undecryptable for ever — its plaintext was deleted after the verified
# encryption, and only brain_triage_finetuned is re-delivered with a renewal. Every licence-derived key this install has used is
# therefore remembered in DATA_DIR/model_key_history.json (0600, next to
# model_key.bin, which holds equally sensitive material) so the at-rest pass can
# re-key such dirs to the current key. The file holds derived keys only (never
# a licence secret), at most _KEY_HISTORY_MAX of them, newest last.
# ---------------------------------------------------------------------------

_KEY_HISTORY_MAX = 8
_KEY_HISTORY_NAME = "model_key_history.json"


def _key_history_path() -> Path:
    from src.utils.paths import MODEL_KEY_PATH
    return Path(MODEL_KEY_PATH).with_name(_KEY_HISTORY_NAME)


def _read_key_history_entries() -> list[dict]:
    """Well-formed entries of the history file ([] when absent/unreadable). Never raises."""
    try:
        path = _key_history_path()
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for e in (data.get("keys") or []) if isinstance(data, dict) else []:
            kh = e.get("key_hex") if isinstance(e, dict) else None
            if isinstance(kh, str) and len(kh) == KEY_BYTES * 2:
                try:
                    k = bytes.fromhex(kh)
                except ValueError:
                    continue
                if key_fingerprint(k) == e.get("fingerprint"):
                    out.append(e)
        return out
    except Exception as e:                                    # corrupt file must not break key paths
        logger.warning("model key history unreadable (reason=key_history_unreadable: %s) — ignored",
                       type(e).__name__)
        return []


def key_history() -> list[bytes]:
    """Keys of earlier licences of this install (oldest first). Never raises."""
    return [bytes.fromhex(e["key_hex"]) for e in _read_key_history_entries()]


def _remember_key(key: bytes, source: str) -> bool:
    """Record a licence-derived key in the history (idempotent, atomic, 0600).
    Returns True when the file was written. Never raises; logs the fingerprint only."""
    try:
        fp = key_fingerprint(key)
        entries = _read_key_history_entries()
        if any(e.get("fingerprint") == fp for e in entries):
            return False
        entries.append({"fingerprint": fp, "key_hex": bytes(key).hex(), "source": source,
                        "first_seen": time.strftime("%Y-%m-%dT%H:%M:%S")})
        entries = entries[-_KEY_HISTORY_MAX:]
        path = _key_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "keys": entries}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(str(path), 0o600)
        except OSError:
            pass
        logger.info("model key history: remembered %s key %s", source, fp)
        return True
    except Exception as e:
        logger.warning("model key history: could not record key (reason=key_history_write_failed: %s: %s)",
                       type(e).__name__, e)
        return False


def key_fingerprint(key: Optional[bytes] = None) -> str:
    """sha256_12 of the key — safe to print/log to confirm two sides agree."""
    return hashlib.sha256(b"fp:" + (key if key is not None else get_model_key())).hexdigest()[:12]


# ============================================================================
# BYTES
# ============================================================================

def _aad(name: str) -> bytes:
    return _AAD_PREFIX + name.encode("utf-8")


def encrypt_bytes(plaintext: bytes, key: bytes, name: str = "") -> bytes:
    """nonce || ciphertext||tag. `name` (the weight file's basename) is bound in
    as associated data so a ciphertext cannot be swapped between files."""
    nonce = os.urandom(NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, _aad(name))


def decrypt_bytes(blob: bytes, key: bytes, name: str = "") -> bytes:
    """Inverse of encrypt_bytes. Raises cryptography.exceptions.InvalidTag on a
    wrong key or any modified byte."""
    if len(blob) < NONCE_BYTES + 16:
        raise ValueError("ciphertext too short")
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    return AESGCM(key).decrypt(nonce, ct, _aad(name))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================================
# FILES
# ============================================================================

def meta_path_for(enc_path: Path) -> Path:
    return enc_path.with_name(enc_path.name[: -len(ENC_SUFFIX)] + META_SUFFIX)


def plaintext_name(enc_path: Path) -> str:
    return enc_path.name[: -len(ENC_SUFFIX)] if enc_path.name.endswith(ENC_SUFFIX) else enc_path.name


def read_meta(enc_path: Path) -> dict:
    """The sidecar of an encrypted file ({plaintext_sha256, size, created_at, alg})."""
    return json.loads(meta_path_for(Path(enc_path)).read_text())


def _meta_for(name: str, plaintext_sha: str, size: int, key: bytes, kdf: Optional[str] = None,
              **extra) -> dict:
    meta = {
        "name": name,
        "plaintext_sha256": plaintext_sha,
        "size": size,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alg": ALG,
        "kdf": kdf or ("raw-key" if os.environ.get("SENTINEL_MODEL_KEY_HEX") else KDF),
        "nonce_bytes": NONCE_BYTES,
        "key_fingerprint": key_fingerprint(key),
    }
    meta.update(extra)
    return meta


def encrypt_file(path: Path, key: Optional[bytes] = None, remove_plaintext: bool = True) -> Path:
    """<path> → <path>.enc + <path>.enc.meta.json; removes the plaintext unless told not to."""
    path = Path(path)
    key = key if key is not None else get_model_key()
    plaintext = path.read_bytes()
    enc_path = path.with_name(path.name + ENC_SUFFIX)
    enc_path.write_bytes(encrypt_bytes(plaintext, key, path.name))
    meta = _meta_for(path.name, sha256_bytes(plaintext), len(plaintext), key)
    meta_path_for(enc_path).write_text(json.dumps(meta, indent=2))
    if remove_plaintext:
        path.unlink()
    return enc_path


def decrypt_file_to_bytes(enc_path: Path, key: Optional[bytes] = None, verify: bool = True) -> bytes:
    """Decrypt into memory (never onto disk). With verify=True the result is
    also checked against the sidecar's plaintext_sha256 when a sidecar exists."""
    enc_path = Path(enc_path)
    key = key if key is not None else get_model_key()
    plaintext = decrypt_bytes(enc_path.read_bytes(), key, plaintext_name(enc_path))
    if verify and meta_path_for(enc_path).exists():
        expected = read_meta(enc_path).get("plaintext_sha256")
        if expected and sha256_bytes(plaintext) != expected:
            raise ValueError(f"{enc_path.name}: decrypted bytes do not match the sidecar plaintext_sha256")
    return plaintext


def decrypt_file(enc_path: Path, key: Optional[bytes] = None, remove_encrypted: bool = True) -> Path:
    """<path>.enc → <path> on disk (the CLI 'decrypt' step for a vendor machine)."""
    enc_path = Path(enc_path)
    out = enc_path.with_name(plaintext_name(enc_path))
    out.write_bytes(decrypt_file_to_bytes(enc_path, key))
    if remove_encrypted:
        enc_path.unlink()
        mp = meta_path_for(enc_path)
        if mp.exists():
            mp.unlink()
    return out


# ============================================================================
# MODEL DIRECTORIES
# ============================================================================

def encrypted_weight_files(model_dir: Path) -> list[Path]:
    d = Path(model_dir)
    return [d / (n + ENC_SUFFIX) for n in WEIGHT_FILES if (d / (n + ENC_SUFFIX)).exists()]


def plaintext_weight_files(model_dir: Path) -> list[Path]:
    d = Path(model_dir)
    return [d / n for n in WEIGHT_FILES if (d / n).exists()]


def is_encrypted_dir(model_dir: Path) -> bool:
    """True when the dir has encrypted weights and NO plaintext weights."""
    return bool(encrypted_weight_files(model_dir)) and not plaintext_weight_files(model_dir)


def plaintext_sha256_of_dir(model_dir: Path) -> Optional[str]:
    """sha256 of the (first) weight file as PLAINTEXT — hashed directly when the
    plaintext is on disk, read from the sidecar when only the .enc is. Stable
    across installs and keys; this is what model_identity reports."""
    for fp in plaintext_weight_files(model_dir):
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    for enc in encrypted_weight_files(model_dir):
        try:
            return read_meta(enc).get("plaintext_sha256")
        except Exception:
            continue
    return None


def encrypt_model_dir(model_dir: Path, key: Optional[bytes] = None) -> list[Path]:
    """Encrypt every known weight file in the dir in place (plaintext removed)."""
    key = key if key is not None else get_model_key()
    done = []
    for fp in plaintext_weight_files(model_dir):
        done.append(encrypt_file(fp, key))
    return done


def decrypt_model_dir(model_dir: Path, key: Optional[bytes] = None) -> list[Path]:
    key = key if key is not None else get_model_key()
    return [decrypt_file(enc, key) for enc in encrypted_weight_files(model_dir)]


# ============================================================================
# AT-REST ENCRYPTION OF A WHOLE models/ TREE (contract D2)
# ============================================================================

AT_REST_ENV = "SENTINEL_ENCRYPT_MODELS_AT_REST"
PROPRIETARY_DIR_SUFFIX = "_finetuned"
# backups of a previous release and gate candidates are never touched in place
_AT_REST_SKIP_MARKERS = ("_prev_", "candidate")
# the only weight file the registry can load from memory (model_registry.load_hf_classifier_dir)
AT_REST_WEIGHT = "model.safetensors"
_TMP_TAG = ".at-rest-tmp"


def at_rest_encryption_policy(models_dir: Optional[Path] = None) -> tuple[bool, str]:
    """(enabled, reason) for encrypting models at rest in THIS process (D2).

    Enabled only with SENTINEL_ENCRYPT_MODELS_AT_REST=1 (set by the Electron
    shell only when app.isPackaged), never with SENTINEL_DEV_INSECURE=1 and
    never for a models dir inside a macOS .app bundle (that would break the
    code signature). reason ∈ enabled | not_requested | dev_insecure |
    macos_app_bundle."""
    if os.environ.get(AT_REST_ENV, "").strip() != "1":
        return False, "not_requested"
    if os.environ.get("SENTINEL_DEV_INSECURE", "").strip() == "1":
        return False, "dev_insecure"
    if models_dir is not None and sys.platform == "darwin":
        try:
            parts = Path(models_dir).resolve().parts
        except OSError:
            parts = Path(models_dir).parts
        if any(p.endswith(".app") for p in parts):
            return False, "macos_app_bundle"
    return True, "enabled"


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _sha256_path(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_encrypted_file(enc_path: Path, key: bytes, name: str, expected_sha: str) -> None:
    """Re-read the ciphertext FROM DISK, decrypt it and compare the plaintext
    sha256. Raises ValueError('verify_mismatch …') / InvalidTag on failure."""
    plain = decrypt_bytes(Path(enc_path).read_bytes(), key, name)
    got = sha256_bytes(plain)
    del plain
    if got != expected_sha:
        raise ValueError(f"verify_mismatch: decrypted {name} sha256 {got[:12]}… != plaintext {expected_sha[:12]}…")


def _write_encrypted_atomically(plain_path: Path, key: bytes, kdf: Optional[str] = None,
                                **meta_extra) -> tuple[Path, Path, str]:
    """Encrypt plain_path to <name>.enc via temp files, verify the temp
    ciphertext by decrypting it back, then rename temp → final (ciphertext
    first, sidecar second). Leaves the plaintext in place. Returns
    (enc_path, meta_path, plaintext_sha). Temp files are removed on failure."""
    name = plain_path.name
    enc_path = plain_path.with_name(name + ENC_SUFFIX)
    meta_path = meta_path_for(enc_path)
    tmp_enc = plain_path.with_name(f"{name}{ENC_SUFFIX}{_TMP_TAG}-{os.getpid()}")
    tmp_meta = plain_path.with_name(f"{name}{META_SUFFIX}{_TMP_TAG}-{os.getpid()}")
    try:
        plaintext = plain_path.read_bytes()
        plain_sha = sha256_bytes(plaintext)
        size = len(plaintext)
        blob = encrypt_bytes(plaintext, key, name)
        del plaintext
        _fsync_write(tmp_enc, blob)
        del blob
        _verify_encrypted_file(tmp_enc, key, name, plain_sha)
        meta = _meta_for(name, plain_sha, size, key, kdf, **meta_extra)
        _fsync_write(tmp_meta, json.dumps(meta, indent=2).encode("utf-8"))
        os.replace(tmp_enc, enc_path)
        try:
            os.replace(tmp_meta, meta_path)
        except BaseException:
            try:                    # never leave a ciphertext without its sidecar
                enc_path.unlink()
            except OSError:
                pass
            raise
        return enc_path, meta_path, plain_sha
    finally:
        for t in (tmp_enc, tmp_meta):
            try:
                if t.exists():
                    t.unlink()
            except OSError:
                pass


def _restore_plaintext(enc_path: Path, plain_path: Path, key: bytes, expected_sha: str) -> None:
    """Put plain_path back from a verified ciphertext (atomic temp + rename)."""
    plain = decrypt_bytes(enc_path.read_bytes(), key, plain_path.name)
    if sha256_bytes(plain) != expected_sha:
        raise ValueError(f"restore of {plain_path.name}: sha256 mismatch")
    tmp = plain_path.with_name(f"{plain_path.name}{_TMP_TAG}-restore-{os.getpid()}")
    _fsync_write(tmp, plain)
    os.replace(tmp, plain_path)


def _encrypt_dir_at_rest(d: Path, key: bytes) -> None:
    """Encrypt d/model.safetensors in place with verify-before-delete.

    On ANY exception the dir is put back the way it was (plaintext present,
    the .enc/.meta written by this call removed) and the exception re-raised
    with a reason code prefix."""
    plain = d / AT_REST_WEIGHT
    enc = plain.with_name(plain.name + ENC_SUFFIX)
    meta = meta_path_for(enc)
    # A stale .enc/.meta from a run that crashed between rename and unlink is
    # simply overwritten: the plaintext on disk is the source of truth.
    created = False
    plain_sha: Optional[str] = None
    plaintext_removed = False
    try:
        try:
            enc_path, _meta_path, plain_sha = _write_encrypted_atomically(plain, key)
            created = True
        except InvalidTag as e:
            raise ValueError("verify_failed: ciphertext did not decrypt with the same key") from e
        # sanity: the sidecar we just wrote must agree with the bytes on disk
        if read_meta(enc_path).get("plaintext_sha256") != plain_sha:
            raise ValueError("verify_mismatch: sidecar plaintext_sha256 disagrees")
        plain.unlink()
        plaintext_removed = True
    except BaseException as e:
        rollback_problems = []
        if plaintext_removed and not plain.exists() and plain_sha:
            try:
                _restore_plaintext(enc, plain, key, plain_sha)
            except Exception as re_err:            # pragma: no cover - double fault
                rollback_problems.append(f"plaintext restore failed: {re_err}")
        if created and plain.exists():
            for f in (enc, meta):
                try:
                    if f.exists():
                        f.unlink()
                except OSError as un:              # pragma: no cover - double fault
                    rollback_problems.append(f"cannot remove {f.name}: {un}")
        if rollback_problems:
            raise RuntimeError("rollback_incomplete: " + "; ".join(rollback_problems)) from e
        raise


def encrypt_dir_at_rest(model_dir: Path, key: Optional[bytes] = None) -> Path:
    """Verify-before-delete encryption of ONE classifier dir's model.safetensors
    (the same procedure encrypt_proprietary_models_at_rest uses per dir; e.g. for
    a kept previous release after a model import). Returns the .enc path.
    Raises ValueError/RuntimeError whose message starts with a reason code
    (verify_mismatch | verify_failed | rollback_incomplete) or OSError; on
    failure the dir is exactly as before (plaintext kept, no partial files)."""
    d = Path(model_dir)
    if not (d / AT_REST_WEIGHT).is_file():
        raise ValueError(f"not_plaintext: {d} has no plaintext {AT_REST_WEIGHT}")
    _encrypt_dir_at_rest(d, key if key is not None else get_model_key())
    return d / (AT_REST_WEIGHT + ENC_SUFFIX)


def _rekey_dir_if_needed(d: Path, key: bytes,
                         old_keys: Union[None, bytes, Iterable[bytes]]) -> bool:
    """d holds only ciphertext. When its sidecar says it was encrypted with one
    of old_keys (the install-secret key, or the key of an earlier licence of
    this install — renewal / expiry) but the current key is different,
    re-encrypt it with the current key, verify, swap atomically. Returns True
    when re-keyed. Raises ValueError('key_mismatch …') when the weights match
    no known key."""
    enc = d / (AT_REST_WEIGHT + ENC_SUFFIX)
    if not enc.exists():
        return False
    meta = read_meta(enc)
    fp_on_disk = meta.get("key_fingerprint")
    if not fp_on_disk or fp_on_disk == key_fingerprint(key):
        return False
    if old_keys is None:
        candidates: list[bytes] = []
    elif isinstance(old_keys, (bytes, bytearray)):
        candidates = [bytes(old_keys)]
    else:
        candidates = [bytes(k) for k in old_keys if k]
    old_key = next((k for k in candidates if k != key and key_fingerprint(k) == fp_on_disk), None)
    if old_key is None:
        raise ValueError(f"key_mismatch: {enc.name} was encrypted with key {fp_on_disk}, "
                         f"this install resolves key {key_fingerprint(key)} ({_key_source_cache or '?'}) "
                         f"and no earlier key of this install matches")
    expected = meta.get("plaintext_sha256")
    plain = decrypt_bytes(enc.read_bytes(), old_key, AT_REST_WEIGHT)
    if expected and sha256_bytes(plain) != expected:
        raise ValueError("verify_mismatch: old ciphertext does not match its sidecar")
    tmp_enc = enc.with_name(enc.name + f"{_TMP_TAG}-rekey-{os.getpid()}")
    tmp_meta = meta_path_for(enc).with_name(meta_path_for(enc).name + f"{_TMP_TAG}-rekey-{os.getpid()}")
    try:
        blob = encrypt_bytes(plain, key, AT_REST_WEIGHT)
        new_meta = _meta_for(AT_REST_WEIGHT, sha256_bytes(plain), len(plain), key, KDF,
                             rekeyed_from=fp_on_disk)
        del plain
        _fsync_write(tmp_enc, blob)
        del blob
        _verify_encrypted_file(tmp_enc, key, AT_REST_WEIGHT, new_meta["plaintext_sha256"])
        _fsync_write(tmp_meta, json.dumps(new_meta, indent=2).encode("utf-8"))
        os.replace(tmp_enc, enc)
        os.replace(tmp_meta, meta_path_for(enc))
        return True
    finally:
        for t in (tmp_enc, tmp_meta):
            try:
                if t.exists():
                    t.unlink()
            except OSError:
                pass


def encrypt_proprietary_models_at_rest(models_dir: Path, key: Optional[bytes] = None) -> dict:
    """Encrypt every proprietary fine-tune under models_dir in place (D2).

    A dir qualifies when its name ends with '_finetuned', it is not a backup
    ('*_prev_*') or gate candidate ('*candidate*'), it is a real directory (not
    a symlink) holding config.json and plaintext model.safetensors — the only
    weight file the registry decrypts into memory. Per dir: encrypt to temp
    files, decrypt the temp ciphertext back and compare sha256 with the
    plaintext, write the sidecar, rename atomically, and only then delete the
    plaintext. Any exception restores the plaintext and removes partial files.
    A dir that is already encrypted with the install-secret key or with the
    key of an EARLIER licence of this install (key_history: renewal, expiry)
    while the current key differs is re-keyed (same verify-then-swap).

    Returns {'encrypted': [...], 'already': [...], 'skipped': [...],
             'rekeyed': [...], 'failed': {dir: '<reason_code>: detail'}}
    with dir NAMES. Never raises. Logs one loud line per outcome."""
    report: dict = {"encrypted": [], "already": [], "skipped": [], "rekeyed": [], "failed": {}}
    root = Path(models_dir)
    try:
        if not root.is_dir():
            logger.warning("model at-rest encryption: models dir %s does not exist — nothing to do", root)
            return report
        key = key if key is not None else get_model_key()
        if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes")
        key = bytes(key)
        entries = sorted(root.iterdir())
    except Exception as e:
        report["failed"][str(root)] = f"key_unavailable: {type(e).__name__}: {e}" \
            if not isinstance(e, OSError) else f"io_error: {e}"
        logger.error("MODEL AT-REST ENCRYPTION FAILED for %s: %s", root, report["failed"][str(root)])
        return report

    # Re-keying only when the key in use IS the key this install resolves from
    # a licence or its install secret (never for an explicit / env key): old
    # candidates are the install-secret key and the keys of earlier licences
    # (renewal, expiry). Ciphertext under no known key is reported as
    # key_mismatch and left untouched.
    old_keys: list[bytes] = []
    try:
        if key == get_model_key() and _key_source_cache in (KEY_SOURCE_LICENSE, KEY_SOURCE_INSTALL):
            install_key = _install_secret_key_if_present()
            if install_key is not None and install_key != key:
                old_keys.append(install_key)
            old_keys.extend(k for k in key_history() if k != key and k not in old_keys)
    except Exception:
        old_keys = []
    for d in entries:
        name = d.name
        try:
            if any(m in name for m in _AT_REST_SKIP_MARKERS):
                if d.is_dir():
                    report["skipped"].append(name)
                continue
            if not name.endswith(PROPRIETARY_DIR_SUFFIX) or not d.is_dir():
                continue
            if d.is_symlink():
                report["skipped"].append(name)
                logger.warning("model at-rest encryption: %s is a symlink — skipped", name)
                continue
            plain_files = plaintext_weight_files(d)
            unsupported = [p.name for p in plain_files if p.name != AT_REST_WEIGHT]
            if unsupported:
                report["skipped"].append(name)
                logger.warning("model at-rest encryption: %s holds %s which the registry cannot load "
                               "encrypted (reason=unsupported_weight_format) — left as is", name, unsupported)
                continue
            if not plain_files:
                if encrypted_weight_files(d):
                    if _rekey_dir_if_needed(d, key, old_keys):
                        report["rekeyed"].append(name)
                        logger.warning("model at-rest encryption: %s RE-KEYED to the current key (%s)",
                                       name, key_fingerprint(key))
                    report["already"].append(name)
                else:
                    report["skipped"].append(name)
                continue
            if not (d / "config.json").exists():
                report["skipped"].append(name)
                logger.warning("model at-rest encryption: %s has no config.json (reason=not_a_model_dir) "
                               "— left as is", name)
                continue
            _encrypt_dir_at_rest(d, key)
            report["encrypted"].append(name)
            logger.warning("model at-rest encryption: %s ENCRYPTED (key %s, plaintext removed after verify)",
                           name, key_fingerprint(key))
        except Exception as e:
            msg = str(e)
            code_ok = bool(re.match(r"^[a-z_]+: ", msg))
            report["failed"][name] = msg if code_ok else (
                f"io_error: {e}" if isinstance(e, OSError) else f"encrypt_failed: {type(e).__name__}: {e}")
            kept = "plaintext kept" if (d / AT_REST_WEIGHT).is_file() else \
                "ciphertext left untouched (no plaintext on disk)"
            logger.error("MODEL AT-REST ENCRYPTION FAILED for %s — %s: %s", name, kept, report["failed"][name])
    logger.warning("model at-rest encryption summary: encrypted=%s already=%s rekeyed=%s skipped=%s failed=%s",
                   report["encrypted"], report["already"], report["rekeyed"], report["skipped"],
                   sorted(report["failed"]))
    return report


# ============================================================================
# PER-CLINIC ENCRYPTED BUNDLE (contract D3 — vendor side)
# ============================================================================

# == scripts/eval_study_level.MANIFEST_FILES (tests/test_model_crypto.py pins the parity)
MANIFEST_FILES = ("config.json", "preprocessor_config.json",
                  "model.safetensors", "pytorch_model.bin", "model.pt")
BUNDLE_ROOT_NAME = "brain_triage_finetuned"
BUNDLE_COPY_FILES = ("config.json", "preprocessor_config.json", "model.safetensors",
                     "study_level_eval.json", "training_metrics.json")
BUNDLE_REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
_FP8_RE = re.compile(r"^[0-9a-f]{8}$")


class BundleExportError(ValueError):
    """export_encrypted_bundle refused or failed; .code is a stable reason code
    (source_missing_files | source_not_plaintext | manifest_mismatch |
    bad_fingerprint | bad_key | verify_failed | zip_verify_failed)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _sidecar_plaintext_sha(model_dir: Path, name: str) -> Optional[str]:
    """== scripts/eval_study_level._sidecar_plaintext_sha."""
    enc = model_dir / (name + ENC_SUFFIX)
    meta = model_dir / (name + META_SUFFIX)
    if enc.exists() and meta.exists():
        try:
            return json.loads(meta.read_text()).get("plaintext_sha256")
        except Exception:
            return None
    return None


def manifest_files_for_dir(model_dir: Path) -> dict:
    """The `files` block scripts/eval_study_level.write_manifest writes for
    model_dir (same rule, re-implemented so src/ never imports scripts/):
    plaintext files are hashed; an encrypted weight is pinned twice — its
    PLAINTEXT sha (from the sidecar) under its plain name and its ciphertext
    sha under '<name>.enc'."""
    model_dir = Path(model_dir)
    files: dict = {}
    for name in MANIFEST_FILES:
        if (model_dir / name).exists():
            files[name] = _sha256_path(model_dir / name)
        elif _sidecar_plaintext_sha(model_dir, name):
            files[name] = _sidecar_plaintext_sha(model_dir, name)
            files[name + ENC_SUFFIX] = _sha256_path(model_dir / (name + ENC_SUFFIX))
    return files


def export_encrypted_bundle(model_dir: Path, key: bytes, out_zip: Path, fingerprint8: str,
                            *, root_name: str = BUNDLE_ROOT_NAME) -> Path:
    """Build the per-clinic model zip (layout in docs/dev/finish_line_contract.md
    'POST /models/import') from a TEMP COPY of model_dir — the source dir is
    never written to.

    Steps: copy config/preprocessor/model.safetensors/study_level_eval/
    training_metrics into a temp dir; when model_dir has a MANIFEST.json, every
    pin it holds for a copied file must match the copy (a corrupted release is
    never shipped); encrypt model.safetensors with the GIVEN key (temp file,
    decrypt-back verify, sidecar records key fingerprint + fingerprint8); write
    MANIFEST.json pinning the plaintext AND the ciphertext sha; zip it as
    <root_name>/<file> (ciphertext stored, json deflated) via a temp file +
    atomic rename; re-open the zip and check the ciphertext pin.

    Raises BundleExportError(code) on any refusal; the temp copy is always removed."""
    model_dir = Path(model_dir)
    out_zip = Path(out_zip)
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
        raise BundleExportError("bad_key", f"key must be {KEY_BYTES} bytes")
    key = bytes(key)
    if not isinstance(fingerprint8, str) or not _FP8_RE.match(fingerprint8):
        raise BundleExportError("bad_fingerprint", "fingerprint8 must be 8 lowercase hex chars")
    missing = [n for n in BUNDLE_REQUIRED_FILES if not (model_dir / n).is_file()]
    if missing:
        if "model.safetensors" in missing and (model_dir / ("model.safetensors" + ENC_SUFFIX)).exists():
            raise BundleExportError("source_not_plaintext",
                                    f"{model_dir} holds only encrypted weights — export needs the plaintext release")
        raise BundleExportError("source_missing_files", f"{model_dir}: missing {missing}")
    source_pins: dict = {}
    if (model_dir / "MANIFEST.json").is_file():
        try:
            source_pins = json.loads((model_dir / "MANIFEST.json").read_text()).get("files") or {}
        except Exception as e:
            raise BundleExportError("manifest_mismatch", f"unreadable {model_dir / 'MANIFEST.json'}: {e}") from e

    work = Path(tempfile.mkdtemp(prefix="sentinel_bundle_"))
    try:
        stage = work / root_name
        stage.mkdir()
        for n in BUNDLE_COPY_FILES:
            if (model_dir / n).is_file():
                shutil.copyfile(model_dir / n, stage / n)
        problems = []
        for n, pin in source_pins.items():
            if (stage / n).exists():
                got = _sha256_path(stage / n)
                if got != pin:
                    problems.append(f"{n}: sha256 {got[:12]}… != pinned {str(pin)[:12]}…")
        if problems:
            raise BundleExportError("manifest_mismatch", "; ".join(problems))
        plain = stage / "model.safetensors"
        try:
            _write_encrypted_atomically(plain, key, KDF, bound_machine_fp8=fingerprint8)
        except (InvalidTag, ValueError) as e:
            raise BundleExportError("verify_failed", str(e)) from e
        plain.unlink()
        files = manifest_files_for_dir(stage)
        manifest = {"model_dir": root_name, "created_at": datetime.now().isoformat(timespec="seconds"),
                    "algorithm": "sha256", "files": files,
                    "encrypted_for_machine_fp8": fingerprint8, "key_fingerprint": key_fingerprint(key)}
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

        out_zip.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = out_zip.with_name(out_zip.name + ".tmp")
        try:
            with zipfile.ZipFile(tmp_zip, "w") as zf:
                for f in sorted(stage.iterdir()):
                    comp = zipfile.ZIP_STORED if f.name.endswith(ENC_SUFFIX) else zipfile.ZIP_DEFLATED
                    zf.write(f, f"{root_name}/{f.name}", compress_type=comp)
            with zipfile.ZipFile(tmp_zip) as zf:
                bad = zf.testzip()
                if bad:
                    raise BundleExportError("zip_verify_failed", f"corrupt member {bad}")
                h = hashlib.sha256()
                with zf.open(f"{root_name}/model.safetensors{ENC_SUFFIX}") as m:
                    for chunk in iter(lambda: m.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != files.get("model.safetensors" + ENC_SUFFIX):
                    raise BundleExportError("zip_verify_failed", "ciphertext in the zip does not match the MANIFEST pin")
            os.replace(tmp_zip, out_zip)
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()
        logger.info("exported encrypted bundle %s (identity %s, machine %s, key %s)", out_zip.name,
                    str(files.get("model.safetensors"))[:12], fingerprint8, key_fingerprint(key))
        return out_zip
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ============================================================================
# CLI
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel model-weight encryption (AES-256-GCM)")
    sub = ap.add_subparsers(dest="cmd")
    p_enc = sub.add_parser("encrypt", help="encrypt model.safetensors/pytorch_model.bin/model.pt in <model_dir>; plaintext is REMOVED")
    p_enc.add_argument("model_dir")
    p_dec = sub.add_parser("decrypt", help="restore plaintext weights in <model_dir> from the .enc files")
    p_dec.add_argument("model_dir")
    sub.add_parser("keyinfo", help="print the key fingerprint (never the key) and its source")
    args = ap.parse_args(argv)

    if args.cmd == "keyinfo":
        try:
            key = get_model_key(refresh=True)
        except ValueError as e:
            print(f"key unavailable: {e}", file=sys.stderr)
            return 1
        src = key_source()
        print(f"key source:      {src} — {_KEY_SOURCE_LABELS.get(src, src)}")
        print(f"key fingerprint: {key_fingerprint(key)}")
        return 0
    if args.cmd in ("encrypt", "decrypt"):
        d = Path(args.model_dir)
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 1
        if args.cmd == "encrypt":
            if not plaintext_weight_files(d):
                print(f"no plaintext weight file ({', '.join(WEIGHT_FILES)}) in {d}", file=sys.stderr)
                return 1
            for out in encrypt_model_dir(d):
                meta = read_meta(out)
                print(f"encrypted {out.name}  ({meta['size'] / 1e6:.1f} MB, plaintext sha256 {meta['plaintext_sha256'][:12]}…, "
                      f"key {meta['key_fingerprint']})")
        else:
            if not encrypted_weight_files(d):
                print(f"no .enc weight file in {d}", file=sys.stderr)
                return 1
            try:
                for out in decrypt_model_dir(d):
                    print(f"decrypted {out.name}")
            except InvalidTag:
                print("decryption FAILED: wrong key (or the file was modified)", file=sys.stderr)
                return 2
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
