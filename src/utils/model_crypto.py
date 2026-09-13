"""
================================================================================
  SENTINEL MEDICAL AI — ENCRYPTED MODEL WEIGHTS AT REST (IP protection)
================================================================================

  The locally fine-tuned detectors (models/*_finetuned) are the product's IP.
  On a clinic PC they sit on a disk anyone with a screwdriver can image, so the
  weight files are stored encrypted and decrypted straight into memory at load
  time — plaintext never touches the disk of the install.

  Cipher       AES-256-GCM (authenticated: a flipped byte fails to decrypt)
  Key          HKDF-SHA256 over  <per-install secret> || <machine fingerprint>
                 • secret: 32 random bytes at DATA_DIR/model_key.bin (0600),
                   created on first use
                 • machine fingerprint: src.utils.license.get_machine_id()
               → the key is bound to THIS install on THIS machine.
               SENTINEL_MODEL_KEY_HEX=<64 hex> overrides with a raw key (the
               release-build step encrypts with the key it provisions for the
               clinic; derive_key() reproduces a clinic's key from its secret +
               fingerprint).
  File format  <name>.enc            = 12-byte nonce || ciphertext || 16-byte GCM tag
               <name>.enc.meta.json  = {plaintext_sha256, size, created_at, alg, ...}
               The sidecar carries the PLAINTEXT sha256 so model identity
               (model_identity.sha256_12) and MANIFEST.json pins stay stable
               across installs, whatever key encrypted the bytes.

  CLI (never run it on the real models/ tree — work on a copy):
    python -m src.utils.model_crypto encrypt <model_dir>   # weights → .enc, plaintext REMOVED
    python -m src.utils.model_crypto decrypt <model_dir>   # reverse
    python -m src.utils.model_crypto keyinfo               # key fingerprint (not the key)
================================================================================
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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


_key_cache: Optional[bytes] = None


def get_model_key(refresh: bool = False) -> bytes:
    """The AES key for this install: SENTINEL_MODEL_KEY_HEX if set, else derived
    from DATA_DIR/model_key.bin + the machine fingerprint. Cached per process."""
    global _key_cache
    if _key_cache is not None and not refresh:
        return _key_cache
    hex_key = os.environ.get("SENTINEL_MODEL_KEY_HEX", "").strip()
    if hex_key:
        try:
            key = bytes.fromhex(hex_key)
        except ValueError as e:
            raise ValueError("SENTINEL_MODEL_KEY_HEX is not valid hex") from e
        if len(key) != KEY_BYTES:
            raise ValueError(f"SENTINEL_MODEL_KEY_HEX must be {KEY_BYTES} bytes ({KEY_BYTES * 2} hex chars)")
    else:
        from src.utils.license import get_machine_id
        key = derive_key(_read_or_create_install_secret(), get_machine_id())
    _key_cache = key
    return key


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


def encrypt_file(path: Path, key: Optional[bytes] = None, remove_plaintext: bool = True) -> Path:
    """<path> → <path>.enc + <path>.enc.meta.json; removes the plaintext unless told not to."""
    path = Path(path)
    key = key if key is not None else get_model_key()
    plaintext = path.read_bytes()
    enc_path = path.with_name(path.name + ENC_SUFFIX)
    enc_path.write_bytes(encrypt_bytes(plaintext, key, path.name))
    meta = {
        "name": path.name,
        "plaintext_sha256": sha256_bytes(plaintext),
        "size": len(plaintext),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alg": ALG,
        "kdf": "raw-key" if os.environ.get("SENTINEL_MODEL_KEY_HEX") else KDF,
        "nonce_bytes": NONCE_BYTES,
        "key_fingerprint": key_fingerprint(key),
    }
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
        src = "SENTINEL_MODEL_KEY_HEX" if os.environ.get("SENTINEL_MODEL_KEY_HEX") else "install secret + machine fingerprint"
        print(f"key source:      {src}")
        print(f"key fingerprint: {key_fingerprint()}")
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
