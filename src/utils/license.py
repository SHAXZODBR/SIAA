"""
================================================================================
  SENTINEL MEDICAL AI — LICENSE / ANTI-PIRACY SYSTEM
================================================================================

  Goal: a clinic that buys Sentinel can install it on the machines they
  paid for. They cannot copy the install to another clinic, and a
  pirate can't crack-and-redistribute the binary.

  Mechanism:
    1. We hold an RSA-2048 keypair. The PRIVATE key never leaves our
       servers. The PUBLIC key is compiled into the app.
    2. When a clinic activates, they send us:
         - their machine fingerprint (hardware-bound hash)
         - their license tier + expiry + customer name
       We sign that JSON blob with our private key and ship the
       signed `license.dat` back to them.
    3. The app loads license.dat at startup, verifies the RSA signature
       with the public key, and refuses to run if:
         - signature invalid
         - machine fingerprint doesn't match this hardware
         - expiry has passed
         - customer name was tampered with
    4. The fingerprint mixes MAC addresses, CPU model, hostname, and
       (on Linux/Mac) /etc/machine-id, so moving the install to another
       PC invalidates the license.

  HOW TO GENERATE THE VENDOR KEYPAIR (once):
    python -m src.utils.license gen-keys [--out-dir <dir>]
    Writes vendor_keys/vendor_private_key.pem (0600, git-ignored) and
    vendor_keys/vendor_public_key.pem; refuses to overwrite an existing
    private key.

  HOW TO ISSUE A LICENSE (for us, the vendor):
    python -m src.utils.license issue \
        --customer "Tashkent Hospital #1" \
        --machine-id <fingerprint-from-clinic> \
        --tier pro \
        --expires 2027-12-31 \
        --private-key vendor_keys/vendor_private_key.pem \
        --output license.dat [--with-model-key]

  PER-CLINIC MODEL KEY (docs/dev/finish_line_contract.md D3):
    --with-model-key puts a fresh random 32-byte `model_key_secret` (64 hex)
    INSIDE the signed payload. The model weights delivered to that clinic are
    encrypted with HKDF(secret || machine_id) (src/utils/model_crypto.derive_key),
    so the vendor can encrypt offline and the clinic never sees plaintext.
    The secret is never printed; only the derived key's fingerprint is.
    packaging/provision_clinic.py does license + encrypted model bundle in one
    step — use it for real deliveries (a bare `issue --with-model-key` renewal
    changes the key and orphans the bundle the clinic already has).
    license_model_key_secret() returns the secret only for a license that
    verifies on THIS machine. A license carrying model_key_secret needs app
    >= 1.1.0 (older verifiers reject the unknown payload field).

  HOW THE CLIENT GETS THEIR FINGERPRINT:
    python -m src.utils.license fingerprint
================================================================================
"""

from __future__ import annotations
import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# model_key_secret: 32 random bytes as 64 lowercase hex chars (contract D3).
MODEL_KEY_SECRET_HEX_LEN = 64
_MODEL_KEY_SECRET_RE = re.compile(r'^[0-9a-f]{64}$')


# ============================================================================
# RSA — uses cryptography (already in requirements.txt)
# ============================================================================

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


# ----- VENDOR PUBLIC KEY ----------------------------------------------------
# We ship this PUBLIC key in the binary. It's NOT a secret.
# The matching PRIVATE key stays on the vendor's signing server only.
#
# To rotate keys: regenerate via `python -m src.utils.license gen-keys
# [--out-dir <dir>]` (writes into vendor_keys/ by default and refuses to
# overwrite an existing private key) and rebuild the app with the new public
# key embedded.
SENTINEL_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyKGs1c1GWwy6bcJHzeUd
f/WEGzXzjF+ATwBnzme35aywXmwp/o/mHt4CwMSoYUsUipxqUnmIVLDDq8HxGLiT
Ocxxi9ZdotgyNqydXkIvTI1t8RezEQrwny/zbbqqFtedLFnAElT+p2zShPqCMSlC
VFtqwAdoWkAR+5U1/7F1/2QkbMk8AYNaBN9YL5apbL5VUPSeWVLcdkgRCUsWeaU7
6venO4Szl58/AOOK6IV6a81429bMyolQbrzg0P2HDuzd7ALz2fml3d8kuem8emmM
qCdX9uUOVhWL6HXVC8mZafBvj6QQsYmrFIDB49/tlWhwPIkpL0jFUoYTCxQgi4ZQ
QwIDAQAB
-----END PUBLIC KEY-----"""


# ============================================================================
# MACHINE FINGERPRINT
# ============================================================================

def get_mac_addresses() -> list[str]:
    """Return the primary MAC address on this host.

    We deliberately use ONLY uuid.getnode() (the primary hardware MAC) and
    skip transient virtual interfaces. This keeps the fingerprint stable
    across reboots — VPN interfaces, Docker bridges, USB-C dongles all flap
    in/out and would invalidate licenses unfairly.
    """
    macs = []
    try:
        primary = uuid.getnode()
        # The locally-administered bit is bit 1 of the FIRST octet (the high
        # octet here = bits 40-47). uuid.getnode() sets it when it had to invent
        # a random MAC (no real NIC). Reject those so the fingerprint is a real,
        # stable hardware address.
        first_octet = (primary >> 40) & 0xff
        locally_administered = bool(first_octet & 0x02)
        if primary != 0 and not locally_administered:
            macs.append(':'.join(f'{(primary >> i) & 0xff:02x}' for i in (40, 32, 24, 16, 8, 0)))
    except Exception:
        pass
    return macs


def get_machine_id() -> str:
    """Compute a stable machine fingerprint.

    Mixes:
      - hostname
      - OS/platform/arch
      - CPU model
      - All MAC addresses
      - /etc/machine-id (Linux) or IOPlatformUUID (macOS)
    """
    parts = [
        socket.gethostname(),
        platform.system(),
        platform.machine(),
        platform.processor() or 'cpu-unknown',
        ','.join(get_mac_addresses()),
    ]

    # Linux machine-id
    for path in ('/etc/machine-id', '/var/lib/dbus/machine-id'):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    parts.append(f.read().strip())
                break
            except Exception:
                pass

    # macOS IOPlatformUUID
    if platform.system() == 'Darwin':
        try:
            import subprocess
            r = subprocess.run(
                ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'],
                capture_output=True, text=True, timeout=3,
            )
            for line in r.stdout.splitlines():
                if 'IOPlatformUUID' in line:
                    parts.append(line.split('"')[-2])
                    break
        except Exception:
            pass

    fingerprint = '|'.join(parts)
    return hashlib.sha256(fingerprint.encode()).hexdigest()


# ============================================================================
# LICENSE FILE FORMAT
# ============================================================================

@dataclass
class License:
    customer: str
    machine_id: str
    tier: str            # 'trial' | 'pro' | 'enterprise'
    issued_at: str       # ISO date
    expires_at: str      # ISO date
    features: list[str]  # ['chest', 'brain_2d', 'head_ct', 'mammography', 'cloud']
    seat_count: int = 1
    notes: str = ''
    # Per-clinic model-encryption secret (64 hex), signed with the rest of the
    # payload. Optional: licenses issued before v1.1.0 do not carry it and
    # verify exactly as before. repr=False keeps it out of logs / tracebacks.
    model_key_secret: Optional[str] = field(default=None, repr=False)

    def to_payload(self) -> dict:
        payload = {
            'customer': self.customer,
            'machine_id': self.machine_id,
            'tier': self.tier,
            'issued_at': self.issued_at,
            'expires_at': self.expires_at,
            'features': sorted(self.features),
            'seat_count': self.seat_count,
            'notes': self.notes,
        }
        # Only present when set, so a license without a model key serializes
        # byte-for-byte as it did before the field existed.
        if self.model_key_secret is not None:
            payload['model_key_secret'] = self.model_key_secret
        return payload


def is_valid_model_key_secret(value: object) -> bool:
    """True for a well-formed model_key_secret: exactly 64 lowercase hex chars."""
    return isinstance(value, str) and bool(_MODEL_KEY_SECRET_RE.match(value))


def generate_model_key_secret() -> str:
    """A fresh model_key_secret: os.urandom(32) as 64 lowercase hex chars."""
    return os.urandom(MODEL_KEY_SECRET_HEX_LEN // 2).hex()


@dataclass
class LicenseValidation:
    valid: bool
    reason: str = ''
    license: Optional[License] = None
    days_remaining: int = 0


def sign_license(license_obj: License, private_key_pem: bytes) -> str:
    """Sign a license payload with the vendor's private key.

    Returns a base64-encoded license file content combining the JSON
    payload and the RSA-PSS signature. This is what gets written to
    license.dat and shipped to the customer.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError('cryptography not installed')
    if license_obj.model_key_secret is not None and not is_valid_model_key_secret(license_obj.model_key_secret):
        # Refuse at the source: a malformed secret would silently fall back to
        # the install key at the clinic and the delivered bundle would not load.
        raise ValueError('model_key_secret must be 64 lowercase hex characters')

    private_key = serialization.load_pem_private_key(
        private_key_pem, password=None, backend=default_backend(),
    )
    payload_json = json.dumps(license_obj.to_payload(), sort_keys=True).encode()
    signature = private_key.sign(
        payload_json,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    bundle = {
        'payload': base64.b64encode(payload_json).decode(),
        'signature': base64.b64encode(signature).decode(),
        'version': 1,
    }
    return base64.b64encode(json.dumps(bundle).encode()).decode()


def verify_license(license_data: str, current_machine_id: Optional[str] = None) -> LicenseValidation:
    """Verify a license string against the embedded vendor public key.

    Checks:
      1. Signature is valid (proves the file was signed by us).
      2. Machine ID matches this hardware.
      3. Expiry hasn't passed.

    Returns LicenseValidation with `valid=True` only if every check passes.
    """
    if not _CRYPTO_AVAILABLE:
        return LicenseValidation(False, 'cryptography library not installed')

    try:
        bundle = json.loads(base64.b64decode(license_data).decode())
        payload_bytes = base64.b64decode(bundle['payload'])
        signature = base64.b64decode(bundle['signature'])
    except Exception as e:
        return LicenseValidation(False, f'malformed license file: {e}')

    # Verify signature
    try:
        public_key = serialization.load_pem_public_key(
            SENTINEL_PUBLIC_KEY_PEM, backend=default_backend(),
        )
        public_key.verify(
            signature,
            payload_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
    except Exception:
        return LicenseValidation(False, 'invalid signature — license is forged or corrupted')

    # Parse payload
    try:
        payload = json.loads(payload_bytes.decode())
        license_obj = License(**payload)
    except Exception as e:
        return LicenseValidation(False, f'unreadable payload: {e}')

    # Machine binding check
    if current_machine_id is None:
        current_machine_id = get_machine_id()
    if license_obj.machine_id != current_machine_id:
        return LicenseValidation(
            False,
            'license is bound to a different machine — contact vendor for a new license',
            license=license_obj,
        )

    # Expiry check
    try:
        expiry = datetime.date.fromisoformat(license_obj.expires_at)
        today = datetime.date.today()
        days_remaining = (expiry - today).days
        if days_remaining < 0:
            return LicenseValidation(
                False,
                f'license expired on {license_obj.expires_at}',
                license=license_obj,
                days_remaining=days_remaining,
            )
        return LicenseValidation(True, 'OK', license=license_obj, days_remaining=days_remaining)
    except Exception as e:
        return LicenseValidation(False, f'invalid expiry date: {e}', license=license_obj)


def get_license_path() -> Path:
    """Where the license file lives on this machine.

    DATA_DIR/license.dat first (the Electron shell writes there via
    SENTINEL_DATA_DIR=userData); falls back to the legacy per-platform config
    location so existing installs keep working."""
    try:
        from src.utils.paths import LICENSE_PATH
        if LICENSE_PATH.exists():
            return LICENSE_PATH
    except Exception:
        pass
    home = Path.home()
    if platform.system() == 'Windows':
        base = Path(os.environ.get('APPDATA', home))
    elif platform.system() == 'Darwin':
        base = home / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME', home / '.config'))
    return base / 'sentinel-medical-ai' / 'license.dat'


def load_and_verify_local_license() -> LicenseValidation:
    """Read license.dat from the standard path and verify it."""
    path = get_license_path()
    if not path.exists():
        return LicenseValidation(False, f'no license file at {path}')
    try:
        data = path.read_text().strip()
    except Exception as e:
        return LicenseValidation(False, f'cannot read license: {e}')
    return verify_license(data)


def license_model_key_secret(validation: Optional[LicenseValidation] = None) -> Optional[bytes]:
    """The per-clinic model-key secret (32 raw bytes) from the local license.

    Returns the secret ONLY when the local license verifies for THIS machine
    (signature, machine binding, expiry) and carries a well-formed
    `model_key_secret`; otherwise None, so src/utils/model_crypto.get_model_key
    falls back to the install secret. Never raises (a broken license file must
    not take the model key path down) and never logs the secret.

    `validation` lets a caller that already verified the license reuse that
    result instead of re-reading the file.
    """
    try:
        result = validation if validation is not None else load_and_verify_local_license()
    except Exception as e:                                   # pragma: no cover - defensive
        logger.warning('model key: license check failed (%s: %s) — not using a license key',
                       type(e).__name__, e)
        return None
    if not result.valid or result.license is None:
        return None
    secret_hex = result.license.model_key_secret
    if secret_hex is None:
        return None
    if not is_valid_model_key_secret(secret_hex):
        logger.warning('model key: license.dat carries a malformed model_key_secret '
                       '(reason=license_model_key_malformed) — not using it')
        return None
    return bytes.fromhex(secret_hex)


# ============================================================================
# CLI for vendor + customer use
# ============================================================================

# Default home of the vendor keypair: <repo>/vendor_keys/. .gitignore ignores
# vendor_private_key.pem in ANY directory; vendor_keys/vendor_public_key.pem is tracked.
VENDOR_KEYS_DIR = Path(__file__).resolve().parents[2] / 'vendor_keys'
PRIVATE_KEY_NAME = 'vendor_private_key.pem'
PUBLIC_KEY_NAME = 'vendor_public_key.pem'


def _generate_keypair(out_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Generate a fresh vendor signing keypair into `out_dir` (default
    <repo>/vendor_keys/, never the current working directory). Run this ONCE;
    embed the public key into SENTINEL_PUBLIC_KEY_PEM and keep the private key
    in a vault.

    The private key is created exclusively (O_EXCL) and owner-only (0600 where
    the platform supports it). An existing private key is never overwritten:
    raises FileExistsError — losing it would orphan every issued licence.
    Returns (private_key_path, public_key_path)."""
    if not _CRYPTO_AVAILABLE:
        print('cryptography library required: pip install cryptography')
        sys.exit(1)

    out = Path(out_dir) if out_dir is not None else VENDOR_KEYS_DIR
    priv_path = out / PRIVATE_KEY_NAME
    pub_path = out / PUBLIC_KEY_NAME
    if priv_path.exists():
        raise FileExistsError(
            f'{priv_path} already exists — refusing to overwrite the vendor private key '
            f'(it signs every issued licence). Move it to an offline backup first, or pass '
            f'--out-dir <empty folder> to generate a new pair elsewhere.')

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    out.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    try:
        fd = os.open(str(priv_path), flags, 0o600)
    except FileExistsError:
        raise FileExistsError(f'{priv_path} already exists — refusing to overwrite the vendor '
                              f'private key') from None
    with os.fdopen(fd, 'wb') as f:
        f.write(priv_pem)
    try:
        os.chmod(str(priv_path), 0o600)     # no-op beyond read-only bits on Windows
    except OSError:
        pass
    pub_path.write_bytes(pub_pem)
    print(f'✓ Generated {priv_path} (KEEP SECRET)')
    print(f'✓ Generated {pub_path} (EMBED IN APP)')
    print('\nReplace SENTINEL_PUBLIC_KEY_PEM in src/utils/license.py with:')
    print(pub_pem.decode())
    return priv_path, pub_path


def _cli_fingerprint():
    fp = get_machine_id()
    print(f'\nMachine fingerprint:\n  {fp}')
    print(f'\nSend this to the vendor to receive your license.dat.')


def _cli_issue(args):
    if not args.private_key:
        print('--private-key required (path to vendor_private_key.pem)')
        sys.exit(1)
    private_key_pem = Path(args.private_key).read_bytes()

    secret_hex = generate_model_key_secret() if getattr(args, 'with_model_key', False) else None
    license_obj = License(
        customer=args.customer,
        machine_id=args.machine_id,
        tier=args.tier,
        issued_at=datetime.date.today().isoformat(),
        expires_at=args.expires,
        features=args.features.split(',') if args.features else ['chest', 'brain_2d', 'head_ct', 'mammography'],
        seat_count=args.seats,
        notes=args.notes or '',
        model_key_secret=secret_hex,
    )
    license_data = sign_license(license_obj, private_key_pem)
    _write_private_text(Path(args.output), license_data)
    print(f'✓ Wrote {args.output}')
    print(f'   Customer:   {license_obj.customer}')
    print(f'   Tier:       {license_obj.tier}')
    print(f'   Expires:    {license_obj.expires_at}')
    print(f'   Features:   {", ".join(license_obj.features)}')
    print(f'   Seats:      {license_obj.seat_count}')
    if secret_hex is not None:
        # Never the secret itself — only the fingerprint of the key it derives
        # for this machine, which `python -m src.utils.model_crypto keyinfo`
        # prints at the clinic so both sides can confirm they agree.
        from src.utils.model_crypto import derive_key, key_fingerprint
        kfp = key_fingerprint(derive_key(bytes.fromhex(secret_hex), license_obj.machine_id))
        print(f'   Model key:  embedded in the signed payload (key fingerprint {kfp})')


def _write_private_text(path: Path, text: str) -> None:
    """Write `text` to `path` readable by the owner only (0600): a license with
    a model_key_secret is key material until it is delivered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def _cli_verify(args):
    path = Path(args.license_file or get_license_path())
    if not path.exists():
        print(f'✗ License file not found: {path}')
        sys.exit(1)
    data = path.read_text().strip()
    result = verify_license(data)
    if result.valid:
        L = result.license
        print('✓ License VALID')
        print(f'   Customer:        {L.customer}')
        print(f'   Tier:            {L.tier}')
        print(f'   Expires:         {L.expires_at} ({result.days_remaining} days remaining)')
        print(f'   Features:        {", ".join(L.features)}')
        print(f'   Bound to:        {L.machine_id[:16]}…')
    else:
        print(f'✗ License INVALID: {result.reason}')
        if result.license:
            L = result.license
            print(f'   Customer in file: {L.customer}')
            print(f'   Expires:          {L.expires_at}')
        sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Sentinel license tool')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('fingerprint', help='Print this machine\'s fingerprint')
    p_gen = sub.add_parser('gen-keys', help='Generate vendor RSA keypair (run once)')
    p_gen.add_argument('--out-dir', default=None,
                       help='folder for the pair (default: <repo>/vendor_keys/); an existing '
                            'vendor_private_key.pem is never overwritten')

    p_issue = sub.add_parser('issue', help='Sign a new license file')
    p_issue.add_argument('--customer', required=True)
    p_issue.add_argument('--machine-id', required=True)
    p_issue.add_argument('--tier', default='pro', choices=['trial', 'pro', 'enterprise'])
    p_issue.add_argument('--expires', required=True, help='YYYY-MM-DD')
    p_issue.add_argument('--features', default='', help='comma-separated')
    p_issue.add_argument('--seats', type=int, default=1)
    p_issue.add_argument('--notes', default='')
    p_issue.add_argument('--private-key', required=True, help='path to vendor_private_key.pem')
    p_issue.add_argument('--output', default='license.dat')
    p_issue.add_argument('--with-model-key', action='store_true',
                         help='embed a fresh per-clinic model_key_secret in the signed payload '
                              '(never printed); prefer packaging/provision_clinic.py for deliveries')

    p_verify = sub.add_parser('verify', help='Verify a license file')
    p_verify.add_argument('--license-file', help='path to license.dat')

    args = parser.parse_args(argv)
    if args.cmd == 'fingerprint':
        _cli_fingerprint()
    elif args.cmd == 'gen-keys':
        try:
            _generate_keypair(Path(args.out_dir) if args.out_dir else None)
        except FileExistsError as e:
            print(f'✗ REFUSED: {e}')
            sys.exit(1)
    elif args.cmd == 'issue':
        _cli_issue(args)
    elif args.cmd == 'verify':
        _cli_verify(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
