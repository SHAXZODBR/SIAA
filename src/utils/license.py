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

  HOW TO ISSUE A LICENSE (for us, the vendor):
    python -m src.utils.license issue \
        --customer "Tashkent Hospital #1" \
        --machine-id <fingerprint-from-clinic> \
        --tier pro \
        --expires 2027-12-31 \
        --output license.dat

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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
# To rotate keys: regenerate via `--gen-keys` and rebuild the app with the
# new public key embedded.
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

    def to_payload(self) -> dict:
        return {
            'customer': self.customer,
            'machine_id': self.machine_id,
            'tier': self.tier,
            'issued_at': self.issued_at,
            'expires_at': self.expires_at,
            'features': sorted(self.features),
            'seat_count': self.seat_count,
            'notes': self.notes,
        }


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
    """Where the license file lives on this machine."""
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


# ============================================================================
# CLI for vendor + customer use
# ============================================================================

def _generate_keypair():
    """Generate a fresh vendor signing keypair. Run this ONCE; embed the
    public key into SENTINEL_PUBLIC_KEY_PEM and keep the private key
    in a vault."""
    if not _CRYPTO_AVAILABLE:
        print('cryptography library required: pip install cryptography')
        sys.exit(1)

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

    Path('vendor_private_key.pem').write_bytes(priv_pem)
    Path('vendor_public_key.pem').write_bytes(pub_pem)
    print('✓ Generated vendor_private_key.pem (KEEP SECRET)')
    print('✓ Generated vendor_public_key.pem (EMBED IN APP)')
    print('\nReplace SENTINEL_PUBLIC_KEY_PEM in src/utils/license.py with:')
    print(pub_pem.decode())


def _cli_fingerprint():
    fp = get_machine_id()
    print(f'\nMachine fingerprint:\n  {fp}')
    print(f'\nSend this to the vendor to receive your license.dat.')


def _cli_issue(args):
    if not args.private_key:
        print('--private-key required (path to vendor_private_key.pem)')
        sys.exit(1)
    private_key_pem = Path(args.private_key).read_bytes()

    license_obj = License(
        customer=args.customer,
        machine_id=args.machine_id,
        tier=args.tier,
        issued_at=datetime.date.today().isoformat(),
        expires_at=args.expires,
        features=args.features.split(',') if args.features else ['chest', 'brain_2d', 'head_ct', 'mammography'],
        seat_count=args.seats,
        notes=args.notes or '',
    )
    license_data = sign_license(license_obj, private_key_pem)
    Path(args.output).write_text(license_data)
    print(f'✓ Wrote {args.output}')
    print(f'   Customer:   {license_obj.customer}')
    print(f'   Tier:       {license_obj.tier}')
    print(f'   Expires:    {license_obj.expires_at}')
    print(f'   Features:   {", ".join(license_obj.features)}')
    print(f'   Seats:      {license_obj.seat_count}')


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


def main():
    parser = argparse.ArgumentParser(description='Sentinel license tool')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('fingerprint', help='Print this machine\'s fingerprint')
    sub.add_parser('gen-keys', help='Generate vendor RSA keypair (run once)')

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

    p_verify = sub.add_parser('verify', help='Verify a license file')
    p_verify.add_argument('--license-file', help='path to license.dat')

    args = parser.parse_args()
    if args.cmd == 'fingerprint':
        _cli_fingerprint()
    elif args.cmd == 'gen-keys':
        _generate_keypair()
    elif args.cmd == 'issue':
        _cli_issue(args)
    elif args.cmd == 'verify':
        _cli_verify(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
