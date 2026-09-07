"""License signing / verification with a throw-away RSA keypair.

The verifier trusts the public key embedded in src.utils.license, so the
tests swap SENTINEL_PUBLIC_KEY_PEM for the throw-away public key. A license
signed with a key the app does not trust must still be rejected.
"""

import base64
import datetime
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.utils import license as lic

MACHINE = "machine-fingerprint-A"


def _keypair() -> tuple[bytes, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub


@pytest.fixture(scope="module")
def vendor_keys():
    return _keypair()


@pytest.fixture
def trusted(monkeypatch, vendor_keys):
    """Make the throw-away public key the one the app trusts."""
    monkeypatch.setattr(lic, "SENTINEL_PUBLIC_KEY_PEM", vendor_keys[1])
    return vendor_keys[0]


def _license(**over) -> lic.License:
    today = datetime.date.today()
    fields = dict(
        customer="Test Clinic",
        machine_id=MACHINE,
        tier="pro",
        issued_at=today.isoformat(),
        expires_at=(today + datetime.timedelta(days=365)).isoformat(),
        features=["brain_2d", "chest"],
        seat_count=1,
        notes="",
    )
    fields.update(over)
    return lic.License(**fields)


def _tamper(license_data: str, *, signature: bool = False, payload: dict | None = None) -> str:
    bundle = json.loads(base64.b64decode(license_data).decode())
    if signature:
        sig = bytearray(base64.b64decode(bundle["signature"]))
        sig[0] ^= 0xFF
        bundle["signature"] = base64.b64encode(bytes(sig)).decode()
    if payload is not None:
        p = json.loads(base64.b64decode(bundle["payload"]).decode())
        p.update(payload)
        bundle["payload"] = base64.b64encode(json.dumps(p, sort_keys=True).encode()).decode()
    return base64.b64encode(json.dumps(bundle).encode()).decode()


def test_valid_license_verifies(trusted):
    data = lic.sign_license(_license(), trusted)
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is True, result.reason
    assert result.reason == "OK"
    assert result.license.customer == "Test Clinic"
    assert result.license.tier == "pro"
    assert 360 <= result.days_remaining <= 365


def test_tampered_signature_fails(trusted):
    data = lic.sign_license(_license(), trusted)
    result = lic.verify_license(_tamper(data, signature=True), current_machine_id=MACHINE)
    assert result.valid is False
    assert "signature" in result.reason


def test_tampered_payload_fails(trusted):
    data = lic.sign_license(_license(), trusted)
    forged = _tamper(data, payload={"customer": "Someone Else", "tier": "enterprise"})
    result = lic.verify_license(forged, current_machine_id=MACHINE)
    assert result.valid is False
    assert "signature" in result.reason


def test_expired_license_fails(trusted):
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    data = lic.sign_license(_license(expires_at=yesterday), trusted)
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is False
    assert "expired" in result.reason
    assert result.days_remaining < 0
    assert result.license is not None           # payload still readable for the UI


def test_last_day_still_valid(trusted):
    today = datetime.date.today().isoformat()
    data = lic.sign_license(_license(expires_at=today), trusted)
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is True and result.days_remaining == 0


def test_wrong_machine_fails(trusted):
    data = lic.sign_license(_license(), trusted)
    result = lic.verify_license(data, current_machine_id="machine-fingerprint-B")
    assert result.valid is False
    assert "different machine" in result.reason


def test_untrusted_key_fails():
    """A license signed by a key the app does not embed is a forgery."""
    priv, _pub = _keypair()
    data = lic.sign_license(_license(), priv)       # SENTINEL_PUBLIC_KEY_PEM untouched
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is False
    assert "signature" in result.reason


def test_malformed_file_fails():
    result = lic.verify_license("not-a-license", current_machine_id=MACHINE)
    assert result.valid is False
    assert "malformed" in result.reason


def test_machine_id_is_stable_and_hex():
    a, b = lic.get_machine_id(), lic.get_machine_id()
    assert a == b and len(a) == 64 and int(a, 16) >= 0
