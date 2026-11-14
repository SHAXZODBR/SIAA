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


# ---------------------------------------------------------------- model_key_secret (contract D3)

SECRET = "ab" * 32                                   # 64 lowercase hex, synthetic


def _payload_of(license_data: str) -> dict:
    bundle = json.loads(base64.b64decode(license_data).decode())
    return json.loads(base64.b64decode(bundle["payload"]).decode())


def test_model_key_secret_round_trip(trusted):
    data = lic.sign_license(_license(model_key_secret=SECRET), trusted)
    assert _payload_of(data)["model_key_secret"] == SECRET          # inside the SIGNED payload
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is True, result.reason
    assert result.license.model_key_secret == SECRET


def test_license_without_secret_serializes_as_before(trusted):
    """No model key → the payload has no extra field, so pre-1.1 licenses and verifiers agree."""
    lo = _license()
    assert "model_key_secret" not in lo.to_payload()
    result = lic.verify_license(lic.sign_license(lo, trusted), current_machine_id=MACHINE)
    assert result.valid is True and result.license.model_key_secret is None


def test_tampered_model_key_secret_fails(trusted):
    data = lic.sign_license(_license(model_key_secret=SECRET), trusted)
    forged = _tamper(data, payload={"model_key_secret": "cd" * 32})
    result = lic.verify_license(forged, current_machine_id=MACHINE)
    assert result.valid is False and "signature" in result.reason
    # adding a secret to a license that had none is a forgery too
    plain = lic.sign_license(_license(), trusted)
    result = lic.verify_license(_tamper(plain, payload={"model_key_secret": SECRET}), current_machine_id=MACHINE)
    assert result.valid is False and "signature" in result.reason


@pytest.mark.parametrize("bad", ["ab" * 31, "AB" * 32, "zz" * 32, "ab" * 32 + "0", ""])
def test_sign_refuses_malformed_secret(trusted, bad):
    with pytest.raises(ValueError):
        lic.sign_license(_license(model_key_secret=bad), trusted)


def test_secret_never_in_repr(trusted):
    lo = _license(model_key_secret=SECRET)
    assert SECRET not in repr(lo)
    result = lic.verify_license(lic.sign_license(lo, trusted), current_machine_id=MACHINE)
    assert SECRET not in repr(result)


def test_generate_model_key_secret_is_fresh_64_hex():
    a, b = lic.generate_model_key_secret(), lic.generate_model_key_secret()
    assert a != b and lic.is_valid_model_key_secret(a) and lic.is_valid_model_key_secret(b)


@pytest.fixture
def local_license(tmp_path, monkeypatch, trusted):
    """A license.dat at a temp path that load_and_verify_local_license reads, on MACHINE."""
    path = tmp_path / "license.dat"
    monkeypatch.setattr(lic, "get_license_path", lambda: path)
    monkeypatch.setattr(lic, "get_machine_id", lambda: MACHINE)

    def write(**over):
        path.write_text(lic.sign_license(_license(**over), trusted))
        return path
    return write


def test_license_model_key_secret_valid_license(local_license):
    local_license(model_key_secret=SECRET)
    assert lic.license_model_key_secret() == bytes.fromhex(SECRET)


@pytest.mark.parametrize("over", [
    {"machine_id": "machine-fingerprint-B"},                                        # other PC
    {"expires_at": (datetime.date.today() - datetime.timedelta(days=1)).isoformat()},  # expired
    {},                                                                              # no secret
])
def test_license_model_key_secret_none_unless_valid_here(local_license, over):
    fields = {"model_key_secret": SECRET} if over else {}
    fields.update(over)
    local_license(**fields)
    assert lic.license_model_key_secret() is None


def test_license_model_key_secret_none_without_file_or_with_garbage(local_license, tmp_path):
    assert lic.license_model_key_secret() is None                    # no file
    (tmp_path / "license.dat").write_text("garbage")
    assert lic.license_model_key_secret() is None


def test_license_model_key_secret_untrusted_signer(local_license, monkeypatch):
    local_license(model_key_secret=SECRET)
    _priv, other_pub = _keypair()
    monkeypatch.setattr(lic, "SENTINEL_PUBLIC_KEY_PEM", other_pub)
    assert lic.license_model_key_secret() is None


def test_cli_issue_with_model_key_never_prints_secret(tmp_path, trusted, monkeypatch, capsys):
    key_file = tmp_path / "vendor_private_key.pem"
    key_file.write_bytes(trusted)
    out = tmp_path / "license.dat"
    lic.main(["issue", "--customer", "Test Clinic", "--machine-id", MACHINE, "--expires", "2031-12-31",
              "--private-key", str(key_file), "--output", str(out), "--with-model-key"])
    printed = capsys.readouterr()
    data = out.read_text()
    secret = _payload_of(data)["model_key_secret"]
    assert lic.is_valid_model_key_secret(secret)
    assert secret not in printed.out and secret not in printed.err
    assert "key fingerprint" in printed.out
    assert (out.stat().st_mode & 0o777) == 0o600
    result = lic.verify_license(data, current_machine_id=MACHINE)
    assert result.valid is True and result.license.model_key_secret == secret
    # without the flag: no secret
    out2 = tmp_path / "plain.dat"
    lic.main(["issue", "--customer", "Test Clinic", "--machine-id", MACHINE, "--expires", "2031-12-31",
              "--private-key", str(key_file), "--output", str(out2)])
    assert "model_key_secret" not in _payload_of(out2.read_text())


# ---------------------------------------------------------------- gen-keys

def test_gen_keys_defaults_to_vendor_keys_not_cwd(monkeypatch, tmp_path, capsys):
    """The pair goes into vendor_keys/ (git-ignored private key), never the cwd."""
    target = tmp_path / "vendor_keys"
    monkeypatch.setattr(lic, "VENDOR_KEYS_DIR", target)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    lic.main(["gen-keys"])
    assert list(cwd.iterdir()) == []
    priv, pub = target / "vendor_private_key.pem", target / "vendor_public_key.pem"
    assert priv.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
    assert pub.read_bytes().startswith(b"-----BEGIN PUBLIC KEY-----")
    import os, sys
    if sys.platform != "win32":
        assert (os.stat(priv).st_mode & 0o777) == 0o600
    assert lic.VENDOR_KEYS_DIR.name == "vendor_keys"


def test_gen_keys_out_dir_and_refuses_to_overwrite_private_key(tmp_path, capsys):
    priv, pub = lic._generate_keypair(tmp_path)
    assert priv == tmp_path / "vendor_private_key.pem"
    before = priv.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        lic._generate_keypair(tmp_path)
    with pytest.raises(SystemExit) as ei:
        lic.main(["gen-keys", "--out-dir", str(tmp_path)])
    assert ei.value.code == 1
    assert "REFUSED" in capsys.readouterr().out
    assert priv.read_bytes() == before


def test_private_key_is_ignored_anywhere_and_public_key_tracked():
    import subprocess
    from pathlib import Path
    repo = Path(lic.__file__).resolve().parents[2]
    gi = (repo / ".gitignore").read_text().splitlines()
    assert "vendor_private_key.pem" in [l.strip() for l in gi]
    try:
        r = subprocess.run(["git", "check-ignore", "--no-index", "vendor_private_key.pem",
                            "some/dir/vendor_private_key.pem", "vendor_keys/vendor_public_key.pem"],
                           cwd=repo, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available")
    ignored = set(r.stdout.split())
    assert {"vendor_private_key.pem", "some/dir/vendor_private_key.pem"} <= ignored
    assert "vendor_keys/vendor_public_key.pem" not in ignored
