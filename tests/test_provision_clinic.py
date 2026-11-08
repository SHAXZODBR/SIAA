"""packaging/provision_clinic.py — license.dat + per-machine encrypted model bundle (contract D3).

Everything is synthetic: a throw-away vendor keypair made by src.utils.license._generate_keypair
(in a temp cwd) and trusted by monkeypatching SENTINEL_PUBLIC_KEY_PEM, a tiny fake model dir with a
valid MANIFEST.json and a passing study_level_eval.json, a fingerprint of 64 hex chars, dates in
2031. No real model, no patient data. The end-to-end test does what the clinic will do: verify the
license for the fingerprint, unzip the bundle, decrypt with derive_key(secret, fingerprint).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import os
import re
import zipfile
from pathlib import Path

import pytest

from src.utils import license as lic
from src.utils import model_crypto as mc

REPO = Path(__file__).resolve().parents[1]
FP = "9a8b7c6d" + "0123456789abcdef" * 3 + "01234567"          # 64 hex, synthetic
assert len(FP) == 64
EXPIRES = "2031-12-31"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prov():
    return _load(REPO / "packaging" / "provision_clinic.py", "provision_clinic")


@pytest.fixture(scope="module")
def vendor_keys(tmp_path_factory):
    """A vendor keypair from license._generate_keypair (it writes into the cwd)."""
    d = tmp_path_factory.mktemp("vendor_keys")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        lic._generate_keypair()
    finally:
        os.chdir(cwd)
    return d / "vendor_private_key.pem", (d / "vendor_public_key.pem").read_bytes()


@pytest.fixture
def trusted(monkeypatch, vendor_keys):
    monkeypatch.setattr(lic, "SENTINEL_PUBLIC_KEY_PEM", vendor_keys[1])
    return vendor_keys[0]


def _modern_eval(files: dict, **over) -> dict:
    """A passing eval in the layout the release-pipeline eval_study_level.py writes."""
    holdout = REPO / "data" / "holdout_patients_v1.txt"
    ev = {
        "gate": {"name": "gate mean @0.50", "sens": 0.9, "spec": 0.5, "acc": 0.8, "balanced": 0.7,
                 "tp": 90, "fn": 10, "tn": 25, "fp": 25, "min_sens": 0.85, "min_spec": 0.4, "passed": True},
        "n_studies": 150, "evaluated_at": "2031-01-02T03:04:05",
        "protocol": {"aggregation": "mean", "threshold": 0.5, "n_central_slices": 5,
                     "preprocessing": "registry_predictor_native_res_bilinear_224"},
        "data": "data/__synthetic_missing_val_dir__/val", "val_patients_sha256": "ab" * 32,
        "holdout_covered": True, "holdout_missing": [],
        "holdout": {"file": str(holdout),
                    "sha256": hashlib.sha256(holdout.read_bytes()).hexdigest() if holdout.is_file() else None},
        "manifest": {"files": files},
    }
    ev.update(over)
    return ev


def make_model(root: Path, *, eval_over: dict | None = None, legacy: bool = False,
               promotion_log: dict | None = None, weights: bytes | None = None) -> tuple[Path, bytes, str]:
    d = root / "brain_triage_finetuned"
    d.mkdir(parents=True)
    weights = weights if weights is not None else os.urandom(25_000)
    (d / "config.json").write_text('{"architectures":["ViTForImageClassification"],"id2label":{"0":"abnormal","1":"normal"}}')
    (d / "preprocessor_config.json").write_text('{"size":224}')
    (d / "model.safetensors").write_bytes(weights)
    (d / "training_metrics.json").write_text('{"epochs": 3}')
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
             for n in ("config.json", "preprocessor_config.json", "model.safetensors")}
    (d / "MANIFEST.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))
    ev = _modern_eval(files, **(eval_over or {}))
    if legacy:
        for k in ("protocol", "data", "val_patients_sha256", "holdout_covered", "holdout_missing", "holdout"):
            ev.pop(k, None)
    (d / "study_level_eval.json").write_text(json.dumps(ev))
    if promotion_log is not None:
        (d / "promotion_log.json").write_text(json.dumps(promotion_log))
    return d, weights, files["model.safetensors"][:12]


def _snapshot(d: Path) -> dict:
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(d.iterdir())}


def _args(model: Path, out: Path, ledger: Path, private_key: Path, **over) -> list[str]:
    a = {"--fingerprint": FP, "--customer": "Synthetic Clinic #9", "--expires": EXPIRES,
         "--model": str(model), "--private-key": str(private_key), "--out": str(out), "--ledger": str(ledger)}
    a.update(over)
    argv: list[str] = []
    for k, v in a.items():
        if v is not None:
            argv += [k, v]
    return argv


def _payload(license_data: str) -> dict:
    bundle = json.loads(base64.b64decode(license_data).decode())
    return json.loads(base64.b64decode(bundle["payload"]).decode())


# ---------------------------------------------------------------- happy path

def test_provision_end_to_end(prov, trusted, tmp_path, capsys):
    model, weights, identity = make_model(tmp_path / "models")
    before = _snapshot(model)
    out, ledger = tmp_path / "deliveries", tmp_path / "licenses" / "ledger.csv"
    assert prov.main(_args(model, out, ledger, trusted)) == 0
    printed = capsys.readouterr()

    folder = out / f"synthetic-clinic-9_{FP[:8]}"
    zip_name = f"brain_triage_finetuned_{identity}_{FP[:8]}.zip"
    assert sorted(p.name for p in out.iterdir()) == [folder.name], "no staging leftovers"
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["license.dat", zip_name, "README_admin_en.txt", "README_admin_ru.txt"])
    assert _snapshot(model) == before, "the repo's plaintext model must never be touched"

    # the license verifies on THAT machine only and carries the secret in the signed payload
    data = (folder / "license.dat").read_text()
    assert (folder / "license.dat").stat().st_mode & 0o777 == 0o600
    res = lic.verify_license(data, current_machine_id=FP)
    assert res.valid, res.reason
    assert res.license.customer == "Synthetic Clinic #9" and res.license.expires_at == EXPIRES
    secret = res.license.model_key_secret
    assert lic.is_valid_model_key_secret(secret) and _payload(data)["model_key_secret"] == secret
    assert not lic.verify_license(data, current_machine_id="0" * 64).valid

    # the clinic side: unzip like /models/import, decrypt with HKDF(secret || fingerprint)
    with zipfile.ZipFile(folder / zip_name) as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "clinic")
    assert "brain_triage_finetuned/model.safetensors" not in names, "plaintext weights in the bundle"
    d = tmp_path / "clinic" / "brain_triage_finetuned"
    key = mc.derive_key(bytes.fromhex(secret), FP)
    assert mc.decrypt_file_to_bytes(d / "model.safetensors.enc", key) == weights
    files = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert files["model.safetensors"][:12] == identity
    assert files["model.safetensors.enc"] == hashlib.sha256((d / "model.safetensors.enc").read_bytes()).hexdigest()

    # READMEs: steps + identity, never the secret
    en = (folder / "README_admin_en.txt").read_text()
    ru = (folder / "README_admin_ru.txt").read_text(encoding="utf-8")
    for text in (en, ru):
        assert identity in text and FP[:8] in text and secret not in text and "license.dat" in text
    assert "Import license.dat" in en and "Import model bundle" in en and "Restart" in en
    assert "Импортировать license.dat" in ru and "Импорт пакета модели" in ru
    # licence → restart → bundle (the key travels in the licence), in both languages
    assert en.index("Import license.dat") < en.index("Restart Sentinel") < en.index("Import model bundle")
    assert ru.index("Импортировать license.dat") < ru.index("Перезапустите") < ru.index("Импорт пакета модели")
    # every UI label quoted must be the app's own dictionary string
    repo = Path(__file__).resolve().parents[1]
    for lang, text in (("en", en), ("ru", ru)):
        dic = (repo / "desktop-app" / "src" / "i18n" / f"{lang}.ts").read_text(encoding="utf-8")
        for ui_key in ("settings.license", "settings.ai", "settings.about", "license.import",
                       "settings.importBundle", "about.exportDiagnostics", "validation.validated"):
            label = re.search(rf"'{re.escape(ui_key)}':\s*'([^']+)'", dic).group(1)
            assert label in text, (lang, ui_key, label)

    # ledger: one row, no secret
    rows = list(csv.DictReader(ledger.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert (row["customer"], row["fp8"], row["identity"], row["expires"], row["tier"]) == \
        ("Synthetic Clinic #9", FP[:8], identity, EXPIRES, "pro")
    assert row["key_fingerprint"] == mc.key_fingerprint(key)
    assert row["bundle_sha256"] == hashlib.sha256((folder / zip_name).read_bytes()).hexdigest()
    assert secret not in ledger.read_text() and "model_key_secret" not in ledger.read_text()

    # the summary never shows the secret or the key
    for stream in (printed.out, printed.err):
        assert secret not in stream and key.hex() not in stream
    assert identity in printed.out and mc.key_fingerprint(key) in printed.out


def test_second_clinic_gets_a_different_secret_and_ledger_appends(prov, trusted, tmp_path):
    model, _, _ = make_model(tmp_path / "models")
    out, ledger = tmp_path / "d", tmp_path / "ledger.csv"
    fp2 = "1" * 64
    assert prov.main(_args(model, out, ledger, trusted)) == 0
    assert prov.main(_args(model, out, ledger, trusted, **{"--fingerprint": fp2, "--customer": "Клиника Ташкент"})) == 0
    s1 = lic.verify_license((out / f"synthetic-clinic-9_{FP[:8]}" / "license.dat").read_text(), FP).license
    s2 = lic.verify_license((out / f"klinika-tashkent_{fp2[:8]}" / "license.dat").read_text(), fp2).license
    assert s1.model_key_secret != s2.model_key_secret
    assert [r["fp8"] for r in csv.DictReader(ledger.open(encoding="utf-8"))] == [FP[:8], fp2[:8]]


def test_legacy_eval_is_accepted_with_a_warning(prov, trusted, tmp_path, capsys):
    """The v1 release predates the pipeline (no protocol block) — provisionable, like publish."""
    model, _, _ = make_model(tmp_path / "models", legacy=True)
    assert prov.main(_args(model, tmp_path / "d", tmp_path / "l.csv", trusted)) == 0
    assert "predates the release pipeline" in capsys.readouterr().out


def test_fingerprint_is_normalised(prov):
    assert prov.validate_fingerprint("  " + FP.upper() + "\n") == FP


def test_fingerprint_accepts_the_cli_output_verbatim(prov):
    """Regression (dress rehearsal 2026-11-09): `fp=$(python -m src.utils.license fingerprint)`
    captures the decorated CLI text, which used to be refused as bad_fingerprint."""
    cli_text = f"\nMachine fingerprint:\n  {FP}\n\nSend this to the vendor to receive your license.dat.\n"
    assert prov.validate_fingerprint(cli_text) == FP
    assert prov.validate_fingerprint(f"fp: {FP.upper()} (sent twice: {FP})") == FP


@pytest.mark.parametrize("raw", ["", "no hex here", "a" * 65, "a" * 63,
                                 f"{'a' * 64} and {'b' * 64}", "x" + "a" * 64])
def test_fingerprint_refuses_absent_or_ambiguous_tokens(prov, raw):
    with pytest.raises(prov.Refused) as ei:
        prov.validate_fingerprint(raw)
    assert ei.value.code == "bad_fingerprint"


@pytest.mark.parametrize("name,slug", [("Synthetic Clinic #9", "synthetic-clinic-9"),
                                       ("Клиника «Шифо» №2", "klinika-shifo-no2"),
                                       ("Ўзбекистон Қон маркази", "ozbekiston-qon-markazi"),
                                       ("!!!", "clinic")])
def test_customer_slug(prov, name, slug):
    assert prov.customer_slug(name) == slug


# ---------------------------------------------------------------- refusals (nothing written)

def _assert_refused(prov, argv, tmp_path, capsys, code: str, out: Path, ledger: Path, exit_code: int = 1):
    assert prov.main(argv) == exit_code
    err = capsys.readouterr().err
    assert f"[{code}]" in err, err
    assert not out.exists() or not any(out.iterdir()), "a refusal must not write a delivery"
    assert not ledger.exists(), "a refusal must not touch the ledger"


@pytest.mark.parametrize("over,code", [
    ({"--fingerprint": "abc123"}, "bad_fingerprint"),
    ({"--fingerprint": "g" * 64}, "bad_fingerprint"),
    ({"--expires": "2020-01-01"}, "bad_expiry"),
    ({"--expires": "31.12.2031"}, "bad_expiry"),
    ({"--customer": "  "}, "bad_customer"),
])
def test_refuses_bad_input(prov, trusted, tmp_path, capsys, over, code):
    model, _, _ = make_model(tmp_path / "models")
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    _assert_refused(prov, _args(model, out, ledger, trusted, **over), tmp_path, capsys, code, out, ledger)


@pytest.mark.parametrize("kwargs,code,needle", [
    ({"eval_over": {"gate": {"name": "gate mean @0.50", "passed": False, "sens": 0.7, "spec": 0.5,
                             "min_sens": 0.85, "min_spec": 0.4}}}, "release_check", "gate.passed"),
    ({"eval_over": {"promotion": {"promote": False, "reason": "worse than current"}}}, "release_check",
     "promotion.promote"),
    ({"eval_over": {"protocol": {"aggregation": "max", "threshold": 0.3,
                                 "preprocessing": "registry_predictor_native_res_bilinear_224"}}},
     "release_check", "aggregation/threshold"),
    ({"eval_over": {"manifest": {"files": {"model.safetensors": "f" * 64}}}}, "release_check", "stale"),
    ({"promotion_log": {"forced": True, "refusal_reasons_overridden": ["promotion.promote is False"]}},
     "forced_promotion", "--force promotion"),
])
def test_refuses_models_that_fail_the_release_rules(prov, trusted, tmp_path, capsys, kwargs, code, needle):
    model, _, _ = make_model(tmp_path / "models", **kwargs)
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    assert prov.main(_args(model, out, ledger, trusted)) == 1
    err = capsys.readouterr().err
    assert f"[{code}]" in err and needle in err, err
    assert not out.exists() and not ledger.exists()


def test_refuses_missing_manifest_or_model(prov, trusted, tmp_path, capsys):
    model, _, _ = make_model(tmp_path / "models")
    (model / "MANIFEST.json").unlink()
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    _assert_refused(prov, _args(model, out, ledger, trusted), tmp_path, capsys, "manifest_missing", out, ledger)
    _assert_refused(prov, _args(tmp_path / "nope", out, ledger, trusted), tmp_path, capsys, "model_missing",
                    out, ledger)


def test_refuses_a_private_key_the_app_does_not_trust(prov, vendor_keys, tmp_path, capsys):
    """SENTINEL_PUBLIC_KEY_PEM untouched: the generated key is not the embedded one → the clinic
    app would reject the license, so nothing is delivered."""
    model, _, _ = make_model(tmp_path / "models")
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    assert prov.main(_args(model, out, ledger, vendor_keys[0])) == 1
    assert "[private_key]" in capsys.readouterr().err
    assert not any(out.iterdir()) and not ledger.exists()
    assert prov.main(_args(model, out, ledger, tmp_path / "missing.pem")) == 1
    assert "[private_key]" in capsys.readouterr().err


def test_refuses_to_overwrite_a_delivery(prov, trusted, tmp_path, capsys):
    model, _, _ = make_model(tmp_path / "models")
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    assert prov.main(_args(model, out, ledger, trusted)) == 0
    first = (out / f"synthetic-clinic-9_{FP[:8]}" / "license.dat").read_text()
    capsys.readouterr()
    assert prov.main(_args(model, out, ledger, trusted)) == 1
    assert "[output_exists]" in capsys.readouterr().err
    assert (out / f"synthetic-clinic-9_{FP[:8]}" / "license.dat").read_text() == first
    assert len(list(csv.DictReader(ledger.open(encoding="utf-8")))) == 1


def test_export_failure_leaves_nothing_behind(prov, trusted, tmp_path, capsys):
    model, _, _ = make_model(tmp_path / "models")
    (model / "model.safetensors").write_bytes(os.urandom(25_000))      # corrupted after the eval/MANIFEST
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    assert prov.main(_args(model, out, ledger, trusted)) == 2
    assert "manifest_mismatch" in capsys.readouterr().err
    assert not any(out.iterdir()), "staging dir must be removed"
    assert not ledger.exists()


def test_self_verification_catches_a_bad_bundle(prov, trusted, tmp_path, capsys, monkeypatch):
    """If the bundle were encrypted with any key but the license-derived one, nothing ships."""
    model, _, _ = make_model(tmp_path / "models")
    real = mc.export_encrypted_bundle
    monkeypatch.setattr(mc, "export_encrypted_bundle",
                        lambda m, key, z, fp8: real(m, bytes(32), z, fp8))
    out, ledger = tmp_path / "d", tmp_path / "l.csv"
    # a self-verification failure is a FAILURE after the checks (exit 2), not an input refusal
    assert prov.main(_args(model, out, ledger, trusted)) == 2
    assert "[verify_failed]" in capsys.readouterr().err
    assert not any(out.iterdir()) and not ledger.exists()
