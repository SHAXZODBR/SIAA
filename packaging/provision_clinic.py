#!/usr/bin/env python3
"""Provision one clinic PC: signed license.dat + a model bundle encrypted for that PC only.

WHAT (docs/dev/finish_line_contract.md D3, vendor side, run on the vendor Mac):
    provision_clinic.py --fingerprint <machine_id> --customer "<clinic>" --expires 2027-12-31
                        [--tier pro] [--features ...] [--model models/brain_triage_finetuned]
                        [--private-key vendor_keys/vendor_private_key.pem] --out <dir>

  writes <out>/<customer-slug>_<fp8>/ :
    license.dat                                     signed; carries a fresh model_key_secret (0600)
    brain_triage_finetuned_<identity12>_<fp8>.zip   the model encrypted with
                                                    HKDF(secret || fingerprint) — layout of
                                                    POST /models/import
    README_admin_ru.txt / README_admin_en.txt       import steps for the clinic admin
  and appends a row to data/licenses/ledger.csv (customer, fp8, identity, issued, expires, tier,
  features, key_fingerprint, bundle_sha256 — never the secret).

WHY: the clinic installer may ship without the validated model. The weights reach a clinic only
as ciphertext bound to that machine, and the key material travels inside the RSA-signed license,
so the vendor can encrypt offline and the clinic never sees plaintext weights. The repo's
plaintext model is never touched (model_crypto.export_encrypted_bundle works on a temp copy).

REFUSALS (exit 1, reason code printed, nothing written):
  bad_fingerprint     not 64 hex chars (what `python -m src.utils.license fingerprint` prints)
  bad_customer        --customer is empty
  bad_expiry          not YYYY-MM-DD, or not after today
  model_missing       --model is not a dir with config/preprocessor/plaintext model.safetensors
  manifest_missing    no MANIFEST.json pin for model.safetensors (identity unknown)
  release_check       study_level_eval.json fails the release rules — promote_model.eval_verdict
                      (hard AND comparison reasons; there is no --force here) for evals written by the
                      release pipeline, plus publish_model_bundle.release_eval_check (gate, floors,
                      production gate name, stale pin, forced promotion_log.json). An eval WITHOUT a
                      `protocol` block predates the pipeline (the v1 release): it is accepted on the
                      gate/floor/pin checks exactly as publish_model_bundle accepts it for the
                      installer, and a warning says so.
  forced_promotion    promotion_log.json records a --force promotion (part of release_check)
  private_key         unreadable private key, or it does not match the public key embedded in
                      src/utils/license.py (the clinic app would reject the license)
  output_exists       <out>/<slug>_<fp8>/ already exists (a delivery is never overwritten)
Failures after the checks (exit 2): bundle export / self-verification failed (manifest_mismatch,
verify_failed, …) — the staging dir is removed. A ledger write failure after the delivery folder
exists also exits 2 and prints the row to add by hand.

SELF-VERIFICATION before the folder is published: the license is verified with the app's own
verify_license() for the given fingerprint, the zip is extracted with
fetch_model_bundle.extract_bundle (what POST /models/import reuses), both MANIFEST pins are checked
and the ciphertext is decrypted with the key derived from the SIGNED license payload — i.e. exactly
what the clinic's app will do.

The secret is never printed or logged; the summary shows only the derived key's fingerprint
(`python -m src.utils.model_crypto keyinfo` prints the same value on the clinic PC).
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils import license as lic            # noqa: E402
from src.utils import model_crypto as mc        # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_FAILED = 2
DEFAULT_MODEL = REPO / 'models' / 'brain_triage_finetuned'
DEFAULT_PRIVATE_KEY = REPO / 'vendor_keys' / 'vendor_private_key.pem'
DEFAULT_LEDGER = REPO / 'data' / 'licenses' / 'ledger.csv'
DEFAULT_FEATURES = 'chest,brain_2d,head_ct,mammography'   # == license.py issue default
LEDGER_FIELDS = ('customer', 'fp8', 'identity', 'issued', 'expires', 'tier',
                 'features', 'key_fingerprint', 'bundle_sha256')
FINGERPRINT_RE = re.compile(r'^[0-9a-f]{64}$')
FINGERPRINT_TOKEN_RE = re.compile(r'(?<![0-9a-z])[0-9a-f]{64}(?![0-9a-z])')
BUNDLE_PREFIX = 'brain_triage_finetuned'

_CYR = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'ғ': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'қ': 'q', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ў': 'o', 'ф': 'f', 'х': 'h', 'ҳ': 'h',
        'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e',
        'ю': 'yu', 'я': 'ya'}


class Refused(Exception):
    """A provisioning refusal: .code is the reason code, nothing has been written."""

    def __init__(self, code: str, message: str):
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message


class VerifyFailed(Refused):
    """Self-verification of the BUILT delivery failed (reason code 'verify_failed').
    Unlike an input refusal this happens after the checks passed — exit 2, the
    staging dir is removed and nothing is delivered."""

    def __init__(self, message: str):
        super().__init__('verify_failed', message)


# ----------------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------------

def validate_fingerprint(raw: str) -> str:
    """The machine fingerprint as get_machine_id() produces it: 64 lowercase hex chars.

    Surrounding whitespace and upper case (copied from an e-mail) are normalised. The whole
    output of `python -m src.utils.license fingerprint` ("Machine fingerprint: <hex> Send this
    to the vendor ...") is also accepted — that is what `fp=$(python -m src.utils.license
    fingerprint)` captures and what a clinic pastes into an e-mail — as long as it contains
    exactly ONE 64-hex token. Zero or several distinct tokens are refused (never guess)."""
    fp = (raw or '').strip().lower()
    if FINGERPRINT_RE.match(fp):
        return fp
    tokens = sorted(set(FINGERPRINT_TOKEN_RE.findall(fp)))
    if len(tokens) == 1:
        return tokens[0]
    raise Refused('bad_fingerprint', 'the fingerprint must be the 64 hex characters printed by '
                                     '`python -m src.utils.license fingerprint` on the clinic PC'
                                     + (' (the text holds several candidates)' if tokens else ''))


def validate_expiry(raw: str, today: Optional[_dt.date] = None) -> str:
    today = today or _dt.date.today()
    try:
        d = _dt.date.fromisoformat((raw or '').strip())
    except ValueError:
        raise Refused('bad_expiry', f'--expires {raw!r} is not YYYY-MM-DD') from None
    if d <= today:
        raise Refused('bad_expiry', f'--expires {d.isoformat()} is not after today ({today.isoformat()})')
    return d.isoformat()


def customer_slug(customer: str) -> str:
    """Filesystem-safe ASCII slug of the clinic name (Russian/Uzbek Cyrillic transliterated)."""
    out = []
    for ch in (customer or '').strip().lower():
        if ch in _CYR:
            out.append(_CYR[ch])
        else:
            out.append(unicodedata.normalize('NFKD', ch).encode('ascii', 'ignore').decode('ascii'))
    slug = re.sub(r'[^a-z0-9]+', '-', ''.join(out).lower()).strip('-')
    return slug[:48].strip('-') or 'clinic'


# ----------------------------------------------------------------------------
# release rules (reused, not re-implemented)
# ----------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def model_identity(model_dir: Path) -> tuple[str, dict]:
    """(identity12, MANIFEST files) — identity is the MANIFEST pin of model.safetensors."""
    mpath = model_dir / 'MANIFEST.json'
    try:
        files = json.loads(mpath.read_text(encoding='utf-8')).get('files') or {}
    except FileNotFoundError:
        raise Refused('manifest_missing', f'{mpath} missing — run scripts/eval_study_level.py '
                                          f'--write-manifest (the identity is its pin)') from None
    except (ValueError, OSError) as e:
        raise Refused('manifest_missing', f'unreadable {mpath}: {e}') from None
    pin = files.get('model.safetensors')
    if not isinstance(pin, str) or not re.match(r'^[0-9a-f]{64}$', pin):
        raise Refused('manifest_missing', f'{mpath} has no sha256 pin for model.safetensors')
    return pin[:12], files


def release_problems(model_dir: Path, identity: str, manifest_files: dict) -> tuple[list[str], list[str]]:
    """(problems, warnings) under the release rules. Empty problems = provisionable."""
    pub = _load_module(HERE / 'publish_model_bundle.py', '_provision_publish_rules')
    problems, summary = pub.release_eval_check(model_dir, identity)
    warnings: list[str] = []
    ev_path = model_dir / 'study_level_eval.json'
    ev: Any = None
    try:
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        ev = None                                  # already reported by release_eval_check
    if isinstance(ev, dict) and 'protocol' in ev:
        promote = _load_module(REPO / 'scripts' / 'promote_model.py', '_provision_promote_rules')
        holdout = promote.DEFAULT_HOLDOUT if Path(promote.DEFAULT_HOLDOUT).is_file() else None
        hard, soft = promote.eval_verdict(ev, {'files': manifest_files}, holdout)
        for r in hard + soft:                      # no --force here: the comparison counts too
            if r not in problems:
                problems.append(r)
    elif isinstance(ev, dict) and summary and summary.get('legacy_eval'):
        warnings.append('study_level_eval.json predates the release pipeline (no protocol block): accepted on '
                        'the gate/floor/pin checks, as publish_model_bundle accepts it for the installer')
    return problems, warnings


# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------

README_EN = """SENTINEL MEDICAL AI — model + licence delivery for {customer}
==========================================================================
Machine fingerprint : {fp8}… (Sentinel accepts this folder on THAT computer only)
Model identity      : {identity}
Licence expires     : {expires}   (tier: {tier})
Key fingerprint     : {key_fp}

Files
  license.dat   — your licence. It also carries the key of the AI model. CONFIDENTIAL: together
                  with the bundle below it is equivalent to the model itself.
  {zip_name}
                — the validated brain-MRI triage model, encrypted for this computer.

Install (administrator account)
  1. Copy this folder to the clinic computer (USB stick or network share).
  2. Open Sentinel → Preferences → License → "Import license.dat" and choose license.dat.
     The licence panel must show "{customer}" and the expiry date {expires}.
  3. Restart Sentinel (the licence carries the key that unlocks the model bundle).
  4. Preferences → AI models → "Import model bundle" and choose
     {zip_name}
     The confirmation shows the model identity — it must be {identity}.
  5. Check: Preferences → AI models shows the brain triage model with identity {identity}
     and status "Validated". The same identity is printed on every AI report.

Troubleshooting
  * "license_model_key_missing": the licence of this delivery is not active yet. Import license.dat,
    restart Sentinel, then import the bundle again.
  * "bundle_not_for_this_machine": the installed licence is not the one from THIS folder, or the bundle
    was made for another computer. Import license.dat from this folder, restart Sentinel, then import
    the bundle again. If it still fails, send the vendor the fingerprint shown in
    Preferences → License — it must start with {fp8}.
  * Preferences → About → "Export diagnostics" creates a zip for vendor support (no patient data).

Keep this folder confidential
  * Sentinel refuses these files on any other computer, but license.dat TOGETHER with the bundle is
    the key to the model weights. Keep the folder on an encrypted USB stick or with the clinic's
    locked backups only; never e-mail it, never copy it to other computers or shared drives.
"""

README_RU = """SENTINEL MEDICAL AI — поставка модели и лицензии для {customer}
==========================================================================
Отпечаток компьютера : {fp8}… (Sentinel принимает папку ТОЛЬКО на этом компьютере)
Идентификатор модели : {identity}
Лицензия действует до: {expires}   (тариф: {tier})
Отпечаток ключа      : {key_fp}

Файлы
  license.dat   — ваша лицензия. Она же содержит ключ ИИ-модели. КОНФИДЕНЦИАЛЬНО: вместе с
                  пакетом ниже она равноценна самой модели.
  {zip_name}
                — валидированная модель триажа МРТ головного мозга, зашифрованная для этого компьютера.

Установка (учётная запись администратора)
  1. Скопируйте эту папку на компьютер клиники (USB-накопитель или сетевая папка).
  2. Откройте Sentinel → Настройки → Лицензия → «Импортировать license.dat» и выберите license.dat.
     В разделе лицензии должно отображаться «{customer}» и дата окончания {expires}.
  3. Перезапустите Sentinel (лицензия содержит ключ, который открывает пакет модели).
  4. Настройки → Модели ИИ → «Импорт пакета модели» и выберите
     {zip_name}
     В подтверждении показан идентификатор модели — он должен быть {identity}.
  5. Проверка: в Настройки → Модели ИИ модель триажа мозга имеет идентификатор {identity}
     и статус «Валидировано». Этот же идентификатор печатается в каждом ИИ-отчёте.

Если что-то не так
  * «license_model_key_missing»: лицензия этой поставки ещё не активна. Импортируйте license.dat,
    перезапустите Sentinel и импортируйте пакет снова.
  * «bundle_not_for_this_machine»: установлена лицензия не из ЭТОЙ папки, или пакет сделан для
    другого компьютера. Импортируйте license.dat из этой папки, перезапустите Sentinel и импортируйте
    пакет снова. Если ошибка остаётся, отправьте поставщику отпечаток из Настройки → Лицензия — он
    должен начинаться с {fp8}.
  * Настройки → О программе → «Экспорт диагностики» создаёт архив для поддержки (без данных пациентов).

Храните папку конфиденциально
  * На других компьютерах Sentinel эти файлы не примет, но license.dat ВМЕСТЕ с пакетом — это ключ к
    весам модели. Храните папку только на зашифрованном USB-носителе или в закрытых резервных копиях
    клиники; не отправляйте её по почте и не копируйте на другие компьютеры и общие папки.
"""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _secret_from_signed_license(license_data: str) -> str:
    """model_key_secret as it sits in the SIGNED payload (not the in-memory value) — the
    self-check must use exactly what the clinic's app will read."""
    bundle = json.loads(base64.b64decode(license_data).decode())
    payload = json.loads(base64.b64decode(bundle['payload']).decode())
    return payload['model_key_secret']


def verify_delivery(zip_path: Path, license_data: str, fingerprint: str, identity: str) -> None:
    """Do what the clinic will do: verify the licence for this fingerprint, extract the bundle
    like POST /models/import, check both MANIFEST pins and decrypt with the licence-derived key.
    Raises Refused('private_key', …) (exit 1: the key pair is wrong, an input problem) or
    VerifyFailed (exit 2: the built delivery would not work at the clinic)."""
    res = lic.verify_license(license_data, current_machine_id=fingerprint)
    if not res.valid:
        if 'signature' in res.reason:
            raise Refused('private_key', f'the clinic app would reject this license: {res.reason} '
                                         '(the private key does not match the public key embedded in '
                                         'src/utils/license.py)')
        raise VerifyFailed(f'the clinic app would reject this license: {res.reason}')
    key = mc.clinic_key_from_secret_hex(_secret_from_signed_license(license_data), fingerprint)
    fetch = _load_module(HERE / 'fetch_model_bundle.py', '_provision_fetch')
    work = Path(tempfile.mkdtemp(prefix='sentinel_provision_verify_'))
    try:
        d = fetch.extract_bundle(zip_path, work)
        files = json.loads((d / 'MANIFEST.json').read_text()).get('files') or {}
        enc = d / 'model.safetensors.enc'
        if (d / 'model.safetensors').exists():
            raise VerifyFailed('the bundle contains PLAINTEXT weights')
        if not enc.is_file() or not mc.meta_path_for(enc).is_file():
            raise VerifyFailed('model.safetensors.enc or its sidecar missing from the bundle')
        if _sha256_file(enc) != files.get('model.safetensors.enc'):
            raise VerifyFailed('ciphertext sha256 != MANIFEST pin')
        if files.get('model.safetensors', '')[:12] != identity:
            raise VerifyFailed('MANIFEST plaintext pin is not the provisioned identity')
        if mc.read_meta(enc).get('plaintext_sha256') != files.get('model.safetensors'):
            raise VerifyFailed('sidecar plaintext_sha256 != MANIFEST pin')
        try:
            plain = mc.decrypt_file_to_bytes(enc, key)        # also checks the sidecar sha
        except Exception as e:
            raise VerifyFailed(f'cannot decrypt with the licence-derived key ({type(e).__name__})') from None
        if hashlib.sha256(plain).hexdigest() != files.get('model.safetensors'):
            raise VerifyFailed('decrypted weights != MANIFEST plaintext pin')
        del plain
    finally:
        shutil.rmtree(work, ignore_errors=True)


def append_ledger(ledger: Path, row: dict) -> None:
    """Append one row (never the secret). An existing ledger keeps its own header."""
    assert 'model_key_secret' not in row
    ledger.parent.mkdir(parents=True, exist_ok=True)
    exists = ledger.is_file() and ledger.stat().st_size > 0
    fields = list(LEDGER_FIELDS)
    if exists:
        with open(ledger, newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), None)
        if header:
            fields = header
    with open(ledger, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------
# main flow
# ----------------------------------------------------------------------------

def provision(*, fingerprint: str, customer: str, expires: str, tier: str, features: str,
              model: Path, private_key: Path, out: Path, ledger: Path) -> dict:
    """Run every check, build + self-verify the delivery, publish the folder, append the ledger.
    Returns the summary dict (no secret). Raises Refused (nothing written) or RuntimeError."""
    fp = validate_fingerprint(fingerprint)
    fp8 = fp[:8]
    customer = (customer or '').strip()
    if not customer:
        raise Refused('bad_customer', '--customer is empty')
    expires = validate_expiry(expires)
    feats = sorted({f.strip() for f in (features or DEFAULT_FEATURES).split(',') if f.strip()})
    model = Path(model)
    if not model.is_dir() or not all((model / n).is_file() for n in mc.BUNDLE_REQUIRED_FILES):
        raise Refused('model_missing', f'{model} must hold {", ".join(mc.BUNDLE_REQUIRED_FILES)} '
                                       f'(plaintext release on the vendor machine)')
    identity, manifest_files = model_identity(model)
    problems, warnings = release_problems(model, identity, manifest_files)
    if problems:
        code = 'forced_promotion' if any('--force promotion' in p for p in problems) else 'release_check'
        raise Refused(code, 'the model is not a validated release:\n      - ' + '\n      - '.join(problems))
    try:
        private_pem = Path(private_key).read_bytes()
    except OSError as e:
        raise Refused('private_key', f'cannot read {private_key}: {e}') from None
    final_dir = Path(out) / f'{customer_slug(customer)}_{fp8}'
    if final_dir.exists():
        raise Refused('output_exists', f'{final_dir} already exists — a delivery is never overwritten '
                                       f'(move it away or choose another --out)')

    secret_hex = lic.generate_model_key_secret()
    key = mc.clinic_key_from_secret_hex(secret_hex, fp)
    key_fp = mc.key_fingerprint(key)
    issued = _dt.date.today().isoformat()
    license_obj = lic.License(customer=customer, machine_id=fp, tier=tier, issued_at=issued,
                              expires_at=expires, features=feats, seat_count=1,
                              notes=f'model {identity}', model_key_secret=secret_hex)
    try:
        license_data = lic.sign_license(license_obj, private_pem)
    except Exception as e:
        raise Refused('private_key', f'cannot sign with {private_key} ({type(e).__name__})') from None
    del secret_hex

    zip_name = f'{BUNDLE_PREFIX}_{identity}_{fp8}.zip'
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{final_dir.name}.staging-', dir=final_dir.parent))
    try:
        try:
            mc.export_encrypted_bundle(model, key, staging / zip_name, fp8)
        except mc.BundleExportError as e:
            raise RuntimeError(str(e)) from None
        verify_delivery(staging / zip_name, license_data, fp, identity)
        lic._write_private_text(staging / 'license.dat', license_data)
        ctx = dict(customer=customer, fp8=fp8, identity=identity, expires=expires, tier=tier,
                   key_fp=key_fp, zip_name=zip_name)
        (staging / 'README_admin_en.txt').write_text(README_EN.format(**ctx), encoding='utf-8')
        (staging / 'README_admin_ru.txt').write_text(README_RU.format(**ctx), encoding='utf-8')
        bundle_sha = _sha256_file(staging / zip_name)
        os.chmod(staging, 0o700)
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    row = {'customer': customer, 'fp8': fp8, 'identity': identity, 'issued': issued, 'expires': expires,
           'tier': tier, 'features': ','.join(feats), 'key_fingerprint': key_fp, 'bundle_sha256': bundle_sha}
    summary = dict(row, out_dir=str(final_dir), zip_name=zip_name, warnings=warnings, ledger=str(ledger),
                   ledger_ok=True)
    try:
        append_ledger(Path(ledger), row)
    except Exception as e:
        summary['ledger_ok'] = False
        summary['ledger_error'] = f'{type(e).__name__}: {e}'
    return summary


def _print_summary(s: dict) -> None:
    bar = '=' * 72
    print(f'\n{bar}\n  CLINIC PROVISIONED  {s["customer"]}\n{bar}')
    for w in s.get('warnings') or []:
        print(f'  ! {w}')
    print(f'  folder          {s["out_dir"]}')
    print(f'  license.dat     tier {s["tier"]}, expires {s["expires"]}, features {s["features"]}')
    print(f'  model bundle    {s["zip_name"]}')
    print(f'  identity        {s["identity"]}')
    print(f'  machine         {s["fp8"]}…')
    print(f'  key fingerprint {s["key_fingerprint"]}  (keyinfo on the clinic PC must print the same)')
    print(f'  bundle sha256   {s["bundle_sha256"]}')
    print(f'  READMEs         README_admin_en.txt, README_admin_ru.txt')
    print(f'  ledger          {s["ledger"]}' + ('' if s['ledger_ok'] else '  — NOT UPDATED'))
    print(bar)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Provision a clinic PC: license.dat + per-machine encrypted model bundle')
    ap.add_argument('--fingerprint', required=True, help='machine id from `python -m src.utils.license fingerprint`')
    ap.add_argument('--customer', required=True, help='clinic name as it will appear on the license')
    ap.add_argument('--expires', required=True, help='YYYY-MM-DD (must be after today)')
    ap.add_argument('--tier', default='pro', choices=['trial', 'pro', 'enterprise'])
    ap.add_argument('--features', default=DEFAULT_FEATURES, help=f'comma-separated (default {DEFAULT_FEATURES})')
    ap.add_argument('--model', default=str(DEFAULT_MODEL), help='validated plaintext release dir')
    ap.add_argument('--private-key', default=str(DEFAULT_PRIVATE_KEY), help='vendor RSA private key (PEM)')
    ap.add_argument('--out', required=True, help='deliveries root; the folder <slug>_<fp8>/ is created inside')
    ap.add_argument('--ledger', default=str(DEFAULT_LEDGER), help=argparse.SUPPRESS)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        s = provision(fingerprint=args.fingerprint, customer=args.customer, expires=args.expires,
                      tier=args.tier, features=args.features, model=Path(args.model),
                      private_key=Path(args.private_key), out=Path(args.out), ledger=Path(args.ledger))
    except VerifyFailed as e:
        print(f'  ✗ FAILED [{e.code}] {e.message} — nothing was delivered', file=sys.stderr)
        return EXIT_FAILED
    except Refused as e:
        print(f'  ✗ REFUSED [{e.code}] {e.message}', file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:
        print(f'  ✗ FAILED [{type(e).__name__}] {e} — nothing was delivered', file=sys.stderr)
        return EXIT_FAILED
    _print_summary(s)
    if not s['ledger_ok']:
        print(f'  ✗ LEDGER NOT UPDATED ({s["ledger_error"]}) — add this row to {s["ledger"]} by hand:\n'
              f'    {",".join(str(s[k]) for k in LEDGER_FIELDS)}', file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
