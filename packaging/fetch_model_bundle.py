#!/usr/bin/env python3
"""Restore the proprietary validated model into models/ for an installer build.

models/brain_triage_finetuned — the Uzbek-trained triage model, the product's only
validated detector — is NOT downloadable: scripts/download_all_models.py cannot fetch
it and it is never committed. An installer build gets it from a private zip whose URL
is the MODEL_BUNDLE_URL secret (a GitHub Release asset, a pre-signed object-store link,
a file:// path on a build PC):

    python packaging/fetch_model_bundle.py --dest models             # CI: URL from the env
    python packaging/fetch_model_bundle.py --dest models --url file:///C:/bundles/triage.zip
    python packaging/fetch_model_bundle.py --dest models --verify-only

Zip layout — the model directory at the top level, exactly as it sits under models/:

    brain_triage_finetuned/MANIFEST.json
    brain_triage_finetuned/config.json
    brain_triage_finetuned/preprocessor_config.json
    brain_triage_finetuned/model.safetensors
    (cd models && zip -r ../brain_triage_bundle.zip brain_triage_finetuned)

A zip made from INSIDE the directory (MANIFEST.json at the zip root) is accepted too.
After extraction every file pinned in MANIFEST.json is re-hashed (SHA-256): a corrupt or
swapped weight file fails the build instead of shipping. Without a URL the script prints
a loud warning and exits 0 — the installer is then built WITHOUT the validated model and
/health reports 'degraded' on every install made from it; --require turns that into a
failure.

Environment:
    MODEL_BUNDLE_URL      the zip URL (or --url)
    MODEL_BUNDLE_TOKEN    bearer token for it (or --token) — needed for a private GitHub
                          Release asset addressed by its API URL
                          https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>
    GITHUB_TOKEN          used automatically for api.github.com URLs when no
                          MODEL_BUNDLE_TOKEN is given (assets of the building repo itself)
    GITHUB_OUTPUT / GITHUB_STEP_SUMMARY   written when present:
                          have_bundle=0|1, model_identity=<sha256_12>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import url2pathname

MODEL_DIR_NAME = 'brain_triage_finetuned'
MANIFEST_NAME = 'MANIFEST.json'
_JUNK_PREFIXES = ('__MACOSX/',)
_JUNK_NAMES = ('.DS_Store', 'Thumbs.db')


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(model_dir: Path) -> tuple[bool, list[str], str]:
    """Re-hash every file MANIFEST.json pins.

    Returns (ok, problems, identity) — identity is the first 12 hex chars of the pinned
    model.safetensors hash, the model identity the app shows and prints on every PDF."""
    manifest = model_dir / MANIFEST_NAME
    if not manifest.is_file():
        return False, [f'{manifest} missing'], ''
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
    except ValueError as e:
        return False, [f'{manifest}: not valid JSON ({e})'], ''
    if data.get('algorithm', 'sha256') != 'sha256':
        return False, [f"unsupported manifest algorithm {data.get('algorithm')!r}"], ''
    files = data.get('files')
    if not isinstance(files, dict) or not files:
        return False, [f'{manifest} lists no files'], ''
    problems = []
    for name, pinned in files.items():
        p = model_dir / name
        if not p.is_file():
            problems.append(f'{name}: missing')
            continue
        actual = sha256_file(p)
        if actual != str(pinned):
            problems.append(f'{name}: sha256 {actual[:12]} != pinned {str(pinned)[:12]}')
    if 'model.safetensors' not in files:
        problems.append('model.safetensors is not pinned in MANIFEST.json')
    return not problems, problems, str(files.get('model.safetensors', ''))[:12]


# ----------------------------------------------------------------------------
# download / extract
# ----------------------------------------------------------------------------

class _CrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host. GitHub answers an
    authenticated asset request with a 302 to a pre-signed object-store URL, and that
    URL rejects requests which ALSO carry a bearer token ('only one auth mechanism')."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            new.remove_header('Authorization')
        return new


def download(url: str, out: Path, token: str = '') -> None:
    parsed = urlparse(url)
    if parsed.scheme == 'file':
        shutil.copyfile(url2pathname(parsed.path), out)
        return
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f'unsupported URL scheme {parsed.scheme!r}')
    req = urllib.request.Request(url, headers={'User-Agent': 'sentinel-installer-build'})
    if token:
        req.add_header('Authorization', f'Bearer {token}')
        req.add_header('Accept', 'application/octet-stream')   # GitHub API: the bytes, not the JSON
    opener = urllib.request.build_opener(_CrossHostRedirect())
    with opener.open(req, timeout=120) as resp, open(out, 'wb') as f:
        shutil.copyfileobj(resp, f, 1 << 20)


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in zf.infolist():
        name = info.filename.replace('\\', '/')
        parts = PurePosixPath(name).parts
        if (not parts or name.startswith('/') or '..' in parts
                or (len(parts[0]) == 2 and parts[0][1] == ':')):
            raise ValueError(f'refusing unsafe zip entry {info.filename!r}')
        if name.startswith(_JUNK_PREFIXES) or parts[-1] in _JUNK_NAMES or info.is_dir():
            continue
        members.append(info)
    return members


def extract_bundle(zip_path: Path, dest: Path) -> Path:
    """Extract so that dest/brain_triage_finetuned/ holds the model. Accepts the documented
    layout (brain_triage_finetuned/... at the zip root) or a zip made from inside the
    directory (MANIFEST.json at the root). Existing files are overwritten."""
    with zipfile.ZipFile(zip_path) as zf:
        members = _safe_members(zf)
        names = [m.filename.replace('\\', '/') for m in members]
        if any(n.startswith(MODEL_DIR_NAME + '/') for n in names):
            target = dest
        elif MANIFEST_NAME in names:
            target = dest / MODEL_DIR_NAME
        else:
            tops = sorted({n.split('/')[0] for n in names})[:10]
            raise ValueError(f'zip holds neither {MODEL_DIR_NAME}/ nor a top-level {MANIFEST_NAME}; '
                             f'top-level entries: {tops}')
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target, members=members)
    return dest / MODEL_DIR_NAME


# ----------------------------------------------------------------------------
# GitHub Actions plumbing (no-ops outside CI)
# ----------------------------------------------------------------------------

def _gh_append(env_var: str, line: str) -> None:
    path = os.environ.get(env_var)
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def _gh_output(key: str, value: str) -> None:
    _gh_append('GITHUB_OUTPUT', f'{key}={value}')


def _gh_summary(line: str) -> None:
    _gh_append('GITHUB_STEP_SUMMARY', line)


def _absent(require: bool, why: str) -> int:
    banner = '=' * 78
    print(banner)
    print('  WARNING: VALIDATED TRIAGE MODEL NOT RESTORED -- ' + why)
    print(f'  The installer is being built WITHOUT models/{MODEL_DIR_NAME}.')
    print("  Every install made from it will report /health status 'degraded' and the")
    print('  validated normal-vs-abnormal triage will be unavailable. Do NOT ship this build')
    print('  to a clinic. Set the MODEL_BUNDLE_URL secret (see docs/admin_install_guide_en.md).')
    print(banner)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::warning title=Validated model missing::{why} - installer built WITHOUT '
              f'models/{MODEL_DIR_NAME}; /health will report degraded on every install.')
    _gh_output('have_bundle', '0')
    _gh_summary(f'- :warning: Validated triage model: **MISSING** ({why}) — installer built '
                f'without `{MODEL_DIR_NAME}`; do not ship to a clinic')
    return 1 if require else 0


def _fail(msg: str) -> int:
    print(f'ERROR: {msg}', file=sys.stderr)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::error title=Validated model bundle::{msg}')
    _gh_output('have_bundle', '0')
    _gh_summary(f'- :x: Validated triage model bundle: **{msg}**')
    return 1


# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=f'Restore models/{MODEL_DIR_NAME} from a private zip and verify its MANIFEST')
    ap.add_argument('--dest', default=str(Path(__file__).resolve().parents[1] / 'models'),
                    help='models/ root the bundle is extracted into (default: <repo>/models)')
    ap.add_argument('--url', default=os.environ.get('MODEL_BUNDLE_URL', ''), help='zip URL (default: $MODEL_BUNDLE_URL)')
    ap.add_argument('--token', default=os.environ.get('MODEL_BUNDLE_TOKEN', ''), help='bearer token (default: $MODEL_BUNDLE_TOKEN)')
    ap.add_argument('--require', action='store_true', help='fail (exit 1) instead of warning when there is no bundle')
    ap.add_argument('--verify-only', action='store_true', help='no download: verify the MANIFEST of an existing dir')
    args = ap.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    model_dir = dest / MODEL_DIR_NAME

    if not args.verify_only:
        url = args.url.strip()
        if not url:
            return _absent(args.require, 'MODEL_BUNDLE_URL is not set')
        token = args.token.strip()
        if not token and urlparse(url).netloc == 'api.github.com':
            token = os.environ.get('GITHUB_TOKEN', '').strip()
        # Never echo the URL itself: a pre-signed link is a secret.
        print(f'downloading validated model bundle from {urlparse(url).netloc or "local file"} '
              f'-> {dest}  (auth: {"bearer token" if token else "none"})')
        with tempfile.TemporaryDirectory(prefix='sentinel-bundle-') as td:
            zip_path = Path(td) / 'bundle.zip'
            try:
                download(url, zip_path, token)
            except Exception as e:
                return _fail(f'download failed: {type(e).__name__}: {e}')
            print(f'  downloaded {zip_path.stat().st_size / 1e6:.0f} MB')
            try:
                extract_bundle(zip_path, dest)
            except (ValueError, zipfile.BadZipFile, OSError) as e:
                return _fail(f'extract failed: {e}')
    elif not model_dir.is_dir():
        return _absent(args.require, f'{model_dir} does not exist')

    ok, problems, identity = verify_manifest(model_dir)
    if not ok:
        return _fail(f'{model_dir}/{MANIFEST_NAME} verification failed: ' + '; '.join(problems))
    size_mb = sum(p.stat().st_size for p in model_dir.rglob('*') if p.is_file()) / 1e6
    print(f'OK validated model present: {model_dir} ({size_mb:.0f} MB), {MANIFEST_NAME} verified, '
          f'identity {identity}')
    _gh_output('have_bundle', '1')
    _gh_output('model_identity', identity)
    _gh_summary(f'- Validated triage model: **present** — `{MODEL_DIR_NAME}` identity `{identity}`, '
                f'{MANIFEST_NAME} verified ({size_mb:.0f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
