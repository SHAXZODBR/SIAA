#!/usr/bin/env python3
"""Restore the proprietary validated model into models/ for an installer build.

models/brain_triage_finetuned — the Uzbek-trained triage model, the product's only
validated detector — is NOT downloadable: scripts/download_all_models.py cannot fetch
it and it is never committed. An installer build gets it from a private bundle, found
in this order:

  1. --url on the command line (an operator restoring from a file:// path on a build PC,
     a GitHub Release asset, a pre-signed object-store link);
  2. by identity, when a models repository is configured (--repo / MODEL_BUNDLE_REPO):
     the release brain-triage-<identity12> of that repository (one release per model,
     named by packaging/publish_model_bundle.py), where <identity12> is --identity or,
     by default, the identity the code pins (src/inference/validated_metrics.py
     FALLBACK_IDENTITY, read as text — the application package is not imported). Its
     manifest.json asset is then downloaded exactly as if its API URL had been given
     with --url (but every part it lists must be on https://api.github.com too), and the
     restored model must have the requested identity. Nothing has to be edited per model
     release. With --require-newest (CI passes it on v* tags) the models repository's
     releases are listed once and the restore fails when a brain-triage-* release was
     created AFTER the requested one — the pin and the restored identity come from the
     same commit, so only this catches a tag pushed without the bump commit.
     --allow-older / MODEL_BUNDLE_ALLOW_OLDER=1 (deliberate rebuild of an older model)
     turns that failure into a loud warning; an unreadable release list always fails;
  3. the MODEL_BUNDLE_URL environment variable (the fallback; the way CI worked before).

    python packaging/fetch_model_bundle.py --dest models             # CI: MODEL_BUNDLE_REPO + token from the env
    python packaging/fetch_model_bundle.py --dest models --require --require-newest   # CI on a v* tag
    python packaging/fetch_model_bundle.py --dest models --repo SHAXZODBR/SIAA-models --identity 0d559766ce58
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
swapped weight file fails the build instead of shipping. Without a bundle source (no URL,
or a configured repository but no token to read it) the script prints a loud warning and
exits 0 — the installer is then built WITHOUT the validated model and /health reports
'degraded' on every install made from it; --require turns that into a failure (CI passes
it on v* tags). When a token exists but the lookup by identity fails (no such release,
token rejected, release without manifest.json) the script fails — it never falls back to
MODEL_BUNDLE_URL silently.

Environment:
    MODEL_BUNDLE_REPO     OWNER/NAME of the private models repository (or --repo);
                          resolution by identity, see above
    MODEL_BUNDLE_URL      the zip / manifest URL used when no repository is configured
                          (an explicit --url always wins)
    MODEL_BUNDLE_ALLOW_OLDER  1 = same as --allow-older (only meaningful with --require-newest)
    MODEL_BUNDLE_TOKEN    bearer token (or --token) — reads the models repository's release
                          and its assets; needed for a private GitHub Release asset addressed
                          by its API URL
                          https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>
    GITHUB_TOKEN          used automatically when no MODEL_BUNDLE_TOKEN is given: for
                          api.github.com URLs, and for the lookup by identity only when the
                          models repository IS the building repository ($GITHUB_REPOSITORY)
    GITHUB_OUTPUT / GITHUB_STEP_SUMMARY   written when present:
                          have_bundle=0|1, model_identity=<sha256_12>,
                          bundle_validated=0|1 (multipart manifests only: 0 when the
                          publisher marked the bundle "validated": false — demo builds)

Multipart manifest detection: a download that does not start with 'PK' is sniffed (first
64 KiB) and, when it looks like JSON, parsed whole — a manifest of any size up to 1 MiB is
accepted (MANIFEST_MAX_BYTES; packaging/publish_model_bundle.py refuses to publish a bigger one).

The bearer token is only ever attached to requests to the host of the bundle URL (the one
given, or the GitHub API URL the lookup resolved); a part on another host is fetched without
it (and refused outright for a bundle found through the models repository). Redirects to
another host drop it too (_CrossHostRedirect).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import url2pathname

MODEL_DIR_NAME = 'brain_triage_finetuned'
MANIFEST_NAME = 'MANIFEST.json'
_JUNK_PREFIXES = ('__MACOSX/',)
_JUNK_NAMES = ('.DS_Store', 'Thumbs.db')
MANIFEST_SNIFF_BYTES = 64 * 1024        # bytes looked at to tell a JSON manifest from a zip
MANIFEST_MAX_BYTES = 1024 * 1024        # largest multipart manifest accepted (publish enforces it too)
TAG_PREFIX = 'brain-triage-'            # == publish_model_bundle.TAG_PREFIX (tested): one release per model
MANIFEST_ASSET = 'manifest.json'        # == publish_model_bundle.MANIFEST_ASSET (tested)
GITHUB_API = 'https://api.github.com'
PIN_FILE = Path(__file__).resolve().parents[1] / 'src' / 'inference' / 'validated_metrics.py'
PIN_NAME = 'FALLBACK_IDENTITY'          # rewritten by scripts/bump_model_release.py
_IDENTITY_RE = re.compile(r'[0-9a-f]{12}')
_REPO_RE = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')


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


# ----------------------------------------------------------------------------
# resolution by identity: release brain-triage-<identity12> of the models repository
# ----------------------------------------------------------------------------

class ResolveError(RuntimeError):
    """The lookup by identity failed although a token was available (hard failure)."""


def _scrub(text: str, token: str) -> str:
    return text.replace(token, '***') if token else text


def read_pinned_identity(path: Path | None = None) -> str:
    """The identity the code pins (validated_metrics.FALLBACK_IDENTITY), read from the source
    text with ast — importing src.inference.validated_metrics would read the model dir and the
    environment at import time. Raises ValueError when the pin is missing or unreadable."""
    path = Path(path) if path is not None else PIN_FILE
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError, SyntaxError, ValueError) as e:
        raise ValueError(f'cannot read the code-pinned identity from {path}: {type(e).__name__}: {e}') from None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == PIN_NAME):
            value = node.value.value if isinstance(node.value, ast.Constant) else None
            if isinstance(value, str) and _IDENTITY_RE.fullmatch(value):
                return value
            raise ValueError(f'{path}: {PIN_NAME} is not a 12-hex-digit string literal')
    raise ValueError(f'{path}: no {PIN_NAME} = \'<12 hex>\' assignment found')


def _http_get(url: str, headers: dict[str, str]) -> bytes:
    """GET `url` and return the body; raises urllib.error.HTTPError / URLError. The single
    network call of the lookup (tests replace it with a fake)."""
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(_CrossHostRedirect())
    with opener.open(req, timeout=60) as resp:
        return resp.read(MANIFEST_MAX_BYTES * 4)


def _api_headers(token: str) -> dict[str, str]:
    return {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'sentinel-installer-build'}


def lookup_release(repo: str, identity: str, token: str) -> dict:
    """The release brain-triage-<identity> of `repo` (GitHub API JSON). Raises ResolveError with
    a message that never contains the token."""
    tag = f'{TAG_PREFIX}{identity}'
    url = f'{GITHUB_API}/repos/{repo}/releases/tags/{tag}'
    try:
        body = _http_get(url, _api_headers(token))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ResolveError(f'no release {tag} in {repo}, or the token cannot read that repository '
                               f'(expired?) (HTTP 404)') from None
        if e.code in (401, 403):
            raise ResolveError(f'token rejected (HTTP {e.code}) while looking up release {tag} in {repo}: '
                               f'renew MODEL_BUNDLE_TOKEN (Contents: read on {repo})') from None
        raise ResolveError(f'GitHub API answered HTTP {e.code} while looking up release {tag} in {repo}') from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, 'reason', e)
        raise ResolveError(f'cannot reach the GitHub API to look up release {tag} in {repo}: '
                           f'{type(e).__name__}: {_scrub(str(reason), token)}') from None
    try:
        release = json.loads(body.decode('utf-8', 'replace'), strict=False)
    except ValueError:
        raise ResolveError(f'the GitHub API answer for release {tag} in {repo} is not JSON') from None
    return release if isinstance(release, dict) else {}


def manifest_url_of(release: dict, repo: str, identity: str) -> str:
    """API URL of the manifest.json asset of `release` (brain-triage-<identity> in `repo`)."""
    tag = f'{TAG_PREFIX}{identity}'
    assets = release.get('assets') if isinstance(release, dict) else None
    asset = next((a for a in assets or [] if isinstance(a, dict) and a.get('name') == MANIFEST_ASSET), None)
    if asset is None:
        raise ResolveError(f'release {tag} exists in {repo} but holds no {MANIFEST_ASSET}: '
                           f're-run packaging/publish_model_bundle.py')
    murl = str(asset.get('url') or '')
    if not murl and asset.get('id') is not None:
        murl = f'{GITHUB_API}/repos/{repo}/releases/assets/{asset["id"]}'
    parsed = urlparse(murl)
    if parsed.scheme != 'https' or parsed.netloc != urlparse(GITHUB_API).netloc:
        # the bearer token is sent to this URL next: only ever to the GitHub API host
        raise ResolveError(f'release {tag} in {repo}: the {MANIFEST_ASSET} asset has no GitHub API URL')
    return murl


def resolve_manifest_url(repo: str, identity: str, token: str) -> str:
    """API URL of the manifest.json asset of release brain-triage-<identity> in `repo`.
    Raises ResolveError with a message that never contains the token."""
    return manifest_url_of(lookup_release(repo, identity, token), repo, identity)


def _created_at(value) -> _dt.datetime | None:
    """GitHub's ISO-8601 created_at ('2031-03-01T10:00:00Z') as an aware datetime, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = _dt.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=_dt.timezone.utc)


def newer_model_releases(repo: str, identity: str, pinned_release: dict, token: str) -> list[str]:
    """Tags of the published brain-triage-* releases of `repo` created AFTER the release of
    `identity` (newest first). One GET of /repos/<repo>/releases?per_page=100 (GitHub lists the
    newest first). Drafts and other tags are ignored. Anything unreadable raises ResolveError
    (never a silent pass); messages never contain the token."""
    tag = f'{TAG_PREFIX}{identity}'
    pinned_at = _created_at(pinned_release.get('created_at') if isinstance(pinned_release, dict) else None)
    if pinned_at is None:
        raise ResolveError(f'release {tag} in {repo} has no readable created_at: cannot compare it with '
                           f'the other releases of {repo}')
    url = f'{GITHUB_API}/repos/{repo}/releases?per_page=100'
    try:
        body = _http_get(url, _api_headers(token))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ResolveError(f'token rejected (HTTP {e.code}) while listing the releases of {repo}') from None
        raise ResolveError(f'GitHub API answered HTTP {e.code} while listing the releases of {repo}') from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, 'reason', e)
        raise ResolveError(f'cannot reach the GitHub API to list the releases of {repo}: '
                           f'{type(e).__name__}: {_scrub(str(reason), token)}') from None
    try:
        releases = json.loads(body.decode('utf-8', 'replace'), strict=False)
    except ValueError:
        raise ResolveError(f'the GitHub API answer listing the releases of {repo} is not JSON') from None
    if not isinstance(releases, list):
        raise ResolveError(f'the GitHub API answer listing the releases of {repo} is not a list')
    newer: list[tuple[_dt.datetime, str]] = []
    for rel in releases:
        if not isinstance(rel, dict):
            raise ResolveError(f'malformed entry in the GitHub API answer listing the releases of {repo}')
        name = rel.get('tag_name')
        if not isinstance(name, str) or not name.startswith(TAG_PREFIX) or name == tag or rel.get('draft') is True:
            continue
        when = _created_at(rel.get('created_at'))
        if when is None:
            raise ResolveError(f'malformed answer listing the releases of {repo}: release {name} has no '
                               f'readable created_at')
        if when > pinned_at:
            newer.append((when, name))
    return [name for _, name in sorted(newer, reverse=True)]


def sniff_manifest(path: Path) -> dict | None:
    """Return the parsed multipart manifest when `path` holds one, else None (a zip or
    anything else). Looks at the first MANIFEST_SNIFF_BYTES: a zip starts with 'PK'; a
    manifest starts with '{'. A JSON-looking file larger than MANIFEST_MAX_BYTES is an
    error, not a zip (the old 4096-byte sniff silently handed an 18-part manifest to
    zipfile -> 'File is not a zip file')."""
    with open(path, 'rb') as f:
        head = f.read(MANIFEST_SNIFF_BYTES)
    if head.startswith(b'PK'):
        return None
    if not head.lstrip(b'\xef\xbb\xbf \t\r\n').startswith(b'{'):
        return None
    size = path.stat().st_size
    if size > MANIFEST_MAX_BYTES:
        raise ValueError(f'bundle manifest is {size} bytes (> {MANIFEST_MAX_BYTES}); re-publish with '
                         f'larger parts (packaging/publish_model_bundle.py --part-mb)')
    try:
        maybe = json.loads(path.read_bytes().decode('utf-8-sig', 'ignore'))
    except ValueError:
        return None
    return maybe if isinstance(maybe, dict) and 'parts' in maybe else None


def download_multipart(manifest_url: str, out: Path, token: str = '', api_parts_only: bool = False) -> dict:
    """MODEL_BUNDLE_URL may point at a small JSON manifest instead of the zip itself —
    GitHub's upload endpoint times out on single large assets, so the zip is shipped as
    <100 MB parts:  {"parts": [<url>, ...], "sha256": "<hex of the whole zip>"}.
    Parts are fetched in order, concatenated, and the whole-zip hash verified. The token goes
    only to parts on the manifest's own host (scheme + host); `api_parts_only` (a bundle the
    models-repository lookup resolved) refuses, before any part is requested, a part that is not
    on https://api.github.com. Returns the manifest (provenance: identity, validated, eval summary)."""
    with tempfile.TemporaryDirectory(prefix='sentinel-bundle-parts-') as td:
        mpath = Path(td) / 'manifest.json'
        download(manifest_url, mpath, token)
        if mpath.stat().st_size > MANIFEST_MAX_BYTES:
            raise ValueError(f'bundle manifest is {mpath.stat().st_size} bytes (> {MANIFEST_MAX_BYTES})')
        try:
            manifest = json.loads(mpath.read_bytes().decode('utf-8-sig'))
            parts = list(manifest['parts']); expected = str(manifest.get('sha256', '')).lower()
        except Exception as e:
            raise ValueError(f'bad bundle manifest: {e}')
        if not parts:
            raise ValueError('bundle manifest lists no parts')
        parts = [str(u) for u in parts]
        base, api = urlparse(manifest_url), urlparse(GITHUB_API)
        if api_parts_only:
            for i, purl in enumerate(parts):
                pp = urlparse(purl)
                if pp.scheme != api.scheme or pp.netloc != api.netloc:
                    # never echo the URL (a pre-signed link is a secret), only its host
                    raise ValueError(f'bundle manifest part {i + 1}/{len(parts)} is on '
                                     f'{pp.scheme or "?"}://{pp.hostname or "?"}, not {GITHUB_API}: a bundle '
                                     f'found through the models repository is only fetched from the GitHub API '
                                     f'(nothing was requested from that host)')
        with open(out, 'wb') as fout:
            for i, purl in enumerate(parts):
                ppath = Path(td) / f'part-{i:02d}'
                pp = urlparse(purl)
                same_host = (pp.scheme, pp.netloc) == (base.scheme, base.netloc)
                download(purl, ppath, token if same_host else '')
                with open(ppath, 'rb') as fin:
                    shutil.copyfileobj(fin, fout, 1 << 20)
                print(f'  part {i + 1}/{len(parts)}: {ppath.stat().st_size / 1e6:.0f} MB')
        if expected:
            actual = sha256_file(out)
            if actual != expected:
                raise ValueError(f'assembled zip sha256 {actual[:12]} != manifest {expected[:12]}')
    return manifest


def _warn_unvalidated(manifest: dict) -> None:
    """The publisher marks a bundle made with --allow-unvalidated as "validated": false.
    Such a build is for demos only — say so loudly (the build itself is not failed)."""
    if manifest.get('validated') is not False:
        _gh_output('bundle_validated', '1')
        return
    why = '; '.join(str(p) for p in manifest.get('validation_problems') or []) or 'no reason recorded'
    banner = '!' * 78
    print(banner)
    print('  WARNING: this bundle was published with --allow-unvalidated ("validated": false).')
    print(f'  Reason(s): {why}')
    print('  DEMO BUILDS ONLY -- do NOT ship an installer made from it to a clinic.')
    print(banner)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::warning title=UNVALIDATED model bundle::published with --allow-unvalidated ({why}) - demo builds only')
    _gh_output('bundle_validated', '0')
    _gh_summary(f'- :warning: Model bundle is **UNVALIDATED** (published with --allow-unvalidated: {why}) — demo builds only')


def _warn_older(msg: str) -> None:
    """--require-newest overridden (--allow-older / MODEL_BUNDLE_ALLOW_OLDER=1): say it loudly."""
    banner = '!' * 78
    print(banner)
    print(f'  WARNING: {msg}')
    print('  Continuing because MODEL_BUNDLE_ALLOW_OLDER=1 / --allow-older: a DELIBERATE rebuild of an')
    print('  older model. Unset it again afterwards.')
    print(banner)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::warning title=Older model than the newest release::{msg} (allowed by MODEL_BUNDLE_ALLOW_OLDER)')
    _gh_summary(f'- :warning: {msg} — allowed by MODEL_BUNDLE_ALLOW_OLDER (deliberate rebuild of an older model)')


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
    print('  to a clinic. Set the MODEL_BUNDLE_TOKEN secret (see docs/admin_install_guide_en.md).')
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
    # --url has no environment default: an explicit --url must be told apart from $MODEL_BUNDLE_URL
    ap.add_argument('--url', default='', help='zip / manifest URL; wins over --repo and $MODEL_BUNDLE_URL')
    ap.add_argument('--repo', default=os.environ.get('MODEL_BUNDLE_REPO', ''), metavar='OWNER/NAME',
                    help=f'models repository: restore release {TAG_PREFIX}<identity> (default: $MODEL_BUNDLE_REPO)')
    ap.add_argument('--identity', default='', metavar='ID12',
                    help=f'model identity to resolve with --repo (default: {PIN_NAME} in src/inference/validated_metrics.py)')
    ap.add_argument('--token', default=os.environ.get('MODEL_BUNDLE_TOKEN', ''), help='bearer token (default: $MODEL_BUNDLE_TOKEN)')
    ap.add_argument('--require', action='store_true', help='fail (exit 1) instead of warning when there is no bundle')
    ap.add_argument('--require-newest', action='store_true',
                    help=f'with a repository lookup: fail (exit 1) when the repository holds a {TAG_PREFIX}* release '
                         f'created after the requested one (CI: v* tags)')
    ap.add_argument('--allow-older', action='store_true',
                    help='with --require-newest: only warn about a newer model release (deliberate rebuild of an '
                         'older model; also MODEL_BUNDLE_ALLOW_OLDER=1)')
    ap.add_argument('--verify-only', action='store_true', help='no download: verify the MANIFEST of an existing dir')
    args = ap.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    model_dir = dest / MODEL_DIR_NAME
    wanted = ''                 # identity requested by the lookup (empty: bundle given by URL)
    via_lookup = False          # bundle URL resolved by the repository lookup (parts: GitHub API only)

    if not args.verify_only:
        token = args.token.strip()
        url = args.url.strip()
        repo = args.repo.strip()
        if not url and repo:
            if not _REPO_RE.fullmatch(repo) or any(p in ('.', '..') for p in repo.split('/')):
                return _fail(f'models repository must be OWNER/NAME, got {repo!r}')
            if not token and repo.lower() == os.environ.get('GITHUB_REPOSITORY', '').strip().lower():
                token = os.environ.get('GITHUB_TOKEN', '').strip()
            if not token:
                return _absent(args.require, f'MODEL_BUNDLE_TOKEN is not set (needed to read release '
                                             f'{TAG_PREFIX}<identity> from {repo})')
            try:
                wanted = args.identity.strip() or read_pinned_identity()
            except ValueError as e:
                return _fail(str(e))
            if not _IDENTITY_RE.fullmatch(wanted):
                return _fail(f'--identity must be 12 lowercase hex digits, got {wanted!r}')
            print(f'resolving release {TAG_PREFIX}{wanted} in {repo} (auth: bearer token)')
            try:
                release = lookup_release(repo, wanted, token)
                url = manifest_url_of(release, repo, wanted)
                newer = newer_model_releases(repo, wanted, release, token) if args.require_newest else []
            except ResolveError as e:
                return _fail(_scrub(str(e), token))
            if newer:
                msg = (f'the code pins {wanted} but the models repository holds a newer model release {newer[0]}: '
                       f'commit and push the scripts/bump_model_release.py changes before tagging')
                if not (args.allow_older or os.environ.get('MODEL_BUNDLE_ALLOW_OLDER', '').strip() == '1'):
                    return _fail(msg)
                _warn_older(msg)
            via_lookup = True
        elif not url:
            url = os.environ.get('MODEL_BUNDLE_URL', '').strip()
        if not url:
            return _absent(args.require, 'MODEL_BUNDLE_URL is not set')
        if args.require_newest and not via_lookup:
            print('--require-newest: bundle given by URL, no models-repository lookup -> nothing to compare')
        if not token and urlparse(url).netloc == 'api.github.com':
            token = os.environ.get('GITHUB_TOKEN', '').strip()
        # Never echo the URL itself: a pre-signed link is a secret.
        print(f'downloading validated model bundle from {urlparse(url).netloc or "local file"} '
              f'-> {dest}  (auth: {"bearer token" if token else "none"})')
        with tempfile.TemporaryDirectory(prefix='sentinel-bundle-') as td:
            zip_path = Path(td) / 'bundle.zip'
            try:
                download(url, zip_path, token)
                # A GitHub asset URL carries no extension: sniff the bytes. A zip starts
                # with 'PK'; a JSON object with a 'parts' list (any size up to
                # MANIFEST_MAX_BYTES) is the multipart manifest (see download_multipart).
                if sniff_manifest(zip_path) is not None:
                    _warn_unvalidated(download_multipart(url, zip_path, token, api_parts_only=via_lookup))
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
    if wanted and identity != wanted:
        return _fail(f'restored identity {identity} != requested identity {wanted} '
                     f'(release {TAG_PREFIX}{wanted} holds another model)')
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
