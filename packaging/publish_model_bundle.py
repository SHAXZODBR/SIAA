#!/usr/bin/env python3
"""Bundle the validated triage model and (optionally) publish it as a private GitHub Release.

WHAT: stage 10 of the release pipeline (docs/dev/release_pipeline_contract.md). Takes
models/brain_triage_finetuned (or a promoted candidate dir), zips it in EXACTLY the layout
packaging/fetch_model_bundle.py restores (`brain_triage_finetuned/…` at the zip root, only the
files the app needs: MANIFEST.json, config.json, preprocessor_config.json, model.safetensors,
study_level_eval.json, training_metrics.json), hashes it, splits it into < 100 MB parts and
writes a multipart manifest `{"parts": [...], "sha256": ...}` next to them. With --github it
creates/updates a GitHub Release on the PRIVATE models repository through the REST API
(urllib only — `gh` is not installed on the build Macs), uploads the parts + manifest, prints
the manifest asset API URL (information only: the installer CI finds the release
brain-triage-<identity12> by the identity the code pins, so nothing is pasted into a secret per
release; the URL is the optional MODEL_BUNDLE_URL fallback) and proves the whole thing works by
restoring it with fetch_model_bundle.py into a temp dir.

WHY: the validated model is never committed and is not downloadable from anywhere public.
GitHub's upload endpoint times out on a single 330 MB asset, hence the parts; the manifest
asset is what CI downloads first (fetch_model_bundle.py sniffs "not a zip, JSON with parts").
Doing this by hand once per release is exactly the kind of step that gets done differently
every time — this script makes it one command, safe to re-run, and verified before any
installer build uses it.

RELEASE SAFETY (refusals exit 1 before anything is written or uploaded):
  * gate — <model>/study_level_eval.json must exist with gate.passed == true, gate.min_sens
    >= 0.85, gate.min_spec >= 0.40 (and sens/spec at or above those floors), promotion.promote
    == true when a promotion block exists, the gate must be the production 'gate mean @0.50',
    an eval that records a `protocol` block must also pass promote_model's protocol and
    release-val-set / locked-holdout rules (an exploratory re-eval with --gate-agg max,
    --gate-threshold 0.3 or --holdout none is refused), and the eval must be for THIS model (its
    MANIFEST pin); <model>/promotion_log.json must not record a --force promotion. --allow-unvalidated
    overrides with a loud banner (DEMO BUILDS ONLY) and the manifest then says
    "validated": false (fetch_model_bundle warns on it in CI). A passing model's manifest
    carries "validated": true plus the eval summary (sens, spec, n, confusion counts).
  * tag/identity — the default tag is brain-triage-<identity12>, so every model gets its own
    release. When the release for the tag already holds a manifest.json, it is downloaded
    (same auth) and its identity compared: the SAME identity with the same zip sha256 and all
    parts present is reused untouched (the manifest asset id does not change;
    --reupload forces a fresh upload); the same identity with a different zip is re-uploaded
    (the manifest asset id changes — CI resolves the release by tag and is unaffected; only a
    hand-set MODEL_BUNDLE_URL fallback would need the new URL); a DIFFERENT identity (or an
    unreadable manifest) is refused unless --replace-existing. Only assets whose names this
    run uploads are ever deleted (part names embed the identity; manifest.json is swapped
    last, after the new parts are up).
  * size — with --github, --part-mb must be >= 20 (and < 100); the uploaded manifest is
    checked against fetch_model_bundle's 1 MiB manifest limit BEFORE any upload, and a
    warning is printed above 32 KiB.

Usage:
    python packaging/publish_model_bundle.py --model models/brain_triage_finetuned --out data/release/v3/
    python packaging/publish_model_bundle.py --model models/brain_triage_finetuned --out data/release/v3/ \\
        --github SHAXZODBR/SIAA-models [--tag brain-triage-<identity12>] [--name "..."] [--notes "..."] [--part-mb 80]
    python packaging/publish_model_bundle.py ... --github SHAXZODBR/SIAA-models --dry-run

Outputs under --out:
    <zip_name>                 brain_triage_finetuned_<identity>.zip (kept: usable as a file:// bundle)
    <zip_name>.part-00 …       the parts (< --part-mb MB each)
    manifest.json              local multipart manifest, parts as file:// URLs (fetch_model_bundle-ready)
    manifest.github.json       (--github) the uploaded manifest + its asset URL (the optional MODEL_BUNDLE_URL fallback)

Token resolution (--github): --token, else $SIAA_MODELS_TOKEN, else $GITHUB_TOKEN, else
`git credential fill` for github.com (the keychain entry `git push` uses). The SOURCE is
printed, the token never is; it is scrubbed from every error message. The repo must be
private (checked first) unless --allow-public is given.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API = 'https://api.github.com'
USER_AGENT = 'sentinel-publish-model-bundle'
BUNDLE_FILES = ('MANIFEST.json', 'config.json', 'preprocessor_config.json', 'model.safetensors',
                'study_level_eval.json', 'training_metrics.json')
REQUIRED_FILES = ('MANIFEST.json', 'config.json', 'preprocessor_config.json', 'model.safetensors')
MANIFEST_ASSET = 'manifest.json'
MB = 1024 * 1024
MIN_PART_MB = 20                   # --github: smaller parts -> too many assets / a manifest CI may not sniff
MAX_PART_MB = 100                  # GitHub asset uploads time out above this
MANIFEST_WARN_BYTES = 32 * 1024    # warn: a manifest this big means an unusually small --part-mb
MANIFEST_MAX_BYTES = 1024 * 1024   # == fetch_model_bundle.MANIFEST_MAX_BYTES (tested); refuse above
GATE_FLOOR_SENS = 0.85             # release floors (docs/dev/release_pipeline_contract.md)
GATE_FLOOR_SPEC = 0.40
EVAL_JSON = 'study_level_eval.json'
PRODUCTION_GATE_NAME = 'gate mean @0.50'   # = scripts/promote_model.PRODUCTION_GATE_NAME
PROMOTION_LOG = 'promotion_log.json'
TAG_PREFIX = 'brain-triage-'
RETRIES = 3
API_TIMEOUT = 60
UPLOAD_TIMEOUT = 900
_SLEEP: Callable[[float], None] = time.sleep     # patched to a no-op in tests
_ACTIVE_TOKEN = ''                                # set by run(); used to scrub unexpected tracebacks


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _load_fetch():
    """packaging/ is not a package: import fetch_model_bundle by path (its zip layout and
    manifest format are the contract this script must produce)."""
    spec = importlib.util.spec_from_file_location('fetch_model_bundle', HERE / 'fetch_model_bundle.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scrub(text: str, token: str) -> str:
    """Never let the token reach stdout/stderr, even inside an exception message."""
    if token and text:
        text = text.replace(token, '***')
        text = text.replace(urllib.parse.quote(token, safe=''), '***')
    return text


class PublishError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# token
# ----------------------------------------------------------------------------

def _git_credential_fill(host: str = 'github.com') -> str:
    """Ask git's credential helper (macOS keychain) for the github.com password/token —
    the same credential `git push` uses. Empty string when there is none."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_ASKPASS='', SSH_ASKPASS='')
    try:
        r = subprocess.run(['git', 'credential', 'fill'], input=f'protocol=https\nhost={host}\n\n',
                           capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError):
        return ''
    if r.returncode != 0:
        return ''
    for line in r.stdout.splitlines():
        if line.startswith('password='):
            return line[len('password='):].strip()
    return ''


def resolve_token(explicit: str = '') -> tuple[str, str]:
    """Return (token, source). Order: --token, $SIAA_MODELS_TOKEN, $GITHUB_TOKEN, git credential fill."""
    if explicit and explicit.strip():
        return explicit.strip(), '--token'
    for var in ('SIAA_MODELS_TOKEN', 'GITHUB_TOKEN'):
        v = os.environ.get(var, '').strip()
        if v:
            return v, f'env {var}'
    v = _git_credential_fill()
    if v:
        return v, 'git credential fill (github.com)'
    return '', 'none'


# ----------------------------------------------------------------------------
# bundle: zip -> sha256 -> parts -> manifest
# ----------------------------------------------------------------------------

def collect_files(model_dir: Path) -> list[Path]:
    """The bundle files that exist, in BUNDLE_FILES order; required ones must be present.
    Everything else in the dir (.enc, _prev dirs, .DS_Store, checkpoints) is excluded."""
    missing = [n for n in REQUIRED_FILES if not (model_dir / n).is_file()]
    if missing:
        raise PublishError(f'{model_dir}: missing required file(s) {missing}')
    return [model_dir / n for n in BUNDLE_FILES if (model_dir / n).is_file()]


def build_zip(files: list[Path], zip_path: Path, root_name: str) -> int:
    """Write <root_name>/<file> entries (the layout fetch_model_bundle documents). The weights
    are stored uncompressed (float tensors do not deflate; deflating 330 MB only costs time)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(zip_path.suffix + '.tmp')
    with zipfile.ZipFile(tmp, 'w') as zf:
        for f in files:
            comp = zipfile.ZIP_STORED if f.suffix == '.safetensors' else zipfile.ZIP_DEFLATED
            zf.write(f, f'{root_name}/{f.name}', compress_type=comp)
    tmp.replace(zip_path)
    return zip_path.stat().st_size


def split_parts(zip_path: Path, part_bytes: int, out_dir: Path) -> list[Path]:
    """<zip>.part-00, .part-01 … each at most part_bytes; concatenating them in order gives
    back the zip byte-for-byte (fetch_model_bundle.download_multipart)."""
    if part_bytes <= 0:
        raise PublishError('--part-mb must be positive')
    for stale in out_dir.glob(zip_path.name + '.part-*'):
        stale.unlink()
    parts: list[Path] = []
    with open(zip_path, 'rb') as fin:
        i = 0
        while True:
            chunk_left = part_bytes
            part = out_dir / f'{zip_path.name}.part-{i:02d}'
            wrote = 0
            with open(part, 'wb') as fout:
                while chunk_left > 0:
                    buf = fin.read(min(1 << 20, chunk_left))
                    if not buf:
                        break
                    fout.write(buf)
                    wrote += len(buf)
                    chunk_left -= len(buf)
            if wrote == 0 and parts:
                part.unlink()
                break
            parts.append(part)
            i += 1
            if wrote < part_bytes:
                break
    return parts


def make_manifest(parts: list[str], sha256: str, identity: str, size_bytes: int, zip_name: str,
                  **extra: Any) -> dict[str, Any]:
    """The multipart manifest fetch_model_bundle.download_multipart reads: only `parts` and
    `sha256` are consumed by it, the rest is provenance for humans and the orchestrator."""
    m: dict[str, Any] = {
        'parts': list(parts),
        'sha256': sha256,
        'identity': identity,
        'created_at': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        'size_bytes': size_bytes,
        'zip_name': zip_name,
    }
    m.update(extra)
    return m


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f       # NaN -> None


def release_eval_check(model_dir: Path, identity: str) -> tuple[list[str], dict[str, Any] | None]:
    """Decide whether `model_dir` may be published as a VALIDATED release.

    Returns (problems, summary): an empty problem list means validated; `summary` is the eval
    summary for the manifest (None when the eval JSON is missing/unreadable). Mirrors
    scripts/promote_model.py's verdict so a --force-promoted, never-evaluated or
    exploratory-re-evaluated model cannot reach the installer by calling this script directly:
    besides gate.passed / promotion / floors / weights pin / forced log it checks the gate name
    ('gate mean @0.50') and — for every eval written by the release-pipeline
    eval_study_level.py (it records a `protocol` block) — promote_model.protocol_reasons()
    (aggregation, threshold, preprocessing) and promote_model.val_set_reasons() (release val
    set / locked holdout coverage). An eval without a `protocol` block predates the release
    pipeline (the v1 release, evaluated 2026-09-07); it is accepted on the gate-name check
    alone and marked `legacy_eval` in the summary."""
    problems: list[str] = []
    summary: dict[str, Any] | None = None
    ev_path = model_dir / EVAL_JSON
    ev: Any = None
    if not ev_path.is_file():
        problems.append(f'{EVAL_JSON} missing — the model was never evaluated against the release gate')
    else:
        try:
            ev = json.loads(ev_path.read_text(encoding='utf-8'))
        except (ValueError, OSError) as e:
            problems.append(f'{EVAL_JSON} unreadable ({e})')
        if ev is not None and not isinstance(ev, dict):
            problems.append(f'{EVAL_JSON} is not a JSON object')
            ev = None
    if isinstance(ev, dict):
        gate = ev.get('gate') if isinstance(ev.get('gate'), dict) else {}
        if not gate:
            problems.append(f'{EVAL_JSON} has no "gate" block')
        if gate.get('passed') is not True:
            problems.append(f'gate.passed is {json.dumps(gate.get("passed"))} (must be true)')
        promo = ev.get('promotion')
        if promo is not None and (not isinstance(promo, dict) or promo.get('promote') is not True):
            reason = promo.get('reason', '') if isinstance(promo, dict) else ''
            problems.append(f'promotion.promote is {json.dumps(promo.get("promote") if isinstance(promo, dict) else promo)}'
                            f' (must be true){": " + str(reason) if reason else ""}')
        for key, metric, floor in (('min_sens', 'sens', GATE_FLOOR_SENS), ('min_spec', 'spec', GATE_FLOOR_SPEC)):
            fl, val = _num(gate.get(key)), _num(gate.get(metric))
            if fl is None:
                problems.append(f'gate.{key} missing — cannot prove the release floor {floor} was applied')
            elif fl < floor - 1e-9:
                problems.append(f'gate.{key} = {fl:g} is below the release floor {floor}')
            if val is None:
                problems.append(f'gate.{metric} missing')
            elif val < floor - 1e-9:
                problems.append(f'gate.{metric} = {val:.3f} is below the release floor {floor}')
        pinned = ((ev.get('manifest') or {}).get('files') or {}).get('model.safetensors') \
            if isinstance(ev.get('manifest'), dict) else None
        if pinned and str(pinned)[:12] != identity:
            problems.append(f'{EVAL_JSON} was produced for model {str(pinned)[:12]}, not {identity} (stale eval)')
        if gate and gate.get('name') != PRODUCTION_GATE_NAME:
            problems.append(f'gate is {json.dumps(gate.get("name"))}, not the production protocol '
                            f'"{PRODUCTION_GATE_NAME}" (eval_study_level.py --gate-agg / --gate-threshold were changed)')
        legacy_eval = 'protocol' not in ev
        if not legacy_eval:
            problems.extend(_release_protocol_problems(ev))
        counts = {k: gate.get(k) for k in ('tp', 'fn', 'tn', 'fp')}
        n = ev.get('n_studies')
        if n is None and all(isinstance(v, int) for v in counts.values()):
            n = sum(counts.values())
        summary = {'sens': _num(gate.get('sens')), 'spec': _num(gate.get('spec')), 'n': n, **counts,
                   'min_sens': _num(gate.get('min_sens')), 'min_spec': _num(gate.get('min_spec')),
                   'passed': gate.get('passed'), 'evaluated_at': ev.get('evaluated_at')}
        for ci in ('sens_ci', 'spec_ci'):
            if gate.get(ci) is not None:
                summary[ci] = gate.get(ci)
        if isinstance(promo, dict):
            summary['promote'] = promo.get('promote')
        if legacy_eval:
            summary['legacy_eval'] = True
        else:
            summary['val_patients_sha256'] = ev.get('val_patients_sha256')
            summary['holdout_covered'] = ev.get('holdout_covered')
    log_path = model_dir / PROMOTION_LOG
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding='utf-8'))
        except (ValueError, OSError) as e:
            problems.append(f'{PROMOTION_LOG} unreadable ({e}) — cannot prove the promotion was not forced')
            log = None
        if isinstance(log, dict) and log.get('forced') is True:
            over = '; '.join(str(r) for r in log.get('refusal_reasons_overridden') or []) or 'no reasons recorded'
            problems.append(f'{PROMOTION_LOG} records a --force promotion (overridden: {over})')
    return problems, summary


def _release_protocol_problems(ev: dict) -> list[str]:
    """promote_model.protocol_reasons + val_set_reasons (minus the gate-name line, reported by
    the caller) — the SAME release rules promote enforces, not a re-implementation."""
    path = Path(__file__).resolve().parent.parent / 'scripts' / 'promote_model.py'
    try:
        spec = importlib.util.spec_from_file_location('_promote_model_rules', path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    except Exception as e:          # never publish as validated when the rules cannot be checked
        return [f'cannot load the release rules from {path} ({type(e).__name__}: {e})']
    holdout = mod.DEFAULT_HOLDOUT if Path(mod.DEFAULT_HOLDOUT).is_file() else None
    reasons = [r for r in mod.protocol_reasons(ev) if not r.startswith('gate is ')]
    reasons += mod.val_set_reasons(ev, holdout)
    return reasons


def default_tag(identity: str) -> str:
    """One release per model: brain-triage-<identity12>. A counter-based tag
    (brain-triage-v<N>) can be reused by accident and overwrite another model's release."""
    return f'{TAG_PREFIX}{identity}'


def _asset_id_from_url(url: str) -> int | None:
    m = re.search(r'/releases/assets/(\d+)/?$', str(url))
    return int(m.group(1)) if m else None


def read_remote_manifest(fetch_mod, url: str, token: str) -> dict[str, Any]:
    """Download an existing manifest.json asset through fetch_model_bundle.download (same
    bearer auth, drops it on the cross-host redirect GitHub answers with). Raises PublishError."""
    with tempfile.TemporaryDirectory(prefix='sentinel-publish-existing-') as td:
        path = Path(td) / 'manifest.json'
        try:
            fetch_mod.download(url, path, token)
        except Exception as e:
            raise PublishError(f'could not download the existing {MANIFEST_ASSET}: {type(e).__name__}: {scrub(str(e), token)}') from None
        if path.stat().st_size > MANIFEST_MAX_BYTES:
            raise PublishError(f'existing {MANIFEST_ASSET} is {path.stat().st_size} bytes — not a bundle manifest')
        try:
            data = json.loads(path.read_bytes().decode('utf-8-sig'))
        except ValueError as e:
            raise PublishError(f'existing {MANIFEST_ASSET} is not valid JSON ({e})') from None
    if not isinstance(data, dict):
        raise PublishError(f'existing {MANIFEST_ASSET} is not a JSON object')
    return data


def verify_restore(fetch_mod, url: str, token: str, expected_identity: str) -> str:
    """Restore through fetch_model_bundle.main into a temp dir; returns the restored identity.
    Raises PublishError on a non-zero exit or an identity mismatch."""
    with tempfile.TemporaryDirectory(prefix='sentinel-publish-verify-') as td:
        argv = ['--dest', td, '--url', url, '--require']
        if token:
            argv += ['--token', token]
        # fetch_model_bundle writes GITHUB_OUTPUT/STEP_SUMMARY when those vars exist — not ours to touch.
        saved = {k: os.environ.pop(k) for k in ('GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY') if k in os.environ}
        try:
            rc = fetch_mod.main(argv)
        finally:
            os.environ.update(saved)
        if rc != 0:
            raise PublishError(f'fetch_model_bundle restore FAILED (exit {rc}) for {urllib.parse.urlparse(url).netloc or "local file"}')
        ok, problems, identity = fetch_mod.verify_manifest(Path(td) / fetch_mod.MODEL_DIR_NAME)
        if not ok:
            raise PublishError('restored dir failed MANIFEST verification: ' + '; '.join(problems))
        if identity != expected_identity:
            raise PublishError(f'restored identity {identity} != source identity {expected_identity}')
        return identity


# ----------------------------------------------------------------------------
# GitHub REST (urllib only)
# ----------------------------------------------------------------------------

class GitHub:
    """Minimal client: JSON calls + streamed asset uploads, 3 attempts with backoff on
    5xx / network errors, json.loads(strict=False) (GitHub bodies may carry control chars)."""

    def __init__(self, token: str, owner: str, repo: str):
        self.token, self.owner, self.repo = token, owner, repo
        self.requests_made: list[tuple[str, str]] = []

    # -- transport -------------------------------------------------------
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {'Authorization': f'Bearer {self.token}', 'Accept': 'application/vnd.github+json',
             'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': USER_AGENT}
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def _retryable(e: BaseException) -> bool:
        if isinstance(e, urllib.error.HTTPError):
            return e.code >= 500
        return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))

    def _send(self, method: str, url: str, body_factory: Callable[[], Any] | None = None,
              headers: dict[str, str] | None = None, timeout: int = API_TIMEOUT,
              before_retry: Callable[[], None] | None = None) -> tuple[int, Any]:
        """Returns (status, parsed JSON or None). Raises PublishError with a scrubbed message."""
        self.requests_made.append((method, url))
        last: BaseException | None = None
        for attempt in range(1, RETRIES + 1):
            body = body_factory() if body_factory else None
            req = urllib.request.Request(url, data=body, method=method, headers=self._headers(headers))
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    status = getattr(resp, 'status', None) or resp.getcode()
                if hasattr(body, 'close'):
                    body.close()
                text = raw.decode('utf-8', 'replace') if raw else ''
                return int(status), (json.loads(text, strict=False) if text.strip() else None)
            except urllib.error.HTTPError as e:
                if hasattr(body, 'close'):
                    body.close()
                if 400 <= e.code < 500:      # 404 / 401 / 422: retrying cannot help
                    detail = ''
                    try:
                        detail = e.read().decode('utf-8', 'replace')[:300]
                    except Exception:
                        pass
                    raise PublishError(f'{method} {self._safe(url)} -> HTTP {e.code} {scrub(detail, self.token)}') from None
                last = e
            except Exception as e:      # URLError, timeouts, connection resets
                if hasattr(body, 'close'):
                    body.close()
                if not self._retryable(e):
                    raise PublishError(f'{method} {self._safe(url)} failed: {type(e).__name__}: {scrub(str(e), self.token)}') from None
                last = e
            if attempt < RETRIES:
                wait = 2.0 ** attempt
                print(f'{Y}  retry {attempt}/{RETRIES - 1} after {type(last).__name__}: '
                      f'{scrub(str(last), self.token)[:120]} (waiting {wait:.0f}s){NC}')
                _SLEEP(wait)
                if before_retry:
                    before_retry()
        raise PublishError(f'{method} {self._safe(url)} failed after {RETRIES} attempts: '
                           f'{type(last).__name__}: {scrub(str(last), self.token)}')

    def _safe(self, url: str) -> str:
        return scrub(url, self.token)

    def get(self, path_or_url: str) -> tuple[int, Any]:
        url = path_or_url if path_or_url.startswith('http') else API + path_or_url
        return self._send('GET', url)

    def get_or_none(self, path: str) -> Any:
        """GET that turns a 404 into None (everything else raises)."""
        try:
            return self.get(path)[1]
        except PublishError as e:
            if 'HTTP 404' in str(e):
                return None
            raise

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode('utf-8')
        return self._send('POST', API + path, lambda: data, {'Content-Type': 'application/json'})[1]

    def delete(self, path: str) -> None:
        self._send('DELETE', API + path)

    # -- release helpers ---------------------------------------------------
    def repo_info(self) -> dict[str, Any]:
        info = self.get_or_none(f'/repos/{self.owner}/{self.repo}')
        if info is None:
            raise PublishError(f'repository {self.owner}/{self.repo} not found (or the token cannot see it): '
                               f'create it as a PRIVATE repo with at least one commit, and give the token '
                               f'Contents read/write on it')
        return info

    def find_release(self, tag: str) -> dict[str, Any] | None:
        return self.get_or_none(f'/repos/{self.owner}/{self.repo}/releases/tags/{urllib.parse.quote(tag, safe="")}')

    def create_release(self, tag: str, name: str, notes: str) -> dict[str, Any]:
        return self.post_json(f'/repos/{self.owner}/{self.repo}/releases',
                              {'tag_name': tag, 'name': name or tag, 'body': notes or '',
                               'draft': False, 'prerelease': False})

    def get_or_create_release(self, tag: str, name: str, notes: str) -> tuple[dict[str, Any], bool]:
        rel = self.find_release(tag)
        if rel is not None:
            return rel, False
        return self.create_release(tag, name, notes), True

    def release_assets(self, release_id: int) -> list[dict[str, Any]]:
        _, assets = self.get(f'/repos/{self.owner}/{self.repo}/releases/{release_id}/assets?per_page=100')
        return list(assets or [])

    def delete_assets_named(self, release_id: int, names: set[str]) -> list[str]:
        deleted = []
        for a in self.release_assets(release_id):
            if a.get('name') in names:
                self.delete(f'/repos/{self.owner}/{self.repo}/releases/assets/{a["id"]}')
                deleted.append(str(a.get('name')))
        return deleted

    def upload_asset(self, release: dict[str, Any], path: Path, name: str) -> dict[str, Any]:
        """Stream the file from disk to upload_url?name=<name>. A failed attempt can leave a
        broken asset with that name behind (GitHub returns 502 mid-upload) — it is deleted
        before every retry so the re-attempt does not hit 422 'already_exists'."""
        base = str(release['upload_url']).split('{', 1)[0]
        url = f'{base}?name={urllib.parse.quote(name, safe="")}'
        size = path.stat().st_size
        headers = {'Content-Type': 'application/octet-stream', 'Content-Length': str(size)}

        def cleanup() -> None:
            try:
                self.delete_assets_named(int(release['id']), {name})
            except PublishError as e:
                print(f'{Y}  (cleanup before retry: {e}){NC}')

        status, asset = self._send('POST', url, lambda: open(path, 'rb'), headers,
                                   timeout=UPLOAD_TIMEOUT, before_retry=cleanup)
        if not isinstance(asset, dict) or 'id' not in asset:
            raise PublishError(f'upload of {name} returned HTTP {status} without an asset id')
        return asset

    def asset_api_url(self, asset_id: int | str) -> str:
        return f'{API}/repos/{self.owner}/{self.repo}/releases/assets/{asset_id}'


# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------

def _build_repo_hint() -> str:
    """owner/repo of THIS checkout's origin (for the 'where to paste the secret' steps)."""
    try:
        r = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], capture_output=True,
                           text=True, timeout=10, cwd=str(REPO))
        m = re.search(r'github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?\s*$', r.stdout or '')
        if r.returncode == 0 and m:
            return f'{m.group(1)}/{m.group(2)}'
    except (OSError, subprocess.SubprocessError):
        pass
    return '<build repo>'


def print_secret_steps(manifest_url: str, owner: str, repo: str, token_source: str) -> None:
    build_repo = _build_repo_hint()
    print()
    print(f'{G}MODEL_BUNDLE_URL={manifest_url}{NC}')      # information only (the optional fallback), one plain line
    print()
    print(f'{B}Nothing to paste per model release: the installer CI (build repo {build_repo}) finds the release '
          f'{TAG_PREFIX}<identity12> in {owner}/{repo} by the identity the code pins.{NC}')
    print(f'  1. MODEL_BUNDLE_TOKEN (https://github.com/{build_repo}/settings/environments -> environment '
          f'"MODEL_BUNDLE_URL") must be able to READ {owner}/{repo} (fine-grained PAT, "Contents: read" on that '
          f'repo); the build job\'s own GITHUB_TOKEN cannot read another repository. Leave it unchanged if the '
          f'existing one still works.')
    print(f'  2. CI reads the models repository from the repository variable MODEL_BUNDLE_REPO (unset: '
          f'SHAXZODBR/SIAA-models); it must name {owner}/{repo}. Only the default tag {TAG_PREFIX}<identity12> '
          f'is found: a release published under another --tag is invisible to CI.')
    print(f'  3. Commit AND push the scripts/bump_model_release.py changes that pin this identity '
          f'(src/inference/validated_metrics.FALLBACK_IDENTITY) BEFORE the version tag: CI resolves the release '
          f'from the pin of the tagged commit.')
    print(f'  4. Push the version tag (git tag vX.Y.Z && git push origin vX.Y.Z): the installers are attached to the '
          f'tag\'s DRAFT release; "Verify bundle identity matches the code" re-checks the restored identity.')
    print(f'  The MODEL_BUNDLE_URL line above is information only: the optional fallback, read only where no models '
          f'repository is configured (e.g. fetch_model_bundle.py on a build PC); it needs no editing per release.')
    print(f'  (this run authenticated with: {token_source})')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _parse_repo(spec: str) -> tuple[str, str]:
    m = re.fullmatch(r'\s*([\w.-]+)/([\w.-]+)\s*', spec or '')
    if not m:
        raise PublishError(f'--github expects owner/repo, got {spec!r}')
    return m.group(1), m.group(2)


def _mb(n: int) -> str:
    return f'{n / 1e6:.1f} MB'


def _banner_unvalidated(problems: list[str]) -> None:
    bar = '!' * 78
    print(f'{R}{bar}')
    print('!! --allow-unvalidated: PUBLISHING A MODEL THAT DID NOT EARN A VALIDATED RELEASE')
    for p in problems:
        print(f'!!   - {p}')
    print('!! manifest.json will say "validated": false and CI prints an UNVALIDATED warning.')
    print('!! DEMO BUILDS ONLY — never point a clinic installer build at this bundle.')
    print(f'{bar}{NC}')


def _gh_manifest_bytes(manifest: dict[str, Any]) -> int:
    return len((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))


def run(args: argparse.Namespace) -> int:
    fetch = _load_fetch()
    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        raise PublishError(f'--model {model_dir} is not a directory')
    files = collect_files(model_dir)
    ok, problems, identity = fetch.verify_manifest(model_dir)
    if not ok:
        raise PublishError(f'{model_dir}/MANIFEST.json verification failed: ' + '; '.join(problems))
    if not (model_dir / 'training_metrics.json').is_file():
        print(f'{Y}WARNING: training_metrics.json not present in {model_dir.name} — bundle will not carry it{NC}')

    # ---- release gate: only a validated model is published (unless --allow-unvalidated) ----
    gate_problems, eval_summary = release_eval_check(model_dir, identity)
    validated = not gate_problems
    if gate_problems:
        if not args.allow_unvalidated:
            raise PublishError(f'{model_dir.name} (identity {identity}) is NOT a validated release — refusing to bundle it:\n  - '
                               + '\n  - '.join(gate_problems)
                               + '\n  Fix: evaluate with scripts/eval_study_level.py --write-manifest and promote with '
                                 'scripts/promote_model.py (no --force). Demo builds only: --allow-unvalidated.')
        _banner_unvalidated(gate_problems)
    provenance: dict[str, Any] = {'validated': validated, 'eval': eval_summary}
    if not validated:
        provenance['validation_problems'] = gate_problems

    root_name = fetch.MODEL_DIR_NAME
    if model_dir.name != root_name:
        print(f'{Y}note: source dir is {model_dir.name}; zip root entry is {root_name}/ (what the installer expects){NC}')
    out_dir = Path(args.out).expanduser().resolve()
    zip_name = f'{root_name}_{identity}.zip'
    zip_path = out_dir / zip_name
    if not args.part_mb > 0:
        raise PublishError('--part-mb must be positive')
    part_bytes = int(args.part_mb * MB)
    if part_bytes <= 0:
        raise PublishError('--part-mb must be positive')
    total = sum(f.stat().st_size for f in files)
    n_parts_est = max(1, -(-total // part_bytes))

    github = bool(args.github)
    owner = repo = ''
    tag = args.tag or ''
    if github:
        owner, repo = _parse_repo(args.github)
        if not tag:
            tag = default_tag(identity)
        if part_bytes >= MAX_PART_MB * MB:
            raise PublishError(f'--part-mb must be < {MAX_PART_MB} (GitHub asset uploads time out above that)')
        if args.part_mb < MIN_PART_MB:
            raise PublishError(f'--part-mb {args.part_mb:g} is below {MIN_PART_MB}: that many parts makes a large '
                               f'manifest and many uploads; use {MIN_PART_MB}..{MAX_PART_MB - 1} (default 80)')
        # the uploaded manifest's size is known before any upload: API URLs with the longest
        # plausible asset ids. fetch_model_bundle refuses a manifest above MANIFEST_MAX_BYTES.
        est = _gh_manifest_bytes(make_manifest(
            [f'{API}/repos/{owner}/{repo}/releases/assets/{"9" * 12}'] * n_parts_est, '0' * 64, identity, total,
            zip_name, model_dir_name=root_name, part_names=[f'{zip_name}.part-{i:02d}' for i in range(n_parts_est)],
            part_sha256=['0' * 64] * n_parts_est, release={'repo': f'{owner}/{repo}', 'tag': tag, 'id': 10 ** 12},
            manifest_asset_url=f'{API}/repos/{owner}/{repo}/releases/assets/{"9" * 12}', **provenance))
        if est > MANIFEST_MAX_BYTES:
            raise PublishError(f'the GitHub manifest would be ~{est} bytes for {n_parts_est} parts (> {MANIFEST_MAX_BYTES}, '
                               f'fetch_model_bundle would refuse it) — use a larger --part-mb')
        if est > MANIFEST_WARN_BYTES:
            print(f'{Y}WARNING: the GitHub manifest will be ~{est // 1024} KiB ({n_parts_est} parts) — '
                  f'unusually large; a --part-mb near 80 keeps it ~2 KiB{NC}')

    print(f'{B}== publish_model_bundle{NC}')
    print(f'  model:     {model_dir}  identity {G}{identity}{NC}')
    for f in files:
        print(f'             {f.name:28s} {_mb(f.stat().st_size):>10s}')
    if validated and eval_summary:
        print(f'  gate:      {G}VALIDATED{NC}  sens {eval_summary["sens"]:.3f} spec {eval_summary["spec"]:.3f} '
              f'n {eval_summary.get("n")}  (floors {eval_summary.get("min_sens")}/{eval_summary.get("min_spec")})')
    else:
        print(f'  gate:      {R}UNVALIDATED (--allow-unvalidated){NC}')
    print(f'  out:       {out_dir}')
    print(f'  zip:       {zip_name}  (~{_mb(total)}, ~{n_parts_est} part(s) of <= {args.part_mb} MB)')

    global _ACTIVE_TOKEN
    token, token_source = '', 'none'
    if github:
        token, token_source = resolve_token(args.token)
        _ACTIVE_TOKEN = token
        print(f'  github:    {owner}/{repo}  tag {tag}{"  (default: brain-triage-<identity>)" if not args.tag else ""}'
              f'  token source: {token_source}')
        if args.tag and identity not in args.tag:
            print(f'{Y}  note: recommended tag is {default_tag(identity)} (one release per model identity){NC}')

    # ---- dry run: plan only, nothing written, no network ---------------------------------
    if args.dry_run:
        print(f'{Y}DRY RUN — nothing written, no network{NC}')
        print(f'  would write: {zip_path}')
        for i in range(n_parts_est):
            print(f'               {zip_path.name}.part-{i:02d}')
        print(f'               {out_dir / "manifest.json"}  (file:// parts, "validated": {json.dumps(validated)})')
        print(f'  would verify: fetch_model_bundle --dest <tmp> --url file://{out_dir / "manifest.json"}')
        if github:
            print(f'  would GET  {API}/repos/{owner}/{repo}  (refuse unless "private": true{", overridden by --allow-public" if args.allow_public else ""})')
            print(f'  would GET  {API}/repos/{owner}/{repo}/releases/tags/{tag}  (POST /releases if 404, draft=false)')
            print(f'  if that release holds {MANIFEST_ASSET}: download it and compare identity — same bundle -> reuse '
                  f'(nothing uploaded); different identity -> refuse{" (overridden: --replace-existing)" if args.replace_existing else " unless --replace-existing"}')
            names = [f'{zip_name}.part-{i:02d}' for i in range(n_parts_est)]
            print(f'  would upload (deleting only same-name assets first): {", ".join(names)}, then {MANIFEST_ASSET}')
            print(f'  would print MODEL_BUNDLE_URL={API}/repos/{owner}/{repo}/releases/assets/<manifest asset id>')
            print(f'  would verify: fetch_model_bundle --dest <tmp> --url <that URL> --token <{token_source}>')
            if not token:
                print(f'{Y}  WARNING: no token found — the real run would fail at the first API call{NC}')
        return 0

    if github and not token:
        raise PublishError('no GitHub token: pass --token, set SIAA_MODELS_TOKEN / GITHUB_TOKEN, or '
                           'store a github.com credential for git (git credential fill returned nothing)')

    # ---- local bundle ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    size = build_zip(files, zip_path, root_name)
    digest = sha256_file(zip_path)
    parts = split_parts(zip_path, part_bytes, out_dir)
    local_manifest = make_manifest([p.as_uri() for p in parts], digest, identity, size, zip_name,
                                   model_dir_name=root_name, part_names=[p.name for p in parts],
                                   part_sha256=[sha256_file(p) for p in parts], **provenance)
    local_manifest_path = out_dir / 'manifest.json'
    local_manifest_path.write_text(json.dumps(local_manifest, indent=2) + '\n', encoding='utf-8')
    lbytes = local_manifest_path.stat().st_size
    if lbytes > MANIFEST_MAX_BYTES:
        raise PublishError(f'local manifest is {lbytes} bytes (> {MANIFEST_MAX_BYTES}, fetch_model_bundle refuses it) — use a larger --part-mb')
    if lbytes > MANIFEST_WARN_BYTES:
        print(f'{Y}WARNING: local manifest is {lbytes // 1024} KiB ({len(parts)} parts) — use a larger --part-mb{NC}')
    print(f'{G}OK{NC} zip {zip_path.name}: {_mb(size)}, sha256 {digest[:12]}…, {len(parts)} part(s) '
          f'({time.time() - t0:.1f}s)')
    print(f'   manifest: {local_manifest_path}  ("validated": {json.dumps(validated)})')

    if args.skip_verify:
        print(f'{Y}   restore check skipped (--skip-verify){NC}')
    else:
        restored = verify_restore(fetch, local_manifest_path.as_uri(), '', identity)
        print(f'{G}OK{NC} local restore via fetch_model_bundle: identity {restored}')

    if not github:
        print(f'{G}DONE{NC} local bundle in {out_dir} — restore it on a build PC with '
              f'packaging/fetch_model_bundle.py --url {local_manifest_path.as_uri()}, '
              f'or re-run with --github owner/repo to publish.')
        return 0

    # ---- GitHub release ----------------------------------------------------------------
    gh = GitHub(token, owner, repo)
    gh_manifest_path = out_dir / 'manifest.github.json'
    try:
        info = gh.repo_info()
        if info.get('private') is not True:
            if not args.allow_public:
                raise PublishError(f'{owner}/{repo} is NOT private ("private": {json.dumps(info.get("private"))}) — '
                                   f'the validated model must not be published to a public repo (override: --allow-public)')
            print(f'{Y}WARNING: {owner}/{repo} is public; continuing because of --allow-public{NC}')
        else:
            print(f'{G}OK{NC} {owner}/{repo} is private')

        # -- tag/identity safety: look BEFORE touching anything --------------------------
        release = gh.find_release(tag)
        created = False
        reuse_url = ''
        if release is not None:
            rid = int(release['id'])
            assets = gh.release_assets(rid)
            masset = next((a for a in assets if a.get('name') == MANIFEST_ASSET), None)
            if masset is not None:
                old_url = gh.asset_api_url(masset['id'])
                try:
                    old = read_remote_manifest(fetch, old_url, token)
                    old_identity, unreadable = str(old.get('identity') or ''), ''
                except PublishError as e:
                    old, old_identity, unreadable = {}, '', str(e)
                if old_identity != identity:
                    what = (f'a {MANIFEST_ASSET} this run cannot read ({unreadable})' if unreadable else
                            f'{MANIFEST_ASSET} for identity {old_identity or "<none recorded>"}, not {identity}')
                    if not args.replace_existing:
                        raise PublishError(f'release {tag} already holds {what} — refusing to replace it: asset '
                                           f'{masset["id"]} may be the live MODEL_BUNDLE_URL, and that model could no '
                                           f'longer be restored for rollback. Use the default tag '
                                           f'{default_tag(identity)} or pass --replace-existing.')
                    print(f'{R}!! --replace-existing: release {tag} holds {what}; its {MANIFEST_ASSET} '
                          f'(asset {masset["id"]}) will be REPLACED after the new parts are up — any MODEL_BUNDLE_URL '
                          f'pointing at it stops working. Its other assets are left in place.{NC}')
                else:
                    present = {int(a['id']) for a in assets if 'id' in a}
                    ids = [_asset_id_from_url(u) for u in old.get('parts') or []]
                    complete = bool(ids) and all(i is not None and i in present for i in ids)
                    same = (complete and str(old.get('sha256', '')).lower() == digest
                            and old.get('validated') == validated)
                    if same and not args.reupload:
                        reuse_url = old_url
                    else:
                        why = ('--reupload' if same else 'incomplete previous upload' if not complete
                               else 'the zip changed (e.g. a re-evaluated study_level_eval.json)')
                        print(f'{Y}   release {tag} holds this identity ({identity}); re-uploading ({why}). The '
                              f'{MANIFEST_ASSET} asset id WILL CHANGE (CI resolves the release by its tag and is '
                              f'unaffected; only a hand-set MODEL_BUNDLE_URL fallback needs the new URL).{NC}')
            print(f'{G}OK{NC} release {tag}: exists (id {rid}) {release.get("html_url", "")}')
        else:
            release, created = gh.create_release(tag, args.name or '', args.notes or ''), True
            rid = int(release['id'])
            print(f'{G}OK{NC} release {tag}: created (id {rid}) {release.get("html_url", "")}')
        if release.get('draft'):
            print(f'{Y}   note: the release is a draft — publish it in the GitHub UI when ready{NC}')

        if reuse_url:
            gh_manifest = dict(old)
            gh_manifest['manifest_asset_url'] = reuse_url
            gh_manifest_path.write_text(json.dumps(gh_manifest, indent=2) + '\n', encoding='utf-8')
            print(f'{G}OK{NC} release {tag} already holds this exact bundle (identity {identity}, sha256 {digest[:12]}…) — '
                  f'nothing uploaded; MODEL_BUNDLE_URL (asset {masset["id"]}) is unchanged')
            manifest_url = reuse_url
        else:
            # parts first (their names embed the identity: only a same-model leftover is replaced) …
            part_names = [p.name for p in parts]
            removed = gh.delete_assets_named(rid, set(part_names))
            part_urls: list[str] = []
            for p in parts:
                t1 = time.time()
                asset = gh.upload_asset(release, p, p.name)
                part_urls.append(gh.asset_api_url(asset['id']))
                print(f'   uploaded {p.name} ({_mb(p.stat().st_size)}) -> asset {asset["id"]} ({time.time() - t1:.0f}s)')

            gh_manifest = make_manifest(part_urls, digest, identity, size, zip_name,
                                        model_dir_name=root_name, part_names=part_names,
                                        part_sha256=local_manifest['part_sha256'],
                                        release={'repo': f'{owner}/{repo}', 'tag': tag, 'id': rid}, **provenance)
            gh_manifest_path.write_text(json.dumps(gh_manifest, indent=2) + '\n', encoding='utf-8')
            gbytes = gh_manifest_path.stat().st_size
            if gbytes > MANIFEST_MAX_BYTES:       # the pre-flight estimate makes this unreachable in practice
                raise PublishError(f'GitHub manifest is {gbytes} bytes (> {MANIFEST_MAX_BYTES}) — not uploaded')
            if gbytes > MANIFEST_WARN_BYTES:
                print(f'{Y}WARNING: GitHub manifest is {gbytes // 1024} KiB (> {MANIFEST_WARN_BYTES // 1024} KiB){NC}')
            # … then swap manifest.json last, so the live one is replaced only once the new parts are up
            removed += gh.delete_assets_named(rid, {MANIFEST_ASSET})
            if removed:
                print(f'   replaced existing asset(s): {", ".join(removed)}')
            masset = gh.upload_asset(release, gh_manifest_path, MANIFEST_ASSET)
            manifest_url = gh.asset_api_url(masset['id'])
            gh_manifest['manifest_asset_url'] = manifest_url
            gh_manifest_path.write_text(json.dumps(gh_manifest, indent=2) + '\n', encoding='utf-8')
            print(f'   uploaded {MANIFEST_ASSET} -> asset {masset["id"]}')
            ours = set(part_names) | {MANIFEST_ASSET}
            others = [a.get('name') for a in gh.release_assets(rid) if a.get('name') not in ours]
            if others:
                print(f'{Y}   note: release also holds asset(s) this run did not touch: {others}{NC}')

        print_secret_steps(manifest_url, owner, repo, token_source)

        if args.skip_verify:
            print(f'{Y}   GitHub restore check skipped (--skip-verify){NC}')
        else:
            try:
                restored = verify_restore(fetch, manifest_url, token, identity)
            except PublishError as e:
                if reuse_url:
                    raise PublishError(f'{e} — the bundle already on release {tag} does not restore; '
                                       f're-run with --reupload') from None
                raise
            print(f'{G}OK{NC} end-to-end restore from GitHub via fetch_model_bundle: identity {restored}')
    except PublishError as e:
        raise PublishError(scrub(str(e), token)) from None

    action = 'verified existing' if reuse_url else 'published'
    print(f'{G}DONE{NC} {action} {zip_name} ({len(parts)} parts, validated={json.dumps(validated)}) in {owner}/{repo} '
          f'release {tag}{" (created)" if created else ""}; local copy + manifests in {out_dir}')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Zip + split the validated model and publish it as a private GitHub Release asset set')
    ap.add_argument('--model', default=str(REPO / 'models' / 'brain_triage_finetuned'),
                    help='model dir to bundle (default: models/brain_triage_finetuned)')
    ap.add_argument('--out', required=True, help='output dir, e.g. data/release/<tag>/')
    ap.add_argument('--part-mb', type=float, default=80,
                    help=f'part size in MiB (default 80; with --github {MIN_PART_MB} <= part-mb < {MAX_PART_MB})')
    ap.add_argument('--github', default='', metavar='OWNER/REPO', help='publish to this (private) GitHub repo')
    ap.add_argument('--tag', default='', help='release tag (default with --github: brain-triage-<identity12>)')
    ap.add_argument('--name', default='', help='release title (default: the tag)')
    ap.add_argument('--notes', default='', help='release body text')
    ap.add_argument('--token', default='', help='GitHub token (else $SIAA_MODELS_TOKEN, $GITHUB_TOKEN, git credential fill)')
    ap.add_argument('--allow-public', action='store_true', help='do not refuse a public repository')
    ap.add_argument('--allow-unvalidated', action='store_true',
                    help='DEMO BUILDS ONLY: bundle a model that failed/never ran the release gate or was --force-promoted '
                         '(manifest says "validated": false)')
    ap.add_argument('--replace-existing', action='store_true',
                    help="replace the manifest.json of a release that holds a DIFFERENT model's bundle")
    ap.add_argument('--reupload', action='store_true',
                    help='re-upload even when the release already holds this exact bundle (changes the manifest asset id)')
    ap.add_argument('--skip-verify', action='store_true', help='skip the fetch_model_bundle restore checks')
    ap.add_argument('--dry-run', action='store_true', help='print the plan; write nothing, no network')
    args = ap.parse_args(argv)
    try:
        return run(args)
    except PublishError as e:
        print(f'{R}ERROR: {scrub(str(e), _ACTIVE_TOKEN)}{NC}', file=sys.stderr)
        return 1
    except Exception as e:      # last line of defence: never let a token leak through a traceback
        print(f'{R}ERROR: {type(e).__name__}: {scrub(str(e), _ACTIVE_TOKEN)}{NC}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
