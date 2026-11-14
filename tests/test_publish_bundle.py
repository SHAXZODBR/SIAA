"""packaging/publish_model_bundle.py — bundle the validated model and publish it as a private
GitHub Release, restorable by packaging/fetch_model_bundle.py.

Everything here is synthetic and offline: a tiny fake model dir with a valid MANIFEST.json,
file:// manifests for local mode, and an in-memory fake of the GitHub REST API (monkeypatched
urllib.request.urlopen / build_opener) that records every request so the tests can prove the
token never appears in a URL, a body or a printed line. No real model, no network.
"""

import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLISH = REPO / 'packaging' / 'publish_model_bundle.py'
FETCH = REPO / 'packaging' / 'fetch_model_bundle.py'
TOKEN = 'ghp_SECRET_token_value_0123456789'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pub():
    return _load(PUBLISH, 'publish_model_bundle')


@pytest.fixture
def fetch():
    return _load(FETCH, 'fetch_model_bundle')


def passing_eval(**gate_overrides) -> dict:
    """A synthetic study_level_eval.json in the layout scripts/eval_study_level.py writes."""
    gate = {'name': 'gate mean @0.50', 'sens': 0.9, 'spec': 0.5, 'acc': 0.8, 'balanced': 0.7,
            'tp': 90, 'fn': 10, 'tn': 25, 'fp': 25, 'min_sens': 0.85, 'min_spec': 0.4, 'passed': True}
    gate.update(gate_overrides)
    return {'gate': gate, 'n_studies': 150, 'n_slices': 750, 'evaluated_at': '2031-01-02T03:04:05'}


def make_model_dir(root: Path, name: str = 'brain_triage_finetuned', weights: bytes = b'',
                   eval_json: dict | None = None, pin_eval: bool = True) -> tuple[Path, str]:
    """Tiny synthetic model dir + a valid MANIFEST.json (hashes computed with hashlib), plus the
    junk a real models/ dir accumulates and the bundle must NOT carry. Returns (dir, identity).
    eval_json defaults to a PASSING gate; with pin_eval its manifest block pins these weights."""
    d = root / name
    d.mkdir(parents=True)
    weights = weights or bytes(range(256)) * 37 + b'\x01\x02\x03'     # odd size: 9475 bytes
    (d / 'config.json').write_text('{"architectures":["ViTForImageClassification"],"id2label":{"0":"abnormal","1":"normal"}}')
    (d / 'preprocessor_config.json').write_text('{"size":224}')
    (d / 'model.safetensors').write_bytes(weights)
    (d / 'training_metrics.json').write_text('{"epochs":12}')
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
             for n in ('config.json', 'preprocessor_config.json', 'model.safetensors')}
    (d / 'MANIFEST.json').write_text(json.dumps({'algorithm': 'sha256', 'files': files}, indent=2))
    ev = passing_eval() if eval_json is None else eval_json
    if pin_eval and isinstance(ev, dict) and 'manifest' not in ev:
        ev = dict(ev, manifest={'path': f'models/{name}/MANIFEST.json', 'files': files})
    (d / 'study_level_eval.json').write_text(json.dumps(ev))
    # junk that must be excluded
    (d / '.DS_Store').write_bytes(b'junk')
    (d / 'model.safetensors.enc').write_bytes(b'encrypted-junk')
    (d / 'checkpoint-100').mkdir()
    (d / 'checkpoint-100' / 'optimizer.pt').write_bytes(b'x')
    (d / '_prev_abc').mkdir()
    (d / '_prev_abc' / 'model.safetensors').write_bytes(b'old')
    return d, files['model.safetensors'][:12]


# ----------------------------------------------------------------------------
# fake GitHub REST API
# ----------------------------------------------------------------------------

class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status
        self.headers = {}

    def getcode(self):
        return self.status


class FakeGitHub:
    """Serves: repo info, release by tag (404 until created), create release, list/delete
    assets, upload assets (records the streamed body), and asset download by API URL
    (Accept: application/octet-stream) — everything publish + fetch need."""

    def __init__(self, owner='o', repo='r', private=True, token=TOKEN):
        self.owner, self.repo, self.private, self.token = owner, repo, private, token
        self.log: list[dict] = []
        self.releases: dict[int, dict] = {}
        self.assets: dict[int, dict] = {}
        self._next = 1000
        self.fail_uploads_once: set[str] = set()      # asset names whose first upload attempt 502s

    # -- helpers
    def _nid(self):
        self._next += 1
        return self._next

    def _release_json(self, rid):
        rel = dict(self.releases[rid])
        rel['assets'] = [self._asset_json(a) for a in self.assets.values() if a['release_id'] == rid]
        return rel

    def _asset_json(self, a):
        return {'id': a['id'], 'name': a['name'], 'size': len(a['data']),
                'url': f'https://api.github.com/repos/{self.owner}/{self.repo}/releases/assets/{a["id"]}'}

    @staticmethod
    def _err(url, code, msg):
        return urllib.error.HTTPError(url, code, msg, {}, io.BytesIO(json.dumps({'message': msg}).encode()))

    @staticmethod
    def _json(obj, status=200):
        return _Resp(json.dumps(obj).encode(), status)

    # -- entry point (urlopen / opener.open)
    def __call__(self, req, timeout=None, **_):
        url = req.full_url
        method = req.get_method()
        body = req.data
        if body is not None and hasattr(body, 'read'):
            body = body.read()
        headers = {k.lower(): v for k, v in req.header_items()}
        self.log.append({'method': method, 'url': url, 'headers': headers, 'body': body})
        assert headers.get('authorization') == f'Bearer {self.token}', 'every call must carry the token'
        p = urllib.parse.urlparse(url)
        path = p.path
        base = f'/repos/{self.owner}/{self.repo}'
        if p.netloc == 'uploads.github.com' and method == 'POST':
            rid = int(path.split('/releases/')[1].split('/')[0])
            name = urllib.parse.parse_qs(p.query)['name'][0]
            assert headers.get('content-type') == 'application/octet-stream'
            assert int(headers['content-length']) == len(body)
            if name in self.fail_uploads_once:
                self.fail_uploads_once.discard(name)
                # GitHub leaves a broken asset behind when the upload dies mid-way
                aid = self._nid()
                self.assets[aid] = {'id': aid, 'name': name, 'data': b'', 'release_id': rid, 'broken': True}
                raise self._err(url, 502, 'Bad Gateway')
            if any(a['name'] == name and a['release_id'] == rid for a in self.assets.values()):
                raise self._err(url, 422, 'Validation Failed: already_exists')
            aid = self._nid()
            self.assets[aid] = {'id': aid, 'name': name, 'data': body, 'release_id': rid}
            return self._json(self._asset_json(self.assets[aid]), 201)
        if p.netloc != 'api.github.com':
            raise self._err(url, 404, 'unknown host')
        if method == 'GET' and path == base:
            return self._json({'full_name': f'{self.owner}/{self.repo}', 'private': self.private})
        if method == 'GET' and path.startswith(base + '/releases/tags/'):
            tag = urllib.parse.unquote(path.rsplit('/', 1)[1])
            for rid, rel in self.releases.items():
                if rel['tag_name'] == tag:
                    return self._json(self._release_json(rid))
            raise self._err(url, 404, 'Not Found')
        if method == 'POST' and path == base + '/releases':
            payload = json.loads(body)
            assert payload['draft'] is False and payload['prerelease'] is False
            rid = self._nid()
            self.releases[rid] = {'id': rid, 'tag_name': payload['tag_name'], 'name': payload.get('name'),
                                  'body': payload.get('body'), 'draft': False, 'prerelease': False,
                                  'html_url': f'https://github.com/{self.owner}/{self.repo}/releases/tag/{payload["tag_name"]}',
                                  'upload_url': f'https://uploads.github.com/repos/{self.owner}/{self.repo}/releases/{rid}/assets{{?name,label}}'}
            return self._json(self._release_json(rid), 201)
        if method == 'GET' and path.startswith(base + '/releases/') and path.endswith('/assets'):
            rid = int(path.split('/releases/')[1].split('/')[0])
            return self._json([self._asset_json(a) for a in self.assets.values() if a['release_id'] == rid])
        if path.startswith(base + '/releases/assets/'):
            aid = int(path.rsplit('/', 1)[1])
            if aid not in self.assets:
                raise self._err(url, 404, 'Not Found')
            if method == 'DELETE':
                del self.assets[aid]
                return _Resp(b'', 204)
            if method == 'GET' and headers.get('accept') == 'application/octet-stream':
                return _Resp(self.assets[aid]['data'], 200)
            if method == 'GET':
                return self._json(self._asset_json(self.assets[aid]))
        raise self._err(url, 404, f'unhandled {method} {path}')


@pytest.fixture
def fake_github(monkeypatch, pub):
    fake = FakeGitHub()

    class _Opener:
        def open(self, req, timeout=None):
            return fake(req, timeout=timeout)

    monkeypatch.setattr(urllib.request, 'urlopen', fake)                 # publish_model_bundle's calls
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *h: _Opener())   # fetch_model_bundle's calls
    monkeypatch.setattr(pub, '_SLEEP', lambda s: None)
    monkeypatch.setattr(pub, 'MIN_PART_MB', 0)      # tiny synthetic models: allow 4 KiB parts in GitHub mode
    for var in ('SIAA_MODELS_TOKEN', 'GITHUB_TOKEN', 'MODEL_BUNDLE_URL', 'MODEL_BUNDLE_TOKEN', 'MODEL_BUNDLE_REPO',
                'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_ACTIONS'):
        monkeypatch.delenv(var, raising=False)
    return fake


# ----------------------------------------------------------------------------
# local mode
# ----------------------------------------------------------------------------

def test_local_mode_end_to_end_restorable_by_fetch(tmp_path, pub, fetch, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'models')
    out = tmp_path / 'release' / 'v9'
    rc = pub.main(['--model', str(model_dir), '--out', str(out), '--part-mb', '0.004'])   # 4 KiB parts -> 3 parts
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err

    zips = sorted(out.glob('*.zip'))
    assert len(zips) == 1 and zips[0].name == f'brain_triage_finetuned_{identity}.zip'
    with zipfile.ZipFile(zips[0]) as zf:
        names = sorted(zf.namelist())
    assert names == sorted(f'brain_triage_finetuned/{n}' for n in pub.BUNDLE_FILES), names
    assert not any('.DS_Store' in n or '.enc' in n or '_prev' in n or 'checkpoint' in n for n in names)

    parts = sorted(out.glob('*.zip.part-*'))
    assert [p.name for p in parts] == [f'{zips[0].name}.part-{i:02d}' for i in range(len(parts))]
    part_limit = int(0.004 * pub.MB)
    assert len(parts) >= 2 and all(p.stat().st_size <= part_limit for p in parts)
    assert all(p.stat().st_size == part_limit for p in parts[:-1])     # only the last part is short
    assert b''.join(p.read_bytes() for p in parts) == zips[0].read_bytes()

    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['parts'] == [p.as_uri() for p in parts]
    assert manifest['sha256'] == hashlib.sha256(zips[0].read_bytes()).hexdigest()
    assert manifest['identity'] == identity
    assert manifest['size_bytes'] == zips[0].stat().st_size
    assert manifest['zip_name'] == zips[0].name
    assert manifest['created_at'].endswith('+00:00')
    assert manifest['validated'] is True
    assert manifest['eval']['sens'] == 0.9 and manifest['eval']['spec'] == 0.5 and manifest['eval']['n'] == 150
    assert (manifest['eval']['tp'], manifest['eval']['fn'], manifest['eval']['tn'], manifest['eval']['fp']) == (90, 10, 25, 25)

    # the installer's restorer accepts it
    dest = tmp_path / 'dest'
    rc = fetch.main(['--dest', str(dest), '--url', (out / 'manifest.json').as_uri(), '--require'])
    assert rc == 0
    ok, problems, restored = fetch.verify_manifest(dest / 'brain_triage_finetuned')
    assert ok and restored == identity, problems
    assert (dest / 'brain_triage_finetuned' / 'model.safetensors').read_bytes() == (model_dir / 'model.safetensors').read_bytes()
    assert (dest / 'brain_triage_finetuned' / 'study_level_eval.json').is_file()
    assert 'OK' in captured.out and identity in captured.out


def test_zip_root_is_brain_triage_finetuned_even_for_a_candidate_dir(tmp_path, pub, fetch):
    model_dir, identity = make_model_dir(tmp_path / 'models', name='brain_triage_candidate_v9')
    out = tmp_path / 'out'
    assert pub.main(['--model', str(model_dir), '--out', str(out)]) == 0
    with zipfile.ZipFile(next(out.glob('*.zip'))) as zf:
        assert all(n.startswith('brain_triage_finetuned/') for n in zf.namelist())
    assert len(list(out.glob('*.part-*'))) == 1     # tiny model -> one part


def test_split_parts_odd_size_reassembles_exactly(tmp_path, pub):
    blob = os.urandom(10_007)
    src = tmp_path / 'b.zip'
    src.write_bytes(blob)
    parts = pub.split_parts(src, 1000, tmp_path)
    assert len(parts) == 11 and parts[-1].stat().st_size == 7
    assert [p.name for p in parts] == [f'b.zip.part-{i:02d}' for i in range(11)]
    assert b''.join(p.read_bytes() for p in parts) == blob
    # exact multiple: no trailing empty part; re-split removes stale parts
    src.write_bytes(blob[:10_000])
    parts = pub.split_parts(src, 1000, tmp_path)
    assert len(parts) == 10 and all(p.stat().st_size == 1000 for p in parts)
    assert len(list(tmp_path.glob('b.zip.part-*'))) == 10
    # smaller than one part
    src.write_bytes(b'abc')
    parts = pub.split_parts(src, 1000, tmp_path)
    assert len(parts) == 1 and parts[0].read_bytes() == b'abc'


def test_missing_required_file_or_bad_manifest_fails(tmp_path, pub, capsys):
    model_dir, _ = make_model_dir(tmp_path / 'models')
    (model_dir / 'model.safetensors').write_bytes(b'tampered')
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o')]) == 1
    assert 'verification failed' in capsys.readouterr().err
    (model_dir / 'preprocessor_config.json').unlink()
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o')]) == 1
    assert 'missing required' in capsys.readouterr().err


def test_dry_run_writes_nothing_and_needs_no_network(tmp_path, pub, monkeypatch, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'models')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network!')))
    monkeypatch.setenv('SIAA_MODELS_TOKEN', TOKEN)
    out = tmp_path / 'out'
    rc = pub.main(['--model', str(model_dir), '--out', str(out), '--github', 'o/r', '--tag', 't1', '--dry-run'])
    o = capsys.readouterr().out
    assert rc == 0 and not out.exists()
    assert 'DRY RUN' in o and 'env SIAA_MODELS_TOKEN' in o and 'releases/assets/<manifest asset id>' in o
    assert TOKEN not in o


# ----------------------------------------------------------------------------
# token resolution
# ----------------------------------------------------------------------------

def test_token_resolution_order(tmp_path, pub, monkeypatch):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    git = fake_bin / 'git'
    git.write_text('#!/bin/sh\n'
                   'if [ "$1" = "credential" ] && [ "$2" = "fill" ]; then\n'
                   '  cat > /dev/null\n'
                   '  printf "protocol=https\\nhost=github.com\\nusername=x\\npassword=ghp_from_keychain\\n"\n'
                   'else\n  exit 1\nfi\n')
    git.chmod(git.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv('PATH', f'{fake_bin}{os.pathsep}{os.environ.get("PATH", "")}')
    monkeypatch.setenv('SIAA_MODELS_TOKEN', 'tok_siaa')
    monkeypatch.setenv('GITHUB_TOKEN', 'tok_gh')

    assert pub.resolve_token('tok_cli') == ('tok_cli', '--token')
    assert pub.resolve_token('') == ('tok_siaa', 'env SIAA_MODELS_TOKEN')
    monkeypatch.delenv('SIAA_MODELS_TOKEN')
    assert pub.resolve_token('') == ('tok_gh', 'env GITHUB_TOKEN')
    monkeypatch.setenv('GITHUB_TOKEN', '   ')      # blank counts as unset
    assert pub.resolve_token('') == ('ghp_from_keychain', 'git credential fill (github.com)')
    git.write_text('#!/bin/sh\nexit 1\n')
    assert pub.resolve_token('') == ('', 'none')


def test_scrub_removes_token_and_its_urlencoded_form(pub):
    tok = 'ghp_abc/def+ghi'
    msg = f'failed https://x?t={urllib.parse.quote(tok, safe="")} Bearer {tok}'
    s = pub.scrub(msg, tok)
    assert tok not in s and urllib.parse.quote(tok, safe='') not in s and s.count('***') == 2
    assert pub.scrub('plain', '') == 'plain'


# ----------------------------------------------------------------------------
# GitHub mode
# ----------------------------------------------------------------------------

def _run_github(pub, model_dir, out, extra=()):
    return pub.main(['--model', str(model_dir), '--out', str(out), '--part-mb', '0.004',
                     '--github', 'o/r', '--tag', 'brain-triage-v9', '--name', 'Brain triage v9',
                     '--notes', 'sens 0.90 spec 0.47', *extra])


def test_github_mode_creates_release_uploads_parts_and_prints_secret_url(tmp_path, pub, fetch, fake_github, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'models')
    out = tmp_path / 'out'
    rc = _run_github(pub, model_dir, out, ['--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    fake = fake_github

    # release: looked up by tag (404) then created, not a draft
    methods = [(e['method'], urllib.parse.urlparse(e['url']).path) for e in fake.log]
    assert ('GET', '/repos/o/r') == methods[0]
    assert ('GET', '/repos/o/r/releases/tags/brain-triage-v9') in methods
    assert ('POST', '/repos/o/r/releases') in methods
    assert len(fake.releases) == 1
    rel = next(iter(fake.releases.values()))
    assert rel['tag_name'] == 'brain-triage-v9' and rel['name'] == 'Brain triage v9' and rel['draft'] is False

    # assets: every part + manifest.json, bodies byte-identical to the local files
    parts = sorted(out.glob('*.zip.part-*'))
    by_name = {a['name']: a for a in fake.assets.values()}
    assert set(by_name) == {p.name for p in parts} | {'manifest.json'}
    for p in parts:
        assert by_name[p.name]['data'] == p.read_bytes()
    uploaded_manifest = json.loads(by_name['manifest.json']['data'])
    expected_part_urls = [f'https://api.github.com/repos/o/r/releases/assets/{by_name[p.name]["id"]}' for p in parts]
    assert uploaded_manifest['parts'] == expected_part_urls
    assert uploaded_manifest['sha256'] == json.loads((out / 'manifest.json').read_text())['sha256']
    assert uploaded_manifest['identity'] == identity

    # the printed secret value is the manifest asset's API URL
    manifest_url = f'https://api.github.com/repos/o/r/releases/assets/{by_name["manifest.json"]["id"]}'
    assert f'MODEL_BUNDLE_URL={manifest_url}' in cap.out
    assert 'settings/environments' in cap.out and 'MODEL_BUNDLE_TOKEN' in cap.out
    assert 'Nothing to paste per model release' in cap.out and 'brain-triage-<identity12> in o/r' in cap.out
    assert 'token source: --token' in cap.out
    local_gh = json.loads((out / 'manifest.github.json').read_text())
    assert local_gh['manifest_asset_url'] == manifest_url and local_gh['parts'] == expected_part_urls

    # the end-to-end verification really pulled the bytes back through the API
    assert ('GET', f'/repos/o/r/releases/assets/{by_name["manifest.json"]["id"]}') in methods
    assert 'end-to-end restore from GitHub' in cap.out

    # the token appears in no URL, no body, no printed line — only in the Authorization header
    for e in fake.log:
        assert TOKEN not in e['url']
        assert e['body'] is None or TOKEN.encode() not in e['body']
    assert TOKEN not in cap.out and TOKEN not in cap.err

    # independent proof: fetch_model_bundle restores from the printed URL with the token
    dest = tmp_path / 'ci_models'
    assert fetch.main(['--dest', str(dest), '--url', manifest_url, '--token', TOKEN, '--require']) == 0
    assert fetch.verify_manifest(dest / 'brain_triage_finetuned')[2] == identity


def _assets(fake):
    return {a['name']: (a['id'], a['data']) for a in fake.assets.values()}


def test_github_mode_rerun_same_bundle_reuses_release_and_keeps_secret_url(tmp_path, pub, fake_github, capsys, monkeypatch):
    """Finding :553 — a same-model re-run must NOT delete + re-upload manifest.json (that
    silently invalidated the asset id pasted into MODEL_BUNDLE_URL). The first run also
    survives a 502 mid-upload (the broken leftover is cleaned before the retry)."""
    monkeypatch.setenv('SIAA_MODELS_TOKEN', TOKEN)
    model_dir, identity = make_model_dir(tmp_path / 'models')
    out = tmp_path / 'out'
    fake_github.fail_uploads_once.add(f'brain_triage_finetuned_{identity}.zip.part-00')
    assert _run_github(pub, model_dir, out) == 0
    cap = capsys.readouterr()
    assert 'retry 1/2' in cap.out and 'token source: env SIAA_MODELS_TOKEN' in cap.out
    assert not any(a.get('broken') for a in fake_github.assets.values())
    first = _assets(fake_github)
    first_url = f'https://api.github.com/repos/o/r/releases/assets/{first["manifest.json"][0]}'
    assert f'MODEL_BUNDLE_URL={first_url}' in cap.out

    # second run, same model: nothing deleted, nothing uploaded, same URL printed, restore re-verified
    fake_github.log.clear()
    assert _run_github(pub, model_dir, out) == 0
    cap = capsys.readouterr()
    methods = [(e['method'], urllib.parse.urlparse(e['url']).netloc) for e in fake_github.log]
    assert not any(m == 'DELETE' for m, _ in methods)
    assert not any(host == 'uploads.github.com' for _, host in methods)
    assert ('POST', 'api.github.com') not in methods                     # no new release either
    assert _assets(fake_github) == first
    assert f'MODEL_BUNDLE_URL={first_url}' in cap.out and 'nothing uploaded' in cap.out
    assert 'end-to-end restore from GitHub' in cap.out and TOKEN not in cap.out + cap.err
    assert json.loads((out / 'manifest.github.json').read_text())['manifest_asset_url'] == first_url

    # --reupload: every asset of THIS run replaced (no duplicates, no leftovers), loud secret warning
    fake_github.log.clear()
    assert _run_github(pub, model_dir, out, ['--reupload']) == 0
    cap = capsys.readouterr()
    second = _assets(fake_github)
    assert set(second) == set(first)
    assert all(second[n][0] != first[n][0] for n in first)
    assert 'replaced existing asset' in cap.out and 'WILL CHANGE' in cap.out
    assert len(fake_github.releases) == 1


def test_github_mode_same_identity_changed_zip_is_reuploaded(tmp_path, pub, fake_github, capsys):
    """Same weights but a re-evaluated study_level_eval.json -> different zip: re-upload (allowed)."""
    model_dir, identity = make_model_dir(tmp_path / 'models')
    assert _run_github(pub, model_dir, tmp_path / 'out', ['--token', TOKEN]) == 0
    first = _assets(fake_github)
    ev = json.loads((model_dir / 'study_level_eval.json').read_text())
    ev['evaluated_at'] = '2031-02-03T04:05:06'
    (model_dir / 'study_level_eval.json').write_text(json.dumps(ev))
    capsys.readouterr()
    assert _run_github(pub, model_dir, tmp_path / 'out', ['--token', TOKEN]) == 0
    cap = capsys.readouterr()
    assert 'the zip changed' in cap.out and 'WILL CHANGE' in cap.out
    assert _assets(fake_github)['manifest.json'][0] != first['manifest.json'][0]
    assert json.loads(_assets(fake_github)['manifest.json'][1])['eval']['evaluated_at'] == '2031-02-03T04:05:06'


def test_github_mode_refuses_tag_holding_a_different_model(tmp_path, pub, fetch, fake_github, capsys):
    """Finding :553 — a reused tag (e.g. brain-triage-v<N> after a _prev_ folder was deleted)
    must not destroy the other model's manifest.json; --replace-existing swaps ONLY the
    assets this run names, and only after the new parts are up."""
    old_dir, old_id = make_model_dir(tmp_path / 'm_old')
    new_dir, new_id = make_model_dir(tmp_path / 'm_new', weights=b'\x07' * 5000)
    assert old_id != new_id
    assert _run_github(pub, old_dir, tmp_path / 'o1', ['--token', TOKEN]) == 0
    before = _assets(fake_github)
    capsys.readouterr()

    fake_github.log.clear()
    rc = _run_github(pub, new_dir, tmp_path / 'o2', ['--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 1, cap.out + cap.err
    assert f'identity {old_id}, not {new_id}' in cap.err and '--replace-existing' in cap.err
    assert _assets(fake_github) == before                                # untouched
    assert not any(e['method'] == 'DELETE' for e in fake_github.log)
    assert not any(urllib.parse.urlparse(e['url']).netloc == 'uploads.github.com' for e in fake_github.log)
    # the existing manifest was read with the same bearer auth, as raw bytes
    mid = before['manifest.json'][0]
    reads = [e for e in fake_github.log if urllib.parse.urlparse(e['url']).path.endswith(f'/releases/assets/{mid}')]
    assert reads and all(e['headers'].get('accept') == 'application/octet-stream' for e in reads)
    assert TOKEN not in cap.out + cap.err

    # explicit override
    fake_github.log.clear()
    rc = _run_github(pub, new_dir, tmp_path / 'o2', ['--token', TOKEN, '--replace-existing'])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    after = _assets(fake_github)
    old_parts = {n for n in before if n != 'manifest.json'}
    assert all(after[n] == before[n] for n in old_parts)                 # never delete what this run did not name
    assert after['manifest.json'][0] != mid
    assert json.loads(after['manifest.json'][1])['identity'] == new_id
    deletes = [urllib.parse.urlparse(e['url']).path for e in fake_github.log if e['method'] == 'DELETE']
    assert deletes == [f'/repos/o/r/releases/assets/{mid}']
    order = [(e['method'], urllib.parse.parse_qs(urllib.parse.urlparse(e['url']).query).get('name', [''])[0])
             for e in fake_github.log if e['method'] in ('DELETE', 'POST')]
    assert order[-2][0] == 'DELETE' and order[-1] == ('POST', 'manifest.json')   # manifest swapped last
    assert 'REPLACED' in cap.out and 'did not touch' in cap.out


def test_github_mode_refuses_unreadable_existing_manifest(tmp_path, pub, fake_github, capsys):
    model_dir, _ = make_model_dir(tmp_path / 'models')
    rid = 555
    fake_github.releases[rid] = {'id': rid, 'tag_name': 'brain-triage-v9', 'name': 'x', 'body': '', 'draft': False,
                                 'prerelease': False, 'html_url': 'h',
                                 'upload_url': f'https://uploads.github.com/repos/o/r/releases/{rid}/assets{{?name,label}}'}
    fake_github.assets[777] = {'id': 777, 'name': 'manifest.json', 'data': b'not json at all', 'release_id': rid}
    rc = _run_github(pub, model_dir, tmp_path / 'out', ['--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 1 and 'cannot read' in cap.err
    assert fake_github.assets[777]['data'] == b'not json at all'


def test_github_default_tag_is_identity_based(tmp_path, pub, fake_github, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'models')
    rc = pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out'), '--part-mb', '0.004',
                   '--github', 'o/r', '--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    rel = next(iter(fake_github.releases.values()))
    assert rel['tag_name'] == f'brain-triage-{identity}' == pub.default_tag(identity)
    assert json.loads(_assets(fake_github)['manifest.json'][1])['release']['tag'] == f'brain-triage-{identity}'
    # an explicit counter tag still works, with a recommendation
    capsys.readouterr()
    assert _run_github(pub, model_dir, tmp_path / 'out2', ['--token', TOKEN]) == 0
    assert f'recommended tag is brain-triage-{identity}' in capsys.readouterr().out


# ----------------------------------------------------------------------------
# release gate (finding :448)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('case,ev,expect', [
    ('gate_failed', passing_eval(passed=False, sens=0.80), 'gate.passed is false'),
    ('promotion_refused', dict(passing_eval(), promotion={'promote': False, 'reason': 'sens dropped'}), 'promotion.promote is false'),
    ('weak_sens_floor', passing_eval(min_sens=0.80), 'gate.min_sens = 0.8 is below the release floor 0.85'),
    ('weak_spec_floor', passing_eval(min_spec=0.30), 'gate.min_spec = 0.3 is below the release floor 0.4'),
    ('sens_below_floor', passing_eval(sens=0.84), 'gate.sens = 0.840 is below the release floor 0.85'),
    ('floors_missing', {'gate': {'passed': True, 'sens': 0.9, 'spec': 0.47}}, 'gate.min_sens missing'),
    ('no_gate', {'results': []}, 'no "gate" block'),
])
def test_publish_refuses_unvalidated_model(tmp_path, pub, monkeypatch, capsys, case, ev, expect):
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network!')))
    model_dir, _ = make_model_dir(tmp_path / 'models', eval_json=ev)
    out = tmp_path / 'out'
    rc = pub.main(['--model', str(model_dir), '--out', str(out), '--github', 'o/r', '--token', TOKEN])
    err = capsys.readouterr().err
    assert rc == 1 and 'NOT a validated release' in err and expect in err, err
    assert not out.exists()                                              # refused before anything is written
    assert pub.main(['--model', str(model_dir), '--out', str(out)]) == 1   # local mode too
    assert not out.exists()


def test_publish_refuses_missing_eval_forced_promotion_and_stale_eval(tmp_path, pub, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'a')
    (model_dir / 'study_level_eval.json').unlink()
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o')]) == 1
    assert 'never evaluated' in capsys.readouterr().err

    model_dir, identity = make_model_dir(tmp_path / 'b')
    (model_dir / 'promotion_log.json').write_text(json.dumps(
        {'forced': True, 'refusal_reasons_overridden': ['gate.passed is false'], 'new_identity': identity}))
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o')]) == 1
    assert 'records a --force promotion' in capsys.readouterr().err
    (model_dir / 'promotion_log.json').write_text(json.dumps({'forced': False, 'new_identity': identity}))
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o')]) == 0     # an honest promotion is fine
    capsys.readouterr()

    ev = passing_eval()
    ev['manifest'] = {'files': {'model.safetensors': 'f' * 64}}              # eval of ANOTHER model
    model_dir, identity = make_model_dir(tmp_path / 'c', eval_json=ev)
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'o2')]) == 1
    assert f'produced for model ffffffffffff, not {identity}' in capsys.readouterr().err


def test_allow_unvalidated_bundles_with_banner_and_validated_false(tmp_path, pub, fetch, monkeypatch, capsys):
    model_dir, identity = make_model_dir(tmp_path / 'models', eval_json=passing_eval(passed=False, sens=0.8))
    (model_dir / 'promotion_log.json').write_text(json.dumps({'forced': True}))
    out = tmp_path / 'out'
    rc = pub.main(['--model', str(model_dir), '--out', str(out), '--allow-unvalidated'])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert '--allow-unvalidated' in cap.out and 'DEMO BUILDS ONLY' in cap.out and 'UNVALIDATED' in cap.out
    m = json.loads((out / 'manifest.json').read_text())
    assert m['validated'] is False and m['eval']['sens'] == 0.8
    assert any('gate.passed' in p for p in m['validation_problems'])
    assert any('--force promotion' in p for p in m['validation_problems'])
    # CI restore of such a bundle warns loudly and reports bundle_validated=0
    gh_out = tmp_path / 'gh_output.txt'
    monkeypatch.setenv('GITHUB_OUTPUT', str(gh_out))
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    assert fetch.main(['--dest', str(tmp_path / 'dest'), '--url', (out / 'manifest.json').as_uri()]) == 0
    o = capsys.readouterr().out
    assert '::warning title=UNVALIDATED model bundle::' in o and 'DEMO BUILDS ONLY' in o
    assert 'bundle_validated=0' in gh_out.read_text() and f'model_identity={identity}' in gh_out.read_text()


def test_release_eval_check_accepts_the_real_validated_model_if_present(pub, fetch):
    real = REPO / 'models' / 'brain_triage_finetuned'
    if not (real / 'study_level_eval.json').is_file() or not (real / 'MANIFEST.json').is_file():
        pytest.skip('validated model not present in this checkout')
    ok, problems, identity = fetch.verify_manifest(real)
    assert ok, problems
    problems, summary = pub.release_eval_check(real, identity)
    assert problems == [] and summary['sens'] >= 0.85 and summary['spec'] >= 0.40 and summary['n'] > 0


# ----------------------------------------------------------------------------
# part size / manifest size (finding :468)
# ----------------------------------------------------------------------------

def test_github_refuses_part_mb_below_minimum_before_network(tmp_path, pub, fake_github, monkeypatch, capsys):
    monkeypatch.setattr(pub, 'MIN_PART_MB', 20)
    model_dir, _ = make_model_dir(tmp_path / 'models')
    rc = pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out'), '--part-mb', '19',
                   '--github', 'o/r', '--token', TOKEN])
    assert rc == 1 and 'below 20' in capsys.readouterr().err and not fake_github.log
    rc = pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out'), '--part-mb', '100',
                   '--github', 'o/r', '--token', TOKEN])
    assert rc == 1 and 'must be < 100' in capsys.readouterr().err and not fake_github.log
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out'), '--part-mb', '0']) == 1
    assert 'positive' in capsys.readouterr().err


def test_github_manifest_size_is_checked_before_any_upload(tmp_path, pub, fake_github, monkeypatch, capsys):
    model_dir, _ = make_model_dir(tmp_path / 'models')
    monkeypatch.setattr(pub, 'MANIFEST_MAX_BYTES', 3000)                 # 3 parts ~= 2.5 KB, 20 parts >> 3 KB
    rc = pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out'), '--part-mb', '0.0005',
                   '--github', 'o/r', '--token', TOKEN])
    assert rc == 1 and 'use a larger --part-mb' in capsys.readouterr().err
    assert not fake_github.log and not (tmp_path / 'out').exists()
    monkeypatch.setattr(pub, 'MANIFEST_MAX_BYTES', 1024 * 1024)
    monkeypatch.setattr(pub, 'MANIFEST_WARN_BYTES', 1000)
    assert _run_github(pub, model_dir, tmp_path / 'out', ['--token', TOKEN]) == 0
    assert 'WARNING: the GitHub manifest will be' in capsys.readouterr().out


def test_publish_and_fetch_agree_on_the_manifest_size_limit(pub, fetch):
    assert pub.MANIFEST_MAX_BYTES == fetch.MANIFEST_MAX_BYTES == 1024 * 1024
    assert fetch.MANIFEST_SNIFF_BYTES == 64 * 1024
    assert pub.MIN_PART_MB == 20


def _manifest_bundle(tmp_path: Path, pad: int) -> tuple[Path, str]:
    """A real multipart bundle (3 file:// parts) whose manifest is padded to `pad` extra bytes —
    as a 18-part GitHub manifest (> 4096 bytes) is in reality."""
    model_dir, identity = make_model_dir(tmp_path / 'src')
    zpath = tmp_path / 'b.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        for n in ('MANIFEST.json', 'config.json', 'preprocessor_config.json', 'model.safetensors'):
            zf.write(model_dir / n, f'brain_triage_finetuned/{n}')
    data = zpath.read_bytes()
    step = -(-len(data) // 3)
    parts = []
    for i in range(3):
        pp = tmp_path / f'b.zip.part-{i:02d}'
        pp.write_bytes(data[i * step:(i + 1) * step])
        parts.append(pp.as_uri())
    man = tmp_path / 'manifest.json'
    man.write_text(json.dumps({'parts': parts, 'sha256': hashlib.sha256(data).hexdigest(), 'identity': identity,
                               'part_names': ['x' * 60] * (pad // 64)}, indent=2))
    return man, identity


@pytest.mark.parametrize('pad', [5_000, 200_000])      # > the old 4096-byte sniff; > the 64 KiB sniff window
def test_fetch_accepts_large_multipart_manifest(tmp_path, fetch, pad, monkeypatch):
    for var in ('GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_ACTIONS'):
        monkeypatch.delenv(var, raising=False)
    man, identity = _manifest_bundle(tmp_path, pad)
    assert man.stat().st_size > max(4096, pad)
    dest = tmp_path / 'dest'
    assert fetch.main(['--dest', str(dest), '--url', man.as_uri(), '--require']) == 0
    assert fetch.verify_manifest(dest / 'brain_triage_finetuned')[2] == identity


def test_fetch_refuses_manifest_over_one_mib_with_a_clear_error(tmp_path, fetch, monkeypatch, capsys):
    for var in ('GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_ACTIONS'):
        monkeypatch.delenv(var, raising=False)
    man, _ = _manifest_bundle(tmp_path, 1_100_000)
    assert man.stat().st_size > fetch.MANIFEST_MAX_BYTES
    assert fetch.main(['--dest', str(tmp_path / 'dest'), '--url', man.as_uri()]) == 1
    err = capsys.readouterr().err
    assert 'bundle manifest is' in err and '--part-mb' in err and 'not a zip' not in err


def test_github_mode_refuses_public_repo_unless_allowed(tmp_path, pub, fake_github, capsys):
    fake_github.private = False
    model_dir, _ = make_model_dir(tmp_path / 'models')
    out = tmp_path / 'out'
    rc = _run_github(pub, model_dir, out, ['--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 1 and 'NOT private' in cap.err
    assert not fake_github.releases and not fake_github.assets
    assert all(e['method'] == 'GET' for e in fake_github.log)
    assert TOKEN not in cap.out + cap.err
    # explicit override
    assert _run_github(pub, model_dir, out, ['--token', TOKEN, '--allow-public']) == 0
    assert 'continuing because of --allow-public' in capsys.readouterr().out


def test_github_mode_without_any_token_fails_before_network(tmp_path, pub, fake_github, monkeypatch, capsys):
    monkeypatch.setenv('PATH', str(tmp_path))            # no git -> credential fill yields nothing
    model_dir, _ = make_model_dir(tmp_path / 'models')
    rc = _run_github(pub, model_dir, tmp_path / 'out')
    cap = capsys.readouterr()
    assert rc == 1 and 'no GitHub token' in cap.err and not fake_github.log


def test_github_mode_api_error_message_is_scrubbed(tmp_path, pub, fake_github, capsys):
    """A 4xx body that echoes the token (as some proxies do) must not reach the terminal."""
    model_dir, _ = make_model_dir(tmp_path / 'models')
    orig = fake_github.__call__

    def leaky(req, timeout=None, **kw):
        if req.get_method() == 'POST' and urllib.parse.urlparse(req.full_url).path == '/repos/o/r/releases':
            raise urllib.error.HTTPError(req.full_url, 403, 'Forbidden', {},
                                         io.BytesIO(f'{{"message":"bad token {TOKEN}"}}'.encode()))
        return orig(req, timeout=timeout, **kw)

    fake_github.__class__ = type('Leaky', (FakeGitHub,), {'__call__': lambda self, req, timeout=None, **kw: leaky(req, timeout, **kw)})
    rc = _run_github(pub, model_dir, tmp_path / 'out', ['--token', TOKEN])
    cap = capsys.readouterr()
    assert rc == 1 and 'HTTP 403' in cap.err and TOKEN not in cap.out + cap.err and '***' in cap.err


# ----------------------------------------------------------------------------
# release protocol (review round 2, publish_model_bundle.py:258)
# ----------------------------------------------------------------------------

def _modern_eval(**over) -> dict:
    """What the release-pipeline eval_study_level.py writes on the legacy val dir."""
    holdout = REPO / 'data' / 'holdout_patients_v1.txt'
    ev = dict(passing_eval(),
              protocol={'aggregation': 'mean', 'threshold': 0.5, 'n_central_slices': 5,
                        'preprocessing': 'registry_predictor_native_res_bilinear_224'},
              data='data/local_finetune_bin/val', val_patients_sha256='ab' * 32,
              holdout_covered=True, holdout_missing=[],
              holdout={'file': str(holdout),
                       'sha256': hashlib.sha256(holdout.read_bytes()).hexdigest() if holdout.is_file() else None})
    ev.update(over)
    return ev


def test_release_eval_check_accepts_a_production_protocol_eval(tmp_path, pub):
    model_dir, identity = make_model_dir(tmp_path / 'm', eval_json=_modern_eval())
    problems, summary = pub.release_eval_check(model_dir, identity)
    assert problems == [] and summary['holdout_covered'] is True and 'legacy_eval' not in summary


@pytest.mark.parametrize('over,expect', [
    ({'gate': dict(passing_eval()['gate'], name='gate max @0.30'),
      'protocol': {'aggregation': 'max', 'threshold': 0.3,
                   'preprocessing': 'registry_predictor_native_res_bilinear_224'}},
     ['not the production protocol', "aggregation/threshold is 'max'"]),
    ({'protocol': {'aggregation': 'mean', 'threshold': 0.5, 'preprocessing': 'pil_bicubic_224'}},
     ['protocol.preprocessing']),
    ({'holdout_covered': None, 'holdout': None}, ['holdout_covered is None']),
    ({'holdout_covered': False, 'holdout_missing': ['9000001']}, ['holdout_covered is False']),
    ({'val_patients_sha256': None}, ['no val_patients_sha256']),
])
def test_publish_refuses_an_exploratory_re_eval(tmp_path, pub, capsys, over, expect):
    """An exploratory eval_study_level.py run that overwrote the release eval (other aggregation /
    threshold / subset / --holdout none) must not be published as validated."""
    model_dir, identity = make_model_dir(tmp_path / 'm', eval_json=_modern_eval(**over))
    problems, _ = pub.release_eval_check(model_dir, identity)
    for e in expect:
        assert any(e in p for p in problems), (e, problems)
    assert pub.main(['--model', str(model_dir), '--out', str(tmp_path / 'out')]) == 1
    assert 'NOT a validated release' in capsys.readouterr().err and not (tmp_path / 'out').exists()


def test_publish_refuses_a_non_production_gate_name_even_on_a_legacy_eval(tmp_path, pub):
    model_dir, identity = make_model_dir(tmp_path / 'm', eval_json=passing_eval(name='gate vote @0.50'))
    problems, _ = pub.release_eval_check(model_dir, identity)
    assert any('not the production protocol' in p for p in problems)


def test_ci_never_ships_an_unvalidated_bundle_on_a_release_tag():
    """fetch_model_bundle emits bundle_validated=0 for an --allow-unvalidated bundle: the workflow must
    refuse it on a v* tag (review round 2) and never upload it as a workflow artifact of the public
    repository (2026-11-10: only model-less installers are artifacts; the draft release refuses it too)."""
    import yaml
    wf = yaml.safe_load((REPO / '.github' / 'workflows' / 'build-windows.yml').read_text(encoding='utf-8'))
    for job in ('windows', 'macos'):
        steps = wf['jobs'][job]['steps']
        names = [s.get('name', '') for s in steps]
        refuse = next(s for s in steps if s.get('name') == 'Refuse an unvalidated model bundle on a release tag')
        assert "steps.bundle.outputs.bundle_validated == '0'" in refuse['if']
        assert "startsWith(github.ref, 'refs/tags/v')" in refuse['if']
        assert names.index(refuse['name']) < names.index('Upload installer artifact')
        assert names.index(refuse['name']) < names.index('Attach to draft GitHub Release (tag builds)')
        upload = next(s for s in steps if s.get('name') == 'Upload installer artifact')
        assert upload['if'] == "steps.bundle.outputs.have_bundle != '1'"          # a model-bearing build: never
        assert upload['with']['name'].endswith('-NO-VALIDATED-MODEL')
        attach = next(s for s in steps if s.get('name') == 'Attach to draft GitHub Release (tag builds)')
        assert "steps.bundle.outputs.bundle_validated != '0'" in attach['if']


def test_publish_and_fetch_share_the_release_tag_and_manifest_asset_names(pub, fetch):
    """CI finds a model's release by name (fetch_model_bundle.resolve_manifest_url): the two scripts
    must never drift apart on the tag prefix or the manifest asset name."""
    assert fetch.TAG_PREFIX == pub.TAG_PREFIX == 'brain-triage-'
    assert fetch.MANIFEST_ASSET == pub.MANIFEST_ASSET == 'manifest.json'
    assert pub.default_tag('0d559766ce58') == f'{fetch.TAG_PREFIX}0d559766ce58'


def test_printed_steps_say_nothing_is_pasted_per_release(pub, monkeypatch, capsys):
    monkeypatch.setattr(pub, '_build_repo_hint', lambda: 'SHAXZODBR/SIAA')
    url = 'https://api.github.com/repos/o/r/releases/assets/123'
    pub.print_secret_steps(url, 'o', 'r', 'env SIAA_MODELS_TOKEN')
    out = capsys.readouterr().out
    assert f'MODEL_BUNDLE_URL={url}' in out                                    # still printed, as information
    assert 'Nothing to paste per model release' in out and 'brain-triage-<identity12> in o/r' in out
    assert 'MODEL_BUNDLE_TOKEN' in out and 'READ o/r' in out and 'Contents: read' in out
    assert 'MODEL_BUNDLE_REPO' in out and 'must name o/r' in out
    assert 'BEFORE the version tag' in out and 'bump_model_release.py' in out and 'FALLBACK_IDENTITY' in out
    assert 'DRAFT release' in out and 'optional fallback' in out and 'needs no editing per release' in out
    assert 'paste the URL' not in out and 'Update -> paste' not in out
    assert 'settings/environments' in out and 'env SIAA_MODELS_TOKEN' in out
    assert TOKEN not in out
