/**
 * Unit tests for electron/adminIpc.js — the file operations behind the
 * 'model-bundle:import' and 'diagnostics:save' IPC channels. Uses throw-away
 * temp folders with synthetic bytes only (no model, no patient data).
 *
 * Run: cd desktop-app && npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
const ipc = require('../electron/adminIpc.js');

const ZIP = Buffer.concat([Buffer.from([0x50, 0x4b, 0x03, 0x04]), Buffer.alloc(2048, 7)]);

async function tmpdir(t) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'sentinel-adminipc-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  return dir;
}

async function leftovers(dir) {
  try {
    return (await fs.readdir(dir)).filter((n) => n.includes('.part-'));
  } catch {
    return [];
  }
}

test('stageModelBundle copies the zip into <dataDir>/model_inbox/<basename>', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'downloads', 'brain_triage_finetuned_0d559766ce58_ab12cd34.zip');
  await fs.mkdir(path.dirname(src), { recursive: true });
  await fs.writeFile(src, ZIP);
  const dataDir = path.join(root, 'userData');

  const res = await ipc.stageModelBundle(src, dataDir);
  assert.equal(res.ok, true, JSON.stringify(res));
  assert.equal(res.path, path.join(dataDir, 'model_inbox', 'brain_triage_finetuned_0d559766ce58_ab12cd34.zip'));
  assert.ok(path.isAbsolute(res.path));
  assert.equal(res.bytes, ZIP.length);
  assert.deepEqual(await fs.readFile(res.path), ZIP);
  assert.deepEqual(await leftovers(path.join(dataDir, 'model_inbox')), []);
});

test('stageModelBundle overwrites an older copy with the same name', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'bundle.zip');
  await fs.writeFile(src, ZIP);
  const inbox = path.join(root, 'data', 'model_inbox');
  await fs.mkdir(inbox, { recursive: true });
  await fs.writeFile(path.join(inbox, 'bundle.zip'), Buffer.from([0x50, 0x4b, 0x03, 0x04, 1]));
  const res = await ipc.stageModelBundle(src, path.join(root, 'data'));
  assert.equal(res.ok, true);
  assert.deepEqual(await fs.readFile(res.path), ZIP);
});

test('stageModelBundle does not copy a file picked from the inbox itself', async (t) => {
  const root = await tmpdir(t);
  const inbox = path.join(root, 'model_inbox');
  await fs.mkdir(inbox);
  const src = path.join(inbox, 'bundle.zip');
  await fs.writeFile(src, ZIP);
  let copies = 0;
  const spy = { ...fs, copyFile: async (...a) => { copies += 1; return fs.copyFile(...a); } };
  const res = await ipc.stageModelBundle(src, root, { fs: spy });
  assert.equal(res.ok, true);
  assert.equal(res.path, src);
  assert.equal(copies, 0);
});

test('stageModelBundle sanitises the destination name', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'clinic bundle (1).zip');
  await fs.writeFile(src, ZIP);
  const res = await ipc.stageModelBundle(src, path.join(root, 'data'));
  assert.equal(res.ok, true);
  assert.equal(path.basename(res.path), 'clinic_bundle_1_.zip');
  assert.equal(path.dirname(res.path), path.join(root, 'data', 'model_inbox'));
});

test('stageModelBundle rejects non-zip, empty and relative inputs with reason codes', async (t) => {
  const root = await tmpdir(t);
  const data = path.join(root, 'data');
  const txt = path.join(root, 'bundle.txt');
  await fs.writeFile(txt, ZIP);
  assert.equal((await ipc.stageModelBundle(txt, data)).error, 'not_a_zip');

  const fake = path.join(root, 'fake.zip');
  await fs.writeFile(fake, '{"not":"a zip"}');
  assert.equal((await ipc.stageModelBundle(fake, data)).error, 'not_a_zip');

  const empty = path.join(root, 'empty.zip');
  await fs.writeFile(empty, '');
  assert.equal((await ipc.stageModelBundle(empty, data)).error, 'empty_file');

  assert.equal((await ipc.stageModelBundle('relative/bundle.zip', data)).error, 'invalid_source');
  assert.equal((await ipc.stageModelBundle(path.join(root, 'missing.zip'), data)).error, 'source_unreadable');
  await fs.mkdir(path.join(root, 'dir.zip'));
  assert.equal((await ipc.stageModelBundle(path.join(root, 'dir.zip'), data)).error, 'source_unreadable');
  assert.equal((await ipc.stageModelBundle(fake, 'relative-data')).error, 'internal_error');
  // Nothing was staged for any rejected input.
  await assert.rejects(fs.stat(path.join(data, 'model_inbox')));
});

test('stageModelBundle refuses an oversized file before copying', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'huge.zip');
  await fs.writeFile(src, ZIP);
  const realStat = fs.stat;
  const big = { ...fs, stat: async (p) => { const s = await realStat(p); return { isFile: () => s.isFile(), size: ipc.MAX_BUNDLE_BYTES + 1 }; } };
  assert.equal((await ipc.stageModelBundle(src, root, { fs: big })).error, 'too_large');
});

test('stageModelBundle cleans up and reports copy_failed when the copy throws', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'bundle.zip');
  await fs.writeFile(src, ZIP);
  const data = path.join(root, 'data');
  const broken = {
    ...fs,
    copyFile: async (from, to) => { await fs.writeFile(to, ZIP.subarray(0, 10)); throw new Error('ENOSPC: no space left on device'); },
  };
  const res = await ipc.stageModelBundle(src, data, { fs: broken });
  assert.equal(res.ok, false);
  assert.equal(res.error, 'copy_failed');
  assert.match(res.detail, /ENOSPC/);
  assert.deepEqual(await leftovers(path.join(data, 'model_inbox')), []);
  await assert.rejects(fs.stat(path.join(data, 'model_inbox', 'bundle.zip')));
});

test('stageModelBundle detects a short copy and never leaves it under the final name', async (t) => {
  const root = await tmpdir(t);
  const src = path.join(root, 'bundle.zip');
  await fs.writeFile(src, ZIP);
  const data = path.join(root, 'data');
  const short = { ...fs, copyFile: async (from, to) => fs.writeFile(to, ZIP.subarray(0, 100)) };
  const res = await ipc.stageModelBundle(src, data, { fs: short });
  assert.equal(res.error, 'copy_incomplete');
  assert.deepEqual(await leftovers(path.join(data, 'model_inbox')), []);
  await assert.rejects(fs.stat(path.join(data, 'model_inbox', 'bundle.zip')));
});

test('validateDiagnosticsPayload accepts IPC byte shapes and rejects non-zips', () => {
  const view = new Uint8Array(Buffer.concat([Buffer.from('xx'), ZIP])).subarray(2);
  for (const bytes of [ZIP, new Uint8Array(ZIP), view, new Uint8Array(ZIP).buffer]) {
    const v = ipc.validateDiagnosticsPayload(bytes);
    assert.equal(v.ok, true);
    assert.equal(v.buffer.length, ZIP.length);
    assert.ok(ipc.isZipMagic(v.buffer));
  }
  assert.equal(ipc.validateDiagnosticsPayload('PK..').error, 'invalid_payload');
  assert.equal(ipc.validateDiagnosticsPayload(undefined).error, 'invalid_payload');
  assert.equal(ipc.validateDiagnosticsPayload(new Uint8Array(0)).error, 'empty_payload');
  assert.equal(ipc.validateDiagnosticsPayload(Buffer.from('{"detail":"Forbidden"}')).error, 'not_a_zip');
});

test('writeDiagnostics writes the bytes and appends .zip when missing', async (t) => {
  const root = await tmpdir(t);
  const res = await ipc.writeDiagnostics(path.join(root, 'support'), new Uint8Array(ZIP));
  assert.equal(res.ok, true);
  assert.equal(res.path, path.join(root, 'support.zip'));
  assert.deepEqual(await fs.readFile(res.path), ZIP);
  const res2 = await ipc.writeDiagnostics(path.join(root, 'sentinel-diagnostics-20311109-1200.ZIP'), ZIP);
  assert.equal(res2.path, path.join(root, 'sentinel-diagnostics-20311109-1200.ZIP'));
  assert.deepEqual(await leftovers(root), []);
});

test('writeDiagnostics reports write_failed / invalid_target without throwing', async (t) => {
  const root = await tmpdir(t);
  assert.equal((await ipc.writeDiagnostics('relative.zip', ZIP)).error, 'invalid_target');
  assert.equal((await ipc.writeDiagnostics(path.join(root, 'x.zip'), Buffer.from('nope'))).error, 'not_a_zip');
  const res = await ipc.writeDiagnostics(path.join(root, 'no-such-dir', 'x.zip'), ZIP);
  assert.equal(res.ok, false);
  assert.equal(res.error, 'write_failed');
  assert.deepEqual(await leftovers(root), []);
});

test('safeFileName mirrors the renderer helper', () => {
  assert.equal(ipc.safeFileName('../../x/sentinel diag.zip', '.zip'), 'sentinel_diag.zip');
  assert.equal(ipc.safeFileName('name', '.zip'), 'name.zip');
  assert.equal(ipc.safeFileName('...', '.zip'), null);
  assert.equal(ipc.safeFileName(42, '.zip'), null);
});
