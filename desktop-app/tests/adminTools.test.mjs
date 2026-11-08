/**
 * Unit tests for src/services/adminTools.ts — the pure logic behind the
 * v1.1.0 administrator tools (AI settings toggle, model-bundle import,
 * diagnostics export) and the D1 "no validated detector" note.
 *
 * Run: cd desktop-app && npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadTs, i18nKeys, readText } from './_loadTs.mjs';

const m = await loadTs('src/services/adminTools.ts');
const EN = i18nKeys('en');

test('no_validated_detector is recognised and mapped to the localized note', () => {
  assert.equal(m.NO_VALIDATED_DETECTOR, 'no_validated_detector');
  assert.equal(m.isNoValidatedDetector('no_validated_detector'), true);
  assert.equal(m.isNoValidatedDetector('  NO_VALIDATED_DETECTOR '), true);
  assert.equal(m.isNoValidatedDetector('wrong_modality'), false);
  assert.equal(m.isNoValidatedDetector(null), false);
  assert.equal(m.rejectionReasonKey('no_validated_detector'), 'findings.noValidatedDetector');
  assert.equal(m.rejectionReasonKey('Unsupported body part: KNEE'), null);
  assert.equal(m.rejectionReasonKey(undefined), null);
});

test('no_detector_ran (pending shown, nothing could run) maps to its own localized note', () => {
  assert.equal(m.NO_DETECTOR_RAN, 'no_detector_ran');
  assert.equal(m.isNoDetectorRan(' No_Detector_Ran '), true);
  assert.equal(m.isNoDetectorRan('no_validated_detector'), false);
  assert.equal(m.rejectionReasonKey('no_detector_ran'), 'findings.noDetectorRan');
  assert.ok(EN.has('findings.noDetectorRan'));
});

test('the D1 note carries the contract wording in English', () => {
  const en = readText('src/i18n/en.ts');
  assert.ok(en.includes(
    "'findings.noValidatedDetector': 'This study type has no validated AI detector in this release — not analysed by AI; "
    + 'the radiologist reads as usual. An administrator can enable unvalidated detectors in Preferences → AI models.',
  ));
});

test('admin role check is exact and case-insensitive', () => {
  assert.equal(m.isAdminRole('admin'), true);
  assert.equal(m.isAdminRole('Admin'), true);
  assert.equal(m.isAdminRole('radiologist'), false);
  assert.equal(m.isAdminRole('administrator-ish'), false);
  assert.equal(m.isAdminRole(undefined), false);
  assert.equal(m.isAdminRole(null), false);
});

test('parseAiSettings accepts the contract shape only', () => {
  assert.deepEqual(m.parseAiSettings({ show_pending_detectors: true, source: 'file' }), { showPendingDetectors: true, source: 'file' });
  assert.deepEqual(m.parseAiSettings({ show_pending_detectors: false, source: 'env' }), { showPendingDetectors: false, source: 'env' });
  // Unknown source degrades to 'default' rather than inventing one.
  assert.deepEqual(m.parseAiSettings({ show_pending_detectors: false, source: 'weird' }), { showPendingDetectors: false, source: 'default' });
  // A non-boolean value must never be guessed into a toggle state.
  assert.equal(m.parseAiSettings({ show_pending_detectors: 'true', source: 'file' }), null);
  assert.equal(m.parseAiSettings({ source: 'file' }), null);
  assert.equal(m.parseAiSettings(null), null);
  assert.equal(m.parseAiSettings('x'), null);
});

test('every AI settings source has a label', () => {
  for (const src of ['file', 'env', 'default']) assert.ok(EN.has(m.AI_SOURCE_KEYS[src]), src);
});

test('parseModelImportResult requires ok + a 12-hex identity', () => {
  assert.deepEqual(
    m.parseModelImportResult({ ok: true, identity: '0D559766CE58', encrypted_at_rest: true, status: 'validated' }),
    { ok: true, identity: '0d559766ce58', encryptedAtRest: true, status: 'validated' },
  );
  assert.throws(() => m.parseModelImportResult({ ok: true, identity: 'abc', status: 'validated' }));
  assert.throws(() => m.parseModelImportResult({ ok: false, identity: '0d559766ce58' }));
  assert.throws(() => m.parseModelImportResult(null));
});

test('extractReasonCode understands every error-body shape the server may use', () => {
  assert.equal(m.extractReasonCode({ detail: { reason: 'bundle_not_for_this_machine', message: 'x' } }), 'bundle_not_for_this_machine');
  assert.equal(m.extractReasonCode({ detail: { code: 'manifest_mismatch' } }), 'manifest_mismatch');
  assert.equal(m.extractReasonCode({ detail: { error: 'not_a_bundle' } }), 'not_a_bundle');
  assert.equal(m.extractReasonCode({ reason: 'path_outside_inbox', detail: 'long text' }), 'path_outside_inbox');
  // The server's _ops_error body: {ok:false, reason, detail:<English text>} → the code, not the text.
  assert.equal(m.extractReasonCode({ ok: false, reason: 'bundle_not_for_this_machine', detail: 'decrypt failed' }), 'bundle_not_for_this_machine');
  assert.equal(m.extractReasonCode({ detail: 'bundle_not_for_this_machine' }), 'bundle_not_for_this_machine');
  assert.equal(m.extractReasonCode({ error: 'boom' }), 'boom');
  assert.equal(m.extractReasonCode('plain'), 'plain');
  assert.equal(m.extractReasonCode({}), null);
  assert.equal(m.extractReasonCode(null), null);
  assert.equal(m.extractReasonCode(new ArrayBuffer(4)), null);
});

test('model import reason codes map to the right failure family', () => {
  const c = m.classifyModelImportFailure;
  assert.equal(c('bundle_not_for_this_machine', 400), 'not_for_this_machine');
  assert.equal(c('decrypt failed: InvalidTag', 400), 'not_for_this_machine');
  assert.equal(c('no_model_key', 400), 'no_model_key');
  assert.equal(c('model_key_missing', 400), 'no_model_key');
  assert.equal(c('not_a_bundle', 400), 'not_a_bundle');
  assert.equal(c('bad_zip', 400), 'not_a_bundle');
  assert.equal(c('manifest_missing', 400), 'not_a_bundle');
  assert.equal(c('manifest_mismatch', 400), 'manifest_mismatch');
  assert.equal(c('ciphertext_sha_mismatch', 400), 'manifest_mismatch');
  assert.equal(c('plaintext_sha256 does not match the manifest pin', 400), 'manifest_mismatch');
  assert.equal(c('path_outside_inbox', 400), 'outside_inbox');
  assert.equal(c(null, 401), 'auth');
  assert.equal(c(null, 403), 'forbidden');
  assert.equal(c(null, null), 'network');
  assert.equal(c('swap_failed', 500), 'unknown');
  // The reason codes src/inference/server.py actually emits (_ModelImportFailure / _ops_error).
  const server = {
    path_missing: 'outside_inbox', path_not_absolute: 'outside_inbox', path_invalid: 'outside_inbox',
    path_outside_inbox: 'outside_inbox', bundle_not_found: 'outside_inbox',
    bundle_not_zip: 'not_a_bundle', bundle_invalid: 'not_a_bundle', bundle_incomplete: 'not_a_bundle',
    bundle_contains_plaintext: 'not_a_bundle', bundle_too_large: 'too_large',
    manifest_invalid: 'manifest_mismatch', manifest_mismatch: 'manifest_mismatch',
    sidecar_invalid: 'manifest_mismatch', sidecar_mismatch: 'manifest_mismatch',
    decrypted_hash_mismatch: 'manifest_mismatch',
    model_key_unavailable: 'no_model_key', bundle_not_for_this_machine: 'not_for_this_machine',
    license_model_key_missing: 'no_model_key',
    import_in_progress: 'in_progress', models_dir_not_writable: 'not_writable',
    model_dir_overridden: 'unknown', reload_failed: 'unknown', install_failed: 'unknown', internal_error: 'unknown',
  };
  for (const [code, fam] of Object.entries(server)) assert.equal(c(code, 400), fam, code);
  assert.equal(c(null, 500), 'unknown');
});

test('every failure family / IPC code maps to an existing i18n key', () => {
  for (const [fam, key] of Object.entries(m.MODEL_IMPORT_ERROR_KEYS)) assert.ok(EN.has(key), `${fam} → ${key}`);
  for (const [code, key] of Object.entries(m.IPC_ERROR_KEYS)) assert.ok(EN.has(key), `${code} → ${key}`);
  assert.equal(m.ipcErrorKey('not_a_zip'), 'settings.ipcErrNotZip');
  assert.equal(m.ipcErrorKey('something_new'), 'settings.ipcErrInternal');
  assert.equal(m.ipcErrorKey(undefined), 'settings.ipcErrInternal');
});

test('every reason code electron/adminIpc.js can return is localized', () => {
  const src = readText('electron/adminIpc.js');
  const codes = new Set([...src.matchAll(/\bfail\('(\w+)'/g)].map((x) => x[1]));
  assert.ok(codes.size >= 8, [...codes].join(','));
  for (const code of codes) assert.ok(code in m.IPC_ERROR_KEYS, `adminIpc code '${code}' missing from IPC_ERROR_KEYS`);
});

test('diagnostics file name comes from Content-Disposition and is sanitised', () => {
  const f = m.filenameFromContentDisposition;
  const fb = 'fallback.zip';
  assert.equal(f('attachment; filename=sentinel-diagnostics-20311109-1200.zip', fb), 'sentinel-diagnostics-20311109-1200.zip');
  assert.equal(f('attachment; filename="sentinel-diagnostics-20311109-1200.zip"', fb), 'sentinel-diagnostics-20311109-1200.zip');
  assert.equal(f("attachment; filename*=UTF-8''sentinel%2Ddiagnostics.zip; filename=\"other.zip\"", fb), 'sentinel-diagnostics.zip');
  assert.equal(f('attachment; filename="../../etc/passwd"', fb), 'passwd.zip');
  assert.equal(f('attachment; filename="C:\\Windows\\evil.zip"', fb), 'evil.zip');
  assert.equal(f('attachment; filename="..."', fb), fb);
  assert.equal(f('attachment', fb), fb);
  assert.equal(f(null, fb), fb);
  assert.equal(f(undefined, fb), fb);
});

test('defaultDiagnosticsName follows the server format', () => {
  assert.equal(m.defaultDiagnosticsName(new Date(2031, 0, 2, 3, 4)), 'sentinel-diagnostics-20310102-0304.zip');
  assert.match(m.defaultDiagnosticsName(), /^sentinel-diagnostics-\d{8}-\d{4}\.zip$/);
});

test('safeFileName strips directories and unsafe characters', () => {
  assert.equal(m.safeFileName('brain_triage_finetuned_0d559766ce58_ab12cd34.zip', '.zip'), 'brain_triage_finetuned_0d559766ce58_ab12cd34.zip');
  assert.equal(m.safeFileName('/tmp/x/a b$c.zip', '.zip'), 'a_b_c.zip');
  assert.equal(m.safeFileName('.hidden', '.zip'), 'hidden.zip');
  assert.equal(m.safeFileName('report', '.zip'), 'report.zip');
  assert.equal(m.safeFileName('zip', '.zip'), null);
  assert.equal(m.safeFileName('', '.zip'), null);
  assert.equal(m.safeFileName(null, '.zip'), null);
});

test('looksLikeZip checks the archive signature', () => {
  assert.equal(m.looksLikeZip(new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0])), true);
  assert.equal(m.looksLikeZip(new Uint8Array([0x50, 0x4b, 0x05, 0x06])), true);
  assert.equal(m.looksLikeZip(new TextEncoder().encode('{"detail":"x"}')), false);
  assert.equal(m.looksLikeZip(new Uint8Array([0x50, 0x4b])), false);
  assert.equal(m.looksLikeZip(null), false);
});
