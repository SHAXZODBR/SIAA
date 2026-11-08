/**
 * Structural regression tests for the v1.1.0 desktop wiring that cannot be
 * exercised without Electron / a running server: the backend environment,
 * the IPC channel names shared by main.js and preload.js, the API routes and
 * methods of the contract, and the admin gating of the new controls.
 *
 * Run: cd desktop-app && npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readText } from './_loadTs.mjs';

const main = readText('electron/main.js');
const preload = readText('electron/preload.js');
const api = readText('src/services/api.ts');
const settingsModal = readText('src/components/SettingsModal.tsx');
const rightPanel = readText('src/components/RightPanel.tsx');
const sidebar = readText('src/components/Sidebar.tsx');

test('backend env encrypts models at rest only in a packaged app (D2)', () => {
  assert.match(main, /SENTINEL_ENCRYPT_MODELS_AT_REST: app\.isPackaged \? '1' : '0',/);
  const env = main.slice(main.indexOf('function backendEnv()'), main.indexOf('function showBackendDialog'));
  assert.ok(env.includes('SENTINEL_ENCRYPT_MODELS_AT_REST'), 'must be set inside backendEnv()');
  // Spread first so the shell's value wins over an inherited environment.
  assert.ok(env.indexOf('...process.env') < env.indexOf('SENTINEL_ENCRYPT_MODELS_AT_REST'));
});

test('IPC channels are registered in main.js and exposed by preload.js', () => {
  for (const ch of ['model-bundle:import', 'diagnostics:save']) {
    assert.ok(main.includes(`ipcMain.handle('${ch}'`), `main.js handles ${ch}`);
    assert.ok(preload.includes(`ipcRenderer.invoke('${ch}'`), `preload.js invokes ${ch}`);
  }
  assert.match(preload, /importModelBundle: \(opts\) =>/);
  assert.match(preload, /saveDiagnostics: \(bytes, suggestedName\) =>/);
  assert.ok(preload.includes("exposeInMainWorld('electronAPI'"), 'existing namespace is kept');
});

test('IPC handlers never throw into the renderer', () => {
  for (const ch of ['model-bundle:import', 'diagnostics:save']) {
    const start = main.indexOf(`ipcMain.handle('${ch}'`);
    const body = main.slice(start, main.indexOf('\n});', start));
    assert.ok(body.includes('try {') && body.includes('} catch (e) {'), `${ch} has a catch-all`);
    assert.ok(body.includes("return { ok: false, error: 'internal_error'"), `${ch} returns {ok:false} on error`);
    assert.ok(body.includes('canceled: true'), `${ch} reports a cancelled dialog`);
  }
});

test('the packaged app stages bundles under its own user-data folder', () => {
  const fn = main.slice(main.indexOf('function adminDataDir'), main.indexOf("ipcMain.handle('model-bundle:import'"));
  assert.ok(fn.includes("app.getPath('userData')"));
  assert.ok(fn.includes('app.isPackaged'), 'renderer hint must be ignored in a packaged app');
  assert.match(readText('electron/adminIpc.js'), /const INBOX_DIR = 'model_inbox';/);
});

test('api.ts follows the server contract routes and methods', () => {
  assert.ok(api.includes("inferenceAPI.get('/settings/ai'"));
  assert.ok(api.includes("inferenceAPI.put('/settings/ai', { show_pending_detectors:"));
  assert.ok(api.includes("inferenceAPI.post('/models/import', { path }"));
  assert.ok(api.includes("inferenceAPI.get('/diagnostics', { responseType: 'arraybuffer'"));
  assert.ok(api.includes("aiSettings: parseAiSettings(d?.ai_settings)"), '/health ai_settings is mapped');
});

test('admin-only controls are gated on the admin role', () => {
  assert.ok(settingsModal.includes('isAdminRole(currentUser?.role)'));
  assert.match(settingsModal, /<PendingDetectorsCard admin=\{admin\}/);
  assert.match(settingsModal, /<ModelBundleImport admin=\{admin\}/);
  assert.ok(settingsModal.includes('<DiagnosticsExport />'));
  // Every admin action button is disabled for non-admins and explains why.
  assert.ok((settingsModal.match(/disabled=\{!admin \|\| !reachable \|\| busy\}/g) || []).length >= 2);
  assert.ok(settingsModal.includes("const disabled = !admin || !reachable || !effective || busy;"));
  assert.ok((settingsModal.match(/<AdminHint \/>/g) || []).length >= 3);
});

test('a setting change refreshes /health and the model list', () => {
  const fn = settingsModal.slice(settingsModal.indexOf('const refreshServerState'), settingsModal.indexOf('const serverLabel'));
  assert.ok(fn.includes('setHealth(await getHealth())') && fn.includes('loadRegistry()'));
  assert.match(settingsModal, /<PendingDetectorsCard admin=\{admin\} onChanged=\{refreshServerState\} \/>/);
  assert.match(settingsModal, /<ModelBundleImport admin=\{admin\} onImported=\{refreshServerState\} \/>/);
});

test('no_validated_detector is rendered as the localized note, not the raw code', () => {
  assert.ok(rightPanel.includes('rejectionReasonKey(reason)'));
  assert.ok(rightPanel.includes('{t(reasonKey)}'));
  assert.equal((sidebar.match(/rejectionReasonKey\(/g) || []).length, 2);
});

test('desktop version is 1.1.0', () => {
  assert.equal(JSON.parse(readText('package.json')).version, '1.1.0');
});

test('toasts are stacked above every modal so in-dialog actions show feedback', () => {
  const toasts = readText('src/components/Toasts.tsx');
  const z = (src) => Math.max(...[...src.matchAll(/z-\[(\d+)\]/g)].map((x) => Number(x[1])));
  const toastZ = z(toasts);
  for (const f of ['SettingsModal.tsx', 'HelpModal.tsx', 'CommandPalette.tsx']) {
    const src = readText(`src/components/${f}`);
    if (/z-\[\d+\]/.test(src)) assert.ok(toastZ > z(src), `Toasts z-[${toastZ}] must exceed ${f} z-[${z(src)}]`);
  }
});


test('report regeneration sends the sequences the server found (D4)', () => {
  // both analysis mappers keep sequences_present, and /report/regenerate forwards it
  assert.equal((api.match(/sequencesPresent: mapSequences\(data\??\.sequences_present\)/g) || []).length, 2);
  const regen = api.slice(api.indexOf('export async function regenerateReport'), api.indexOf('export async function signReport'));
  assert.ok(regen.includes('sequences?: string[]'), 'regenerateReport takes optional sequences');
  assert.ok(regen.includes('...(sequences ? { sequences } : {})'), 'sent only when known');
  assert.ok(rightPanel.includes('result.sequencesPresent,'), 'RightPanel passes the result sequences');
});
