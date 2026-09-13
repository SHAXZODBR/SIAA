const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');

let mainWindow;
let inferenceProcess = null;

// Single instance lock
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    title: 'Sentinel Medical AI',
    backgroundColor: '#0f172a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false,
  });

  // Load React app
  const isDev = !app.isPackaged;
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.maximize();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ===== IPC Handlers =====

// Open DICOM file dialog
ipcMain.handle('open-dicom-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Open DICOM File',
    filters: [
      { name: 'DICOM Files', extensions: ['dcm', 'dicom'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    properties: ['openFile', 'multiSelections'],
  });
  return result.filePaths;
});

// Open folder dialog
ipcMain.handle('open-dicom-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Open DICOM Folder',
    properties: ['openDirectory'],
  });
  return result.filePaths[0] || null;
});

// Get app info
ipcMain.handle('get-app-info', () => ({
  version: app.getVersion(),
  name: app.getName(),
  platform: process.platform,
  arch: process.arch,
}));

// ===== PDF EXPORT — Native Chromium rendering (full Unicode support) =====
// Renders an HTML string to a PDF using Chromium's printToPDF.
// This solves the Cyrillic/Uzbek garbled-text problem that jsPDF has.
ipcMain.handle('export-pdf', async (_, { html, filename, suggestedDir }) => {
  try {
    // Create a hidden BrowserWindow to render the HTML
    const pdfWin = new BrowserWindow({
      show: false,
      webPreferences: { offscreen: true, sandbox: true },
    });

    // Load the HTML as a data URL (no temp file needed)
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
    await pdfWin.loadURL(dataUrl);

    // Wait for fonts/images to settle
    await new Promise((r) => setTimeout(r, 250));

    const pdfBuffer = await pdfWin.webContents.printToPDF({
      pageSize: 'A4',
      printBackground: true,
      margins: { marginType: 'custom', top: 0.5, bottom: 0.5, left: 0.5, right: 0.5 },
    });

    pdfWin.destroy();

    // Ask user where to save
    const defaultPath = path.join(
      suggestedDir || app.getPath('downloads'),
      filename || `report_${Date.now()}.pdf`
    );
    const saveResult = await dialog.showSaveDialog(mainWindow, {
      title: 'Save Report PDF',
      defaultPath,
      filters: [{ name: 'PDF', extensions: ['pdf'] }],
    });

    if (saveResult.canceled || !saveResult.filePath) {
      return { success: false, canceled: true };
    }

    fs.writeFileSync(saveResult.filePath, pdfBuffer);

    // Reveal in OS file manager
    shell.showItemInFolder(saveResult.filePath);

    return { success: true, filePath: saveResult.filePath };
  } catch (e) {
    console.error('PDF export failed:', e);
    return { success: false, error: e.message };
  }
});

// ===== LICENSE / MACHINE BINDING =====
// Returns a stable machine fingerprint used to validate the license.
// Combines hardware identifiers so the license can't be moved between machines.
ipcMain.handle('get-machine-id', () => {
  const networkInterfaces = os.networkInterfaces();
  const macs = [];
  for (const name of Object.keys(networkInterfaces)) {
    for (const net of networkInterfaces[name]) {
      if (!net.internal && net.mac && net.mac !== '00:00:00:00:00:00') {
        macs.push(net.mac);
      }
    }
  }
  macs.sort();

  const fingerprint = [
    os.hostname(),
    os.platform(),
    os.arch(),
    os.cpus()[0]?.model || 'cpu-unknown',
    String(os.totalmem()),
    macs.join(','),
  ].join('|');

  return crypto.createHash('sha256').update(fingerprint).digest('hex');
});

// Read license file from app data directory
ipcMain.handle('read-license', async () => {
  try {
    const licensePath = path.join(app.getPath('userData'), 'license.dat');
    if (!fs.existsSync(licensePath)) return null;
    return fs.readFileSync(licensePath, 'utf8');
  } catch {
    return null;
  }
});

// Write license file (after activation)
ipcMain.handle('write-license', async (_, licenseData) => {
  try {
    const licensePath = path.join(app.getPath('userData'), 'license.dat');
    fs.writeFileSync(licensePath, licenseData, 'utf8');
    return { success: true };
  } catch (e) {
    return { success: false, error: e.message };
  }
});

// ===== Documentation =====
// A packaged build may ship a docs/ folder next to the backend
// (electron-builder extraResources); in dev the repo root is used. When no
// folder exists the renderer falls back to its in-app quick guide.
function docsDir() {
  const candidates = [
    path.join(process.resourcesPath || '', 'docs'),
    path.join(__dirname, '..', '..', 'docs'),
  ];
  return candidates.find((p) => { try { return p && fs.statSync(p).isDirectory(); } catch { return false; } }) || null;
}

ipcMain.handle('open-docs', async () => {
  const dir = docsDir();
  if (!dir) return { opened: false, path: null };
  try {
    const err = await shell.openPath(dir);
    if (err) return { opened: false, path: dir, error: err };
    return { opened: true, path: dir };
  } catch (e) {
    return { opened: false, path: dir, error: String(e) };
  }
});

// ===== Server-side machine fingerprint =====
// Licenses are bound to the fingerprint computed by the Python server
// (src/utils/license.py::get_machine_id), which is NOT the same recipe as
// 'get-machine-id' above. The setup wizard therefore asks the server's own
// CLI for it, so the value the clinic sends to the vendor is the one that
// verify_license() will compare against.
function backendPaths() {
  const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
  const packagedDir = path.join(resourcesPath, 'backend');
  const devDir = path.join(__dirname, '..', '..');
  const backendDir = fs.existsSync(path.join(packagedDir, 'src', 'utils', 'license.py')) ? packagedDir
    : fs.existsSync(path.join(devDir, 'src', 'utils', 'license.py')) ? devDir : null;
  if (!backendDir) return null;
  const candidates = [
    path.join(backendDir, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
    process.env.SENTINEL_PYTHON || '',
    process.platform === 'win32' ? 'python.exe' : 'python3',
  ].filter(Boolean);
  const pythonExe = candidates.find((p) => { try { return fs.existsSync(p); } catch { return false; } }) || candidates[candidates.length - 1];
  return { backendDir, pythonExe };
}

ipcMain.handle('get-server-fingerprint', () => new Promise((resolve) => {
  const bp = backendPaths();
  if (!bp) return resolve({ fingerprint: null, error: 'backend not found' });
  let out = '';
  let done = false;
  const finish = (result) => { if (!done) { done = true; resolve(result); } };
  try {
    const child = spawn(bp.pythonExe, ['-m', 'src.utils.license', 'fingerprint'], {
      cwd: bp.backendDir,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const timer = setTimeout(() => { try { child.kill(); } catch {} finish({ fingerprint: null, error: 'timeout' }); }, 15000);
    child.stdout.on('data', (d) => { out += String(d); });
    child.stderr.on('data', (d) => { out += String(d); });
    child.on('error', (err) => { clearTimeout(timer); finish({ fingerprint: null, error: String(err) }); });
    child.on('exit', () => {
      clearTimeout(timer);
      const m = out.match(/\b([0-9a-f]{64})\b/);
      finish(m ? { fingerprint: m[1], error: null } : { fingerprint: null, error: out.trim().slice(0, 400) || 'no output' });
    });
  } catch (e) {
    finish({ fingerprint: null, error: String(e) });
  }
}));

// ===== Backend lifecycle =====
// In a packaged build the desktop app launches the inference server itself so
// the user doesn't have to start Python/Ollama by hand. In dev we assume the
// developer ran `python run_server.py` already.
//
// - SENTINEL_DATA_DIR points the server's DB / audit log / license / logs at
//   the per-user app-data folder (writable, survives reinstall).
// - SENTINEL_REQUIRE_AUTH=1 keeps login mandatory in every packaged build.
// - stdout/stderr are appended to <userData>/logs/backend.log for IT support.
// - An unexpected exit is reported to the doctor and the server is restarted
//   up to MAX_BACKEND_RESTARTS times with exponential backoff.
const MAX_BACKEND_RESTARTS = 3;
const BACKEND_STABLE_MS = 120000; // an exit after this long of uptime resets the restart counter
let backendRestarts = 0;
let backendStartedAt = 0;
let backendLog = null;
let quitting = false;

function backendLogPath() {
  return path.join(app.getPath('userData'), 'logs', 'backend.log');
}

function openBackendLog() {
  try {
    fs.mkdirSync(path.dirname(backendLogPath()), { recursive: true });
    return fs.createWriteStream(backendLogPath(), { flags: 'a' });
  } catch (e) {
    console.error('Could not open backend log:', e);
    return null;
  }
}

function logBackend(line) {
  if (!backendLog) backendLog = openBackendLog();
  if (backendLog) backendLog.write(`[${new Date().toISOString()}] [electron] ${line}\n`);
}

function backendEnv() {
  return {
    ...process.env,
    SENTINEL_REQUIRE_AUTH: '1',                    // enforce auth in production
    SENTINEL_DATA_DIR: app.getPath('userData'),    // DB, audit log, license, logs
  };
}

function showBackendDialog(type, title, message, detail) {
  const opts = { type, title, message, detail, buttons: ['OK'] };
  const p = mainWindow && !mainWindow.isDestroyed()
    ? dialog.showMessageBox(mainWindow, opts)
    : dialog.showMessageBox(opts);
  p.catch(() => {});
}

function spawnBackend() {
  const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
  const backendDir = path.join(resourcesPath, 'backend');
  const serverScript = path.join(backendDir, 'run_server.py');
  const candidates = [
    path.join(backendDir, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
    process.platform === 'win32' ? 'python.exe' : 'python3',
  ];
  const pythonExe = candidates.find((p) => { try { return fs.existsSync(p); } catch { return false; } }) || candidates[candidates.length - 1];

  if (!fs.existsSync(serverScript)) {
    logBackend(`server script not found: ${serverScript}`);
    dialog.showErrorBox('Backend not found',
      'The Sentinel inference server was not found in this install. ' +
      'Reinstall the full package, or start the server manually.');
    return;
  }

  if (!backendLog) backendLog = openBackendLog();

  try {
    const child = spawn(pythonExe, [serverScript], {
      cwd: backendDir,
      env: backendEnv(),
      detached: false,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    inferenceProcess = child;
    backendStartedAt = Date.now();
    logBackend(`backend start attempt=${backendRestarts + 1} pid=${child.pid} python=${pythonExe}`);

    if (backendLog) {
      child.stdout.pipe(backendLog, { end: false });
      child.stderr.pipe(backendLog, { end: false });
    } else {
      child.stdout.resume();
      child.stderr.resume();
    }

    child.on('error', (err) => {
      logBackend(`spawn error: ${err}`);
      if (!quitting) dialog.showErrorBox('Backend failed to start', String(err));
    });

    child.on('exit', (code, signal) => {
      logBackend(`backend exit code=${code} signal=${signal}`);
      if (inferenceProcess === child) inferenceProcess = null;
      if (quitting) return;
      handleBackendExit(code, signal);
    });
  } catch (e) {
    logBackend(`spawn threw: ${e}`);
    dialog.showErrorBox('Backend failed to start', String(e));
  }
}

function handleBackendExit(code, signal) {
  // A server that ran for a while before dying gets a fresh restart budget.
  if (Date.now() - backendStartedAt > BACKEND_STABLE_MS) backendRestarts = 0;

  const reason = code !== null && code !== undefined ? `exit code ${code}` : `signal ${signal}`;
  if (backendRestarts >= MAX_BACKEND_RESTARTS) {
    logBackend(`giving up after ${MAX_BACKEND_RESTARTS} restarts`);
    showBackendDialog(
      'error',
      'AI server stopped',
      `The Sentinel inference server stopped ${MAX_BACKEND_RESTARTS} times in a row (${reason}) and will not be restarted automatically.`,
      `Log file: ${backendLogPath()}\n\nContact IT support. Analysis is unavailable until the server is running again.`,
    );
    return;
  }

  backendRestarts += 1;
  const delayMs = 2000 * Math.pow(2, backendRestarts - 1); // 2s, 4s, 8s
  logBackend(`scheduling restart ${backendRestarts}/${MAX_BACKEND_RESTARTS} in ${delayMs}ms`);
  showBackendDialog(
    'warning',
    'AI server restarting',
    `The Sentinel inference server stopped unexpectedly (${reason}). Restarting in ${delayMs / 1000}s (attempt ${backendRestarts} of ${MAX_BACKEND_RESTARTS}).`,
    `Log file: ${backendLogPath()}`,
  );
  setTimeout(() => {
    if (!quitting && !inferenceProcess) spawnBackend();
  }, delayMs);
}

function startBackend() {
  if (!app.isPackaged) return; // dev: backend run manually

  // Already running? (a previous instance, or user started it)
  const http = require('http');
  const check = http.get('http://127.0.0.1:8000/health', (res) => {
    res.resume();
    logBackend('backend already running on :8000 — not spawning');
  });
  check.setTimeout(2000, () => check.destroy(new Error('health check timeout')));
  check.on('error', () => spawnBackend());
}

function stopBackend() {
  quitting = true;
  if (inferenceProcess) {
    try { inferenceProcess.kill(); } catch {}
    inferenceProcess = null;
  }
  if (backendLog) {
    try { backendLog.end(); } catch {}
    backendLog = null;
  }
}

// ===== App Lifecycle =====
app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  // On macOS the app (and its backend) stays alive until the user quits;
  // reopening a window via the Dock must not hit a dead server.
  if (process.platform !== 'darwin') {
    stopBackend();
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackend();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
