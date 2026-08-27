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

// ===== Backend lifecycle =====
// In a packaged build the desktop app launches the inference server itself so
// the user doesn't have to start Python/Ollama by hand. In dev we assume the
// developer ran `python run_server.py` already.
function startBackend() {
  if (!app.isPackaged) return; // dev: backend run manually

  // Already running? (a previous instance, or user started it)
  const http = require('http');
  const check = http.get('http://127.0.0.1:8000/health', () => { /* up */ });
  check.on('error', () => {
    // Not up — spawn it. Look for a bundled python venv or system python.
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
    const serverScript = path.join(resourcesPath, 'backend', 'run_server.py');
    const candidates = [
      path.join(resourcesPath, 'backend', 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
      process.platform === 'win32' ? 'python.exe' : 'python3',
    ];
    let pythonExe = candidates.find((p) => { try { return fs.existsSync(p); } catch { return false; } }) || candidates[candidates.length - 1];

    if (!fs.existsSync(serverScript)) {
      dialog.showErrorBox('Backend not found',
        'The Sentinel inference server was not found in this install. ' +
        'Reinstall the full package, or start the server manually.');
      return;
    }
    try {
      inferenceProcess = spawn(pythonExe, [serverScript], {
        cwd: path.join(resourcesPath, 'backend'),
        env: { ...process.env, SENTINEL_REQUIRE_AUTH: '1' },  // enforce auth in production
        detached: false,
        stdio: 'ignore',
      });
      inferenceProcess.on('error', (err) => {
        dialog.showErrorBox('Backend failed to start', String(err));
      });
    } catch (e) {
      dialog.showErrorBox('Backend failed to start', String(e));
    }
  });
}

// ===== App Lifecycle =====
app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (inferenceProcess) {
    try { inferenceProcess.kill(); } catch {}
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (inferenceProcess) {
    try { inferenceProcess.kill(); } catch {}
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
