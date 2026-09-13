const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openDicomFile: () => ipcRenderer.invoke('open-dicom-file'),
  openDicomFolder: () => ipcRenderer.invoke('open-dicom-folder'),
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  onNotification: (callback) => ipcRenderer.on('notification', (_, data) => callback(data)),

  // PDF export — uses Chromium's native printToPDF (full Unicode support)
  exportPDF: (html, filename, suggestedDir) =>
    ipcRenderer.invoke('export-pdf', { html, filename, suggestedDir }),

  // License/security
  getMachineId: () => ipcRenderer.invoke('get-machine-id'),
  readLicense: () => ipcRenderer.invoke('read-license'),
  writeLicense: (data) => ipcRenderer.invoke('write-license', data),
  // Fingerprint computed by the Python server (the one licenses are bound to)
  getServerFingerprint: () => ipcRenderer.invoke('get-server-fingerprint'),

  // Help → Documentation (bundled docs/ folder, if any)
  openDocs: () => ipcRenderer.invoke('open-docs'),
});
