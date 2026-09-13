/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set to '1' to show the built-in demo worklist (never in a clinical build). */
  readonly VITE_DEMO_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface ElectronAPI {
  openDicomFile: () => Promise<string[]>;
  openDicomFolder: () => Promise<string | null>;
  getAppInfo: () => Promise<{ version: string; name: string; platform: string; arch: string }>;
  onNotification: (callback: (data: unknown) => void) => void;
  exportPDF: (
    html: string,
    filename?: string,
    suggestedDir?: string,
  ) => Promise<{ success: boolean; filePath?: string; canceled?: boolean; error?: string }>;
  getMachineId: () => Promise<string>;
  readLicense: () => Promise<string | null>;
  writeLicense: (data: string) => Promise<{ success: boolean; error?: string }>;
  /** Fingerprint from the Python server's own CLI — the value licenses are bound to. */
  getServerFingerprint?: () => Promise<{ fingerprint: string | null; error: string | null }>;
  /** Opens the bundled docs/ folder in the OS file manager; opened=false when none is bundled. */
  openDocs?: () => Promise<{ opened: boolean; path: string | null; error?: string }>;
}

interface Window {
  electronAPI?: ElectronAPI;
}
