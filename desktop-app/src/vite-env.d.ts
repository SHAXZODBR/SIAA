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
  readFileBase64?: (filePath: string) => Promise<string | null>;
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
  /** Language for main-process dialogs (file pickers, backend crash boxes). */
  setLanguage?: (lang: 'ru' | 'uz' | 'en') => void;
  /**
   * Admin: pick a model bundle (*.zip) and copy it to DATA_DIR/model_inbox/.
   * dataDirHint (the server's /health data_dir) is honoured only in dev builds;
   * a packaged app always stages under its own user-data folder. Never rejects.
   */
  importModelBundle?: (opts?: { dataDirHint?: string | null }) => Promise<ShellFileResult>;
  /** Admin: save-dialog for the diagnostics zip; writes it and reveals it in the file manager. Never rejects. */
  saveDiagnostics?: (bytes: Uint8Array, suggestedName: string) => Promise<ShellFileResult>;
}

/**
 * Result of a shell file operation (electron/adminIpc.js). `error` is a reason
 * code (not_a_zip, empty_file, too_large, source_unreadable, copy_failed,
 * copy_incomplete, write_failed, invalid_payload, empty_payload, invalid_source,
 * invalid_target, internal_error); `detail` is technical text for the console.
 */
type ShellFileResult =
  | { ok: true; path: string; name?: string; bytes?: number }
  | { ok: false; canceled?: boolean; error?: string; detail?: string };

interface Window {
  electronAPI?: ElectronAPI;
}
