/**
 * Pure helpers behind the administrator tools in Preferences
 * (AI settings toggle, model-bundle import, diagnostics export) and the
 * localisation of server reason codes shown to doctors.
 *
 * This module deliberately has NO runtime imports (type-only imports are
 * erased), so `npm test` can load it directly with Node's built-in test runner
 * and TypeScript type stripping — see desktop-app/tests/adminTools.test.mjs.
 * Keep it that way: no axios, no store, no React here.
 */
import type { I18nKey } from '../i18n/en';
import type { AiSettings, AiSettingsSource, ModelImportResult } from '../types';

/** Server rejection reason (decision D1): the study type has no validated detector. */
export const NO_VALIDATED_DETECTOR = 'no_validated_detector';

/** True when the server returned the D1 "no validated detector" not-analysed state. */
export function isNoValidatedDetector(reason: string | null | undefined): boolean {
  return typeof reason === 'string' && reason.trim().toLowerCase() === NO_VALIDATED_DETECTOR;
}

/**
 * Server reason: pending detectors are shown, yet no AI detector could run on
 * the study (model not installed, or no suitable series) — accepted, not analysed.
 */
export const NO_DETECTOR_RAN = 'no_detector_ran';

/** True when the server returned the "no detector could run" not-analysed state. */
export function isNoDetectorRan(reason: string | null | undefined): boolean {
  return typeof reason === 'string' && reason.trim().toLowerCase() === NO_DETECTOR_RAN;
}

/**
 * Doctor-facing text for a rejection / requires-review reason. Known machine
 * codes become a localized sentence (i18n key); anything else is shown as the
 * server sent it (it is already human text from the server). Returns null when
 * there is nothing to show.
 */
export function rejectionReasonKey(reason: string | null | undefined): I18nKey | null {
  if (isNoValidatedDetector(reason)) return 'findings.noValidatedDetector';
  if (isNoDetectorRan(reason)) return 'findings.noDetectorRan';
  return null;
}

// ─── Roles ──────────────────────────────────────────────────────────────────

/** Admin-only controls are enabled for role === 'admin' (server enforces it too). */
export function isAdminRole(role: string | null | undefined): boolean {
  return typeof role === 'string' && role.trim().toLowerCase() === 'admin';
}

// ─── /settings/ai ───────────────────────────────────────────────────────────

const AI_SOURCES: AiSettingsSource[] = ['file', 'env', 'default'];

/**
 * Validates a GET/PUT /settings/ai body (or /health "ai_settings").
 * Returns null when the shape is not the contract's — callers must then treat
 * the setting as unknown rather than guess a value.
 */
export function parseAiSettings(d: unknown): AiSettings | null {
  if (!d || typeof d !== 'object') return null;
  const o = d as Record<string, unknown>;
  if (typeof o.show_pending_detectors !== 'boolean') return null;
  const source = AI_SOURCES.includes(o.source as AiSettingsSource) ? (o.source as AiSettingsSource) : 'default';
  return { showPendingDetectors: o.show_pending_detectors, source };
}

export const AI_SOURCE_KEYS: Record<AiSettingsSource, I18nKey> = {
  file: 'settings.aiSourceFile',
  env: 'settings.aiSourceEnv',
  default: 'settings.aiSourceDefault',
};

// ─── POST /models/import ────────────────────────────────────────────────────

/**
 * Validates the 200 body of POST /models/import. Throws when the server says
 * ok but the identity is missing — a "success" toast without an identity
 * would hide a broken install.
 */
export function parseModelImportResult(d: unknown): ModelImportResult {
  const o = (d && typeof d === 'object' ? d : {}) as Record<string, unknown>;
  const identity = typeof o.identity === 'string' ? o.identity.trim() : '';
  if (o.ok !== true || !/^[0-9a-f]{12}$/i.test(identity)) {
    throw new Error(`unexpected /models/import response: ${JSON.stringify(d)?.slice(0, 200)}`);
  }
  return {
    ok: true,
    identity: identity.toLowerCase(),
    encryptedAtRest: o.encrypted_at_rest === true,
    status: typeof o.status === 'string' ? o.status : 'unknown',
  };
}

/** Reason-code families a failed bundle import is reported under. */
export type ModelImportFailure =
  | 'not_for_this_machine'
  | 'no_model_key'
  | 'in_progress'
  | 'too_large'
  | 'not_writable'
  | 'not_a_bundle'
  | 'manifest_mismatch'
  | 'outside_inbox'
  | 'forbidden'
  | 'auth'
  | 'network'
  | 'staging_failed'
  | 'unknown';

export const MODEL_IMPORT_ERROR_KEYS: Record<ModelImportFailure, I18nKey> = {
  not_for_this_machine: 'settings.importErrNotForMachine',
  no_model_key: 'settings.importErrNoKey',
  in_progress: 'settings.importErrInProgress',
  too_large: 'settings.ipcErrTooLarge',
  not_writable: 'settings.importErrNotWritable',
  not_a_bundle: 'settings.importErrNotBundle',
  manifest_mismatch: 'settings.importErrManifest',
  outside_inbox: 'settings.importErrInbox',
  forbidden: 'error.forbidden',
  auth: 'error.auth',
  network: 'health.aiServerDown',
  staging_failed: 'settings.importErrStaging',
  unknown: 'settings.importErrUnknown',
};

/**
 * Pulls the machine reason code out of an error body. The server may send
 * `{"reason": "..."}`, `{"detail": "..."}` or FastAPI's
 * `{"detail": {"reason"|"code"|"error": "..."}}` — all are accepted.
 */
export function extractReasonCode(data: unknown): string | null {
  if (!data || typeof data !== 'object') return typeof data === 'string' && data.trim() ? data.trim() : null;
  const o = data as Record<string, unknown>;
  const pick = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v.trim() : null);
  const detail = o.detail;
  if (detail && typeof detail === 'object') {
    const d = detail as Record<string, unknown>;
    const fromDetail = pick(d.reason) || pick(d.code) || pick(d.error);
    if (fromDetail) return fromDetail;
  }
  return pick(o.reason) || pick(o.code) || pick(detail) || pick(o.error);
}

/**
 * Maps a reason code (+ HTTP status) to a failure family. Order matters:
 * "wrong machine" first (the most actionable for the vendor), then no model
 * key on this PC (license.dat not imported yet), operational refusals (import
 * already running, too large, models folder read-only), a bad path / file not
 * in DATA_DIR/model_inbox, then a structurally broken zip, then checksum /
 * manifest / sidecar disagreements. Server codes: src/inference/server.py
 * _ModelImportFailure; error body {ok:false, reason, detail}.
 */
export function classifyModelImportFailure(
  reason: string | null | undefined,
  httpStatus: number | null | undefined,
): ModelImportFailure {
  const r = (reason || '').toLowerCase();
  if (r) {
    if (r.includes('not_for_this_machine') || r.includes('invalidtag') || r.includes('invalid_tag')) return 'not_for_this_machine';
    if (/model_key|no_key|key_unavailable|key_missing|no_license/.test(r)) return 'no_model_key';
    if (r.includes('in_progress')) return 'in_progress';
    if (r.includes('too_large')) return 'too_large';
    if (/not_writable|read_only|readonly/.test(r)) return 'not_writable';
    if (/^path_|inbox|outside|bundle_not_found/.test(r)) return 'outside_inbox';
    if (/not_a_bundle|not_a?_?zip|bad_?zip|invalid_(zip|bundle|layout)|bundle_invalid|bundle_layout|incomplete|contains_plaintext|missing|no_manifest|empty/.test(r)) return 'not_a_bundle';
    if (/manifest|sidecar|sha|mismatch|checksum|hash|integrity/.test(r)) return 'manifest_mismatch';
  }
  if (httpStatus === 401) return 'auth';
  if (httpStatus === 403) return 'forbidden';
  if (httpStatus == null || httpStatus === 0) return reason ? 'unknown' : 'network';
  return 'unknown';
}

// ─── GET /diagnostics ───────────────────────────────────────────────────────

/** `sentinel-diagnostics-<YYYYMMDD-HHMM>.zip` in local time (server format). */
export function defaultDiagnosticsName(now: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `sentinel-diagnostics-${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}.zip`;
}

/**
 * Reduces any suggested name to a safe plain file name ending in `ext`:
 * directory parts dropped, only [A-Za-z0-9._-] kept, no leading dots.
 * Returns null when nothing usable is left.
 */
export function safeFileName(name: string | null | undefined, ext: string): string | null {
  if (typeof name !== 'string') return null;
  const base = name.split(/[\\/]/).pop() || '';
  let cleaned = base.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+/, '').slice(0, 120);
  if (!cleaned || cleaned === ext.replace(/^\./, '')) return null;
  if (!cleaned.toLowerCase().endsWith(ext.toLowerCase())) cleaned += ext;
  return cleaned;
}

/**
 * File name from a Content-Disposition header (RFC 6266: `filename*=UTF-8''…`
 * wins over `filename="…"`), sanitised with safeFileName; `fallback` when the
 * header is missing or unusable.
 */
export function filenameFromContentDisposition(header: string | null | undefined, fallback: string): string {
  if (typeof header === 'string' && header) {
    const star = /filename\*\s*=\s*(?:[\w-]+)?'[^']*'([^;]+)/i.exec(header);
    if (star) {
      try {
        const s = safeFileName(decodeURIComponent(star[1].trim().replace(/^"|"$/g, '')), '.zip');
        if (s) return s;
      } catch {
        // malformed percent-encoding — fall through to the plain parameter
      }
    }
    const plain = /filename\s*=\s*(?:"([^"]*)"|([^;\s]+))/i.exec(header);
    const s = safeFileName(plain ? (plain[1] ?? plain[2]) : null, '.zip');
    if (s) return s;
  }
  return fallback;
}

/** Local-file-header (PK\x03\x04) or empty-archive (PK\x05\x06) signature. */
export function looksLikeZip(bytes: Uint8Array | null | undefined): boolean {
  if (!bytes || bytes.length < 4 || bytes[0] !== 0x50 || bytes[1] !== 0x4b) return false;
  return (bytes[2] === 0x03 && bytes[3] === 0x04) || (bytes[2] === 0x05 && bytes[3] === 0x06);
}

// ─── Electron IPC error codes (electron/adminIpc.js) ────────────────────────

/** Reason codes returned by the shell's 'model-bundle:import' / 'diagnostics:save'. */
export const IPC_ERROR_KEYS: Record<string, I18nKey> = {
  not_a_zip: 'settings.ipcErrNotZip',
  empty_file: 'settings.ipcErrEmpty',
  too_large: 'settings.ipcErrTooLarge',
  source_unreadable: 'settings.ipcErrUnreadable',
  copy_failed: 'settings.ipcErrCopy',
  copy_incomplete: 'settings.ipcErrCopy',
  write_failed: 'settings.ipcErrWrite',
  invalid_payload: 'settings.ipcErrPayload',
  empty_payload: 'settings.ipcErrPayload',
  invalid_source: 'settings.ipcErrUnreadable',
  invalid_target: 'settings.ipcErrWrite',
  internal_error: 'settings.ipcErrInternal',
};

/** Localized key for a shell IPC error code (unknown codes → generic). */
export function ipcErrorKey(code: string | null | undefined): I18nKey {
  return (code && IPC_ERROR_KEYS[code]) || 'settings.ipcErrInternal';
}
