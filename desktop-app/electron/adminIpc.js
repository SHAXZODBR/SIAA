'use strict';
/**
 * File operations behind the admin IPC channels of the Electron shell
 * (registered in electron/main.js):
 *
 *   'model-bundle:import'  stageModelBundle()  — copy a picked *.zip into
 *                          <DATA_DIR>/model_inbox/<basename> so the server's
 *                          POST /models/import accepts the path.
 *   'diagnostics:save'     writeDiagnostics()  — write the GET /diagnostics zip
 *                          to the path chosen in the save dialog.
 *
 * Kept free of `require('electron')` so `npm test` (node --test) can exercise
 * it directly. Every function resolves to {ok:true, ...} or
 * {ok:false, error:<reason code>, detail:<technical text>} and never throws:
 * the renderer maps the reason codes to localized messages
 * (src/services/adminTools.ts IPC_ERROR_KEYS — keep both lists in sync).
 *
 * Writes go to a temp file in the destination folder first and are renamed
 * into place only after the size is verified, so a half-copied bundle never
 * sits under its final name where the server could pick it up.
 */
const path = require('path');
const fsp = require('fs/promises');

/** Upper bound for a model bundle (the ViT triage bundle is ~350 MB). */
const MAX_BUNDLE_BYTES = 8 * 1024 * 1024 * 1024;
/** Upper bound for a diagnostics zip (log tails are capped at 2 MB each server-side). */
const MAX_DIAGNOSTICS_BYTES = 512 * 1024 * 1024;
const INBOX_DIR = 'model_inbox';

function fail(error, detail) {
  return { ok: false, error, detail: detail ? String(detail).slice(0, 500) : undefined };
}

/**
 * Plain, safe file name ending in `ext`: directory parts dropped, only
 * [A-Za-z0-9._-] kept, no leading dots/underscores. null when nothing is left.
 * Mirrors safeFileName() in src/services/adminTools.ts.
 */
function safeFileName(name, ext) {
  if (typeof name !== 'string') return null;
  const base = name.split(/[\\/]/).pop() || '';
  let cleaned = base.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+/, '').slice(0, 120);
  if (!cleaned || cleaned === ext.replace(/^\./, '')) return null;
  if (!cleaned.toLowerCase().endsWith(ext.toLowerCase())) cleaned += ext;
  return cleaned;
}

/** PK\x03\x04 (local file header) or PK\x05\x06 (empty archive). */
function isZipMagic(buf) {
  if (!buf || buf.length < 4 || buf[0] !== 0x50 || buf[1] !== 0x4b) return false;
  return (buf[2] === 0x03 && buf[3] === 0x04) || (buf[2] === 0x05 && buf[3] === 0x06);
}

/** Uint8Array / ArrayBuffer / Buffer (as delivered over IPC) → Buffer, else null. */
function toBuffer(bytes) {
  if (Buffer.isBuffer(bytes)) return bytes;
  if (bytes instanceof Uint8Array) return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes instanceof ArrayBuffer) return Buffer.from(bytes);
  return null;
}

function tempSibling(target) {
  return `${target}.part-${process.pid}-${Date.now()}`;
}

async function readHead(fs, file, n) {
  const fh = await fs.open(file, 'r');
  try {
    const buf = Buffer.alloc(n);
    const { bytesRead } = await fh.read(buf, 0, n, 0);
    return buf.subarray(0, bytesRead);
  } finally {
    await fh.close();
  }
}

/**
 * Copies `sourcePath` (a model bundle zip) to <dataDir>/model_inbox/<safe basename>.
 * Resolves {ok:true, path, name, bytes} with the absolute destination path.
 */
async function stageModelBundle(sourcePath, dataDir, { fs = fsp } = {}) {
  if (typeof sourcePath !== 'string' || !sourcePath || !path.isAbsolute(sourcePath)) {
    return fail('invalid_source', `not an absolute path: ${sourcePath}`);
  }
  if (typeof dataDir !== 'string' || !dataDir || !path.isAbsolute(dataDir)) {
    return fail('internal_error', `data dir is not an absolute path: ${dataDir}`);
  }
  if (path.extname(sourcePath).toLowerCase() !== '.zip') return fail('not_a_zip', path.basename(sourcePath));

  let size;
  try {
    const st = await fs.stat(sourcePath);
    if (!st.isFile()) return fail('source_unreadable', 'not a regular file');
    size = st.size;
    if (size === 0) return fail('empty_file');
    if (size > MAX_BUNDLE_BYTES) return fail('too_large', `${size} bytes`);
    if (!isZipMagic(await readHead(fs, sourcePath, 4))) return fail('not_a_zip', 'missing zip signature');
  } catch (e) {
    return fail('source_unreadable', e && e.message);
  }

  const name = safeFileName(path.basename(sourcePath), '.zip');
  if (!name) return fail('invalid_source', 'unusable file name');
  const inbox = path.join(dataDir, INBOX_DIR);
  const dest = path.join(inbox, name);

  // Picked straight from the inbox: nothing to copy.
  if (path.resolve(sourcePath) === path.resolve(dest)) return { ok: true, path: dest, name, bytes: size };

  const tmp = tempSibling(dest);
  try {
    await fs.mkdir(inbox, { recursive: true });
    await fs.copyFile(sourcePath, tmp);
    const copied = (await fs.stat(tmp)).size;
    if (copied !== size) {
      await fs.unlink(tmp).catch(() => {});
      return fail('copy_incomplete', `${copied} of ${size} bytes`);
    }
    await fs.rename(tmp, dest);
  } catch (e) {
    await fs.unlink(tmp).catch(() => {});
    return fail('copy_failed', e && e.message);
  }
  return { ok: true, path: dest, name, bytes: size };
}

/**
 * Checks a diagnostics payload before the save dialog is shown.
 * Resolves {ok:true, buffer} or a failure.
 */
function validateDiagnosticsPayload(bytes) {
  const buf = toBuffer(bytes);
  if (!buf) return fail('invalid_payload', typeof bytes);
  if (buf.length === 0) return fail('empty_payload');
  if (buf.length > MAX_DIAGNOSTICS_BYTES) return fail('too_large', `${buf.length} bytes`);
  if (!isZipMagic(buf)) return fail('not_a_zip', 'missing zip signature');
  return { ok: true, buffer: buf };
}

/**
 * Writes the diagnostics zip to `targetPath` (from the save dialog; `.zip`
 * appended when the user typed no extension). Resolves {ok:true, path, bytes}.
 */
async function writeDiagnostics(targetPath, bytes, { fs = fsp } = {}) {
  const v = validateDiagnosticsPayload(bytes);
  if (!v.ok) return v;
  if (typeof targetPath !== 'string' || !targetPath || !path.isAbsolute(targetPath)) {
    return fail('invalid_target', `not an absolute path: ${targetPath}`);
  }
  const target = path.extname(targetPath).toLowerCase() === '.zip' ? targetPath : `${targetPath}.zip`;
  const tmp = tempSibling(target);
  try {
    await fs.writeFile(tmp, v.buffer);
    await fs.rename(tmp, target);
  } catch (e) {
    await fs.unlink(tmp).catch(() => {});
    return fail('write_failed', e && e.message);
  }
  return { ok: true, path: target, bytes: v.buffer.length };
}

module.exports = {
  INBOX_DIR,
  MAX_BUNDLE_BYTES,
  MAX_DIAGNOSTICS_BYTES,
  safeFileName,
  isZipMagic,
  toBuffer,
  stageModelBundle,
  validateDiagnosticsPayload,
  writeDiagnostics,
};
