/**
 * Test helper: loads a dependency-free TypeScript module (e.g.
 * src/services/adminTools.ts) by transpiling it with the project's own
 * `typescript` devDependency and importing the result from a data: URL.
 * Works on every Node version CI uses (no reliance on Node's type stripping)
 * and adds no third-party test dependency.
 */
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const ts = require('typescript');

export const DESKTOP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export async function loadTs(relPath) {
  const file = path.join(DESKTOP, relPath);
  const src = readFileSync(file, 'utf8');
  const out = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020, isolatedModules: true },
    fileName: file,
  }).outputText;
  if (/^\s*import\s+(?!type\b)[^;]*from\s+['"]/m.test(out)) {
    throw new Error(`${relPath} has runtime imports — it must stay dependency-free to be unit-tested`);
  }
  return import(`data:text/javascript;base64,${Buffer.from(out).toString('base64')}`);
}

/** Keys of desktop-app/src/i18n/<lang>.ts (flat `'key': '...'` entries). */
export function i18nKeys(lang = 'en') {
  const src = readFileSync(path.join(DESKTOP, 'src', 'i18n', `${lang}.ts`), 'utf8');
  return new Set([...src.matchAll(/^\s*'([^']+)':\s*'/gm)].map((m) => m[1]));
}

export function readText(relPath) {
  return readFileSync(path.join(DESKTOP, relPath), 'utf8');
}
