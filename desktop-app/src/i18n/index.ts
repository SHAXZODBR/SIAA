/**
 * Interface localization (RU default · UZ Latin · EN).
 *
 *   t(key, params?)          — reads the language from the zustand store
 *                              (for code that runs outside React).
 *   translate(lang, key, …)  — explicit language.
 *   useT()                   — hook; the returned `t` re-renders the component
 *                              whenever settings.language changes.
 *   useLang()                — hook returning the current Lang.
 *   formatDate / formatDateTime / formatTime / formatNumber
 *                            — Intl formatting in the selected locale.
 *
 * Fallback order for a missing key: selected language → English → the key.
 */
import { useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import type { Lang } from '../types';
import { en } from './en';
import { ru } from './ru';
import { uz } from './uz';
import type { I18nKey } from './en';

export type { I18nKey };
export type I18nParams = Record<string, string | number>;

export const LANGS: Lang[] = ['ru', 'uz', 'en'];
export const LOCALES: Record<Lang, string> = { ru: 'ru-RU', uz: 'uz-UZ', en: 'en-US' };

const DICTS: Record<Lang, Record<I18nKey, string>> = { en, ru, uz };

function interpolate(template: string, params?: I18nParams): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (m, name: string) =>
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : m,
  );
}

export function isLang(v: unknown): v is Lang {
  return v === 'ru' || v === 'uz' || v === 'en';
}

export function translate(lang: Lang, key: I18nKey, params?: I18nParams): string {
  const dict = DICTS[lang] || en;
  const raw = dict[key] ?? en[key] ?? key;
  return interpolate(raw, params);
}

/** Current language from the store (safe outside React). */
export function currentLang(): Lang {
  const l = useAppStore.getState().settings.language;
  return isLang(l) ? l : 'ru';
}

export function t(key: I18nKey, params?: I18nParams): string {
  return translate(currentLang(), key, params);
}

export function useLang(): Lang {
  const l = useAppStore((s) => s.settings.language);
  return isLang(l) ? l : 'ru';
}

export type TFn = (key: I18nKey, params?: I18nParams) => string;

export function useT(): TFn {
  const lang = useLang();
  return useCallback<TFn>((key, params) => translate(lang, key, params), [lang]);
}

/** Native language names / short badges for switchers. */
export function langName(l: Lang): string {
  return translate(l, l === 'ru' ? 'lang.ru' : l === 'uz' ? 'lang.uz' : 'lang.en');
}
export function langShort(l: Lang): string {
  return translate(l, l === 'ru' ? 'lang.short.ru' : l === 'uz' ? 'lang.short.uz' : 'lang.short.en');
}

// ─── Intl formatting ────────────────────────────────────────────────────────

function toDate(v: string | number | Date | null | undefined): Date | null {
  if (v === null || v === undefined || v === '') return null;
  if (v instanceof Date) return isNaN(v.getTime()) ? null : v;
  const s = String(v).trim();
  // DICOM DA 'YYYYMMDD'
  if (/^\d{8}$/.test(s)) {
    const d = new Date(`${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T00:00:00`);
    return isNaN(d.getTime()) ? null : d;
  }
  // Plain ISO date 'YYYY-MM-DD' → local midnight (avoid UTC shift to the previous day)
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    const d = new Date(`${s}T00:00:00`);
    return isNaN(d.getTime()) ? null : d;
  }
  const d = new Date(v as string | number);
  return isNaN(d.getTime()) ? null : d;
}

function safeFormat(fn: () => string, fallback: string): string {
  try { return fn(); } catch { return fallback; }
}

/** Date only, e.g. 15.01.2025 (ru) · 15/01/2025 (uz) · 1/15/2025 (en). Unparseable input is returned as-is. */
export function formatDate(v: string | number | Date | null | undefined, lang: Lang): string {
  const d = toDate(v);
  if (!d) return v == null ? '' : String(v);
  return safeFormat(() => new Intl.DateTimeFormat(LOCALES[lang], { year: 'numeric', month: '2-digit', day: '2-digit' }).format(d), String(v));
}

/** Date + time (minutes). */
export function formatDateTime(v: string | number | Date | null | undefined, lang: Lang): string {
  const d = toDate(v);
  if (!d) return v == null ? '' : String(v);
  return safeFormat(
    () => new Intl.DateTimeFormat(LOCALES[lang], { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(d),
    String(v),
  );
}

/** Time only (HH:MM, 24h in every supported locale). */
export function formatTime(v: string | number | Date | null | undefined, lang: Lang, withSeconds = false): string {
  const d = toDate(v);
  if (!d) return v == null ? '' : String(v);
  return safeFormat(
    () => new Intl.DateTimeFormat(LOCALES[lang], { hour: '2-digit', minute: '2-digit', ...(withSeconds ? { second: '2-digit' } : {}), hour12: false }).format(d),
    String(v),
  );
}

export function formatNumber(n: number, lang: Lang, options?: Intl.NumberFormatOptions): string {
  return safeFormat(() => new Intl.NumberFormat(LOCALES[lang], options).format(n), String(n));
}

/** 0.83 → "83 %" style percentage in the selected locale. */
export function formatPercent(fraction: number, lang: Lang, digits = 0): string {
  return safeFormat(
    () => new Intl.NumberFormat(LOCALES[lang], { style: 'percent', maximumFractionDigits: digits }).format(fraction),
    `${Math.round(fraction * 100)}%`,
  );
}

/** Number of keys per dictionary — surfaced in Settings → Language for honesty. */
export function dictionarySizes(): Record<Lang, number> {
  return { ru: Object.keys(ru).length, uz: Object.keys(uz).length, en: Object.keys(en).length };
}
