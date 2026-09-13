import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { getLicenseStatus } from '../services/api';
import type { LicenseStatus } from '../types';
import { useT, useLang, formatDate } from '../i18n';
import type { I18nKey } from '../i18n';

const MODE_KEYS: Record<string, I18nKey> = {
  licensed: 'license.mode.licensed',
  unlicensed: 'license.mode.unlicensed',
  demo: 'license.mode.demo',
  dev: 'license.mode.dev',
};

/**
 * Client-side sanity check before a file is written as license.dat: the
 * server's sign_license() emits base64(JSON{payload, signature, version}).
 * The signature itself is only verified by the server.
 */
export function looksLikeLicense(text: string): boolean {
  try {
    const bundle = JSON.parse(atob(text.trim()));
    return !!bundle && typeof bundle.payload === 'string' && typeof bundle.signature === 'string';
  } catch {
    return false;
  }
}

interface FingerprintState {
  loading: boolean;
  value: string | null;
  error: string | null;
}

/**
 * License status (GET /license/status), the server-computed machine
 * fingerprint with a copy button, and "Import license.dat" through the
 * Electron bridge. Shared by the setup wizard and Preferences → License.
 */
export default function LicensePanel() {
  const { addNotification, settings } = useAppStore();
  const t = useT();
  const lang = useLang();
  const [status, setStatus] = useState<LicenseStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [fp, setFp] = useState<FingerprintState>({ loading: false, value: null, error: null });
  const [copied, setCopied] = useState(false);
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const bridge = typeof window !== 'undefined' ? window.electronAPI : undefined;
  const canImport = !!bridge?.writeLicense;

  const refreshStatus = useCallback(async () => {
    setLoadingStatus(true);
    try {
      setStatus(await getLicenseStatus());
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  useEffect(() => { refreshStatus(); }, [refreshStatus, settings.inferenceUrl]);

  const loadFingerprint = useCallback(async () => {
    if (!bridge?.getServerFingerprint) {
      setFp({ loading: false, value: null, error: 'no-bridge' });
      return;
    }
    setFp({ loading: true, value: null, error: null });
    try {
      const res = await bridge.getServerFingerprint();
      setFp({ loading: false, value: res?.fingerprint || null, error: res?.fingerprint ? null : (res?.error || 'unknown') });
    } catch (e) {
      setFp({ loading: false, value: null, error: String(e) });
    }
  }, [bridge]);

  useEffect(() => { loadFingerprint(); }, [loadFingerprint]);

  const copyFingerprint = async () => {
    if (!fp.value) return;
    try {
      await navigator.clipboard.writeText(fp.value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.warn('Clipboard write failed:', e);
    }
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file || !bridge?.writeLicense) return;
    setImporting(true);
    try {
      const text = (await file.text()).trim();
      if (!looksLikeLicense(text)) {
        addNotification({ type: 'error', title: t('license.importInvalid') });
        return;
      }
      const res = await bridge.writeLicense(text);
      if (res?.success) {
        setImported(true);
        addNotification({ type: 'success', title: t('license.imported'), message: t('license.restartHint') });
      } else {
        addNotification({ type: 'error', title: t('license.importFailed'), message: res?.error });
      }
    } catch (err) {
      console.error('License import failed:', err);
      addNotification({ type: 'error', title: t('license.importFailed'), message: String(err) });
    } finally {
      setImporting(false);
    }
  };

  const modeKey = status?.mode ? MODE_KEYS[status.mode] : undefined;
  const valid = !!status?.reachable && status.valid;
  const pill = !status?.reachable
    ? 'bg-ink-800 text-ink-400 border border-ink-700'
    : valid ? 'severity-normal' : 'severity-moderate';

  return (
    <div className="space-y-4">
      {/* Status from the server */}
      <div className="p-4 bg-ink-850 rounded-lg border border-ink-800">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm font-semibold text-ink-100">{t('license.status')}</div>
          <div className="flex items-center gap-2">
            <span className={`severity-pill ${pill}`}>
              {!status?.reachable ? t('common.offline') : valid ? t('license.valid') : t('license.invalid')}
            </span>
            <button onClick={refreshStatus} disabled={loadingStatus} className="btn-ghost !py-0.5 !px-2 text-[10px]">
              {loadingStatus ? t('settings.testing') : t('settings.refresh')}
            </button>
          </div>
        </div>

        {!status?.reachable ? (
          <div className="text-xs text-ink-400">{t('license.serverUnreachable')}</div>
        ) : (
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
            <Field label={t('license.mode')} value={modeKey ? t(modeKey) : (status.mode || t('common.unknown'))} />
            <Field label={t('license.customer')} value={status.customer || '—'} />
            <Field label={t('license.tier')} value={status.tier || '—'} />
            <Field label={t('license.expires')} value={status.expiresAt ? formatDate(status.expiresAt, lang) : '—'} />
            {status.features.length > 0 && (
              <Field label={t('license.features')} value={status.features.join(', ')} wide />
            )}
            {!valid && status.reason && <Field label={t('license.reason')} value={status.reason} wide />}
            {!valid && status.demoCallsToday != null && status.demoLimit != null && (
              <div className="col-span-2 text-[11px] text-moderate">
                {t('license.demoCalls', { n: status.demoCallsToday, limit: status.demoLimit })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Machine fingerprint (server recipe) */}
      <div className="p-4 bg-ink-850 rounded-lg border border-ink-800">
        <div className="text-sm font-semibold text-ink-100 mb-1">{t('license.fingerprint')}</div>
        <div className="text-[11px] text-ink-500 mb-3">{t('license.fingerprintHint')}</div>
        {fp.loading ? (
          <div className="text-xs text-ink-400">{t('license.fingerprintLoading')}</div>
        ) : fp.value ? (
          <div className="flex items-center gap-2">
            <code className="flex-1 min-w-0 px-2 py-1.5 bg-ink-950 border border-ink-800 rounded text-[11px] font-mono text-ink-200 break-all select-all">
              {fp.value}
            </code>
            <button onClick={copyFingerprint} className="btn-secondary flex-shrink-0">
              {copied ? t('common.copied') : t('common.copy')}
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-xs text-ink-400">{t('license.fingerprintUnavailable')}</div>
            {fp.error && fp.error !== 'no-bridge' && (
              <div className="text-[10px] text-ink-600 font-mono break-all">{fp.error}</div>
            )}
            {bridge?.getServerFingerprint && (
              <button onClick={loadFingerprint} className="btn-ghost !px-2 text-[10px]">{t('common.retry')}</button>
            )}
          </div>
        )}
      </div>

      {/* Import license.dat */}
      <div className="p-4 bg-ink-850 rounded-lg border border-ink-800">
        <div className="text-sm font-semibold text-ink-100 mb-1">{t('license.import')}</div>
        <div className="text-[11px] text-ink-500 mb-3">{t('license.importHint')}</div>
        {canImport ? (
          <div className="flex items-center gap-3">
            <input ref={fileRef} type="file" accept=".dat,.lic,.txt,*/*" onChange={handleImportFile} style={{ display: 'none' }} />
            <button onClick={() => fileRef.current?.click()} disabled={importing} className="btn-primary">
              {t('license.import')}
            </button>
            {imported && <span className="text-[11px] text-normal">{t('license.imported')} · {t('license.restartHint')}</span>}
          </div>
        ) : (
          <div className="text-xs text-ink-400">{t('license.notElectron')}</div>
        )}
      </div>
    </div>
  );
}

function Field({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? 'col-span-2 min-w-0' : 'min-w-0'}>
      <div className="text-ink-500 mb-0.5">{label}</div>
      <div className="text-ink-200 font-mono break-words">{value}</div>
    </div>
  );
}
