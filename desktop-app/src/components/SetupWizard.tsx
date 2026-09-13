import React, { useEffect, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { probeHealth } from '../services/api';
import type { HealthStatus, Lang } from '../types';
import { useT, LANGS, langName, langShort } from '../i18n';
import type { I18nKey } from '../i18n';
import { PRODUCT_NAME } from '../services/appInfo';
import LicensePanel from './LicensePanel';

const STEPS = ['language', 'clinic', 'server', 'license', 'support'] as const;
type StepId = (typeof STEPS)[number];

const STEP_LABEL: Record<StepId, I18nKey> = {
  language: 'wizard.stepLanguage',
  clinic: 'wizard.stepClinic',
  server: 'wizard.stepServer',
  license: 'wizard.stepLicense',
  support: 'wizard.stepSupport',
};

interface Draft {
  clinicName: string;
  clinicAddress: string;
  clinicPhone: string;
  inferenceUrl: string;
  supportContact: string;
}

/**
 * First-run wizard. Shown after login until settings.setupComplete is true
 * and a clinic name exists (App.tsx), or when re-run from Preferences.
 * Everything it collects is persisted through the store's updateSettings;
 * the interface language is applied immediately so the wizard itself
 * re-renders in the chosen language.
 */
export default function SetupWizard() {
  const { settings, updateSettings, closeSetup, addNotification } = useAppStore();
  const t = useT();
  const [stepIdx, setStepIdx] = useState(0);
  const [draft, setDraft] = useState<Draft>({
    clinicName: settings.clinicName,
    clinicAddress: settings.clinicAddress,
    clinicPhone: settings.clinicPhone,
    inferenceUrl: settings.inferenceUrl,
    supportContact: settings.supportContact,
  });
  const [clinicError, setClinicError] = useState(false);
  const [probe, setProbe] = useState<{ state: 'idle' | 'testing' | 'done'; health: HealthStatus | null }>({ state: 'idle', health: null });

  const step = STEPS[stepIdx];
  const isLast = stepIdx === STEPS.length - 1;
  // Re-run from Preferences: the workstation is already configured, so Cancel is allowed.
  const canCancel = settings.setupComplete && !!settings.clinicName.trim();

  const patch = (p: Partial<Draft>) => setDraft((d) => ({ ...d, ...p }));

  const commitDraft = (extra: Partial<typeof settings> = {}) => {
    updateSettings({
      clinicName: draft.clinicName.trim(),
      clinicAddress: draft.clinicAddress.trim(),
      clinicPhone: draft.clinicPhone.trim(),
      inferenceUrl: draft.inferenceUrl.trim() || settings.inferenceUrl,
      supportContact: draft.supportContact.trim(),
      ...extra,
    });
  };

  const next = () => {
    if (step === 'clinic' && !draft.clinicName.trim()) {
      setClinicError(true);
      return;
    }
    if (step === 'server') {
      // Apply the URL now so the license step (and the health poll) talk to the right server.
      updateSettings({ inferenceUrl: draft.inferenceUrl.trim() || settings.inferenceUrl });
    }
    if (isLast) {
      finish();
      return;
    }
    setStepIdx((i) => Math.min(STEPS.length - 1, i + 1));
  };

  const back = () => setStepIdx((i) => Math.max(0, i - 1));

  const finish = () => {
    if (!draft.clinicName.trim()) {
      setStepIdx(STEPS.indexOf('clinic'));
      setClinicError(true);
      return;
    }
    commitDraft({ setupComplete: true });
    closeSetup();
    addNotification({ type: 'success', title: t('wizard.done'), message: t('wizard.rerunHint') });
  };

  const cancel = () => {
    if (!canCancel) return;
    closeSetup();
  };

  const testConnection = async () => {
    setProbe({ state: 'testing', health: null });
    const h = await probeHealth(draft.inferenceUrl);
    setProbe({ state: 'done', health: h });
  };

  // Escape = cancel (re-run mode only)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') cancel(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [canCancel]);

  useEffect(() => { setClinicError(false); }, [step]);

  const canSkip = step === 'server' || step === 'license' || step === 'support';

  return (
    <div className="h-screen w-screen bg-ink-950 text-ink-200 flex items-center justify-center overflow-hidden" lang={settings.language}>
      <div className="absolute inset-0 bg-grid opacity-60" />
      <div className="relative w-[960px] max-w-[96vw] h-[640px] max-h-[92vh] glass rounded-xl border border-ink-700/50 shadow-2xl overflow-hidden flex animate-scale-in">
        {/* Stepper */}
        <aside className="w-64 border-r border-ink-800 bg-ink-950/40 flex flex-col">
          <div className="p-5 border-b border-ink-800">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-7 h-7 bg-gradient-to-br from-accent-400 to-accent-700 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <span className="text-xs font-semibold text-ink-200 tracking-wide">{PRODUCT_NAME}</span>
            </div>
            <h1 className="text-base font-bold text-ink-100">{t('wizard.title')}</h1>
            <p className="text-[11px] text-ink-500 mt-0.5">{t('wizard.subtitle')}</p>
          </div>
          <ol className="flex-1 p-3 space-y-1">
            {STEPS.map((s, i) => {
              const active = i === stepIdx;
              const done = i < stepIdx;
              return (
                <li key={s}>
                  <button
                    type="button"
                    onClick={() => i < stepIdx && setStepIdx(i)}
                    disabled={i > stepIdx}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm text-left transition-colors ${
                      active ? 'bg-accent-600/20 text-white' : done ? 'text-ink-300 hover:bg-ink-800/50' : 'text-ink-600 cursor-default'
                    }`}
                  >
                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold font-mono flex-shrink-0 ${
                      active ? 'bg-accent-600 text-white' : done ? 'bg-normal/20 text-normal' : 'bg-ink-800 text-ink-500'
                    }`}>
                      {done ? '✓' : i + 1}
                    </span>
                    <span>{t(STEP_LABEL[s])}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          <div className="p-4 border-t border-ink-800 text-[10px] text-ink-500">
            {t('wizard.step', { n: stepIdx + 1, total: STEPS.length })}
          </div>
        </aside>

        {/* Content */}
        <form
          className="flex-1 flex flex-col min-w-0"
          onSubmit={(e) => { e.preventDefault(); next(); }}
        >
          <div className="flex-1 overflow-y-auto p-8">
            {step === 'language' && (
              <StepFrame title={t('wizard.languageTitle')} hint={t('wizard.languageHint')}>
                <div className="grid grid-cols-3 gap-3">
                  {LANGS.map((l: Lang) => (
                    <button
                      key={l}
                      type="button"
                      onClick={() => updateSettings({ language: l })}
                      className={`p-4 rounded-lg border text-left transition-all ${
                        settings.language === l
                          ? 'border-accent-500 bg-accent-900/20 shadow-glow-accent'
                          : 'border-ink-800 hover:border-ink-600 bg-ink-850'
                      }`}
                    >
                      <div className="text-[10px] font-bold text-ink-500 tracking-wider mb-1">{langShort(l)}</div>
                      <div className="text-base font-semibold text-ink-100">{langName(l)}</div>
                    </button>
                  ))}
                </div>
              </StepFrame>
            )}

            {step === 'clinic' && (
              <StepFrame title={t('wizard.clinicTitle')} hint={t('wizard.clinicHint')}>
                <div className="space-y-4 max-w-lg">
                  <Labeled label={t('settings.clinicName')} desc={t('settings.clinicNameDesc')} required>
                    <input
                      autoFocus
                      value={draft.clinicName}
                      onChange={(e) => { patch({ clinicName: e.target.value }); setClinicError(false); }}
                      className={`input-medical ${clinicError ? '!border-critical' : ''}`}
                    />
                    {clinicError && <div className="text-[11px] text-critical mt-1">{t('wizard.clinicNameRequired')}</div>}
                  </Labeled>
                  <Labeled label={t('settings.clinicAddress')} desc={t('settings.clinicAddressDesc')}>
                    <input value={draft.clinicAddress} onChange={(e) => patch({ clinicAddress: e.target.value })} className="input-medical" />
                  </Labeled>
                  <Labeled label={t('settings.clinicPhone')} desc={t('settings.clinicPhoneDesc')}>
                    <input value={draft.clinicPhone} onChange={(e) => patch({ clinicPhone: e.target.value })} className="input-medical" />
                  </Labeled>
                </div>
              </StepFrame>
            )}

            {step === 'server' && (
              <StepFrame title={t('wizard.serverTitle')} hint={t('wizard.serverHint')}>
                <div className="space-y-4 max-w-lg">
                  <Labeled label={t('settings.inferenceUrl')} desc={t('settings.inferenceUrlDesc')}>
                    <div className="flex items-center gap-2">
                      <input
                        autoFocus
                        value={draft.inferenceUrl}
                        onChange={(e) => { patch({ inferenceUrl: e.target.value }); setProbe({ state: 'idle', health: null }); }}
                        className="input-medical !font-mono"
                        spellCheck={false}
                      />
                      <button type="button" onClick={testConnection} disabled={probe.state === 'testing'} className="btn-secondary whitespace-nowrap">
                        {probe.state === 'testing' ? t('settings.testing') : t('settings.testConnection')}
                      </button>
                    </div>
                  </Labeled>
                  {probe.state === 'done' && probe.health && <HealthSummary health={probe.health} />}
                </div>
              </StepFrame>
            )}

            {step === 'license' && (
              <StepFrame title={t('wizard.licenseTitle')} hint={t('wizard.licenseHint')}>
                <LicensePanel />
              </StepFrame>
            )}

            {step === 'support' && (
              <StepFrame title={t('wizard.supportTitle')} hint={t('wizard.supportHint')}>
                <div className="max-w-lg">
                  <Labeled label={t('settings.supportContact')} desc={t('settings.supportContactDesc')}>
                    <input
                      autoFocus
                      value={draft.supportContact}
                      onChange={(e) => patch({ supportContact: e.target.value })}
                      placeholder={t('wizard.supportPlaceholder')}
                      className="input-medical"
                    />
                  </Labeled>
                </div>
              </StepFrame>
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-4 border-t border-ink-800 bg-ink-950/30 flex items-center gap-2">
            {canCancel && (
              <button type="button" onClick={cancel} className="btn-ghost">{t('common.cancel')}</button>
            )}
            <div className="flex-1" />
            {stepIdx > 0 && (
              <button type="button" onClick={back} className="btn-secondary">{t('common.back')}</button>
            )}
            {canSkip && !isLast && (
              <button type="button" onClick={() => setStepIdx((i) => i + 1)} className="btn-ghost">{t('common.skip')}</button>
            )}
            <button type="submit" className="btn-primary min-w-[120px]">
              {isLast ? t('wizard.finish') : t('common.next')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function StepFrame({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="animate-fade-in">
      <h2 className="text-xl font-bold text-ink-100 mb-1">{title}</h2>
      <p className="text-xs text-ink-400 mb-6 max-w-xl leading-relaxed">{hint}</p>
      {children}
    </div>
  );
}

function Labeled({ label, desc, required, children }: { label: string; desc?: string; required?: boolean; children: React.ReactNode }) {
  const t = useT();
  return (
    <label className="block">
      <div className="text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">
        {label}{required && <span className="ml-1 text-critical normal-case font-normal">· {t('common.required')}</span>}
      </div>
      {children}
      {desc && <div className="text-[10px] text-ink-500 mt-1">{desc}</div>}
    </label>
  );
}

/** Test-connection result: server state, version, device, models, LLM, license mode. */
export function HealthSummary({ health }: { health: HealthStatus }) {
  const t = useT();
  const models = Object.entries(health.models || {});
  const ok = health.reachable && health.status === 'ok';
  const stateLabel = !health.reachable
    ? t('settings.connectionFailed')
    : health.status === 'ok' ? t('settings.connectionOk') : health.status === 'degraded' ? t('common.degraded') : t('common.error');
  const pill = !health.reachable
    ? 'severity-critical'
    : ok ? 'severity-normal' : health.status === 'degraded' ? 'severity-moderate' : 'severity-critical';
  const licenseKey: I18nKey | null = health.license.mode === 'licensed' ? 'license.mode.licensed'
    : health.license.mode === 'unlicensed' ? 'license.mode.unlicensed'
      : health.license.mode === 'demo' ? 'license.mode.demo'
        : health.license.mode === 'dev' ? 'license.mode.dev' : null;

  return (
    <div className="p-4 bg-ink-850 rounded-lg border border-ink-800 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-ink-100">{t('wizard.serverStatus')}</span>
        <span className={`severity-pill ${pill}`}>{stateLabel}</span>
      </div>
      {health.reachable && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
          <KV label={t('settings.serverVersion')} value={health.version || '—'} />
          <KV label={t('settings.device')} value={health.device || '—'} />
          <KV
            label={t('settings.llm')}
            value={health.llm.backend ? `${health.llm.backend} · ${health.llm.reachable ? t('common.reachable') : t('common.unreachable')}` : t('common.unavailable')}
          />
          <KV label={t('license.mode')} value={licenseKey ? t(licenseKey) : (health.license.mode || t('common.unknown'))} />
          <KV label={t('settings.authRequired')} value={health.authRequired ? t('common.yes') : t('common.no')} />
          {health.dataDir && <KV label={t('settings.dataDir')} value={health.dataDir} />}
          <div className="col-span-2">
            <div className="text-ink-500 mb-1">{t('settings.loadedModels')}</div>
            {models.length === 0 ? (
              <div className="text-ink-400">{t('settings.noModelStatus')}</div>
            ) : (
              <div className="space-y-1">
                {models.map(([key, m]) => (
                  <div key={key} className="flex items-center justify-between gap-2 px-2 py-1 bg-ink-900 rounded border border-ink-800">
                    <span className="font-mono text-ink-200 truncate">{key}</span>
                    <span className={`severity-pill text-[9px] flex-shrink-0 ${m.loaded ? 'severity-normal' : 'severity-critical'}`}>
                      {m.loaded ? t('settings.loaded') : t('settings.notLoaded')}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-ink-500 mb-0.5">{label}</div>
      <div className="text-ink-200 font-mono break-all">{value}</div>
    </div>
  );
}
