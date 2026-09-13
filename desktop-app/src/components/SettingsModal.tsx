import React, { useCallback, useEffect, useState } from 'react';
import { useAppStore } from '../store/appStore';
import type { AppSettings, AvailableModel, HealthStatus, Lang } from '../types';
import { probeHealth, listAvailableModels } from '../services/api';
import { statusWord } from '../services/findingTranslations';
import { useT, useLang, LANGS, langName, dictionarySizes } from '../i18n';
import type { I18nKey } from '../i18n';
import { PRODUCT_NAME, VENDOR_NAME, VENDOR_SITE, getPackageVersion } from '../services/appInfo';
import LicensePanel from './LicensePanel';
import { HealthSummary } from './SetupWizard';

type SectionId = 'general' | 'connections' | 'language' | 'ai' | 'license' | 'shortcuts' | 'about';

const SECTIONS: { id: SectionId; labelKey: I18nKey; icon: string }[] = [
  { id: 'general', labelKey: 'settings.general', icon: '⚕' },
  { id: 'connections', labelKey: 'settings.connections', icon: '⇄' },
  { id: 'language', labelKey: 'settings.language', icon: '🌐' },
  { id: 'ai', labelKey: 'settings.ai', icon: '🧠' },
  { id: 'license', labelKey: 'settings.license', icon: '🔑' },
  { id: 'shortcuts', labelKey: 'settings.shortcuts', icon: '⌨' },
  { id: 'about', labelKey: 'settings.about', icon: 'ⓘ' },
];

function isSection(v: string): v is SectionId {
  return SECTIONS.some((s) => s.id === v);
}

/**
 * Preferences. Only settings that this build actually honours are shown —
 * clinic identity (letterhead), server endpoints, language, live model and
 * license status, the real shortcut table and product identity.
 */
export default function SettingsModal() {
  const { settingsOpen, settingsSection, closeSettings, settings, updateSettings } = useAppStore();
  const t = useT();
  const [section, setSection] = useState<SectionId>(isSection(settingsSection) ? settingsSection : 'general');

  // Follow openSettings('shortcuts') / ('about') from menus and the palette.
  useEffect(() => {
    if (settingsOpen) setSection(isSection(settingsSection) ? settingsSection : 'general');
  }, [settingsOpen, settingsSection]);

  useEffect(() => {
    if (!settingsOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') closeSettings(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [settingsOpen, closeSettings]);

  if (!settingsOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-ink-950/70 backdrop-blur-sm animate-fade-in"
      onClick={closeSettings}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('settings.title')}
        className="w-[920px] h-[660px] max-w-[95vw] max-h-[90vh] glass rounded-xl border border-ink-700/50 shadow-2xl overflow-hidden flex animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Sidebar */}
        <div className="w-56 border-r border-ink-800 bg-ink-950/30 flex flex-col">
          <div className="p-4 border-b border-ink-800">
            <h2 className="text-sm font-bold text-ink-100">{t('settings.title')}</h2>
            <p className="text-[10px] text-ink-500 mt-0.5">{PRODUCT_NAME} v{getPackageVersion()}</p>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                onClick={() => setSection(s.id)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors ${
                  section === s.id ? 'bg-accent-600/20 text-white' : 'text-ink-300 hover:bg-ink-800/50 hover:text-white'
                }`}
              >
                <span className="text-base w-5 text-center">{s.icon}</span>
                <span>{t(s.labelKey)}</span>
              </button>
            ))}
          </div>
          <div className="p-3 border-t border-ink-800">
            <button onClick={closeSettings} className="w-full btn-ghost text-center">{t('common.close')}</button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-w-0">
          {section === 'general' && <GeneralSection settings={settings} update={updateSettings} />}
          {section === 'connections' && <ConnectionsSection settings={settings} update={updateSettings} />}
          {section === 'language' && <LanguageSection settings={settings} update={updateSettings} />}
          {section === 'ai' && <AISection />}
          {section === 'license' && <LicenseSection />}
          {section === 'shortcuts' && <ShortcutsSection />}
          {section === 'about' && <AboutSection />}
        </div>
      </div>
    </div>
  );
}

// ─── Building blocks ─────────────────────────────────────────────────────────

interface SectionProps {
  settings: AppSettings;
  update: (patch: Partial<AppSettings>) => void;
}

function SectionHeader({ title, desc }: { title: string; desc?: string }) {
  return (
    <div className="p-6 border-b border-ink-800">
      <h3 className="text-lg font-bold text-ink-100">{title}</h3>
      {desc && <p className="text-xs text-ink-400 mt-0.5">{desc}</p>}
    </div>
  );
}

function Row({ label, desc, children, stacked }: { label: string; desc?: string; children: React.ReactNode; stacked?: boolean }) {
  return (
    <div className={`py-3 border-b border-ink-800/50 ${stacked ? 'space-y-2' : 'flex items-start justify-between gap-4'}`}>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-ink-100">{label}</div>
        {desc && <div className="text-xs text-ink-500 mt-0.5">{desc}</div>}
      </div>
      <div className={stacked ? '' : 'flex-shrink-0'}>{children}</div>
    </div>
  );
}

// ─── Clinic ──────────────────────────────────────────────────────────────────

function GeneralSection({ settings, update }: SectionProps) {
  const t = useT();
  const { openSetup } = useAppStore();
  const missing = !settings.clinicName.trim();
  return (
    <div>
      <SectionHeader title={t('settings.generalTitle')} desc={t('settings.generalDesc')} />
      <div className="p-6 space-y-1">
        <Row label={t('settings.clinicName')} desc={t('settings.clinicNameDesc')}>
          <input
            type="text"
            value={settings.clinicName}
            onChange={(e) => update({ clinicName: e.target.value })}
            className={`input-medical !py-1 !w-72 ${missing ? '!border-critical' : ''}`}
          />
        </Row>
        <Row label={t('settings.clinicAddress')} desc={t('settings.clinicAddressDesc')}>
          <input type="text" value={settings.clinicAddress} onChange={(e) => update({ clinicAddress: e.target.value })} className="input-medical !py-1 !w-72" />
        </Row>
        <Row label={t('settings.clinicPhone')} desc={t('settings.clinicPhoneDesc')}>
          <input type="text" value={settings.clinicPhone} onChange={(e) => update({ clinicPhone: e.target.value })} className="input-medical !py-1 !w-72" />
        </Row>
        <Row label={t('settings.supportContact')} desc={t('settings.supportContactDesc')}>
          <input
            type="text"
            value={settings.supportContact}
            onChange={(e) => update({ supportContact: e.target.value })}
            placeholder={t('wizard.supportPlaceholder')}
            className="input-medical !py-1 !w-72"
          />
        </Row>
        <Row label={t('settings.runSetup')} desc={t('settings.runSetupDesc')}>
          <button onClick={openSetup} className="btn-secondary">{t('settings.runSetup')}</button>
        </Row>
      </div>
    </div>
  );
}

// ─── Connections ─────────────────────────────────────────────────────────────

function ConnectionsSection({ settings, update }: SectionProps) {
  const t = useT();
  const { orthancConnected, health } = useAppStore();
  const [probe, setProbe] = useState<{ state: 'idle' | 'testing' | 'done'; health: HealthStatus | null }>({ state: 'idle', health: null });

  const test = async () => {
    setProbe({ state: 'testing', health: null });
    const h = await probeHealth(settings.inferenceUrl);
    setProbe({ state: 'done', health: h });
  };

  const aiLabel = !health || !health.reachable
    ? t('common.offline')
    : health.status === 'ok' ? t('common.ready') : health.status === 'degraded' ? t('common.degraded') : t('common.error');
  const aiColor = !health || !health.reachable ? 'text-ink-500' : health.status === 'ok' ? 'text-normal' : health.status === 'degraded' ? 'text-moderate' : 'text-critical';

  return (
    <div>
      <SectionHeader title={t('settings.connectionsTitle')} desc={t('settings.connectionsDesc')} />
      <div className="p-6 space-y-1">
        <Row label={t('settings.inferenceUrl')} desc={t('settings.inferenceUrlDesc')} stacked>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={settings.inferenceUrl}
              onChange={(e) => { update({ inferenceUrl: e.target.value }); setProbe({ state: 'idle', health: null }); }}
              className="input-medical !py-1 !font-mono"
              spellCheck={false}
            />
            <button onClick={test} disabled={probe.state === 'testing'} className="btn-secondary whitespace-nowrap">
              {probe.state === 'testing' ? t('settings.testing') : t('settings.testConnection')}
            </button>
          </div>
          {probe.state === 'done' && probe.health && <HealthSummary health={probe.health} />}
        </Row>
        <Row label={t('settings.orthancUrl')} desc={t('settings.orthancUrlDesc')}>
          <input
            type="text"
            value={settings.orthancUrl}
            onChange={(e) => update({ orthancUrl: e.target.value })}
            className="input-medical !py-1 !w-72 !font-mono"
            spellCheck={false}
          />
        </Row>

        <div className="mt-6 p-3 bg-ink-850 rounded-lg border border-ink-800">
          <div className="text-xs font-semibold text-ink-100 mb-2">{t('settings.connectionStatus')}</div>
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className={`flex items-center gap-1.5 ${orthancConnected ? 'text-normal' : 'text-ink-500'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${orthancConnected ? 'bg-normal animate-pulse' : 'bg-ink-600'}`} />
              {t('settings.pacs')}: {orthancConnected ? t('common.online') : t('common.offline')}
            </div>
            <div className={`flex items-center gap-1.5 ${aiColor}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${health?.reachable && health.status === 'ok' ? 'bg-normal animate-pulse' : 'bg-ink-600'}`} />
              {t('settings.aiServer')}: {aiLabel}
            </div>
            {health?.reachable && (
              <>
                <div className="text-ink-400">{t('settings.authRequired')}: {health.authRequired ? t('common.yes') : t('common.no')}</div>
                {health.dataDir && <div className="text-ink-400 font-mono truncate col-span-2" title={health.dataDir}>{t('settings.dataDir')}: {health.dataDir}</div>}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Language ────────────────────────────────────────────────────────────────

function LanguageSection({ settings, update }: SectionProps) {
  const t = useT();
  const sizes = dictionarySizes();
  return (
    <div>
      <SectionHeader title={t('settings.languageTitle')} desc={t('settings.languageDesc')} />
      <div className="p-6 space-y-2">
        <div className="text-sm font-medium text-ink-100 mb-2">{t('settings.interfaceLanguage')}</div>
        {LANGS.map((l: Lang) => (
          <button
            key={l}
            onClick={() => update({ language: l })}
            className={`w-full p-3 flex items-center justify-between rounded-lg border transition-all ${
              settings.language === l ? 'border-accent-500 bg-accent-900/20' : 'border-ink-800 hover:border-ink-700 bg-ink-850'
            }`}
          >
            <div className="text-left">
              <div className="text-sm font-semibold text-ink-100">{langName(l)}</div>
              <div className="text-[10px] text-ink-500 font-mono">{sizes[l]} / {sizes.en}</div>
            </div>
            {settings.language === l && (
              <svg className="w-5 h-5 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </button>
        ))}
        <div className="mt-4 p-3 bg-ink-850 rounded-lg border border-ink-800 text-[11px] text-ink-400 leading-relaxed">
          {t('settings.reportLanguageNote')}
        </div>
      </div>
    </div>
  );
}

// ─── AI models ───────────────────────────────────────────────────────────────

function AISection() {
  const { health, settings } = useAppStore();
  const t = useT();
  const lang = useLang();
  const models = health?.models ? Object.entries(health.models) : [];
  const [registry, setRegistry] = useState<AvailableModel[] | null>(null);
  const [loadingRegistry, setLoadingRegistry] = useState(false);

  const loadRegistry = useCallback(async () => {
    setLoadingRegistry(true);
    try {
      setRegistry(await listAvailableModels());
    } catch (e) {
      console.warn('models/available failed:', e);
      setRegistry([]);
    } finally {
      setLoadingRegistry(false);
    }
  }, []);

  useEffect(() => { if (health?.reachable) loadRegistry(); }, [health?.reachable, loadRegistry]);

  const serverLabel = !health || !health.reachable
    ? t('common.offline')
    : health.status === 'ok' ? t('common.ready') : health.status === 'degraded' ? t('common.degraded') : t('common.error');
  const serverPill = !health || !health.reachable
    ? 'bg-ink-800 text-ink-400 border border-ink-700'
    : health.status === 'ok' ? 'severity-normal' : health.status === 'degraded' ? 'severity-moderate' : 'severity-critical';

  return (
    <div>
      <SectionHeader title={t('settings.aiTitle')} desc={t('settings.aiDesc')} />
      <div className="p-6 space-y-4">
        <div className="p-4 bg-ink-850 rounded-lg border border-ink-800">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-sm font-semibold text-ink-100">{t('settings.inferenceServer')}</div>
              <div className="text-[10px] text-ink-500 font-mono">{settings.inferenceUrl}</div>
            </div>
            <span className={`severity-pill ${serverPill}`}>{serverLabel}</span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-[10px]">
            <div>
              <div className="text-ink-500">{t('settings.serverVersion')}</div>
              <div className="text-ink-200 font-mono">{health?.version || '—'}</div>
            </div>
            <div>
              <div className="text-ink-500">{t('settings.device')}</div>
              <div className="text-ink-200 font-mono">{health?.device || '—'}</div>
            </div>
            <div>
              <div className="text-ink-500">{t('settings.llm')}</div>
              <div className="text-ink-200 font-mono">
                {health?.llm.backend ? `${health.llm.backend} · ${health.llm.reachable ? t('common.reachable') : t('common.unreachable')}` : '—'}
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2">{t('settings.loadedModels')}</div>
          {models.length === 0 ? (
            <div className="p-3 bg-ink-850 rounded-lg border border-ink-800 text-xs text-ink-400">{t('settings.noModelStatus')}</div>
          ) : (
            <div className="space-y-1">
              {models.map(([key, m]) => (
                <div key={key} className="flex items-center justify-between px-3 py-2 bg-ink-850 rounded-lg border border-ink-800 text-xs">
                  <div className="min-w-0">
                    <div className="text-ink-100 font-mono truncate">{key}</div>
                    {m.reason && <div className="text-[10px] text-ink-500 truncate">{m.reason}</div>}
                  </div>
                  <span className={`severity-pill flex-shrink-0 ${m.loaded ? 'severity-normal' : 'severity-critical'}`}>
                    {m.loaded ? t('settings.loaded') : t('settings.notLoaded')}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider">{t('settings.availableModels')}</div>
              <div className="text-[10px] text-ink-500">{t('settings.availableModelsDesc')}</div>
            </div>
            <button onClick={loadRegistry} disabled={loadingRegistry || !health?.reachable} className="btn-ghost !px-2 text-[10px]">
              {loadingRegistry ? t('settings.testing') : t('settings.refresh')}
            </button>
          </div>
          {!health?.reachable ? (
            <div className="p-3 bg-ink-850 rounded-lg border border-ink-800 text-xs text-ink-400">{t('health.aiServerDown')}</div>
          ) : registry === null || registry.length === 0 ? (
            <div className="p-3 bg-ink-850 rounded-lg border border-ink-800 text-xs text-ink-400">{t('settings.noModelStatus')}</div>
          ) : (
            <div className="space-y-1">
              {registry.map((m) => (
                <div key={m.key} className="px-3 py-2 bg-ink-850 rounded-lg border border-ink-800 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-ink-100 truncate">{m.name}</div>
                      <div className="text-[10px] text-ink-500 font-mono truncate">
                        {m.key}{m.tier ? ` · ${t('settings.tier')}: ${m.tier}` : ''}{m.downloadMb != null ? ` · ${t('settings.downloadMb', { mb: m.downloadMb })}` : ''}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      {m.validationStatus && (
                        <span className={`severity-pill text-[9px] ${m.validationStatus === 'validated' ? 'severity-normal' : m.validationStatus === 'pending' ? 'severity-moderate' : 'bg-ink-800 text-ink-400 border border-ink-700'}`}>
                          {statusWord(m.validationStatus, lang)}
                        </span>
                      )}
                    </div>
                  </div>
                  {!m.depsOk && (
                    <div className="text-[10px] text-moderate mt-1">{t('settings.depsMissing', { reason: m.depsReason || '—' })}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="p-3 bg-ink-850 rounded-lg border border-ink-800 text-[11px] text-ink-400 leading-relaxed">
          {t('settings.aiFixedNote')}
        </div>
      </div>
    </div>
  );
}

// ─── License ─────────────────────────────────────────────────────────────────

function LicenseSection() {
  const t = useT();
  return (
    <div>
      <SectionHeader title={t('settings.licenseTitle')} desc={t('settings.licenseDesc')} />
      <div className="p-6">
        <LicensePanel />
      </div>
    </div>
  );
}

// ─── Shortcuts — every binding here exists in useKeyboardShortcuts ───────────

const SHORTCUT_GROUPS: { groupKey: I18nKey; items: [I18nKey, string][] }[] = [
  { groupKey: 'palette.cat.tools', items: [['tool.pan', 'P'], ['tool.zoom', 'Z'], ['tool.wwwl', 'W'], ['tool.length', 'L'], ['tool.angle', 'G'], ['tool.rotate', 'R'], ['tool.invert', 'I']] },
  { groupKey: 'palette.cat.view', items: [['menu.heatmapOverlay', 'H'], ['menu.toggleSidebar', '⌘B'], ['menu.toggleRightPanel', '⌘⇧B'], ['menu.toggleSeriesStrip', 'T'], ['menu.fitToWindow', 'F'], ['menu.fullscreen', 'F11']] },
  { groupKey: 'palette.cat.presets', items: [['preset.chest', '1'], ['preset.lung', '2'], ['preset.bone', '3'], ['preset.soft', '4'], ['preset.brain', '5']] },
  { groupKey: 'palette.cat.file', items: [['menu.uploadFiles', '⌘O'], ['menu.uploadFolder', '⌘⇧O'], ['menu.exportPdf', '⌘P'], ['menu.preferences', '⌘,'], ['menu.commandPalette', '⌘K']] },
  { groupKey: 'palette.cat.help', items: [['menu.documentation', 'F1'], ['menu.keyboardShortcuts', '⌘/']] },
];

function ShortcutsSection() {
  const t = useT();
  return (
    <div>
      <SectionHeader title={t('settings.shortcutsTitle')} desc={t('settings.shortcutsDesc')} />
      <div className="p-6 space-y-4">
        {SHORTCUT_GROUPS.map((g) => (
          <div key={g.groupKey}>
            <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2">{t(g.groupKey)}</div>
            <div className="bg-ink-850 rounded-lg border border-ink-800 overflow-hidden">
              {g.items.map(([labelKey, key], i) => (
                <div key={labelKey} className={`flex items-center justify-between px-3 py-2 ${i > 0 ? 'border-t border-ink-800' : ''}`}>
                  <span className="text-sm text-ink-200">{t(labelKey)}</span>
                  <span className="kbd">{key}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── About ───────────────────────────────────────────────────────────────────

function AboutSection() {
  const t = useT();
  return (
    <div>
      <SectionHeader title={t('settings.aboutTitle', { product: PRODUCT_NAME })} />
      <div className="p-6">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 bg-gradient-to-br from-accent-400 to-accent-700 rounded-2xl flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h3 className="text-xl font-bold text-ink-100">{PRODUCT_NAME}</h3>
            <div className="text-xs text-ink-400">{t('common.version')} {getPackageVersion()}</div>
            <div className="text-[10px] text-ink-500 mt-1">{t('about.copyright', { vendor: VENDOR_NAME, site: VENDOR_SITE })}</div>
          </div>
        </div>
        <div className="p-4 bg-ink-850 rounded-lg border border-ink-800 text-xs text-ink-300 leading-relaxed">
          {t('about.description', { product: PRODUCT_NAME })}
        </div>
        <div className="mt-3 p-3 bg-ink-850 rounded-lg border border-ink-800 text-[11px] text-ink-400 leading-relaxed">
          {t('product.intendedUse')}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 text-[10px]">
          <div className="p-2 bg-ink-850 rounded border border-ink-800">
            <div className="text-ink-500">{t('about.builtWith')}</div>
            <div className="text-ink-200 font-mono">PyTorch · FastAPI · Electron · React</div>
          </div>
          <div className="p-2 bg-ink-850 rounded border border-ink-800">
            <div className="text-ink-500">{t('about.imagingStack')}</div>
            <div className="text-ink-200 font-mono">MONAI · TorchXRayVision</div>
          </div>
        </div>
      </div>
    </div>
  );
}
