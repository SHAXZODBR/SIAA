import React, { useEffect, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useT } from '../i18n';
import type { I18nKey } from '../i18n';
import { PRODUCT_NAME } from '../services/appInfo';

/** Quick-guide sections, in workflow order: sign in → upload → findings → report/sign → export → problems. */
const SECTIONS: { title: I18nKey; body: I18nKey }[] = [
  { title: 'help.sec1Title', body: 'help.sec1Body' },
  { title: 'help.sec2Title', body: 'help.sec2Body' },
  { title: 'help.sec3Title', body: 'help.sec3Body' },
  { title: 'help.sec4Title', body: 'help.sec4Body' },
  { title: 'help.sec5Title', body: 'help.sec5Body' },
  { title: 'help.sec6Title', body: 'help.sec6Body' },
];

/**
 * In-app quick guide (Help → Documentation / F1). Always available in the
 * interface language; the full manuals (docs/user_manual_{ru,en}.md) open
 * through the Electron bridge when a docs/ folder is bundled.
 */
export default function HelpModal() {
  const { helpOpen, closeHelp, openSettings, settings } = useAppStore();
  const t = useT();
  const [docsMissing, setDocsMissing] = useState(false);
  const [opening, setOpening] = useState(false);
  const bridge = typeof window !== 'undefined' ? window.electronAPI : undefined;
  const canOpenDocs = !!bridge?.openDocs;

  useEffect(() => {
    if (!helpOpen) return;
    setDocsMissing(false);
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') closeHelp(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [helpOpen, closeHelp]);

  if (!helpOpen) return null;

  const openDocs = async () => {
    if (!bridge?.openDocs || opening) return;
    setOpening(true);
    try {
      const res = await bridge.openDocs();
      if (!res?.opened) setDocsMissing(true);
    } catch (e) {
      console.warn('open-docs failed:', e);
      setDocsMissing(true);
    } finally {
      setOpening(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[95] flex items-center justify-center bg-ink-950/70 backdrop-blur-sm animate-fade-in"
      onClick={closeHelp}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('help.title')}
        className="w-[760px] max-w-[95vw] max-h-[88vh] glass rounded-xl border border-ink-700/50 shadow-2xl overflow-hidden flex flex-col animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-ink-800 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-ink-100">{t('help.title')}</h2>
            <p className="text-xs text-ink-400 mt-0.5">{t('help.subtitle', { product: PRODUCT_NAME })}</p>
          </div>
          <button onClick={closeHelp} className="text-ink-500 hover:text-ink-200 text-xl leading-none" aria-label={t('common.close')} title={t('common.close')}>
            ×
          </button>
        </div>

        {/* Sections */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {SECTIONS.map((s, i) => (
            <section key={s.title} className="flex gap-3">
              <div className="flex-shrink-0 w-7 h-7 rounded-md bg-accent-600/20 text-accent-300 text-xs font-bold flex items-center justify-center font-mono">
                {i + 1}
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-semibold text-ink-100 mb-1">{t(s.title)}</h3>
                <p className="text-xs text-ink-300 leading-relaxed">{t(s.body)}</p>
              </div>
            </section>
          ))}

          <div className="p-3 bg-ink-850 rounded-lg border border-ink-800 text-[11px] text-ink-400 leading-relaxed">
            {t('product.intendedUse')}
            {settings.supportContact && (
              <span className="block mt-1 text-ink-300">{t('common.support')}: {settings.supportContact}</span>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-ink-800 flex items-center justify-between gap-3 bg-ink-950/30">
          <div className="flex items-center gap-2 min-w-0">
            {canOpenDocs && (
              <button onClick={openDocs} disabled={opening} className="btn-secondary">
                {t('help.openFolder')}
              </button>
            )}
            <button onClick={() => { closeHelp(); openSettings('shortcuts'); }} className="btn-ghost">
              {t('menu.keyboardShortcuts')}
            </button>
            {docsMissing && <span className="text-[11px] text-moderate truncate">{t('help.docsMissing')}</span>}
          </div>
          <button onClick={closeHelp} className="btn-primary">{t('common.close')}</button>
        </div>
      </div>
    </div>
  );
}
