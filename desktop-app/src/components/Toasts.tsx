import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { useT } from '../i18n';

const AUTO_DISMISS_MS = 8000;

/**
 * Doctor-facing toast stack (bottom-right). Technical detail stays in the console;
 * these only ever show localized, actionable text.
 */
export default function Toasts() {
  const { notifications, dismissNotification } = useAppStore();
  const t = useT();
  const visible = notifications.slice(0, 4);

  useEffect(() => {
    if (visible.length === 0) return;
    const timers = visible.map((n) => {
      const remaining = Math.max(500, AUTO_DISMISS_MS - (Date.now() - n.timestamp));
      return setTimeout(() => dismissNotification(n.id), remaining);
    });
    return () => timers.forEach(clearTimeout);
  }, [visible.map((n) => n.id).join('|')]);

  if (visible.length === 0) return null;

  const styles: Record<string, string> = {
    info: 'border-accent-500/40 bg-ink-900',
    ai: 'border-accent-500/40 bg-ink-900',
    success: 'border-normal/40 bg-ink-900',
    warning: 'border-moderate/50 bg-ink-900',
    error: 'border-critical/50 bg-ink-900',
  };
  const dots: Record<string, string> = {
    info: 'bg-accent-400', ai: 'bg-accent-400', success: 'bg-normal', warning: 'bg-moderate', error: 'bg-critical',
  };

  return (
    <div className="fixed bottom-10 right-3 z-[80] w-80 space-y-2 pointer-events-none">
      {visible.map((n) => (
        <div
          key={n.id}
          role="status"
          className={`pointer-events-auto p-3 rounded-lg border shadow-xl animate-slide-up ${styles[n.type] || styles.info}`}
        >
          <div className="flex items-start gap-2">
            <div className={`w-2 h-2 mt-1.5 rounded-full flex-shrink-0 ${dots[n.type] || dots.info}`} />
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-ink-100">{n.title}</div>
              {n.message && <div className="text-[11px] text-ink-400 mt-0.5 break-words">{n.message}</div>}
              {n.action && (
                <button onClick={n.action.onClick} className="mt-1.5 text-[11px] text-accent-400 hover:text-accent-300">
                  {n.action.label}
                </button>
              )}
            </div>
            <button
              onClick={() => dismissNotification(n.id)}
              className="text-ink-500 hover:text-ink-200 text-sm leading-none"
              aria-label={t('common.dismiss')}
              title={t('common.dismiss')}
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
