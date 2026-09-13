import React, { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { useT, LANGS, langShort, langName, formatDate } from '../i18n';
import type { I18nKey } from '../i18n';
import { PRODUCT_NAME, getAppVersion } from '../services/appInfo';

interface MenuItem {
  label: string;
  shortcut?: string;
  action?: () => void;
  separator?: boolean;
  checked?: boolean;
}

interface Menu {
  id: string;
  label: string;
  items: MenuItem[];
}

const ROLE_KEYS: Record<string, I18nKey> = {
  admin: 'role.admin',
  radiologist: 'role.radiologist',
  technician: 'role.technician',
};

export function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

/** Reset zoom / pan / rotation / inversion / window to the identity view. */
export function resetImage() {
  const s = useAppStore.getState();
  s.setViewerZoom(1);
  s.setPanOffset({ x: 0, y: 0 });
  s.setRotation(0);
  if (s.invert) s.toggleInvert();
  s.setWindowLevel({ center: 128, width: 256 });
}

/**
 * Menu bar. Every item here is wired to a real action — nothing decorative.
 * Also hosts the compact interface-language switcher (changes re-render the
 * whole UI immediately and set the default report language).
 */
export default function TopBar() {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const { currentUser, clearAuth, settings, updateSettings, sidebarOpen, rightPanelOpen, thumbnailStripOpen, showHeatmap } = useAppStore();
  const t = useT();
  const menuRef = useRef<HTMLDivElement>(null);
  const [version, setVersion] = useState('');

  useEffect(() => {
    getAppVersion().then(setVersion).catch(() => {});
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setActiveMenu(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const store = () => useAppStore.getState();

  const menus: Menu[] = [
    {
      id: 'file',
      label: t('menu.file'),
      items: [
        { label: t('menu.uploadFolder'), shortcut: '⌘⇧O', action: () => store().requestUpload('folder') },
        { label: t('menu.uploadFiles'), shortcut: '⌘O', action: () => store().requestUpload('files') },
        { separator: true, label: '' },
        { label: t('menu.exportPdf'), shortcut: '⌘P', action: () => store().requestExport() },
        { separator: true, label: '' },
        { label: t('menu.preferences'), shortcut: '⌘,', action: () => store().openSettings() },
        { label: t('menu.setupWizard'), action: () => store().openSetup() },
        { separator: true, label: '' },
        { label: t('menu.signOut'), action: () => clearAuth() },
      ],
    },
    {
      id: 'view',
      label: t('menu.view'),
      items: [
        { label: t('menu.fitToWindow'), shortcut: 'F', action: () => { store().setViewerZoom(1); store().setPanOffset({ x: 0, y: 0 }); } },
        { label: t('menu.resetImage'), action: resetImage },
        { label: t('menu.rotate'), shortcut: 'R', action: () => store().setRotation((store().rotation + 90) % 360) },
        { label: t('menu.invert'), shortcut: 'I', checked: store().invert, action: () => store().toggleInvert() },
        { label: t('menu.fullscreen'), shortcut: 'F11', action: toggleFullscreen },
        { separator: true, label: '' },
        { label: t('menu.toggleSidebar'), shortcut: '⌘B', checked: sidebarOpen, action: () => store().toggleSidebar() },
        { label: t('menu.toggleRightPanel'), shortcut: '⌘⇧B', checked: rightPanelOpen, action: () => store().toggleRightPanel() },
        { label: t('menu.toggleSeriesStrip'), shortcut: 'T', checked: thumbnailStripOpen, action: () => store().toggleThumbnailStrip() },
        { separator: true, label: '' },
        { label: t('menu.heatmapOverlay'), shortcut: 'H', checked: showHeatmap, action: () => store().toggleHeatmap() },
      ],
    },
    {
      id: 'tools',
      label: t('menu.tools'),
      items: [
        { label: t('tool.pan'), shortcut: 'P', action: () => store().setActiveTool('pan') },
        { label: t('tool.zoom'), shortcut: 'Z', action: () => store().setActiveTool('zoom') },
        { label: t('tool.wwwl'), shortcut: 'W', action: () => store().setActiveTool('wwwl') },
        { separator: true, label: '' },
        { label: t('tool.length'), shortcut: 'L', action: () => store().setActiveTool('length') },
        { label: t('tool.angle'), shortcut: 'G', action: () => store().setActiveTool('angle') },
        { separator: true, label: '' },
        { label: t('menu.commandPalette'), shortcut: '⌘K', action: () => store().setCommandPaletteOpen(true) },
      ],
    },
    {
      id: 'help',
      label: t('menu.help'),
      items: [
        { label: t('menu.documentation'), shortcut: 'F1', action: () => store().openHelp() },
        { label: t('menu.keyboardShortcuts'), shortcut: '⌘/', action: () => store().openSettings('shortcuts') },
        { separator: true, label: '' },
        { label: t('menu.about', { product: PRODUCT_NAME }), action: () => store().openSettings('about') },
      ],
    },
  ];

  const roleKey = currentUser ? ROLE_KEYS[currentUser.role] : undefined;

  return (
    <div className="h-9 bg-ink-950 border-b border-ink-800 flex items-center px-2 text-sm select-none drag relative z-40">
      {/* Logo / Brand */}
      <div className="flex items-center gap-2 px-2 no-drag">
        <div className="w-5 h-5 bg-gradient-to-br from-accent-400 to-accent-600 rounded flex items-center justify-center">
          <svg className="w-3 h-3 text-white" viewBox="0 0 24 24" fill="none">
            <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <span className="text-xs font-semibold text-ink-200 tracking-wide">SENTINEL</span>
        {version && <span className="text-[10px] text-ink-500 font-mono">v{version}</span>}
      </div>

      {/* Menu Items */}
      <div ref={menuRef} className="flex items-center gap-0 ml-4 no-drag">
        {menus.map((menu) => {
          const isActive = activeMenu === menu.id;
          return (
            <div key={menu.id} className="relative">
              <button
                onClick={() => setActiveMenu(isActive ? null : menu.id)}
                onMouseEnter={() => activeMenu && setActiveMenu(menu.id)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  isActive
                    ? 'bg-ink-800 text-white'
                    : 'text-ink-300 hover:bg-ink-800/50 hover:text-white'
                }`}
              >
                {menu.label}
              </button>

              {isActive && (
                <div className="absolute left-0 top-full mt-0.5 w-72 glass rounded-lg border border-ink-700/50 shadow-xl py-1 animate-scale-in z-50">
                  {menu.items.map((item, idx) => {
                    if (item.separator) {
                      return <div key={idx} className="my-1 h-px bg-ink-700/50 mx-2" />;
                    }
                    return (
                      <button
                        key={idx}
                        onClick={() => {
                          item.action?.();
                          setActiveMenu(null);
                        }}
                        className="w-full flex items-center justify-between px-3 py-1.5 text-xs transition-colors text-ink-200 hover:bg-accent-600/20 hover:text-white"
                      >
                        <span className="flex items-center gap-2">
                          <span className={`w-3 text-[10px] ${item.checked ? 'text-accent-400' : 'text-transparent'}`}>✓</span>
                          {item.label}
                        </span>
                        {item.shortcut && (
                          <span className="text-[10px] text-ink-500 font-mono ml-4">{item.shortcut}</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Patient Context (shows when study selected) */}
      <PatientContext />

      {/* Quick Actions */}
      <div className="flex items-center gap-1 no-drag">
        {/* Compact language switcher — re-renders the whole UI and sets the default report language */}
        <div className="flex items-center gap-0.5 mr-1" role="group" aria-label={t('topbar.language')} title={t('topbar.language')}>
          {LANGS.map((l) => (
            <button
              key={l}
              onClick={() => updateSettings({ language: l })}
              title={langName(l)}
              className={`px-1.5 py-0.5 text-[10px] font-semibold rounded transition-colors ${
                settings.language === l ? 'bg-accent-600 text-white' : 'text-ink-400 hover:text-ink-100 hover:bg-ink-800'
              }`}
            >
              {langShort(l)}
            </button>
          ))}
        </div>

        <button
          onClick={() => useAppStore.getState().setCommandPaletteOpen(true)}
          className="px-2 py-1 flex items-center gap-2 text-xs text-ink-400 hover:text-ink-100 hover:bg-ink-800 rounded transition-colors"
          title={t('menu.commandPalette')}
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span className="text-[10px] font-mono px-1 py-0.5 bg-ink-800 rounded">⌘K</span>
        </button>

        {/* User profile → Sign out */}
        <button
          onClick={() => clearAuth()}
          title={t('menu.signOut')}
          className="flex items-center gap-2 ml-2 px-2 py-1 hover:bg-ink-800/50 rounded cursor-pointer transition-colors text-left"
        >
          <div className="w-6 h-6 bg-gradient-to-br from-accent-500 to-accent-700 rounded-full flex items-center justify-center text-[10px] font-bold text-white">
            {currentUser?.fullName?.split(' ').map((n) => n[0]).join('').substring(0, 2) || currentUser?.username?.slice(0, 2).toUpperCase() || '·'}
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-[11px] font-medium text-ink-100">{currentUser?.fullName || currentUser?.username}</span>
            <span className="text-[9px] text-ink-500 uppercase tracking-wider">{roleKey ? t(roleKey) : currentUser?.role}</span>
          </div>
        </button>
      </div>
    </div>
  );
}

function PatientContext() {
  const { selectedStudyId, studies, settings } = useAppStore();
  const t = useT();
  const study = studies.find((s) => s.id === selectedStudyId);

  if (!study) return null;

  return (
    <div className="flex items-center gap-4 px-4 py-1 bg-ink-900/50 rounded border border-ink-800 mr-2 no-drag">
      <div className="flex flex-col leading-tight">
        <span className="text-[9px] text-ink-500 uppercase tracking-wider">{t('topbar.patient')}</span>
        <span className="text-xs font-medium text-ink-100 font-mono">{study.patientId}</span>
      </div>
      <div className="h-6 w-px bg-ink-700" />
      <div className="flex flex-col leading-tight">
        <span className="text-[9px] text-ink-500 uppercase tracking-wider">{t('topbar.study')}</span>
        <span className="text-xs font-medium text-ink-100">{study.modality} — {study.bodyPart}</span>
      </div>
      {study.studyDate && (
        <>
          <div className="h-6 w-px bg-ink-700" />
          <div className="flex flex-col leading-tight">
            <span className="text-[9px] text-ink-500 uppercase tracking-wider">{t('topbar.date')}</span>
            <span className="text-xs font-medium text-ink-100">{formatDate(study.studyDate, settings.language)}</span>
          </div>
        </>
      )}
    </div>
  );
}
