import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useAppStore } from '../store/appStore';
import { useT } from '../i18n';
import { PRODUCT_NAME } from '../services/appInfo';
import { toggleFullscreen } from './TopBar';

interface Command {
  id: string;
  label: string;
  category: string;
  shortcut?: string;
  action: () => void;
  keywords?: string;
}

/**
 * ⌘K palette. Every command here performs a real action; anything the build
 * cannot do (prior-study comparison, printing) is simply not listed.
 */
export default function CommandPalette() {
  const { commandPaletteOpen, setCommandPaletteOpen } = useAppStore();
  const t = useT();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands: Command[] = useMemo(() => {
    const s = () => useAppStore.getState();
    const cat = {
      tools: t('palette.cat.tools'),
      view: t('palette.cat.view'),
      presets: t('palette.cat.presets'),
      file: t('palette.cat.file'),
      panel: t('palette.cat.panel'),
      settings: t('palette.cat.settings'),
      help: t('palette.cat.help'),
    };
    return [
      // Tools
      { id: 'pan', label: t('palette.pan'), category: cat.tools, shortcut: 'P', action: () => s().setActiveTool('pan'), keywords: 'move grab hand pan перемещение siljitish' },
      { id: 'zoom', label: t('palette.zoom'), category: cat.tools, shortcut: 'Z', action: () => s().setActiveTool('zoom'), keywords: 'magnify enlarge zoom масштаб masshtab' },
      { id: 'wwwl', label: t('palette.wwwl'), category: cat.tools, shortcut: 'W', action: () => s().setActiveTool('wwwl'), keywords: 'brightness contrast window level окно oyna' },
      { id: 'length', label: t('palette.length'), category: cat.tools, shortcut: 'L', action: () => s().setActiveTool('length'), keywords: 'ruler distance length длина uzunlik' },
      { id: 'angle', label: t('palette.angle'), category: cat.tools, shortcut: 'G', action: () => s().setActiveTool('angle'), keywords: 'degree angle угол burchak' },
      { id: 'rotate', label: t('palette.rotate'), category: cat.tools, shortcut: 'R', action: () => s().setRotation((s().rotation + 90) % 360), keywords: 'rotate turn поворот burish' },
      { id: 'invert', label: t('palette.invert'), category: cat.tools, shortcut: 'I', action: () => s().toggleInvert(), keywords: 'invert negative инверсия inversiya' },

      // View
      { id: 'heatmap', label: t('palette.heatmap'), category: cat.view, shortcut: 'H', action: () => s().toggleHeatmap(), keywords: 'overlay attention grad-cam heatmap тепловая issiqlik' },
      { id: 'sidebar', label: t('palette.sidebar'), category: cat.view, shortcut: '⌘B', action: () => s().toggleSidebar(), keywords: 'worklist sidebar список ro‘yxat' },
      { id: 'rightpanel', label: t('palette.rightPanel'), category: cat.view, shortcut: '⌘⇧B', action: () => s().toggleRightPanel(), keywords: 'panel findings report панель panel' },
      { id: 'strip', label: t('palette.seriesStrip'), category: cat.view, shortcut: 'T', action: () => s().toggleThumbnailStrip(), keywords: 'series strip серии seriyalar' },
      { id: 'fit', label: t('palette.fit'), category: cat.view, shortcut: 'F', action: () => { s().setViewerZoom(1); s().setPanOffset({ x: 0, y: 0 }); }, keywords: 'fit reset zoom вписать moslashtirish' },
      { id: 'fullscreen', label: t('palette.fullscreen'), category: cat.view, shortcut: 'F11', action: toggleFullscreen, keywords: 'fullscreen полноэкранный to‘liq ekran' },

      // Window Presets
      { id: 'preset-chest', label: t('palette.presetChest'), category: cat.presets, shortcut: '1', action: () => s().setWindowLevel({ center: 50, width: 350 }), keywords: 'window level chest грудная ko‘krak' },
      { id: 'preset-lung', label: t('palette.presetLung'), category: cat.presets, shortcut: '2', action: () => s().setWindowLevel({ center: -500, width: 1500 }), keywords: 'window level lung лёгкие o‘pka' },
      { id: 'preset-bone', label: t('palette.presetBone'), category: cat.presets, shortcut: '3', action: () => s().setWindowLevel({ center: 400, width: 1800 }), keywords: 'window level bone кости suyak' },
      { id: 'preset-soft', label: t('palette.presetSoft'), category: cat.presets, shortcut: '4', action: () => s().setWindowLevel({ center: 50, width: 400 }), keywords: 'window level soft tissue мягкие yumshoq' },
      { id: 'preset-brain', label: t('palette.presetBrain'), category: cat.presets, shortcut: '5', action: () => s().setWindowLevel({ center: 40, width: 80 }), keywords: 'window level brain мозг miya' },

      // File
      { id: 'upload-folder', label: t('palette.uploadFolder'), category: cat.file, shortcut: '⌘⇧O', action: () => s().requestUpload('folder'), keywords: 'open upload folder dicom study загрузить папка yuklash papka' },
      { id: 'upload-files', label: t('palette.uploadFiles'), category: cat.file, shortcut: '⌘O', action: () => s().requestUpload('files'), keywords: 'open upload files dicom загрузить файлы yuklash fayllar' },
      { id: 'export-pdf', label: t('palette.exportPdf'), category: cat.file, shortcut: '⌘P', action: () => s().requestExport(), keywords: 'export pdf report print экспорт eksport' },

      // Right panel
      { id: 'tab-findings', label: t('palette.findingsTab'), category: cat.panel, action: () => { s().setRightPanelTab('findings'); if (!s().rightPanelOpen) s().toggleRightPanel(); }, keywords: 'findings находки topilmalar' },
      { id: 'tab-report', label: t('palette.reportTab'), category: cat.panel, action: () => { s().setRightPanelTab('report'); if (!s().rightPanelOpen) s().toggleRightPanel(); }, keywords: 'report отчёт hisobot' },
      { id: 'tab-chat', label: t('palette.chatTab'), category: cat.panel, action: () => { s().setRightPanelTab('chat'); if (!s().rightPanelOpen) s().toggleRightPanel(); }, keywords: 'ask ai chat спросить so‘rash' },
      { id: 'tab-details', label: t('palette.detailsTab'), category: cat.panel, action: () => { s().setRightPanelTab('info'); if (!s().rightPanelOpen) s().toggleRightPanel(); }, keywords: 'details info детали tafsilotlar' },

      // Settings
      { id: 'settings', label: t('palette.settings'), category: cat.settings, shortcut: '⌘,', action: () => s().openSettings(), keywords: 'configuration options preferences настройки sozlamalar' },
      { id: 'setup', label: t('palette.setup'), category: cat.settings, action: () => s().openSetup(), keywords: 'setup wizard first run мастер usta' },
      { id: 'language-ru', label: t('palette.langRu'), category: cat.settings, action: () => s().updateSettings({ language: 'ru' }), keywords: 'russian русский rus' },
      { id: 'language-uz', label: t('palette.langUz'), category: cat.settings, action: () => s().updateSettings({ language: 'uz' }), keywords: 'uzbek узбекский o‘zbek' },
      { id: 'language-en', label: t('palette.langEn'), category: cat.settings, action: () => s().updateSettings({ language: 'en' }), keywords: 'english английский ingliz' },
      { id: 'logout', label: t('palette.signOut'), category: cat.settings, action: () => s().clearAuth(), keywords: 'sign out logout выйти chiqish' },

      // Help
      { id: 'docs', label: t('palette.docs'), category: cat.help, shortcut: 'F1', action: () => s().openHelp(), keywords: 'help documentation guide справка yordam' },
      { id: 'shortcuts', label: t('menu.keyboardShortcuts'), category: cat.help, shortcut: '⌘/', action: () => s().openSettings('shortcuts'), keywords: 'keyboard shortcuts hotkeys горячие tezkor' },
      { id: 'about', label: t('palette.about', { product: PRODUCT_NAME }), category: cat.help, action: () => s().openSettings('about'), keywords: 'about version о программе haqida' },
    ];
  }, [t]);

  const filtered = useMemo(() => {
    if (!query) return commands;
    const q = query.toLowerCase();
    return commands.filter((c) =>
      c.label.toLowerCase().includes(q) ||
      c.category.toLowerCase().includes(q) ||
      c.keywords?.toLowerCase().includes(q),
    );
  }, [commands, query]);

  const grouped = useMemo(() => {
    const groups: Record<string, Command[]> = {};
    for (const cmd of filtered) {
      if (!groups[cmd.category]) groups[cmd.category] = [];
      groups[cmd.category].push(cmd);
    }
    return groups;
  }, [filtered]);

  // Escape closes (⌘K is handled globally in useKeyboardShortcuts)
  useEffect(() => {
    if (!commandPaletteOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setCommandPaletteOpen(false);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [commandPaletteOpen, setCommandPaletteOpen]);

  useEffect(() => {
    if (commandPaletteOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [commandPaletteOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const executeCommand = (cmd: Command) => {
    setCommandPaletteOpen(false);
    cmd.action();
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter' && filtered[selectedIndex]) {
      e.preventDefault();
      executeCommand(filtered[selectedIndex]);
    }
  };

  if (!commandPaletteOpen) return null;

  let globalIndex = 0;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-ink-950/70 backdrop-blur-sm animate-fade-in"
      onClick={() => setCommandPaletteOpen(false)}
    >
      <div
        className="w-[640px] max-w-[90vw] glass rounded-xl border border-ink-700/50 shadow-2xl overflow-hidden animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-ink-800">
          <svg className="w-5 h-5 text-ink-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder={t('palette.placeholder')}
            className="flex-1 bg-transparent text-sm text-ink-100 placeholder-ink-500 focus:outline-none"
          />
          <span className="text-[10px] font-mono text-ink-500 bg-ink-800 px-2 py-0.5 rounded">ESC</span>
        </div>

        {/* Results */}
        <div className="max-h-[60vh] overflow-y-auto">
          {Object.entries(grouped).length === 0 ? (
            <div className="p-8 text-center text-sm text-ink-500">
              {t('palette.noResults', { query })}
            </div>
          ) : (
            Object.entries(grouped).map(([category, cmds]) => (
              <div key={category}>
                <div className="px-4 py-1.5 text-[10px] font-bold text-ink-500 uppercase tracking-wider bg-ink-950/50">
                  {category}
                </div>
                {cmds.map((cmd) => {
                  const idx = globalIndex++;
                  const isSelected = idx === selectedIndex;
                  return (
                    <button
                      key={cmd.id}
                      onClick={() => executeCommand(cmd)}
                      onMouseEnter={() => setSelectedIndex(idx)}
                      className={`w-full flex items-center justify-between px-4 py-2 text-sm transition-colors ${
                        isSelected
                          ? 'bg-accent-600/20 text-white'
                          : 'text-ink-200 hover:bg-ink-800/50'
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span className={`w-1 h-1 rounded-full ${isSelected ? 'bg-accent-400' : 'bg-transparent'}`} />
                        {cmd.label}
                      </span>
                      {cmd.shortcut && (
                        <span className="text-[10px] font-mono text-ink-500">{cmd.shortcut}</span>
                      )}
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-ink-800 flex items-center justify-between text-[10px] text-ink-500">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="kbd">↑</span><span className="kbd">↓</span> {t('palette.navigate')}</span>
            <span className="flex items-center gap-1"><span className="kbd">↵</span> {t('palette.select')}</span>
            <span className="flex items-center gap-1"><span className="kbd">ESC</span> {t('palette.close')}</span>
          </div>
          <span className="text-accent-400 font-semibold">SENTINEL</span>
        </div>
      </div>
    </div>
  );
}
