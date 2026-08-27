import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useAppStore } from '../store/appStore';

interface Command {
  id: string;
  label: string;
  category: string;
  shortcut?: string;
  icon?: JSX.Element;
  action: () => void;
  keywords?: string;
}

export default function CommandPalette() {
  const { commandPaletteOpen, setCommandPaletteOpen, toggleHeatmap, toggleSidebar, setActiveTool, setWindowLevel, openSettings, setCurrentUser } = useAppStore();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands: Command[] = useMemo(() => [
    // Tools
    { id: 'pan', label: 'Switch to Pan Tool', category: 'Tools', shortcut: 'P', action: () => setActiveTool('pan'), keywords: 'move grab hand' },
    { id: 'zoom', label: 'Switch to Zoom Tool', category: 'Tools', shortcut: 'Z', action: () => setActiveTool('zoom'), keywords: 'magnify enlarge' },
    { id: 'wwwl', label: 'Switch to Window/Level', category: 'Tools', shortcut: 'W', action: () => setActiveTool('wwwl'), keywords: 'brightness contrast' },
    { id: 'length', label: 'Measure Length', category: 'Tools', shortcut: 'L', action: () => setActiveTool('length'), keywords: 'ruler distance' },
    { id: 'angle', label: 'Measure Angle', category: 'Tools', shortcut: 'G', action: () => setActiveTool('angle'), keywords: 'degree' },
    { id: 'ellipse', label: 'Ellipse ROI', category: 'Tools', shortcut: 'E', action: () => setActiveTool('ellipse'), keywords: 'circle region of interest' },

    // View
    { id: 'heatmap', label: 'Toggle AI Heatmap', category: 'View', shortcut: 'H', action: toggleHeatmap, keywords: 'overlay attention grad-cam' },
    { id: 'sidebar', label: 'Toggle Sidebar', category: 'View', shortcut: '⌘B', action: toggleSidebar, keywords: 'worklist' },
    { id: 'fullscreen', label: 'Toggle Fullscreen', category: 'View', shortcut: 'F11', action: () => { if (!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); } },

    // Window Presets
    { id: 'preset-chest', label: 'Apply Chest Preset', category: 'Presets', action: () => setWindowLevel({ center: 50, width: 350 }), keywords: 'window level chest' },
    { id: 'preset-lung', label: 'Apply Lung Preset', category: 'Presets', action: () => setWindowLevel({ center: -500, width: 1500 }), keywords: 'window level lung' },
    { id: 'preset-bone', label: 'Apply Bone Preset', category: 'Presets', action: () => setWindowLevel({ center: 400, width: 1800 }), keywords: 'window level bone' },
    { id: 'preset-soft', label: 'Apply Soft Tissue Preset', category: 'Presets', action: () => setWindowLevel({ center: 50, width: 400 }), keywords: 'window level soft' },
    { id: 'preset-brain', label: 'Apply Brain Preset', category: 'Presets', action: () => setWindowLevel({ center: 40, width: 80 }), keywords: 'window level brain' },

    // AI
    { id: 'ai-run', label: 'Re-run AI Analysis', category: 'AI', shortcut: '⌘↩', action: () => {}, keywords: 'analyze predict' },
    { id: 'ai-compare', label: 'Compare with Prior Study', category: 'AI', action: () => {}, keywords: 'diff change' },

    // File
    { id: 'open', label: 'Open DICOM File...', category: 'File', shortcut: '⌘O', action: async () => {
      // @ts-ignore
      const files = await window.electronAPI?.openDicomFile();
      console.log(files);
    }},
    { id: 'export-pdf', label: 'Export Report as PDF', category: 'File', shortcut: '⌘P', action: () => {} },
    { id: 'print', label: 'Print Report', category: 'File', action: () => window.print() },

    // Navigation
    { id: 'next-study', label: 'Next Study', category: 'Navigation', shortcut: '→', action: () => {} },
    { id: 'prev-study', label: 'Previous Study', category: 'Navigation', shortcut: '←', action: () => {} },

    // Settings
    { id: 'settings', label: 'Open Preferences', category: 'Settings', shortcut: '⌘,', action: openSettings, keywords: 'configuration options' },
    { id: 'language-ru', label: 'Switch to Russian', category: 'Settings', action: () => useAppStore.getState().updateSettings({ language: 'ru' }) },
    { id: 'language-uz', label: 'Switch to Uzbek', category: 'Settings', action: () => useAppStore.getState().updateSettings({ language: 'uz' }) },
    { id: 'language-en', label: 'Switch to English', category: 'Settings', action: () => useAppStore.getState().updateSettings({ language: 'en' }) },
    { id: 'logout', label: 'Sign Out', category: 'Settings', action: () => setCurrentUser(null) },

    // Help
    { id: 'docs', label: 'Open Documentation', category: 'Help', shortcut: 'F1', action: () => {} },
    { id: 'about', label: 'About Sentinel Medical AI', category: 'Help', action: () => {} },
  ], []);

  const filtered = useMemo(() => {
    if (!query) return commands;
    const q = query.toLowerCase();
    return commands.filter(c =>
      c.label.toLowerCase().includes(q) ||
      c.category.toLowerCase().includes(q) ||
      c.keywords?.toLowerCase().includes(q)
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

  // Global Cmd+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(!commandPaletteOpen);
      }
      if (e.key === 'Escape' && commandPaletteOpen) {
        setCommandPaletteOpen(false);
      }
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
    cmd.action();
    setCommandPaletteOpen(false);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(i => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(i => Math.max(0, i - 1));
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
        onClick={e => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-ink-800">
          <svg className="w-5 h-5 text-ink-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Search commands, tools, presets..."
            className="flex-1 bg-transparent text-sm text-ink-100 placeholder-ink-500 focus:outline-none"
          />
          <span className="text-[10px] font-mono text-ink-500 bg-ink-800 px-2 py-0.5 rounded">ESC</span>
        </div>

        {/* Results */}
        <div className="max-h-[60vh] overflow-y-auto">
          {Object.entries(grouped).length === 0 ? (
            <div className="p-8 text-center text-sm text-ink-500">
              No commands found for "{query}"
            </div>
          ) : (
            Object.entries(grouped).map(([category, cmds]) => (
              <div key={category}>
                <div className="px-4 py-1.5 text-[10px] font-bold text-ink-500 uppercase tracking-wider bg-ink-950/50">
                  {category}
                </div>
                {cmds.map(cmd => {
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
            <span className="flex items-center gap-1"><span className="kbd">↑</span><span className="kbd">↓</span> navigate</span>
            <span className="flex items-center gap-1"><span className="kbd">↵</span> select</span>
            <span className="flex items-center gap-1"><span className="kbd">ESC</span> close</span>
          </div>
          <span className="text-accent-400 font-semibold">SENTINEL</span>
        </div>
      </div>
    </div>
  );
}
