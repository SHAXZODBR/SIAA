import React from 'react';
import { useAppStore } from '../store/appStore';
import { useT } from '../i18n';
import type { I18nKey } from '../i18n';
import { toggleFullscreen } from './TopBar';

interface Tool {
  id: string;
  icon: JSX.Element;
  labelKey: I18nKey;
  shortcut?: string;
  group: 'navigation' | 'measurement' | 'ai';
}

/**
 * Only tools that the viewer actually implements are listed here:
 * pan / zoom / window-level / rotate / invert (navigation), length / angle
 * (screen-pixel measurements) and the real AI heatmap toggle.
 */
const TOOLS: Tool[] = [
  // Navigation
  {
    id: 'pan', labelKey: 'tool.pan', shortcut: 'P', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M8 9l4-4 4 4m0 6l-4 4-4-4m-4-4h12M4 12h16" />
    </svg>,
  },
  {
    id: 'zoom', labelKey: 'tool.zoom', shortcut: 'Z', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
    </svg>,
  },
  {
    id: 'wwwl', labelKey: 'tool.wwwl', shortcut: 'W', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M20 7l-8 4-8-4m16 0l-8-4-8 4m16 0v10l-8 4m0 0l-8-4V7m8 4v10" />
    </svg>,
  },
  {
    id: 'rotate', labelKey: 'tool.rotate', shortcut: 'R', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 4v5h5M20 20v-5h-5M5.5 9.5a7 7 0 0111 0M18.5 14.5a7 7 0 01-11 0" />
    </svg>,
  },
  {
    id: 'invert', labelKey: 'tool.invert', shortcut: 'I', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
    </svg>,
  },
  // Measurement
  {
    id: 'length', labelKey: 'tool.length', shortcut: 'L', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 20l5-5m0 0l5-5m-5 5l5 5m-5-5L4 10m10 10l5-5m-5 5v-5m5 5h-5" />
    </svg>,
  },
  {
    id: 'angle', labelKey: 'tool.angle', shortcut: 'G', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 20h16M4 20V8m0 12l10-12" />
    </svg>,
  },
  // AI
  {
    id: 'heatmap', labelKey: 'tool.heatmap', shortcut: 'H', group: 'ai',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>,
  },
];

export const WINDOW_PRESETS: { id: string; labelKey: I18nKey; center: number; width: number }[] = [
  { id: 'chest', labelKey: 'preset.chest', center: 50, width: 350 },
  { id: 'lung', labelKey: 'preset.lung', center: -500, width: 1500 },
  { id: 'bone', labelKey: 'preset.bone', center: 400, width: 1800 },
  { id: 'soft', labelKey: 'preset.soft', center: 50, width: 400 },
  { id: 'brain', labelKey: 'preset.brain', center: 40, width: 80 },
];

export default function Toolbar() {
  const { activeTool, setActiveTool, showHeatmap, toggleHeatmap, setWindowLevel, invert } = useAppStore();
  const t = useT();

  const handleToolClick = (toolId: string) => {
    const s = useAppStore.getState();
    if (toolId === 'heatmap') {
      toggleHeatmap();
    } else if (toolId === 'invert') {
      s.toggleInvert();
    } else if (toolId === 'rotate') {
      s.setRotation((s.rotation + 90) % 360);
    } else {
      setActiveTool(toolId);
    }
  };

  const isActive = (tool: Tool) =>
    tool.id === 'heatmap' ? showHeatmap : tool.id === 'invert' ? invert : tool.id === 'rotate' ? false : activeTool === tool.id;

  const groupedTools = {
    navigation: TOOLS.filter((tl) => tl.group === 'navigation'),
    measurement: TOOLS.filter((tl) => tl.group === 'measurement'),
    ai: TOOLS.filter((tl) => tl.group === 'ai'),
  };

  return (
    <div className="h-12 bg-ink-900 border-b border-ink-800 flex items-center px-3 gap-1 select-none">
      {/* Navigation tools */}
      <ToolGroup tools={groupedTools.navigation} isActive={isActive} onSelect={handleToolClick} />

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* Measurement tools */}
      <ToolGroup tools={groupedTools.measurement} isActive={isActive} onSelect={handleToolClick} />

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* AI tools — highlighted */}
      <div className="flex items-center gap-1 bg-accent-900/20 rounded-md px-1">
        <span className="text-[9px] font-bold text-accent-400 uppercase tracking-wider px-1">AI</span>
        <ToolGroup tools={groupedTools.ai} isActive={isActive} onSelect={handleToolClick} />
      </div>

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* Window/Level Presets */}
      <div className="flex items-center gap-1">
        <span className="text-[9px] font-bold text-ink-500 uppercase tracking-wider px-1">{t('tool.presets')}</span>
        <div className="flex gap-0.5">
          {WINDOW_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setWindowLevel({ center: preset.center, width: preset.width })}
              className="px-2 py-1 text-[10px] font-medium text-ink-300 hover:text-white hover:bg-ink-800 rounded transition-colors"
              title={t('tool.presetTitle', { center: preset.center, width: preset.width })}
            >
              {t(preset.labelKey)}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1" />

      {/* Viewer info on right */}
      <ViewerInfo />

      {/* Fullscreen button */}
      <button
        onClick={toggleFullscreen}
        className="tool-btn"
        title={`${t('tool.fullscreen')} (F11)`}
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M4 8V4m0 0h4M4 4l5 5m11-5v4m0-4h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
    </div>
  );
}

function ToolGroup({ tools, isActive, onSelect }: { tools: Tool[]; isActive: (tool: Tool) => boolean; onSelect: (id: string) => void }) {
  const t = useT();
  return (
    <div className="flex items-center gap-0.5">
      {tools.map((tool) => {
        const active = isActive(tool);
        const label = t(tool.labelKey);
        return (
          <button
            key={tool.id}
            onClick={() => onSelect(tool.id)}
            className={`tool-btn group relative ${active ? 'active' : ''}`}
            title={`${label}${tool.shortcut ? ` (${tool.shortcut})` : ''}`}
          >
            {tool.icon}
            {/* Tooltip */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 px-2 py-1 bg-ink-950 text-xs text-ink-200 rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50 transition-opacity border border-ink-700">
              {label}
              {tool.shortcut && <span className="ml-2 text-ink-500 font-mono text-[10px]">{tool.shortcut}</span>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ViewerInfo() {
  const { viewerZoom, windowLevel, rotation } = useAppStore();
  const t = useT();
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-ink-500 mr-2">
      <div className="flex items-center gap-1" title={t('tool.zoom')}>
        <span className="text-ink-600">{t('tool.zoomShort')}:</span>
        <span className="text-ink-300">{Math.round(viewerZoom * 100)}%</span>
      </div>
      <div className="flex items-center gap-1" title={t('tool.wwwl')}>
        <span className="text-ink-600">{t('tool.widthShort')}:</span>
        <span className="text-ink-300">{windowLevel.width}</span>
      </div>
      <div className="flex items-center gap-1" title={t('tool.wwwl')}>
        <span className="text-ink-600">{t('tool.levelShort')}:</span>
        <span className="text-ink-300">{windowLevel.center}</span>
      </div>
      {rotation !== 0 && (
        <div className="flex items-center gap-1" title={t('tool.rotate')}>
          <span className="text-ink-300">{rotation}°</span>
        </div>
      )}
    </div>
  );
}
