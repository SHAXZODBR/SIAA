import React from 'react';
import { useAppStore } from '../store/appStore';

interface Tool {
  id: string;
  icon: JSX.Element;
  label: string;
  shortcut?: string;
  group: 'navigation' | 'measurement' | 'annotation' | 'ai';
}

const TOOLS: Tool[] = [
  // Navigation
  {
    id: 'pan', label: 'Pan', shortcut: 'P', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M8 9l4-4 4 4m0 6l-4 4-4-4m-4-4h12M4 12h16" />
    </svg>,
  },
  {
    id: 'zoom', label: 'Zoom', shortcut: 'Z', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
    </svg>,
  },
  {
    id: 'wwwl', label: 'Window/Level', shortcut: 'W', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M20 7l-8 4-8-4m16 0l-8-4-8 4m16 0v10l-8 4m0 0l-8-4V7m8 4v10" />
    </svg>,
  },
  {
    id: 'rotate', label: 'Rotate', shortcut: 'R', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 4v5h5M20 20v-5h-5M5.5 9.5a7 7 0 0111 0M18.5 14.5a7 7 0 01-11 0" />
    </svg>,
  },
  {
    id: 'invert', label: 'Invert', shortcut: 'I', group: 'navigation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
    </svg>,
  },
  // Measurement
  {
    id: 'length', label: 'Length', shortcut: 'L', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 20l5-5m0 0l5-5m-5 5l5 5m-5-5L4 10m10 10l5-5m-5 5v-5m5 5h-5" />
    </svg>,
  },
  {
    id: 'angle', label: 'Angle', shortcut: 'G', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 20h16M4 20V8m0 12l10-12" />
    </svg>,
  },
  {
    id: 'ellipse', label: 'Ellipse ROI', shortcut: 'E', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <ellipse cx="12" cy="12" rx="9" ry="6" strokeWidth={1.5} />
    </svg>,
  },
  {
    id: 'rectangle', label: 'Rectangle ROI', shortcut: 'B', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <rect x="4" y="6" width="16" height="12" strokeWidth={1.5} />
    </svg>,
  },
  {
    id: 'freehand', label: 'Freehand', group: 'measurement',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
    </svg>,
  },
  // Annotation
  {
    id: 'arrow', label: 'Arrow', shortcut: 'A', group: 'annotation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M17 8l4 4m0 0l-4 4m4-4H3" />
    </svg>,
  },
  {
    id: 'text', label: 'Text', shortcut: 'X', group: 'annotation',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 6h16M4 12h16M4 18h12" />
    </svg>,
  },
  // AI
  {
    id: 'heatmap', label: 'AI Heatmap', shortcut: 'H', group: 'ai',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>,
  },
  {
    id: 'segmentation', label: 'Segmentation', shortcut: 'S', group: 'ai',
    icon: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
    </svg>,
  },
];

const WINDOW_PRESETS = [
  { name: 'Chest', center: 50, width: 350 },
  { name: 'Lung', center: -500, width: 1500 },
  { name: 'Bone', center: 400, width: 1800 },
  { name: 'Soft Tissue', center: 50, width: 400 },
  { name: 'Brain', center: 40, width: 80 },
  { name: 'Abdomen', center: 60, width: 400 },
  { name: 'Mediastinum', center: 50, width: 400 },
];

export default function Toolbar() {
  const { activeTool, setActiveTool, showHeatmap, toggleHeatmap, setWindowLevel, windowLevel } = useAppStore();

  const handleToolClick = (toolId: string) => {
    if (toolId === 'heatmap') {
      toggleHeatmap();
    } else if (toolId === 'invert') {
      useAppStore.getState().toggleInvert();
    } else {
      setActiveTool(toolId);
    }
  };

  const groupedTools = {
    navigation: TOOLS.filter(t => t.group === 'navigation'),
    measurement: TOOLS.filter(t => t.group === 'measurement'),
    annotation: TOOLS.filter(t => t.group === 'annotation'),
    ai: TOOLS.filter(t => t.group === 'ai'),
  };

  return (
    <div className="h-12 bg-ink-900 border-b border-ink-800 flex items-center px-3 gap-1 select-none">
      {/* Navigation tools */}
      <ToolGroup tools={groupedTools.navigation} activeTool={activeTool} onSelect={handleToolClick} showHeatmap={showHeatmap} />

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* Measurement tools */}
      <ToolGroup tools={groupedTools.measurement} activeTool={activeTool} onSelect={handleToolClick} showHeatmap={showHeatmap} />

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* Annotation tools */}
      <ToolGroup tools={groupedTools.annotation} activeTool={activeTool} onSelect={handleToolClick} showHeatmap={showHeatmap} />

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* AI tools — highlighted */}
      <div className="flex items-center gap-1 bg-accent-900/20 rounded-md px-1">
        <span className="text-[9px] font-bold text-accent-400 uppercase tracking-wider px-1">AI</span>
        <ToolGroup tools={groupedTools.ai} activeTool={activeTool} onSelect={handleToolClick} showHeatmap={showHeatmap} />
      </div>

      <div className="h-6 w-px bg-ink-800 mx-1" />

      {/* Window/Level Presets */}
      <div className="flex items-center gap-1">
        <span className="text-[9px] font-bold text-ink-500 uppercase tracking-wider px-1">Presets</span>
        <div className="flex gap-0.5">
          {WINDOW_PRESETS.slice(0, 5).map(preset => (
            <button
              key={preset.name}
              onClick={() => setWindowLevel({ center: preset.center, width: preset.width })}
              className="px-2 py-1 text-[10px] font-medium text-ink-300 hover:text-white hover:bg-ink-800 rounded transition-colors"
              title={`Center: ${preset.center}, Width: ${preset.width}`}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1" />

      {/* Viewer info on right */}
      <ViewerInfo />

      {/* Fullscreen button */}
      <button
        onClick={() => {
          if (!document.fullscreenElement) document.documentElement.requestFullscreen();
          else document.exitFullscreen();
        }}
        className="tool-btn"
        title="Fullscreen (F11)"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M4 8V4m0 0h4M4 4l5 5m11-5v4m0-4h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
    </div>
  );
}

function ToolGroup({ tools, activeTool, onSelect, showHeatmap }: any) {
  return (
    <div className="flex items-center gap-0.5">
      {tools.map((tool: Tool) => {
        const isActive = tool.id === 'heatmap' ? showHeatmap : activeTool === tool.id;
        return (
          <button
            key={tool.id}
            onClick={() => onSelect(tool.id)}
            className={`tool-btn group relative ${isActive ? 'active' : ''}`}
            title={`${tool.label}${tool.shortcut ? ` (${tool.shortcut})` : ''}`}
          >
            {tool.icon}
            {/* Tooltip */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 px-2 py-1 bg-ink-950 text-xs text-ink-200 rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50 transition-opacity border border-ink-700">
              {tool.label}
              {tool.shortcut && <span className="ml-2 text-ink-500 font-mono text-[10px]">{tool.shortcut}</span>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ViewerInfo() {
  const { viewerZoom, windowLevel } = useAppStore();
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-ink-500 mr-2">
      <div className="flex items-center gap-1">
        <span className="text-ink-600">Z:</span>
        <span className="text-ink-300">{Math.round(viewerZoom * 100)}%</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-ink-600">W:</span>
        <span className="text-ink-300">{windowLevel.width}</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-ink-600">L:</span>
        <span className="text-ink-300">{windowLevel.center}</span>
      </div>
    </div>
  );
}
