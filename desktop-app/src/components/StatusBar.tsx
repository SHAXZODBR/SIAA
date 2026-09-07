import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { getAppVersion } from '../services/appInfo';

export default function StatusBar() {
  const { orthancConnected, health, studies, toggleSidebar, sidebarOpen, toggleRightPanel, rightPanelOpen, activeTool } = useAppStore();
  const [time, setTime] = useState(new Date());
  const [version, setVersion] = useState('');

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    getAppVersion().then(setVersion).catch(() => {});
    return () => clearInterval(timer);
  }, []);

  const pendingCount = studies.filter(s => s.aiStatus === 'pending').length;
  const processingCount = studies.filter(s => s.aiStatus === 'processing').length;
  const completeCount = studies.filter(s => s.aiStatus === 'complete').length;

  const aiState = !health || !health.reachable
    ? { label: 'offline', color: 'text-ink-500', dot: 'bg-ink-600' }
    : health.status === 'ok'
      ? { label: 'ready', color: 'text-normal', dot: 'bg-normal animate-pulse' }
      : health.status === 'degraded'
        ? { label: 'degraded', color: 'text-moderate', dot: 'bg-moderate' }
        : { label: 'error', color: 'text-critical', dot: 'bg-critical' };

  return (
    <div className="h-7 bg-ink-950 border-t border-ink-800 flex items-center px-2 text-[11px] select-none">
      {/* Sidebar toggles */}
      <button
        onClick={toggleSidebar}
        className={`px-2 py-0.5 rounded transition-colors ${sidebarOpen ? 'text-accent-400 bg-accent-500/10' : 'text-ink-500 hover:text-ink-200 hover:bg-ink-800'}`}
        title="Toggle sidebar (⌘B)"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <div className="w-px h-4 bg-ink-800 mx-1" />

      {/* Tool indicator */}
      <div className="flex items-center gap-1 px-2">
        <span className="text-ink-600">Tool:</span>
        <span className="font-mono text-ink-300 capitalize">{activeTool}</span>
      </div>

      <div className="w-px h-4 bg-ink-800 mx-1" />

      {/* PACS status */}
      <div className={`flex items-center gap-1.5 px-2 ${orthancConnected ? 'text-normal' : 'text-ink-500'}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${orthancConnected ? 'bg-normal animate-pulse' : 'bg-ink-600'}`} />
        <span className="text-ink-500">PACS:</span>
        <span className="font-mono">{orthancConnected ? 'online' : 'offline'}</span>
      </div>

      {/* AI server status (from /health) */}
      <div className={`flex items-center gap-1.5 px-2 ${aiState.color}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${aiState.dot}`} />
        <span className="text-ink-500">AI:</span>
        <span className="font-mono">{aiState.label}</span>
      </div>

      <div className="w-px h-4 bg-ink-800 mx-1" />

      {/* Queue */}
      {(pendingCount > 0 || processingCount > 0 || completeCount > 0) && (
        <div className="flex items-center gap-3 px-2">
          {processingCount > 0 && (
            <div className="flex items-center gap-1 text-accent-400">
              <div className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-pulse" />
              <span className="font-mono">{processingCount}</span> analyzing
            </div>
          )}
          {pendingCount > 0 && (
            <div className="flex items-center gap-1 text-ink-400">
              <span className="font-mono">{pendingCount}</span> pending
            </div>
          )}
          <div className="flex items-center gap-1 text-normal">
            <span className="font-mono">{completeCount}</span> analyzed
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Real server facts from /health */}
      <div className="flex items-center gap-3 text-ink-500">
        {health?.device && (
          <div className="flex items-center gap-1" title="Inference device (from /health)">
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span className="font-mono">{health.device}</span>
          </div>
        )}
        {health?.llm && (
          <div className={`flex items-center gap-1 ${health.llm.reachable ? '' : 'text-moderate'}`} title="Report assistant (LLM)">
            <span className="text-ink-600">LLM:</span>
            <span className="font-mono">{health.llm.reachable ? (health.llm.backend || 'ready') : 'unavailable'}</span>
          </div>
        )}
        {(health?.version || version) && (
          <div className="flex items-center gap-1 font-mono" title="Server / app version">
            {health?.version ? `srv ${health.version}` : ''}{health?.version && version ? ' · ' : ''}{version ? `app ${version}` : ''}
          </div>
        )}
      </div>

      <div className="w-px h-4 bg-ink-800 mx-1" />

      {/* Time */}
      <div className="px-2 font-mono text-ink-300">
        {time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}
      </div>

      {/* Right panel toggle */}
      <button
        onClick={toggleRightPanel}
        className={`px-2 py-0.5 rounded transition-colors ${rightPanelOpen ? 'text-accent-400 bg-accent-500/10' : 'text-ink-500 hover:text-ink-200 hover:bg-ink-800'}`}
        title="Toggle right panel (⌘⇧B)"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
      </button>
    </div>
  );
}
