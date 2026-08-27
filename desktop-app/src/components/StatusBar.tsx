import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';

export default function StatusBar() {
  const { orthancConnected, inferenceConnected, studies, toggleSidebar, sidebarOpen, toggleRightPanel, rightPanelOpen, activeTool } = useAppStore();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pendingCount = studies.filter(s => s.aiStatus === 'pending').length;
  const processingCount = studies.filter(s => s.aiStatus === 'processing').length;
  const completeCount = studies.filter(s => s.aiStatus === 'complete').length;

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

      {/* AI status */}
      <div className={`flex items-center gap-1.5 px-2 ${inferenceConnected ? 'text-normal' : 'text-ink-500'}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${inferenceConnected ? 'bg-normal animate-pulse' : 'bg-ink-600'}`} />
        <span className="text-ink-500">AI:</span>
        <span className="font-mono">{inferenceConnected ? 'ready' : 'offline'}</span>
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
            <span className="font-mono">{completeCount}</span> done today
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* System metrics */}
      <div className="flex items-center gap-3 text-ink-500">
        <div className="flex items-center gap-1" title="Storage used">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
          </svg>
          <span className="font-mono">12.4 GB / 500 GB</span>
        </div>
        <div className="flex items-center gap-1" title="GPU usage">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <span className="font-mono">GTX 1650 · 32%</span>
        </div>
        <div className="flex items-center gap-1" title="Active users">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          <span className="font-mono">1</span>
        </div>
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
