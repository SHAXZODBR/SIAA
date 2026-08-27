import React, { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/appStore';

interface MenuItem {
  label: string;
  shortcut?: string;
  action?: () => void;
  separator?: boolean;
  disabled?: boolean;
  icon?: string;
}

interface Menu {
  label: string;
  items: MenuItem[];
}

export default function TopBar() {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const { currentUser, setCurrentUser, settings } = useAppStore();
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setActiveMenu(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const menus: Menu[] = [
    {
      label: 'File',
      items: [
        { label: 'Open DICOM...', shortcut: '⌘O', action: () => triggerOpenFile() },
        { label: 'Open Folder...', shortcut: '⌘⇧O', action: () => triggerOpenFolder() },
        { separator: true, label: '' },
        { label: 'Import from PACS', shortcut: '⌘I' },
        { label: 'Export Study...', shortcut: '⌘E' },
        { separator: true, label: '' },
        { label: 'Export Report as PDF', shortcut: '⌘P' },
        { label: 'Print...', shortcut: '⌘⇧P' },
        { separator: true, label: '' },
        { label: 'Preferences...', shortcut: '⌘,', action: () => useAppStore.getState().openSettings() },
        { separator: true, label: '' },
        { label: 'Sign Out', action: () => setCurrentUser(null) },
      ],
    },
    {
      label: 'Edit',
      items: [
        { label: 'Undo', shortcut: '⌘Z' },
        { label: 'Redo', shortcut: '⌘⇧Z' },
        { separator: true, label: '' },
        { label: 'Copy Measurements', shortcut: '⌘C' },
        { label: 'Paste Annotations', shortcut: '⌘V' },
        { separator: true, label: '' },
        { label: 'Clear All Annotations' },
        { label: 'Reset Image', shortcut: '⌘R' },
      ],
    },
    {
      label: 'View',
      items: [
        { label: 'Fit to Window', shortcut: 'F', action: () => useAppStore.getState().setViewerZoom(1) },
        { label: 'Actual Size (1:1)', shortcut: '1' },
        { label: 'Fullscreen', shortcut: 'F11', action: () => toggleFullscreen() },
        { separator: true, label: '' },
        { label: 'Toggle Sidebar', shortcut: '⌘B', action: () => useAppStore.getState().toggleSidebar() },
        { label: 'Toggle Right Panel', shortcut: '⌘⇧B' },
        { label: 'Toggle Thumbnails', shortcut: 'T' },
        { separator: true, label: '' },
        { label: 'Heatmap Overlay', shortcut: 'H', action: () => useAppStore.getState().toggleHeatmap() },
        { label: 'Show Annotations', shortcut: 'A' },
        { label: 'Show Measurements', shortcut: 'M' },
        { separator: true, label: '' },
        { label: 'Split View (2)', shortcut: '⌘2' },
        { label: 'Split View (4)', shortcut: '⌘4' },
      ],
    },
    {
      label: 'Tools',
      items: [
        { label: 'Pan', shortcut: 'P' },
        { label: 'Zoom', shortcut: 'Z' },
        { label: 'Window/Level', shortcut: 'W' },
        { separator: true, label: '' },
        { label: 'Length Measurement', shortcut: 'L' },
        { label: 'Angle Measurement', shortcut: 'G' },
        { label: 'Elliptical ROI', shortcut: 'E' },
        { label: 'Rectangle ROI', shortcut: 'R' },
        { label: 'Freehand ROI', shortcut: 'R' },
        { separator: true, label: '' },
        { label: 'Text Annotation', shortcut: 'X' },
        { label: 'Arrow Annotation' },
        { separator: true, label: '' },
        { label: 'Re-run AI Analysis', shortcut: '⌘⇧A' },
        { label: 'DICOM Anonymize...' },
      ],
    },
    {
      label: 'AI',
      items: [
        { label: 'Run Analysis', shortcut: '⌘Return' },
        { label: 'Compare with Prior Study' },
        { separator: true, label: '' },
        { label: 'Show Confidence Scores', shortcut: 'C' },
        { label: 'Show Grad-CAM Heatmap', shortcut: 'H' },
        { label: 'Show Segmentation Masks', shortcut: 'S' },
        { separator: true, label: '' },
        { label: 'Export AI Report' },
        { label: 'Submit Feedback' },
      ],
    },
    {
      label: 'Admin',
      items: [
        { label: 'User Management', disabled: currentUser?.role !== 'admin' },
        { label: 'Audit Log', disabled: currentUser?.role !== 'admin' },
        { separator: true, label: '' },
        { label: 'Clinic Settings', disabled: currentUser?.role !== 'admin' },
        { label: 'Model Configuration', disabled: currentUser?.role !== 'admin' },
        { separator: true, label: '' },
        { label: 'Database Backup', disabled: currentUser?.role !== 'admin' },
        { label: 'System Diagnostics' },
      ],
    },
    {
      label: 'Help',
      items: [
        { label: 'Documentation', shortcut: 'F1' },
        { label: 'Keyboard Shortcuts', shortcut: '⌘/', action: () => useAppStore.getState().setCommandPaletteOpen(true) },
        { separator: true, label: '' },
        { label: 'Report a Problem' },
        { label: 'Contact Support' },
        { separator: true, label: '' },
        { label: 'About Sentinel Medical AI' },
      ],
    },
  ];

  const triggerOpenFile = async () => {
    // @ts-ignore
    const files = await window.electronAPI?.openDicomFile();
    if (files) console.log('Selected:', files);
  };

  const triggerOpenFolder = async () => {
    // @ts-ignore
    const folder = await window.electronAPI?.openDicomFolder();
    if (folder) console.log('Selected:', folder);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

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
        <span className="text-[10px] text-ink-500 font-mono">v1.0.0</span>
      </div>

      {/* Menu Items */}
      <div ref={menuRef} className="flex items-center gap-0 ml-4 no-drag">
        {menus.map((menu) => {
          const isActive = activeMenu === menu.label;
          return (
            <div key={menu.label} className="relative">
              <button
                onClick={() => setActiveMenu(isActive ? null : menu.label)}
                onMouseEnter={() => activeMenu && setActiveMenu(menu.label)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  isActive
                    ? 'bg-ink-800 text-white'
                    : 'text-ink-300 hover:bg-ink-800/50 hover:text-white'
                }`}
              >
                {menu.label}
              </button>

              {isActive && (
                <div className="absolute left-0 top-full mt-0.5 w-64 glass rounded-lg border border-ink-700/50 shadow-xl py-1 animate-scale-in z-50">
                  {menu.items.map((item, idx) => {
                    if (item.separator) {
                      return <div key={idx} className="my-1 h-px bg-ink-700/50 mx-2" />;
                    }
                    return (
                      <button
                        key={idx}
                        disabled={item.disabled}
                        onClick={() => {
                          item.action?.();
                          setActiveMenu(null);
                        }}
                        className={`w-full flex items-center justify-between px-3 py-1.5 text-xs transition-colors ${
                          item.disabled
                            ? 'text-ink-600 cursor-not-allowed'
                            : 'text-ink-200 hover:bg-accent-600/20 hover:text-white'
                        }`}
                      >
                        <span>{item.label}</span>
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
        <button
          onClick={() => useAppStore.getState().setCommandPaletteOpen(true)}
          className="px-2 py-1 flex items-center gap-2 text-xs text-ink-400 hover:text-ink-100 hover:bg-ink-800 rounded transition-colors"
          title="Command Palette"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span className="text-[10px] font-mono px-1 py-0.5 bg-ink-800 rounded">⌘K</span>
        </button>

        <button className="tool-btn !w-7 !h-7" title="Notifications">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
          <span className="absolute top-1 right-1 w-1.5 h-1.5 bg-accent-500 rounded-full" />
        </button>

        {/* User profile */}
        <div className="flex items-center gap-2 ml-2 px-2 py-1 hover:bg-ink-800/50 rounded cursor-pointer transition-colors">
          <div className="w-6 h-6 bg-gradient-to-br from-accent-500 to-accent-700 rounded-full flex items-center justify-center text-[10px] font-bold text-white">
            {currentUser?.fullName?.split(' ').map(n => n[0]).join('').substring(0, 2) || 'DR'}
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-[11px] font-medium text-ink-100">{currentUser?.fullName}</span>
            <span className="text-[9px] text-ink-500 uppercase tracking-wider">{currentUser?.role}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientContext() {
  const { selectedStudyId, studies } = useAppStore();
  const study = studies.find(s => s.id === selectedStudyId);

  if (!study) return null;

  return (
    <div className="flex items-center gap-4 px-4 py-1 bg-ink-900/50 rounded border border-ink-800 mr-2 no-drag">
      <div className="flex flex-col leading-tight">
        <span className="text-[9px] text-ink-500 uppercase tracking-wider">Patient</span>
        <span className="text-xs font-medium text-ink-100 font-mono">{study.patientId}</span>
      </div>
      <div className="h-6 w-px bg-ink-700" />
      <div className="flex flex-col leading-tight">
        <span className="text-[9px] text-ink-500 uppercase tracking-wider">Study</span>
        <span className="text-xs font-medium text-ink-100">{study.modality} — {study.bodyPart}</span>
      </div>
      <div className="h-6 w-px bg-ink-700" />
      <div className="flex flex-col leading-tight">
        <span className="text-[9px] text-ink-500 uppercase tracking-wider">Date</span>
        <span className="text-xs font-medium text-ink-100">{new Date(study.studyDate).toLocaleDateString()}</span>
      </div>
    </div>
  );
}
