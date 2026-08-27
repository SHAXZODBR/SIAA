import { create } from 'zustand';
import type { Study, AIResult, Report, User, Finding, AppSettings } from '../types';

interface AppState {
  // Current user
  currentUser: User | null;
  setCurrentUser: (user: User | null) => void;

  // Studies
  studies: Study[];
  selectedStudyId: string | null;
  setStudies: (studies: Study[]) => void;
  addStudy: (study: Study) => void;
  updateStudyStatus: (id: string, status: Study['aiStatus']) => void;
  selectStudy: (id: string | null) => void;

  // AI Results
  aiResults: Record<string, AIResult>;
  setAIResult: (studyId: string, result: AIResult) => void;

  // Reports
  reports: Record<string, Report>;
  setReport: (studyId: string, report: Report) => void;

  // Viewer state
  showHeatmap: boolean;
  toggleHeatmap: () => void;
  selectedFinding: Finding | null;
  selectFinding: (finding: Finding | null) => void;
  viewerZoom: number;
  setViewerZoom: (zoom: number) => void;
  panOffset: { x: number; y: number };
  setPanOffset: (offset: { x: number; y: number }) => void;
  rotation: number;
  setRotation: (rotation: number) => void;
  invert: boolean;
  toggleInvert: () => void;
  windowLevel: { center: number; width: number };
  setWindowLevel: (wl: { center: number; width: number }) => void;
  activeTool: string;
  setActiveTool: (tool: string) => void;

  // UI state
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  rightPanelOpen: boolean;
  toggleRightPanel: () => void;
  rightPanelTab: 'findings' | 'report' | 'chat' | 'info' | 'compare';
  setRightPanelTab: (tab: 'findings' | 'report' | 'chat' | 'info' | 'compare') => void;
  thumbnailStripOpen: boolean;
  toggleThumbnailStrip: () => void;
  modalityFilter: string;
  setModalityFilter: (modality: string) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;

  // Modals
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
  settingsOpen: boolean;
  openSettings: () => void;
  closeSettings: () => void;

  // Notifications
  notifications: Notification[];
  addNotification: (n: Omit<Notification, 'id' | 'timestamp'>) => void;
  dismissNotification: (id: string) => void;

  // Connection status
  orthancConnected: boolean;
  inferenceConnected: boolean;
  setConnectionStatus: (orthanc: boolean, inference: boolean) => void;

  // Settings
  settings: AppSettings;
  updateSettings: (settings: Partial<AppSettings>) => void;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'ai';
  title: string;
  message?: string;
  timestamp: number;
  read?: boolean;
  action?: { label: string; onClick: () => void };
}

export const useAppStore = create<AppState>((set) => ({
  // User
  currentUser: null,
  setCurrentUser: (user) => set({ currentUser: user }),

  // Studies
  studies: [],
  selectedStudyId: null,
  setStudies: (studies) => set({ studies }),
  addStudy: (study) => set((s) => ({ studies: [study, ...s.studies] })),
  updateStudyStatus: (id, status) =>
    set((s) => ({
      studies: s.studies.map((st) => (st.id === id ? { ...st, aiStatus: status } : st)),
    })),
  selectStudy: (id) => set({ selectedStudyId: id, selectedFinding: null }),

  // AI Results
  aiResults: {},
  setAIResult: (studyId, result) =>
    set((s) => ({ aiResults: { ...s.aiResults, [studyId]: result } })),

  // Reports
  reports: {},
  setReport: (studyId, report) =>
    set((s) => ({ reports: { ...s.reports, [studyId]: report } })),

  // Viewer
  showHeatmap: true,
  toggleHeatmap: () => set((s) => ({ showHeatmap: !s.showHeatmap })),
  selectedFinding: null,
  selectFinding: (finding) => set({ selectedFinding: finding }),
  viewerZoom: 1,
  setViewerZoom: (zoom) => set({ viewerZoom: zoom }),
  panOffset: { x: 0, y: 0 },
  setPanOffset: (offset) => set({ panOffset: offset }),
  rotation: 0,
  setRotation: (rotation) => set({ rotation }),
  invert: false,
  toggleInvert: () => set((s) => ({ invert: !s.invert })),
  windowLevel: { center: 50, width: 350 },  // Chest preset default
  setWindowLevel: (wl) => set({ windowLevel: wl }),
  activeTool: 'pan',
  setActiveTool: (tool) => set({ activeTool: tool }),

  // UI
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  rightPanelOpen: true,
  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  rightPanelTab: 'findings',
  setRightPanelTab: (tab) => set({ rightPanelTab: tab }),
  thumbnailStripOpen: false,
  toggleThumbnailStrip: () => set((s) => ({ thumbnailStripOpen: !s.thumbnailStripOpen })),
  modalityFilter: 'all',
  setModalityFilter: (modality) => set({ modalityFilter: modality }),
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  // Modals
  commandPaletteOpen: false,
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
  settingsOpen: false,
  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),

  // Notifications
  notifications: [],
  addNotification: (n) => set((s) => ({
    notifications: [
      { ...n, id: Math.random().toString(36).slice(2), timestamp: Date.now() },
      ...s.notifications,
    ].slice(0, 50), // Keep last 50
  })),
  dismissNotification: (id) => set((s) => ({
    notifications: s.notifications.filter(n => n.id !== id),
  })),

  // Connection
  orthancConnected: false,
  inferenceConnected: false,
  setConnectionStatus: (orthanc, inference) =>
    set({ orthancConnected: orthanc, inferenceConnected: inference }),

  // Settings
  settings: {
    orthancUrl: 'http://localhost:8042',
    inferenceUrl: 'http://localhost:8000',
    storagePath: './data',
    language: 'ru',
    autoAnalyze: true,
    refreshInterval: 30,
    theme: 'dark',
  },
  updateSettings: (newSettings) =>
    set((s) => ({ settings: { ...s.settings, ...newSettings } })),
}));
