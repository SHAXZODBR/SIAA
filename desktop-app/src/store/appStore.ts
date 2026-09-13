import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type {
  Study, AIResult, Report, User, Finding, AppSettings, HealthStatus, SignedReport, Lang,
} from '../types';

interface AppState {
  // Auth (memory only — never persisted)
  currentUser: User | null;
  authToken: string | null;
  mustChangePassword: boolean;
  setAuth: (token: string, user: User, mustChangePassword: boolean) => void;
  setMustChangePassword: (v: boolean) => void;
  clearAuth: () => void;
  /** @deprecated kept for older call sites; prefer setAuth/clearAuth */
  setCurrentUser: (user: User | null) => void;

  // Studies
  studies: Study[];
  selectedStudyId: string | null;
  setStudies: (studies: Study[]) => void;
  addStudy: (study: Study) => void;
  updateStudy: (id: string, patch: Partial<Study>) => void;
  updateStudyStatus: (id: string, status: Study['aiStatus']) => void;
  selectStudy: (id: string | null) => void;

  // AI Results
  aiResults: Record<string, AIResult>;
  setAIResult: (studyId: string, result: AIResult) => void;

  // Reports
  reports: Record<string, Report>;
  setReport: (studyId: string, report: Report) => void;

  // Server-confirmed signatures, keyed `${studyId}:${language}`
  signedReports: Record<string, SignedReport>;
  setSignedReport: (studyId: string, language: Lang, signed: SignedReport) => void;

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
  settingsSection: string;
  openSettings: (section?: string) => void;
  closeSettings: () => void;
  /** First-run wizard forced open from Settings ("Run setup again"). */
  setupOpen: boolean;
  openSetup: () => void;
  closeSetup: () => void;
  helpOpen: boolean;
  openHelp: () => void;
  closeHelp: () => void;
  /** Cross-component requests (menu / palette → Sidebar upload inputs, → ReportTab export). */
  uploadRequest: { kind: 'folder' | 'files'; nonce: number } | null;
  requestUpload: (kind: 'folder' | 'files') => void;
  exportRequest: number;
  requestExport: () => void;

  // Notifications
  notifications: Notification[];
  addNotification: (n: Omit<Notification, 'id' | 'timestamp'>) => void;
  dismissNotification: (id: string) => void;

  // Connection status
  orthancConnected: boolean;
  inferenceConnected: boolean;
  setConnectionStatus: (orthanc: boolean, inference: boolean) => void;
  health: HealthStatus | null;
  setHealth: (health: HealthStatus | null) => void;

  // Settings (persisted to localStorage)
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

export const DEFAULT_SETTINGS: AppSettings = {
  orthancUrl: 'http://localhost:8042',
  inferenceUrl: 'http://127.0.0.1:8000',
  language: 'ru',
  clinicName: '',
  clinicAddress: '',
  clinicPhone: '',
  supportContact: '',
  theme: 'dark',
  setupComplete: false,
};

export const signedKey = (studyId: string, language: Lang) => `${studyId}:${language}`;

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Auth
      currentUser: null,
      authToken: null,
      mustChangePassword: false,
      setAuth: (token, user, mustChangePassword) =>
        set({ authToken: token, currentUser: user, mustChangePassword }),
      setMustChangePassword: (v) => set({ mustChangePassword: v }),
      // Logging out also drops every patient-identifying object held in memory.
      clearAuth: () =>
        set({
          authToken: null,
          currentUser: null,
          mustChangePassword: false,
          studies: [],
          selectedStudyId: null,
          aiResults: {},
          reports: {},
          signedReports: {},
          selectedFinding: null,
        }),
      setCurrentUser: (user) => set({ currentUser: user }),

      // Studies
      studies: [],
      selectedStudyId: null,
      setStudies: (studies) => set({ studies }),
      addStudy: (study) => set((s) => ({ studies: [study, ...s.studies.filter((st) => st.id !== study.id)] })),
      updateStudy: (id, patch) =>
        set((s) => ({
          studies: s.studies.map((st) => (st.id === id ? { ...st, ...patch } : st)),
        })),
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

      signedReports: {},
      setSignedReport: (studyId, language, signed) =>
        set((s) => ({ signedReports: { ...s.signedReports, [signedKey(studyId, language)]: signed } })),

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
      windowLevel: { center: 128, width: 256 },  // identity for 8-bit preview PNGs
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
      settingsSection: 'general',
      openSettings: (section) => set((s) => ({ settingsOpen: true, settingsSection: section || s.settingsSection })),
      closeSettings: () => set({ settingsOpen: false }),
      setupOpen: false,
      openSetup: () => set({ setupOpen: true, settingsOpen: false, commandPaletteOpen: false }),
      closeSetup: () => set({ setupOpen: false }),
      helpOpen: false,
      openHelp: () => set({ helpOpen: true, commandPaletteOpen: false }),
      closeHelp: () => set({ helpOpen: false }),
      uploadRequest: null,
      requestUpload: (kind) => set({ uploadRequest: { kind, nonce: Date.now() } }),
      exportRequest: 0,
      requestExport: () => set((s) => ({ exportRequest: s.exportRequest + 1, rightPanelOpen: true, rightPanelTab: 'report' })),

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
      health: null,
      setHealth: (health) => set({ health }),

      // Settings
      settings: { ...DEFAULT_SETTINGS },
      updateSettings: (newSettings) =>
        set((s) => ({ settings: { ...s.settings, ...newSettings } })),
    }),
    {
      name: 'sentinel-settings',
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // Only preferences are persisted — never the token, studies or results.
      partialize: (s) => ({ settings: s.settings }),
      merge: (persisted, current) => {
        const p = (persisted as Partial<AppState> | undefined)?.settings || {};
        return { ...current, settings: { ...DEFAULT_SETTINGS, ...current.settings, ...p } };
      },
    },
  ),
);
