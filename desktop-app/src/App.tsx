import React, { useEffect, useRef } from 'react';
import { useAppStore } from './store/appStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { getHealth, checkOrthancHealth, listStudies, getStudy } from './services/api';
import { L } from './services/findingTranslations';

import TopBar from './components/TopBar';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import DicomViewer from './components/DicomViewer';
import RightPanel from './components/RightPanel';
import StatusBar from './components/StatusBar';
import ThumbnailStrip from './components/ThumbnailStrip';
import LoginScreen from './components/LoginScreen';
import CommandPalette from './components/CommandPalette';
import SettingsModal from './components/SettingsModal';
import Toasts from './components/Toasts';

const HEALTH_POLL_MS = 15000;

export default function App() {
  const {
    currentUser, mustChangePassword, setConnectionStatus, setHealth, health,
    sidebarOpen, rightPanelOpen, addNotification, settings,
    studies, setStudies, selectedStudyId, aiResults, setAIResult, setSignedReport, setReport, updateStudy,
  } = useAppStore();
  const lang = settings.language;
  const loggedIn = !!currentUser && !mustChangePassword;

  // Setup keyboard shortcuts
  useKeyboardShortcuts();

  // Poll /health every 15s (public route — also drives the login screen status line)
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const [h, orthanc] = await Promise.all([getHealth(), checkOrthancHealth()]);
      if (cancelled) return;
      setHealth(h);
      setConnectionStatus(orthanc, h.reachable && h.status === 'ok');
    };
    poll();
    const interval = setInterval(poll, HEALTH_POLL_MS);
    return () => { cancelled = true; clearInterval(interval); };
  }, [settings.inferenceUrl, setHealth, setConnectionStatus]);

  // Welcome notification + worklist restore (GET /studies) after login
  const restoredForUser = useRef<string | null>(null);
  useEffect(() => {
    if (!loggedIn || !currentUser) return;
    addNotification({
      type: 'info',
      title: `${currentUser.fullName || currentUser.username}`,
      message: `${currentUser.role}`,
    });
    if (restoredForUser.current === currentUser.id) return;
    restoredForUser.current = currentUser.id;
    listStudies(100)
      .then((restored) => {
        if (restored.length === 0) return;
        const current = useAppStore.getState().studies;
        const known = new Set(current.map((s) => s.id));
        const merged = [...current, ...restored.filter((s) => !known.has(s.id))];
        setStudies(merged);
        addNotification({ type: 'success', title: L('studiesRestored', lang), message: `${restored.length}` });
      })
      .catch((e) => {
        console.warn('Worklist restore failed:', e?.message || e);
      });
  }, [loggedIn, currentUser?.id]);

  // Lazily fetch the persisted AI result for a restored study (GET /study/{id})
  const fetchedDetail = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!loggedIn || !selectedStudyId) return;
    if (aiResults[selectedStudyId]) return;
    const study = studies.find((s) => s.id === selectedStudyId);
    if (!study || !study.restored || fetchedDetail.current.has(selectedStudyId)) return;
    fetchedDetail.current.add(selectedStudyId);
    getStudy(selectedStudyId)
      .then((detail) => {
        updateStudy(selectedStudyId, { ...detail.study, id: selectedStudyId, restored: true });
        if (detail.result) setAIResult(selectedStudyId, detail.result);
        for (const s of detail.signed) setSignedReport(selectedStudyId, s.language, s);
        const draft = detail.drafts[0];
        if (draft && (draft.reportText || draft.aiDraftText)) {
          const now = new Date().toISOString();
          setReport(selectedStudyId, {
            id: `rep-${selectedStudyId}`,
            studyId: selectedStudyId,
            doctorId: '',
            reportText: draft.reportText || draft.aiDraftText,
            aiDraftText: draft.aiDraftText || draft.reportText,
            language: draft.language,
            isSigned: detail.signed.length > 0,
            createdAt: now,
            updatedAt: now,
          });
        }
      })
      .catch((e) => {
        fetchedDetail.current.delete(selectedStudyId);
        console.warn('Study detail fetch failed:', e?.message || e);
        addNotification({ type: 'warning', title: L('aiServerError', lang) });
      });
  }, [loggedIn, selectedStudyId, studies, aiResults]);

  // Login gate (also blocks until a temporary password has been replaced)
  if (!loggedIn) {
    return (
      <>
        <LoginScreen />
        <Toasts />
      </>
    );
  }

  const healthBanner = !health || !health.reachable
    ? L('aiServerDown', lang)
    : health.status === 'degraded'
      ? L('aiServerDegraded', lang)
      : health.status === 'error'
        ? L('aiServerError', lang)
        : null;

  return (
    <div className="h-screen w-screen flex flex-col bg-ink-950 text-ink-200 overflow-hidden">
      {/* Server health banner — full width, above everything */}
      {healthBanner && (
        <div className="w-full px-4 py-1.5 bg-critical/20 border-b border-critical/40 text-xs text-critical flex items-center gap-2 select-none">
          <span className="w-2 h-2 rounded-full bg-critical animate-pulse flex-shrink-0" />
          <span className="font-semibold">{healthBanner}</span>
          {settings.supportContact && <span className="text-ink-300 ml-2">{L('support', lang)}: {settings.supportContact}</span>}
        </div>
      )}

      {/* Top menu bar */}
      <TopBar />

      {/* Tool bar */}
      <Toolbar />

      {/* Main 3-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Study worklist */}
        {sidebarOpen && (
          <div className="w-72 flex-shrink-0 animate-fade-in">
            <Sidebar />
          </div>
        )}

        {/* Center: DICOM Viewer */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 relative">
            <DicomViewer />
          </div>

          {/* Bottom series strip (real series only) */}
          <ThumbnailStrip />
        </div>

        {/* Right: Findings + Report + Info */}
        {rightPanelOpen && (
          <div className="w-96 flex-shrink-0 animate-fade-in">
            <RightPanel />
          </div>
        )}
      </div>

      {/* Bottom status bar */}
      <StatusBar />

      {/* Overlays */}
      <CommandPalette />
      <SettingsModal />
      <Toasts />
    </div>
  );
}
