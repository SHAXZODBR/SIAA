import React, { useEffect, useRef } from 'react';
import { useAppStore } from './store/appStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { getHealth, checkOrthancHealth, listStudies, getStudy } from './services/api';
import { useT } from './i18n';
import type { I18nKey } from './i18n';

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
import SetupWizard from './components/SetupWizard';
import HelpModal from './components/HelpModal';
import Toasts from './components/Toasts';

const HEALTH_POLL_MS = 15000;

const ROLE_KEYS: Record<string, I18nKey> = {
  admin: 'role.admin',
  radiologist: 'role.radiologist',
  technician: 'role.technician',
};

export default function App() {
  const {
    currentUser, mustChangePassword, setConnectionStatus, setHealth, health,
    sidebarOpen, rightPanelOpen, addNotification, settings, setupOpen,
    studies, setStudies, selectedStudyId, aiResults, setAIResult, setSignedReport, setReport, updateStudy,
  } = useAppStore();
  const t = useT();
  const loggedIn = !!currentUser && !mustChangePassword;
  // First-run wizard: shown after login until the clinic is configured, or when re-run from Settings.
  const needsSetup = setupOpen || !settings.setupComplete || !settings.clinicName.trim();

  // Setup keyboard shortcuts
  useKeyboardShortcuts();

  // Native dialogs (file pickers, backend crash boxes) live in the Electron main
  // process — keep them in the same language as the interface.
  useEffect(() => {
    try { window.electronAPI?.setLanguage?.(settings.language); } catch { /* browser build */ }
  }, [settings.language]);

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
    const roleKey = ROLE_KEYS[currentUser.role];
    addNotification({
      type: 'info',
      title: `${currentUser.fullName || currentUser.username}`,
      message: roleKey ? t(roleKey) : currentUser.role,
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
        addNotification({ type: 'success', title: t('health.studiesRestored'), message: `${restored.length}` });
      })
      .catch((e) => {
        console.warn('Worklist restore failed:', e?.message || e);
      });
  }, [loggedIn, currentUser?.id]);

  // Lazily fetch the persisted AI result for a restored study (GET /study/{id})
  const fetchedDetail = useRef<Set<string>>(new Set());
  const fetchedPreview = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!loggedIn || !selectedStudyId) return;
    const study = studies.find((s) => s.id === selectedStudyId);
    const existing = aiResults[selectedStudyId];
    if (existing) {
      // Result already in memory but without its slice preview: when the server
      // says it stored one (has_preview), fetch GET /study/{id} once and merge it.
      if (existing.previewBase64 || !study?.hasPreview || fetchedPreview.current.has(selectedStudyId)) return;
      const id = selectedStudyId;
      fetchedPreview.current.add(id);
      getStudy(id)
        .then((detail) => {
          const preview = detail.result?.previewBase64;
          if (!preview) {
            updateStudy(id, { hasPreview: false }); // truly absent — viewer shows "image unavailable"
            return;
          }
          const current = useAppStore.getState().aiResults[id];
          if (current && !current.previewBase64) setAIResult(id, { ...current, previewBase64: preview });
        })
        .catch((e) => {
          fetchedPreview.current.delete(id);
          updateStudy(id, { hasPreview: false });
          console.warn('Study preview fetch failed:', e?.message || e);
          addNotification({ type: 'warning', title: t('health.aiServerError') });
        });
      return;
    }
    if (!study || !study.restored || fetchedDetail.current.has(selectedStudyId)) return;
    fetchedDetail.current.add(selectedStudyId);
    getStudy(selectedStudyId)
      .then((detail) => {
        // GET /study/{id} already carries the preview when the server stored one,
        // so no second fetch is needed; a result without it is truly image-less.
        fetchedPreview.current.add(selectedStudyId);
        updateStudy(selectedStudyId, {
          ...detail.study, id: selectedStudyId, restored: true,
          ...(detail.result ? { hasPreview: !!detail.result.previewBase64 } : {}),
        });
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
        addNotification({ type: 'warning', title: t('health.aiServerError') });
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

  // First-run setup — full screen, before the worklist
  if (needsSetup) {
    return (
      <>
        <SetupWizard />
        <Toasts />
      </>
    );
  }

  const healthBanner = !health || !health.reachable
    ? t('health.aiServerDown')
    : health.status === 'degraded'
      ? t('health.aiServerDegraded')
      : health.status === 'error'
        ? t('health.aiServerError')
        : null;

  return (
    <div className="h-screen w-screen flex flex-col bg-ink-950 text-ink-200 overflow-hidden" lang={settings.language}>
      {/* Server health banner — full width, above everything */}
      {healthBanner && (
        <div className="w-full px-4 py-1.5 bg-critical/20 border-b border-critical/40 text-xs text-critical flex items-center gap-2 select-none">
          <span className="w-2 h-2 rounded-full bg-critical animate-pulse flex-shrink-0" />
          <span className="font-semibold">{healthBanner}</span>
          {settings.supportContact && <span className="text-ink-300 ml-2">{t('common.support')}: {settings.supportContact}</span>}
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
      <HelpModal />
      <Toasts />
    </div>
  );
}
