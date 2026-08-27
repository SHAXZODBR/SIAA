import React, { useEffect } from 'react';
import { useAppStore } from './store/appStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { checkInferenceHealth, checkOrthancHealth } from './services/api';

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

export default function App() {
  const { currentUser, setConnectionStatus, sidebarOpen, rightPanelOpen, thumbnailStripOpen, addNotification } = useAppStore();

  // Setup keyboard shortcuts
  useKeyboardShortcuts();

  // Check connections periodically
  useEffect(() => {
    if (!currentUser) return;

    const checkConnections = async () => {
      try {
        const [orthanc, inference] = await Promise.all([
          checkOrthancHealth(),
          checkInferenceHealth(),
        ]);
        setConnectionStatus(orthanc, inference);
      } catch {
        setConnectionStatus(false, false);
      }
    };

    checkConnections();
    const interval = setInterval(checkConnections, 30000);
    return () => clearInterval(interval);
  }, [currentUser, setConnectionStatus]);

  // Welcome notification
  useEffect(() => {
    if (currentUser) {
      addNotification({
        type: 'info',
        title: 'Welcome back',
        message: `Signed in as ${currentUser.fullName}`,
      });
    }
  }, [currentUser]);

  // Login gate
  if (!currentUser) {
    return <LoginScreen />;
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-ink-950 text-ink-200 overflow-hidden">
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

          {/* Bottom thumbnail strip */}
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
    </div>
  );
}
