import { useEffect } from 'react';
import { useAppStore } from '../store/appStore';

/**
 * Global keyboard shortcuts. Every binding here has a real action and is
 * listed in Settings → Shortcuts; tools that the viewer does not implement
 * have no binding.
 */
export function useKeyboardShortcuts() {
  const {
    setActiveTool,
    toggleHeatmap,
    toggleSidebar,
    toggleRightPanel,
    toggleThumbnailStrip,
    setCommandPaletteOpen,
    openSettings,
    openHelp,
    requestUpload,
    requestExport,
    setViewerZoom,
    setPanOffset,
    setWindowLevel,
  } = useAppStore();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't handle shortcuts when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return;

      const key = e.key.toLowerCase();
      const cmd = e.metaKey || e.ctrlKey;

      if (e.key === 'F1') {
        e.preventDefault();
        openHelp();
        return;
      }

      // Modifier shortcuts
      if (cmd) {
        switch (key) {
          case 'k':
            e.preventDefault();
            setCommandPaletteOpen(true);
            return;
          case ',':
            e.preventDefault();
            openSettings();
            return;
          case '/':
            e.preventDefault();
            openSettings('shortcuts');
            return;
          case 'b':
            e.preventDefault();
            if (e.shiftKey) toggleRightPanel();
            else toggleSidebar();
            return;
          case 'o':
            e.preventDefault();
            requestUpload(e.shiftKey ? 'folder' : 'files');
            return;
          case 'p':
            e.preventDefault();
            requestExport();
            return;
        }
        return;
      }

      // Single-key tool shortcuts
      if (!e.altKey) {
        const s = useAppStore.getState();
        switch (key) {
          case 'p': setActiveTool('pan'); break;
          case 'z': setActiveTool('zoom'); break;
          case 'w': setActiveTool('wwwl'); break;
          case 'l': setActiveTool('length'); break;
          case 'g': setActiveTool('angle'); break;
          case 'r': s.setRotation((s.rotation + 90) % 360); break;
          case 'i': s.toggleInvert(); break;
          case 't': toggleThumbnailStrip(); break;
          case 'h': toggleHeatmap(); break;
          case 'f': setViewerZoom(1); setPanOffset({ x: 0, y: 0 }); break;
          case '1': setWindowLevel({ center: 50, width: 350 }); break;  // Chest
          case '2': setWindowLevel({ center: -500, width: 1500 }); break; // Lung
          case '3': setWindowLevel({ center: 400, width: 1800 }); break;  // Bone
          case '4': setWindowLevel({ center: 50, width: 400 }); break;    // Soft
          case '5': setWindowLevel({ center: 40, width: 80 }); break;     // Brain
          case 'escape':
            setCommandPaletteOpen(false);
            break;
        }
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [setActiveTool, toggleHeatmap, toggleSidebar, toggleRightPanel, toggleThumbnailStrip, setCommandPaletteOpen, openSettings, openHelp, requestUpload, requestExport, setViewerZoom, setPanOffset, setWindowLevel]);
}
