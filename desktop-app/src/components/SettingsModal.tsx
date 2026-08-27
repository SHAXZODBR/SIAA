import React, { useState } from 'react';
import { useAppStore } from '../store/appStore';

const SECTIONS = [
  { id: 'general', label: 'General', icon: '⚙' },
  { id: 'connections', label: 'Connections', icon: '🔗' },
  { id: 'appearance', label: 'Appearance', icon: '🎨' },
  { id: 'language', label: 'Language', icon: '🌐' },
  { id: 'ai', label: 'AI Model', icon: '🧠' },
  { id: 'security', label: 'Security', icon: '🔒' },
  { id: 'notifications', label: 'Notifications', icon: '🔔' },
  { id: 'shortcuts', label: 'Shortcuts', icon: '⌨' },
  { id: 'about', label: 'About', icon: 'ⓘ' },
];

export default function SettingsModal() {
  const { settingsOpen, closeSettings, settings, updateSettings } = useAppStore();
  const [section, setSection] = useState('general');

  if (!settingsOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-ink-950/70 backdrop-blur-sm animate-fade-in"
      onClick={closeSettings}
    >
      <div
        className="w-[900px] h-[640px] max-w-[95vw] max-h-[90vh] glass rounded-xl border border-ink-700/50 shadow-2xl overflow-hidden flex animate-scale-in"
        onClick={e => e.stopPropagation()}
      >
        {/* Sidebar */}
        <div className="w-56 border-r border-ink-800 bg-ink-950/30 flex flex-col">
          <div className="p-4 border-b border-ink-800">
            <h2 className="text-sm font-bold text-ink-100">Preferences</h2>
            <p className="text-[10px] text-ink-500 mt-0.5">Sentinel Medical AI v1.0.0</p>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {SECTIONS.map(s => (
              <button
                key={s.id}
                onClick={() => setSection(s.id)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors ${
                  section === s.id
                    ? 'bg-accent-600/20 text-white'
                    : 'text-ink-300 hover:bg-ink-800/50 hover:text-white'
                }`}
              >
                <span className="text-base">{s.icon}</span>
                <span>{s.label}</span>
              </button>
            ))}
          </div>
          <div className="p-3 border-t border-ink-800">
            <button onClick={closeSettings} className="w-full btn-ghost text-center">
              Close
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {section === 'general' && <GeneralSection settings={settings} update={updateSettings} />}
          {section === 'connections' && <ConnectionsSection settings={settings} update={updateSettings} />}
          {section === 'appearance' && <AppearanceSection />}
          {section === 'language' && <LanguageSection settings={settings} update={updateSettings} />}
          {section === 'ai' && <AISection />}
          {section === 'security' && <SecuritySection />}
          {section === 'notifications' && <NotificationsSection />}
          {section === 'shortcuts' && <ShortcutsSection />}
          {section === 'about' && <AboutSection />}
        </div>
      </div>
    </div>
  );
}

function SectionHeader({ title, desc }: { title: string; desc?: string }) {
  return (
    <div className="p-6 border-b border-ink-800">
      <h3 className="text-lg font-bold text-ink-100">{title}</h3>
      {desc && <p className="text-xs text-ink-400 mt-0.5">{desc}</p>}
    </div>
  );
}

function Row({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="py-3 border-b border-ink-800/50 flex items-start justify-between gap-4">
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-ink-100">{label}</div>
        {desc && <div className="text-xs text-ink-500 mt-0.5">{desc}</div>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative w-10 h-5 rounded-full transition-colors ${
        checked ? 'bg-accent-600' : 'bg-ink-700'
      }`}
    >
      <span
        className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-md transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-0.5'
        }`}
      />
    </button>
  );
}

function GeneralSection({ settings, update }: any) {
  return (
    <div>
      <SectionHeader title="General" desc="Common preferences for Sentinel" />
      <div className="p-6 space-y-1">
        <Row label="Auto-analyze new studies" desc="Automatically run AI on new incoming DICOM files">
          <Toggle checked={settings.autoAnalyze} onChange={(v) => update({ autoAnalyze: v })} />
        </Row>
        <Row label="Refresh interval" desc="How often to check for new studies (seconds)">
          <input
            type="number"
            value={settings.refreshInterval}
            onChange={(e) => update({ refreshInterval: +e.target.value })}
            className="input-medical !w-20 !py-1 !text-center"
          />
        </Row>
        <Row label="Storage path" desc="Local directory for studies and reports">
          <input
            type="text"
            value={settings.storagePath}
            onChange={(e) => update({ storagePath: e.target.value })}
            className="input-medical !py-1 !w-64"
          />
        </Row>
        <Row label="Open last study on startup" desc="Resume where you left off">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
        <Row label="Enable telemetry" desc="Help improve Sentinel (anonymous usage data only)">
          <Toggle checked={false} onChange={() => {}} />
        </Row>
      </div>
    </div>
  );
}

function ConnectionsSection({ settings, update }: any) {
  return (
    <div>
      <SectionHeader title="Connections" desc="Configure PACS and AI server endpoints" />
      <div className="p-6 space-y-1">
        <Row label="Orthanc PACS URL" desc="Local PACS server receiving DICOM files">
          <input
            type="text"
            value={settings.orthancUrl}
            onChange={(e) => update({ orthancUrl: e.target.value })}
            className="input-medical !py-1 !w-72 !font-mono"
          />
        </Row>
        <Row label="AI Inference URL" desc="Local FastAPI server for DenseNet121">
          <input
            type="text"
            value={settings.inferenceUrl}
            onChange={(e) => update({ inferenceUrl: e.target.value })}
            className="input-medical !py-1 !w-72 !font-mono"
          />
        </Row>
        <Row label="DICOM AE Title" desc="Application Entity Title for DICOM C-STORE">
          <input type="text" defaultValue="SENTINEL" className="input-medical !py-1 !w-40 !font-mono" />
        </Row>
        <Row label="DICOM Port" desc="Port for incoming DICOM C-STORE transfers">
          <input type="number" defaultValue={4242} className="input-medical !py-1 !w-24 !text-center !font-mono" />
        </Row>
        <div className="mt-6 p-3 bg-ink-850 rounded-lg border border-ink-800">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-normal" />
            <span className="text-xs font-semibold text-ink-100">Connection Status</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="flex items-center gap-1.5 text-normal">
              <div className="w-1.5 h-1.5 rounded-full bg-normal animate-pulse" />
              Orthanc: Connected
            </div>
            <div className="flex items-center gap-1.5 text-normal">
              <div className="w-1.5 h-1.5 rounded-full bg-normal animate-pulse" />
              AI Server: Ready
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function AppearanceSection() {
  return (
    <div>
      <SectionHeader title="Appearance" desc="Customize how Sentinel looks" />
      <div className="p-6 space-y-4">
        <div>
          <div className="text-sm font-medium text-ink-100 mb-2">Theme</div>
          <div className="grid grid-cols-3 gap-2">
            {[
              { id: 'dark', name: 'Dark (Medical)', preview: 'bg-gradient-to-br from-ink-900 to-ink-950' },
              { id: 'darker', name: 'Pitch Black', preview: 'bg-black' },
              { id: 'midnight', name: 'Midnight Blue', preview: 'bg-gradient-to-br from-slate-900 to-blue-950' },
            ].map(t => (
              <button key={t.id} className="rounded-lg overflow-hidden border-2 border-accent-500 p-2 text-left">
                <div className={`h-16 rounded ${t.preview} mb-2`} />
                <div className="text-xs font-medium text-ink-100">{t.name}</div>
              </button>
            ))}
          </div>
        </div>
        <Row label="Compact mode" desc="Reduce spacing for denser information display">
          <Toggle checked={false} onChange={() => {}} />
        </Row>
        <Row label="Animations" desc="Smooth transitions and effects">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
      </div>
    </div>
  );
}

function LanguageSection({ settings, update }: any) {
  const languages = [
    { id: 'ru', name: 'Русский', native: 'Russian', progress: 100 },
    { id: 'uz', name: "O'zbek", native: 'Uzbek', progress: 100 },
    { id: 'en', name: 'English', native: 'English', progress: 100 },
  ];
  return (
    <div>
      <SectionHeader title="Language & Region" desc="Choose the interface and report language" />
      <div className="p-6 space-y-2">
        <div className="text-sm font-medium text-ink-100 mb-2">Interface Language</div>
        {languages.map(l => (
          <button
            key={l.id}
            onClick={() => update({ language: l.id })}
            className={`w-full p-3 flex items-center justify-between rounded-lg border transition-all ${
              settings.language === l.id
                ? 'border-accent-500 bg-accent-900/20'
                : 'border-ink-800 hover:border-ink-700 bg-ink-850'
            }`}
          >
            <div className="text-left">
              <div className="text-sm font-semibold text-ink-100">{l.name}</div>
              <div className="text-[10px] text-ink-500">{l.native} · {l.progress}% translated</div>
            </div>
            {settings.language === l.id && (
              <svg className="w-5 h-5 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function AISection() {
  return (
    <div>
      <SectionHeader title="AI Model" desc="Configure the AI analysis pipeline" />
      <div className="p-6 space-y-1">
        <div className="p-4 bg-ink-850 rounded-lg border border-ink-800 mb-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-sm font-semibold text-ink-100">DenseNet121 + nnU-Net</div>
              <div className="text-[10px] text-ink-500 font-mono">models/densenet/best_model.pt</div>
            </div>
            <span className="severity-pill severity-normal">Loaded</span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-[10px]">
            <div>
              <div className="text-ink-500">Architecture</div>
              <div className="text-ink-200 font-mono">DenseNet121</div>
            </div>
            <div>
              <div className="text-ink-500">Version</div>
              <div className="text-ink-200 font-mono">v1.0.0</div>
            </div>
            <div>
              <div className="text-ink-500">Classes</div>
              <div className="text-ink-200 font-mono">14</div>
            </div>
          </div>
        </div>

        <Row label="Confidence threshold" desc="Minimum confidence for reporting findings">
          <input type="number" step="0.01" defaultValue={0.5} min={0} max={1} className="input-medical !py-1 !w-20 !text-center !font-mono" />
        </Row>
        <Row label="Enable Grad-CAM heatmaps" desc="Generate visual explanations for AI decisions">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
        <Row label="Test-time augmentation" desc="Run multiple augmentations, average results (+2-5% accuracy)">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
        <Row label="Use GPU acceleration" desc="NVIDIA CUDA / Apple MPS">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
      </div>
    </div>
  );
}

function SecuritySection() {
  return (
    <div>
      <SectionHeader title="Security & Privacy" desc="Protect patient data and control access" />
      <div className="p-6 space-y-1">
        <Row label="Require login" desc="Authenticate users before accessing studies">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
        <Row label="Auto-lock after idle" desc="Minutes of inactivity before locking">
          <input type="number" defaultValue={15} className="input-medical !py-1 !w-20 !text-center" />
        </Row>
        <Row label="Audit logging" desc="Record all user actions for compliance">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
        <Row label="Database encryption" desc="Use OS disk encryption (FileVault / BitLocker)">
          <span className="severity-pill severity-moderate">OS-level</span>
        </Row>
        <Row label="Two-factor authentication" desc="Require 2FA for admin accounts">
          <Toggle checked={false} onChange={() => {}} />
        </Row>
        <Row label="DICOM anonymization" desc="Strip PII from exported DICOM files">
          <Toggle checked={true} onChange={() => {}} />
        </Row>
      </div>
    </div>
  );
}

function NotificationsSection() {
  return (
    <div>
      <SectionHeader title="Notifications" desc="Control alerts and sounds" />
      <div className="p-6 space-y-1">
        <Row label="New study alerts" desc="Notify when new studies arrive"><Toggle checked={true} onChange={() => {}} /></Row>
        <Row label="Critical finding alerts" desc="Priority alerts for severe pathologies"><Toggle checked={true} onChange={() => {}} /></Row>
        <Row label="Sound on new study" desc="Play chime when study arrives"><Toggle checked={true} onChange={() => {}} /></Row>
        <Row label="Desktop notifications" desc="System-level notifications"><Toggle checked={true} onChange={() => {}} /></Row>
      </div>
    </div>
  );
}

function ShortcutsSection() {
  const shortcuts = [
    { group: 'Tools', items: [['Pan', 'P'], ['Zoom', 'Z'], ['Window/Level', 'W'], ['Length', 'L'], ['Angle', 'G'], ['Ellipse ROI', 'E']] },
    { group: 'View', items: [['Toggle Heatmap', 'H'], ['Toggle Sidebar', '⌘B'], ['Fullscreen', 'F11'], ['Fit to Window', 'F']] },
    { group: 'AI', items: [['Re-run Analysis', '⌘↩'], ['Show Confidence', 'C'], ['Show Segmentation', 'S']] },
    { group: 'File', items: [['Open', '⌘O'], ['Export PDF', '⌘P'], ['Preferences', '⌘,'], ['Command Palette', '⌘K']] },
  ];
  return (
    <div>
      <SectionHeader title="Keyboard Shortcuts" desc="Speed up your workflow" />
      <div className="p-6 space-y-4">
        {shortcuts.map(g => (
          <div key={g.group}>
            <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2">{g.group}</div>
            <div className="bg-ink-850 rounded-lg border border-ink-800 overflow-hidden">
              {g.items.map(([label, key], i) => (
                <div key={label} className={`flex items-center justify-between px-3 py-2 ${i > 0 ? 'border-t border-ink-800' : ''}`}>
                  <span className="text-sm text-ink-200">{label}</span>
                  <span className="kbd">{key}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AboutSection() {
  return (
    <div>
      <SectionHeader title="About Sentinel Medical AI" />
      <div className="p-6">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 bg-gradient-to-br from-accent-400 to-accent-700 rounded-2xl flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h3 className="text-xl font-bold text-ink-100">Sentinel Medical AI</h3>
            <div className="text-xs text-ink-400">Version 1.0.0 (build 20240101)</div>
            <div className="text-[10px] text-ink-500 mt-1">© 2024 SIA Medical AI · siaa.uz</div>
          </div>
        </div>
        <div className="p-4 bg-ink-850 rounded-lg border border-ink-800 text-xs text-ink-300 leading-relaxed">
          Sentinel Medical AI is an on-premise radiology AI assistant built for clinics in Central Asia.
          It automatically analyzes DICOM medical images using Stanford-validated DenseNet121 architecture,
          providing radiologists with AI-assisted findings, heatmaps, and structured reports in Russian, Uzbek, and English.
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 text-[10px]">
          <div className="p-2 bg-ink-850 rounded border border-ink-800">
            <div className="text-ink-500">Built with</div>
            <div className="text-ink-200 font-mono">PyTorch · Electron · React</div>
          </div>
          <div className="p-2 bg-ink-850 rounded border border-ink-800">
            <div className="text-ink-500">AI Framework</div>
            <div className="text-ink-200 font-mono">MONAI · nnU-Net</div>
          </div>
        </div>
        <div className="mt-4 flex gap-2">
          <button className="btn-secondary flex-1">Check for Updates</button>
          <button className="btn-secondary flex-1">View License</button>
        </div>
      </div>
    </div>
  );
}
