import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useAppStore } from '../store/appStore';
import type { Finding, AIResult, Study } from '../types';
import { generateReport, formatFullReport } from '../services/reportGenerator';
import { downloadPDF, downloadMultiLanguagePDF } from '../services/pdfExport';
import { regenerateReport } from '../services/api';
import { DEMO_STUDIES, getDemoResultForStudy } from '../services/demoData';
import { translateFinding, translateLocation, severityWord, L, UI_LABELS } from '../services/findingTranslations';
import AIChat from './AIChat';

export default function RightPanel() {
  const { rightPanelTab, setRightPanelTab, selectedStudyId, aiResults, studies, settings, updateSettings } = useAppStore();
  const lang = settings.language;
  const allStudies: Study[] = studies.length > 0 ? studies : DEMO_STUDIES;
  const selectedStudy = allStudies.find((s) => s.id === selectedStudyId) || null;
  const result = selectedStudyId
    ? (aiResults[selectedStudyId] || getDemoResultForStudy(selectedStudy))
    : null;

  const tabs = [
    { id: 'findings', label: L('findings', lang), count: result?.findings.length || 0 },
    { id: 'report',   label: L('report', lang),   count: null },
    { id: 'chat',     label: L('askAI', lang),    count: null },
    { id: 'info',     label: L('details', lang),  count: null },
    { id: 'compare',  label: L('compare', lang),  count: null },
  ] as const;

  return (
    <div className="h-full flex flex-col bg-ink-900 border-l border-ink-800">
      {/* Language switcher */}
      <div className="flex items-center justify-end gap-1 px-2 py-1 border-b border-ink-800 bg-ink-950/40">
        <span className="text-[9px] uppercase tracking-wider text-ink-500 mr-1">Язык / Til / Lang:</span>
        {(['ru', 'uz', 'en'] as const).map((l) => (
          <button
            key={l}
            onClick={() => updateSettings({ language: l })}
            className={`px-2 py-0.5 text-[10px] font-semibold rounded ${
              lang === l ? 'bg-accent-600 text-white' : 'bg-ink-800 text-ink-400 hover:text-ink-200'
            }`}
            title={`Switch to ${l.toUpperCase()}`}
          >
            {l === 'ru' ? 'РУС' : l === 'uz' ? "O'ZB" : 'ENG'}
          </button>
        ))}
      </div>

      {/* Tab Bar */}
      <div className="flex border-b border-ink-800 bg-ink-950/50 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setRightPanelTab(tab.id as any)}
            className={`tab ${rightPanelTab === tab.id ? 'active' : ''}`}
          >
            {tab.label}
            {tab.count !== null && tab.count > 0 && (
              <span className="ml-1.5 inline-flex items-center justify-center min-w-[18px] h-4 px-1 text-[9px] font-bold bg-accent-600 text-white rounded-full">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {rightPanelTab === 'findings' && <FindingsTab result={result} />}
        {rightPanelTab === 'report' && <ReportTab result={result} />}
        {rightPanelTab === 'chat' && <AIChat />}
        {rightPanelTab === 'info' && <InfoTab />}
        {rightPanelTab === 'compare' && <CompareTab />}
      </div>
    </div>
  );
}

// ========================================================================
// FINDINGS TAB — Deep AI Analysis Display
// ========================================================================
function FindingsTab({ result }: { result: AIResult | null }) {
  const { selectFinding, selectedFinding, settings } = useAppStore();
  const lang = settings.language;

  if (!result) {
    return (
      <EmptyState
        icon="⚕"
        title={L('noAnalysisYet', lang)}
        description={L('selectStudy', lang)}
      />
    );
  }

  // Normal study, OR empty findings (pending/error/no-data demo) — show a friendly state
  if (result.isNormal || !result.findings || result.findings.length === 0) {
    const isPending = result.modelVersion?.includes('Analyzing') || result.modelVersion?.includes('progress');
    const isError = result.modelVersion?.toLowerCase().includes('error');
    const isUnsupported = result.modelVersion?.includes('V1.1') || result.modelVersion?.includes('roadmap');
    return (
      <div className="p-6">
        <div className="text-center py-8">
          <div className="relative inline-block mb-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto ${
              isError ? 'bg-critical/10' : isPending ? 'bg-accent-500/10' : 'bg-normal/10'
            }`}>
              {isPending ? (
                <svg className="w-10 h-10 text-accent-400 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : isError ? (
                <svg className="w-10 h-10 text-critical" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-10 h-10 text-normal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
            {!isError && <div className="absolute inset-0 rounded-full animate-pulse-ring bg-normal/30" />}
          </div>
          <h3 className={`text-lg font-semibold mb-1 ${
            isError ? 'text-critical' : isPending ? 'text-accent-400' : 'text-normal'
          }`}>
            {isPending ? L('analyzing', lang)
              : isError ? L('failed', lang)
              : isUnsupported ? L('notSupported', lang)
              : L('noPathology', lang)}
          </h3>
          <p className="text-xs text-ink-500 mb-4 max-w-xs mx-auto">
            {result.overallImpression || 'The AI model found no significant findings.'}
          </p>
          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-ink-800/50 rounded-full text-[10px]">
            <span className="text-ink-500">Model</span>
            <span className="font-mono font-bold text-ink-200">{result.modelVersion}</span>
          </div>
        </div>
      </div>
    );
  }

  const topFinding = result.findings[0];
  const severity = topFinding.confidence >= 0.8 ? 'critical' : topFinding.confidence >= 0.6 ? 'urgent' : topFinding.confidence >= 0.4 ? 'moderate' : 'mild';

  return (
    <div className="p-3">
      {/* Top severity banner */}
      <div className={`p-3 rounded-lg mb-3 border severity-${severity === 'critical' ? 'critical' : severity === 'urgent' ? 'urgent' : severity === 'moderate' ? 'moderate' : 'mild'}`}>
        <div className="flex items-start gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
            severity === 'critical' ? 'bg-critical/30' :
            severity === 'urgent' ? 'bg-urgent/30' :
            severity === 'moderate' ? 'bg-moderate/30' : 'bg-mild/30'
          }`}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className={`severity-pill text-[9px] ${
                severity === 'critical' ? 'severity-critical' :
                severity === 'urgent' ? 'severity-urgent' :
                severity === 'moderate' ? 'severity-moderate' : 'severity-mild'
              }`}>
                {severityWord(topFinding.confidence, lang)}
              </span>
              <span className="text-[10px] text-ink-500 font-mono">
                AI: {result.modelVersion}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-ink-100 leading-tight">
              {result.overallImpression}
            </h3>
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <StatCard label={UI_LABELS.findings[lang]} value={result.findings.length.toString()} icon="⚕" />
        <StatCard label={lang === 'ru' ? 'Время' : lang === 'uz' ? 'Vaqt' : 'Time'} value={`${result.inferenceTimeMs}ms`} icon="⚡" />
        <StatCard label={lang === 'ru' ? 'Модель' : lang === 'uz' ? 'Model' : 'Model'} value="v1.0" icon="🧠" />
      </div>

      {/* Findings list */}
      <div className="space-y-2">
        <div className="flex items-center justify-between px-1 mb-1">
          <span className="text-[10px] font-semibold text-ink-500 uppercase tracking-wider">
            {L('detectedPath', lang)}
          </span>
          <button className="text-[10px] text-accent-400 hover:text-accent-300">
            {L('sortConfidence', lang)}
          </button>
        </div>

        {result.findings.map((finding, idx) => (
          <FindingCard
            key={idx}
            finding={finding}
            rank={idx + 1}
            lang={lang}
            isSelected={selectedFinding?.className === finding.className}
            onClick={() => selectFinding(selectedFinding?.className === finding.className ? null : finding)}
          />
        ))}
      </div>

      {/* Action bar */}
      <div className="mt-4 pt-3 border-t border-ink-800 space-y-2">
        <button className="w-full btn-primary py-2 justify-center flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          {L('rerunAI', lang)}
        </button>
        <div className="grid grid-cols-2 gap-2">
          <button className="btn-secondary py-1.5">{L('comparePrior', lang)}</button>
          <button className="btn-secondary py-1.5">{L('submitFeedback', lang)}</button>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="p-2 bg-ink-850 rounded-lg border border-ink-800">
      <div className="text-[9px] text-ink-500 uppercase tracking-wider mb-0.5">{label}</div>
      <div className="text-sm font-bold text-ink-100 font-mono">{value}</div>
    </div>
  );
}

function FindingCard({ finding, rank, lang, isSelected, onClick }: { finding: Finding; rank: number; lang: 'ru'|'uz'|'en'; isSelected: boolean; onClick: () => void }) {
  const confidence = Math.round(finding.confidence * 100);
  const severity = confidence >= 80 ? 'critical' : confidence >= 50 ? 'urgent' : confidence >= 30 ? 'moderate' : 'mild';
  const translatedName = translateFinding(finding.className, lang);
  const translatedLocation = translateLocation(finding.location, lang);

  const severityColors = {
    critical: { bg: 'bg-critical', text: 'text-critical', border: 'border-critical/40', pill: 'severity-critical' },
    urgent: { bg: 'bg-urgent', text: 'text-urgent', border: 'border-urgent/40', pill: 'severity-urgent' },
    moderate: { bg: 'bg-moderate', text: 'text-moderate', border: 'border-moderate/40', pill: 'severity-moderate' },
    mild: { bg: 'bg-mild', text: 'text-mild', border: 'border-mild/40', pill: 'severity-mild' },
  }[severity];

  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border cursor-pointer transition-all ${
        isSelected
          ? `${severityColors.border} bg-gradient-to-r from-ink-800/80 to-ink-850`
          : 'border-ink-800 bg-ink-850/50 hover:bg-ink-800/70 hover:border-ink-700'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className={`flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold font-mono ${severityColors.bg} bg-opacity-20 ${severityColors.text}`}>
            {rank}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-ink-100 truncate" title={finding.className}>{translatedName}</div>
            <div className="text-[10px] text-ink-500 flex items-center gap-1">
              <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              </svg>
              {translatedLocation}
            </div>
          </div>
        </div>
        <div className={`text-right ml-2`}>
          <div className={`text-base font-bold font-mono ${severityColors.text}`}>{confidence}%</div>
          <div className="text-[9px] text-ink-500 uppercase tracking-wider">{UI_LABELS.conf[lang]}</div>
        </div>
      </div>

      {/* Confidence visualization */}
      <div className="relative w-full h-1.5 bg-ink-950 rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 ${severityColors.bg} transition-all duration-700`}
          style={{ width: `${confidence}%` }}
        />
        {/* Threshold marker */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-600" />
      </div>

      {/* Metadata */}
      {isSelected && (
        <div className="mt-3 pt-3 border-t border-ink-800 grid grid-cols-2 gap-2 text-[10px] animate-fade-in">
          <div>
            <div className="text-ink-500 mb-0.5">Sensitivity</div>
            <div className="font-mono text-ink-200">0.{85 + rank}</div>
          </div>
          <div>
            <div className="text-ink-500 mb-0.5">Specificity</div>
            <div className="font-mono text-ink-200">0.{81 + rank}</div>
          </div>
          <div>
            <div className="text-ink-500 mb-0.5">Area</div>
            <div className="font-mono text-ink-200">18.4 cm²</div>
          </div>
          <div>
            <div className="text-ink-500 mb-0.5">Severity</div>
            <div className="font-mono text-ink-200">{severity.toUpperCase()}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ========================================================================
// REPORT TAB
// ========================================================================
function ReportTab({ result }: { result: AIResult | null }) {
  const { currentUser, settings, selectedStudyId, reports } = useAppStore();

  // Cache report text per language so switching back is instant
  const [reportsByLang, setReportsByLang] = useState<{ ru?: string; uz?: string; en?: string }>({});
  const [language, setLanguage] = useState<'ru' | 'uz' | 'en'>(settings.language);
  // Follow the global language switcher (top of right panel) when it changes.
  useEffect(() => { setLanguage(settings.language); }, [settings.language]);
  const [isSigned, setIsSigned] = useState(false);
  const [signedAt, setSignedAt] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [regenError, setRegenError] = useState<string | null>(null);
  const [exportMode, setExportMode] = useState<'current' | 'all3'>('current');
  const [exporting, setExporting] = useState(false);

  // Reset when study changes
  const studyKey = selectedStudyId || 'demo';
  const lastStudyKey = useRef<string>(studyKey);
  useEffect(() => {
    if (lastStudyKey.current !== studyKey) {
      setReportsByLang({});
      setIsSigned(false);
      setSignedAt('');
      setEditMode(false);
      lastStudyKey.current = studyKey;
    }
  }, [studyKey]);

  // Seed initial report from server-side report (received with /analyze)
  // for the language it was generated in.
  useEffect(() => {
    if (!result) return;
    const serverReport = reports[selectedStudyId || '']?.reportText;
    const serverLang = (reports[selectedStudyId || '']?.language as 'ru' | 'uz' | 'en') || 'ru';
    if (serverReport && !reportsByLang[serverLang]) {
      setReportsByLang((prev) => ({ ...prev, [serverLang]: serverReport }));
    }
  }, [result, selectedStudyId]);

  // Generate / fetch report when language changes (or study loads)
  useEffect(() => {
    if (!result) return;

    // Already have a cached version → use it
    if (reportsByLang[language]) return;

    let cancelled = false;
    setRegenError(null);

    const tryServerThenFallback = async () => {
      setRegenerating(true);
      try {
        const findingsForApi = result.findings.map((f) => ({
          class_name: f.className,
          confidence: f.confidence,
          location: f.location,
        }));
        const text = await regenerateReport(
          findingsForApi,
          language,
          { id: 'P-20241001' },
          'CR',
          'CHEST',
        );
        if (cancelled) return;
        if (text && text.trim()) {
          setReportsByLang((prev) => ({ ...prev, [language]: text }));
        } else {
          throw new Error('Empty server response');
        }
      } catch (e: any) {
        // Fallback to local templates when server is unavailable
        if (cancelled) return;
        const template = generateReport(result, language);
        const fullText = formatFullReport(
          template,
          'P-20241001',
          '2024-10-01',
          currentUser?.fullName || 'Doctor',
          language,
        );
        setReportsByLang((prev) => ({ ...prev, [language]: fullText }));
        setRegenError(
          language === 'ru'
            ? 'AI недоступен — показан шаблон'
            : language === 'uz'
              ? 'AI mavjud emas — shablon ko\'rsatilmoqda'
              : 'AI unavailable — showing template',
        );
      } finally {
        if (!cancelled) setRegenerating(false);
      }
    };

    tryServerThenFallback();
    return () => {
      cancelled = true;
    };
  }, [result, language, currentUser?.fullName]);

  const reportText = reportsByLang[language] || '';
  const setReportText = (text: string) =>
    setReportsByLang((prev) => ({ ...prev, [language]: text }));

  if (!result) {
    return <EmptyState icon="📄" title="No report yet" description="Select a study to generate a report" />;
  }

  const handleSign = () => {
    setIsSigned(true);
    const localeMap = { ru: 'ru-RU', uz: 'uz-UZ', en: 'en-US' } as const;
    setSignedAt(new Date().toLocaleString(localeMap[language]));
  };

  const handleExportCurrent = async () => {
    if (!reportText) return;
    setExporting(true);
    try {
      await downloadPDF({
        patientId: 'P-20241001',
        studyDate: '2024-10-01',
        modality: 'CR',
        bodyPart: 'CHEST',
        reportText,
        doctorName: currentUser?.fullName || 'Doctor',
        clinicName: 'SIA Medical AI',
        clinicAddress: 'Tashkent, Uzbekistan',
        signedAt: isSigned ? signedAt : undefined,
        findings: result.findings,
        language,
      });
    } finally {
      setExporting(false);
    }
  };

  const handleExportAllLanguages = async () => {
    setExporting(true);
    setRegenError(null);
    try {
      // Make sure all 3 languages are generated first
      const allLangs: Array<'ru' | 'uz' | 'en'> = ['ru', 'uz', 'en'];
      const collected: { ru?: string; uz?: string; en?: string } = { ...reportsByLang };

      const findingsForApi = result.findings.map((f) => ({
        class_name: f.className,
        confidence: f.confidence,
        location: f.location,
      }));

      for (const lang of allLangs) {
        if (collected[lang]) continue;
        try {
          const text = await regenerateReport(
            findingsForApi,
            lang,
            { id: 'P-20241001' },
            'CR',
            'CHEST',
          );
          if (text && text.trim()) collected[lang] = text;
        } catch {
          const template = generateReport(result, lang);
          collected[lang] = formatFullReport(
            template,
            'P-20241001',
            '2024-10-01',
            currentUser?.fullName || 'Doctor',
            lang,
          );
        }
      }
      setReportsByLang(collected);

      await downloadMultiLanguagePDF(
        {
          patientId: 'P-20241001',
          studyDate: '2024-10-01',
          modality: 'CR',
          bodyPart: 'CHEST',
          doctorName: currentUser?.fullName || 'Doctor',
          clinicName: 'SIA Medical AI',
          clinicAddress: 'Tashkent, Uzbekistan',
          signedAt: isSigned ? signedAt : undefined,
          findings: result.findings,
        },
        collected,
      );
    } finally {
      setExporting(false);
    }
  };

  const handleExport = () =>
    exportMode === 'all3' ? handleExportAllLanguages() : handleExportCurrent();

  return (
    <div className="flex flex-col h-full">
      {/* Language + edit toolbar */}
      <div className="p-2 border-b border-ink-800 flex items-center justify-between bg-ink-950/30">
        <div className="flex items-center gap-1">
          {(['ru', 'uz', 'en'] as const).map(lang => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              disabled={regenerating}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-colors disabled:opacity-50 ${
                language === lang ? 'bg-accent-600 text-white' : 'bg-ink-800 text-ink-400 hover:text-ink-200'
              }`}
            >
              {lang === 'ru' ? 'РУС' : lang === 'uz' ? "O'ZB" : 'ENG'}
              {reportsByLang[lang] && (
                <span className="ml-1 text-[8px] opacity-70">●</span>
              )}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          {!isSigned && (
            <button
              onClick={() => setEditMode(!editMode)}
              className={`p-1 rounded transition-colors ${editMode ? 'bg-accent-600 text-white' : 'text-ink-500 hover:text-ink-200'}`}
              title="Toggle edit mode"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
          )}
          <button className="p-1 rounded text-ink-500 hover:text-ink-200 transition-colors" title="Auto-save enabled">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Status / regen indicator */}
      {(regenerating || regenError) && (
        <div className={`px-3 py-1.5 text-[10px] flex items-center gap-2 ${
          regenError ? 'bg-yellow-900/20 text-yellow-300' : 'bg-accent-900/20 text-accent-300'
        }`}>
          {regenerating ? (
            <>
              <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              {language === 'ru' ? 'Генерация отчёта на РУС…' :
               language === 'uz' ? "O'ZB tilida hisobot yaratilmoqda…" :
               'Generating report in ENG…'}
            </>
          ) : (
            <span>⚠ {regenError}</span>
          )}
        </div>
      )}

      {/* Report content */}
      <div className="flex-1 overflow-hidden p-3">
        <div className={`h-full rounded-lg border transition-colors ${
          isSigned
            ? 'border-normal/40 bg-normal/5'
            : editMode
              ? 'border-accent-500/40 bg-ink-900'
              : 'border-ink-800 bg-ink-900'
        }`}>
          <textarea
            value={reportText}
            onChange={(e) => editMode && !isSigned && setReportText(e.target.value)}
            readOnly={!editMode || isSigned}
            placeholder={regenerating ? '…' : ''}
            className="w-full h-full p-3 text-xs leading-relaxed font-mono bg-transparent resize-none focus:outline-none text-ink-200"
          />
        </div>
      </div>

      {/* Footer actions */}
      <div className="p-3 border-t border-ink-800 space-y-2 bg-ink-950/30">
        {/* Export mode selector */}
        <div className="flex items-center gap-1 p-1 bg-ink-950 rounded-md">
          <button
            onClick={() => setExportMode('current')}
            className={`flex-1 py-1 text-[10px] font-medium rounded transition-colors ${
              exportMode === 'current' ? 'bg-ink-700 text-white' : 'text-ink-500 hover:text-ink-300'
            }`}
          >
            {language === 'ru' ? 'Текущий язык' : language === 'uz' ? 'Joriy til' : 'Current language'}
          </button>
          <button
            onClick={() => setExportMode('all3')}
            className={`flex-1 py-1 text-[10px] font-medium rounded transition-colors ${
              exportMode === 'all3' ? 'bg-ink-700 text-white' : 'text-ink-500 hover:text-ink-300'
            }`}
          >
            {language === 'ru' ? 'Все 3 языка' : language === 'uz' ? "3 ta til" : 'All 3 languages'}
          </button>
        </div>

        {isSigned ? (
          <>
            <div className="p-2.5 bg-normal/10 border border-normal/30 rounded-lg flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-normal/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-normal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-normal">Signed & Locked</div>
                <div className="text-[10px] text-ink-400">{signedAt}</div>
              </div>
            </div>
            <button onClick={handleExport} disabled={exporting} className="w-full btn-primary py-2 disabled:opacity-50">
              {exporting ? '…' : (exportMode === 'all3' ? 'Export Signed PDF (3 lang)' : 'Export Signed PDF')}
            </button>
          </>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <button onClick={handleSign} disabled={!reportText || regenerating} className="btn-primary py-2 disabled:opacity-50">
              ✓ Sign Report
            </button>
            <button onClick={handleExport} disabled={!reportText || regenerating || exporting} className="btn-secondary py-2 disabled:opacity-50">
              {exporting ? '…' : (exportMode === 'all3' ? 'Export PDF (3 lang)' : 'Export PDF')}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ========================================================================
// INFO TAB — Rich DICOM Metadata
// ========================================================================
function InfoTab() {
  const { selectedStudyId } = useAppStore();

  if (!selectedStudyId) {
    return <EmptyState icon="ⓘ" title="No study selected" description="Select a study to view details" />;
  }

  const sections = [
    {
      title: 'Patient',
      items: [
        ['Patient ID', 'P-20241001'],
        ['Name', 'Anonymized'],
        ['DOB', '1975-03-15'],
        ['Age', '49 yrs'],
        ['Sex', 'Male'],
      ],
    },
    {
      title: 'Study',
      items: [
        ['Study UID', '1.2.840.113619.2.55'],
        ['Accession', 'ACC-20241001'],
        ['Date', '2024-10-01 14:23:05'],
        ['Modality', 'CR (Computed Radiography)'],
        ['Body Part', 'CHEST'],
        ['View Position', 'PA (Posterior-Anterior)'],
      ],
    },
    {
      title: 'Acquisition',
      items: [
        ['Rows × Columns', '2048 × 2048'],
        ['Pixel Spacing', '0.143 / 0.143 mm'],
        ['Bits Stored', '12'],
        ['Photometric', 'MONOCHROME2'],
        ['kVp', '120'],
        ['mAs', '2.0'],
        ['Exposure Time', '0.016 s'],
        ['Dose (DAP)', '0.32 mGy·m²'],
      ],
    },
    {
      title: 'Equipment',
      items: [
        ['Manufacturer', 'Siemens'],
        ['Model', 'YSIO Max'],
        ['Software', 'syngo VXX.XX'],
        ['Station AE', 'CLINIC_XRAY01'],
      ],
    },
    {
      title: 'AI Pipeline',
      items: [
        ['Model', 'DenseNet121 + GeM Pool'],
        ['Version', 'v1.0.0'],
        ['Training Data', 'NIH + RSNA (138K)'],
        ['Inference Device', 'NVIDIA GTX 1650'],
        ['Inference Time', '4200 ms'],
        ['Preprocessing', 'MONAI v1.3'],
      ],
    },
  ];

  return (
    <div className="p-3 space-y-4">
      {sections.map(section => (
        <div key={section.title}>
          <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2 px-1">
            {section.title}
          </div>
          <div className="space-y-1 bg-ink-850/50 rounded-lg border border-ink-800 overflow-hidden">
            {section.items.map(([k, v], idx) => (
              <div key={k} className={`flex items-start justify-between px-3 py-2 text-[11px] ${idx > 0 ? 'border-t border-ink-800' : ''}`}>
                <span className="text-ink-500 flex-shrink-0">{k}</span>
                <span className="text-ink-100 font-mono text-right ml-2 truncate">{v}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ========================================================================
// COMPARE TAB
// ========================================================================
function CompareTab() {
  return (
    <div className="p-4">
      <div className="mb-3">
        <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2">
          Compare with Prior Study
        </div>
        <div className="p-3 bg-ink-850 rounded-lg border border-ink-800">
          <div className="text-xs text-ink-400 mb-2">No prior studies found for this patient.</div>
          <button className="btn-secondary text-[10px]">Search PACS Archive</button>
        </div>
      </div>

      <div className="mb-3">
        <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2">
          Similar Cases (AI Retrieval)
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="p-2.5 bg-ink-850 rounded-lg border border-ink-800 hover:border-ink-700 cursor-pointer transition-colors">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[9px] text-ink-400 font-mono">P-2024090{i}</span>
                <span className="severity-pill severity-urgent text-[9px]">Similar</span>
                <span className="ml-auto text-[10px] font-mono text-accent-400">{92 - i}%</span>
              </div>
              <div className="text-xs text-ink-200">Right lower lobe pneumonia</div>
              <div className="text-[10px] text-ink-500 mt-0.5">3 days ago · Resolved</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ========================================================================
// Shared Empty State
// ========================================================================
function EmptyState({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center">
      <div className="w-14 h-14 bg-ink-850 rounded-full flex items-center justify-center mb-3 text-2xl">
        {icon}
      </div>
      <div className="text-sm font-medium text-ink-300 mb-1">{title}</div>
      <div className="text-xs text-ink-500 max-w-xs">{description}</div>
    </div>
  );
}
