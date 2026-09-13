import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useAppStore, signedKey } from '../store/appStore';
import type { Finding, AIResult, Study, Lang, ModelIdentity, SignedReport } from '../types';
import { generateReport, formatFullReport } from '../services/reportGenerator';
import { downloadPDF, downloadMultiLanguagePDF, sanitizeFilePart } from '../services/pdfExport';
import type { PDFBaseData, PDFLanguageSection } from '../services/pdfExport';
import { regenerateReport, signReport, saveReportCorrection, describeApiError } from '../services/api';
import { attachReportPdf } from '../services/api';
import { DEMO_MODE, DEMO_STUDIES, getDemoResultForStudy } from '../services/demoData';
import {
  translateFinding, translateLocation, findingUrgency, urgencyWord, statusWord,
} from '../services/findingTranslations';
import type { FindingUrgency } from '../services/findingTranslations';
import { getAppVersion } from '../services/appInfo';
import { useT, LANGS, langShort, formatDate, formatDateTime } from '../i18n';
import AIChat from './AIChat';

/** Validation-status badge: validated = green, pending = amber, experimental = grey. */
function statusPillClass(status: string): string {
  if (status === 'validated') return 'severity-normal';
  if (status === 'pending') return 'severity-moderate';
  return 'bg-ink-800 text-ink-400 border border-ink-700';
}

/** Urgency badge — derived from positive + validation status, never from the score. */
function urgencyPillClass(urgency: FindingUrgency): string {
  if (urgency === 'review') return 'severity-critical';
  if (urgency === 'unvalidated') return 'severity-moderate';
  return 'bg-ink-800 text-ink-500 border border-ink-700';
}

function isSameFinding(a: Finding | null, b: Finding): boolean {
  return !!a && a.className === b.className && (a.detector || null) === (b.detector || null);
}

export default function RightPanel() {
  const { rightPanelTab, setRightPanelTab, selectedStudyId, aiResults, studies, settings } = useAppStore();
  const t = useT();
  const lang = settings.language;
  const allStudies: Study[] = studies.length > 0 ? studies : (DEMO_MODE ? DEMO_STUDIES : []);
  const study = allStudies.find((s) => s.id === selectedStudyId) || null;
  const result: AIResult | null = selectedStudyId
    ? (aiResults[selectedStudyId] || getDemoResultForStudy(study))
    : null;

  // The Compare tab has no real data source in this build (no PACS prior lookup), so it is not offered.
  const tabs = [
    { id: 'findings', label: t('panel.findings'), count: result ? result.findings.filter((f) => f.positive).length : 0 },
    { id: 'report',   label: t('panel.report'),   count: null },
    { id: 'chat',     label: t('panel.askAi'),    count: null },
    { id: 'info',     label: t('panel.details'),  count: null },
  ] as const;

  return (
    <div className="h-full flex flex-col bg-ink-900 border-l border-ink-800">
      {/* Tab Bar */}
      <div className="flex border-b border-ink-800 bg-ink-950/50 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setRightPanelTab(tab.id)}
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
        {rightPanelTab === 'findings' && <FindingsTab result={result} lang={lang} />}
        {rightPanelTab === 'report' && <ReportTab result={result} study={study} lang={lang} />}
        {rightPanelTab === 'chat' && <AIChat />}
        {rightPanelTab === 'info' && <InfoTab study={study} result={result} lang={lang} />}
        {rightPanelTab === 'compare' && <CompareTab />}
      </div>
    </div>
  );
}

// ========================================================================
// FINDINGS TAB — server result, status + urgency badges, model identity
// ========================================================================
function FindingsTab({ result, lang }: { result: AIResult | null; lang: Lang }) {
  const { selectFinding, selectedFinding } = useAppStore();
  const t = useT();

  // Flagged findings first, then by confidence. Presentation order only — urgency never comes from the score.
  const ordered = useMemo(() => {
    if (!result) return [] as Finding[];
    return [...result.findings].sort((a, b) => (Number(b.positive) - Number(a.positive)) || (b.confidence - a.confidence));
  }, [result]);

  if (!result) {
    return (
      <EmptyState
        icon="⚕"
        title={t('findings.noAnalysisYet')}
        description={t('findings.selectStudy')}
      />
    );
  }

  if (result.rejected || result.requiresReview) {
    return <NotAnalyzedState result={result} />;
  }

  const overall = result.overallAssessment;
  const positives = ordered.filter((f) => f.positive);
  const flagged = overall ? overall.abnormalFlagged : positives.length > 0;
  const flaggedCount = positives.length;
  // The body sentence is composed on the client in the UI language — the
  // server's overall text is English and stays available as technical detail.
  const flaggedNames = Array.from(new Set(
    (positives.length > 0 ? positives.map((f) => f.className) : (overall?.flags || []))
      .map((name) => translateFinding(name, lang)),
  ));
  const overallSummary = flagged
    ? t('findings.flaggedFor', { findings: flaggedNames.join('; ') || '—' })
    : t('findings.noFlagNotNormal');
  const serverText = overall?.text || result.overallImpression || '';

  return (
    <div className="p-3 space-y-3">
      <DisclaimerBanner text={t('findings.disclaimer')} />

      {/* Overall assessment — the server's rule; never a "normal" certificate */}
      <div className={`p-3 rounded-lg border ${flagged ? 'severity-critical' : 'border-ink-700 bg-ink-850/60 text-ink-200'}`}>
        <div className="text-[10px] font-bold uppercase tracking-wider mb-1 opacity-80">{t('findings.overallAssessment')}</div>
        <div className="text-sm font-semibold leading-tight">
          {flagged ? t('findings.abnormalFlagged') : t('findings.noFindingFlagged')}
        </div>
        <p className="text-xs mt-1 opacity-90">{overallSummary}</p>
        {serverText && (
          <details className="mt-1.5 text-[10px] opacity-70">
            <summary className="cursor-pointer select-none">{t('findings.technicalDetail')}</summary>
            <p className="mt-1 font-mono break-words" lang="en">{serverText}</p>
          </details>
        )}
      </div>

      {/* Stats grid — real numbers from the result only */}
      <div className="grid grid-cols-3 gap-2">
        <StatCard label={t('findings.flagged')} value={`${flaggedCount} / ${ordered.length}`} />
        <StatCard label={t('findings.time')} value={result.inferenceTimeMs ? `${result.inferenceTimeMs} ms` : '—'} />
        <StatCard label={t('findings.threshold')} value={result.threshold != null ? result.threshold.toFixed(2) : '—'} />
      </div>

      {/* Findings list */}
      <div className="space-y-2">
        <div className="px-1 text-[10px] font-semibold text-ink-500 uppercase tracking-wider">
          {t('findings.detected')}
        </div>
        {ordered.length === 0 && (
          <div className="p-3 rounded-lg border border-ink-800 bg-ink-850/50 text-xs text-ink-400">
            {t('findings.noFindingFlagged')}
          </div>
        )}
        {ordered.map((finding, idx) => (
          <FindingCard
            key={`${finding.detector || ''}:${finding.className}:${idx}`}
            finding={finding}
            rank={idx + 1}
            lang={lang}
            threshold={result.threshold}
            isSelected={isSameFinding(selectedFinding, finding)}
            onClick={() => selectFinding(isSameFinding(selectedFinding, finding) ? null : finding)}
          />
        ))}
      </div>

      <ModelIdentityList
        models={result.modelIdentity}
        fallback={result.modelVersion}
        appVersion={result.appVersion}
        lang={lang}
      />
    </div>
  );
}

function NotAnalyzedState({ result }: { result: AIResult }) {
  const t = useT();
  const reason = result.rejectionReason || '';
  return (
    <div className="p-4 space-y-3">
      <div className="p-4 rounded-lg border severity-moderate">
        <div className="text-sm font-semibold leading-snug">
          {t('findings.notAnalyzed')}{reason ? `: ${reason}` : ''}
        </div>
      </div>
      <DisclaimerBanner text={t('findings.disclaimer')} />
    </div>
  );
}

function DisclaimerBanner({ text }: { text: string }) {
  const t = useT();
  return (
    <div className="px-3 py-2 rounded-lg border border-moderate/40 bg-moderate/10 text-[11px] text-ink-200 leading-snug">
      {text || t('findings.intendedUse')}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-2 bg-ink-850 rounded-lg border border-ink-800">
      <div className="text-[9px] text-ink-500 uppercase tracking-wider mb-0.5">{label}</div>
      <div className="text-sm font-bold text-ink-100 font-mono">{value}</div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-ink-500 mb-0.5">{label}</div>
      <div className="font-mono text-ink-200 break-all">{value}</div>
    </div>
  );
}

function FindingCard({ finding, rank, lang, threshold, isSelected, onClick }: {
  finding: Finding; rank: number; lang: Lang; threshold: number | null; isSelected: boolean; onClick: () => void;
}) {
  const t = useT();
  const confidence = Math.round(finding.confidence * 100);
  const urgency = findingUrgency(finding);
  const name = translateFinding(finding.className, lang);
  const location = translateLocation(finding.location || '', lang);
  const barColor = urgency === 'review' ? 'bg-critical' : urgency === 'unvalidated' ? 'bg-moderate' : 'bg-ink-500';
  const thresholdPct = threshold != null ? Math.round(threshold * 100) : null;

  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border cursor-pointer transition-all ${
        isSelected
          ? 'border-accent-500/40 bg-gradient-to-r from-ink-800/80 to-ink-850'
          : 'border-ink-800 bg-ink-850/50 hover:bg-ink-800/70 hover:border-ink-700'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold font-mono bg-ink-800 text-ink-300">
            {rank}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-ink-100 truncate" title={finding.className}>{name}</div>
            {location && <div className="text-[10px] text-ink-500 truncate">{location}</div>}
          </div>
        </div>
        <div className="text-right ml-2 flex-shrink-0">
          <div className={`text-base font-bold font-mono ${finding.positive ? 'text-ink-100' : 'text-ink-500'}`}>{confidence}%</div>
          <div className="text-[9px] text-ink-500 uppercase tracking-wider">{t('findings.conf')}</div>
        </div>
      </div>

      {/* Badges: urgency (rule-based) + validation status of the detector */}
      <div className="flex flex-wrap items-center gap-1 mb-2">
        <span className={`severity-pill text-[9px] ${urgencyPillClass(urgency)}`}>{urgencyWord(urgency, lang)}</span>
        <span className={`severity-pill text-[9px] ${statusPillClass(finding.status)}`}>{statusWord(finding.status, lang)}</span>
      </div>

      {/* Confidence visualization with the real decision threshold */}
      <div className="relative w-full h-1.5 bg-ink-950 rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 ${barColor} transition-all duration-700`}
          style={{ width: `${confidence}%` }}
        />
        {thresholdPct != null && (
          <div
            className="absolute inset-y-0 w-px bg-ink-300"
            style={{ left: `${thresholdPct}%` }}
            title={`${t('findings.threshold')} ${thresholdPct}%`}
          />
        )}
      </div>

      {/* Metadata — only what the server reported */}
      {isSelected && (
        <div className="mt-3 pt-3 border-t border-ink-800 grid grid-cols-2 gap-2 text-[10px] animate-fade-in">
          <Meta label={t('findings.detector')} value={finding.detector || '—'} />
          <Meta label={t('findings.sequence')} value={finding.sequenceUsed || '—'} />
          <Meta label={t('findings.status')} value={statusWord(finding.status, lang)} />
          <Meta label={t('findings.class')} value={finding.className} />
          {thresholdPct != null && <Meta label={t('findings.threshold')} value={`${thresholdPct}%`} />}
        </div>
      )}
    </div>
  );
}

function ModelIdentityList({ models, fallback, appVersion, lang }: {
  models: ModelIdentity[]; fallback: string; appVersion: string | null; lang: Lang;
}) {
  const t = useT();
  return (
    <div className="pt-3 border-t border-ink-800">
      <div className="px-1 text-[10px] font-semibold text-ink-500 uppercase tracking-wider mb-2">{t('findings.models')}</div>
      {models.length === 0 ? (
        <div className="px-1 text-[10px] text-ink-500 font-mono">{fallback || '—'}</div>
      ) : (
        <div className="space-y-1">
          {models.map((m) => (
            <div key={m.key || m.displayName} className="px-2 py-1.5 rounded-md bg-ink-850/60 border border-ink-800 flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <div className="text-[11px] text-ink-200 truncate" title={m.source || undefined}>{m.displayName}</div>
                <div className="text-[9px] text-ink-500 font-mono">{m.sha256_12 || '—'}</div>
                {m.validationNote && <div className="text-[9px] text-ink-500 mt-0.5 leading-snug">{m.validationNote}</div>}
              </div>
              <span className={`severity-pill text-[9px] flex-shrink-0 ${statusPillClass(m.status)}`}>{statusWord(m.status, lang)}</span>
            </div>
          ))}
        </div>
      )}
      {appVersion && (
        <div className="px-1 mt-2 text-[9px] text-ink-600 font-mono">{t('findings.aiServerVersion')}: {appVersion}</div>
      )}
    </div>
  );
}

// ========================================================================
// REPORT TAB — real patient fields, server-side signing, honest exports
// ========================================================================
interface ReportCache {
  studyId: string;
  /** Current (possibly doctor-edited) text per language */
  texts: Partial<Record<Lang, string>>;
  /** Untouched AI draft per language — sent to /report/sign as ai_draft_text */
  drafts: Partial<Record<Lang, string>>;
}
const EMPTY_CACHE: ReportCache = { studyId: '', texts: {}, drafts: {} };

function ReportTab({ result, study, lang }: { result: AIResult | null; study: Study | null; lang: Lang }) {
  const { currentUser, settings, reports, signedReports, setSignedReport, addNotification, exportRequest } = useAppStore();
  const t = useT();
  const studyId = study?.id || '';
  const storedReport = reports[studyId];

  // Cache is keyed by study so switching studies never shows another patient's text.
  const [cache, setCache] = useState<ReportCache>(EMPTY_CACHE);
  const texts = cache.studyId === studyId ? cache.texts : {};
  const drafts = cache.studyId === studyId ? cache.drafts : {};

  const [language, setLanguage] = useState<Lang>(settings.language);
  // Follow the global interface language (top bar switcher) — it is the default report language.
  useEffect(() => { setLanguage(settings.language); }, [settings.language]);
  const [editMode, setEditMode] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [regenError, setRegenError] = useState<string | null>(null);
  const [exportMode, setExportMode] = useState<'current' | 'opened'>('current');
  const [exporting, setExporting] = useState(false);
  const [signing, setSigning] = useState(false);

  useEffect(() => { setEditMode(false); setRegenError(null); }, [studyId]);

  const signedFor = (l: Lang): SignedReport | undefined => signedReports[signedKey(studyId, l)];
  const signed = signedFor(language);

  // The language the report was originally produced in; every other language is an auto-translation.
  const originalLang: Lang = storedReport?.language || result?.reportLanguage || settings.language;

  const setText = (l: Lang, text: string) => setCache((c) => {
    const base = c.studyId === studyId ? c : { studyId, texts: {}, drafts: {} };
    return { studyId, texts: { ...base.texts, [l]: text }, drafts: base.drafts };
  });
  const seedText = (l: Lang, text: string, draft: string) => setCache((c) => {
    const base = c.studyId === studyId ? c : { studyId, texts: {}, drafts: {} };
    if (base.texts[l]) return base;
    return { studyId, texts: { ...base.texts, [l]: text }, drafts: { ...base.drafts, [l]: draft } };
  });

  // Seed from the server-side report (received with /analyze/study or restored via GET /study/{id})
  // and from any server-confirmed signatures.
  useEffect(() => {
    if (!studyId) return;
    if (storedReport?.reportText) {
      seedText(storedReport.language, storedReport.reportText, storedReport.aiDraftText || storedReport.reportText);
    }
    for (const l of LANGS) {
      const s = signedReports[signedKey(studyId, l)];
      if (s?.reportText) seedText(l, s.reportText, s.reportText);
    }
  }, [studyId, storedReport, signedReports]);

  // Generate the report for the selected language when it has not been opened yet.
  const cachedText = texts[language];
  useEffect(() => {
    if (!result || !study || !studyId) return;
    if (result.rejected || result.requiresReview) return;
    if (signed || cachedText) return;
    if (storedReport?.reportText && storedReport.language === language) return; // seeded above

    let cancelled = false;
    setRegenError(null);
    setRegenerating(true);
    const modality = result.modality || study.modality;
    const bodyPart = result.bodyPart || study.bodyPart;

    (async () => {
      try {
        const findingsForApi = result.findings.map((f) => ({
          class_name: f.className,
          confidence: f.confidence,
          location: f.location,
          positive: f.positive,
          status: f.status,
          detector: f.detector,
        }));
        // The server strips PHI from LLM prompts; the id is only used for its own bookkeeping.
        const text = await regenerateReport(
          findingsForApi,
          language,
          { id: study.patientId, age: study.patientAge ?? undefined, sex: study.patientSex ?? undefined, study_date: study.studyDate },
          modality,
          bodyPart,
        );
        if (cancelled) return;
        if (!text || !text.trim()) throw new Error('Empty report from server');
        seedText(language, text, text);
      } catch (e) {
        if (cancelled) return;
        const info = describeApiError(e);
        console.warn('Report generation failed — using local template:', info.code, info.detail);
        const template = generateReport(result, language, modality, bodyPart);
        const fullText = formatFullReport(template, study.patientId, study.studyDate || '—', currentUser?.fullName || '', language);
        seedText(language, fullText, fullText);
        setRegenError(t('health.llmUnavailable'));
      } finally {
        if (!cancelled) setRegenerating(false);
      }
    })();

    return () => { cancelled = true; setRegenerating(false); };
  }, [result, studyId, language, cachedText, signed, storedReport]);

  const reportText = signed ? signed.reportText : (cachedText || '');
  const draftText = drafts[language] || '';
  const edited = !signed && !!draftText && reportText !== draftText;
  const notAnalyzed = !!result && (result.rejected || result.requiresReview);

  // ----- PDF export ----------------------------------------------------------
  const sectionFor = (l: Lang): PDFLanguageSection | null => {
    const s = signedFor(l);
    const text = s ? s.reportText : texts[l];
    if (!text) return null;
    return {
      reportText: text,
      doctorName: s ? (s.signer.fullName || s.signer.username) : (currentUser?.fullName || ''),
      signedAt: s?.signedAt,
      sha256: s?.sha256,
      // A signed translation was reviewed by the signer; only unsigned ones carry the tag.
      autoTranslated: !s && l !== originalLang,
    };
  };
  const openedLangs = LANGS.filter((l) => !!sectionFor(l));
  const allOpenedSigned = openedLangs.length > 0 && openedLangs.every((l) => !!signedFor(l));
  const canExport = !!result && !notAnalyzed && !!study && (exportMode === 'opened' ? openedLangs.length > 0 : !!reportText);

  const buildPdfBase = async (): Promise<PDFBaseData> => ({
    patientId: study!.patientId,
    patientName: study!.patientName,
    studyDate: study!.studyDate || '—',
    accessionNumber: study!.accessionNumber,
    studyInstanceUid: study!.studyInstanceUid,
    modality: result!.modality || study!.modality,
    bodyPart: result!.bodyPart || study!.bodyPart,
    clinicName: settings.clinicName || t('login.clinicPlaceholder'),
    clinicAddress: [settings.clinicAddress, settings.clinicPhone].filter(Boolean).join(' · '),
    findings: result!.findings,
    modelIdentity: result!.modelIdentity,
    threshold: result!.threshold,
    appVersion: await getAppVersion(),
    disclaimer: result!.disclaimer,
  });

  // <studyInstanceUid last 12 chars | patientId>_<studyDate>_<lang>[_DRAFT].pdf
  const pdfFilename = (langPart: string, draft: boolean) => {
    const uid = study?.studyInstanceUid || '';
    const idPart = uid ? uid.slice(-12) : study?.patientId || 'study';
    return `${sanitizeFilePart(idPart)}_${sanitizeFilePart(study?.studyDate || 'nodate')}_${langPart}${draft ? '_DRAFT' : ''}.pdf`;
  };

  const handleExport = async () => {
    if (exporting || !canExport) return;
    setExporting(true);
    try {
      const base = await buildPdfBase();
      let res;
      if (exportMode === 'opened') {
        const sections: Partial<Record<Lang, PDFLanguageSection>> = {};
        for (const l of openedLangs) sections[l] = sectionFor(l)!;
        res = await downloadMultiLanguagePDF(base, sections, pdfFilename(openedLangs.join('-'), !allOpenedSigned));
      } else {
        const section = sectionFor(language);
        if (!section) return;
        res = await downloadPDF({ ...base, ...section, language }, pdfFilename(language, !signed));
      }
      if (res.success) addNotification({ type: 'success', title: t('report.pdfSaved'), message: res.filePath });
      // Signed single-language export → file the PDF on the server / PACS (Encapsulated PDF).
      if (res.success && exportMode !== 'opened' && signed && res.filePath) {
        const sr = signedFor(language);
        const bridge = (window as any).electronAPI;
        if (sr?.reportId && bridge?.readFileBase64) {
          try {
            const b64 = await bridge.readFileBase64(res.filePath);
            if (b64) {
              const r = await attachReportPdf(sr.reportId, b64, true);
              addNotification(r.pushed
                ? { type: 'success', title: t('report.sentToPacs') }
                : { type: 'warning', title: t('report.pdfStoredOnServer'), message: r.error || undefined });
            }
          } catch (e) {
            console.error('PACS filing failed:', e);
            addNotification({ type: 'warning', title: t('report.pacsFailed') });
          }
        }
      }
      else if (!res.canceled) addNotification({ type: 'error', title: t('report.exportFailed'), message: res.error });
    } catch (e) {
      console.error('PDF export failed:', e);
      addNotification({ type: 'error', title: t('report.exportFailed') });
    } finally {
      setExporting(false);
    }
  };

  // File → Export PDF / ⌘P / palette: run the same export as the button.
  const seenExportRequest = useRef(exportRequest);
  useEffect(() => {
    if (exportRequest === seenExportRequest.current) return;
    seenExportRequest.current = exportRequest;
    if (!result) { addNotification({ type: 'warning', title: t('report.noReportYet'), message: t('report.selectStudy') }); return; }
    if (!canExport) { addNotification({ type: 'warning', title: t('report.noReportYet') }); return; }
    handleExport();
  }, [exportRequest]);

  if (!result) {
    return <EmptyState icon="📄" title={t('report.noReportYet')} description={t('report.selectStudy')} />;
  }
  if (notAnalyzed || !study) {
    return <NotAnalyzedState result={result} />;
  }

  const handleToggleEdit = () => {
    // Leaving edit mode with changes: keep the correction for the fine-tuning set (best effort).
    if (editMode && edited && studyId) {
      saveReportCorrection(studyId, draftText, reportText, language).catch((e) => {
        console.warn('save_correction failed:', describeApiError(e).detail);
      });
    }
    setEditMode(!editMode);
  };

  const handleSign = async () => {
    if (!reportText || signing) return;
    setSigning(true);
    try {
      const s = await signReport(study.id, reportText, language, draftText || reportText);
      setSignedReport(study.id, language, s);
      setEditMode(false);
      addNotification({
        type: 'success',
        title: t('report.signedLocked'),
        message: `${s.signer.fullName || s.signer.username} · ${s.sha256.slice(0, 12)}`,
      });
    } catch (e) {
      const info = describeApiError(e);
      console.error('Sign failed:', info.code, info.detail);
      if (info.code === 'conflict') addNotification({ type: 'warning', title: t('report.alreadySigned') });
      else if (info.code === 'forbidden') addNotification({ type: 'warning', title: t('report.noPermissionSign') });
      else if (info.code === 'network') addNotification({ type: 'error', title: t('health.aiServerDown') });
      else if (info.code === 'auth') addNotification({ type: 'warning', title: t('health.sessionExpired') });
      else addNotification({ type: 'error', title: t('report.signFailed'), message: info.detail });
    } finally {
      setSigning(false);
    }
  };

  const exportLabel = exportMode === 'opened'
    ? `${allOpenedSigned ? t('report.exportSigned') : t('report.exportDraft')} (${openedLangs.length})`
    : (signed ? t('report.exportSigned') : t('report.exportDraft'));

  return (
    <div className="flex flex-col h-full">
      {/* Language + edit toolbar */}
      <div className="p-2 border-b border-ink-800 flex items-center justify-between bg-ink-950/30">
        <div className="flex items-center gap-1">
          {LANGS.map((l) => (
            <button
              key={l}
              onClick={() => setLanguage(l)}
              disabled={regenerating}
              className={`px-2.5 py-1 text-[10px] font-bold rounded transition-colors disabled:opacity-50 ${
                language === l ? 'bg-accent-600 text-white' : 'bg-ink-800 text-ink-400 hover:text-ink-200'
              }`}
              title={signedFor(l) ? t('report.signedLocked') : (texts[l] ? t('report.draft') : '')}
            >
              {langShort(l)}
              {signedFor(l) ? (
                <span className="ml-1 text-[8px] text-normal">✓</span>
              ) : texts[l] ? (
                <span className="ml-1 text-[8px] opacity-70">●</span>
              ) : null}
            </button>
          ))}
        </div>
        {!signed && (
          <button
            onClick={handleToggleEdit}
            className={`p-1 rounded transition-colors ${editMode ? 'bg-accent-600 text-white' : 'text-ink-500 hover:text-ink-200'}`}
            title={t('report.edit')}
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
        )}
      </div>

      {/* Status strip */}
      {(regenerating || regenError || edited || (!signed && reportText && language !== originalLang)) && (
        <div className="px-3 py-1.5 text-[10px] flex flex-col gap-0.5 bg-ink-950/40 border-b border-ink-800">
          {regenerating && (
            <span className="flex items-center gap-2 text-accent-300">
              <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              {t('report.generating')} · {langShort(language)}…
            </span>
          )}
          {regenError && <span className="text-yellow-300">⚠ {regenError}</span>}
          {!signed && reportText && language !== originalLang && (
            <span className="text-moderate">⚠ {t('report.autoTranslated')}</span>
          )}
          {edited && <span className="text-ink-400">{t('report.draftEdited')}</span>}
        </div>
      )}

      {/* Report content */}
      <div className="flex-1 overflow-hidden p-3 relative">
        <div className={`h-full rounded-lg border transition-colors ${
          signed
            ? 'border-normal/40 bg-normal/5'
            : editMode
              ? 'border-accent-500/40 bg-ink-900'
              : 'border-ink-800 bg-ink-900'
        }`}>
          <textarea
            value={reportText}
            onChange={(e) => editMode && !signed && setText(language, e.target.value)}
            readOnly={!editMode || !!signed}
            placeholder={regenerating ? '…' : ''}
            className="w-full h-full p-3 text-xs leading-relaxed font-mono bg-transparent resize-none focus:outline-none text-ink-200"
          />
        </div>
        {/* Draft watermark — the same text the PDF banner prints */}
        {!signed && reportText && (
          <div className="absolute top-4 right-5 pointer-events-none select-none px-2 py-0.5 rounded border border-critical/40 bg-ink-950/70 text-[9px] font-bold tracking-wider text-critical/80 uppercase rotate-[-3deg]">
            {t('report.draftWatermark')}
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="p-3 border-t border-ink-800 space-y-2 bg-ink-950/30">
        {/* Export scope: this language, or every language the doctor opened */}
        <div className="flex items-center gap-1 p-1 bg-ink-950 rounded-md">
          <button
            onClick={() => setExportMode('current')}
            className={`flex-1 py-1 text-[10px] font-medium rounded transition-colors ${
              exportMode === 'current' ? 'bg-ink-700 text-white' : 'text-ink-500 hover:text-ink-300'
            }`}
          >
            {t('report.currentLanguage')}
          </button>
          <button
            onClick={() => setExportMode('opened')}
            className={`flex-1 py-1 text-[10px] font-medium rounded transition-colors ${
              exportMode === 'opened' ? 'bg-ink-700 text-white' : 'text-ink-500 hover:text-ink-300'
            }`}
          >
            {t('report.openedLanguages')} ({openedLangs.length})
          </button>
        </div>

        {signed ? (
          <>
            <div className="p-2.5 bg-normal/10 border border-normal/30 rounded-lg flex items-start gap-2">
              <div className="w-8 h-8 rounded-full bg-normal/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-normal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-normal">{t('report.signedLocked')}</div>
                <div className="text-[10px] text-ink-300 truncate">
                  {t('report.signedBy')}: {signed.signer.fullName || signed.signer.username} · {formatDateTime(signed.signedAt, lang)}
                </div>
                <div className="text-[9px] text-ink-500 font-mono break-all" title={signed.sha256}>{t('report.sha')}: {signed.sha256}</div>
              </div>
            </div>
            <button onClick={handleExport} disabled={exporting || !canExport} className="w-full btn-primary py-2 disabled:opacity-50">
              {exporting ? t('report.exporting') : exportLabel}
            </button>
          </>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={handleSign}
              disabled={!reportText || regenerating || signing}
              className="btn-primary py-2 disabled:opacity-50"
            >
              {signing ? t('report.signing') : `✓ ${t('report.sign')}`}
            </button>
            <button
              onClick={handleExport}
              disabled={!canExport || regenerating || exporting}
              className="btn-secondary py-2 disabled:opacity-50"
            >
              {exporting ? t('report.exporting') : exportLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ========================================================================
// INFO TAB — real DICOM header fields from the Study (nulls hidden)
// ========================================================================
function InfoTab({ study, result, lang }: { study: Study | null; result: AIResult | null; lang: Lang }) {
  const { health } = useAppStore();
  const t = useT();

  if (!study) {
    return <EmptyState icon="ⓘ" title={t('details.noStudySelected')} description={t('details.selectStudy')} />;
  }

  type Row = [string, string];
  const row = (label: string, value: unknown): Row | null =>
    value === null || value === undefined || value === '' ? null : [label, String(value)];
  const compact = (rows: (Row | null)[]): Row[] => rows.filter((r): r is Row => !!r);

  const sections = [
    {
      title: t('details.patient'),
      items: compact([
        row(t('details.patientId'), study.patientId),
        row(t('details.patientName'), study.patientName),
        row(t('details.sex'), study.patientSex),
        row(t('details.age'), study.patientAge),
        row(t('details.birthDate'), study.patientBirthDate ? formatDate(study.patientBirthDate, lang) : null),
      ]),
    },
    {
      title: t('details.study'),
      items: compact([
        row(t('details.studyUid'), study.studyInstanceUid),
        row(t('details.accession'), study.accessionNumber),
        row(t('details.studyDate'), study.studyDate ? formatDate(study.studyDate, lang) : null),
        row(t('details.modality'), study.modality !== 'N/A' ? study.modality : null),
        row(t('details.bodyPart'), study.bodyPart !== '—' ? study.bodyPart : null),
        row(t('details.description'), study.studyDescription),
        row(t('details.filesCount'), study.numFiles),
        row(t('details.series'), study.seriesDescriptions?.length ? study.seriesDescriptions.join(', ') : null),
      ]),
    },
    {
      title: t('details.equipment'),
      items: compact([
        row(t('details.manufacturer'), study.manufacturer),
        row(t('details.scannerModel'), study.scannerModel),
      ]),
    },
    {
      title: t('details.aiPipeline'),
      items: compact([
        ...(result?.modelIdentity || []).map((m) => row(m.displayName, `${m.sha256_12 || '—'} · ${statusWord(m.status, lang)}`)),
        row(t('details.threshold'), result?.threshold != null ? result.threshold.toFixed(2) : null),
        row(t('details.inferenceTime'), result?.inferenceTimeMs ? `${result.inferenceTimeMs} ms` : null),
        row(t('details.analyzedAt'), formatDateTime(result?.createdAt || study.aiAnalyzedAt, lang) || null),
        row(t('details.device'), health?.device),
        row(t('findings.aiServerVersion'), result?.appVersion || health?.version),
      ]),
    },
  ].filter((s) => s.items.length > 0);

  return (
    <div className="p-3 space-y-4">
      {sections.map((section) => (
        <div key={section.title}>
          <div className="text-[10px] font-bold text-ink-500 uppercase tracking-wider mb-2 px-1">
            {section.title}
          </div>
          <div className="space-y-1 bg-ink-850/50 rounded-lg border border-ink-800 overflow-hidden">
            {section.items.map(([k, v], idx) => (
              <div key={`${k}-${idx}`} className={`flex items-start justify-between px-3 py-2 text-[11px] ${idx > 0 ? 'border-t border-ink-800' : ''}`}>
                <span className="text-ink-500 flex-shrink-0">{k}</span>
                <span className="text-ink-100 font-mono text-right ml-2 break-all">{v}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ========================================================================
// COMPARE TAB — no prior-study / similar-case data source exists in this build
// ========================================================================
function CompareTab() {
  const t = useT();
  return <EmptyState icon="⇄" title={t('panel.compare')} description={t('details.compareUnavailable')} />;
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
