import React, { useState, useMemo, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import type { Study, AIResult, Lang } from '../types';
import { analyzeStudy, describeApiError } from '../services/api';
import { DEMO_MODE, DEMO_STUDIES } from '../services/demoData';
import { findingUrgency, translateFinding, statusWord } from '../services/findingTranslations';
import { useT, formatDate, formatTime } from '../i18n';
import type { TFn, I18nKey } from '../i18n';

type SortKey = 'time' | 'patient' | 'modality' | 'status';

const SORT_KEYS: Record<SortKey, I18nKey> = {
  time: 'worklist.sortTime',
  patient: 'worklist.sortPatient',
  modality: 'worklist.sortModality',
  status: 'worklist.sortStatus',
};

interface UploadProgress {
  phase: 'idle' | 'collecting' | 'uploading' | 'analyzing';
  count: number;
  percent: number;
}

// ─── File collection helpers ────────────────────────────────────────────────

/** Client-side filter: .dcm / .dicom / extensionless files (skips hidden files and DICOMDIR). */
export function isDicomCandidate(name: string): boolean {
  const base = (name || '').split('/').pop() || '';
  if (!base || base.startsWith('.')) return false;
  if (base.toUpperCase() === 'DICOMDIR') return false;
  const dot = base.lastIndexOf('.');
  if (dot === -1) return true;
  const ext = base.slice(dot + 1).toLowerCase();
  return ext === 'dcm' || ext === 'dicom';
}

function filePath(f: File): string {
  return (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
}

function filterAndSort(files: File[]): File[] {
  return files
    .filter((f) => isDicomCandidate(filePath(f)))
    .sort((a, b) => filePath(a).localeCompare(filePath(b), undefined, { numeric: true }));
}

/** Recursively read a dropped FileSystemEntry (folder drops). */
async function readEntryFiles(entry: any): Promise<File[]> {
  if (!entry) return [];
  if (entry.isFile) {
    return new Promise<File[]>((resolve) => entry.file((f: File) => resolve([f]), () => resolve([])));
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const out: File[] = [];
    const readBatch = () => new Promise<any[]>((resolve) => reader.readEntries(resolve, () => resolve([])));
    let batch = await readBatch();
    while (batch.length > 0) {
      for (const child of batch) out.push(...(await readEntryFiles(child)));
      batch = await readBatch();
    }
    return out;
  }
  return [];
}

async function collectDroppedFiles(dt: DataTransfer): Promise<File[]> {
  const items = dt.items ? Array.from(dt.items) : [];
  const entries = items
    .map((it) => (typeof (it as any).webkitGetAsEntry === 'function' ? (it as any).webkitGetAsEntry() : null))
    .filter(Boolean);
  if (entries.length > 0) {
    const nested = await Promise.all(entries.map(readEntryFiles));
    return nested.flat();
  }
  return Array.from(dt.files || []);
}

/** Placeholder result for a 422 requires_review answer (no study fields are returned). */
function reviewOnlyResult(studyId: string, reason: string): AIResult {
  const now = new Date().toISOString();
  return {
    id: studyId, studyId, findings: [], inferenceTimeMs: 0, modelVersion: '', isNormal: false,
    overallImpression: '', createdAt: now, overallAssessment: null, disclaimer: '', modelIdentity: [],
    threshold: null, requiresReview: true, rejected: true, rejectionReason: reason, appVersion: null,
  };
}

export default function Sidebar() {
  const {
    studies, selectedStudyId, selectStudy, modalityFilter, setModalityFilter, searchQuery, setSearchQuery,
    addStudy, setAIResult, setReport, settings, health, addNotification, aiResults, uploadRequest,
  } = useAppStore();
  const t = useT();
  const lang = settings.language;
  const [sortKey, setSortKey] = useState<SortKey>('time');
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const [progress, setProgress] = useState<UploadProgress>({ phase: 'idle', count: 0, percent: 0 });
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const folderInputRef = React.useRef<HTMLInputElement>(null);

  const uploading = progress.phase !== 'idle';
  const serverReady = !!health && health.reachable && health.status === 'ok';

  // Menu / palette / ⌘O requests open the same hidden inputs as the buttons.
  useEffect(() => {
    if (!uploadRequest || uploading) return;
    if (!serverReady) { addNotification({ type: 'error', title: t('health.aiServerDown') }); return; }
    (uploadRequest.kind === 'folder' ? folderInputRef : fileInputRef).current?.click();
  }, [uploadRequest?.nonce]);

  const handleUploadFiles = async (rawFiles: File[]) => {
    const files = filterAndSort(rawFiles);
    if (files.length === 0) {
      addNotification({ type: 'warning', title: t('worklist.noDicomFiles') });
      return;
    }
    setProgress({ phase: 'uploading', count: files.length, percent: 0 });
    try {
      const analysis = await analyzeStudy(files, settings.language, (pct) => {
        setProgress({ phase: pct >= 100 ? 'analyzing' : 'uploading', count: files.length, percent: pct });
      });
      setProgress({ phase: 'analyzing', count: files.length, percent: 100 });

      // The Study is built from the SERVER response — never from the button or filename.
      const { study, result, reportText, reportLanguage } = analysis;
      addStudy(study);
      setAIResult(study.id, result);
      if (reportText) {
        const now = new Date().toISOString();
        setReport(study.id, {
          id: `rep-${study.id}`,
          studyId: study.id,
          doctorId: 'ai',
          reportText,
          aiDraftText: reportText,
          language: reportLanguage,
          isSigned: false,
          createdAt: now,
          updatedAt: now,
        });
      }
      selectStudy(study.id);
      if (result.rejected || result.requiresReview) {
        addNotification({ type: 'warning', title: t('worklist.notAnalyzed'), message: result.rejectionReason || undefined });
      }
    } catch (e) {
      const info = describeApiError(e);
      console.error('Upload/analysis failed:', info.code, info.detail, e);
      if (info.code === 'review') {
        // Server refused to analyze (wrong modality / body part) — keep the study visible for a full read.
        const studyId = `review-${Date.now()}`;
        const now = new Date().toISOString();
        const study: Study = {
          id: studyId, patientId: 'UNKNOWN', modality: 'N/A', bodyPart: '—', studyDate: '',
          receivedAt: now, dicomPath: '', aiStatus: 'complete', numFiles: files.length,
        };
        addStudy(study);
        setAIResult(studyId, reviewOnlyResult(studyId, info.detail));
        selectStudy(studyId);
        addNotification({ type: 'warning', title: t('worklist.notAnalyzed'), message: info.detail });
      } else if (info.code === 'network') {
        addNotification({ type: 'error', title: t('health.aiServerDown') });
      } else if (info.code === 'auth') {
        addNotification({ type: 'warning', title: t('health.sessionExpired') });
      } else {
        addNotification({ type: 'error', title: t('health.analysisFailed'), message: info.detail });
      }
    } finally {
      setProgress({ phase: 'idle', count: 0, percent: 0 });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files ? Array.from(e.target.files) : [];
    if (list.length > 0) handleUploadFiles(list);
    // Reset input so the same selection can be re-uploaded
    e.target.value = '';
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (uploading || !serverReady) return;
    setProgress({ phase: 'collecting', count: 0, percent: 0 });
    try {
      const files = await collectDroppedFiles(e.dataTransfer);
      await handleUploadFiles(files);
    } finally {
      setProgress((p) => (p.phase === 'collecting' ? { phase: 'idle', count: 0, percent: 0 } : p));
    }
  };

  const displayStudies = studies.length > 0 ? studies : (DEMO_MODE ? DEMO_STUDIES : []);

  const filtered = useMemo(() => {
    const result = displayStudies.filter((s) => {
      if (modalityFilter !== 'all' && s.modality !== modalityFilter) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        return s.patientId.toLowerCase().includes(q) ||
               (s.patientName || '').toLowerCase().includes(q) ||
               (s.accessionNumber || '').toLowerCase().includes(q) ||
               s.bodyPart.toLowerCase().includes(q) ||
               s.modality.toLowerCase().includes(q);
      }
      return true;
    });

    if (sortKey === 'time') result.sort((a, b) => b.receivedAt.localeCompare(a.receivedAt));
    if (sortKey === 'patient') result.sort((a, b) => a.patientId.localeCompare(b.patientId));
    if (sortKey === 'modality') result.sort((a, b) => a.modality.localeCompare(b.modality));
    if (sortKey === 'status') result.sort((a, b) => a.aiStatus.localeCompare(b.aiStatus));

    return result;
  }, [displayStudies, modalityFilter, searchQuery, sortKey]);

  const stats = useMemo(() => ({
    total: displayStudies.length,
    pending: displayStudies.filter((s) => s.aiStatus === 'pending').length,
    processing: displayStudies.filter((s) => s.aiStatus === 'processing').length,
    complete: displayStudies.filter((s) => s.aiStatus === 'complete').length,
    error: displayStudies.filter((s) => s.aiStatus === 'error').length,
  }), [displayStudies]);

  const modalities = useMemo(() => {
    const set = new Set(displayStudies.map((s) => s.modality));
    return ['all', ...Array.from(set).sort()];
  }, [displayStudies]);

  const progressLabel = (() => {
    if (progress.phase === 'collecting') return t('worklist.collecting');
    if (progress.phase === 'uploading') return t('worklist.uploadingCount', { count: progress.count, percent: progress.percent });
    if (progress.phase === 'analyzing') return t('worklist.analyzingCount', { count: progress.count });
    return '';
  })();

  return (
    <div className="h-full flex flex-col bg-ink-900 border-r border-ink-800">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b border-ink-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-ink-100 uppercase tracking-wider flex items-center gap-2">
            <svg className="w-4 h-4 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            {t('worklist.title')}
          </h2>

          {/* View mode toggle */}
          <div className="flex items-center bg-ink-800 rounded-md p-0.5">
            <button
              onClick={() => setViewMode('list')}
              className={`p-1 rounded transition-colors ${viewMode === 'list' ? 'bg-ink-700 text-accent-400' : 'text-ink-500 hover:text-ink-200'}`}
              title={t('worklist.listView')}
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1 rounded transition-colors ${viewMode === 'grid' ? 'bg-ink-700 text-accent-400' : 'text-ink-500 hover:text-ink-200'}`}
              title={t('worklist.gridView')}
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative mb-2">
          <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t('worklist.searchPlaceholder')}
            className="w-full pl-8 pr-3 py-1.5 text-xs bg-ink-950/50 border border-ink-700 rounded-md text-ink-100 placeholder-ink-500 focus:outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20"
          />
        </div>

        {/* Modality Filter Pills (real modalities only) */}
        <div className="flex gap-1 overflow-x-auto">
          {modalities.map((m) => {
            const count = m === 'all' ? displayStudies.length : displayStudies.filter((s) => s.modality === m).length;
            return (
              <button
                key={m}
                onClick={() => setModalityFilter(m)}
                className={`px-2 py-0.5 text-[10px] font-medium rounded whitespace-nowrap transition-colors ${
                  modalityFilter === m
                    ? 'bg-accent-600 text-white'
                    : 'bg-ink-800 text-ink-400 hover:text-ink-200 hover:bg-ink-700'
                }`}
              >
                {m === 'all' ? t('worklist.all') : m} <span className="opacity-60">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Stats strip */}
      <div className="px-3 py-2 border-b border-ink-800 grid grid-cols-4 gap-1">
        <StatChip label={t('worklist.total')} value={stats.total} color="text-ink-300" />
        <StatChip label={t('worklist.pending')} value={stats.pending} color="text-ink-400" />
        <StatChip label={t('worklist.analyzing')} value={stats.processing} color="text-accent-400" pulse={stats.processing > 0} />
        <StatChip label={t('worklist.done')} value={stats.complete} color="text-normal" />
      </div>

      {/* Sort bar */}
      <div className="px-3 py-1.5 border-b border-ink-800 flex items-center gap-2 text-[10px] text-ink-500">
        <span className="uppercase tracking-wider">{t('worklist.sort')}</span>
        {(['time', 'patient', 'modality', 'status'] as SortKey[]).map((k) => (
          <button
            key={k}
            onClick={() => setSortKey(k)}
            className={`capitalize ${sortKey === k ? 'text-accent-400 font-medium' : 'text-ink-500 hover:text-ink-300'}`}
          >
            {t(SORT_KEYS[k])}
          </button>
        ))}
      </div>

      {/* Study List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filtered.map((study) => (
          viewMode === 'list'
            ? <StudyCard key={study.id} study={study} result={aiResults[study.id]} lang={lang} t={t} selected={study.id === selectedStudyId} onClick={() => selectStudy(study.id)} />
            : <StudyThumb key={study.id} study={study} selected={study.id === selectedStudyId} onClick={() => selectStudy(study.id)} />
        ))}

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <svg className="w-12 h-12 text-ink-700 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm text-ink-400">{displayStudies.length === 0 ? t('worklist.noStudies') : t('worklist.noMatch')}</p>
            <p className="text-xs text-ink-600 mt-1">{displayStudies.length === 0 ? t('worklist.noStudiesHint') : t('worklist.noMatchHint')}</p>
          </div>
        )}
      </div>

      {/* Upload Section — whole study (folder or many files); the SERVER routes by DICOM headers */}
      <div
        className={`p-2 border-t border-ink-800 space-y-2 transition-colors ${dragOver ? 'bg-accent-900/20' : ''}`}
        onDragOver={(e) => { e.preventDefault(); if (!dragOver) setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".dcm,.dicom,application/dicom,*/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        />

        {uploading ? (
          <div className="w-full py-2 px-3 bg-ink-800 rounded-md text-xs text-accent-300">
            <div className="flex items-center gap-2 mb-1.5">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>{progressLabel}</span>
            </div>
            <div className="h-1 bg-ink-950 rounded-full overflow-hidden">
              <div
                className={`h-full bg-accent-500 transition-all ${progress.phase === 'analyzing' ? 'animate-pulse' : ''}`}
                style={{ width: `${progress.phase === 'analyzing' ? 100 : progress.percent}%` }}
              />
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-1">
            <button
              onClick={() => folderInputRef.current?.click()}
              disabled={!serverReady}
              title={serverReady ? t('worklist.uploadStudy') : t('health.aiServerDown')}
              className="py-2 bg-accent-600 hover:bg-accent-500 disabled:bg-ink-700 disabled:text-ink-500 disabled:cursor-not-allowed text-white text-xs font-medium rounded-md transition-all flex items-center justify-center gap-1.5 shadow-sm"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              {t('worklist.uploadFolder')}
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!serverReady}
              title={serverReady ? t('worklist.uploadStudy') : t('health.aiServerDown')}
              className="py-2 bg-ink-800 hover:bg-ink-700 disabled:bg-ink-800/50 disabled:text-ink-600 disabled:cursor-not-allowed text-ink-100 text-xs font-medium rounded-md transition-all flex items-center justify-center gap-1.5 border border-ink-700"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {t('worklist.uploadFiles')}
            </button>
          </div>
        )}

        <div className="text-[10px] text-ink-500 text-center">
          {serverReady ? t('worklist.dropHint') : t('health.aiServerDown')}
        </div>
      </div>
    </div>
  );
}

function StatChip({ label, value, color, pulse }: { label: string; value: number; color: string; pulse?: boolean }) {
  return (
    <div className="flex flex-col items-center py-1 bg-ink-950/50 rounded">
      <span className={`text-sm font-bold ${color} ${pulse ? 'animate-pulse' : ''}`}>{value}</span>
      <span className="text-[9px] text-ink-600 uppercase tracking-wider">{label}</span>
    </div>
  );
}

/** Worklist badge derived from the real AI result — never random. */
function resultBadge(result: AIResult | undefined, lang: Lang, t: TFn) {
  if (!result) return null;
  if (result.rejected || result.requiresReview) {
    return { label: t('worklist.needsReview'), cls: 'severity-moderate', top: null as null | { name: string; confidence: number } };
  }
  const positives = result.findings.filter((f) => f.positive);
  if (positives.length === 0) return null;
  const top = positives.reduce((a, b) => (b.confidence > a.confidence ? b : a));
  const urgency = findingUrgency(top);
  return {
    label: urgency === 'review' ? t('worklist.needsReview') : `${t('worklist.needsReview')} · ${statusWord(top.status, lang)}`,
    cls: urgency === 'review' ? 'severity-critical' : 'severity-moderate',
    top: { name: translateFinding(top.className, lang), confidence: top.confidence },
  };
}

const STATUS_KEYS: Record<Study['aiStatus'], I18nKey> = {
  pending: 'worklist.status.pending',
  processing: 'worklist.status.processing',
  complete: 'worklist.status.complete',
  error: 'worklist.status.error',
};

function StudyCard({ study, result, lang, t, selected, onClick }: { study: Study; result?: AIResult; lang: Lang; t: TFn; selected: boolean; onClick: () => void }) {
  const statusConfig = {
    pending: { color: 'bg-ink-500', textColor: 'text-ink-400' },
    processing: { color: 'bg-accent-500 animate-pulse', textColor: 'text-accent-400' },
    complete: { color: 'bg-normal', textColor: 'text-normal' },
    error: { color: 'bg-critical', textColor: 'text-critical' },
  }[study.aiStatus];

  const modalityColors: Record<string, string> = {
    CR: 'border-l-accent-500', DX: 'border-l-accent-500',
    CT: 'border-l-purple-500', MR: 'border-l-emerald-500',
    US: 'border-l-yellow-500',
  };

  const badge = resultBadge(result, lang, t);

  return (
    <div
      onClick={onClick}
      className={`px-3 py-2 rounded-lg border-l-2 ${modalityColors[study.modality] || 'border-l-ink-600'} cursor-pointer transition-all group ${
        selected
          ? 'bg-accent-900/20 border border-accent-500/40 shadow-sm'
          : 'bg-ink-850/40 hover:bg-ink-800/70 border border-transparent'
      }`}
    >
      <div className="flex items-start justify-between mb-1">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[10px] font-bold text-ink-300 bg-ink-800 px-1.5 py-0.5 rounded">{study.modality}</span>
          {badge && (
            <span className={`severity-pill ${badge.cls} truncate`}>{badge.label}</span>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <span className={`w-1.5 h-1.5 rounded-full ${statusConfig.color}`} />
          <span className={`text-[9px] font-medium uppercase tracking-wider ${statusConfig.textColor}`}>
            {t(STATUS_KEYS[study.aiStatus])}
          </span>
        </div>
      </div>

      <div className="text-sm font-semibold text-ink-100 font-mono mb-0.5 truncate">{study.patientId}</div>

      <div className="flex items-center justify-between text-[10px] text-ink-500">
        <span className="truncate">{study.bodyPart}{study.numFiles ? ` · ${study.numFiles} ${t('common.files')}` : ''}</span>
        <span className="font-mono flex-shrink-0">
          {study.studyDate ? formatDate(study.studyDate, lang) : formatTime(study.receivedAt, lang)}
        </span>
      </div>

      {badge?.top && (
        <div className="mt-1.5 flex items-center gap-1.5">
          <span className="text-[9px] text-ink-400 truncate flex-1">{badge.top.name}</span>
          <span className="text-[9px] text-ink-400 font-mono">{Math.round(badge.top.confidence * 100)}%</span>
        </div>
      )}
    </div>
  );
}

function StudyThumb({ study, selected, onClick }: { study: Study; selected: boolean; onClick: () => void }) {
  const statusConfig = {
    pending: 'bg-ink-500',
    processing: 'bg-accent-500 animate-pulse',
    complete: 'bg-normal',
    error: 'bg-critical',
  }[study.aiStatus];

  return (
    <div
      onClick={onClick}
      className={`aspect-square relative rounded-lg overflow-hidden cursor-pointer transition-all group ${
        selected ? 'ring-2 ring-accent-500' : 'hover:ring-1 hover:ring-ink-600'
      }`}
    >
      <div className="w-full h-full bg-gradient-to-br from-ink-800 to-ink-950 flex items-center justify-center">
        <span className="text-2xl font-bold text-ink-700">{study.modality}</span>
      </div>
      <div className="absolute inset-0 bg-gradient-to-t from-ink-950 via-transparent to-transparent p-1.5 flex flex-col justify-end">
        <div className="text-[10px] font-mono text-ink-100 truncate">{study.patientId}</div>
        <div className="text-[9px] text-ink-400 truncate">{study.bodyPart}</div>
      </div>
      <div className={`absolute top-1 right-1 w-2 h-2 rounded-full ${statusConfig}`} />
    </div>
  );
}
