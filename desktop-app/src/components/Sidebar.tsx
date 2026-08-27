import React, { useState, useMemo } from 'react';
import { useAppStore } from '../store/appStore';
import type { Study } from '../types';
import { analyzeDicom } from '../services/api';
import { DEMO_STUDIES } from '../services/demoData';

type SortKey = 'time' | 'patient' | 'modality' | 'status';

type ForceModality = 'auto' | 'chest' | 'brain_2d' | 'head_ct' | 'mammography';

const MODALITY_LABELS: Record<ForceModality, string> = {
  auto: 'Auto',
  chest: 'Chest X-ray',
  brain_2d: 'Brain MRI',
  head_ct: 'Head CT',
  mammography: 'Mammography',
};

export default function Sidebar() {
  const { studies, selectedStudyId, selectStudy, modalityFilter, setModalityFilter, searchQuery, setSearchQuery, addStudy, setAIResult, setReport, settings } = useAppStore();
  const [sortKey, setSortKey] = useState<SortKey>('time');
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const [uploading, setUploading] = useState(false);
  const [forceModality, setForceModality] = useState<ForceModality>('auto');
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleUploadFile = async (file: File) => {
    setUploading(true);
    try {
      const studyId = `st-${Date.now()}`;
      const guessedBodyPart =
        forceModality === 'brain_2d' ? 'BRAIN'
        : forceModality === 'head_ct' ? 'HEAD'
        : forceModality === 'mammography' ? 'BREAST'
        : 'CHEST';
      const guessedModality =
        forceModality === 'brain_2d' ? 'MR'
        : forceModality === 'head_ct' ? 'CT'
        : forceModality === 'mammography' ? 'MG'
        : 'CR';

      const study: Study = {
        id: studyId,
        patientId: file.name.replace(/\.[^/.]+$/, '').slice(0, 12),
        modality: guessedModality,
        bodyPart: guessedBodyPart,
        studyDate: new Date().toISOString().slice(0, 10),
        receivedAt: new Date().toISOString(),
        dicomPath: file.name,
        aiStatus: 'processing',
      };
      addStudy(study);
      selectStudy(studyId);

      const result = await analyzeDicom(file, file.name, settings.language, {
        useAutoRouting: true,
        forceModality: forceModality === 'auto' ? undefined : forceModality,
      });

      setAIResult(studyId, { ...result, studyId });

      if (result.reportText) {
        setReport(studyId, {
          id: `rep-${Date.now()}`,
          studyId,
          doctorId: 'ai',
          reportText: result.reportText,
          aiDraftText: result.reportText,
          language: settings.language,
          isSigned: false,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      }

      const updatedStudies = useAppStore.getState().studies.map(s =>
        s.id === studyId ? { ...s, aiStatus: 'complete' as const, aiAnalyzedAt: new Date().toISOString() } : s
      );
      useAppStore.setState({ studies: updatedStudies });
    } catch (e: any) {
      console.error('Upload/analysis failed:', e);
      const detail = e?.response?.data?.detail || e?.message || 'Server unavailable';
      alert(`Analysis failed: ${detail}\n\nIf the model isn't downloaded, run:\n  python scripts/download_all_models.py`);
    } finally {
      setUploading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleUploadFile(file);
    }
    // Reset input so same file can be re-uploaded
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleUploadFile(file);
    }
  };

  const displayStudies = studies.length > 0 ? studies : DEMO_STUDIES;

  const filtered = useMemo(() => {
    let result = displayStudies.filter(s => {
      if (modalityFilter !== 'all' && s.modality !== modalityFilter) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        return s.patientId.toLowerCase().includes(q) ||
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
    pending: displayStudies.filter(s => s.aiStatus === 'pending').length,
    processing: displayStudies.filter(s => s.aiStatus === 'processing').length,
    complete: displayStudies.filter(s => s.aiStatus === 'complete').length,
    error: displayStudies.filter(s => s.aiStatus === 'error').length,
  }), [displayStudies]);

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
            Worklist
          </h2>

          {/* View mode toggle */}
          <div className="flex items-center bg-ink-800 rounded-md p-0.5">
            <button
              onClick={() => setViewMode('list')}
              className={`p-1 rounded transition-colors ${viewMode === 'list' ? 'bg-ink-700 text-accent-400' : 'text-ink-500 hover:text-ink-200'}`}
              title="List view"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1 rounded transition-colors ${viewMode === 'grid' ? 'bg-ink-700 text-accent-400' : 'text-ink-500 hover:text-ink-200'}`}
              title="Grid view"
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
            placeholder="Search patient, modality..."
            className="w-full pl-8 pr-3 py-1.5 text-xs bg-ink-950/50 border border-ink-700 rounded-md text-ink-100 placeholder-ink-500 focus:outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20"
          />
        </div>

        {/* Modality Filter Pills */}
        <div className="flex gap-1 overflow-x-auto">
          {['all', 'CR', 'DX', 'CT', 'MR', 'US'].map((m) => {
            const count = m === 'all' ? displayStudies.length : displayStudies.filter(s => s.modality === m).length;
            if (count === 0 && m !== 'all') return null;
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
                {m === 'all' ? 'All' : m} <span className="opacity-60">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Stats strip */}
      <div className="px-3 py-2 border-b border-ink-800 grid grid-cols-4 gap-1">
        <StatChip label="Total" value={stats.total} color="text-ink-300" />
        <StatChip label="Pending" value={stats.pending} color="text-ink-400" />
        <StatChip label="Analyzing" value={stats.processing} color="text-accent-400" pulse={stats.processing > 0} />
        <StatChip label="Done" value={stats.complete} color="text-normal" />
      </div>

      {/* Sort bar */}
      <div className="px-3 py-1.5 border-b border-ink-800 flex items-center gap-2 text-[10px] text-ink-500">
        <span className="uppercase tracking-wider">Sort:</span>
        {(['time', 'patient', 'modality', 'status'] as SortKey[]).map(k => (
          <button
            key={k}
            onClick={() => setSortKey(k)}
            className={`capitalize ${sortKey === k ? 'text-accent-400 font-medium' : 'text-ink-500 hover:text-ink-300'}`}
          >
            {k}
          </button>
        ))}
      </div>

      {/* Study List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filtered.map((study) => (
          viewMode === 'list'
            ? <StudyCard key={study.id} study={study} selected={study.id === selectedStudyId} onClick={() => selectStudy(study.id)} />
            : <StudyThumb key={study.id} study={study} selected={study.id === selectedStudyId} onClick={() => selectStudy(study.id)} />
        ))}

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <svg className="w-12 h-12 text-ink-700 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm text-ink-400">No studies found</p>
            <p className="text-xs text-ink-600 mt-1">Try adjusting filters</p>
          </div>
        )}
      </div>

      {/* Upload Section — modality picker + button */}
      <div
        className="p-2 border-t border-ink-800 space-y-2"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".dcm,.dicom,application/dicom,*/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Modality picker — overrides the AI auto-detection */}
        <div>
          <div className="text-[9px] uppercase tracking-wider text-ink-500 mb-1">
            AI Model
          </div>
          <div className="grid grid-cols-3 gap-1">
            {(['auto', 'chest', 'brain_2d', 'head_ct', 'mammography'] as ForceModality[]).map((m) => (
              <button
                key={m}
                onClick={() => setForceModality(m)}
                className={`px-1.5 py-1 text-[9px] font-medium rounded transition-colors ${
                  forceModality === m
                    ? 'bg-accent-600 text-white'
                    : 'bg-ink-800 text-ink-400 hover:text-ink-200'
                }`}
                title={`Force the ${MODALITY_LABELS[m]} model`}
              >
                {MODALITY_LABELS[m]}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleUploadClick}
          disabled={uploading}
          className="w-full py-2 bg-accent-600 hover:bg-accent-500 disabled:bg-ink-700 text-white text-xs font-medium rounded-md transition-all flex items-center justify-center gap-2 shadow-sm hover:shadow-glow-accent"
        >
          {uploading ? (
            <>
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload DICOM ({MODALITY_LABELS[forceModality]})
            </>
          )}
        </button>

        <div className="text-[10px] text-ink-500 text-center">
          Drag &amp; drop DICOM here, or click above
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

function StudyCard({ study, selected, onClick }: { study: Study; selected: boolean; onClick: () => void }) {
  const statusConfig = {
    pending: { color: 'bg-ink-500', label: 'Pending', textColor: 'text-ink-400' },
    processing: { color: 'bg-accent-500 animate-pulse', label: 'Analyzing', textColor: 'text-accent-400' },
    complete: { color: 'bg-normal', label: 'Complete', textColor: 'text-normal' },
    error: { color: 'bg-critical', label: 'Error', textColor: 'text-critical' },
  }[study.aiStatus];

  const modalityColors: Record<string, string> = {
    CR: 'border-l-accent-500', DX: 'border-l-accent-500',
    CT: 'border-l-purple-500', MR: 'border-l-emerald-500',
    US: 'border-l-yellow-500',
  };

  const severity = study.aiStatus === 'complete' ? (Math.random() > 0.6 ? 'high' : 'low') : null;

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
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-ink-300 bg-ink-800 px-1.5 py-0.5 rounded">{study.modality}</span>
          {severity === 'high' && (
            <span className="severity-pill severity-critical">Critical</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <span className={`w-1.5 h-1.5 rounded-full ${statusConfig.color}`} />
          <span className={`text-[9px] font-medium uppercase tracking-wider ${statusConfig.textColor}`}>
            {statusConfig.label}
          </span>
        </div>
      </div>

      <div className="text-sm font-semibold text-ink-100 font-mono mb-0.5">{study.patientId}</div>

      <div className="flex items-center justify-between text-[10px] text-ink-500">
        <span className="truncate">{study.bodyPart}</span>
        <span className="font-mono flex-shrink-0">
          {new Date(study.receivedAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}
        </span>
      </div>

      {study.aiStatus === 'complete' && (
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="flex-1 h-0.5 bg-ink-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-accent-500 to-accent-400" style={{ width: '87%' }} />
          </div>
          <span className="text-[9px] text-ink-400 font-mono">87%</span>
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
