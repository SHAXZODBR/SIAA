/**
 * Demo data — ONLY shown when the app is built with VITE_DEMO_MODE=1.
 * In a clinical build DEMO_MODE is false, `getDemoResultForStudy` returns null
 * and the worklist starts empty.
 */

import type { Study, AIResult, Finding, FindingStatus } from '../types';

export const DEMO_MODE: boolean = import.meta.env.VITE_DEMO_MODE === '1';

// ─── Demo studies that appear in the worklist (demo builds only) ───────────
export const DEMO_STUDIES: Study[] = [
  { id: 'demo-001', patientId: 'DEMO-0001', modality: 'MR', bodyPart: 'BRAIN',
    studyDate: '2025-01-15', receivedAt: '2025-01-15T09:15:00', dicomPath: '', aiStatus: 'complete', numFiles: 24 },
  { id: 'demo-002', patientId: 'DEMO-0002', modality: 'MR', bodyPart: 'BRAIN',
    studyDate: '2025-01-15', receivedAt: '2025-01-15T09:45:00', dicomPath: '', aiStatus: 'processing' },
  { id: 'demo-003', patientId: 'DEMO-0003', modality: 'CR', bodyPart: 'CHEST',
    studyDate: '2025-01-15', receivedAt: '2025-01-15T10:20:00', dicomPath: '', aiStatus: 'complete', numFiles: 1 },
];

function demoFinding(
  className: string, finding: string, confidence: number, positive: boolean,
  status: FindingStatus, detector: string, location: string,
): Finding {
  return { className, finding, confidence, positive, status, detector, sequenceUsed: null, heatmapBase64: '', location };
}

function demoResult(partial: Partial<AIResult> & Pick<AIResult, 'id' | 'findings' | 'overallImpression'>): AIResult {
  return {
    studyId: '',
    inferenceTimeMs: 0,
    modelVersion: 'demo',
    isNormal: false,
    createdAt: new Date().toISOString(),
    overallAssessment: null,
    disclaimer: 'DEMO DATA — not a clinical result.',
    modelIdentity: [],
    threshold: null,
    requiresReview: false,
    rejected: false,
    rejectionReason: null,
    appVersion: null,
    ...partial,
  };
}

export const DEMO_BRAIN_PANEL: AIResult = demoResult({
  id: 'demo-brain',
  findings: [
    demoFinding('abnormal', 'Abnormal (triage)', 0.81, true, 'validated', 'triage', 'Brain parenchyma'),
    demoFinding('glioma_tumor', 'Glioma', 0.64, true, 'pending', 'tumor', 'Left frontal lobe'),
  ],
  inferenceTimeMs: 5200,
  modelVersion: 'Brain panel (demo)',
  overallImpression: 'DEMO: abnormal flagged by the triage detector — requires radiologist review.',
  overallAssessment: { abnormalFlagged: true, flags: ['abnormal'], text: 'DEMO: abnormal flagged by the triage detector — requires radiologist review.' },
});

export const DEMO_CHEST: AIResult = demoResult({
  id: 'demo-chest',
  findings: [
    demoFinding('Pneumonia', 'Pneumonia', 0.57, true, 'pending', 'chest', 'Right lower zone'),
  ],
  inferenceTimeMs: 4200,
  modelVersion: 'Chest X-ray (demo)',
  overallImpression: 'DEMO: pending-validation detector flagged pneumonia — NOT a normal read.',
  overallAssessment: { abnormalFlagged: true, flags: ['Pneumonia'], text: 'DEMO: pending-validation detector flagged pneumonia — NOT a normal read.' },
});

// ─── Pick demo per (study modality, body part) + status ─────────────────────
export function getDemoResultForStudy(
  study: { id?: string; modality: string; bodyPart: string; aiStatus?: string } | null,
): AIResult | null {
  if (!DEMO_MODE || !study) return null;
  // For pending/processing studies, don't show fake findings — show empty.
  if (study.aiStatus === 'pending' || study.aiStatus === 'processing' || study.aiStatus === 'error') {
    return null;
  }

  const bp = (study.bodyPart || '').toUpperCase();
  const mod = (study.modality || '').toUpperCase();

  if (bp.includes('BRAIN') && (mod === 'MR' || mod === 'MRI')) return { ...DEMO_BRAIN_PANEL, studyId: study.id || '' };
  if (bp.includes('CHEST')) return { ...DEMO_CHEST, studyId: study.id || '' };
  return null;
}
