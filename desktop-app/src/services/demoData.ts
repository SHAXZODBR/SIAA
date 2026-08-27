/**
 * Shared demo data — used by Sidebar (list of fake studies) and RightPanel
 * (modality-appropriate findings). Single source of truth so they stay in sync.
 */

import type { Study, AIResult } from '../types';

// ─── Demo studies that appear in the worklist before any real upload ────────
export const DEMO_STUDIES: Study[] = [
  { id: 'st-001', patientId: 'P-20241001', modality: 'CR', bodyPart: 'CHEST PA',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T09:15:00', dicomPath: '', aiStatus: 'complete' },
  { id: 'st-002', patientId: 'P-20241002', modality: 'CT', bodyPart: 'CHEST',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T09:45:00', dicomPath: '', aiStatus: 'processing' },
  { id: 'st-003', patientId: 'P-20241003', modality: 'CR', bodyPart: 'CHEST AP',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T10:20:00', dicomPath: '', aiStatus: 'pending' },
  { id: 'st-004', patientId: 'P-20241004', modality: 'MR', bodyPart: 'BRAIN',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T11:00:00', dicomPath: '', aiStatus: 'complete' },
  { id: 'st-005', patientId: 'P-20241005', modality: 'CR', bodyPart: 'CHEST PA',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T11:30:00', dicomPath: '', aiStatus: 'error' },
  { id: 'st-006', patientId: 'P-20241006', modality: 'CT', bodyPart: 'HEAD',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T12:15:00', dicomPath: '', aiStatus: 'complete' },
  { id: 'st-007', patientId: 'P-20241007', modality: 'CR', bodyPart: 'CHEST PA',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T13:05:00', dicomPath: '', aiStatus: 'complete' },
  { id: 'st-008', patientId: 'P-20241008', modality: 'MG', bodyPart: 'BREAST',
    studyDate: '2024-10-01', receivedAt: '2024-10-01T14:00:00', dicomPath: '', aiStatus: 'pending' },
];

// ─── Modality-aware demo AI results ─────────────────────────────────────────
export const DEMO_CHEST_PNEUMONIA: AIResult = {
  id: 'demo-chest-pneu', studyId: '',
  findings: [
    { className: 'Pneumonia',     confidence: 0.87, heatmapBase64: '', location: 'Right lower zone' },
    { className: 'Effusion',      confidence: 0.62, heatmapBase64: '', location: 'Right lower zone' },
    { className: 'Consolidation', confidence: 0.45, heatmapBase64: '', location: 'Right lower zone' },
    { className: 'Cardiomegaly',  confidence: 0.31, heatmapBase64: '', location: 'Central / mediastinal region' },
  ],
  inferenceTimeMs: 4200, modelVersion: 'Chest X-ray (demo)', isNormal: false,
  overallImpression: 'Findings suggestive of right lower lobe pneumonia with pleural effusion.',
  createdAt: new Date().toISOString(),
};

export const DEMO_CHEST_NORMAL: AIResult = {
  id: 'demo-chest-norm', studyId: '',
  findings: [],
  inferenceTimeMs: 3100, modelVersion: 'Chest X-ray (demo)', isNormal: true,
  overallImpression: 'No significant pathological findings.',
  createdAt: new Date().toISOString(),
};

export const DEMO_CHEST_TB: AIResult = {
  id: 'demo-chest-tb', studyId: '',
  findings: [
    { className: 'Infiltration',  confidence: 0.78, heatmapBase64: '', location: 'Right upper zone' },
    { className: 'Consolidation', confidence: 0.55, heatmapBase64: '', location: 'Right upper zone' },
    { className: 'Cavitation',    confidence: 0.42, heatmapBase64: '', location: 'Right apex' },
  ],
  inferenceTimeMs: 4500, modelVersion: 'Chest X-ray + TB classifier (demo)', isNormal: false,
  overallImpression: 'Pattern suggestive of TB. Strongly recommend sputum testing.',
  createdAt: new Date().toISOString(),
};

export const DEMO_BRAIN_GLIOMA: AIResult = {
  id: 'demo-brain', studyId: '',
  findings: [
    { className: 'glioma_tumor',     confidence: 0.87, heatmapBase64: '', location: 'Left frontal lobe' },
    { className: 'meningioma_tumor', confidence: 0.11, heatmapBase64: '', location: 'Left frontal lobe' },
    { className: 'no_tumor',         confidence: 0.02, heatmapBase64: '', location: 'Brain parenchyma' },
  ],
  inferenceTimeMs: 5200, modelVersion: 'Brain Tumor Classifier (demo)', isNormal: false,
  overallImpression: 'Findings suggestive of glioma in the left frontal lobe. Tumor volume: 12.4 cm³.',
  createdAt: new Date().toISOString(),
};

export const DEMO_HEAD_CT_BLEED: AIResult = {
  id: 'demo-head-ct', studyId: '',
  findings: [
    { className: 'intraventricular', confidence: 0.76, heatmapBase64: '', location: 'Lateral ventricles' },
    { className: 'any',              confidence: 0.72, heatmapBase64: '', location: 'Intracranial' },
    { className: 'normal',           confidence: 0.19, heatmapBase64: '', location: 'Brain parenchyma' },
  ],
  inferenceTimeMs: 3600, modelVersion: 'Head CT Hemorrhage (demo)', isNormal: false,
  overallImpression: 'Intraventricular hemorrhage detected. URGENT neurosurgical consultation.',
  createdAt: new Date().toISOString(),
};

export const DEMO_MAMMO: AIResult = {
  id: 'demo-mammo', studyId: '',
  findings: [
    { className: 'mass',          confidence: 0.68, heatmapBase64: '', location: 'Upper outer quadrant' },
    { className: 'calcification', confidence: 0.34, heatmapBase64: '', location: 'Upper outer quadrant' },
  ],
  inferenceTimeMs: 2900, modelVersion: 'Mammography (demo)', isNormal: false,
  overallImpression: 'Suspicious mass with associated calcifications. BI-RADS 4 — biopsy recommended.',
  createdAt: new Date().toISOString(),
};

export const DEMO_SPINE_UNSUPPORTED: AIResult = {
  id: 'demo-spine', studyId: '',
  findings: [],
  inferenceTimeMs: 0, modelVersion: 'Spine MRI — V1.1 roadmap', isNormal: true,
  overallImpression: 'Spine MRI is not yet supported in Sentinel V1.0. Coming in V1.1.',
  createdAt: new Date().toISOString(),
};

export const DEMO_ABDOMEN_UNSUPPORTED: AIResult = {
  id: 'demo-abdomen', studyId: '',
  findings: [],
  inferenceTimeMs: 0, modelVersion: 'Abdomen CT — V1.1 roadmap', isNormal: true,
  overallImpression: 'Abdomen CT is not yet supported in Sentinel V1.0. Coming in V1.1.',
  createdAt: new Date().toISOString(),
};

// ─── Pick demo per (study modality, body part) + status ─────────────────────
export function getDemoResultForStudy(
  study: { id?: string; modality: string; bodyPart: string; aiStatus?: string } | null,
): AIResult | null {
  if (!study) return null;
  // For pending/processing studies, don't show fake findings — show empty.
  if (study.aiStatus === 'pending' || study.aiStatus === 'processing') {
    return {
      id: 'demo-pending', studyId: study.id || '',
      findings: [], inferenceTimeMs: 0,
      modelVersion: 'AI analysis in progress…', isNormal: true,
      overallImpression: 'Analysis will appear here once complete.',
      createdAt: new Date().toISOString(),
    };
  }
  if (study.aiStatus === 'error') {
    return {
      id: 'demo-error', studyId: study.id || '',
      findings: [], inferenceTimeMs: 0,
      modelVersion: 'Error — analysis failed', isNormal: true,
      overallImpression: 'AI analysis failed. Click "Re-run AI Analysis" to try again.',
      createdAt: new Date().toISOString(),
    };
  }

  const bp = (study.bodyPart || '').toUpperCase();
  const mod = (study.modality || '').toUpperCase();

  // Brain MRI → glioma demo
  if (bp.includes('BRAIN') && (mod === 'MR' || mod === 'MRI')) return DEMO_BRAIN_GLIOMA;
  // Head CT → hemorrhage demo
  if ((bp.includes('HEAD') || bp.includes('BRAIN') || bp.includes('SKULL')) && mod === 'CT') return DEMO_HEAD_CT_BLEED;
  // Spine → not supported
  if (bp.includes('SPINE')) return DEMO_SPINE_UNSUPPORTED;
  // Abdomen → not supported
  if (bp.includes('ABDOMEN')) return DEMO_ABDOMEN_UNSUPPORTED;
  // Mammography
  if (mod === 'MG' || mod === 'MAMMO' || bp.includes('BREAST') || bp.includes('MAMMARY')) return DEMO_MAMMO;
  // Chest variants — vary by study ID so different rows show different findings
  if (bp.includes('CHEST') || bp.includes('THORAX') || bp.includes('LUNG')) {
    if (study.id === 'st-003' || study.id === 'st-007') return DEMO_CHEST_NORMAL;
    if (study.id === 'st-005') return DEMO_CHEST_TB;
    return DEMO_CHEST_PNEUMONIA;
  }

  // Default
  return DEMO_CHEST_PNEUMONIA;
}
