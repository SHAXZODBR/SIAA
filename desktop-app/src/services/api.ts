/**
 * API service for communicating with:
 * 1. Sentinel FastAPI inference server (localhost:8000)
 * 2. Orthanc PACS server (localhost:8042)
 */

import axios from 'axios';
import type { AIResult, Study, Finding } from '../types';

// ===== Inference Server API =====

const inferenceAPI = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,
});

export async function checkInferenceHealth(): Promise<boolean> {
  try {
    const res = await inferenceAPI.get('/health');
    return res.data.status === 'ok' && res.data.model_loaded;
  } catch {
    return false;
  }
}

export async function analyzeDicom(
  file: File | Blob,
  filename: string,
  language: 'ru' | 'uz' | 'en' = 'ru',
  options: { useAutoRouting?: boolean; forceModality?: string } = {},
): Promise<AIResult & { reportText?: string; gemmaAvailable?: boolean }> {
  const formData = new FormData();
  formData.append('file', file, filename);

  // Use the new multi-modality auto-routing endpoint by default;
  // it falls back to chest if it can't detect anything else.
  const useAuto = options.useAutoRouting !== false;
  const path = useAuto ? '/analyze/auto' : '/analyze';
  const params = new URLSearchParams({ language });
  if (options.forceModality) params.set('force_modality', options.forceModality);

  const res = await inferenceAPI.post(`${path}?${params.toString()}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 90000,
  });

  const data = res.data;
  return {
    id: data.study_id,
    studyId: data.study_id,
    findings: data.findings.map((f: any) => ({
      className: f.class_name,
      confidence: f.confidence,
      heatmapBase64: f.heatmap_base64,
      location: f.location,
    })),
    inferenceTimeMs: data.inference_time_ms,
    modelVersion: data.model_version,
    isNormal: data.normal,
    overallImpression: data.overall_impression,
    createdAt: new Date().toISOString(),
    reportText: data.report_text,
    gemmaAvailable: data.gemma_available,
  };
}

/**
 * List which pretrained models are available (and which need downloading).
 * Used in Settings to show modality status.
 */
export async function listAvailableModels(): Promise<any[]> {
  try {
    const res = await inferenceAPI.get('/models/available');
    return res.data.models || [];
  } catch {
    return [];
  }
}

/**
 * Regenerate report in a different language using Gemma 3.
 */
export async function regenerateReport(
  findings: any[],
  language: 'ru' | 'uz' | 'en',
  patientInfo?: any,
  modality: string = 'CR',
  bodyPart: string = 'CHEST',
): Promise<string> {
  const res = await inferenceAPI.post('/report/regenerate', {
    findings,
    language,
    patient_info: patientInfo,
    modality,
    body_part: bodyPart,
  }, { timeout: 60000 });
  return res.data.report;
}

/**
 * Ask AI a follow-up question about the analysis.
 */
export async function askAIQuestion(
  question: string,
  findings: any[],
  reportText: string,
  language: 'ru' | 'uz' | 'en' = 'ru',
): Promise<string> {
  const res = await inferenceAPI.post('/report/ask', {
    question,
    findings,
    report_text: reportText,
    language,
  }, { timeout: 60000 });
  return res.data.answer;
}

/**
 * Save doctor's correction of the AI-generated report.
 * Used to collect training data for fine-tuning.
 */
export async function saveReportCorrection(
  studyId: string,
  originalReport: string,
  correctedReport: string,
  doctorId: string,
  language: 'ru' | 'uz' | 'en' = 'ru',
): Promise<void> {
  await inferenceAPI.post('/report/save_correction', null, {
    params: {
      study_id: studyId,
      original_report: originalReport,
      corrected_report: correctedReport,
      doctor_id: doctorId,
      language,
    },
  });
}

// ===== Orthanc PACS API =====

const orthancAPI = axios.create({
  baseURL: 'http://127.0.0.1:8042',
  timeout: 10000,
});

export async function checkOrthancHealth(): Promise<boolean> {
  try {
    const res = await orthancAPI.get('/system');
    return res.status === 200;
  } catch {
    return false;
  }
}

export async function getOrthancStudies(): Promise<string[]> {
  try {
    const res = await orthancAPI.get('/studies');
    return res.data;
  } catch {
    return [];
  }
}

export async function getOrthancStudyDetails(studyId: string): Promise<any> {
  const res = await orthancAPI.get(`/studies/${studyId}`);
  return res.data;
}

export async function getOrthancStudyInstances(studyId: string): Promise<string[]> {
  const res = await orthancAPI.get(`/studies/${studyId}/instances`);
  return res.data;
}

export async function downloadOrthancInstance(instanceId: string): Promise<Blob> {
  const res = await orthancAPI.get(`/instances/${instanceId}/file`, {
    responseType: 'blob',
  });
  return res.data;
}

// ===== Orthanc Watcher =====

let watcherInterval: NodeJS.Timeout | null = null;
let knownStudies = new Set<string>();

export function startOrthancWatcher(
  onNewStudy: (studyId: string, details: any) => void,
  intervalMs: number = 10000,
): void {
  if (watcherInterval) {
    clearInterval(watcherInterval);
  }

  watcherInterval = setInterval(async () => {
    try {
      const studies = await getOrthancStudies();
      for (const id of studies) {
        if (!knownStudies.has(id)) {
          knownStudies.add(id);
          const details = await getOrthancStudyDetails(id);
          onNewStudy(id, details);
        }
      }
    } catch {
      // Orthanc not available — silently retry
    }
  }, intervalMs);
}

export function stopOrthancWatcher(): void {
  if (watcherInterval) {
    clearInterval(watcherInterval);
    watcherInterval = null;
  }
}
