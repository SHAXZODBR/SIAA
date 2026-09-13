/**
 * API service for communicating with:
 * 1. Sentinel FastAPI inference server (API contract v1 — pilot)
 * 2. Orthanc PACS server (localhost:8042)
 *
 * The base URL comes from settings.inferenceUrl; every clinical route carries
 * `Authorization: Bearer <jwt>` from the store. A 401 anywhere clears auth and
 * returns the app to the login screen.
 */

import axios, { AxiosError } from 'axios';
import { useAppStore } from '../store/appStore';
import type {
  AIResult, Study, Finding, FindingStatus, ModelIdentity, OverallAssessment,
  HealthStatus, SignedReport, User, Lang, LicenseStatus, AvailableModel,
} from '../types';
import type { I18nKey } from '../i18n/en';

// ===== Inference Server API =====

const inferenceAPI = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,
});

function normalizeBaseUrl(url: string): string {
  const trimmed = (url || '').trim().replace(/\/+$/, '');
  return trimmed || 'http://127.0.0.1:8000';
}

inferenceAPI.interceptors.request.use((config) => {
  const { settings, authToken } = useAppStore.getState();
  config.baseURL = normalizeBaseUrl(settings.inferenceUrl);
  if (authToken) {
    config.headers.set('Authorization', `Bearer ${authToken}`);
  }
  return config;
});

inferenceAPI.interceptors.response.use(
  (res) => res,
  (error: AxiosError) => {
    const status = error.response?.status;
    const url = error.config?.url || '';
    if (status === 401 && !url.includes('/auth/login')) {
      // Session expired / token rejected — back to the login screen.
      useAppStore.getState().clearAuth();
    }
    return Promise.reject(error);
  },
);

export type ApiErrorCode = 'network' | 'auth' | 'forbidden' | 'conflict' | 'review' | 'server';

export interface ApiErrorInfo {
  code: ApiErrorCode;
  /** Technical detail from the server (stays in the console / toast body). */
  detail: string;
  /** Localized doctor-facing message key — render with t(info.messageKey). */
  messageKey: I18nKey;
}

const ERROR_MESSAGE_KEYS: Record<ApiErrorCode, I18nKey> = {
  network: 'health.aiServerDown',
  auth: 'health.sessionExpired',
  forbidden: 'error.forbidden',
  conflict: 'error.conflict',
  review: 'worklist.notAnalyzed',
  server: 'error.server',
};

/** Doctor-facing classification of a failed request (technical detail stays in console). */
export function describeApiError(e: unknown): ApiErrorInfo {
  const err = e as AxiosError<any>;
  const status = err?.response?.status;
  const data = err?.response?.data;
  const detail: string =
    (typeof data?.detail === 'string' && data.detail) ||
    (typeof data?.reason === 'string' && data.reason) ||
    err?.message || 'Unknown error';
  const build = (code: ApiErrorCode, d = detail): ApiErrorInfo => ({ code, detail: d, messageKey: ERROR_MESSAGE_KEYS[code] });
  if (!err?.response) return build('network');
  if (status === 401) return build('auth');
  if (status === 403) return build('forbidden');
  if (status === 409) return build('conflict');
  if (status === 422 && data?.requires_review) return build('review', data?.reason || detail);
  return build('server');
}

// ----- Auth -----------------------------------------------------------------

export interface LoginResult {
  token: string;
  user: User;
  mustChangePassword: boolean;
}

function mapUser(u: any): User {
  return {
    id: String(u?.id ?? ''),
    username: String(u?.username ?? ''),
    fullName: String(u?.full_name ?? u?.fullName ?? u?.username ?? ''),
    role: String(u?.role ?? 'radiologist'),
  };
}

export async function login(username: string, password: string): Promise<LoginResult> {
  const res = await inferenceAPI.post('/auth/login', { username, password });
  const data = res.data || {};
  const token: string = data.access_token;
  if (!token) throw new Error('Login response did not include access_token');
  return {
    token,
    user: mapUser(data.user),
    mustChangePassword: !!data.must_change_password,
  };
}

export async function changePassword(oldPassword: string, newPassword: string): Promise<void> {
  await inferenceAPI.post('/auth/change_password', {
    old_password: oldPassword,
    new_password: newPassword,
  });
}

// ----- Health ---------------------------------------------------------------

export function unreachableHealth(): HealthStatus {
  return {
    reachable: false,
    status: 'error',
    version: null,
    device: null,
    authRequired: true,
    models: {},
    llm: { backend: null, reachable: false },
    license: { mode: null },
    dataDir: null,
  };
}

function mapHealth(d: any): HealthStatus {
  const models: HealthStatus['models'] = {};
  if (d?.models && typeof d.models === 'object') {
    for (const [k, v] of Object.entries<any>(d.models)) {
      models[k] = { loaded: !!v?.loaded, reason: v?.reason ?? null };
    }
  }
  const status = d?.status === 'ok' || d?.status === 'degraded' ? d.status : 'error';
  return {
    reachable: true,
    status,
    version: d?.version ?? null,
    device: d?.device ?? null,
    authRequired: d?.auth_required !== false,
    models,
    llm: { backend: d?.llm?.backend ?? null, reachable: !!d?.llm?.reachable },
    license: { mode: d?.license?.mode ?? null },
    dataDir: d?.data_dir ?? null,
  };
}

export async function getHealth(): Promise<HealthStatus> {
  try {
    const res = await inferenceAPI.get('/health', { timeout: 5000 });
    return mapHealth(res.data);
  } catch {
    return unreachableHealth();
  }
}

export async function checkInferenceHealth(): Promise<boolean> {
  const h = await getHealth();
  return h.reachable && h.status === 'ok';
}

// ----- Response mapping -----------------------------------------------------

const FINDING_STATUSES: FindingStatus[] = ['validated', 'pending', 'experimental'];

function mapFinding(f: any, fallbackStatus: FindingStatus): Finding {
  const status: FindingStatus = FINDING_STATUSES.includes(f?.status) ? f.status : fallbackStatus;
  return {
    className: String(f?.class_name ?? ''),
    finding: String(f?.finding ?? f?.class_name ?? ''),
    confidence: typeof f?.confidence === 'number' ? f.confidence : Number(f?.confidence) || 0,
    positive: !!f?.positive,
    status,
    detector: f?.detector ?? null,
    sequenceUsed: f?.sequence_used ?? null,
    heatmapBase64: typeof f?.heatmap_base64 === 'string' ? f.heatmap_base64 : '',
    location: String(f?.location ?? ''),
  };
}

function mapModelIdentity(m: any): ModelIdentity {
  return {
    key: String(m?.key ?? ''),
    displayName: String(m?.display_name ?? m?.key ?? ''),
    source: m?.source ?? null,
    sha256_12: m?.sha256_12 ?? null,
    status: m?.status ?? 'experimental',
    validationNote: m?.validation_note ?? null,
  };
}

function mapOverall(o: any): OverallAssessment | null {
  if (!o || typeof o !== 'object') return null;
  return {
    abnormalFlagged: !!o.abnormal_flagged,
    flags: Array.isArray(o.flags) ? o.flags.map(String) : [],
    text: String(o.text ?? ''),
  };
}

function parseMaybeJson<T>(v: any, fallback: T): T {
  if (v == null) return fallback;
  if (typeof v === 'string') {
    try { return JSON.parse(v) as T; } catch { return fallback; }
  }
  return v as T;
}

/** DICOM DA 'YYYYMMDD' → 'YYYY-MM-DD'; ISO strings pass through. */
export function normalizeDicomDate(v: any): string {
  if (!v) return '';
  const s = String(v).trim();
  if (/^\d{8}$/.test(s)) return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`;
  return s;
}

function modelVersionLabel(models: ModelIdentity[]): string {
  if (!models.length) return '';
  return models.map((m) => `${m.displayName}${m.sha256_12 ? ` (${m.sha256_12})` : ''}`).join(', ');
}

export interface StudyAnalysis {
  study: Study;
  result: AIResult;
  reportText: string | null;
  reportLanguage: Lang;
  gemmaAvailable: boolean;
}

function mapStudyResponse(data: any): StudyAnalysis {
  const now = new Date().toISOString();
  const studyId: string = String(data?.study_id ?? `st-${Date.now()}`);
  const modelIdentity = Array.isArray(data?.model_identity) ? data.model_identity.map(mapModelIdentity) : [];
  const modality = String(data?.modality ?? '').toUpperCase() || 'N/A';
  const bodyPart = String(data?.body_part ?? '') || '—';

  const study: Study = {
    id: studyId,
    patientId: data?.patient_id ? String(data.patient_id) : 'UNKNOWN',
    modality,
    bodyPart,
    studyDate: normalizeDicomDate(data?.study_date),
    receivedAt: now,
    dicomPath: '',
    aiStatus: 'complete',
    aiAnalyzedAt: now,
    studyInstanceUid: data?.study_instance_uid ?? null,
    accessionNumber: data?.accession_number ?? null,
    patientName: data?.patient_name ?? null,
    patientSex: data?.patient_sex ?? null,
    patientAge: data?.patient_age ?? null,
    patientBirthDate: data?.patient_birth_date ? normalizeDicomDate(data.patient_birth_date) : null,
    studyDescription: data?.study_description ?? null,
    manufacturer: data?.manufacturer ?? null,
    scannerModel: data?.scanner_model ?? null,
    numFiles: typeof data?.num_files === 'number' ? data.num_files : null,
    seriesDescriptions: Array.isArray(data?.series_descriptions) ? data.series_descriptions.map(String) : [],
  };

  const findings: Finding[] = Array.isArray(data?.findings)
    ? data.findings.map((f: any) => mapFinding(f, 'experimental'))
    : [];

  const reportLanguage: Lang = (['ru', 'uz', 'en'] as Lang[]).includes(data?.report_language) ? data.report_language : 'ru';

  const result: AIResult = {
    id: studyId,
    studyId,
    findings,
    inferenceTimeMs: Number(data?.inference_time_ms) || 0,
    modelVersion: modelVersionLabel(modelIdentity),
    isNormal: false, // the product never certifies normal
    overallImpression: String(data?.overall_impression ?? ''),
    createdAt: now,
    previewBase64: data?.preview_base64 || undefined,
    modality,
    bodyPart,
    overallAssessment: mapOverall(data?.overall_assessment),
    disclaimer: String(data?.disclaimer ?? ''),
    modelIdentity,
    threshold: typeof data?.threshold === 'number' ? data.threshold : null,
    requiresReview: !!data?.requires_review,
    rejected: !!data?.rejected,
    rejectionReason: data?.rejection_reason ?? null,
    appVersion: data?.app_version ?? null,
    reportLanguage,
    gemmaAvailable: !!data?.gemma_available,
  };

  return {
    study,
    result,
    reportText: typeof data?.report_text === 'string' && data.report_text.trim() ? data.report_text : null,
    reportLanguage,
    gemmaAvailable: !!data?.gemma_available,
  };
}

// ----- Analyze --------------------------------------------------------------

/**
 * Whole-study analysis — POST /analyze/study with repeated `files` fields.
 * The server routes by DICOM headers; the client never guesses modality.
 */
export async function analyzeStudy(
  files: File[],
  language: Lang = 'ru',
  onUploadProgress?: (percent: number) => void,
): Promise<StudyAnalysis> {
  const formData = new FormData();
  for (const f of files) formData.append('files', f, f.name);

  const params = new URLSearchParams({ language });
  const res = await inferenceAPI.post(`/analyze/study?${params.toString()}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000,
    onUploadProgress: (evt) => {
      if (onUploadProgress && evt.total) {
        onUploadProgress(Math.min(100, Math.round((evt.loaded * 100) / evt.total)));
      }
    },
  });
  return mapStudyResponse(res.data);
}

/**
 * Single-file analysis via the legacy /analyze/auto route (kept for compatibility).
 * New code should use analyzeStudy().
 */
export async function analyzeDicom(
  file: File | Blob,
  filename: string,
  language: Lang = 'ru',
  options: { useAutoRouting?: boolean; forceModality?: string } = {},
): Promise<AIResult & { reportText?: string; gemmaAvailable?: boolean }> {
  const formData = new FormData();
  formData.append('file', file, filename);

  const useAuto = options.useAutoRouting !== false;
  const path = useAuto ? '/analyze/auto' : '/analyze';
  const params = new URLSearchParams({ language });
  if (options.forceModality) params.set('force_modality', options.forceModality);

  const res = await inferenceAPI.post(`${path}?${params.toString()}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 90000,
  });

  const data = res.data;
  // Backend modality looks like "MR/brain_2d" → split into DICOM modality + body part
  const rawModality: string = data.modality || '';
  const [modPart, keyPart] = rawModality.split('/');
  const bodyPartMap: Record<string, string> = {
    brain_2d: 'BRAIN', brain: 'BRAIN', head_ct: 'HEAD',
    chest: 'CHEST', mammography: 'BREAST',
  };
  const modality = (modPart || 'CR').toUpperCase();
  const bodyPart = bodyPartMap[keyPart] || (modality === 'MR' ? 'BRAIN' : 'CHEST');
  const modelIdentity = Array.isArray(data.model_identity) ? data.model_identity.map(mapModelIdentity) : [];
  const now = new Date().toISOString();
  return {
    id: data.study_id,
    studyId: data.study_id,
    findings: (data.findings || []).map((f: any) => mapFinding(f, 'pending')),
    inferenceTimeMs: data.inference_time_ms,
    modelVersion: modelVersionLabel(modelIdentity) || data.model_version || '',
    isNormal: false,
    overallImpression: data.overall_impression,
    createdAt: now,
    reportText: data.report_text,
    gemmaAvailable: data.gemma_available,
    previewBase64: data.preview_base64,
    modality,
    bodyPart,
    overallAssessment: mapOverall(data.overall_assessment),
    disclaimer: String(data.disclaimer ?? ''),
    modelIdentity,
    threshold: typeof data.threshold === 'number' ? data.threshold : null,
    requiresReview: !!data.requires_review,
    rejected: !!data.rejected,
    rejectionReason: data.rejection_reason ?? null,
    appVersion: data.app_version ?? null,
  };
}

// ----- Studies (worklist persistence) ---------------------------------------

function mapStudyRow(row: any): Study {
  const id = String(row?.study_id ?? row?.id ?? '');
  return {
    id,
    patientId: row?.patient_id ? String(row.patient_id) : 'UNKNOWN',
    modality: String(row?.modality ?? '').toUpperCase() || 'N/A',
    bodyPart: String(row?.body_part ?? '') || '—',
    studyDate: normalizeDicomDate(row?.study_date),
    receivedAt: row?.created_at ? String(row.created_at) : new Date().toISOString(),
    dicomPath: '',
    aiStatus: 'complete',
    studyInstanceUid: row?.study_instance_uid ?? null,
    accessionNumber: row?.accession_number ?? null,
    patientName: row?.patient_name ?? null,
    patientSex: row?.patient_sex ?? null,
    patientAge: row?.patient_age ?? null,
    patientBirthDate: row?.patient_birth_date ? normalizeDicomDate(row.patient_birth_date) : null,
    studyDescription: row?.study_description ?? null,
    manufacturer: row?.manufacturer ?? null,
    scannerModel: row?.scanner_model ?? null,
    numFiles: typeof row?.num_files === 'number' ? row.num_files : null,
    seriesDescriptions: Array.isArray(row?.series_descriptions) ? row.series_descriptions.map(String) : [],
    restored: true,
  };
}

export async function listStudies(limit = 100): Promise<Study[]> {
  const res = await inferenceAPI.get('/studies', { params: { limit } });
  const rows: any[] = Array.isArray(res.data) ? res.data : (res.data?.studies || []);
  return rows.map(mapStudyRow).filter((s) => s.id);
}

export interface StudyDetail {
  study: Study;
  result: AIResult | null;
  signed: SignedReport[];
  drafts: { language: Lang; aiDraftText: string; reportText: string }[];
}

function mapSignedRow(r: any, studyId: string): SignedReport | null {
  if (!r || !(r.is_signed === 1 || r.is_signed === true || r.signed_at)) return null;
  const language: Lang = (['ru', 'uz', 'en'] as Lang[]).includes(r.language) ? r.language : 'ru';
  const signer = r.signer || {};
  return {
    reportId: String(r.report_id ?? r.id ?? ''),
    studyId,
    language,
    signedAt: String(r.signed_at ?? ''),
    signer: {
      id: String(signer.id ?? r.signer_id ?? ''),
      username: String(signer.username ?? r.signer_username ?? ''),
      fullName: String(signer.full_name ?? r.signer_full_name ?? signer.username ?? r.signer_id ?? ''),
    },
    sha256: String(r.sha256 ?? ''),
    modelIdentity: parseMaybeJson<any[]>(r.model_identity, []).map(mapModelIdentity),
    reportText: String(r.report_text ?? ''),
  };
}

export async function getStudy(studyId: string): Promise<StudyDetail> {
  const res = await inferenceAPI.get(`/study/${encodeURIComponent(studyId)}`);
  const data = res.data || {};
  const study = mapStudyRow(data.study || { study_id: studyId });
  study.restored = true;

  let result: AIResult | null = null;
  const ai = data.ai_result;
  if (ai) {
    const modelIdentity = parseMaybeJson<any[]>(ai.model_identity, []).map(mapModelIdentity);
    const findings = parseMaybeJson<any[]>(ai.findings, []).map((f) => mapFinding(f, 'experimental'));
    const overall = mapOverall(parseMaybeJson<any>(ai.overall, null));
    result = {
      id: study.id,
      studyId: study.id,
      findings,
      inferenceTimeMs: Number(ai.inference_ms ?? ai.inference_time_ms) || 0,
      modelVersion: modelVersionLabel(modelIdentity),
      isNormal: false,
      overallImpression: overall?.text || '',
      createdAt: String(ai.created_at ?? study.receivedAt),
      previewBase64: undefined, // preview slices are not persisted server-side
      modality: study.modality,
      bodyPart: study.bodyPart,
      overallAssessment: overall,
      disclaimer: String(ai.disclaimer ?? ''),
      modelIdentity,
      threshold: typeof ai.threshold === 'number' ? ai.threshold : null,
      requiresReview: !!ai.requires_review,
      rejected: !!ai.rejected,
      rejectionReason: ai.rejection_reason ?? null,
      appVersion: ai.app_version ?? null,
    };
  }

  const reports: any[] = Array.isArray(data.reports) ? data.reports : [];
  const signed = reports.map((r) => mapSignedRow(r, study.id)).filter((x): x is SignedReport => !!x);
  const drafts = reports.map((r) => ({
    language: ((['ru', 'uz', 'en'] as Lang[]).includes(r?.language) ? r.language : 'ru') as Lang,
    aiDraftText: String(r?.ai_draft_text ?? ''),
    reportText: String(r?.report_text ?? ''),
  }));
  return { study, result, signed, drafts };
}

/**
 * Every detector the server knows about (GET /models/available) — shown in
 * Settings → AI models. Returns [] when the server is unreachable.
 */
export async function listAvailableModels(): Promise<AvailableModel[]> {
  try {
    const res = await inferenceAPI.get('/models/available', { timeout: 8000 });
    const rows: any[] = Array.isArray(res.data?.models) ? res.data.models : [];
    return rows.map((m) => ({
      key: String(m?.key ?? ''),
      name: String(m?.name ?? m?.key ?? ''),
      tier: m?.tier ?? null,
      validationStatus: m?.validation_status ?? null,
      license: m?.license ?? null,
      is3d: !!m?.is_3d,
      depsOk: m?.deps_ok !== false,
      depsReason: m?.deps_reason || null,
      downloadMb: typeof m?.download_mb === 'number' ? m.download_mb : null,
      classes: Array.isArray(m?.classes) ? m.classes.map(String) : [],
    })).filter((m) => m.key);
  } catch {
    return [];
  }
}

export function unreachableLicense(): LicenseStatus {
  return {
    reachable: false, valid: false, mode: null, customer: null, tier: null,
    expiresAt: null, features: [], reason: null, demoCallsToday: null, demoLimit: null,
  };
}

/** GET /license/status — public route; never throws. */
export async function getLicenseStatus(): Promise<LicenseStatus> {
  try {
    const res = await inferenceAPI.get('/license/status', { timeout: 5000 });
    const d = res.data || {};
    const info = d.info && typeof d.info === 'object' ? d.info : {};
    return {
      reachable: true,
      valid: !!d.valid,
      mode: info.mode ?? null,
      customer: info.customer ?? null,
      tier: info.tier ?? null,
      expiresAt: info.expires_at ?? null,
      features: Array.isArray(info.features) ? info.features.map(String) : [],
      reason: info.reason ?? null,
      demoCallsToday: typeof d.demo_calls_today === 'number' ? d.demo_calls_today : null,
      demoLimit: typeof d.demo_limit === 'number' ? d.demo_limit : null,
    };
  } catch {
    return unreachableLicense();
  }
}

/**
 * Probe an arbitrary base URL (Setup wizard "Test connection") without
 * touching settings.inferenceUrl. Never throws.
 */
export async function probeHealth(baseUrl: string): Promise<HealthStatus> {
  try {
    const res = await axios.get(`${normalizeBaseUrl(baseUrl)}/health`, { timeout: 5000 });
    return mapHealth(res.data);
  } catch {
    return unreachableHealth();
  }
}

// ----- Reports --------------------------------------------------------------

/**
 * Regenerate report in a different language using the local LLM.
 */
export async function regenerateReport(
  findings: any[],
  language: Lang,
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
 * Sign a report on the server — POST /report/sign. Requires the sign_report permission.
 * A second sign for the same study+language returns 409.
 */
export async function signReport(
  studyId: string,
  reportText: string,
  language: Lang,
  aiDraftText: string,
): Promise<SignedReport> {
  const res = await inferenceAPI.post('/report/sign', {
    study_id: studyId,
    report_text: reportText,
    language,
    ai_draft_text: aiDraftText,
  }, { timeout: 30000 });
  const d = res.data || {};
  const signer = d.signer || {};
  return {
    reportId: String(d.report_id ?? ''),
    studyId: String(d.study_id ?? studyId),
    language,
    signedAt: String(d.signed_at ?? new Date().toISOString()),
    signer: {
      id: String(signer.id ?? ''),
      username: String(signer.username ?? ''),
      fullName: String(signer.full_name ?? signer.username ?? ''),
    },
    sha256: String(d.sha256 ?? ''),
    modelIdentity: Array.isArray(d.model_identity) ? d.model_identity.map(mapModelIdentity) : [],
    reportText,
  };
}

/**
 * Ask AI a follow-up question about the analysis.
 */
export async function askAIQuestion(
  question: string,
  findings: any[],
  reportText: string,
  language: Lang = 'ru',
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
 * Save doctor's correction of the AI-generated report (JSON body).
 * Used to collect training data for fine-tuning.
 */
export async function saveReportCorrection(
  studyId: string,
  originalReport: string,
  correctedReport: string,
  language: Lang = 'ru',
): Promise<void> {
  await inferenceAPI.post('/report/save_correction', {
    study_id: studyId,
    original_report: originalReport,
    corrected_report: correctedReport,
    language,
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

let watcherInterval: ReturnType<typeof setInterval> | null = null;
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
