// ===== Core Types for Sentinel Medical AI =====

export type Lang = 'ru' | 'uz' | 'en';

/** Per-detector validation status reported by the server (API contract v1). */
export type FindingStatus = 'validated' | 'pending' | 'experimental';

export interface Study {
  id: string;
  orthancId?: string;
  patientId: string;
  modality: 'CR' | 'CT' | 'MR' | 'DX' | string;
  bodyPart: string;
  studyDate: string;
  receivedAt: string;
  dicomPath: string;
  aiStatus: 'pending' | 'processing' | 'complete' | 'error';
  aiAnalyzedAt?: string;
  // Real DICOM header fields (null when the server could not read them)
  studyInstanceUid?: string | null;
  accessionNumber?: string | null;
  patientName?: string | null;
  patientSex?: string | null;
  patientAge?: string | null;
  patientBirthDate?: string | null;
  studyDescription?: string | null;
  manufacturer?: string | null;
  scannerModel?: string | null;
  numFiles?: number | null;
  seriesDescriptions?: string[];
  /** true when restored from GET /studies (result must be lazily fetched) */
  restored?: boolean;
}

export interface Finding {
  className: string;
  /** Human-readable label from the server (falls back to className) */
  finding: string;
  confidence: number;
  positive: boolean;
  status: FindingStatus;
  detector: string | null;
  sequenceUsed: string | null;
  heatmapBase64: string;
  location: string;
}

export interface ModelIdentity {
  key: string;
  displayName: string;
  source: string | null;
  sha256_12: string | null;
  status: FindingStatus | string;
  validationNote: string | null;
}

export interface OverallAssessment {
  abnormalFlagged: boolean;
  flags: string[];
  text: string;
}

export interface AIResult {
  id: string;
  studyId: string;
  findings: Finding[];
  inferenceTimeMs: number;
  modelVersion: string;
  /** ALWAYS false for the brain panel — the product never certifies normal */
  isNormal: boolean;
  overallImpression: string;
  createdAt: string;
  previewBase64?: string;  // data-URI PNG of the actual analyzed slice (real scan)
  modality?: string;       // real DICOM modality from backend, e.g. 'MR'
  bodyPart?: string;       // resolved body part, e.g. 'BRAIN'
  overallAssessment: OverallAssessment | null;
  disclaimer: string;
  modelIdentity: ModelIdentity[];
  threshold: number | null;
  requiresReview: boolean;
  rejected: boolean;
  rejectionReason: string | null;
  appVersion: string | null;
  reportLanguage?: Lang;
  gemmaAvailable?: boolean;
}

export interface Report {
  id: string;
  studyId: string;
  doctorId: string;
  reportText: string;
  aiDraftText: string;
  pdfPath?: string;
  language: Lang;
  isSigned: boolean;
  signedAt?: string;
  createdAt: string;
  updatedAt: string;
}

/** Server-confirmed signature (POST /report/sign 200) */
export interface SignedReport {
  reportId: string;
  studyId: string;
  language: Lang;
  signedAt: string;
  signer: { id: string; username: string; fullName: string };
  sha256: string;
  modelIdentity: ModelIdentity[];
  reportText: string;
}

export interface User {
  id: string;
  username: string;
  fullName: string;
  role: 'admin' | 'radiologist' | 'technician' | string;
  lastLogin?: string;
}

export interface HealthStatus {
  /** false when /health could not be reached at all */
  reachable: boolean;
  status: 'ok' | 'degraded' | 'error';
  version: string | null;
  device: string | null;
  authRequired: boolean;
  models: Record<string, { loaded: boolean; reason: string | null }>;
  llm: { backend: string | null; reachable: boolean };
  license: { mode: string | null };
  dataDir: string | null;
}

export interface AppSettings {
  orthancUrl: string;
  inferenceUrl: string;
  language: Lang;
  clinicName: string;
  clinicAddress: string;
  /** Printed on the PDF letterhead next to the address. */
  clinicPhone: string;
  supportContact: string;
  theme: 'dark' | 'light';
  /** true once the first-run wizard has been completed on this workstation. */
  setupComplete: boolean;
}

/** GET /license/status */
export interface LicenseStatus {
  reachable: boolean;
  valid: boolean;
  mode: string | null;
  customer: string | null;
  tier: string | null;
  expiresAt: string | null;
  features: string[];
  reason: string | null;
  demoCallsToday: number | null;
  demoLimit: number | null;
}

/** One entry of GET /models/available */
export interface AvailableModel {
  key: string;
  name: string;
  tier: string | null;
  validationStatus: string | null;
  license: string | null;
  is3d: boolean;
  depsOk: boolean;
  depsReason: string | null;
  downloadMb: number | null;
  classes: string[];
}

// Modality icon mapping
export const MODALITY_ICONS: Record<string, string> = {
  CR: 'X-Ray',
  DX: 'X-Ray',
  CT: 'CT',
  MR: 'MRI',
  US: 'Ultra',
  PT: 'PET',
};

// Confidence level thresholds
export const CONFIDENCE = {
  HIGH: 0.8,
  MEDIUM: 0.5,
  LOW: 0.3,
} as const;
