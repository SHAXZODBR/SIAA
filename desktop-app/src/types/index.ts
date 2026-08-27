// ===== Core Types for Sentinel Medical AI =====

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
}

export interface Finding {
  className: string;
  confidence: number;
  heatmapBase64: string;
  location: string;
}

export interface AIResult {
  id: string;
  studyId: string;
  findings: Finding[];
  inferenceTimeMs: number;
  modelVersion: string;
  isNormal: boolean;
  overallImpression: string;
  createdAt: string;
}

export interface Report {
  id: string;
  studyId: string;
  doctorId: string;
  reportText: string;
  aiDraftText: string;
  pdfPath?: string;
  language: 'ru' | 'uz' | 'en';
  isSigned: boolean;
  signedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface User {
  id: string;
  username: string;
  fullName: string;
  role: 'admin' | 'radiologist' | 'technician';
  lastLogin?: string;
}

export interface AppSettings {
  orthancUrl: string;
  inferenceUrl: string;
  storagePath: string;
  language: 'ru' | 'uz' | 'en';
  autoAnalyze: boolean;
  refreshInterval: number;
  theme: 'dark' | 'light';
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
