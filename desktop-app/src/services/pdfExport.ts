/**
 * PDF Report Export for Sentinel Medical AI.
 *
 * Strategy:
 *   - Build a clean HTML document with proper Unicode fonts (Inter / system).
 *   - Hand it to Electron's main process which renders to PDF via Chromium's
 *     native printToPDF. Chromium has full Unicode support — no garbled
 *     Cyrillic / Uzbek text.
 *   - Browser fallback (when not in Electron): open print dialog.
 *
 * Honesty rules baked into the layout:
 *   - A PDF without a SERVER-confirmed signature carries a visible
 *     "AI DRAFT — NOT SIGNED" line (the caller also suffixes the filename).
 *   - Model identity (display name + sha256 + validation status), the decision
 *     threshold and the app version are printed on every report.
 *   - Finding colours follow the urgency rule (positive + validated → review),
 *     never the raw confidence value.
 *   - Letterhead comes from Settings; nothing about the clinic is invented.
 */

import type { Finding, ModelIdentity, Lang } from '../types';
import { translateFinding, translateLocation, findingUrgency, statusWord } from './findingTranslations';
import { translate } from '../i18n';
import { PRODUCT_NAME, VENDOR_NAME, VENDOR_SITE } from './appInfo';

export interface PDFReportData {
  patientId: string;
  patientName?: string | null;
  studyDate: string;
  accessionNumber?: string | null;
  studyInstanceUid?: string | null;
  modality: string;
  bodyPart: string;
  reportText: string;
  /** Printed on the signature line: the server-confirmed signer, or the logged-in doctor for drafts. */
  doctorName: string;
  clinicName: string;
  clinicAddress: string;
  /** Server-confirmed signature timestamp. Undefined ⇒ the PDF is an AI draft. */
  signedAt?: string;
  sha256?: string;
  findings: Finding[];
  modelIdentity?: ModelIdentity[];
  threshold?: number | null;
  appVersion?: string;
  /** This language is not the one the report was originally produced in. */
  autoTranslated?: boolean;
  disclaimer?: string;
  language?: Lang;
}

/** Per-language part of a multi-language export. */
export type PDFLanguageSection = Pick<PDFReportData, 'reportText' | 'doctorName' | 'signedAt' | 'sha256' | 'autoTranslated'>;

/** Study-level part shared by every language section. */
export type PDFBaseData = Omit<PDFReportData, keyof PDFLanguageSection | 'language'>;

export interface PDFExportResult {
  success: boolean;
  filePath?: string;
  canceled?: boolean;
  error?: string;
}

const LANG_ORDER: Lang[] = ['ru', 'uz', 'en'];
const LANG_MARKERS: Record<Lang, string> = { ru: 'РУССКИЙ', uz: "O'ZBEK", en: 'ENGLISH' };
const LOCALES: Record<Lang, string> = { ru: 'ru-RU', uz: 'uz-UZ', en: 'en-US' };

// Translations for static PDF labels
const LABELS = {
  ru: {
    patientId: 'ID пациента',
    patientName: 'Пациент',
    studyDate: 'Дата исследования',
    accession: 'Направление',
    studyUid: 'Study UID',
    modality: 'Модальность',
    bodyPart: 'Область',
    aiFindings: 'РЕЗУЛЬТАТЫ ИИ-АНАЛИЗА',
    finding: 'Находка',
    status: 'Статус',
    confidence: 'Достоверность',
    location: 'Локализация',
    flagged: 'отмечено',
    notFlagged: 'не отмечено',
    notSigned: 'НЕ ПОДПИСАНО',
    signed: 'ПОДПИСАНО',
    signedBy: 'Подписал',
    hash: 'SHA-256',
    radiologist: 'Врач-рентгенолог',
    models: 'Модели ИИ',
    threshold: 'Порог решения',
    appVersion: 'Версия приложения',
    notReported: 'не указано',
    generatedBy: 'Создано системой',
    advisory: 'результаты ИИ носят рекомендательный характер и требуют подтверждения врачом.',
  },
  uz: {
    patientId: 'Bemor ID',
    patientName: 'Bemor',
    studyDate: 'Tekshiruv sanasi',
    accession: "Yo'llanma",
    studyUid: 'Study UID',
    modality: 'Modallik',
    bodyPart: 'Soha',
    aiFindings: 'AI-TAHLIL NATIJALARI',
    finding: 'Topilma',
    status: 'Holat',
    confidence: 'Ishonchlilik',
    location: 'Joylashuv',
    flagged: 'belgilangan',
    notFlagged: 'belgilanmagan',
    notSigned: 'IMZOLANMAGAN',
    signed: 'IMZOLANGAN',
    signedBy: 'Imzolagan',
    hash: 'SHA-256',
    radiologist: 'Rentgenolog shifokor',
    models: 'AI modellari',
    threshold: 'Qaror chegarasi',
    appVersion: 'Ilova versiyasi',
    notReported: "ko'rsatilmagan",
    generatedBy: 'Tizim tomonidan yaratilgan',
    advisory: "AI natijalari tavsiya xarakteriga ega va shifokor tasdig'ini talab qiladi.",
  },
  en: {
    patientId: 'Patient ID',
    patientName: 'Patient',
    studyDate: 'Study date',
    accession: 'Accession',
    studyUid: 'Study UID',
    modality: 'Modality',
    bodyPart: 'Body part',
    aiFindings: 'AI ANALYSIS FINDINGS',
    finding: 'Finding',
    status: 'Status',
    confidence: 'Confidence',
    location: 'Location',
    flagged: 'flagged',
    notFlagged: 'not flagged',
    notSigned: 'NOT SIGNED',
    signed: 'SIGNED',
    signedBy: 'Signed by',
    hash: 'SHA-256',
    radiologist: 'Radiologist',
    models: 'AI models',
    threshold: 'Decision threshold',
    appVersion: 'App version',
    notReported: 'not reported',
    generatedBy: 'Generated by',
    advisory: 'AI results are advisory and require physician confirmation.',
  },
} as const;

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatDateTime(iso: string, lang: Lang): string {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(LOCALES[lang]);
}

/** Safe file-name fragment: letters, digits, dot, dash, underscore only. */
export function sanitizeFilePart(value: string): string {
  const cleaned = (value || '').replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
  return cleaned || 'x';
}

function footerText(appVersion: string | undefined, lang: Lang): string {
  const L = LABELS[lang];
  const product = `${PRODUCT_NAME}${appVersion ? ` v${appVersion}` : ''}`;
  return `${L.generatedBy} ${product} · ${VENDOR_NAME} · ${VENDOR_SITE} — ${L.advisory}`;
}

function buildSinglePagePDFHTML(data: PDFReportData): string {
  const lang: Lang = data.language || 'ru';
  const L = LABELS[lang];
  const reportLines = data.reportText.split('\n');

  // Format report — detect headers (uppercase ending with ':')
  const reportHTML = reportLines
    .map((rawLine) => {
      const line = rawLine.trim();
      if (!line) return '<div class="spacer"></div>';
      const isHeader =
        line === line.toUpperCase() && line.length > 3 && line.endsWith(':');
      if (isHeader) {
        return `<h3 class="section-header">${escapeHtml(line)}</h3>`;
      }
      // Separator lines like "---"
      if (/^-{3,}$/.test(line)) {
        return '<hr class="report-hr"/>';
      }
      return `<p class="report-para">${escapeHtml(line)}</p>`;
    })
    .join('\n');

  // Findings list — colour by urgency (positive + validated → review), never by the score itself
  const findingsHTML =
    data.findings && data.findings.length > 0
      ? `
      <div class="findings-block">
        <h2 class="findings-title">${L.aiFindings}</h2>
        <table class="findings-table">
          <tr class="findings-head">
            <th>${L.finding}</th><th>${L.status}</th><th></th><th>${L.confidence}</th><th>${L.location}</th>
          </tr>
          ${data.findings
            .map((f) => {
              const conf = Math.round(f.confidence * 100);
              const urgency = findingUrgency(f);
              const color = urgency === 'review' ? '#dc2626' : urgency === 'unvalidated' ? '#d97706' : '#94a3b8';
              return `
                <tr>
                  <td class="finding-name">${escapeHtml(translateFinding(f.className, lang))}
                    <span class="finding-flag" style="color:${color};">· ${f.positive ? L.flagged : L.notFlagged}</span></td>
                  <td class="finding-status">${escapeHtml(statusWord(f.status, lang))}</td>
                  <td class="finding-bar-cell">
                    <div class="bar-bg">
                      <div class="bar-fill" style="width:${conf}%;background:${color};"></div>
                    </div>
                  </td>
                  <td class="finding-conf">${conf}%</td>
                  <td class="finding-loc">${escapeHtml(translateLocation(f.location || '', lang))}</td>
                </tr>`;
            })
            .join('')}
        </table>
      </div>`
      : '';

  const models = data.modelIdentity || [];
  const modelsHTML = `
    <section class="meta-block">
      <div class="meta-title">${L.models}</div>
      ${models.length === 0
        ? `<div class="meta-row">${L.notReported}</div>`
        : models
            .map(
              (m) => `<div class="meta-row"><span class="meta-name">${escapeHtml(m.displayName)}</span>
                 <span class="mono">${escapeHtml(m.sha256_12 || '—')}</span>
                 <span class="meta-status">${escapeHtml(statusWord(m.status, lang))}</span></div>`,
            )
            .join('')}
      <div class="meta-row">${L.threshold}: <span class="mono">${data.threshold != null ? data.threshold.toFixed(2) : L.notReported}</span>
        &nbsp;·&nbsp; ${L.appVersion}: <span class="mono">${escapeHtml(data.appVersion || L.notReported)}</span></div>
    </section>`;

  const draftBanner = data.signedAt
    ? ''
    : `<div class="draft-banner">${escapeHtml(translate(lang, 'report.draftWatermark'))}</div>`;
  const translatedBanner = data.autoTranslated
    ? `<div class="translated-banner">${escapeHtml(translate(lang, 'report.autoTranslated'))}</div>`
    : '';
  const disclaimerHTML = data.disclaimer
    ? `<p class="disclaimer">${escapeHtml(data.disclaimer)}</p>`
    : '';

  const sigBlock = data.signedAt
    ? `<div class="sig signed">
         <div class="sig-line"></div>
         <div class="sig-name">${escapeHtml(data.doctorName)}</div>
         <div class="sig-status">✓ ${L.signed}: ${escapeHtml(formatDateTime(data.signedAt, lang))}</div>
         ${data.sha256 ? `<div class="sig-hash">${L.hash}: <span class="mono">${escapeHtml(data.sha256)}</span></div>` : ''}
       </div>`
    : `<div class="sig">
         <div class="sig-line"></div>
         <div class="sig-name">${escapeHtml(data.doctorName || L.radiologist)}</div>
         <div class="sig-status not-signed">${L.notSigned} — ${escapeHtml(translate(lang, 'report.draftWatermark'))}</div>
       </div>`;

  const optionalRow = (label: string, value: string | null | undefined, mono = false) =>
    value
      ? `<div class="pi"><span class="pi-k">${label}:</span> <span class="pi-v${mono ? ' mono' : ''}">${escapeHtml(value)}</span></div>`
      : '';

  return `
    <header class="letterhead">
      <div class="lh-left">
        <div class="brand">${escapeHtml(PRODUCT_NAME)}</div>
        <div class="clinic">${escapeHtml(data.clinicName)}</div>
        ${data.clinicAddress ? `<div class="clinic-addr">${escapeHtml(data.clinicAddress)}</div>` : ''}
      </div>
      <div class="lh-right">
        <div>${L.patientId}: <strong>${escapeHtml(data.patientId)}</strong></div>
        <div>${L.studyDate}: <strong>${escapeHtml(data.studyDate)}</strong></div>
        <div>${escapeHtml(data.modality)} — ${escapeHtml(data.bodyPart)}</div>
      </div>
    </header>

    ${draftBanner}
    ${translatedBanner}

    <section class="patient-box">
      <div class="pi"><span class="pi-k">${L.patientId}:</span> <span class="pi-v">${escapeHtml(data.patientId)}</span></div>
      <div class="pi"><span class="pi-k">${L.studyDate}:</span> <span class="pi-v">${escapeHtml(data.studyDate)}</span></div>
      ${optionalRow(L.patientName, data.patientName)}
      ${optionalRow(L.accession, data.accessionNumber)}
      <div class="pi"><span class="pi-k">${L.modality}:</span> <span class="pi-v">${escapeHtml(data.modality)}</span></div>
      <div class="pi"><span class="pi-k">${L.bodyPart}:</span> <span class="pi-v">${escapeHtml(data.bodyPart)}</span></div>
      ${optionalRow(L.studyUid, data.studyInstanceUid, true)}
    </section>

    ${findingsHTML}

    <section class="report-body">
      ${reportHTML}
    </section>

    ${disclaimerHTML}
    ${modelsHTML}

    ${sigBlock}
  `;
}

function wrapInDocument(bodyContent: string, footer: string, lang: Lang): string {
  // Self-contained HTML doc with embedded CSS.
  // Uses system fonts that have full Cyrillic + Uzbek Latin coverage on
  // every modern OS (San Francisco on macOS, Segoe UI on Windows).
  return `<!DOCTYPE html>
<html lang="${lang}">
<head>
<meta charset="utf-8" />
<title>${escapeHtml(PRODUCT_NAME)} Report</title>
<style>
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", "Liberation Sans", "DejaVu Sans", Arial, sans-serif;
    color: #0f172a;
    font-size: 10pt;
    line-height: 1.45;
  }
  body { padding: 0 18mm 12mm 18mm; }
  .mono { font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 8.5pt; }

  .letterhead {
    background: #0f172a;
    color: #fff;
    margin: -8mm -18mm 6mm -18mm;
    padding: 8mm 18mm;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .brand { font-size: 14pt; font-weight: 700; letter-spacing: 0.5px; }
  .clinic { font-size: 9pt; opacity: 0.95; margin-top: 2pt; }
  .clinic-addr { font-size: 8pt; opacity: 0.8; }
  .lh-right { text-align: right; font-size: 8.5pt; line-height: 1.5; }

  .draft-banner {
    border: 2px solid #dc2626;
    color: #dc2626;
    background: #fef2f2;
    text-align: center;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4pt 8pt;
    margin-bottom: 6pt;
    font-size: 10pt;
  }
  .translated-banner {
    border: 1px solid #d97706;
    color: #92400e;
    background: #fffbeb;
    text-align: center;
    font-weight: 600;
    padding: 3pt 8pt;
    margin-bottom: 6pt;
    font-size: 8.5pt;
  }

  .patient-box {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5pt 8pt;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4pt 16pt;
    margin-bottom: 8pt;
    font-size: 9pt;
  }
  .pi-k { color: #475569; font-weight: 600; }
  .pi-v { color: #0f172a; word-break: break-all; }

  .findings-block { margin: 6pt 0 10pt 0; }
  .findings-title {
    font-size: 10pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 4pt 0;
    padding-bottom: 2pt;
    border-bottom: 2px solid #0f172a;
    letter-spacing: 0.3px;
  }
  .findings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
  }
  .findings-table tr { border-bottom: 1px solid #f1f5f9; }
  .findings-table td, .findings-table th { padding: 3pt 4pt; vertical-align: middle; }
  .findings-head th { text-align: left; font-size: 7.5pt; color: #64748b; text-transform: uppercase; letter-spacing: 0.3px; }
  .finding-name { font-weight: 600; width: 30%; }
  .finding-flag { font-weight: 500; font-size: 8pt; }
  .finding-status { width: 18%; font-size: 8.5pt; color: #334155; }
  .finding-bar-cell { width: 22%; }
  .bar-bg { background: #e2e8f0; height: 5pt; border-radius: 2pt; overflow: hidden; }
  .bar-fill { height: 100%; }
  .finding-conf { width: 9%; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .finding-loc { width: 21%; color: #64748b; font-size: 8.5pt; }

  .report-body { margin-top: 6pt; }
  .section-header {
    font-size: 10pt;
    font-weight: 700;
    color: #0f172a;
    margin: 8pt 0 2pt 0;
    letter-spacing: 0.3px;
  }
  .report-para { margin: 2pt 0; text-align: justify; }
  .report-hr { border: none; border-top: 1px dashed #cbd5e1; margin: 6pt 0; }
  .spacer { height: 3pt; }

  .disclaimer { font-size: 8pt; color: #475569; font-style: italic; margin: 8pt 0 0 0; }

  .meta-block {
    margin-top: 8pt;
    padding: 4pt 8pt;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    font-size: 8pt;
    color: #334155;
  }
  .meta-title { font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; color: #64748b; font-size: 7.5pt; margin-bottom: 2pt; }
  .meta-row { margin: 1pt 0; }
  .meta-name { font-weight: 600; }
  .meta-status { color: #64748b; }

  .sig { margin-top: 12pt; }
  .sig-line { border-top: 1px solid #475569; width: 60mm; margin-bottom: 3pt; }
  .sig-name { font-weight: 700; font-size: 10pt; }
  .sig-status { color: #64748b; font-size: 8.5pt; margin-top: 1pt; }
  .sig.signed .sig-status { color: #16a34a; font-weight: 600; }
  .sig-status.not-signed { color: #dc2626; font-weight: 600; }
  .sig-hash { font-size: 7.5pt; color: #475569; margin-top: 1pt; word-break: break-all; }

  .page-footer {
    position: fixed;
    bottom: 4mm;
    left: 18mm;
    right: 18mm;
    font-size: 7pt;
    color: #94a3b8;
    text-align: center;
    border-top: 1px solid #e2e8f0;
    padding-top: 2pt;
  }

  /* Multi-language separator */
  .lang-page-break { page-break-after: always; }
  .lang-marker {
    background: #1e293b;
    color: #fff;
    display: inline-block;
    padding: 2pt 8pt;
    border-radius: 3pt;
    font-size: 8pt;
    font-weight: 700;
    margin-bottom: 4pt;
    letter-spacing: 0.5px;
  }

  @page { size: A4; margin: 8mm 0; }
  @media print {
    body { padding: 0 18mm 12mm 18mm; }
  }
</style>
</head>
<body>
${bodyContent}
<footer class="page-footer">${escapeHtml(footer)}</footer>
</body>
</html>`;
}

/**
 * Build the full HTML for a single-language PDF.
 */
export function buildPDFHTML(data: PDFReportData): string {
  const lang: Lang = data.language || 'ru';
  return wrapInDocument(buildSinglePagePDFHTML(data), footerText(data.appVersion, lang), lang);
}

/**
 * Build the full HTML for a multi-language PDF — one section per language the
 * doctor actually opened. Each section carries its own signature state and
 * auto-translation tag.
 */
export function buildMultiLanguagePDFHTML(
  base: PDFBaseData,
  sections: Partial<Record<Lang, PDFLanguageSection>>,
): string {
  const langs = LANG_ORDER.filter((l) => !!sections[l]?.reportText);
  const parts = langs.map((lang, idx) => {
    const bodyHtml = buildSinglePagePDFHTML({ ...base, ...sections[lang]!, language: lang });
    const breakClass = idx < langs.length - 1 ? 'lang-page-break' : '';
    return `
        <div class="lang-section ${breakClass}">
          <div style="text-align:center;margin:4pt 0;">
            <span class="lang-marker">${LANG_MARKERS[lang]}</span>
          </div>
          ${bodyHtml}
        </div>`;
  });
  const footerLang: Lang = langs[0] || 'ru';
  return wrapInDocument(parts.join('\n'), footerText(base.appVersion, footerLang), footerLang);
}

async function renderHtmlToPdf(html: string, filename: string): Promise<PDFExportResult> {
  // Electron path — native PDF rendering with full Cyrillic/Uzbek support
  const electronAPI = window.electronAPI;
  if (electronAPI?.exportPDF) {
    return await electronAPI.exportPDF(html, filename);
  }

  // Browser fallback — open print preview
  const w = window.open('', '_blank');
  if (!w) return { success: false, error: 'Popup blocked' };
  w.document.write(html);
  w.document.close();
  setTimeout(() => {
    w.focus();
    w.print();
  }, 400);
  return { success: true };
}

/**
 * Trigger the PDF download. In Electron, uses native printToPDF (full Unicode).
 * In a normal browser, opens the print dialog with the styled HTML.
 */
export async function downloadPDF(data: PDFReportData, filename?: string): Promise<PDFExportResult> {
  const safeName =
    filename ||
    `${sanitizeFilePart(data.patientId)}_${sanitizeFilePart(data.studyDate)}_${data.language || 'ru'}${data.signedAt ? '' : '_DRAFT'}.pdf`;
  return renderHtmlToPdf(buildPDFHTML(data), safeName);
}

/**
 * Download one PDF containing every opened language, one section each.
 */
export async function downloadMultiLanguagePDF(
  base: PDFBaseData,
  sections: Partial<Record<Lang, PDFLanguageSection>>,
  filename?: string,
): Promise<PDFExportResult> {
  const safeName =
    filename || `${sanitizeFilePart(base.patientId)}_${sanitizeFilePart(base.studyDate)}_multilang_DRAFT.pdf`;
  return renderHtmlToPdf(buildMultiLanguagePDFHTML(base, sections), safeName);
}
