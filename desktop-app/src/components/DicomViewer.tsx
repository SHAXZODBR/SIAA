import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import type { Finding, Study, AIResult, Lang } from '../types';
import { DEMO_MODE, DEMO_STUDIES, getDemoResultForStudy } from '../services/demoData';
import { translateFinding, statusWord } from '../services/findingTranslations';
import { translate, useT, formatDate } from '../i18n';
import { PRODUCT_NAME } from '../services/appInfo';

/**
 * DICOM preview viewer.
 *
 * Shows the REAL slice returned by the server (preview_base64) and, on top of
 * it, the REAL per-finding heatmap (heatmap_base64) when one exists. Nothing is
 * synthesised: no fake anatomy, no bounding boxes guessed from text, no pixel
 * probe, no px/mm scale. Demo drawings exist only in VITE_DEMO_MODE builds.
 */
export default function DicomViewer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    selectedStudyId, showHeatmap, selectedFinding,
    viewerZoom, setViewerZoom,
    panOffset, setPanOffset,
    rotation, invert,
    windowLevel, setWindowLevel,
    aiResults, studies, activeTool, settings,
  } = useAppStore();
  const lang = settings.language;

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [measurements, setMeasurements] = useState<any[]>([]);
  const [measurementInProgress, setMeasurementInProgress] = useState<any>(null);
  // Cache of decoded images (preview + heatmaps), keyed by their data-URI
  const imgCache = useRef<Map<string, HTMLImageElement>>(new Map());
  const [imgTick, setImgTick] = useState(0);

  const getImage = useCallback((uri: string): HTMLImageElement | null => {
    if (!uri) return null;
    const src = uri.startsWith('data:') ? uri : `data:image/png;base64,${uri}`;
    let img = imgCache.current.get(src);
    if (!img) {
      img = new Image();
      img.onload = () => setImgTick((t) => t + 1);
      img.src = src;
      imgCache.current.set(src, img);
    }
    return img.complete && img.naturalWidth > 0 ? img : null;
  }, []);

  const allStudies = studies.length > 0 ? studies : (DEMO_MODE ? DEMO_STUDIES : []);
  const selStudy = allStudies.find((s) => s.id === selectedStudyId) || null;
  const result: AIResult | null = selectedStudyId
    ? (aiResults[selectedStudyId] || getDemoResultForStudy(selStudy))
    : null;

  // ===== Main Image Rendering =====
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = container.clientWidth * dpr;
    canvas.height = container.clientHeight * dpr;
    canvas.style.width = `${container.clientWidth}px`;
    canvas.style.height = `${container.clientHeight}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const w = container.clientWidth;
    const h = container.clientHeight;

    // Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    if (!selectedStudyId) {
      drawEmptyState(ctx, w, h, lang);
      return;
    }

    // Save context for transformations
    ctx.save();
    ctx.translate(w / 2 + panOffset.x, h / 2 + panOffset.y);
    ctx.scale(viewerZoom, viewerZoom);
    ctx.rotate((rotation * Math.PI) / 180);
    ctx.translate(-w / 2, -h / 2);

    // Apply window/level as filter (identity at center 128 / width 256)
    const contrast = windowLevel.width / 256;
    const brightness = (128 - windowLevel.center) / 128;
    ctx.filter = `contrast(${contrast}) brightness(${1 + brightness}) ${invert ? 'invert(1)' : ''}`;

    const previewImg = result?.previewBase64 ? getImage(result.previewBase64) : null;
    if (previewImg) {
      drawScanImage(ctx, w, h, previewImg);
    } else if (DEMO_MODE && selStudy && selStudy.id.startsWith('demo-')) {
      drawDemoPlaceholder(ctx, w, h, lang);
    } else {
      ctx.filter = 'none';
      drawNoPreview(ctx, w, h, lang, !!selStudy?.restored);
    }
    ctx.restore();

    // Real metadata overlays (not affected by transform)
    drawCornerOverlays(ctx, w, h, selStudy, result, lang);
  }, [selectedStudyId, studies, aiResults, imgTick, viewerZoom, panOffset, rotation, invert, windowLevel, lang]);

  // ===== Overlay Rendering (real AI heatmaps + measurements) =====
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = container.clientWidth * dpr;
    canvas.height = container.clientHeight * dpr;
    canvas.style.width = `${container.clientWidth}px`;
    canvas.style.height = `${container.clientHeight}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = container.clientWidth;
    const h = container.clientHeight;
    ctx.clearRect(0, 0, w, h);

    if (!selectedStudyId || !result) {
      drawMeasurements(ctx, measurements, measurementInProgress);
      return;
    }

    const heatFinding = pickHeatmapFinding(result.findings, selectedFinding, showHeatmap);
    const previewImg = result.previewBase64 ? getImage(result.previewBase64) : null;
    const heatImg = heatFinding ? getImage(heatFinding.heatmapBase64) : null;

    if (heatFinding && heatImg) {
      ctx.save();
      ctx.translate(w / 2 + panOffset.x, h / 2 + panOffset.y);
      ctx.scale(viewerZoom, viewerZoom);
      ctx.rotate((rotation * Math.PI) / 180);
      ctx.translate(-w / 2, -h / 2);
      // Align the heatmap to the preview rectangle (same fit), or fit it on its own.
      const rect = fitRect(w, h, previewImg || heatImg);
      ctx.globalAlpha = 0.45;
      ctx.drawImage(heatImg, rect.x, rect.y, rect.w, rect.h);
      ctx.globalAlpha = 1;
      ctx.restore();

      // Label — which finding / detector status this heatmap belongs to
      const label = `${translateFinding(heatFinding.className, lang)} · ${statusWord(heatFinding.status, lang)} · ${Math.round(heatFinding.confidence * 100)}%`;
      ctx.font = '600 11px Inter, sans-serif';
      const tw = ctx.measureText(label).width + 14;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      roundRect(ctx, 8, h - 34, tw, 22, 4);
      ctx.fill();
      ctx.fillStyle = heatFinding.status === 'validated' ? '#4ade80' : heatFinding.status === 'pending' ? '#fbbf24' : '#94a3b8';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, 15, h - 23);
    }

    drawMeasurements(ctx, measurements, measurementInProgress);
  }, [selectedStudyId, aiResults, imgTick, showHeatmap, selectedFinding, viewerZoom, panOffset, rotation, measurements, measurementInProgress, lang]);

  // ===== Mouse Handlers =====
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!selectedStudyId) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    setIsDragging(true);

    if (activeTool === 'length' || activeTool === 'angle') {
      setMeasurementInProgress({ type: activeTool, start: { x, y }, end: { x, y } });
    }
  }, [activeTool, panOffset, selectedStudyId]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (!isDragging) return;

    if (activeTool === 'pan' || !activeTool) {
      setPanOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    } else if (activeTool === 'wwwl') {
      const dx = e.movementX;
      const dy = e.movementY;
      setWindowLevel({
        width: Math.max(1, windowLevel.width + dx * 4),
        center: windowLevel.center + dy * 4,
      });
    } else if (measurementInProgress) {
      setMeasurementInProgress({ ...measurementInProgress, end: { x, y } });
    }
  }, [isDragging, dragStart, activeTool, windowLevel, setPanOffset, setWindowLevel, measurementInProgress]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    if (measurementInProgress) {
      setMeasurements([...measurements, measurementInProgress]);
      setMeasurementInProgress(null);
    }
  }, [measurementInProgress, measurements]);

  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setViewerZoom(Math.max(0.1, Math.min(10, viewerZoom + delta * viewerZoom)));
  }, [viewerZoom, setViewerZoom]);

  const handleDoubleClick = () => {
    setViewerZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Clear measurements when the study changes
  useEffect(() => { setMeasurements([]); setMeasurementInProgress(null); }, [selectedStudyId]);

  const getCursor = () => {
    if (!selectedStudyId) return 'default';
    if (activeTool === 'pan' || !activeTool) return isDragging ? 'grabbing' : 'grab';
    if (activeTool === 'zoom') return 'zoom-in';
    if (activeTool === 'wwwl') return 'ns-resize';
    return 'crosshair';
  };

  return (
    <div
      ref={containerRef}
      className="h-full w-full relative bg-black select-none"
      style={{ cursor: getCursor() }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
    >
      {/* Main image canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 dicom-canvas" />

      {/* Overlay canvas for AI heatmap + measurements */}
      <canvas ref={overlayCanvasRef} className="absolute inset-0 pointer-events-none" />

      {/* AI status indicator (top center) — only when a study is selected */}
      {selectedStudyId && <AIStatusBadge result={result} lang={lang} />}
    </div>
  );
}

// ===== Helper: AI Status Badge =====
function AIStatusBadge({ result }: { result: AIResult | null; lang: Lang }) {
  const t = useT();
  if (!result) return null;

  const review = result.rejected || result.requiresReview;
  const flagged = result.overallAssessment?.abnormalFlagged ?? result.findings.some((f) => f.positive);
  const validatedFlag = result.findings.some((f) => f.positive && f.status === 'validated');

  const dot = review ? 'bg-moderate' : flagged ? (validatedFlag ? 'bg-critical shadow-glow-critical' : 'bg-moderate') : 'bg-ink-500';
  const text = review ? t('viewer.notAnalyzed') : flagged ? t('viewer.abnormalFlagged') : t('viewer.noFindingFlagged');

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 glass px-3 py-1.5 rounded-full border border-ink-700/50 animate-slide-down max-w-[80%]">
      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${dot} ${flagged && !review ? 'animate-pulse' : ''}`} />
      <span className="text-xs font-medium text-ink-100 truncate">{text}</span>
      {result.inferenceTimeMs > 0 && (
        <span className="text-[10px] text-ink-500 font-mono flex-shrink-0">{result.inferenceTimeMs}ms</span>
      )}
    </div>
  );
}

// ===== Selection of the heatmap to show =====
function pickHeatmapFinding(findings: Finding[], selected: Finding | null, showHeatmap: boolean): Finding | null {
  if (!showHeatmap) return null;
  if (selected) {
    const match = findings.find((f) => f.className === selected.className && f.detector === selected.detector);
    return match && match.heatmapBase64 ? match : null;
  }
  const candidates = findings.filter((f) => f.positive && f.heatmapBase64);
  if (candidates.length === 0) return null;
  return candidates.reduce((a, b) => (b.confidence > a.confidence ? b : a));
}

// ===== Drawing Functions =====

function fitRect(w: number, h: number, img: HTMLImageElement) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const scale = (Math.min(w, h) * 0.94) / Math.max(iw, ih, 1);
  const dw = iw * scale;
  const dh = ih * scale;
  return { x: (w - dw) / 2, y: (h - dh) / 2, w: dw, h: dh };
}

function drawEmptyState(ctx: CanvasRenderingContext2D, w: number, h: number, lang: Lang) {
  // Grid pattern
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const cx = w / 2;
  const cy = h / 2;

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy - 20, 24, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - 8, cy - 20);
  ctx.lineTo(cx + 8, cy - 20);
  ctx.moveTo(cx, cy - 28);
  ctx.lineTo(cx, cy - 12);
  ctx.stroke();

  ctx.fillStyle = '#64748b';
  ctx.font = '500 14px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(translate(lang, 'viewer.selectStudy'), cx, cy + 20);

  ctx.fillStyle = '#475569';
  ctx.font = '400 11px Inter, sans-serif';
  ctx.fillText(translate(lang, 'viewer.dropHint'), cx, cy + 40);
}

function drawNoPreview(ctx: CanvasRenderingContext2D, w: number, h: number, lang: Lang, restored: boolean) {
  ctx.fillStyle = '#64748b';
  ctx.font = '500 13px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(translate(lang, 'viewer.noImagePreview'), w / 2, h / 2);
  if (restored) {
    ctx.fillStyle = '#475569';
    ctx.font = '400 11px Inter, sans-serif';
    ctx.fillText(translate(lang, 'viewer.restoredHint'), w / 2, h / 2 + 20);
  }
}

/** Draw a REAL scan image (the actual analyzed slice), fit to the viewport on black. */
function drawScanImage(ctx: CanvasRenderingContext2D, w: number, h: number, img: HTMLImageElement) {
  const r = fitRect(w, h, img);
  if (!r.w || !r.h) return;
  ctx.drawImage(img, r.x, r.y, r.w, r.h);
}

/** Demo builds only: a clearly-labelled placeholder, not a fake anatomy render. */
function drawDemoPlaceholder(ctx: CanvasRenderingContext2D, w: number, h: number, lang: Lang) {
  const cx = w / 2, cy = h / 2;
  const r = Math.min(w, h) * 0.35;
  ctx.fillStyle = '#111827';
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = '#64748b';
  ctx.font = '700 16px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(translate(lang, 'viewer.demo'), cx, cy);
}

function drawCornerOverlays(ctx: CanvasRenderingContext2D, w: number, h: number, study: Study | null, result: AIResult | null, lang: Lang) {
  ctx.font = '500 10px JetBrains Mono, monospace';
  ctx.fillStyle = '#64748b';
  ctx.textBaseline = 'alphabetic';

  // Top-left: patient (real fields only; skip unknowns)
  ctx.textAlign = 'left';
  const left: string[] = [];
  if (study?.patientId) left.push(study.patientId);
  if (study?.patientName) left.push(study.patientName);
  const demo: string[] = [];
  if (study?.patientSex) demo.push(study.patientSex);
  if (study?.patientAge) demo.push(study.patientAge);
  if (demo.length) left.push(demo.join(' · '));
  if (study?.patientBirthDate) left.push(`${translate(lang, 'viewer.dob')}: ${formatDate(study.patientBirthDate, lang)}`);
  left.forEach((line, i) => ctx.fillText(line, 8, 14 + i * 12));

  // Top-right: study
  ctx.textAlign = 'right';
  const right: string[] = [];
  if (study) right.push(`${study.modality}${study.bodyPart && study.bodyPart !== '—' ? ` — ${study.bodyPart}` : ''}`);
  if (study?.studyDate) right.push(formatDate(study.studyDate, lang));
  if (study?.studyDescription) right.push(study.studyDescription);
  if (study?.accessionNumber) right.push(`${translate(lang, 'viewer.acc')} ${study.accessionNumber}`);
  if (study?.numFiles) right.push(translate(lang, 'viewer.files', { count: study.numFiles }));
  right.forEach((line, i) => ctx.fillText(line, w - 8, 14 + i * 12));

  // Bottom-right: models actually used (identity from the server)
  ctx.textAlign = 'right';
  ctx.fillText(PRODUCT_NAME.toUpperCase(), w - 8, h - 14);
  if (result?.modelIdentity?.length) {
    ctx.fillStyle = '#3b82f6';
    const models = result.modelIdentity.slice(0, 3);
    models.forEach((m, i) => {
      const line = `${m.displayName}${m.sha256_12 ? ` ${m.sha256_12}` : ''} · ${statusWord(m.status, lang)}`;
      ctx.fillText(line, w - 8, h - 26 - (models.length - 1 - i) * 12);
    });
    ctx.fillStyle = '#64748b';
  }
}

function drawMeasurements(ctx: CanvasRenderingContext2D, measurements: any[], inProgress: any) {
  const all = [...measurements];
  if (inProgress) all.push(inProgress);

  for (const m of all) {
    if (m.type === 'length') {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(m.start.x, m.start.y);
      ctx.lineTo(m.end.x, m.end.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(m.start.x, m.start.y, 3, 0, Math.PI * 2);
      ctx.arc(m.end.x, m.end.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#3b82f6';
      ctx.fill();

      // Screen pixels — no pixel spacing is available for the preview, so never claim mm.
      const dx = m.end.x - m.start.x;
      const dy = m.end.y - m.start.y;
      const length = Math.sqrt(dx * dx + dy * dy);
      const mx = (m.start.x + m.end.x) / 2;
      const my = (m.start.y + m.end.y) / 2;
      ctx.font = '600 11px JetBrains Mono, monospace';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`${Math.round(length)} px`, mx + 8, my - 4);
    }
  }
}

// ===== Helpers =====

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
