import React from 'react';
import { useAppStore } from '../store/appStore';
import { useT } from '../i18n';

/**
 * Series strip at the bottom — lists the REAL series descriptions the server
 * read from the uploaded study. Per-slice thumbnails are not available from
 * the API, so nothing is fabricated: the strip stays hidden until real series
 * information exists.
 */
export default function ThumbnailStrip() {
  const { thumbnailStripOpen, selectedStudyId, studies } = useAppStore();
  const t = useT();
  const study = studies.find((s) => s.id === selectedStudyId);
  const series = study?.seriesDescriptions || [];

  if (!thumbnailStripOpen || !study || series.length === 0) return null;

  return (
    <div className="h-16 bg-ink-950 border-t border-ink-800 flex items-center px-3 gap-2 overflow-x-auto animate-slide-up">
      <div className="flex-shrink-0 pr-3 border-r border-ink-800 h-full flex flex-col justify-center">
        <div className="text-[9px] text-ink-500 uppercase tracking-wider font-semibold">{t('series.title')}</div>
        <div className="text-[10px] text-ink-500">
          {series.length}{study.numFiles ? ` · ${t('series.files', { count: study.numFiles })}` : ''}
        </div>
      </div>

      {series.map((name, i) => (
        <div
          key={`${name}-${i}`}
          className="flex-shrink-0 px-2 py-1 rounded bg-ink-900 border border-ink-800 text-[10px] font-mono text-ink-300 max-w-[180px] truncate"
          title={name}
        >
          {name || t('series.unnamed', { n: i + 1 })}
        </div>
      ))}
    </div>
  );
}
