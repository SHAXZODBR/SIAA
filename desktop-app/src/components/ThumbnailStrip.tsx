import React from 'react';
import { useAppStore } from '../store/appStore';

/**
 * Thumbnail strip at the bottom — shows series/slices in current study.
 * Used for CT/MRI studies with multiple slices.
 */
export default function ThumbnailStrip() {
  const { thumbnailStripOpen, selectedStudyId } = useAppStore();

  if (!thumbnailStripOpen || !selectedStudyId) return null;

  // Demo: 12 thumbnails
  const thumbnails = Array.from({ length: 12 }, (_, i) => i);
  const [activeIdx, setActiveIdx] = React.useState(5);

  return (
    <div className="h-24 bg-ink-950 border-t border-ink-800 flex items-center px-3 gap-2 overflow-x-auto animate-slide-up">
      <div className="flex-shrink-0 pr-3 border-r border-ink-800 h-full flex flex-col justify-center">
        <div className="text-[9px] text-ink-500 uppercase tracking-wider font-semibold">Series</div>
        <div className="text-sm font-mono text-ink-200">Axial T1</div>
        <div className="text-[10px] text-ink-500">12 slices</div>
      </div>

      {thumbnails.map(i => (
        <button
          key={i}
          onClick={() => setActiveIdx(i)}
          className={`flex-shrink-0 w-16 h-16 relative rounded overflow-hidden transition-all ${
            i === activeIdx
              ? 'ring-2 ring-accent-500 scale-105'
              : 'ring-1 ring-ink-800 hover:ring-ink-600 opacity-60 hover:opacity-100'
          }`}
        >
          <div className="w-full h-full bg-gradient-to-br from-ink-700 to-ink-900 flex items-center justify-center">
            <div className="w-10 h-10 rounded-full bg-ink-800/50" />
          </div>
          <div className="absolute bottom-0 inset-x-0 bg-ink-950/80 px-1 py-0.5">
            <div className="text-[9px] font-mono text-center text-ink-300">{i + 1}/12</div>
          </div>
        </button>
      ))}

      <div className="flex-1" />

      <div className="flex items-center gap-1 flex-shrink-0 pl-3 border-l border-ink-800 h-full">
        <button className="tool-btn" title="Previous series">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
        <button className="tool-btn" title="Next series">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  );
}
