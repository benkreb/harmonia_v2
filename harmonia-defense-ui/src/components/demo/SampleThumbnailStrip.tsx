import type { CuratedSample } from "../../data/types";
import { sampleImage } from "../../utils/formatters";

interface SampleThumbnailStripProps {
  samples: CuratedSample[];
  currentIdx: number;
  onSelect: (idx: number) => void;
}

export function SampleThumbnailStrip({ samples, currentIdx, onSelect }: SampleThumbnailStripProps) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-center gap-3">
        {samples.map((sample, idx) => (
          <button
            key={sample.sample_idx}
            onClick={() => onSelect(idx)}
            className={`relative h-16 w-16 overflow-hidden rounded-xl border transition ${
              idx === currentIdx ? "scale-110 border-blue-600 ring-2 ring-blue-600" : "border-slate-200 opacity-80 hover:opacity-100"
            }`}
          >
            <img src={sampleImage(sample.sample_idx, "gt")} alt={`sample ${sample.sample_idx}`} className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-3 text-center text-xs font-semibold uppercase tracking-widest text-slate-400">
        <span>Easy cases</span>
        <span>Typical</span>
        <span>Hard cases</span>
      </div>
    </div>
  );
}
