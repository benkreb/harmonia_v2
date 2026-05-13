import { useEffect, useState } from "react";
import type { CuratedSample } from "../../data/types";
import { SamplePanelGrid } from "./SamplePanelGrid";
import { SampleThumbnailStrip } from "./SampleThumbnailStrip";

interface SampleBrowserProps {
  samples: CuratedSample[];
}

export function SampleBrowser({ samples }: SampleBrowserProps) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const sample = samples[currentIdx];

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "j") {
        event.preventDefault();
        setCurrentIdx((idx) => Math.min(samples.length - 1, idx + 1));
      } else if (event.key === "ArrowLeft" || event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCurrentIdx((idx) => Math.max(0, idx - 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [samples.length]);

  return (
    <div className="flex h-full flex-col gap-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-4xl font-bold tracking-tight">See the predictions for yourself</h2>
          <p className="mt-1 text-lg text-slate-600">15 curated samples - easy, typical, and hard cases by C1 Dice.</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="rounded-full border border-slate-200 bg-white px-4 py-2 font-semibold shadow-sm" onClick={() => setCurrentIdx((idx) => Math.max(0, idx - 1))}>
            Prev
          </button>
          <span className="font-mono text-sm text-slate-500">
            {currentIdx + 1}/{samples.length}
          </span>
          <button className="rounded-full bg-blue-600 px-4 py-2 font-semibold text-white shadow-sm" onClick={() => setCurrentIdx((idx) => Math.min(samples.length - 1, idx + 1))}>
            Next
          </button>
        </div>
      </div>

      <SamplePanelGrid sample={sample} />
      <SampleThumbnailStrip samples={samples} currentIdx={currentIdx} onSelect={setCurrentIdx} />
      <div className="text-center text-sm text-slate-500">
        Patient {sample.patient_id} · {sample.pathology} · Density {sample.breast_density ?? "n/a"} · Category {sample.category}
      </div>
    </div>
  );
}
