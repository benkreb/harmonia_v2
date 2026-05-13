import { configOrder } from "../../data/configMeta";
import type { CuratedSample } from "../../data/types";
import { sampleImage } from "../../utils/formatters";
import { ConfigChip } from "../atoms/ConfigChip";
import { DiceBadge } from "./DiceBadge";

interface SamplePanelGridProps {
  sample: CuratedSample;
}

export function SamplePanelGrid({ sample }: SamplePanelGridProps) {
  const panels = [
    { key: "input", label: "Input", kind: "base" as const },
    { key: "gt", label: "Ground Truth", kind: "base" as const },
    ...configOrder.map((cfg) => ({ key: cfg.toLowerCase(), label: cfg, cfg, kind: "prediction" as const })),
  ];

  return (
    <div className="grid grid-cols-7 gap-3">
      {panels.map((panel) => (
        <div key={panel.key} className="relative rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
          <img src={sampleImage(sample.sample_idx, panel.key)} alt={`${panel.label} sample ${sample.sample_idx}`} className="aspect-square w-full rounded-xl object-cover" />
          <div className="mt-3 flex items-center justify-between gap-2">
            {panel.kind === "prediction" ? <ConfigChip configId={panel.cfg} variant="soft" /> : <span className="text-sm font-semibold text-slate-700">{panel.label}</span>}
            {panel.kind === "prediction" && <DiceBadge value={sample.dice[panel.cfg]} configId={panel.cfg} />}
          </div>
        </div>
      ))}
    </div>
  );
}
