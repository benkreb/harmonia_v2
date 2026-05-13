import { AlertTriangle } from "lucide-react";
import { ConfigChip } from "../atoms/ConfigChip";
import { rawToConfigId } from "../../data/configMeta";
import type { ConfigRow } from "../../data/types";
import { fmtMetric, fmtSec } from "../../utils/formatters";

interface ComparisonTableProps {
  configs: ConfigRow[];
}

export function ComparisonTable({ configs }: ComparisonTableProps) {
  const c1Sens = configs.find((row) => row.id === "C1")?.sens.mean ?? 0;
  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-100 text-xs uppercase tracking-widest text-slate-500">
          <tr>
            <th className="px-4 py-3">Config</th>
            <th className="px-4 py-3">Dice</th>
            <th className="px-4 py-3">IoU</th>
            <th className="px-4 py-3">Sensitivity</th>
            <th className="px-4 py-3">Specificity</th>
            <th className="px-4 py-3">Precision</th>
            <th className="px-4 py-3">Best</th>
            <th className="px-4 py-3">Time / round</th>
          </tr>
        </thead>
        <tbody>
          {configs.map((row) => {
            const cfg = rawToConfigId[row.id];
            const sensWarn = row.id !== "C1" && (row.sens.mean ?? 0) < c1Sens - 0.01;
            return (
              <tr key={row.id} className={row.id === "C1" ? "border-l-4 border-blue-700 bg-blue-50/60" : "border-t border-slate-100"}>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <ConfigChip configId={cfg} />
                    <span className="font-semibold text-slate-800">{row.id === "C1" ? "Reference" : row.scenario}</span>
                  </div>
                </td>
                <MetricCell mean={row.dice.mean} std={row.dice.std} />
                <MetricCell mean={row.iou.mean} std={row.iou.std} />
                <td className="px-4 py-3 font-mono">
                  <span>{fmtMetric(row.sens.mean)}</span>
                  <span className="ml-1 text-slate-400">±{fmtMetric(row.sens.std)}</span>
                  {sensWarn && <AlertTriangle className="ml-2 inline h-4 w-4 text-amber-500" />}
                </td>
                <MetricCell mean={row.spec.mean} std={row.spec.std} />
                <MetricCell mean={row.prec.mean} std={row.prec.std} />
                <td className="px-4 py-3 font-mono">{row.n_rounds_to_best ?? "-"}</td>
                <td className="px-4 py-3 font-mono">{fmtSec(row.wall_time_avg_sec)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricCell({ mean, std }: { mean: number | null; std: number | null }) {
  return (
    <td className="px-4 py-3 font-mono">
      <span>{fmtMetric(mean)}</span>
      <span className="ml-1 text-slate-400">±{fmtMetric(std)}</span>
    </td>
  );
}
