import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { palette } from "../../data/palette";
import type { BenchmarkData } from "../../data/types";

interface ConvergenceChartProps {
  convergence: BenchmarkData["convergence"];
  c1Reference: number;
  height?: number;
}

const series = [
  ["f1_iid", "F1"],
  ["f2_iid_phe", "F2"],
  ["f3_noniid", "F3"],
  ["f4_noniid_phe", "F4"],
] as const;

export function ConvergenceChart({ convergence, c1Reference, height = 380 }: ConvergenceChartProps) {
  const rounds = Array.from({ length: Math.max(...Object.values(convergence).map((values) => values?.length ?? 0), 1) }, (_, idx) => {
    const round = idx + 1;
    const row: Record<string, number | null> = { round };
    for (const [key, label] of series) {
      row[label] = convergence[key]?.find((entry) => entry.round === round)?.val_dice ?? null;
    }
    return row;
  });

  return (
    <div className="h-full w-full" style={{ minHeight: height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rounds} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
          <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
          <XAxis dataKey="round" tick={{ fill: "#64748B" }} />
          <YAxis domain={[0.85, 0.93]} tick={{ fill: "#64748B" }} tickFormatter={(value) => value.toFixed(2)} />
          <Tooltip formatter={(value: number) => value?.toFixed?.(4) ?? "-"} />
          <Legend />
          <ReferenceLine y={c1Reference} stroke={palette.config.C1} strokeDasharray="6 4" label="C1 test Dice" />
          {series.map(([, label]) => (
            <Line
              key={label}
              type="monotone"
              dataKey={label}
              stroke={palette.config[label]}
              strokeWidth={3}
              dot={false}
              connectNulls
              isAnimationActive
              animationDuration={800}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
