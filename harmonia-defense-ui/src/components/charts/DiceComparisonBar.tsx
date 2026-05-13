import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { rawToConfigId } from "../../data/configMeta";
import { palette } from "../../data/palette";
import type { ConfigRow } from "../../data/types";

interface DiceComparisonBarProps {
  configs: ConfigRow[];
  metric?: "dice" | "iou";
}

export function DiceComparisonBar({ configs, metric = "dice" }: DiceComparisonBarProps) {
  const data = configs.map((row) => {
    const id = rawToConfigId[row.id];
    return { id, value: row[metric].mean ?? 0, fill: palette.config[id] };
  });

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
        <XAxis dataKey="id" tick={{ fill: "#64748B" }} />
        <YAxis domain={[0.82, 0.93]} tick={{ fill: "#64748B" }} />
        <Tooltip formatter={(value: number) => value.toFixed(4)} />
        <Bar dataKey="value" radius={[8, 8, 0, 0]} isAnimationActive>
          {data.map((entry) => (
            <Cell key={entry.id} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
