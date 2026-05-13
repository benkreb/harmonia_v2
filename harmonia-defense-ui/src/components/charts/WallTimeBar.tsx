import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { rawToConfigId } from "../../data/configMeta";
import { palette } from "../../data/palette";
import type { ConfigRow } from "../../data/types";

interface WallTimeBarProps {
  configs: ConfigRow[];
}

export function WallTimeBar({ configs }: WallTimeBarProps) {
  const data = configs
    .filter((row) => row.id !== "C1")
    .map((row) => {
      const id = rawToConfigId[row.id];
      return { id, value: row.wall_time_avg_sec ?? 0, fill: palette.config[id] };
    });

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
        <XAxis dataKey="id" tick={{ fill: "#64748B" }} />
        <YAxis tick={{ fill: "#64748B" }} label={{ value: "sec/round", angle: -90, position: "insideLeft" }} />
        <Tooltip formatter={(value: number) => `${value.toFixed(1)} sec`} />
        <Bar dataKey="value" radius={[8, 8, 0, 0]} isAnimationActive>
          {data.map((entry) => (
            <Cell key={entry.id} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
