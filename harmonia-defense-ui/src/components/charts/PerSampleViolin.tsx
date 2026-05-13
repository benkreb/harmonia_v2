import { configOrder } from "../../data/configMeta";
import { palette } from "../../data/palette";
import type { ConfigId, PerSampleArrays } from "../../data/types";

interface PerSampleViolinProps {
  perSample: Record<ConfigId, PerSampleArrays>;
}

function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

export function PerSampleViolin({ perSample }: PerSampleViolinProps) {
  const width = 760;
  const laneHeight = 34;
  const x0 = 90;
  const chartWidth = width - x0 - 20;
  const domain: [number, number] = [0.65, 1.0];
  const scale = (value: number) => x0 + ((value - domain[0]) / (domain[1] - domain[0])) * chartWidth;

  return (
    <svg viewBox={`0 0 ${width} ${laneHeight * configOrder.length + 34}`} className="h-[230px] w-full">
      <line x1={x0} x2={x0 + chartWidth} y1={laneHeight * configOrder.length + 4} y2={laneHeight * configOrder.length + 4} stroke="#CBD5E1" />
      {[0.7, 0.8, 0.9, 1.0].map((tick) => (
        <g key={tick}>
          <line x1={scale(tick)} x2={scale(tick)} y1={8} y2={laneHeight * configOrder.length + 4} stroke="#E2E8F0" />
          <text x={scale(tick)} y={laneHeight * configOrder.length + 24} textAnchor="middle" className="fill-slate-500 text-xs">
            {tick.toFixed(1)}
          </text>
        </g>
      ))}
      {configOrder.map((cfg, lane) => {
        const values = perSample[cfg].dice;
        const q1 = quantile(values, 0.25);
        const med = quantile(values, 0.5);
        const q3 = quantile(values, 0.75);
        const y = 18 + lane * laneHeight;
        const bins = Array.from({ length: 30 }, (_, idx) => {
          const min = domain[0] + (idx / 30) * (domain[1] - domain[0]);
          const max = domain[0] + ((idx + 1) / 30) * (domain[1] - domain[0]);
          return values.filter((value) => value >= min && value < max).length;
        });
        const maxBin = Math.max(...bins, 1);
        return (
          <g key={cfg}>
            <text x={18} y={y + 4} className="fill-slate-700 text-sm font-bold">
              {cfg}
            </text>
            {bins.map((count, idx) => {
              const x = x0 + (idx / bins.length) * chartWidth;
              const h = 6 + (count / maxBin) * 18;
              return (
                <rect
                  key={idx}
                  x={x}
                  y={y - h / 2}
                  width={chartWidth / bins.length - 1}
                  height={h}
                  rx={3}
                  fill={palette.config[cfg]}
                  opacity={0.22 + (count / maxBin) * 0.55}
                />
              );
            })}
            <line x1={scale(q1)} x2={scale(q3)} y1={y} y2={y} stroke={palette.config[cfg]} strokeWidth={5} strokeLinecap="round" />
            <circle cx={scale(med)} cy={y} r={5} fill="#0F172A" />
          </g>
        );
      })}
    </svg>
  );
}
