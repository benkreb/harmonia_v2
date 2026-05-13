import type { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string;
  caption?: string;
  icon?: ReactNode;
  accent?: string;
}

export function MetricCard({ label, value, caption, icon, accent = "#2563EB" }: MetricCardProps) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm" style={{ borderLeft: `6px solid ${accent}` }}>
      <div className="mb-4 flex items-center justify-between text-sm font-semibold uppercase tracking-widest text-slate-500">
        <span>{label}</span>
        {icon}
      </div>
      <div className="font-mono text-5xl font-bold tracking-tight text-slate-900">{value}</div>
      {caption && <p className="mt-3 text-sm leading-relaxed text-slate-500">{caption}</p>}
    </div>
  );
}
