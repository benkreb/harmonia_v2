import { Database, Lock, Network, Server } from "lucide-react";

export function ArchitectureDiagram() {
  return (
    <div className="relative mx-auto h-[360px] w-[920px]">
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 920 360">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#2563EB" />
          </marker>
        </defs>
        <path d="M460 116 C280 170 230 190 205 246" stroke="#2563EB" strokeWidth="3" strokeDasharray="8 8" fill="none" markerEnd="url(#arrow)" />
        <path d="M460 116 C640 170 690 190 715 246" stroke="#2563EB" strokeWidth="3" strokeDasharray="8 8" fill="none" markerEnd="url(#arrow)" />
        <path d="M205 246 C250 198 310 164 460 116" stroke="#7C3AED" strokeWidth="3" strokeDasharray="8 8" fill="none" markerEnd="url(#arrow)" opacity="0.65" />
        <path d="M715 246 C670 198 610 164 460 116" stroke="#7C3AED" strokeWidth="3" strokeDasharray="8 8" fill="none" markerEnd="url(#arrow)" opacity="0.65" />
      </svg>

      <Node className="left-[330px] top-4 w-[260px]" title="FL Server" subtitle="FedAvg + PHE aggregation" icon={<Server />} />
      <Node className="bottom-8 left-4 w-[290px]" title="Client 1" subtitle="Local training · patient shard" icon={<Database />} />
      <Node className="bottom-8 right-4 w-[290px]" title="Client 2" subtitle="Local training · patient shard" icon={<Database />} />

      <div className="absolute left-[370px] top-[185px] flex items-center gap-2 rounded-full border border-blue-200 bg-white px-4 py-2 text-sm font-semibold text-blue-700 shadow-sm">
        <Lock className="h-4 w-4" />
        CKKS encrypted parameters
      </div>
      <div className="absolute left-[390px] top-[255px] flex items-center gap-2 text-sm font-medium text-slate-500">
        <Network className="h-4 w-4" />
        Raw mammograms never leave clients
      </div>
    </div>
  );
}

function Node({ title, subtitle, icon, className }: { title: string; subtitle: string; icon: React.ReactNode; className: string }) {
  return (
    <div className={`absolute rounded-3xl border border-slate-200 bg-white p-6 text-center shadow-lg ${className}`}>
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 text-blue-700">{icon}</div>
      <div className="text-xl font-bold text-slate-900">{title}</div>
      <div className="mt-1 text-sm text-slate-500">{subtitle}</div>
    </div>
  );
}
