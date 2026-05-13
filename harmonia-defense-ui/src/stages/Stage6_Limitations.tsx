import { AlertTriangle } from "lucide-react";
import { Card } from "../components/atoms/Card";

const limitations = [
  {
    title: "Single seed per configuration",
    badge: "Statistical",
    body: "Each F-configuration was run once. Dice gaps are point estimates; confidence intervals require multiple reruns.",
  },
  {
    title: "PHE bytes are a simulation artifact",
    badge: "Measurement",
    body: "The local sequential fallback measures wall-time overhead, not true wire-level CKKS ciphertext overhead over gRPC.",
  },
  {
    title: "Secret key held server-side",
    badge: "Trust model",
    body: "This measures PHE overhead, but not a strict production trust boundary. Future work should use threshold encryption.",
  },
];

export function Stage6_Limitations() {
  return (
    <div className="flex h-full flex-col justify-center gap-10">
      <header>
        <h1 className="text-5xl font-bold tracking-tight">What this study does not claim</h1>
        <p className="mt-3 text-2xl text-slate-600">Honest framing of the boundaries of our findings.</p>
      </header>
      <div className="grid grid-cols-3 gap-6">
        {limitations.map((item) => (
          <Card key={item.title} className="min-h-[300px]">
            <AlertTriangle className="mb-5 h-10 w-10 text-amber-500" />
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-bold uppercase tracking-widest text-amber-700">{item.badge}</span>
            <h2 className="mt-5 text-2xl font-bold">{item.title}</h2>
            <p className="mt-4 text-lg leading-relaxed text-slate-600">{item.body}</p>
          </Card>
        ))}
      </div>
      <p className="rounded-3xl bg-slate-100 p-6 text-lg italic leading-relaxed text-slate-600">
        Federated configurations also show a sensitivity drop of about 1.2% versus centralized. We surface this because clinical AI
        needs honest failure framing, not only best-case metrics.
      </p>
    </div>
  );
}
