import { CheckCircle2 } from "lucide-react";

const takeaways = [
  ["Clinical parity", "Federated learning reaches Dice within 0.003 of centralized training on identical held-out data."],
  ["Robust to Non-IID", "An 80/20 patient-level skew leaves performance within the same clinical-quality band."],
  ["Privacy is affordable", "TenSEAL CKKS adds wall-time overhead without measurable Dice drift in IID or Non-IID settings."],
];

export function Stage7_Conclusion() {
  return (
    <div className="flex h-full flex-col justify-center">
      <h1 className="text-6xl font-extrabold tracking-tight">What we showed</h1>
      <div className="mt-12 grid gap-5">
        {takeaways.map(([title, body]) => (
          <div key={title} className="flex items-start gap-5 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <CheckCircle2 className="mt-1 h-9 w-9 flex-none text-emerald-500" />
            <div>
              <h2 className="text-3xl font-bold">{title}</h2>
              <p className="mt-2 text-xl leading-relaxed text-slate-600">{body}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-10 grid grid-cols-3 gap-5 text-center">
        <div className="rounded-2xl bg-blue-50 p-5 font-semibold text-blue-800">GPU droplet deployment</div>
        <div className="rounded-2xl bg-blue-50 p-5 font-semibold text-blue-800">Multi-rerun variance study</div>
        <div className="rounded-2xl bg-blue-50 p-5 font-semibold text-blue-800">Threshold encryption</div>
      </div>
      <footer className="mt-12 text-center">
        <p className="text-4xl font-semibold">Thank you</p>
        <p className="mt-2 text-2xl text-slate-500">Questions?</p>
      </footer>
    </div>
  );
}
