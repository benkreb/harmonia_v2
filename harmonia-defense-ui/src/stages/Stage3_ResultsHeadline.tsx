import type { BenchmarkData } from "../data/types";
import { ConvergenceChart } from "../components/charts/ConvergenceChart";
import { Number } from "../components/atoms/Number";

interface Stage3Props {
  benchmark: BenchmarkData;
}

export function Stage3_ResultsHeadline({ benchmark }: Stage3Props) {
  const c1 = benchmark.configs.find((row) => row.id === "C1")!;
  const f1 = benchmark.configs.find((row) => row.id === "f1_iid")!;
  const f3 = benchmark.configs.find((row) => row.id === "f3_noniid")!;
  const iidGap = (c1.dice.mean ?? 0) - (f1.dice.mean ?? 0);
  const noniidGap = (c1.dice.mean ?? 0) - (f3.dice.mean ?? 0);

  return (
    <div className="grid h-full grid-cols-[0.42fr_0.58fr] items-center gap-10">
      <section>
        <p className="text-sm font-bold uppercase tracking-[0.35em] text-blue-600">Headline</p>
        <h1 className="mt-5 text-6xl font-extrabold tracking-tight text-slate-950">Federated learning achieves clinical parity</h1>
        <div className="mt-10 font-mono text-8xl font-extrabold text-blue-600">
          <Number value={iidGap} decimals={3} />
        </div>
        <p className="mt-3 text-xl font-medium text-slate-600">Dice gap between centralized and federated IID training</p>
        <p className="mt-6 text-lg text-slate-500">Non-IID Dice gap: <span className="font-mono font-bold text-slate-900">{noniidGap.toFixed(3)}</span></p>
        <p className="mt-2 text-sm text-slate-400">Test set: 260 held-out patient samples · identical across all configurations</p>
      </section>
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-2xl font-bold">Validation Dice over training rounds</h2>
        <ConvergenceChart convergence={benchmark.convergence} c1Reference={benchmark.c1_reference.dice_mean} />
      </section>
    </div>
  );
}
