import type { BenchmarkData } from "../data/types";
import { DiceComparisonBar } from "../components/charts/DiceComparisonBar";
import { PerSampleViolin } from "../components/charts/PerSampleViolin";
import { WallTimeBar } from "../components/charts/WallTimeBar";
import { ComparisonTable } from "../components/tables/ComparisonTable";

interface Stage4Props {
  benchmark: BenchmarkData;
}

export function Stage4_Drilldown({ benchmark }: Stage4Props) {
  return (
    <div className="flex h-full flex-col gap-5">
      <header>
        <h1 className="text-5xl font-bold tracking-tight">Per-Config Drilldown</h1>
        <p className="mt-2 text-xl text-slate-600">The models differ by thousandths, not by orders of magnitude.</p>
      </header>
      <ComparisonTable configs={benchmark.configs} />
      <div className="grid grid-cols-2 gap-5">
        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="mb-2 text-xl font-bold">Dice comparison</h2>
          <DiceComparisonBar configs={benchmark.configs} metric="dice" />
        </div>
        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="mb-2 text-xl font-bold">Per-round training time</h2>
          <WallTimeBar configs={benchmark.configs} />
        </div>
      </div>
      <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-xl font-bold">Per-sample Dice distribution</h2>
          <span className="text-sm text-slate-500">260-sample held-out test set</span>
        </div>
        <PerSampleViolin perSample={benchmark.per_sample} />
      </div>
    </div>
  );
}
