import { ArrowRight, Hospital, ShieldAlert } from "lucide-react";
import { Card } from "../components/atoms/Card";

export function Stage1_Problem() {
  return (
    <div className="flex h-full flex-col justify-between">
      <header>
        <h1 className="text-5xl font-bold tracking-tight">The Problem</h1>
        <p className="mt-3 text-2xl font-medium text-slate-600">Medical AI needs data. Data needs to stay where it lives.</p>
      </header>

      <main className="grid grid-cols-2 gap-8">
        <Card className="border-l-8 border-l-blue-600">
          <Hospital className="mb-6 h-12 w-12 text-blue-600" />
          <h2 className="text-3xl font-bold">Breast cancer screening at scale</h2>
          <ul className="mt-6 space-y-4 text-xl leading-relaxed text-slate-600">
            <li>1 in 8 women diagnosed in their lifetime</li>
            <li>Mammography is the gold standard for early detection</li>
            <li>Mass segmentation is critical for computer-aided diagnosis</li>
          </ul>
        </Card>
        <Card className="border-l-8 border-l-violet-600">
          <ShieldAlert className="mb-6 h-12 w-12 text-violet-600" />
          <h2 className="text-3xl font-bold">The data dilemma</h2>
          <ul className="mt-6 space-y-4 text-xl leading-relaxed text-slate-600">
            <li>Hospitals cannot share raw patient images</li>
            <li>Each hospital alone has too little data</li>
            <li>Centralized training requires uploading everything</li>
          </ul>
        </Card>
      </main>

      <footer className="flex items-center justify-center gap-5 text-3xl font-semibold text-slate-800">
        <ArrowRight className="h-10 w-10 text-blue-600" />
        Can we train one model across hospitals without sharing raw data? <span className="text-blue-600">Yes.</span>
      </footer>
    </div>
  );
}
