import { Cpu, Network, Shield } from "lucide-react";
import { MetricCard } from "../components/atoms/MetricCard";
import { ArchitectureDiagram } from "../components/diagram/ArchitectureDiagram";
import { ConfigMatrix } from "../components/diagram/ConfigMatrix";

export function Stage2_Approach() {
  return (
    <div className="flex h-full flex-col gap-7">
      <header>
        <h1 className="text-5xl font-bold tracking-tight">The Approach</h1>
        <p className="mt-3 text-2xl font-medium text-slate-600">A segmentation model trained federated, then aggregated with a privacy layer.</p>
      </header>
      <ArchitectureDiagram />
      <div className="grid grid-cols-3 gap-5">
        <MetricCard label="Model" value="ConnectedUnet" caption="EfficientNet-B3 encoder · 256×256 grayscale input · binary tumor mask" icon={<Cpu />} />
        <MetricCard label="Federated Learning" value="FedAvg" caption="2 clients · patient-level partitions · 20 rounds max · local epochs" icon={<Network />} accent="#0891B2" />
        <MetricCard label="Privacy Layer" value="CKKS" caption="TenSEAL · poly modulus 8192 · homomorphic weighted aggregation" icon={<Shield />} accent="#7C3AED" />
      </div>
      <div>
        <p className="mb-4 text-center text-lg font-semibold text-slate-600">We evaluate five configurations:</p>
        <ConfigMatrix />
      </div>
    </div>
  );
}
