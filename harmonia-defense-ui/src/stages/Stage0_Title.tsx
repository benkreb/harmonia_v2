import { motion } from "framer-motion";
import { Network } from "lucide-react";

export function Stage0_Title() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8 flex h-28 w-28 items-center justify-center rounded-[2rem] bg-blue-600 text-white shadow-xl">
        <Network className="h-14 w-14" />
      </motion.div>
      <p className="text-sm font-bold uppercase tracking-[0.35em] text-blue-600">Capstone Project</p>
      <h1 className="mt-5 text-7xl font-extrabold tracking-tight text-slate-950">Harmonia Vision</h1>
      <p className="mt-5 max-w-4xl text-2xl font-medium leading-relaxed text-slate-600">
        Privacy-Preserving Federated Learning for Breast Cancer Mass Segmentation
      </p>
      <div className="mt-12 grid w-full max-w-4xl grid-cols-3 gap-4 text-sm font-semibold text-slate-500">
        <span>Berk Reb · Software Engineering</span>
        <span>Supervisor: Waddah</span>
        <span>Istinye University · May 2026</span>
      </div>
      <p className="absolute bottom-12 animate-pulse text-sm text-slate-400">Press → to begin</p>
    </div>
  );
}
