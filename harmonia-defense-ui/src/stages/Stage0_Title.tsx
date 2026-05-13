import { motion } from "framer-motion";
import { Network } from "lucide-react";

export function Stage0_Title() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-6 flex h-20 w-20 items-center justify-center rounded-[1.5rem] bg-blue-600 text-white shadow-xl xl:h-28 xl:w-28 xl:rounded-[2rem]">
        <Network className="h-10 w-10 xl:h-14 xl:w-14" />
      </motion.div>
      <p className="text-sm font-bold uppercase tracking-[0.35em] text-blue-600">Capstone Project</p>
      <h1 className="mt-4 text-5xl font-extrabold tracking-tight text-slate-950 xl:text-7xl">Harmonia Vision</h1>
      <p className="mt-4 max-w-4xl text-xl font-medium leading-relaxed text-slate-600 xl:text-2xl">
        Privacy-Preserving Federated Learning for Breast Cancer Mass Segmentation
      </p>
      <div className="mt-8 grid w-full max-w-6xl grid-cols-3 items-start gap-4 text-xs font-semibold leading-tight text-slate-500 xl:mt-12 xl:text-sm">
        <div className="space-y-1 rounded-2xl bg-white/70 p-4 shadow-sm">
          <p>Berk GÜNBERK · Software Engineering</p>
          <p>Kutay ORALLI · Software Engineering</p>
          <p>Eyüpcan IŞIKGÖR · Software Engineering</p>
        </div>
        <div className="space-y-1 rounded-2xl bg-white/70 p-4 shadow-sm">
          <p>Beril ÇİTİL · Software Engineering</p>
          <p>İrem BATIGÜN · Computer Engineering</p>
          <p>Supervisor: Tarık ZİYADOĞDU</p>
        </div>
        <div className="rounded-2xl bg-white/70 p-4 shadow-sm">Istinye University · May 2026</div>
      </div>
      <p className="mt-6 animate-pulse text-sm text-slate-400">Press → to begin</p>
    </div>
  );
}
