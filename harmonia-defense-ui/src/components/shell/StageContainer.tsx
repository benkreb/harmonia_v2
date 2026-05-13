import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { stageVariants } from "../../utils/animations";

interface StageContainerProps {
  children: ReactNode;
  direction: number;
  background?: "default" | "gradient" | "subtle";
}

const backgrounds = {
  default: "bg-slate-50",
  gradient: "bg-gradient-to-br from-slate-50 via-white to-blue-50",
  subtle: "bg-gradient-to-br from-white via-slate-50 to-slate-100",
};

export function StageContainer({ children, direction, background = "default" }: StageContainerProps) {
  return (
    <motion.section
      custom={direction}
      variants={stageVariants}
      initial="enter"
      animate="center"
      exit="exit"
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`h-screen w-screen overflow-hidden px-16 py-12 text-slate-900 ${backgrounds[background]}`}
    >
      <div className="mx-auto flex h-full max-w-screen-2xl flex-col">{children}</div>
    </motion.section>
  );
}
