import { animate, useMotionValue, useTransform, motion } from "framer-motion";
import { useEffect } from "react";

interface NumberProps {
  value: number;
  decimals?: number;
  duration?: number;
}

export function Number({ value, decimals = 3, duration = 0.8 }: NumberProps) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (latest) => latest.toFixed(decimals));

  useEffect(() => {
    const controls = animate(count, value, { duration, ease: "easeOut" });
    return controls.stop;
  }, [count, duration, value]);

  return <motion.span>{rounded}</motion.span>;
}
