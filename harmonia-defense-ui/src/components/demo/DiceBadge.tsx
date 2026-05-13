import { palette } from "../../data/palette";
import type { ConfigId } from "../../data/types";

export function DiceBadge({ value, configId }: { value: number; configId: ConfigId }) {
  return (
    <span className="rounded-full px-2.5 py-1 font-mono text-xs font-bold text-white shadow" style={{ backgroundColor: palette.config[configId] }}>
      Dice {value.toFixed(3)}
    </span>
  );
}
