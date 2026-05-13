import { configMeta } from "../../data/configMeta";
import { palette } from "../../data/palette";
import type { ConfigId } from "../../data/types";

interface ConfigChipProps {
  configId: ConfigId;
  variant?: "solid" | "outline" | "soft";
}

export function ConfigChip({ configId, variant = "solid" }: ConfigChipProps) {
  const color = palette.config[configId];
  const label = configMeta[configId].id;
  const style =
    variant === "solid"
      ? { backgroundColor: color, color: "white", borderColor: color }
      : variant === "soft"
        ? { backgroundColor: `${color}18`, color, borderColor: `${color}44` }
        : { backgroundColor: "white", color, borderColor: color };

  return (
    <span className="inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-bold tabular-nums" style={style}>
      {label}
    </span>
  );
}
