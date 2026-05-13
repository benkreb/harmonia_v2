import type { ConfigId, RawConfigId } from "../data/types";
import { rawToConfigId } from "../data/configMeta";

export function fmtMetric(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "-";
}

export function fmtSec(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value < 60) return `${value.toFixed(0)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export function fmtTimer(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function toConfigId(raw: RawConfigId): ConfigId {
  return rawToConfigId[raw];
}

export function sampleImage(sampleIdx: number, panel: string): string {
  return `./predictions/sample_${sampleIdx.toString().padStart(3, "0")}_${panel}.png`;
}
