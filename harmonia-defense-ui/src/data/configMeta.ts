import { Network, Server, Shield } from "lucide-react";
import type { ComponentType } from "react";
import type { ConfigId, RawConfigId } from "./types";

export interface ConfigMeta {
  id: ConfigId;
  rawId: RawConfigId;
  label: string;
  scenario: string;
  phe: boolean;
  icon: ComponentType<{ className?: string }>;
}

export const configMeta: Record<ConfigId, ConfigMeta> = {
  C1: { id: "C1", rawId: "C1", label: "Centralized", scenario: "Reference", phe: false, icon: Server },
  F1: { id: "F1", rawId: "f1_iid", label: "FL IID", scenario: "IID", phe: false, icon: Network },
  F2: { id: "F2", rawId: "f2_iid_phe", label: "FL IID + PHE", scenario: "IID + PHE", phe: true, icon: Shield },
  F3: { id: "F3", rawId: "f3_noniid", label: "FL Non-IID", scenario: "Non-IID", phe: false, icon: Network },
  F4: { id: "F4", rawId: "f4_noniid_phe", label: "FL Non-IID + PHE", scenario: "Non-IID + PHE", phe: true, icon: Shield },
};

export const rawToConfigId: Record<RawConfigId, ConfigId> = {
  C1: "C1",
  f1_iid: "F1",
  f2_iid_phe: "F2",
  f3_noniid: "F3",
  f4_noniid_phe: "F4",
};

export const configOrder: ConfigId[] = ["C1", "F1", "F2", "F3", "F4"];
export const flOrder: Exclude<ConfigId, "C1">[] = ["F1", "F2", "F3", "F4"];
