export type ConfigId = "C1" | "F1" | "F2" | "F3" | "F4";
export type RawConfigId = "C1" | "f1_iid" | "f2_iid_phe" | "f3_noniid" | "f4_noniid_phe";

export interface StageNote {
  title: string;
  targetSec: number;
  note: string;
}

export interface MetricStat {
  mean: number | null;
  std: number | null;
}

export interface ConfigRow {
  id: RawConfigId;
  scenario: "centralized" | "iid" | "noniid_80_20";
  phe: boolean;
  dice: MetricStat;
  iou: MetricStat;
  sens: MetricStat;
  spec: MetricStat;
  prec: MetricStat;
  pixel_acc: number | null;
  wall_time_total_sec: number | null;
  wall_time_avg_sec: number | null;
  n_rounds: number | null;
  n_rounds_to_best: number | null;
  total_comm_mb: number | null;
  comm_per_round_mb: number | null;
}

export interface RoundRecord {
  round: number;
  val_dice: number;
  val_iou: number;
  val_loss: number;
  train_loss: number | null;
  wall_time_sec: number | null;
}

export interface PerSampleArrays {
  dice: number[];
  iou: number[];
}

export interface BenchmarkData {
  configs: ConfigRow[];
  convergence: Partial<Record<"f1_iid" | "f2_iid_phe" | "f3_noniid" | "f4_noniid_phe", RoundRecord[]>>;
  per_sample: Record<ConfigId, PerSampleArrays>;
  c1_reference: { dice_mean: number };
  generated_at: string;
}

export interface CuratedSample {
  sample_idx: number;
  category: "best" | "median" | "worst";
  patient_id: string;
  pathology: string;
  breast_density: number | null;
  dice: Record<ConfigId, number>;
  iou: Record<ConfigId, number>;
}

export interface CuratedSamples {
  samples: CuratedSample[];
  n_samples: number;
}
