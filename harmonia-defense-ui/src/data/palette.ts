export const palette = {
  bg: "#FAFBFC",
  surface: "#FFFFFF",
  surfaceAlt: "#F1F5F9",
  border: "#E2E8F0",
  textPrimary: "#0F172A",
  textSecondary: "#475569",
  textTertiary: "#94A3B8",
  brand: "#2563EB",
  brandLight: "#3B82F6",
  brandDark: "#1D4ED8",
  config: {
    C1: "#1E40AF",
    F1: "#2563EB",
    F2: "#7C3AED",
    F3: "#0891B2",
    F4: "#6366F1",
  },
  success: "#10B981",
  warning: "#F59E0B",
  danger: "#EF4444",
  overlayGT: "#DC2626",
  overlayPred: "#06B6D4",
} as const;

export type ConfigColorKey = keyof typeof palette.config;
