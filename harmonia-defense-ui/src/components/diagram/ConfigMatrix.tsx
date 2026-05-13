import { configMeta, configOrder } from "../../data/configMeta";
import { palette } from "../../data/palette";

export function ConfigMatrix() {
  return (
    <div className="flex items-center justify-center gap-4">
      {configOrder.map((cfg) => {
        const meta = configMeta[cfg];
        const Icon = meta.icon;
        return (
          <div key={cfg} className="min-w-[150px] rounded-2xl border border-slate-200 bg-white p-4 text-center shadow-sm">
            <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-xl text-white" style={{ backgroundColor: palette.config[cfg] }}>
              <Icon className="h-5 w-5" />
            </div>
            <div className="font-bold text-slate-900">{cfg}</div>
            <div className="text-xs text-slate-500">{meta.label}</div>
          </div>
        );
      })}
    </div>
  );
}
