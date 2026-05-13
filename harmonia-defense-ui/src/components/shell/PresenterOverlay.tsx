import { fmtTimer } from "../../utils/formatters";

interface PresenterOverlayProps {
  enabled: boolean;
  stageIndex: number;
  totalStages: number;
  elapsedSec: number;
  targetSec: number;
  speakerNote: string;
}

export function PresenterOverlay({
  enabled,
  stageIndex,
  totalStages,
  elapsedSec,
  targetSec,
  speakerNote,
}: PresenterOverlayProps) {
  if (!enabled) return null;

  const overTime = elapsedSec > targetSec * 1.5;
  return (
    <aside className="fixed bottom-6 right-6 z-50 w-[420px] rounded-2xl bg-slate-950/90 p-5 text-slate-100 shadow-2xl backdrop-blur">
      <div className="mb-3 flex items-center justify-between text-sm text-slate-300">
        <span>
          Stage {stageIndex + 1}/{totalStages}
        </span>
        <span className={overTime ? "font-semibold text-red-300" : ""}>
          {fmtTimer(elapsedSec)} / {fmtTimer(targetSec)}
        </span>
      </div>
      <p className="text-sm leading-relaxed text-slate-100">{speakerNote}</p>
    </aside>
  );
}
