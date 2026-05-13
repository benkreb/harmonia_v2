interface ProgressBarProps {
  current: number;
  total: number;
}

export function ProgressBar({ current, total }: ProgressBarProps) {
  return (
    <div className="fixed left-0 top-0 z-40 flex h-1 w-full gap-0.5 bg-transparent">
      {Array.from({ length: total }, (_, idx) => (
        <div
          key={idx}
          className={`h-full flex-1 transition-colors duration-300 ${
            idx < current ? "bg-blue-600/45" : idx === current ? "bg-blue-600" : "bg-slate-200"
          }`}
        />
      ))}
    </div>
  );
}
