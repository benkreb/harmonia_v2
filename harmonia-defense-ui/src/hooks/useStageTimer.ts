import { useEffect, useState } from "react";

export function useStageTimer(currentStage: number) {
  const [elapsedSec, setElapsedSec] = useState(0);

  useEffect(() => {
    setElapsedSec(0);
    const id = window.setInterval(() => setElapsedSec((value) => value + 1), 1000);
    return () => window.clearInterval(id);
  }, [currentStage]);

  return { elapsedSec };
}
