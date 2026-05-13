import { useCallback, useState } from "react";

export function useStageRouter(stageCount: number) {
  const [currentStage, setCurrentStage] = useState(0);
  const [direction, setDirection] = useState(1);

  const goTo = useCallback(
    (idx: number) => {
      setCurrentStage((current) => {
        const next = Math.max(0, Math.min(stageCount - 1, idx));
        setDirection(next >= current ? 1 : -1);
        return next;
      });
    },
    [stageCount],
  );

  const goNext = useCallback(() => {
    setCurrentStage((current) => {
      setDirection(1);
      return Math.min(stageCount - 1, current + 1);
    });
  }, [stageCount]);

  const goPrev = useCallback(() => {
    setCurrentStage((current) => {
      setDirection(-1);
      return Math.max(0, current - 1);
    });
  }, []);

  return { currentStage, direction, goNext, goPrev, goTo };
}
