import { useEffect } from "react";
import type { useStageRouter } from "./useStageRouter";

type Router = ReturnType<typeof useStageRouter>;

export function useKeyboardNav(router: Router, stageCount: number): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA") return;

      if (["ArrowRight", " ", "PageDown"].includes(event.key)) {
        event.preventDefault();
        router.goNext();
      } else if (["ArrowLeft", "PageUp"].includes(event.key)) {
        event.preventDefault();
        router.goPrev();
      } else if (/^[1-8]$/.test(event.key)) {
        router.goTo(Math.min(stageCount - 1, Number(event.key) - 1));
      } else if (event.key === "0") {
        document.body.style.zoom = "100%";
        window.scrollTo(0, 0);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [router, stageCount]);
}
