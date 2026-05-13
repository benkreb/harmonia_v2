import { AnimatePresence, MotionConfig } from "framer-motion";
import { useEffect } from "react";
import { ProgressBar } from "./components/shell/ProgressBar";
import { PresenterOverlay } from "./components/shell/PresenterOverlay";
import { StageContainer } from "./components/shell/StageContainer";
import { stageNotes } from "./data/stageNotes";
import { useBenchmarkData } from "./hooks/useBenchmarkData";
import { useCuratedSamples } from "./hooks/useCuratedSamples";
import { useKeyboardNav } from "./hooks/useKeyboardNav";
import { usePresenterMode } from "./hooks/usePresenterMode";
import { useStageRouter } from "./hooks/useStageRouter";
import { useStageTimer } from "./hooks/useStageTimer";
import { Stage0_Title } from "./stages/Stage0_Title";
import { Stage1_Problem } from "./stages/Stage1_Problem";
import { Stage2_Approach } from "./stages/Stage2_Approach";
import { Stage3_ResultsHeadline } from "./stages/Stage3_ResultsHeadline";
import { Stage4_Drilldown } from "./stages/Stage4_Drilldown";
import { Stage5_InteractiveDemo } from "./stages/Stage5_InteractiveDemo";
import { Stage6_Limitations } from "./stages/Stage6_Limitations";
import { Stage7_Conclusion } from "./stages/Stage7_Conclusion";

const STAGE_COUNT = 8;

export default function App() {
  const router = useStageRouter(STAGE_COUNT);
  const presenter = usePresenterMode();
  const { elapsedSec } = useStageTimer(router.currentStage);
  const benchmark = useBenchmarkData();
  const curated = useCuratedSamples();

  useKeyboardNav(router, STAGE_COUNT);

  useEffect(() => {
    (window as unknown as { __goToStage?: (idx: number) => void }).__goToStage = router.goTo;
  }, [router.goTo]);

  const note = stageNotes[router.currentStage];
  const stage = renderStage(router.currentStage, benchmark, curated);

  return (
    <MotionConfig reducedMotion="user" transition={{ duration: 0.25, ease: "easeOut" }}>
      <ProgressBar current={router.currentStage} total={STAGE_COUNT} />
      <AnimatePresence mode="wait" custom={router.direction} initial={false}>
        <StageContainer key={router.currentStage} direction={router.direction} background={router.currentStage === 0 ? "gradient" : "default"}>
          {stage}
        </StageContainer>
      </AnimatePresence>
      <PresenterOverlay
        enabled={presenter.enabled}
        stageIndex={router.currentStage}
        totalStages={STAGE_COUNT}
        elapsedSec={elapsedSec}
        targetSec={note.targetSec}
        speakerNote={note.note}
      />
    </MotionConfig>
  );
}

function renderStage(stage: number, benchmark: ReturnType<typeof useBenchmarkData>, curated: ReturnType<typeof useCuratedSamples>) {
  if (!benchmark || !curated) {
    if (stage <= 2 || stage >= 6) {
      // Static stages can render before JSON finishes loading.
    } else {
      return <div className="flex h-full items-center justify-center text-3xl font-semibold text-slate-500">Loading defense data...</div>;
    }
  }

  switch (stage) {
    case 0:
      return <Stage0_Title />;
    case 1:
      return <Stage1_Problem />;
    case 2:
      return <Stage2_Approach />;
    case 3:
      return benchmark ? <Stage3_ResultsHeadline benchmark={benchmark} /> : null;
    case 4:
      return benchmark ? <Stage4_Drilldown benchmark={benchmark} /> : null;
    case 5:
      return curated ? <Stage5_InteractiveDemo curated={curated} /> : null;
    case 6:
      return <Stage6_Limitations />;
    case 7:
      return <Stage7_Conclusion />;
    default:
      return <Stage0_Title />;
  }
}
