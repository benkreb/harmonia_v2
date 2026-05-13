import type { CuratedSamples } from "../data/types";
import { SampleBrowser } from "../components/demo/SampleBrowser";

interface Stage5Props {
  curated: CuratedSamples;
}

export function Stage5_InteractiveDemo({ curated }: Stage5Props) {
  return <SampleBrowser samples={curated.samples} />;
}
