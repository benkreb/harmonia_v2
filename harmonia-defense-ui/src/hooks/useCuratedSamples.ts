import { useEffect, useState } from "react";
import type { CuratedSamples } from "../data/types";

export function useCuratedSamples(): CuratedSamples | null {
  const [data, setData] = useState<CuratedSamples | null>(null);

  useEffect(() => {
    fetch("./data/samples_curated.json")
      .then((response) => response.json())
      .then(setData)
      .catch((error) => {
        console.error("Failed to load samples_curated.json", error);
      });
  }, []);

  return data;
}
