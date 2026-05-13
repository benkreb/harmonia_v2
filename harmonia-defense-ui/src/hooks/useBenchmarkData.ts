import { useEffect, useState } from "react";
import type { BenchmarkData } from "../data/types";

export function useBenchmarkData(): BenchmarkData | null {
  const [data, setData] = useState<BenchmarkData | null>(null);

  useEffect(() => {
    fetch("./data/benchmark.json")
      .then((response) => response.json())
      .then(setData)
      .catch((error) => {
        console.error("Failed to load benchmark.json", error);
      });
  }, []);

  return data;
}
