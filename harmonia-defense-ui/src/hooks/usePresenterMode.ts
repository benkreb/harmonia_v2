import { useCallback, useEffect, useState } from "react";

export function usePresenterMode() {
  const [enabled, setEnabled] = useState(() => new URLSearchParams(window.location.search).get("presenter") === "1");

  const toggle = useCallback(() => setEnabled((current) => !current), []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "p") toggle();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggle]);

  return { enabled, toggle };
}
