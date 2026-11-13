// One hover model for every chart and table: the candidate under the pointer (by number), and one tooltip that follows
// the pointer, rendered once per page.

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

type HoverState = { n: number | null; x: number; y: number };
type HoverApi = { hover: HoverState; set: (n: number | null, e?: { clientX: number; clientY: number }) => void };

const HoverContext = createContext<HoverApi>({ hover: { n: null, x: 0, y: 0 }, set: () => {} });

export function HoverProvider({ children }: { children: ReactNode }) {
  const [hover, setHover] = useState<HoverState>({ n: null, x: 0, y: 0 });
  const set = useCallback((n: number | null, e?: { clientX: number; clientY: number }) => {
    setHover((h) => (n === null ? { ...h, n: null } : { n, x: e?.clientX ?? h.x, y: e?.clientY ?? h.y }));
  }, []);
  const api = useMemo(() => ({ hover, set }), [hover, set]);
  return <HoverContext.Provider value={api}>{children}</HoverContext.Provider>;
}

export function useHover(): HoverApi {
  return useContext(HoverContext);
}

/** The tooltip for the hovered candidate, next to the pointer (kept inside the window). */
export function Tooltip({ render }: { render: (n: number) => ReactNode }) {
  const { hover } = useHover();
  if (hover.n === null) return null;
  const left = Math.min(hover.x + 16, window.innerWidth - 340);
  const top = Math.min(hover.y + 16, window.innerHeight - 260);
  return (
    <div className="tooltip" style={{ left, top }}>
      {render(hover.n)}
    </div>
  );
}
