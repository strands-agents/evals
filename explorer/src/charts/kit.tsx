// Chart primitives shared by every view: the width hook, readable ticks, and the one candidate mark.

import { useEffect, useRef, useState } from "react";
import type { MarkKind } from "../model/study";
import { INK, MARK, MUTED, SURFACE } from "../encoding/theme";

export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(900);
  useEffect(() => {
    if (!ref.current) return;
    setW(Math.max(320, ref.current.getBoundingClientRect().width));
    const ro = new ResizeObserver((entries) => setW(Math.max(320, entries[0].contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Readable ticks: on log scales only 1-2-5 steps (then decades), so labels never collide. */
export function niceTicks(scale: { ticks: (n: number) => number[] }, log: boolean | undefined, count = 5, integer = false): number[] {
  let all: number[] = scale.ticks(count);
  if (integer) all = all.filter((v) => Number.isInteger(v));
  if (!log) return all;
  const lead = (v: number) => Math.round(v / 10 ** Math.floor(Math.log10(v)));
  const t125 = all.filter((v) => [1, 2, 5].includes(lead(v)));
  return t125.length > 6 ? all.filter((v) => lead(v) === 1) : t125;
}

/** A mark's radius for the space each candidate has (smaller when candidates are dense). */
export function markRadius(step: number): number {
  return step >= 14 ? 4.5 : step >= 7 ? 3.5 : 2.5;
}

/** One candidate's mark. Filled: comparable and feasible; hollow: breaks a constraint; small grey dot: not comparable.
 * A candidate that measured nothing has no mark. ``best``: a ring (the best candidate, or a member of the Pareto
 * set). */
export function Mark({ x, y, kind, r, best, hovered }: { x: number; y: number; kind: MarkKind; r: number; best?: boolean; hovered?: boolean }) {
  if (kind === "unmeasured") return null;
  return (
    <g>
      {best && <circle cx={x} cy={y} r={r + 3.5} fill="none" stroke={INK} strokeWidth={1.5} />}
      {kind === "ineligible" && <circle cx={x} cy={y} r={Math.max(2, r - 1.5)} fill={MUTED} stroke={SURFACE} strokeWidth={1.5} />}
      {kind === "infeasible" && <circle cx={x} cy={y} r={r - 0.6} fill={SURFACE} stroke={MARK} strokeWidth={1.8} />}
      {kind === "ok" && <circle cx={x} cy={y} r={r} fill={MARK} stroke={SURFACE} strokeWidth={2} />}
      {hovered && <circle cx={x} cy={y} r={r + 6.5} fill="none" stroke={INK} strokeOpacity={0.35} strokeWidth={1} />}
    </g>
  );
}

/** The glyph for each mark kind, for legends (drawn by the same component, so they cannot drift). */
export function MarkGlyph({ kind, best }: { kind: MarkKind; best?: boolean }) {
  return (
    <svg width={22} height={18} aria-hidden>
      <Mark x={11} y={9} kind={kind} r={4.5} best={best} />
    </svg>
  );
}
