// Small helpers for the two-dimensional charts (trade-off scatters, slices, traces): value domains, nearest-point hit
// testing, the 2-D front of two objectives, and labels placed without overlapping.

import type { Goal, Objective, StudyModel } from "../model/study";
import { textWidth } from "./axis";

/** A value's domain: the metric's declared range where it has one, else the values padded by a tenth (as the candidate
 * panels do), always wide enough for every value and every ``extra`` (a limit). */
export function valueDomain(values: number[], range: [number | null, number | null], extra: number[] = []): [number, number] {
  const all = [...values, ...extra].filter((v) => Number.isFinite(v));
  let [lo, hi] = range;
  if (lo === null || hi === null) {
    const min = all.length ? Math.min(...all) : 0;
    const max = all.length ? Math.max(...all) : 1;
    const pad = (max - min) * 0.1 || Math.abs(max) * 0.1 || 1;
    lo ??= min - pad;
    hi ??= max + pad;
  }
  if (all.length) {
    lo = Math.min(lo, ...all);
    hi = Math.max(hi, ...all);
  }
  if (lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

/** The domain of a goal over every candidate (including its limit, for a constraint). */
export function goalDomain(study: StudyModel, goal: Goal): [number, number] {
  const values = study.candidates.flatMap((t) => (study.markKind(t) === "unmeasured" ? [] : [study.mean(t, goal.metric)])).filter((v): v is number => v !== null);
  return valueDomain(values, study.range(goal.metric), goal.role === "constraint" ? [goal.constraint.limit] : []);
}

/** The item closest to (x, y) within ``max`` pixels, or null. */
export function nearest<T>(items: T[], px: (t: T) => number, py: (t: T) => number, x: number, y: number, max = 18): T | null {
  let best: T | null = null;
  let d = max * max;
  for (const it of items) {
    const dx = px(it) - x;
    const dy = py(it) - y;
    const dd = dx * dx + dy * dy;
    if (dd <= d) {
      d = dd;
      best = it;
    }
  }
  return best;
}

/** A value oriented so that larger is better. */
export function oriented(o: Objective, v: number): number {
  return o.kind === "Maximize" ? v : -v;
}

/** The candidates no other candidate beats on both objectives (as the framework's Pareto does: at least as good on both
 * and not equal), among the candidates the record's best is drawn from: comparable and meeting every constraint. */
export function front2(study: StudyModel, xo: Objective, yo: Objective): number[] {
  const pool = study.candidates.flatMap((t) => {
    if (study.markKind(t) !== "ok") return [];
    const x = study.mean(t, xo.metric);
    const y = study.mean(t, yo.metric);
    return x === null || y === null ? [] : [{ n: t.number, u: oriented(xo, x), v: oriented(yo, y) }];
  });
  return pool.filter((a) => !pool.some((b) => b.u >= a.u && b.v >= a.v && (b.u !== a.u || b.v !== a.v))).map((a) => a.n);
}

/**
 * The boundary of the region the given points dominate, in pixels, on axes oriented so that better is right and up:
 * from the left edge at the best y, a step at each point, down to the bottom edge. Returns the line and the region
 * (closed along the bottom-left), or null without points.
 */
export function staircase(points: { px: number; py: number }[], left: number, bottom: number): { line: string; region: string } | null {
  if (!points.length) return null;
  // from the right: a point is on the boundary when it is higher (smaller py) than every point to its right
  const byX = [...points].sort((a, b) => b.px - a.px || a.py - b.py);
  const edge: { px: number; py: number }[] = [];
  let top = Infinity;
  for (const p of byX) {
    if (p.py < top) {
      edge.push(p);
      top = p.py;
    }
  }
  edge.reverse(); // left to right: highest first
  let line = `M${left},${edge[0].py}`;
  edge.forEach((p, i) => {
    line += `H${p.px}`;
    line += `V${i + 1 < edge.length ? edge[i + 1].py : bottom}`;
  });
  return { line, region: `${line}H${left}Z` };
}

type Box = { x0: number; y0: number; x1: number; y1: number };

/** Places labels next to their points, skipping any that would overlap another label, a mark it is not for, or leave
 * the frame. Earlier labels win. */
export function placeLabels(
  labels: { key: string; text: string; x: number; y: number; r: number; cls?: string }[],
  frame: Box,
  marks: { x: number; y: number }[] = [],
  font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif",
): { key: string; text: string; x: number; y: number; anchor: "start" | "end"; cls?: string }[] {
  const placed: Box[] = [];
  const out: { key: string; text: string; x: number; y: number; anchor: "start" | "end"; cls?: string }[] = [];
  const hits = (b: Box) => placed.some((p) => b.x0 < p.x1 && b.x1 > p.x0 && b.y0 < p.y1 && b.y1 > p.y0);
  const covers = (b: Box, self: { x: number; y: number }) =>
    marks.some((m) => (m.x !== self.x || m.y !== self.y) && m.x > b.x0 - 3 && m.x < b.x1 + 3 && m.y > b.y0 - 3 && m.y < b.y1 + 3);
  for (const l of labels) {
    const w = textWidth(l.text, font);
    const h = 12;
    const gap = l.r + 4;
    const tries: { x: number; y: number; anchor: "start" | "end"; box: Box }[] = [
      { x: l.x + gap, y: l.y - gap + 2, anchor: "start", box: { x0: l.x + gap, y0: l.y - gap - h + 5, x1: l.x + gap + w, y1: l.y - gap + 5 } },
      { x: l.x - gap, y: l.y - gap + 2, anchor: "end", box: { x0: l.x - gap - w, y0: l.y - gap - h + 5, x1: l.x - gap, y1: l.y - gap + 5 } },
      { x: l.x + gap, y: l.y + gap + 8, anchor: "start", box: { x0: l.x + gap, y0: l.y + gap - 1, x1: l.x + gap + w, y1: l.y + gap + h - 1 } },
      { x: l.x - gap, y: l.y + gap + 8, anchor: "end", box: { x0: l.x - gap - w, y0: l.y + gap - 1, x1: l.x - gap, y1: l.y + gap + h - 1 } },
    ];
    const ok = tries.find(
      (c) => c.box.x0 >= frame.x0 && c.box.x1 <= frame.x1 && c.box.y0 >= frame.y0 && c.box.y1 <= frame.y1 && !hits(c.box) && !covers(c.box, l),
    );
    if (!ok) continue;
    placed.push(ok.box);
    out.push({ key: l.key, text: l.text, x: ok.x, y: ok.y, anchor: ok.anchor, cls: l.cls });
  }
  return out;
}

/** A label cut to fit a width, with an ellipsis. */
export function fit(text: string, width: number, font?: string): string {
  if (textWidth(text, font) <= width) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (textWidth(`${text.slice(0, mid)}…`, font) <= width) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo)}…` : "…";
}
