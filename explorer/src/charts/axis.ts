// The candidate axis every candidate chart shares: same margins and scale, so the schedule track and every panel line
// up.

import { scaleLinear } from "d3-scale";
import type { StudyModel } from "../model/study";

export const RIGHT = 88; // reference-line labels (baseline, limit, best)
export const LABEL_FONT = "11.5px system-ui, -apple-system, 'Segoe UI', sans-serif";

let canvas: HTMLCanvasElement | null = null;
export function textWidth(s: string, font = LABEL_FONT): number {
  canvas ??= document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  ctx.font = font;
  return ctx.measureText(s).width;
}

/** The left margin: wide enough for the longest row label of the schedule track (and the y ticks). */
export function leftMargin(labels: string[]): number {
  return Math.ceil(Math.max(72, ...labels.map((l) => textWidth(l) + 40)));
}

/** The last candidate slot the axis shows: the highest candidate number proposed so far. */
export function lastSlot(study: StudyModel): number {
  return Math.max(0, ...study.candidates.map((t) => t.number));
}

export type CandidateAxis = { x: (n: number) => number; step: number; left: number; right: number; ticks: number[] };

export function candidateAxis(study: StudyModel, width: number, left: number): CandidateAxis {
  const last = lastSlot(study);
  const scale = scaleLinear().domain([-0.5, last + 0.5]).range([left, width - RIGHT]);
  const step = scale(1) - scale(0);
  const every = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find((k) => k * step >= 26) ?? 1000;
  const ticks: number[] = [];
  for (let n = 0; n <= last; n += every) ticks.push(n);
  return { x: (n) => scale(n), step, left, right: width - RIGHT, ticks };
}
