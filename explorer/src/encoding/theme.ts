// The explorer's visual grammar, one encoding per concept:
// - a candidate is a column on the shared candidate axis; its mark's y is the mean, a thin whisker its 95% interval;
// - feasibility is shape, never color: filled = comparable and feasible, hollow = breaks a constraint, small grey dot =
//   not comparable; nothing measured = no mark (an × in the status row);
// - the baseline is candidate 0's mark plus a dashed reference line; best (or the Pareto set) is a ring;
// - a constraint is a red limit line with a faint wash on its infeasible side (red is reserved for limits);
// - hue identifies blocks, and only blocks: the schedule track's change rows and block chips; marks stay neutral;
// - categorical choices are never a hue: they are text (key=option), positions, or a highlight.

import type { StudyModel } from "../model/study";

export const INK = "#0b0b0b";
export const INK_2 = "#52514e";
export const MUTED = "#898781";
export const GRID = "#e8e7e1";
export const AXIS = "#c3c2b7";
export const SURFACE = "#fcfcfb";
export const MARK = "#2b2a27";
export const LIMIT = "#d03b3b";
export const LIMIT_WASH = "rgba(208, 59, 59, 0.06)";

// Categorical slots in fixed order (validated: adjacent CVD ΔE ≥ 9.1, normal-vision ≥ 19.6); red is left out (limits).
const BLOCK_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"];

/** A block's color: its slot by position in the study's block order; past the palette, neutral (its label carries it). */
export function blockColor(study: StudyModel, block: string): string {
  const i = study.blockOrder.indexOf(block);
  return i >= 0 && i < BLOCK_SLOTS.length ? BLOCK_SLOTS[i] : MUTED;
}
