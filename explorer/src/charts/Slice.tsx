// One variable of the space against the candidates. A slice: the variable's value (x: its declared domain, or one lane
// per option) against a goal (y), every candidate the variable was active in as the shared mark (inactive candidates
// are absent). A trace: the variable's value (y) against the candidate number on the shared candidate axis, aligned
// with the schedule track. Options are positions (lanes), never hues; a text variable's values are its versions (v0 is
// the default).

import { scaleLinear, scaleLog } from "d3-scale";
import { useRef } from "react";
import { INK_2, LIMIT, LIMIT_WASH, MARK, MUTED } from "../encoding/theme";
import { axisFormat } from "../format";
import { limitText } from "../model/goals";
import type { Goal, StudyModel } from "../model/study";
import type { SpaceRow } from "../record/schema";
import type { Candidate } from "../record/parse";
import { useHover } from "../interaction/hover";
import { useGo } from "../router";
import { textWidth, type CandidateAxis } from "./axis";
import { markRadius, Mark, niceTicks } from "./kit";
import { fit, nearest } from "./plot";

const FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
/** Lanes shown before only the options that occur are kept. */
const MAX_LANES = 8;

export type Lane = { value: string; label: string; candidates: number };

/** A categorical variable's lanes: every option while there are few, else only those some candidate chose (``hidden``:
 * how many options are left out). Text variables: one lane per version. */
export function lanesOf(study: StudyModel, row: SpaceRow): { lanes: Lane[]; hidden: number } {
  const count = (v: string) => study.candidates.filter((t) => row.key in t.values && String(t.values[row.key]) === v).length;
  if (row.kind === "text") {
    const versions = study.textVersions(row.key);
    return { lanes: versions.map((v, i) => ({ value: `v${i}`, label: `v${i}`, candidates: v.candidates.length })), hidden: 0 };
  }
  const all = (row.options ?? []).map((c) => ({ value: String(c), label: String(c), candidates: count(String(c)) }));
  if (all.length <= MAX_LANES) return { lanes: all, hidden: 0 };
  const used = all.filter((l) => l.candidates > 0);
  return { lanes: used, hidden: all.length - used.length };
}

/** The lane a candidate's value falls in (text: its version), or null when the variable was not active in it. */
function laneValue(study: StudyModel, row: SpaceRow, t: Candidate): string | null {
  if (!(row.key in t.values)) return null;
  const v = t.values[row.key];
  if (row.kind === "text") return `v${study.textVersions(row.key).findIndex((x) => x.text === v)}`;
  return String(v);
}

export const isNumeric = (row: SpaceRow) => row.kind === "float" || row.kind === "int";

/** A numeric variable's scale over its declared domain (widened to every value tried). */
function numericScale(study: StudyModel, row: SpaceRow, range: [number, number]) {
  const vs = study.candidates.map((t) => t.values[row.key]).filter((v): v is number => typeof v === "number");
  let lo = Math.min(row.min_value ?? Infinity, ...vs);
  let hi = Math.max(row.max_value ?? -Infinity, ...vs);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) [lo, hi] = [0, 1];
  if (lo === hi) [lo, hi] = [lo - 1, hi + 1];
  if (row.log && lo > 0) return { scale: scaleLog().domain([lo, hi]).range(range), log: true };
  return { scale: scaleLinear().domain([lo, hi]).range(range), log: false };
}

function lanePositions(lanes: Lane[], a: number, b: number) {
  const step = (b - a) / Math.max(1, lanes.length);
  const at = new Map(lanes.map((l, i) => [l.value, a + step * (i + 0.5)]));
  return { at, step };
}

/** A spread within a lane that keeps candidates apart without meaning anything (the same for a candidate
 * everywhere). */
function jitter(n: number): number {
  return ((n * 0.6180339887) % 1) - 0.5;
}

type Pt = { t: Candidate; px: number; py: number };

export function SlicePlot({ study, row, goal, width, height, yDomain }: {
  study: StudyModel;
  row: SpaceRow;
  goal: Goal;
  width: number;
  height: number;
  yDomain: [number, number];
}) {
  const { hover, set } = useHover();
  const go = useGo();
  const svg = useRef<SVGSVGElement>(null);
  const numeric = isNumeric(row);
  const { lanes, hidden } = numeric ? { lanes: [], hidden: 0 } : lanesOf(study, row);
  const m = { top: 8, right: 12, bottom: numeric ? 26 : 40, left: 44 };
  const left = m.left;
  const right = width - m.right;
  const top = m.top;
  const bottom = height - m.bottom;
  const y = scaleLinear().domain(yDomain).nice(4).range([bottom, top]);
  const [d0, d1] = y.domain();
  const cy = (v: number) => y(Math.max(d0, Math.min(d1, v)));
  const yt = y.ticks(4);
  const yf = axisFormat(yt);
  const limit = goal.role === "constraint" ? goal.constraint : null;

  const xs = numeric ? numericScale(study, row, [left + 6, right - 6]) : null;
  const lp = numeric ? null : lanePositions(lanes, left, right);
  const crowd = lanes.reduce((m, l) => Math.max(m, l.candidates), 0);
  const spread = lp ? Math.min(lp.step * 0.75, 24 + 4 * Math.sqrt(crowd)) : 0;
  const pts: Pt[] = study.candidates.flatMap((t) => {
    if (study.markKind(t) === "unmeasured" || !(row.key in t.values)) return [];
    const v = study.mean(t, goal.metric);
    if (v === null) return [];
    let px: number;
    if (xs) {
      const raw = t.values[row.key];
      if (typeof raw !== "number") return [];
      px = xs.scale(raw);
    } else {
      const at = lp!.at.get(laneValue(study, row, t) ?? "");
      if (at === undefined) return [];
      px = at + jitter(t.number) * spread;
    }
    return [{ t, px, py: cy(v) }];
  });
  const r = markRadius(pts.length > 120 ? 6 : pts.length > 40 ? 9 : 14);
  const base = study.baseline ? study.mean(study.baseline, goal.metric) : null;
  const byN = new Map(pts.map((p) => [p.t.number, p]));
  const hovered = hover.n !== null ? byN.get(hover.n) : undefined;
  const ordered = [...pts].sort((a, b) => Number(study.best.has(a.t.number)) - Number(study.best.has(b.t.number)));

  const xt = xs ? niceTicks(xs.scale as never, xs.log, Math.max(2, Math.min(6, Math.round((right - left) / 70))), row.kind === "int") : [];
  const xf = axisFormat(xt);
  const near = (e: React.MouseEvent) => {
    const b = svg.current!.getBoundingClientRect();
    return nearest(pts, (p) => p.px, (p) => p.py, e.clientX - b.left, e.clientY - b.top, 14);
  };

  return (
    <svg
      ref={svg}
      width={width}
      height={height}
      onMouseMove={(e) => {
        const p = near(e);
        set(p ? p.t.number : null, e);
      }}
      onMouseLeave={() => set(null)}
      onClick={(e) => {
        const p = near(e);
        if (p) go(study.id, { page: "candidate", n: p.t.number });
      }}
      style={{ cursor: pts.length ? "pointer" : undefined }}
    >
      {limit && (
        <rect
          x={left}
          width={right - left}
          y={limit.kind === "AtMost" ? top : cy(limit.limit)}
          height={Math.max(0, limit.kind === "AtMost" ? cy(limit.limit) - top : bottom - cy(limit.limit))}
          fill={LIMIT_WASH}
        />
      )}
      {yt.map((v) => (
        <g key={v}>
          <line x1={left} x2={right} y1={y(v)} y2={y(v)} className="gridline" />
          <text x={left - 7} y={y(v)} dy="0.32em" textAnchor="end" className="tick">
            {yf(v)}
          </text>
        </g>
      ))}
      {xs &&
        xt.map((v) => (
          <text key={v} x={xs.scale(v)} y={bottom + 19} textAnchor="middle" className="tick">
            {xf(v)}
          </text>
        ))}
      {xs && <line x1={left} x2={right} y1={bottom} y2={bottom} className="axisline" />}
      {lp &&
        lanes.map((l, i) => (
          <g key={l.value}>
            {i > 0 && <line x1={left + lp.step * i} x2={left + lp.step * i} y1={top} y2={bottom} className="gridline" />}
            <text x={lp.at.get(l.value)} y={bottom + 20} textAnchor="middle" className={l.candidates ? "lane-label" : "lane-label none"}>
              <title>{`${l.label}: ${l.candidates} candidate${l.candidates === 1 ? "" : "s"}`}</title>
              {fit(l.label, lp.step - 6, FONT)}
            </text>
            <text x={lp.at.get(l.value)} y={bottom + 33} textAnchor="middle" className="tick">
              {`${l.candidates} candidate${l.candidates === 1 ? "" : "s"}`}
            </text>
          </g>
        ))}
      {lp && <line x1={left} x2={right} y1={bottom} y2={bottom} className="axisline" />}
      {base !== null && <line x1={left} x2={right} y1={cy(base)} y2={cy(base)} stroke={MARK} strokeOpacity={0.5} strokeDasharray="4 4" />}
      {limit && (
        <g>
          <line x1={left} x2={right} y1={cy(limit.limit)} y2={cy(limit.limit)} stroke={LIMIT} strokeWidth={1.5} />
          <title>{limitText(limit)}</title>
        </g>
      )}
      {ordered.map((p) => (
        <Mark key={p.t.number} x={p.px} y={p.py} kind={study.markKind(p.t)} r={r} best={study.best.has(p.t.number)} />
      ))}
      {hovered && <Mark x={hovered.px} y={hovered.py} kind={study.markKind(hovered.t)} r={r} best={study.best.has(hovered.t.number)} hovered />}
      {!pts.length && (
        <text x={(left + right) / 2} y={(top + bottom) / 2} textAnchor="middle" className="tick" fill={MUTED}>
          no values
        </text>
      )}
      {hidden > 0 && (
        <text x={right} y={top + 10} textAnchor="end" className="tick">
          +{hidden} never chosen
        </text>
      )}
    </svg>
  );
}

/** The left margin a trace's lane labels need (with the schedule track's labels, for alignment). */
export function traceLabelWidth(study: StudyModel, rows: SpaceRow[]): number {
  let w = 0;
  for (const row of rows) {
    if (isNumeric(row) || row.kind === "text") continue;
    for (const l of lanesOf(study, row).lanes) w = Math.max(w, textWidth(l.label, FONT));
  }
  return Math.min(220, w + 24);
}

export function traceHeight(study: StudyModel, row: SpaceRow): number {
  if (isNumeric(row) || row.kind === "text") return 120;
  const n = lanesOf(study, row).lanes.length;
  return Math.max(54, Math.min(200, n * 20)) + 16;
}

export function TracePlot({ study, row, axis, width }: { study: StudyModel; row: SpaceRow; axis: CandidateAxis; width: number }) {
  const { hover, set } = useHover();
  const go = useGo();
  const height = traceHeight(study, row);
  const top = 8;
  const bottom = height - 8;
  const numeric = isNumeric(row);
  const text = row.kind === "text";
  const r = markRadius(axis.step);
  const xs = Math.min(3, r * 0.8); // the half-size of a × (nothing measured)

  // y: the numeric value over its domain; the version index of a text; one lane per option otherwise
  let yOf: (t: Candidate) => number | null;
  let ticks: { y: number; label: string }[] = [];
  let lanes: Lane[] = [];
  if (numeric) {
    const { scale, log } = numericScale(study, row, [bottom - 4, top + 4]);
    yOf = (t) => (typeof t.values[row.key] === "number" ? scale(t.values[row.key] as number) : null);
    const tv = niceTicks(scale as never, log, 3, row.kind === "int");
    const f = axisFormat(tv);
    ticks = tv.map((v) => ({ y: scale(v), label: f(v) }));
  } else if (text) {
    const versions = study.textVersions(row.key);
    const scale = scaleLinear().domain([0, Math.max(1, versions.length - 1)]).range([bottom - 4, top + 4]);
    yOf = (t) => {
      if (!(row.key in t.values)) return null;
      const i = versions.findIndex((v) => v.text === t.values[row.key]);
      return i < 0 ? null : scale(i);
    };
    ticks = scale.ticks(Math.min(5, versions.length)).filter(Number.isInteger).map((v) => ({ y: scale(v), label: `v${v}` }));
  } else {
    lanes = lanesOf(study, row).lanes;
    const step = (bottom - top) / Math.max(1, lanes.length);
    const at = new Map(lanes.map((l, i) => [l.value, top + step * (i + 0.5)]));
    yOf = (t) => at.get(laneValue(study, row, t) ?? "") ?? null;
    const every = Math.ceil(12 / step);
    ticks = lanes.flatMap((l, i) => (i % every === 0 ? [{ y: at.get(l.value)!, label: fit(l.label, axis.left - 18, FONT) }] : []));
  }
  const base = study.baseline ? yOf(study.baseline) : null;

  return (
    <svg width={width} height={height} onMouseLeave={() => set(null)}>
      {ticks.map((k) => (
        <g key={`${k.label}${k.y}`}>
          <line x1={axis.left} x2={axis.right} y1={k.y} y2={k.y} className="gridline" />
          <text x={axis.left - 8} y={k.y} dy="0.32em" textAnchor="end" className={lanes.length ? "lane-label" : "tick"}>
            {k.label}
          </text>
        </g>
      ))}
      {hover.n !== null && <rect x={axis.x(hover.n) - axis.step / 2} y={top - 6} width={axis.step} height={bottom - top + 12} className="hover-col" />}
      {base !== null && numeric && <line x1={axis.left} x2={axis.right} y1={base} y2={base} stroke={MARK} strokeOpacity={0.5} strokeDasharray="4 4" />}
      {study.candidates.map((t) => {
        const py = yOf(t);
        if (py === null) return null;
        const px = axis.x(t.number);
        // a candidate that measured nothing still tried the value: its × (as in the schedule track's status row)
        if (study.markKind(t) === "unmeasured")
          return <path key={t.number} d={`M${px - xs},${py - xs}L${px + xs},${py + xs}M${px - xs},${py + xs}L${px + xs},${py - xs}`} stroke={INK_2} strokeWidth={xs > 2.5 ? 1.5 : 1.2} />;
        return <Mark key={t.number} x={px} y={py} kind={study.markKind(t)} r={r} best={study.best.has(t.number)} hovered={hover.n === t.number} />;
      })}
      {study.candidates.map((t) => (
        <rect
          key={t.number}
          x={axis.x(t.number) - axis.step / 2}
          y={0}
          width={axis.step}
          height={height}
          fill="transparent"
          style={{ cursor: "pointer" }}
          onMouseMove={(e) => set(t.number, e)}
          onClick={() => go(study.id, { page: "candidate", n: t.number })}
        />
      ))}
    </svg>
  );
}

