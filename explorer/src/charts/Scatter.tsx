// Two objectives against each other, with the shared candidate marks. Both axes are oriented so that better is right
// and up (an arrow says so), so a front is always the upper-right edge and the region it dominates the lower-left. The
// staircase steps through the candidates it is given (the recorded best set, or the 2-D front of the pair), constraints
// on either metric are red limit lines, and the baseline is a dashed crosshair. Used full size (ticks, titles, labels
// on the baseline and the best set) and compact (a cell of the pair matrix: ticks only on the edges it is given).

import { scaleLinear } from "d3-scale";
import { useRef } from "react";
import { INK, INK_2, LIMIT, LIMIT_WASH, MARK, MUTED } from "../encoding/theme";
import { axisFormat } from "../format";
import { limitText } from "../model/goals";
import type { Goal, StudyModel } from "../model/study";
import type { Candidate } from "../record/parse";
import { useHover } from "../interaction/hover";
import { useGo } from "../router";
import { Mark } from "./kit";
import { nearest, placeLabels, staircase } from "./plot";

export type ObjectiveGoal = Extract<Goal, { role: "objective" }>;

type Pt = { t: Candidate; px: number; py: number; x: number; y: number; xci: [number, number] | null; yci: [number, number] | null };

export type ScatterProps = {
  study: StudyModel;
  xg: ObjectiveGoal;
  yg: ObjectiveGoal;
  width: number;
  height: number;
  xDomain: [number, number];
  yDomain: [number, number];
  /** The candidates the staircase steps through; null: no staircase. */
  front: number[] | null;
  compact?: boolean;
  xTicks?: boolean;
  yTicks?: boolean;
  intervals?: boolean;
  /** Whether a candidate matches the highlight (null: no highlight). */
  match?: ((t: Candidate) => boolean) | null;
  selected?: boolean;
  /** A click on no candidate (compact: select this pair). */
  onPick?: () => void;
};

const FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

/** The candidates a scatter of these two metrics plots: a mark and a value on both. */
export function plotted(study: StudyModel, xm: string, ym: string): Candidate[] {
  return study.candidates.filter((t) => study.markKind(t) !== "unmeasured" && study.mean(t, xm) !== null && study.mean(t, ym) !== null);
}

export function Scatter(p: ScatterProps) {
  const { study, xg, yg, width, height, compact } = p;
  const { hover, set } = useHover();
  const go = useGo();
  const svg = useRef<SVGSVGElement>(null);
  const m = compact
    ? { top: 6, right: 6, bottom: p.xTicks ? 22 : 6, left: p.yTicks ? 42 : 6 }
    : { top: 28, right: 28, bottom: 44, left: 62 };
  const left = m.left;
  const right = width - m.right;
  const top = m.top;
  const bottom = height - m.bottom;
  const xMax = xg.objective.kind === "Maximize";
  const yMax = yg.objective.kind === "Maximize";
  const x = scaleLinear().domain(p.xDomain).nice(compact ? 3 : 8).range(xMax ? [left, right] : [right, left]);
  const y = scaleLinear().domain(p.yDomain).nice(compact ? 3 : 8).range(yMax ? [bottom, top] : [top, bottom]);
  const [x0, x1] = x.domain();
  const [y0, y1] = y.domain();
  const cx = (v: number) => x(Math.max(x0, Math.min(x1, v)));
  const cy = (v: number) => y(Math.max(y0, Math.min(y1, v)));
  const xt = x.ticks(compact ? 3 : Math.max(3, Math.min(8, Math.round((right - left) / 90))));
  const yt = y.ticks(compact ? 3 : Math.max(3, Math.min(7, Math.round((bottom - top) / 60))));
  const xf = axisFormat(xt);
  const yf = axisFormat(yt);
  const r = compact ? (right - left < 130 || study.candidates.length > 150 ? 2.5 : 3.5) : study.candidates.length > 150 ? 3.5 : 4.5;

  const pts: Pt[] = plotted(study, xg.metric, yg.metric).map((t) => {
    const ex = study.est(t, xg.metric)!;
    const ey = study.est(t, yg.metric)!;
    return { t, x: ex.mean!, y: ey.mean!, px: cx(ex.mean!), py: cy(ey.mean!), xci: ex.ci, yci: ey.ci };
  });
  const byN = new Map(pts.map((q) => [q.t.number, q]));
  const match = p.match ?? null;
  const dim = (t: Candidate) => match !== null && !match(t);
  const stairs = p.front ? staircase(p.front.flatMap((n) => (byN.has(n) ? [byN.get(n)!] : [])), left, bottom) : null;
  const base = study.baseline ? byN.get(study.baseline.number) : undefined;

  // constraints on either axis' metric: a limit line, washed on its infeasible side
  const limits = study.constraints.flatMap((g) => {
    const out: { key: string; axis: "x" | "y"; at: number; from: number; to: number; text: string }[] = [];
    const c = g.constraint;
    if (g.metric === xg.metric && c.limit >= x0 && c.limit <= x1) out.push({ key: `x${g.index}`, axis: "x", at: x(c.limit), from: x(c.limit), to: x(c.kind === "AtMost" ? x1 : x0), text: limitText(c) });
    if (g.metric === yg.metric && c.limit >= y0 && c.limit <= y1) out.push({ key: `y${g.index}`, axis: "y", at: y(c.limit), from: y(c.limit), to: y(c.kind === "AtMost" ? y1 : y0), text: limitText(c) });
    return out;
  });

  const order = [...pts].sort((a, b) => Number(!dim(a.t)) - Number(!dim(b.t)) || Number(study.best.has(a.t.number)) - Number(study.best.has(b.t.number)));
  const hovered = hover.n !== null ? byN.get(hover.n) : undefined;

  const labels = compact
    ? []
    : placeLabels(
        [
          ...(base ? [{ key: "baseline", text: "baseline", x: base.px, y: base.py, r: r + 3.5 }] : []),
          ...pts.filter((q) => study.best.has(q.t.number) && q !== base).map((q) => ({ key: `n${q.t.number}`, text: `#${q.t.number}`, x: q.px, y: q.py, r: r + 3.5, cls: "strong" })),
        ],
        { x0: left, y0: top, x1: right, y1: bottom },
        pts.map((q) => ({ x: q.px, y: q.py })),
        FONT,
      );

  const local = (e: React.MouseEvent) => {
    const b = svg.current!.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top] as const;
  };
  const near = (e: React.MouseEvent) => {
    const [mx, my] = local(e);
    return nearest(pts, (q) => q.px, (q) => q.py, mx, my, compact ? 10 : 16);
  };

  return (
    <svg
      ref={svg}
      width={width}
      height={height}
      className={compact ? `pair-cell${p.selected ? " on" : ""}` : undefined}
      onMouseMove={(e) => {
        const q = near(e);
        set(q ? q.t.number : null, e);
      }}
      onMouseLeave={() => set(null)}
      onClick={(e) => {
        const q = near(e);
        if (q) go(study.id, { page: "candidate", n: q.t.number });
        else p.onPick?.();
      }}
      style={{ cursor: "pointer" }}
    >
      {compact && <rect x={left} y={top} width={right - left} height={bottom - top} className="cell-frame" />}
      {limits.map((l) =>
        l.axis === "x" ? (
          <rect key={`w${l.key}`} x={Math.min(l.from, l.to)} width={Math.abs(l.to - l.from)} y={top} height={bottom - top} fill={LIMIT_WASH} />
        ) : (
          <rect key={`w${l.key}`} x={left} width={right - left} y={Math.min(l.from, l.to)} height={Math.abs(l.to - l.from)} fill={LIMIT_WASH} />
        ),
      )}
      {(!compact || p.xTicks) &&
        xt.map((v) => (
          <g key={`x${v}`}>
            {!compact && <line x1={x(v)} x2={x(v)} y1={top} y2={bottom} className="gridline" />}
            <text x={x(v)} y={bottom + 14} textAnchor={x(v) < left + 14 ? "start" : x(v) > right - 14 ? "end" : "middle"} className="tick">
              {xf(v)}
            </text>
          </g>
        ))}
      {(!compact || p.yTicks) &&
        yt.map((v) => (
          <g key={`y${v}`}>
            {!compact && <line x1={left} x2={right} y1={y(v)} y2={y(v)} className="gridline" />}
            <text x={left - 7} y={y(v)} dy="0.32em" textAnchor="end" className="tick">
              {yf(v)}
            </text>
          </g>
        ))}
      {stairs && (
        <g>
          <path d={stairs.region} fill={INK} fillOpacity={0.04} />
          <path d={stairs.line} fill="none" stroke={INK} strokeOpacity={0.65} strokeWidth={compact ? 1.2 : 1.5} strokeLinejoin="round" />
        </g>
      )}
      {limits.map((l) =>
        l.axis === "x" ? (
          <g key={l.key}>
            <line x1={l.at} x2={l.at} y1={top} y2={bottom} stroke={LIMIT} strokeWidth={1.5} />
            {!compact && (
              <text x={l.at + 5} y={top + 11} className="ref">
                {l.text}
              </text>
            )}
          </g>
        ) : (
          <g key={l.key}>
            <line x1={left} x2={right} y1={l.at} y2={l.at} stroke={LIMIT} strokeWidth={1.5} />
            {!compact && (
              <text x={right - 5} y={l.at - 5} textAnchor="end" className="ref">
                {l.text}
              </text>
            )}
          </g>
        ),
      )}
      {base && !compact && (
        <g stroke={MARK} strokeOpacity={0.4} strokeDasharray="4 4">
          <line x1={base.px} x2={base.px} y1={top} y2={bottom} />
          <line x1={left} x2={right} y1={base.py} y2={base.py} />
        </g>
      )}
      {p.intervals &&
        pts.map((q) => (
          <g key={`ci${q.t.number}`} stroke={MARK} strokeOpacity={dim(q.t) ? 0.08 : 0.3} strokeWidth={1.2}>
            {q.xci && <line x1={cx(q.xci[0])} x2={cx(q.xci[1])} y1={q.py} y2={q.py} />}
            {q.yci && <line x1={q.px} x2={q.px} y1={cy(q.yci[0])} y2={cy(q.yci[1])} />}
          </g>
        ))}
      {order.map((q) => (
        <g key={q.t.number} opacity={dim(q.t) ? 0.16 : 1}>
          <Mark x={q.px} y={q.py} kind={study.markKind(q.t)} r={r} best={study.best.has(q.t.number)} />
        </g>
      ))}
      {hovered && <Mark x={hovered.px} y={hovered.py} kind={study.markKind(hovered.t)} r={r} best={study.best.has(hovered.t.number)} hovered />}
      {labels.map((l) => (
        <text key={l.key} x={l.x} y={l.y} textAnchor={l.anchor} className={`ref${l.cls ? ` ${l.cls}` : ""}`} opacity={l.key.startsWith("n") && dim(study.byNumber.get(Number(l.key.slice(1)))!) ? 0.3 : 1}>
          {l.text}
        </text>
      ))}
      {!compact && (
        <g>
          <text x={left} y={14} className="axis-title">
            {yg.metric}
            <tspan fill={MUTED} dx={8}>
              better ↑
            </tspan>
          </text>
          <text x={right} y={height - 6} textAnchor="end" className="axis-title">
            {xg.metric}
            <tspan fill={MUTED} dx={8}>
              better →
            </tspan>
          </text>
        </g>
      )}
      {!pts.length && (
        <text x={(left + right) / 2} y={(top + bottom) / 2} textAnchor="middle" className="tick" fill={INK_2}>
          no values
        </text>
      )}
    </svg>
  );
}
