// One value per candidate on the shared candidate axis: an objective, a constraint, any other metric, or what the
// methods were told. The panels of a page are stacked on one axis, so a column is the same candidate in every panel.

import { scaleLinear } from "d3-scale";
import { axisFormat } from "../format";
import { limitText } from "../model/goals";
import type { Goal, StudyModel } from "../model/study";
import type { Candidate } from "../record/parse";
import { INK, LIMIT, LIMIT_WASH, MARK, MUTED } from "../encoding/theme";
import { useHover } from "../interaction/hover";
import { useGo } from "../router";
import type { CandidateAxis } from "./axis";
import { Mark, markRadius } from "./kit";

export type Value = { mean: number; ci: [number, number] | null };

const TOP = 10;
const BOTTOM = 8;

function spread(labels: { y: number; text: string; cls: string }[], gap = 13) {
  const out = [...labels].sort((a, b) => a.y - b.y);
  for (let i = 1; i < out.length; i++) if (out[i].y - out[i - 1].y < gap) out[i] = { ...out[i], y: out[i - 1].y + gap };
  return out;
}

export function CandidatePanel({ study, axis, width, height = 150, value, range, goal, bestLine }: {
  study: StudyModel;
  axis: CandidateAxis;
  width: number;
  height?: number;
  value: (t: Candidate) => Value | null;
  range: [number | null, number | null];
  goal?: Goal;
  bestLine?: boolean;
}) {
  const { hover, set } = useHover();
  const go = useGo();
  const values = study.candidates.map(value);
  const means = values.flatMap((v) => (v ? [v.mean] : []));
  const limit = goal?.role === "constraint" ? goal.constraint : null;
  if (limit) means.push(limit.limit);
  // fit the values (and the limit), never past the metric's declared range
  const [lo, hi] = range;
  const min = means.length ? Math.min(...means) : lo ?? 0;
  const max = means.length ? Math.max(...means) : hi ?? 1;
  const pad = (max - min) * 0.1 || Math.abs(max) * 0.1 || 1;
  const fitted = scaleLinear().domain([min - pad, max + pad]).nice(4).domain();
  const y0 = lo === null ? fitted[0] : Math.max(lo, fitted[0]);
  const y1 = hi === null ? fitted[1] : Math.min(hi, fitted[1]);
  const y = scaleLinear().domain([y0, y1]).range([height - BOTTOM, TOP]);
  const [d0, d1] = y.domain();
  const clamp = (v: number) => Math.max(d0, Math.min(d1, v));
  const ticks = y.ticks(4);
  const fmt = axisFormat(ticks);
  const base = study.baseline ? value(study.baseline) : null;
  const r = markRadius(axis.step);
  const whiskers = axis.step >= 6;

  // the value of the best candidate after each candidate (from the iteration records), when best is one candidate
  let path = "";
  let bestNow: number | null = null;
  if (bestLine && !study.bestIsSet) {
    let prev: number | null = null;
    study.candidates.forEach((t, i) => {
      const b = study.bestAfter(t.number)[0];
      const v = b === undefined ? null : value(b)?.mean ?? null;
      if (v === null) return;
      const xa = axis.x(t.number) - (i === 0 ? 0 : axis.step / 2);
      const xb = axis.x(t.number) + axis.step / 2;
      path += prev === null ? `M${xa},${y(v)}H${xb}` : `V${y(v)}H${xb}`;
      prev = v;
    });
    bestNow = prev;
  }

  const refs = spread([
    ...(base ? [{ y: y(clamp(base.mean)), text: "baseline", cls: "ref" }] : []),
    ...(limit ? [{ y: y(clamp(limit.limit)), text: limitText(limit), cls: "ref" }] : []),
    ...(bestNow !== null ? [{ y: y(clamp(bestNow)), text: "best", cls: "ref strong" }] : []),
  ]);

  return (
    <svg width={width} height={height} onMouseLeave={() => set(null)}>
      {limit && (
        <rect
          x={axis.left}
          width={axis.right - axis.left}
          y={limit.kind === "AtMost" ? TOP : y(clamp(limit.limit))}
          height={Math.max(0, limit.kind === "AtMost" ? y(clamp(limit.limit)) - TOP : height - BOTTOM - y(clamp(limit.limit)))}
          fill={LIMIT_WASH}
        />
      )}
      {ticks.map((v) => (
        <g key={v}>
          <line x1={axis.left} x2={axis.right} y1={y(v)} y2={y(v)} className="gridline" />
          <text x={axis.left - 8} y={y(v)} dy="0.32em" textAnchor="end" className="tick">
            {fmt(v)}
          </text>
        </g>
      ))}
      {hover.n !== null && <rect x={axis.x(hover.n) - axis.step / 2} y={TOP - 6} width={axis.step} height={height - TOP - BOTTOM + 12} className="hover-col" />}
      {base && <line x1={axis.left} x2={axis.right} y1={y(clamp(base.mean))} y2={y(clamp(base.mean))} stroke={MARK} strokeOpacity={0.55} strokeDasharray="4 4" />}
      {limit && <line x1={axis.left} x2={axis.right} y1={y(clamp(limit.limit))} y2={y(clamp(limit.limit))} stroke={LIMIT} strokeWidth={1.5} />}
      {path && <path d={path} fill="none" stroke={INK} strokeOpacity={0.6} strokeWidth={1.5} />}
      {refs.map((l) => (
        <text key={l.text} x={axis.right + 8} y={l.y} dy="0.32em" className={l.cls}>
          {l.text}
        </text>
      ))}
      {study.candidates.map((t, i) => {
        const v = values[i];
        if (!v) return null;
        const cx = axis.x(t.number);
        return (
          <g key={t.number}>
            {whiskers && v.ci && <line x1={cx} x2={cx} y1={y(clamp(v.ci[0]))} y2={y(clamp(v.ci[1]))} stroke={MARK} strokeOpacity={0.3} strokeWidth={1.2} />}
            <Mark x={cx} y={y(clamp(v.mean))} kind={study.markKind(t)} r={r} best={study.best.has(t.number)} hovered={hover.n === t.number} />
          </g>
        );
      })}
      {values.every((v) => !v) && (
        <text x={(axis.left + axis.right) / 2} y={height / 2} textAnchor="middle" className="tick" fill={MUTED}>
          no values
        </text>
      )}
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
