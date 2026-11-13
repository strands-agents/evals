// The effects model's estimates (effects.json): for a goal metric, each level's change from its variable's reference
// level, with a 90% credible interval and the probability it is better for the goal; and whether the model fits.
// An interval that includes zero is drawn faint: the data do not settle that effect.

import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { textWidth } from "../charts/axis";
import { useWidth } from "../charts/kit";
import { INK, MARK, MUTED } from "../encoding/theme";
import { axisFormat, num } from "../format";
import { goalTitle } from "../model/goals";
import type { StudyModel } from "../model/study";
import { pBetter, type Effect, type Effects, type EffectsGoal } from "../record/effects";

const ROW = 22;
const HEAD = 22;

function signed(v: number): string {
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${num(Math.abs(v))}`;
}

export function FitTag({ effects, metric }: { effects: Effects; metric: string }) {
  const d = effects.metrics[metric]?.diagnostics;
  if (!d) return null;
  return (
    <span className={`tag${d.flagged ? " warn" : ""}`} title={d.messages.join("\n") || undefined}>
      {d.flagged ? "fit flagged" : "fit ok"}
    </span>
  );
}

/** The forest plot for one goal: rows grouped by variable, in the study's space order. */
export function EffectsPlot({ study, effects, goal, keys }: { study: StudyModel; effects: Effects; goal: EffectsGoal; keys?: string[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const order = new Map(study.space.map((s, i) => [s.key, i]));
  const rows = effects.effects
    .filter((e) => e.metric === goal.metric && (!keys || keys.includes(e.key)))
    .sort((a, b) => (order.get(a.key) ?? 0) - (order.get(b.key) ?? 0));
  if (!rows.length) return <div className="muted small">No effects</div>;
  const groups: { key: string; reference: string; rows: Effect[] }[] = [];
  for (const e of rows) {
    const last = groups.at(-1);
    if (last?.key === e.key) last.rows.push(e);
    else groups.push({ key: e.key, reference: e.reference, rows: [e] });
  }
  const lo = Math.min(0, ...rows.map((e) => e.ci90[0]));
  const hi = Math.max(0, ...rows.map((e) => e.ci90[1]));
  const mono = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  const sans = "12px system-ui, -apple-system, 'Segoe UI', sans-serif";
  const texts = keys
    ? groups.flatMap((g) => g.rows.map((e) => textWidth(`${e.level} vs ${g.reference}`, sans) + 12))
    : groups.flatMap((g) => [textWidth(g.key, mono) + textWidth(` vs ${g.reference}`, sans) + 12, ...g.rows.map((e) => textWidth(e.level, sans) + 28)]);
  const right = Math.ceil(textWidth("change · P(better)", "11px system-ui, -apple-system, sans-serif")) + 18;
  const label = Math.min(Math.max(60, ...texts), width * 0.45);
  const x = scaleLinear().domain([lo, hi]).nice(4).range([label, Math.max(label + 60, width - right)]);
  const ticks = x.ticks(Math.max(2, Math.floor((width - right - label) / 70)));
  const fmt = axisFormat(ticks);
  let y = HEAD;
  const placed = groups.map((g) => {
    const top = y;
    y += ROW * (g.rows.length + (keys ? 0 : 1));
    return { g, top };
  });
  const height = y + 6;
  return (
    <div ref={ref} className="effects-plot">
      <svg width={width} height={height}>
        {ticks.map((v, i) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={HEAD - 4} y2={height - 4} className="gridline" />
            <text x={x(v)} y={12} textAnchor={i === ticks.length - 1 ? "end" : i === 0 ? "start" : "middle"} className="tick">
              {fmt(v)}
            </text>
          </g>
        ))}
        <line x1={x(0)} x2={x(0)} y1={HEAD - 4} y2={height - 4} stroke={INK} strokeOpacity={0.5} />
        <text x={width - right + 10} y={12} className="tick">
          change · P(better)
        </text>
        {placed.map(({ g, top }) => (
          <g key={g.key}>
            {!keys && (
              <text x={4} y={top + ROW / 2} dy="0.32em" className="effects-key">
                {g.key} <tspan className="effects-ref">vs {g.reference}</tspan>
              </text>
            )}
            {g.rows.map((e, i) => {
              const cy = top + ROW * (i + (keys ? 0 : 1)) + ROW / 2;
              const settled = e.ci90[0] > 0 || e.ci90[1] < 0;
              const p = pBetter(e, goal.better);
              return (
                <g key={e.level} opacity={settled ? 1 : 0.45}>
                  <title>{`${e.key} = ${e.level} vs ${g.reference}: ${signed(e.mean)} [${num(e.ci90[0])} – ${num(e.ci90[1])}], P(better) ${p.toFixed(2)}`}</title>
                  <text x={keys ? 4 : 20} y={cy} dy="0.32em" className="effects-level">
                    {keys ? `${e.level} vs ${g.reference}` : e.level}
                  </text>
                  <line x1={x(e.ci90[0])} x2={x(e.ci90[1])} y1={cy} y2={cy} stroke={MARK} strokeWidth={2} strokeLinecap="round" />
                  <circle cx={x(e.mean)} cy={cy} r={4} fill={MARK} stroke="#fff" strokeWidth={1.5} />
                  <text x={width - right + 8} y={cy} dy="0.32em" className="tick" fill={MUTED}>
                    {signed(e.mean)} · {p.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

/** The Overview's Effects section: one goal at a time. */
export function EffectsSection({ study, effects }: { study: StudyModel; effects: Effects | null }) {
  const goals = effects?.goals ?? [];
  const [pick, setPick] = useState(0);
  if (!effects)
    return (
      <div className="muted small">
        Not fitted · <span className="mono">python -m strands_evals.optimization.effects {study.id}</span>
      </div>
    );
  if (!goals.length || !effects.effects.length) return <div className="muted small">{effects.reason || "No goals"}</div>;
  const goal = goals[Math.min(pick, goals.length - 1)];
  const model = goal.role === "objective" ? study.objectives[goal.index] : study.constraints[goal.index];
  const m = effects.metrics[goal.metric];
  return (
    <div className="effects">
      <div className="effects-bar">
        <select value={pick} onChange={(e) => setPick(Number(e.target.value))}>
          {goals.map((g, i) => {
            const sg = g.role === "objective" ? study.objectives[g.index] : study.constraints[g.index];
            return (
              <option key={i} value={i}>
                {sg ? goalTitle(sg) : g.metric}
              </option>
            );
          })}
        </select>
        <FitTag effects={effects} metric={goal.metric} />
        <span className="muted small">{m?.observations} observations · 90% intervals</span>
        {effects.omitted.length > 0 && (
          <span className="muted small" title={effects.omitted.map((o) => `${o.key}: ${o.why}`).join("\n")}>
            · not modeled: {effects.omitted.map((o) => o.key).join(", ")}
          </span>
        )}
      </div>
      {model && <EffectsPlot study={study} effects={effects} goal={goal} />}
    </div>
  );
}
