import { Glyph, Legend, LegendItem } from "../components/Legend";
// What one objective costs in another. Two objectives: one scatter. Three or more: every pair as a small multiple, and
// the same scatter for the pair selected. Marks are the shared candidate marks (rings: the recorded best set); the
// staircase steps through the recorded Pareto set (two objectives) or the 2-D front of the pair (more), named so. A
// highlight picks any variable and a value or range: matching candidates stay ink, the rest fade. Options are never a
// hue.

import { useMemo, useState } from "react";
import { useWidth } from "../charts/kit";
import { front2, valueDomain } from "../charts/plot";
import { plotted, Scatter, type ObjectiveGoal } from "../charts/Scatter";
import { CandidateTip } from "../components/CandidateTip";
import { INK, MARK } from "../encoding/theme";
import { num } from "../format";
import { goalTitle } from "../model/goals";
import type { StudyModel } from "../model/study";
import type { SpaceRow } from "../record/schema";
import type { Candidate } from "../record/parse";
import { Tooltip } from "../interaction/hover";
import { href } from "../router";
import "./TradeOff.css";

// ---- the highlight ------------------------------------------------------------------------------------------------

type Highlight = { key: string; value: string } | { key: string; lo: number; hi: number };

const NUMERIC = new Set(["float", "int"]);

/** The highlight's values for a categorical or text variable: each option (or text version) with its candidates. */
function valuesOf(study: StudyModel, row: SpaceRow): { value: string; label: string; candidates: number }[] {
  if (row.kind === "text")
    return study.textVersions(row.key).map((v, i) => ({ value: `v${i}`, label: `v${i}${v.first !== null ? ` · first #${v.first}` : ""}`, candidates: v.candidates.length }));
  return (row.options ?? []).map((c) => {
    const value = String(c);
    return { value, label: value, candidates: study.candidates.filter((t) => row.key in t.values && String(t.values[row.key]) === value).length };
  });
}

function matcher(study: StudyModel, h: Highlight | null): ((t: Candidate) => boolean) | null {
  if (!h) return null;
  const row = study.spaceByKey.get(h.key);
  if (!row) return null;
  if ("lo" in h) return (t) => typeof t.values[h.key] === "number" && (t.values[h.key] as number) >= h.lo && (t.values[h.key] as number) <= h.hi;
  if (row.kind === "text") {
    const text = study.textVersions(h.key)[Number(h.value.slice(1))]?.text;
    return (t) => t.values[h.key] === text;
  }
  return (t) => h.key in t.values && String(t.values[h.key]) === h.value;
}

function highlightText(h: Highlight): string {
  return "lo" in h ? `${num(h.lo)} ≤ ${h.key} ≤ ${num(h.hi)}` : `${h.key} = ${h.value}`;
}

function HighlightControl({ study, h, set }: { study: StudyModel; h: Highlight | null; set: (h: Highlight | null) => void }) {
  const row = h ? study.spaceByKey.get(h.key) : undefined;
  const groups = [
    ...study.blockOrder.map((b) => ({ block: b as string | null, rows: study.space.filter((s) => study.blockOf.get(s.key) === b) })),
    { block: null, rows: study.space.filter((s) => !study.blockOf.has(s.key)) },
  ].filter((g) => g.rows.length);
  const pick = (key: string) => {
    const r = study.spaceByKey.get(key);
    if (!r) return set(null);
    if (NUMERIC.has(r.kind)) {
      const vs = study.candidates.map((t) => t.values[key]).filter((v): v is number => typeof v === "number");
      set({ key, lo: vs.length ? Math.min(...vs) : r.min_value ?? 0, hi: vs.length ? Math.max(...vs) : r.max_value ?? 0 });
    } else {
      const vals = valuesOf(study, r);
      set({ key, value: (vals.find((v) => v.candidates > 0) ?? vals[0])?.value ?? "" });
    }
  };
  const match = matcher(study, h);
  const count = match ? study.candidates.filter(match).length : 0;
  return (
    <span className="ctl">
      <span className="ctl-k">highlight</span>
      <select value={h?.key ?? ""} onChange={(e) => pick(e.target.value)} className={h ? "" : "idle"}>
        <option value="">none</option>
        {groups.map((g) => (
          <optgroup key={g.block ?? ""} label={g.block ?? "no block"}>
            {g.rows.map((s) => (
              <option key={s.key} value={s.key}>
                {s.key}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      {h && row && "value" in h && (
        <>
          <span className="ctl-k">=</span>
          <select value={h.value} onChange={(e) => set({ key: h.key, value: e.target.value })}>
            {valuesOf(study, row).map((v) => (
              <option key={v.value} value={v.value}>
                {v.label} ({v.candidates})
              </option>
            ))}
          </select>
        </>
      )}
      {h && row && "lo" in h && (
        <>
          <span className="ctl-k">from</span>
          <input type="number" step="any" value={h.lo} onChange={(e) => set({ ...h, lo: Number(e.target.value) })} />
          <span className="ctl-k">to</span>
          <input type="number" step="any" value={h.hi} onChange={(e) => set({ ...h, hi: Number(e.target.value) })} />
        </>
      )}
      {h && (
        <>
          <span className="ctl-count">
            {count} of {study.candidates.length} candidates
          </span>
          <button className="ctl-clear" onClick={() => set(null)} title="clear">
            ×
          </button>
        </>
      )}
    </span>
  );
}

// ---- the page -----------------------------------------------------------------------------------------------------

function TradeOffLegend({ study, frontName, limits, intervals, h }: { study: StudyModel; frontName: string | null; limits: boolean; intervals: boolean; h: Highlight | null }) {
  return (
    <Legend study={study} interval={intervals} limit={limits}>
      {frontName && <LegendItem glyph={<Glyph><path d="M2,5H11V12H20" fill="none" stroke={INK} strokeOpacity={0.65} strokeWidth={1.5} /></Glyph>} text={frontName} />}
      {frontName && <LegendItem glyph={<Glyph><rect x={3} y={4} width={16} height={10} rx={2} fill={INK} fillOpacity={0.07} /></Glyph>} text="dominated" />}
      {h && <LegendItem glyph={<Glyph><circle cx={11} cy={9} r={4.5} fill={MARK} opacity={0.16} /></Glyph>} text={`not ${highlightText(h)}`} />}
    </Legend>
  );
}

function Stat({ k, v, title, n, study }: { k: string; v: string; title?: string; n?: number; study?: StudyModel }) {
  return (
    <span title={title}>
      <span className="k">{k}</span> <b>{v}</b>
      {n !== undefined && study && <a href={href(study.id, { page: "candidate", n })}> #{n}</a>}
    </span>
  );
}

export function TradeOff({ study }: { study: StudyModel }) {
  const [ref, measured] = useWidth<HTMLDivElement>();
  const [pair, setPair] = useState<[number, number]>([0, 1]);
  const [intervals, setIntervals] = useState(false);
  const [h, setH] = useState<Highlight | null>(null);
  const objectives = study.objectives;

  // one domain per objective, shared by every chart it is on (its declared range, its limits)
  const domains = useMemo(
    () =>
      objectives.map((g) => {
        const vs = study.candidates.flatMap((t) => (study.markKind(t) === "unmeasured" ? [] : [study.mean(t, g.metric)])).filter((v): v is number => v !== null);
        const limits = study.constraints.filter((c) => c.metric === g.metric).map((c) => c.constraint.limit);
        return valueDomain(vs, study.range(g.metric), limits);
      }),
    [study, objectives],
  );

  if (objectives.length < 2) return <div className="empty">one objective</div>;

  const width = measured;
  const [xi, yi] = pair;
  const xg = objectives[xi];
  const yg = objectives[yi];
  const many = objectives.length > 2;
  const match = matcher(study, h);
  const best = [...study.best];
  const front = (a: ObjectiveGoal, b: ObjectiveGoal) => (!study.bestIsSet ? null : many ? front2(study, a.objective, b.objective) : best);
  const detailFront = front(xg, yg);
  const frontName = !study.bestIsSet ? null : many ? `front of ${yg.metric} vs ${xg.metric}` : "Pareto front";
  const shown = plotted(study, xg.metric, yg.metric);
  const missing = study.candidates.filter((t) => !shown.includes(t));
  const limitsShown = study.constraints.some((c) => c.metric === xg.metric || c.metric === yg.metric);

  // layout: the pair matrix beside the detail while its cells stay readable, else above it
  const n = objectives.length;
  const TICK_W = 42;
  const TICK_H = 22;
  const NAME_W = 110; // the last row's name, right of its cells
  const besideCell = Math.min(180, Math.floor((width * 0.42 - TICK_W - NAME_W - (n - 1) * 6) / (n - 1)));
  const beside = many && width >= 1000 && besideCell >= 110;
  const cell = beside ? besideCell : Math.max(76, Math.min(180, Math.floor((width - TICK_W - NAME_W - (n - 1) * 6) / (n - 1))));
  const matrixW = many ? TICK_W + (n - 1) * cell + (n - 1) * 6 + NAME_W : 0;
  const detailW = beside ? width - matrixW - 20 : width;
  const detailH = Math.round(Math.max(360, Math.min(600, (detailW - 2) * 0.62)));

  const swap = (k: number, axis: "x" | "y") => setPair(axis === "x" ? [k, yi] : [xi, k]);

  const stats = (
    <div className="panel-stats">
      {many && detailFront && <Stat k={`front of ${yg.metric} vs ${xg.metric}`} v={String(detailFront.length)} />}
      {study.bestIsSet ? <Stat k="Pareto set" v={String(best.length)} /> : best[0] !== undefined && <Stat k="best" v="" n={best[0]} study={study} />}
      {missing.length > 0 && (
        <Stat
          k="not plotted"
          v={String(missing.length)}
          title={`candidates ${missing.map((t) => t.number).join(", ")}`}
        />
      )}
    </div>
  );

  return (
    <div ref={ref} className="tradeoff">
      <div className="controls">
        {many && (
          <>
            <label className="ctl">
              <span className="ctl-k">x</span>
              <select value={xi} onChange={(e) => swap(Number(e.target.value), "x")}>
                {objectives.map((g, k) => (k === yi ? null : <option key={k} value={k}>{goalTitle(g)}</option>))}
              </select>
            </label>
            <label className="ctl">
              <span className="ctl-k">y</span>
              <select value={yi} onChange={(e) => swap(Number(e.target.value), "y")}>
                {objectives.map((g, k) => (k === xi ? null : <option key={k} value={k}>{goalTitle(g)}</option>))}
              </select>
            </label>
          </>
        )}
        <HighlightControl study={study} h={h} set={setH} />
        <label className="ctl check">
          <input type="checkbox" checked={intervals} onChange={(e) => setIntervals(e.target.checked)} />
          95% intervals
        </label>
      </div>
      <TradeOffLegend study={study} frontName={frontName} limits={limitsShown} intervals={intervals} h={h} />
      <div className={beside ? "to-layout beside" : "to-layout"}>
        {many && (
          <section className="pair-matrix" style={{ width: matrixW }}>
            <div className="group-head">
              Pairs <span className="count">{(n * (n - 1)) / 2}</span>
            </div>
            <div
              className="pair-grid"
              style={{ gridTemplateColumns: `${cell + TICK_W}px ${n > 2 ? `repeat(${n - 2}, ${cell}px) ` : ""}${NAME_W}px`, gridTemplateRows: `auto repeat(${n - 1}, auto)` }}
            >
              {objectives.map((gi, i) =>
                objectives.map((gj, j) => {
                  const w = cell + (j === 0 ? TICK_W : 0);
                  const hgt = cell + (i === n - 1 ? TICK_H : 0);
                  // the name of objective i: above its column (x) and right of its row (y)
                  if (i === j)
                    return (
                      <div
                        key={`${i}-${j}`}
                        className={`pair-name${i === 0 ? " top" : i === n - 1 ? " last" : ""}`}
                        style={{ width: i === n - 1 ? undefined : w, height: i === 0 ? undefined : hgt, paddingLeft: j === 0 ? TICK_W + 4 : 6, paddingBottom: i === n - 1 ? TICK_H : 4 }}
                        title={goalTitle(gi)}
                      >
                        {goalTitle(gi)}
                      </div>
                    );
                  if (j > i) return <div key={`${i}-${j}`} />;
                  const on = (xi === j && yi === i) || (xi === i && yi === j);
                  return (
                    <Scatter
                      key={`${i}-${j}`}
                      study={study}
                      xg={gj}
                      yg={gi}
                      width={w}
                      height={hgt}
                      xDomain={domains[j]}
                      yDomain={domains[i]}
                      front={front(gj, gi)}
                      compact
                      xTicks={i === n - 1}
                      yTicks={j === 0}
                      match={match}
                      selected={on}
                      onPick={() => setPair([j, i])}
                    />
                  );
                }),
              )}
            </div>
          </section>
        )}
        <section className="panel to-detail" style={{ width: beside ? detailW : undefined }}>
          <div className="panel-head">
            <h3>
              {goalTitle(yg)} <span className="vs">vs</span> {goalTitle(xg)}
            </h3>
            {stats}
          </div>
          <Scatter
            study={study}
            xg={xg}
            yg={yg}
            width={detailW - 2}
            height={detailH}
            xDomain={domains[xi]}
            yDomain={domains[yi]}
            front={detailFront}
            intervals={intervals}
            match={match}
          />
        </section>
      </div>
      <Tooltip render={(k) => <CandidateTip study={study} n={k} />} />
    </div>
  );
}
