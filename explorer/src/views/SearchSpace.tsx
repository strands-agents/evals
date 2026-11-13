import { EffectsPlot, FitTag } from "../components/Effects";
import type { Effects } from "../record/effects";
import { Glyph, Legend, LegendItem } from "../components/Legend";
// Which values work, and what was tried: the space as a tree (selector › option › the variables under it, at any
// depth), grouped under each block in block order. Each numeric or categorical variable is a slice (its value against a
// goal) or a trace (its value over the candidates, aligned with the schedule track); each text variable is its version
// lineage, each version's diff against the version it was rewritten from. Variables no candidate had are one line.

import { useState, type ReactNode } from "react";
import { leftMargin, textWidth, candidateAxis, type CandidateAxis } from "../charts/axis";
import { MarkGlyph, useWidth } from "../charts/kit";
import { goalDomain } from "../charts/plot";
import { scheduleLabels, ScheduleTrack } from "../charts/ScheduleTrack";
import { isNumeric, lanesOf, SlicePlot, traceLabelWidth, TracePlot } from "../charts/Slice";
import { TextDiff } from "../components/Diff";
import { BlockChip, CandidateLink } from "../components/common";
import { CandidateTip } from "../components/CandidateTip";
import { INK_2 } from "../encoding/theme";
import { num } from "../format";
import { better, goalTitle } from "../model/goals";
import { SELECTORS, type Choice, type Goal, type StudyModel } from "../model/study";
import type { SpaceRow } from "../record/schema";
import { Tooltip } from "../interaction/hover";
import "./SearchSpace.css";

type Mode = "slice" | "trace";

// ---- the tree -----------------------------------------------------------------------------------------------------

type Node =
  | { kind: "var"; row: SpaceRow; options: { option: string; children: Node[] }[] }
  | { kind: "when"; cond: Choice; children: Node[] };

/** A block's variables as a tree: a selector's options hold the variables that exist under them; a condition on a
 * selector of another block heads its own group. */
function tree(study: StudyModel, rows: SpaceRow[], prefix: Choice[] = []): Node[] {
  const depth = prefix.length;
  const conds = new Map(rows.map((r) => [r.key, study.conditions(r.key)]));
  const direct = rows.filter((r) => conds.get(r.key)!.length === depth);
  const groups = new Map<string, { cond: Choice; rows: SpaceRow[] }>();
  for (const r of rows) {
    const c = conds.get(r.key)!;
    if (c.length === depth) continue;
    const next = c[depth];
    const id = `${next.key}\u0000${next.option}`;
    if (!groups.has(id)) groups.set(id, { cond: next, rows: [] });
    groups.get(id)!.rows.push(r);
  }
  const nodes: Node[] = direct.map((row) => {
    const options: { option: string; children: Node[] }[] = [];
    if (SELECTORS.has(row.kind))
      for (const o of row.options ?? []) {
        const id = `${row.key}\u0000${String(o)}`;
        const g = groups.get(id);
        if (!g) continue;
        groups.delete(id);
        options.push({ option: String(o), children: tree(study, g.rows, [...prefix, g.cond]) });
      }
    return { kind: "var" as const, row, options };
  });
  for (const g of groups.values()) nodes.push({ kind: "when", cond: g.cond, children: tree(study, g.rows, [...prefix, g.cond]) });
  return nodes;
}

// ---- facts about a variable ---------------------------------------------------------------------------------------

function domainText(row: SpaceRow): string {
  if (isNumeric(row)) {
    const rowStep = typeof row.step === "number" ? row.step : null;
    const step = rowStep !== null && !(row.kind === "int" && rowStep === 1) ? ` step ${num(rowStep)}` : "";
    return `${row.kind} ${num(row.min_value)} – ${num(row.max_value)}${row.log ? " log" : ""}${step}`;
  }
  if (row.kind === "text") return row.size_cap ? `text ≤ ${num(row.size_cap)} chars` : "text";
  if (row.kind === "skills") return "skill library";
  const n = row.options?.length ?? 0;
  return `${row.kind} · ${n} option${n === 1 ? "" : "s"}`;
}

function defaultText(row: SpaceRow): string {
  if (row.kind === "text" || row.kind === "skills") return "v0";
  return typeof row.default === "number" ? num(row.default) : String(row.default);
}

function counts(study: StudyModel, key: string) {
  let active = 0, changed = 0, inherited = 0, defaulted = 0;
  for (const t of study.candidates) {
    if (key in t.values) active++;
    if (t.changedKeys.includes(key)) changed++;
    if (key in t.inherited) inherited++;
    if (t.defaulted.includes(key)) defaulted++;
  }
  return { active, changed, inherited, defaulted };
}

function Facts({ study, row }: { study: StudyModel; row: SpaceRow }) {
  const c = counts(study, row.key);
  return (
    <span className="var-facts">
      <span>{domainText(row)}</span>
      <span>
        <span className="k">default</span> {defaultText(row)}
      </span>
      <span>
        <span className="k">active</span> <b>{c.active}</b> of {study.candidates.length}
      </span>
      {c.changed > 0 && (
        <span>
          <span className="k">changed</span> <b>{c.changed}</b>
        </span>
      )}
      {c.inherited > 0 && (
        <span>
          <span className="k">inherited</span> <b>{c.inherited}</b>
        </span>
      )}
      {c.defaulted > 0 && (
        <span>
          <span className="k">defaulted</span> <b>{c.defaulted}</b>
        </span>
      )}
    </span>
  );
}

// ---- text versions ------------------------------------------------------------------------------------------------

/** Lines added and removed from a to b (a longest-common-subsequence line diff). */
function lineChanges(a: string, b: string): { added: number; removed: number } {
  const la = a.split("\n");
  const lb = b.split("\n");
  const n = la.length;
  const m = lb.length;
  if (n * m > 4_000_000) {
    const left = new Map<string, number>();
    for (const l of la) left.set(l, (left.get(l) ?? 0) + 1);
    let same = 0;
    for (const l of lb) {
      const k = left.get(l) ?? 0;
      if (k > 0) {
        same++;
        left.set(l, k - 1);
      }
    }
    return { added: m - same, removed: n - same };
  }
  let prev = new Int32Array(m + 1);
  let cur = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) cur[j] = la[i - 1] === lb[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    [prev, cur] = [cur, prev];
  }
  const same = prev[m];
  return { added: m - same, removed: n - same };
}

type Version = {
  i: number;
  text: string;
  first: number | null;
  candidates: number[];
  parent: number | null; // the version it was rewritten from (recorded: the first candidate's origin.from_candidate)
  changes: { added: number; removed: number } | null;
  best: { v: number; n: number } | null;
  inBest: boolean;
};

function bestOf(study: StudyModel, goal: Goal, candidates: number[]): { v: number; n: number } | null {
  let out: { v: number; n: number } | null = null;
  const isBetter = (a: number, b: number) =>
    goal.role === "objective" ? better(goal.objective, a, b) : goal.constraint.kind === "AtMost" ? a < b : a > b;
  for (const n of candidates) {
    const t = study.byNumber.get(n);
    if (!t || study.markKind(t) === "unmeasured") continue;
    const v = study.mean(t, goal.metric);
    if (v === null) continue;
    if (!out || isBetter(v, out.v)) out = { v, n };
  }
  return out;
}

function lineage(study: StudyModel, key: string, goal: Goal): Version[] {
  const versions = study.textVersions(key);
  const index = new Map(versions.map((v, i) => [v.text, i]));
  return versions.map((v, i) => {
    let parent: number | null = null;
    if (v.first !== null) {
      // the version the first candidate rewrote from: its parent candidate's text
      const parentId = study.byNumber.get(v.first)?.parentId;
      const held = parentId ? study.byId.get(parentId)?.values[key] : undefined;
      const text = held === null ? "" : held;
      if (typeof text === "string") parent = index.get(text) ?? null;
    }
    if (parent === i) parent = null;
    return {
      i,
      text: v.text,
      first: v.first,
      candidates: v.candidates,
      parent,
      changes: parent !== null ? lineChanges(versions[parent].text, v.text) : null,
      best: bestOf(study, goal, v.candidates),
      inBest: v.candidates.some((n) => study.best.has(n)),
    };
  });
}

function CandidateList({ study, candidates }: { study: StudyModel; candidates: number[] }) {
  const SHOW = 8;
  if (!candidates.length) return <span className="muted">none</span>;
  return (
    <span className="candidate-list">
      {candidates.slice(0, SHOW).map((n, k) => (
        <span key={n}>
          {k > 0 && ", "}
          <CandidateLink study={study} n={n}>
            {n}
          </CandidateLink>
        </span>
      ))}
      {candidates.length > SHOW && <span className="muted"> +{candidates.length - SHOW}</span>}
    </span>
  );
}

function TextCard({ study, row, goal, mode, axis, width }: { study: StudyModel; row: SpaceRow; goal: Goal; mode: Mode; axis: CandidateAxis; width: number }) {
  const [sel, setSel] = useState<number | null>(null);
  const versions = lineage(study, row.key, goal);
  const v = sel !== null ? versions[sel] : null;
  const skills = row.kind === "skills";
  return (
    <section className="var-card wide">
      <VarHead study={study} row={row} />
      {mode === "trace" && (
        <div className="trace-wrap">
          <TracePlot study={study} row={row} axis={axis} width={width} />
        </div>
      )}
      <div className="lineage">
        <table className="grid-table">
          <thead>
            <tr>
              <th>version</th>
              <th className="num">first</th>
              <th>from</th>
              <th className="num">{skills ? "skills" : "lines"}</th>
              <th>candidates</th>
              <th className="num">best {goal.metric}</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((x) => (
              <tr
                key={x.i}
                className={sel === x.i ? "on" : ""}
                tabIndex={0}
                aria-expanded={sel === x.i}
                onClick={() => setSel(sel === x.i ? null : x.i)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSel(sel === x.i ? null : x.i);
                  }
                }}
              >
                <td className="ver">
                  <span className="caret">{sel === x.i ? "▾" : "▸"}</span>
                  <b>v{x.i}</b>
                  {x.inBest && (
                    <span className="in-best" title={study.bestIsSet ? "Pareto set" : "best"}>
                      <MarkGlyph kind="ok" best />
                    </span>
                  )}
                </td>
                <td className="num">{x.first === null ? <span className="muted">default</span> : <CandidateLink study={study} n={x.first}>{x.first}</CandidateLink>}</td>
                <td>{x.parent !== null ? `v${x.parent}` : <span className="muted">–</span>}</td>
                <td className="num">
                  {x.changes ? (
                    <>
                      <span className="n-add">+{x.changes.added}</span> <span className="n-del">−{x.changes.removed}</span>
                    </>
                  ) : (
                    <span className="muted">–</span>
                  )}
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  <CandidateList study={study} candidates={x.candidates} />
                </td>
                <td className="num">
                  {x.best ? (
                    <>
                      <b>{num(x.best.v)}</b>{" "}
                      <span onClick={(e) => e.stopPropagation()}>
                        <CandidateLink study={study} n={x.best.n}>
                          #{x.best.n}
                        </CandidateLink>
                      </span>
                    </>
                  ) : (
                    <span className="muted">–</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {v && (
          <div className="version-diff">
            <div className="diff-head">
              <b>v{v.i}</b>
              {v.parent !== null ? (
                <span className="muted"> vs v{v.parent}</span>
              ) : (
                <span className="muted">{v.first === null ? " · default" : " · no parent"}</span>
              )}
              {!skills && <span className="muted"> · {v.text.length.toLocaleString()} chars</span>}
            </div>
            {skills ? (
              <div className="skill-decisions">
                <div className="mono small">{v.text || "empty"}</div>
              </div>
            ) : v.parent !== null ? (
              <TextDiff before={versions[v.parent].text} after={v.text} />
            ) : (
              <pre className="textbox">{v.text}</pre>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

// ---- variables ----------------------------------------------------------------------------------------------------

function VarHead({ study, row }: { study: StudyModel; row: SpaceRow }) {
  return (
    <div className="var-head">
      <h3 className="mono">{row.key}</h3>
      <Facts study={study} row={row} />
    </div>
  );
}

function SliceCard({ study, row, goal, yDomain, page, effects }: { study: StudyModel; row: SpaceRow; goal: Goal; yDomain: [number, number]; page: number; effects: Effects | null }) {
  const fitted = effects?.goals.find((g) => g.role === goal.role && g.index === goal.index);
  const has = fitted && effects?.effects.some((e) => e.metric === fitted.metric && e.key === row.key);
  const [ref, w] = useWidth<HTMLDivElement>();
  // options whose names do not fit one column get two
  let wide = false;
  if (!isNumeric(row) && page >= 800) {
    const lanes = lanesOf(study, row).lanes;
    const need = lanes.reduce((m, l) => Math.max(m, textWidth(l.label) + 14), 0) * lanes.length + 60;
    wide = need > 380;
  }
  return (
    <section className="var-card" style={wide ? { gridColumn: "span 2" } : undefined}>
      <VarHead study={study} row={row} />
      <div ref={ref} className="plot-wrap">
        <SlicePlot study={study} row={row} goal={goal} width={w} height={200} yDomain={yDomain} />
      </div>
      {effects && fitted && has && (
        <div className="var-effects">
          <div className="var-effects-head">
            effect · <FitTag effects={effects} metric={fitted.metric} />
          </div>
          <EffectsPlot study={study} effects={effects} goal={fitted} keys={[row.key]} />
        </div>
      )}
    </section>
  );
}

function TraceCard({ study, row, axis, width }: { study: StudyModel; row: SpaceRow; axis: CandidateAxis; width: number }) {
  return (
    <section className="var-card wide">
      <VarHead study={study} row={row} />
      <div className="trace-wrap">
        <TracePlot study={study} row={row} axis={axis} width={width} />
      </div>
    </section>
  );
}

function NeverActive({ row }: { row: SpaceRow }) {
  return (
    <div className="never">
      <span className="mono">{row.key}</span>
      <span className="muted">{domainText(row)}</span>
      <span className="muted">default {defaultText(row)}</span>
      <span className="tag">never active</span>
    </div>
  );
}

type Ctx = { study: StudyModel; goal: Goal; mode: Mode; yDomain: [number, number]; axis: CandidateAxis; width: number; effects: Effects | null };

function CondHead({ study, cond, of, indent }: { study: StudyModel; cond: Choice; of?: boolean; indent: number }) {
  const chosen = study.candidates.filter((t) => cond.key in t.values && String(t.values[cond.key]) === cond.option).length;
  return (
    <div className="cond-head" style={indent ? { marginLeft: indent } : undefined}>
      <span className="choices">
        {of ? <span className="sep">› </span> : <span className="sep">when </span>}
        <span className="mono">{cond.key}</span> = <b>{cond.option}</b>
      </span>
      <span className="muted small nowrap">
        chosen in {chosen} of {study.candidates.length}
      </span>
    </div>
  );
}

function Level({ ctx, nodes, depth = 0 }: { ctx: Ctx; nodes: Node[]; depth?: number }) {
  const { study } = ctx;
  const vars = nodes.filter((n): n is Extract<Node, { kind: "var" }> => n.kind === "var");
  const active = (r: SpaceRow) => study.candidates.some((t) => r.key in t.values);
  const plots = vars.filter((n) => active(n.row));
  const never = vars.filter((n) => !active(n.row));
  const texts = plots.filter((n) => n.row.kind === "text" || n.row.kind === "skills");
  const others = plots.filter((n) => n.row.kind !== "text" && n.row.kind !== "skills");
  // a trace keeps the candidate axis aligned with the schedule track, so its branches indent only their headings
  const indent = ctx.mode === "trace" ? depth * 18 : 0;
  const branches: { key: string; simple: boolean; el: ReactNode }[] = [];
  const branch = (key: string, cond: Choice, of: boolean, children: Node[]) => {
    const leaves = children.every((c) => c.kind === "var" && !c.options.length);
    const shown = children.filter((c) => c.kind === "var" && active(c.row));
    // a branch holding one plotted variable sits in a grid with its siblings (a slice only: a trace keeps full width)
    const simple = ctx.mode === "slice" && leaves && children.length === 1 && shown.length === 1 && shown.every((c) => c.kind === "var" && c.row.kind !== "text" && c.row.kind !== "skills");
    branches.push({
      key,
      simple,
      el: (
        <div key={key} className={simple ? "branch simple" : "branch"}>
          <CondHead study={study} cond={cond} of={of} indent={indent} />
          <Level ctx={ctx} nodes={children} depth={depth + 1} />
        </div>
      ),
    });
  };
  for (const v of vars) for (const o of v.options) branch(`${v.row.key}=${o.option}`, { key: v.row.key, option: o.option }, true, o.children);
  for (const n of nodes) if (n.kind === "when") branch(`when ${n.cond.key}=${n.cond.option}`, n.cond, false, n.children);
  // consecutive simple branches share a grid
  const sub: ReactNode[] = [];
  for (let i = 0; i < branches.length; ) {
    if (!branches[i].simple) {
      sub.push(branches[i].el);
      i++;
      continue;
    }
    let j = i;
    while (j < branches.length && branches[j].simple) j++;
    sub.push(
      <div key={`grid ${branches[i].key}`} className="branch-grid">
        {branches.slice(i, j).map((b) => b.el)}
      </div>,
    );
    i = j;
  }
  return (
    <div className="level">
      {others.length > 0 &&
        (ctx.mode === "slice" ? (
          <div className="slice-grid">
            {others.map((n) => (
              <SliceCard key={n.row.key} study={study} row={n.row} goal={ctx.goal} yDomain={ctx.yDomain} page={ctx.width} effects={ctx.effects} />
            ))}
          </div>
        ) : (
          others.map((n) => <TraceCard key={n.row.key} study={study} row={n.row} axis={ctx.axis} width={ctx.width} />)
        ))}
      {texts.map((n) => (
        <TextCard key={n.row.key} study={study} row={n.row} goal={ctx.goal} mode={ctx.mode} axis={ctx.axis} width={ctx.width} />
      ))}
      {never.map((n) => (
        <NeverActive key={n.row.key} row={n.row} />
      ))}
      {sub}
    </div>
  );
}

function BlockHead({ study, block }: { study: StudyModel; block: string | null }) {
  if (block === null)
    return (
      <div className="block-head">
        <span className="muted">no block</span>
      </div>
    );
  const b = study.h.blocks[block];
  const settings = Object.entries(b.config ?? {}).filter(([, v]) => v !== null && v !== undefined);
  const changed = study.candidates.filter((t) => study.changed(t).includes(block)).length;
  return (
    <div className="block-head">
      <BlockChip study={study} block={block} />
      <span className="method">{b.proposer}</span>
      {settings.map(([k, v]) => (
        <span key={k} className="muted small">
          {k} {Array.isArray(v) ? v.map(String).join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v)}
        </span>
      ))}
      <span className="muted small">changed in {changed} of {study.candidates.length}</span>
    </div>
  );
}

function SpaceLegend({ study, goal, mode }: { study: StudyModel; goal: Goal; mode: Mode }) {
  const kinds = new Set(study.candidates.map((t) => study.markKind(t)));
  return (
    <Legend study={study} limit={mode === "slice" && goal.role === "constraint"}>
      {mode === "trace" && kinds.has("unmeasured") && <LegendItem glyph={<Glyph><path d="M8,6L14,12M8,12L14,6" stroke={INK_2} strokeWidth={1.5} /></Glyph>} text="not measured" />}
    </Legend>
  );
}

export function SearchSpace({ study, effects }: { study: StudyModel; effects: Effects | null }) {
  const [ref, measured] = useWidth<HTMLDivElement>();
  const [goalId, setGoalId] = useState(study.objectives.length ? "o0" : "c0");
  const [mode, setMode] = useState<Mode>("slice");
  const goals = study.goals;
  const goal = goals.find((g) => `${g.role[0]}${g.index}` === goalId) ?? goals[0];
  const width = measured - 2; // inside the cards' borders
  const left = Math.max(leftMargin(scheduleLabels(study)), traceLabelWidth(study, study.space));
  const axis = candidateAxis(study, width, left);

  const sections = [
    ...study.blockOrder.map((b) => ({ block: b as string | null, rows: study.space.filter((s) => study.blockOf.get(s.key) === b) })),
    { block: null, rows: study.space.filter((s) => !study.blockOf.has(s.key)) },
  ].filter((s) => s.rows.length);

  if (!goal) return <div className="empty">no goals</div>;
  const ctx: Ctx = { study, goal, mode, yDomain: goalDomain(study, goal), axis, width, effects };

  return (
    <div ref={ref} className={`space ${mode}`}>
      <div className="controls">
        <div className="seg" role="tablist">
          <button className={mode === "slice" ? "on" : ""} onClick={() => setMode("slice")}>
            slice
          </button>
          <button className={mode === "trace" ? "on" : ""} onClick={() => setMode("trace")}>
            trace
          </button>
        </div>
        {mode === "slice" && (
          <label className="ctl">
            <span className="ctl-k">against</span>
            <select value={goalId} onChange={(e) => setGoalId(e.target.value)}>
              {study.objectives.length > 0 && (
                <optgroup label="objectives">
                  {study.objectives.map((g) => (
                    <option key={`o${g.index}`} value={`o${g.index}`}>
                      {goalTitle(g)}
                    </option>
                  ))}
                </optgroup>
              )}
              {study.constraints.length > 0 && (
                <optgroup label="constraints">
                  {study.constraints.map((g) => (
                    <option key={`c${g.index}`} value={`c${g.index}`}>
                      {goalTitle(g)}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>
        )}
        {mode === "trace" && study.space.some((s) => s.kind === "text") && (
          <label className="ctl">
            <span className="ctl-k">best by</span>
            <select value={goalId} onChange={(e) => setGoalId(e.target.value)}>
              {goals.map((g) => (
                <option key={`${g.role}${g.index}`} value={`${g.role[0]}${g.index}`}>
                  {goalTitle(g)}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <SpaceLegend study={study} goal={goal} mode={mode} />
      {mode === "trace" && (
        <div className="track-card">
          <ScheduleTrack study={study} axis={axis} width={width} />
        </div>
      )}
      {sections.map((s) => (
        <section key={s.block ?? ""} className="block-section">
          <BlockHead study={study} block={s.block} />
          <Level ctx={ctx} nodes={tree(study, s.rows)} />
        </section>
      ))}
      <Tooltip render={(n) => <CandidateTip study={study} n={n} />} />
    </div>
  );
}
