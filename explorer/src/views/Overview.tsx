import { EffectsSection } from "../components/Effects";
import type { Effects } from "../record/effects";
// Did it work, and what is the answer: every goal at the baseline and at the best (objectives, then constraints,
// always both), the best candidates against the baseline, what the selected one changed, and the run's health and cost.
// Everything is read from the record: best is the record's best set, "meets" is a recorded violation <= 0.

import { useState, type ReactNode } from "react";
import { TextDiff } from "../components/Diff";
import { BlockChip, Changed, Choices, EstimateText, CandidateLink, valueText } from "../components/common";
import { CandidateTip } from "../components/CandidateTip";
import { ci, num, words } from "../format";
import { goalTitle } from "../model/goals";
import { SELECTORS, type StudyModel } from "../model/study";
import type { Candidate } from "../record/parse";
import type { TrialStatus } from "../record/schema";
import { Tooltip, useHover } from "../interaction/hover";
import "./Overview.css";

type ObjectiveGoal = StudyModel["objectives"][number];
type ConstraintGoal = StudyModel["constraints"][number];

/** The framework's usage metric for a trial's model tokens (trials.USAGE); summed only when the study has it. */
const TOKENS_METRIC = "tokens";
const ROWS = 10;
const STATUS_ORDER: TrialStatus[] = ["completed", "agent_failure", "infrastructure_invalid", "evaluator_invalid"];

// ---- local helpers ------------------------------------------------------------------------------------------------

/** A change, signed: +0.25, −1,232, ±0. */
function signed(v: number): string {
  if (v === 0) return "±0";
  return v > 0 ? `+${num(v)}` : `−${num(-v)}`;
}

/** How far a candidate is from meeting every constraint: how many it breaks, then by how much in total (the sum of its
 * violations, as the optimizer ranks infeasible candidates; the units are the constraints' metrics). */
function broken(study: StudyModel, t: Candidate): number {
  return study.constraints.filter((g) => (study.violation(t, g.index) ?? 0) > 0).length;
}
function excess(study: StudyModel, t: Candidate): number {
  return study.constraints.reduce((s, g) => {
    const v = study.violation(t, g.index);
    return s + (v !== null && v > 0 ? v : 0);
  }, 0);
}

/** Order candidates by the first objective, best first; unmeasured last; then by number. */
function byFirstObjective(study: StudyModel) {
  const g = study.objectives[0];
  return (a: Candidate, b: Candidate): number => {
    if (!g) return a.number - b.number;
    const va = study.mean(a, g.metric);
    const vb = study.mean(b, g.metric);
    if (va === null || vb === null) return va === vb ? a.number - b.number : va === null ? 1 : -1;
    if (va !== vb) return g.objective.kind === "Maximize" ? vb - va : va - vb;
    return a.number - b.number;
  };
}

/** Text that may wrap only after _ / - or a dot that ends a name (metric names and keys), never inside a word or a
 * number. */
function Breakable({ text }: { text: string }) {
  const parts = text.split(/(?<=[_/-])|(?<=\.)(?=[^\d])/);
  return (
    <>
      {parts.map((p, i) => (
        <span key={i}>
          {p}
          {i < parts.length - 1 && <wbr />}
        </span>
      ))}
    </>
  );
}

function Met({ v }: { v: number | null }) {
  if (v === null) return null;
  return v <= 0 ? <span className="good">✓</span> : <span className="bad">✗</span>;
}

function Section({ title, count, aside, children }: { title: ReactNode; count?: number; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="ov-section">
      <div className="ov-head">
        <span>
          {title}
          {count !== undefined && <span className="count">{count}</span>}
        </span>
        {aside}
      </div>
      {children}
    </section>
  );
}

// ---- 1. the headline: each objective, then each constraint --------------------------------------------------------

function ObjectiveTile({ study, g, best, note }: { study: StudyModel; g: ObjectiveGoal; best: Candidate[]; note: string | null }) {
  const base = study.baseline ? study.est(study.baseline, g.metric) : null;
  const b = base?.mean ?? null;
  let label: ReactNode = "best";
  let value = "–";
  let foot: ReactNode = note ? <span className="muted">{note}</span> : null;
  if (best.length > 1) {
    // a set: the Pareto set, or several candidates that are each best (equal on every objective)
    const vs = best.map((t) => study.mean(t, g.metric)).filter((v): v is number => v !== null);
    const lo = Math.min(...vs);
    const hi = Math.max(...vs);
    label = study.bestIsSet ? (
      `Pareto set · ${best.length.toLocaleString()} candidates`
    ) : best.length <= 3 ? (
      <>
        best ·{" "}
        {best.map((t, i) => (
          <span key={t.number}>
            {i > 0 && ", "}
            <CandidateLink study={study} n={t.number} />
          </span>
        ))}
      </>
    ) : (
      `best · ${best.length.toLocaleString()} candidates`
    );
    value = vs.length ? (lo === hi ? num(lo) : `${num(lo)} – ${num(hi)}`) : "–";
    foot = vs.length && b !== null ? <span className="ov-change">{lo === hi ? signed(lo - b) : `${signed(lo - b)} to ${signed(hi - b)}`}</span> : null;
  } else if (best.length) {
    const t = best[0];
    const e = study.est(t, g.metric);
    label = (
      <>
        {study.bestIsSet ? "Pareto set" : "best"} · <CandidateLink study={study} n={t.number} />
      </>
    );
    value = num(e?.mean ?? null);
    foot =
      e && e.mean !== null ? (
        <span>
          {b !== null && <span className="ov-change">{signed(e.mean - b)}</span>}
          {e.ci && <span className="ci">{ci(e.ci)}</span>}
        </span>
      ) : null;
  }
  return (
    <div className="ov-tile">
      <div className="ov-tile-title">{goalTitle(g)}</div>
      <div className="ov-tile-body">
        <div className="ov-cell">
          <span className="ov-k">baseline</span>
          <span className="ov-v from">{num(b)}</span>
        </div>
        <span className="ov-arrow">→</span>
        <div className="ov-cell">
          <span className="ov-k">{label}</span>
          <span className="ov-v">{value}</span>
        </div>
      </div>
      {foot && <div className="ov-foot">{foot}</div>}
    </div>
  );
}

function ConstraintTile({ study, g }: { study: StudyModel; g: ConstraintGoal }) {
  const comparable = study.candidates.filter((t) => study.comparable(t) && study.violation(t, g.index) !== null);
  const meet = comparable.filter((t) => study.violation(t, g.index)! <= 0).length;
  const base = study.baseline;
  return (
    <div className="ov-tile">
      <div className="ov-tile-title">{goalTitle(g)}</div>
      <div className="ov-tile-body">
        <div className="ov-cell">
          <span className="ov-k">baseline</span>
          <span className="ov-v from">
            {num(base ? study.mean(base, g.metric) : null)} {base && <Met v={study.violation(base, g.index)} />}
          </span>
        </div>
        <div className="ov-cell ov-meet">
          <span className="ov-k">met by</span>
          <span className="ov-v">
            {meet.toLocaleString()} <span className="ov-of">of {comparable.length.toLocaleString()} comparable</span>
          </span>
        </div>
      </div>
    </div>
  );
}

function Headline({ study, best, note }: { study: StudyModel; best: Candidate[]; note: string | null }) {
  return (
    <>
      <Section title="Objectives" count={study.objectives.length}>
        {study.objectives.length ? (
          <div className="ov-tiles">
            {study.objectives.map((g) => (
              <ObjectiveTile key={g.index} study={study} g={g} best={best} note={note} />
            ))}
          </div>
        ) : (
          <div className="ov-empty">No objectives</div>
        )}
      </Section>
      <Section title="Constraints" count={study.constraints.length}>
        {study.constraints.length ? (
          <div className="ov-tiles">
            {study.constraints.map((g) => (
              <ConstraintTile key={g.index} study={study} g={g} />
            ))}
          </div>
        ) : (
          <div className="ov-empty">No constraints</div>
        )}
      </Section>
    </>
  );
}

// ---- 2. the best candidates -----------------------------------------------------------------------------------------

type Ranked = { kind: "best" | "closest" | "none"; candidates: Candidate[]; note: string | null };

function ranked(study: StudyModel): Ranked {
  const order = byFirstObjective(study);
  const best = [...study.best].map((n) => study.byNumber.get(n)).filter((t): t is Candidate => !!t).sort(order);
  if (best.length) return { kind: "best", candidates: best, note: null };
  if (!study.candidates.length) return { kind: "none", candidates: [], note: "No candidates" };
  const comparable = study.candidates.filter((t) => study.comparable(t));
  if (!comparable.length) return { kind: "none", candidates: [], note: "No comparable candidates" };
  if (study.constraints.length && !comparable.some((t) => study.markKind(t) === "ok"))
    return {
      kind: "closest",
      candidates: [...comparable].sort((a, b) => broken(study, a) - broken(study, b) || excess(study, a) - excess(study, b) || order(a, b)),
      note: "No feasible candidates",
    };
  return { kind: "none", candidates: [], note: "No best candidate" };
}

function CandidateTable({ study, rows, selected, onSelect, inBest }: { study: StudyModel; rows: Candidate[]; selected: number | null; onSelect: (n: number) => void; inBest: (n: number) => boolean }) {
  const { set } = useHover();
  const withChoices = study.space.some((s) => SELECTORS.has(s.kind));
  const objectives = study.objectives;
  const constraints = study.constraints;
  return (
    <div className="ov-scroll">
      <table className={`grid-table ov-table${study.goals.length > 4 ? " ov-many" : ""}`}>
        <thead>
          <tr>
            <th rowSpan={2} className="ov-candidate">
              Candidate
            </th>
            <th colSpan={Math.max(1, objectives.length)} className="ov-group">
              Objectives
            </th>
            <th colSpan={Math.max(1, constraints.length)} className="ov-group">
              Constraints
            </th>
            {withChoices && (
              <th rowSpan={2} className="ov-fill">
                Choices
              </th>
            )}
            <th rowSpan={2} className={withChoices ? "" : "ov-fill"}>
              Changed
            </th>
          </tr>
          <tr>
            {objectives.length ? (
              objectives.map((g, i) => (
                <th key={`o${g.index}`} className={`num ov-goal${i === 0 ? " ov-first" : ""}`}>
                  <Breakable text={goalTitle(g)} />
                </th>
              ))
            ) : (
              <th className="ov-goal ov-first muted">none</th>
            )}
            {constraints.length ? (
              constraints.map((g, i) => (
                <th key={`c${g.index}`} className={`num ov-goal${i === 0 ? " ov-first" : ""}`}>
                  <Breakable text={goalTitle(g)} />
                </th>
              ))
            ) : (
              <th className="ov-goal ov-first muted">none</th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => {
            const isBase = t.role === "baseline";
            return (
              <tr
                key={t.number}
                className={`${isBase ? "ov-base" : "ov-pick"}${t.number === selected ? " on" : ""}`}
                onClick={isBase ? undefined : () => onSelect(t.number)}
                onMouseMove={(e) => set(t.number, e)}
                onMouseLeave={() => set(null)}
              >
                <td className="num ov-candidate">
                  <CandidateLink study={study} n={t.number}>
                    {t.number}
                  </CandidateLink>
                  {inBest(t.number) && isBase && <span className="tag">{study.bestIsSet ? "Pareto set" : "best"}</span>}
                  {study.markKind(t) === "unmeasured" && <span className="tag warn">{t.unscored ? "unscored" : "not measured"}</span>}
                  {t.label && t.label.outcome !== "feasible" && <span className="tag warn">{words(t.label.outcome)}</span>}
                </td>
                {objectives.length ? (
                  objectives.map((g, i) => (
                    <td key={`o${g.index}`} className={`num ov-est${i === 0 ? " ov-first" : ""}`}>
                      <EstimateText e={study.est(t, g.metric)} />
                    </td>
                  ))
                ) : (
                  <td className="ov-first" />
                )}
                {constraints.length ? (
                  constraints.map((g, i) => (
                    <td key={`c${g.index}`} className={`num${i === 0 ? " ov-first" : ""}`}>
                      <b>{num(study.mean(t, g.metric))}</b> <Met v={study.violation(t, g.index)} />
                    </td>
                  ))
                ) : (
                  <td className="ov-first" />
                )}
                {withChoices && (
                  <td className="ov-choices ov-fill">
                    <Choices study={study} t={t} />
                  </td>
                )}
                <td className={`ov-changed${withChoices ? "" : " ov-fill"}`}>
                  <Changed study={study} t={t} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BestCandidates({ study, list, selected, onSelect }: { study: StudyModel; list: Ranked; selected: number | null; onSelect: (n: number) => void }) {
  const [all, setAll] = useState(false);
  const base = study.baseline;
  const others = list.candidates.filter((t) => t !== base);
  const shown = all ? others : others.slice(0, ROWS);
  const rows = [...(base ? [base] : []), ...shown];
  const title = list.kind === "closest" ? "Closest candidates" : study.bestIsSet ? "Pareto set" : list.candidates.length > 1 ? "Best candidates" : "Best candidate";
  const toggle =
    others.length > ROWS ? (
      <button className="ov-toggle" onClick={() => setAll(!all)}>
        {all ? "Show fewer" : `Show all ${others.length.toLocaleString()}`}
      </button>
    ) : null;
  return (
    <Section title={title} count={list.kind === "none" ? 0 : list.candidates.length} aside={toggle}>
      <div className="ov-card">
        {list.note && <div className="ov-note">{list.note}</div>}
        {rows.length > 0 && <CandidateTable study={study} rows={rows} selected={selected} onSelect={onSelect} inBest={(n) => list.kind === "best" && study.best.has(n)} />}
      </div>
    </Section>
  );
}

// ---- 3. the selected candidate against the baseline -----------------------------------------------------------------

function same(a: unknown, b: unknown): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

function VsBaseline({ study, n }: { study: StudyModel; n: number }) {
  const t = study.byNumber.get(n);
  const base = study.baseline;
  if (!t || !base) return null;
  const differs = study.space.filter((s) => !same(t.values[s.key], base.values[s.key]));
  const isText = (key: string) => typeof t.values[key] === "string" && typeof base.values[key] === "string";
  const values = differs.filter((s) => !(s.kind === "text" && isText(s.key)));
  const texts = differs.filter((s) => s.kind === "text" && isText(s.key));
  return (
    <Section
      title={
        <>
          <CandidateLink study={study} n={t.number}>
            Candidate {t.number}
          </CandidateLink>{" "}
          vs baseline
        </>
      }
    >
      <div className="ov-card">
        {!differs.length && <div className="ov-note">Same as baseline</div>}
        {values.length > 0 && (
          <table className="grid-table ov-params">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Block</th>
                <th className="num">Baseline</th>
                <th className="num">Candidate {t.number}</th>
              </tr>
            </thead>
            <tbody>
              {values.map((s) => {
                const block = study.blockOf.get(s.key);
                return (
                  <tr key={s.key}>
                    <td className="mono ov-key">{s.key}</td>
                    <td>{block ? <BlockChip study={study} block={block} /> : <span className="muted">–</span>}</td>
                    <td className="num ov-was">{valueText(study, s, base.values[s.key])}</td>
                    <td className="num">
                      <b>{valueText(study, s, t.values[s.key])}</b>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {texts.map((s) => {
          const block = study.blockOf.get(s.key);
          return (
            <div key={s.key} className="ov-text">
              <div className="ov-text-head">
                <span className="mono ov-key">{s.key}</span>
                {block && <BlockChip study={study} block={block} />}
                <span className="muted small tnum">
                  {valueText(study, s, base.values[s.key])} → {valueText(study, s, t.values[s.key])}
                </span>
              </div>
              <TextDiff before={String(base.values[s.key])} after={String(t.values[s.key])} />
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ---- 4. health and cost --------------------------------------------------------------------------------------------

function Stat({ k, v, sub, zero }: { k: ReactNode; v: ReactNode; sub?: boolean; zero?: boolean }) {
  return (
    <div className={`ov-stat${sub ? " sub" : ""}${zero ? " zero" : ""}`}>
      <span>{k}</span>
      <b>{v}</b>
    </div>
  );
}

function statusCounts(statuses: string[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const s of statuses) counts.set(s, (counts.get(s) ?? 0) + 1);
  const rank = (s: string) => {
    const i = STATUS_ORDER.indexOf(s as TrialStatus);
    return i < 0 ? STATUS_ORDER.length : i;
  };
  return [...counts.entries()].sort((a, b) => rank(a[0]) - rank(b[0]));
}

function Health({ study }: { study: StudyModel }) {
  const candidates = study.candidates;
  const outcomes = (["feasible", "infeasible", "ineligible"] as const).map(
    (s) => [s, candidates.filter((t) => t.label?.outcome === s).length] as const,
  );
  const unmeasured = candidates.filter((t) => study.markKind(t) === "unmeasured").length;
  const comparable = candidates.filter((t) => study.comparable(t));
  const trials = candidates.flatMap((t) => t.trials);
  const measured = trials.filter((a) => a.status === "completed");
  const lost = trials.filter((a) => a.status !== "completed");
  const searchTokens = candidates.reduce((s, t) => s + study.searchTokens(t), 0);
  const hasTokens = TOKENS_METRIC in study.h.metrics;
  const trialTokens = trials.reduce((s, a) => s + (study.trialMetrics(a)[TOKENS_METRIC] ?? 0), 0);
  const stop = study.record.stop;
  const budget = study.config.budget;
  const rollouts = study.record.iterations.reduce((s, it) => s + it.record.rollouts_spent, 0) + (study.baseline?.trials.reduce((s, a) => s + a.rollouts, 0) ?? 0);
  return (
    <Section title="Health and cost">
      <div className="ov-health">
        <div className="ov-tile">
          <div className="ov-tile-title">Candidates</div>
          <div className="ov-big">{candidates.length.toLocaleString()}</div>
          {outcomes.map(([s, c]) => (
            <Stat key={s} k={s} v={c.toLocaleString()} zero={c === 0} />
          ))}
          <Stat k="not measured" v={unmeasured.toLocaleString()} zero={unmeasured === 0} />
          <div className="ov-rule" />
          <Stat k="comparable" v={comparable.length.toLocaleString()} zero={comparable.length === 0} />
        </div>
        <div className="ov-tile">
          <div className="ov-tile-title">Trials</div>
          <div className="ov-big">{trials.length.toLocaleString()}</div>
          <Stat k="completed" v={measured.length.toLocaleString()} zero={measured.length === 0} />
          <Stat k="not measured" v={lost.length.toLocaleString()} zero={lost.length === 0} />
          {statusCounts(lost.map((a) => a.status)).map(([s, c]) => (
            <Stat key={s} k={words(s)} v={c.toLocaleString()} sub />
          ))}
        </div>
        <div className="ov-tile">
          <div className="ov-tile-title">Model tokens</div>
          <Stat k="proposers" v={num(searchTokens)} zero={searchTokens === 0} />
          {hasTokens && <Stat k="trials" v={num(trialTokens)} zero={trialTokens === 0} />}
        </div>
        <div className="ov-tile">
          <div className="ov-tile-title">Run</div>
          <Stat k="state" v={study.state} />
          <Stat k="rollouts" v={`${num(rollouts)} of ${num(budget.max_rollouts)}`} />
          {budget.max_iterations !== null && <Stat k="iterations" v={`${study.record.iterations.length.toLocaleString()} of ${num(budget.max_iterations)}`} />}
          {study.record.resumes.length > 0 && <Stat k="resumes" v={study.record.resumes.length.toLocaleString()} />}
          {stop && <Stat k="stopped on" v={stop.reason} />}
          {stop?.message && <pre className="ov-error mono small">{stop.message}</pre>}
        </div>
      </div>
    </Section>
  );
}

// ---- the page ------------------------------------------------------------------------------------------------------

export function Overview({ study, effects }: { study: StudyModel; effects: Effects | null }) {
  const list = ranked(study);
  const [picked, setPicked] = useState<number | null>(null);
  const candidates = list.candidates.filter((t) => t.role !== "baseline");
  const selected = picked !== null && candidates.some((t) => t.number === picked) ? picked : candidates[0]?.number ?? null;
  const best = list.kind === "best" ? list.candidates : [];
  return (
    <div className="overview">
      <Headline study={study} best={best} note={list.note} />
      <BestCandidates study={study} list={list} selected={selected} onSelect={setPicked} />
      <Section title="Effects">
        <EffectsSection study={study} effects={effects} />
      </Section>
      {selected !== null && <VsBaseline study={study} n={selected} />}
      <Health study={study} />
      <Tooltip render={(n) => <CandidateTip study={study} n={n} />} />
    </div>
  );
}
