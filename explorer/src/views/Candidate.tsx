// One candidate: what it was, how it was proposed, and how each case went. Everything is the record's: the schedule
// path, the proposing block and its rationale, every variable's provenance, the change from its parent candidate,
// each goal with its interval, and every trial by case and run index.

import { Fragment, type ReactNode } from "react";
import { BlockChip, Changed, Choices, EstimateText, SchedulePath, CandidateLink, valueText } from "../components/common";
import { TextDiff } from "../components/Diff";
import { MUTED, MARK } from "../encoding/theme";
import { ci, num, words } from "../format";
import { goalTitle } from "../model/goals";
import type { Goal, Scope, StudyModel } from "../model/study";
import type { TrialRecord } from "../record/schema";
import type { Candidate as CandidateT } from "../record/parse";
import { href } from "../router";
import { breakable, provenance, roleText } from "./Candidates";
import "./Candidate.css";

// ---- small shared pieces (also used by the trial page) --------------------------------------------------------------

export function trialHref(study: StudyModel, t: CandidateT, a: TrialRecord): string {
  return href(study.id, { page: "trial", n: t.number, case: a.case_name, rep: a.run_index });
}

/** The first line of an error, for grouping and previews. */
export function firstLine(s: string | null): string {
  if (!s) return "";
  const line = s.split("\n").find((l) => l.trim()) ?? "";
  return line.length > 220 ? `${line.slice(0, 220)}…` : line;
}

/** A trial's value of a metric, from its evaluations and usage. `failure` says the trial's agent raised, so an
 * evaluator metric carries the configured agent_failure_score, not a measurement. */
export function trialValue(study: StudyModel, a: TrialRecord, metric: string): { v: number | null; failure: boolean } {
  const v = study.trialMetrics(a)[metric];
  const failure = a.status === "agent_failure" && study.h.metrics[metric]?.kind === "evaluator";
  return { v: v === undefined ? null : v, failure };
}

/** The score an evaluator takes on an agent_failure trial, as configured: "worst", a number, or nothing. */
export function agentFailureText(study: StudyModel): string | null {
  const v = study.config.agent_failure_score;
  if (v === null || v === undefined) return null;
  return typeof v === "number" ? num(v) : String(v);
}

export function Count({ n }: { n: number }) {
  return <span className="count">{n}</span>;
}

/** A recorded value of any shape, verbatim: strings and numbers as written, objects as key = value. */
export function Verbatim({ v }: { v: unknown }) {
  if (v === null) return <span className="muted">null</span>;
  if (v === undefined) return <span className="muted">–</span>;
  if (typeof v === "string") return <span className="text">{v}</span>;
  if (typeof v === "number" || typeof v === "boolean") return <span className="tnum">{String(v)}</span>;
  if (Array.isArray(v)) return <span className="mono small">{JSON.stringify(v)}</span>;
  const entries = Object.entries(v as Record<string, unknown>);
  if (!entries.length) return <span className="muted">{"{}"}</span>;
  return (
    <span className="kv-inline">
      {entries.map(([k, x]) => (
        <span key={k}>
          <span className="mono">{k}</span> = <Verbatim v={x} />
        </span>
      ))}
    </span>
  );
}

function Check({ v }: { v: number | null }) {
  if (v === null) return <span className="muted" title="not measured">–</span>;
  return v <= 0 ? <span className="good" title={`violation ${num(v)}`}>✓</span> : <span className="bad" title={`violation ${num(v)}`}>✗</span>;
}

// ---- header ---------------------------------------------------------------------------------------------------------

function Head({ study, t }: { study: StudyModel; t: CandidateT }) {
  const prev = study.byNumber.get(t.number - 1);
  const next = study.byNumber.get(t.number + 1);
  const bestWord = study.bestIsSet ? "Pareto set" : "best";
  const spans = study.bestSpans.get(t.number) ?? [];
  const past = spans.filter((s) => s.to !== null);
  return (
    <div className="candidate-head">
      <div className="candidate-nav">
        {prev ? <a href={href(study.id, { page: "candidate", n: prev.number })}>‹ candidate {prev.number}</a> : <span />}
        {next ? <a href={href(study.id, { page: "candidate", n: next.number })}>candidate {next.number} ›</a> : <span />}
      </div>
      <div className="candidate-title">
        <h2>Candidate {t.number}</h2>
        {study.best.has(t.number) && <span className="tag">{bestWord}</span>}
        {t.label ? (
          <span className={`tag${t.label.outcome === "feasible" ? "" : " warn"}`}>{words(t.label.outcome)}</span>
        ) : (
          <span className="tag warn">{t.unscored ? "unscored" : "not measured"}</span>
        )}
        {t.rescored && <span className="tag">re-scored</span>}
        {past.length > 0 && (
          <span className="muted small">
            {bestWord} after candidate{past.length > 1 || past[0].to! - 1 > past[0].from ? "s" : ""} {past.map((s) => (s.to! - 1 > s.from ? `${s.from}–${s.to! - 1}` : `${s.from}`)).join(", ")}
          </span>
        )}
      </div>
      <div className="candidate-sub">
        <span>
          <SchedulePath study={study} t={t} />
          {t.role !== "baseline" && (
            <>
              {study.path(t).some((s) => s.depth > 0) ? " · " : ""}
              {roleText(t)}
            </>
          )}
        </span>
        {t.role !== "baseline" && <Changed study={study} t={t} />}
        <Choices study={study} t={t} />
      </div>
      {t.rationale && (
        <div className="candidate-reason">
          <span className="k">rationale</span> <span className="text">{t.rationale}</span>
        </div>
      )}
      {t.label && t.label.outcome !== "feasible" && t.label.reason && (
        <div className="candidate-reason">
          <span className="k">reason</span> <span className="text">{t.label.reason}</span>
        </div>
      )}
    </div>
  );
}

// ---- goals ----------------------------------------------------------------------------------------------------------

function GoalRows({ study, t, goals, base }: { study: StudyModel; t: CandidateT; goals: Goal[]; base: CandidateT | undefined }) {
  return (
    <>
      {goals.map((g) => {
        const e = study.est(t, g.metric);
        return (
          <tr key={`${g.role}${g.index}`}>
            <td className="goal-name">{breakable(goalTitle(g))}</td>
            <td className="num">
              <EstimateText e={e} />
            </td>
            <td className="num muted small">{e ? `${e.n} case${e.n === 1 ? "" : "s"}` : ""}</td>
            <td className="check">{g.role === "constraint" ? <Check v={study.violation(t, g.index)} /> : null}</td>
            {base && <td className="num muted">{num(study.mean(base, g.metric))}</td>}
          </tr>
        );
      })}
    </>
  );
}

function Goals({ study, t }: { study: StudyModel; t: CandidateT }) {
  const base = t.parentId === null ? undefined : study.byId.get(t.parentId);
  const head = (
    <thead>
      <tr>
        <th />
        <th className="num">candidate {t.number}</th>
        <th className="num">n</th>
        <th />
        {base && (
          <th className="num">
            <CandidateLink study={study} n={base.number}>candidate {base.number}</CandidateLink> <span className="muted">(parent)</span>
          </th>
        )}
      </tr>
    </thead>
  );
  return (
    <section className="card candidate-goals">
      <h2>Goals</h2>
      <div className="group-head">
        Objectives <Count n={study.objectives.length} />
      </div>
      {study.objectives.length ? (
        <table className="grid-table goal-table">
          {head}
          <tbody>
            <GoalRows study={study} t={t} goals={study.objectives} base={base} />
          </tbody>
        </table>
      ) : (
        <div className="none-box">No objectives</div>
      )}
      <div className="group-head">
        Constraints <Count n={study.constraints.length} />
      </div>
      {study.constraints.length ? (
        <table className="grid-table goal-table">
          {head}
          <tbody>
            <GoalRows study={study} t={t} goals={study.constraints} base={base} />
          </tbody>
        </table>
      ) : (
        <div className="none-box">No constraints</div>
      )}
      <div className="facts">
        <span>
          <span className="k">outcome</span> {t.label ? words(t.label.outcome) : t.unscored ? "unscored (budget)" : "not measured"}
        </span>
        {t.label?.is_from_fit && (
          <span>
            <span className="k">estimates</span> from the fitted score model
          </span>
        )}
      </div>
    </section>
  );
}

// ---- the schedule ---------------------------------------------------------------------------------------------------

function candidateRange(study: StudyModel, s: Scope): ReactNode {
  if (!s.candidates.length) return <span className="muted">none</span>;
  const first = s.candidates[0];
  const last = s.candidates[s.candidates.length - 1];
  return (
    <>
      <CandidateLink study={study} n={first}>{first}</CandidateLink>
      {last !== first && (
        <>
          –<CandidateLink study={study} n={last}>{last}</CandidateLink>
        </>
      )}
      <span className="muted"> · {s.candidates.length}</span>
    </>
  );
}

function Schedule({ study, t }: { study: StudyModel; t: CandidateT }) {
  const path = study.path(t);
  const here = path.at(-1);
  const role = roleText(t);
  const best = study.bestAfter(t.number);
  return (
    <section className="card candidate-schedule">
      <h2>Schedule</h2>
      {path.length > 0 ? (
        <div className="path-list">
          {path.map((s) => (
            <div key={s.id} className={`scope-row${s.id === here?.id ? " here" : ""}`} style={{ marginLeft: (s.depth - 1) * 16 }}>
              <div className="scope-line">
                <span className="scope-name">
                  {s.outer ? <BlockChip study={study} block={s.outer} /> : null} <b>round {study.roundNumber(s)}</b>
                  <span className="mono small muted"> · {s.node}</span>
                </span>
                <span className="scope-candidates small">
                  <span className="k">candidates</span>
                  {candidateRange(study, s)}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="muted small">no round</div>
      )}
      <dl className="facts-dl">
        <dt>role</dt>
        <dd>{role}</dd>
        <dt>proposed by</dt>
        <dd>
          {t.origin === "baseline" ? <span className="muted">baseline</span> : <BlockChip study={study} block={t.origin} />}
        </dd>
        <dt>best after</dt>
        <dd>
          {best.length ? (
            best.map((b, i) => (
              <Fragment key={b.number}>
                {i > 0 && ", "}
                <CandidateLink study={study} n={b.number}>{b.number}</CandidateLink>
              </Fragment>
            ))
          ) : (
            <span className="muted">none</span>
          )}
        </dd>
      </dl>
    </section>
  );
}

// ---- how it was proposed --------------------------------------------------------------------------------------------

function VarTable({ study, t, keys, from }: { study: StudyModel; t: CandidateT; keys: string[]; from?: (k: string) => ReactNode }) {
  return (
    <table className="grid-table var-table">
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td className="mono small">{k}</td>
            <td className="tnum">
              {!(k in t.values) ? (
                <span className="muted">–</span>
              ) : study.spaceByKey.get(k)?.kind === "text" && typeof t.values[k] === "string" ? (
                <details className="text-value">
                  <summary>{valueText(study, study.spaceByKey.get(k), t.values[k])}</summary>
                  <div className="textbox">{String(t.values[k])}</div>
                </details>
              ) : (
                valueText(study, study.spaceByKey.get(k), t.values[k])
              )}
            </td>
            {from && <td className="small">{from(k)}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Proposal({ study, t }: { study: StudyModel; t: CandidateT }) {
  const inherited = Object.keys(t.inherited);
  const held = Object.keys(t.values).filter((k) => provenance(study, t, k)?.how === "held");
  const parent = t.parentId === null ? undefined : study.byId.get(t.parentId);
  const proposer = t.origin === "baseline" ? null : study.h.blocks[t.origin]?.proposer ?? null;
  const tokens = study.searchTokens(t);
  return (
    <section className="card candidate-proposal">
      <h2>Proposal</h2>
      <div className="pblock">
        <div className="pblock-head">
          {t.origin === "baseline" ? <span className="tag">baseline</span> : <BlockChip study={study} block={t.origin} />}
          {proposer && <b className="mono">{proposer}</b>}
          {tokens > 0 && <span className="muted small tnum">{num(tokens)} proposer tokens</span>}
        </div>
        {t.rationale && (
          <div className="pblock-body">
            <span className="text">{t.rationale}</span>
          </div>
        )}
        <div className="pblock-body">
          <div className="pblock-vars">
            <div className="sub-head">
              changed <Count n={t.changedKeys.length} />
            </div>
            {t.changedKeys.length ? <VarTable study={study} t={t} keys={t.changedKeys} /> : <div className="muted small">none</div>}
          </div>
        </div>
      </div>
      {(inherited.length > 0 || t.defaulted.length > 0 || held.length > 0) && (
        <div className="pvars">
          {inherited.length > 0 && (
            <div>
              <div className="sub-head">
                inherited <Count n={inherited.length} />
              </div>
              <VarTable study={study} t={t} keys={inherited} />
            </div>
          )}
          {t.defaulted.length > 0 && (
            <div>
              <div className="sub-head">
                defaulted <Count n={t.defaulted.length} />
              </div>
              <VarTable study={study} t={t} keys={t.defaulted} />
            </div>
          )}
          {held.length > 0 && (
            <details>
              <summary className="sub-head">
                held from {parent === undefined ? "–" : `candidate ${parent.number}`} <Count n={held.length} />
              </summary>
              <VarTable study={study} t={t} keys={held} />
            </details>
          )}
        </div>
      )}
    </section>
  );
}

// ---- the change from its base candidate -----------------------------------------------------------------------------

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function BaseChanges({ study, t }: { study: StudyModel; t: CandidateT }) {
  if (t.parentId === null) return null;
  const base = study.byId.get(t.parentId);
  if (!base) return null;
  const order = [...study.space.map((s) => s.key)];
  for (const k of [...Object.keys(base.values), ...Object.keys(t.values)]) if (!order.includes(k)) order.push(k);
  const changed = order.filter((k) => (k in t.values || k in base.values) && !same(t.values[k], base.values[k]));
  const texts = changed.filter((k) => study.spaceByKey.get(k)?.kind === "text" && (typeof t.values[k] === "string" || typeof base.values[k] === "string"));
  return (
    <section className="card candidate-changes">
      <h2>
        Changes from <CandidateLink study={study} n={base.number} /> <span className="muted small">parent</span>
      </h2>
      {changed.length === 0 ? (
        <div className="muted">no changes</div>
      ) : (
        <>
          <table className="grid-table change-table">
            <thead>
              <tr>
                <th />
                <th>variable</th>
                <th>candidate {base.number}</th>
                <th>candidate {t.number}</th>
              </tr>
            </thead>
            <tbody>
              {changed.map((k) => {
                const s = study.spaceByKey.get(k);
                const b = study.blockOf.get(k);
                return (
                  <tr key={k}>
                    <td>{b && <BlockChip study={study} block={b} />}</td>
                    <td className="mono small">{k}</td>
                    <td className="tnum muted">{k in base.values ? valueText(study, s, base.values[k]) : "–"}</td>
                    <td className="tnum">{k in t.values ? valueText(study, s, t.values[k]) : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {texts.map((k) => (
            <div key={k} className="text-change">
              <div className="sub-head mono">{k}</div>
              {typeof base.values[k] === "string" && typeof t.values[k] === "string" ? (
                <TextDiff before={base.values[k] as string} after={t.values[k] as string} />
              ) : (
                <div className="textbox">{String(t.values[k] ?? base.values[k])}</div>
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
}

// ---- cases ----------------------------------------------------------------------------------------------------------

type FailGroup = { status: string; error: string | null; evaluatorErrors: string[]; trials: TrialRecord[] };

function failureGroups(t: CandidateT): FailGroup[] {
  const groups = new Map<string, FailGroup>();
  for (const a of t.trials) {
    const failedEvaluators = Object.keys(a.evaluator_errors ?? {});
    if (a.status === "completed" && !failedEvaluators.length) continue;
    const key = JSON.stringify([a.status, a.error, failedEvaluators]);
    if (!groups.has(key)) groups.set(key, { status: a.status, error: a.error, evaluatorErrors: failedEvaluators, trials: [] });
    groups.get(key)!.trials.push(a);
  }
  return [...groups.values()].sort((x, y) => y.trials.length - x.trials.length);
}

function groupWords(g: FailGroup): string {
  const parts = [words(g.status)];
  if (g.evaluatorErrors.length) parts.push(`evaluator failed: ${g.evaluatorErrors.join(", ")}`);
  return parts.join(" · ");
}

function Failures({ study, t }: { study: StudyModel; t: CandidateT }) {
  const groups = failureGroups(t);
  if (!groups.length) return null;
  return (
    <div className="failures">
      {groups.map((g, i) => {
        const byCase = new Map<string, TrialRecord[]>();
        for (const a of g.trials) byCase.set(a.case_name, [...(byCase.get(a.case_name) ?? []), a]);
        return (
          <div key={i} className="failure">
            <span className="fail-count tnum">
              {g.trials.length}/{t.trials.length}
            </span>
            <span className="fail-what">
              <b>{groupWords(g)}</b>
              {g.status === "agent_failure" && agentFailureText(study) && <span className="muted small"> agent_failure_score: {agentFailureText(study)}</span>}
              {g.error && (
                <div className="mono small fail-error" title={g.error}>
                  {firstLine(g.error)}
                </div>
              )}
            </span>
            <span className="fail-where small">
              {[...byCase.entries()].map(([c, as], j) => (
                <span key={c}>
                  {j > 0 && " · "}
                  {c}{" "}
                  {as.map((a, k) => (
                    <Fragment key={a.run_index}>
                      {k > 0 && ", "}
                      <a href={trialHref(study, t, a)}>{a.run_index}</a>
                    </Fragment>
                  ))}
                </span>
              ))}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** One trial's value of a metric, as a link to the trial: the number, × when nothing was measured, n/a when the
 * metric is unavailable. */
function RepValue({ study, t, a, metric }: { study: StudyModel; t: CandidateT; a: TrialRecord; metric: string }) {
  const title = `${a.case_name} · run ${a.run_index} · ${words(a.status)}${a.error ? `\n${firstLine(a.error)}` : ""}`;
  const { v, failure } = trialValue(study, a, metric);
  if (a.status === "infrastructure_invalid" || a.status === "evaluator_invalid") return <a className="rep x" href={trialHref(study, t, a)} title={title}>×</a>;
  if (v === null)
    return (
      <a className="rep na" href={trialHref(study, t, a)} title={`${title}\n${metric} ${failure ? "agent_failure: null" : "unavailable"}`}>
        n/a
      </a>
    );
  return (
    <a className={`rep${a.status === "completed" ? "" : " flagged"}`} href={trialHref(study, t, a)} title={`${title}\n${metric} = ${v}${failure ? " (agent_failure)" : ""}`}>
      {num(v)}
    </a>
  );
}

const STRIP_W = 70;
const STRIP_H = 14;

/** Many runs per case: one dot per measured run on a small strip (the metric's range, else this candidate's values),
 * the case mean as a tick; each dot opens its trial. */
function RepStrip({ study, t, as, metric, domain }: { study: StudyModel; t: CandidateT; as: TrialRecord[]; metric: string; domain: [number, number] }) {
  const [lo, hi] = domain;
  const x = (v: number) => (hi === lo ? STRIP_W / 2 : 4 + ((v - lo) / (hi - lo)) * (STRIP_W - 8));
  const vals = as.map((a) => ({ a, ...trialValue(study, a, metric) })).filter((x): x is { a: TrialRecord; v: number; failure: boolean } => x.v !== null);
  const mean = vals.length ? vals.reduce((s, x) => s + x.v, 0) / vals.length : null;
  return (
    <svg width={STRIP_W} height={STRIP_H} className="strip">
      <line x1={2} x2={STRIP_W - 2} y1={STRIP_H / 2} y2={STRIP_H / 2} stroke={MUTED} strokeOpacity={0.4} />
      {mean !== null && <line x1={x(mean)} x2={x(mean)} y1={1} y2={STRIP_H - 1} stroke={MARK} strokeWidth={1.5} />}
      {vals.map(({ a, v, failure }) => (
        <a key={a.run_index} href={trialHref(study, t, a)}>
          <title>{`${a.case_name} · run ${a.run_index} · ${words(a.status)} · ${metric} = ${v}${failure ? " (agent_failure)" : ""}`}</title>
          <circle cx={x(v)} cy={STRIP_H / 2} r={2.6} fill={a.status === "completed" ? MARK : "#fff"} stroke={MARK} strokeWidth={1.2} fillOpacity={0.75} />
        </a>
      ))}
    </svg>
  );
}

/** A case's mean of a metric over its measured runs. */
function caseMean(study: StudyModel, as: TrialRecord[], metric: string): number | null {
  const vs = as.map((a) => trialValue(study, a, metric).v).filter((v): v is number => v !== null);
  return vs.length ? vs.reduce((s, v) => s + v, 0) / vs.length : null;
}

function Cases({ study, t }: { study: StudyModel; t: CandidateT }) {
  const cases = [...study.h.cases];
  for (const a of t.trials) if (!cases.includes(a.case_name)) cases.push(a.case_name);
  const byCase = new Map(cases.map((c) => [c, t.trials.filter((a) => a.case_name === c).sort((x, y) => x.run_index - y.run_index)]));
  const reps = Math.max(0, ...[...byCase.values()].map((as) => as.length));
  const dots = reps > 3;
  const domain = (metric: string): [number, number] => {
    const vs = t.trials.map((a) => trialValue(study, a, metric).v).filter((v): v is number => v !== null);
    const [rlo, rhi] = study.range(metric);
    return [rlo ?? Math.min(...vs), rhi ?? Math.max(...vs)];
  };
  const goalCols: (Goal | null)[] = [...(study.objectives.length ? study.objectives : [null]), ...(study.constraints.length ? study.constraints : [null])];
  const ncols = goalCols.length;
  return (
    <section className="card candidate-cases">
      <h2>
        Cases <Count n={cases.length} />
        <span className="muted small"> · {t.trials.length} trials · {study.h.repetitions} run{study.h.repetitions === 1 ? "" : "s"} per case</span>
      </h2>
      {t.trials.length === 0 ? (
        <div className="muted">no trials</div>
      ) : (
        <>
          <Failures study={study} t={t} />
          <div className="cases-scroll">
            <table className="grid-table case-table">
              <thead>
                <tr className="case-groups">
                  <th colSpan={2} />
                  <th colSpan={Math.max(1, study.objectives.length)} className="gstart">
                    Objectives <Count n={study.objectives.length} />
                  </th>
                  <th colSpan={Math.max(1, study.constraints.length)} className="gstart">
                    Constraints <Count n={study.constraints.length} />
                  </th>
                </tr>
                <tr>
                  <th>case</th>
                  <th>trials</th>
                  {goalCols.map((g, i) => {
                    const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                    return (
                      <th key={g ? `${g.role}${g.index}` : `none${i}`} className={`num${start ? " gstart" : ""}`}>
                        {g ? <div className="case-head">{breakable(goalTitle(g))}</div> : <span className="muted">none</span>}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => {
                  const as = byCase.get(c)!;
                  const scored = as.filter((a) => a.status === "completed" || a.status === "agent_failure");
                  return (
                    <tr key={c}>
                      <td className="mono small case-name">{c}</td>
                      <td className="nowrap">
                        {as.length === 0 && <span className="muted">none</span>}
                        {as.map((a) => {
                          const measured = a.status === "completed" || a.status === "agent_failure";
                          return (
                            <a
                              key={a.run_index}
                              href={trialHref(study, t, a)}
                              className={`rep-link${a.status === "completed" ? "" : measured ? " flagged" : " x"}`}
                              title={`run ${a.run_index} · ${words(a.status)}${a.error ? `\n${firstLine(a.error)}` : ""}`}
                            >
                              {a.run_index}
                              {measured ? "" : "×"}
                            </a>
                          );
                        })}
                      </td>
                      {scored.length === 0 && as.length > 0 ? (
                        <td colSpan={ncols} className="gstart muted small">
                          {as.length}/{as.length} {[...new Set(as.map((a) => words(a.status)))].join(", ")}
                        </td>
                      ) : (
                        goalCols.map((g, i) => {
                          const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                          const key = g ? `${g.role}${g.index}` : `none${i}`;
                          if (!g) return <td key={key} className={start ? "gstart" : ""} />;
                          const mean = caseMean(study, as, g.metric);
                          return (
                            <td key={key} className={`num${start ? " gstart" : ""}`}>
                              <span className="cell">
                                {dots ? (
                                  <RepStrip study={study} t={t} as={as} metric={g.metric} domain={domain(g.metric)} />
                                ) : (
                                  as.length > 1 && (
                                    <span className="reps">
                                      {as.map((a) => (
                                        <RepValue key={a.run_index} study={study} t={t} a={a} metric={g.metric} />
                                      ))}
                                    </span>
                                  )
                                )}
                                {as.length === 1 && !dots ? (
                                  <RepValue study={study} t={t} a={as[0]} metric={g.metric} />
                                ) : (
                                  <b className="mean" title="case mean">
                                    {num(mean)}
                                  </b>
                                )}
                              </span>
                            </td>
                          );
                        })
                      )}
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2} className="muted small">
                    estimate
                  </td>
                  {goalCols.map((g, i) => {
                    const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                    const key = g ? `${g.role}${g.index}` : `none${i}`;
                    if (!g) return <td key={key} className={start ? "gstart" : ""} />;
                    const e = study.est(t, g.metric);
                    return (
                      <td key={key} className={`num${start ? " gstart" : ""}`}>
                        <EstimateText e={e} />
                        {g.role === "constraint" && (
                          <>
                            {" "}
                            <Check v={study.violation(t, g.index)} />
                          </>
                        )}
                      </td>
                    );
                  })}
                </tr>
              </tfoot>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

// ---- every metric -----------------------------------------------------------------------------------------------------

function AllMetrics({ study, t }: { study: StudyModel; t: CandidateT }) {
  const measured = t.label?.measured ?? {};
  const metrics = [...study.metrics];
  for (const m of Object.keys(measured)) if (!metrics.includes(m)) metrics.push(m);
  const cases = study.h.cases.filter((c) => t.trials.some((a) => a.case_name === c));
  if (!metrics.length || !Object.keys(measured).length) return null;
  return (
    <details className="card candidate-metrics">
      <summary>
        <h2>
          All metrics <Count n={metrics.length} />
        </h2>
      </summary>
      <div className="cases-scroll">
        <table className="grid-table">
          <thead>
            <tr>
              <th>metric</th>
              <th>role</th>
              <th className="num">estimate</th>
              <th className="num">n</th>
              {cases.map((c) => (
                <th key={c} className="num mono">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const e = study.est(t, m);
              return (
                <tr key={m}>
                  <td className="mono small">{m}</td>
                  <td className="muted small">{study.h.metrics[m]?.kind ?? "–"}</td>
                  <td className="num">
                    {e && e.mean !== null ? (
                      <span>
                        {num(e.mean)} <span className="ci">{ci(e.ci)}</span>
                      </span>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </td>
                  <td className="num muted small">{e?.n ?? ""}</td>
                  {cases.map((c) => (
                    <td key={c} className="num">
                      {num(caseMean(study, t.trials.filter((a) => a.case_name === c), m))}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function Candidate({ study, n }: { study: StudyModel; n: number }) {
  const t = study.byNumber.get(n);
  if (!t) return <div className="empty">No candidate {n}</div>;
  return (
    <div className="candidate">
      <Head study={study} t={t} />
      <div className="candidate-top">
        <Goals study={study} t={t} />
        <Schedule study={study} t={t} />
      </div>
      <Proposal study={study} t={t} />
      <BaseChanges study={study} t={t} />
      <Cases study={study} t={t} />
      <AllMetrics study={study} t={t} />
    </div>
  );
}
