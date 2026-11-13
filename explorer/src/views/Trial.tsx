// One trial: a case at one run index, under one candidate. Its status, its metrics (the goals' first, then the rest),
// what could not be measured and why, the evaluators' reasons, the output, and the transcript in the record's format.

import { Changed, Choices, Long, SchedulePath, CandidateLink } from "../components/common";
import { TranscriptView } from "../components/Transcript";
import { num, words } from "../format";
import { goalTitle } from "../model/goals";
import type { Goal, StudyModel } from "../model/study";
import { useTranscript } from "../record/load";
import type { TrialRecord } from "../record/schema";
import type { Candidate } from "../record/parse";
import type { StudyRef } from "../source";
import { agentFailureText, trialHref, trialValue, Count } from "./Candidate";
import { breakable, roleText } from "./Candidates";
import "./Candidate.css";
import "./Trial.css";

function Nav({ study, t, i }: { study: StudyModel; t: Candidate; i: number }) {
  const prev = t.trials[i - 1];
  const next = t.trials[i + 1];
  const label = (a: TrialRecord) => `${a.case_name} · ${a.run_index}`;
  return (
    <div className="candidate-nav">
      {prev ? <a href={trialHref(study, t, prev)}>‹ {label(prev)}</a> : <span />}
      {next ? <a href={trialHref(study, t, next)}>{label(next)} ›</a> : <span />}
    </div>
  );
}

/** A metric's value in this trial, and whether it is the configured agent_failure score, not a measurement. */
function ValueCell({ study, a, metric }: { study: StudyModel; a: TrialRecord; metric: string }) {
  const { v, failure } = trialValue(study, a, metric);
  if (failure)
    return (
      <td className="num">
        {v !== null ? <b>{num(v)}</b> : <span className="muted">null</span>}
        <span className="muted small"> agent_failure</span>
      </td>
    );
  return (
    <td className="num">
      {v !== null ? <b>{num(v)}</b> : <span className="muted">{metric in (a.evaluator_errors ?? {}) ? "evaluator failed" : "–"}</span>}
    </td>
  );
}

function GoalValues({ study, goals, a }: { study: StudyModel; goals: Goal[]; a: TrialRecord }) {
  return (
    <table className="grid-table at-metrics">
      <tbody>
        {goals.map((g) => (
          <tr key={`${g.role}${g.index}`}>
            <td className="goal-name">{breakable(goalTitle(g))}</td>
            <ValueCell study={study} a={a} metric={g.metric} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Metrics({ study, a }: { study: StudyModel; a: TrialRecord }) {
  const values = study.trialMetrics(a);
  const goalMetrics = new Set(study.goals.map((g) => g.metric));
  const order = study.metrics.filter((m) => !goalMetrics.has(m));
  for (const m of Object.keys(values)) if (!goalMetrics.has(m) && !order.includes(m)) order.push(m);
  const rest = order.filter((m) => m in values || trialValue(study, a, m).failure);
  const failed = Object.entries(a.evaluator_errors ?? {});
  return (
    <section className="card">
      <h2>Metrics</h2>
      <div className="group-head">
        Objectives <Count n={study.objectives.length} />
      </div>
      {study.objectives.length ? <GoalValues study={study} goals={study.objectives} a={a} /> : <div className="none-box">No objectives</div>}
      <div className="group-head">
        Constraints <Count n={study.constraints.length} />
      </div>
      {study.constraints.length ? <GoalValues study={study} goals={study.constraints} a={a} /> : <div className="none-box">No constraints</div>}
      {rest.length > 0 && (
        <>
          <div className="group-head">
            Other metrics <Count n={rest.length} />
          </div>
          <table className="grid-table at-metrics">
            <tbody>
              {rest.map((m) => (
                <tr key={m}>
                  <td className="mono small">{m}</td>
                  <td className="muted small">{study.h.metrics[m]?.kind ?? ""}</td>
                  <ValueCell study={study} a={a} metric={m} />
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      <div className="group-head">
        Evaluator errors <Count n={failed.length} />
      </div>
      {failed.length ? (
        <div className="chips">
          {failed.map(([name, err]) => (
            <span key={name} className="tag mono" title={err}>
              {name}
            </span>
          ))}
        </div>
      ) : (
        <div className="muted small">none</div>
      )}
    </section>
  );
}

function TranscriptCard({ studyRef, a }: { studyRef: StudyRef; a: TrialRecord }) {
  const hash = typeof a.transcript === "string" ? a.transcript : a.transcript?.hash ?? null;
  const transcript = useTranscript(studyRef, hash);
  return (
    <section className="card at-transcript">
      <h2>
        Transcript
        {hash && <span className="muted small mono"> · {hash}</span>}
      </h2>
      {!hash ? (
        <div className="muted">no transcript</div>
      ) : transcript === null ? (
        <div className="muted">loading…</div>
      ) : transcript === "missing" ? (
        <div className="muted">transcripts/{hash}.json missing</div>
      ) : (
        <TranscriptView transcript={transcript} />
      )}
    </section>
  );
}

export function Trial({ study, studyRef, n, kase, rep }: { study: StudyModel; studyRef: StudyRef; n: number; kase: string; rep: number }) {
  const t = study.byNumber.get(n);
  const i = t ? t.trials.findIndex((a) => a.case_name === kase && a.run_index === rep) : -1;
  if (!t) return <div className="empty">No candidate {n}</div>;
  if (i < 0)
    return (
      <div className="empty">
        No trial {kase} · {rep} in <CandidateLink study={study} n={n} />
      </div>
    );
  const a = t.trials[i];
  const reasons = Object.entries(study.trialReasons(a));
  return (
    <div className="trial candidate">
      <div className="candidate-head">
        <Nav study={study} t={t} i={i} />
        <div className="candidate-title">
          <h2>
            <CandidateLink study={study} n={t.number}>Candidate {t.number}</CandidateLink> <span className="sep">›</span> <span className="mono">{a.case_name}</span> <span className="sep">›</span> run {a.run_index}
          </h2>
          <span className={`tag${a.status === "completed" ? "" : " warn"}`}>{words(a.status)}</span>
          <span className="tag">{a.split}</span>
          {a.attempts > 1 && <span className="tag warn">attempt {a.attempts}</span>}
          {a.status === "agent_failure" && agentFailureText(study) && <span className="muted small">agent_failure_score: {agentFailureText(study)}</span>}
          <span className="muted small">
            trial {i + 1} of {t.trials.length}
          </span>
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
      </div>
      {a.error !== null && (
        <section className="card at-error">
          <h2>Error</h2>
          <pre className="textbox mono small">{a.error}</pre>
        </section>
      )}
      <div className="at-top">
        <Metrics study={study} a={a} />
        <div>
          <section className="card">
            <h2>
              Reasons <Count n={reasons.length} />
            </h2>
            {reasons.length ? (
              <table className="grid-table at-reasons">
                <tbody>
                  {reasons.map(([m, r]) => (
                    <tr key={m}>
                      <td className="mono small">{m}</td>
                      <td>
                        <Long text={r} limit={600} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="muted small">none</div>
            )}
          </section>
          <section className="card">
            <h2>Output</h2>
            {a.output === null ? <div className="muted small">none</div> : a.output === "" ? <div className="muted small">empty</div> : <pre className="textbox">{a.output}</pre>}
          </section>
        </div>
      </div>
      <TranscriptCard studyRef={studyRef} a={a} />
    </div>
  );
}
