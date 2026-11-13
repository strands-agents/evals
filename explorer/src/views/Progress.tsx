// How every objective and constraint evolved under the schedule: one column of panels on one candidate axis, under a
// schedule track that stays in view. Objectives, then constraints (always shown), then every other metric (folded).

import type { ReactNode } from "react";
import { leftMargin, candidateAxis, type CandidateAxis } from "../charts/axis";
import { useWidth } from "../charts/kit";
import { scheduleLabels, ScheduleTrack } from "../charts/ScheduleTrack";
import { CandidatePanel, type Value } from "../charts/CandidatePanel";
import { Legend } from "../components/Legend";
import { CandidateTip } from "../components/CandidateTip";
import { num } from "../format";
import { goalTitle } from "../model/goals";
import type { Goal, StudyModel } from "../model/study";
import type { Candidate } from "../record/parse";
import { Tooltip } from "../interaction/hover";
import { href } from "../router";
import "./Progress.css";

type Panel = {
  key: string;
  title: string;
  stats: { k: string; v: string; n?: number[] }[];
  value: (t: Candidate) => Value | null;
  range: [number | null, number | null];
  goal?: Goal;
  bestLine?: boolean;
};

function metricValue(study: StudyModel, metric: string) {
  return (t: Candidate): Value | null => {
    const e = study.est(t, metric);
    return e && e.mean !== null ? { mean: e.mean, ci: e.ci } : null;
  };
}

function baselineStat(study: StudyModel, value: (t: Candidate) => Value | null) {
  const b = study.baseline ? value(study.baseline) : null;
  return b ? [{ k: "baseline", v: num(b.mean) }] : [];
}

function panels(study: StudyModel) {
  const best = [...study.best].map((n) => study.byNumber.get(n)!).filter(Boolean);
  const objectives: Panel[] = study.objectives.map((g) => {
    const value = metricValue(study, g.metric);
    const vs = best.map((t) => value(t)?.mean).filter((v): v is number => v !== undefined && v !== null);
    const stats = study.bestIsSet
      ? vs.length
        ? [{ k: `Pareto set (${best.length})`, v: vs.length > 1 ? `${num(Math.min(...vs))} – ${num(Math.max(...vs))}` : num(vs[0]) }]
        : []
      : best[0] && value(best[0])
        ? [{ k: "best", v: num(value(best[0])!.mean), n: best.map((t) => t.number) }]
        : [];
    return { key: `o${g.index}`, title: goalTitle(g), stats: [...stats, ...baselineStat(study, value)], value, range: study.range(g.metric), goal: g, bestLine: true };
  });
  if (study.toldScalar) {
    const value = (t: Candidate): Value | null => {
      const v = study.objectiveOf(t);
      return v === null ? null : { mean: v, ci: null };
    };
    objectives.unshift({ key: "objective", title: study.toldScalar, stats: baselineStat(study, value), value, range: [null, null], bestLine: true });
  }
  const constraints: Panel[] = study.constraints.map((g) => {
    const value = metricValue(study, g.metric);
    const judged = study.candidates.filter((t) => study.comparable(t) && study.violation(t, g.index) !== null);
    const met = judged.filter((t) => study.violation(t, g.index)! <= 0).length;
    return { key: `c${g.index}`, title: goalTitle(g), stats: [{ k: "met", v: `${met} of ${judged.length}` }, ...baselineStat(study, value)], value, range: study.range(g.metric), goal: g };
  });
  const goalMetrics = new Set(study.goals.map((g) => g.metric));
  const others: Panel[] = study.metrics
    .filter((m) => !goalMetrics.has(m))
    .map((m) => {
      const value = metricValue(study, m);
      return { key: `m:${m}`, title: m, stats: baselineStat(study, value), value, range: study.range(m) };
    });
  return { objectives, constraints, others };
}

function PanelCard({ study, axis, width, p }: { study: StudyModel; axis: CandidateAxis; width: number; p: Panel }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{p.title}</h3>
        <div className="panel-stats">
          {p.stats.map((s) => (
            <span key={s.k}>
              <span className="k">{s.k}</span> <b>{s.v}</b>
              {s.n?.map((n) => (
                <a key={n} href={href(study.id, { page: "candidate", n })}>
                  {" "}#{n}
                </a>
              ))}
            </span>
          ))}
        </div>
      </div>
      <CandidatePanel study={study} axis={axis} width={width} value={p.value} range={p.range} goal={p.goal} bestLine={p.bestLine} />
    </section>
  );
}

function Group({ title, count, children }: { title: string; count: number; children: ReactNode }) {
  return (
    <div className="panel-group">
      <div className="group-head">
        {title} <span className="count">{count}</span>
      </div>
      {children}
    </div>
  );
}

function ProgressLegend({ study }: { study: StudyModel }) {
  return <Legend study={study} interval bestLine limit={study.constraints.length > 0} />;
}

export function Progress({ study }: { study: StudyModel }) {
  const [ref, measured] = useWidth<HTMLDivElement>();
  const width = measured - 2; // inside the cards' borders
  const left = leftMargin(scheduleLabels(study));
  const axis = candidateAxis(study, width, left);
  const { objectives, constraints, others } = panels(study);
  return (
    <div ref={ref} className="progress">
      <ProgressLegend study={study} />
      <div className="track-card">
        <ScheduleTrack study={study} axis={axis} width={width} />
      </div>
      <Group title="Objectives" count={study.objectives.length}>
        {objectives.map((p) => (
          <PanelCard key={p.key} study={study} axis={axis} width={width} p={p} />
        ))}
      </Group>
      <Group title="Constraints" count={study.constraints.length}>
        {constraints.length ? constraints.map((p) => <PanelCard key={p.key} study={study} axis={axis} width={width} p={p} />) : <div className="empty-panel">No constraints</div>}
      </Group>
      {others.length > 0 && (
        <details className="panel-group">
          <summary className="group-head">
            Other metrics <span className="count">{others.length}</span>
          </summary>
          {others.map((p) => (
            <PanelCard key={p.key} study={study} axis={axis} width={width} p={p} />
          ))}
        </details>
      )}
      <Tooltip render={(n) => <CandidateTip study={study} n={n} />} />
    </div>
  );
}
