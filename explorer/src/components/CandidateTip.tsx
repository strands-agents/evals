// The one tooltip for a candidate, everywhere: what it is, where it ran, what it changed, and how it did on each goal.

import { num, words } from "../format";
import { goalTitle } from "../model/goals";
import type { StudyModel } from "../model/study";
import { Changed, Choices, EstimateText, SchedulePath } from "./common";

export function CandidateTip({ study, n }: { study: StudyModel; n: number }) {
  const t = study.byNumber.get(n);
  if (!t) return null;
  const measured = t.trials.filter((a) => a.status === "completed").length;
  const lost = t.trials.find((a) => a.status !== "completed");
  const outcome = t.label?.outcome ?? null;
  const unmeasured = study.markKind(t) === "unmeasured";
  return (
    <div className="tip">
      <div className="tip-head">
        <b>Candidate {t.number}</b>
        {study.best.has(t.number) && <span className="tag">{study.bestIsSet ? "Pareto set" : "best"}</span>}
        {outcome !== null && outcome !== "feasible" && <span className="tag warn">{words(outcome)}</span>}
        {unmeasured && <span className="tag warn">{t.unscored ? "unscored" : "not measured"}</span>}
      </div>
      <div className="tip-sub">
        <SchedulePath study={study} t={t} />
      </div>
      <div className="tip-sub">
        <Changed study={study} t={t} />
      </div>
      <div className="tip-sub">
        <Choices study={study} t={t} />
      </div>
      {outcome !== null && outcome !== "feasible" && t.label?.reason && <div className="tip-sub bad">{t.label.reason}</div>}
      {lost && (
        <div className="tip-sub muted">
          {measured}/{t.trials.length} measured · {words(lost.status)}
        </div>
      )}
      {study.objectives.length > 0 && !unmeasured && (
        <div className="tip-group">
          {study.objectives.map((g) => (
            <div key={`o${g.index}`} className="tip-row">
              <span>{goalTitle(g)}</span>
              <EstimateText e={study.est(t, g.metric)} />
            </div>
          ))}
        </div>
      )}
      {study.constraints.length > 0 && !unmeasured && (
        <div className="tip-group">
          {study.constraints.map((g) => {
            const v = study.violation(t, g.index);
            return (
              <div key={`c${g.index}`} className="tip-row">
                <span>{goalTitle(g)}</span>
                <span className="tnum">
                  <b>{num(study.mean(t, g.metric))}</b> {v === null ? "" : v <= 0 ? <span className="good">✓</span> : <span className="bad">✗</span>}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
