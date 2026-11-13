// The run's name, state, goals (objectives, then constraints, always both) and the tabs.

import { num } from "../format";
import { arrow, op } from "../model/goals";
import type { StudyModel } from "../model/study";
import { href, type Route } from "../router";
import { Pill } from "./common";

export function StudyHeader({ study, route }: { study: StudyModel; route: Route }) {
  const tabs: { page: "overview" | "progress" | "tradeoff" | "space" | "candidates" | "setup"; label: string }[] = [
    { page: "setup", label: "Setup" },
    { page: "overview", label: "Overview" },
    { page: "progress", label: "Progress" },
    { page: "candidates", label: "Candidates" },
    { page: "space", label: "Tunables" },
    ...(study.multi ? [{ page: "tradeoff" as const, label: "Trade-off" }] : []),
  ];
  const active = route.page === "candidate" || route.page === "trial" ? "candidates" : route.page;
  const stop = study.record.stop;
  return (
    <header className="study-head">
      <div className="title-row">
        <h1>{study.name}</h1>
        {study.state === "running" && <Pill kind="running">running</Pill>}
        {study.state === "finished" && <Pill kind="finished">finished{stop ? ` · ${stop.reason}` : ""}</Pill>}
        {study.state === "stopped" && stop && (
          <Pill kind="stopped" title={stop.message}>
            stopped: {stop.reason}
            {stop.message ? ` · ${stop.message.length > 80 ? `${stop.message.slice(0, 80)}…` : stop.message}` : ""}
          </Pill>
        )}
      </div>
      <div className="goal-row">
        <div className="goal-group">
          <span className="role objective">objectives</span>
          {study.h.objectives.map((o, k) => (
            <span key={k} className="goal-chip">
              {arrow(o)} <b>{o.metric}</b>
            </span>
          ))}
          {study.multi && <span className="qualifier">{study.h.combine.kind}</span>}
        </div>
        <div className="goal-group">
          <span className="role constraint">constraints</span>
          {study.h.constraints.map((c, k) => (
            <span key={k} className="goal-chip">
              <b>{c.metric}</b> {op(c)} {num(c.limit)}
            </span>
          ))}
          {study.h.constraints.length === 0 && <span className="qualifier">none</span>}
        </div>
      </div>
      <nav className="tabs">
        {tabs.map((t) => (
          <a key={t.page} href={href(study.id, { page: t.page })} className={active === t.page ? "on" : ""}>
            {t.label}
          </a>
        ))}
      </nav>
    </header>
  );
}
