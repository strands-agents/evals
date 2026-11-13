// Small pieces every view uses, so a fact is always shown the same way.

import type { ReactNode } from "react";
import { blockColor } from "../encoding/theme";
import { ci, num } from "../format";
import type { StudyModel, ViewEstimate } from "../model/study";
import type { Candidate } from "../record/parse";
import type { SpaceRow } from "../record/schema";
import { href } from "../router";

export function Pill({ kind, children, title }: { kind: string; children: ReactNode; title?: string }) {
  return (
    <span className={`pill ${kind}`} title={title}>
      <span className="dot" />
      {children}
    </span>
  );
}

/** A block, named with its color key (the one place hue appears outside the schedule track). */
export function BlockChip({ study, block }: { study: StudyModel; block: string }) {
  return (
    <span className="block-chip">
      <i style={{ background: blockColor(study, block) }} />
      {block}
    </span>
  );
}

/** The blocks a candidate changed, or "baseline". */
export function Changed({ study, t }: { study: StudyModel; t: Candidate }) {
  if (t.role === "baseline") return null;
  return (
    <span className="chips">
      {study.changed(t).map((b) => (
        <BlockChip key={b} study={study} block={b} />
      ))}
    </span>
  );
}

/** A candidate's choices at every depth, as text: conversation_manager = summarizing › style = brief. */
export function Choices({ study, t }: { study: StudyModel; t: Candidate }) {
  const choices = study.choices(t);
  if (!choices.length) return null;
  return (
    <span className="choices">
      {choices.map((c, i) => (
        <span key={c.key}>
          {i > 0 && <span className="sep"> › </span>}
          <span className="mono">{c.key}</span> = <b>{c.option}</b>
        </span>
      ))}
    </span>
  );
}

/** Where a candidate sits in the schedule: "structure round 2 › numbers round 2.1". */
export function SchedulePath({ study, t }: { study: StudyModel; t: Candidate }) {
  if (t.role === "baseline") return <span>baseline</span>;
  const rounds = study.path(t).filter((s) => s.depth > 0);
  if (!rounds.length) return null;
  return (
    <span>
      {rounds.map((s, i) => (
        <span key={s.id}>
          {i > 0 && " › "}
          {s.outer} round {study.roundNumber(s)}
        </span>
      ))}
    </span>
  );
}

export function EstimateText({ e }: { e: ViewEstimate | null }) {
  if (!e || e.mean === null) return <span className="muted">–</span>;
  return (
    <span className="tnum">
      <b>{num(e.mean)}</b> <span className="ci">{ci(e.ci)}</span>
    </span>
  );
}

/** A parameter value for a table cell. Text is shown by its version (v0 is the default) and length. */
export function valueText(study: StudyModel, s: SpaceRow | undefined, v: unknown): string {
  if (v === undefined) return "–";
  if (s?.kind === "skills") {
    if (typeof v !== "string" || !v) return "empty";
    const i = study.textVersions(s.key).findIndex((x) => x.text === v);
    return `v${i} · ${v.split("/").filter(Boolean).pop()?.slice(0, 8)}`;
  }
  if (s?.kind === "text" && typeof v === "string") {
    const i = study.textVersions(s.key).findIndex((x) => x.text === v);
    return `v${i} · ${v.length.toLocaleString()} chars`;
  }
  if (typeof v === "number") return s?.kind === "int" ? Math.round(v).toLocaleString() : num(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

export function CandidateLink({ study, n, children }: { study: StudyModel; n: number | null | undefined; children?: ReactNode }) {
  if (n === null || n === undefined) return <span className="muted">–</span>;
  return <a href={href(study.id, { page: "candidate", n })}>{children ?? `candidate ${n}`}</a>;
}

/** Long text: shown whole when short, else a one-line preview that opens. */
export function Long({ text, limit = 1200, mono }: { text: string; limit?: number; mono?: boolean }) {
  const cls = mono ? "text mono small" : "text";
  if (text.length <= limit) return <div className={cls}>{text}</div>;
  return (
    <details>
      <summary>{text.slice(0, 180).replace(/\s+/g, " ")}…</summary>
      <div className={cls}>{text}</div>
    </details>
  );
}
