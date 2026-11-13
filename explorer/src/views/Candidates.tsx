// Find and compare candidates: one row per candidate. Sticky on the left: the candidate, where it ran, what it changed,
// how it is drawn (the mark glyph). Then the objectives, the constraints (always, "none" when there are none), and
// every variable under its block (collapsible). Filters above; sortable headers; hover shows the candidate, a click
// opens it.

import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { MarkGlyph } from "../charts/kit";
import { levelLabel } from "../charts/ScheduleTrack";
import { BlockChip, Changed, SchedulePath, valueText } from "../components/common";
import { CandidateTip } from "../components/CandidateTip";
import { blockColor } from "../encoding/theme";
import { ci, num, words } from "../format";
import { Tooltip, useHover } from "../interaction/hover";
import { goalTitle } from "../model/goals";
import { SELECTORS, type MarkKind, type StudyModel } from "../model/study";
import type { Candidate } from "../record/parse";
import { href, useGo } from "../router";
import "./Candidates.css";

// ---- columns ----------------------------------------------------------------------------------------------------------

type Sortable = number | string | null;

type Col = {
  id: string;
  group: string; // the header group it sits under
  head: ReactNode;
  title?: string;
  cls?: string;
  sort?: (t: Candidate) => Sortable;
  cell: (t: Candidate) => ReactNode;
};

type Group = { id: string; head: ReactNode; cls?: string };

const KIND_ORDER: Record<MarkKind, number> = { ok: 0, infeasible: 1, ineligible: 2, unmeasured: 3 };

/** Break long identifiers at their separators, never inside a word. */
export function breakable(s: string): ReactNode {
  const parts = s.split(/(?<=[._])/);
  return parts.map((p, i) => (
    <span key={i}>
      {p}
      {i < parts.length - 1 && <wbr />}
    </span>
  ));
}

/** A block's variable headers: the full keys, less a dotted prefix common to every key of the block, when removing it
 * leaves every key non-empty and no remainder could be read as another variable of the study. */
export function elision(keys: string[], allKeys: Set<string>): string {
  if (keys.length < 2) return "";
  const split = keys.map((k) => k.split("."));
  let c = 0;
  while (split.every((s) => s.length > c + 1 && s[c] === split[0][c])) c++;
  if (c === 0) return "";
  const prefix = `${split[0].slice(0, c).join(".")}.`;
  const rest = keys.map((k) => k.slice(prefix.length));
  if (rest.some((r) => allKeys.has(r))) return "";
  return prefix;
}

/** How a variable got its value in a candidate, from the record: changed by its proposing block, inherited,
 * at its default, or held from the parent candidate. */
export function provenance(study: StudyModel, t: Candidate, key: string): { how: "changed" | "held" | "inherited" | "defaulted"; from: number | null } | null {
  if (!(key in t.values)) return null;
  if (t.changedKeys.includes(key)) return { how: "changed", from: null };
  if (key in t.inherited) return { how: "inherited", from: null };
  if (t.defaulted.includes(key)) return { how: "defaulted", from: null };
  return { how: "held", from: t.parentId === null ? null : study.byId.get(t.parentId)?.number ?? null };
}

function provenanceText(p: ReturnType<typeof provenance>): string {
  if (!p) return "not in values";
  if (p.how === "changed") return "changed";
  if (p.how === "inherited") return "inherited";
  if (p.how === "defaulted") return "defaulted";
  return p.from === null ? "held" : `held from candidate ${p.from}`;
}

/** A candidate's role, in the record's words. */
export function roleText(t: Candidate): string {
  return words(t.role);
}

function sortValue(v: unknown, textIndex?: (s: string) => number): Sortable {
  if (v === undefined || v === null) return null;
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") return textIndex ? textIndex(v) : v;
  return JSON.stringify(v);
}

function StatusGlyph({ study, t }: { study: StudyModel; t: Candidate }) {
  const kind = study.markKind(t);
  const best = study.best.has(t.number);
  const title = [
    t.label ? `outcome: ${words(t.label.outcome)}` : t.unscored ? "unscored (budget)" : "not measured",
    t.label?.reason ?? "",
    best ? (study.bestIsSet ? "Pareto set" : "best") : "",
  ].filter(Boolean).join("\n");
  return (
    <span className="tr-glyph" title={title}>
      {kind === "unmeasured" ? <span className="tr-x">×</span> : <MarkGlyph kind={kind} best={best} />}
      {kind === "unmeasured" && <span className="tr-status">{t.unscored ? "unscored" : "not measured"}</span>}
    </span>
  );
}

function columns(study: StudyModel, collapsed: Set<string>): { cols: Col[]; groups: (Group & { span: number })[]; sticky: number } {
  const cols: Col[] = [
    { id: "n", group: "candidate", head: "#", cls: "num tr-n", sort: (t) => t.number, cell: (t) => <a href={href(study.id, { page: "candidate", n: t.number })} onClick={(e) => e.stopPropagation()}>{t.number}</a> },
    {
      id: "path",
      group: "candidate",
      head: "schedule",
      cls: "tr-path",
      cell: (t) => (
        <>
          <SchedulePath study={study} t={t} />
          {t.role !== "baseline" && (
            <span className="muted">
              {study.path(t).some((s) => s.depth > 0) ? " · " : ""}
              {roleText(t)}
            </span>
          )}
        </>
      ),
    },
    { id: "changed", group: "candidate", head: "changed", cls: "tr-changed", cell: (t) => (t.role === "baseline" ? <span className="muted">–</span> : <Changed study={study} t={t} />) },
    {
      id: "status",
      group: "candidate",
      head: "",
      title: "status · feasible · best",
      cls: "tr-statuscol",
      sort: (t) => (study.best.has(t.number) ? 0 : 10) + KIND_ORDER[study.markKind(t)],
      cell: (t) => <StatusGlyph study={study} t={t} />,
    },
  ];
  const sticky = cols.length;
  const groups: Group[] = [
    { id: "candidate", head: "", cls: "tr-sticky-group" },
    { id: "objectives", head: <>Objectives <span className="count">{study.objectives.length}</span></>, cls: "tr-goal-group" },
    { id: "constraints", head: <>Constraints <span className="count">{study.constraints.length}</span></>, cls: "tr-goal-group" },
  ];

  for (const g of study.objectives) {
    cols.push({
      id: `o${g.index}`,
      group: "objectives",
      head: breakable(goalTitle(g)),
      title: goalTitle(g),
      cls: "num tr-goal",
      sort: (t) => study.mean(t, g.metric),
      cell: (t) => {
        const e = study.est(t, g.metric);
        if (!e || e.mean === null) return <span className="muted">–</span>;
        return <span title={e.ci ? `95% interval ${ci(e.ci)} · n ${e.n}` : `n ${e.n}`}>{num(e.mean)}</span>;
      },
    });
  }
  if (!study.objectives.length) cols.push({ id: "o-none", group: "objectives", head: <span className="muted">none</span>, cls: "tr-none", cell: () => null });

  for (const g of study.constraints) {
    cols.push({
      id: `c${g.index}`,
      group: "constraints",
      head: breakable(goalTitle(g)),
      title: goalTitle(g),
      cls: "num tr-goal",
      sort: (t) => study.mean(t, g.metric),
      cell: (t) => {
        const e = study.est(t, g.metric);
        const v = study.violation(t, g.index);
        return (
          <span title={e?.ci ? `95% interval ${ci(e.ci)} · n ${e.n}` : undefined}>
            {e && e.mean !== null ? num(e.mean) : <span className="muted">–</span>}
            <span className="tr-check">{v === null ? "" : v <= 0 ? <span className="good">✓</span> : <span className="bad">✗</span>}</span>
          </span>
        );
      },
    });
  }
  if (!study.constraints.length) cols.push({ id: "c-none", group: "constraints", head: <span className="muted">none</span>, cls: "tr-none", cell: () => null });

  const allKeys = new Set(study.space.map((s) => s.key));
  const inBlocks = new Set<string>();
  for (const b of study.blockOrder) {
    const keys = study.h.blocks[b]?.keys ?? [];
    keys.forEach((k) => inBlocks.add(k));
    const prefix = elision(keys, allKeys);
    const open = !collapsed.has(b);
    groups.push({
      id: `b:${b}`,
      head: (
        <span className="tr-block-head">
          <svg className="tr-caret" width={10} height={10} viewBox="0 0 10 10" aria-hidden>
            <path d={open ? "M1.5 3.5 L5 7 L8.5 3.5" : "M3.5 1.5 L7 5 L3.5 8.5"} fill="none" stroke="currentColor" strokeWidth={1.6} />
          </svg>
          <BlockChip study={study} block={b} />
          {open && prefix && <span className="mono tr-prefix">{prefix}…</span>}
          {!open && <span className="count">{keys.length}</span>}
        </span>
      ),
      cls: "tr-block-group",
    });
    if (!open) {
      cols.push({
        id: `b:${b}`,
        group: `b:${b}`,
        head: "",
        title: `${b} · ${keys.length} variable${keys.length === 1 ? "" : "s"}`,
        cls: "tr-folded",
        sort: (t) => (study.changed(t).includes(b) ? 0 : 1),
        cell: (t) => (study.changed(t).includes(b) ? <i className="tr-tick" style={{ background: blockColor(study, b) }} /> : null),
      });
      continue;
    }
    for (const key of keys) cols.push(paramCol(study, `b:${b}`, key, prefix));
    if (!keys.length) cols.push({ id: `b:${b}:none`, group: `b:${b}`, head: <span className="muted">none</span>, cls: "tr-none", cell: () => null });
  }
  // variables the record has outside every block (none in a well-formed study, but never hidden)
  const loose = study.space.filter((s) => !inBlocks.has(s.key)).map((s) => s.key);
  if (loose.length) {
    groups.push({ id: "loose", head: <span className="muted">in no block</span>, cls: "tr-block-group" });
    for (const key of loose) cols.push(paramCol(study, "loose", key, ""));
  }

  cols.forEach((c, i) => {
    if (i >= sticky && (i === sticky || cols[i - 1].group !== c.group)) c.cls = `${c.cls ?? ""} gstart`;
  });
  // group spans, in column order
  const spans = groups
    .map((g) => ({ ...g, span: cols.filter((c) => c.group === g.id).length }))
    .filter((g) => g.span > 0);
  return { cols, groups: spans, sticky };
}

function paramCol(study: StudyModel, group: string, key: string, prefix: string): Col {
  const row = study.spaceByKey.get(key);
  const versions = row?.kind === "text" ? study.textVersions(key) : null;
  const textIndex = versions ? (s: string) => versions.findIndex((v) => v.text === s) : undefined;
  const shown = prefix ? key.slice(prefix.length) : key;
  return {
    id: `p:${key}`,
    group,
    head: (
      <>
        {prefix && <span className="muted">…</span>}
        {breakable(shown)}
      </>
    ),
    title: key,
    cls: row && (row.kind === "float" || row.kind === "int") ? "num tr-param" : "tr-param",
    sort: (t) => sortValue(t.values[key], textIndex),
    cell: (t) => {
      const p = provenance(study, t, key);
      if (!p) return <span className="muted">–</span>;
      return (
        <span className={p.how === "changed" ? "tr-drawn" : "tr-kept"} title={provenanceText(p)}>
          {valueText(study, row, t.values[key])}
        </span>
      );
    },
  };
}

// ---- filters ----------------------------------------------------------------------------------------------------------

type Filters = {
  outcome: Set<MarkKind>;
  best: "now" | "ever" | null;
  rounds: Record<number, number | null>; // level index -> scope id
  changed: Set<string>;
  choices: Record<string, string | null>;
};

const EMPTY: Filters = { outcome: new Set(), best: null, rounds: {}, changed: new Set(), choices: {} };

const OUTCOME_WORDS: Record<MarkKind, string> = { ok: "feasible", infeasible: "infeasible", ineligible: "ineligible", unmeasured: "not measured" };

function matcher(study: StudyModel, f: Filters) {
  const rounds = Object.values(f.rounds).filter((id): id is number => id !== null);
  const choices = Object.entries(f.choices).filter((e): e is [string, string] => e[1] !== null);
  return (t: Candidate) => {
    if (f.outcome.size && !f.outcome.has(study.markKind(t))) return false;
    if (f.best === "now" && !study.best.has(t.number)) return false;
    if (f.best === "ever" && !study.bestSpans.has(t.number)) return false;
    for (const id of rounds) if (!study.path(t).some((s) => s.id === id)) return false;
    if (f.changed.size && !study.changed(t).some((b) => f.changed.has(b))) return false;
    for (const [k, o] of choices) if (!(k in t.values) || String(t.values[k]) !== o) return false;
    return true;
  };
}

function toggle<T>(s: Set<T>, v: T): Set<T> {
  const out = new Set(s);
  if (out.has(v)) out.delete(v);
  else out.add(v);
  return out;
}

function Chip({ on, onClick, children, count, title }: { on: boolean; onClick: () => void; children: ReactNode; count?: number; title?: string }) {
  return (
    <button type="button" className={`fchip${on ? " on" : ""}`} onClick={onClick} title={title} aria-pressed={on} disabled={count === 0 && !on}>
      {children}
      {count !== undefined && <span className="count">{count}</span>}
    </button>
  );
}

function FilterBar({ study, f, set, shown }: { study: StudyModel; f: Filters; set: (f: Filters) => void; shown: number }) {
  const count = (pred: (t: Candidate) => boolean) => study.candidates.filter(pred).length;
  const outcomes = (["ok", "infeasible", "ineligible", "unmeasured"] as MarkKind[]).filter((k) => study.candidates.some((t) => study.markKind(t) === k));
  const bestWord = study.bestIsSet ? "Pareto set" : "best";
  const selectors = study.space.filter((s) => SELECTORS.has(s.kind));
  const changedBlocks = study.blockOrder.filter((b) => study.candidates.some((t) => study.changed(t).includes(b)));
  const active = f.outcome.size + (f.best ? 1 : 0) + Object.values(f.rounds).filter((v) => v !== null).length + f.changed.size + Object.values(f.choices).filter((v) => v !== null).length;
  return (
    <div className="filters">
      <div className="frow">
        <span className="fgroup">
          <span className="flabel">outcome</span>
          {outcomes.map((k) => (
            <Chip key={k} on={f.outcome.has(k)} onClick={() => set({ ...f, outcome: toggle(f.outcome, k) })} count={count((t) => study.markKind(t) === k)}>
              {OUTCOME_WORDS[k]}
            </Chip>
          ))}
        </span>
        <span className="fgroup">
          <span className="flabel">{bestWord}</span>
          <Chip on={f.best === "now"} onClick={() => set({ ...f, best: f.best === "now" ? null : "now" })} count={study.best.size}>
            now
          </Chip>
          <Chip on={f.best === "ever"} onClick={() => set({ ...f, best: f.best === "ever" ? null : "ever" })} count={study.bestSpans.size} title="best_after">
            ever
          </Chip>
        </span>
        {changedBlocks.length > 0 && (
          <span className="fgroup">
            <span className="flabel">changed</span>
            {changedBlocks.map((b) => (
              <Chip key={b} on={f.changed.has(b)} onClick={() => set({ ...f, changed: toggle(f.changed, b) })} count={count((t) => study.changed(t).includes(b))}>
                <BlockChip study={study} block={b} />
              </Chip>
            ))}
          </span>
        )}
      </div>
      {(study.levels.length > 0 || selectors.length > 0) && (
        <div className="frow">
          {study.levels.map((level, i) => (
            <label key={`${level.depth}|${level.node}|${level.outer}`} className="fgroup">
              <span className="flabel">{levelLabel(level)}</span>
              <select
                value={f.rounds[i] ?? ""}
                onChange={(e) => set({ ...f, rounds: { ...f.rounds, [i]: e.target.value === "" ? null : Number(e.target.value) } })}
              >
                <option value="">any</option>
                {level.scopes.map((s) => (
                  <option key={s.id} value={s.id}>
                    round {study.roundNumber(s)} ({s.candidates.length})
                  </option>
                ))}
              </select>
            </label>
          ))}
          {selectors.map((s) => (
            <label key={s.key} className="fgroup">
              <span className="flabel mono">{s.key}</span>
              <span className="muted">=</span>
              <select value={f.choices[s.key] ?? ""} onChange={(e) => set({ ...f, choices: { ...f.choices, [s.key]: e.target.value === "" ? null : e.target.value } })}>
                <option value="">any</option>
                {(s.options ?? []).map((c) => (
                  <option key={String(c)} value={String(c)}>
                    {String(c)} ({count((t) => s.key in t.values && String(t.values[s.key]) === String(c))})
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}
      <div className="frow fsummary">
        <span>
          {shown === study.candidates.length ? `${study.candidates.length} candidates` : `${shown} of ${study.candidates.length} candidates`}
        </span>
        {active > 0 && (
          <button type="button" className="fclear" onClick={() => set(EMPTY)}>
            clear
          </button>
        )}
      </div>
    </div>
  );
}

// ---- the table --------------------------------------------------------------------------------------------------------

type Sort = { col: string; dir: 1 | -1 };

function compare(a: Sortable, b: Sortable): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "number") return -1;
  if (typeof b === "number") return 1;
  return a.localeCompare(b);
}

const Body = memo(function Body({ rows, cols, lefts, sticky, onHover, onOpen }: {
  rows: Candidate[];
  cols: Col[];
  lefts: number[];
  sticky: number;
  onHover: (n: number | null, e?: { clientX: number; clientY: number }) => void;
  onOpen: (n: number) => void;
}) {
  return (
    <tbody onMouseLeave={() => onHover(null)}>
      {rows.map((t) => (
        <tr key={t.number} onMouseMove={(e) => onHover(t.number, e)} onClick={() => onOpen(t.number)} className={t.role === "baseline" ? "tr-baseline" : undefined}>
          {cols.map((c, i) => (
            <td key={c.id} className={`${c.cls ?? ""}${i < sticky ? " stick" : ""}${i === sticky - 1 ? " stick-last" : ""}`} style={i < sticky ? { left: lefts[i] ?? 0 } : undefined}>
              {c.cell(t)}
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
});

/** The body, given only the hover setter (stable), so moving the pointer re-renders the tooltip, not the table. */
function HoverBody(props: Omit<Parameters<typeof Body>[0], "onHover">) {
  const { set } = useHover();
  return <Body {...props} onHover={set} />;
}

/** Forget the hovered candidate when the rows under the pointer change (filters, sort) and when the table goes away. */
function HoverReset({ on }: { on: unknown }) {
  const { set } = useHover();
  useEffect(() => {
    set(null);
    return () => set(null);
  }, [on, set]);
  return null;
}

export function Candidates({ study }: { study: StudyModel }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [filters, setFilters] = useState<Filters>(EMPTY);
  const [sort, setSort] = useState<Sort>({ col: "n", dir: 1 });
  const { cols, groups, sticky } = useMemo(() => columns(study, collapsed), [study, collapsed]);
  const rows = useMemo(() => {
    const match = matcher(study, filters);
    const out = study.candidates.filter(match);
    const col = cols.find((c) => c.id === sort.col);
    if (col?.sort) {
      const key = col.sort;
      const vals = new Map(out.map((t) => [t.number, key(t)]));
      out.sort((a, b) => {
        const va = vals.get(a.number)!;
        const vb = vals.get(b.number)!;
        if (va === null || vb === null) return compare(va, vb) || a.number - b.number; // missing values last either way
        return compare(va, vb) * sort.dir || a.number - b.number;
      });
    }
    return out;
  }, [study, filters, sort, cols]);

  // sticky columns: each one's left edge is the width of those before it, measured from the header
  const headRefs = useRef<(HTMLTableCellElement | null)[]>([]);
  const [lefts, setLefts] = useState<number[]>([]);
  useLayoutEffect(() => {
    const ws = headRefs.current.slice(0, sticky).map((th) => th?.getBoundingClientRect().width ?? 0);
    const next = ws.map((_, i) => ws.slice(0, i).reduce((a, b) => a + b, 0));
    if (next.some((v, i) => Math.abs(v - (lefts[i] ?? -1)) > 0.5)) setLefts(next);
  });

  const go = useGo();
  const onOpen = useMemo(() => (n: number) => go(study.id, { page: "candidate", n }), [go, study.id]);
  const sortBy = (c: Col) => {
    if (!c.sort) return;
    setSort((s) => (s.col === c.id ? { col: c.id, dir: s.dir === 1 ? -1 : 1 } : { col: c.id, dir: 1 }));
  };
  const fold = (g: string) => {
    if (!g.startsWith("b:")) return;
    setCollapsed((s) => toggle(s, g.slice(2)));
  };

  if (!study.candidates.length) return <div className="empty">No candidates</div>;
  return (
    <div className="candidates">
      <FilterBar study={study} f={filters} set={setFilters} shown={rows.length} />
      <div className="tr-scroll">
        <table className="tr-table">
          <thead>
            <tr className="tr-groups">
              {groups.map((g) => (
                <th
                  key={g.id}
                  colSpan={g.span}
                  className={`${g.cls ?? ""}${g.id === "candidate" ? " stick stick-last" : ""}`}
                  style={g.id === "candidate" ? { left: 0 } : undefined}
                  onClick={() => fold(g.id)}
                  title={g.id.startsWith("b:") ? (collapsed.has(g.id.slice(2)) ? "unfold" : "fold") : undefined}
                >
                  {g.head}
                </th>
              ))}
            </tr>
            <tr className="tr-heads">
              {cols.map((c, i) => (
                <th
                  key={c.id}
                  ref={(el) => {
                    headRefs.current[i] = el;
                  }}
                  className={`${c.cls ?? ""}${i < sticky ? " stick" : ""}${i === sticky - 1 ? " stick-last" : ""}${c.sort ? " sortable" : ""}${sort.col === c.id ? " sorted" : ""}`}
                  style={i < sticky ? { left: lefts[i] ?? 0 } : undefined}
                  title={c.title}
                  onClick={() => sortBy(c)}
                >
                  <div className="tr-head">
                    {c.head}
                    {sort.col === c.id && c.sort && <span className="tr-arrow">{sort.dir === 1 ? "▲" : "▼"}</span>}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <HoverBody rows={rows} cols={cols} lefts={lefts} sticky={sticky} onOpen={onOpen} />
        </table>
        {!rows.length && <div className="empty">No matches</div>}
      </div>
      <HoverReset on={rows} />
      <Tooltip render={(n) => <CandidateTip study={study} n={n} />} />
    </div>
  );
}
