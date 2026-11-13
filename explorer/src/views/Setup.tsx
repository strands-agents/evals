// What the run was configured to do, exactly as its record says: the goals and how they combine, the schedule
// (with its nested inner schedule), the blocks, the search space as a tree, and the experiment.

import type { ReactNode } from "react";
import { BlockChip, valueText } from "../components/common";
import { num } from "../format";
import { arrow, goalTitle, op } from "../model/goals";
import { SELECTORS, type StudyModel } from "../model/study";
import type { ScheduleDesc, SpaceRow } from "../record/schema";
import "./Setup.css";

// ---- local helpers ------------------------------------------------------------------------------------------------

/** A recorded setting's value, as written: numbers formatted, null as "–", lists and objects as JSON-ish text. */
function settingText(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return num(v);
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return `[${v.map(settingText).join(", ")}]`;
  return JSON.stringify(v);
}

/** A metric's range, as an interval: [0, 1], [0, ∞). */
function rangeText([lo, hi]: [number | null, number | null]): string {
  return `${lo === null ? "(−∞" : `[${num(lo)}`}, ${hi === null ? "∞)" : `${num(hi)}]`}`;
}

/** The settings of a recorded description (a schedule node, a method, combine): every field but the named ones. */
function settings(desc: Record<string, unknown>, skip: string[]): [string, unknown][] {
  return Object.entries(desc).filter(([k]) => !skip.includes(k));
}

function Settings({ study, items, blocks = [] }: { study: StudyModel; items: [string, unknown][]; blocks?: string[] }) {
  if (!items.length) return null;
  return (
    <span className="su-settings">
      {items.map(([k, v]) => (
        <span key={k} className="su-setting">
          <span className="su-sk">{k}</span>
          <span className="su-eq">=</span>
          {blocks.includes(k) && (typeof v === "string" || Array.isArray(v)) ? (
            <span className="chips">
              {(Array.isArray(v) ? v : [v]).map((b) => (study.h.blocks[String(b)] ? <BlockChip key={String(b)} study={study} block={String(b)} /> : <span key={String(b)}>{settingText(b)}</span>))}
              {Array.isArray(v) && !v.length && <span className="su-sv">[]</span>}
            </span>
          ) : (
            <span className="su-sv">{settingText(v)}</span>
          )}
        </span>
      ))}
    </span>
  );
}

/** Text that may wrap only after _ / - or a dot that ends a name (keys and metric names), never inside a word or a
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

function Section({ title, count, children }: { title: string; count?: number; children: ReactNode }) {
  return (
    <section className="su-section">
      <div className="su-head">
        {title}
        {count !== undefined && <span className="count">{count}</span>}
      </div>
      {children}
    </section>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="su-empty">{children}</div>;
}

// ---- 1. goals ------------------------------------------------------------------------------------------------------

/** The objective settings shown: each only when some objective differs from its default (weight 1, scale None,
 * as Maximize/Minimize declare them). */
const OBJECTIVE_DEFAULTS: { key: "weight" | "scale"; value: number | null }[] = [
  { key: "weight", value: 1 },
  { key: "scale", value: null },
];

function Goals({ study }: { study: StudyModel }) {
  const shown = OBJECTIVE_DEFAULTS.filter((d) => study.h.objectives.some((o) => o[d.key] !== d.value));
  const hidden = OBJECTIVE_DEFAULTS.filter((d) => !shown.includes(d));
  const combine = study.h.combine;
  return (
    <>
      <div className="su-goals-grid">
      <Section title="Objectives" count={study.objectives.length}>
        {study.objectives.length ? (
          <div className="su-card">
            <table className="grid-table su-table">
              <thead>
                <tr>
                  <th className="num su-n">#</th>
                  <th>Kind</th>
                  <th>Metric</th>
                  {shown.map((d) => (
                    <th key={d.key} className="num">
                      {d.key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {study.objectives.map((g) => (
                  <tr key={g.index}>
                    <td className="num muted su-n">{g.index + 1}</td>
                    <td>
                      {arrow(g.objective)} {g.objective.kind}
                    </td>
                    <td className="mono su-wrap">
                      <Breakable text={g.metric} />
                    </td>
                    {shown.map((d) => (
                      <td key={d.key} className={`num${g.objective[d.key] === d.value ? " muted" : ""}`}>
                        {settingText(g.objective[d.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {hidden.length > 0 && (
              <div className="su-foot">
                {hidden.map((d) => `${d.key} = ${settingText(d.value)}`).join(" · ")}
              </div>
            )}
          </div>
        ) : (
          <Empty>No objectives</Empty>
        )}
      </Section>
      <Section title="Constraints" count={study.constraints.length}>
        {study.constraints.length ? (
          <div className="su-card">
            <table className="grid-table su-table">
              <thead>
                <tr>
                  <th className="num su-n">#</th>
                  <th>Metric</th>
                  <th>Operator</th>
                  <th className="num">Limit</th>
                  <th>Name</th>
                </tr>
              </thead>
              <tbody>
                {study.constraints.map((g) => (
                  <tr key={g.index}>
                    <td className="num muted su-n">{g.index + 1}</td>
                    <td className="mono su-wrap">
                      <Breakable text={g.metric} />
                    </td>
                    <td>
                      {op(g.constraint)} <span className="muted">{g.constraint.kind}</span>
                    </td>
                    <td className="num">{num(g.constraint.limit)}</td>
                    <td className="mono su-wrap">{g.constraint.name}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No constraints</Empty>
        )}
      </Section>
      </div>
      <Section title="Combine">
        <div className="su-card su-pad">
          <div className="su-node-row">
            <b className="su-kind">{combine.kind}</b>
            <Settings study={study} items={settings(combine, ["kind"])} />
          </div>
        </div>
      </Section>
    </>
  );
}

// ---- 2. the schedule -------------------------------------------------------------------------------------------------

/** The schedule settings that name blocks (Nested's outer, Alternate's blocks). */
const BLOCK_SETTINGS = ["outer", "blocks"];

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

function ScheduleView({ study, desc, name }: { study: StudyModel; desc: ScheduleDesc; name: string | null }) {
  const inner = (desc.config.inner ?? null) as ScheduleDesc | null;
  const rounds = name === null ? study.scopes.size - 1 : 0;
  return (
    <div className="su-node">
      <div className="su-node-row">
        {name !== null && <span className="su-child">{name}</span>}
        <b className="su-kind">{desc.name}</b>
        <Settings study={study} items={settings(desc.config, ["inner"])} blocks={BLOCK_SETTINGS} />
        {name === null && <span className="su-stats">{rounds > 0 ? plural(rounds, "round", "rounds") : "no rounds"}</span>}
      </div>
      {inner && (
        <div className="su-children">
          <ScheduleView study={study} desc={inner} name="inner" />
        </div>
      )}
    </div>
  );
}

// ---- 3. blocks -----------------------------------------------------------------------------------------------------

function Blocks({ study }: { study: StudyModel }) {
  const names = [...study.blockOrder, ...Object.keys(study.h.blocks).filter((b) => !study.blockOrder.includes(b))];
  return (
    <div className="su-card">
      <table className="grid-table su-table">
        <thead>
          <tr>
            <th>Block</th>
            <th>Proposer</th>
            <th>Variables</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const b = study.h.blocks[name];
            if (!b) return null;
            return (
              <tr key={name}>
                <td>
                  <BlockChip study={study} block={name} />
                </td>
                <td>
                  <div className="su-method">
                    <b className="su-kind">{b.proposer}</b>
                    <Settings study={study} items={settings(b.config ?? {}, [])} />
                  </div>
                </td>
                <td className="su-vars">
                  {b.keys.map((k) => (
                    <div key={k} className="mono su-wrap">
                      <Breakable text={k} />
                    </div>
                  ))}
                  {!b.keys.length && <span className="muted">none</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---- 4. the search space, as a tree --------------------------------------------------------------------------------

function optionText(v: unknown): string {
  return typeof v === "string" ? v : settingText(v);
}

function Domain({ s }: { s: SpaceRow }) {
  if (s.kind === "float" || s.kind === "int") {
    const step = typeof s.step === "number" ? s.step : null;
    return (
      <span className="tnum">
        [{num(s.min_value)}, {num(s.max_value)}]{s.log && <span className="su-flag">log</span>}
        {step !== null && <span className="su-flag">step {num(step)}</span>}
      </span>
    );
  }
  if (s.kind === "text") return <span className="tnum">{s.size_cap !== null ? `≤ ${num(s.size_cap)} chars` : "text"}</span>;
  return (
    <span className="su-options">
      {(s.options ?? []).map((c, i) => (
        <span key={i} className="su-option">
          {optionText(c)}
        </span>
      ))}
    </span>
  );
}

/** A text default, recorded only as a fingerprint: the version name, the length, and the hash's start. */
function textDefault(s: SpaceRow): string {
  const d = s.default as { sha256?: string; length?: number } | null;
  if (d && typeof d === "object") return `v0 · ${num(d.length ?? null)} chars${d.sha256 ? ` · ${d.sha256.slice(0, 8)}` : ""}`;
  return typeof s.default === "string" ? s.default : "–";
}

type Line = { kind: "row"; row: SpaceRow; depth: number } | { kind: "option"; key: string; option: string; depth: number };

/** The space in tree order: every row whose active_when is null, then under each selector, each of its options
 * that has variables, and those variables (recursively); a row whose condition names no selector in the space stays
 * at the top, so every row appears once. */
function spaceLines(study: StudyModel): Line[] {
  const under = new Map<string, SpaceRow[]>();
  const top: SpaceRow[] = [];
  for (const s of study.space) {
    const at = s.active_when as { key: string; option: string } | null;
    if (at && study.spaceByKey.has(at.key)) {
      const k = JSON.stringify([at.key, at.option]);
      under.set(k, [...(under.get(k) ?? []), s]);
    } else top.push(s);
  }
  const out: Line[] = [];
  const seen = new Set<string>();
  const visit = (s: SpaceRow, depth: number) => {
    if (seen.has(s.key)) return;
    seen.add(s.key);
    out.push({ kind: "row", row: s, depth });
    if (!SELECTORS.has(s.kind)) return;
    const options = (s.options ?? []).map(String);
    const named = new Set(options);
    // options in the selector's order, then any condition naming an option the selector does not list
    const extra = [...under.keys()].map((k) => JSON.parse(k) as [string, string]).filter(([key, o]) => key === s.key && !named.has(o)).map(([, o]) => o);
    for (const option of [...options, ...extra]) {
      const rows = under.get(JSON.stringify([s.key, option]));
      if (!rows?.length) continue;
      out.push({ kind: "option", key: s.key, option, depth: depth + 1 });
      for (const r of rows) visit(r, depth + 2);
    }
  };
  for (const s of top) visit(s, 0);
  for (const s of study.space) visit(s, 0); // anything under a condition that never resolved (a cycle)
  return out;
}

function Space({ study }: { study: StudyModel }) {
  const lines = spaceLines(study);
  const indent = (depth: number) => ({ paddingLeft: 16 + depth * 14 });
  return (
    <div className="su-card">
      <table className="grid-table su-table su-space">
        <thead>
          <tr>
            <th>Key</th>
            <th>Type</th>
            <th>Domain</th>
            <th>Default</th>
            <th>Block</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((l) => {
            if (l.kind === "option")
              return (
                <tr key={`${l.key}=${l.option}`} className="su-opt-row">
                  <td colSpan={5} style={indent(l.depth)}>
                    <span className="su-branch" />
                    <span className="mono">
                      <Breakable text={l.key} />
                    </span>{" "}
                    = <b>{l.option}</b>
                  </td>
                </tr>
              );
            const s = l.row;
            const block = study.blockOf.get(s.key);
            return (
              <tr key={s.key} className={SELECTORS.has(s.kind) ? "su-selector" : ""}>
                <td className="mono su-wrap su-key" style={indent(l.depth)}>
                  <Breakable text={s.key} />
                </td>
                <td className="su-type">{s.kind}</td>
                <td>
                  <Domain s={s} />
                </td>
                <td className="su-default">
                  {s.kind === "text" || s.kind === "skills" ? <span className="tnum">{textDefault(s)}</span> : <span className="tnum">{valueText(study, s, s.default)}</span>}
                </td>
                <td>{block ? <BlockChip study={study} block={block} /> : <span className="muted">–</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!lines.length && <div className="su-foot">No variables</div>}
    </div>
  );
}

// ---- 5. the experiment ---------------------------------------------------------------------------------------------

function KV({ k, children }: { k: string; children: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{children}</dd>
    </>
  );
}

function Experiment({ study }: { study: StudyModel }) {
  const h = study.h;
  const config = study.config;
  const goalsOf = (m: string) => study.goals.filter((g) => g.metric === m);
  const splits: ["train" | "validation" | "test", string[]][] = [
    ["train", config.split.train],
    ["validation", config.split.validation],
    ["test", config.split.test],
  ];
  return (
    <div className="su-card su-pad">
      <dl className="su-kv">
        <KV k="agent">
          <span className="mono su-wrap">{h.agent}</span>
        </KV>
        <KV k="cases">
          <span className="tnum">{h.cases.length.toLocaleString()}</span>
          <span className="su-cases">
            {splits.map(([name, cases]) =>
              cases.map((c) => (
                <span key={c} className="su-option mono" title={name}>
                  {c}
                </span>
              )),
            )}
          </span>
        </KV>
        <KV k="split">
          <span className="tnum">{splits.map(([name, cases]) => `${name} ${cases.length}`).join(" · ")}</span>
        </KV>
        <KV k="runs per candidate">
          <span className="tnum">{num(h.repetitions)}</span>
        </KV>
        <KV k="budget">
          <span className="tnum">
            {num(config.budget.max_rollouts)} rollouts
            {config.budget.max_iterations !== null ? ` · ${num(config.budget.max_iterations)} iterations` : ""}
          </span>
        </KV>
        <KV k="final pick">
          <span className="tnum">
            {config.final_pick.rule.name} · {num(config.final_pick.n_finalists)} finalists
          </span>
        </KV>
        <KV k="score model">{config.score_model.name}</KV>
        <KV k="selection">{config.selection.name}</KV>
        {config.agent_failure_score !== null && <KV k="agent_failure_score">{String(config.agent_failure_score)}</KV>}
        <KV k="state">
          {study.state}
          {study.record.stop ? ` · ${study.record.stop.reason}` : ""}
          {study.record.resumes.length ? ` · ${study.record.resumes.length} resume${study.record.resumes.length === 1 ? "" : "s"}` : ""}
        </KV>
      </dl>
      <table className="grid-table su-table su-metrics">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Kind</th>
            <th>Roles</th>
            <th>Range</th>
            <th>Goals</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(h.metrics).map(([m, d]) => (
            <tr key={m}>
              <td className="mono su-wrap">
                <Breakable text={m} />
              </td>
              <td>{d.kind}</td>
              <td>{d.roles.length ? d.roles.join(", ") : <span className="muted">–</span>}</td>
              <td className="tnum">{rangeText(d.range)}</td>
              <td>
                <span className="su-goals">
                  {goalsOf(m).map((g) => (
                    <span key={`${g.role}${g.index}`} className={`su-goal ${g.role}`}>
                      {goalTitle(g)}
                    </span>
                  ))}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- the page ------------------------------------------------------------------------------------------------------

export function Setup({ study }: { study: StudyModel }) {
  return (
    <div className="setup">
      <Goals study={study} />
      <Section title="Schedule">
        <div className="su-card su-pad su-tree">
          {study.h.schedule ? (
            <ScheduleView study={study} desc={study.h.schedule} name={null} />
          ) : (
            <div className="su-node-row">
              <b className="su-kind">joint</b>
              <span className="su-stats">every block, every iteration</span>
            </div>
          )}
        </div>
      </Section>
      <Section title="Blocks" count={Object.keys(study.h.blocks).length}>
        <Blocks study={study} />
      </Section>
      <Section title="Search space" count={study.space.length}>
        <Space study={study} />
      </Section>
      <Section title="Experiment">
        <Experiment study={study} />
      </Section>
    </div>
  );
}
