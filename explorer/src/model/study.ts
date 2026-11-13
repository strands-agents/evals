// A run as the views read it: the record's facts, indexed, plus the few things that are pure functions of them
// (the round tree from each candidate's round path, the best-so-far spans from the iteration records, text
// versions). Nothing here is guessed, colored or worded: encodings live in encoding/, words in model/goals.ts.

import type { Candidate, Lifecycle, RunRecord } from "../record/parse";
import type {
  BlockDesc,
  Config,
  Measured,
  MetricDesc,
  ScheduleDesc,
  SpaceRow,
  TestReport,
  TrialRecord,
  ValidationReport,
} from "../record/schema";

export const SELECTORS = new Set(["one_of", "choice"]);

export type Objective = { kind: "Maximize" | "Minimize"; metric: string; weight: number; scale: number | null };
export type Constraint = { kind: "AtMost" | "AtLeast"; metric: string; limit: number; name: string };

/** One goal of the run: an objective or a constraint, by its position. */
export type Goal =
  | { role: "objective"; index: number; metric: string; objective: Objective }
  | { role: "constraint"; index: number; metric: string; constraint: Constraint };

/** How a candidate is drawn: feasible, breaking a constraint, not comparable (a train case unmeasured), or
 * nothing measured yet (unscored, or no label). */
export type MarkKind = "ok" | "infeasible" | "ineligible" | "unmeasured";

/** What charts read for one metric of one candidate: the measured mean, its 95% interval, and the case count. */
export type ViewEstimate = { mean: number | null; ci: [number, number] | null; n: number };

/** One Nested round (or the run's root): synthesized from each candidate's round path. */
export type Scope = {
  id: number;
  path: number[]; // [] for the root
  node: string;
  outer: string | null;
  round: number | null; // the last element of the path
  parent: Scope | null;
  depth: number;
  children: Scope[];
  candidates: number[]; // candidate numbers, in order
};

export type Level = { depth: number; node: string; outer: string | null; scopes: Scope[] };

export type Choice = { key: string; option: string };

/** The combine setting, parsed from the config's objective string. */
export type Combine = { kind: string; best_is_set: boolean };

export type Header = {
  agent: string;
  objectives: Objective[];
  constraints: Constraint[];
  combine: Combine;
  schedule: ScheduleDesc | null;
  blocks: Record<string, BlockDesc>;
  block_order: string[];
  space: SpaceRow[];
  metrics: Record<string, MetricDesc>;
  cases: string[];
  repetitions: number;
};

function parseObjectives(config: Config): Objective[] {
  const out: Objective[] = [];
  for (const m of config.objective.matchAll(/\b(Maximize|Minimize)\(\s*'?([\w.]+)'?/g)) {
    const metric = m[2];
    const desc = config.metrics[metric];
    out.push({ kind: m[1] as Objective["kind"], metric, weight: desc?.weight ?? 1, scale: desc?.scale ?? null });
  }
  return out;
}

function parseConstraints(config: Config): Constraint[] {
  return config.constraints.map((name) => {
    const m = name.match(/^([\w.]+)\s*(<=|>=)\s*(-?\d+(?:\.\d+)?)$/);
    if (!m) return { kind: "AtMost" as const, metric: name, limit: Number.NaN, name };
    return { kind: m[2] === "<=" ? ("AtMost" as const) : ("AtLeast" as const), metric: m[1], limit: Number(m[3]), name };
  });
}

export class StudyModel {
  readonly h: Header;
  readonly config: Config;
  readonly candidates: Candidate[];
  readonly byNumber: Map<number, Candidate>;
  readonly byId: Map<string, Candidate>;
  readonly space: SpaceRow[];
  readonly spaceByKey: Map<string, SpaceRow>;
  readonly blockOf: Map<string, string>;
  readonly blockOrder: string[];
  readonly goals: Goal[];
  readonly objectives: Extract<Goal, { role: "objective" }>[];
  readonly constraints: Extract<Goal, { role: "constraint" }>[];
  readonly scopes: Map<number, Scope>;
  readonly roots: Scope[];
  readonly levels: Level[];
  /** The best candidates now: the last iteration's `best_after`, or the validation pick once there is one. */
  readonly best: Set<number>;
  /** For each candidate ever in the best set: the candidate number after which it entered and left (null: still in). */
  readonly bestSpans: Map<number, { from: number; to: number | null }[]>;
  /** The best set after each iteration, keyed by the last candidate number proposed by then. */
  private bestTimeline: { after: number; best: number[] }[] = [];
  /** The end phase, when the run reached it. */
  readonly finalists: Candidate[];
  readonly validation: ValidationReport | null;
  readonly test: TestReport | null;
  readonly testFront: Record<string, TestReport> | null;

  constructor(readonly record: RunRecord, readonly id: string, readonly name: string) {
    const config = record.config;
    this.config = config;
    this.candidates = record.candidates;
    this.byNumber = new Map(this.candidates.map((t) => [t.number, t]));
    this.byId = record.byId;
    this.space = config.space.map((s) =>
      Array.isArray(s.active_when) ? { ...s, active_when: { key: s.active_when[0], option: s.active_when[1] } } : s,
    );
    this.spaceByKey = new Map(this.space.map((s) => [s.key, s]));
    this.blockOf = new Map();
    for (const b of config.search.blocks) for (const k of b.keys) this.blockOf.set(k, b.name);
    this.blockOrder = config.search.blocks.map((b) => b.name);
    const blocks = Object.fromEntries(config.search.blocks.map((b) => [b.name, b]));
    const objectives = parseObjectives(config);
    const constraints = parseConstraints(config);
    const kind = config.objective.match(/^\s*([A-Za-z_]+)/)?.[1] ?? "Maximize";
    this.h = {
      agent: Object.values(config.agent ?? {}).filter(Boolean).join(" "),
      objectives,
      constraints,
      combine: { kind, best_is_set: kind === "Pareto" },
      schedule: config.search.schedule,
      blocks,
      block_order: this.blockOrder,
      space: this.space,
      metrics: config.metrics,
      cases: [...config.split.train, ...config.split.validation, ...config.split.test],
      repetitions: config.runs_per_candidate,
    };
    this.objectives = objectives.map((objective, index) => ({ role: "objective" as const, index, metric: objective.metric, objective }));
    this.constraints = constraints.map((constraint, index) => ({ role: "constraint" as const, index, metric: constraint.metric, constraint }));
    this.goals = [...this.objectives, ...this.constraints];

    // the round tree, from each candidate's round path
    this.scopes = new Map();
    this.roots = [];
    const byPath = new Map<string, Scope>();
    const outerOf = (depth: number): string | null => {
      // the default schedule nests one level per structural block; the config records the outer block's name
      const s = config.search.schedule;
      if (!s || depth !== 1) return null;
      return (s.config.outer as string | undefined) ?? null;
    };
    const root: Scope = { id: 0, path: [], node: config.search.schedule?.name ?? "joint", outer: null, round: null, parent: null, depth: 0, children: [], candidates: [] };
    this.scopes.set(0, root);
    this.roots.push(root);
    byPath.set("", root);
    let nextId = 1;
    const scopeFor = (path: number[]): Scope => {
      const key = path.join(".");
      const seen = byPath.get(key);
      if (seen) return seen;
      const parent = scopeFor(path.slice(0, -1));
      const scope: Scope = {
        id: nextId++,
        path,
        node: config.search.schedule?.name ?? "round",
        outer: outerOf(path.length),
        round: path[path.length - 1] ?? null,
        parent,
        depth: path.length,
        children: [],
        candidates: [],
      };
      parent.children.push(scope);
      byPath.set(key, scope);
      this.scopes.set(scope.id, scope);
      return scope;
    };
    for (const t of this.candidates) {
      const scope = t.round && t.round.length ? scopeFor(t.round) : root;
      for (let s: Scope | null = scope; s; s = s.parent) s.candidates.push(t.number);
    }
    const levels = new Map<string, Level>();
    for (const s of this.scopes.values()) {
      if (s.depth === 0) continue;
      const key = `${s.depth}|${s.node}|${s.outer}`;
      if (!levels.has(key)) levels.set(key, { depth: s.depth, node: s.node, outer: s.outer, scopes: [] });
      levels.get(key)!.scopes.push(s);
    }
    this.levels = [...levels.values()].sort((a, b) => a.depth - b.depth);

    // best now, and when each candidate was in the best set, in candidate-number coordinates
    const numberOf = (id: string) => this.byId.get(id)?.number;
    const last = record.iterations.at(-1);
    const bestIds = record.picked ? (record.picked.front.length ? record.picked.front : [record.picked.champion_id]) : last?.best_after ?? [];
    this.best = new Set(bestIds.map(numberOf).filter((n): n is number => n !== undefined));
    this.bestSpans = new Map();
    let prev = new Set<number>();
    for (const it of record.iterations) {
      const after = Math.max(0, ...it.record.proposals.map((p) => p.number ?? 0));
      const now = new Set(it.best_after.map(numberOf).filter((n): n is number => n !== undefined));
      this.bestTimeline.push({ after, best: [...now] });
      for (const n of now) if (!prev.has(n)) this.bestSpans.set(n, [...(this.bestSpans.get(n) ?? []), { from: after, to: null }]);
      for (const n of prev)
        if (!now.has(n)) {
          const spans = this.bestSpans.get(n)!;
          spans[spans.length - 1].to = after;
        }
      prev = now;
    }

    this.validation = record.picked;
    this.test = record.test;
    this.testFront = record.testFront;
    this.finalists = (record.finalists ?? []).map((id) => this.byId.get(id)).filter((t): t is Candidate => !!t);
  }

  // ---- the run -----------------------------------------------------------------------------------------------------
  get state(): Lifecycle {
    return this.record.state;
  }
  get baseline(): Candidate | undefined {
    return this.candidates.find((t) => t.role === "baseline");
  }
  get multi(): boolean {
    return this.objectives.length > 1;
  }
  get bestIsSet(): boolean {
    return this.h.combine.best_is_set;
  }
  /** The combined scalar the search ranks on, when it is not a single metric: shown as its own panel. */
  get toldScalar(): string | null {
    return this.h.combine.kind === "Maximize" || this.h.combine.kind === "Minimize" ? null : "objective";
  }
  /** Every metric any candidate has measured values for: goals first, then evaluators, then usage. */
  get metrics(): string[] {
    const seen = new Set(this.candidates.flatMap((t) => Object.keys(t.label?.measured ?? {})));
    const goals = [...new Set(this.goals.map((g) => g.metric))];
    const rest = Object.keys(this.h.metrics).filter((m) => seen.has(m) && !goals.includes(m));
    const role = (m: string) => (this.h.metrics[m].kind === "evaluator" ? 0 : 1);
    return [...goals, ...rest.sort((a, b) => role(a) - role(b))];
  }
  range(metric: string): [number | null, number | null] {
    return this.h.metrics[metric]?.range ?? [null, null];
  }

  // ---- one candidate -----------------------------------------------------------------------------------------------
  measured(t: Candidate, metric: string): Measured | null {
    return t.label?.measured?.[metric] ?? null;
  }
  est(t: Candidate, metric: string): ViewEstimate | null {
    const m = this.measured(t, metric);
    return m ? { mean: m.mean, ci: m.ci95, n: m.n } : null;
  }
  mean(t: Candidate, metric: string): number | null {
    return this.measured(t, metric)?.mean ?? null;
  }
  /** The train objective: the mean of the per-run objectives the score event recorded. */
  objectiveOf(t: Candidate): number | null {
    const runs = Object.values(t.scores.train?.objective_by_run ?? {}).flat();
    if (!runs.length) return null;
    return runs.reduce((a, b) => a + b, 0) / runs.length;
  }
  /** How far over the line the candidate's measured mean is: positive means violated. */
  violation(t: Candidate, index: number): number | null {
    const c = this.constraints[index]?.constraint;
    if (!c) return null;
    const mean = this.mean(t, c.metric);
    if (mean === null || Number.isNaN(c.limit)) return null;
    return c.kind === "AtMost" ? mean - c.limit : c.limit - mean;
  }
  /** The probability the ranking saw of this candidate meeting constraint `index`. */
  pMet(t: Candidate, index: number): number | null {
    const name = this.constraints[index]?.constraint.name;
    return name == null ? null : t.label?.p_met?.[name] ?? null;
  }
  markKind(t: Candidate): MarkKind {
    if (!t.label) return "unmeasured";
    if (t.label.outcome === "ineligible") return "ineligible";
    return t.label.outcome === "infeasible" ? "infeasible" : "ok";
  }
  /** Whether the candidate can be ranked against the others: labeled, and not ineligible. */
  comparable(t: Candidate): boolean {
    const k = this.markKind(t);
    return k === "ok" || k === "infeasible";
  }
  /** The best set once candidate `n` had been proposed, from the iteration records. */
  bestAfter(n: number): Candidate[] {
    let at: number[] = [];
    for (const b of this.bestTimeline) {
      if (b.after > n) break;
      at = b.best;
    }
    return at.map((k) => this.byNumber.get(k)).filter((t): t is Candidate => !!t);
  }
  /** The blocks the candidate's proposers changed, in block order. */
  changed(t: Candidate): string[] {
    const touched = new Set(t.changedKeys.map((k) => this.blockOf.get(k)).filter(Boolean));
    return this.blockOrder.filter((b) => touched.has(b) || t.origin === b);
  }
  /** Every choice the candidate made, outermost first. */
  choices(t: Candidate): Choice[] {
    return this.space.filter((s) => SELECTORS.has(s.kind) && s.key in t.values).map((s) => ({ key: s.key, option: String(t.values[s.key]) }));
  }
  /** The rounds the candidate ran in, outermost first. */
  path(t: Candidate): Scope[] {
    if (!t.round || !t.round.length) return [];
    const out: Scope[] = [];
    for (let i = 1; i <= t.round.length; i++) {
      const s = [...this.scopes.values()].find((x) => x.path.join(".") === t.round!.slice(0, i).join("."));
      if (s) out.push(s);
    }
    return out;
  }
  /** A round's number, counted within its enclosing round: "3", or "3.2" for the second round inside round 3. */
  roundNumber(scope: Scope): string {
    return scope.path.map((n) => n + 1).join(".");
  }
  /** The model usage the proposers spent on this candidate (reflection's own calls). */
  searchTokens(t: Candidate): number {
    return t.searchUsageTokens;
  }

  // ---- trials ------------------------------------------------------------------------------------------------------
  /** A trial's per-metric values: each evaluator's first score, plus the usage numbers. */
  trialMetrics(r: TrialRecord): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [name, rows] of Object.entries(r.evaluations ?? {})) {
      const score = (Array.isArray(rows) ? rows[0] : rows)?.score;
      if (typeof score === "number") out[name] = score;
    }
    const u = r.usage ?? {};
    for (const k of ["input_tokens", "output_tokens", "tokens", "seconds"] as const) {
      const v = u[k];
      if (typeof v === "number") out[k] = v;
    }
    return out;
  }
  /** A trial's judge reasons, per evaluator. */
  trialReasons(r: TrialRecord): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [name, rows] of Object.entries(r.evaluations ?? {})) {
      const reason = (Array.isArray(rows) ? rows[0] : rows)?.reason;
      if (reason) out[name] = reason;
    }
    return out;
  }

  // ---- the space ---------------------------------------------------------------------------------------------------
  /** The choices a variable exists under, outermost first. */
  conditions(key: string): Choice[] {
    const out: Choice[] = [];
    let at = (this.spaceByKey.get(key)?.active_when ?? null) as Choice | null;
    while (at) {
      out.unshift({ key: at.key, option: at.option });
      at = (this.spaceByKey.get(at.key)?.active_when ?? null) as Choice | null;
    }
    return out;
  }

  private versions = new Map<string, { text: string; first: number | null; candidates: number[] }[]>();

  /** Every distinct value of a text variable, in order of first use (v0 is the baseline's). The config stores only
   * a fingerprint of the default text, so the baseline candidate's value is the first full text we have. */
  textVersions(key: string): { text: string; first: number | null; candidates: number[] }[] {
    const cached = this.versions.get(key);
    if (cached) return cached;
    const base = this.baseline?.values[key];
    const out: { text: string; first: number | null; candidates: number[] }[] = [
      { text: typeof base === "string" ? base : "", first: null, candidates: [] },
    ];
    for (const t of this.candidates) {
      const v = t.values[key];
      if (typeof v !== "string") continue;
      let found = out.find((o) => o.text === v);
      if (!found) out.push((found = { text: v, first: t.number, candidates: [] }));
      if (found.first === null && t.role !== "baseline") found.first = t.number;
      found.candidates.push(t.number);
    }
    this.versions.set(key, out);
    return out;
  }
}
