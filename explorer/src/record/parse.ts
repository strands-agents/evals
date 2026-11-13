// Reading an events.jsonl strictly: a strands_evals.optimization run record (run.json schema_version 1), every
// line one event; only a final line still being written may be incomplete. A resume discards the search events of
// the killed iteration, exactly as the optimizer's own reader does, so a resumed run parses to the same state —
// but keeps its trial events, which that reader keeps too (they are the cache the re-run reads from).

import type {
  Config,
  Event,
  EvidenceEvent,
  FitEvent,
  IterationEndEvent,
  Label,
  Measured,
  ObserveEvent,
  ProposeEvent,
  ResumeEvent,
  ScoreEvent,
  Split,
  StopEvent,
  TestReport,
  TrialRecord,
  ValidationReport,
} from "./schema";
import { SCHEMA_VERSION } from "./schema";

export class RecordError extends Error {
  constructor(message: string, readonly line?: number) {
    super(line === undefined ? message : `line ${line}: ${message}`);
  }
}

export type Lifecycle = "running" | "finished" | "stopped";

/** One candidate, assembled from its propose, trial, score, and label events. */
export type Candidate = {
  id: string;
  number: number;
  parentId: string | null;
  /** "baseline" or the proposing block's name. */
  origin: string;
  rationale: string;
  values: Record<string, unknown>;
  inherited: Record<string, unknown>;
  defaulted: string[];
  changedKeys: string[];
  round: number[] | null;
  role: "baseline" | "opener" | "step";
  rescored: boolean;
  label: Label | null;
  scores: Partial<Record<Split, ScoreEvent>>;
  trials: TrialRecord[];
  /** Set when the budget could not afford scoring it. */
  unscored: boolean;
  /** The observation usage the proposers spent on it (reflection's own model calls). */
  searchUsageTokens: number;
};

export type RunRecord = {
  config: Config;
  candidates: Candidate[];
  byId: Map<string, Candidate>;
  iterations: IterationEndEvent[];
  observes: ObserveEvent[];
  evidences: EvidenceEvent[];
  fits: FitEvent[];
  resumes: ResumeEvent[];
  stop: StopEvent | null;
  finalists: string[] | null;
  picked: ValidationReport | null;
  test: TestReport | null;
  testFront: Record<string, TestReport> | null;
  testReason: string | null;
  state: Lifecycle;
  events: number;
};

const CORE: Record<string, string[]> = {
  config: ["config"],
  propose: ["candidate_id", "number", "parent_id", "origin", "rescore", "changed_keys"],
  trial: ["record"],
  score: ["candidate_id", "split", "objective_by_run", "case_names"],
  fit: ["candidates", "is_fitted"],
  label: ["candidate_id", "label"],
  iteration_end: ["record", "best_after"],
  finalists: ["candidate_ids"],
  picked: ["report"],
  test: ["report"],
  stop: ["reason"],
  resume: ["spent"],
};

export function parseRecord(text: string): RunRecord {
  const lines = text.split("\n");
  const events: { e: Event; line: number; killed?: boolean }[] = [];
  const resumes: ResumeEvent[] = [];
  let lastClosed = -1; // index into `events` of the last iteration_end (or stop)

  lines.forEach((raw, i) => {
    if (!raw.trim()) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      if (i === lines.length - 1) return; // the line being written
      throw new RecordError("not valid JSON", i + 1);
    }
    const e = parsed as Event & Record<string, unknown>;
    if (typeof e.event !== "string") throw new RecordError("an event without an event name", i + 1);
    if (!events.length) {
      if (e.event !== "config") throw new RecordError("the record does not start with a config event", i + 1);
      const v = (e as { config?: { schema_version?: unknown } }).config?.schema_version;
      if (v !== SCHEMA_VERSION) throw new RecordError(`run record schema_version ${JSON.stringify(v)}; this explorer reads ${SCHEMA_VERSION}`, i + 1);
    }
    const missing = (CORE[e.event] ?? []).filter((k) => !(k in e));
    if (missing.length) throw new RecordError(`${e.event} event without ${missing.join(", ")}`, i + 1);
    if (e.event === "resume") {
      /* The killed iteration is re-run, so its search events are dropped, as the optimizer's own reader
         drops them. Its trial events are not: those are the cache the re-run reads instead of paying for
         the rollouts again, so dropping one shows a trial the run did record as never run. They stay where
         they are in the file, which is what decides precedence when the re-run records the same trial too.
         The config event is kept whatever happens: a run killed inside its first iteration has no
         iteration_end to truncate to, and slicing it away leaves a record that starts with a propose. */
      resumes.push(e as ResumeEvent);
      const kept = events.splice(Math.max(lastClosed, 0) + 1).filter((k) => k.e.event === "trial");
      for (const k of kept) k.killed = true;
      events.push(...kept);
      return;
    }
    events.push({ e, line: i + 1 });
    if (e.event === "iteration_end" || e.event === "stop") lastClosed = events.length - 1;
  });
  if (!events.length) throw new RecordError("empty record");

  const config = (events[0].e as { config: Config }).config;
  const record: RunRecord = {
    config,
    candidates: [],
    byId: new Map(),
    iterations: [],
    observes: [],
    evidences: [],
    fits: [],
    resumes,
    stop: null,
    finalists: null,
    picked: null,
    test: null,
    testFront: null,
    testReason: null,
    state: "running",
    events: events.length + resumes.length,
  };

  const candidate = (id: string, line: number): Candidate => {
    const t = record.byId.get(id);
    if (!t) throw new RecordError(`candidate ${id} before its propose event`, line);
    return t;
  };

  const addTrial = (t: Candidate, r: TrialRecord) => {
    if (r.transcript && typeof r.transcript === "object") r.transcript = r.transcript.hash;
    // a re-run or a resume replaces the trial with the same key
    const at = t.trials.findIndex((x) => x.case_name === r.case_name && x.run_index === r.run_index && x.split === r.split);
    if (at >= 0) t.trials[at] = r;
    else t.trials.push(r);
  };

  /* Trials kept from a killed iteration whose propose went with the rest of it. The re-run proposes the
     same candidate again — same values, so the same id — and reads these from the cache rather than
     running them, so they are held until that propose arrives. */
  const held = new Map<string, TrialRecord[]>();
  const flushHeld = (t: Candidate) => {
    for (const r of held.get(t.id) ?? []) addTrial(t, r);
    held.delete(t.id);
  };

  for (const { e, line, killed } of events.slice(1)) {
    switch (e.event) {
      case "config":
        throw new RecordError("a second config event: one record holds one run", line);
      case "propose": {
        const p = e as ProposeEvent;
        const seen = record.byId.get(p.candidate_id);
        if (seen) {
          seen.rescored = seen.rescored || p.rescore;
          flushHeld(seen);
          break; // a rescore repeats the id; the first propose holds the only copy of the values
        }
        const t: Candidate = {
          id: p.candidate_id,
          number: p.number,
          parentId: p.parent_id,
          origin: p.origin,
          rationale: p.rationale ?? "",
          values: p.values ?? {},
          inherited: p.inherited ?? {},
          defaulted: p.defaulted ?? [],
          changedKeys: p.changed_keys ?? [],
          round: p.round ?? null,
          role: p.origin === "baseline" ? "baseline" : p.role === "opener" ? "opener" : "step",
          rescored: p.rescore,
          label: null,
          scores: {},
          trials: [],
          unscored: false,
          searchUsageTokens: 0,
        };
        record.byId.set(t.id, t);
        record.candidates.push(t);
        flushHeld(t);
        break;
      }
      case "trial": {
        const r = (e as { record: TrialRecord }).record;
        /* Only a trial kept from a killed iteration may name a candidate not proposed yet; anywhere else
           that is a record out of order. One whose candidate the re-run never proposes again is left
           held, and goes unshown with the rest of the iteration it belongs to. */
        if (killed && !record.byId.has(r.candidate_id)) {
          if (r.transcript && typeof r.transcript === "object") r.transcript = r.transcript.hash;
          held.set(r.candidate_id, [...(held.get(r.candidate_id) ?? []), r]);
          break;
        }
        addTrial(candidate(r.candidate_id, line), r);
        break;
      }
      case "score": {
        const s = e as ScoreEvent;
        candidate(s.candidate_id, line).scores[s.split] = s;
        break;
      }
      case "label": {
        const l = e as { candidate_id: string; label: Label };
        candidate(l.candidate_id, line).label = l.label;
        break;
      }
      case "fit":
        record.fits.push(e as FitEvent);
        break;
      case "observe": {
        const o = e as ObserveEvent;
        record.observes.push(o);
        const id = o.observation.candidate_id;
        if (id && record.byId.has(id)) {
          const t = record.byId.get(id)!;
          t.searchUsageTokens += o.observation.patch?.usage?.tokens ?? 0;
        }
        break;
      }
      case "evidence":
        record.evidences.push(e as EvidenceEvent);
        break;
      case "iteration_end":
        record.iterations.push(e as IterationEndEvent);
        break;
      case "unscored": {
        const u = e as { candidate_id: string };
        if (record.byId.has(u.candidate_id)) record.byId.get(u.candidate_id)!.unscored = true;
        break;
      }
      case "stop":
        record.stop = e as StopEvent;
        break;
      case "finalists":
        record.finalists = (e as { candidate_ids: string[] }).candidate_ids;
        break;
      case "picked":
        record.picked = (e as { report: ValidationReport }).report;
        break;
      case "test": {
        const t = e as { report: TestReport | null; front?: Record<string, TestReport> | null; reason?: string | null };
        record.test = t.report;
        record.testFront = t.front ?? null;
        record.testReason = t.reason ?? null;
        break;
      }
      // select, invalid, duplicate, proposer_retry, screened, proposer_error: context events the views
      // read through the iteration records; kept out of the model until a view needs them.
      default:
        break;
    }
  }

  record.state = record.stop === null ? "running" : record.stop.reason === "proposer_error" ? "stopped" : "finished";
  return record;
}

/** The per-metric measurements of one candidate, from its label (the read-only snapshot written for consumers). */
export function measuredOf(t: Candidate): Record<string, Measured> {
  return t.label?.measured ?? {};
}
