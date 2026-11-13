// The run record (strands_evals.optimization, run.json schema_version 1), exactly as the optimizer writes it:
// a run directory holding run.json, an append-only events.jsonl (one JSON object per line, the only source of
// truth), transcripts/<hash>.json, effects.json, and the review files. This file is the explorer's read contract
// for that record.

export const SCHEMA_VERSION = 1;

// ---- run.json (also the first event's `config`) --------------------------------------------------------------------

export type MetricDesc = {
  kind: "evaluator" | "usage";
  roles: ("objective" | "constraint")[];
  range: [number | null, number | null];
  weight: number;
  scale: number | null;
};

/** One declared tunable. `default` for text kinds is a fingerprint (sha256 + length), never the text itself. */
export type SpaceRow = {
  key: string;
  name: string;
  kind: "text" | "int" | "float" | "one_of" | "choice" | "skills" | string;
  default: unknown;
  min_value: number | null;
  max_value: number | null;
  log: boolean;
  options: unknown[] | null;
  size_cap: number | null;
  edit_instructions: string | null;
  /** Written as a two-element [key, option] array; StudyModel normalizes it to the object form. */
  active_when: { key: string; option: string } | [string, string] | null;
  source?: { path: string; pointer: string | null; as_path: boolean } | null;
  [detail: string]: unknown;
};

export type BlockDesc = { name: string; keys: string[]; proposer: string; config: Record<string, unknown> };
export type ScheduleDesc = { name: string; config: Record<string, unknown> };
export type SearchDesc = { blocks: BlockDesc[]; schedule: ScheduleDesc | null; redraws?: number };

export type SplitDesc = {
  settings: Record<string, unknown>;
  seed: number | null;
  train: string[];
  validation: string[];
  test: string[];
};

export type Config = {
  schema_version: number;
  name: string | null;
  agent: Record<string, string | null>;
  workdir: string;
  code: { version: string | null; commit: string | null; dirty: boolean | null };
  /** Human-readable, e.g. "WeightedSum(Maximize(ExactAnswer), Maximize(Helpfulness))". Directions are parsed from it. */
  objective: string;
  /** Human-readable, e.g. "tokens <= 600". Metric, operator, and limit are parsed from each. */
  constraints: string[];
  metrics: Record<string, MetricDesc>;
  space: SpaceRow[];
  search: SearchDesc;
  split: SplitDesc;
  budget: { max_rollouts: number; max_iterations: number | null };
  final_pick: { n_finalists: number; rule: { name: string; config: Record<string, unknown> } };
  selection: { name: string; config: Record<string, unknown> };
  score_model: { name: string; config: Record<string, unknown> };
  runs_per_candidate: number;
  proposals_per_iteration: number;
  rerun_passes: number;
  feasibility_threshold: number;
  agent_failure_score: string | number | null;
  seed: number;
  max_workers: number;
};

// ---- shared pieces ---------------------------------------------------------------------------------------------------

export type Usage = {
  input_tokens: number | null;
  output_tokens: number | null;
  tokens: number | null;
  cache_read_tokens?: number | null;
  cache_write_tokens?: number | null;
  model_calls?: number | null;
  seconds?: number | null;
  models?: string[] | null;
};

/** The score model's belief about one metric. */
export type Estimate = { mean: number | null; sd: number | null };
/** What was actually measured for one metric: the per-case mean, its 95% interval, and the cases behind it. */
export type Measured = { mean: number | null; ci95: [number, number] | null; n: number };

export type Outcome = "feasible" | "infeasible" | "ineligible" | "invalid" | "unscored" | "duplicate";

export type Label = {
  outcome: "feasible" | "infeasible" | "ineligible";
  reason: string;
  unmet_constraints: string[];
  estimates: Record<string, Estimate>;
  p_met: Record<string, number>;
  is_from_fit: boolean;
  measured: Record<string, Measured>;
};

export type TrialStatus = "completed" | "agent_failure" | "infrastructure_invalid" | "evaluator_invalid";
export type Split = "train" | "validation" | "test";

export type Evaluation = { score: number | null; test_pass?: boolean | null; reason?: string | null; label?: string | null };

export type TrialRecord = {
  candidate_id: string;
  case_name: string;
  run_index: number;
  split: Split;
  status: TrialStatus;
  output: string | null;
  error: string | null;
  error_type: string | null;
  /** Per evaluator, that evaluator's outputs for the trial (score, test_pass, reason, label); a lone object when
   * the evaluator returned one row. */
  evaluations: Record<string, Evaluation | Evaluation[]>;
  evaluator_errors: Record<string, string>;
  details?: unknown;
  usage: Usage;
  judge_usage: Record<string, Usage>;
  rollouts: number;
  attempts: number;
  previous_attempts: unknown[];
  /** Written as { hash, format }; parse keeps only the hash (the transcripts/<hash>.json path needs nothing else). */
  transcript: string | { hash: string; format?: string } | null;
  interactions?: unknown;
  environment_state?: unknown;
};

export type Patch = { updates: Record<string, unknown>; rationale: string; state: unknown; usage: Usage | null };

export type PairedComparison = { mean_diff: number; ci_low: number; ci_high: number; n_cases: number };

export type ValidationReport = {
  champion_id: string;
  front: string[];
  objectives: Record<string, number>;
  estimates: Record<string, Record<string, Estimate>>;
  infeasible: Record<string, string>;
  measured: Record<string, Record<string, Measured>>;
  problem: string | null;
  reason: string;
};

export type TestReport = {
  champion_id: string;
  baseline_id: string;
  objective: PairedComparison;
  difference: Record<string, PairedComparison>;
  ratio: Record<string, PairedComparison>;
  missing?: Record<string, number>;
  means?: Record<string, Record<string, Measured>>;
};

// ---- events.jsonl ----------------------------------------------------------------------------------------------------

type Base = { event: string; iteration: number; time: string };

export type ConfigEvent = Base & { event: "config"; config: Config };

export type ProposeEvent = Base & {
  event: "propose";
  candidate_id: string;
  number: number;
  parent_id: string | null;
  /** "baseline" or the proposing block's name. */
  origin: string;
  rationale: string;
  /** The candidate's full values; absent on a rescore (the first propose holds the only copy). */
  values?: Record<string, unknown>;
  inherited: Record<string, unknown>;
  defaulted: string[];
  rescore: boolean;
  changed_keys: string[];
  /** The Nested round path, outermost first, or null outside a round. */
  round: number[] | null;
  role: "opener" | null;
  patch?: Patch;
};

export type TrialEvent = Base & { event: "trial"; record: TrialRecord };

export type ScoreEvent = Base & {
  event: "score";
  candidate_id: string;
  split: Split;
  objective_by_run: Record<string, number[]>;
  statuses: Partial<Record<TrialStatus, number>>;
  unavailable: Record<string, unknown>;
  runs: number;
  rollouts: number;
  case_names: string[];
};

export type FitEvent = Base & {
  event: "fit";
  candidates: string[];
  is_fitted: boolean;
  flagged: string[];
  error: string | null;
  diagnostics: Record<string, unknown>;
};

export type LabelEvent = Base & { event: "label"; candidate_id: string; label: Label };

export type SelectEvent = Base & { event: "select"; parent_id: string; strategy: string };

export type Observation = {
  patch: Patch;
  candidate_id: string | null;
  values: Record<string, unknown> | null;
  source_id?: string | null;
  objective?: number | null;
  violations?: (number | null)[] | null;
  term_values?: number[] | null;
  outcome?: Outcome | null;
  reason?: string | null;
};

export type ObserveEvent = Base & { event: "observe"; block: string; observation: Observation };

export type Evidence = {
  candidate_id: string;
  values: Record<string, unknown>;
  origin: string;
  objective: number | null;
  term_values: number[] | null;
  violations: (number | null)[] | null;
  outcome: Outcome;
};

export type EvidenceEvent = Base & { event: "evidence"; block: string; evidence: Evidence };

export type ProposalOutcome = {
  candidate_id: string | null;
  number: number | null;
  parent_id: string | null;
  origin: string;
  changed_keys: string[];
  rationale: string;
  outcome: Outcome;
  objective: number | null;
};

export type IterationRecord = {
  iteration: number;
  parent_id: string | null;
  round: number[] | null;
  role: "opener" | null;
  blocks: string[];
  rollouts_spent: number;
  proposals: ProposalOutcome[];
};

export type IterationEndEvent = Base & {
  event: "iteration_end";
  record: IterationRecord;
  best_after: string[];
  exhausted: boolean | string[];
};

export type InvalidEvent = Base & { event: "invalid"; block: string; redraw: number; reason: string; updates: Record<string, unknown> };
export type DuplicateEvent = Base & { event: "duplicate"; block: string; redraw: number; candidate_id: string };
export type UnscoredEvent = Base & { event: "unscored"; candidate_id: string; pending: number; remaining: number; reserve: number };
export type ProposerRetryEvent = Base & { event: "proposer_retry"; block: string; redraw: number; error: string; error_type: string };
export type ScreenedEvent = Base & { event: "screened"; block: string; key: string; reason: string };
export type ProposerErrorEvent = Base & { event: "proposer_error"; block: string; error: string; error_type?: string };

export type ResumeEvent = Base & { event: "resume"; spent: number; budget: Config["budget"]; code: Config["code"] };
export type StopEvent = Base & { event: "stop"; reason: "budget" | "iterations" | "exhausted" | "proposer_error" | "setup"; message: string };

export type FinalistsEvent = Base & { event: "finalists"; candidate_ids: string[] };
export type PickedEvent = Base & { event: "picked"; report: ValidationReport };
export type TestEvent = Base & { event: "test"; report: TestReport | null; front: Record<string, TestReport> | null; reason: string | null };

export type Event =
  | ConfigEvent
  | ProposeEvent
  | TrialEvent
  | ScoreEvent
  | FitEvent
  | LabelEvent
  | SelectEvent
  | ObserveEvent
  | EvidenceEvent
  | IterationEndEvent
  | InvalidEvent
  | DuplicateEvent
  | UnscoredEvent
  | ProposerRetryEvent
  | ScreenedEvent
  | ProposerErrorEvent
  | ResumeEvent
  | StopEvent
  | FinalistsEvent
  | PickedEvent
  | TestEvent;

/** A transcript: the trial's stored trajectory (a strands-evals Session, or the rollout's own list). */
export type Transcript = unknown;
