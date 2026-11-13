// effects.json, written next to a run's record by strands_evals/optimization/effects.py: the effect of each option
// on each goal metric, from an effects model fitted to the record.

import { useEffect, useState } from "react";
import type { StudyRef } from "../source";

export type EffectsFactor = { key: string; kind: string; levels: string[]; reference: string };
/** One level's change from its variable's reference level, for one metric. */
export type Effect = { metric: string; key: string; level: string; reference: string; mean: number; ci90: [number, number]; p_positive: number };
export type EffectsDiagnostics = { n_holdout: number; z_mean: number; z_sq_sum: number; flagged: boolean; messages: string[] };
export type MetricEffects = {
  scale: { unit: number; offset: number };
  noise: number;
  observations: number;
  diagnostics: EffectsDiagnostics;
  [detail: string]: unknown;
};
export type EffectsGoal = { role: "objective" | "constraint"; index: number; metric: string; better: "higher" | "lower" };
export type Effects = {
  effects: Effect[];
  factors: EffectsFactor[];
  omitted: { key: string; why: string }[];
  goals: EffectsGoal[];
  metrics: Record<string, MetricEffects>;
  /** Why `effects` is empty, when it is. */
  reason: string | null;
};

/** The run's effects.json, or null when it has none. Re-read when the record grows (`events`). */
export function useEffects(ref: StudyRef | null, events: number): Effects | null {
  const [state, setState] = useState<Effects | null>(null);
  useEffect(() => {
    setState(null);
    if (!ref) return;
    ref.read("effects.json").then((text) => {
      if (text === null) return;
      try {
        const e = JSON.parse(text) as Effects;
        if (Array.isArray(e.effects) && Array.isArray(e.goals) && e.metrics) setState(e);
      } catch {
        // not an effects file (a dev server's fallback page): no effects
      }
    });
  }, [ref, events]);
  return state;
}

/** The probability that a level is better than the reference for a goal. */
export function pBetter(e: Effect, better: "higher" | "lower"): number {
  return better === "higher" ? e.p_positive : 1 - e.p_positive;
}
