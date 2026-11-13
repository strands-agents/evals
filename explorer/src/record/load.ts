// Loading records from a folder: each run parsed once, re-read only while it runs (and only when the file grew).

import { useEffect, useState } from "react";
import { StudyModel } from "../model/study";
import type { StudyRef } from "../source";
import { parseRecord, RecordError } from "./parse";
import type { Transcript } from "./schema";

export type Loaded = { ok: true; study: StudyModel } | { ok: false; error: string };

async function load(ref: StudyRef): Promise<{ text: string; loaded: Loaded } | null> {
  const text = await ref.read("events.jsonl");
  if (text === null) return null;
  try {
    return { text, loaded: { ok: true, study: new StudyModel(parseRecord(text), ref.id, ref.name) } };
  } catch (e) {
    return { text, loaded: { ok: false, error: e instanceof RecordError || e instanceof Error ? e.message : String(e) } };
  }
}

/** One study, kept fresh every few seconds while it runs. */
export function useStudy(ref: StudyRef | null): Loaded | null {
  const [state, setState] = useState<Loaded | null>(null);
  useEffect(() => {
    setState(null);
    if (!ref) return;
    let alive = true;
    let size = -1;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const got = await load(ref);
      if (!alive) return;
      if (got && got.text.length !== size) {
        size = got.text.length;
        setState(got.loaded);
      }
      const running = got?.loaded.ok && got.loaded.study.state === "running";
      if (ref.live && running) timer = setTimeout(tick, 5000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [ref]);
  return state;
}

/** Every study in the folder, for the sidebar; the running ones re-read every 15 s. */
export function useStudies(refs: StudyRef[]): Map<string, Loaded> {
  const [all, setAll] = useState<Map<string, Loaded>>(new Map());
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const sizes = new Map<string, number>();
    const running = new Set(refs.map((r) => r.id));
    const tick = async () => {
      const next = await Promise.all(refs.filter((r) => running.has(r.id)).map(async (r) => [r, await load(r)] as const));
      if (!alive) return;
      setAll((prev) => {
        const out = new Map(prev);
        for (const [r, got] of next) {
          if (!got) continue;
          if (got.text.length !== sizes.get(r.id)) out.set(r.id, got.loaded);
          sizes.set(r.id, got.text.length);
          if (!(got.loaded.ok && got.loaded.study.state === "running") || !r.live) running.delete(r.id);
        }
        return out;
      });
      if (running.size) timer = setTimeout(tick, 15000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [refs]);
  return all;
}

export function useTranscript(ref: StudyRef | null, hash: string | null): Transcript | null | "missing" {
  const [state, setState] = useState<Transcript | null | "missing">(null);
  useEffect(() => {
    setState(null);
    if (!ref || !hash) return;
    ref.read(`transcripts/${hash}.json`).then((t) => setState(t === null ? "missing" : (JSON.parse(t) as Transcript)));
  }, [ref, hash]);
  return state;
}
