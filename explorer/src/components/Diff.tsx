import type { ReactNode } from "react";

type Op<T> = { op: "same" | "add" | "del"; v: T };

/** The longest-common-subsequence edit script from a to b (fine for prompts: up to a few thousand tokens). */
function diff<T>(a: T[], b: T[]): Op<T>[] | null {
  const n = a.length;
  const m = b.length;
  if (n * m > 6_000_000) return null;
  const w = m + 1;
  const L = new Int32Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i * w + j] = a[i] === b[j] ? L[(i + 1) * w + j + 1] + 1 : Math.max(L[(i + 1) * w + j], L[i * w + j + 1]);
  const out: Op<T>[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ op: "same", v: a[i] });
      i++;
      j++;
    } else if (L[(i + 1) * w + j] >= L[i * w + j + 1]) out.push({ op: "del", v: a[i++] });
    else out.push({ op: "add", v: b[j++] });
  }
  while (i < n) out.push({ op: "del", v: a[i++] });
  while (j < m) out.push({ op: "add", v: b[j++] });
  return out;
}

/**
 * The change from one text to another, readably: a line diff (unchanged stretches folded) for multi-line text, a word
 * diff for short text. Removed text is red with "−", added text green with "+".
 */
export function TextDiff({ before, after }: { before: string; after: string }) {
  if (before === after) return <div className="muted small">unchanged</div>;
  const la = before.split("\n");
  const lb = after.split("\n");
  if (la.length <= 3 && lb.length <= 3) {
    const ops = diff(before.split(/(\s+)/), after.split(/(\s+)/));
    if (ops)
      return (
        <div className="words">
          {ops.map((o, k) => (o.op === "same" ? <span key={k}>{o.v}</span> : o.op === "add" ? <ins key={k}>{o.v}</ins> : <del key={k}>{o.v}</del>))}
        </div>
      );
  }
  const ops = diff(la, lb) ?? [...la.map((v) => ({ op: "del" as const, v })), ...lb.map((v) => ({ op: "add" as const, v }))];
  const rows: ReactNode[] = [];
  const CONTEXT = 2;
  let k = 0;
  while (k < ops.length) {
    if (ops[k].op !== "same") {
      rows.push(<div key={k} className={ops[k].op}>{ops[k].v || " "}</div>);
      k++;
      continue;
    }
    let e = k;
    while (e < ops.length && ops[e].op === "same") e++;
    const run = e - k;
    const head = k === 0 ? 0 : CONTEXT;
    const tail = e === ops.length ? 0 : CONTEXT;
    if (run > head + tail + 2) {
      for (let q = k; q < k + head; q++) rows.push(<div key={q} className="same">{ops[q].v || " "}</div>);
      rows.push(<div key={`gap${k}`} className="gap">… {run - head - tail} unchanged line{run - head - tail === 1 ? "" : "s"} …</div>);
      for (let q = e - tail; q < e; q++) rows.push(<div key={q} className="same">{ops[q].v || " "}</div>);
    } else for (let q = k; q < e; q++) rows.push(<div key={q} className="same">{ops[q].v || " "}</div>);
    k = e;
  }
  return <div className="diff">{rows}</div>;
}
