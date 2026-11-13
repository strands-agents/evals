import { format } from "d3-format";

const compact = format(".3~s");
const three = format(".3~r");

/** A number for people: integers with separators, others to 3 significant digits, very large ones compact (1.23M). */
export function num(v: unknown): string {
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "–";
  if (typeof v !== "number") return String(v);
  const a = Math.abs(v);
  if (a >= 1_000_000) return compact(v).replace("G", "B");
  if (a !== 0 && a < 0.001) return v.toExponential(1);
  if (Number.isInteger(v)) return v.toLocaleString();
  if (a >= 1000) return Math.round(v).toLocaleString();
  return three(v);
}

/** One format for every tick of an axis: compact (50k) when the axis reaches 10,000, else plain. */
export function axisFormat(ticks: number[]): (v: number) => string {
  const top = Math.max(...ticks.map(Math.abs), 0);
  if (top >= 10_000) return (v) => format("~s")(v).replace("G", "B");
  return (v) => num(v);
}

export function ci(c: [number, number] | null | undefined): string {
  if (!c) return "";
  return `[${num(c[0])} – ${num(c[1])}]`;
}

export function when(ts: string | number | null | undefined): string {
  if (ts === null || ts === undefined) return "–";
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function ago(iso: string): string {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 2 * 86400) return `${(s / 3600).toFixed(1)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

export function secs(s: number | null | undefined): string {
  if (s === null || s === undefined) return "–";
  if (s < 90) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

/** A record's status words, as written, made readable: "agent_failure" → "agent failure". */
export function words(s: string): string {
  return s.replaceAll("_", " ");
}
