// The runs in the open folder, grouped by the folder they sit in, each with its state from the record.

import { useRef } from "react";
import { arrow } from "../model/goals";
import type { Loaded } from "../record/load";
import { href } from "../router";
import { canPick, type StudyRef } from "../source";

function group(refs: StudyRef[]): [string, StudyRef[]][] {
  const out = new Map<string, StudyRef[]>();
  for (const r of [...refs].sort((a, b) => a.id.localeCompare(b.id))) {
    const dir = r.id.includes("/") ? r.id.slice(0, r.id.lastIndexOf("/")) : "";
    out.set(dir, [...(out.get(dir) ?? []), r]);
  }
  return [...out.entries()];
}

function Item({ r, loaded, current }: { r: StudyRef; loaded: Loaded | undefined; current: boolean }) {
  const study = loaded?.ok ? loaded.study : null;
  const state = study?.state ?? (loaded && !loaded.ok ? "error" : "loading");
  return (
    <a className={`study-item${current ? " on" : ""}`} href={href(r.id, { page: "overview" })} title={r.id}>
      <span className="study-name">
        <i className={`status-dot ${state}`} title={state} />
        {r.name.replace(/([-+])/g, "$1\u200b")}
      </span>
      <span className="study-meta">
        {study && (
          <>
            {study.candidates.length} candidates · {study.h.objectives.map((o) => `${arrow(o)}${o.metric}`).join(" ")}
            {study.h.constraints.length > 0 && ` · ${study.h.constraints.length} constraint${study.h.constraints.length > 1 ? "s" : ""}`}
          </>
        )}
        {loaded && !loaded.ok && <span className="bad">{loaded.error}</span>}
      </span>
    </a>
  );
}

export function Sidebar({ folder, refs, loaded, current, onPick, onFiles }: {
  folder: string | null;
  refs: StudyRef[];
  loaded: Map<string, Loaded>;
  current: string | null;
  onPick: () => void;
  onFiles: (files: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const open = () => (canPick ? onPick() : input.current?.click());
  return (
    <aside className="sidebar">
      <button className="open-folder" onClick={open} title="Open a folder of studies">
        {folder ?? "Open folder…"}
      </button>
      <input ref={input} type="file" hidden multiple {...({ webkitdirectory: "", directory: "" } as object)} onChange={(e) => e.target.files && onFiles([...e.target.files])} />
      <nav>
        {group(refs).map(([dir, items]) => (
          <div key={dir} className="study-group">
            {dir && <div className="group-name">{dir}</div>}
            {items.map((r) => (
              <Item key={r.id} r={r} loaded={loaded.get(r.id)} current={r.id === current} />
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
