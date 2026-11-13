// Studies on disk, read in the browser: a folder the user picks (File System Access API, remembered between visits),
// or a folder selected or dropped as files (other browsers; a snapshot). A run is a directory with an events.jsonl.
// An embedded explorer reads one study the page holds instead.

export type StudyRef = {
  id: string; // the study directory's path inside the picked folder
  name: string;
  read(file: string): Promise<string | null>; // "study.jsonl" or "transcripts/<hash>.json"
  modified(): Promise<number>; // ms since the epoch, of study.jsonl
  live: boolean; // re-reading gives fresh contents (a picked folder) rather than a snapshot
};

export type Folder = { name: string; studies: StudyRef[]; live: boolean };

export const canPick = typeof window !== "undefined" && "showDirectoryPicker" in window;

// ---- a picked folder ------------------------------------------------------------------------------------------------
type Dir = FileSystemDirectoryHandle;

async function child(dir: Dir, path: string): Promise<File | null> {
  const parts = path.split("/");
  let d = dir;
  try {
    for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p);
    return await (await d.getFileHandle(parts[parts.length - 1])).getFile();
  } catch {
    return null;
  }
}

async function scan(dir: Dir, prefix: string, depth: number, out: StudyRef[]): Promise<void> {
  if (await child(dir, "events.jsonl")) {
    out.push({
      id: prefix || dir.name,
      name: dir.name,
      live: true,
      read: async (file) => (await child(dir, file))?.text() ?? null,
      modified: async () => (await child(dir, "events.jsonl"))?.lastModified ?? 0,
    });
    return;
  }
  if (depth === 0) return;
  for await (const entry of (dir as any).values() as AsyncIterable<FileSystemHandle>) {
    if (entry.kind === "directory" && !entry.name.startsWith(".") && entry.name !== "transcripts")
      await scan(entry as Dir, prefix ? `${prefix}/${entry.name}` : entry.name, depth - 1, out);
  }
}

export async function openHandle(dir: Dir): Promise<Folder> {
  const studies: StudyRef[] = [];
  await scan(dir, "", 3, studies);
  return { name: dir.name, studies, live: true };
}

export async function pickFolder(): Promise<Folder> {
  const dir: Dir = await (window as any).showDirectoryPicker({ id: "studies", mode: "read" });
  await remember(dir);
  return openHandle(dir);
}

// ---- the folder picked last time (IndexedDB keeps the handle; the browser asks again for permission) ---------------
const DB = "study-explorer";

function idb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore("handles");
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function remember(dir: Dir): Promise<void> {
  const db = await idb();
  await new Promise<void>((resolve) => {
    const tx = db.transaction("handles", "readwrite");
    tx.objectStore("handles").put(dir, "folder");
    tx.oncomplete = () => resolve();
  });
}

export async function lastFolder(): Promise<Dir | null> {
  if (!canPick) return null;
  try {
    const db = await idb();
    return await new Promise((resolve) => {
      const r = db.transaction("handles").objectStore("handles").get("folder");
      r.onsuccess = () => resolve((r.result as Dir) ?? null);
      r.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

export async function permitted(dir: Dir, ask: boolean): Promise<boolean> {
  const h = dir as any;
  if ((await h.queryPermission({ mode: "read" })) === "granted") return true;
  return ask && (await h.requestPermission({ mode: "read" })) === "granted";
}

// ---- files selected or dropped (a snapshot) -------------------------------------------------------------------------
export function fromFiles(files: File[]): Folder {
  const byPath = new Map<string, File>();
  for (const f of files) byPath.set((f as any).webkitRelativePath || f.name, f);
  const studies: StudyRef[] = [];
  for (const [path, file] of byPath) {
    if (!path.endsWith("events.jsonl")) continue;
    const dir = path.slice(0, -"events.jsonl".length).replace(/\/$/, "");
    const id = dir.split("/").slice(1).join("/") || dir;
    studies.push({
      id,
      name: dir.split("/").pop() || dir,
      live: false,
      read: async (name) => byPath.get(dir ? `${dir}/${name}` : name)?.text() ?? null,
      modified: async () => file.lastModified,
    });
  }
  const root = (files[0] as any)?.webkitRelativePath?.split("/")[0] ?? "files";
  return { name: root, studies, live: false };
}

// ---- a study the page holds (an embedded explorer) ------------------------------------------------------------------
/** A study as a page embeds it: its files' contents. The record and each transcript are text, or gzipped and
 * base64-encoded (``recordGz``, ``transcriptsGz``; a transcript is inflated when its trial is opened). */
export type EmbeddedStudy = {
  id: string;
  record?: string; // events.jsonl
  recordGz?: string;
  effects?: unknown | null; // effects.json, parsed
  transcripts?: Record<string, string>; // hash -> transcripts/<hash>.json
  transcriptsGz?: Record<string, string>;
};

/** Text that was gzipped and base64-encoded. */
export async function inflate(base64Gzip: string): Promise<string> {
  const bytes = Uint8Array.from(atob(base64Gzip), (c) => c.charCodeAt(0));
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
}

export function fromEmbedded(data: EmbeddedStudy): StudyRef {
  const text = (plain: string | undefined, gz: string | undefined) => (plain !== undefined ? plain : gz !== undefined ? inflate(gz) : null);
  return {
    id: data.id,
    name: data.id.split("/").pop() || data.id,
    live: false,
    read: async (file) => {
      if (file === "events.jsonl") return text(data.record, data.recordGz);
      if (file === "effects.json") return data.effects == null ? null : JSON.stringify(data.effects);
      const hash = /^transcripts\/(.+)\.json$/.exec(file)?.[1];
      return hash === undefined ? null : text(data.transcripts?.[hash], data.transcriptsGz?.[hash]);
    },
    modified: async () => 0,
  };
}

// ---- runs served by `strands-evals ui` (the page lives at /optimization/ on the same origin) ------------------------
export async function fromUiServer(token: string): Promise<Folder> {
  const q = `?t=${encodeURIComponent(token)}`;
  const listed = await fetch(`/api/optimization/runs${q}`, { cache: "no-store" });
  if (!listed.ok) throw new Error(`the server refused the runs list (${listed.status})`);
  const { root, runs } = (await listed.json()) as { root: string; runs: { id: string }[] };
  const studies: StudyRef[] = runs.map(({ id }) => ({
    id,
    name: id,
    live: true,
    read: async (file) => {
      const r = await fetch(`/api/optimization/file${q}&run=${encodeURIComponent(id)}&path=${encodeURIComponent(file)}`, { cache: "no-store" });
      return r.ok ? r.text() : null;
    },
    modified: async () => Date.now(),
  }));
  return { name: root.split("/").pop() || root, studies, live: true };
}

// ---- development only: studies served by the Vite dev server (used for headless checks) ---------------------------
export async function fromDevServer(root: string, ids: string[]): Promise<Folder> {
  const studies: StudyRef[] = ids.map((id) => {
    const base = `/@fs${root}/${id}`;
    return {
      id,
      name: id.split("/").pop() || id,
      live: true,
      read: async (file) => {
        const r = await fetch(`${base}/${file}`, { cache: "no-store" });
        return r.ok ? r.text() : null;
      },
      modified: async () => Date.now(),
    };
  });
  return { name: root.split("/").pop() || root, studies, live: true };
}
