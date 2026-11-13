import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Sidebar } from "./components/Sidebar";
import { useStudies } from "./record/load";
import { HashRouter, useRoute } from "./router";
import { canPick, fromDevServer, fromFiles, fromUiServer, lastFolder, openHandle, permitted, pickFolder, type Folder } from "./source";
import "./styles.css";
import { Boundary, StudyView } from "./StudyView";

const TOKEN = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("t") : null;

function App() {
  const [folder, setFolder] = useState<Folder | null>(null);
  const [reopen, setReopen] = useState<FileSystemDirectoryHandle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const route = useRoute();

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const token = q.get("t");
    if (token) {
      fromUiServer(token).then(setFolder).catch((e) => setError(String(e)));
      return;
    }
    const devRoot = q.get("devroot");
    if (import.meta.env.DEV && devRoot) {
      fromDevServer(devRoot, (q.get("devstudies") ?? "").split(",").filter(Boolean)).then(setFolder);
      return;
    }
    lastFolder().then(async (dir) => {
      if (!dir) return;
      if (await permitted(dir, false)) setFolder(await openHandle(dir));
      else setReopen(dir);
    });
  }, []);

  const pick = useCallback(async () => {
    try {
      setError(null);
      setFolder(await pickFolder());
      setReopen(null);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String(e));
    }
  }, []);
  const again = useCallback(async () => {
    if (reopen && (await permitted(reopen, true))) {
      setFolder(await openHandle(reopen));
      setReopen(null);
    }
  }, [reopen]);
  const files = useCallback((list: File[]) => setFolder(fromFiles(list)), []);

  const refs = folder?.studies ?? [];
  const loaded = useStudies(refs);
  const ref = refs.find((r) => r.id === route.study) ?? refs[0] ?? null;

  return (
    <>
      {TOKEN && (
        <nav className="evals-nav">
          <a className="brand" href={`/?t=${encodeURIComponent(TOKEN)}`}>
            <b>strands</b>-evals
          </a>
          <span className="tabs">
            <a href={`/?t=${encodeURIComponent(TOKEN)}`}>Runs</a>
            <a href={`/?t=${encodeURIComponent(TOKEN)}#tab=experiment`}>Experiment</a>
            <a href={`/?t=${encodeURIComponent(TOKEN)}#tab=report`}>Report</a>
            <span className="on">Optimization</span>
          </span>
        </nav>
      )}
      <div
      className="shell"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) files([...e.dataTransfer.files]);
      }}
    >
      <Sidebar folder={folder?.name ?? null} refs={refs} loaded={loaded} current={ref?.id ?? null} onPick={pick} onFiles={files} />
      <div className="content">
        {ref ? (
          <StudyView key={ref.id} studyRef={ref} />
        ) : (
          <div className="pick">
            {reopen ? (
              <button className="primary" onClick={again}>
                Reopen {reopen.name}
              </button>
            ) : folder ? (
              <span className="muted">No studies in {folder.name}.</span>
            ) : canPick ? (
              <button className="primary" onClick={pick}>
                Open a folder of studies
              </button>
            ) : (
              <span className="muted">Open a folder of studies from the sidebar, or drop one here.</span>
            )}
            {error && <div className="bad">{error}</div>}
          </div>
        )}
      </div>
    </div>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <Boundary route="app">
    <HashRouter>
      <App />
    </HashRouter>
  </Boundary>,
);
