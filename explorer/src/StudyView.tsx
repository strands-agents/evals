// One study: its header and tabs, and the page the route names. The app shows it beside the sidebar; an embedded
// explorer shows it alone.

import { Component, type ReactNode } from "react";
import { StudyHeader } from "./components/StudyHeader";
import { HoverProvider } from "./interaction/hover";
import { useEffects } from "./record/effects";
import { useStudy } from "./record/load";
import { useRoute } from "./router";
import type { StudyRef } from "./source";
import { Trial } from "./views/Trial";
import { Overview } from "./views/Overview";
import { Progress } from "./views/Progress";
import { SearchSpace } from "./views/SearchSpace";
import { Setup } from "./views/Setup";
import { TradeOff } from "./views/TradeOff";
import { Candidate } from "./views/Candidate";
import { Candidates } from "./views/Candidates";

export class Boundary extends Component<{ children: ReactNode; route: string }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: { route: string }) {
    if (prev.route !== this.props.route && this.state.error) this.setState({ error: null });
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="card">
        <h2>This view failed</h2>
        <pre className="mono small" id="render-error">{String(this.state.error.stack ?? this.state.error)}</pre>
      </div>
    );
  }
}

export function StudyView({ studyRef }: { studyRef: StudyRef }) {
  const loaded = useStudy(studyRef);
  const route = useRoute();
  const effects = useEffects(studyRef, loaded?.ok ? loaded.study.record.events : 0);
  if (!loaded) return <div className="empty">loading…</div>;
  if (!loaded.ok)
    return (
      <div className="card">
        <h2>{studyRef.id}</h2>
        <pre className="mono small bad">{loaded.error}</pre>
      </div>
    );
  const study = loaded.study;
  return (
    <HoverProvider>
      <StudyHeader study={study} route={route} />
      <main>
        <Boundary route={JSON.stringify(route)}>
          {route.page === "overview" && <Overview study={study} effects={effects} />}
          {route.page === "progress" && <Progress study={study} />}
          {route.page === "tradeoff" && <TradeOff study={study} />}
          {route.page === "space" && <SearchSpace study={study} effects={effects} />}
          {route.page === "candidates" && <Candidates study={study} />}
          {route.page === "candidate" && <Candidate study={study} n={route.n} />}
          {route.page === "trial" && <Trial study={study} studyRef={studyRef} n={route.n} kase={route.case} rep={route.rep} />}
          {route.page === "setup" && <Setup study={study} />}
        </Boundary>
      </main>
    </HoverProvider>
  );
}
