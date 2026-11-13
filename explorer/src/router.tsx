// The route lives in the hash and includes the study, so switching studies never reloads the page (a folder of files
// that was dropped or selected stays loaded): #/s/<study>/<page>[/...]. An embedded explorer keeps its route in memory
// instead, so several on one page navigate independently. Views read the route and navigate through the router they are
// rendered in; their links are href()s under either.

import { createContext, useContext, useEffect, useLayoutEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";

export type Page =
  | { page: "overview" }
  | { page: "progress" }
  | { page: "tradeoff" }
  | { page: "space" }
  | { page: "candidates" }
  | { page: "candidate"; n: number }
  | { page: "trial"; n: number; case: string; rep: number }
  | { page: "setup" };

export type Route = { study: string | null } & Page;

const PAGES = new Set(["overview", "progress", "tradeoff", "space", "candidates", "setup"]);

function int(s: string | undefined): number | null {
  return s !== undefined && /^\d+$/.test(s) ? Number(s) : null;
}

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] !== "s" || !parts[1]) return { study: null, page: "overview" };
  const study = parts[1];
  const [page, a, b, c] = parts.slice(2);
  if (page === "candidate" && int(a) !== null) return { study, page: "candidate", n: int(a)! };
  if (page === "trial" && int(a) !== null && b !== undefined && int(c) !== null) return { study, page: "trial", n: int(a)!, case: b, rep: int(c)! };
  if (page && PAGES.has(page)) return { study, page: page as "overview" };
  return { study, page: "overview" };
}

export function href(study: string, p: Page): string {
  const base = `#/s/${encodeURIComponent(study)}`;
  if (p.page === "candidate") return `${base}/candidate/${p.n}`;
  if (p.page === "trial") return `${base}/trial/${p.n}/${encodeURIComponent(p.case)}/${p.rep}`;
  return `${base}/${p.page}`;
}

type Router = { route: Route; go: (study: string, p: Page) => void };

const RouterContext = createContext<Router | null>(null);

function useRouter(): Router {
  const router = useContext(RouterContext);
  if (!router) throw new Error("no router: render inside a HashRouter or a MemoryRouter");
  return router;
}

export function useRoute(): Route {
  return useRouter().route;
}

export function useGo(): Router["go"] {
  return useRouter().go;
}

function goHash(study: string, p: Page) {
  window.location.hash = href(study, p);
}

/** The app's router: the route in the window's hash. */
export function HashRouter({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => {
      const next = parse(window.location.hash);
      setRoute((prev) => {
        if (prev.study !== next.study || prev.page !== next.page) window.scrollTo(0, 0);
        return next;
      });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const router = useMemo(() => ({ route, go: goHash }), [route]);
  return <RouterContext.Provider value={router}>{children}</RouterContext.Provider>;
}

/** A route kept in memory: an embedded explorer's, which the page around it can navigate too. */
export type MemoryRoute = { get: () => Route; go: (study: string, p: Page) => void; subscribe: (on: () => void) => () => void };

export function memoryRoute(initial: Route): MemoryRoute {
  let route = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => route,
    go: (study, p) => {
      route = { ...p, study };
      for (const on of listeners) on();
    },
    subscribe: (on) => {
      listeners.add(on);
      return () => listeners.delete(on);
    },
  };
}

/** An embedded explorer's router: the route in ``memory``, for the explorer inside ``root`` (the element that scrolls).
 * A click on a link inside ``root`` routes here instead of changing the window's hash, and a new page scrolls ``root``,
 * not the window, to the top. */
export function MemoryRouter({ memory, root, children }: { memory: MemoryRoute; root: HTMLElement; children: ReactNode }) {
  const route = useSyncExternalStore(memory.subscribe, memory.get);
  useEffect(() => {
    // capturing, so a link inside an element that stops clicks (a table row that opens on click) is routed too
    const follow = (e: MouseEvent) => {
      const to = (e.target as Element).closest("a")?.getAttribute("href");
      if (!to?.startsWith("#/")) return;
      e.preventDefault();
      const next = parse(to);
      if (next.study !== null) memory.go(next.study, next);
    };
    root.addEventListener("click", follow, true);
    return () => root.removeEventListener("click", follow, true);
  }, [memory, root]);
  useLayoutEffect(() => {
    root.scrollTo(0, 0);
  }, [root, route.study, route.page]);
  const router = useMemo(() => ({ route, go: memory.go }), [route, memory]);
  return <RouterContext.Provider value={router}>{children}</RouterContext.Provider>;
}
