'use strict';

/* ── the address ──
   One path names what is open. Served, it is the URL's path; with no server it follows `#`, since a
   page opened from disk or a static host has only the one address. */

const OPT_SEGMENT = {
  'opt-setup': 'setup', 'opt-overview': 'overview', 'opt-progress': 'progress',
  'opt-candidates': 'candidates', 'opt-tunables': 'tunables', 'opt-trade': 'trade-off',
};
const OPT_VIEW = Object.fromEntries(Object.entries(OPT_SEGMENT).map(([view, segment]) => [segment, view]));

/* A run is named by its file or directory under the root, kept as path segments. The root itself is `.`. */
const idPath = id => id === '.' ? '' : '/' + String(id).split('/').map(encodeURIComponent).join('/');
const idOf = segments => segments.join('/') || '.';

/** The path for a view, given what is open in it. */
function routeFor(view) {
  const opt = OPT_RUNS[optSel];
  if (view === 'report' && repDisplaced && opt) {
    return `/optimization${idPath(opt.id)}/candidates/${encodeURIComponent(repDisplaced.cand)}/report`;
  }
  if (SECTION_OF[view] === 'optimization') {
    return opt && OPT_SEGMENT[view] ? `/optimization${idPath(opt.id)}/${OPT_SEGMENT[view]}` : '/optimization';
  }
  const run = RUNS[selRun];
  if (!run || (view !== 'experiment' && view !== 'report')) return '/runs';
  const name = view === 'experiment' && cases[sel] && cases[sel].name;
  return `/runs${idPath(run.id)}/${view}${name ? '/cases/' + encodeURIComponent(name) : ''}`;
}

/** What a path asks for: `{ tab, run, case, opt, cand }`, with only the parts it names. Anything
    it cannot read is the run list. */
function parseRoute(path) {
  const [root, ...rest] = path.split('/').filter(Boolean).map(segment => {
    try { return decodeURIComponent(segment); } catch { return segment; }
  });
  const last = rest[rest.length - 1];
  if (root === 'optimization') {
    if (rest.length >= 3 && last === 'report' && rest[rest.length - 3] === 'candidates') {
      return { tab: 'report', opt: idOf(rest.slice(0, -3)), cand: rest[rest.length - 2] };
    }
    return OPT_VIEW[last] ? { tab: OPT_VIEW[last], opt: idOf(rest.slice(0, -1)) } : { tab: 'opt-runs' };
  }
  if (root === 'runs' && rest.length > 1) {
    if (last === 'experiment' || last === 'report') return { tab: last, run: rest.slice(0, -1).join('/') };
    if (rest.length > 3 && rest[rest.length - 2] === 'cases' && rest[rest.length - 3] === 'experiment') {
      return { tab: 'experiment', run: rest.slice(0, -3).join('/'), case: last };
    }
  }
  return { tab: 'runs' };
}

const currentRoute = () => MODE === 'server' ? location.pathname : location.hash.slice(1);
const readAddress = () => parseRoute(currentRoute());

/* The route the page last wrote or followed, and whether it is following one now. */
let lastRoute = null, following = false;

/* A move between views or runs is a step Back can undo. `replace` corrects the address in place,
   as does anything written before the page is ready. */
function writeAddress(view, replace) {
  if (following) return;
  const route = routeFor(view);
  if (route === lastRoute) return;
  lastRoute = route;
  const push = !replace && app.dataset.ready;
  const url = MODE === 'server' ? route + location.search : `${location.pathname}${location.search}#${route}`;
  /* On file:// the origin is opaque and the History API throws SecurityError: fall back to the hash. */
  try {
    history[push ? 'pushState' : 'replaceState'](null, '', url);
  } catch {
    if (push) location.hash = route; else location.replace('#' + route);
  }
}
