'use strict';

/* ── the address ──
   One path names what is open. Served, it is the URL's path; with no server it follows `#`, since a
   page opened from disk or a static host has only the one address. */

const OPT_SEGMENT = {
  'opt-setup': 'setup', 'opt-overview': 'overview', 'opt-progress': 'progress',
  'opt-candidates': 'candidates', 'opt-tunables': 'tunables', 'opt-trade': 'trade-off',
};
const OPT_VIEW = Object.fromEntries(Object.entries(OPT_SEGMENT).map(([view, segment]) => [segment, view]));

/* A file or a run directory is named by its path under the root, kept as path segments. The root itself is `.`. */
const idPath = id => id === '.' ? '' : '/' + String(id).split('/').map(encodeURIComponent).join('/');
const idOf = segments => segments.join('/') || '.';

/* A comparison names its reports in the query, `?r=<path>&r=<path>`. Every other page is its path alone. */
const comparedQuery = paths => (paths.length ? '?' + paths.map(path => 'r=' + encodeURIComponent(path)).join('&') : '');
const comparedIn = query => new URLSearchParams(query).getAll('r');

/** The path for a view, given what is open in it. */
function routeFor(view) {
  const opt = OPT_RUNS[optSel];
  if (view === 'report' && repCand && opt) {
    return `/optimization${idPath(opt.id)}/candidates/${encodeURIComponent(repCand)}/report`;
  }
  if (SECTION_OF[view] === 'optimization') {
    return opt && OPT_SEGMENT[view] ? `/optimization${idPath(opt.id)}/${OPT_SEGMENT[view]}` : '/optimization';
  }
  if (view === 'report' && repMeta) return `/reports${idPath(repMeta.path)}`;
  if (view === 'compare') return '/reports/compare' + comparedQuery(CMP.map(c => c.meta.path));
  if (view === 'experiment' && expMeta) {
    const name = cases[sel] && cases[sel].name;
    return `/experiments${idPath(expMeta.path)}${name ? '/cases/' + encodeURIComponent(name) : ''}`;
  }
  return view === 'experiments' ? '/experiments' : '/reports';
}

/** What a path asks for: `{ tab, report, compare, experiment, case, opt, cand }`, with only the parts it names.
    A path that names nothing leaves `tab` out, for the page to choose its own first view. */
function parseRoute(route) {
  const [path, query = ''] = route.split('?');
  if (path.replace(/\/+$/, '') === '/reports/compare') return { tab: 'compare', compare: comparedIn(query) };
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
  if (root === 'reports') return rest.length ? { tab: 'report', report: rest.join('/') } : { tab: 'reports' };
  if (root === 'experiments') {
    if (!rest.length) return { tab: 'experiments' };
    /* A file's path ends in `.json`, so `cases` after one can only be the view of a single case. */
    const named = rest.length > 2 && rest[rest.length - 2] === 'cases' && /\.json$/i.test(rest[rest.length - 3]);
    return named ? { tab: 'experiment', experiment: rest.slice(0, -2).join('/'), case: last }
      : { tab: 'experiment', experiment: rest.join('/') };
  }
  return {};
}

const currentRoute = () => MODE === 'server'
  ? location.pathname + comparedQuery(comparedIn(location.search)) : location.hash.slice(1);
const readAddress = () => parseRoute(currentRoute());

/* The route the page last wrote or followed, and whether it is following one now. */
let lastRoute = null, following = false;

/* A move between pages is a step Back can undo. `replace` corrects the address in place,
   as does anything written before the page is ready. */
function writeAddress(view, replace) {
  if (following) return;
  const route = routeFor(view);
  if (route === lastRoute) return;
  lastRoute = route;
  const push = !replace && app.dataset.ready;
  /* Whatever else the address carries stays; the reports of a comparison left behind do not. */
  const others = new URLSearchParams(location.search);
  others.delete('r');
  const kept = others.toString() && (route.includes('?') ? '&' : '?') + others;
  const url = MODE === 'server' ? route + kept : `${location.pathname}${location.search}#${route}`;
  /* On file:// the origin is opaque and the History API throws SecurityError: fall back to the hash. */
  try {
    history[push ? 'pushState' : 'replaceState'](null, '', url);
  } catch {
    if (push) location.hash = route; else location.replace('#' + route);
  }
}
