'use strict';

/* ── the address ──
   The URL's path names what is open. */

const OPT_SEGMENT = {
  'opt-setup': 'setup', 'opt-overview': 'result', 'opt-progress': 'iterations',
  'opt-candidates': 'candidates', 'opt-trade': 'trade-off',
};
const OPT_VIEW = Object.fromEntries(Object.entries(OPT_SEGMENT).map(([view, segment]) => [segment, view]));

/* A file or a run directory is named by its path under the root, kept as path segments. The root itself is `.`. */
const idPath = id => id === '.' ? '' : '/' + String(id).split('/').map(encodeURIComponent).join('/');
const idOf = segments => segments.join('/') || '.';
/* A report row is named by its case and its evaluator, the pair that tells it from every other row. */
const rowPath = r => (r ? `/rows/${encodeURIComponent(r.name)}${r.evaluator == null ? '' : '/' + encodeURIComponent(r.evaluator)}` : '');

/** The path for a view, given what is open in it. */
function routeFor(view) {
  if (view === 'missing') return MISSING.route;
  const opt = OPT_RUNS[optSel];
  if (view === 'report' && repCand && opt) {
    return `/optimization${idPath(opt.id)}/candidates/${encodeURIComponent(repCand)}/report`;
  }
  if (SECTION_OF[view] === 'optimization') {
    /* The candidate opened last, and the report row opened last below, are the ones the address names. */
    const cand = view === 'opt-candidates' && OPT && [...optOpen].reverse().find(id => OPT.proposals.some(p => p.candidate_id === id));
    return opt && OPT_SEGMENT[view] ? `/optimization${idPath(opt.id)}/${OPT_SEGMENT[view]}${cand ? '/' + encodeURIComponent(cand) : ''}` : '/optimization';
  }
  if (view === 'report' && repMeta) return `/reports${idPath(repMeta.path)}${rowPath(ROWS[[...openRows].pop()])}`;
  if (view === 'experiment' && expMeta) {
    const name = cases[sel] && cases[sel].name;
    return `/experiments${idPath(expMeta.path)}${name ? '/cases/' + encodeURIComponent(name) : ''}`;
  }
  if (view === 'session' && sesMeta) return `/sessions${idPath(sesMeta.path)}`;
  return view === 'experiments' ? '/experiments' : view === 'sessions' ? '/sessions' : '/reports';
}

/** What a path asks for: `{ tab, report, row, evaluator, experiment, case, session, opt, cand, candidate }`,
    with only the parts it names.
    A path that names nothing leaves `tab` out, for the page to choose its own first view; one that
    names no page the app has gives `missing`. */
function parseRoute(route) {
  const [path] = route.split('?');
  const [root, ...rest] = path.split('/').filter(Boolean).map(segment => {
    try { return decodeURIComponent(segment); } catch { return segment; }
  });
  const last = rest[rest.length - 1];
  if (root === 'optimization') {
    if (rest.length >= 3 && last === 'report' && rest[rest.length - 3] === 'candidates') {
      return { tab: 'report', opt: idOf(rest.slice(0, -3)), cand: rest[rest.length - 2] };
    }
    if (OPT_VIEW[last]) return { tab: OPT_VIEW[last], opt: idOf(rest.slice(0, -1)) };
    if (rest.length >= 2 && rest[rest.length - 2] === 'candidates') {
      return { tab: 'opt-candidates', opt: idOf(rest.slice(0, -2)), candidate: last };
    }
    /* A run directory with no view named opens on its Result view. */
    return rest.length ? { tab: 'opt-overview', opt: idOf(rest) } : { tab: 'opt-runs' };
  }
  if (root === 'reports') {
    /* As with a case below: `rows` after a file's `.json` can only be one row of it, with or without an evaluator. */
    const at = [2, 3].find(n => rest.length > n && rest[rest.length - n] === 'rows' && /\.json$/i.test(rest[rest.length - n - 1]));
    if (at) return { tab: 'report', report: rest.slice(0, -at).join('/'), row: rest[rest.length - at + 1], evaluator: rest[rest.length - at + 2] };
    return rest.length ? { tab: 'report', report: rest.join('/') } : { tab: 'reports' };
  }
  if (root === 'sessions') return rest.length ? { tab: 'session', session: rest.join('/') } : { tab: 'sessions' };
  if (root === 'experiments') {
    if (!rest.length) return { tab: 'experiments' };
    /* A file's path ends in `.json`, so `cases` after one can only be the view of a single case. */
    const named = rest.length > 2 && rest[rest.length - 2] === 'cases' && /\.json$/i.test(rest[rest.length - 3]);
    return named ? { tab: 'experiment', experiment: rest.slice(0, -2).join('/'), case: last }
      : { tab: 'experiment', experiment: rest.join('/') };
  }
  /* `/index.html` is the page itself. Anything else names no page. */
  return !root || root === 'index.html' ? {} : { tab: 'missing' };
}

const currentRoute = () => location.pathname;
const readAddress = () => parseRoute(currentRoute());

/* The route the page last wrote or followed, and whether it is following one now. */
let lastRoute = null, following = false;

/* A move between pages is a step Back can undo. `replace` corrects the address in place,
   as does anything written before the page is ready. */
function writeAddress(view, replace) {
  if (following) return;
  const route = routeFor(view);
  if (route === lastRoute) return;
  const push = !replace && app.dataset.ready;
  /* Whatever else the address carries stays; the reports of a comparison left behind do not. */
  const others = new URLSearchParams(location.search);
  others.delete('r');
  const kept = others.toString() && (route.includes('?') ? '&' : '?') + others;
  /* Safari refuses a page more than 100 of these in 30 seconds. The address is then left as it was,
     and the next write puts it right. */
  try {
    history[push ? 'pushState' : 'replaceState'](null, '', route + kept);
    lastRoute = route;
  } catch { /* nothing was written */ }
}
