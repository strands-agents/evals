'use strict';

const app = document.querySelector('.app');
const $ = id => document.getElementById(id);
const arrayOf = v => Array.isArray(v) ? v : [];
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = n => (typeof n === 'number' && isFinite(n)) ? n.toFixed(2) : '—';
const KB = n => n == null ? '' : (n < 1024 ? n + ' B' : (n / 1024).toFixed(1) + ' KB');
/* `n` may come from a file, and the result is often set as HTML. */
const plural = (n, w) => `${esc(n)} ${w}${n === 1 ? '' : 's'}`;
const nice = v => Number.isInteger(v) ? String(v)
  : Math.abs(v) >= 100 ? String(Math.round(v)) : (Math.round(v * 100) / 100).toFixed(2);
/* Four decimals: [0.004, 0.21] must not print as [0.00, 0.21], which reads as touching zero. */
const fine = v => Math.abs(v) >= 10 ? nice(v) : String(Number(v.toFixed(4)));
const compact = n => Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(2) + 'M'
  : Math.abs(n) >= 1e4 ? Math.round(n / 1e3) + 'k' : String(Math.round(n));

const when = secs => {
  if (!secs) return '—';
  const d = new Date(secs * 1000);
  return isNaN(d.getTime()) ? '—'
    : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
};

/* A trajectory entry is `list[Any]`: a tool name, a recorded call, or an unknown shape. */
function stepText(t) {
  if (t == null) return '—';
  if (typeof t === 'string') return t;
  if (typeof t !== 'object') return String(t);
  const name = t.name || t.tool || t.tool_name || t.toolName || t.function;
  if (typeof name === 'string') {
    const args = t.arguments || t.args || t.input || t.parameters;
    if (args && typeof args === 'object') {
      const inner = Object.keys(args).slice(0, 3).map(k => `${k}=${JSON.stringify(args[k])}`).join(', ');
      return `${name}(${inner}${Object.keys(args).length > 3 ? ', …' : ''})`;
    }
    return `${name}()`;
  }
  return JSON.stringify(t);
}
const sameStep = (a, b) => stepText(a) === stepText(b);

/** A 0..1 score as its number beside a proportional bar. */
const scoreBar = score => `<span class="bar2"><span class="num">${esc(fmt(score))}</span><span class="track">`
  + `<span class="fill" style="width:${Math.round(Math.max(0, Math.min(1, score || 0)) * 100)}%"></span>`
  + '</span></span>';

/** How `b` differs from `a`, in order: `[['same' | 'del' | 'ins', item], ...]`. */
function diffSeq(a, b) {
  const lcs = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let x = a.length - 1; x >= 0; x--)
    for (let y = b.length - 1; y >= 0; y--)
      lcs[x][y] = a[x] === b[y] ? lcs[x + 1][y + 1] + 1 : Math.max(lcs[x + 1][y], lcs[x][y + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { out.push(['same', a[i]]); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) out.push(['del', a[i++]]);
    else out.push(['ins', b[j++]]);
  }
  while (i < a.length) out.push(['del', a[i++]]);
  while (j < b.length) out.push(['ins', b[j++]]);
  return out;
}

/** Replace an element's markup without losing the keyboard's place in it: focus goes back to the
    element of the same tag and data attributes, when there still is one. */
function redraw(el, html) {
  const had = el.contains(document.activeElement) ? document.activeElement : null;
  el.innerHTML = html;
  if (!had) return;
  const mark = JSON.stringify(had.dataset);
  const same = [...el.querySelectorAll(had.localName)].find(x => JSON.stringify(x.dataset) === mark);
  if (same) same.focus();
}

/** Fill a picker, keeping `want` chosen if it is still on offer. Returns what ended up chosen.
    A few short choices sit side by side; more, or `asList`, open as a list under a button. */
function fillPick(id, options, want, asList) {
  const chosen = options.some(o => o.value === want) ? want : (options[0] || {}).value;
  const choice = o => `<button type="button" data-value="${esc(o.value)}" aria-pressed="${o.value === chosen}"`
    + `${o.value === chosen ? ' class="on"' : ''}>${esc(o.label)}</button>`;
  const few = !asList && options.length <= 4 && options.reduce((n, o) => n + String(o.label).length, 0) <= 44;
  const el = $(id);
  el.value = chosen;
  el.className = few ? 'pick seg' : 'pick';
  el.innerHTML = few ? options.map(choice).join('')
    : `<details><summary>${esc((options.find(o => o.value === chosen) || {}).label)}</summary>`
      + `<div class="pick-list">${options.map(choice).join('')}</div></details>`;
  return chosen;
}
/* The pickers and questions below are drawn by the page. A browser's own menus and dialogs can be
   suppressed or misplaced by an extension or an attached tool, and then the control does nothing. */
const closePicks = except => app.querySelectorAll('.pick details[open]').forEach(d => { if (d !== except) d.open = false; });
app.addEventListener('click', e => {
  const choice = e.target.closest('.pick button[data-value]');
  closePicks(choice ? null : e.target.closest('.pick details'));
  if (!choice) return;
  const pick = choice.closest('.pick');
  pick.value = choice.dataset.value;
  pick.dispatchEvent(new Event('change', { bubbles: true }));
  const again = $(pick.id).querySelector('.on, summary');
  if (again) again.focus();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePicks(); });

/** Ask a yes-or-no question in the page. Resolves true when `yes` is chosen; Escape and a click outside are a no. */
function ask(message, yes, no) {
  return new Promise(resolve => {
    const box = document.createElement('div');
    box.className = 'ask';
    box.setAttribute('role', 'alertdialog');
    box.setAttribute('aria-modal', 'true');
    box.innerHTML = `<div class="ask-card"><p>${esc(message)}</p><div class="ask-acts">`
      + `<button class="btn" data-answer="no">${esc(no)}</button>`
      + `<button class="btn pri" data-answer="yes">${esc(yes)}</button></div></div>`;
    const asker = document.activeElement;
    const onKey = e => {
      if (e.key === 'Escape') return done(false);
      if (e.key !== 'Tab') return;
      /* Nothing behind the question can be reached while it is up: Tab goes from one answer to the other. */
      e.preventDefault();
      const answers = [...box.querySelectorAll('[data-answer]')];
      answers[(answers.indexOf(document.activeElement) + 1) % answers.length].focus();
    };
    const done = answer => {
      box.remove();
      document.removeEventListener('keydown', onKey);
      asker.focus();
      resolve(answer);
    };
    box.onclick = e => {
      const button = e.target.closest('[data-answer]');
      if (button || e.target === box) done(!!button && button.dataset.answer === 'yes');
    };
    document.addEventListener('keydown', onKey);
    app.appendChild(box);
    box.querySelector('[data-answer="no"]').focus();
  });
}

/* The Strands frog, for a page or a table with nothing in it. */
const FROG = '<svg class="frog" viewBox="0 0 32 32" fill="currentColor" aria-hidden="true"><path d="'
  + 'M32 15.24H30.48V25.91H32V15.24ZM30.48 25.91H28.95V27.43H30.48V25.91ZM30.48 12.19H28.95V15.24H30.48V12.19ZM28.95 27.43H27.43V28.96H28.95V27.43ZM28.95 9.15H27.43V12.19H28.95V9.15ZM27.43 28.96H25.91V30.48H27.43V28.96ZM27.43 7.62H25.91V9.15H27.43V7.62ZM25.91 3.05H24.38V7.62H25.91V3.05ZM16.76 30.48V28.96H19.81V27.43H15.24V28.96H7.62V30.48H15.24V32H25.91V30.48H16.76ZM24.38 1.53H22.86V3.05H24.38V1.53ZM22.86 15.24H19.81V16.76H22.86V15.24ZM21.34 25.91H19.81V27.43H21.34V25.91ZM22.86 0H18.29V1.53H22.86V0ZM19.81 24.38H18.29V25.91H19.81V24.38ZM19.81 16.76H18.29V18.29H19.81V16.76ZM21.34 3.05H18.29V6.1H21.34V3.05ZM18.29 18.29H16.76V24.38H18.29V18.29ZM18.29 1.53H16.76V3.05H18.29V1.53ZM16.76 3.05H15.24V6.1H16.76V3.05ZM15.24 25.91H13.72V27.43H15.24V25.91ZM13.72 21.34H12.19V25.91H13.72V21.34ZM15.24 6.1H12.19V7.62H15.24V6.1ZM12.19 19.81H10.67V21.34H12.19V19.81ZM12.19 9.15H10.67V10.67H12.19V9.15ZM12.19 3.05H10.67V6.1H12.19V3.05ZM10.67 1.53H9.15V3.05H10.67V1.53ZM10.67 10.67H6.1V12.19H10.67V10.67ZM7.62 27.43H6.1V28.96H7.62V27.43ZM9.15 0H4.57V1.53H9.15V0ZM1.53 30.48V28.96H0V32H7.62V30.48H1.53ZM10.67 18.29H3.05V19.81H10.67V18.29ZM7.62 3.05H4.57V6.1H7.62V3.05ZM4.57 27.43H6.1V25.91H3.05V27.43H1.53V28.96H4.57V27.43ZM4.57 7.62H3.05V10.67H4.57V7.62ZM4.57 1.53H3.05V3.05H4.57V1.53ZM3.05 24.38H1.53V25.91H3.05V24.38ZM3.05 10.67H1.53V13.72H3.05V10.67ZM3.05 3.05H1.53V7.62H3.05V3.05ZM3.05 21.34V19.81H1.53V13.72H0V24.38H1.53V21.34H3.05Z"/></svg>';
/** A page or a whole table with nothing to show: the frog, a headline, then the reason. Both are markup. */
const emptyState = (title, body) => `<div class="nores">${FROG}<b>${title}</b>${body}</div>`;

const json = v => `<span class="mono">${esc(JSON.stringify(v, null, 1))}</span>`;
/* A case's input and outputs are whatever the task takes and returns: text, a dict, a list. */
const valueText = v => v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v, null, 2);
const isConversation = v => Array.isArray(v) && v.length > 0
  && v.every(m => m && typeof m === 'object' && typeof m.role === 'string' && 'content' in m);
/** A value of any shape as markup: text as it is, a conversation or a recorded session as one,
    anything else as indented JSON. */
function showValue(v) {
  if (typeof v === 'string') return esc(v);
  if (isConversation(v) || sessionSpans(v)) return `<div class="trace">${traceView(v, 'the value as recorded')}</div>`;
  return `<pre class="val">${esc(valueText(v))}</pre>`;
}
const pre = v => v == null ? '&mdash;' : json(v);

/** Put text on the clipboard. `text` may be a promise of it, as when the server is writing it: Safari
    lets a page write only inside the click that asked, so the clipboard is handed the promise where it takes one. */
async function copyText(text) {
  try {
    if (window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
      const blob = Promise.resolve(text).then(now => new Blob([now], { type: 'text/plain' }));
      await navigator.clipboard.write([new ClipboardItem({ 'text/plain': blob })]);
      return true;
    }
  } catch { /* fall through to the ways that wait for the text */ }
  let now;
  try { now = await text; } catch { return false; }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(now);
      return true;
    }
  } catch { /* fall through to the selection trick */ }
  /* `navigator.clipboard` needs a secure context, which a server reached by a name that is not loopback is not. */
  try {
    const ta = Object.assign(document.createElement('textarea'), { value: now });
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const done = document.execCommand && document.execCommand('copy');
    ta.remove();
    return !!done;
  } catch { return false; }
}

/* ══════════ transport ══════════ */

/* The server writes the token into the document; a link opened with `?t=` still works without that. */
const TOKEN = (document.querySelector('meta[name="strands-evals-token"]') || {}).content
  || new URLSearchParams(location.search).get('t') || '';
let ROOT = '', READONLY = true, DEPTH = null;

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { 'X-Evals-Token': TOKEN } }, opts || {}));
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    const err = new Error((body && body.error) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

/* ══════════ chrome ══════════ */

/* Every view the page has: `show()` knows no other. */
const SECTION_OF = {
  reports: 'evals', report: 'evals', experiments: 'evals', experiment: 'evals',
  sessions: 'evals', session: 'evals',
  'opt-runs': 'optimization', 'opt-setup': 'optimization', 'opt-overview': 'optimization',
  'opt-progress': 'optimization', 'opt-candidates': 'optimization',
  'opt-trade': 'optimization', missing: 'evals',
};
const VIEW_NAME = {
  reports: 'Reports', report: 'Report', experiments: 'Experiments', experiment: 'Cases',
  sessions: 'Sessions', session: 'Session', 'opt-runs': 'Runs',
  'opt-setup': 'Setup', 'opt-overview': 'Result', 'opt-progress': 'Iterations',
  'opt-candidates': 'Candidates', 'opt-trade': 'Trade-off', missing: 'Not found',
};
/* The list a page is opened from: the sidebar marks that, since the page itself is not in it. */
const LIST_OF = { report: 'reports', experiment: 'experiments', session: 'sessions' };
const listOf = view => (view === 'missing' ? MISSING.list
  : (view === 'report' && repCand) || SECTION_OF[view] === 'optimization' ? 'opt-runs' : LIST_OF[view] || view);
/* Reports when there are any, since most folders hold results; a folder of experiments alone opens on those,
   and one of sessions alone on those. */
const homeView = () => (REPORTS.length ? 'reports' : EXPERIMENTS.length ? 'experiments'
  : sessionRows().length ? 'sessions' : 'reports');

/* Counts the moves between views: a file that arrives after one is not opened over where the page went. */
let moves = 0;
/* What the `missing` view last said was not here: `{ route, list }`, set each time it is shown. */
let MISSING = null;

function show(view, opts) {
  moves += 1;
  /* The list of runs can be shown empty; a view of one run needs a run. */
  if (!SECTION_OF[view] || (SECTION_OF[view] === 'optimization' && view !== 'opt-runs' && !OPT_RUNS.length)) view = homeView();
  /* A page with nothing open in it is its list. */
  if (view === 'report' && !REP) view = 'reports';
  if (view === 'experiment' && !EXP) view = 'experiments';
  if (view === 'session' && !sesMeta) view = 'sessions';
  const moved = !$('v-' + view).classList.contains('on');
  app.querySelectorAll('.tab').forEach(t => {
    const here = t.dataset.view === view;
    t.classList.toggle('on', here || t.dataset.view === listOf(view));
    if (here) t.setAttribute('aria-current', 'page');
    else t.removeAttribute('aria-current');
  });
  app.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === 'v-' + view));
  const lists = view === 'reports' || view === 'experiments' || view === 'sessions';
  $('skips').hidden = !lists;
  $('stores').hidden = !lists;
  /* The bar names the folder being read, on every page: the file a page shows is named once, in
     the page's own header. */
  $('path').textContent = ROOT;
  /* A page of a run marks that run in the Runs list from then on. */
  if (view !== 'opt-runs' && view !== 'missing' && listOf(view) === 'opt-runs' && OPT_RUNS[optSel]) optSeen = OPT_RUNS[optSel].id;
  renderPageHead(view);
  renderGuide(view);
  markRunList();
  markRunRows();
  markFileTrees();
  setNav(false);
  if (opts && opts.silent) return;
  /* A view moved to starts at its top, where its heading and any notice are. */
  if (moved) window.scrollTo(0, 0);
  writeAddress(view);
}
/* The sidebar lists start folded. Clicking a view's row opens the view and unfolds its list; the
   caret beside the row folds it back, or unfolds it without opening anything. */
const TREE_OF = { reports: 'reports-tree', experiments: 'experiments-tree', 'opt-runs': 'runs-sub' };
function setFolded(tree, folded) {
  $(tree).classList.toggle('folded', folded);
  app.querySelector(`.fold[data-fold="${tree}"]`).setAttribute('aria-expanded', String(!folded));
}
app.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  const tree = TREE_OF[t.dataset.view];
  /* Clicking the row of the view already showing folds its list back and forth: the caret is a
     small target, and the row is already going nowhere. */
  if (tree) setFolded(tree, currentView() === t.dataset.view && !$(tree).classList.contains('folded'));
  show(t.dataset.view);
});
app.addEventListener('click', e => {
  const fold = e.target.closest('.fold[data-fold]');
  if (fold) setFolded(fold.dataset.fold, fold.getAttribute('aria-expanded') === 'true');
});
/* On a narrow window the sidebar is a drawer, opened by the button in the bar. Going anywhere, Escape,
   or a click outside it puts it away again. On a wide one the button is not shown and this does nothing. */
function setNav(open) {
  app.dataset.nav = open ? 'open' : 'closed';
  app.querySelector('[data-nav-toggle]').setAttribute('aria-expanded', String(open));
}
app.addEventListener('click', e => {
  if (e.target.closest('[data-nav-toggle]')) return setNav(app.dataset.nav !== 'open');
  /* The caret folds a list without going anywhere, so the drawer stays for what it unfolded. */
  if (app.dataset.nav === 'open' && !e.target.closest('.sections')) setNav(false);
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && app.dataset.nav === 'open') setNav(false); });

/* The first stop for the keyboard: past the bar and the sidebar, to what the page is showing. */
app.querySelector('[data-skip]').onclick = e => {
  e.preventDefault();
  app.querySelector('main').focus();
};

/* Rows, chips and chart marks that act as buttons carry tabindex="0": Enter or Space clicks them. */
app.addEventListener('keydown', e => {
  if ((e.key !== 'Enter' && e.key !== ' ') || e.target.getAttribute('tabindex') !== '0') return;
  e.preventDefault();
  e.target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
});
const currentView = () => app.querySelector('.view.on').id.slice(2);
