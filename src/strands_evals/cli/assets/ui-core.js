/* Two modes. server: served by `strands-evals ui`, Save writes the loaded file in place.
 * standalone: opened from disk, Save downloads a copy because file:// cannot write back. */

'use strict';

const app = document.querySelector('.app');
const $ = id => document.getElementById(id);
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
    const onKey = e => { if (e.key === 'Escape') done(false); };
    const done = answer => {
      box.remove();
      document.removeEventListener('keydown', onKey);
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

/* ══════════ transport ══════════ */

/* The server writes the token into the document; a link opened with `?t=` still works without that. */
const TOKEN = (document.querySelector('meta[name="strands-evals-token"]') || {}).content
  || new URLSearchParams(location.search).get('t') || '';
let MODE = 'standalone', ROOT = '', READONLY = true, DEPTH = null;

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

const SECTION_OF = {
  reports: 'evals', report: 'evals', experiments: 'evals', experiment: 'evals',
  'opt-runs': 'optimization', 'opt-setup': 'optimization', 'opt-overview': 'optimization',
  'opt-progress': 'optimization', 'opt-candidates': 'optimization', 'opt-tunables': 'optimization',
  'opt-trade': 'optimization',
};
/* The list a page is opened from: the sidebar marks that, since the page itself is not in it. */
const listOf = view => view === 'report' ? (repCand ? 'opt-runs' : 'reports')
  : view === 'experiment' ? 'experiments'
    : view === 'session' ? 'sessions'
      : SECTION_OF[view] === 'optimization' ? 'opt-runs' : view;
const VIEW_NAME = {
  reports: 'Reports', report: 'Report', experiments: 'Experiments', experiment: 'Cases', 'opt-runs': 'Runs',
  'opt-setup': 'Setup', 'opt-overview': 'Overview', 'opt-progress': 'Progress',
  'opt-candidates': 'Candidates', 'opt-tunables': 'Tunables', 'opt-trade': 'Trade-off',
};
/* Reports when there are any, since most folders hold results; a folder of experiments alone opens on those,
   and one of sessions alone on those. */
const homeView = () => (REPORTS.length ? 'reports' : EXPERIMENTS.length ? 'experiments'
  : sessionRows().length ? 'sessions' : 'reports');

function show(view, opts) {
  if (!SECTION_OF[view] || (SECTION_OF[view] === 'optimization' && !OPT_RUNS.length)) view = homeView();
  /* A page with nothing open in it is its list. */
  if (view === 'report' && !REP) view = 'reports';
  if (view === 'experiment' && !EXP) view = 'experiments';
  if (view === 'session' && !sesMeta) view = 'sessions';
  if (view === 'sessions' && !sessionRows().length) view = homeView();
  const section = SECTION_OF[listOf(view)];
  const moved = !$('v-' + view).classList.contains('on');
  app.querySelectorAll('.sec').forEach(s => s.classList.toggle('on', s.dataset.sec === section));
  app.querySelectorAll('.tab').forEach(t => {
    const here = t.dataset.view === view;
    t.classList.toggle('on', here || t.dataset.view === listOf(view));
    if (here) t.setAttribute('aria-current', 'page');
    else t.removeAttribute('aria-current');
  });
  app.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === 'v-' + view));
  const lists = view === 'reports' || view === 'experiments' || view === 'sessions';
  $('folder').hidden = !lists;
  $('skips').hidden = !lists;
  $('stores').hidden = !lists;
  const run = OPT_RUNS[optSel];
  const folder = ROOT || folderLabel || 'no folder opened';
  const where = {
    reports: folder,
    experiments: folder,
    experiment: expMeta && expMeta.path,
    report: repMeta && repMeta.path,
    sessions: folder,
    session: sesMeta && sesMeta.path,
    'opt-runs': folder,
    'opt-setup': (run && (run.dir === '.' ? 'run.json' : run.dir + '/run.json')) || 'no optimizer run loaded',
    'opt-tunables': (run && (run.dir === '.' ? 'run.json' : run.dir + '/run.json')) || 'no optimizer run loaded',
    'opt-overview': (run && run.events_path) || 'no optimizer run loaded',
    'opt-progress': (run && run.events_path) || 'no optimizer run loaded',
    'opt-candidates': (run && run.events_path) || 'no optimizer run loaded',
    'opt-trade': (run && run.events_path) || 'no optimizer run loaded',
  }[view];
  $('path').textContent = where;
  renderPageHead(view);
  renderGuide(view);
  if (opts && opts.silent) return;
  /* A view moved to starts at its top, where its heading and any notice are. */
  if (moved) window.scrollTo(0, 0);
  writeAddress(view);
}
app.querySelectorAll('.tab').forEach(t => t.onclick = () => show(t.dataset.view));
app.querySelectorAll('.sec').forEach(s => s.onclick = () => {
  const first = [...app.querySelectorAll('.tab')].find(t => t.dataset.sec === s.dataset.sec && !t.hidden);
  if (first) show(first.dataset.view);
});
/* Rows, chips and chart marks that act as buttons carry tabindex="0": Enter or Space clicks them. */
app.addEventListener('keydown', e => {
  if ((e.key !== 'Enter' && e.key !== ' ') || e.target.getAttribute('tabindex') !== '0') return;
  e.preventDefault();
  e.target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
});
const currentView = () => app.querySelector('.view.on').id.slice(2);
