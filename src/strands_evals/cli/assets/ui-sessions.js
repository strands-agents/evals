'use strict';

/* ── recorded sessions ──
   A session is the traces of one conversation, as `strands-evals fetch -o` writes it. A diagnosis is
   what `strands-evals diagnose -o` found in one: it shows on its session's page, or on its own when
   that session is not in the folder. */

let SESSIONS = [], DIAGNOSES = [], STORES = [];
let sesMeta = null, SES = null;              // the open session, or lone diagnosis, and its document
let sesFound = [];                           // the diagnoses shown with it, as `{ meta, doc }`
let sesMark = '';                            // what was on disk when it was read

/* Registered from here so that the tables in `ui-core.js` and `ui-guide.js` hold no line per feature. */
Object.assign(SECTION_OF, { sessions: 'evals', session: 'evals' });
Object.assign(VIEW_NAME, { sessions: 'Sessions', session: 'Session' });
Object.assign(TERMS, {
  session: ['Session', 'Everything recorded for one conversation with an agent: one or more traces, saved as a file.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  trace: ['Trace', 'One run of the agent inside a session, made of spans: each agent invocation, model call and tool call.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  diagnosis: ['Diagnosis', 'What the detectors found in a session: the spans where something failed, and the root cause behind each.', 'detectors/'],
});
Object.assign(VIEW_GUIDE, {
  sessions: {
    steps: [
      'Each row is a <b>session</b>: the recorded traces of one conversation, saved with <code>strands-evals fetch ... -o FILE</code> or from code.',
      'Select a row to read it, one trace after another.',
      'A <b>diagnosis</b> shows on the page of the session it was made from. One whose session is not in this folder is listed here on its own.',
    ],
    terms: ['session', 'trace', 'diagnosis'],
    links: [['cli/', 'CLI reference'], ['detectors/', 'Diagnosing failures']],
  },
  session: {
    steps: [
      'Read it top to bottom. Each <b>trace</b> has its own heading and what it cost; the time between traces is not counted.',
      'A trace with several agent invocations is split into turns: the prompt and the reply first, then the model and tool calls behind them.',
      'No diagnosis yet? Make one, then Rescan: <code>strands-evals diagnose FILE -o NAME.diagnosis.json</code>',
      '<b>Copy as Markdown</b> at the top copies the session and its diagnoses as text for a coding agent.',
    ],
    terms: ['session', 'trace', 'diagnosis'],
    links: [['detectors/', 'Diagnosing failures'], ['cli/', 'CLI reference']],
  },
});

/* ══════════ reading a folder without the server ══════════
   `sessionMeta`, `diagnosisMeta`, `sessionOf` and `isTaskResult` mirror `_session_summary`,
   `_diagnosis_summary`, `session_of` and `is_task_result` in `_ui_server.py`. */

const sessionMeta = obj => {
  const traces = sessionTraces(obj);
  return {
    session_id: obj.session_id, traces: traces.length,
    spans: traces.reduce((n, tr) => n + tr.spans.length, 0), seconds: traceStats(obj).seconds,
  };
};
const diagnosisMeta = obj => ({
  session_id: typeof obj.session_id === 'string' ? obj.session_id : null,
  failures: (obj.failures || []).length, root_causes: (obj.root_causes || []).length,
});

/* The nearest session with the diagnosis's `session_id`. Two equally near are not guessed between. */
function sessionOf(diagnosis, sessions) {
  const named = sessions.filter(s => s.meta.obj.session_id === diagnosis.meta.obj.session_id);
  const nearest = Math.min(...named.map(s => stepsApart(s.dir, diagnosis.dir)));
  const closest = named.filter(s => stepsApart(s.dir, diagnosis.dir) === nearest);
  return closest.length === 1 ? closest[0] : null;
}

/* One case's saved task output, as `LocalFileTaskResultStore` writes it, and how such a file opens. */
const TASK_RESULT_HEAD = /^\s*\{\s*"input"\s*:/;
const isTaskResult = obj => !!obj && typeof obj === 'object' && !Array.isArray(obj)
  && ['input', 'actual_output', 'expected_output', 'actual_trajectory'].every(key => key in obj);

/** The sessions and diagnoses among the files a folder read found, paired the way the server pairs them. */
function sessionsFrom(found) {
  const newest = kind => found.filter(f => f.kind === kind)
    .sort((a, b) => b.meta.modified - a.meta.modified || (a.meta.path < b.meta.path ? -1 : 1));
  const sessions = newest('session'), diagnosed = new Map();
  const diagnoses = newest('diagnosis').map(d => {
    const s = sessionOf(d, sessions);
    if (s) diagnosed.set(s.meta.path, (diagnosed.get(s.meta.path) || []).concat(d.meta.path));
    return Object.assign(d.meta, { dir: d.dir, session: s ? s.meta.path : null });
  });
  return {
    sessions: sessions.map(s => Object.assign(s.meta, { dir: s.dir, diagnoses: diagnosed.get(s.meta.path) || [] })),
    diagnoses,
  };
}

/* ══════════ the list ══════════ */

/* What the Sessions list holds: every session, then each diagnosis whose session is not in the folder. */
const sessionRows = () => SESSIONS.concat(DIAGNOSES.filter(d => !d.session));
const failuresFound = n => (n ? `<span class="tag warn">${plural(n, 'failure')} found</span>`
  : '<span class="tag ok">no failures found</span>');
const madeOf = s => (s.diagnoses || []).map(path => DIAGNOSES.find(d => d.path === path)).filter(Boolean);
const lasted = seconds => (seconds == null ? '—' : `${nice(seconds)}s`);

function renderSessions() {
  const rows = sessionRows();
  const lone = rows.length - SESSIONS.length;
  app.querySelector('.tab[data-view="sessions"]').hidden = !rows.length;
  $('sessions-count').innerHTML = plural(SESSIONS.length, 'session')
    + (lone ? ` · ${lone} ${lone === 1 ? 'diagnosis' : 'diagnoses'} without ${lone === 1 ? 'its' : 'their'} session` : '')
    + (truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '');
  $('sessions-rows').innerHTML = rows.map(m => {
    const made = m.traces == null ? null : madeOf(m);
    return `<tr class="row${sesMeta && sesMeta.path === m.path ? ' on' : ''}" data-session="${esc(m.path)}" tabindex="0">
        <td>${filed(m)}</td>
        <td class="ev">` + (made
      ? `${plural(m.traces, 'trace')} · ${plural(m.spans, 'span')}`
        + (made.length ? ` · ${failuresFound(made.reduce((n, d) => n + d.failures, 0))}` : '')
      : '<span class="tag" data-tip="A diagnosis whose session is not in this folder. It opens on its own.">diagnosis</span> '
        + `${plural(m.failures, 'failure')} · ${plural(m.root_causes, 'root cause')}`) + `</td>
        <td class="ev">${made ? lasted(m.seconds) : '—'}</td>
        <td class="ev">${when(m.modified)}</td></tr>`;
  }).join('');
  $('stores').innerHTML = STORES.length
    ? `<p class="stores">${STORES.length === 1 ? 'A task result store was' : `${STORES.length} task result stores were`} found: `
      + STORES.map(s => `<code>${esc(s.dir)}</code> holds ${plural(s.files, 'saved result')}`).join(', ')
      + '. A store keeps what the task returned for each case, so its files are not listed here.</p>'
    : '';
}

/** Take what a scan or a folder read found. The lists are drawn here, since nothing else holds them. */
function setSessions(found) {
  SESSIONS = found.sessions || [];
  DIAGNOSES = found.diagnoses || [];
  STORES = found.stores || [];
  renderSessions();
}

/* ══════════ one session ══════════ */

function closeSession() {
  sesMeta = SES = null;
  sesFound = [];
}

function sessionHead() {
  const m = listed(sessionRows(), sesMeta);
  const made = m.traces == null ? null : madeOf(m);
  return [[['Sessions', 'sessions'], [m.path]], made
    ? `${plural(m.traces, 'trace')} · ${plural(m.spans, 'span')}${m.seconds == null ? '' : ` · ${lasted(m.seconds)}`}`
      + ` · session <b>${esc(m.session_id)}</b>`
      + (made.length ? ` · ${failuresFound(made.reduce((n, d) => n + d.failures, 0))}` : '')
    : `A diagnosis${m.session_id ? ` of session <b>${esc(m.session_id)}</b>, which is not in this folder` : ' that does not name its session'}`
      + ` · ${plural(m.failures, 'failure')} · ${plural(m.root_causes, 'root cause')}`];
}

function renderSession() {
  $('ses-found').innerHTML = sesFound.map(f => `<section class="found"><h3>${esc(f.meta.path)}</h3>`
    + `${diagnosisView(f.doc, { spans: sessionSpans(SES) })}</section>`).join('');
  $('ses-trace').innerHTML = sesMeta.traces == null
    ? `<details class="raw"><summary>the file as written</summary>`
      + `<pre class="out">${esc(JSON.stringify(SES, null, 1).slice(0, 200000))}</pre></details>`
    : traceView(SES);
}

/* What is on disk for a session and its diagnoses, to tell when the open copy has gone stale. */
const sessionMark = m => [m.modified].concat((m.diagnoses || []).map(path =>
  (DIAGNOSES.find(d => d.path === path) || {}).modified)).join();

async function loadSession(meta) {
  const doc = await document_for(meta);
  const found = meta.traces == null ? [{ meta, doc }]
    : await Promise.all(madeOf(meta).map(async d => ({ meta: d, doc: await document_for(d) })));
  sesMeta = meta; SES = doc; sesFound = found;
  sesMark = sessionMark(meta);
  renderSessions();
  renderSession();
}

/** Open a session on its own page. A diagnosis opens the session it was made from, or itself when that is not here. */
async function openSession(path) {
  const named = sessionRows().find(m => m.path === path)
    || SESSIONS.find(s => s.path === (DIAGNOSES.find(d => d.path === path) || {}).session);
  if (!named) return;
  try { await loadSession(named); } catch (err) { return openFailed(path, err); }
  scanError = '';
  renderLists();
  show('session');
}

/** Re-read the open session when its file, or a diagnosis made of it, changed on disk. */
async function refreshSession() {
  const now = sesMeta && sessionRows().find(m => m.path === sesMeta.path);
  if (now && sessionMark(now) !== sesMark) await loadSession(now);
}

app.addEventListener('click', e => {
  const named = e.target.closest('[data-session]');
  if (!named) return;
  e.preventDefault();
  openSession(named.dataset.session);
});
