'use strict';

/* ── recorded sessions ──
   A session is the traces of one conversation, as `strands-evals fetch -o` writes it. A diagnosis is
   what `strands-evals diagnose -o` found in one: it shows on its session's page, or on its own when
   that session is not in the folder. */

let SESSIONS = [], DIAGNOSES = [], STORES = [];
let sesMeta = null, SES = null;              // the open session, or lone diagnosis, and its document
let sesFound = [];                           // the diagnoses shown with it, as `{ meta, doc }`
let sesMark = '';                            // what was on disk when it was read

/* ══════════ reading a folder without the server ══════════
   `sessionMeta`, `diagnosisMeta`, `sessionOf` and `isTaskResult` mirror `_session_summary`,
   `_diagnosis_summary`, `session_of` and `is_task_result` in `_ui_discovery.py`. */

const sessionMeta = obj => ({
  session_id: obj.session_id, traces: sessionTraces(obj).length,
  spans: sessionSpans(obj).length, seconds: traceStats(obj).seconds,
});
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
/* Which of the two a row of that list is. */
const isSession = m => m.traces != null;
const madeOf = s => (s.diagnoses || []).map(path => DIAGNOSES.find(d => d.path === path)).filter(Boolean);

/* What a row holds, as the list and the page header both say it. */
const sessionHolds = m => `${plural(m.traces, 'trace')} · ${plural(m.spans, 'span')}`;
const diagnosisHolds = m => `${plural(m.failures, 'failure')} · ${plural(m.root_causes, 'root cause')}`;
/* What the diagnoses made of a session found between them, or nothing when there is none. */
function foundTag(s) {
  const made = madeOf(s);
  if (!made.length) return '';
  const failures = made.reduce((n, d) => n + d.failures, 0);
  return ' · ' + (failures ? `<span class="tag warn">${plural(failures, 'failure')} found</span>`
    : '<span class="tag ok">no failures found</span>');
}

function renderSessions() {
  const rows = sessionRows();
  const lone = rows.length - SESSIONS.length;
  app.querySelector('.tab[data-view="sessions"]').hidden = !rows.length;
  $('sessions-count').innerHTML = plural(SESSIONS.length, 'session')
    + (lone ? ` · ${lone} ${lone === 1 ? 'diagnosis' : 'diagnoses'} without ${lone === 1 ? 'its' : 'their'} session` : '')
    + (truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '');
  $('sessions-rows').innerHTML = rows.map(m =>
    `<tr class="row${sesMeta && sesMeta.path === m.path ? ' on' : ''}" data-session="${esc(m.path)}" tabindex="0">
        <td>${filed(m)}</td>
        <td class="ev">` + (isSession(m) ? sessionHolds(m) + foundTag(m)
      : '<span class="tag" data-tip="A diagnosis whose session is not in this folder. It opens on its own.">diagnosis</span> '
        + diagnosisHolds(m)) + `</td>
        <td class="ev">${m.seconds == null ? '—' : secondsText(m.seconds)}</td>
        <td class="ev">${when(m.modified)}</td></tr>`).join('')
    || blank('No sessions in this pond yet', (MODE === 'server' ? `No recorded session or diagnosis under ${esc(ROOT)}. `
      : 'No recorded session or diagnosis among the files that were read. ')
      + 'Save one with <code>strands-evals fetch ... -o FILE</code>'
      + `${MODE === 'server' ? ', then Rescan' : ''}. ${docLink('cli/', 'empty-sessions', 'CLI reference')}`);
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
  return [[['Sessions', 'sessions'], [m.path]], isSession(m)
    ? `${sessionHolds(m)}${m.seconds == null ? '' : ` · ${secondsText(m.seconds)}`}`
      + ` · session <b>${esc(m.session_id)}</b>${foundTag(m)}`
    : `A diagnosis${m.session_id ? ` of session <b>${esc(m.session_id)}</b>, which is not in this folder` : ' that does not name its session'}`
      + ` · ${diagnosisHolds(m)}`];
}

function renderSession() {
  $('ses-found').innerHTML = sesFound.map(f => `<section class="found"><h3>${esc(f.meta.path)}</h3>`
    + `${diagnosisView(f.doc, { spans: sessionSpans(SES) })}</section>`).join('');
  $('ses-trace').innerHTML = isSession(sesMeta) ? traceView(SES) : rawView(SES, 'the file as written');
}

/* What is on disk for a session and its diagnoses, to tell when the open copy has gone stale. */
const sessionMark = m => [m.modified].concat((m.diagnoses || []).map(path =>
  (DIAGNOSES.find(d => d.path === path) || {}).modified)).join();

/* A session's document and those of the diagnoses made of it, as `{ doc, found }`. */
async function readSession(meta) {
  const doc = await document_for(meta);
  const found = isSession(meta) ? await Promise.all(madeOf(meta).map(async d => ({ meta: d, doc: await document_for(d) })))
    : [{ meta, doc }];
  return { doc, found };
}

function loadSession(meta, read) {
  sesMeta = meta; SES = read.doc; sesFound = read.found;
  sesMark = sessionMark(meta);
  renderSessions();
  renderSession();
}

/** Open a session on its own page. A diagnosis opens the session it was made from, or itself when that is not here. */
async function openSession(path) {
  const named = sessionRows().find(m => m.path === path)
    || SESSIONS.find(s => s.path === (DIAGNOSES.find(d => d.path === path) || {}).session);
  const read = named && await readToOpen(path, () => readSession(named));
  if (!read) return false;
  loadSession(named, read);
  scanError = '';
  renderLists();
  show('session');
  return true;
}

/** Re-read the open session when its file, or a diagnosis made of it, changed on disk. */
async function refreshSession() {
  const was = sesMeta, now = was && sessionRows().find(m => m.path === was.path);
  if (!now || sessionMark(now) === sesMark) return;
  const read = await readSession(now);
  if (sesMeta === was) loadSession(now, read);
}

app.addEventListener('click', e => {
  const named = e.target.closest('[data-session]');
  if (!named) return;
  e.preventDefault();
  openSession(named.dataset.session);
});
