'use strict';

/* ── recorded sessions ──
   A session is the traces of one conversation, as `strands-evals fetch -o` writes it. A diagnosis is
   what `strands-evals diagnose -o` found in one: it shows on its session's page, or on its own when
   that session is not in the folder. */

let SESSIONS = [], DIAGNOSES = [], STORES = [];
let sesMeta = null, SES = null;              // the open session, or lone diagnosis, and its document
let sesFound = [];                           // the diagnoses shown with it, as `{ meta, doc }`
let sesMark = '';                            // what was on disk when it was read

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
  return ' · ' + (failures ? `<span class="tag bad">${plural(failures, 'failure')} found</span>`
    : '<span class="tag ok">no failures found</span>');
}

function renderSessions() {
  const all = sessionRows(), rows = arranged('sessions', all);
  const lone = all.length - SESSIONS.length;
  app.querySelector('.tab[data-view="sessions"]').hidden = !all.length;
  $('sessions-count').innerHTML = (LIST_STATE.sessions.q ? `${rows.length} of ` : '') + plural(SESSIONS.length, 'session')
    + (lone ? ` · ${lone} ${lone === 1 ? 'diagnosis' : 'diagnoses'} without ${lone === 1 ? 'its' : 'their'} session` : '')
    + (truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '');
  $('sessions-rows').innerHTML = rows.map(m =>
    `<tr class="row${sesMeta && sesMeta.path === m.path ? ' on' : ''}" data-session="${esc(m.path)}" tabindex="0" aria-describedby="row-hint">
        <td>${filed(m)}</td>
        <td class="ev">` + (isSession(m) ? sessionHolds(m) + foundTag(m)
      : '<span class="tag" data-tip="A diagnosis whose session is not in this folder. It opens on its own.">diagnosis</span> '
        + diagnosisHolds(m)) + `</td>
        <td class="ev">${m.seconds == null ? '—' : secondsText(m.seconds)}</td>
        <td class="ev">${when(m.modified)}</td></tr>`).join('')
    || (all.length ? noMatch('sessions', 4) : '')
    || blank('No sessions here yet', `No recorded session or diagnosis under ${esc(ROOT)}. `
      + 'Save one with <code>strands-evals fetch ... -o FILE</code>, then Rescan. '
      + docLink('cli/', 'empty-sessions', 'CLI reference'));
  $('stores').innerHTML = STORES.length
    ? `<p class="stores">${STORES.length === 1 ? 'A task result store was' : `${STORES.length} task result stores were`} found: `
      + STORES.map(s => `<code>${esc(s.dir)}</code> holds ${plural(s.files, 'saved result')}`).join(', ')
      + '. A store keeps what the task returned for each case, so its files are not listed here.</p>'
    : '';
}

/** Take what a scan found. The lists are drawn here, since nothing else holds them. */
function setSessions(found) {
  SESSIONS = found.sessions || [];
  DIAGNOSES = found.diagnoses || [];
  STORES = found.stores || [];
  renderSessions();
}

/* ══════════ one session ══════════ */

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
