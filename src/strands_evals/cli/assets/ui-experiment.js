'use strict';

let EXP = null, expMeta = null;              // the loaded experiment document + its file
let cases = [], sel = 0;                     // working copy, selection
let evals = [], origEvals = '[]', evOpen = -1;

/* `savedCases` is the file as last loaded or saved. `orig[i]` is the saved serialization of
   `cases[i]`, or NEW_CASE for a case that is not in the file yet. */
const NEW_CASE = '\u0000';
let savedCases = [], orig = [];

function markSaved() {
  savedCases = cases.map(c => JSON.stringify(c));
  orig = savedCases.slice();
  origEvals = JSON.stringify(evals);
}

const dirtyCases = () => cases.map((c, i) => JSON.stringify(c) !== orig[i] ? i : -1).filter(i => i >= 0);
const removedCases = () => savedCases.length - orig.filter(s => s !== NEW_CASE).length;
const evalsDirty = () => JSON.stringify(evals) !== origEvals;
const countDirty = () => dirtyCases().length + removedCases() + (evalsDirty() ? 1 : 0);

function loadExperiment(meta, doc) {
  expMeta = meta; EXP = doc;
  cases = (doc.cases || []).map(c => JSON.parse(JSON.stringify(c)));
  evals = (doc.evaluators || []).map(e => JSON.parse(JSON.stringify(e)));
  markSaved();
  sel = 0; evOpen = -1;
  $('case-find').value = '';
  $('exp-sub').textContent = READONLY && MODE === 'server'
    ? 'Read-only: the server was started with --read-only.'
    : (MODE === 'server'
      ? `Edits save back to ${meta.path}.`
      : 'No server, so Save downloads a copy for you to put back yourself.');
  renderEvaluators(); renderForm(); refreshDirty(); note('');
}

function clearExperiment() {
  EXP = null; expMeta = null; cases = []; evals = []; sel = 0; evOpen = -1;
  markSaved();
  $('evs').innerHTML = '<span class="empty">no experiment loaded</span>';
  $('evnotes').innerHTML = '';
  $('evform').hidden = true;
  $('form-title').textContent = '—';
  $('form-sid').textContent = '';
  $('form').innerHTML = '<div class="nores"><b>No experiment in this run</b>'
    + 'This folder had a report but no experiment file beside it.</div>';
  $('case-list').innerHTML = '';
  $('case-count').textContent = '';
  $('exp-sub').textContent = '';
  refreshDirty(); note('');
}

function note(msg, kind) {
  const el = $('saved');
  el.innerHTML = msg ? esc(msg) : '';
  el.classList.toggle('err', kind === 'err');
  const offer = { stale: ['Reload from disk', () => selectRun(selRun, { force: true })],
    gone: ['Download a copy', downloadExperiment] }[kind];
  if (offer) {
    const btn = document.createElement('button');
    btn.className = 'btn reload';
    btn.textContent = offer[0];
    btn.onclick = offer[1];
    el.appendChild(btn);
  }
}

/* ══════════ evaluators ══════════ */

const CONFIG_KEYS = e => Object.keys(e).filter(k => k !== 'evaluator_type' && k !== 'name');

function evalNotes(index) {
  const notes = (expMeta && expMeta.evaluator_notes) || [];
  return index == null ? notes : notes.filter(n => n.index === index);
}

/* A chip names its settings and cuts a long value short: the whole of it is in the panel the chip opens. */
const CHIP_VALUE = 44;
const chipValue = v => {
  const text = JSON.stringify(v);
  return text.length > CHIP_VALUE ? text.slice(0, CHIP_VALUE).trimEnd() + '…' : text;
};
/* Prose settings get a text block to be read and edited in; the rest stay one line. */
const isProse = (key, v) => typeof v === 'string' && (v.length > 60 || v.includes('\n') || key === 'rubric' || key === 'system_prompt');

function renderEvaluators() {
  $('evs').innerHTML = evals.length ? evals.map((e, i) => {
    const type = e.evaluator_type || 'unknown';
    const cfg = CONFIG_KEYS(e)
      .map(k => `<span class="ck">${esc(k)}</span>=${esc(chipValue(e[k]))}`).join(' · ');
    const notes = evalNotes(i);
    const worst = notes.some(n => n.level === 'error') ? 'error' : (notes.length ? 'warning' : '');
    return `<span class="evchip edit${i === evOpen ? ' on' : ''}${worst ? ' ' + worst : ''}" tabindex="0" role="button"`
      + ` aria-expanded="${i === evOpen}" data-ev="${i}"`
      + ` title="${worst ? esc(notes.map(n => n.message).join('; ')) : 'Select to read and edit its settings'}">`
      + `<b>${esc(e.name || type)}</b>`
      + (e.name ? `<span class="t">${esc(type)}</span>` : '')
      + (cfg ? `<span class="c mono">${cfg}</span>` : '')   // already escaped, per key and value
      + (worst ? `<span class="c">${worst === 'error' ? 'will not load' : 'custom'}</span>` : '')
      + '</span>';
  }).join('') : '<span class="empty">this experiment declares no evaluators</span>';

  const errs = evalNotes().filter(n => n.level === 'error');
  $('evnotes').innerHTML = errs.length
    ? `<span class="tag warn">${plural(errs.length, 'evaluator problem')}</span> `
      + errs.map(n => `<code>${esc(n.evaluator_type)}</code> ${esc(n.message)}`).join(' · ')
      + ' — <span class="path">the file saves, but <code>run</code> and <code>validate</code>'
      + ' will fail to load it until this is fixed.</span>'
    : '';
  renderEvalForm();
}

/* What an evaluator checks, in the library's own words when the server supplied them, and its docs page. */
function evaluatorSummary(type) {
  const help = helpFor('evaluator', type);
  return `<p class="evsum">${esc(help.text)} ${docLink(help.doc[0], help.doc[1], 'Read the docs')}</p>`;
}

/* Only keys already in the file are editable: `Experiment.from_dict` passes every key except
   `evaluator_type` to the constructor, so a new key would raise TypeError on load. */
function renderEvalForm() {
  const box = $('evform');
  if (evOpen < 0 || !evals[evOpen]) { box.hidden = true; box.innerHTML = ''; return; }
  const e = evals[evOpen];
  const keys = ['name'].concat(CONFIG_KEYS(e));
  const editable = keys.filter(k => e[k] == null || ['string', 'number', 'boolean'].includes(typeof e[k]));
  const frozen = keys.filter(k => !editable.includes(k));
  box.hidden = false;
  box.innerHTML = `<div class="evhead"><b>${esc(e.name || e.evaluator_type)}</b>`
    + `<span class="path mono">${esc(e.evaluator_type)}</span><span class="grow"></span>`
    + '<button class="btn" data-evclose="1">Close</button></div>'
    + evaluatorSummary(e.evaluator_type)
    + '<div class="grid">'
    + editable.map(k => {
      const v = e[k];
      const prose = isProse(k, v);
      const type = typeof v === 'boolean' ? 'checkbox' : (typeof v === 'number' ? 'number' : 'text');
      const attrs = type === 'checkbox' ? (v ? 'checked' : '') : `value="${esc(v == null ? '' : v)}"`;
      const rows = prose ? Math.min(16, Math.max(3, Math.ceil(v.length / 95) + v.split('\n').length - 1)) : 0;
      return `<span class="evlab${prose ? ' top' : ''}"><label for="ev-${esc(k)}">${esc(k)}</label>`
        + `${helpFor('setting', `${e.evaluator_type}.${k}`) ? helpButton('setting', `${e.evaluator_type}.${k}`, k) : ''}</span>`
        + (prose
          ? `<textarea class="f" id="ev-${esc(k)}" rows="${rows}" data-evk="${esc(k)}">${esc(v)}</textarea>`
          : `<input class="f" id="ev-${esc(k)}" type="${type}" data-evk="${esc(k)}" ${attrs}`
            + (k === 'name' ? ' placeholder="defaults to the class name"' : '') + '>');
    }).join('')
    + '</div>'
    + (frozen.length
      ? `<div class="hint">Not editable here: <code>${frozen.map(esc).join('</code>, <code>')}</code>`
        + ' — structured values are left exactly as the file had them.</div>'
      : '')
    + '<div class="hint">Only keys already in the file are shown; a key the '
    + 'constructor never accepted would make the experiment fail to load.</div>';
}

$('evs').onclick = e => {
  const chip = e.target.closest('[data-ev]');
  if (!chip) return;
  const i = +chip.dataset.ev;
  evOpen = evOpen === i ? -1 : i;
  renderEvaluators();
};
$('evform').addEventListener('click', e => {
  if (e.target.closest('[data-evclose]')) { evOpen = -1; renderEvaluators(); }
});
$('evform').addEventListener('input', e => {
  const k = e.target.dataset.evk;
  if (k == null || evOpen < 0) return;
  const el = e.target;
  let value;
  if (el.type === 'checkbox') value = el.checked;
  else if (el.type === 'number') {
    if (el.value === '') value = null;
    else if (!isFinite(Number(el.value))) { el.classList.add('invalid'); return; }
    else value = Number(el.value);
  } else value = el.value === '' ? null : el.value;
  el.classList.remove('invalid');
  if (value === null && k === 'name') delete evals[evOpen].name;
  else evals[evOpen][k] = value;
  refreshDirty(); note('');
});

/* ══════════ case editor ══════════ */

const FIELDS = [
  { k: 'name', l: 'Name', t: 'line', hint: 'Identifies the case in the report.' },
  { k: 'input', l: 'Input', t: 'area', rows: 2, required: true, hint: 'What the agent receives. Required.' },
  { k: 'expected_output', l: 'Expected output', t: 'area', rows: 3,
    hint: 'Leave empty and an output evaluator reports not_applicable.' },
  { k: 'expected_assertion', l: 'Expected assertion', t: 'area', rows: 2,
    hint: 'A natural-language claim a judge can check.' },
  { k: 'expected_trajectory', l: 'Expected trajectory', t: 'steps' },
  { k: 'expected_interactions', l: 'Expected interactions', t: 'json' },
  { k: 'expected_environment_state', l: 'Expected env state', t: 'json' },
  { k: 'metadata', l: 'Metadata', t: 'meta',
    hint: 'Free-form keys that travel with the case: status, synthetic, source ids, grouping tags.' },
];

const isDraft = c => !!(c && c.metadata && c.metadata.status === 'draft');

const PROP_GLYPHS = [['expected_output', 'O'], ['expected_assertion', 'A'],
  ['expected_trajectory', 'T'], ['expected_interactions', 'I'],
  ['expected_environment_state', 'E'], ['metadata', 'M']];
/* An empty dict or list means the same as an absent key to every evaluator. */
const props = c => PROP_GLYPHS
  .filter(([k]) => c[k] != null && (typeof c[k] !== 'object' || Object.keys(c[k]).length))
  .map(([k, g]) => `<span title="This case sets ${k}.">${g}</span>`).join('');

function renderList() {
  const q = $('case-find').value.trim().toLowerCase();
  const shown = cases.map((c, i) => ({ c, i })).filter(({ c }) =>
    !q || ((c.name || '') + ' ' + (c.input || '')).toLowerCase().includes(q));
  $('case-list').innerHTML = shown.map(({ c, i }) => {
    const edited = JSON.stringify(c) !== orig[i];
    return `<li data-i="${i}" tabindex="0" class="${i === sel ? 'on' : ''}" title="${edited ? 'unsaved changes' : esc(c.name || '(unnamed)')}">`
      + `<span class="dot${edited ? ' edited' : ''}"></span>`
      + `<span class="nm">${esc(c.name || '(unnamed)')}</span>`
      + (isDraft(c) ? '<span class="badge" data-term="draft">draft</span>' : '')
      + `<span class="props mono">${props(c)}</span></li>`;
  }).join('') || '<li class="empty">No match</li>';
  const d = dirtyCases().length;
  $('case-count').textContent = `${shown.length} of ${plural(cases.length, 'case')}` + (d ? ` · ${d} edited` : '');
  renderReview();
}

function renderForm() {
  renderList();
  const c = cases[sel];
  if (!c) {
    $('form-title').textContent = '—';
    $('form-sid').textContent = '';
    $('form').innerHTML = EXP
      ? '<div class="nores"><b>No cases</b>This experiment’s case list is empty. Use <b>+ Case</b> to add one.</div>'
      : '<div class="nores"><b>No experiment loaded</b>Pick a run on the Runs tab.</div>';
    return;
  }
  $('form-title').textContent = c.name || '(unnamed)';
  $('form-sid').textContent = String(c.session_id || '').slice(0, 8);
  $('form').innerHTML = FIELDS.map(f => {
    const v = c[f.k];
    let ctl;
    if (f.t === 'steps') {
      ctl = stepsControl(f.k, v);
    } else if (f.t === 'meta') {
      ctl = metaControl(f.k, v);
    } else if (f.t === 'json') {
      ctl = v == null
        ? `<textarea class="f mono" rows="2" data-k="${f.k}" data-json="1" spellcheck="false" placeholder="empty"></textarea>`
        : `<textarea class="f mono" rows="${Math.min(10, JSON.stringify(v, null, 2).split('\n').length)}" data-k="${f.k}" data-json="1" spellcheck="false">${esc(JSON.stringify(v, null, 2))}</textarea>`;
    } else if (f.t === 'line') {
      ctl = `<input class="f" data-k="${f.k}" value="${esc(v == null ? '' : v)}" placeholder="unnamed">`;
    } else {
      ctl = `<textarea class="f" rows="${f.rows}" data-k="${f.k}"${f.required ? ' data-required="1"' : ''} placeholder="empty">${esc(v == null ? '' : v)}</textarea>`;
    }
    return `<div class="field" role="group" aria-labelledby="label-${f.k}">`
      + `<div class="flabel"><label id="label-${f.k}">${f.l}</label>${helpButton('field', f.k, f.l)}</div>${ctl}`
      + (f.hint ? `<div class="hint">${f.hint}</div>` : '') + '</div>';
  }).join('');
}

/* `expected_trajectory` is `list[Any]`: strings, recorded calls, or not a list at all. */
function stepsControl(key, v) {
  if (v == null) {
    return `<div class="chips" data-chips="${key}">`
      + '<span class="empty">none — a trajectory evaluator reports not_applicable</span>'
      + `<button class="addchip" data-add="${key}">+ tool</button></div>`;
  }
  if (!Array.isArray(v)) {
    return `<textarea class="f mono" rows="4" data-k="${key}" data-json="1" spellcheck="false">${esc(JSON.stringify(v, null, 2))}</textarea>`
      + '<div class="hint">This trajectory is not a list, so it is shown as JSON rather than as steps.</div>';
  }
  const structured = v.some(t => typeof t !== 'string');
  return `<div class="chips" data-chips="${key}">`
    + v.map((t, j) => `<span class="chip"><span class="ord">${j + 1}</span>${esc(stepText(t))}`
      + `<button class="x" type="button" data-rm="${j}" aria-label="remove step ${j + 1}">&times;</button></span>`).join('')
    + (structured ? '' : `<button class="addchip" data-add="${key}">+ tool</button>`)
    + '</div>'
    + (structured
      ? '<div class="hint">Recorded calls, summarised. Remove still works; adding one is only '
        + 'offered for plain tool names, since this page cannot invent a valid call shape.</div>'
      : '');
}

/* ── metadata editor ── */
/* A string that would not survive the input as typed ("42", "true", one with a line break) is shown quoted. */
function metaVal(v) {
  return typeof v === 'string' && parseMetaValue(v) === v && !/[\r\n]/.test(v) ? v : JSON.stringify(v);
}

function parseMetaValue(raw) {
  const t = raw.trim();
  if (t === '') return '';
  try { return JSON.parse(t); } catch { return raw; }
}

function readMetaRows(box) {
  const entries = [];
  box.querySelectorAll('.meta-row').forEach(r => {
    const typedKey = r.querySelector('[data-meta-k]').value, typedValue = r.querySelector('[data-meta-v]').value;
    /* `data-was` is the pair as the file had it: a field left alone gives that back, untrimmed and unparsed. */
    const was = r.dataset.was ? JSON.parse(r.dataset.was) : null;
    const key = was && typedKey === was[0] ? was[0] : typedKey.trim();
    if (!key && !(was && was[0] === '')) return;      // a row still being typed into is not a key yet
    entries.push([key, was && typedValue === metaVal(was[1]) ? was[1] : parseMetaValue(typedValue)]);
  });
  return Object.fromEntries(entries);   // unlike assignment, keeps a `__proto__` key as data
}

const metaRow = (k, v) => `<div class="meta-row" data-was="${esc(JSON.stringify([k, v]))}">`
  + `<input class="f mk" data-meta-k="1" value="${esc(k)}" placeholder="key" aria-label="metadata key">`
  + `<input class="f" data-meta-v="1" value="${esc(metaVal(v))}" placeholder="value" aria-label="metadata value">`
  + '<button class="x" type="button" data-meta-rm="1" aria-label="remove this key">&times;</button></div>';
const blankMetaRow = () => metaRow('', '').replace(/ data-was="[^"]*"/, '');

function metaControl(key, v) {
  if (v != null && (typeof v !== 'object' || Array.isArray(v))) {
    return `<textarea class="f mono" rows="2" data-k="${key}" data-json="1" spellcheck="false">${esc(JSON.stringify(v, null, 2))}</textarea>`
      + '<div class="hint">This metadata is not an object, so it is shown as JSON rather than as keys.</div>';
  }
  return `<div class="meta-rows" data-meta="${key}">`
    + Object.entries(v || {}).map(([mk, mv]) => metaRow(mk, mv)).join('')
    + `<button class="addchip" data-meta-add="${key}">+ key</button></div>`;
}

/* ── review pass over drafts ── */
let review = false;
const draftIdx = () => cases.map((c, i) => (isDraft(c) ? i : -1)).filter(i => i >= 0);

function renderReview() {
  const bar = $('review-bar');
  if (!bar) return;
  const drafts = draftIdx();
  const btn = $('review-case');
  btn.textContent = drafts.length ? `Review drafts (${drafts.length})` : 'Review drafts';
  btn.disabled = !drafts.length || (MODE === 'server' && READONLY);
  if (!review) { bar.hidden = true; return; }
  if (!drafts.length) {
    review = false; bar.hidden = true;
    note('every draft is resolved — save to write it out');
    return;
  }
  if (!drafts.includes(sel)) { sel = drafts[0]; renderForm(); return; }
  bar.hidden = false;
  bar.innerHTML = `<span><b>${drafts.length}</b> ${drafts.length === 1 ? 'draft' : 'drafts'} left`
    + `, <b>${cases.length - drafts.length}</b> kept</span>`
    + '<span class="grow"></span>'
    + '<button class="btn" data-rv="accept">Accept (a)</button>'
    + '<button class="btn" data-rv="drop">Drop (d)</button>'
    + '<button class="btn" data-rv="acceptall">Accept all</button>'
    + '<button class="btn" data-rv="exit">Exit (esc)</button>';
}

function reviewAct(act) {
  if (!review) return;
  if (act === 'exit') { review = false; renderReview(); return; }
  /* `note('')` must precede the re-render: `renderReview` announces completion through `note`. */
  if (act === 'acceptall') {
    draftIdx().forEach(i => { cases[i].metadata = { ...cases[i].metadata, status: 'accepted' }; });
    note(''); renderForm(); refreshDirty(); return;
  }
  const c = cases[sel];
  if (!c) return;
  if (act === 'accept') c.metadata = { ...c.metadata, status: 'accepted' };
  if (act === 'drop') {
    cases.splice(sel, 1); orig.splice(sel, 1);
    sel = Math.max(0, Math.min(sel, cases.length - 1));
  }
  const drafts = draftIdx();
  if (drafts.length) sel = drafts[0];
  note(''); renderForm(); refreshDirty();
}

$('review-case').onclick = () => { review = true; renderReview(); };
$('review-bar').onclick = e => {
  const b = e.target.closest('[data-rv]');
  if (b) reviewAct(b.dataset.rv);
};

document.addEventListener('keydown', e => {
  if (!review || currentView() !== 'experiment') return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'a') { e.preventDefault(); reviewAct('accept'); }
  else if (e.key === 'd') { e.preventDefault(); reviewAct('drop'); }
  else if (e.key === 'Escape') reviewAct('exit');
});

$('case-find').oninput = renderList;
$('case-list').onclick = e => {
  const li = e.target.closest('li[data-i]');
  if (!li) return;
  sel = +li.dataset.i;
  renderForm(); writeAddress(currentView(), true);
};

$('form').addEventListener('input', e => {
  /* Metadata rows carry no `data-k`: the dict is rebuilt from every row on each keystroke. */
  const box = e.target.closest('[data-meta]');
  if (box && cases[sel] && e.target.dataset.k == null) {
    const next = readMetaRows(box);
    if (Object.keys(next).length) cases[sel][box.dataset.meta] = next;
    else delete cases[sel][box.dataset.meta];
    note(''); refreshDirty();
    return;
  }
  const k = e.target.dataset.k;
  if (k == null || !cases[sel]) return;
  const el = e.target, raw = el.value;
  if (el.dataset.json) {
    if (raw.trim() === '') { cases[sel][k] = null; el.classList.remove('invalid'); }
    else {
      try { cases[sel][k] = JSON.parse(raw); el.classList.remove('invalid'); }
      catch { el.classList.add('invalid'); return; }
    }
  } else if (el.dataset.required) {
    // `Case.input` is required: keep the empty string, not null, and let preflight() report it.
    cases[sel][k] = raw;
  } else {
    cases[sel][k] = raw.trim() === '' ? null : raw;
  }
  if (k === 'name') { $('form-title').textContent = cases[sel].name || '(unnamed)'; }
  note(''); refreshDirty();
});

$('form').addEventListener('click', e => {
  if (!cases[sel]) return;
  /* Added in the DOM only: an empty row must not reach the case as a `""` key. */
  const metaAdd = e.target.closest('[data-meta-add]');
  if (metaAdd) {
    metaAdd.insertAdjacentHTML('beforebegin', blankMetaRow());
    const fresh = metaAdd.previousElementSibling.querySelector('[data-meta-k]');
    if (fresh) fresh.focus();
    return;
  }
  const metaRm = e.target.closest('[data-meta-rm]');
  if (metaRm) {
    const box = metaRm.closest('[data-meta]');
    metaRm.closest('.meta-row').remove();
    const next = readMetaRows(box);
    if (Object.keys(next).length) cases[sel][box.dataset.meta] = next;
    else delete cases[sel][box.dataset.meta];
    note(''); refreshDirty();
    return;
  }
  const rm = e.target.closest('[data-rm]');
  const add = e.target.closest('[data-add]');
  if (rm) {
    const k = rm.closest('[data-chips]').dataset.chips;
    cases[sel][k].splice(+rm.dataset.rm, 1);
    if (!cases[sel][k].length) cases[sel][k] = null;
    renderForm(); note(''); refreshDirty();
  } else if (add) {
    const k = add.dataset.add;
    const name = prompt('Tool name');
    if (!name || !name.trim()) return;
    (cases[sel][k] = cases[sel][k] || []).push(name.trim());
    renderForm(); note(''); refreshDirty();
  }
});

const uuid = () => (crypto.randomUUID ? crypto.randomUUID()
  : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, ch => {
    const r = crypto.getRandomValues(new Uint8Array(1))[0] % 16;
    return (ch === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  }));

$('add-case').onclick = () => {
  if (!EXP) return;
  const names = new Set(cases.map(c => c.name));
  let n = cases.length + 1, name = `case_${n}`;
  while (names.has(name)) name = `case_${++n}`;
  cases.push({ name, session_id: uuid(), input: '' });
  orig.push(NEW_CASE);
  sel = cases.length - 1;
  renderForm(); refreshDirty();
  note(`added ${name} — fill in its input, then save`);
};

$('del-case').onclick = () => {
  const c = cases[sel];
  if (!c) return;
  if (!confirm(`Delete case ${c.name || '(unnamed)'}? It is removed from the file when you save.`)) return;
  cases.splice(sel, 1); orig.splice(sel, 1);
  sel = Math.max(0, Math.min(sel, cases.length - 1));
  renderForm(); refreshDirty();
  note(`removed ${c.name || '(unnamed)'} — save to write it out`);
};

function refreshDirty() {
  const d = countDirty();
  const el = $('dirty');
  el.textContent = d ? `${plural(d, 'unsaved change')}` : 'No unsaved changes';
  el.classList.toggle('yes', !!d);
  $('save').textContent = d ? `Save (${d})` : 'Save';
  const blocked = !d || !EXP || (MODE === 'server' && READONLY);
  $('save').disabled = blocked;
  $('save').title = (MODE === 'server' && READONLY) ? 'the server was started with --read-only' : '';
  $('revert').disabled = !d;
  $('add-case').disabled = !EXP;
  $('del-case').disabled = !cases.length;
  renderList();
}

/* `to_file` serializes with allow_nan=False, so a non-finite number makes the real write raise. */
function nonFinite(v, path) {
  if (typeof v === 'number') return isFinite(v) ? null : path;
  if (Array.isArray(v)) {
    for (let i = 0; i < v.length; i++) { const r = nonFinite(v[i], `${path}[${i}]`); if (r) return r; }
    return null;
  }
  if (v && typeof v === 'object') {
    for (const k of Object.keys(v)) { const r = nonFinite(v[k], `${path}.${k}`); if (r) return r; }
    return null;
  }
  return null;
}

function preflight() {
  for (const [i, c] of cases.entries()) {
    const bad = nonFinite(c, `cases[${i}]`);
    if (bad) return `${bad} is not finite — the library's writer rejects it`;
    if (c.input == null || String(c.input).trim() === '') {
      return `case ${c.name || i} has no input, which Case requires`;
    }
  }
  const names = cases.map(c => c.name).filter(Boolean);
  const dupe = names.find((n, i) => names.indexOf(n) !== i);
  if (dupe) return `two cases are named ${dupe}; the report would not tell them apart`;
  return null;
}

$('save').onclick = async () => {
  const problem = preflight();
  if (problem) return note(problem, 'err');
  const document_ = { cases, evaluators: evals };

  if (MODE === 'server') {
    /* The editor is inert until the answer arrives, and the answer belongs to the run that asked. */
    const meta = expMeta, editor = $('v-experiment');
    $('save').disabled = true;
    editor.inert = true;
    try {
      const res = await api('/api/experiment?path=' + encodeURIComponent(meta.path), {
        method: 'PUT',
        headers: { 'X-Evals-Token': TOKEN, 'Content-Type': 'application/json' },
        body: JSON.stringify({ document: document_, modified: meta.modified }),
      });
      meta.modified = res.modified;
      if (res.evaluator_notes) meta.evaluator_notes = res.evaluator_notes;
      if (expMeta === meta) {
        markSaved();
        renderEvaluators(); refreshDirty();
        note(`saved ${res.path} · ${plural(res.cases, 'case')}`);
      }
      rescan(true);
    } catch (err) {
      if (expMeta === meta) {
        refreshDirty();
        note(err.message, { 409: 'stale', 404: 'gone' }[err.status] || 'err');
      }
    } finally {
      editor.inert = false;
    }
    return;
  }

  downloadExperiment();
  markSaved();
  refreshDirty();
};

function downloadExperiment() {
  const text = JSON.stringify({ cases, evaluators: evals }, null, 2) + '\n';
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = Object.assign(document.createElement('a'), {
    href: url, download: (expMeta && expMeta.name) || 'experiment.json',
  });
  a.click();
  if (URL.revokeObjectURL) URL.revokeObjectURL(url);
  note(`downloaded ${a.download} · ${plural(cases.length, 'case')} · ${KB(text.length)}`);
}

$('revert').onclick = () => {
  const d = countDirty();
  cases = savedCases.map(s => JSON.parse(s));
  orig = savedCases.slice();
  evals = JSON.parse(origEvals);
  sel = Math.min(sel, Math.max(0, cases.length - 1));
  renderEvaluators(); renderForm(); refreshDirty();
  note(`reverted ${plural(d, 'change')}`);
};

window.addEventListener('beforeunload', e => {
  if (countDirty()) { e.preventDefault(); e.returnValue = ''; }
});
