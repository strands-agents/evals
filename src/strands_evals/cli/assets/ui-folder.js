'use strict';

/* ══════════ standalone folder reading ══════════ */

const SCAN_DEPTH = 3;

function classify(o) {
  if (!isObject(o)) return null;
  if (Array.isArray(o.cases) && Array.isArray(o.evaluators)) return 'experiment';
  if (typeof o.overall_score === 'number' && Array.isArray(o.scores)
      && Array.isArray(o.test_passes) && Array.isArray(o.cases)) return 'report';
  if (typeof o.session_id === 'string' && Array.isArray(o.traces)) {
    return o.traces.every(t => t && typeof t === 'object' && Array.isArray(t.spans)) ? 'session' : null;
  }
  const on = (list, key) => Array.isArray(list) && list.every(x => x && typeof x === 'object' && typeof x[key] === 'string');
  const found = on(o.failures, 'span_id'), explained = on(o.root_causes, 'failure_span_id');
  if (found && typeof o.session_id === 'string' && (o.root_causes == null || explained)) return 'diagnosis';
  /* `--rca-only` writes root causes alone, with no session named. */
  if (o.failures == null && explained && o.root_causes.length) return 'diagnosis';
  return null;
}

/* Same rule as the server's `_stem`: the token only comes off at a separator or the whole name. */
const stem = n => n.replace(/\.json$/i, '').replace(/(^|[._-])(experiment|exp|report|rep)$/i, '');
/* The server orders paths by code point, which `localeCompare` does not. */
const byPath = (a, b) => (a.meta.path < b.meta.path ? -1 : 1);
const newestFirst = (a, b) => b.meta.modified - a.meta.modified || byPath(a, b);

function reportMeta(obj) {
  const detailed = obj.detailed_results || [], passes = obj.test_passes || [], rows = obj.cases || [];
  let failing = 0, notApplicable = 0;
  rows.forEach((_, i) => {
    const outs = Array.isArray(detailed[i]) ? detailed[i] : [];
    const declined = outs.length > 0 && outs.every(o => o && o.label === 'not_applicable' && o.test_pass);
    if (declined) notApplicable++;
    else if (!passes[i]) failing++;
  });
  return {
    overall_score: obj.overall_score, rows: rows.length, failing, not_applicable: notApplicable,
    evaluators: [...new Set(rows.filter(r => r && typeof r === 'object').map(r => String(r.evaluator)))].sort(),
  };
}

/* How many directory steps separate two folders. */
function stepsApart(left, right) {
  const a = left === '.' ? [] : left.split('/'), b = right === '.' ? [] : right.split('/');
  let shared = 0;
  while (shared < Math.min(a.length, b.length) && a[shared] === b[shared]) shared++;
  return a.length + b.length - 2 * shared;
}

/* The experiment a report was made from, or null when that cannot be told. Mirrors `experiment_of`
   in `_ui_server.py`: beside it under the same name, else the nearest one holding all its cases. */
function experimentOf(report, experiments, alone) {
  const beside = experiments.filter(e => e.dir === report.dir).sort(byPath);
  const named = beside.filter(e => stem(e.meta.name) === stem(report.meta.name));
  if (named.length === 1) return named[0];
  if (alone && beside.length === 1) return beside[0];
  const holding = experiments.filter(e => linkStatus(e.meta.obj, report.meta.obj).status === 'verified');
  const nearest = Math.min(...holding.map(e => stepsApart(e.dir, report.dir)));
  const closest = holding.filter(e => stepsApart(e.dir, report.dir) === nearest);
  return closest.length === 1 ? closest[0] : null;
}

/* Every identifier a case of an experiment can be referred to by: report rows prefer `session_id`,
   optimizer splits hold whichever the run was given. */
function caseIds(experiment) {
  const ids = new Set();
  for (const c of experiment.cases || []) {
    if (!c || typeof c !== 'object') continue;
    for (const key of ['session_id', 'name']) if (c[key] != null) ids.add(String(c[key]));
  }
  return ids;
}

// Same pairing check as the server.
function linkStatus(expObj, repObj) {
  const keys = caseIds(expObj);
  const rows = new Set();
  (repObj.cases || []).forEach(r => {
    const k = r && (r.session_id || r.name);
    if (k != null) rows.add(String(k));
  });
  if (!keys.size || !rows.size) return { status: 'unverified', detail: 'no case identifiers to compare' };
  const missing = [...rows].filter(k => !keys.has(k));
  if (!missing.length) return { status: 'verified', detail: `all ${rows.size} report case(s) exist in this experiment` };
  if (missing.length === rows.size) return { status: 'mismatch', detail: `none of the ${rows.size} report case(s) appear in this experiment` };
  return { status: 'partial', detail: `${missing.length} of ${rows.size} report case(s) are not in this experiment` };
}

/* JSON Lines. Kept in step with `read_event_log` in `_ui_server.py`. */
function readEventLines(text) {
  const lines = String(text).split('\n');
  const complete = lines.length - 1;
  const events = [], bad = [];
  lines.forEach((line, i) => {
    const number = i + 1;
    if (!line.trim()) return;
    let event;
    try {
      event = JSON.parse(line);
    } catch (err) {
      if (number > complete) return;        // partial final line of a live log
      bad.push({ line: number, why: 'invalid JSON: ' + err.message });
      return;
    }
    if (isObject(event)) events.push(event);
    else bad.push({ line: number, why: 'not a JSON object' });
  });
  return { events, bad };
}

const dirOf = path => path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '.';

/* Must match the summary the server computes for the same rows. */
function logSummary(events) {
  const iterations = new Set();
  const endReports = {};
  let champion = null, stop = null, candidates = 0, trials = 0;
  for (const e of events) {
    switch (e.event) {
      case 'iteration_end': iterations.add(e.iteration); break;
      case 'propose': candidates++; break;
      case 'trial': trials++; break;
      case 'picked': case 'test':
        if (e.report && e.report.champion_id) champion = e.report.champion_id;
        if (e.report) endReports[e.event] = e.report;
        break;
      case 'stop': stop = e.reason || null; break;
    }
  }
  /* `test` is the held-out split, `picked` the validation split. Same precedence as the result card. */
  const endReport = endReports.test || endReports.picked || null;
  const objective = endReport && endReport.objective;
  return {
    iterations: iterations.size, candidates, trials,
    champion_id: champion, stop_reason: stop,
    result: objective && typeof objective === 'object'
      ? { objective, held_out: !!endReports.test } : null,
  };
}

function tunedAgainst(split, experiments) {
  const wanted = new Set();
  for (const part of ['train', 'validation', 'test']) {
    for (const id of splitCases(split, part)) {
      if (typeof id === 'string' || typeof id === 'number') wanted.add(String(id));
    }
  }
  if (!wanted.size) return null;

  /* The largest overlap wins, and a tie goes to the first path, as in the server's `tuned_against`. */
  let best = null;
  for (const exp of [...experiments].sort(byPath)) {
    const have = caseIds(exp.meta.obj);
    const matched = [...wanted].filter(id => have.has(id)).length;
    if (matched && (!best || matched > best.matched)) {
      best = {
        path: exp.meta.path, matched, split_cases: wanted.size, experiment_cases: exp.meta.obj.cases.length,
        status: matched === wanted.size ? 'verified' : 'partial',
      };
    }
  }
  return best;
}

/* An optimizer run is a directory holding both a run.json and an events.jsonl. */
function runDirs(entries) {
  const dirsWith = name => new Set(entries.filter(e => e.path.split('/').pop() === name).map(e => dirOf(e.path)));
  const logs = dirsWith('events.jsonl');
  return new Set([...dirsWith('run.json')].filter(dir => logs.has(dir)));
}

/* What a run directory keeps for itself, which the server leaves out of its walk. `effects.json` is read instead. */
const RUN_OWN_DIRS = ['transcripts', 'candidates', 'front'];
const RUN_OWN_FILES = ['best.values.json', 'result.json'];

/* `{ run, rest }` when `path` is a run directory's own storage: the run, and the path inside it. */
function runStorage(path, runs) {
  for (const run of runs) {
    const rest = run === '.' ? path : path.startsWith(run + '/') ? path.slice(run.length + 1) : null;
    if (rest != null && (RUN_OWN_DIRS.some(own => rest.startsWith(own + '/')) || RUN_OWN_FILES.includes(rest.toLowerCase()))) {
      return { run, rest };
    }
  }
  return null;
}

function optimizerRunsFrom(configs, logs, transcripts, effects, experiments) {
  const runs = [...configs.keys()].sort().map(dir => {
    const config = configs.get(dir), log = logs.get(dir);
    return {
      id: dir,
      dir,
      run_id: config.obj.run_id || (dir === '.' ? 'this folder' : dir.slice(dir.lastIndexOf('/') + 1)),
      name: config.obj.name || null,
      objective: config.obj.objective || null,
      constraints: config.obj.constraints || [],
      is_pareto: configFacts(config.obj).isPareto,
      budget: config.obj.budget || {},
      config: config.obj,
      events_path: log.path,
      events: log.events.length,
      events_size: log.size,
      modified: Math.max(config.modified, log.modified),
      bad_lines: log.bad,
      truncated: false,
      eventsCache: log.events,
      /* Held as Files: a served run reads a transcript on request instead. */
      transcripts: transcripts.get(dir),
      fromFolder: true,
      has_effects: effects.has(dir),
      effectsDoc: effects.get(dir) || null,
      ...logSummary(log.events),
      tuned_against: tunedAgainst(config.obj.split, experiments),
    };
  });
  return runs.sort((a, b) => b.modified - a.modified);
}

/* The directories the server's walk leaves out, to the same depth. `rootDepth` is 1 when every path
   starts with the picked folder's own name, as a directory <input> reports them. */
const SKIPPED_DIRS = ['node_modules', '__pycache__'];
function scanned(path, rootDepth) {
  const dirs = path.split('/').slice(rootDepth, -1);
  return dirs.length <= SCAN_DEPTH && !dirs.some(d => d.startsWith('.') || SKIPPED_DIRS.includes(d));
}

const UNRECOGNISED = 'not an experiment or a report';

async function ingest(all, rootDepth = 0) {
  /* In path order, so a directory's files are read in the order the server reads them. */
  const entries = all.filter(e => scanned(e.path, rootDepth))
    .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  const runs = runDirs(entries);
  const found = [], skipped = [];
  const stores = new Map();                     // directory -> the saved task results it holds
  const logs = new Map(), configs = new Map(), effects = new Map();
  const transcripts = new Map([...runs].map(run => [run, new Map()]));
  const modified = file => (file.lastModified ? file.lastModified / 1000 : 0);

  for (const { file, path } of entries) {
    const own = runStorage(path, runs);
    if (own) {
      /* `transcripts/<hash>.json` is kept as a File, unparsed, until a trial that names it is opened. */
      const held = /^transcripts\/([^/]+)\.json$/i.exec(own.rest);
      if (held) transcripts.get(own.run).set(held[1], file);
      continue;
    }
    const here = dirOf(path), name = file.name.toLowerCase();
    if (name.endsWith('.jsonl')) {
      if (name !== 'events.jsonl') continue;
      if (!runs.has(here)) skipped.push({ path, why: 'no run.json beside it' });
      else logs.set(here, { path, size: file.size, modified: modified(file), ...readEventLines(await file.text()) });
      continue;
    }

    const text = await file.text();
    /* One result read in full shows the directory is a store; the rest are told by how they open. */
    if (stores.has(here) && TASK_RESULT_HEAD.test(text)) {
      stores.set(here, stores.get(here) + 1);
      continue;
    }
    let obj = null, why = null;
    try { obj = JSON.parse(text); } catch (err) { why = 'invalid JSON: ' + err.message; }
    if (runs.has(here) && name === 'effects.json') {
      effects.set(here, obj);
      continue;
    }
    const kind = why ? null : classify(obj);
    if (!kind && !why && isTaskResult(obj)) {
      stores.set(here, (stores.get(here) || 0) + 1);
      continue;
    }
    if (!kind && runs.has(here) && name === 'run.json' && isObject(obj)) {
      configs.set(here, { obj, modified: modified(file) });
      continue;
    }
    if (!kind) {
      skipped.push({ path, why: why || UNRECOGNISED });
      continue;
    }
    const meta = { path, name: file.name, size: file.size, modified: modified(file), obj };
    if (kind === 'experiment') Object.assign(meta, { cases: obj.cases.length, evaluators: obj.evaluators.filter(isObject) });
    else if (kind === 'report') Object.assign(meta, reportMeta(obj));
    else Object.assign(meta, kind === 'session' ? sessionMeta(obj) : diagnosisMeta(obj));
    found.push({ kind, meta, dir: here });
  }

  const filesOf = kind => found.filter(f => f.kind === kind).sort(newestFirst);
  const experiments = filesOf('experiment'), made = new Map();
  const reports = filesOf('report').map(r => {
    const alone = found.filter(f => f.kind === 'report' && f.dir === r.dir).length === 1;
    const e = experimentOf(r, experiments, alone);
    if (e) made.set(e.meta.path, (made.get(e.meta.path) || []).concat(r.meta.path));
    return Object.assign(r.meta, {
      dir: r.dir, experiment: e ? e.meta.path : null, link: e ? linkStatus(e.meta.obj, r.meta.obj) : null,
    });
  });
  return {
    experiments: experiments.map(e => Object.assign(e.meta, { dir: e.dir, reports: made.get(e.meta.path) || [] })),
    reports, skipped, ...sessionsFrom(found),
    optimizerRuns: optimizerRunsFrom(configs, logs, transcripts, effects, experiments),
    stores: [...stores].sort(([a], [b]) => (a < b ? -1 : 1)).map(([dir, files]) => ({ dir, files })),
  };
}

/* The folder replaces whatever was open, so nothing from before it may stay on screen. */
async function showFolder(label, entries, rootDepth) {
  const found = await ingest(entries, rootDepth);
  folderLabel = label;
  showingSample = false;
  EXPERIMENTS = found.experiments;
  REPORTS = found.reports;
  skipped = found.skipped;
  clearExperiment(); clearReport(); closeSession();
  setSessions(found);
  renderLists();
  await loadOptimizerRuns(found.optimizerRuns);
  show(homeView());
}

async function pickFolder() {
  if (!await mayDiscard('open a folder')) return;
  if (window.showDirectoryPicker) {
    let dir;
    try { dir = await window.showDirectoryPicker(); } catch { return; }   // cancelled
    const entries = [];
    const walk = async (handle, prefix, depth) => {
      for await (const [name, h] of handle.entries()) {
        if (h.kind === 'file') {
          if (/\.jsonl?$/i.test(name)) entries.push({ file: await h.getFile(), path: prefix + name });
        } else if (depth < SCAN_DEPTH && scanned(prefix + name + '/x', 0)) {
          await walk(h, prefix + name + '/', depth + 1);
        }
      }
    };
    await walk(dir, '', 0);
    return showFolder(dir.name, entries);
  }
  // showDirectoryPicker needs a secure context, which file:// is not: fall back to a directory <input>.
  $('dirinput').click();
}

app.querySelector('[data-act="pick"]').onclick = pickFolder;

$('dirinput').onchange = async e => {
  const files = [...e.target.files].filter(f => /\.jsonl?$/i.test(f.name));
  const first = (files[0] && files[0].webkitRelativePath) || '';
  const label = first.includes('/') ? first.slice(0, first.indexOf('/')) : 'selected folder';
  const rooted = first.includes('/') && files.every(f => (f.webkitRelativePath || '').startsWith(label + '/'));
  await showFolder(label, files.map(f => ({ file: f, path: f.webkitRelativePath || f.name })), rooted ? 1 : 0);
};
