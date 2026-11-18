'use strict';

/* ══════════ standalone folder reading ══════════ */

const SCAN_DEPTH = 3;

function classify(o) {
  if (!o || typeof o !== 'object' || Array.isArray(o)) return null;
  if (Array.isArray(o.cases) && Array.isArray(o.evaluators)) return 'experiment';
  if (typeof o.overall_score === 'number' && Array.isArray(o.scores)
      && Array.isArray(o.test_passes) && Array.isArray(o.cases)) return 'report';
  return null;
}

/* Same rule as the server's `_stem`: the token only comes off at a separator or the whole name. */
const stem = n => n.replace(/\.json$/i, '').replace(/(^|[._-])(experiment|exp|report|rep)$/i, '');

function reportMeta(obj) {
  const detailed = obj.detailed_results || [], passes = obj.test_passes || [], rows = obj.cases || [];
  let failing = 0, notApplicable = 0;
  rows.forEach((_, i) => {
    const outs = detailed[i] || [];
    const declined = outs.length > 0 && outs.every(o => o && o.label === 'not_applicable' && o.test_pass);
    if (declined) notApplicable++;
    else if (i < passes.length && !passes[i]) failing++;
  });
  return {
    overall_score: obj.overall_score, rows: rows.length, failing, not_applicable: notApplicable,
    evaluators: [...new Set(rows.filter(r => r && typeof r === 'object').map(r => String(r.evaluator)))].sort(),
  };
}

// Same pairing check as the server.
function linkStatus(expObj, repObj) {
  const keys = new Set();
  (expObj.cases || []).forEach(c => {
    if (c && c.session_id != null) keys.add(String(c.session_id));
    if (c && c.name != null) keys.add(String(c.name));
  });
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
    if (event && typeof event === 'object' && !Array.isArray(event)) events.push(event);
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

/* Two record shapes: the optimizer writes fractions at `split.train` and the drawn names
   under `split.cases.train`; earlier records put the names at `split.train` directly. */

function tunedAgainst(split, experiments) {
  const wanted = new Set();
  for (const part of ['train', 'validation', 'test']) {
    for (const id of splitCases(split, part)) {
      if (typeof id === 'string' || typeof id === 'number') wanted.add(String(id));
    }
  }
  if (!wanted.size) return null;

  let best = null;
  for (const exp of [...experiments].sort((a, b) => a.meta.path.localeCompare(b.meta.path))) {
    const expCases = exp.meta.obj.cases || [];
    /* Counts identifiers in common, the same way the server's `tuned_against` does. */
    const have = new Set();
    for (const c of expCases) {
      if (!c || typeof c !== 'object') continue;
      for (const k of ['session_id', 'name']) if (c[k] != null) have.add(String(c[k]));
    }
    let matched = 0;
    wanted.forEach(id => { if (have.has(id)) matched++; });
    if (matched && (!best || matched > best.matched)) {
      best = {
        path: exp.meta.path, matched, split_cases: wanted.size, experiment_cases: expCases.length,
        status: matched === wanted.size ? 'verified' : 'partial',
      };
    }
  }
  return best;
}

/* An optimizer run is a directory holding both a run.json and an events.jsonl. */
function optimizerRunsFrom(configs, logs, transcripts, effects) {
  const runs = [];
  [...configs.keys()].sort().forEach(dir => {
    if (!logs.has(dir)) return;
    const config = configs.get(dir), log = logs.get(dir);
    runs.push({
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
      /* Always a Map for a folder run, even empty: a served run reads transcripts on request. */
      transcripts: transcripts.get(dir) || new Map(),
      fromFolder: true,
      has_effects: effects.has(dir),
      effectsDoc: (effects.get(dir) || {}).obj || null,
      ...logSummary(log.events),
      tuned_against: null,          // filled in by the caller, which holds the experiments
    });
  });
  runs.sort((a, b) => b.modified - a.modified);
  return runs;
}

/* The directories the server's walk leaves out, to the same depth. `rootDepth` is 1 when every path
   starts with the picked folder's own name, as a directory <input> reports them. */
const SKIPPED_DIRS = ['node_modules', '__pycache__'];
function scanned(path, rootDepth) {
  const dirs = path.split('/').slice(rootDepth, -1);
  return dirs.length <= SCAN_DEPTH && !dirs.some(d => d.startsWith('.') || SKIPPED_DIRS.includes(d));
}

/* A run directory's own storage, which the server prunes too. `transcripts/` is held aside below instead. */
const RUN_OWN_DIRS = ['candidates', 'front'];
function withoutRunStorage(entries) {
  const dirsWith = name => new Set(entries.filter(e => e.path.split('/').pop() === name).map(e => dirOf(e.path)));
  const logs = dirsWith('events.jsonl');
  const runs = [...dirsWith('run.json')].filter(dir => logs.has(dir)).map(dir => (dir === '.' ? '' : dir + '/'));
  return entries.filter(e => !runs.some(run => RUN_OWN_DIRS.some(own => e.path.startsWith(run + own + '/'))));
}

async function ingest(all, rootDepth = 0) {
  const entries = withoutRunStorage(all.filter(e => scanned(e.path, rootDepth)));
  const found = [];
  skipped = [];
  const logs = new Map(), configs = new Map(), transcripts = new Map(), effects = new Map();
  const owned = [];                             // `best.values.json` and `result.json`, held aside the same way

  for (const { file, path } of entries) {
    /* `transcripts/<hash>.json` belongs to the run directory above it and is kept as a File, unparsed. */
    const held = /(^|\/)transcripts\/([^/]+)\.json$/i.exec(path);
    if (held) {
      const dir = dirOf(dirOf(path));
      if (!transcripts.has(dir)) transcripts.set(dir, new Map());
      transcripts.get(dir).set(held[2], file);
      continue;
    }
    if (/\.jsonl$/i.test(file.name)) {
      if (file.name.toLowerCase() !== 'events.jsonl') {
        skipped.push({ path, why: 'not an optimizer event log' });
        continue;
      }
      const { events, bad } = readEventLines(await file.text());
      logs.set(dirOf(path), {
        path, events, bad, size: file.size,
        modified: file.lastModified ? file.lastModified / 1000 : 0,
      });
      continue;
    }

    let obj = null, why = null;
    try { obj = JSON.parse(await file.text()); } catch (err) { why = 'invalid JSON: ' + err.message; }
    const kind = why ? null : classify(obj);
    if (!kind) {
      /* Hold a run.json aside rather than skipping it: whether it is an optimizer run config
         depends on a sibling that may not have been read yet. */
      if (!why && obj && typeof obj === 'object' && file.name.toLowerCase() === 'run.json') {
        configs.set(dirOf(path), {
          path, obj, size: file.size,
          modified: file.lastModified ? file.lastModified / 1000 : 0,
        });
        continue;
      }
      /* Same hold-aside for `effects.json` and `best.values.json`. */
      if (!why && obj && file.name.toLowerCase() === 'effects.json') {
        effects.set(dirOf(path), { path, obj });
        continue;
      }
      if (!why && obj && ['best.values.json', 'result.json'].includes(file.name.toLowerCase())) {
        owned.push({ path, dir: dirOf(path) });
        continue;
      }
      skipped.push({ path, why: why || 'not an experiment or a report' });
      continue;
    }
    const meta = {
      path, name: file.name, size: file.size,
      modified: file.lastModified ? file.lastModified / 1000 : 0, obj,
    };
    if (kind === 'experiment') Object.assign(meta, { cases: (obj.cases || []).length, evaluators: obj.evaluators || [] });
    else Object.assign(meta, reportMeta(obj));
    found.push({ kind, meta, dir: dirOf(path) });
  }

  const optimizerRuns = optimizerRunsFrom(configs, logs, transcripts, effects);
  const experiments = found.filter(f => f.kind === 'experiment');
  optimizerRuns.forEach(run => {
    run.tuned_against = tunedAgainst((run.config || {}).split, experiments);
  });
  const paired = new Set(optimizerRuns.map(r => r.dir));
  configs.forEach((config, dir) => {
    if (!paired.has(dir)) skipped.push({ path: config.path, why: 'not an experiment or a report' });
  });
  logs.forEach((log, dir) => {
    if (!paired.has(dir)) skipped.push({ path: log.path, why: 'no run.json beside it' });
  });
  effects.forEach((held, dir) => {
    if (!paired.has(dir)) skipped.push({ path: held.path, why: 'not an experiment or a report' });
  });
  owned.forEach(held => {
    if (!paired.has(held.dir)) skipped.push({ path: held.path, why: 'not an experiment or a report' });
  });

  const byDir = new Map();
  found.forEach(f => { if (!byDir.has(f.dir)) byDir.set(f.dir, []); byDir.get(f.dir).push(f); });

  const runs = [];
  [...byDir.keys()].sort().forEach(dir => {
    const files = byDir.get(dir);
    const exps = files.filter(f => f.kind === 'experiment');
    const reps = files.filter(f => f.kind === 'report');
    const taken = new Set();
    exps.forEach(e => {
      let m = reps.find(r => !taken.has(r) && stem(r.meta.name) === stem(e.meta.name));
      if (!m && exps.length === 1 && reps.length === 1) m = reps[0];
      if (m) taken.add(m);
      runs.push({
        id: e.meta.path, dir, experiment: e.meta, report: m ? m.meta : null,
        link: m ? linkStatus(e.meta.obj, m.meta.obj) : null,
      });
    });
    reps.filter(r => !taken.has(r)).forEach(r =>
      runs.push({ id: r.meta.path, dir, experiment: null, report: r.meta, link: null }));
  });
  runs.sort((a, b) => Math.max((b.experiment || {}).modified || 0, (b.report || {}).modified || 0)
    - Math.max((a.experiment || {}).modified || 0, (a.report || {}).modified || 0));
  return { runs, optimizerRuns };
}

async function pickFolder() {
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
    folderLabel = dir.name;
    showingSample = false;
    const found = await ingest(entries);
    RUNS = found.runs;
    selRun = -1;
    renderRuns();
    await loadOptimizerRuns(found.optimizerRuns);
    show('runs');
    return;
  }
  // showDirectoryPicker needs a secure context, which file:// is not: fall back to a directory <input>.
  $('dirinput').click();
}

app.querySelector('[data-act="pick"]').onclick = pickFolder;

$('dirinput').onchange = async e => {
  const files = [...e.target.files].filter(f => /\.jsonl?$/i.test(f.name));
  const first = (files[0] && files[0].webkitRelativePath) || '';
  folderLabel = first.includes('/') ? first.slice(0, first.indexOf('/')) : 'selected folder';
  showingSample = false;
  const rooted = first.includes('/') && files.every(f => (f.webkitRelativePath || '').startsWith(folderLabel + '/'));
  const found = await ingest(files.map(f => ({ file: f, path: f.webkitRelativePath || f.name })), rooted ? 1 : 0);
  RUNS = found.runs;
  selRun = -1;
  renderRuns();
  await loadOptimizerRuns(found.optimizerRuns);
  show('runs');
};
