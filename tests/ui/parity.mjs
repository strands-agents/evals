/* The server's discovery (`_ui_server.py`) and the page's no-server folder reader (`ui-folder.js`)
 * are two implementations of one contract. This runs both over the same tree and compares them. */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { page, runner } from './harness.mjs';

const { ok, done } = runner('parity');
const PY = process.env.PYTHON || 'python3';
const TOKEN = 'parity-token';
const FIXTURE = new URL('./fixtures/optimizer-run/', import.meta.url).pathname;

const cases = (...names) => names.map(name => ({ name, session_id: `sid-${name}`, input: `ask ${name}` }));
const experiment = (...names) => ({ cases: cases(...names), evaluators: [{ evaluator_type: 'OutputEvaluator', rubric: 'r' }] });
function report(names, { failing = [], declined = [] } = {}) {
  const rows = names.map(name => ({ name, session_id: `sid-${name}`, evaluator: 'OutputEvaluator' }));
  const verdict = name => declined.includes(name)
    ? { score: 0, test_pass: true, label: 'not_applicable' }
    : { score: failing.includes(name) ? 0 : 1, test_pass: !failing.includes(name), label: 'judged' };
  return {
    overall_score: 0.5, cases: rows,
    scores: names.map(name => verdict(name).score),
    test_passes: names.map(name => verdict(name).test_pass),
    detailed_results: names.map(name => [verdict(name)]),
  };
}

/* A recorded session: `seconds` is what each trace took, in whole milliseconds, added up. */
const stamp = seconds => new Date(Date.UTC(2026, 0, 5, 9) + seconds * 1000).toISOString().replace('Z', '456Z');
const span = (id, start, end) => ({ span_type: 'inference', messages: [], span_info: { span_id: id, session_id: 's', start_time: stamp(start), end_time: stamp(end) } });
const session = (id, ...traces) => ({ session_id: id, traces: traces.map((spans, i) => ({ trace_id: `${id}-${i}`, session_id: id, spans })) });
const diagnosis = id => ({ session_id: id, failures: [{ span_id: 'a', category: ['x'], confidence: [0.5], evidence: ['e'] }], root_causes: [] });
const taskResult = name => ({ input: name, actual_output: 'out', name, expected_output: null, actual_trajectory: null });

/* One file per rule the two sides must agree on. */
const TREE = {
  'paired/experiment.json': experiment('a', 'b', 'c'),
  'paired/report.json': report(['a', 'b', 'c'], { failing: ['b'], declined: ['c'] }),
  'stems/nightly.experiment.json': experiment('n1'),
  'stems/nightly.report.json': report(['n1']),
  'stems/smoke.json': experiment('s1'),
  'stems/myexperiment.json': experiment('m1'),
  'mismatch/experiment.json': experiment('x'),
  'mismatch/report.json': report(['y', 'z']),
  'partial/experiment.json': experiment('p1', 'p2'),
  'partial/report.json': report(['p1', 'gone']),
  'orphan/report.json': report(['o1']),
  'many/suite.json': experiment('q1', 'q2'),
  'many/other.json': experiment('r1'),
  'many/suite.report.json': report(['q1', 'q2']),
  'many/monday.report.json': report(['q1']),
  'near/suite.json': experiment('k1'),
  'near/runs/k.report.json': report(['k1']),
  'far/copy.json': experiment('k1'),
  'tie/left/e.json': experiment('t1'),
  'tie/right/e.json': experiment('t1'),
  'tie/shared.report.json': report(['t1']),
  'split-source/experiment.json': experiment('train_case_01', 'train_case_02', 'val_case_01', 'test_case_01'),
  'Zsplit/experiment.json': experiment('train_case_01', 'train_case_02', 'val_case_01', 'test_case_01'),
  'odd/experiment.json': { cases: cases('odd1'), evaluators: ['not an object', { evaluator_type: 'OutputEvaluator', rubric: 'r' }] },
  'odd/report.json': { ...report(['odd1']), detailed_results: ['not a list'] },
  'recorded/any-name.json': session('s1', [span('a', 0, 1.2345), span('b', 2, 3.5)], [span('c', 600, 600.25)], []),
  'recorded/s1.diagnosis.json': diagnosis('s1'),
  'recorded/untimed.json': session('s2', [{ span_type: 'inference', messages: [] }]),
  'recorded/elsewhere/s9.diagnosis.json': diagnosis('s9'),
  'recorded/causes.json': { root_causes: [{ failure_span_id: 'a', location: 'b' }] },
  'recorded/no-spans.session.json': { session_id: 's3', traces: [{ trace_id: 't' }] },
  'elsewhere/transcripts/s4.json': session('s4', [span('a', 0, 1)]),
  'twice/left/s.json': session('twin', [span('a', 0, 1)]),
  'twice/right/s.json': session('twin', [span('a', 0, 1)]),
  'twice/twin.diagnosis.json': diagnosis('twin'),
  'cached/a.json': taskResult('a'),
  'cached/b.json': taskResult('b'),
  'cached/c.report.json': report(['c1']),
  'cached/z-notes.json': { input: 'opens like a result, in a store' },
  'loose/row.json': { input: 'opens like a result, and is not one' },
  'notes.json': { just: 'notes' },
  'broken.json': '{"cases": [',
  'a/b/c/too-deep/experiment.json': experiment('deep'),
  'node_modules/pkg/experiment.json': experiment('dependency'),
  '__pycache__/experiment.json': experiment('cache'),
  '.hidden/experiment.json': experiment('hidden'),
};

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ui-parity-'));
for (const [rel, content] of Object.entries(TREE)) {
  fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
  fs.writeFileSync(path.join(root, rel), typeof content === 'string' ? content : JSON.stringify(content));
}
/* A run directory with everything a real run leaves beside its log. */
fs.cpSync(FIXTURE, path.join(root, 'tuning'), { recursive: true });
fs.writeFileSync(path.join(root, 'tuning/result.json'), JSON.stringify({ champion: 'c10' }));
fs.writeFileSync(path.join(root, 'tuning/best.values.json'), JSON.stringify({ temperature: 0.2 }));
fs.mkdirSync(path.join(root, 'tuning/candidates/c1'), { recursive: true });
fs.writeFileSync(path.join(root, 'tuning/candidates/c1/agent.json'), JSON.stringify({ copied: true }));

/* What a folder picker hands the page: every .json and .jsonl under the root. */
const picked = {};
(function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (/\.jsonl?$/i.test(entry.name)) picked[path.relative(root, full).split(path.sep).join('/')] = fs.readFileSync(full, 'utf8');
  }
})(root);

const port = await new Promise(resolve => {
  const probe = net.createServer();
  probe.listen(0, '127.0.0.1', () => { const { port } = probe.address(); probe.close(() => resolve(port)); });
});
const proc = spawn(PY, [new URL('./serve.py', import.meta.url).pathname, root, String(port), TOKEN], { stdio: ['ignore', 'pipe', 'inherit'] });
await new Promise((resolve, reject) => {
  proc.stdout.on('data', chunk => String(chunk).includes('ready') && resolve());
  proc.on('exit', code => reject(new Error('server exited ' + code)));
});

/* Only what both sides decide. Sizes, mtimes and in-memory documents differ by construction. */
const pick = (obj, keys) => obj && Object.fromEntries(keys.map(k => [k, obj[k] === undefined ? null : obj[k]]));
const byPath = (a, b) => a.path.localeCompare(b.path);
const normalize = ({ experiments, reports, skipped, optimizer, sessions, diagnoses, stores }) => ({
  sessions: sessions.map(s => ({ ...pick(s, ['path', 'name', 'dir', 'session_id', 'traces', 'spans', 'seconds']), diagnoses: [...s.diagnoses].sort() }))
    .sort(byPath),
  diagnoses: diagnoses.map(d => pick(d, ['path', 'name', 'dir', 'session_id', 'failures', 'root_causes', 'session'])).sort(byPath),
  stores: stores.map(s => ({ ...pick(s, ['dir', 'files']), path: s.dir })).sort(byPath),
  /* Each side lists newest first by its own clock, so the order is not compared. */
  experiments: experiments.map(e => ({ ...pick(e, ['path', 'name', 'dir', 'cases', 'evaluators']), reports: [...e.reports].sort() }))
    .sort(byPath),
  reports: reports.map(r => ({
    ...pick(r, ['path', 'name', 'dir', 'overall_score', 'rows', 'failing', 'not_applicable', 'evaluators', 'experiment']),
    link: pick(r.link, ['status', 'detail']),
  })).sort(byPath),
  /* The two JSON parsers word a syntax error differently; that it was one is what must agree. */
  skipped: skipped.map(s => ({ path: s.path, why: s.why.replace(/^(invalid JSON).*/, '$1') }))
    .sort(byPath),
  optimizer: optimizer.map(r => pick(r, ['id', 'dir', 'run_id', 'name', 'objective', 'constraints', 'is_pareto',
    'budget', 'events_path', 'events', 'iterations', 'candidates', 'trials', 'champion_id', 'result', 'stop_reason',
    'bad_lines', 'truncated', 'tuned_against'])).sort((a, b) => a.id.localeCompare(b.id)),
});

try {
  const served = await (await fetch(`http://127.0.0.1:${port}/api/scan`, { headers: { 'X-Evals-Token': TOKEN } })).json();
  const p = await page();
  await p.openFolder(picked);
  const folder = JSON.parse(p.window.eval('JSON.stringify({ experiments: EXPERIMENTS, reports: REPORTS, skipped, optimizer: OPT_RUNS,'
    + ' sessions: SESSIONS, diagnoses: DIAGNOSES, stores: STORES })'));

  const server = normalize({ ...served, optimizer: served.optimizer_runs });
  const client = normalize(folder);
  ok(server.experiments.length >= 12 && server.reports.length >= 9 && server.optimizer.length === 1,
    'the tree exercises pairing and a run directory',
    `${server.experiments.length} experiments, ${server.reports.length} reports, ${server.optimizer.length} optimizer runs`);
  const of = Object.fromEntries(server.reports.map(r => [r.path, r.experiment]));
  const paired = ['many/monday.report.json', 'near/runs/k.report.json', 'tie/shared.report.json', 'orphan/report.json'].map(r => of[r]).join();
  ok(paired === 'many/suite.json,near/suite.json,,', 'by cases a report goes to the one experiment that holds them, the nearer of two, and to none on a tie',
    paired);
  const recorded = Object.fromEntries(server.sessions.map(s => [s.path, s]));
  ok(recorded['recorded/any-name.json'].seconds === 3.75 && recorded['recorded/untimed.json'].seconds === null
    && recorded['recorded/any-name.json'].diagnoses.join() === 'recorded/s1.diagnosis.json',
  'a session is timed trace by trace and holds the diagnosis made of it', JSON.stringify(recorded['recorded/any-name.json']));
  const lone = server.diagnoses.filter(d => d.session === null).map(d => d.path).join();
  ok(lone === 'recorded/causes.json,recorded/elsewhere/s9.diagnosis.json,twice/twin.diagnosis.json',
    'a diagnosis stands alone when no session has its id, when it names none, and when two equally near do', lone);
  ok(JSON.stringify(server.stores) === '[{"dir":"cached","files":3,"path":"cached"}]',
    'a store counts its results and what opens like one after them, and a look-alike elsewhere is not a store', JSON.stringify(server.stores));
  const key = entry => entry.id || entry.path;
  for (const part of ['experiments', 'reports', 'skipped', 'optimizer', 'sessions', 'diagnoses', 'stores']) {
    const keys = list => list.map(key).join(' ');
    ok(keys(server[part]) === keys(client[part]), `${part}: both sides find the same entries`,
      `\n  server: ${keys(server[part])}\n  page:   ${keys(client[part])}`);
    for (const entry of server[part]) {
      const other = client[part].find(x => key(x) === key(entry));
      if (!other) continue;
      for (const field of Object.keys(entry)) {
        ok(JSON.stringify(entry[field]) === JSON.stringify(other[field]), `${part} ${key(entry)}: ${field} agrees`,
          `\n  server: ${JSON.stringify(entry[field])}\n  page:   ${JSON.stringify(other[field])}`);
      }
    }
  }
  done(p.errs);
} finally {
  proc.kill();
  fs.rmSync(root, { recursive: true, force: true });
}
