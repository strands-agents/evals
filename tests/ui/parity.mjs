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
  'split-source/experiment.json': experiment('train_case_01', 'train_case_02', 'val_case_01', 'test_case_01'),
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
const normalize = ({ runs, skipped, optimizer }) => ({
  runs: runs.map(r => ({
    id: r.id, dir: r.dir,
    experiment: pick(r.experiment, ['path', 'name', 'cases', 'evaluators']),
    report: pick(r.report, ['path', 'name', 'overall_score', 'rows', 'failing', 'not_applicable', 'evaluators']),
    link: pick(r.link, ['status', 'detail']),
  })).sort((a, b) => a.id.localeCompare(b.id)),
  /* The two JSON parsers word a syntax error differently; that it was one is what must agree. */
  skipped: skipped.map(s => ({ path: s.path, why: s.why.replace(/^(invalid JSON).*/, '$1') }))
    .sort((a, b) => a.path.localeCompare(b.path)),
  optimizer: optimizer.map(r => pick(r, ['id', 'dir', 'run_id', 'name', 'objective', 'constraints', 'is_pareto',
    'budget', 'events_path', 'events', 'iterations', 'candidates', 'trials', 'champion_id', 'result', 'stop_reason',
    'bad_lines', 'truncated', 'tuned_against'])).sort((a, b) => a.id.localeCompare(b.id)),
});

try {
  const served = await (await fetch(`http://127.0.0.1:${port}/api/runs`, { headers: { 'X-Evals-Token': TOKEN } })).json();
  const p = await page();
  await p.openFolder(picked);
  const folder = JSON.parse(p.window.eval('JSON.stringify({ runs: RUNS, skipped, optimizer: OPT_RUNS })'));

  const server = normalize({ runs: served.runs, skipped: served.skipped, optimizer: served.optimizer_runs });
  const client = normalize(folder);
  ok(server.runs.length >= 8 && server.optimizer.length === 1, 'the tree exercises pairing and a run directory',
    `${server.runs.length} runs, ${server.optimizer.length} optimizer runs`);
  const key = entry => entry.id || entry.path;
  for (const part of ['runs', 'skipped', 'optimizer']) {
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
