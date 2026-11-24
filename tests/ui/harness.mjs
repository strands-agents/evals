import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { JSDOM } from 'jsdom';

const ASSET = process.env.UI_HTML
  ? new URL(process.env.UI_HTML, `file://${process.cwd()}/`)
  : new URL('../../src/strands_evals/cli/assets/ui.html', import.meta.url);
/* Inlines the stylesheet and scripts beside the page, the way `load_asset` does for the server. */
const beside = name => fs.readFileSync(new URL(name, ASSET), 'utf8');
const HTML = fs.readFileSync(ASSET, 'utf8')
  .replace(/<link rel="stylesheet" href="([\w.-]+)">/g, (_, name) => `<style>\n${beside(name)}</style>`)
  .replace(/<script src="([\w.-]+)"><\/script>/g, (_, name) => `<script>\n${beside(name)}</script>`);

/* Every page built. When the suite reports, each one's errors count against it and its window is closed:
   a page polls on a timer, which would keep Node alive. */
const PAGES = [];

/* What the pages are waiting on: requests in flight, and steps through history the browser has not taken yet. */
const PENDING = new Set();
function track(work) {
  const waiting = {};
  PENDING.add(waiting);
  const over = () => PENDING.delete(waiting);
  work.then(over, over);
  return work;
}
/* The next step through history, which jsdom takes a few timers later. */
const arrival = window => new Promise(resolve => window.addEventListener('popstate', resolve, { once: true }));

/* Requests to the real server. They always answer, so they are waited for however slow the machine is. */
const REAL = new Set();
/* `UI_TEST_DELAY=<ms>` holds each of them back, to run the suites as a slow machine would. */
const DELAY = Number(process.env.UI_TEST_DELAY) || 0;

/** Ask the real server at `base`. The answer is handed over with its body already read, so the page's
    own read of it is not a second wait that nothing tracks. */
export function ask(base, url, opts = {}) {
  const work = (async () => {
    if (DELAY) await new Promise(resolve => setTimeout(resolve, DELAY));
    const got = await fetch(base + url, opts);
    const body = [204, 304].includes(got.status) ? null : await got.arrayBuffer();
    return new Response(body, { status: got.status, statusText: got.statusText, headers: got.headers });
  })();
  REAL.add(work);
  const over = () => REAL.delete(work);
  work.then(over, over);
  return work;
}

/* Turns of the event loop until nothing is pending. Work on a fake server that is still pending after
   `STUCK_TURNS` turns and `STUCK_MS` is being held by the test on purpose, and is not waited for again. */
const STUCK_TURNS = 40, STUCK_MS = 300;
async function settled() {
  const began = Date.now();
  for (let quiet = 0, turns = 0; quiet < 3; turns += 1) {
    await new Promise(resolve => setTimeout(resolve, 0));
    if (!PENDING.size && !REAL.size) quiet += 1;
    else if (!REAL.size && turns >= STUCK_TURNS && Date.now() - began >= STUCK_MS) PENDING.clear();
    else quiet = 0;
  }
}

/** Let the page act on what was just done to it. The wait ends when its work has, not after a fixed
    time, so a busy machine only makes it longer. */
export async function tick(n = 3) {
  await new Promise(resolve => setTimeout(resolve, n * 5));
  await settled();
}

/** Poll until `cond()` holds; real HTTP round trips outlast a fixed tick. */
export async function waitFor(cond, what = 'condition', ms = 5000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (cond()) return true;
    await tick(2);
  }
  throw new Error('timed out waiting for ' + what);
}

/** Build the page. `fetchImpl` is installed before any script runs, so boot() sees it.
    `cookie` is one the browser already holds, as from an earlier visit. `visible` makes the
    document one that is being looked at; without it the page's polling timer does nothing. */
export async function page({ fetchImpl, url = 'http://127.0.0.1:9999/?t=tok', token, cookie, visible } = {}) {
  const errs = [];
  /* `token` stands in for the server writing it into the document. */
  const html = token ? HTML.replace('name="strands-evals-token" content=""', `name="strands-evals-token" content="${token}"`) : HTML;
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: !!visible,
    url,
    beforeParse(window) {
      if (fetchImpl) {
        window.fetch = (...a) => {
          const answer = fetchImpl(...a);
          track(Promise.resolve(answer));
          return answer;
        };
      }
      for (const step of ['back', 'forward', 'go']) {
        const take = window.history[step].bind(window.history);
        window.history[step] = (...a) => { track(arrival(window)); return take(...a); };
      }
      if (cookie) window.document.cookie = cookie;
      // jsdom has no download: keep the blob the page made for `downloaded()`.
      window.__blobs = [];
      window.URL.createObjectURL = blob => { window.__blobs.push(blob); return 'blob:stub'; };
      window.URL.revokeObjectURL = () => {};
      // The page draws its own questions and menus: a native dialog is a bug.
      for (const name of ['alert', 'confirm', 'prompt']) {
        window[name] = () => { throw new Error(`the page opened a native ${name}()`); };
      }
      // jsdom has no layout to scroll: count the calls instead.
      window.__scrolls = 0;
      window.scrollTo = () => { window.__scrolls += 1; };
      // And record the name on the anchor the page would have clicked.
      window.__downloads = [];
      window.HTMLAnchorElement.prototype.click = function () { window.__downloads.push(this.download); };
    },
  });
  const { window } = dom;
  window.addEventListener('error', e => errs.push(String((e.error && e.error.message) || e.error)));
  const d = window.document;
  await tick();
  const listed = (rows, match) => [...d.querySelectorAll(`${rows} tr.row`)]
    .find(row => !match || row.textContent.includes(match));
  const api = {
    window, d, errs, dom,
    $: s => d.querySelector(s),
    $$: s => [...d.querySelectorAll(s)],
    /** The text of a selector's first match, or of an element, with whitespace collapsed. */
    text: s => ((typeof s === 'string' ? d.querySelector(s) : s) || {}).textContent?.replace(/\s+/g, ' ').trim(),
    click: async s => {
      const el = typeof s === 'string' ? d.querySelector(s) : s;
      el.dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true }));
      await tick();
    },
    /** Type `value` into a field and leave it, as a reader does before acting on anything else. */
    type: async (s, value) => {
      const el = typeof s === 'string' ? d.querySelector(s) : s;
      el.value = value;
      el.dispatchEvent(new window.Event('input', { bubbles: true }));
      el.dispatchEvent(new window.Event('change', { bubbles: true }));
      await tick();
    },
    /** What hovering each match of a selector says: the text of its `<title>`. */
    tips: s => [...d.querySelectorAll(s)].map(el => el.querySelector('title').textContent),
    /** The view that is showing, as its element's id. */
    view: () => d.querySelector('.view.on').id,
    /** Go Back or Forward by `delta` steps, and let the page follow. */
    travel: async delta => {
      window.history.go(delta);
      await tick();
    },
    /** Go to an address of the page as a link in it would, and let the page follow. */
    visit: async route => {
      window.visit(route);
      await tick();
    },
    /** Ask the server whether the folder changed, as the page's timer does. */
    poll: async () => {
      await window.pollNow();
      await tick();
    },
    /** Open an experiment's cases from the Experiments list: the first, or the one whose row says `match`. */
    openExperiment: match => api.click(listed('#experiments-rows', match)),
    /** Open a report from the Reports list: the first, or the one whose row says `match`. */
    openReport: match => api.click(listed('#reports-rows', match)),
    /** Open a session from the Sessions list: the first, or the one whose row says `match`. */
    openSession: match => api.click(listed('#sessions-rows', match)),
    /** The parsed document behind the page's most recent download. */
    downloaded: async () => {
      const blob = window.__blobs[window.__blobs.length - 1];
      if (!blob) return null;
      // jsdom's Blob has no `text()`; FileReader is how this jsdom reads one.
      const text = await new Promise((resolve, reject) => {
        const reader = new window.FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsText(blob);
      });
      return JSON.parse(text);
    },
    /** A keystroke on `target`; with none, on the document, where the shortcuts listen. */
    key: async (key, target = d) => {
      target.dispatchEvent(new window.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
      await tick();
    },
  };
  PAGES.push(api);
  return api;
}

// A copy, as off the wire: the page must not share objects with the test.
const answer = (body, status = 200) =>
  ({ ok: status < 400, status, statusText: '', json: async () => JSON.parse(JSON.stringify(body)) });

/** A `fetchImpl` that answers as the server would, recording each request in `.sent`.
    `scan` is what the folder holds (`{ experiments, reports }`), or a function returning `[body, status]`
    to fail a scan; `put` is such a function too, to hold a save. `file` may be a function of the path
    asked for; `mark` returns the tree's fingerprint. */
export function fakeServer({ config = { root: '/served', depth: 3, read_only: false }, scan = {}, file, put, mark } = {}) {
  const sent = [];
  const fetchImpl = async (url, opts = {}) => {
    sent.push({ url: String(url), method: opts.method || 'GET', headers: opts.headers || {} });
    const path = String(url).split('?')[0];
    if (path === '/api/config') return answer(config);
    if (path === '/api/changes') return answer({ fingerprint: mark ? mark() : 'still' });
    if (path === '/api/scan') {
      return typeof scan === 'function' ? answer(...scan())
        : answer({ ...config, fingerprint: mark ? mark() : 'still', experiments: [], reports: [], skipped: [], optimizer_runs: [], ...scan });
    }
    if (opts.method === 'PUT') return answer(...await put(JSON.parse(opts.body)));
    return answer(typeof file === 'function' ? file(new URL(String(url), 'http://x').searchParams.get('path')) : file);
  };
  return Object.assign(fetchImpl, { sent });
}

/* One Python process serves every folder a suite asks for: starting one costs more than a suite's pages do. */
let farm = null;
function servers() {
  if (farm) return farm;
  const proc = spawn(process.env.PYTHON || 'python3', [new URL('./serve.py', import.meta.url).pathname],
    { stdio: ['pipe', 'pipe', 'inherit'] });
  const answers = readline.createInterface({ input: proc.stdout })[Symbol.asyncIterator]();
  const gone = new Promise((_, reject) => proc.on('exit', code => reject(new Error('the test server exited ' + code))));
  gone.catch(() => {});
  process.on('exit', () => proc.kill());
  farm = { proc, answers, gone, last: Promise.resolve() };
  return farm;
}

/** Start the real server over `root`: `{ base }` once it is listening. */
export async function realServer(root, token, { readOnly = false } = {}) {
  const { proc, answers, gone } = servers();
  /* One at a time, so each answer is read by the request that asked. */
  const asked = farm.last.then(async () => {
    proc.stdin.write(JSON.stringify({ root, token, read_only: readOnly }) + '\n');
    const { value } = await Promise.race([answers.next(), gone]);
    return { base: `http://127.0.0.1:${JSON.parse(value).port}` };
  });
  farm.last = asked.catch(() => {});
  return asked;
}

const ROOTS = [];
/** Write `files`, a map of path to document or to text, as a folder; or copy the folder at a path. Removed when the suite reports.
    Every file has the same time, so a list is in path order unless a test says otherwise. */
export function folder(files) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ui-test-')));
  ROOTS.push(root);
  if (typeof files === 'string') {
    fs.cpSync(files, root, { recursive: true });
    return root;
  }
  for (const [rel, held] of Object.entries(files)) {
    const at = path.join(root, rel);
    fs.mkdirSync(path.dirname(at), { recursive: true });
    fs.writeFileSync(at, typeof held === 'string' ? held : JSON.stringify(held));
    fs.utimesSync(at, 1700000000, 1700000000);
  }
  return root;
}

/** A folder of one experiment and the report made from it, with enough in both to show every part of the page. */
export const SAMPLE = new URL('./fixtures/sample/', import.meta.url).pathname;

const TOKEN = 'test-token';
/** A page over the real server reading `files`, as `folder()` takes them, opened at `route`. Its `root` is the
    folder, `saved()` the experiment file its last accepted Save wrote, and `puts()` how many saves it sent. */
export async function served(files, { route = '/', readOnly, ...options } = {}) {
  const root = folder(files);
  const { base } = await realServer(root, TOKEN, { readOnly });
  const sent = { puts: 0, saved: null };
  const fetchImpl = async (url, opts = {}) => {
    const answer = await ask(base, url, opts);
    if (opts.method === 'PUT') {
      sent.puts += 1;
      if (answer.ok) sent.saved = new URL(base + url).searchParams.get('path');
    }
    return answer;
  };
  const p = await page({ fetchImpl, url: base + route, token: TOKEN, ...options });
  await waitFor(() => p.$('.app').dataset.ready, 'boot to finish');
  return Object.assign(p, {
    root, base, token: TOKEN,
    puts: () => sent.puts,
    saved: () => (sent.saved ? JSON.parse(fs.readFileSync(path.join(root, sent.saved), 'utf8')) : null),
    /** Write a file of the folder, as something other than the page would. */
    write: (rel, held) => {
      fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
      fs.writeFileSync(path.join(root, rel), typeof held === 'string' ? held : JSON.stringify(held));
    },
  });
}

export function runner(name) {
  let fail = 0, pass = 0;
  const ok = (cond, msg, extra) => {
    if (cond) { pass++; } else { fail++; console.log(`FAIL ${msg} :: ${extra}`); }
  };
  return {
    ok,
    /** Passes when the two are the same once written as JSON; a failure prints what was got. */
    same: (actual, expected, msg) => ok(JSON.stringify(actual) === JSON.stringify(expected), msg, JSON.stringify(actual)),
    /** Report the suite. An error on any page it built fails it. */
    done() {
      const errs = PAGES.flatMap(p => p.errs);
      PAGES.forEach(p => p.window.close());
      if (farm) farm.proc.kill();
      ROOTS.forEach(root => fs.rmSync(root, { recursive: true, force: true }));
      if (errs.length) { console.log('page errors:', errs); fail += errs.length; }
      console.log(`${fail ? 'FAILED' : 'ok'}  ${name}: ${pass} passed${fail ? `, ${fail} failed` : ''}`);
      if (fail) process.exitCode = 1;
    },
  };
}
