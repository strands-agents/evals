import fs from 'node:fs';
import { JSDOM } from 'jsdom';

const ASSET = process.env.UI_HTML
  ? new URL(process.env.UI_HTML, `file://${process.cwd()}/`)
  : new URL('../../src/strands_evals/cli/assets/ui.html', import.meta.url);
/* Inlines the stylesheet and scripts beside the page, the way `load_asset` does for the server. */
const beside = name => fs.readFileSync(new URL(name, ASSET), 'utf8');
const HTML = fs.readFileSync(ASSET, 'utf8')
  .replace(/<link rel="stylesheet" href="([\w.-]+)">/g, (_, name) => `<style>\n${beside(name)}</style>`)
  .replace(/<script src="([\w.-]+)"><\/script>/g, (_, name) => `<script>\n${beside(name)}</script>`);

/** A File stand-in: jsdom cannot build a real FileList, and only `text()` is used. */
export class FakeFile {
  constructor(path, text) {
    this.webkitRelativePath = path;
    this.name = path.split('/').pop();
    this._t = text;
    this.size = text.length;
    this.lastModified = 1700000000000;
  }
  async text() { return this._t; }
}

/* Every window built, closed when the suite reports: a served page polls on a timer, which would keep Node alive. */
const WINDOWS = [];

export const tick = (n = 3) => new Promise(resolve => setTimeout(resolve, n * 5));

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
    `phone` answers the page's small-window media query, which jsdom cannot. `visible` makes the
    document one that is being looked at; without it a served page's polling timer does nothing. */
export async function page({ fetchImpl, url = 'http://127.0.0.1:9999/?t=tok', token, phone, visible } = {}) {
  const errs = [];
  /* `token` stands in for the server writing it into the document, as it does for a served page. */
  const html = token ? HTML.replace('name="strands-evals-token" content=""', `name="strands-evals-token" content="${token}"`) : HTML;
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: !!visible,
    url,
    beforeParse(window) {
      if (fetchImpl) window.fetch = (...a) => fetchImpl(...a);
      if (phone) window.matchMedia = query => ({ matches: /max-width/.test(query) });
      // A download is the only place a standalone page emits a document: keep the blob for `saved()`.
      window.__blobs = [];
      window.URL.createObjectURL = blob => { window.__blobs.push(blob); return 'blob:stub'; };
      window.URL.revokeObjectURL = () => {};
      window.confirm = () => true;
      // jsdom has no layout to scroll: count the calls instead.
      window.__scrolls = 0;
      window.scrollTo = () => { window.__scrolls += 1; };
      // jsdom has no download: record the anchor the page would have clicked.
      window.__downloads = [];
      window.HTMLAnchorElement.prototype.click = function () { window.__downloads.push(this.download); };
    },
  });
  const { window } = dom;
  WINDOWS.push(window);
  window.addEventListener('error', e => errs.push(String((e.error && e.error.message) || e.error)));
  const d = window.document;
  await tick();
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
    type: async (s, value) => {
      const el = typeof s === 'string' ? d.querySelector(s) : s;
      el.value = value;
      el.dispatchEvent(new window.Event('input', { bubbles: true }));
      await tick();
    },
    /** The parsed document behind the page's most recent download. */
    saved: async () => {
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
    /** Feed a tree into the standalone folder reader. */
    openFolder: async files => {
      const input = d.querySelector('#dirinput');
      Object.defineProperty(input, 'files', {
        value: Object.entries(files).map(([p, o]) =>
          new FakeFile(p, typeof o === 'string' ? o : JSON.stringify(o))),
        configurable: true,
      });
      await input.onchange({ target: input });
      await tick();
    },
  };
  return api;
}

// A copy, as off the wire: the page must not share objects with the test.
const answer = (body, status = 200) =>
  ({ ok: status < 400, status, statusText: '', json: async () => JSON.parse(JSON.stringify(body)) });

/** A `fetchImpl` that answers as the server would, recording each request in `.sent`.
    `runs` and `put` may be functions returning `[body, status]`, to fail a scan or hold a save.
    `file` may be a function of the path asked for; `mark` returns the tree's fingerprint. */
export function fakeServer({ config = { root: '/served', depth: 3, read_only: false }, runs = [], file, put, mark } = {}) {
  const sent = [];
  const fetchImpl = async (url, opts = {}) => {
    sent.push({ url: String(url), method: opts.method || 'GET', headers: opts.headers || {} });
    const path = String(url).split('?')[0];
    if (path === '/api/config') return answer(config);
    if (path === '/api/changes') return answer({ fingerprint: mark ? mark() : 'still' });
    if (path === '/api/runs') {
      return typeof runs === 'function' ? answer(...runs())
        : answer({ ...config, fingerprint: mark ? mark() : 'still', runs, skipped: [], optimizer_runs: [] });
    }
    if (opts.method === 'PUT') return answer(...await put(JSON.parse(opts.body)));
    return answer(typeof file === 'function' ? file(new URL(String(url), 'http://x').searchParams.get('path')) : file);
  };
  return Object.assign(fetchImpl, { sent });
}

export function runner(name) {
  let fail = 0, pass = 0;
  const ok = (cond, msg, extra) => {
    if (cond) { pass++; } else { fail++; console.log(`FAIL ${msg} :: ${extra}`); }
  };
  return {
    ok,
    done(errs) {
      WINDOWS.forEach(window => window.close());
      if (errs && errs.length) { console.log('page errors:', errs); fail += errs.length; }
      console.log(`${fail ? 'FAILED' : 'ok'}  ${name}: ${pass} passed${fail ? `, ${fail} failed` : ''}`);
      if (fail) process.exitCode = 1;
    },
  };
}
