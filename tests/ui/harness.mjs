import fs from 'node:fs';
import { JSDOM } from 'jsdom';

export const ASSET = process.env.UI_HTML
  ? new URL(process.env.UI_HTML, `file://${process.cwd()}/`)
  : new URL('../../src/strands_evals/cli/assets/ui.html', import.meta.url);
export const HTML = fs.readFileSync(ASSET, 'utf8');

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

/** Build the page. `fetchImpl` is installed before any script runs, so boot() sees it. */
export async function page({ fetchImpl, url = 'http://127.0.0.1:9999/?t=tok' } = {}) {
  const errs = [];
  const dom = new JSDOM(HTML, {
    runScripts: 'dangerously',
    url,
    beforeParse(window) {
      if (fetchImpl) window.fetch = (...a) => fetchImpl(...a);
      window.URL.createObjectURL = () => 'blob:stub';
      window.URL.revokeObjectURL = () => {};
      window.confirm = () => true;
      // jsdom has no download: record the anchor the page would have clicked.
      window.__downloads = [];
      window.HTMLAnchorElement.prototype.click = function () { window.__downloads.push(this.download); };
    },
  });
  const { window } = dom;
  window.addEventListener('error', e => errs.push(String((e.error && e.error.message) || e.error)));
  const d = window.document;
  await tick();
  const api = {
    window, d, errs, dom,
    $: s => d.querySelector(s),
    $$: s => [...d.querySelectorAll(s)],
    text: s => (d.querySelector(s) || {}).textContent?.replace(/\s+/g, ' ').trim(),
    click: async s => {
      const el = typeof s === 'string' ? d.querySelector(s) : s;
      el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
      await tick();
    },
    type: async (s, value) => {
      const el = d.querySelector(s);
      el.value = value;
      el.dispatchEvent(new window.Event('input', { bubbles: true }));
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

export function runner(name) {
  let fail = 0, pass = 0;
  const ok = (cond, msg, extra) => {
    if (cond) { pass++; } else { fail++; console.log(`FAIL ${msg} :: ${extra}`); }
  };
  return {
    ok,
    done(errs) {
      if (errs && errs.length) { console.log('page errors:', errs); fail += errs.length; }
      console.log(`${fail ? 'FAILED' : 'ok'}  ${name}: ${pass} passed${fail ? `, ${fail} failed` : ''}`);
      if (fail) process.exitCode = 1;
    },
  };
}
