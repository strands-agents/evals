/* The file opened from disk — `file://`, no server, no origin.
   Distinct from `standalone.mjs`: that suite boots at an http URL with `fetch` simply absent,
   which reaches standalone mode without ever touching the APIs an opaque origin restricts.
   Sharing the page as a single file is a supported way to use it, so the restricted ones get
   exercised here. `history.replaceState` is the sharp edge: it rejects a URL it cannot verify
   against an opaque origin, and thrown from an async caller that aborted rendering outright. */
import { page, runner } from './harness.mjs';

const { ok, done } = runner('file-origin');

/* What a browser does with fetch('/api/config') on a page loaded from disk: the path resolves
   against the file: scheme and is refused before any network activity. */
const diskFetch = async () => {
  throw new TypeError('Failed to fetch: URL scheme "file" is not supported');
};

const p = await page({ url: 'file:///home/someone/Downloads/ui.html', fetchImpl: diskFetch });
const { $, $$, text, click, window, errs } = p;

// ── it boots at all ──
/* Each of these read zero / empty when replaceState escapes, so they are the regression guard. */
ok($$('#runs-rows tr.run').length === 1, 'bundled run rendered from disk', $$('#runs-rows tr.run').length);
ok(text('.tab.on') === 'Runs', 'lands on Runs', text('.tab.on'));
ok($$('#case-list li[data-i]').length === 13, 'sample cases loaded', $$('#case-list li[data-i]').length);
ok(!/could not load/.test($('#runs-rows').textContent), 'no load-error row', $('#runs-rows').textContent.slice(0, 80));

// ── standalone, because the origin cannot serve an API ──
ok(/standalone file/.test(text('#mode-tag')), 'mode tag names standalone', text('#mode-tag'));
ok($('[data-act="rescan"]').hidden === true, 'Rescan hidden without a server');

// ── deep links survive the restriction ──
/* The hash is the whole point of writeHash: a colleague is meant to be able to send a URL.
   replaceState cannot set it here, so the fallback has to, and must not throw doing it. */
await click('#runs-rows tr.run');
ok(/run=/.test(window.location.hash), 'hash records the run from disk', window.location.hash);
ok(/tab=/.test(window.location.hash), 'hash records the tab', window.location.hash);

const beforeTab = window.location.hash;
await click('.tab[data-view="report"]');
ok(window.location.hash !== beforeTab, 'hash follows tab changes', window.location.hash);
ok(/tab=report/.test(window.location.hash), 'and names the new tab', window.location.hash);

// ── a hash written from disk is readable back, so the link actually round-trips ──
const shared = window.location.hash;
const reopened = await page({ url: `file:///home/someone/Downloads/ui.html${shared}`, fetchImpl: diskFetch });
ok(reopened.text('.tab.on') === 'Report', 'shared link reopens on its tab', reopened.text('.tab.on'));
ok(reopened.errs.length === 0, 'reopening from a shared link raises nothing', reopened.errs.join('; '));

// ── editing still works; only persistence differs ──
ok($$('#case-list li[data-i]').length > 0 && $$('input, textarea').length > 0, 'case editor reachable');
ok($('[data-act="pick"]').hidden === false, 'Open folder offered for real files');

done(errs);
