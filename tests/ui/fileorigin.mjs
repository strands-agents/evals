/* The page opened from disk: `file://`, an opaque origin, where `history.replaceState` rejects the
   URL. `standalone.mjs` boots at an http URL and never meets that. */
import { page, runner, tick } from './harness.mjs';

const { ok, done } = runner('file-origin');
const ON_DISK = 'file:///home/someone/Downloads/ui.html';
// What a browser does with fetch('/api/config') from disk: refused before any network activity.
const diskFetch = async () => {
  throw new TypeError('Failed to fetch: URL scheme "file" is not supported');
};

const { $, $$, text, click, window, errs } = await page({ url: ON_DISK, fetchImpl: diskFetch });

ok($$('#runs-rows tr.run').length === 1 && $$('#case-list li[data-i]').length === 13, 'the sample renders from disk',
  text('#runs-rows'));
ok(/standalone/.test(text('#mode-tag')) && $('[data-act="rescan"]').hidden, 'in standalone mode, with no Rescan',
  text('#mode-tag'));

// replaceState cannot write the link here, so the hash fallback has to.
await click('#runs-rows tr.run');
ok(window.location.hash.startsWith('#/runs/experiment.json/experiment'), 'the address names the run and the view, after the #',
  window.location.hash);
await click('.tab[data-view="report"]');
ok(window.location.hash === '#/runs/experiment.json/report', 'and follows a tab change', window.location.hash);
window.history.back();
await tick(6);
ok($('.view.on').id === 'v-experiment' && window.location.hash.startsWith('#/runs/experiment.json/experiment'), 'which Back undoes, from disk too',
  `${$('.view.on').id} ${window.location.hash}`);
window.history.forward();
await tick(6);

const reopened = await page({ url: ON_DISK + window.location.hash, fetchImpl: diskFetch });
ok(reopened.text('.tab.on') === 'Report', 'a link written from disk reopens on its tab', reopened.text('.tab.on'));

done(errs.concat(reopened.errs));
