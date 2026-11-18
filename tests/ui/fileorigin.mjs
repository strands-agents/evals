/* The page opened from disk: `file://`, an opaque origin, where `history.replaceState` rejects the
   URL. `standalone.mjs` boots at an http URL and never meets that. */
import { page, runner } from './harness.mjs';

const { ok, done } = runner('file-origin');
const ON_DISK = 'file:///home/someone/Downloads/ui.html';
// What a browser does with fetch('/api/config') from disk: refused before any network activity.
const diskFetch = async () => {
  throw new TypeError('Failed to fetch: URL scheme "file" is not supported');
};

const { $, $$, text, click, openExperiment, travel, view, window } = await page({ url: ON_DISK, fetchImpl: diskFetch });

ok($$('#reports-rows tr.row').length === 1 && $$('#experiments-rows tr.row').length === 1, 'the sample is listed from disk',
  text('#reports-rows'));
ok(/standalone/.test(text('#mode-tag')) && $('[data-act="rescan"]').hidden, 'in standalone mode, with no Rescan',
  text('#mode-tag'));

// replaceState cannot write the address here, so the hash fallback has to.
await openExperiment();
ok($$('#case-list li[data-i]').length === 13 && window.location.hash.startsWith('#/experiments/experiment.json'),
  'the address names the experiment that is open, after the #', window.location.hash);
await click('#facts a[data-report]');
ok(window.location.hash === '#/reports/report.json', 'and follows to its report', window.location.hash);
await travel(-1);
ok(view() === 'v-experiment' && window.location.hash.startsWith('#/experiments/experiment.json'), 'which Back undoes, from disk too',
  `${view()} ${window.location.hash}`);
await travel(1);

const reopened = await page({ url: ON_DISK + window.location.hash, fetchImpl: diskFetch });
ok(reopened.view() === 'v-report' && reopened.text('#r-score') === '0.92', 'an address written from disk reopens the report it names',
  reopened.view());

done();
