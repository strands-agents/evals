/* The Evals section beyond the bundled sample: the report table, the run list, links, the case
   editor and what Save writes, served mode against a fake server, keyboard and theme. */
import { fakeServer, page, runner, tick } from './harness.mjs';

const { ok, done } = runner('extras');
const pages = [];
const open = async opts => { const p = await page(opts); pages.push(p); return p; };
/** A page with no server, opened at an address of its own: there the route follows `#`. */
const linked = route => open({ url: `http://127.0.0.1:9999/#${route}` });
/** A page with this experiment, and its report if given, read from a folder and opened. */
const opened = async (experiment, report) => {
  const p = await open();
  await p.openFolder({ 'run/experiment.json': experiment, ...(report && { 'run/report.json': report }) });
  await p.click('#runs-rows tr.run');
  return p;
};
const experiment = (...cases) => ({ cases, evaluators: [] });
const CASE = { name: 'a', session_id: 's-a', input: 'q' };
const report = (score, ...cases) => ({
  overall_score: score, cases: cases.map(c => ({ evaluator: 'X', ...c })),
  scores: cases.map(() => score), test_passes: cases.map(() => score === 1),
});

// ── report table: the row cap ──
{
  const cases = Array.from({ length: 420 }, (_, i) => ({ name: `case_${i}`, session_id: `sid-${i}`, input: `q${i}` }));
  const p = await opened(experiment(...cases), report(1, ...cases));
  await p.click('.tab[data-view="report"]');
  const shown = () => p.$$('#rep-rows tr.case').length;
  ok(shown() === 300, 'the first page is capped at 300 rows', shown());
  ok(!p.$('#rep-more').hidden && /120 more/.test(p.text('#rep-more')), 'with a button naming how many are left',
    p.text('#rep-more'));
  ok(p.text('#rep-count') === '420 of 420 rows', 'the count is the total, not the rows shown', p.text('#rep-count'));
  await p.click('#rep-more');
  ok(shown() === 420 && p.$('#rep-more').hidden, 'show more reveals the rest and goes away', shown());
}

// ── report table: a report with only its required arrays ──
{
  const p = await opened(experiment(CASE), report(1, { ...CASE, actual_output: 'yes' }));
  await p.click('.tab[data-view="report"]');
  ok(p.$$('#rep-rows tr.case').length === 1 && p.text('#r-score') === '1.00', 'renders its row and score',
    p.text('#v-report'));
  await p.click('#rep-rows tr.case');
  ok(/returned no outputs/.test(p.text('#rep-rows tr.detail')),
    'and says the row has no outputs rather than inventing a verdict', p.text('#rep-rows tr.detail'));
}

// ── run list: pairing, half-present runs, and the cells that jump to a tab ──
{
  const p = await open();
  await p.openFolder({
    'paired/experiment.json': experiment({ ...CASE, expected_output: 'yes' }),
    'paired/report.json': report(0.25, CASE),
    'paired/experiment.saved.json': experiment(CASE),
    'mismatch/experiment.json': experiment(CASE),
    'mismatch/report.json': report(0, { name: 'somebody-elses-case', session_id: 's-z' }),
    'only-report/report.json': report(0.4, { name: 'a' }),
    'only-exp/experiment.json': experiment(CASE),
  });
  const PAIRED = 'paired / experiment.json', ONLY_EXP = 'only-exp / experiment.json';
  const row = label => p.$$('#runs-rows tr.run').find(r => p.text(r.children[0]) === label);
  const cell = (label, i) => row(label).children[i];
  const shows = (label, i, what) => what.test(p.text(cell(label, i)));
  const runs = () => p.click('.tab[data-view="runs"]');

  ok(shows(PAIRED, 2, /pairing verified/), 'experiment.json pairs with the report.json beside it', p.text(row(PAIRED)));
  ok(shows('paired / experiment.saved.json', 2, /not run yet/), 'leaving the other experiment in the folder unpaired',
    p.text('#runs-rows'));
  ok(shows('mismatch / experiment.json', 2, /does not match/), 'a report for other cases is flagged', p.text('#runs-rows'));
  ok(shows('only-report / report.json', 1, /none/), 'a report alone is a run with no experiment', p.text('#runs-rows'));
  const jumps = label => [...row(label).children].map(td => td.dataset.goto || '-').join();
  ok(jumps(PAIRED) === '-,experiment,report,-' && jumps(ONLY_EXP) === '-,experiment,-,-',
    'only a cell with something behind it is marked as a jump', `${jumps(PAIRED)} | ${jumps(ONLY_EXP)}`);

  await p.click(row('only-report / report.json'));
  ok(p.text('.tab.on') === 'Report' && p.text('#r-score') === '0.40', 'a report-only run opens on its report',
    p.text('.tab.on'));
  await p.click('.tab[data-view="experiment"]');
  ok(/No experiment in this run/.test(p.text('#form')) && p.$('#save').disabled && p.$('#add-case').disabled,
    'and its editor says why it is empty, with nothing to press', p.text('#form'));

  await runs();
  await p.click(cell(ONLY_EXP, 2));
  ok(p.text('.tab.on') === 'Experiment', 'a cell that is not a jump does what the row does', p.text('.tab.on'));
  await p.click('.tab[data-view="report"]');
  ok(/No report in this run/.test(p.text('#rep-rows')), 'and a run with no report says so', p.text('#rep-rows'));

  await runs();
  await p.click(cell(PAIRED, 2));
  ok(p.text('.tab.on') === 'Report' && p.text('#r-score') === '0.25', 'the report cell opens a run on its report',
    p.text('.tab.on'));
  await runs();
  await p.click(cell(PAIRED, 2));
  ok(p.text('.tab.on') === 'Report', 'also when that run is the one already open', p.text('.tab.on'));

  // Re-reading the open run would drop the edit without asking.
  await p.click('.tab[data-view="experiment"]');
  await p.type('#form [data-k="expected_output"]', 'edited, not saved');
  await runs();
  await p.click(cell(PAIRED, 3));
  ok(p.$('#v-experiment').classList.contains('on') && p.$('#form [data-k="expected_output"]').value === 'edited, not saved',
    'the open run’s row goes back to it with the unsaved edit intact', p.$('#form [data-k="expected_output"]').value);
}

// ── deep links ──
{
  const p = await linked('/runs/experiment.json/experiment/cases/refund_edge_case');
  ok(p.text('.tab.on') === 'Experiment' && p.text('#case-list li.on .nm') === 'refund_edge_case',
    'an address opens the view and the case it names', `${p.text('.tab.on')} ${p.text('#case-list li.on')}`);
  await p.click('.tab[data-view="report"]');
  ok(p.window.location.hash === '#/runs/experiment.json/report', 'and another view of the same run is a path beside it',
    p.window.location.hash);
}
for (const [route, view, rewritten, what] of [
  ['/nope/at/all', 'runs', '/runs', 'an address the page does not have falls back to the run list'],
  ['/runs/experiment.json/experiment/cases/%E0%A4%A', 'experiment', '/runs/experiment.json/experiment/cases/order_status_lookup',
    'a malformed escape names no case, so the first is shown'],
  ['/optimization/tuning/overview', 'runs', '/runs', 'an Optimization address with no optimizer run falls back to the run list'],
]) {
  const p = await linked(route);
  const on = p.$$('.view.on').map(v => v.id).join();
  ok(p.$('.app').dataset.ready === '1' && on === `v-${view}` && p.window.location.hash === '#' + rewritten,
    `${route}: ${what}, and the address is rewritten to match`, `${on} ${p.window.location.hash}`);
}

// ── case editor: deleting a case ──
{
  const p = await opened(experiment({ name: 'keep', input: 'a' }, { name: 'gone', input: 'b' }));
  const remove = async () => { await p.click('#case-list li[data-i="1"]'); await p.click('#del-case'); };
  await remove();
  ok(p.text('#dirty') === '1 unsaved change', 'deleting a case counts as a change', p.text('#dirty'));
  await p.click('#revert');
  ok(p.$$('#case-list li[data-i]').length === 2 && p.text('#dirty') === 'No unsaved changes',
    'Revert brings it back, leaving nothing to save', p.text('#dirty'));
  await remove();
  await p.click('#save');
  const names = (await p.saved()).cases.map(c => c.name).join();
  ok(names === 'keep', 'and Save writes the file without it', names);
  ok(p.text('#dirty') === 'No unsaved changes', 'after which the delete is no longer pending', p.text('#dirty'));
}

// ── case editor: an evaluator setting is text, whatever it contains ──
{
  const markup = '"><img src=x onerror="window.__xss=1">', value = 'say <b>yes</b>';
  const p = await opened({ cases: [CASE], evaluators: [{ evaluator_type: 'Contains', [markup]: value }] });
  ok(p.text('#evs .evchip .c') === `${markup}=${JSON.stringify(value)}`, 'its name and value are text on the chip',
    p.$('#evs .evchip').innerHTML);
  await p.click('#evs .evchip');
  ok(p.$$('#evform label').some(l => l.textContent === markup) && !p.$('#evform img'), 'and in the form it opens',
    p.$('#evform').innerHTML);
}

const DRAFTS = experiment(
  { name: 'kept', input: 'q0', expected_output: 'a', metadata: { status: 'accepted', source: 'session-7' } },
  { name: 'draft_a', input: 'q1', metadata: { status: 'draft', turns: 3 } },
  { name: 'draft_b', input: 'q2', metadata: { status: 'draft' } },
  { name: 'odd', input: 'q3', metadata: 'not an object' },
  { name: 'hollow', input: 'q4', metadata: {}, expected_trajectory: [] },
);
const badges = p =>
  p.$$('#case-list li[data-i]').filter(li => li.querySelector('.badge')).map(li => li.dataset.i).join();

// ── metadata ──
{
  const p = await opened(DRAFTS);
  const pick = i => p.click(`#case-list li[data-i="${i}"]`);
  const rows = () => p.$$('.meta-row');
  const field = (n, which) => rows()[n].querySelector(`[data-meta-${which}]`);
  const written = async i => { await p.click('#save'); return JSON.stringify((await p.saved()).cases[i].metadata); };
  const glyphs = () => p.$$('#case-list .props').map(el => el.textContent).join();

  ok(glyphs() === 'OM,M,M,M,', 'the list marks the properties a case carries, and an empty one is not carried', glyphs());
  ok(badges(p) === '1,2', 'and badges each draft', badges(p));

  await pick(1);
  const shown = rows().map((_, n) => `${field(n, 'k').value}=${field(n, 'v').value}`).join();
  ok(shown === 'status=draft,turns=3', 'one row per key, with the value unquoted', shown);
  await p.type(field(1, 'v'), '4');
  ok(p.text('#dirty') === '1 unsaved change' && p.text('#save') === 'Save (1)',
    'editing a value is an unsaved change, counted on Save', `${p.text('#dirty')} | ${p.text('#save')}`);
  await p.type(field(0, 'v'), 'hello world');
  let file = await written(1);
  ok(file === '{"status":"hello world","turns":4}', 'a value that parses as JSON is written as JSON, any other as text', file);

  await p.type(field(1, 'k'), 'turn_count');
  file = await written(1);
  ok(file === '{"status":"hello world","turn_count":4}', 'a key can be renamed', file);
  await p.type(field(1, 'k'), '');
  file = await written(1);
  ok(file === '{"status":"hello world"}', 'emptying a key drops it', file);
  await p.type(field(0, 'k'), '');
  file = await written(1);
  ok(file === undefined && glyphs() === 'OM,,M,M,', 'and emptying the last one drops the metadata itself',
    `${file} ${glyphs()}`);

  await pick(2);
  await p.click('[data-meta-add]');
  ok(rows().length === 2 && p.text('#dirty') === 'No unsaved changes', '+ key offers a blank row, which is not an edit',
    `${rows().length} rows, ${p.text('#dirty')}`);
  await p.type(field(1, 'k'), 'source');
  await p.type(field(1, 'v'), 'session-9');
  file = await written(2);
  ok(file === '{"status":"draft","source":"session-9"}', 'a filled row becomes a key', file);
  await p.click(field(0, 'rm'));
  file = await written(2);
  ok(file === '{"source":"session-9"}', 'and the × removes one', file);

  await pick(3);
  ok(!p.$('.meta-rows') && p.$('textarea[data-k="metadata"]').value === '"not an object"',
    'metadata that is not an object is shown as JSON, not rows', p.text('#form'));
  await pick(4);
  ok(rows().length === 0 && !!p.$('[data-meta-add]'), 'and an empty one as no rows, with + key offered', rows().length);
}
{
  const metadata = {
    edit: 'x', id: '42', flag: 'true', big: '9007199254740993', note: 'line1\nline2', ' padded ': 'v', '': 'empty key',
  };
  const p = await opened(experiment({ name: 'a', input: 'q', metadata }));
  await p.type('.meta-row [data-meta-v]', 'y');
  await p.click('#save');
  const file = JSON.stringify((await p.saved()).cases[0].metadata);
  ok(file === JSON.stringify({ ...metadata, edit: 'y' }),
    'editing one key leaves numeric and boolean strings, line breaks, padded keys and an empty key as they were', file);
}

// ── review pass over drafts ──
{
  const p = await opened(DRAFTS);
  const bar = p.$('#review-bar'), selected = () => p.$('#case-list li.on').dataset.i;
  ok(p.text('#review-case') === 'Review drafts (2)' && bar.hidden,
    'the button counts the drafts, and the bar waits to be asked for', p.text('#review-case'));
  await p.click('#review-case');
  ok(/2 drafts left, 3 kept/.test(p.text(bar)) && selected() === '1',
    'starting the pass counts both sides and selects the first draft', `${p.text(bar)} | selected ${selected()}`);
  await p.key('a');
  ok(/1 draft left, 4 kept/.test(p.text(bar)) && selected() === '2' && badges(p) === '2',
    'a accepts the draft and moves to the next', `${p.text(bar)} | selected ${selected()} | drafts ${badges(p)}`);
  await p.key('d');
  ok(p.$$('#case-list li').length === 4 && bar.hidden && p.$('#review-case').disabled,
    'd drops the case, and with no drafts left the pass closes', p.$$('#case-list li').length);
  ok(/every draft is resolved/.test(p.text('#saved')), 'saying so', p.text('#saved'));
  await p.click('#save');
  const file = (await p.saved()).cases.map(c => `${c.name}:${JSON.stringify(c.metadata)}`);
  ok(file.length === 4 && file[1] === 'draft_a:{"status":"accepted","turns":3}',
    'Save writes the file without the dropped case, the accepted one keeping its other keys', file.join(' '));
}
{
  const p = await opened(DRAFTS);
  const bar = p.$('#review-bar');
  await p.click('#review-case');
  await p.key('Escape');
  ok(bar.hidden && badges(p) === '1,2' && p.text('#dirty') === 'No unsaved changes',
    'Escape leaves the pass, deciding nothing', `drafts ${badges(p)}, ${p.text('#dirty')}`);
  await p.click('#review-case');
  await p.key('d', p.$('#case-find'));
  ok(p.$$('#case-list li').length === 5, 'a letter typed into a field is not a decision', p.$$('#case-list li').length);
  await p.click('[data-rv="acceptall"]');
  ok(badges(p) === '' && bar.hidden, 'Accept all resolves every draft and ends the pass', badges(p));
}

// ── review pass: its shortcuts belong to the Experiment view ──
{
  const p = await opened(experiment({ ...CASE, metadata: { status: 'draft' } }, { name: 'b', session_id: 's-b', input: 'q' }));
  await p.click('#review-case');
  await p.click('.tab[data-view="runs"]');
  await p.key('d');
  await p.click('.tab[data-view="experiment"]');
  ok(p.$$('#case-list li[data-i]').length === 2, 'd pressed on another view drops nothing', p.$$('#case-list li[data-i]').length);
}

// ── a long rubric: cut short on its chip, whole in a text block ──
{
  const rubric = 'Score 1.0 when the reply is polite and names the policy. '.repeat(6).trim();
  const p = await opened({ cases: [CASE], evaluators: [{ evaluator_type: 'OutputEvaluator', rubric, include_inputs: true }] });
  const chip = () => p.$('#evs .evchip');
  ok(p.text(chip()).length < 120 && p.text(chip()).includes('…') && chip().getAttribute('aria-expanded') === 'false',
    'a long rubric is cut short on its chip', p.text(chip()));
  await p.click(chip());
  const area = p.$('#evform textarea[data-evk="rubric"]');
  ok(area && area.value === rubric && p.$('#evform input[data-evk="include_inputs"]').type === 'checkbox'
    && chip().getAttribute('aria-expanded') === 'true',
  'and the chip opens a text block holding all of it, with the short settings still one line each');
  await p.type(area, 'Score 1.0 when correct.\nScore 0 otherwise.');
  await p.click('#save');
  const saved = (await p.saved()).evaluators[0].rubric;
  ok(saved === 'Score 1.0 when correct.\nScore 0 otherwise.', 'what is typed there is saved, line break included', saved);
}

// ── history: a move between views or runs is a step, and Back and Forward retrace them ──
{
  const two = { name: 'a2', session_id: 's-a2', input: 'q' };
  const p = await open();
  await p.openFolder({
    'one/experiment.json': experiment(CASE, two), 'one/report.json': report(1, CASE),
    'two/experiment.json': experiment({ name: 'b', session_id: 's-b', input: 'q' }),
  });
  const history = p.window.history;
  const run = name => p.$$('#runs-rows tr.run').find(r => r.textContent.includes(name));
  const at = () => `${p.$('.view.on').id} ${p.text('#path')}`;
  const travel = async delta => { history.go(delta); await tick(6); return at(); };
  const start = history.length;
  await p.click(run('one'));
  const scrolls = p.window.__scrolls;
  await p.click('.tab[data-view="report"]');
  await p.click('.tab[data-view="report"]');
  ok(p.window.__scrolls === scrolls + 1, 'a view moved to starts at its top; the one already open stays put', p.window.__scrolls - scrolls);
  await p.click('.tab[data-view="runs"]');
  await p.click(run('two'));
  ok(history.length === start + 4, 'each move between views or runs adds a step', history.length - start);
  const retraced = [await travel(-1), await travel(-1), await travel(-1), await travel(3)];
  ok(retraced.join() === 'v-runs one,v-report one/report.json,v-experiment one/experiment.json,v-experiment two/experiment.json',
    'Back walks them in reverse, across runs, and Forward returns', retraced.join());

  await travel(-3);
  const steps = history.length;
  await p.click(p.$$('#case-list li[data-i]')[1]);
  ok(history.length === steps && p.window.location.hash === '#/runs/one/experiment.json/experiment/cases/a2',
    'picking a case corrects the address without adding a step', p.window.location.hash);

  await p.click('.tab[data-view="runs"]');
  await p.click(run('two'));
  const reread = [await travel(-2), p.$('#form [data-k="name"]').value, await travel(2)];
  ok(reread.join() === 'v-experiment one/experiment.json,a2,v-experiment two/experiment.json',
    'Back reopens the case that was being read, and the way Forward is still there', reread.join());

  p.window.location.hash = '#/runs/one/experiment.json/report';
  await tick(6);
  ok(at() === 'v-report one/report.json', 'an address typed into the open page is followed', at());

  await p.click(run('two'));
  await p.type('#form [data-k="input"]', 'unsaved');
  p.window.confirm = () => false;
  await travel(-1);
  ok(p.$('#form [data-k="input"]').value === 'unsaved' && p.window.location.hash === '#/runs/two/experiment.json/report',
    'Back to another run does not drop unsaved edits when that is declined, and the address says where the page still is',
    p.window.location.hash);

  const fresh = await linked('/runs/experiment.json/report');
  ok(fresh.window.history.length === 1, 'opening a link adds no step of its own', fresh.window.history.length);
}

// ── served mode, against a fake server ──
const served = put => fakeServer({
  runs: [{ id: 'e.json', dir: '.', report: null, link: null,
    experiment: { path: 'e.json', name: 'e.json', size: 1, modified: 5, cases: 1, evaluators: [] } }],
  file: { path: 'e.json', modified: 5, document: experiment({ name: 'a', input: 'q' }) },
  put,
});
{
  const p = await open({ fetchImpl: fakeServer({ runs: () => [{ error: 'scan blew up' }, 500] }) });
  ok(/^served/.test(p.text('#mode-tag')) && !p.$('#runs-rows tr.run'),
    'a failed scan leaves the page in served mode, not on the sample', p.text('#mode-tag'));
  ok(/scan blew up/.test(p.text('#runs-rows')), 'and is reported', p.text('#runs-rows'));
}
{
  let release;
  const held = new Promise(resolve => { release = resolve; });
  const answers = [
    async () => { await held; return [{ path: 'e.json', modified: 6, cases: 1 }]; },
    () => [{ error: 'e.json is no longer on disk' }, 404],
  ];
  const p = await open({ fetchImpl: served(() => answers.shift()()) });
  await p.click('#runs-rows tr.run');
  await p.type('#form [data-k="input"]', 'changed');
  await p.click('#save');
  ok(p.$('#v-experiment').inert === true, 'the editor is inert while a save is in flight');
  release();
  await tick(4);
  ok(p.$('#v-experiment').inert === false && /saved e\.json/.test(p.text('#saved')), 'and usable again once it lands',
    p.text('#saved'));

  await p.type('#form [data-k="input"]', 'worth keeping');
  await p.click('#save');
  await tick(4);
  const offer = p.$('#saved button');
  ok(offer && offer.textContent === 'Download a copy', 'a save answered 404 offers the edits as a download', p.text('#saved'));
  await p.click(offer);
  ok((await p.saved()).cases[0].input === 'worth keeping' && p.text('#dirty') === '1 unsaved change',
    'which holds them, while the page still counts them as unsaved', p.text('#dirty'));
}
{
  const server = served();
  const p = await open({ fetchImpl: server, url: 'http://127.0.0.1:9999/', token: 'from-the-document' });
  await p.click('#runs-rows tr.run');
  const tokens = [...new Set(server.sent.map(r => r.headers['X-Evals-Token']))].join();
  ok(tokens === 'from-the-document', 'with no token in the link, every request carries the one in the document', tokens);
}

// ── keyboard, the current view, and the theme ──
{
  const p = await open();
  await p.key('Enter', p.$('#runs-rows tr.run'));
  const current = p.$$('.tab[aria-current="page"]').map(t => t.dataset.view).join();
  ok(p.$('#v-experiment').classList.contains('on'), 'Enter on a row opens it');
  ok(current === 'experiment', 'and only the open view is marked as current', current);

  const root = p.d.documentElement;
  const luminance = hex => {
    const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map(v => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const token = name => p.window.getComputedStyle(p.$('.app')).getPropertyValue(name).trim();
  const contrast = () => {
    const [hi, lo] = [luminance(token('--ink3')), luminance(token('--card'))].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const lightCard = token('--card');
  ok(root.dataset.theme === 'light' && contrast() >= 4.5, 'the muted ink clears 4.5:1 on the light card',
    `${token('--ink3')} on ${lightCard}`);
  await p.click('[data-theme-toggle]');
  ok(root.dataset.theme === 'dark' && p.d.cookie.includes('strands-evals-theme=dark'),
    'the toggle flips the theme and keeps the choice in a cookie', `${root.dataset.theme} | ${p.d.cookie}`);
  ok(token('--card') !== lightCard && contrast() >= 4.5, 'and the muted ink clears 4.5:1 on the dark card too',
    `${token('--ink3')} on ${token('--card')}`);
}

done(pages.flatMap(p => p.errs));
