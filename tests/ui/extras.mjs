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
  await p.openExperiment();
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
  await p.openReport();
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
  await p.openReport();
  ok(p.$$('#rep-rows tr.case').length === 1 && p.text('#r-score') === '1.00', 'renders its row and score',
    p.text('#v-report'));
  await p.click('#rep-rows tr.case');
  ok(/returned no outputs/.test(p.text('#rep-rows tr.detail')),
    'and says the row has no outputs rather than inventing a verdict', p.text('#rep-rows tr.detail'));
}

// ── the two lists: what each report was made from, and what each experiment has made ──
{
  const p = await open();
  await p.openFolder({
    'paired/experiment.json': experiment({ ...CASE, expected_output: 'yes' }),
    'paired/report.json': report(0.25, CASE),
    'paired/experiment.saved.json': experiment(CASE),
    'mismatch/experiment.json': experiment(CASE),
    'mismatch/report.json': report(0, { name: 'somebody-elses-case', session_id: 's-z' }),
    'only-report/report.json': report(0.4, { name: 'nobody-holds-this' }),
    'only-exp/experiment.json': experiment(CASE),
  });
  const row = path => p.$$('tr.row').find(r => (r.dataset.report || r.dataset.experiment) === path);
  const says = path => p.text(row(path).children[2]);
  const on = () => p.$('.view.on').id;
  const edit = () => p.$('#form [data-k="expected_output"]').value;
  const experiments = () => p.click('.tab[data-view="experiments"]');

  ok(p.text('#reports-count') === '3 reports' && p.text('#experiments-count') === '4 experiments',
    'every report and every experiment file is listed once', `${p.text('#reports-count')} ${p.text('#experiments-count')}`);
  ok(says('paired/report.json') === 'paired/experiment.json pairing verified', 'report.json is of the experiment.json beside it',
    says('paired/report.json'));
  ok(says('paired/experiment.json') === 'report.json 0.25' && says('paired/experiment.saved.json') === 'not run yet',
    'which lists it with its score, leaving the other experiment in the folder not run', says('paired/experiment.json'));
  ok(says('mismatch/report.json') === 'mismatch/experiment.json does not match', 'a report for other cases is flagged',
    says('mismatch/report.json'));
  ok(says('only-report/report.json') === 'run from code', 'a report no experiment file holds was run from code',
    says('only-report/report.json'));

  await p.click(row('only-report/report.json'));
  ok(on() === 'v-report' && p.text('#r-score') === '0.40' && p.text('.tab.on') === 'Reports',
    'a row opens its report, with the sidebar still on Reports', `${on()} ${p.text('.tab.on')}`);

  await p.click(row('paired/report.json').querySelector('a[data-experiment]'));
  ok(on() === 'v-experiment' && p.text('#crumbs') === 'Experiments/paired/experiment.json' && p.text('.tab.on') === 'Experiments',
    'the experiment named in a report’s row opens the experiment, not the report', `${on()} ${p.text('#crumbs')}`);
  ok(p.text('#facts') === '1 case · 0 evaluators · Report: paired/report.json 0.25', 'whose header links the report made from it',
    p.text('#facts'));
  await p.click('#facts a[data-report]');
  ok(on() === 'v-report' && p.text('#facts') === '0.25 overall · 1 row · 1 failing · of paired/experiment.json pairing verified',
    'and the report’s header links back to the experiment it is of', p.text('#facts'));

  await p.click('#facts a[data-experiment]');
  await p.type('#form [data-k="expected_output"]', 'edited, not saved');
  await p.click('#facts a[data-report]');
  ok(on() === 'v-report' && !p.$('.ask'), 'opening a report asks nothing: an unsaved edit is not in its way', on());
  await experiments();
  await p.click(row('paired/experiment.json'));
  ok(on() === 'v-experiment' && edit() === 'edited, not saved', 'and the open experiment’s row goes back to it with the edit intact',
    edit());

  await experiments();
  row('only-exp/experiment.json').focus();
  await p.click(row('only-exp/experiment.json'));
  ok(/^1 unsaved change will be lost if you open another experiment\./.test(p.text('.ask')), 'another experiment asks first',
    p.text('.ask'));
  ok(p.d.activeElement.dataset.answer === 'no', 'with the keyboard on the answer that loses nothing', p.d.activeElement.textContent);
  await p.click('.ask [data-answer="no"]');
  ok(on() === 'v-experiments' && edit() === 'edited, not saved', 'Stay leaves the edit where it was', `${on()} ${edit()}`);
  ok(p.d.activeElement === row('only-exp/experiment.json'), 'and puts the keyboard back on the row that asked', p.d.activeElement.tagName);
  await p.click(row('only-exp/experiment.json'));
  await p.click('.ask [data-answer="yes"]');
  ok(on() === 'v-experiment' && p.text('#facts code') === 'strands-evals run experiment.json --agent module:attr -o report.json'
    && /· Not run yet\. To run it: /.test(p.text('#facts')),
  'Discard and open opens it, and an experiment with no report says how to run it', p.text('#facts'));
}

// ── deep links ──
{
  const p = await linked('/experiments/experiment.json/cases/refund_edge_case');
  ok(p.$('.view.on').id === 'v-experiment' && p.text('#case-list li.on .nm') === 'refund_edge_case',
    'an address opens the experiment and the case it names', `${p.$('.view.on').id} ${p.text('#case-list li.on')}`);
  await p.click('#facts a[data-report]');
  ok(p.window.location.hash === '#/reports/report.json', 'and its report has an address of its own', p.window.location.hash);
}
for (const [route, view, rewritten, what] of [
  ['/reports', 'reports', '/reports', 'the list of reports'],
  ['/experiments', 'experiments', '/experiments', 'the list of experiments'],
  ['/reports/report.json', 'report', '/reports/report.json', 'one report'],
  ['/experiments/experiment.json', 'experiment', '/experiments/experiment.json/cases/order_status_lookup',
    'one experiment, on its first case'],
  ['/nope/at/all', 'reports', '/reports', 'an address the page does not have falls back to Reports'],
  ['/reports/gone.json', 'reports', '/reports', 'a report that is not in the folder falls back to the list'],
  ['/experiments/gone.json', 'experiments', '/experiments', 'as does an experiment that is not'],
  ['/experiments/experiment.json/cases/%E0%A4%A', 'experiment', '/experiments/experiment.json/cases/order_status_lookup',
    'a malformed escape names no case, so the first is shown'],
  ['/optimization/tuning/overview', 'reports', '/reports', 'an Optimization address with no optimizer run falls back to Reports'],
]) {
  const p = await linked(route);
  const on = p.$$('.view.on').map(v => v.id).join();
  ok(p.$('.app').dataset.ready === '1' && on === `v-${view}` && p.window.location.hash === '#' + rewritten,
    `${route}: ${what}, and the address says so`, `${on} ${p.window.location.hash}`);
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

// ── review pass: its shortcuts belong to the experiment page ──
{
  const p = await opened(experiment({ ...CASE, metadata: { status: 'draft' } }, { name: 'b', session_id: 's-b', input: 'q' }));
  await p.click('#review-case');
  await p.click('.tab[data-view="experiments"]');
  await p.key('d');
  await p.openExperiment();
  ok(p.$$('#case-list li[data-i]').length === 2, 'd pressed on another view drops nothing', p.$$('#case-list li[data-i]').length);
}

// ── a report with no experiment of its name is paired by its cases, or not at all ──
{
  const T = { name: 't', session_id: 's-t', input: 'q' };
  const p = await open();
  await p.openFolder({
    'suite/checkout.json': experiment(CASE),
    'suite/billing.json': experiment({ name: 'z', session_id: 's-z', input: 'q' }),
    'suite/checkout.report.json': report(1, CASE),
    'suite/tuesday.report.json': report(0.5, CASE),
    'solo/cases.json': experiment(T),
    'solo/last-night.json': report(0.75, { name: 'somebody-elses-case' }),
    'twins/left/e.json': experiment(T),
    'twins/right/e.json': experiment(T),
    'twins/shared.report.json': report(1, T),
    'code/adhoc.report.json': report(0.25, { name: 'made-in-code', session_id: 's-code' }),
  });
  const from = path => p.text(p.$(`#reports-rows tr.row[data-report="${path}"]`).children[2]);
  const made = path => p.$$(`#experiments-rows tr.row[data-experiment="${path}"] a[data-report]`).map(a => a.textContent).join();

  ok(from('suite/tuesday.report.json') === 'suite/checkout.json pairing verified',
    'a second report goes to the experiment that holds its cases, not to another one beside it', from('suite/tuesday.report.json'));
  ok(made('suite/checkout.json') === 'checkout.report.json,tuesday.report.json' && made('suite/billing.json') === '',
    'so one experiment lists both of its reports and the other none', `${made('suite/checkout.json')} | ${made('suite/billing.json')}`);
  ok(from('solo/last-night.json') === 'solo/cases.json does not match',
    'the only report and the only experiment in a folder go together, and a mismatch is said', from('solo/last-night.json'));
  ok(from('twins/shared.report.json') === 'run from code' && made('twins/left/e.json') + made('twins/right/e.json') === '',
    'two experiments equally near that both hold the cases are a tie, which is not guessed', from('twins/shared.report.json'));

  await p.openReport('adhoc');
  ok(p.$('.view.on').id === 'v-report' && p.text('#crumbs') === 'Reports/code/adhoc.report.json' && p.text('#r-score') === '0.25',
    'a report made in code opens on its own, named in the page header', `${p.text('#crumbs')} ${p.text('#r-score')}`);
  ok(p.text('#facts') === '0.25 overall · 1 row · 1 failing · run from code' && !p.$('#facts a'),
    'which says it was run from code and links no experiment', p.text('#facts'));
  await p.click('.tab[data-view="experiments"]');
  ok(!p.$('#experiments-rows tr.row.on') && p.text('#experiments-count') === '5 experiments', 'and opens no experiment with it',
    p.text('#experiments-count'));

  await p.openExperiment('checkout');
  const links = p.$$('#facts a[data-report]').map(a => a.textContent).join();
  ok(links === 'suite/checkout.report.json,suite/tuesday.report.json' && /^1 case · 0 evaluators · Reports: /.test(p.text('#facts')),
    'an experiment’s header links each report made from it', p.text('#facts'));
  await p.click(p.$$('#facts a[data-report]')[1]);
  ok(p.text('#crumbs') === 'Reports/suite/tuesday.report.json' && p.text('#r-score') === '0.50', 'and each link opens that report',
    p.text('#crumbs'));
  await p.click('#crumbs a');
  ok(p.$('.view.on').id === 'v-reports' && p.$('#reports-rows tr.row.on').dataset.report === 'suite/tuesday.report.json',
    'the header’s first crumb leads back to the list, where the open report is marked', p.$('.view.on').id);
}

// ── delete and add-a-tool need no dialog: one is undone, the other typed in place ──
{
  const p = await opened(experiment(CASE, { name: 'b', session_id: 's-b', input: 'q', expected_trajectory: ['lookup'] }));
  const listed = () => p.$$('#case-list li[data-i] .nm').map(el => el.textContent).join();
  await p.click('#del-case');
  ok(listed() === 'b' && /^Removed a\./.test(p.text('#saved')), 'Delete removes the case at once and says so', `${listed()} ${p.text('#saved')}`);
  await p.click('#saved button');
  ok(listed() === 'a,b' && p.text('#case-list li.on .nm') === 'a' && p.window.eval('countDirty()') === 0,
    'and Undo puts it back where it was, with nothing left to save', listed());

  await p.click(p.$$('#case-list li[data-i]')[1]);
  const tools = () => p.text('#form [data-chips="expected_trajectory"]');
  await p.click('#form [data-add]');
  await p.type('#form input[data-add-to]', 'refund');
  await p.key('Enter', p.$('#form input[data-add-to]'));
  ok(/lookup/.test(tools()) && /refund/.test(tools()) && !p.$('#form input[data-add-to]'), 'a tool is typed where the button was and added on Enter', tools());
  await p.click('#form [data-add]');
  await p.type('#form input[data-add-to]', 'never');
  await p.key('Escape', p.$('#form input[data-add-to]'));
  ok(!/never/.test(tools()) && !p.$('#form input[data-add-to]'), 'and Escape gives up without adding it', tools());
}

// ── opening a folder replaces what is open, so it asks before unsaved edits go with it ──
{
  const p = await opened(experiment(CASE));
  let pickers = 0;
  p.$('#dirinput').click = () => { pickers += 1; };
  await p.click('[data-act="pick"]');
  ok(pickers === 1 && !p.$('.ask'), 'Open folder goes straight to the picker when nothing is unsaved', pickers);
  await p.type('#form [data-k="input"]', 'unsaved');
  await p.click('[data-act="pick"]');
  ok(/^1 unsaved change will be lost if you open a folder\./.test(p.text('.ask')) && pickers === 1,
    'and asks first when something is', p.text('.ask'));
  await p.click('.ask [data-answer="no"]');
  ok(pickers === 1 && p.$('#form [data-k="input"]').value === 'unsaved', 'Stay opens no picker and keeps the edit', pickers);
  await p.click('[data-act="pick"]');
  await p.click('.ask [data-answer="yes"]');
  ok(pickers === 2, 'Discard and open goes on to it', pickers);
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

// ── history: a move between pages is a step, and Back and Forward retrace them ──
{
  const two = { name: 'a2', session_id: 's-a2', input: 'q' };
  const p = await open();
  await p.openFolder({
    'one/experiment.json': experiment(CASE, two), 'one/report.json': report(1, CASE),
    'two/experiment.json': experiment(two),
  });
  const history = p.window.history;
  const at = () => `${p.$('.view.on').id} ${p.text('#path')}`;
  const travel = async delta => { history.go(delta); await tick(6); return at(); };
  const experiments = () => p.click('.tab[data-view="experiments"]');
  const start = history.length;
  await p.openExperiment('one /');
  const scrolls = p.window.__scrolls;
  await p.click('#facts a[data-report]');
  await p.openReport('one /');
  ok(p.window.__scrolls === scrolls + 1, 'a page moved to starts at its top; the one already open stays put', p.window.__scrolls - scrolls);
  await experiments();
  const list = at();
  await p.openExperiment('two /');
  ok(history.length === start + 4, 'each move between pages adds a step', history.length - start);
  const retraced = [await travel(-1), await travel(-1), await travel(-1), await travel(3)];
  ok(retraced.join() === `${list},v-report one/report.json,v-experiment one/experiment.json,v-experiment two/experiment.json`
    && /^v-experiments /.test(list),
  'Back walks them in reverse, across reports and experiments, and Forward returns', retraced.join());

  await travel(-3);
  const steps = history.length;
  await p.click(p.$$('#case-list li[data-i]')[1]);
  ok(history.length === steps && p.window.location.hash === '#/experiments/one/experiment.json/cases/a2',
    'picking a case corrects the address without adding a step', p.window.location.hash);

  await experiments();
  await p.openExperiment('two /');
  const reread = [await travel(-2), p.$('#form [data-k="name"]').value, await travel(2)];
  ok(reread.join() === 'v-experiment one/experiment.json,a2,v-experiment two/experiment.json',
    'Back reopens the case that was being read, and the way Forward is still there', reread.join());

  p.window.location.hash = '#/reports/one/report.json';
  await tick(6);
  ok(at() === 'v-report one/report.json', 'an address typed into the open page is followed', at());

  await experiments();
  await p.openExperiment('one /');
  await p.type('#form [data-k="input"]', 'unsaved');
  const back = await travel(-2);
  ok(back === 'v-report one/report.json' && !p.$('.ask'), 'Back to a report asks nothing: unsaved edits are not in its way', back);
  await travel(-1);
  ok(/1 unsaved change will be lost if you open another experiment\./.test(p.text('.ask')),
    'Back to another experiment asks before dropping them', p.text('.ask'));
  await p.click('.ask [data-answer="no"]');
  ok(!p.$('.ask') && p.$('#form [data-k="input"]').value === 'unsaved'
    && p.window.location.hash === '#/experiments/one/experiment.json/cases/a',
  'Stay keeps them, and the address says where the page still is', p.window.location.hash);
  await experiments();
  await p.openExperiment('two /');
  await p.click('.ask [data-answer="yes"]');
  await tick(6);
  ok(!p.$('.ask') && p.text('#path') === 'two/experiment.json', 'choosing another experiment asks too, and Discard and open opens it',
    p.text('#path'));

  const fresh = await linked('/reports/report.json');
  ok(fresh.window.history.length === 1, 'opening a link adds no step of its own', fresh.window.history.length);
}

// ── history: Back pressed again before the step before it has loaded ──
{
  const reports = ['a.json', 'b.json'].map(path =>
    ({ path, name: path, dir: '.', size: 1, modified: 5, rows: 1, overall_score: 1, experiment: null, link: null }));
  const server = fakeServer({ scan: { reports }, file: path => ({ path, modified: 5, document: report(1, CASE) }) });
  let slow = false, release;
  const held = new Promise(resolve => { release = resolve; });
  const p = await open({ fetchImpl: async (url, opts) => {
    if (slow && String(url).includes('path=a.json')) await held;
    return server(url, opts);
  } });
  await p.openReport('a.json');
  await p.openReport('b.json');
  const steps = p.window.history.length;
  slow = true;
  p.window.history.back();
  await tick();
  p.window.history.back();
  await tick(6);
  release();
  await tick(6);
  const at = `${p.$('.view.on').id} ${p.window.location.pathname}`;
  ok(at === 'v-reports /reports' && p.window.history.length === steps,
    'the page ends where the second Back went, not on the report the first was still loading', `${at}, ${p.window.history.length - steps} steps added`);
}

// ── served mode, against a fake server ──
const served = put => fakeServer({
  scan: { experiments: [{ path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 1, evaluators: [], reports: [] }] },
  file: { path: 'e.json', modified: 5, document: experiment({ name: 'a', input: 'q' }) },
  put,
});
{
  const p = await open({ fetchImpl: fakeServer({ scan: () => [{ error: 'scan blew up' }, 500] }) });
  ok(/^served/.test(p.text('#mode-tag')) && !p.$('tr.row'),
    'a failed scan leaves the page in served mode, not on the sample', p.text('#mode-tag'));
  ok(/scan blew up/.test(p.text('#reports-rows')) && /scan blew up/.test(p.text('#experiments-rows')),
    'and is reported on both lists', p.text('#reports-rows'));
}
{
  let release;
  const held = new Promise(resolve => { release = resolve; });
  const answers = [
    async () => { await held; return [{ path: 'e.json', modified: 6, cases: 1 }]; },
    () => [{ error: 'e.json is no longer on disk' }, 404],
  ];
  const p = await open({ fetchImpl: served(() => answers.shift()()) });
  await p.openExperiment();
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
  await p.openExperiment();
  const tokens = [...new Set(server.sent.map(r => r.headers['X-Evals-Token']))].join();
  ok(tokens === 'from-the-document', 'with no token in the link, every request carries the one in the document', tokens);
}

// ── keyboard, the current page, and the theme ──
{
  const p = await open();
  await p.key('Enter', p.$('#experiments-rows tr.row'));
  const marked = () => `${p.$$('.tab.on').map(t => t.dataset.view).join()} | ${p.$$('.tab[aria-current="page"]').map(t => t.dataset.view).join()}`;
  ok(p.$('#v-experiment').classList.contains('on'), 'Enter on a row opens it');
  ok(marked() === 'experiments | ', 'the sidebar marks the list an open page came from, without calling it the current page', marked());
  await p.click('.tab[data-view="experiments"]');
  ok(marked() === 'experiments | experiments', 'which it is once the list itself is showing', marked());

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
