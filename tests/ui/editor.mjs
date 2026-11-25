/* The cases editor: an edit is written as it was made, or Save says why not. */
import { fakeServer, page, runner, served, tick } from './harness.mjs';

const { ok, same, done } = runner('editor');

const draft = name => ({ name, session_id: `s-${name}`, input: 'q', metadata: { status: 'draft' } });
const EXPERIMENT = {
  cases: [{ name: 'plain', session_id: 's-1', input: 'q' }, { name: 'padded', session_id: 's-2', input: '\nstarts on its second line' }],
  evaluators: [{ evaluator_type: 'Threshold', threshold: 0.5 }],
};
const DRAFTS = { cases: [draft('a'), draft('b')], evaluators: [] };

async function opened(files, match) {
  const p = await served(files);
  await p.openExperiment(match);
  return p;
}
const field = (p, k) => p.$(`#form [data-k="${k}"]`);
const state = p => [p.text('#dirty'), p.$('#save').disabled];

// ── a field typed into and emptied again ──
{
  const p = await opened({ 'e/experiment.json': EXPERIMENT });
  await p.type(field(p, 'expected_output'), 'x');
  const typed = state(p);
  await p.type(field(p, 'expected_output'), '');
  same([typed, state(p)], [['1 unsaved change', false], ['No unsaved changes', true]],
    'typing into an empty field is a change, and emptying it again is none: the file never had the key');

  await p.type(field(p, 'input'), '');
  await p.type(field(p, 'expected_output'), 'x');
  await p.click('#save');
  ok(/has no input/.test(p.text('#saved')) && !p.puts(), 'an emptied input is kept as text and Save says the case needs one',
    p.text('#saved'));
}

// ── text that starts with a line break ──
{
  const p = await opened({ 'e/experiment.json': EXPERIMENT });
  await p.click(p.$$('#case-list li[data-i]')[1]);
  await p.type(field(p, 'input'), field(p, 'input').value + ', and goes on');
  await p.click('#save');
  same((await p.saved()).cases[1].input, '\nstarts on its second line, and goes on', 'an edit to it keeps the line break it starts with');
}

// ── an evaluator setting that is not a number ──
{
  const p = await opened({ 'e/experiment.json': EXPERIMENT });
  await p.type(field(p, 'input'), 'changed');
  await p.click('#evs [data-ev="0"]');
  const box = p.$('#evform [data-evk="threshold"]');
  /* A browser reports text that is no number as bad input with an empty value; jsdom only empties it. */
  Object.defineProperty(box, 'validity', { configurable: true, value: { badInput: true } });
  await p.type(box, '');
  await p.click('#save');
  ok(box.classList.contains('invalid') && !p.puts() && /evaluator setting is not a number/.test(p.text('#saved')),
    'it is marked, and Save refuses rather than write the setting as cleared', p.text('#saved'));
  delete box.validity;
  await p.type(box, '0.7');
  await p.click('#save');
  same((await p.saved()).evaluators[0].threshold, 0.7, 'once it is a number again, Save writes it');
}

// ── the review pass belongs to the experiment it was started in ──
{
  const p = await opened({ 'one/experiment.json': DRAFTS, 'two/experiment.json': DRAFTS }, 'one');
  await p.click('#review-case');
  const reviewing = !p.$('#review-bar').hidden;
  await p.click('.tab[data-view="experiments"]');
  await p.openExperiment('two');
  same([reviewing, p.$('#review-bar').hidden, p.text('#crumbs b')], [true, true, 'two/experiment.json'],
    'opening another experiment does not carry the review over to its drafts');
}

// ── a review key pressed while a save is on its way ──
{
  let release = null;
  const p = await page({ fetchImpl: fakeServer({
    scan: { experiments: [{ path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 2, evaluators: [], reports: [] }] },
    file: { path: 'e.json', modified: 5, document: DRAFTS },
    put: () => new Promise(resolve => { release = () => resolve([{ path: 'e.json', modified: 6, cases: 2 }]); }),
  }) });
  await p.openExperiment();
  await p.click('#review-case');
  await p.key('a');
  await p.click('#save');
  await p.key('d');
  release();
  await tick(4);
  same([p.$$('#case-list li[data-i]').length, p.text('#dirty')], [2, 'No unsaved changes'],
    'it waits for the answer: nothing is dropped that the save did not send, and nothing is left looking saved');
}

// ── the address names the selected case ──
{
  const three = { cases: ['a', 'b', 'c'].map(name => ({ name, session_id: `s-${name}`, input: 'q' })), evaluators: [] };
  const p = await page({ fetchImpl: fakeServer({
    scan: { experiments: [{ path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 3, evaluators: [], reports: [] }] },
    file: { path: 'e.json', modified: 5, document: three },
  }) });
  await p.openExperiment();
  /* Leave for the Reports list and come Back: the step that fails when the address names a case that is gone. */
  const awayAndBack = async () => {
    await p.click('.tab[data-view="reports"]');
    await p.travel(-1);
    return [p.view(), p.text('#form-title'), p.window.location.pathname];
  };
  await p.click(p.$$('#case-list li[data-i]')[1]);
  await p.type(field(p, 'name'), 'b2');
  same(await awayAndBack(), ['v-experiment', 'b2', '/experiments/e.json/cases/b2'],
    'a renamed case takes its address with it, so Back returns to the case');
  await p.click('#del-case');
  same(await awayAndBack(), ['v-experiment', 'c', '/experiments/e.json/cases/c'],
    'deleting it moves the address to the case selected in its place');
  await p.click('#saved .reload');
  same(await awayAndBack(), ['v-experiment', 'b2', '/experiments/e.json/cases/b2'], 'and Undo moves it back');
  await p.click('#add-case');
  same(await awayAndBack(), ['v-experiment', 'case_4', '/experiments/e.json/cases/case_4'], 'a case that was added is named too');
  await p.click('#revert');
  same(await awayAndBack(), ['v-experiment', 'c', '/experiments/e.json/cases/c'], 'as is the one selected after Revert');
}

// ── a name being typed, in a browser that rations how often the address may be written ──
{
  const three = { cases: ['a', 'b', 'c'].map(name => ({ name, session_id: `s-${name}`, input: 'q' })), evaluators: [] };
  const p = await page({ fetchImpl: fakeServer({
    scan: { experiments: [{ path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 3, evaluators: [], reports: [] }] },
    file: { path: 'e.json', modified: 5, document: three },
  }) });
  await p.openExperiment();
  await p.click(p.$$('#case-list li[data-i]')[1]);

  /* Safari raises this after 100 history calls in 30 seconds. `rationed` is when the page has used them up. */
  const written = [];
  let rationed = false;
  for (const call of ['pushState', 'replaceState']) {
    const really = p.window.history[call].bind(p.window.history);
    p.window.history[call] = (...args) => {
      if (rationed) throw new p.window.DOMException(`Attempt to use history.${call}() more than 100 times per 30 seconds`, 'SecurityError');
      written.push(call);
      return really(...args);
    };
  }
  const name = field(p, 'name');
  const key = async (value, event = 'input') => {
    name.value = value;
    name.dispatchEvent(new p.window.Event(event, { bubbles: true }));
    await tick();
  };
  const at = () => p.window.location.pathname;

  for (const typed of ['b', 'bx', 'bxy', 'bxyz']) await key(typed);
  same([written.length, p.text('#form-title'), p.text('#case-list li.on .nm')], [0, 'bxyz', 'bxyz'],
    'each key of a name moves the form and the list, and none of them writes the address');
  await key('bxyz', 'change');
  same([written.join(), at()], ['replaceState', '/experiments/e.json/cases/bxyz'], 'leaving the field writes it once, in place');
  await key('bq');
  await new Promise(resolve => setTimeout(resolve, 650));
  same([written.join(), at()], ['replaceState,replaceState', '/experiments/e.json/cases/bq'],
    'and so does a pause in the typing, for a reader who goes Back without leaving the field');

  rationed = true;
  await key('r2');
  await key('r2', 'change');
  await p.click('#del-case');
  same([p.errs.length, at(), /^Removed r2\./.test(p.text('#saved')), !!p.$('#saved .reload')],
    [0, '/experiments/e.json/cases/bq', true, true],
    'a browser that will write no more addresses leaves this one behind, and the editor carries on: Delete still offers Undo');
  rationed = false;
  await p.click('#saved .reload');
  same(at(), '/experiments/e.json/cases/r2', 'and the next write the browser allows puts the address right');
}

// ── a second save, straight after the first ──
{
  let stamp = 5;
  const sent = [];
  const p = await page({ fetchImpl: fakeServer({
    scan: { experiments: [{ path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 2, evaluators: [], reports: [] }] },
    file: { path: 'e.json', modified: 5, document: EXPERIMENT },
    /* As the server does: a save is refused unless it names the time the file was last written. */
    put: body => {
      sent.push(body.modified);
      return body.modified === stamp ? [{ path: 'e.json', modified: ++stamp, cases: 2 }] : [{ error: 'e.json changed on disk since it was loaded' }, 409];
    },
  }) });
  await p.openExperiment();
  await p.type(field(p, 'input'), 'one');
  await p.click('#save');
  await p.type(field(p, 'input'), 'two');
  await p.click('#save');
  same([sent, p.text('#saved')], [[5, 6], 'saved e.json · 2 cases'], 'it is checked against the time the first save left the file with, not the time it was read');
}

done();
