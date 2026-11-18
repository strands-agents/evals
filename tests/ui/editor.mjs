/* The cases editor: an edit is written as it was made, or Save says why not. */
import { fakeServer, page, runner, tick } from './harness.mjs';

const { ok, same, done } = runner('editor');

const draft = name => ({ name, session_id: `s-${name}`, input: 'q', metadata: { status: 'draft' } });
const EXPERIMENT = {
  cases: [{ name: 'plain', session_id: 's-1', input: 'q' }, { name: 'padded', session_id: 's-2', input: '\nstarts on its second line' }],
  evaluators: [{ evaluator_type: 'Threshold', threshold: 0.5 }],
};
const DRAFTS = { cases: [draft('a'), draft('b')], evaluators: [] };

async function opened(files, match) {
  const p = await page();
  await p.openFolder(files);
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
  ok(/has no input/.test(p.text('#saved')) && !p.window.__blobs.length, 'an emptied input is kept as text and Save says the case needs one',
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
  ok(box.classList.contains('invalid') && !p.window.__blobs.length && /evaluator setting is not a number/.test(p.text('#saved')),
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
  same([reviewing, p.$('#review-bar').hidden, p.text('#path')], [true, true, 'two/experiment.json'],
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

done();
