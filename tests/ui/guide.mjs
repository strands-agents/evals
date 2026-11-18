/* Guidance and outbound links: the per-view guide, the ? help on fields and settings, and the
   documentation links with their campaign tags. */
import { fakeServer, page, runner, tick } from './harness.mjs';

const { ok, done } = runner('guide');
const pages = [];
const open = async opts => { const p = await page(opts); pages.push(p); return p; };

const DOCS = 'https://strandsagents.com/docs/user-guide/evals-sdk/';
const TAGS = 'utm_source=strands-evals-ui&utm_medium=product&utm_campaign=evals-ui&utm_content=';
/* A docs link as `page|placement|anchor`, or null when it is not tagged the way every one must be. */
const parse = a => {
  const m = new RegExp(`^${DOCS.replace(/[.]/g, '\\.')}([\\w/-]*)\\?${TAGS}([\\w-]+)(#[\\w-]+)?$`).exec(a.href);
  return m && [m[1], m[2], m[3] || ''].join('|');
};
const docLinks = p => p.$$(`a[href^="${DOCS}"]`);
const EXPERIMENT = {
  cases: [{ name: 'a', session_id: 's-a', input: 'q' }],
  evaluators: [{ evaluator_type: 'Contains', value: 'x', case_sensitive: true }, { evaluator_type: 'AcmeSecretJudge', threshold: 1 }],
};
const helpText = p => p.text('.help-pop');
const helpLink = p => parse(p.$('.help-pop a'));

// ── every link into the docs is tagged, names its placement, and opens beside the page ──
{
  const p = await open();
  const links = docLinks(p);
  const untagged = links.filter(a => !parse(a)).map(a => a.href);
  ok(links.length >= 20 && untagged.length === 0, 'every docs link carries the campaign tags and a placement',
    `${links.length} links, untagged: ${untagged.join(' ')}`);
  ok(links.every(a => a.target === '_blank' && a.rel.includes('noopener')), 'and opens in a new tab without an opener');
  const chrome = p.$$('.bar a, .sections a').map(a => parse(a) || a.href);
  ok(JSON.stringify(chrome) === JSON.stringify([
    '|nav-docs|', 'quickstart/|nav-quickstart|', 'evaluators/|nav-evaluators|',
    'https://github.com/strands-agents/evals',
    '|nav-slug|', 'how-evaluation-works/|sidebar-how-it-works|', 'cli/|sidebar-cli|',
    'https://github.com/strands-agents/evals/issues',
  ]), 'the bar and the sidebar link to the docs and the repository', chrome.join(' , '));
  ok(links.some(a => /\?utm_source.*#what-an-evaluation-is-made-of$/.test(a.href)),
    'an anchor comes after the tags, where a browser reads it');
  ok(!p.d.documentElement.innerHTML.includes('t=tok'), 'no link carries the page token');
}

// ── the guide: on every view, open until it is turned off, and that choice is kept ──
{
  const p = await open();
  const panels = () => p.$$('details.guide');
  const views = p.$$('main > section.view').map(s => s.id);
  ok(panels().length === views.length && views.every(id => p.$(`#${id} > details.guide`)),
    'every view has a guide', `${panels().length} guides for ${views.length} views`);
  ok(panels().every(d => d.open && d.querySelector('ol li') && docLinks({ $$: s => [...d.querySelectorAll(s)] }).length),
    'open to begin with, with steps and a way into the docs');
  ok(parse(p.$('#v-report details.guide a')) === 'how-evaluation-works/|guide-report|#the-levels-an-evaluator-scores-at',
    'its links name the view they sit on', parse(p.$('#v-report details.guide a')));

  await p.click('[data-guide-toggle]');
  ok(panels().every(d => !d.open) && p.$('.app').dataset.guide === 'off' && p.d.cookie.includes('strands-evals-guide=off'),
    'the Guide switch closes them all and remembers', p.d.cookie);
  await p.click('[data-guide-toggle]');
  ok(panels().every(d => d.open), 'and opens them again');

  const small = await open({ phone: true });
  ok(small.$$('details.guide').every(d => !d.open) && small.$('.app').dataset.guide === 'off',
    'on a phone-sized window they start closed, so the view is not pushed off the first screen');
  await small.click('[data-guide-toggle]');
  ok(small.$$('details.guide').every(d => d.open), 'and the Guide switch still opens them');
}

// ── a ? on a case field says what it is, what it takes, and where the docs cover it ──
{
  const p = await open();
  await p.openFolder({ 'run/experiment.json': EXPERIMENT });
  await p.click('#runs-rows tr.run');
  const fields = p.$$('#form .help').map(b => b.dataset.help);
  ok(JSON.stringify(fields) === JSON.stringify(['name', 'input', 'expected_output', 'expected_assertion',
    'expected_trajectory', 'expected_interactions', 'expected_environment_state', 'metadata'].map(k => `field:${k}`)),
  'every case field has one', fields.join());

  const button = p.$('.help[data-help="field:expected_trajectory"]');
  await p.click(button);
  ok(!p.$('.help-pop').hidden && button.getAttribute('aria-expanded') === 'true'
    && /steps you expect the agent to take/.test(helpText(p)) && /A list of tool names/.test(helpText(p)),
  'it opens with what the field is and what it accepts', helpText(p));
  ok(helpLink(p) === 'evaluators/trajectory_evaluator/|field-expected_trajectory|', 'and links to the page about it', helpLink(p));
  ok(button.nextElementSibling === p.$('.help-pop'), 'placed after the button, so Tab reaches its link next');
  await p.key('Escape');
  ok(p.$('.help-pop').hidden && button.getAttribute('aria-expanded') === 'false', 'Escape closes it');
  await p.click(button);
  await p.click(button);
  ok(p.$('.help-pop').hidden, 'and so does the button itself');
}

// ── evaluators: described from the library when a server supplies it, linked to their own page either way ──
{
  const catalog = { Contains: { summary: 'Checks if actual_output contains a substring.', settings: [
    { name: 'value', required: true, default: null }, { name: 'case_sensitive', required: false, default: 'true' },
  ] } };
  const run = { id: 'e.json', dir: '.', report: null, link: null,
    experiment: { path: 'e.json', name: 'e.json', size: 1, modified: 5, cases: 1, evaluators: EXPERIMENT.evaluators } };
  const p = await open({ fetchImpl: fakeServer({
    config: { root: '/served', depth: 3, read_only: false, evaluators: catalog }, runs: [run], file: { path: 'e.json', modified: 5, document: EXPERIMENT },
  }) });
  await p.click('#runs-rows tr.run');
  await p.click(p.$$('#evs .evchip')[0]);
  ok(/Checks if actual_output contains a substring\./.test(p.text('#evform .evsum')), 'the form says what the evaluator checks',
    p.text('#evform .evsum'));
  ok(parse(p.$('#evform .evsum a')) === 'evaluators/deterministic_evaluators/|evaluator-Contains|', 'with its page',
    parse(p.$('#evform .evsum a')));
  await p.click('.help[data-help="setting:Contains.case_sensitive"]');
  ok(/letter case/.test(helpText(p)) && /Default: true\./.test(helpText(p)), 'a setting says what it does and its default', helpText(p));
  await p.click('.help[data-help="setting:Contains.value"]');
  ok(/Required\./.test(helpText(p)), 'or that it is required', helpText(p));

  await p.click(p.$$('#evs .evchip')[1]);
  ok(/Not one of the built-in evaluators/.test(p.text('#evform .evsum')), 'an evaluator the library does not ship says so',
    p.text('#evform .evsum'));
  ok(parse(p.$('#evform .evsum a')) === 'evaluators/custom_evaluator/|evaluator-custom|',
    'and its name stays out of the link', parse(p.$('#evform .evsum a')));
}
{
  const p = await open();
  await p.openFolder({ 'run/experiment.json': { ...EXPERIMENT, evaluators: [{ evaluator_type: 'ToolSelectionAccuracyEvaluator' }, { evaluator_type: 'HelpfulnessEvaluator' }] } });
  await p.click('#runs-rows tr.run');
  const pagesFor = [];
  for (const chip of [0, 1]) {
    await p.click(p.$$('#evs .evchip')[chip]);
    pagesFor.push(parse(p.$('#evform .evsum a')));
  }
  ok(JSON.stringify(pagesFor) === JSON.stringify([
    'evaluators/tool_selection_evaluator/|evaluator-ToolSelectionAccuracyEvaluator|',
    'evaluators/helpfulness_evaluator/|evaluator-HelpfulnessEvaluator|',
  ]), 'with no server, a built-in still links to its own page', pagesFor.join(' , '));

  await p.click('.tab[data-view="report"]');
  ok(parse(p.$('#rep-rows a')) === 'cli/run/|empty-report|#experiment-file-run', 'an experiment with no report links to how to run it',
    parse(p.$('#rep-rows a')));
}

// ── tooltips: a label explains itself on hover or focus, from the glossary or its own text ──
{
  const p = await open();
  await p.openFolder({ 'run/experiment.json': EXPERIMENT });
  const tip = p.$('.tip');
  const over = async (el, type = 'mouseover') => {
    el.dispatchEvent(new p.window.MouseEvent(type, { bubbles: true }));
    await tick();
  };
  ok(tip.hidden, 'nothing shows until something is pointed at');

  await p.click('.tab[data-view="report"]');
  await over(p.$('#v-report th[data-term="score"]'));
  ok(!tip.hidden && tip.querySelector('b').textContent === 'Score' && /usually from 0 to 1/.test(tip.textContent),
    'a column head shows its glossary entry', tip.textContent);
  await over(p.$('#rep-find'));
  ok(tip.hidden, 'and it goes when the pointer moves on');

  await p.click('#runs-rows tr.run');
  const chip = p.$('#evs .evchip');
  const title = chip.title;
  await over(chip);
  ok(title && tip.textContent === title && !chip.hasAttribute('title'),
    'a plain title is shown the same way, and not a second time by the browser', `${title} / ${tip.textContent}`);

  const tab = p.$('.tab[data-view="experiment"]');
  tab.dispatchEvent(new p.window.FocusEvent('focusin', { bubbles: true }));
  await tick();
  ok(!tip.hidden && /cases and evaluators of the open experiment/.test(tip.textContent), 'focus shows it too', tip.textContent);
  await p.key('Escape');
  ok(tip.hidden, 'and Escape dismisses it');

  const explained = sel => p.$$(sel).every(el => el.dataset.term || el.dataset.tip || el.title);
  ok(explained('main th') && explained('.sections .tab') && explained('.seg button:not([data-f="all"])'),
    'every column head, view and filter says what it is');
}

done(pages.flatMap(p => p.errs));
