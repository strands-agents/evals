/* Guidance and outbound links: the per-view guide, the ? help on fields and settings, and the
   documentation links with their campaign tags. */
import { SAMPLE, fakeServer, page, runner, served, tick } from './harness.mjs';

const { ok, done } = runner('guide');

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
  evaluators: [{ evaluator_type: 'Contains', value: 'x', 'value.unit': 'word', case_sensitive: true }, { evaluator_type: 'AcmeSecretJudge', threshold: 1 }],
};
const helpText = p => p.text('.help-pop');
const helpLink = p => parse(p.$('.help-pop a'));

// ── every link into the docs is tagged, names its placement, and opens beside the page ──
{
  const p = await served(SAMPLE);
  const views = p.$$('main > section.view').map(s => s.id.slice(2));
  /* The chrome's links, and the guide's for each view in turn: the panel holds one view at a time. */
  const links = views.flatMap(view => {
    p.window.renderGuide(view);
    return docLinks(p).map(a => ({ href: a.href, tagged: parse(a), blank: a.target === '_blank' && a.rel.includes('noopener') }));
  });
  const untagged = links.filter(a => !a.tagged).map(a => a.href);
  ok(new Set(links.map(a => a.href)).size >= 20 && untagged.length === 0, 'every docs link carries the campaign tags and a placement',
    `${new Set(links.map(a => a.href)).size} links, untagged: ${untagged.join(' ')}`);
  ok(links.every(a => a.blank), 'and opens in a new tab without an opener');
  /* The file tree's leaves are in the sidebar and are links, but they open a file in the page. */
  const chrome = p.$$('.bar a, .sections a:not(.runrow)').map(a => parse(a) || a.href);
  ok(JSON.stringify(chrome) === JSON.stringify([
    '|nav-docs|', 'quickstart/|nav-quickstart|', 'evaluators/|nav-evaluators|',
    'https://github.com/strands-agents/evals',
    '|nav-slug|', 'how-evaluation-works/|sidebar-how-it-works|', 'cli/|sidebar-cli|',
    'https://github.com/strands-agents/evals/issues',
  ]), 'the bar and the sidebar link to the docs and the repository', chrome.join(' , '));
  ok(links.some(a => /\?utm_source.*#what-an-evaluation-is-made-of$/.test(a.href)),
    'an anchor comes after the tags, where a browser reads it');
  ok(!p.$$('a').some(a => a.href.includes(p.token)), 'no link carries the page token');
  const unguided = views.filter(view => {
    p.$('#guide-body').innerHTML = '';
    p.window.renderGuide(view);
    return !p.$('#guide-body ol li') || !p.$(`#guide-body a[href^="${DOCS}"]`);
  });
  ok(unguided.length === 0, 'every view has a guide of its own, with steps and a way into the docs', unguided.join());
}

// ── the guide: a panel beside the content, closed until asked for, and that choice is kept ──
{
  const p = await served(SAMPLE);
  const state = () => [p.$('#guide').hidden, p.$('.app').dataset.guide, p.$('[data-guide-toggle]').getAttribute('aria-expanded')].join();
  ok(state() === 'true,off,false', 'the guide starts closed', state());
  await p.click('[data-guide-toggle]');
  ok(state() === 'false,on,true' && p.d.cookie.includes('strands-evals-guide=on'),
    'the Guide switch opens it beside the content and remembers', `${state()} ${p.d.cookie}`);
  ok(p.text('#guide-body h2') === 'Reports' && p.$$('#guide-body ol li').length === 3 && p.$$('#guide-body dt').length === 3
    && parse(p.$('#guide-body a')) === 'quickstart/|guide-reports|',
  'it explains the view that is open: what to do, the terms it uses, and where the docs say more', p.text('#guide-body').slice(0, 120));
  await p.openReport();
  ok(p.text('#guide-body h2') === 'Report'
    && parse(p.$('#guide-body a')) === 'how-evaluation-works/|guide-report|#the-levels-an-evaluator-scores-at',
  'and follows to the next view, its links naming the view they are for', p.text('#guide-body h2'));
  await p.click('[data-guide-close]');
  ok(state() === 'true,off,false' && p.d.cookie.includes('strands-evals-guide=off'), 'its own close button closes it and remembers that too',
    `${state()} ${p.d.cookie}`);

  const later = await served(SAMPLE, { cookie: 'strands-evals-guide=on' });
  ok(!later.$('#guide').hidden && later.text('#guide-body h2') === 'Reports', 'a page opened after it was left open starts with it open',
    later.$('#guide').hidden);
}

// ── a ? on a case field says what it is, what it takes, and where the docs cover it ──
{
  const p = await served({ 'run/experiment.json': EXPERIMENT });
  await p.openExperiment();
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

// ── evaluators: described from the library, and linked to their own page ──
{
  const catalog = { Contains: { summary: 'Checks if actual_output contains a substring.', settings: [
    { name: 'value', required: true, default: null }, { name: 'case_sensitive', required: false, default: 'true' },
  ] } };
  const listed = { path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 1, evaluators: EXPERIMENT.evaluators, reports: [] };
  const p = await page({ fetchImpl: fakeServer({
    config: { root: '/served', depth: 3, read_only: false, evaluators: catalog },
    scan: { experiments: [listed] }, file: { path: 'e.json', modified: 5, document: EXPERIMENT },
  }) });
  await p.openExperiment();
  await p.click(p.$$('#evs .evchip')[0]);
  ok(/Checks if actual_output contains a substring\./.test(p.text('#evform .evsum')), 'the form says what the evaluator checks',
    p.text('#evform .evsum'));
  ok(parse(p.$('#evform .evsum a')) === 'evaluators/deterministic_evaluators/|evaluator-Contains|', 'with its page',
    parse(p.$('#evform .evsum a')));
  await p.click('.help[data-help="setting:Contains.case_sensitive"]');
  ok(/letter case/.test(helpText(p)) && /Default: true\./.test(helpText(p)), 'a setting says what it does and its default', helpText(p));
  await p.click('.help[data-help="setting:Contains.value"]');
  ok(/Required\./.test(helpText(p)), 'or that it is required', helpText(p));
  ok(!p.$('.help[data-help="setting:Contains.value.unit"]'), 'a setting is not explained as another whose name it starts with');

  await p.click(p.$$('#evs .evchip')[1]);
  ok(/Not one of the built-in evaluators/.test(p.text('#evform .evsum')), 'an evaluator the library does not ship says so',
    p.text('#evform .evsum'));
  ok(parse(p.$('#evform .evsum a')) === 'evaluators/custom_evaluator/|evaluator-custom|',
    'and its name stays out of the link', parse(p.$('#evform .evsum a')));
}
{
  const p = await served({ 'run/experiment.json': { ...EXPERIMENT, evaluators: [{ evaluator_type: 'ToolSelectionAccuracyEvaluator' }, { evaluator_type: 'HelpfulnessEvaluator' }] } });
  await p.openExperiment();
  const pagesFor = [];
  for (const chip of [0, 1]) {
    await p.click(p.$$('#evs .evchip')[chip]);
    pagesFor.push(parse(p.$('#evform .evsum a')));
  }
  ok(JSON.stringify(pagesFor) === JSON.stringify([
    'evaluators/tool_selection_evaluator/|evaluator-ToolSelectionAccuracyEvaluator|',
    'evaluators/helpfulness_evaluator/|evaluator-HelpfulnessEvaluator|',
  ]), 'a built-in links to its own page', pagesFor.join(' , '));

  ok(parse(p.$('#facts a')) === 'cli/run/|unrun-experiment|#experiment-file-run', 'an experiment with no report links to how to run it',
    parse(p.$('#facts a')));
}

// ── tooltips: a label explains itself on hover or focus, from the glossary or its own text ──
{
  const p = await served({ 'run/experiment.json': EXPERIMENT });
  const tip = p.$('.tip');
  const over = async (el, type = 'mouseover') => {
    el.dispatchEvent(new p.window.MouseEvent(type, { bubbles: true }));
    await tick();
  };
  /* The pointer must rest on an element for half a second before its tip shows. */
  const rested = () => new Promise(resolve => setTimeout(resolve, 600));
  ok(tip.hidden, 'nothing shows until something is pointed at');

  await over(p.$('#v-report th[data-term="score"]'));
  ok(tip.hidden, 'pointing at a column head shows nothing at once: crossing the page pops nothing');
  await rested();
  ok(!tip.hidden && tip.querySelector('b').textContent === 'Score' && /usually from 0 to 1/.test(tip.textContent),
    'after a rest it shows its glossary entry', tip.textContent);
  await over(p.$('#rep-find'));
  ok(tip.hidden, 'and it goes when the pointer moves on');
  await rested();
  ok(tip.hidden, 'a rest on something with nothing to say shows nothing');

  await p.openExperiment();
  const chip = p.$('#evs .evchip');
  const title = chip.title;
  await over(chip);
  await rested();
  ok(title && tip.textContent === title && !chip.hasAttribute('title'),
    'a plain title is shown the same way, and not a second time by the browser', `${title} / ${tip.textContent}`);

  const tab = p.$('.tab[data-view="experiments"]');
  tab.dispatchEvent(new p.window.FocusEvent('focusin', { bubbles: true }));
  await tick();
  ok(!tip.hidden && /Every experiment file found in this folder/.test(tip.textContent), 'focus shows it too', tip.textContent);
  await p.key('Escape');
  ok(tip.hidden, 'and Escape dismisses it');

  const explained = sel => p.$$(sel).every(el => el.dataset.term || el.dataset.tip || el.title);
  ok(explained('main th') && explained('.sections .tab') && explained('.ctl .seg button:not([data-f="all"])'),
    'every column head, view and filter says what it is');
}

done();
