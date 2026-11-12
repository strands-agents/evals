/* Standalone mode: the file opened with no server behind it. */
import { page, runner } from './harness.mjs';

const { ok, done } = runner('standalone');
const p = await page();                       // no fetch -> /api/config fails -> standalone
const { $, $$, text, click, type, openFolder, window, errs } = p;

// ── boot ──
ok(text('.tab.on') === 'Runs', 'lands on Runs', text('.tab.on'));
ok($$('.view.on').length === 1, 'exactly one view visible', $$('.view.on').length);
ok(/standalone file/.test(text('#mode-tag')), 'mode tag names standalone', text('#mode-tag'));
ok(/saves download/.test(text('#mode-tag')), 'mode tag says saves download', text('#mode-tag'));
ok($('[data-act="rescan"]').hidden === true, 'Rescan hidden without a server');
ok($('[data-act="pick"]').hidden === false, 'Open folder offered');
ok(/showing the sample it carries/.test(text('#runs-sub')), 'says the data is the sample', text('#runs-sub'));
ok(/Open a folder/.test(text('#runs-sub')), 'and names the way out of it', text('#runs-sub'));

// ── the bundled sample is one run, and its pairing is checked not assumed ──
const runs = $$('#runs-rows tr.run');
ok(runs.length === 1, 'one bundled run', runs.length);
const cells = [...runs[0].children].map(c => c.textContent.replace(/\s+/g, ' ').trim());
ok(/13 cases/.test(cells[1]) && /3 evaluators/.test(cells[1]), 'counts cases + evaluators', cells[1]);
ok(/0\.92/.test(cells[2]) && /5 failing/.test(cells[2]), 'summarises the report', cells[2]);
ok(/pairing verified/.test(cells[2]), 'pairing is verified, not assumed', cells[2]);

// ── the sample is already loaded, so the page is useful on open ──
ok($$('#case-list li[data-i]').length === 13, '13 cases loaded at boot', $$('#case-list li[data-i]').length);
ok(text('#r-score') === '0.92', 'hero score', text('#r-score'));
const stats = $$('#r-stats .stat').map(s => s.querySelector('.v').textContent + '=' + s.querySelector('.k').textContent);
ok(stats[0] === '31 / 36=Rows passing', 'rows passing counted with is_applicable', stats[0]);
ok(stats[3] === '3=Not applicable', 'not-applicable count', stats[3]);

// ── evaluators, including the editable config ──
const chips = $$('#evs .evchip').map(c => c.textContent.replace(/\s+/g, ' ').trim());
ok(chips.length === 3, 'three evaluator chips', chips.length);
ok(chips[0] === 'OutputEvaluator', 'unnamed evaluator shows its class', chips[0]);
ok(/threshold=0.9/.test(chips[1]), 'non-default config shown', chips[1]);
ok(chips[2].startsWith('support_rubric') && /RubricEvaluator/.test(chips[2]), 'name AND class', chips[2]);
/* Only the server can check an evaluator — it has the registry and the constructors. With no
   server the page says nothing rather than guessing from the class name. */
ok(text('#evnotes') === '', 'no evaluator verdicts without a server', text('#evnotes'));

await click('#evs .evchip');
ok($('#evform').hidden === false, 'clicking a chip opens its config');
await click('[data-evclose]');
ok($('#evform').hidden === true, 'and closes again');

await click($$('#evs .evchip')[1]);
ok($('#evform [data-evk="threshold"]'), 'threshold is editable', $('#evform').textContent.slice(0, 80));
await type('#evform [data-evk="threshold"]', '0.75');
ok(/1 unsaved change/.test(text('#dirty')), 'editing config marks dirty', text('#dirty'));
ok($('#save').disabled === false, 'Save enabled by a config edit');
await click('#revert');
ok(text('#dirty') === 'No unsaved changes', 'revert restores evaluators', text('#dirty'));
ok(/threshold=0.9/.test($$('#evs .evchip')[1].textContent), 'chip shows the original value');

// ── add / delete a case ──
const before = $$('#case-list li[data-i]').length;
await click('#add-case');
ok($$('#case-list li[data-i]').length === before + 1, 'add case appends', $$('#case-list li[data-i]').length);
ok($('#save').disabled === false, 'a new case is dirty immediately');
await click('#save');
ok(/has no input/.test(text('#saved')), 'refuses to write a case with no input', text('#saved'));
await type('#form [data-k="input"]', 'a brand new question');
await click('#save');
ok(/downloaded/.test(text('#saved')), 'standalone save downloads', text('#saved'));
ok(text('#dirty') === 'No unsaved changes', 'clean after save', text('#dirty'));
await click('#del-case');
ok($$('#case-list li[data-i]').length === before, 'delete removes it', $$('#case-list li[data-i]').length);

// ── duplicate names are refused: the report could not tell them apart ──
await click($$('#case-list li[data-i]')[0]);
const second = $$('#case-list li[data-i]')[1].querySelector('.nm').textContent;
await type('#form [data-k="name"]', second);
await click('#save');
ok(/two cases are named/.test(text('#saved')), 'duplicate names refused', text('#saved'));
await click('#revert');

// ── trajectories that are not lists of strings ──
await openFolder({
  'tree/experiment.json': {
    cases: [
      { name: 'dicts', session_id: 'sid-1', input: 'hi',
        expected_trajectory: [{ name: 'search', arguments: { q: 'rome' } }, { tool: 'reply' }] },
      { name: 'session', session_id: 'sid-2', input: 'hi',
        expected_trajectory: { session_id: 's1', spans: [{ name: 'agent' }] } },
      { name: 'strings', session_id: 'sid-3', input: 'hi', expected_trajectory: ['search', 'reply'] },
    ],
    evaluators: [{ evaluator_type: 'ToolTrajectoryEvaluator' }],
  },
});
ok($$('#runs-rows tr.run').length === 1, 'opened folder lists its run', $$('#runs-rows tr.run').length);
/* Having opened a folder, the page is reading real files. Still calling that the bundled
   sample is the only signal a reader has that the click did anything, so it has to flip. */
ok(!/showing the sample it carries/.test(text('#runs-sub')), 'stops claiming the sample', text('#runs-sub'));
ok(/Reading the folder you opened/.test(text('#runs-sub')), 'confirms the folder', text('#runs-sub'));
ok(/save as a download/.test(text('#runs-sub')), 'says where edits go', text('#runs-sub'));
await click('#runs-rows tr.run');
ok($$('#case-list li[data-i]').length === 3, 'three cases from the folder', $$('#case-list li[data-i]').length);

const trajectoryText = () => {
  const field = $$('#form .field').find(f => /Expected trajectory/.test(f.querySelector('label').textContent));
  return field.textContent.replace(/\s+/g, ' ').trim();
};
ok(/search\(q="rome"\)/.test(trajectoryText()), 'a recorded call is named, not [object Object]', trajectoryText());
ok(!/object Object/.test(trajectoryText()), 'no [object Object] anywhere', trajectoryText());
ok(/reply\(\)/.test(trajectoryText()), 'an entry with only a tool name still reads', trajectoryText());

await click($$('#case-list li[data-i]')[1]);
ok(!/^Expected trajectory none/.test(trajectoryText()), 'a Session trajectory is not reported as absent', trajectoryText());
ok(/session_id/.test(trajectoryText()), 'a Session trajectory is shown as JSON', trajectoryText());

await click($$('#case-list li[data-i]')[2]);
ok($$('#form .chips[data-chips] .chip').length === 2, 'string trajectory still renders chips',
  $$('#form .chips .chip').length);
ok($('#form .chips .addchip'), 'and can be added to');

done(errs);
