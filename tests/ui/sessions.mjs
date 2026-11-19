/* Sessions: the recorded traces of a conversation as a third kind of thing in the Evals section, the
   diagnoses made of them, and the task result stores the scan leaves out. */
import { fakeServer, page, runner, served, tick } from './harness.mjs';

const { ok, same: eq, done } = runner('sessions');

/* Timestamps as the library writes them: UTC, six fractional digits. */
const at = seconds => new Date(Date.UTC(2026, 0, 5, 9) + seconds * 1000).toISOString().replace('Z', '000Z');
const info = (id, start, end, parent = null) =>
  ({ span_id: id, session_id: 's', parent_span_id: parent, start_time: at(start), end_time: at(end) });
const agent = (id, start, end, prompt, parent) => ({
  span_info: info(id, start, end, parent), span_type: 'invoke_agent',
  user_prompt: prompt, agent_response: `reply to ${prompt}`, available_tools: [],
});
const model = (id, start, end, parent) => ({
  span_info: info(id, start, end, parent), span_type: 'inference',
  messages: [{ role: 'assistant', content: [{ content_type: 'text', text: `thinking in ${id}` }] }],
});
const tool = (id, start, end, links) => ({
  span_info: info(id, start, end, links.parent), span_type: 'execute_tool', agent_span_id: links.agent || null,
  tool_call: { name: 'lookup_order', arguments: { order: id } }, tool_result: { content: 'in transit' },
});

/* Three traces ten minutes apart, as an agent invoked once per message is recorded. */
const APART = {
  session_id: 's-apart',
  traces: [0, 1, 2].map(k => ({
    trace_id: `t-${k}`, session_id: 's-apart',
    spans: [agent(`a${k}`, k * 600, k * 600 + 4, `question ${k}`), model(`m${k}`, k * 600 + 0.5, k * 600 + 1.5, `a${k}`),
      tool(`x${k}`, k * 600 + 1.5, k * 600 + 3, { parent: `a${k}` })],
  })),
};
/* One trace holding three turns, each agent span written after the spans under it, as spans are exported. */
const TOGETHER = {
  session_id: 's-together',
  traces: [{
    trace_id: 't-all', session_id: 's-together',
    spans: [
      model('M0', 0.5, 1.5, 'A0'), tool('T0', 1.5, 3, { agent: 'A0' }), agent('A0', 0, 4, 'first'),
      model('M1', 10.5, 11.5, 'A1'), tool('T1', 11.5, 13, { agent: 'A1' }), model('unlinked', 12, 12.5), agent('A1', 10, 14, 'second'),
      model('M2', 20.5, 21.5, 'A2'), agent('helper', 21, 23, 'delegated', 'A2'), tool('T2', 21.5, 22.5, { parent: 'helper' }),
      agent('A2', 20, 24, 'third'), model('late', 100, 101), model('later', 200, 202),
    ],
  }],
};
const PLAIN = { session_id: 's-plain', traces: [{ trace_id: 't', session_id: 's-plain', spans: APART.traces[0].spans }] };
const diagnosis = session_id => ({
  session_id,
  failures: [{ span_id: 'T1', category: ['wrong_tool'], confidence: [0.9], evidence: ['It looked the order up twice.'] }],
  root_causes: [{
    failure_span_id: 'T1', location: 'M1', causality: 'PRIMARY_FAILURE', propagation_impact: [],
    failure_detection_timing: 'ONLY_AT_TASK_END', completion_status: 'PARTIAL_SUCCESS',
    root_cause_explanation: 'No refund tool is described.', fix_type: 'TOOL_DESCRIPTION_FIX', fix_recommendation: 'Describe a refund tool.',
  }],
});
const result = name => ({ input: `ask ${name}`, actual_output: 'answer', name, expected_output: null, expected_trajectory: null, actual_trajectory: null });

const FOLDER = {
  'traces/a.json': APART,                                       // no name says what it is: the shape does
  'chat/turns.session.json': TOGETHER,
  'chat/turns.diagnosis.json': diagnosis('s-together'),
  'plain/one.json': PLAIN,
  'elsewhere/lost.diagnosis.json': diagnosis('s-gone'),
  'elsewhere/causes.json': { root_causes: diagnosis('x').root_causes },   // `diagnose --rca-only`
  'lookalike.session.json': { session_id: 's-x', traces: [{ name: 'a trace with no spans list' }] },
  'store/a.json': result('a'), 'store/b.json': result('b'), 'store/c.json': result('c'),
  'store/notes.json': { just: 'notes' },
  'inputs/row.json': { input: 'q', output: 'a' },               // opens like a task result, and is not one
};

const cells = (p, match) => p.$$('#sessions-rows tr.row').find(r => r.textContent.includes(match)).querySelectorAll('td');
const rowOf = (p, match) => [...cells(p, match)].slice(1, 3).map(td => p.text(td));
const tab = p => p.$('.tab[data-view="sessions"]');
const texts = (p, selector) => p.$$(selector).map(el => p.text(el));
/* A diagnosis on a session's page: how much it found, then each part under its heading. */
const found = p => {
  const diag = p.$('#ses-found .diag');
  return [p.text(diag.querySelector('h4 + p')), Object.fromEntries([...diag.querySelectorAll('h5')].map(h => [h.textContent, p.text(h.nextElementSibling)]))];
};

// ── the list: sessions told by their shape, with what each recorded ──
{
  const bare = await served({ 'e.json': { cases: [{ name: 'a', input: 'q' }], evaluators: [] } });
  ok(tab(bare).hidden && bare.view() === 'v-experiments' && !bare.text('#stores'),
    'a folder with no session in it has no Sessions item in the sidebar', `${tab(bare).hidden} ${bare.view()}`);
  const p = await served(FOLDER);
  ok(!tab(p).hidden && p.text(tab(p)) === 'Sessions', 'one appears once a session is found');
  await p.click(tab(p));
  ok(p.view() === 'v-sessions' && tab(p).getAttribute('aria-current') === 'page' && p.window.location.pathname === '/sessions',
    'and it opens the list, at an address of its own', `${p.view()} ${p.window.location.pathname}`);
  eq(p.text('#sessions-count'), '3 sessions · 2 diagnoses without their session', 'the list counts what it holds');
  eq(rowOf(p, 'a.json'), ['3 traces · 9 spans', '12s'],
    'a session is recognised whatever its file is called, and its duration leaves out the wait between traces');
  eq(rowOf(p, 'turns.session.json'), ['1 trace · 13 spans · 1 failure found', '202s'], 'a diagnosis of a session is counted on its row');
  ok(p.text(cells(p, 'turns.session.json')[0]).startsWith('chat / turns.session.json'), 'a row names the file and its folder',
    p.text(cells(p, 'turns.session.json')[0]));
  ok(!p.$$('#sessions-rows tr.row').some(r => r.textContent.includes('turns.diagnosis.json')),
    'that diagnosis has no row of its own');
  eq(rowOf(p, 'lost.diagnosis.json'), ['diagnosis 1 failure · 1 root cause', '—'],
    'a diagnosis whose session is not in the folder is listed on its own');
  eq(rowOf(p, 'causes.json'), ['diagnosis 0 failures · 1 root cause', '—'], 'so is one that holds root causes alone');
  const skips = texts(p, '#skips li');
  ok(skips.some(s => s.startsWith('lookalike.session.json')) && !p.$$('#sessions-rows tr.row').some(r => r.textContent.includes('lookalike')),
    'a file named like a session whose traces hold no spans is not one', skips.join(' | '));

  // ── a task result store: said to be there, and its files neither listed nor skipped ──
  eq(p.text('#stores'), 'A task result store was found: store holds 3 saved results. A store keeps what the task returned for each case, so its files are not listed here.',
    'the list says a task result store was found, and how much it holds');
  ok(!skips.some(s => /^store\/[abc]\.json/.test(s)), 'its results are not reported as skipped', skips.join(' | '));
  ok(skips.some(s => s.startsWith('store/notes.json')) && skips.some(s => s.startsWith('inputs/row.json')),
    'another file in the store is, and so is a file that only opens like a result', skips.join(' | '));

  // ── a session of several traces: one after another, each under its own heading with what it cost ──
  await p.openSession('a.json');
  ok(p.view() === 'v-session' && p.window.location.pathname === '/sessions/traces/a.json' && tab(p).classList.contains('on'),
    'a row opens the session on its own page, named in the address', `${p.view()} ${p.window.location.pathname}`);
  eq([p.text('#crumbs'), p.text('#facts'), p.text('#path')],
    ['Sessions/traces/a.json', '3 traces · 9 spans · 12s · session s-apart', 'traces/a.json'], 'the page header names it and says what it holds');
  eq(p.text('#ses-trace .tr-count'), '3 traces · 9 spans of a recorded session', 'the session is counted as a whole');
  eq(texts(p, '#ses-trace .tr-head'), ['Trace 1 of 3t-0', 'Trace 2 of 3t-1', 'Trace 3 of 3t-2'], 'each trace has its own heading');
  eq(texts(p, '#ses-trace .tr-sum'), ['3 model calls · 3 tool calls · 12s', ...Array(3).fill('1 model call · 1 tool call · 4s')],
    'the session has a cost line and so does each trace');
  eq(p.$$('#ses-trace .tr-trace').map(tr => [tr.dataset.trace, [...tr.querySelectorAll(':scope > .msg > .who span')].map(el => p.text(el)).join(' ')]),
    [['t-0', '#0 #1 #2'], ['t-1', '#3 #4 #5'], ['t-2', '#6 #7 #8']], 'spans stay under their own trace, numbered through the session');
  ok(!p.$('#ses-trace .tr-turn') && !p.$('#ses-found').children.length, 'a trace of one agent invocation is not split into turns');
  eq(texts(p, '#actions [data-md], #v-session [data-md]'), ['Copy as Markdown', 'Copy path'],
    'the session is copied from its header, by the one button that says so');

  await p.click('#crumbs a');
  ok(p.view() === 'v-sessions' && p.$('#sessions-rows tr.row.on').dataset.session === 'traces/a.json',
    'the breadcrumb leads back to the list, with the open session marked');

  // ── one trace of one invocation: no headings, and its cost said once ──
  await p.openSession('one.json');
  eq([p.text('#ses-trace .tr-count'), texts(p, '#ses-trace .tr-sum'), p.$$('#ses-trace .tr-trace').length, p.$$('#ses-trace .tr-head, #ses-trace .tr-turn').length],
    ['3 spans of a recorded session', ['1 model call · 1 tool call · 4s'], 1, 0],
    'a session of one trace is counted and costed once, with no trace heading and no turns');

  // ── several agent invocations in one trace: turns ──
  await p.openSession('turns.session.json');
  ok(p.$$('#ses-trace .tr-trace').length === 1 && !p.$('#ses-trace .tr-head'), 'a single trace needs no trace heading');
  eq(p.$$('#ses-trace .tr-turnhead').map(h => `${p.text(h.querySelector('b'))}: ${p.text(h.querySelector('span'))}`),
    ['Turn 1 of 3: 1 model call · 1 tool call · 4s', 'Turn 2 of 3: 2 model calls · 1 tool call · 4s',
      'Turn 3 of 3: 1 model call · 1 tool call · 4s', 'Outside any turn: 2 model calls · 3s'],
    'a trace with several agent invocations reads as turns, each with what it cost, the wait between spans outside any turn left out');
  const turns = p.$$('#ses-trace .tr-turn').map(t => [...t.querySelectorAll(':scope > .msg')].map(m => p.text(m.querySelector('.who span'))).join(' '));
  eq(turns, ['#2 #0 #1', '#6 #3 #4 #5', '#10 #7 #8 #9', '#11 #12'],
    'each turn opens on its agent invocation, then the spans it led to: by link, through a nested agent, or by when they ran');
  ok(p.text(p.$('#ses-trace .tr-turn .msg')).includes('first') && p.text(p.$$('#ses-trace .tr-turn')[2]).includes('delegated'),
    'the prompt leads its turn, and a delegated invocation stays inside the turn that made it');

  // ── the diagnosis of a session shows on its page ──
  eq([p.text('#ses-found h3'), found(p)[0]], ['chat/turns.diagnosis.json', '1 failure · 1 root cause · session s-together'],
    'a diagnosis made of the session is shown above its traces, with the file it came from');
  eq(found(p)[1], {
    'What failed': 'at #4 tool call lookup_orderwrong tool90% sureIt looked the order up twice.',
    Why: 'primary failure began at #3 model call, seen at #4 tool call lookup_orderNo refund tool is described.'
      + 'Noticed: only at task end · Outcome: partial success',
    'What to do': 'tool description fix Describe a refund tool.',
  }, 'it reads as what failed, why and what to do, each span named by its number in the trace below');
  await p.click('#ses-found .spanref');
  eq(texts(p, '#ses-trace .msg.lit .who'), ['#4tool'], 'a span the diagnosis names marks its card in the trace');
  ok(p.text('#facts').endsWith('session s-together · 1 failure found'), 'and the header says a diagnosis found something', p.text('#facts'));

  // ── a diagnosis with no session to show ──
  await p.click('#crumbs a');
  await p.openSession('lost.diagnosis.json');
  eq([p.view(), p.window.location.pathname, p.text('#facts')],
    ['v-session', '/sessions/elsewhere/lost.diagnosis.json', 'A diagnosis of session s-gone, which is not in this folder · 1 failure · 1 root cause'],
    'a diagnosis on its own opens as a page that says which session it is missing');
  ok(found(p)[1]['What failed'].startsWith('at T1') && !p.$('#ses-found .spanref') && p.$('#ses-trace details.raw') && !p.$('#ses-trace .tr-trace'),
    'it shows what was found with its spans named, the file as written, and no trace', JSON.stringify(found(p)));
  await p.click('#crumbs a');
  await p.openSession('causes.json');
  ok(p.text('#facts').startsWith('A diagnosis that does not name its session'), 'root causes alone open too', p.text('#facts'));
  eq([found(p)[0], Object.keys(found(p)[1])], ['0 failures · 1 root cause', ['Why', 'What to do']], 'as why and what to do, with nothing under what failed');
}

// ── a message that is not one ──
{
  const p = await served({ 'odd.json': { session_id: 's-odd', traces: [{ spans: [
    { span_type: 'inference', messages: [null, { role: 'assistant', content: 'still here' }] }] }] } });
  await p.openSession();
  eq([p.view(), texts(p, '#ses-trace .msg.nested')], ['v-session', ['unknown', 'assistantstill here']],
    'a message that is not one is shown as unknown, and the rest of its span with it');
}

/* ── addresses, and keeping up with the folder ── */
const entry = (path, modified, more) => ({ path, name: path.split('/').pop(), dir: path.split('/').slice(0, -1).join('/') || '.', size: 1, modified, ...more });
const sessionEntry = (path, modified, doc, diagnoses = []) => entry(path, modified, {
  session_id: doc.session_id, traces: doc.traces.length, spans: doc.traces.reduce((n, t) => n + t.spans.length, 0), seconds: 12, diagnoses,
});
function serve(tree) {
  const server = fakeServer({
    mark: () => tree.mark,
    scan: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark, experiments: [], reports: [], skipped: [],
      optimizer_runs: [], sessions: tree.sessions, diagnoses: tree.diagnoses || [], stores: [] }],
    file: path => ({ path, ...tree.files[path] }),
  });
  return Object.assign(server, { asked: () => server.sent.splice(0).map(r => r.url.split('?')[0]).join(' ') });
}

{
  const tree = {
    mark: 'm1',
    sessions: [sessionEntry('traces/a.json', 5, APART, ['traces/a.diagnosis.json'])],
    diagnoses: [entry('traces/a.diagnosis.json', 5, { session_id: 's-apart', failures: 1, root_causes: 1, session: 'traces/a.json' })],
    files: { 'traces/a.json': { modified: 5, document: APART }, 'traces/a.diagnosis.json': { modified: 5, document: diagnosis('s-apart') } },
  };
  const openAt = async path => {
    const p = await page({ fetchImpl: serve(tree), url: `http://127.0.0.1:9999${path}` });
    return [p, `${p.view()} ${p.window.location.pathname}`];
  };
  const [linked, where] = await openAt('/sessions/traces/a.json');
  ok(where === 'v-session /sessions/traces/a.json' && linked.$$('#ses-trace .tr-trace').length === 3 && linked.$('#ses-found .diag'),
    'the address of a session reopens it, with its diagnosis', where);
  eq((await openAt('/sessions/traces/a.diagnosis.json'))[1], 'v-session /sessions/traces/a.json',
    'the address of a diagnosis opens the session it was made from');
  const [gone, lost] = await openAt('/sessions/gone.json');
  eq([lost, gone.text('#missing b'), tab(gone).classList.contains('on')], ['v-missing /sessions/gone.json', 'That session hopped away', true],
    'the address of a session that is gone says so and stays, with Sessions still marked');
  eq((await openAt('/'))[1], 'v-sessions /sessions', 'a folder holding only sessions opens on them');

  const [p] = await openAt('/sessions');
  await p.openSession();
  await p.travel(-1);
  eq(`${p.view()} ${p.window.location.pathname}`, 'v-sessions /sessions', 'Back from a session returns to the list');
  await p.travel(1);
  eq(`${p.view()} ${p.window.location.pathname}`, 'v-session /sessions/traces/a.json', 'and Forward opens it again');
}

{
  const tree = { mark: 'm1', sessions: [], files: {} };
  const server = serve(tree);
  const p = await page({ fetchImpl: server, url: 'http://127.0.0.1:9999/sessions', visible: true });
  const poll = async () => { await p.window.pollNow(); await tick(); };
  eq([tab(p).hidden, p.view(), p.window.location.pathname, p.text('#sessions-rows')],
    [true, 'v-sessions', '/sessions', 'No sessions in this pond yetNo recorded session or diagnosis under /served. '
      + 'Save one with strands-evals fetch ... -o FILE, then Rescan. CLI reference'],
    'the address of the list in a folder with no session shows it empty, and says how to fill it');

  tree.sessions.push(sessionEntry('new/a.json', 5, PLAIN));
  tree.files['new/a.json'] = { modified: 5, document: PLAIN };
  tree.mark += '+';
  await poll();
  ok(!tab(p).hidden && p.$$('#sessions-rows tr.row').length === 1, 'a session written while the page is open appears with nothing clicked');

  await p.click(tab(p));
  await p.openSession();
  server.asked();
  await poll();
  eq(server.asked(), '/api/changes', 'an open session is not fetched again while nothing changes');

  tree.sessions[0] = sessionEntry('new/a.json', 6, APART);
  tree.files['new/a.json'] = { modified: 6, document: APART };
  tree.mark += '+';
  await poll();
  eq([p.$$('#ses-trace .tr-trace').length, p.text('#facts')], [3, '3 traces · 9 spans · 12s · session s-apart'],
    'an open session that is written again is re-read, and its header follows');

  tree.diagnoses = [entry('new/a.diagnosis.json', 7, { session_id: 's-apart', failures: 1, root_causes: 0, session: 'new/a.json' })];
  tree.sessions[0].diagnoses = ['new/a.diagnosis.json'];
  const { root_causes: _, ...failuresOnly } = diagnosis('s-apart');   // `diagnose --detect-only`
  tree.files['new/a.diagnosis.json'] = { modified: 7, document: failuresOnly };
  tree.mark += '+';
  await poll();
  ok(p.text('#facts').endsWith('1 failure found'), 'a diagnosis made of the open session shows up on its page', p.text('#facts'));
  eq([found(p)[0], Object.keys(found(p)[1])], ['1 failure · 0 root causes · session s-apart', ['What failed']], 'failures alone read as what failed');
}

done();
