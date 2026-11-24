'use strict';

/* ── a recorded conversation ──
   Everything that draws a trace goes through `traceView`, so a new way of showing one lands here. */

/* No "show more" on purpose: the whole document is in the raw JSON at the foot of the transcript. */
function clip(text, limit) {
  const s = text == null ? '' : String(text);
  const n = limit || 1200;
  if (s.length <= n) return esc(s);
  return esc(s.slice(0, n))
    + `<span class="path"> … ${plural(s.length - n, 'character')} more, in the raw JSON below</span>`;
}

/* Tool output is usually a JSON string, and a lone `{"output": "..."}` is an envelope around text. */
function blockText(v) {
  if (typeof v !== 'string') return JSON.stringify(v === undefined ? null : v, null, 1);
  const s = v.trim();
  if (!/^[[{]/.test(s) || !/[\]}]$/.test(s)) return v;
  try {
    const o = JSON.parse(s);
    if (o && typeof o === 'object' && !Array.isArray(o)
      && Object.keys(o).length === 1 && typeof o.output === 'string') return o.output;
    return JSON.stringify(o, null, 1);
  } catch { return v; }
}

const toolCall = (name, input) =>
  `<div class="call"><span class="ckind">tool call <b>${esc(name || '?')}</b></span>`
  + `<pre>${clip(JSON.stringify(input === undefined ? null : input, null, 1), 700)}</pre></div>`;

const toolResult = (text, status) =>
  `<div class="call"><span class="ckind">tool result${status
    ? ` · <b class="warn">${esc(status)}</b>` : ''}</span><pre>${clip(text, 700)}</pre></div>`;

/* Two block vocabularies: the Strands blocks a rollout records (`toolUse`, `toolResult`,
   `reasoningContent`) and the `content_type` ones a recorded session's spans carry.
   A tool result in either is read as `[text, status]`, the status null when the call succeeded. */
const strandsResult = r => [
  (r.content || []).map(c => blockText(c && typeof c === 'object'
    ? ('json' in c ? c.json : typeof c.text === 'string' ? c.text : c) : c)).join('\n'),
  r.status && r.status !== 'success' ? r.status : null,
];
const recordedResult = r => [typeof r.error === 'string' ? r.error : blockText(r.content), r.error ? 'error' : null];

function contentBlock(b) {
  if (b == null || typeof b !== 'object') return `<div class="say">${clip(b)}</div>`;
  if (typeof b.text === 'string') return `<div class="say">${clip(b.text)}</div>`;
  if (b.toolUse) return toolCall(b.toolUse.name, b.toolUse.input);
  if (b.content_type === 'tool_use') return toolCall(b.name, b.arguments);
  if (b.toolResult) return toolResult(...strandsResult(b.toolResult));
  if (b.content_type === 'tool_result') return toolResult(...recordedResult(b));
  if (b.reasoningContent) {
    return `<div class="say think">${clip(((b.reasoningContent.reasoningText || {}).text) || '')}</div>`;
  }
  return `<pre>${clip(JSON.stringify(b), 700)}</pre>`;
}

const usageLine = u => {
  if (!u || typeof u !== 'object') return '';
  const parts = [];
  if (typeof u.inputTokens === 'number') parts.push(`${compact(u.inputTokens)} in`);
  if (typeof u.outputTokens === 'number') parts.push(`${compact(u.outputTokens)} out`);
  if (u.cacheReadInputTokens) parts.push(`${compact(u.cacheReadInputTokens)} cache read`);
  return parts.length ? parts.join(' · ') + ' tokens' : '';
};

/* A message's content as blocks: a list as it is, anything else as one block of text. */
const messageBlocks = m => (Array.isArray(m && m.content) ? m.content
  : [{ text: !m || m.content == null ? '' : String(m.content) }]);

function messageCard(m, nested) {
  const role = (m && m.role) || 'unknown';
  const known = ['user', 'assistant', 'tool', 'system'].includes(role) ? ` ${role}` : '';
  const u = usageLine(((m && m.metadata) || {}).usage);
  return `<div class="msg${known}${nested ? ' nested' : ''}"><div class="who">`
    + `<b>${esc(role)}</b>${u ? `<span>${esc(u)}</span>` : ''}</div>`
    + messageBlocks(m).map(contentBlock).join('') + '</div>';
}

function spanCard(s, i) {
  const kind = (s && s.span_type) || 'span';
  const head = n => `<div class="who"><span>#${i}</span><b>${esc(n)}</b></div>`;
  if (kind === 'inference' && Array.isArray(s.messages)) {
    return `<div class="msg">${head('inference')}`
      + s.messages.map(m => messageCard(m, true)).join('') + '</div>';
  }
  if (kind === 'execute_tool') {
    const call = s.tool_call || {};
    return `<div class="msg">${head('tool')}${toolCall(call.name, call.arguments)}`
      + toolResult(...recordedResult(s.tool_result || {})) + '</div>';
  }
  if (kind === 'invoke_agent') {
    const say = (label, text) => (text
      ? `<div class="call"><span class="ckind">${label}</span><pre>${clip(text, 700)}</pre></div>` : '');
    return `<div class="msg">${head('invoke agent')}${say('system prompt', s.system_prompt)}`
      + say('prompt', s.user_prompt) + say('response', s.agent_response) + '</div>';
  }
  return `<div class="msg">${head(kind)}<pre>${clip(JSON.stringify(s), 700)}</pre></div>`;
}

/** The traces of a strands-evals `Session` as `[{ id, spans }]`, or null when `doc` is not one. */
const sessionTraces = doc => doc && typeof doc === 'object' && !Array.isArray(doc) && Array.isArray(doc.traces)
  ? doc.traces.map(tr => ({ id: tr && tr.trace_id, spans: tr && Array.isArray(tr.spans) ? tr.spans : [] }))
  : null;

/** The spans of a strands-evals `Session`, or null when `doc` is not one. */
const sessionSpans = doc => {
  const traces = sessionTraces(doc);
  return traces && traces.flatMap(tr => tr.spans);
};

const spanStart = s => Date.parse(((s && s.span_info) || {}).start_time);
const spanEnd = s => Date.parse(((s && s.span_info) || {}).end_time);

/** What a trace says it cost: `{ seconds, tokens, modelCalls, toolCalls }`, each null where it is not
    recorded, or null when `doc` is not a trace. A session's seconds are what its traces took, added up:
    the wait between two traces is nobody's cost. A message list has `replies`, its assistant messages,
    in place of `modelCalls`, which it does not record. */
function traceStats(doc) {
  const traces = sessionTraces(doc);
  if (traces) {
    const spans = traces.flatMap(tr => tr.spans);
    const took = traces.map(tr => {
      const times = tr.spans.flatMap(s => [spanStart(s), spanEnd(s)]).filter(Number.isFinite);
      return times.length > 1 ? Math.max(...times) - Math.min(...times) : null;
    }).filter(ms => ms != null);
    return {
      seconds: took.length ? took.reduce((sum, ms) => sum + ms, 0) / 1000 : null,
      tokens: null,
      modelCalls: spans.filter(s => s && s.span_type === 'inference').length,
      toolCalls: spans.filter(s => s && s.span_type === 'execute_tool').length,
    };
  }
  if (!Array.isArray(doc)) return null;
  const used = doc.map(m => ((m && m.metadata) || {}).usage).filter(u => u && typeof u === 'object');
  return {
    seconds: null,
    tokens: used.length ? used.reduce((sum, u) => sum + (u.inputTokens || 0) + (u.outputTokens || 0), 0) : null,
    replies: doc.filter(m => m && m.role === 'assistant').length,
    toolCalls: doc.reduce((n, m) => n + (Array.isArray(m && m.content) ? m.content.filter(b => b && b.toolUse).length : 0), 0),
  };
}

const secondsText = seconds => `${nice(seconds)}s`;
/** `2 model calls · 1 tool call · 4.2s · 3.1k tokens`, leaving out what was not recorded. */
const costText = stats => !stats ? '' : [
  stats.modelCalls ? plural(stats.modelCalls, 'model call') : '',
  stats.replies ? plural(stats.replies, 'assistant message') : '',
  stats.toolCalls ? plural(stats.toolCalls, 'tool call') : '',
  stats.seconds != null ? secondsText(stats.seconds) : '',
  stats.tokens != null ? `${compact(stats.tokens)} tokens` : '',
].filter(Boolean).join(' · ');
/* Each of `groups` is timed on its own, as a trace is: the wait between two of them is not counted. */
const spansCost = groups => costText(traceStats({ traces: groups.map(spans => ({ spans })) }));

/** The turns of one trace: `{ turns: [{ agent, spans }], loose }`, or null when it holds fewer than two.
    A turn is an agent invocation that is not inside another one. A span belongs to the turn its
    `agent_span_id` or its parents lead to, and failing those to the one it started during. */
function traceTurns(spans) {
  const byId = new Map(spans.filter(s => s && s.span_info && s.span_info.span_id).map(s => [s.span_info.span_id, s]));
  const turnOf = s => {
    let turn = null, at = s;
    for (let hops = 0; at && hops < spans.length; hops++) {
      at = byId.get(at.agent_span_id) || byId.get((at.span_info || {}).parent_span_id);
      if (at && at.span_type === 'invoke_agent') turn = at;
    }
    return turn;
  };
  const agents = spans.filter(s => s && s.span_type === 'invoke_agent' && !turnOf(s));
  if (agents.length < 2) return null;
  const turns = new Map(agents.map(agent => [agent, { agent, spans: [] }]));
  const loose = [];
  spans.forEach(s => {
    if (turns.has(s)) return;
    const turn = (s && turnOf(s)) || agents.find(a => spanStart(s) >= spanStart(a) && spanStart(s) <= spanEnd(a));
    (turn ? turns.get(turn).spans : loose).push(s);
  });
  return { turns: [...turns.values()], loose };
}

/** One trace of a session. `first` is where its spans stand in the session's count, and `of` how many
    traces the session holds. */
function traceSection(trace, at, of, first) {
  const number = new Map(trace.spans.map((s, i) => [s, first + i]));
  const cards = spans => spans.map(s => spanCard(s, number.get(s))).join('');
  const turn = (label, spans, timed = [spans]) => `<div class="tr-turn"><div class="tr-turnhead"><b>${label}</b>`
    + `<span>${esc(spansCost(timed))}</span></div>${cards(spans)}</div>`;
  const turns = traceTurns(trace.spans);
  const cost = spansCost([trace.spans]);
  return `<section class="tr-trace" data-trace="${esc(trace.id == null ? '' : trace.id)}">`
    + (of > 1 ? `<h4 class="tr-head">Trace ${at + 1} of ${of}<span class="path">${esc(trace.id == null ? '' : trace.id)}</span></h4>`
      + (cost ? `<div class="tr-sum">${esc(cost)}</div>` : '') : '')
    + (!trace.spans.length ? '<div class="empty">This trace was recorded with no spans in it.</div>'
      : !turns ? trace.spans.map((s, i) => spanCard(s, first + i)).join('')
        : turns.turns.map((t, k) => turn(`Turn ${k + 1} of ${turns.turns.length}`, [t.agent, ...t.spans])).join('')
          + (turns.loose.length ? turn('Outside any turn', turns.loose, turns.loose.map(s => [s])) : ''))
    + '</section>';
}

/** A document as it was written, one disclosure away. */
const rawView = (doc, label) => `<details class="raw"><summary>${esc(label)}</summary>`
  + `<pre class="out">${esc(JSON.stringify(doc, null, 1).slice(0, 200000))}</pre></details>`;

/* Two shapes: a strands-evals `Session` (traces of spans) or a list of Strands messages. Read
   off the document, not the pointer's `format`, because the two can disagree. */
function traceView(doc, rawLabel = 'the file as written') {
  const raw = rawView(doc, rawLabel);
  const cost = costText(traceStats(doc));
  const sum = cost ? `<div class="tr-sum">${esc(cost)}</div>` : '';
  const traces = sessionTraces(doc);
  if (traces) {
    const spans = sessionSpans(doc).length;
    if (!spans) {
      return '<div class="empty">This session was recorded with no spans in it.</div>' + raw;
    }
    let first = 0;
    return `<div class="tr-count">${traces.length > 1 ? plural(traces.length, 'trace') + ' · ' : ''}`
      + `${plural(spans, 'span')} of a recorded session</div>${sum}${traceCopy(doc)}`
      + traces.map((tr, at) => {
        const section = traceSection(tr, at, traces.length, first);
        first += tr.spans.length;
        return section;
      }).join('') + raw;
  }
  if (Array.isArray(doc)) {
    if (!doc.length) return '<div class="empty">This transcript is an empty list.</div>' + raw;
    return `<div class="tr-count">${plural(doc.length, 'message')}</div>${sum}${traceCopy(doc)}`
      + doc.map(m => messageCard(m)).join('') + raw;
  }
  return '<div class="empty">This file is neither a recorded session nor a list of messages, '
    + 'so it is shown as it was written rather than as a conversation.</div>' + raw;
}
