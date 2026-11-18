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
   `reasoningContent`) and the `content_type` ones a recorded session's spans carry. */
function contentBlock(b) {
  if (b == null || typeof b !== 'object') return `<div class="say">${clip(b)}</div>`;
  if (typeof b.text === 'string') return `<div class="say">${clip(b.text)}</div>`;
  if (b.toolUse) return toolCall(b.toolUse.name, b.toolUse.input);
  if (b.content_type === 'tool_use') return toolCall(b.name, b.arguments);
  if (b.toolResult) {
    const parts = (b.toolResult.content || []).map(c => blockText(c && typeof c === 'object'
      ? ('json' in c ? c.json : typeof c.text === 'string' ? c.text : c) : c));
    const status = b.toolResult.status && b.toolResult.status !== 'success' ? b.toolResult.status : null;
    return toolResult(parts.join('\n'), status);
  }
  if (b.content_type === 'tool_result') {
    return toolResult(typeof b.error === 'string' ? b.error : blockText(b.content),
      b.error ? 'error' : null);
  }
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

function messageCard(m, nested) {
  const role = (m && m.role) || 'unknown';
  const known = ['user', 'assistant', 'tool', 'system'].includes(role) ? ` ${role}` : '';
  const content = Array.isArray(m.content) ? m.content
    : [{ text: m.content == null ? '' : String(m.content) }];
  const u = usageLine((m.metadata || {}).usage);
  return `<div class="msg${known}${nested ? ' nested' : ''}"><div class="who">`
    + `<b>${esc(role)}</b>${u ? `<span>${esc(u)}</span>` : ''}</div>`
    + content.map(b => contentBlock(b)).join('') + '</div>';
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
    const res = s.tool_result || {};
    return `<div class="msg">${head('tool')}${toolCall(call.name, call.arguments)}`
      + toolResult(typeof res.error === 'string' ? res.error : blockText(res.content),
        res.error ? 'error' : null) + '</div>';
  }
  if (kind === 'invoke_agent') {
    const say = (label, text) => (text
      ? `<div class="call"><span class="ckind">${label}</span><pre>${clip(text, 700)}</pre></div>` : '');
    return `<div class="msg">${head('invoke agent')}${say('system prompt', s.system_prompt)}`
      + say('prompt', s.user_prompt) + say('response', s.agent_response) + '</div>';
  }
  return `<div class="msg">${head(kind)}<pre>${clip(JSON.stringify(s), 700)}</pre></div>`;
}

/** The spans of a strands-evals `Session`, or null when `doc` is not one. */
const sessionSpans = doc => doc && typeof doc === 'object' && !Array.isArray(doc) && Array.isArray(doc.traces)
  ? doc.traces.flatMap(tr => (tr && Array.isArray(tr.spans) ? tr.spans : []))
  : null;

/** What a trace says it cost: `{ seconds, tokens, modelCalls, toolCalls }`, each null where it is not
    recorded, or null when `doc` is not a trace. A session times its spans; a message list counts tokens. */
function traceStats(doc) {
  const spans = sessionSpans(doc);
  if (spans) {
    const times = spans.flatMap(s => {
      const info = (s && s.span_info) || {};
      return [Date.parse(info.start_time), Date.parse(info.end_time)];
    }).filter(Number.isFinite);
    return {
      seconds: times.length > 1 ? (Math.max(...times) - Math.min(...times)) / 1000 : null,
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
    modelCalls: doc.filter(m => m && m.role === 'assistant').length,
    toolCalls: doc.reduce((n, m) => n + (Array.isArray(m && m.content) ? m.content.filter(b => b && b.toolUse).length : 0), 0),
  };
}

/** `2 model calls · 1 tool call · 4.2s · 3.1k tokens`, leaving out what was not recorded. */
const costText = stats => !stats ? '' : [
  stats.modelCalls ? plural(stats.modelCalls, 'model call') : '',
  stats.toolCalls ? plural(stats.toolCalls, 'tool call') : '',
  stats.seconds != null ? `${nice(stats.seconds)}s` : '',
  stats.tokens != null ? `${compact(stats.tokens)} tokens` : '',
].filter(Boolean).join(' · ');

/* Two shapes: a strands-evals `Session` (traces of spans) or a list of Strands messages. Read
   off the document, not the pointer's `format`, because the two can disagree. */
function traceView(doc, rawLabel = 'the file as written') {
  const raw = `<details class="raw"><summary>${esc(rawLabel)}</summary>`
    + `<pre class="out">${esc(JSON.stringify(doc, null, 1).slice(0, 200000))}</pre></details>`;
  const cost = costText(traceStats(doc));
  const sum = cost ? `<div class="tr-sum">${esc(cost)}</div>` : '';
  const spans = sessionSpans(doc);
  if (spans) {
    if (!spans.length) {
      return '<div class="empty">This session was recorded with no spans in it.</div>' + raw;
    }
    return `<div class="tr-count">${plural(spans.length, 'span')} of a recorded session</div>${sum}`
      + spans.map((s, i) => spanCard(s, i)).join('') + raw;
  }
  if (Array.isArray(doc)) {
    if (!doc.length) return '<div class="empty">This transcript is an empty list.</div>' + raw;
    return `<div class="tr-count">${plural(doc.length, 'message')}</div>${sum}`
      + doc.map(m => messageCard(m)).join('') + raw;
  }
  return '<div class="empty">This file is neither a recorded session nor a list of messages, '
    + 'so it is shown as it was written rather than as a conversation.</div>' + raw;
}
