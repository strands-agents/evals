// Transcripts as the record stores them: a strands-evals Session (format "session": traces of spans), or the
// rollout's own list of Strands messages (format "list").
export type ContentBlock = {
  text?: string;
  toolUse?: { toolUseId: string; name: string; input: unknown };
  toolResult?: { toolUseId: string; status?: string; content: { text?: string; json?: unknown }[] };
  reasoningContent?: { reasoningText?: { text: string } };
  [k: string]: unknown;
};
export type StrandsMessage = { role: string; content: ContentBlock[]; metadata?: { usage?: Record<string, number> } };

export type Span = {
  span_type?: string;
  messages?: StrandsMessage[];
  tool_call?: { name?: string; arguments?: unknown; tool_call_id?: string };
  tool_result?: { content?: unknown; error?: string | null; tool_call_id?: string };
  system_prompt?: string;
  user_prompt?: string;
  agent_response?: string;
  [k: string]: unknown;
};
export type Session = { session_id?: string; traces: { spans: Span[] }[] };
