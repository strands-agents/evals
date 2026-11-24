import { trace } from '@opentelemetry/api'
import { Model } from '@strands-agents/sdk'

export const EXACT = 'Give the exact answer.'

const text = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value ?? '')

/** Answers "question N" right when the system prompt asks for the exact answer (except every 10th), else every 3rd. */
export class Deterministic extends Model {
  config: Record<string, unknown> = { modelId: 'scripted-deterministic' }
  updateConfig(config: Record<string, unknown>): void { Object.assign(this.config, config) }
  getConfig(): any { return this.config }
  async *stream(messages: any[], options?: any): AsyncIterable<any> {
    const last = messages[messages.length - 1]
    const question = (last?.content ?? []).map((block: any) => block.text ?? '').join('')
    const n = Number(question.match(/\d+/)?.[0] ?? 0)
    const right = text(options?.systemPrompt).includes(EXACT) ? n % 10 !== 0 : n % 3 === 0
    const reply = right ? question.replace('question', 'answer') : "I'm not sure."
    // As a provider SDK does (the Anthropic SDK's `anthropic.messages.create`): its own `chat` span around this call,
    // which the runner must not count a second time.
    const provider = trace.getTracer('fake-provider-sdk').startSpan('fake.messages.create', {
      attributes: { 'gen_ai.operation.name': 'chat', 'gen_ai.usage.input_tokens': 680, 'gen_ai.usage.output_tokens': 20 },
    })
    yield { type: 'modelMessageStartEvent', role: 'assistant' }
    yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: reply } }
    yield { type: 'modelContentBlockStopEvent' }
    yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
    yield { type: 'modelMetadataEvent', usage: { inputTokens: 680, outputTokens: 20, totalTokens: 700 } }
    provider.end()
  }
}
