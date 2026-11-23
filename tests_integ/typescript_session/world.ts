import { Model } from '@strands-agents/sdk'

export const EXACT = 'Give the exact answer.'

const text = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value ?? '')

/**
 * Answers "question N" right only from the second user turn of one conversation, and then as the plain
 * scripted model does: every question when the system prompt asks for the exact answer (except every 10th),
 * else every 3rd. A fresh agent per call never reaches a second turn, so a right answer proves the
 * conversation was held open. "slow N" takes N milliseconds to answer, whatever the cancel signal says;
 * "abortable N" takes as long unless the cancel signal stops it first.
 */
export class TwoTurnDeterministic extends Model {
  config: Record<string, unknown> = { modelId: 'scripted-two-turn' }
  updateConfig(config: Record<string, unknown>): void { Object.assign(this.config, config) }
  getConfig(): any { return this.config }
  async *stream(messages: any[], options?: any): AsyncIterable<any> {
    const userTurns = messages.filter((message: any) => message?.role === 'user').length
    const last = messages[messages.length - 1]
    const question = (last?.content ?? []).map((block: any) => block.text ?? '').join('')
    const slow = question.match(/^slow (\d+)$/)
    if (slow) await new Promise((done) => setTimeout(done, Number(slow[1]))) // a model call that takes a while
    const abortable = question.match(/^abortable (\d+)$/)
    if (abortable) {
      await new Promise<void>((done, fail) => {
        const timer = setTimeout(done, Number(abortable[1]))
        options?.cancelSignal?.addEventListener('abort', () => {
          clearTimeout(timer)
          fail(new Error('cancelled'))
        })
      })
    }
    const n = Number(question.match(/\d+/)?.[0] ?? 0)
    const right = text(options?.systemPrompt).includes(EXACT) ? n % 10 !== 0 : n % 3 === 0
    const reply = userTurns >= 2 && right ? question.replace('question', 'answer') : "I'm not sure."
    yield { type: 'modelMessageStartEvent', role: 'assistant' }
    yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: reply } }
    yield { type: 'modelContentBlockStopEvent' }
    yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
    yield { type: 'modelMetadataEvent', usage: { inputTokens: 680, outputTokens: 20, totalTokens: 700 } }
  }
}
