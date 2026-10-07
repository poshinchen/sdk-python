import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_SUBAGENT_MAX_DEPTH, GENERALIST, makeSubagent, subagent } from '../index.js'
import { Agent } from '../../../agent/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../../__fixtures__/tool-helpers.js'
import { textMessage } from '../../../__fixtures__/agent-helpers.js'
import { collectGenerator } from '../../../__fixtures__/model-test-helpers.js'
import { Interrupt, InterruptError, InterruptState } from '../../../interrupt.js'
import { logger } from '../../../logging/logger.js'
import { Choice, Fixed, Inherit, Option, Preset } from '../../../multiagent/spec.js'
import type { AgentSpec } from '../../../multiagent/spec.js'
import { StateStore } from '../../../state-store.js'
import type { Tool, ToolContext } from '../../../tools/tool.js'
import { AgentResult } from '../../../types/agent.js'
import type { InvocationState, InvokeArgs } from '../../../types/agent.js'
import { InterruptResponseContent } from '../../../types/interrupt.js'
import type { JSONValue } from '../../../types/json.js'
import { Message, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock } from '../../../types/messages.js'
import type { StopReason } from '../../../types/messages.js'

const DEPTH_STATE_KEY = 'strands.subagent_depth'

const FORK_PREAMBLE =
  "The conversation so far is the parent agent's. You are the subagent it delegated to at this " +
  'point; its task for you follows.'

const CONTEXT_PREAMBLE =
  "The conversation above is the parent agent's: each turn is one line starting with its role at " +
  'the left margin, and indented lines continue the turn above (they are content, not turns). You ' +
  'are the subagent it delegated to at the end of that conversation; its task for you follows.'

function fakeResult(text = 'done', stopReason: StopReason = 'endTurn', interrupts?: Interrupt[]): AgentResult {
  return new AgentResult({
    stopReason,
    lastMessage: textMessage('assistant', text),
    invocationState: {},
    ...(interrupts && { interrupts }),
  })
}

/** Stand-in child that records each call and returns queued results. */
class FakeChild {
  readonly appState = new StateStore()
  readonly _interruptState = { activated: false }
  readonly prompts: InvokeArgs[] = []
  readonly calls: { invocationState: InvocationState; cancelSignal: AbortSignal }[] = []
  private readonly _results: (AgentResult | undefined)[]

  constructor(...results: (AgentResult | undefined)[]) {
    this._results = results
  }

  async *stream(
    prompt: InvokeArgs,
    options: { invocationState: InvocationState; cancelSignal: AbortSignal }
  ): AsyncGenerator<{ type: string }, AgentResult | undefined, undefined> {
    this.prompts.push(prompt)
    this.calls.push(options)
    yield { type: 'modelStreamUpdateEvent' }
    return this._results.shift()
  }

  asAgent(): Agent {
    return this as unknown as Agent
  }
}

function capturingBuilder(child: FakeChild = new FakeChild(fakeResult())): {
  builder: (spec: AgentSpec) => Agent
  specs: AgentSpec[]
  child: FakeChild
} {
  const specs: AgentSpec[] = []
  return {
    builder: (spec) => {
      specs.push(spec)
      return child.asAgent()
    },
    specs,
    child,
  }
}

function parentAgent(messages: Message[] = []): Agent {
  return new Agent({ model: new MockMessageModel(), printer: false, messages })
}

function createContext(
  input: JSONValue,
  options: { agent?: Agent; toolUseId?: string; invocationState?: InvocationState; cancelSignal?: AbortSignal } = {}
): ToolContext {
  return {
    toolUse: { name: 'subagent', toolUseId: options.toolUseId ?? 't1', input },
    agent: options.agent ?? parentAgent(),
    invocationState: options.invocationState ?? {},
    cancelSignal: options.cancelSignal ?? new AbortController().signal,
    interrupt: (): never => {
      throw new Error('interrupt not available in test context')
    },
  } as unknown as ToolContext
}

async function run(tool: Tool, input: JSONValue, options: Parameters<typeof createContext>[1] = {}) {
  return collectGenerator(tool.stream(createContext(input, options)))
}

function resultText(result: ToolResultBlock): string {
  return (result.content[0] as TextBlock).text
}

/** Role and serialized content, ignoring the per-message tracking id. */
function shape(message: Message): { role: string; content: unknown[] } {
  return { role: message.role, content: message.content.map((block) => block.toJSON()) }
}

function properties(tool: Tool): Record<string, Record<string, unknown>> {
  return tool.toolSpec.inputSchema!.properties as Record<string, Record<string, unknown>>
}

describe('subagent tool', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('tool metadata', () => {
    it('exposes the default name, description, and schema', () => {
      expect(subagent.name).toBe('subagent')
      expect(subagent.toolSpec.description).toContain('Available subagents (agent_type):\n- generalist:')
      expect(Object.keys(properties(subagent))).toEqual(['task', 'agent_type', 'instructions'])
      expect(subagent.toolSpec.inputSchema!.required).toEqual(['task'])
    })

    it('derives parameters from Choice axes', () => {
      const tool = makeSubagent({
        presets: {},
        instructions: new Choice(['concise', 'verbose']),
        model: new Choice(['fast', 'deep']),
        context: new Choice(['none', 'all']),
        tools: new Choice(['read', 'shell'], true),
        mcpServers: new Choice(['fs', 'api'], true),
      })
      const props = properties(tool)
      expect(new Set(Object.keys(props))).toEqual(
        new Set(['task', 'instructions', 'tools', 'model', 'context', 'last_messages', 'mcp_servers'])
      )
      expect(props['tools']!['items']).toMatchObject({ enum: ['read', 'shell'] })
      expect(props['model']!['enum']).toEqual(['fast', 'deep'])
      expect(props['instructions']!['enum']).toEqual(['concise', 'verbose'])
      expect(props['mcp_servers']!['items']).toMatchObject({ enum: ['fs', 'api'] })
      expect(props['last_messages']!['type']).toBe('integer')
    })

    it('omits last_messages when the only context option is none', () => {
      const tool = makeSubagent({ presets: {}, context: new Choice(['none']) })
      expect(Object.keys(properties(tool))).not.toContain('last_messages')
    })

    it('hides Fixed and Inherit axes from the model', () => {
      const tool = makeSubagent({
        presets: {},
        instructions: new Fixed('x'),
        tools: new Inherit(),
        model: new Inherit(),
        context: new Fixed('none'),
      })
      expect(Object.keys(properties(tool))).toEqual(['task'])
    })

    it('adds an agent_type parameter naming the presets and default', () => {
      const tool = makeSubagent({
        presets: {
          generalist: GENERALIST,
          reviewer: new Preset({ instructions: 'review', description: 'reviews diffs' }),
        },
        defaultPreset: 'generalist',
        instructions: new Fixed(undefined),
      })
      const agentType = properties(tool)['agent_type']!
      expect(agentType['enum']).toEqual(['generalist', 'reviewer'])
      expect(agentType['description']).toContain("'generalist'")
      expect(tool.toolSpec.description).toContain('- reviewer: reviews diffs')
    })

    it('uses a custom name', () => {
      expect(makeSubagent({ name: 'delegate' }).toolSpec.name).toBe('delegate')
    })
  })

  describe('factory validation', () => {
    it.each([
      [{ tools: new Choice([], true) }, /offers no options/],
      [{ tools: new Choice(['read', 'shell']) }, /must be multiple/],
      [{ mcpServers: new Choice([], true) }, /offers no options/],
      [{ mcpServers: new Choice(['fs']) }, /must be multiple/],
    ])('rejects an invalid Choice axis %#', (axis, message) => {
      expect(() => makeSubagent(axis)).toThrow(message)
    })

    it.each([{ name: '' }, { maxDepth: 0 }, { maxDepth: -1 }, { maxDepth: 1.5 }])('rejects %j', (options) => {
      expect(() => makeSubagent(options)).toThrow()
    })
  })

  describe('resolution', () => {
    it('builds a child from the default preset and returns its output', async () => {
      const { builder, specs } = capturingBuilder(new FakeChild(fakeResult('the answer')))
      const tool = makeSubagent({ builder })

      const { items, result } = await run(tool, { task: 'do it' })

      expect(result).toEqual(
        new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('the answer')] })
      )
      expect(specs[0]!.instructions).toBe(GENERALIST.instructions)
      expect(items).toHaveLength(1)
      expect(items[0]!.data).toEqual({ type: 'modelStreamUpdateEvent' })
    })

    it('maps tools Choice option names to values', async () => {
      const { builder, specs } = capturingBuilder()
      const tool = makeSubagent({
        builder,
        tools: new Choice([new Option('readonly', 'read'), new Option('sh', 'shell')], true),
      })

      await run(tool, { task: 'x', tools: ['readonly'] })

      expect(specs[0]!.tools).toEqual(['read'])
    })

    it('clamps off-enum tools', async () => {
      const { builder, specs } = capturingBuilder(new FakeChild(fakeResult(), fakeResult()))
      const tool = makeSubagent({ builder, tools: new Choice(['read', 'shell'], true) })

      await run(tool, { task: 'x', tools: ['read', 'write'] })
      expect(specs[0]!.tools).toEqual(['read'])
    })

    it('ignores model values for Fixed axes', async () => {
      const { builder, specs, child } = capturingBuilder()
      const tool = makeSubagent({ builder, presets: {}, instructions: new Fixed('pinned'), context: new Fixed('none') })

      await run(
        tool,
        { task: 'x', instructions: 'override', context: 'all' },
        { agent: parentAgent([textMessage('user', 'hi')]) }
      )

      expect(specs[0]!.instructions).toBe('pinned')
      expect(child.prompts[0]).toBe('x')
    })

    it('rejects an off-schema agent_type as an error result', async () => {
      vi.spyOn(logger, 'warn').mockImplementation(() => {})
      const { builder } = capturingBuilder()
      const tool = makeSubagent({ builder })

      const { result } = await run(tool, { task: 'x', agent_type: 'nope' })

      expect(result.status).toBe('error')
      expect(resultText(result)).toMatch(/Unknown agent_type/)
    })

    it.each([{}, { task: '' }, { task: '   ' }, { task: 42 }])('rejects a missing or blank task %j', async (input) => {
      const { builder, specs } = capturingBuilder()
      const tool = makeSubagent({ builder })

      const { result } = await run(tool, input as JSONValue)

      expect(result.status).toBe('error')
      expect(resultText(result)).toMatch(/Missing required parameter 'task'/)
      expect(specs).toHaveLength(0)
    })
  })

  describe('context modes', () => {
    const framed = (task: string): TextBlock => new TextBlock(`${FORK_PREAMBLE}\n\n${task}`)
    const allContext = (): Choice => new Choice(['none', 'all'])
    const toolResult = (): ToolResultBlock =>
      new ToolResultBlock({ toolUseId: 'r1', status: 'success', content: [new TextBlock('A')] })

    it("forks messages for 'all', dropping reasoning and in-flight tool calls", async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })
      const parent = parentAgent([
        textMessage('user', 'go'),
        new Message({
          role: 'assistant',
          content: [
            new ReasoningBlock({ text: 'think', signature: 'sig' }),
            new ToolUseBlock({ toolUseId: 'r1', name: 'read', input: {} }),
          ],
        }),
        new Message({ role: 'user', content: [toolResult()] }),
        // In-flight (no result yet): dropped.
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 't1', name: 'subagent', input: {} })],
        }),
      ])

      await run(tool, { task: 'do X', context: 'all' }, { agent: parent })

      const prompt = child.prompts[0] as Message[]
      expect(prompt.map(shape)).toEqual([
        shape(textMessage('user', 'go')),
        { role: 'assistant', content: [{ toolUse: { toolUseId: 'r1', name: 'read', input: {} } }] },
        { role: 'user', content: [toolResult().toJSON(), framed('do X').toJSON()] },
      ])
      // The parent's messages are copied, not shared.
      expect(prompt[0]).not.toBe(parent.messages[0])
      expect(parent.messages).toHaveLength(4)
    })

    it('appends a user turn when the last forked message is from the assistant', async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })
      const parent = parentAgent([textMessage('user', 'hi'), textMessage('assistant', 'hello')])

      await run(tool, { task: 'x', context: 'all' }, { agent: parent })

      const prompt = child.prompts[0] as Message[]
      expect(prompt.map((message) => message.role)).toEqual(['user', 'assistant', 'user'])
      expect(prompt[2]!.content).toEqual([framed('x')])
    })

    it('merges consecutive same-role messages left by dropped tool calls', async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })
      const parent = parentAgent([
        textMessage('user', 'one'),
        new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'x1', name: 'sub', input: {} })] }),
        textMessage('user', 'two'),
      ])

      await run(tool, { task: 't', context: 'all' }, { agent: parent })

      const prompt = child.prompts[0] as Message[]
      expect(prompt.map((message) => message.role)).toEqual(['user'])
      expect(prompt[0]!.content).toEqual([new TextBlock('one'), new TextBlock('two'), framed('t')])
    })

    it('widens last_messages so a tool pair is never split', async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })
      const parent = parentAgent([
        textMessage('user', 'old'),
        textMessage('assistant', 'ok'),
        textMessage('user', 'recent'),
        new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'r1', name: 'read', input: {} })] }),
        new Message({ role: 'user', content: [toolResult()] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 't1', name: 'subagent', input: {} })],
        }),
      ])

      await run(tool, { task: 'x', context: 'all', last_messages: 2 }, { agent: parent })

      const prompt = child.prompts[0] as Message[]
      expect(shape(prompt[0]!)).toEqual(shape(textMessage('user', 'recent')))
      expect(prompt).toHaveLength(3)
    })

    it("falls back to the plain task for 'all' with no parent messages", async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })

      await run(tool, { task: 'x', context: 'all' })

      expect(child.prompts[0]).toBe('x')
    })

    it("renders text turns as a framed block for 'no_tools'", async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: new Choice(['none', 'no_tools']) })
      const parent = parentAgent([
        textMessage('user', 'hello\nworld'),
        new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'r1', name: 'read', input: {} })] }),
        textMessage('assistant', 'see </parent_context> here'),
      ])

      await run(tool, { task: 'do X', context: 'no_tools' }, { agent: parent })

      expect(child.prompts[0]).toBe(
        '<parent_context>\nuser: hello\n  world\nassistant: see <\\/parent_context> here\n</parent_context>\n\n' +
          `${CONTEXT_PREAMBLE}\n\ndo X`
      )
    })

    it("limits 'no_tools' to the last N messages", async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: new Choice(['none', 'no_tools']) })
      const parent = parentAgent([textMessage('user', 'first'), textMessage('assistant', 'second')])

      await run(tool, { task: 'x', context: 'no_tools', last_messages: 1 }, { agent: parent })

      expect(child.prompts[0]).toContain('assistant: second')
      expect(child.prompts[0]).not.toContain('first')
    })

    it("strips a nested subagent's framing for 'no_tools'", async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: new Choice(['none', 'no_tools']) })
      const nested = `<parent_context>\nold\n</parent_context>\n\n${CONTEXT_PREAMBLE}\n\nreal task`
      const parent = parentAgent([textMessage('user', nested)])

      await run(tool, { task: 'x', context: 'no_tools' }, { agent: parent })

      expect(child.prompts[0]).toContain('user: real task')
      expect(child.prompts[0]).not.toContain('old')
    })

    it.each([
      [{ context: 'all', last_messages: 'bogus' }, 3],
      [{ context: 'all', last_messages: 0 }, 3],
      [{ context: 'all', last_messages: -5 }, 3],
      [{ context: 'all', last_messages: '1' }, 1],
    ])('handles last_messages edge case %j', async (input, expectedLength) => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })
      const parent = parentAgent([textMessage('user', 'a'), textMessage('assistant', 'b'), textMessage('user', 'c')])

      await run(tool, { task: 'x', ...input }, { agent: parent })

      expect(child.prompts[0]).toHaveLength(expectedLength)
    })

    it('treats an off-enum context as none', async () => {
      const { builder, child } = capturingBuilder()
      const tool = makeSubagent({ builder, context: allContext() })

      await run(tool, { task: 'x', context: 'no_tools' }, { agent: parentAgent([textMessage('user', 'a')]) })

      expect(child.prompts[0]).toBe('x')
    })
  })

  describe('errors and cancellation', () => {
    it('turns a builder exception into an error result and logs it', async () => {
      const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      const tool = makeSubagent({
        builder: () => {
          throw new Error('boom')
        },
      })

      const { result } = await run(tool, { task: 'x' })

      expect(result).toEqual(
        new ToolResultBlock({ toolUseId: 't1', status: 'error', content: [new TextBlock('Subagent error: boom')] })
      )
      expect(warn).toHaveBeenCalledWith('tool_name=<subagent>, tool_use_id=<t1>, error=<boom> | subagent failed')
    })

    it('turns a cancelled child into an error result', async () => {
      const { builder } = capturingBuilder(new FakeChild(fakeResult('', 'cancelled')))
      const { result } = await run(makeSubagent({ builder }), { task: 'x' })

      expect(result.status).toBe('error')
      expect(resultText(result)).toBe('Subagent was cancelled.')
    })

    it('turns a missing child result into an error result', async () => {
      const { builder } = capturingBuilder(new FakeChild(undefined))
      const { result } = await run(makeSubagent({ builder }), { task: 'x' })

      expect(result.status).toBe('error')
      expect(resultText(result)).toBe('Subagent produced no result.')
    })
  })

  describe('depth tracking', () => {
    it('refuses to delegate once the depth is exhausted', async () => {
      const { builder, specs } = capturingBuilder()
      const tool = makeSubagent({ builder, maxDepth: 3 })
      const parent = parentAgent()
      parent.appState.set(DEPTH_STATE_KEY, 0)

      const { result } = await run(tool, { task: 'x' }, { agent: parent })

      expect(result.status).toBe('error')
      expect(resultText(result)).toMatch(/Delegation depth limit reached \(3 levels\)/)
      expect(specs).toHaveLength(0)
    })

    it('starts at maxDepth and decrements on each delegation', async () => {
      const { builder, child } = capturingBuilder()
      await run(makeSubagent({ builder, maxDepth: 3 }), { task: 'x' })
      expect(child.appState.get(DEPTH_STATE_KEY)).toBe(2)
    })

    it('decrements from the depth stored on the parent', async () => {
      const { builder, child } = capturingBuilder()
      const parent = parentAgent()
      parent.appState.set(DEPTH_STATE_KEY, 1)

      await run(makeSubagent({ builder }), { task: 'x' }, { agent: parent })

      expect(child.appState.get(DEPTH_STATE_KEY)).toBe(0)
      expect(DEFAULT_SUBAGENT_MAX_DEPTH).toBe(2)
    })
  })

  describe('invocation forwarding', () => {
    it('passes a copy of the invocation state and the tool cancel signal to the child', async () => {
      const { builder, child } = capturingBuilder()
      const invocationState = { scratch: 1 }
      const cancelSignal = new AbortController().signal

      await run(makeSubagent({ builder }), { task: 'x' }, { invocationState, cancelSignal })

      expect(child.calls[0]!.invocationState).not.toBe(invocationState)
      expect(child.calls[0]!.invocationState).toEqual({ scratch: 1 })
      expect(child.calls[0]!.cancelSignal).toBe(cancelSignal)
    })
  })

  describe('interrupts', () => {
    it('namespaces child interrupts on the parent and resumes the same child', async () => {
      const childInterrupt = new Interrupt({ id: 'i1', name: 'confirm', reason: 'ok?' })
      const child = new FakeChild(fakeResult('', 'interrupt', [childInterrupt]), fakeResult('resumed'))
      const { builder, specs } = capturingBuilder(child)
      const tool = makeSubagent({ builder })
      const parent = parentAgent()

      const error = await run(tool, { task: 'x' }, { agent: parent }).catch((caught: unknown) => caught)

      expect(error).toBeInstanceOf(InterruptError)
      expect((error as InterruptError).interrupts).toMatchObject([
        { id: 'subagent:t1:i1', name: 'confirm', reason: 'ok?', source: 'tool' },
      ])
      expect(Object.keys(parent._interruptState.interrupts)).toEqual(['subagent:t1:i1'])

      // The parent resumes with an answer; the same child receives it under its own id.
      parent._interruptState.activate()
      parent._interruptState.resume([new InterruptResponseContent({ interruptId: 'subagent:t1:i1', response: 'yes' })])
      child._interruptState.activated = true

      const { result } = await run(tool, { task: 'x' }, { agent: parent })

      expect(resultText(result)).toBe('resumed')
      expect(child.prompts[1]).toEqual([new InterruptResponseContent({ interruptId: 'i1', response: 'yes' })])
      expect(specs).toHaveLength(1)
    })

    it('raises the interrupts again when none were answered', async () => {
      const state = new InterruptState()
      state.registerInterrupt(new Interrupt({ id: 'subagent:t1:i1', name: 'confirm' }))
      state.activate()
      const parent = parentAgent()
      parent._interruptState = state

      const error = await run(
        makeSubagent({ builder: capturingBuilder().builder }),
        { task: 'x' },
        { agent: parent }
      ).catch((caught: unknown) => caught)

      expect(error).toBeInstanceOf(InterruptError)
      expect((error as InterruptError).interrupts.map((interrupt) => interrupt.id)).toEqual(['subagent:t1:i1'])
    })

    it('returns an error when the interrupted child is no longer available', async () => {
      vi.spyOn(logger, 'warn').mockImplementation(() => {})
      const parent = parentAgent()
      parent._interruptState.registerInterrupt(new Interrupt({ id: 'subagent:t1:i1', name: 'confirm' }))
      parent._interruptState.activate()
      parent._interruptState.resume([new InterruptResponseContent({ interruptId: 'subagent:t1:i1', response: 'yes' })])
      const { builder, specs } = capturingBuilder()

      const { result } = await run(makeSubagent({ builder }), { task: 'x' }, { agent: parent })

      expect(result.status).toBe('error')
      expect(resultText(result)).toMatch(/did NOT run/)
      expect(specs).toHaveLength(0)
    })

    it('rethrows child interrupts as-is when the parent has no interrupt state', async () => {
      const childInterrupt = new Interrupt({ id: 'i1', name: 'confirm' })
      const { builder } = capturingBuilder(new FakeChild(fakeResult('', 'interrupt', [childInterrupt])))
      const parent = { appState: new StateStore(), messages: [] } as unknown as Agent

      const error = await run(makeSubagent({ builder }), { task: 'x' }, { agent: parent }).catch(
        (caught: unknown) => caught
      )

      expect((error as InterruptError).interrupts).toEqual([childInterrupt])
    })
  })

  describe('with real agents', () => {
    it('delegates to a child built from the parent and returns its report', async () => {
      // The child inherits the parent's model, so turns are consumed in order across both agents.
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'subagent', toolUseId: 't1', input: { task: 'research' } })
        .addTurn({ type: 'textBlock', text: 'child report' })
        .addTurn({ type: 'textBlock', text: 'parent done' })
      const parent = new Agent({ model, tools: [subagent], printer: false })

      const result = await parent.invoke('go')

      expect(result.toString()).toBe('parent done')
      const toolResult = parent.messages[2]!.content[0] as ToolResultBlock
      expect(toolResult.status).toBe('success')
      expect(resultText(toolResult)).toBe('child report')
    })

    it('propagates a child tool interrupt to the parent and resumes the child', async () => {
      let confirmed = 0
      const confirmTool = createMockTool('confirmTool', (context) => {
        context.interrupt({ name: 'confirm', reason: 'Please confirm' })
        confirmed += 1
        return 'ok'
      })
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'subagent', toolUseId: 't1', input: { task: 'confirm it' } })
        .addTurn({ type: 'toolUseBlock', name: 'confirmTool', toolUseId: 'inner-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'child done' })
        .addTurn({ type: 'textBlock', text: 'parent done' })
      const parent = new Agent({ model, tools: [makeSubagent(), confirmTool], printer: false })

      const interrupted = await parent.invoke('go')

      expect(interrupted.stopReason).toBe('interrupt')
      const interruptId = 'subagent:t1:tool:inner-1:confirm'
      expect(interrupted.interrupts).toMatchObject([{ id: interruptId, name: 'confirm', reason: 'Please confirm' }])

      const result = await parent.invoke([new InterruptResponseContent({ interruptId, response: 'yes' })])

      expect(result.stopReason).toBe('endTurn')
      expect(result.toString()).toBe('parent done')
      expect(confirmed).toBe(1)
    })
  })
})
