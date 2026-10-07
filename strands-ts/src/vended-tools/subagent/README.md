# Subagent Tool

Delegates a self-contained task to a child agent that runs in its own context and returns a final report.

Use this when a subtask would otherwise flood the parent's context with intermediate work and only the conclusion matters. The model writes the task on each call; the child cannot ask follow-up questions, so the task must carry all the context it needs. Each call builds a fresh child, runs it to completion, and returns its final message as the tool result.

`makeSubagent` takes axis policies from `@strands-agents/sdk/multiagent` (`Fixed`, `Inherit`, `Open`, `Choice`) that control which parameters the model sees and which values it can supply for the child's `instructions`, `tools`, `mcpServers`, `model`, and `context`. Named roles are bundled into `Preset`s that the model selects via `agent_type`.

## Features

- **Fresh Context per Call**: Each delegation runs in a newly built child agent, so intermediate work never enters the parent's conversation
- **Authority Modes**: Each axis is pinned, inherited, free-form, or limited to a developer-set set of options
- **Presets**: Named roles the model picks via `agent_type`; defaults to a built-in `generalist`
- **Context Sharing**: Optionally share the parent's conversation (`'all'` or text-only `'no_tools'`), bounded by `last_messages`
- **Depth Limit**: Nested delegation is capped (default: 2 levels)
- **Interrupt Propagation**: Child interrupts surface on the parent and resume the same child when answered

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { subagent } from '@strands-agents/sdk/vended-tools/subagent'

const agent = new Agent({ systemPrompt: 'You are a manager.', tools: [subagent] })
await agent.invoke('Research the latest TypeScript 5.x features and summarize them.')
```

Presets, a model choice, shared context, and a tool subset:

```typescript
import { Agent } from '@strands-agents/sdk'
import { Choice, Option, Preset } from '@strands-agents/sdk/multiagent'
import { makeSubagent } from '@strands-agents/sdk/vended-tools/subagent'

const subagent = makeSubagent({
  presets: {
    researcher: new Preset({
      instructions: 'You research topics thoroughly.',
      description: 'deep research on a topic',
    }),
    reviewer: new Preset({
      instructions: 'You review code for correctness and style.',
      description: 'code review',
    }),
  },
  model: new Choice([
    new Option('fast', 'us.anthropic.claude-sonnet-4-20250514-v1:0', 'quick tasks'),
    new Option('deep', 'us.anthropic.claude-opus-4-20250514-v1:0', 'hard problems'),
  ]),
  context: new Choice(['none', 'all', 'no_tools']),
  tools: new Choice(['read', 'shell', 'write'], true),
  maxDepth: 3,
})
const agent = new Agent({ tools: [subagent] })
await agent.invoke('Review the changes in src/main.ts for correctness.')
```

Custom builder:

```typescript
import { Agent } from '@strands-agents/sdk'
import type { AgentSpec } from '@strands-agents/sdk/multiagent'
import { makeSubagent } from '@strands-agents/sdk/vended-tools/subagent'

const subagent = makeSubagent({
  builder: (spec: AgentSpec) => new Agent({ systemPrompt: spec.instructions ?? '', printer: false }),
})
```

## API

### `subagent`

The default tool, produced by `makeSubagent()`: the `generalist` preset, free-form `instructions`, inherited tools, MCP servers, and model, and no shared context.

### `makeSubagent(options?)`

| Option          | Type                         | Default                      | Description                                                                            |
| --------------- | ---------------------------- | ---------------------------- | -------------------------------------------------------------------------------------- |
| `builder`       | `(spec: AgentSpec) => Agent` | (inherits from parent)       | Turns a resolved spec into a child agent.                                              |
| `presets`       | `Record<string, Preset>`     | `{ generalist: GENERALIST }` | Named roles the model selects via `agent_type`. Pass `{}` to disable presets.          |
| `defaultPreset` | `string`                     | first preset                 | Preset used when the model omits `agent_type`.                                         |
| `instructions`  | `Open \| Choice \| Fixed`    | `new Open()`                 | Policy for the child's system prompt.                                                  |
| `tools`         | `Choice \| Fixed \| Inherit` | `new Inherit()`              | Policy for the child's tools. A `Choice` must be `multiple`.                           |
| `mcpServers`    | `Choice \| Fixed \| Inherit` | `new Inherit()`              | Policy for the child's MCP servers (by `clientName`). A `Choice` must be `multiple`.   |
| `model`         | `Inherit \| Choice \| Fixed` | `new Inherit()`              | Policy for the child's model.                                                          |
| `context`       | `Fixed \| Choice`            | `new Fixed('none')`          | How much of the parent's conversation the child sees: `'none'`, `'all'`, `'no_tools'`. |
| `maxDepth`      | `number`                     | `2`                          | Upper bound on nested delegation levels. Must be a positive integer.                   |
| `name`          | `string`                     | `subagent`                   | Tool name.                                                                             |

Throws if `name` is empty, `maxDepth` is not a positive integer, or a `tools` / `mcpServers` `Choice` has no options or is not `multiple`.

The default builder gives each child the parent's model, tools, and MCP servers (narrowed by the resolved spec), sandbox, printer setting, and trace attributes, plus `contextManager: 'auto'` unless the child's model is stateful.

### `GENERALIST`

The built-in general-purpose `Preset` used when no `presets` are supplied.

### `DEFAULT_SUBAGENT_DESCRIPTION` / `DEFAULT_SUBAGENT_MAX_DEPTH`

The base tool description and the default `maxDepth`.

### Input

Only `task` is always present; the other parameters appear depending on the configured policies.

| Property        | Type       | Required | Description                                                                     |
| --------------- | ---------- | -------- | ------------------------------------------------------------------------------- |
| `task`          | `string`   | Yes      | The self-contained task, including all context the subagent needs.              |
| `agent_type`    | `string`   | No       | Preset name. Present when presets are configured.                               |
| `instructions`  | `string`   | No       | System prompt for the child. Present when `instructions` is `Open` or `Choice`. |
| `tools`         | `string[]` | No       | Subset of tools to grant. Present when `tools` is a `Choice`.                   |
| `mcp_servers`   | `string[]` | No       | Subset of MCP servers to grant. Present when `mcpServers` is a `Choice`.        |
| `model`         | `string`   | No       | Model option name. Present when `model` is a `Choice`.                          |
| `context`       | `string`   | No       | Context option name. Present when `context` is a `Choice`.                      |
| `last_messages` | `number`   | No       | Share only the last N parent messages. Present when context can be shared.      |

### Output

Returns the child's final response as text. Returns an error result (without building a child) when `task` is missing, `agent_type` is unknown, or the depth limit is reached; also when the child throws or is cancelled.

### Interrupts

When the child stops on an interrupt, the interrupt is raised on the parent with its id namespaced as `subagent:<toolUseId>:<childInterruptId>` (with `toolUseId` URI-encoded). Resuming the parent with a response resumes the same child in process. The interrupted child is held in memory only, so it cannot be resumed after a process restart.
