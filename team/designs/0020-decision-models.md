# Decision Models

**Date**: 2026-10-05

**Issue**: [#4551](https://github.com/strands-agents/harness-sdk/issues/4551)

**Scope**: TypeScript SDK first, stable. Python parity follows.

## Problem

Agents keep asking a model small closed questions. Which specialist handles this request? Which model serves this turn? Is this message urgent? The SDK has no primitive for it, so every call site reaches for a general `Model` through `structured_output` (Python) or `StructuredOutputTool` + forced `toolChoice` (TypeScript), hand-rolls prompt-injection defense, and parses the reply.

The one call site that does this today, `ClassifierStrategy` ([`strands-ts/src/models/routing/classifier-strategy.ts`](../../strands-ts/src/models/routing/classifier-strategy.ts), 374 lines; mirror in Python at 327), spends most of its code bounding untrusted inputs, wrapping them in an injection-defense envelope, and parsing a structured answer back out. A second decision point would repeat all of that. Its call also opens no span, and its tokens reach neither the agent's metrics nor a trace — the same gap [#4005](https://github.com/strands-agents/harness-sdk/pull/4005) describes.

## Proposal

A `DecisionModel` abstraction and one implementation that runs on any Strands `Model`. V1 ships the primitive only. A `DecisionStrategy` routing adapter and the `ClassifierStrategy` rewrite follow.

### Interface

```ts
export abstract class DecisionModel {
  async ask(
    state: DecisionInput,
    questions: Record<string, Question>,
    options?: { cancelSignal?: AbortSignal },
  ): Promise<DecisionResult> {
    // Validates questions synchronously, opens a `decision` span,
    // delegates to _ask, records usage, closes the span.
  }
  protected abstract _ask(
    state: DecisionInput,
    questions: Record<string, Question>,
    options?: { cancelSignal?: AbortSignal },
  ): Promise<DecisionResult>
}

export interface Question {
  instructions: string
  options: 'bool' | readonly string[]
  descriptions?: Record<string, string>
  allowUncertain?: boolean  // default true
}

export type DecisionAnswer = boolean | string | Uncertain
export class Uncertain { constructor(public readonly reason?: string) {} }

export interface DecisionResult {
  answers: Record<string, DecisionAnswer>
  usage: { inputTokens: number; outputTokens: number; totalTokens: number }
  modelId?: string
}

export type DecisionInput = string | ContentBlock[] | Message[]
```

An implementation returns an answer for every question id — a `boolean`, one of the question's string-literal options, or an `Uncertain` instance. Failures raise; there is no partial result. The abstract base class opens a `decision` span and reports usage on every call so a concrete implementation does not have to; the trade-off is that every `DecisionModel` subclass is traced the same way (`Model` leaves that to its caller, deliberately — `DecisionModel` goes the other way because `DecisionStrategy` is not guaranteed to exist as the owning caller).

### `LLMDecisionModel`

The one v1 implementation. Builds a Zod schema per call where each field is a question id, constructs a `StructuredOutputTool` from the schema, places `state` as the user turn, puts the question list in the system prompt, and calls `model.stream(messages, { toolSpecs, toolChoice: { tool: { name } }, systemPrompt })`. The tool-use input is read, validated through the same Zod schema, and mapped back to typed answers. On `allowUncertain: true` (the default) the field type adds an `{ kind: 'uncertain', reason?: string }` variant, and the system-prompt rule tells the model to pick it when evidence is insufficient.

**State shape.** `DecisionInput` is `string | ContentBlock[] | Message[]` — the subset of the agent's `InvokeArgs` that makes sense for a stateless decision. Multimodal passes through: an image or document in the user turn reaches the model the same way an agent's user turn does. No character budgets. If a caller sends too much, the model's `ContextWindowOverflowError` surfaces unchanged.

**Prompt-injection defense.** On by default. The system prompt carries the `MANDATORY RULES` block inherited from `ClassifierStrategy`'s scaffolding, telling the model to treat the user turn as evidence rather than instructions. Callers with developer-controlled state construct the model with `untrustedState: false` to skip the envelope and save tokens. Because `state` rides in the user turn, the message/system split carries the trust boundary; no `<untrusted_context>` XML wrapper or `&<>` escaping is needed.

**Transport.** TypeScript's `Model` contract has no `structuredOutput` method, so `LLMDecisionModel` uses `StructuredOutputTool` + forced `toolChoice`, matching `ClassifierStrategy`'s current approach. Python's `LLMDecisionModel` will instead call `model.structured_output` with a dynamic Pydantic model built the same way. The public interface stays identical; the internal transport is each language's existing pattern.

### Errors

Caller errors raise synchronously before any model call: empty `questions`, a question with `options: []`, duplicate option values, or boolean questions with non-boolean description keys all throw `DecisionConfigError`. Runtime errors raise from `ask`: a model throttle or context overflow re-raises unchanged, a missing tool-use block or failing schema validation throws `DecisionValidationError` with the original cause chained. Callers that want decline-on-error wrap the call. No retries; retry policy is cross-cutting and belongs at the `Model` layer or in a caller wrapper, not inside every `DecisionModel`.

`Uncertain` is a result, not an error — it means the model worked, the question was well-formed, and the answer is "I can't tell." A caller that wants to treat uncertainty as failure narrows at the use site. The span records which question ids resolved to uncertain so observers can measure confidence patterns per model.

### Observability

Every `DecisionResult` carries input, output, and total tokens read from the model's final metadata event. The `decision` span opened by the base class records `gen_ai.operation.name="decision"`, request and response model ids, input and output tokens, the question ids as a span event, and the answers as a span event. Agent-level `accumulatedUsage` wiring lands when [#4005](https://github.com/strands-agents/harness-sdk/pull/4005) merges — the base class picks up the auxiliary-call hook under `source='decision'` without changing the public surface.

### File layout

`strands-ts/src/decisions/` as a top-level directory, same tier as `models` and `tools`. Internal layout: `types.ts` (`Question`, `DecisionResult`, `Uncertain`, errors), `base.ts` (`DecisionModel` abstract), `llm.ts` (`LLMDecisionModel`), with `_schema.ts` and `_prompt.ts` as `@internal` helpers kept out of the barrel. Public exports from the root `index.ts`: `DecisionModel`, `LLMDecisionModel`, `Uncertain`, `DecisionConfigError`, `DecisionValidationError`, and the `Question` / `DecisionResult` / `DecisionAnswer` / `DecisionInput` types.

## Alternatives

**A System One model as a `Model` provider.** No new type, works anywhere a `Model` does. Rejected because `Model` streams text and tool calls for arbitrary schemas, and a System One model generates neither; the provider would fake a stream and reject most requests at runtime.

**A typed schema class whose fields are the questions.** A single Zod or Pydantic class returned validated. More static safety, but a schema compiler plus a run-time options builder is a lot of surface for what `Record<string, Question>` already covers. Easy to layer on top of `ask` later.

**Partial results.** `DecisionResult.answers` widened to `boolean | string | Uncertain | DecisionError` per question. Rejected: a lot of per-answer-error surface for a case callers can approximate by splitting their question set.

**Experimental namespace.** `strands-ts/src/experimental/decisions`. Rejected because the planned `ClassifierStrategy` rewrite will depend on `LLMDecisionModel`, and splitting a stable caller's internals into `experimental` either requires a documented exception or a promotion dance. Shipping stable from day one is the honest version; shape changes go through the normal API bar-raising process.

**Character budgets.** `maxStateChars`, `maxQuestionChars` constructor options matching `ClassifierStrategy`'s current knobs. Rejected: `AgentInput`-shaped state does not need `ClassifierStrategy`'s original hedge against unknown message sizes, and the downstream model's `ContextWindowOverflowError` already surfaces when limits are hit.

**Decision hooks in v1.** `BeforeDecisionEvent` and `AfterDecisionEvent`. Rejected: hook names cross-cut both SDKs permanently once added, no concrete caller has asked for one, and the span plus the result already cover observability. Add when a real caller surfaces a need.

## Developer Experience

```ts
import { Agent } from 'strands-agents'
import { LLMDecisionModel, Uncertain } from 'strands-agents'
import { BedrockModel } from 'strands-agents'

const TEAMS = ['billing', 'technical', 'account'] as const
const decider = new LLMDecisionModel(
  new BedrockModel({ modelId: 'us.amazon.nova-2-lite-v1:0' }),
)

const result = await decider.ask(
  'I was charged twice and now I can\'t log in.',
  {
    team: { instructions: 'Which team should handle this ticket?', options: TEAMS },
    urgent: { instructions: 'Does the customer need an answer today?', options: 'bool' },
  },
)

if (result.answers.team instanceof Uncertain) {
  // fall back to a default team, or escalate
} else {
  // narrowed to 'billing' | 'technical' | 'account'
}
result.usage  // { inputTokens: 212, outputTokens: 31, totalTokens: 243 }
```

## Future Work

`DecisionStrategy` lands as a thin `RoutingStrategy` passthrough that calls `ask` with a single enum question whose values are candidate names; `ClassifierStrategy` becomes a preset built on it, dropping ~325–375 lines of scaffolding across the two SDKs. A `SystemOneDecisionModel` implementation calls a System One endpoint ([Jev](https://docs.typesafe.ai/concepts/system-one), [Strands Decider](https://github.com/strands-labs/strands-decider) — `POST /v1/systemone`) and surfaces its native per-question confidence through the same `Uncertain` sentinel. The Python port mirrors the TypeScript design, swapping the `StructuredOutputTool` transport for `model.structured_output` with a dynamic Pydantic model.
