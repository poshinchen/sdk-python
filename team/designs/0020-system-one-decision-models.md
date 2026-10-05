# Decision Models

**Date**: 2026-10-03

**Issue**: [#4551](https://github.com/strands-agents/harness-sdk/issues/4551)

**Scope**: Python SDK, experimental.

## Problem

Agents keep asking a model small closed questions. Which specialist handles this request? Which model serves this turn? Is this message urgent? The SDK has no type for it, so every call site asks a general `Model` for structured output and parses the reply by hand.

The one built-in example, `ClassifierStrategy` ([`classifier_strategy.py`](../../strands-py/src/strands/models/routing/classifier_strategy.py)), spends most of its code bounding the request and keeping message content out of its instructions. Its call opens no span, and its tokens reach neither the agent's metrics nor a trace — the same gap [#4005](https://github.com/strands-agents/harness-sdk/pull/4005) describes. A second decision point would repeat all of that.

A faster option exists but is unreachable from here. System One models like [Jev](https://docs.typesafe.ai/concepts/system-one) and [Strands Decider](https://github.com/strands-labs/strands-decider) take state plus questions and return an answer per question without generating text. The SDK's decision points take a `Model`, which is a streaming chat contract, so today a developer who wants one writes their own HTTP client outside any SDK abstraction.

## Proposal

A `DecisionModel` interface, with an LLM implementation and an experimental System One one, plus a `DecisionStrategy` that lets `ModelRouter` use either.

```python
@dataclass(frozen=True)
class Question:
    instructions: str                                # "Which team should handle this ticket?"
    options: type[Enum] | type[bool] = bool          # bool: yes or no. Enum class: one of its members.
    descriptions: Mapping[Enum, str] | None = None   # optional, what each member means

@dataclass(frozen=True)
class DecisionResult:
    answers: Mapping[str, bool | Enum]               # one answer per question id
    usage: Usage
    model_id: str | None = None

class DecisionModel(abc.ABC):
    @abc.abstractmethod
    async def ask(self, state: str | Mapping[str, Any], questions: Mapping[str, Question], **kwargs) -> DecisionResult: ...
```

An implementation returns an answer for every question id — a `bool`, or a member of that question's `Enum`. One request or one per question is up to the implementation. Failures raise; there is no partial result.

`LLMDecisionModel(model)` asks any Strands `Model` through a tool call and reads answers from the tool input. State goes in message content, questions in instructions, so text in the state cannot rewrite a question. `SystemOneDecisionModel(...)` calls a System One endpoint — `POST /v1/systemone`, the shape Jev serves and Strands Decider documents. `DecisionStrategy(decision_model)` is a `RoutingStrategy` that asks one question whose options are the candidates' names and descriptions; any error becomes a decline, so the router falls back to its default just as `ClassifierStrategy` does.

Uncertain answers are out of scope. Only System One models measure confidence, and how a caller handles uncertainty needs its own design.

### Token usage and traces

Every `DecisionResult` carries input, output and total tokens — `LLMDecisionModel` reads them from the metadata event, `SystemOneDecisionModel` from the response body. Implementations do not open spans; the caller does. `DecisionStrategy` opens a `decision` span as a child of the active span through a new `Tracer.start_decision_span`/`end_decision_span` pair that mirrors the existing memory spans. The span carries `gen_ai.operation.name="decision"`, request and response model ids, input and output tokens, the question ids, and the answers as a span event. Usage counts toward `accumulated_usage` through the auxiliary-call path from #4005, under source `decision`; until that lands it stays on the result and the span.

## Alternatives

**A System One model as a `Model` provider.** No new type, works anywhere a `Model` does. But `Model` streams text and tool calls for any schema, and a System One model generates neither, so the provider would fake a stream and reject most requests at run time.

**A typed Pydantic schema whose fields become questions.** Statically typed answers, but a schema compiler and a run-time options builder is a lot of surface for what a mapping of `Question`s already covers. Easy to layer on top of `ask` later.

## Developer Experience

```python
from enum import Enum

from strands import Agent
from strands.experimental.decisions import DecisionStrategy, LLMDecisionModel, Question
from strands.models import BedrockModel, ModelRouter, RoutingCandidate

class Team(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"

decider = LLMDecisionModel(BedrockModel(model_id="us.amazon.nova-2-lite-v1:0"))

result = await decider.ask(
    state="I was charged twice and now I can't log in.",
    questions={
        "team": Question("Which team should handle this ticket?", options=Team),
        "urgent": Question("Does the customer need an answer today?"),
    },
)
result.answers["team"]    # Team.BILLING
result.answers["urgent"]  # True
result.usage              # {"inputTokens": 212, "outputTokens": 31, "totalTokens": 243}

router = ModelRouter(
    models=[
        RoutingCandidate(BedrockModel(model_id="global.anthropic.claude-sonnet-5-5"), name="complex",
                         description="Multi-step reasoning, code generation"),
        RoutingCandidate(BedrockModel(model_id="us.amazon.nova-2-lite-v1:0"), name="routine",
                         description="Direct questions, short summaries"),
    ],
    strategy=DecisionStrategy(decider),
)
agent = Agent(model=router)
```

An `Enum` with no members, or empty `questions`, raises `ValueError` before any request. A missing or wrong-typed answer raises `ValueError`. A throttled request raises `ModelThrottledException`. `DecisionStrategy` turns every error into a decline.
