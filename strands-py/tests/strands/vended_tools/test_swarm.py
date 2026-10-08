"""Tests for the swarm vended tool.

The swarm tool is a thin shim over :class:`~strands.multiagent.Swarm`: it owns
spec validation, agent construction, and result mapping. Tests mock the SDK
Swarm class so no model calls are made.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.agent import AgentResult
from strands.agent.state import AgentState
from strands.multiagent.base import NodeResult, Status
from strands.multiagent.spec import Choice, Fixed, Inherit, Open, Preset
from strands.multiagent.swarm import SwarmResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.types.tools import ToolContext
from strands.vended_tools.swarm import make_swarm, swarm
from strands.vended_tools.swarm.swarm import (
    _DEPTH_STATE_KEY,
    _build_agent_item_schema,
    _build_description,
    _resolve_specs,
)
from strands.vended_tools.swarm.types import DEFAULT_MAX_AGENTS

_swarm_module = importlib.import_module("strands.vended_tools.swarm.swarm")


def _mock_parent(tool_names: list[str] | None = None, state: dict | None = None) -> SimpleNamespace:
    registry = ToolRegistry()
    if tool_names:
        for name in tool_names:
            t = Mock()
            t.tool_name = name
            registry.registry[name] = t
    return SimpleNamespace(
        tool_registry=registry,
        model=Mock(),
        sandbox=None,
        callback_handler=None,
        trace_attributes=None,
        state=AgentState(state),
    )


def _ctx(parent=None):
    if parent is None:
        parent = _mock_parent()
    return ToolContext(tool_use={"name": "swarm", "toolUseId": "id", "input": {}}, agent=parent, invocation_state={})


def _spec(name, **kw):
    return {"name": name, **kw}


def _result(*, status=Status.COMPLETED, text="Done!", node="writer"):
    r = AgentResult(
        message={"role": "assistant", "content": [{"text": text}]},
        stop_reason="end_turn",
        state={},
        metrics=EventLoopMetrics(),
    )
    sr = SwarmResult(status=status, results={node: NodeResult(result=r, status=Status.COMPLETED)}, execution_count=1)
    return sr


def _kwargs():
    return dict(
        max_agents=DEFAULT_MAX_AGENTS,
        presets={},
        default_preset=None,
        instructions=Open(),
        tools=None,
        mcp_servers=Inherit(),
        model=Inherit(),
    )


def _patch(result=None):
    """Patch Swarm and _default_builder so no model calls are made."""
    if result is None:
        result = _result()
    sp = patch.object(_swarm_module, "Swarm")

    def _builder(spec):
        child = Mock(name=spec.name)
        child.state = AgentState()
        child.tool_registry = ToolRegistry()
        return child

    bp = patch.object(_swarm_module, "_default_builder", return_value=_builder)

    class _Ctx:
        def __enter__(self):
            self.cls = sp.__enter__()
            bp.__enter__()
            self.cls.return_value.invoke_async = AsyncMock(return_value=result)
            return self.cls

        def __exit__(self, *a):
            bp.__exit__(*a)
            sp.__exit__(*a)

    return _Ctx()


class TestBuildAgentItemSchema:
    def test_visible_and_hidden_axes(self):
        # Open + no presets → instructions required.
        schema = _build_agent_item_schema(
            presets={}, instructions=Open(), tools=None, mcp_servers=Inherit(), model=None
        )
        assert schema["required"] == ["name", "instructions"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["instructions"]["type"] == "string"

        # Open + presets → instructions not required (preset provides a default).
        with_presets = _build_agent_item_schema(
            presets={"w": Preset(description="W")},
            instructions=Open(),
            tools=None,
            mcp_servers=Inherit(),
            model=None,
        )
        assert with_presets["required"] == ["name"]

        # Fixed/Inherit → hidden.
        hidden = _build_agent_item_schema(
            presets={},
            instructions=Fixed("x"),
            tools=Inherit(),
            mcp_servers=Inherit(),
            model=Fixed("m"),
        )
        assert set(hidden["properties"]) == {"name"}

    def test_choice_axes_and_presets(self):
        presets = {"alpha": Preset(description="First"), "beta": Preset(description="Second")}
        schema = _build_agent_item_schema(
            presets=presets,
            instructions=Choice(["A", "B"]),
            tools=Choice(["calc", "fetch"], multiple=True),
            mcp_servers=Choice(["docs", "github"], multiple=True),
            model=Choice(["fast", "smart"]),
        )
        assert schema["properties"]["instructions"]["enum"] == ["A", "B"]
        assert schema["properties"]["tools"]["items"]["enum"] == ["calc", "fetch"]
        assert schema["properties"]["mcp_servers"]["items"]["enum"] == ["docs", "github"]
        assert schema["properties"]["model"]["enum"] == ["fast", "smart"]
        assert sorted(schema["properties"]["agent_type"]["enum"]) == ["alpha", "beta"]


class TestBuildDescription:
    def test_with_and_without_presets(self):
        assert _build_description("Base.", {}) == "Base."
        desc = _build_description("Base.", {"w": Preset(description="Writes.")})
        assert "w" in desc and "Writes." in desc


class TestResolveSpecs:
    @pytest.mark.parametrize(
        "agents,match",
        [
            ("bad", "must be a list"),
            ([], "At least 1"),
            ([{"name": "a"}] * 25, "At most 20"),
            (["not a dict"], "must be a dict"),
            ([{}], "non-empty 'name'"),
            ([{"name": "a"}, {"name": "a"}], "Duplicate"),
        ],
    )
    def test_rejects_invalid_input(self, agents, match):
        with pytest.raises(ValueError, match=match):
            _resolve_specs(agents, **_kwargs())

    def test_resolves_instructions_and_names(self):
        specs = _resolve_specs([_spec("a", instructions="Do."), _spec("b")], **_kwargs())
        assert [s.name for s in specs] == ["a", "b"]
        assert specs[0].instructions == "Do."

    def test_presets_and_defaults(self):
        kw = _kwargs()
        kw["presets"] = {"writer": Preset(instructions="Write.")}
        kw["default_preset"] = "writer"
        assert _resolve_specs([_spec("a", agent_type="writer")], **kw)[0].instructions == "Write."
        assert _resolve_specs([_spec("b")], **kw)[0].instructions == "Write."  # default
        with pytest.raises(ValueError, match="Unknown agent_type"):
            _resolve_specs([_spec("c", agent_type="nope")], **kw)

    def test_choice_tools_filtered(self):
        kw = _kwargs()
        kw["tools"] = Choice(["calc", "fetch"], multiple=True)
        assert _resolve_specs([_spec("a", tools=["calc", "unknown"])], **kw)[0].tools == ["calc"]


class TestMakeSwarm:
    def test_default_instance(self):
        from strands.tools.decorator import DecoratedFunctionTool

        assert isinstance(swarm, DecoratedFunctionTool) and swarm.tool_name == "swarm"
        assert make_swarm(name="team").tool_name == "team"

    @pytest.mark.parametrize(
        "kw",
        [
            {"max_agents": 0},
            {"max_agents": True},
            {"max_depth": 0},
            {"max_depth": True},
            {"tools": Choice([], multiple=True)},
            {"tools": Choice(["a"])},  # multiple=False on a list axis
            {"mcp_servers": Choice([], multiple=True)},
            {"mcp_servers": Choice(["a"])},  # multiple=False on a list axis
        ],
    )
    def test_rejects_invalid_limits(self, kw):
        with pytest.raises(ValueError):
            make_swarm(**kw)

    def test_schema_reflects_configuration(self):
        agents = swarm.tool_spec["inputSchema"]["json"]["properties"]["agents"]
        assert agents["minItems"] == 1 and agents["maxItems"] == DEFAULT_MAX_AGENTS
        item = agents["items"]
        assert item["required"] == ["name", "instructions"] and item["additionalProperties"] is False
        assert "instructions" in item["properties"]

        # Custom max_agents
        assert make_swarm(max_agents=3).tool_spec["inputSchema"]["json"]["properties"]["agents"]["maxItems"] == 3

        # Choice instructions
        ci = make_swarm(instructions=Choice(["A"])).tool_spec["inputSchema"]["json"]["properties"]["agents"]["items"]
        assert ci["properties"]["instructions"]["enum"] == ["A"]

        # Fixed instructions hidden
        fi = make_swarm(instructions=Fixed("x")).tool_spec["inputSchema"]["json"]["properties"]["agents"]["items"]
        assert "instructions" not in fi["properties"]

        # Choice mcp_servers visible
        ms = make_swarm(mcp_servers=Choice(["docs", "gh"], multiple=True))
        msi = ms.tool_spec["inputSchema"]["json"]["properties"]["agents"]["items"]
        assert msi["properties"]["mcp_servers"]["items"]["enum"] == ["docs", "gh"]

        # Fixed mcp_servers hidden
        msf = make_swarm(mcp_servers=Fixed([])).tool_spec["inputSchema"]["json"]["properties"]["agents"]["items"]
        assert "mcp_servers" not in msf["properties"]

    def test_presets_in_schema_and_description(self):
        t = make_swarm(presets={"writer": Preset(description="Writes."), "coder": Preset(description="Codes.")})
        item = t.tool_spec["inputSchema"]["json"]["properties"]["agents"]["items"]
        assert sorted(item["properties"]["agent_type"]["enum"]) == ["coder", "writer"]
        assert "writer" in t.tool_spec["description"]


class TestSwarmToolExecution:
    @pytest.mark.asyncio
    async def test_success(self):
        with _patch() as cls:
            result = await swarm(task="go", agents=[_spec("w", instructions="Write.")], tool_context=_ctx())
        assert result == "writer: Done!"
        cls.return_value.invoke_async.assert_awaited_once_with("go")

    @pytest.mark.asyncio
    async def test_forwards_limits(self):
        custom = make_swarm(max_handoffs=5, max_iterations=10, execution_timeout=60.0, node_timeout=30.0)
        with _patch() as cls:
            await custom(task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx())
        kw = cls.call_args[1]
        assert (kw["max_handoffs"], kw["max_iterations"], kw["execution_timeout"], kw["node_timeout"]) == (
            5,
            10,
            60.0,
            30.0,
        )

    @pytest.mark.asyncio
    async def test_input_validation_errors(self):
        with pytest.raises(ValueError, match="must be a list"):
            await swarm(task="t", agents="bad", tool_context=_ctx())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Duplicate"):
            await swarm(task="t", agents=[_spec("a"), _spec("a")], tool_context=_ctx())
        with pytest.raises(ValueError, match="At most 2"):
            await make_swarm(max_agents=2)(task="t", agents=[_spec("a"), _spec("b"), _spec("c")], tool_context=_ctx())

    @pytest.mark.asyncio
    async def test_failed_status_raises(self):
        failed = _result(status=Status.FAILED, text="partial")
        failed.execution_count = 20
        with _patch(failed), pytest.raises(RuntimeError, match="status=failed"):
            await make_swarm(max_handoffs=10, max_iterations=20)(
                task="t",
                agents=[_spec("a", instructions="Do.")],
                tool_context=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_builder_and_nodes(self):
        built = []
        sentinels = [Mock(name="a"), Mock(name="b")]
        it = iter(sentinels)

        def builder(spec):
            built.append(spec)
            return next(it)

        custom = make_swarm(builder=builder, instructions=Fixed("Locked."))
        with _patch() as cls:
            await custom(task="t", agents=[_spec("a"), _spec("b")], tool_context=_ctx())
        assert [s.name for s in built] == ["a", "b"]
        assert all(s.instructions == "Locked." for s in built)
        assert cls.call_args[1]["nodes"] == sentinels


class TestDepthGuard:
    @pytest.mark.asyncio
    async def test_exhausted_raises(self):
        parent = _mock_parent(state={_DEPTH_STATE_KEY: 0})
        with pytest.raises(RuntimeError, match="depth limit reached"):
            await swarm(task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx(parent))

    @pytest.mark.asyncio
    async def test_propagates_decremented_depth(self):
        children = []

        def builder(spec):
            child = Mock(name=spec.name)
            child.state = AgentState()
            child.tool_registry = ToolRegistry()
            children.append(child)
            return child

        # First call (no stored depth) → uses max_depth.
        custom = make_swarm(builder=builder, max_depth=3)
        with _patch():
            await custom(task="t", agents=[_spec("a", instructions="X")], tool_context=_ctx())
        assert children[0].state.get(_DEPTH_STATE_KEY) == 2

        # Stored depth on parent → uses that instead.
        children.clear()
        parent = _mock_parent(state={_DEPTH_STATE_KEY: 2})
        with _patch():
            await custom(
                task="t", agents=[_spec("a", instructions="X"), _spec("b", instructions="Y")], tool_context=_ctx(parent)
            )
        assert children[0].state.get(_DEPTH_STATE_KEY) == 1
        assert children[1].state.get(_DEPTH_STATE_KEY) == 1


class TestToolExclusion:
    @pytest.mark.asyncio
    async def test_handoff_to_agent_excluded(self):
        children = []

        def builder(spec):
            child = Mock(name=spec.name)
            child.state = AgentState()
            child.tool_registry = ToolRegistry()
            child.tool_registry.registry["handoff_to_agent"] = Mock()
            children.append(child)
            return child

        with _patch():
            await make_swarm(builder=builder)(task="t", agents=[_spec("a", instructions="X")], tool_context=_ctx())
        assert "handoff_to_agent" not in children[0].tool_registry.registry


class TestExports:
    def test_importable(self):
        from strands.vended_tools import make_swarm as ms
        from strands.vended_tools import swarm as s

        assert s is swarm and ms is make_swarm
