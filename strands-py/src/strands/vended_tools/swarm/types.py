"""Shared types and constants for the swarm tool."""

DEFAULT_SWARM_DESCRIPTION = (
    "Spin up a team of specialised AI agents that collaborate autonomously via handoffs "
    "to solve a complex task. Each agent has its own role (system prompt) and optional tool "
    "access drawn from the caller's tools. The team self-organises: agents hand off to one "
    "another and signal completion without central control. Returns a consolidated result "
    "from the entire team's work."
)
"""Description for the default swarm tool."""

DEFAULT_MAX_AGENTS = 20
"""Upper bound on the number of agents a single swarm invocation may create."""

DEFAULT_MAX_DEPTH = 2
"""Upper bound on the number of nested swarm levels."""
