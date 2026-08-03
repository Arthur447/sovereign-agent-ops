"""The ops tool registry — capability, risk and inverse, declared together.

## The enforcement point is here, not in the agent

The reversibility gate could live in the remediation agent, next to the
code that decides what to do. It does not, on purpose.

An agent-side check is advisory. It protects against an agent that
reasons badly, which is the easy failure, and not at all against an
agent that has been prompt-injected through a log line it read, which is
the failure that actually ends up in the incident report. The tool server
is the only place that sees every call regardless of who made it or why,
so it is the only place a guarantee can be made rather than requested.

The agent's remaining job is translation: it turns the server's refusal
into an A2A `TASK_STATE_AUTH_REQUIRED` its client understands. That is a
protocol concern, and it belongs on the agent side.

## Why risk is declared next to the tool

`ToolRisk` is written once, by whoever writes the tool, and reviewed as
part of the same diff. The alternative — a central policy file mapping
tool names to risk levels — drifts the first time a tool changes what it
does without changing its name, and nothing detects the drift.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sovops.policy.reversibility import Reversibility, ToolRisk


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: what it does, what it costs if wrong, how to undo it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk: ToolRisk
    handler: Callable[..., dict[str, Any]]
    # Read-only tools skip the gate entirely — there is nothing to undo
    # about a read. Keeping the flag explicit rather than deriving it from
    # `REVERSIBLE` matters: a reversible *write* still consumes episode
    # budget, and conflating the two would give an agent an unlimited
    # supply of "harmless" restarts.
    read_only: bool = False


class ToolRegistry:
    """Holds the tool set and validates it at registration.

    Validation at registration rather than at call time is the whole
    point: a tool whose declared inverse does not exist should fail when
    the server boots, in CI, not at 3am when the gate reaches for a
    rollback that was never implemented.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def validate(self) -> None:
        """Whole-registry invariants, checked once the set is complete."""
        for spec in self._tools.values():
            if spec.read_only and spec.risk.reversibility is not Reversibility.REVERSIBLE:
                raise ValueError(
                    f"{spec.name!r} is read_only but declares "
                    f"{spec.risk.reversibility}; a read changes nothing"
                )
            inverse = spec.risk.rollback_tool
            if inverse and inverse not in self._tools:
                raise ValueError(
                    f"{spec.name!r} declares inverse {inverse!r}, which is not "
                    "registered — a rollback plan that cannot be executed is not "
                    "a rollback plan"
                )

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}") from None

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def read_only_names(self) -> list[str]:
        """The scope handed to the triage agent.

        Triage gets exactly this set. The privilege boundary between the
        two agents is enforced by which token the server accepts for which
        tool, so an over-eager diagnostician cannot restart anything even
        if it decides that would be informative.
        """
        return sorted(s.name for s in self._tools.values() if s.read_only)
