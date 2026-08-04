"""An agent reaches its tools over the wire, or not at all.

The tool server owns the reversibility gate, the episode budget and the
audit ledger, and every one of them sits downstream of
`OpsToolService.call`. A caller sharing the interpreter can skip all
three by importing the handler — and the audit consequence is the bad
one: tampering with a hash-chained ledger is hard, never appearing in it
is free.

Stated in a docstring that is a convention. Here it fails in CI.

## What this guard does not do

It stops the bypass from being *written*. It does not stop a module
already loaded in the same interpreter from being reached at runtime —
that impossibility comes from deploying the tool server as a separate
process, which is a deployment property, not a test one. Two different
controls, two different failures; neither replaces the other.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "sovops"
AGENTS = SRC / "agents"

# The whole server-side package. Not a list of tool names: a rule that
# enumerates what is forbidden is a rule that a new tool escapes on the
# day it is written.
FORBIDDEN_ROOT = "sovops.mcp_server"


def agent_modules() -> list[Path]:
    return sorted(AGENTS.rglob("*.py"))


def imported_modules(source: str) -> set[str]:
    """Every module named by an import, full dotted path.

    Parsed rather than grepped, for the same reason as `test_dep_guard`:
    a docstring that mentions the package is not an import, and a guard
    that fires on prose gets disabled within a week.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_no_agent_imports_the_tool_server():
    offenders = [
        f"{path.relative_to(SRC)} imports {module!r}"
        for path in agent_modules()
        for module in sorted(imported_modules(path.read_text(encoding="utf-8")))
        if module == FORBIDDEN_ROOT or module.startswith(f"{FORBIDDEN_ROOT}.")
    ]
    assert not offenders, (
        "agents must reach tools through sovops.mcp_client over the transport, "
        "which is where the tier check, the gate and the audit row happen:\n  "
        + "\n  ".join(offenders)
    )


def test_the_client_is_not_in_the_forbidden_package():
    """The rule has no exception, and that is deliberate.

    `MCPClient` first lived in `sovops/mcp_server/`, which would have
    forced an allowlist — and an exception carved into a security rule is
    where the next bypass goes to hide. It also mixed two roles that never
    meet in this system: agents are clients of the tool server and never
    servers of it. (A2A keeps both halves together, correctly, because
    there each agent is both.)
    """
    from sovops.mcp_client import MCPClient

    assert MCPClient.__module__.startswith("sovops.mcp_client")


def test_the_guard_can_actually_fail():
    """A guard that cannot fail is decoration."""
    assert "sovops.mcp_server.server" in imported_modules(
        "from sovops.mcp_server.server import OpsToolService"
    )
    assert "sovops.mcp_server" not in imported_modules("# sovops.mcp_server in a comment")
