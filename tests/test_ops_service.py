"""The tool service: tiers, the gate, approvals, and what lands in the ledger.

Uses stub handlers rather than Docker. What is under test is the policy
path, and binding it to a container runtime would make the suite slow and
would fail for reasons that have nothing to do with the rules.
"""

from __future__ import annotations

import pytest

from sovops.audit.ledger import InMemoryAuditStore, Ledger
from sovops.mcp_server.registry import ToolRegistry, ToolSpec
from sovops.mcp_server.server import OpsToolService, Tier, ToolAccessDenied
from sovops.policy.reversibility import (
    BlastRadius,
    Reversibility,
    ReversibilityGate,
    Scope,
    ToolRisk,
)

READ_TOKEN = "tok-triage"
WRITE_TOKEN = "tok-remediation"

_SVC = {
    "type": "object",
    "properties": {"service": {"type": "string"}},
    "required": ["service"],
    "additionalProperties": False,
}


def build_service() -> tuple[OpsToolService, Ledger, list[str]]:
    calls: list[str] = []
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_metrics",
            description="read",
            input_schema=_SVC,
            risk=ToolRisk(Reversibility.REVERSIBLE, BlastRadius(Scope.CONTAINER, 0)),
            handler=lambda service: (
                calls.append(f"get_metrics:{service}"),
                {"service": service, "memory_limit_bytes": 536870912},
            )[1],
            read_only=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_service",
            description="reversible",
            input_schema=_SVC,
            risk=ToolRisk(Reversibility.REVERSIBLE, BlastRadius(Scope.CONTAINER, 1)),
            handler=lambda service: (
                calls.append(f"restart:{service}"),
                {"restarted": service},
            )[1],
        )
    )
    registry.register(
        ToolSpec(
            name="drop_volume",
            description="irreversible",
            input_schema={
                "type": "object",
                "properties": {"volume": {"type": "string"}},
                "required": ["volume"],
                "additionalProperties": False,
            },
            risk=ToolRisk(Reversibility.IRREVERSIBLE, BlastRadius(Scope.DATA, 1)),
            handler=lambda volume: (calls.append(f"drop:{volume}"), {"dropped": volume})[1],
        )
    )
    registry.validate()

    ledger = Ledger(InMemoryAuditStore())
    service = OpsToolService(
        registry=registry,
        gate=ReversibilityGate(auto_apply_ceiling=4, episode_ceiling=8),
        ledger=ledger,
        tiers={READ_TOKEN: Tier.READ_ONLY, WRITE_TOKEN: Tier.OPERATOR},
    )
    return service, ledger, calls


def call(svc: OpsToolService, token: str, tool: str, **arguments):
    """Positional params are deliberately not named `service` — the tools
    take a `service` argument, and a collision here would shadow it."""
    return svc.call(
        token=token,
        tool=tool,
        arguments=arguments,
        trace_id="trace-1",
        task_id="task-1",
        agent_id="test",
    )


def test_triage_tier_sees_only_read_tools():
    service, _, _ = build_service()
    names = {s.name for s in service.visible_tools(service.tier_for(READ_TOKEN))}
    assert names == {"get_metrics"}


def test_triage_tier_cannot_invoke_a_write_tool_it_can_name():
    """Hiding a tool is not access control; invocation is the boundary."""
    service, _, calls = build_service()
    with pytest.raises(ToolAccessDenied):
        call(service, READ_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    assert calls == []


def test_denied_call_is_audited_even_though_it_never_ran():
    service, ledger, _ = build_service()
    with pytest.raises(ToolAccessDenied):
        call(service, READ_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    rows = ledger._store.all_rows()  # noqa: SLF001 - asserting on the evidence trail
    assert [r.decision for r in rows] == ["denied_by_tier"]


def test_write_without_episode_is_refused():
    service, _, calls = build_service()
    outcome = call(service, WRITE_TOKEN, "restart_service", service="target-api")
    assert outcome.status == "error"
    assert "episode_id is required" in outcome.reason
    assert calls == []


def test_reversible_write_executes_autonomously():
    service, _, calls = build_service()
    outcome = call(
        service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1"
    )
    assert outcome.status == "ok"
    assert calls == ["restart:target-api"]


def test_irreversible_write_is_held_and_never_executes():
    service, _, calls = build_service()
    outcome = call(service, WRITE_TOKEN, "drop_volume", volume="pgdata", episode_id="inc-1")
    assert outcome.status == "approval_required"
    assert outcome.approval_ref
    assert calls == []


def test_approval_releases_exactly_the_refused_call():
    service, _, calls = build_service()
    held = call(service, WRITE_TOKEN, "drop_volume", volume="pgdata", episode_id="inc-1")

    approved = call(
        service,
        WRITE_TOKEN,
        "drop_volume",
        volume="pgdata",
        episode_id="inc-1",
        approval_ref=held.approval_ref,
        approver="arthur",
    )
    assert approved.status == "ok"
    assert calls == ["drop:pgdata"]


def test_approval_does_not_authorise_different_arguments():
    """An approval is consent to an action, not a blank cheque."""
    service, _, calls = build_service()
    held = call(service, WRITE_TOKEN, "drop_volume", volume="pgdata", episode_id="inc-1")

    swapped = call(
        service,
        WRITE_TOKEN,
        "drop_volume",
        volume="production-backups",
        episode_id="inc-1",
        approval_ref=held.approval_ref,
        approver="arthur",
    )
    assert swapped.status == "error"
    assert "does not authorise this call" in swapped.reason
    assert calls == []


def test_approval_requires_an_approver_identity():
    service, _, calls = build_service()
    held = call(service, WRITE_TOKEN, "drop_volume", volume="pgdata", episode_id="inc-1")
    outcome = call(
        service,
        WRITE_TOKEN,
        "drop_volume",
        volume="pgdata",
        episode_id="inc-1",
        approval_ref=held.approval_ref,
    )
    assert outcome.status == "error"
    assert calls == []


def test_episode_budget_stops_a_runaway_agent():
    """Individually permitted restarts, one confused episode."""
    service, _, calls = build_service()
    for _ in range(8):
        call(service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    blocked = call(
        service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1"
    )
    assert blocked.status == "approval_required"
    assert len(calls) == 8


def test_budget_is_per_episode_not_global():
    service, _, _ = build_service()
    for _ in range(8):
        call(service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    fresh = call(
        service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-2"
    )
    assert fresh.status == "ok"


def test_ledger_chain_verifies_after_a_full_episode():
    service, ledger, _ = build_service()
    call(service, WRITE_TOKEN, "get_metrics", service="target-api")
    call(service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    held = call(service, WRITE_TOKEN, "drop_volume", volume="pgdata", episode_id="inc-1")
    call(
        service,
        WRITE_TOKEN,
        "drop_volume",
        volume="pgdata",
        episode_id="inc-1",
        approval_ref=held.approval_ref,
        approver="arthur",
    )

    ok, bad_index = ledger.verify()
    assert ok, f"chain broken at row {bad_index}"


def test_tampering_with_a_row_is_detected():
    service, ledger, _ = build_service()
    call(service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1")
    call(service, WRITE_TOKEN, "restart_service", service="target-db", episode_id="inc-1")

    # Tamper with the store's own list, not the copy `all_rows()` hands out.
    # That the accessor returns a copy is itself the first line of defence:
    # a caller cannot rewrite history through the read path.
    from dataclasses import asdict

    from sovops.audit.ledger import AuditRow

    store = ledger._store  # noqa: SLF001 - simulating an operator with DB access
    store.rows[0] = AuditRow(**{**asdict(store.rows[0]), "decision": "read"})

    ok, bad_index = ledger.verify()
    assert not ok
    assert bad_index == 0


def test_read_accessor_cannot_rewrite_history():
    """`all_rows()` hands out a copy — tampering through it is a no-op."""
    service, ledger, _ = build_service()
    call(service, WRITE_TOKEN, "restart_service", service="target-api", episode_id="inc-1")

    from dataclasses import asdict

    from sovops.audit.ledger import AuditRow

    handed_out = ledger._store.all_rows()  # noqa: SLF001
    handed_out[0] = AuditRow(**{**asdict(handed_out[0]), "decision": "forged"})

    ok, _ = ledger.verify()
    assert ok


def test_write_schema_carries_the_platform_fields():
    """Tool authors write the domain schema; the platform injects governance."""
    service, _, _ = build_service()
    spec = next(s for s in service.visible_tools(Tier.OPERATOR) if s.name == "restart_service")
    schema = service.exposed_schema(spec)
    assert "episode_id" in schema["properties"]
    assert "episode_id" in schema["required"]
    assert "approval_ref" in schema["properties"]


def test_read_schema_is_left_alone():
    """A read has no episode and no approval path; injecting them would lie."""
    service, _, _ = build_service()
    spec = next(s for s in service.visible_tools(Tier.OPERATOR) if s.name == "get_metrics")
    assert service.exposed_schema(spec) == _SVC
