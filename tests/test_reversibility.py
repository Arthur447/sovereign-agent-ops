"""The gate's behaviour, stated as tests.

These are the rules an auditor would ask about, written so that changing
one is a visible diff rather than a quiet regression.
"""

from __future__ import annotations

import pytest

from sovops.policy.reversibility import (
    BlastRadius,
    Decision,
    Reversibility,
    ReversibilityGate,
    Scope,
    ToolRisk,
)


def risk(rev: Reversibility, scope: Scope, n: int = 1, inverse: str | None = None) -> ToolRisk:
    return ToolRisk(rev, BlastRadius(scope, n), rollback_tool=inverse)


def test_reversible_container_action_is_autonomous():
    gate = ReversibilityGate()
    ruling = gate.rule(
        tool="restart_service",
        risk=risk(Reversibility.REVERSIBLE, Scope.CONTAINER),
        budget=gate.new_episode(),
    )
    assert ruling.decision is Decision.AUTO_APPLY


def test_compensable_action_carries_its_inverse():
    gate = ReversibilityGate()
    ruling = gate.rule(
        tool="scale_memory",
        risk=risk(Reversibility.COMPENSABLE, Scope.SERVICE, inverse="restore_memory"),
        budget=gate.new_episode(),
    )
    assert ruling.decision is Decision.APPLY_WITH_ROLLBACK
    assert ruling.rollback_tool == "restore_memory"


def test_irreversible_always_escalates():
    gate = ReversibilityGate(auto_apply_ceiling=10_000, episode_ceiling=10_000)
    ruling = gate.rule(
        tool="drop_volume",
        risk=risk(Reversibility.IRREVERSIBLE, Scope.DATA),
        budget=gate.new_episode(),
    )
    assert ruling.decision is Decision.REQUIRE_HUMAN
    # Even with the ceilings raised absurdly high, no threshold buys this.
    assert "cannot be undone" in ruling.reason


def test_wide_scope_escalates_even_when_reversible():
    """Restarting the cluster is reversible and still not the agent's call."""
    gate = ReversibilityGate(auto_apply_ceiling=4)
    ruling = gate.rule(
        tool="restart_service",
        risk=risk(Reversibility.REVERSIBLE, Scope.CLUSTER),
        budget=gate.new_episode(),
    )
    assert ruling.decision is Decision.REQUIRE_HUMAN


def test_episode_budget_accumulates_across_individually_safe_actions():
    """The failure mode this exists for: many small steps, one large episode."""
    gate = ReversibilityGate(auto_apply_ceiling=4, episode_ceiling=3)
    budget = gate.new_episode()
    container = risk(Reversibility.REVERSIBLE, Scope.CONTAINER)

    for _ in range(3):
        ruling = gate.rule(tool="restart_service", risk=container, budget=budget)
        assert ruling.decision is Decision.AUTO_APPLY
        budget.charge("restart_service", container.blast.weight)

    fourth = gate.rule(tool="restart_service", risk=container, budget=budget)
    assert fourth.decision is Decision.REQUIRE_HUMAN
    assert "episode blast budget exhausted" in fourth.reason


def test_compensable_without_inverse_is_rejected_at_declaration():
    """A compensable tool with no inverse is an irreversible one, mislabelled."""
    with pytest.raises(ValueError, match="must declare `rollback_tool`"):
        ToolRisk(Reversibility.COMPENSABLE, BlastRadius(Scope.SERVICE, 1))


def test_inverse_on_a_non_compensable_tool_is_rejected():
    with pytest.raises(ValueError, match="meaningless"):
        ToolRisk(
            Reversibility.REVERSIBLE,
            BlastRadius(Scope.CONTAINER, 1),
            rollback_tool="restart_service",
        )


def test_every_ruling_carries_a_reason():
    """An escalation that cannot explain itself trains reflexive approval."""
    gate = ReversibilityGate()
    budget = gate.new_episode()
    for rev, scope, inverse in [
        (Reversibility.REVERSIBLE, Scope.CONTAINER, None),
        (Reversibility.COMPENSABLE, Scope.SERVICE, "restore_memory"),
        (Reversibility.IRREVERSIBLE, Scope.DATA, None),
    ]:
        ruling = gate.rule(tool="t", risk=risk(rev, scope, inverse=inverse), budget=budget)
        assert ruling.reason.strip()
