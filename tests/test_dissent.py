"""The action is pinned; disagreement is recorded, not obeyed.

Found by the live eval suite: `qwen3:1.7b` answered `scale_memory` to a
config drift, which addresses nothing. Validation allowed it because it
only checked the allowlist — so a 1.7B could override a deterministic
candidate, which is the non-determinism that removing parameter
generation was supposed to eliminate.
"""

from __future__ import annotations

import pytest

from sovops.agents.remediation.agent import PlanRejected, _validate_plan


def test_model_cannot_override_the_deterministic_candidate():
    plan = _validate_plan(
        {"tool": "scale_memory", "service": "target-api", "rationale": "because"},
        expected_service="target-api",
        candidate="restart_service",
    )
    assert plan["tool"] == "restart_service"


def test_the_disagreement_is_recorded():
    """A model wanting a different action is a signal worth seeing — it is
    just not a signal worth acting on unreviewed."""
    plan = _validate_plan(
        {"tool": "scale_memory", "service": "target-api", "rationale": "because"},
        expected_service="target-api",
        candidate="restart_service",
    )
    assert "scale_memory" in plan["dissent"]
    assert "restart_service" in plan["dissent"]


def test_agreement_records_no_dissent():
    plan = _validate_plan(
        {"tool": "restart_service", "service": "target-api", "rationale": "because"},
        expected_service="target-api",
        candidate="restart_service",
    )
    assert "dissent" not in plan


def test_a_tool_outside_the_allowlist_is_still_rejected_outright():
    """Pinning is not a licence to accept anything: an out-of-allowlist tool
    means the model is answering a different question, and the plan is
    rejected rather than silently corrected."""
    with pytest.raises(PlanRejected, match="not in this agent's allowlist"):
        _validate_plan(
            {"tool": "rm_rf", "service": "target-api", "rationale": "because"},
            expected_service="target-api",
            candidate="restart_service",
        )
