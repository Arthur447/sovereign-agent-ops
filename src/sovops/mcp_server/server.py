"""The ops MCP server — tool transport, privilege tiers, and the gate.

## What this file is responsible for

Three things that must not be separable:

1. **Serving tools** over MCP (Streamable HTTP, so writes travel over an
   authenticated transport — the stdio/no-auth shortcut is only honest
   for a read-only server).
2. **Deciding who may call what.** The bearer token maps to a tier, and
   the tier determines both which tools appear in `tools/list` and which
   may be invoked. Hiding a tool is not access control, so both are
   enforced; listing is a convenience, invocation is the boundary.
3. **Running the reversibility gate before any write executes.**

They live together because separating them creates a path where a tool
runs without a ruling. There is exactly one place a handler is invoked in
this file, and it is downstream of both checks.

## Episodes

Every write call names the incident it belongs to (`episode_id`, injected
into the exposed schema — the domain tools do not know about it). The
server keeps the cumulative blast budget per episode, which is what makes
the cumulative rule enforceable at all: an agent cannot reset its own
budget by forgetting, because the budget does not live in the agent.

A write call with no episode is refused. An action nobody can attribute
to an incident is an action nobody will review.

## Approvals

A refusal returns an `approval_ref`. When a human approves, the agent
re-calls the same tool with that ref and the approver's identity; the
server checks the ref is pending and that the call matches what was
refused — same tool, same arguments — then executes. Re-checking the
arguments matters: an approval is consent to a specific action, and a ref
that authorises whatever the agent sends next is a bypass with extra
steps.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sovops.audit.ledger import Ledger, hash_params
from sovops.mcp_server.registry import ToolRegistry, ToolSpec
from sovops.policy.reversibility import Decision, EpisodeBudget, ReversibilityGate

logger = logging.getLogger(__name__)


class Tier:
    """The two privilege tiers. Names are what land in the audit rows."""

    READ_ONLY = "read_only"
    OPERATOR = "operator"


class ToolAccessDenied(PermissionError):
    """The caller's tier does not include this tool."""


@dataclass
class PendingApproval:
    """A refused call, held so a human decision can release exactly it."""

    ref: str
    tool: str
    params_hash: str
    episode_id: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallOutcome:
    """What the MCP layer returns to the agent.

    `status` is the field the agent branches on. `approval_required` is
    not an error — the system worked exactly as designed — so it is not
    raised as one. Treating a policy refusal as a failure is how agents
    learn to retry around it.
    """

    status: str  # "ok" | "approval_required" | "error"
    result: dict[str, Any] | None = None
    approval_ref: str | None = None
    reason: str | None = None
    decision: str | None = None
    rollback: dict[str, Any] | None = None


class OpsToolService:
    """Tier checks + gate + audit around the registry. Transport-agnostic.

    Deliberately separate from the MCP wiring so the eval harness can
    drive it directly: the harness must exercise the *policy*, and forcing
    every synthetic incident through an HTTP transport would make the
    suite slow and would test the SDK rather than this code.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gate: ReversibilityGate,
        ledger: Ledger,
        tiers: dict[str, str],
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._ledger = ledger
        self._tiers = tiers  # token -> tier
        self._episodes: dict[str, EpisodeBudget] = {}
        self._pending: dict[str, PendingApproval] = {}

    # -- visibility ---------------------------------------------------------

    def tier_for(self, token: str) -> str:
        tier = self._tiers.get(token)
        if tier is None:
            raise ToolAccessDenied("unauthenticated")
        return tier

    def visible_tools(self, tier: str) -> list[ToolSpec]:
        if tier == Tier.OPERATOR:
            return self._registry.all()
        return [s for s in self._registry.all() if s.read_only]

    def exposed_schema(self, spec: ToolSpec) -> dict[str, Any]:
        """The domain schema, plus the fields the platform requires.

        Injected here rather than written into each tool so a tool author
        cannot forget them, and so the domain schemas stay readable as
        descriptions of the operation rather than of the governance.
        """
        if spec.read_only:
            return spec.input_schema

        schema = json.loads(json.dumps(spec.input_schema))
        schema["properties"]["episode_id"] = {
            "type": "string",
            "description": "Incident this action belongs to. Required: actions are attributed.",
        }
        schema["properties"]["approval_ref"] = {
            "type": "string",
            "description": (
                "Approval reference, when re-calling after a human "
                "authorised a refused action."
            ),
        }
        schema["properties"]["approver"] = {
            "type": "string",
            "description": "Identity of the human who approved. Required with approval_ref.",
        }
        schema["required"] = [*schema.get("required", []), "episode_id"]
        return schema

    # -- invocation ---------------------------------------------------------

    def call(
        self,
        *,
        token: str,
        tool: str,
        arguments: dict[str, Any],
        trace_id: str,
        task_id: str,
        agent_id: str,
    ) -> CallOutcome:
        tier = self.tier_for(token)
        spec = self._registry.get(tool)

        if spec not in self.visible_tools(tier):
            # Audited: a call outside the caller's tier is the signal that
            # matters most after an incident, and dropping it because it
            # never executed would lose exactly the interesting event.
            self._ledger.record(
                trace_id=trace_id,
                task_id=task_id,
                agent_id=agent_id,
                tool=tool,
                params=arguments,
                reversibility=str(spec.risk.reversibility),
                blast_weight=spec.risk.blast.weight,
                decision="denied_by_tier",
                reason=f"tier {tier!r} may not call {tool!r}",
            )
            raise ToolAccessDenied(f"tier {tier!r} may not call {tool!r}")

        if spec.read_only:
            return self._execute(
                spec,
                arguments,
                trace_id=trace_id,
                task_id=task_id,
                agent_id=agent_id,
                decision="read",
                reason="read-only tool, no gate applies",
            )

        episode_id = arguments.get("episode_id")
        if not episode_id:
            return CallOutcome(
                status="error",
                reason=(
                    "episode_id is required for write tools; "
                    "unattributed actions are not permitted"
                ),
            )

        approval_ref = arguments.get("approval_ref")
        domain_args = {
            k: v
            for k, v in arguments.items()
            if k not in {"episode_id", "approval_ref", "approver"}
        }
        budget = self._episodes.setdefault(episode_id, self._gate.new_episode())

        # -- approved re-call ------------------------------------------------
        if approval_ref:
            approver = arguments.get("approver")
            pending = self._pending.get(approval_ref)
            if pending is None:
                return CallOutcome(
                    status="error", reason=f"unknown approval_ref {approval_ref!r}"
                )
            if not approver:
                return CallOutcome(
                    status="error", reason="approval_ref requires an approver identity"
                )
            if pending.tool != tool or pending.params_hash != hash_params(domain_args):
                # The approval was for a different action. Refusing here is
                # the difference between consent and a blank cheque.
                self._ledger.record(
                    trace_id=trace_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    tool=tool,
                    params=domain_args,
                    reversibility=str(spec.risk.reversibility),
                    blast_weight=spec.risk.blast.weight,
                    decision="approval_mismatch",
                    reason=(
                        f"approval {approval_ref!r} authorised {pending.tool!r} "
                        "with different arguments"
                    ),
                    approver=approver,
                )
                return CallOutcome(
                    status="error",
                    reason=f"approval {approval_ref!r} does not authorise this call",
                )

            del self._pending[approval_ref]
            budget.charge(tool, spec.risk.blast.weight)
            return self._execute(
                spec,
                domain_args,
                trace_id=trace_id,
                task_id=task_id,
                agent_id=agent_id,
                decision="human_approved",
                reason=f"approved by {approver}: {pending.reason}",
                approver=approver,
            )

        # -- the gate --------------------------------------------------------
        ruling = self._gate.rule(tool=tool, risk=spec.risk, budget=budget)

        if ruling.decision is Decision.REQUIRE_HUMAN:
            ref = f"apr-{uuid.uuid4().hex[:12]}"
            self._pending[ref] = PendingApproval(
                ref=ref,
                tool=tool,
                params_hash=hash_params(domain_args),
                episode_id=episode_id,
                reason=ruling.reason,
                arguments=domain_args,
            )
            self._ledger.record(
                trace_id=trace_id,
                task_id=task_id,
                agent_id=agent_id,
                tool=tool,
                params=domain_args,
                reversibility=str(spec.risk.reversibility),
                blast_weight=spec.risk.blast.weight,
                decision=str(ruling.decision),
                reason=ruling.reason,
                rollback_ref=ref,
            )
            return CallOutcome(
                status="approval_required",
                approval_ref=ref,
                reason=ruling.reason,
                decision=str(ruling.decision),
            )

        rollback_plan: dict[str, Any] | None = None
        if ruling.decision is Decision.APPLY_WITH_ROLLBACK:
            rollback_plan = self._build_rollback(spec, domain_args, ruling.rollback_tool)

        budget.charge(tool, spec.risk.blast.weight)
        outcome = self._execute(
            spec,
            domain_args,
            trace_id=trace_id,
            task_id=task_id,
            agent_id=agent_id,
            decision=str(ruling.decision),
            reason=ruling.reason,
            rollback_ref=ruling.rollback_tool,
        )
        outcome.decision = str(ruling.decision)
        outcome.rollback = rollback_plan
        return outcome

    def _build_rollback(
        self, spec: ToolSpec, arguments: dict[str, Any], inverse: str | None
    ) -> dict[str, Any]:
        """Capture the inverse call *before* the forward one runs.

        Reading current state after the change has already happened would
        record the new value as the thing to restore — a rollback plan
        that restores the damage.
        """
        current = {}
        if inverse and "service" in arguments:
            try:
                current = self._registry.get("get_metrics").handler(arguments["service"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not capture pre-state for rollback: %s", exc)
        return {
            "tool": inverse,
            "arguments": {
                "service": arguments.get("service"),
                "limit_mb": int(current.get("memory_limit_bytes", 0) / 1_048_576) or None,
            },
            "captured_from": current or None,
        }

    def _execute(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        *,
        trace_id: str,
        task_id: str,
        agent_id: str,
        decision: str,
        reason: str,
        approver: str | None = None,
        rollback_ref: str | None = None,
    ) -> CallOutcome:
        try:
            result = spec.handler(**arguments)
            outcome_note = "ok"
        except Exception as exc:  # noqa: BLE001
            result, outcome_note = None, f"error: {exc}"

        self._ledger.record(
            trace_id=trace_id,
            task_id=task_id,
            agent_id=agent_id,
            tool=spec.name,
            params=arguments,
            reversibility=str(spec.risk.reversibility),
            blast_weight=spec.risk.blast.weight,
            decision=decision,
            reason=reason,
            approver=approver,
            rollback_ref=rollback_ref,
            outcome=outcome_note,
        )

        if result is None:
            return CallOutcome(status="error", reason=outcome_note, decision=decision)
        return CallOutcome(status="ok", result=result, decision=decision)

    # -- introspection for the demo and the eval report ---------------------

    def episode_budget(self, episode_id: str) -> EpisodeBudget | None:
        return self._episodes.get(episode_id)

    def pending_approvals(self) -> list[PendingApproval]:
        return list(self._pending.values())
