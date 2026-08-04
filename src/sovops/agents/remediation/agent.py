"""The remediation agent — holds the write credential, owns the action.

## Its whole job in one sentence

Turn a diagnosis into at most one tool call, and turn the tool server's
refusal into a protocol state its caller can act on.

It does not diagnose. It receives a `FailureMode` and acts on it. That
separation is not tidiness: the triage agent holds a read-only MCP token
and this one holds an operator token, so a diagnostic path that has been
talked into something by a hostile log line still cannot reach a write.
The A2A boundary is where the credential changes.

## Why the plan is model-written but code-validated

Which tool addresses which failure mode is close to a lookup table, and a
lookup table would be more predictable. But the *parameters* are not — how
much memory to grant a service that just OOMed depends on what it was
using, and that is a judgement.

So the model proposes `(tool, arguments, rationale)` under a schema whose
tool field is an enum of what this agent may call, and `_validate_plan`
rejects anything outside the allowlist before it travels. The model can
be wrong; it cannot be wrong in a way that reaches an unlisted tool.

## The AUTH_REQUIRED mapping

When the gate refuses, this agent does not retry, does not pick a smaller
action, and does not report failure. It moves its task to
`TASK_STATE_AUTH_REQUIRED`, puts the approval reference and the gate's
reason in the status message, and stops. A human answering later sends a
second message on the same task id, and `handle` resumes from the pending
approval. That is the entire HITL mechanism — no approval service, no
side channel, one paused task.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sovops.a2a.types import (
    ActorContext,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from sovops.agents.triage.rules import FailureMode
from sovops.gateway.backends import extract_json
from sovops.gateway.base import LlmGateway, TaskClass
from sovops.mcp_client import MCPClient

logger = logging.getLogger(__name__)

AGENT_ID = "remediation-agent"

# What this agent may call. The enum in the plan schema is built from this
# list, so widening the agent's reach means editing one line that a
# reviewer sees, not discovering it in a prompt.
ALLOWED_TOOLS = ("restart_service", "scale_memory", "drop_volume")

# The candidate action per failure mode. Not the final answer — the model
# still chooses arguments — but it bounds the search so a 1.7B is not
# asked to invent an ops playbook from scratch.
CANDIDATE_TOOL: dict[str, str] = {
    FailureMode.MEMORY_EXHAUSTION: "scale_memory",
    FailureMode.CRASH_LOOP: "restart_service",
    FailureMode.LATENCY_DEGRADATION: "restart_service",
    FailureMode.CONFIG_DRIFT: "restart_service",
    # No tool in this registry frees disk without destroying data, so the
    # honest candidate is the irreversible one — and the gate will refuse
    # it. That is the correct outcome, not a gap: the alternative is an
    # agent that silently does nothing about a filling disk.
    FailureMode.DISK_PRESSURE: "drop_volume",
}

# The model names the action and justifies it. It does NOT name the
# parameters.
#
# That is a change made after measuring, and it is the same move as the
# deterministic classifier in triage. The first live run had `qwen3:1.7b`
# fail plan validation on 3 of 5 attempts, and — more tellingly — copy the
# prompt's `<service>-data` placeholder into the plan *verbatim* as a
# volume name. A garbage volume name that passes a non-empty check is
# worse than a rejected plan: it reaches a human for approval looking
# legitimate.
#
# So parameters are derived in code. "Give the service headroom over
# current usage" is arithmetic, and "which volume belongs to target-db" is
# a lookup. Neither is judgement, and asking a 1.7B for them buys
# non-determinism for nothing.
#
# What the model still does is real: it confirms the action against the
# evidence, and it writes the `rationale` a human reads at 3am when
# deciding whether to approve. That text is the whole value of the call.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": list(ALLOWED_TOOLS)},
        "service": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["tool", "service", "rationale"],
    "additionalProperties": False,
}

# Which volume belongs to which service. An explicit map, not a naming
# convention: a convention silently produces a plausible name for a
# service that has no volume, and the first time that matters is when a
# human approves the deletion of something that does not exist — or worse,
# of something that does and should not have been reachable.
MANAGED_VOLUMES: dict[str, str] = {
    "target-db": "sovops_pgdata",
}

# Headroom multiplier for `scale_memory`, and the clamp the tool enforces.
MEMORY_HEADROOM_FACTOR = 2
MEMORY_MIN_MB, MEMORY_MAX_MB = 64, 4096
MEMORY_DEFAULT_MB = 512

PLAN_SYSTEM = (
    "You are an SRE remediation planner. Given a diagnosed failure mode and "
    "current service metrics, confirm ONE action from the allowed list and "
    "explain why it addresses the diagnosed cause. A human may read your "
    "rationale before authorising the action, so make it specific and cite "
    "the evidence. Do not invent parameters — they are computed for you. "
    "Reply only with JSON matching the schema. /no_think"
)


def build_card(url: str) -> AgentCard:
    return AgentCard(
        name="remediation-agent",
        description=(
            "Applies at most one remediation per incident, under a reversibility "
            "gate enforced by the tool server. Escalates to a human whenever the "
            "action leaves the reversible perimeter."
        ),
        version="0.1.0",
        url=url,
        skills=[
            AgentSkill(
                id="remediate",
                name="Remediate a diagnosed incident",
                description=(
                    "Takes a failure mode and service metrics, applies one bounded "
                    "action or pauses for human authorisation."
                ),
                tags=["ops", "remediation", "hitl"],
            )
        ],
    )


class PlanRejected(RuntimeError):
    """The proposed plan did not survive validation. Never reaches a tool."""


def _validate_plan(
    plan: dict[str, Any], *, expected_service: str, candidate: str
) -> dict[str, Any]:
    """Exact checks in code. The model's job was judgement, not compliance.

    The action is **pinned to the deterministic candidate**. Letting the
    model override it would reintroduce exactly the non-determinism that
    removing parameter generation was meant to eliminate — and it did:
    the live suite caught `qwen3:1.7b` answering `scale_memory` to a
    config drift, which addresses nothing.

    A disagreement is not discarded, though. It is recorded as `dissent`
    and travels to the audit row and to the human's approval payload.
    The model wanting a different action is a signal worth seeing; it is
    just not a signal worth acting on unreviewed.
    """
    tool = plan.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise PlanRejected(f"tool {tool!r} is not in this agent's allowlist {ALLOWED_TOOLS}")

    if tool != candidate:
        plan["dissent"] = (
            f"model proposed {tool!r} against the indicated action {candidate!r}; "
            f"the indicated action stands"
        )
        plan["tool"] = candidate

    service = plan.get("service")
    if service != expected_service:
        # A plan that wanders to a different service than the one the
        # incident is about is the single most dangerous failure here, and
        # it is cheap to catch.
        raise PlanRejected(
            f"plan targets {service!r} but the incident is about {expected_service!r}"
        )

    if not str(plan.get("rationale", "")).strip():
        # The rationale is the only part of the plan the model actually
        # produces, and it is what a human reads before authorising. An
        # empty one means the call bought nothing.
        raise PlanRejected(f"{tool} plan carries no rationale")

    return plan


def derive_arguments(tool: str, service: str, signals: dict[str, Any]) -> dict[str, Any]:
    """Compute the tool's parameters. No model involved.

    Raises `PlanRejected` rather than guessing when the parameter cannot
    be derived — a `drop_volume` on a service with no registered volume
    is not a call to make up a name for.
    """
    if tool == "restart_service":
        return {"service": service}

    if tool == "scale_memory":
        current_bytes = int(signals.get("memory_limit_bytes") or 0)
        current_mb = current_bytes // 1_048_576 if current_bytes else MEMORY_DEFAULT_MB
        target = max(MEMORY_MIN_MB, min(current_mb * MEMORY_HEADROOM_FACTOR, MEMORY_MAX_MB))
        return {"service": service, "limit_mb": target}

    if tool == "drop_volume":
        volume = MANAGED_VOLUMES.get(service)
        if not volume:
            raise PlanRejected(
                f"no managed volume registered for {service!r}; "
                "refusing to name one that may not exist"
            )
        return {"service": service, "volume": volume}

    raise PlanRejected(f"no argument derivation for tool {tool!r}")


class RemediationAgent:
    """A2A handler. One instance per process."""

    def __init__(self, *, tools: MCPClient, gateway: LlmGateway) -> None:
        # The operator credential is not held here: it lives inside the
        # client, which is the only handle this agent has to the tool
        # server. An agent cannot present a credential it was never given.
        self._tools = tools
        self._gateway = gateway

    async def handle(self, task: Task, message: Message, actor: ActorContext) -> None:
        payload = message.first_data()

        # A message carrying an approval is a human answering a pause.
        if "approval_ref" in payload:
            return self._resume(task, payload, actor)

        failure_mode = payload.get("failure_mode", FailureMode.UNKNOWN)
        service = payload.get("service", "")
        signals = payload.get("signals", {})
        episode_id = payload.get("episode_id") or task.id

        if failure_mode == FailureMode.HEALTHY:
            task.status = TaskStatus(
                TaskState.COMPLETED,
                message=_agent_message(
                    {"action": "none", "reason": "service is healthy, nothing to remediate"}
                ),
            )
            return

        if failure_mode == FailureMode.UNKNOWN:
            # Defence in depth: triage already refuses to delegate an
            # unclassified incident, but this agent must not treat "no
            # diagnosis" as "nothing wrong" whatever its caller does.
            # Closing here would retire an open incident nobody looked at.
            task.status = TaskStatus(
                TaskState.INPUT_REQUIRED,
                message=_agent_message(
                    {
                        "action": "none",
                        "awaiting": "human_diagnosis",
                        "reason": (
                            "delegated without a diagnosis; no candidate action can "
                            "be derived from an unclassified incident"
                        ),
                    }
                ),
            )
            return

        candidate = CANDIDATE_TOOL.get(failure_mode)
        if candidate is None:
            task.status = TaskStatus(
                TaskState.REJECTED,
                message=_agent_message({"reason": f"no candidate action for {failure_mode!r}"}),
            )
            return

        try:
            plan = self._plan(failure_mode, service, signals, candidate)
        except Exception as exc:  # noqa: BLE001 - PlanRejected and backend errors alike
            task.status = TaskStatus(
                TaskState.FAILED, message=_agent_message({"error": f"planning failed: {exc}"})
            )
            return

        task.metadata["plan"] = plan
        task.metadata["episode_id"] = episode_id
        self._apply(task, plan, episode_id=episode_id, actor=actor)

    # -- planning -----------------------------------------------------------

    def _plan(
        self, failure_mode: str, service: str, signals: dict, candidate: str
    ) -> dict[str, Any]:
        derived = derive_arguments(candidate, service, signals)
        prompt = (
            f"Diagnosed failure mode: {failure_mode}\n"
            f"Service: {service}\n"
            f"Current metrics: {json.dumps(signals, default=str)}\n"
            f"The indicated action is `{candidate}` with parameters "
            f"{json.dumps({k: v for k, v in derived.items() if k != 'service'})}, "
            f"already computed.\n"
            f"Confirm the action and explain, citing the metrics above, why it "
            f"addresses the diagnosed cause."
        )
        completion = self._gateway.complete(
            task_class=TaskClass.PLAN,
            system=PLAN_SYSTEM,
            user=prompt,
            json_schema=PLAN_SCHEMA,
        )
        plan = extract_json(completion.text)
        plan.setdefault("service", service)
        plan.setdefault("tool", candidate)
        validated = _validate_plan(plan, expected_service=service, candidate=candidate)

        # Re-derived rather than reusing `derived`: cheap, and it keeps the
        # arguments provably tied to the tool that survived validation.
        validated["arguments"] = derive_arguments(validated["tool"], service, signals)
        validated["_model"] = completion.model
        validated["_sovereign"] = completion.sovereign
        validated["_latency_ms"] = completion.latency_ms
        return validated

    # -- application --------------------------------------------------------

    def _apply(
        self, task: Task, plan: dict[str, Any], *, episode_id: str, actor: ActorContext
    ) -> None:
        arguments = dict(plan["arguments"]) | {"episode_id": episode_id}

        outcome = self._tools.call(
            tool=plan["tool"], arguments=arguments, task_id=task.id
        )

        if outcome.status == "approval_required":
            # The pause. Everything the human needs is in this message.
            task.status = TaskStatus(
                TaskState.AUTH_REQUIRED,
                message=_agent_message(
                    {
                        "awaiting": "human_authorisation",
                        "approval_ref": outcome.approval_ref,
                        "reason": outcome.reason,
                        "proposed_tool": plan["tool"],
                        "proposed_arguments": {
                            k: v for k, v in arguments.items() if k != "episode_id"
                        },
                        "rationale": plan.get("rationale", ""),
                        # Surfaced to the human deciding: "the model wanted
                        # something else" is exactly the kind of context an
                        # approver should have before saying yes.
                        "dissent": plan.get("dissent"),
                        "requested_by": actor.actor,
                    }
                ),
            )
            return

        if outcome.status == "error":
            task.status = TaskStatus(
                TaskState.FAILED, message=_agent_message({"error": outcome.reason})
            )
            return

        task.status = TaskStatus(
            TaskState.COMPLETED,
            message=_agent_message(
                {
                    "applied": plan["tool"],
                    "arguments": {k: v for k, v in arguments.items() if k != "episode_id"},
                    "decision": outcome.decision,
                    "result": outcome.result,
                    "rollback": outcome.rollback,
                    "rationale": plan.get("rationale", ""),
                    "dissent": plan.get("dissent"),
                }
            ),
        )

    def _resume(self, task: Task, payload: dict[str, Any], actor: ActorContext) -> None:
        """A human answered. Approve and apply, or decline and close."""
        plan = task.metadata.get("plan")
        episode_id = task.metadata.get("episode_id", task.id)
        if not plan:
            task.status = TaskStatus(
                TaskState.FAILED,
                message=_agent_message({"error": "no plan on this task to approve"}),
            )
            return

        if not payload.get("approved", False):
            task.status = TaskStatus(
                TaskState.REJECTED,
                message=_agent_message(
                    {
                        "declined_by": payload.get("approver", actor.actor),
                        "reason": payload.get("reason", "declined by operator"),
                    }
                ),
            )
            return

        # The arguments stored on the task at plan time, not recomputed:
        # the human approved *these*, and re-deriving could produce a
        # different value if the metrics moved between the pause and the
        # approval. The tool server checks the hash matches anyway, so a
        # drift would be rejected — but failing there would look like a
        # bug rather than the safety property it is.
        arguments = dict(plan["arguments"]) | {
            "episode_id": episode_id,
            "approval_ref": payload["approval_ref"],
            "approver": payload.get("approver", actor.actor),
        }

        outcome = self._tools.call(
            tool=plan["tool"], arguments=arguments, task_id=task.id
        )

        if outcome.status != "ok":
            task.status = TaskStatus(
                TaskState.FAILED, message=_agent_message({"error": outcome.reason})
            )
            return

        task.status = TaskStatus(
            TaskState.COMPLETED,
            message=_agent_message(
                {
                    "applied": plan["tool"],
                    "decision": "human_approved",
                    "approver": arguments["approver"],
                    "result": outcome.result,
                }
            ),
        )


def _agent_message(data: dict[str, Any]) -> Message:
    return Message(role=Role.AGENT, parts=[Part(data=data)])
