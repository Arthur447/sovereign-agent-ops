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
from sovops.mcp_server.server import OpsToolService

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

# The union of every allowed tool's arguments. A field missing here is a
# tool that can be *chosen* but never *parameterised* — the plan travels,
# the call fails on a missing argument, and the failure surfaces as a
# broken agent rather than as the schema gap it is. Found exactly that way
# by the demo: `drop_volume` had no `volume`.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": list(ALLOWED_TOOLS)},
        "service": {"type": "string"},
        "limit_mb": {"type": "integer", "minimum": 64, "maximum": 4096},
        "volume": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["tool", "service", "rationale"],
    "additionalProperties": False,
}

# Which arguments each tool actually needs. Validated before the plan
# leaves the agent, so a call is never dispatched with a missing one.
REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "restart_service": ("service",),
    "scale_memory": ("service", "limit_mb"),
    "drop_volume": ("volume",),
}

PLAN_SYSTEM = (
    "You are an SRE remediation planner. Given a diagnosed failure mode and "
    "current service metrics, choose ONE action and its parameters. "
    "Reply only with JSON matching the schema. Be conservative: prefer the "
    "smallest action that addresses the diagnosed cause. /no_think"
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


def _validate_plan(plan: dict[str, Any], *, expected_service: str) -> dict[str, Any]:
    """Exact checks in code. The model's job was judgement, not compliance."""
    tool = plan.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise PlanRejected(f"tool {tool!r} is not in this agent's allowlist {ALLOWED_TOOLS}")

    service = plan.get("service")
    if service != expected_service:
        # A plan that wanders to a different service than the one the
        # incident is about is the single most dangerous failure here, and
        # it is cheap to catch.
        raise PlanRejected(
            f"plan targets {service!r} but the incident is about {expected_service!r}"
        )

    missing = [arg for arg in REQUIRED_ARGS.get(tool, ()) if not plan.get(arg)]
    if missing:
        # Caught here rather than at dispatch: a plan that cannot be
        # executed should fail as an invalid plan, not as a tool error
        # three layers down where the cause is no longer obvious.
        raise PlanRejected(f"{tool} requires {missing} — the plan does not name them")

    if tool == "scale_memory":
        limit = plan.get("limit_mb")
        if not isinstance(limit, int) or not (64 <= limit <= 4096):
            raise PlanRejected(f"scale_memory needs a limit_mb in [64, 4096], got {limit!r}")

    return plan


class RemediationAgent:
    """A2A handler. One instance per process."""

    def __init__(self, *, ops: OpsToolService, gateway: LlmGateway, token: str) -> None:
        self._ops = ops
        self._gateway = gateway
        self._token = token  # operator tier

    async def handle(self, task: Task, message: Message, actor: ActorContext) -> None:
        payload = message.first_data()

        # A message carrying an approval is a human answering a pause.
        if "approval_ref" in payload:
            return self._resume(task, payload, actor)

        failure_mode = payload.get("failure_mode", FailureMode.UNKNOWN)
        service = payload.get("service", "")
        signals = payload.get("signals", {})
        episode_id = payload.get("episode_id") or task.id

        if failure_mode in (FailureMode.HEALTHY, FailureMode.UNKNOWN):
            task.status = TaskStatus(
                TaskState.COMPLETED,
                message=_agent_message(
                    {"action": "none", "reason": f"nothing to remediate for {failure_mode!r}"}
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
        prompt = (
            f"Diagnosed failure mode: {failure_mode}\n"
            f"Service: {service}\n"
            f"Current metrics: {json.dumps(signals, default=str)}\n"
            f"The indicated action for this failure mode is `{candidate}`.\n"
            f"Choose the parameters. If the action is scale_memory, pick a new "
            f"limit_mb that gives headroom over current usage without being wasteful. "
            f"If the action is drop_volume, name the volume as `<service>-data`; it "
            f"will not execute without a human authorising this exact volume."
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
        plan["_model"] = completion.model
        plan["_sovereign"] = completion.sovereign
        plan["_latency_ms"] = completion.latency_ms
        return _validate_plan(plan, expected_service=service)

    # -- application --------------------------------------------------------

    def _apply(
        self, task: Task, plan: dict[str, Any], *, episode_id: str, actor: ActorContext
    ) -> None:
        arguments = {
            k: v
            for k, v in plan.items()
            if k in {"service", "limit_mb", "volume"} and v is not None
        }
        arguments["episode_id"] = episode_id

        outcome = self._ops.call(
            token=self._token,
            tool=plan["tool"],
            arguments=arguments,
            trace_id=task.metadata.get("trace_id", task.id),
            task_id=task.id,
            agent_id=AGENT_ID,
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

        arguments = {
            k: v
            for k, v in plan.items()
            if k in {"service", "limit_mb", "volume"} and v is not None
        }
        arguments |= {
            "episode_id": episode_id,
            "approval_ref": payload["approval_ref"],
            "approver": payload.get("approver", actor.actor),
        }

        outcome = self._ops.call(
            token=self._token,
            tool=plan["tool"],
            arguments=arguments,
            trace_id=task.metadata.get("trace_id", task.id),
            task_id=task.id,
            agent_id=AGENT_ID,
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
