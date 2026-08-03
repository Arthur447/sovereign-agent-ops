"""The whole loop, in-process: incident → triage → A2A → remediation → MCP.

The A2A hop is real — a JSON-RPC request over HTTP against the peer's
actual ASGI app, through `httpx.ASGITransport`. What is faked is the
model backend, because a 70-second CPU inference per case would make this
suite unusable and it is not what these assertions are about.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from sovops.a2a.client import A2AClient
from sovops.a2a.server import build_a2a_app
from sovops.a2a.types import ActorContext, TaskState
from sovops.agents.remediation.agent import RemediationAgent
from sovops.agents.remediation.agent import build_card as remediation_card
from sovops.agents.triage.agent import TriageAgent
from sovops.audit.ledger import InMemoryAuditStore, Ledger
from sovops.gateway.base import ClassRoute, LlmGateway, ModelSpec, RoutingPolicy, TaskClass
from sovops.mcp_server.registry import ToolRegistry, ToolSpec
from sovops.mcp_server.server import OpsToolService, Tier
from sovops.policy.reversibility import (
    BlastRadius,
    Reversibility,
    ReversibilityGate,
    Scope,
    ToolRisk,
)

READ_TOKEN = "tok-triage"
WRITE_TOKEN = "tok-remediation"
A2A_TOKEN = "tok-a2a"


class FakeBackend:
    """Returns whatever the test queued. Records what it was asked."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, *, model, system, user, max_tokens, json_schema=None):
        self.prompts.append(user)
        payload = self._responses.pop(0) if self._responses else {}
        return json.dumps(payload), 100, 20


def build_stack(*, plan_responses: list[dict], applied: list[str]):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_metrics",
            description="read",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            risk=ToolRisk(Reversibility.REVERSIBLE, BlastRadius(Scope.CONTAINER, 0)),
            handler=lambda service: {"service": service, "memory_limit_bytes": 536870912},
            read_only=True,
        )
    )
    registry.register(
        ToolSpec(
            name="restart_service",
            description="reversible",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            risk=ToolRisk(Reversibility.REVERSIBLE, BlastRadius(Scope.CONTAINER, 1)),
            handler=lambda service: (
                applied.append(f"restart:{service}"),
                {"restarted": service},
            )[1],
        )
    )
    registry.register(
        ToolSpec(
            name="restore_memory",
            description="inverse",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}, "limit_mb": {"type": "integer"}},
                "required": ["service", "limit_mb"],
                "additionalProperties": False,
            },
            risk=ToolRisk(
                Reversibility.COMPENSABLE,
                BlastRadius(Scope.SERVICE, 1),
                rollback_tool="scale_memory",
            ),
            handler=lambda service, limit_mb: {"service": service, "limit_mb": limit_mb},
        )
    )
    registry.register(
        ToolSpec(
            name="scale_memory",
            description="compensable",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}, "limit_mb": {"type": "integer"}},
                "required": ["service", "limit_mb"],
                "additionalProperties": False,
            },
            risk=ToolRisk(
                Reversibility.COMPENSABLE,
                BlastRadius(Scope.SERVICE, 1),
                rollback_tool="restore_memory",
            ),
            handler=lambda service, limit_mb: (
                applied.append(f"scale:{service}:{limit_mb}"),
                {"service": service, "limit_mb": limit_mb},
            )[1],
        )
    )
    registry.register(
        ToolSpec(
            name="drop_volume",
            description="irreversible",
            input_schema={
                "type": "object",
                "properties": {"volume": {"type": "string"}, "service": {"type": "string"}},
                "required": ["volume"],
                "additionalProperties": False,
            },
            risk=ToolRisk(Reversibility.IRREVERSIBLE, BlastRadius(Scope.DATA, 1)),
            handler=lambda volume, service=None: (
                applied.append(f"drop:{volume}"),
                {"dropped": volume},
            )[1],
        )
    )
    registry.validate()

    ledger = Ledger(InMemoryAuditStore())
    ops = OpsToolService(
        registry=registry,
        gate=ReversibilityGate(auto_apply_ceiling=4, episode_ceiling=16),
        ledger=ledger,
        tiers={READ_TOKEN: Tier.READ_ONLY, WRITE_TOKEN: Tier.OPERATOR},
    )

    local = ModelSpec(
        id="fake-local",
        backend="fake",
        sovereign=True,
        eur_per_mtok_in=0.0,
        eur_per_mtok_out=0.0,
    )
    backend = FakeBackend(plan_responses)
    gateway = LlmGateway(
        policy=RoutingPolicy(
            {
                TaskClass.TRIAGE: ClassRoute(primary=local),
                TaskClass.PLAN: ClassRoute(primary=local),
                TaskClass.JUDGE: ClassRoute(primary=local),
            }
        ),
        backends={"fake": backend},
    )

    remediation = RemediationAgent(ops=ops, gateway=gateway, token=WRITE_TOKEN)
    remediation_app = build_a2a_app(
        card=remediation_card("http://remediation.test"),
        handler=remediation.handle,
        tokens={A2A_TOKEN: "triage-agent"},
    )
    client = A2AClient(
        base_url="http://remediation.test",
        token=A2A_TOKEN,
        actor=ActorContext(actor="triage-agent", on_behalf_of="alertmanager"),
        http_client=TestClient(remediation_app, base_url="http://remediation.test"),
    )
    triage = TriageAgent(ops=ops, gateway=gateway, token=READ_TOKEN, remediation=client)
    triage_app = build_a2a_app(
        card=remediation_card("http://triage.test"),
        handler=triage.handle,
        tokens={A2A_TOKEN: "alertmanager"},
    )
    return triage_app, ledger, backend, ops


def post_incident(app, payload: dict) -> dict:
    resp = TestClient(app, base_url="http://triage.test").post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "m1",
                    "role": "ROLE_USER",
                    "parts": [{"data": payload}],
                }
            },
        },
        headers={"Authorization": f"Bearer {A2A_TOKEN}", "X-Sovops-Actor": "alertmanager"},
    )
    body = resp.json()
    assert "error" not in body, body
    return body["result"]


def test_agent_card_is_served_unauthenticated():
    app, *_ = build_stack(plan_responses=[], applied=[])
    card = TestClient(app).get("/.well-known/agent-card.json").json()
    assert card["protocolVersion"] == "1.0"
    assert card["interfaces"][0]["transport"] == "JSONRPC"
    assert "bearer" in card["securitySchemes"]


def test_unauthenticated_rpc_is_rejected():
    app, *_ = build_stack(plan_responses=[], applied=[])
    resp = TestClient(app).post(
        "/", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/list", "params": {}}
    )
    assert resp.status_code == 401


def test_crash_loop_is_remediated_autonomously():
    """Reversible action, within budget: no human involved."""
    applied: list[str] = []
    app, ledger, backend, _ = build_stack(
        plan_responses=[
            {
                "tool": "restart_service",
                "service": "target-worker",
                "rationale": "restart clears the failed process",
            }
        ],
        applied=applied,
    )
    task = post_incident(
        app,
        {
            "service": "target-worker",
            "signals": {"restart_count": 9, "exit_code": 1, "memory_percent": "20%"},
        },
    )

    assert task["status"]["state"] == str(TaskState.COMPLETED)
    assert applied == ["restart:target-worker"]
    ok, _ = ledger.verify()
    assert ok


def test_crash_loop_diagnosis_never_reaches_the_model():
    """The rules settle it; a model call here would be waste and risk."""
    app, _, backend, _ = build_stack(
        plan_responses=[
            {"tool": "restart_service", "service": "target-worker", "rationale": "restart"}
        ],
        applied=[],
    )
    post_incident(
        app,
        {
            "service": "target-worker",
            "signals": {"restart_count": 9, "exit_code": 1, "memory_percent": "20%"},
        },
    )
    # Exactly one model call: the remediation plan. Triage used rules only.
    assert len(backend.prompts) == 1
    assert "Diagnosed failure mode" in backend.prompts[0]


def test_disk_pressure_pauses_for_a_human():
    """The demo's centrepiece: the only available action is irreversible."""
    applied: list[str] = []
    app, ledger, _, ops = build_stack(
        plan_responses=[
            {
                "tool": "drop_volume",
                "service": "target-db",
                "volume": "target-db-data",
                "rationale": "reclaim space by dropping the stale volume",
            }
        ],
        applied=applied,
    )
    task = post_incident(
        app,
        {
            "service": "target-db",
            "signals": {"disk_percent": "96%", "restart_count": 0},
        },
    )

    assert task["status"]["state"] == str(TaskState.AUTH_REQUIRED)
    assert applied == [], "an irreversible action must not run before approval"

    detail = task["status"]["message"]["parts"][0]["data"]
    assert detail["awaiting"] == "human_authorisation"
    assert detail["diagnosis"] == "disk_pressure"
    assert ops.pending_approvals(), "the refusal must be recoverable by an approval"


def test_the_refusal_is_in_the_ledger_with_its_reason():
    app, ledger, _, _ = build_stack(
        plan_responses=[
            {
                "tool": "drop_volume",
                "service": "target-db",
                "volume": "target-db-data",
                "rationale": "x",
            }
        ],
        applied=[],
    )
    post_incident(app, {"service": "target-db", "signals": {"disk_percent": "96%"}})

    rows = ledger._store.all_rows()  # noqa: SLF001
    refusals = [r for r in rows if r.decision == "require_human"]
    assert len(refusals) == 1
    assert "cannot be undone" in refusals[0].reason
    ok, _ = ledger.verify()
    assert ok


def test_parameters_are_derived_in_code_not_taken_from_the_model():
    """The model names the action; the code names the arguments.

    Measured reason: `qwen3:1.7b` copied the prompt's `<service>-data`
    placeholder into a plan verbatim as a volume name. A garbage name that
    passes a non-empty check is worse than a rejected plan — it reaches a
    human for approval looking legitimate.
    """
    from sovops.agents.remediation.agent import MANAGED_VOLUMES, derive_arguments

    assert derive_arguments("drop_volume", "target-db", {}) == {
        "service": "target-db",
        "volume": MANAGED_VOLUMES["target-db"],
    }
    # 512 MiB currently → 1024 MiB, arithmetic, no model involved.
    assert derive_arguments(
        "scale_memory", "target-worker", {"memory_limit_bytes": 536870912}
    ) == {"service": "target-worker", "limit_mb": 1024}


def test_no_volume_is_invented_for_an_unregistered_service():
    """Refusing beats guessing: a plausible name for a volume that may not
    exist is exactly what a human would approve without noticing."""
    from sovops.agents.remediation.agent import PlanRejected, derive_arguments

    with pytest.raises(PlanRejected, match="no managed volume registered"):
        derive_arguments("drop_volume", "target-api", {})


def test_plan_without_a_rationale_is_rejected():
    """The rationale is the only thing the model still produces, and it is
    what the human reads before authorising. An empty one means the call
    bought nothing."""
    applied: list[str] = []
    app, _, _, _ = build_stack(
        plan_responses=[
            {"tool": "restart_service", "service": "target-worker", "rationale": ""}
        ],
        applied=applied,
    )
    task = post_incident(
        app, {"service": "target-worker", "signals": {"restart_count": 9, "exit_code": 1}}
    )
    assert task["status"]["state"] == str(TaskState.FAILED)
    assert applied == []


def test_healthy_service_produces_no_action_and_no_model_call():
    applied: list[str] = []
    app, _, backend, _ = build_stack(plan_responses=[], applied=applied)
    task = post_incident(
        app,
        {
            "service": "target-api",
            "signals": {"restart_count": 0, "memory_percent": "30%", "p99_latency_ms": 90},
        },
    )
    assert task["status"]["state"] == str(TaskState.COMPLETED)
    assert applied == []
    assert backend.prompts == []


def test_plan_targeting_another_service_is_rejected_before_it_travels():
    """The most dangerous planning failure, caught in code."""
    applied: list[str] = []
    app, _, _, _ = build_stack(
        plan_responses=[
            {"tool": "restart_service", "service": "target-db", "rationale": "wandered off"}
        ],
        applied=applied,
    )
    task = post_incident(
        app,
        {
            "service": "target-worker",
            "signals": {"restart_count": 9, "exit_code": 1},
        },
    )
    assert task["status"]["state"] == str(TaskState.FAILED)
    assert applied == []


def test_triage_cannot_write_even_if_it_tries(caplog):
    """The privilege boundary, asserted directly rather than trusted."""
    from sovops.mcp_server.server import ToolAccessDenied

    _, ledger, _, ops = build_stack(plan_responses=[], applied=[])
    with pytest.raises(ToolAccessDenied):
        ops.call(
            token=READ_TOKEN,
            tool="restart_service",
            arguments={"service": "target-api", "episode_id": "inc-1"},
            trace_id="t",
            task_id="k",
            agent_id="triage-agent",
        )
