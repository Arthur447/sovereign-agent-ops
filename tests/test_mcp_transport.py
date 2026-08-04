"""The tool server over its transport: same guarantees, now unreachable
by any other route.

These assertions were previously true only by convention — nothing
stopped a caller from importing the handler and skipping the gate, and
the ledger records nothing about a call that never reached the service.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from sovops.evals.harness import READ_TOKEN, WRITE_TOKEN, Recorder, build_ops, build_registry
from sovops.mcp_server.app import build_mcp_app
from sovops.mcp_server.client import MCPClient
from sovops.mcp_server.server import ToolAccessDenied

SIGNALS = {"restart_count": 7, "memory_percent": "99.4%", "memory_limit_bytes": 536870912}


@pytest.fixture
def stack():
    recorder = Recorder()
    ops, ledger = build_ops(build_registry(recorder, SIGNALS))
    app = build_mcp_app(ops)
    http = TestClient(app, base_url="http://tools.test")

    def client_for(token: str, agent_id: str) -> MCPClient:
        return MCPClient(
            base_url="http://tools.test", token=token, agent_id=agent_id, http_client=http
        )

    return client_for, ledger, recorder, http


def test_an_unknown_token_is_refused_before_any_tool_is_named(stack):
    client_for, _, _, _ = stack
    with pytest.raises(ToolAccessDenied):
        client_for("not-a-token", "impostor").list_tools()


def test_listing_is_filtered_by_tier(stack):
    client_for, _, _, _ = stack
    read_only = {t["name"] for t in client_for(READ_TOKEN, "triage-agent").list_tools()}
    operator = {t["name"] for t in client_for(WRITE_TOKEN, "remediation-agent").list_tools()}

    assert read_only == {"get_metrics", "get_logs"}
    assert "scale_memory" in operator


def test_hiding_a_tool_is_not_access_control(stack):
    """Triage cannot see `scale_memory` in the listing. That is a
    convenience. The assertion that matters is that naming it anyway
    fails."""
    client_for, ledger, recorder, _ = stack
    with pytest.raises(ToolAccessDenied):
        client_for(READ_TOKEN, "triage-agent").call(
            tool="scale_memory",
            arguments={"service": "target-worker", "limit_mb": 1024, "episode_id": "e1"},
            task_id="t1",
        )

    assert recorder.tools == []
    rows = ledger._store.all_rows()
    assert [r.decision for r in rows] == ["denied_by_tier"]


def test_the_gate_still_rules_across_the_transport(stack):
    client_for, ledger, recorder, _ = stack
    outcome = client_for(WRITE_TOKEN, "remediation-agent").call(
        tool="scale_memory",
        arguments={"service": "target-worker", "limit_mb": 1024, "episode_id": "e1"},
        task_id="t1",
    )

    assert outcome.status == "ok"
    assert outcome.decision == "apply_with_rollback"
    assert recorder.tools == ["scale_memory"]
    assert ledger.verify()[0]


def test_an_irreversible_action_is_refused_over_the_wire_too(stack):
    """A refusal is not an error: `isError` stays false and the agent gets
    an approval reference, because the system did exactly what it was
    built to do."""
    client_for, _, recorder, _ = stack
    outcome = client_for(WRITE_TOKEN, "remediation-agent").call(
        tool="drop_volume",
        arguments={"volume": "pgdata", "episode_id": "e1"},
        task_id="t1",
    )

    assert outcome.status == "approval_required"
    assert outcome.approval_ref
    assert recorder.tools == []


def test_a_call_that_names_no_task_is_refused(stack):
    """Attribution is not optional. Without it the audit row cannot be
    tied to an incident, and an action nobody can attribute is an action
    nobody will review."""
    _, _, _, http = stack
    resp = http.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {"name": "get_metrics", "arguments": {"service": "target-worker"}},
        },
        headers={"Authorization": f"Bearer {READ_TOKEN}"},
    )
    assert "error" in resp.json()
