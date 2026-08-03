"""Five-minute demo: two incidents, two outcomes.

Incident 1 is a crash loop. Reversible action, within budget — the agents
handle it end to end and nobody is woken up.

Incident 2 is disk pressure. The only action that frees space destroys
data, so the gate refuses, the task parks in `TASK_STATE_AUTH_REQUIRED`,
and the demo asks. Approving resumes the *same task*, and the approval
releases exactly the refused call — change one argument and it is rejected.

Run with the deterministic backend by default so the demo takes seconds.
`--live` drives the local model instead, which is the honest version and
takes minutes on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys

from starlette.testclient import TestClient

from sovops.a2a.client import A2AClient
from sovops.a2a.server import build_a2a_app
from sovops.a2a.types import ActorContext, TaskState
from sovops.agents.remediation.agent import RemediationAgent
from sovops.agents.remediation.agent import build_card as remediation_card
from sovops.agents.triage.agent import TriageAgent
from sovops.agents.triage.agent import build_card as triage_card
from sovops.evals.harness import (
    READ_TOKEN,
    WRITE_TOKEN,
    Recorder,
    build_gateway,
    build_ops,
    build_registry,
)
from sovops.evals.runner import StubBackend
from sovops.telemetry import tracing

A2A_TOKEN = "demo-a2a"

BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[0m",
)


def rule(title: str) -> None:
    print(f"\n{BOLD}{'─' * 78}\n{title}\n{'─' * 78}{RESET}")


def build(signals: dict, backend):
    recorder = Recorder()
    registry = build_registry(recorder, signals)
    ops, ledger = build_ops(registry)
    gateway = build_gateway(backend)

    remediation = RemediationAgent(ops=ops, gateway=gateway, token=WRITE_TOKEN)
    remediation_app = build_a2a_app(
        card=remediation_card("http://remediation.demo"),
        handler=remediation.handle,
        tokens={A2A_TOKEN: "triage-agent"},
    )
    peer = A2AClient(
        base_url="http://remediation.demo",
        token=A2A_TOKEN,
        actor=ActorContext(actor="triage-agent", on_behalf_of="alertmanager"),
        http_client=TestClient(remediation_app, base_url="http://remediation.demo"),
    )
    triage = TriageAgent(ops=ops, gateway=gateway, token=READ_TOKEN, remediation=peer)
    triage_app = build_a2a_app(
        card=triage_card("http://triage.demo"),
        handler=triage.handle,
        tokens={A2A_TOKEN: "alertmanager"},
    )
    return triage_app, remediation_app, ops, ledger, recorder, gateway


def send(app, base: str, payload: dict, task_id: str | None = None) -> dict:
    message = {"messageId": "demo", "role": "ROLE_USER", "parts": [{"data": payload}]}
    if task_id:
        message["taskId"] = task_id
    resp = TestClient(app, base_url=base).post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {"message": message},
        },
        headers={"Authorization": f"Bearer {A2A_TOKEN}", "X-Sovops-Actor": "alertmanager"},
    )
    body = resp.json()
    if "error" in body:
        raise SystemExit(f"transport error: {body['error']}")
    return body["result"]


def detail(task: dict) -> dict:
    message = task.get("status", {}).get("message") or {}
    for part in message.get("parts", []):
        if "data" in part:
            return part["data"]
    return {}


def show_ledger(ledger) -> None:
    print(f"\n{DIM}audit ledger:{RESET}")
    for row in ledger._store.all_rows():  # noqa: SLF001 - the demo is about the evidence
        colour = GREEN if row.decision in {"read", "auto_apply", "human_approved"} else YELLOW
        print(
            f"  {DIM}#{row.seq}{RESET} {colour}{row.decision:<18}{RESET} {row.tool:<16} "
            f"{DIM}{row.reason[:64]}{RESET}"
        )
    ok, bad = ledger.verify()
    verdict = f"{GREEN}intact{RESET}" if ok else f"{RED}BROKEN at row {bad}{RESET}"
    print(f"  {DIM}chain:{RESET} {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use the local model (minutes)")
    args = parser.parse_args(argv)
    backend = None if args.live else StubBackend()
    tracing.configure("sovops-demo")

    # ---------------------------------------------------------------- 1 ----
    rule("INCIDENT 1 — target-worker crash loop  (reversible, within budget)")
    signals = {
        "restart_count": 12,
        "exit_code": 1,
        "memory_percent": "22%",
        "oom_killed": False,
    }
    print(f"{DIM}signals:{RESET} {json.dumps(signals)}")

    triage_app, _, ops, ledger, recorder, gateway = build(signals, backend)
    task = send(
        triage_app, "http://triage.demo", {"service": "target-worker", "signals": signals}
    )
    info = detail(task)

    print(
        f"\n  diagnosis        {BOLD}{info.get('diagnosis')}{RESET} "
        f"{DIM}(source: {info.get('source')}){RESET}"
    )
    print(f"  delegated to     {info.get('delegated_to')} {DIM}via A2A message/send{RESET}")
    print(f"  final state      {GREEN}{task['status']['state']}{RESET}")
    print(f"  applied          {recorder.tools}")
    print(f"  model calls      {gateway.calls}  {DIM}cost EUR {gateway.cost_eur:.4f}{RESET}")
    show_ledger(ledger)

    # ---------------------------------------------------------------- 2 ----
    rule("INCIDENT 2 — target-db disk pressure  (only action is irreversible)")
    signals = {"disk_percent": "96%", "restart_count": 0, "memory_percent": "44%"}
    print(f"{DIM}signals:{RESET} {json.dumps(signals)}")

    triage_app, remediation_app, ops, ledger, recorder, gateway = build(signals, backend)
    task = send(triage_app, "http://triage.demo", {"service": "target-db", "signals": signals})
    info = detail(task)

    print(
        f"\n  diagnosis        {BOLD}{info.get('diagnosis')}{RESET} "
        f"{DIM}(source: {info.get('source')}){RESET}"
    )
    print(f"  final state      {YELLOW}{task['status']['state']}{RESET}")
    print(f"  applied          {recorder.tools}  {DIM}← nothing ran{RESET}")

    remote = info.get("detail", {})
    print(f"\n  {YELLOW}the gate refused:{RESET} {remote.get('reason')}")
    print(
        f"  proposed         {remote.get('proposed_tool')} {remote.get('proposed_arguments')}"
    )
    print(f"  approval_ref     {remote.get('approval_ref')}")
    show_ledger(ledger)

    # -- the wrong approval --------------------------------------------------
    rule("An approval authorises ONE action — not the next one")
    pending = ops.pending_approvals()[0]
    tampered = ops.call(
        token=WRITE_TOKEN,
        tool=pending.tool,
        arguments={
            **pending.arguments,
            "volume": "production-backups",
            "episode_id": pending.episode_id,
            "approval_ref": pending.ref,
            "approver": "arthur",
        },
        trace_id="demo",
        task_id="demo",
        agent_id="demo",
    )
    print(f"  same ref, different volume → {RED}{tampered.status}{RESET}: {tampered.reason}")

    # -- the right approval --------------------------------------------------
    rule("Human approves — the SAME task resumes")
    resumed = send(
        remediation_app,
        "http://remediation.demo",
        {"approval_ref": remote.get("approval_ref"), "approved": True, "approver": "arthur"},
        task_id=info.get("remediation_task"),
    )
    print(f"  remediation task {resumed['id']}")
    print(f"  final state      {GREEN}{resumed['status']['state']}{RESET}")
    print(f"  applied          {recorder.tools}")
    print(f"  approver         {detail(resumed).get('approver')}")
    show_ledger(ledger)

    rule("Summary")
    print(f"  incident 1  autonomous, {GREEN}no human{RESET} — reversible and within budget")
    print(
        f"  incident 2  paused at {YELLOW}{TaskState.AUTH_REQUIRED}{RESET}, "
        f"resumed by an approval scoped to one action"
    )
    print(
        f"  inference   {'local (qwen3:1.7b)' if args.live else 'deterministic stub'}, "
        f"{GREEN}never left sovereign{RESET}"
    )
    print("  evidence    hash-chained, refusals included\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
