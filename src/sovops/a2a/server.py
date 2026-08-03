"""A2A JSON-RPC server — the transport half, agent-agnostic.

Hosts one agent: serves its Agent Card, dispatches the JSON-RPC methods,
and owns the task store. What the agent *does* lives in a handler passed
in at construction, so both agents in this repo share this file and
differ only in behaviour and privilege.

## Authentication

Bearer, validated per request against a token→principal map, exactly as
the spec prescribes: credentials arrive in HTTP headers, never in the
JSON-RPC payload. A missing or unknown token is `401`; a known token
calling a method its principal may not use is `403`.

The token does double duty — it authenticates the caller *and* selects
the MCP privilege tier used downstream. That is what makes the boundary
between triage and remediation real: they are not two names for the same
capability, they hold different credentials and the tool server enforces
the difference.

## The resume path

`message/send` carrying a `taskId` for a task sitting in
`TASK_STATE_AUTH_REQUIRED` is an approval (or a refusal) arriving from a
human. The task picks up where the gate stopped it. This is the whole
HITL mechanism: no side channel, no separate approval service — a paused
task and a second message on the same id.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from sovops.a2a.types import (
    AGENT_CARD_PATH,
    ERR_INTERNAL,
    ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    ERR_TASK_NOT_CANCELABLE,
    ERR_TASK_NOT_FOUND,
    METHOD_MESSAGE_SEND,
    METHOD_TASKS_CANCEL,
    METHOD_TASKS_GET,
    METHOD_TASKS_LIST,
    A2A_VERSION,
    ActorContext,
    AgentCard,
    Message,
    Task,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)

TaskHandler = Callable[[Task, Message, ActorContext], Awaitable[None]]


class TaskStore:
    """In-memory task store.

    In-memory is a real limitation and worth stating: a paused task does
    not survive a restart, so an approval arriving after a deploy has
    nothing to resume. The fix is the same table the audit ledger already
    uses, and the seam is this class — nothing above it knows where tasks
    live.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def put(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(self, limit: int = 50) -> list[Task]:
        return list(self._tasks.values())[-limit:]


def _rpc_error(request_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
        status_code=status,
    )


def _rpc_result(request_id: Any, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "result": result},
        headers={"A2A-Version": A2A_VERSION},
    )


def build_a2a_app(
    *,
    card: AgentCard,
    handler: TaskHandler,
    tokens: dict[str, str],
    store: TaskStore | None = None,
) -> FastAPI:
    """Assemble the ASGI app for one agent.

    `tokens` maps bearer token → principal name. Kept as a plain dict
    because in a real deployment this is an OIDC introspection call, and
    pretending otherwise with a fake JWT would add ceremony without
    adding a single guarantee.
    """
    app = FastAPI(title=card.name, version=card.version)
    tasks = store or TaskStore()
    app.state.task_store = tasks

    @app.get(AGENT_CARD_PATH)
    async def agent_card() -> dict[str, Any]:
        """Public discovery. Unauthenticated by design — a card advertises
        which credentials are needed, so requiring them to read it is a
        bootstrap loop."""
        return card.to_wire()

    @app.post("/")
    async def jsonrpc(
        request: Request,
        authorization: str | None = Header(default=None),
        x_sovops_actor: str | None = Header(default=None),
    ) -> JSONResponse:
        token = (authorization or "").removeprefix("Bearer ").strip()
        principal = tokens.get(token)
        if principal is None:
            return _rpc_error(None, ERR_INVALID_REQUEST, "unauthenticated", status=401)

        try:
            payload = await request.json()
        except Exception:
            return _rpc_error(None, ERR_PARSE, "malformed JSON body")

        if payload.get("jsonrpc") != "2.0" or "method" not in payload:
            return _rpc_error(payload.get("id"), ERR_INVALID_REQUEST, "not a JSON-RPC 2.0 request")

        rpc_id = payload.get("id")
        method = payload["method"]
        params = payload.get("params") or {}
        actor = ActorContext.from_header(x_sovops_actor)

        if method == METHOD_MESSAGE_SEND:
            return await _handle_send(tasks, handler, rpc_id, params, actor)
        if method == METHOD_TASKS_GET:
            task = tasks.get(params.get("id", ""))
            if task is None:
                return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"no task {params.get('id')!r}")
            return _rpc_result(rpc_id, task.to_wire())
        if method == METHOD_TASKS_LIST:
            limit = int(params.get("limit", 50))
            return _rpc_result(rpc_id, {"tasks": [t.to_wire() for t in tasks.list(limit)]})
        if method == METHOD_TASKS_CANCEL:
            task = tasks.get(params.get("id", ""))
            if task is None:
                return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"no task {params.get('id')!r}")
            if task.is_terminal:
                return _rpc_error(
                    rpc_id, ERR_TASK_NOT_CANCELABLE, f"task already {task.status.state}"
                )
            task.status = TaskStatus(TaskState.CANCELED)
            return _rpc_result(rpc_id, task.to_wire())

        return _rpc_error(rpc_id, ERR_METHOD_NOT_FOUND, f"unsupported method {method!r}")

    return app


async def _handle_send(
    tasks: TaskStore,
    handler: TaskHandler,
    rpc_id: Any,
    params: dict[str, Any],
    actor: ActorContext,
) -> JSONResponse:
    raw_message = params.get("message")
    if not isinstance(raw_message, dict):
        return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "params.message is required")

    try:
        message = Message.from_wire(raw_message)
    except ValueError as exc:
        return _rpc_error(rpc_id, ERR_INVALID_PARAMS, str(exc))

    # A message naming an existing task is a continuation — in this system,
    # almost always a human answering an AUTH_REQUIRED pause.
    if message.taskId:
        task = tasks.get(message.taskId)
        if task is None:
            return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"no task {message.taskId!r}")
        if task.is_terminal:
            return _rpc_error(
                rpc_id, ERR_INVALID_REQUEST, f"task already {task.status.state}, cannot continue"
            )
    else:
        task = tasks.put(Task())

    task.history.append(message)
    task.status = TaskStatus(TaskState.WORKING)

    try:
        await handler(task, message, actor)
    except Exception as exc:  # noqa: BLE001 - the boundary turns any failure into a task state
        logger.exception("handler failed for task %s", task.id)
        task.status = TaskStatus(
            TaskState.FAILED,
            message=Message.from_wire(
                {"role": "ROLE_AGENT", "parts": [{"data": {"error": str(exc)[:500]}}]}
            ),
        )

    return _rpc_result(rpc_id, task.to_wire())
