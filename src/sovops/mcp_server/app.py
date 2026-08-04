"""The MCP transport — what turns "the gate is on the server" into a fact.

## Why this file was missing, and what its absence cost

`OpsToolService` implemented every MCP *semantic*: a tool registry with
JSON argument schemas, `tools/list` filtered by privilege tier, the
reversibility gate, the episode budget, the audit ledger. What it had no
transport, so agents reached it by calling the Python object directly.

In one process that is not a boundary. Any code sharing the interpreter
can write `from sovops.mcp_server.ops_tools import drop_volume` and run
the handler, and then:

- the gate is never consulted — it lives in `OpsToolService.call`, which
  was skipped;
- **no audit row is written** — every `_ledger.record` in this package is
  downstream of that same method.

The second one is the serious one. Tampering with an append-only, hash-
chained ledger is hard; never appearing in it is free. A ledger that can
be bypassed is a ledger that records only well-behaved callers.

The realistic bypass is not an attacker either — a prompt injection
cannot do this, since the model emits one enum value and never code. It
is the next person under a deadline who notices the direct import works.
A rule enforced by the discipline of whoever writes the next module is
not a rule.

## What this layer is allowed to decide: nothing

It translates *form*, never *meaning*. It pulls the bearer string out of
an HTTP header and hands it to `OpsToolService`, which is the component
that answers for the mapping credential → privilege. Resolving the tier
here and passing the *result* down would make the service trust its
caller about the caller's own rights — a confused deputy, and an audit
ledger recording claims rather than credentials.

So there is no policy in this file. If a decision ever appears here, it
has leaked out of the component that owns it.

## What is implemented, and what is not

`tools/list` and `tools/call` over JSON-RPC 2.0 with bearer auth. That is
the subset this system uses. **Not** implemented: the `initialize`
handshake and capability negotiation, notifications, `resources/*`,
`prompts/*`, and SSE streaming. Named here rather than implied, because
the defect this file fixes was a docstring that claimed a transport the
module did not have.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from opentelemetry import context

from sovops.a2a.types import (
    ERR_INTERNAL,
    ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
)
from sovops.mcp_server.server import OpsToolService, ToolAccessDenied
from sovops.telemetry import tracing

METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"

# Task and agent identity ride beside the protocol, as they do on the A2A
# side: MCP carries no notion of "which incident is this, and who is
# asking". Without them an audit row cannot be attributed, and an
# unattributable action is one nobody will review.
HEADER_TASK = "X-Sovops-Task"
HEADER_AGENT = "X-Sovops-Agent"


def _error(request_id: Any, code: int, message: str, *, status: int = 200) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
        status_code=status,
    )


def _result(request_id: Any, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def build_mcp_app(service: OpsToolService) -> FastAPI:
    """Assemble the ASGI app in front of one tool service.

    No token map is passed in: the service already holds it, and a second
    copy here would be a second answer to the same question.
    """
    app = FastAPI(title="sovops-ops-tools", version="0.1.0")
    app.state.service = service

    @app.post("/")
    async def jsonrpc(
        request: Request,
        authorization: str | None = Header(default=None),
        x_sovops_task: str | None = Header(default=None),
        x_sovops_agent: str | None = Header(default=None),
    ) -> JSONResponse:
        token = (authorization or "").removeprefix("Bearer ").strip()

        # Adopt the caller's trace so a tool call hangs under the incident
        # that caused it rather than starting a second root.
        otel_token = context.attach(tracing.extract_context(dict(request.headers)))
        try:
            try:
                payload = await request.json()
            except Exception:
                return _error(None, ERR_PARSE, "malformed JSON body")

            if payload.get("jsonrpc") != "2.0" or "method" not in payload:
                return _error(
                    payload.get("id"), ERR_INVALID_REQUEST, "not a JSON-RPC 2.0 request"
                )

            rpc_id = payload.get("id")
            method = payload["method"]
            params = payload.get("params") or {}

            # The tier is resolved by the service, never here. An
            # unknown token fails at this first question, before any tool
            # name is even read.
            try:
                tier = service.tier_for(token)
            except ToolAccessDenied as exc:
                return _error(rpc_id, ERR_INVALID_REQUEST, str(exc), status=401)

            if method == METHOD_TOOLS_LIST:
                return _result(
                    rpc_id,
                    {
                        "tools": [
                            {
                                "name": spec.name,
                                "description": spec.description,
                                "inputSchema": service.exposed_schema(spec),
                            }
                            for spec in service.visible_tools(tier)
                        ]
                    },
                )

            if method == METHOD_TOOLS_CALL:
                name = params.get("name")
                if not name:
                    return _error(rpc_id, ERR_INVALID_PARAMS, "params.name is required")
                if not x_sovops_task or not x_sovops_agent:
                    return _error(
                        rpc_id,
                        ERR_INVALID_PARAMS,
                        f"{HEADER_TASK} and {HEADER_AGENT} are required: "
                        "an action nobody can attribute is an action nobody will review",
                    )
                try:
                    outcome = service.call(
                        token=token,
                        tool=name,
                        arguments=params.get("arguments") or {},
                        trace_id=tracing.current_trace_id(),
                        task_id=x_sovops_task,
                        agent_id=x_sovops_agent,
                    )
                except ToolAccessDenied as exc:
                    # Already audited by the service as `denied_by_tier`.
                    return _error(rpc_id, ERR_INVALID_REQUEST, str(exc), status=403)
                except KeyError as exc:
                    return _error(rpc_id, ERR_INVALID_PARAMS, str(exc))
                except Exception as exc:  # noqa: BLE001
                    return _error(rpc_id, ERR_INTERNAL, str(exc))

                # `isError` is false for a policy refusal: the system did
                # exactly what it was built to do. Reporting a refusal as
                # a failure is how agents learn to retry around it.
                return _result(
                    rpc_id,
                    {
                        "isError": outcome.status == "error",
                        "structuredContent": asdict(outcome),
                    },
                )

            return _error(rpc_id, ERR_METHOD_NOT_FOUND, f"unsupported method {method!r}")
        finally:
            context.detach(otel_token)

    return app
