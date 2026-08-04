"""MCP client — one per agent, because one per *credential*.

The token is held by the client rather than passed per call. That is the
point of the object: "triage may not write" stops being a rule the agent
must remember to respect and becomes a property of the only handle it
has to the tool server. An agent cannot present a credential it was
never given.

Errors are translated back into the exception the in-process contract
raised (`ToolAccessDenied`), so a tier violation reads the same to the
caller whether it was refused across a socket or in memory. The client's
job is to make the remote call obey the local contract; inventing a
second error vocabulary would push protocol details into agent code.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from sovops.mcp_server.app import (
    HEADER_AGENT,
    HEADER_TASK,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
)
from sovops.mcp_server.server import CallOutcome, ToolAccessDenied
from sovops.telemetry import tracing

DEFAULT_TIMEOUT_S = 60.0


class MCPError(RuntimeError):
    """The tool server returned a JSON-RPC error that is not a refusal."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class MCPClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        agent_id: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._agent_id = agent_id
        self._timeout_s = timeout_s
        # Injected so tests drive the real ASGI app in-process through a
        # Starlette `TestClient`. That is not a shortcut around the
        # boundary: the request travels the same routing, the same auth
        # and the same handlers — it simply skips the socket.
        self._http_client = http_client

    @contextmanager
    def _client(self) -> Iterator[httpx.Client]:
        if self._http_client is not None:
            yield self._http_client
            return
        with httpx.Client(timeout=self._timeout_s) as client:
            yield client

    def _rpc(self, method: str, params: dict[str, Any], *, task_id: str) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": method,
            "params": params,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            HEADER_TASK: task_id,
            HEADER_AGENT: self._agent_id,
        }

        with tracing.span(
            f"mcp.client {method}",
            **{"sovops.peer": self._base_url, "sovops.agent_id": self._agent_id},
        ):
            # Injected inside the span, not before it — otherwise the
            # server's span hangs beside this call instead of under it.
            tracing.inject_context(headers)
            with self._client() as client:
                resp = client.post(f"{self._base_url}/", json=body, headers=headers)

        payload = resp.json()
        if "error" in payload:
            err = payload["error"]
            if resp.status_code in (401, 403):
                raise ToolAccessDenied(err.get("message", "denied"))
            raise MCPError(err.get("code", -1), err.get("message", "unknown error"))
        return payload["result"]

    def list_tools(self, *, task_id: str = "discovery") -> list[dict[str, Any]]:
        return self._rpc(METHOD_TOOLS_LIST, {}, task_id=task_id)["tools"]

    def call(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        task_id: str,
    ) -> CallOutcome:
        result = self._rpc(
            METHOD_TOOLS_CALL,
            {"name": tool, "arguments": arguments},
            task_id=task_id,
        )
        return CallOutcome(**result["structuredContent"])
