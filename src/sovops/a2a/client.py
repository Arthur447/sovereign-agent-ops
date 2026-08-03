"""A2A JSON-RPC client.

Small on purpose. The interesting behaviour of an A2A client is not the
HTTP — it is knowing that a returned task may be *paused* rather than
done, and that a pause is resumed by a second `message/send` on the same
task id. `send` returns the raw `Task`; deciding what a non-terminal
state means is the caller's business, because in this system the answer
differs: the triage agent surfaces the pause to its own caller, and the
demo script asks a human.

`discover()` fetches the Agent Card. It is used at startup rather than at
call time — a card is a description of a peer, not a per-request lookup,
and re-fetching it on every call would turn a discovery mechanism into a
hot path.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from sovops.a2a.types import (
    A2A_VERSION,
    AGENT_CARD_PATH,
    METHOD_MESSAGE_SEND,
    METHOD_TASKS_CANCEL,
    METHOD_TASKS_GET,
    ActorContext,
    Message,
    Part,
    Role,
)
from sovops.telemetry import tracing

DEFAULT_TIMEOUT_S = 600.0  # sovereign inference on CPU is slow; see README


class A2AError(RuntimeError):
    """The peer returned a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class A2AClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        actor: ActorContext | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._actor = actor or ActorContext(actor="system")
        self._timeout_s = timeout_s
        # Injected so the end-to-end tests can drive the peer's real ASGI app
        # in-process by passing a Starlette `TestClient` (itself an
        # `httpx.Client`). Spawning uvicorn in a thread would exercise the
        # same code plus a socket, and fail for port reasons instead of
        # protocol ones.
        self._http_client = http_client

    @contextmanager
    def _client(self) -> Iterator[httpx.Client]:
        """Yield a client, closing it only if we created it."""
        if self._http_client is not None:
            yield self._http_client
            return
        with httpx.Client(timeout=self._timeout_s) as client:
            yield client

    def discover(self) -> dict[str, Any]:
        with self._client() as client:
            resp = client.get(f"{self._base_url}{AGENT_CARD_PATH}")
        resp.raise_for_status()
        return resp.json()

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": method,
            "params": params,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "A2A-Version": A2A_VERSION,
            # Not an A2A field. See `ActorContext` — the protocol carries
            # no identity, so delegation rides beside it and lands in the
            # audit ledger.
            "X-Sovops-Actor": self._actor.to_header(),
        }
        # W3C traceparent. A2A defines no trace field either, but the spec
        # recommends OpenTelemetry propagation over the HTTP headers —
        # without it each agent emits a disconnected trace and one incident
        # looks like two.
        tracing.inject_context(headers)

        with tracing.span(
            f"a2a.{method}",
            **{"sovops.peer": self._base_url, "sovops.actor": self._actor.actor},
        ):
            with self._client() as client:
                resp = client.post(f"{self._base_url}/", json=body, headers=headers)
        payload = resp.json()
        if "error" in payload:
            err = payload["error"]
            raise A2AError(err.get("code", -1), err.get("message", "unknown error"))
        return payload["result"]

    def send_data(self, data: dict[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
        """Send one structured payload. Returns the peer's Task.

        `data` and not `text`: an incident is structured, and serialising
        it to prose so the peer can parse it back is a lossy round-trip
        with nothing gained.
        """
        message = Message(
            role=Role.USER,
            parts=[Part(data=data)],
            taskId=task_id,
        )
        return self._rpc(METHOD_MESSAGE_SEND, {"message": message.to_wire()})

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._rpc(METHOD_TASKS_GET, {"id": task_id})

    def cancel(self, task_id: str) -> dict[str, Any]:
        return self._rpc(METHOD_TASKS_CANCEL, {"id": task_id})
