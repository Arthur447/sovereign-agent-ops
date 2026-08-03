"""A2A wire types, hand-written against the v1.0 specification.

## Why hand-written and not an SDK

The MCP side of this repo uses the official SDK, because a tool server is
commodity plumbing and reimplementing JSON-RPC framing teaches nothing.
A2A is the opposite case here: the whole point is to understand what the
protocol actually guarantees, and an SDK hides exactly the parts worth
knowing — how a task is addressed, what carries identity (nothing does),
and which state the HITL pause lives in.

So this file is a faithful subset: the canonical data structures, the
JSON-RPC binding, and nothing else. gRPC and HTTP+JSON bindings exist in
the spec and are deliberately absent.

## Fidelity notes

- Task states use the spec's `TASK_STATE_*` enum names verbatim rather
  than shortened ones, so a payload from this server is comparable
  against the spec text without a translation table.
- `Part` is a one-of: exactly one of `text`, `raw`, `url`, `data`. This
  system sends `data` almost exclusively — an incident payload is
  structured, and serialising it to prose so another agent can parse it
  back is a loss with no compensating benefit.
- `AgentCard.securitySchemes` reuses the OpenAPI shapes, as the spec
  requires.

## The identity gap, made explicit

Per the spec's "opaque agent" model, A2A payloads carry **no user or
client identity**. Credentials are obtained out of band and travel in
HTTP headers. That leaves a real hole: when triage delegates to
remediation, nothing in the protocol says *on whose behalf*.

`ActorContext` below is this repo's answer, and it is deliberately
outside the A2A types — it rides in an `X-Sovops-Actor` header and lands
in the audit ledger. Naming it as an extension rather than pretending
A2A provides it is the honest framing: the protocol does not solve
delegation, and anyone who says it does has not read section 8.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

A2A_VERSION = "1.0"
AGENT_CARD_PATH = "/.well-known/agent-card.json"


class TaskState(StrEnum):
    UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    REJECTED = "TASK_STATE_REJECTED"
    # The state this entire system is built around: the agent has decided
    # what to do, is permitted to say so, and is not permitted to do it
    # without a human. Not an error — a pause with a resume path.
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


TERMINAL_STATES = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
)
INTERRUPTED_STATES = frozenset({TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED})


class Role(StrEnum):
    USER = "ROLE_USER"
    AGENT = "ROLE_AGENT"


@dataclass(frozen=True)
class Part:
    """Exactly one of text / raw / url / data."""

    text: str | None = None
    raw: str | None = None  # base64
    url: str | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        present = [f for f in ("text", "raw", "url", "data") if getattr(self, f) is not None]
        if len(present) != 1:
            raise ValueError(
                f"a Part carries exactly one of text/raw/url/data, got {present or 'none'}"
            )

    def to_wire(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> Part:
        return cls(
            **{
                k: v
                for k, v in payload.items()
                if k in {"text", "raw", "url", "data", "metadata"}
            }
        )


@dataclass(frozen=True)
class Message:
    parts: list[Part]
    role: Role = Role.USER
    messageId: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    contextId: str | None = None
    taskId: str | None = None
    metadata: dict[str, Any] | None = None
    referenceTaskIds: list[str] | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "messageId": self.messageId,
            "role": str(self.role),
            "parts": [p.to_wire() for p in self.parts],
        }
        for key in ("contextId", "taskId", "metadata", "referenceTaskIds"):
            value = getattr(self, key)
            if value is not None:
                wire[key] = value
        return wire

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> Message:
        return cls(
            parts=[Part.from_wire(p) for p in payload.get("parts", [])],
            role=Role(payload.get("role", Role.USER)),
            messageId=payload.get("messageId") or f"msg-{uuid.uuid4().hex[:12]}",
            contextId=payload.get("contextId"),
            taskId=payload.get("taskId"),
            metadata=payload.get("metadata"),
            referenceTaskIds=payload.get("referenceTaskIds"),
        )

    def first_data(self) -> dict[str, Any]:
        for part in self.parts:
            if part.data is not None:
                return part.data
        raise ValueError("message carries no data part")


@dataclass
class TaskStatus:
    state: TaskState
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    message: Message | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"state": str(self.state), "timestamp": self.timestamp}
        if self.message is not None:
            wire["message"] = self.message.to_wire()
        return wire


@dataclass
class Artifact:
    """A task's output. Parts, same as a message, but durable and addressable."""

    artifactId: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:12]}")
    name: str | None = None
    parts: list[Part] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "artifactId": self.artifactId,
            "parts": [p.to_wire() for p in self.parts],
        }
        if self.name:
            wire["name"] = self.name
        return wire


@dataclass
class Task:
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    contextId: str = field(default_factory=lambda: f"ctx-{uuid.uuid4().hex[:12]}")
    status: TaskStatus = field(default_factory=lambda: TaskStatus(TaskState.SUBMITTED))
    artifacts: list[Artifact] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "contextId": self.contextId,
            "status": self.status.to_wire(),
            "artifacts": [a.to_wire() for a in self.artifacts],
            "history": [m.to_wire() for m in self.history],
            "metadata": self.metadata,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status.state in TERMINAL_STATES

    @property
    def awaits_human(self) -> bool:
        return self.status.state in INTERRUPTED_STATES


# --------------------------------------------------------------------------
# Agent Card
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentCard:
    """Served at `/.well-known/agent-card.json` (RFC 8615).

    `securitySchemes` uses the OpenAPI shapes the spec borrows. Here it is
    a bearer scheme: the token both authenticates the caller and selects
    its MCP privilege tier downstream, which is the mechanism that makes
    the A2A boundary a privilege boundary rather than a naming convention.
    """

    name: str
    description: str
    version: str
    url: str
    skills: list[AgentSkill] = field(default_factory=list)
    streaming: bool = False
    pushNotifications: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocolVersion": A2A_VERSION,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "provider": {"organization": "sovereign-agent-ops", "url": self.url},
            "interfaces": [{"transport": "JSONRPC", "url": self.url}],
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": self.pushNotifications,
            },
            "securitySchemes": {
                "bearer": {"type": "http", "scheme": "bearer", "bearerFormat": "opaque"}
            },
            "security": [{"bearer": []}],
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": [
                {"id": s.id, "name": s.name, "description": s.description, "tags": s.tags}
                for s in self.skills
            ],
        }


@dataclass(frozen=True)
class ActorContext:
    """Who this work is ultimately on behalf of. NOT part of A2A.

    A2A's opaque-agent model means the protocol carries no identity, so a
    delegated call cannot say who authorised it. This repo propagates an
    actor out of band in `X-Sovops-Actor` and writes it to the audit
    ledger alongside the trace id.

    It is a stopgap, and worth naming as one: an opaque header is not a
    verifiable delegation, and a downstream agent has to trust its caller
    completely. The real fix is a signed delegation token — the direction
    the ecosystem's agent-identity work is heading — and this is the seam
    where it would land.
    """

    actor: str
    on_behalf_of: str | None = None

    def to_header(self) -> str:
        return f"{self.actor}" + (f";obo={self.on_behalf_of}" if self.on_behalf_of else "")

    @classmethod
    def from_header(cls, raw: str | None) -> ActorContext:
        if not raw:
            return cls(actor="anonymous")
        actor, _, rest = raw.partition(";")
        obo = rest.removeprefix("obo=") if rest.startswith("obo=") else None
        return cls(actor=actor.strip() or "anonymous", on_behalf_of=obo)


# --------------------------------------------------------------------------
# JSON-RPC method names (spec §9)
# --------------------------------------------------------------------------

METHOD_MESSAGE_SEND = "message/send"
METHOD_MESSAGE_STREAM = "message/sendStreaming"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_LIST = "tasks/list"
METHOD_TASKS_CANCEL = "tasks/cancel"
METHOD_TASKS_SUBSCRIBE = "tasks/subscribe"
METHOD_AGENT_EXTENDED_CARD = "agent/getExtendedAgentCard"

# JSON-RPC error codes: the standard range plus A2A's reserved block.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_TASK_NOT_FOUND = -32001
ERR_TASK_NOT_CANCELABLE = -32002
