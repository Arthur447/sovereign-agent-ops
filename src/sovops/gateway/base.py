"""The LLM gateway contract — the one door, provider-shaped.

## Why this file exists at all

The obvious way to build an agent is to import a vendor SDK and call it.
That works until the day the deployment target is a SecNumCloud/HDS
environment, and then it does not work at all: the constraint is not
"we prefer open weights", it is "customer data must not transit a
non-sovereign endpoint". A codebase that hard-codes one vendor cannot
express that rule — it can only be audited for it, forever, by humans.

So every model call in this repo goes through `LlmGateway`. No module
outside `sovops.gateway` imports `httpx` against a model endpoint or a
vendor SDK; `tests/test_dep_guard.py` enforces it.

## The part that is not just an interface

Being "model-agnostic" is table stakes and mostly uninteresting — two
implementations behind a Protocol. The design decision here is the
**posture**: the sovereign backend is the default, and leaving it is an
explicit, audited, per-call event carrying a reason, not a silent
fallback when the local model is slow or unsure.

That inversion matters. A gateway that quietly falls back to a hosted
API the first time the local 3B struggles is a compliance incident
waiting to be discovered in a log six months later. `EscalationDenied`
is raised rather than degraded — refusing is the safe failure, and the
caller has to decide what to do about it in the open.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from sovops.telemetry import tracing


class TaskClass(StrEnum):
    """What a call is *for*. Routing keys off this, never off a model name.

    Callers name the job; the routing policy names the model. A node that
    picks its own model has quietly moved a governance decision into
    application code.
    """

    TRIAGE = "triage"  # classify an incident into a known failure mode
    PLAN = "plan"  # synthesise a remediation plan
    JUDGE = "judge"  # eval-time grading of a run


@dataclass(frozen=True)
class ModelSpec:
    """A model the platform is allowed to call.

    `sovereign` is the field the whole posture hangs on: it asserts the
    weights run on infrastructure we control. It is declared per model in
    `routing.yaml`, not inferred from the backend type — self-hosting a
    model does not make an endpoint sovereign if that endpoint is someone
    else's cloud, and that distinction is exactly the one an auditor asks
    about.
    """

    id: str
    backend: str  # "openai_compat" | "anthropic"
    sovereign: bool
    eur_per_mtok_in: float
    eur_per_mtok_out: float
    max_output_tokens: int = 4096

    def cost_eur(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.eur_per_mtok_in + output_tokens * self.eur_per_mtok_out
        ) / 1_000_000


@dataclass(frozen=True)
class Completion:
    """One model response, with everything observability needs attached.

    Cost and latency are computed here rather than by the caller because
    a number that each call site derives on its own is a number that will
    disagree with itself by the third call site. `sovereign` rides along
    so a span or an audit row can record *where the inference happened*
    without re-deriving it from the model id.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_eur: float
    latency_ms: int
    sovereign: bool


class EscalationDenied(RuntimeError):
    """A call asked to leave the sovereign default without authorisation.

    Deliberately an exception and not a degraded return value: the caller
    must handle the refusal explicitly. Silent degradation is how a
    sovereignty guarantee becomes a sovereignty claim.
    """


class Backend(Protocol):
    """A concrete inference backend. Two exist: OpenAI-compatible and Anthropic.

    The OpenAI-compatible one covers both vLLM and Ollama — they expose
    the same `/v1/chat/completions` shape, so the local development path
    and the production self-hosted path run *the same code*, which is the
    only reason a laptop is an acceptable place to build this.
    """

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> tuple[str, int, int]: ...


class LlmGateway:
    """Routes a `TaskClass` to a model, meters the call, enforces posture."""

    def __init__(
        self,
        *,
        policy: RoutingPolicy,
        backends: dict[str, Backend],
    ) -> None:
        self._policy = policy
        self._backends = backends

    def complete(
        self,
        *,
        task_class: TaskClass,
        system: str,
        user: str,
        json_schema: dict | None = None,
        allow_escalation: bool = False,
        escalation_reason: str | None = None,
    ) -> Completion:
        """One call, routed by task class.

        `allow_escalation` is opt-in per call and requires a reason. The
        pair exists so that "we left sovereign inference for this run"
        is a fact recorded at the decision point, not reconstructed later
        from a model id in a log line.
        """
        spec = self._policy.resolve(
            task_class,
            allow_escalation=allow_escalation,
            escalation_reason=escalation_reason,
        )
        backend = self._backends[spec.backend]

        with tracing.span(
            f"llm.{task_class}",
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": spec.id,
                "sovops.task_class": str(task_class),
                "sovops.escalated": allow_escalation,
                "sovops.escalation_reason": escalation_reason,
            },
        ) as current:
            started = time.perf_counter()
            text, tok_in, tok_out = backend.complete(
                model=spec.id,
                system=system,
                user=user,
                max_tokens=spec.max_output_tokens,
                json_schema=json_schema,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)

            completion = Completion(
                text=text,
                model=spec.id,
                input_tokens=tok_in,
                output_tokens=tok_out,
                cost_eur=spec.cost_eur(tok_in, tok_out),
                latency_ms=latency_ms,
                sovereign=spec.sovereign,
            )
            tracing.record_completion(current, completion)
            return completion


@dataclass(frozen=True)
class ClassRoute:
    """The routing rule for one task class."""

    primary: ModelSpec
    escalation: ModelSpec | None = None


class RoutingPolicy:
    """Task class → model, with the sovereign default enforced.

    Loaded from `routing.yaml` so that changing which model handles which
    job is a reviewable config diff rather than a code change — the same
    reason permissions live in a manifest and not in `if` statements.
    """

    def __init__(self, routes: dict[TaskClass, ClassRoute]) -> None:
        self._routes = routes

    def resolve(
        self,
        task_class: TaskClass,
        *,
        allow_escalation: bool,
        escalation_reason: str | None,
    ) -> ModelSpec:
        route = self._routes.get(task_class)
        if route is None:
            raise KeyError(f"no route declared for task class {task_class!r}")

        if not allow_escalation:
            return route.primary

        if route.escalation is None:
            raise EscalationDenied(
                f"task class {task_class!r} declares no escalation target; "
                "the sovereign primary is the only permitted model"
            )
        if not escalation_reason:
            raise EscalationDenied(
                f"escalation to {route.escalation.id!r} requires an explicit "
                "reason — an unexplained departure from sovereign inference "
                "is not auditable"
            )
        return route.escalation

    @property
    def routes(self) -> dict[TaskClass, ClassRoute]:
        return dict(self._routes)
