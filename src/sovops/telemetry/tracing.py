"""OpenTelemetry wiring — one trace per incident, across both agents.

## Why tracing and not more logging

Cost and latency are already recorded per model call, and every action is
already in the audit ledger. What neither answers is *where the time and
the money went inside one incident* — which is the question asked when an
agent takes four minutes to restart a container, and the one a log file
answers only by manual correlation.

A trace answers it structurally: the incident is the root span, the A2A
hop is a child, the model call and the tool call are children of that.
Cost rides as a span attribute rather than in a parallel ledger, so the
observability answer and the billing answer come from one place and
cannot drift.

## The A2A hop

A2A carries no trace context of its own, but the spec explicitly
recommends OpenTelemetry propagation over the HTTP headers, which is what
`inject_context` / `extract_context` do here. Without it each agent
produces its own disconnected trace and the whole exercise is pointless —
two traces for one incident is worse than none, because it looks like
coverage.

## Degrading

If no collector is configured the provider is still installed with no
exporter, so instrumented code runs unchanged and costs almost nothing.
Telemetry that crashes the thing it observes is worse than absent
telemetry, and an ops agent is exactly where that lesson gets expensive.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure(service_name: str) -> None:
    """Install the provider once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OTLP exporter unavailable, tracing locally only: %s", exc)

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def tracer(name: str = "sovops"):
    return trace.get_tracer(name)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """A span that records exceptions and re-raises.

    Swallowing here would make a failing agent look healthy in the trace,
    which is the one thing an observability layer must never do.
    """
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise


def current_trace_id() -> str:
    """The active trace id as hex, or empty if untraced.

    Written into every audit row, so a ledger entry and a trace can be
    joined after the fact — the audit answers *what was permitted*, the
    trace answers *what it cost*, and an incident review needs both.
    """
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else ""


def inject_context(headers: dict[str, str]) -> dict[str, str]:
    """Add W3C traceparent to outgoing A2A headers."""
    inject(headers)
    return headers


def extract_context(headers: Any):
    """Recover the caller's context from incoming A2A headers."""
    return extract(headers)


def record_completion(current: trace.Span, completion: Any) -> None:
    """Attach model economics to the active span.

    `gen_ai.*` follows the OpenTelemetry semantic conventions for
    generative AI so these spans are readable by tooling that has never
    heard of this repo. `sovops.sovereign` is ours, because no convention
    covers "did the inference stay on infrastructure we control" — and in
    this deployment that is the attribute an auditor searches on.
    """
    current.set_attribute("gen_ai.request.model", completion.model)
    current.set_attribute("gen_ai.usage.input_tokens", completion.input_tokens)
    current.set_attribute("gen_ai.usage.output_tokens", completion.output_tokens)
    current.set_attribute("sovops.cost_eur", completion.cost_eur)
    current.set_attribute("sovops.latency_ms", completion.latency_ms)
    current.set_attribute("sovops.sovereign", completion.sovereign)
