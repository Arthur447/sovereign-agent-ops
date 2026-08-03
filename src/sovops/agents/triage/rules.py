"""Deterministic pre-classification — the model only adjudicates ambiguity.

## Why this exists (it was not the original plan)

The first end-to-end run of the sovereign path measured two things worth
writing down rather than working around:

- ~70s per call for a 400-token prompt on 8 CPU cores. Real, not a cold
  start — it stayed there warm.
- `qwen3:1.7b` classified a latency incident (`cpu=91%, p99=4200ms,
  mem=41%`) as `memory_exhaustion`. Not a prompt bug: a 1.7B model
  reading a closed-set classification off six numbers is simply not
  reliable at it.

The tempting fix is to escalate to a bigger hosted model. The routing
policy forbids it for `triage`, on purpose: an incident payload carries
production logs and hostnames, which is the data class the sovereignty
constraint exists to protect. "The local model was not good enough" is
not an exception clause for that.

So the fix is architectural instead. Most of these classifications are
not judgement at all — `oom_killed=true` *is* memory exhaustion, and
asking a language model to confirm it buys non-determinism for nothing.
The rules below settle every case that arithmetic can settle, and the
model is called only when the evidence genuinely conflicts.

This is the same shape as the deterministic funnel diagnosis in the LPM
loop: compute what can be computed, escalate only the mismatch case.

## What this buys

Latency: unambiguous incidents resolve in microseconds and never reach
the model at all. Accuracy: the cases that do reach it are the ones where
a wrong answer was always going to need review anyway. And the eval
suite can now measure the two paths separately — a rule that fires and a
model that guesses are different kinds of correct, and averaging them
would hide which one is carrying the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FailureMode(StrEnum):
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CRASH_LOOP = "crash_loop"
    DISK_PRESSURE = "disk_pressure"
    LATENCY_DEGRADATION = "latency_degradation"
    CONFIG_DRIFT = "config_drift"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


class Source(StrEnum):
    """Which path produced a diagnosis. Reported, never averaged away."""

    RULE = "rule"
    MODEL = "model"


@dataclass(frozen=True)
class Diagnosis:
    failure_mode: FailureMode
    confidence: float
    source: Source
    evidence: list[str]

    @property
    def is_ambiguous(self) -> bool:
        return self.failure_mode is FailureMode.UNKNOWN


# Thresholds are named constants rather than inline literals so the eval
# suite can sweep them, and so a change to one is a reviewable diff with
# a name attached.
OOM_MEMORY_PERCENT = 95.0
CRASH_LOOP_RESTARTS = 5
DISK_PRESSURE_PERCENT = 90.0
LATENCY_P99_MS = 2000.0
HIGH_CPU_PERCENT = 80.0


def _percent(raw: Any) -> float:
    """Docker reports percentages as '41.20%'. Parse defensively, not cleverly."""
    if raw is None:
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    try:
        return float(str(raw).strip().rstrip("%") or 0.0)
    except ValueError:
        return 0.0


def classify(signals: dict[str, Any]) -> Diagnosis:
    """Settle what arithmetic can settle. Return UNKNOWN to defer to the model.

    Rule order encodes precedence, and the precedence is deliberate: an
    OOM kill outranks a restart count, because a container that OOMs
    *also* restarts, and diagnosing the symptom instead of the cause
    produces a remediation that restarts a container which will OOM again
    in four minutes.
    """
    evidence: list[str] = []

    oom = bool(signals.get("oom_killed"))
    mem_pct = _percent(signals.get("memory_percent"))
    restarts = int(signals.get("restart_count") or 0)
    disk_pct = _percent(signals.get("disk_percent"))
    p99 = float(signals.get("p99_latency_ms") or 0.0)
    cpu_pct = _percent(signals.get("cpu_percent"))
    exit_code = signals.get("exit_code")
    config_changed = bool(signals.get("config_changed_recently"))

    # 1. OOM is definitional. No model needed, ever.
    if oom or mem_pct >= OOM_MEMORY_PERCENT:
        if oom:
            evidence.append("oom_killed=true")
        if mem_pct >= OOM_MEMORY_PERCENT:
            evidence.append(f"memory_percent={mem_pct:.1f} >= {OOM_MEMORY_PERCENT}")
        return Diagnosis(FailureMode.MEMORY_EXHAUSTION, 1.0, Source.RULE, evidence)

    # 2. Disk pressure is equally definitional and independent of the rest.
    if disk_pct >= DISK_PRESSURE_PERCENT:
        return Diagnosis(
            FailureMode.DISK_PRESSURE,
            1.0,
            Source.RULE,
            [f"disk_percent={disk_pct:.1f} >= {DISK_PRESSURE_PERCENT}"],
        )

    # 3. Repeated restarts with a non-zero exit and no memory signal.
    if restarts >= CRASH_LOOP_RESTARTS and exit_code not in (0, None):
        return Diagnosis(
            FailureMode.CRASH_LOOP,
            1.0,
            Source.RULE,
            [f"restart_count={restarts} >= {CRASH_LOOP_RESTARTS}", f"exit_code={exit_code}"],
        )

    # 4. Slow but alive, and not restarting — a performance problem.
    if p99 >= LATENCY_P99_MS and restarts == 0:
        evidence.append(f"p99_latency_ms={p99:.0f} >= {LATENCY_P99_MS}")
        if cpu_pct >= HIGH_CPU_PERCENT:
            evidence.append(f"cpu_percent={cpu_pct:.1f} >= {HIGH_CPU_PERCENT}")
        return Diagnosis(FailureMode.LATENCY_DEGRADATION, 1.0, Source.RULE, evidence)

    # 5. Recent config change plus any degradation, with no resource cause.
    if config_changed and (restarts > 0 or p99 >= LATENCY_P99_MS):
        return Diagnosis(
            FailureMode.CONFIG_DRIFT,
            0.9,
            Source.RULE,
            ["config_changed_recently=true", f"restart_count={restarts}"],
        )

    # 6. Everything nominal.
    if restarts == 0 and p99 < LATENCY_P99_MS and mem_pct < 80 and disk_pct < 80:
        return Diagnosis(
            FailureMode.HEALTHY,
            1.0,
            Source.RULE,
            ["no threshold breached"],
        )

    # 7. Genuinely ambiguous — conflicting or partial evidence. This is the
    #    only case worth a model call, and it is the case a model is
    #    actually better at than a threshold.
    return Diagnosis(
        FailureMode.UNKNOWN,
        0.0,
        Source.RULE,
        [
            f"restart_count={restarts}",
            f"memory_percent={mem_pct:.1f}",
            f"p99_latency_ms={p99:.0f}",
            f"cpu_percent={cpu_pct:.1f}",
            f"exit_code={exit_code}",
        ],
    )
