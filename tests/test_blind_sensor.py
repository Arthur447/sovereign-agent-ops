"""A blind agent must not report all-green.

Found by walking the loop with no signals injected: every threshold in
`classify` reads a missing field as 0, so an empty dict satisfied rule 6
and came back `healthy` with evidence "no threshold breached". The audit
row for the same run said `get_metrics:read:error: metrics endpoint
unreachable`, and the task still closed COMPLETED.

Same family as the unclassifiable incident that used to close COMPLETED:
absence of information became a reassuring terminal state. There the
model could not decide; here the sensor could not answer.
"""

from __future__ import annotations

from sovops.agents.triage.rules import FailureMode, classify

HEALTHY = {
    "restart_count": 0,
    "memory_percent": "30%",
    "p99_latency_ms": 95,
    "disk_percent": "40%",
}


def test_no_signals_at_all_is_not_healthy():
    assert classify({}).failure_mode is FailureMode.BLIND


def test_blindness_never_reaches_the_model():
    """The first version of this fix returned UNKNOWN, which means "ask the
    model". The stub, handed no evidence, answered `crash_loop` and the
    loop restarted a service on it — trading a silent no-op for an action
    taken on an invention. `is_ambiguous` is what keeps BLIND off that
    path."""
    assert classify({}).is_ambiguous is False
    assert classify({"restart_count": 3, "memory_percent": "85%"}).is_ambiguous is True


def test_the_evidence_says_blind_rather_than_nominal():
    diagnosis = classify({})
    assert "blind" in diagnosis.evidence[0]


def test_a_sensor_that_answered_still_settles_healthy():
    """The guard must not turn every healthy service into a page."""
    assert classify(HEALTHY).failure_mode is FailureMode.HEALTHY


def test_only_logs_is_still_blind():
    """`get_logs` succeeding does not make the metrics reading exist —
    the fields rule 6 asserts about are the ones that must be present."""
    assert classify({"logs": "INFO nothing to see"}).failure_mode is FailureMode.BLIND


def test_a_partial_reading_missing_a_guaranteed_field_is_blind():
    """`get_metrics` returns restart count and memory together or not at
    all; one without the other means the reading is malformed."""
    assert classify({"restart_count": 0}).failure_mode is FailureMode.BLIND


def test_definitional_rules_still_fire_without_a_full_reading():
    """The guard sits at rule 6, not at the top: an OOM kill is memory
    exhaustion whatever else is missing, and deferring it to a human
    because the payload was partial would be the opposite mistake."""
    assert classify({"oom_killed": True}).failure_mode is FailureMode.MEMORY_EXHAUSTION
