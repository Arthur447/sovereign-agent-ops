"""Tamper-evident audit ledger.

## Why a hash chain and not just a table

"Every agent action is traced" is satisfied by any append-only table
until someone asks the question a regulated audit actually asks: *how do
you know a row was not removed or edited after the fact?* A plain table
answers "because we don't do that". A chained one answers "because the
next row's hash would not verify", and that answer survives the person
giving it leaving the company.

Each row carries `prev_hash` and its own `row_hash`, computed over the
canonical serialisation of its content plus the previous hash. Deleting
or editing any row breaks verification from that point forward, and
`verify()` reports the first index where the chain fails.

This costs one SHA-256 per action and buys the difference between a log
and evidence. It is deliberately not a blockchain: there is no consensus
and no distribution, because the threat model is "an operator or a
compromised agent edits history", not "a hostile majority".

## What lands here, and when

Every gate ruling — including the refusals. An audit trail that records
only what was done cannot answer "did the agent try to delete the
volume", which is the question after an incident. The ruling is written
*before* the tool call, and the outcome patched onto it after, so an
action that crashed mid-flight is still visible as attempted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditRow:
    """One recorded agent action or refusal.

    `params_hash` rather than `params`: parameters can carry hostnames,
    connection strings and customer identifiers, and an audit ledger that
    is itself a data-leak surface has traded one compliance problem for
    another. The hash proves *which* parameters were used to anyone who
    can reproduce them, without storing them.
    """

    seq: int
    ts: str
    trace_id: str
    task_id: str
    agent_id: str
    tool: str
    params_hash: str
    reversibility: str
    blast_weight: int
    decision: str
    reason: str
    approver: str | None = None
    rollback_ref: str | None = None
    outcome: str | None = None
    prev_hash: str = GENESIS_HASH
    row_hash: str = ""

    def content_digest(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "row_hash"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_params(params: dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditStore(Protocol):
    def append(self, row: AuditRow) -> AuditRow: ...
    def all_rows(self) -> list[AuditRow]: ...


@dataclass
class InMemoryAuditStore:
    """Used by the eval harness and the tests.

    The eval suite runs hundreds of incidents; pointing them at the real
    ledger would mix synthetic actions into the evidence trail, which is
    exactly the kind of contamination the trail exists to prevent.
    """

    rows: list[AuditRow] = field(default_factory=list)

    def append(self, row: AuditRow) -> AuditRow:
        self.rows.append(row)
        return row

    def all_rows(self) -> list[AuditRow]:
        return list(self.rows)


class Ledger:
    """Appends chained rows and verifies the chain."""

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    def record(
        self,
        *,
        trace_id: str,
        task_id: str,
        agent_id: str,
        tool: str,
        params: dict[str, Any],
        reversibility: str,
        blast_weight: int,
        decision: str,
        reason: str,
        approver: str | None = None,
        rollback_ref: str | None = None,
        outcome: str | None = None,
    ) -> AuditRow:
        existing = self._store.all_rows()
        prev_hash = existing[-1].row_hash if existing else GENESIS_HASH

        row = AuditRow(
            seq=len(existing),
            ts=datetime.now(UTC).isoformat(),
            trace_id=trace_id,
            task_id=task_id,
            agent_id=agent_id,
            tool=tool,
            params_hash=hash_params(params),
            reversibility=reversibility,
            blast_weight=blast_weight,
            decision=decision,
            reason=reason,
            approver=approver,
            rollback_ref=rollback_ref,
            outcome=outcome,
            prev_hash=prev_hash,
        )
        sealed = AuditRow(**{**asdict(row), "row_hash": row.content_digest()})
        return self._store.append(sealed)

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain. Returns `(ok, first_bad_index)`.

        Checks both links: that each row's stored hash matches its own
        content, and that it names its predecessor correctly. Only
        checking the second would let an edited row pass as long as
        whoever edited it also recomputed the forward pointer.
        """
        prev = GENESIS_HASH
        for i, row in enumerate(self._store.all_rows()):
            if row.prev_hash != prev:
                return False, i
            if row.row_hash != row.content_digest():
                return False, i
            prev = row.row_hash
        return True, None
