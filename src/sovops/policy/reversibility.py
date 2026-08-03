"""The reversibility gate — where "autonomy with guardrails" stops being a slogan.

## The rule this encodes

An autonomous remediation agent needs a rule for when it may act alone.
"Ask a human for risky things" is not a rule, it is a wish: risk is not a
property anyone can evaluate consistently at 3am, and an agent asked to
self-assess it will be confidently wrong in both directions.

The rule used here is narrower and mechanical: **an action may be taken
autonomously if and only if undoing it costs nothing.** Everything else
escalates. Reversibility is a property of the *tool*, declared once when
the tool is written and reviewed like a permission — not inferred per
call by the model that wants to use it.

That gives three tiers:

- `REVERSIBLE`   — the system returns to its prior state on its own, or
                   the action is idempotent and state-preserving.
                   Restarting a stateless container. Scaling replicas up.
- `COMPENSABLE`  — undone only by a specific inverse action, which must
                   exist and must be recorded *before* the action runs.
                   Changing a config value. Scaling memory limits.
- `IRREVERSIBLE` — no inverse exists. Deleting a volume. Rotating a
                   credential other systems hold. Draining to a state
                   whose prior contents are gone.

## Why blast radius is separate, and cumulative

Reversibility alone is not enough: restarting one container is reversible
and restarting every container in the cluster is *also* reversible, and
only one of those is something an agent should do unattended at 3am.

So the gate takes a second dimension, and — this is the part usually
missed — it accumulates over the episode. Twenty individually-tiny
actions inside one incident are not twenty small decisions; they are one
large one, taken by an agent that never noticed it was escalating. The
gate is therefore stateful per task: `EpisodeBudget` carries what has
already been applied, and a call that fits on its own can still be
refused because of what preceded it.

## The mapping to the protocol

The output is not a boolean. `Decision.REQUIRE_HUMAN` is returned to the
A2A layer, which puts the task into `TASK_STATE_AUTH_REQUIRED` — a state
the protocol already defines, that a client can poll, and that resumes
when a human answers. The regulatory constraint ("validation beyond the
reversible perimeter") thus lives in the task state machine rather than
in a convention someone has to remember.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    COMPENSABLE = "compensable"
    IRREVERSIBLE = "irreversible"


class Scope(StrEnum):
    """What an action touches, ordered by how much of the estate it can take down.

    `DATA` is deliberately not on the same axis as the infrastructure
    scopes: it is not "bigger than a cluster", it is a different kind of
    loss. Infrastructure comes back; data does not. It gets its own rule
    below rather than a larger number here, because ranking it on the
    same scale would let a permissive threshold authorise it by accident.
    """

    CONTAINER = "container"
    SERVICE = "service"
    NODE = "node"
    CLUSTER = "cluster"
    DATA = "data"


_INFRA_WEIGHT: dict[Scope, int] = {
    Scope.CONTAINER: 1,
    Scope.SERVICE: 4,
    Scope.NODE: 16,
    Scope.CLUSTER: 64,
}


@dataclass(frozen=True)
class BlastRadius:
    scope: Scope
    n_targets: int = 1

    @property
    def weight(self) -> int:
        """A single comparable number for the budget.

        Weights are geometric rather than linear because the operational
        difference between one container and one service is not the same
        as between four containers and five. Restarting a service is a
        category change, not four times a container.
        """
        if self.scope is Scope.DATA:
            return _INFRA_WEIGHT[Scope.CLUSTER] * max(1, self.n_targets)
        return _INFRA_WEIGHT[self.scope] * max(1, self.n_targets)


@dataclass(frozen=True)
class ToolRisk:
    """The risk declaration attached to an MCP tool.

    Written once, next to the tool, reviewed like a permission change.
    `rollback_tool` is mandatory for `COMPENSABLE` and validated at
    registration — a compensable action whose inverse does not exist is
    an irreversible action that has not been labelled honestly, and the
    registry refuses it rather than discovering it during an incident.
    """

    reversibility: Reversibility
    blast: BlastRadius
    rollback_tool: str | None = None

    def __post_init__(self) -> None:
        if self.reversibility is Reversibility.COMPENSABLE and not self.rollback_tool:
            raise ValueError(
                "a COMPENSABLE tool must declare `rollback_tool`; without a named "
                "inverse it is IRREVERSIBLE and must say so"
            )
        if self.reversibility is not Reversibility.COMPENSABLE and self.rollback_tool:
            raise ValueError(
                f"`rollback_tool` is meaningless for {self.reversibility}; "
                "only COMPENSABLE actions are undone by an inverse call"
            )


class Decision(StrEnum):
    AUTO_APPLY = "auto_apply"
    APPLY_WITH_ROLLBACK = "apply_with_rollback"
    REQUIRE_HUMAN = "require_human"
    DENY = "deny"


@dataclass(frozen=True)
class Ruling:
    """The gate's answer, carrying the reason.

    The reason is not decoration: it is what lands in the audit row and
    in the message shown to the human being woken up. An escalation that
    cannot explain itself trains operators to approve reflexively, which
    is worse than not escalating at all.
    """

    decision: Decision
    reason: str
    rollback_tool: str | None = None
    budget_after: int = 0


@dataclass
class EpisodeBudget:
    """Cumulative blast radius already spent inside one incident.

    Lives for the duration of one A2A task. The ceiling is what stops an
    agent from taking a hundred individually-permissible actions during a
    single confused episode — the failure mode that looks like a runaway
    loop from the outside and like compliance from inside each call.
    """

    ceiling: int = 16
    spent: int = 0
    applied: list[str] = field(default_factory=list)

    def would_exceed(self, cost: int) -> bool:
        return self.spent + cost > self.ceiling

    def charge(self, tool: str, cost: int) -> None:
        self.spent += cost
        self.applied.append(tool)


class ReversibilityGate:
    """Decides, per call, whether the agent may act alone.

    `auto_apply_ceiling` is the largest single-action weight that may run
    unattended. The default of 4 admits container-level work and one
    service-level action, and refers everything above to a human.
    """

    def __init__(self, *, auto_apply_ceiling: int = 4, episode_ceiling: int = 16) -> None:
        self._auto_ceiling = auto_apply_ceiling
        self._episode_ceiling = episode_ceiling

    def new_episode(self) -> EpisodeBudget:
        return EpisodeBudget(ceiling=self._episode_ceiling)

    def rule(self, *, tool: str, risk: ToolRisk, budget: EpisodeBudget) -> Ruling:
        cost = risk.blast.weight

        # Data is categorical. No threshold authorises destroying it, and
        # no accumulation of prior good behaviour earns the right.
        if risk.blast.scope is Scope.DATA and risk.reversibility is not Reversibility.REVERSIBLE:
            return Ruling(
                decision=Decision.REQUIRE_HUMAN,
                reason=(
                    f"{tool} touches data irreversibly; no autonomy threshold "
                    "applies to loss that cannot be undone"
                ),
                budget_after=budget.spent,
            )

        if risk.reversibility is Reversibility.IRREVERSIBLE:
            return Ruling(
                decision=Decision.REQUIRE_HUMAN,
                reason=f"{tool} is irreversible (scope={risk.blast.scope}, n={risk.blast.n_targets})",
                budget_after=budget.spent,
            )

        # The episode check runs before the per-call check: an action that
        # is fine in isolation is still refused if this incident has
        # already moved too much of the estate.
        if budget.would_exceed(cost):
            return Ruling(
                decision=Decision.REQUIRE_HUMAN,
                reason=(
                    f"episode blast budget exhausted: {budget.spent}+{cost} > "
                    f"{budget.ceiling} after {len(budget.applied)} action(s) "
                    f"({', '.join(budget.applied) or 'none'})"
                ),
                budget_after=budget.spent,
            )

        if cost > self._auto_ceiling:
            return Ruling(
                decision=Decision.REQUIRE_HUMAN,
                reason=(
                    f"single-action blast weight {cost} exceeds the autonomous "
                    f"ceiling {self._auto_ceiling} (scope={risk.blast.scope}, "
                    f"n={risk.blast.n_targets})"
                ),
                budget_after=budget.spent,
            )

        if risk.reversibility is Reversibility.COMPENSABLE:
            return Ruling(
                decision=Decision.APPLY_WITH_ROLLBACK,
                reason=(
                    f"{tool} is compensable within budget; inverse "
                    f"{risk.rollback_tool!r} recorded before apply"
                ),
                rollback_tool=risk.rollback_tool,
                budget_after=budget.spent + cost,
            )

        return Ruling(
            decision=Decision.AUTO_APPLY,
            reason=f"{tool} is reversible, weight {cost} within ceiling {self._auto_ceiling}",
            budget_after=budget.spent + cost,
        )
