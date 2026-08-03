"""Loads `routing.yaml` into a `RoutingPolicy`.

Separate from `base.py` so the policy object stays constructible in
tests without a file on disk, and so the YAML shape can change without
touching the enforcement logic.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from sovops.gateway.base import ClassRoute, ModelSpec, RoutingPolicy, TaskClass

DEFAULT_ROUTING_PATH = Path(__file__).resolve().parents[3] / "routing.yaml"


def load_routing_policy(path: Path | str | None = None) -> RoutingPolicy:
    raw = yaml.safe_load(Path(path or DEFAULT_ROUTING_PATH).read_text(encoding="utf-8"))

    models = {
        key: ModelSpec(
            id=spec["id"],
            backend=spec["backend"],
            sovereign=bool(spec["sovereign"]),
            eur_per_mtok_in=float(spec["eur_per_mtok_in"]),
            eur_per_mtok_out=float(spec["eur_per_mtok_out"]),
            max_output_tokens=int(spec.get("max_output_tokens", 4096)),
        )
        for key, spec in raw["models"].items()
    }

    routes: dict[TaskClass, ClassRoute] = {}
    for class_name, rule in raw["classes"].items():
        escalation_key = rule.get("escalation")
        primary = models[rule["primary"]]

        # A non-sovereign primary is a configuration error, not a choice:
        # the default path is what runs when nobody thought about it, and
        # it must be the safe one.
        if not primary.sovereign:
            raise ValueError(
                f"task class {class_name!r} declares a non-sovereign primary "
                f"({primary.id!r}); the sovereign default is not optional"
            )

        routes[TaskClass(class_name)] = ClassRoute(
            primary=primary,
            escalation=models[escalation_key] if escalation_key else None,
        )

    return RoutingPolicy(routes)
