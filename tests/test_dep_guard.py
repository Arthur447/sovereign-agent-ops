"""The gateway is the only door to a model — asserted, not trusted.

`gateway/base.py` states that no module outside `sovops.gateway` reaches a
model endpoint directly. Stated in a docstring, that is a convention
someone has to remember; here it fails in CI instead.

The rule is deliberately narrower than "nobody imports an HTTP client":
`a2a/client.py` legitimately speaks HTTP to a peer agent, and forbidding
that would be a rule people route around. What is forbidden is reaching
*inference* outside the one place that applies the routing policy, meters
the cost, and enforces the sovereign posture.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "sovops"
GATEWAY = SRC / "gateway"

# Vendor inference SDKs. An import here routes a call past the routing
# policy, the cost meter, and the sovereignty check in one move.
VENDOR_SDKS = frozenset(
    {
        "anthropic",
        "openai",
        "ollama",
        "mistralai",
        "cohere",
        "litellm",
        "transformers",
        "vllm",
        "huggingface_hub",
    }
)

# Addressing one of these is calling a model, whatever client does it —
# and this repo calls models over raw HTTP rather than through an SDK, so
# the endpoint path is the real signal, not the import.
MODEL_ENDPOINTS = ("/chat/completions", "/v1/messages", "/api/generate", "/v1/completions")


def modules_outside_the_gateway() -> list[Path]:
    return [p for p in sorted(SRC.rglob("*.py")) if GATEWAY not in p.parents]


def imported_roots(source: str) -> set[str]:
    """Top-level package of every import in a module.

    Parsed rather than grepped: a comment mentioning `anthropic` is not an
    import, and a guard that fires on prose gets disabled within a week.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_vendor_inference_sdk_outside_the_gateway():
    offenders = [
        f"{path.relative_to(SRC)} imports {root!r}"
        for path in modules_outside_the_gateway()
        for root in sorted(imported_roots(path.read_text(encoding="utf-8")) & VENDOR_SDKS)
    ]
    assert not offenders, (
        "inference must go through sovops.gateway, which is where the routing "
        "policy, the cost meter and the sovereign check live:\n  " + "\n  ".join(offenders)
    )


def test_no_model_endpoint_is_addressed_outside_the_gateway():
    offenders = [
        f"{path.relative_to(SRC)} addresses {endpoint!r}"
        for path in modules_outside_the_gateway()
        for endpoint in MODEL_ENDPOINTS
        if endpoint in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "a model endpoint reached outside the gateway is an unmetered, "
        "unrouted call:\n  " + "\n  ".join(offenders)
    )


def test_the_gateway_itself_holds_the_door():
    """The rule is meaningless unless inference actually happens somewhere.

    Without this, both guards above would pass on a repo that calls no
    model at all — "nobody bypasses the gateway" and "nobody uses a model"
    are not the same claim, and only one of them is worth asserting.
    """
    inside = "\n".join(p.read_text(encoding="utf-8") for p in GATEWAY.rglob("*.py"))
    assert any(endpoint in inside for endpoint in MODEL_ENDPOINTS), (
        "no model endpoint inside sovops.gateway — the guards are passing "
        "vacuously"
    )


def test_the_guard_can_actually_fail():
    """A guard that cannot fail is decoration.

    Asserts the detector on a module that violates the rule, so a refactor
    that quietly breaks the parsing turns this file green for the wrong
    reason.
    """
    assert imported_roots("import anthropic") & VENDOR_SDKS
    assert imported_roots("from openai import OpenAI") & VENDOR_SDKS
    assert not imported_roots("# anthropic is mentioned in a comment") & VENDOR_SDKS
