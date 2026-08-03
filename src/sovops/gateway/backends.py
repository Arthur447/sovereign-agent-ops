"""The two concrete backends.

`OpenAICompatBackend` is the sovereign path. It speaks the OpenAI
`/v1/chat/completions` shape, which vLLM and Ollama both expose — so the
laptop development path and the production self-hosted path are the same
code with a different base URL. That equivalence is the only honest way
to build this without a GPU: what changes between here and a vLLM
deployment is a URL and a model id, not a call site.

`AnthropicBackend` is the escalation path, reached only through an
explicit opt-in with a reason (see `base.RoutingPolicy.resolve`). It
exists to make the *routing decision* real — a "model-agnostic" gateway
with one implementation proves nothing.

Both return `(text, input_tokens, output_tokens)` and nothing else.
Cost, latency and posture are the gateway's business, so a backend
cannot disagree with another about how they are computed.
"""
from __future__ import annotations

import json
import os

import httpx

DEFAULT_TIMEOUT_S = 300.0  # CPU inference on a 3B is slow; this is not a bug


class BackendError(RuntimeError):
    """The backend could not produce a completion."""


class OpenAICompatBackend:
    """vLLM / Ollama / any OpenAI-compatible endpoint.

    Structured output is requested through `response_format`, which both
    vLLM (via its guided-decoding integration) and Ollama support. When
    the server honours it, the response body is schema-valid and there is
    nothing to parse defensively. When it does not, `_extract_json`
    salvages the first JSON object — the fallback exists because a local
    3B is materially worse at instruction-following than a frontier model
    and pretending otherwise would make the eval suite measure the
    parser rather than the agent.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("SOVOPS_INFERENCE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("SOVOPS_INFERENCE_KEY", "not-needed")
        self._timeout_s = timeout_s

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"inference endpoint unreachable: {exc}") from exc

        if resp.status_code >= 400:
            raise BackendError(f"inference endpoint returned {resp.status_code}: {resp.text[:400]}")

        body = resp.json()
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        return (
            text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


class AnthropicBackend:
    """The escalation path. Called over plain HTTP, no SDK.

    No vendor SDK on purpose: an SDK import is the thing that creeps into
    call sites, and the dep-guard test that keeps this file the only door
    is easier to state and enforce when the door is `httpx`.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, *, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout_s = timeout_s

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> tuple[str, int, int]:
        if not self._api_key:
            raise BackendError(
                "ANTHROPIC_API_KEY is unset — the escalation path is configured "
                "but not credentialed. The sovereign primary still works."
            )

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if json_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }

        try:
            resp = httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
                },
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"anthropic endpoint unreachable: {exc}") from exc

        if resp.status_code >= 400:
            raise BackendError(f"anthropic returned {resp.status_code}: {resp.text[:400]}")

        body = resp.json()
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return (
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )


def extract_json(text: str) -> dict:
    """Best-effort recovery of a JSON object from a model response.

    Used only when a backend did not honour `response_format`. Kept in
    one place, and *not* inside the backends, so the eval suite can
    report how often it fired: a rising salvage rate is a signal about
    the local model, and it should be visible rather than absorbed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, depth = text.find("{"), 0
        if start == -1:
            raise BackendError(f"no JSON object in response: {text[:200]!r}") from None
        for i, ch in enumerate(text[start:], start):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise BackendError(f"malformed JSON object: {exc}") from exc
                break
        else:
            raise BackendError("unterminated JSON object in response") from None

    if not isinstance(parsed, dict):
        raise BackendError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
