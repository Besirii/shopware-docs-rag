"""LLM providers behind one tiny interface: `generate_json(system, user, schema)`.

Each provider gets structured output through its *native* mechanism, so we
never parse free text with regexes:

* Anthropic - `output_config.format` (JSON schema structured outputs). Current
  Claude models reject `temperature` and forced `tool_choice`, so neither is
  sent; `effort` controls cost/latency instead. Server-side refusal fallbacks
  are enabled on the models that support them.
* OpenAI    - `response_format={"type": "json_schema", "strict": true}`.
* Ollama    - `format=<json schema>` (grammar-constrained decoding), plus an
  explicit `num_ctx`: Ollama's default context window is small and silently
  truncates a RAG prompt from the *front* - i.e. it drops the documentation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import ClassVar, Protocol

import httpx

from app.config import LLMProvider, Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


class LLMClient(Protocol):
    provider: str
    model: str

    def generate_json(self, system: str, user: str, schema: dict, schema_name: str, max_tokens: int) -> LLMResult: ...

    def health(self) -> dict: ...


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort for models that wrap JSON in prose or code fences.
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise LLMError(f"model did not return JSON: {text[:200]!r}") from None


class AnthropicLLM:
    provider = "anthropic"
    # Models that accept the server-side refusal fallback (`fallbacks: "default"`).
    _FALLBACK_MODELS: ClassVar[set[str]] = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

    def __init__(self, model: str, api_key: str | None, effort: str, timeout_s: float,
                 workspace_id: str | None = None) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self._anthropic = anthropic
        # Keys that are not scoped to a workspace must name one on every request.
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant auth login` profile.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=3,
                                           default_headers=headers)

    def generate_json(self, system: str, user: str, schema: dict, schema_name: str, max_tokens: int) -> LLMResult:
        t0 = time.perf_counter()
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if re.match(r"claude-(opus|sonnet|haiku|fable|mythos)-5", self.model):
            output_config["effort"] = self.effort
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        try:
            if self.model in self._FALLBACK_MODELS:
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:  # 4xx/5xx after SDK retries
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"Anthropic connection error: {e}") from e

        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise LLMError(f"model refused the request (category={category})")
        if resp.stop_reason == "max_tokens":
            raise LLMError("answer hit max_tokens - raise LLM_MAX_TOKENS")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return LLMResult(
            data=_parse_json(text),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def health(self) -> dict:
        # Don't spend tokens on a health check - only verify the model id exists.
        try:
            self._client.models.retrieve(self.model)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


class OpenAILLM:
    provider = "openai"

    def __init__(self, model: str, api_key: str | None, temperature: float, seed: int, timeout_s: float,
                 base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=3)

    def generate_json(self, system: str, user: str, schema: dict, schema_name: str, max_tokens: int) -> LLMResult:
        import openai

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
                temperature=self.temperature,
                seed=self.seed,
                max_completion_tokens=max_tokens,
            )
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("answer hit max_tokens - raise LLM_MAX_TOKENS")
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"model refused: {choice.message.refusal}")
        return LLMResult(
            data=_parse_json(choice.message.content or ""),
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def health(self) -> dict:
        try:
            self._client.models.retrieve(self.model)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str, base_url: str, temperature: float, seed: int, timeout_s: float,
                 num_ctx: int = 8192, repeat_penalty: float = 1.1) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        # Small models under grammar-constrained JSON decoding sometimes loop on a
        # token until Ollama aborts ("token repeat limit reached"); a mild repeat
        # penalty prevents it, and a stronger one is used for a single retry.
        self.repeat_penalty = repeat_penalty
        self._http = httpx.Client(timeout=timeout_s)
        self._thinking: bool | None = None

    def _supports_thinking(self) -> bool:
        """Reasoning models (qwen3, ...) would spend tokens 'thinking' before the JSON.

        For grounded extraction and judging we want the direct answer, so we
        turn thinking off - but sending `think` to a model without that
        capability is an error, hence the capability check (cached).
        """
        if self._thinking is None:
            try:
                info = self._http.post(f"{self.base_url}/api/show", json={"model": self.model}, timeout=10).json()
                self._thinking = "thinking" in (info.get("capabilities") or [])
            except httpx.HTTPError:
                self._thinking = False
        return self._thinking

    def generate_json(self, system: str, user: str, schema: dict, schema_name: str, max_tokens: int) -> LLMResult:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens,
                        "num_ctx": self.num_ctx, "repeat_penalty": self.repeat_penalty},
        }
        if self._supports_thinking():
            body["think"] = False
        data = self._post_chat(body)
        if data.get("_repeat_abort"):
            log.warning("ollama aborted a repetition loop; retrying once with a stronger repeat penalty")
            body["options"] = {**body["options"], "repeat_penalty": max(1.2, self.repeat_penalty + 0.1)}
            data = self._post_chat(body)
            if data.get("_repeat_abort"):
                raise LLMError("Ollama aborted generation twice (token repeat limit)")
        if data.get("done_reason") == "length":
            raise LLMError("answer hit num_predict - raise LLM_MAX_TOKENS")
        return LLMResult(
            data=_parse_json(data["message"]["content"]),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            model=data.get("model", self.model),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _post_chat(self, body: dict) -> dict:
        try:
            r = self._http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama connection error: {e}") from e
        if r.status_code == 500 and "repeat limit" in r.text:
            return {"_repeat_abort": True}
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()

    def health(self) -> dict:
        try:
            tags = self._http.get(f"{self.base_url}/api/tags", timeout=5).json()
            names = {m["name"] for m in tags.get("models", [])}
            present = self.model in names or f"{self.model}:latest" in names
            return {"ok": present, "error": None if present else f"model {self.model} not pulled"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


def make_llm(settings: Settings, provider: LLMProvider | None = None, model: str | None = None) -> LLMClient | None:
    """Build the configured client. Returns None for the 'extractive' (no-LLM) provider."""
    provider = provider or settings.llm_provider
    model = model or (settings.resolved_llm_model if provider == settings.llm_provider else None)
    from app.config import DEFAULT_LLM_MODELS

    model = model or DEFAULT_LLM_MODELS[provider]
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    if provider == "anthropic":
        return AnthropicLLM(model, key, settings.anthropic_effort, settings.llm_timeout_s,
                            workspace_id=settings.anthropic_workspace_id)
    if provider == "openai":
        return OpenAILLM(model, key, settings.llm_temperature, settings.llm_seed, settings.llm_timeout_s,
                         base_url=settings.llm_base_url)
    if provider == "ollama":
        return OllamaLLM(model, settings.ollama_base_url, settings.llm_temperature,
                         settings.llm_seed, settings.llm_timeout_s, num_ctx=settings.ollama_num_ctx,
                         repeat_penalty=settings.ollama_repeat_penalty)
    return None
