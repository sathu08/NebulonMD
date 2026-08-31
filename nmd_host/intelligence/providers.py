"""LLM provider interface + swappable adapters.

All adapters are optional-dependency: the library is only imported when the
provider is constructed, so the rule-based pipeline never needs them. Adapters
are selected via ``NMD_LLM_PROVIDER`` (openai | anthropic | gemini | qwen |
nvidia | ollama) plus the matching API key/model env vars.

Every adapter returns a standardized ``LLMResponse`` (no provider-specific
response shape) and raises only the typed ``LLM*Error`` hierarchy below, so
Step 6 callers (agent runtime) never depend on an SDK-specific contract.
``provider_from_env()`` wraps the adapter in bounded, backoff rate-limit
retries (``NMD_LLM_MAX_RETRIES``).
"""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel
from typing import Optional, Protocol, Type

from nmd_host.utils.env_helpers import env_float as _env_float
from nmd_host.utils.env_helpers import env_int as _env_int


_CONFIG_ERROR = (
    "{name} provider: set NMD_LLM_PROVIDER={provider} and the required "
    "key/model (see nmd_host/intelligence/providers.py)"
)


class LLMProviderError(RuntimeError):
    """Base class for every LLM failure (predictable for Step 6 callers)."""

    pass


class LLMTimeoutError(LLMProviderError):
    """The provider did not answer within its configured time budget."""

    pass


class LLMUnavailableError(LLMProviderError):
    """The provider endpoint was unreachable or returned a 5xx."""

    pass


class LLMRateLimitError(LLMProviderError):
    """The provider rate-limited the request; retry after backoff."""

    pass


class LLMTokenLimitError(LLMProviderError):
    """Request/context exceeded the provider's token or context window."""

    pass


class LLMInvalidResponseError(LLMProviderError):
    """The provider replied with something unusable (not the requested shape)."""

    pass


class LLMResponse(BaseModel):
    """Common provider response: text plus provenance of which provider.

    Adapters never leak SDK-specific shapes; they map everything into this
    model, so the extraction layer and Step 6 agent only ever see ``.text``.
    """

    text: str = ""
    provider: str = ""


def _classify_sdk_error(exc: Exception) -> LLMProviderError:
    """Map an SDK exception to the typed ``LLM*Error`` contract.

    SDKs are optional imports, so classification is by exception type name
    plus message markers rather than importing vendor modules.
    """
    name = type(exc).__name__.lower()
    message = str(exc)
    if "timeout" in name or "deadline" in name:
        return LLMTimeoutError(f"LLM request timed out ({type(exc).__name__}: {message})")
    if "ratelimit" in name or message.startswith("429"):
        return LLMRateLimitError(f"LLM rate limit exceeded ({type(exc).__name__}: {message})")
    if "token" in name or "context" in name or "truncation" in name:
        return LLMTokenLimitError(
            f"LLM context/token limit reached ({type(exc).__name__}: {message})"
        )
    if (
        "connection" in name
        or "unavailable" in name
        or "internalservererror" in name
        or "connecttimeout" in name
    ):
        return LLMUnavailableError(f"LLM provider unavailable ({type(exc).__name__}: {message})")
    return LLMProviderError(f"LLM provider error: {type(exc).__name__}: {message}")


class LLMProvider(Protocol):
    """Common contract: complete a prompt and parse structured JSON."""

    def complete(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        ...

    def structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        """Return the LLM reply parsed to a plain JSON dict.

        Validation of the *contents* is the extraction layer's job, so a
        partially-invalid reply can be sanitized item by item instead of
        being rejected wholesale.
        """
        ...


def _extract_json(raw: str) -> dict:
    """Parse model JSON out of an LLM reply (tolerates fences/prose)."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise LLMInvalidResponseError(f"no JSON object found in LLM reply: {raw[:120]!r}")
    try:
        data = json.loads(text[first: last + 1])
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(f"invalid JSON from LLM: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMInvalidResponseError(f"LLM reply is not a JSON object: {raw[:120]!r}")
    return data


def _parse_json(raw: str, schema: Type[BaseModel]) -> BaseModel:
    return schema.model_validate(_extract_json(raw))


class BaseLLMProvider:
    """Shared ``structured()`` for adapters that only implement ``complete``."""

    def structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        system = (
            "You are a memory extraction system. Return ONLY valid JSON "
            "matching the requested schema. No markdown, no prose."
        )
        raw = self.complete(prompt, system=system, temperature=0.0)
        text = raw.text if isinstance(raw, LLMResponse) else raw
        return _extract_json(text)


class RetryingLLMProvider:
    """Bounded, exponential-backoff retry on ``LLMRateLimitError``.

    Wraps any ``LLMProvider`` so the extraction layer and Step 6 agent get
    predictable rate-limit handling without embedding backoff in the agent.
    Only rate limits are retried (transient by nature); timeouts and
    invalid responses surface immediately to the caller.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self._provider = provider
        self._max_retries = max(0, max_retries)
        self._base_delay = max(0.0, base_delay)

    @property
    def provider(self) -> str:
        return getattr(self._provider, "provider", "")

    @property
    def model(self) -> Optional[str]:
        return getattr(self._provider, "model", None)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                return self._provider.complete(
                    prompt, system=system, temperature=temperature
                )
            except LLMRateLimitError as exc:
                if attempt + 1 < attempts:
                    delay = self._base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise exc
        raise LLMRateLimitError("unreachable")

    def structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        return self._provider.structured(prompt, schema)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / any OpenAI-compatible endpoint."""

    provider = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model = model or os.environ.get("NMD_LLM_MODEL", "gpt-4o-mini")
        key = api_key or os.environ.get("NMD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="OpenAI", provider="openai"))
        from openai import OpenAI  # optional dependency

        self._client = OpenAI(api_key=key, base_url=base_url)

    def complete(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        extra_body = getattr(self, "_extra_body", None)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=_env_float("NMD_LLM_TIMEOUT", 60.0),
                **({"extra_body": extra_body} if extra_body else {}),
            )
        except Exception as exc:
            raise _classify_sdk_error(exc) from exc
        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider,
        )


class OllamaProvider(OpenAIProvider):
    """Local Ollama through its OpenAI-compatible endpoint."""

    provider = "ollama"

    def __init__(self, model: Optional[str] = None, base_url: str = "http://localhost:11434/v1") -> None:
        super().__init__(
            model=model or os.environ.get("NMD_LLM_MODEL", "qwen2.5"),
            api_key="ollama",
            base_url=base_url,
        )


class NvidiaProvider(OpenAIProvider):
    """NVIDIA NIM via the AI Integrate API's OpenAI-compatible endpoint."""

    provider = "nvidia"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("NMD_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="NVIDIA", provider="nvidia"))
        super().__init__(
            model=model or os.environ.get("NMD_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            api_key=key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        thinking_on = os.environ.get("NMD_LLM_THINKING", "").strip().lower() in ("1", "true", "yes")
        if not thinking_on:
            self._extra_body = {"chat_template_kwargs": {"thinking": False}}


class QwenProvider(OpenAIProvider):
    """Alibaba Qwen via DashScope's OpenAI-compatible endpoint."""

    provider = "qwen"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("NMD_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="Qwen", provider="qwen"))
        super().__init__(
            model=model or os.environ.get("NMD_LLM_MODEL", "qwen-plus"),
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude."""

    provider = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.environ.get("NMD_LLM_MODEL", "claude-sonnet-4-20250514")
        key = api_key or os.environ.get("NMD_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="Anthropic", provider="anthropic"))
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Failed to import anthropic") from e

        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise _classify_sdk_error(exc) from exc
        return LLMResponse(
            text="".join(block.text for block in response.content if block.type == "text"),
            provider=self.provider,
        )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini."""

    provider = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.environ.get("NMD_LLM_MODEL", "gemini-2.0-flash")
        key = api_key or os.environ.get("NMD_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="Gemini", provider="gemini"))
        from google import genai  # optional dependency

        self._client = genai.Client(api_key=key)

    def complete(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        from google import genai

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system, temperature=temperature
                ),
            )
        except Exception as exc:
            raise _classify_sdk_error(exc) from exc
        return LLMResponse(text=response.text or "", provider=self.provider)


_PROVIDERS = {
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "qwen": QwenProvider,
    "nvidia": NvidiaProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def provider_from_env(max_retries: Optional[int] = None) -> LLMProvider:
    """Build the provider named by ``NMD_LLM_PROVIDER`` from environment config.

    The adapter is wrapped in a ``RetryingLLMProvider`` (bounded backoff on
    rate limits, ``NMD_LLM_MAX_RETRIES`` attempts) unless ``max_retries=0``.
    """
    name = os.environ.get("NMD_LLM_PROVIDER", "").strip().lower()
    if name not in _PROVIDERS:
        raise LLMProviderError(
            f"NMD_LLM_PROVIDER={name or '(unset)'!r}; expected one of "
            f"{sorted(_PROVIDERS)}"
        )
    provider = _PROVIDERS[name]()
    if max_retries is None:
        max_retries = _env_int("NMD_LLM_MAX_RETRIES", 3)
    if max_retries > 0:
        return RetryingLLMProvider(provider, max_retries=max_retries)
    return provider


__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "GeminiProvider",
    "LLMInvalidResponseError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMTokenLimitError",
    "LLMUnavailableError",
    "NvidiaProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "QwenProvider",
    "RetryingLLMProvider",
    "provider_from_env",
]
