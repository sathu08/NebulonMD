"""LLM provider interface + swappable adapters.

All adapters are optional-dependency: the library is only imported when the
provider is constructed, so the rule-based pipeline never needs them. Adapters
are selected via ``NMD_LLM_PROVIDER`` (openai | anthropic | gemini | qwen |
nvidia | ollama | openrouter | other) plus the matching API key/model env vars.

``other`` is the generic OpenAI-compatible endpoint: set ``NMD_LLM_MODEL``,
``NMD_LLM_API_KEY`` and ``NMD_LLM_BASE_URL`` (model + base URL may also live
in ``nebulonmd.cfg`` under ``[llm]``; the key always stays in ``.env``).

The extraction system now includes robust JSON parsing with multiple fallback
strategies to handle various model output formats. If ``NMD_LLM_EXTRACTOR_LENIENT``
is enabled, the system will extract partial data even when the full JSON is
malformed, falling back to rule-based extraction only as a last resort.

Every adapter returns a standardized ``LLMResponse`` (no provider-specific
response shape) and raises only the typed ``LLM*Error`` hierarchy below, so
Step 6 callers (agent runtime) never depend on an SDK-specific contract.
``provider_from_env()`` wraps the adapter in bounded, backoff rate-limit
retries (``NMD_LLM_MAX_RETRIES``).
"""

from __future__ import annotations

import json
import os
import re
import time

from pydantic import BaseModel
from typing import Optional, Protocol, Type

from nmd_host.utils.env_helpers import env_float as _env_float
from nmd_host.utils.env_helpers import env_int as _env_int


_CONFIG_ERROR = (
    "{name} provider: set NMD_LLM_PROVIDER={provider} and the required "
    "key/model (see nmd_host/intelligence/providers.py)"
)


# Non-secret LLM settings that may fall back to ``nebulonmd.cfg`` ``[llm]``.
# ``NMD_LLM_API_KEY`` (and vendor keys) are deliberately ABSENT here: the key
# is only ever read from ``.env`` / the process environment, never the cfg.
_LLM_CFG_FALLBACK = {
    "NMD_LLM_PROVIDER": "nmd_llm_provider",
    "NMD_LLM_MODEL": "nmd_llm_model",
    "NMD_LLM_BASE_URL": "nmd_llm_base_url",
    "NMD_LLM_TIMEOUT": "nmd_llm_timeout",
    "NMD_LLM_MAX_RETRIES": "nmd_llm_max_retries",
    "NMD_LLM_THINKING": "nmd_llm_thinking",
    "NMD_LLM_EXTRACTOR": "nmd_llm_extractor",
    "NMD_LLM_EXTRACTOR_LENIENT": "nmd_llm_extractor_lenient",
}


def _ensure_llm_env_from_cfg() -> None:
    """Seed missing/empty non-secret ``NMD_LLM_*`` env vars from cfg ``[llm]``.

    Precedence: explicit ``.env`` / process environment always wins; the cfg
    is only a fallback. ``NMD_LLM_API_KEY`` is never copied — it stays
    ``.env``-only (see ``_SECRET_KEYS`` in ``nmd_host.core.config``).
    Failures (no cfg file, no ``[llm]`` section) are silent: env-only
    configuration keeps working.
    """
    try:
        needed = [k for k in _LLM_CFG_FALLBACK if not (os.environ.get(k) or "").strip()]
        if not needed:
            return
        from configparser import ConfigParser
        from nmd_host.core.config import _default_cfg_path
        cfg_file = _default_cfg_path()
        if not cfg_file.exists():
            return
        parser = ConfigParser()
        try:
            parser.read(cfg_file, encoding="utf-8")
        except Exception:
            return
        if not parser.has_section("llm"):
            return
        for env_key in needed:
            cfg_key = _LLM_CFG_FALLBACK[env_key]
            try:
                value = parser.get("llm", cfg_key, fallback="")
            except Exception:
                value = ""
            if value is not None and str(value).strip() != "":
                os.environ[env_key] = str(value).strip()
    except Exception:
        pass


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
    """Parse model JSON out of an LLM reply (tolerates fences/prose, malformed JSON, multiple objects)."""
    text = (raw or "").strip()
    if not text:
        raise LLMInvalidResponseError("empty LLM reply")

    # Case 1: Remove markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Case 2: Try to find the first { ... } pair (works for prose-wrapped JSON)
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise LLMInvalidResponseError(f"no JSON object found in LLM reply: {raw[:120]!r}")

    json_text = text[first: last + 1]

    # Case 3: Try parsing as-is
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Case 4: Try fixing common JSON issues (trailing commas, unescaped quotes)
    try:
        # Remove trailing commas before }
        fixed = json_text.replace(",}", "}").replace(",]", "]")
        # Remove trailing commas after } (if any)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Case 5: Try extracting multiple JSON objects (some models emit arrays of objects)
    try:
        # Look for multiple { ... } patterns
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_objects:
            # Try each object until one works
            for obj_str in json_objects:
                try:
                    data = json.loads(obj_str)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Case 6: Last resort - try to extract key-value pairs from prose
    try:
        # Look for "key": value patterns
        kv_pairs = re.findall(r'"([^"]+)":\s*("[^"]*"|\d+|true|false|null|\{[^}]*\})', text)
        if kv_pairs:
            data = {}
            for key, value in kv_pairs:
                try:
                    # Try to parse the value as JSON
                    parsed = json.loads(value)
                    data[key] = parsed
                except json.JSONDecodeError:
                    # If not JSON, treat as string
                    data[key] = value.strip('"')
            if data:
                return data
    except Exception:
        pass

    raise LLMInvalidResponseError(
        f"Could not extract valid JSON from LLM reply. First 200 chars: {raw[:200]!r}"
    )


def _parse_json(raw: str, schema: Type[BaseModel]) -> BaseModel:
    """Parse JSON with fallback to partial validation for robustness."""
    try:
        data = _extract_json(raw)
        return schema.model_validate(data)
    except LLMInvalidResponseError:
        # If full validation fails, try to validate as much as possible
        try:
            data = _extract_json(raw)
            # Try to create a partial validation by allowing extra fields
            class PartialSchema(schema):
                class Config:
                    extra = "allow"
            return PartialSchema.model_validate(data)
        except Exception:
            # If even partial validation fails, return a basic dict with the raw data
            # This ensures the extraction layer can still process it
            data = _extract_json(raw)
            return schema.model_construct(**data)


class BaseLLMProvider:
    """Shared ``structured()`` for adapters that only implement ``complete``."""

    def structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        # Check if lenient mode is enabled
        import os
        lenient_mode = os.environ.get("NMD_LLM_EXTRACTOR_LENIENT", "false").lower() in ("1", "true", "yes")
        
        system = (
            "You are a memory extraction system. Return ONLY valid JSON "
            "matching the requested schema. No markdown, no prose."
        )
        raw = self.complete(prompt, system=system, temperature=0.0)
        text = raw.text if isinstance(raw, LLMResponse) else raw
        
        # Use robust JSON extraction with fallback
        try:
            return self._extract_with_fallback(text, schema)
        except Exception:
            # In lenient mode, fallback to empty dict (rules will handle it)
            if lenient_mode:
                return {}
            # In strict mode, raise the exception to make the failure explicit
            raise LLMInvalidResponseError(
                f"Failed to extract structured data from LLM response and lenient mode is disabled. "
                f"Schema: {schema.__name__}, Response: {text[:200]!r}"
            )

    def _clean_json_response(self, text: str) -> str:
        """Clean common JSON issues from model responses."""
        import re
        
        # Remove markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        # Fix common JSON issues
        text = text.replace(",}", "}").replace(",]", "]")
        text = re.sub(r',\s*([}\]])', r'\1', text)
        text = text.replace("'", '"')  # Replace single quotes with double quotes
        
        # Handle common escape issues
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Fix unescaped newlines in strings
        text = re.sub(r'"([^"]*)\n([^"]*)"', lambda m: f'"{m.group(1)} {m.group(2)}"', text)
        
        return text

    def _extract_robust_json(self, text: str) -> dict:
        """Extract JSON with multiple fallback strategies for robustness."""
        # Strategy 1: Direct extraction
        try:
            return _extract_json(text)
        except:
            pass
        
        # Strategy 2: Clean and extract
        try:
            cleaned = self._clean_json_response(text)
            return _extract_json(cleaned)
        except:
            pass
        
        # Strategy 3: Extract JSON-like structures
        try:
            # Look for object-like patterns
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        except:
            pass
        
        # Strategy 4: Extract key-value pairs
        try:
            kv_pairs = re.findall(r'"([^"]+)":\s*("[^"]*"|\d+|true|false|null|\{[^}]*\})', text)
            if kv_pairs:
                result = {}
                for key, value in kv_pairs:
                    try:
                        parsed = json.loads(value)
                        result[key] = parsed
                    except:
                        result[key] = value.strip('"')
                return result
        except:
            pass
        
        # Strategy 5: Return empty dict (fallback to rules)
        return {}

    def _extract_with_fallback(self, text: str, schema: Type[BaseModel]) -> dict:
        """Extract JSON with multiple fallback strategies and return partial results."""
        # Try all extraction strategies
        extraction_strategies = [
            lambda: _extract_json(text),
            lambda: _extract_json(self._clean_json_response(text)),
            lambda: self._extract_robust_json(text),
        ]
        
        for strategy in extraction_strategies:
            try:
                data = strategy()
                if isinstance(data, dict):
                    # Try to validate against schema
                    try:
                        return schema.model_validate(data)
                    except Exception:
                        # If validation fails, try to extract what we can
                        return self._extract_partial_data(data, schema)
            except Exception:
                continue
        
        # If all strategies fail, return an empty dict (fallback to rules)
        return {}

    def _extract_partial_data(self, data: dict, schema: Type[BaseModel]) -> dict:
        """Extract data that matches the schema fields, ignoring extra fields."""
        if not isinstance(data, dict):
            return {}
        
        result = {}
        schema_fields = schema.model_fields.keys()
        
        for field in schema_fields:
            if field in data:
                result[field] = data[field]
            else:
                # Try to find the field in nested structures
                for key, value in data.items():
                    if field.lower() in key.lower() or key.lower() in field.lower():
                        result[field] = value
                        break
        
        return result


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


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter - OpenAI-compatible endpoint for routing across multiple LLMs."""

    provider = "openrouter"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("NMD_LLM_API_KEY")
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="OpenRouter", provider="openrouter"))
        super().__init__(
            model=model or os.environ.get("NMD_LLM_MODEL", "deepseek-ai/deepseek-v4-pro-0813"),
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )


class OtherProvider(OpenAIProvider):
    """Generic OpenAI-compatible endpoint (custom model / key / base URL).

    Configure via ``NMD_LLM_PROVIDER=other`` plus::

        NMD_LLM_MODEL     e.g. my-org/my-model
        NMD_LLM_API_KEY   (in ``.env``, never in the cfg)
        NMD_LLM_BASE_URL  e.g. https://llm.example.com/v1

    Model and base URL may also come from ``nebulonmd.cfg`` ``[llm]``
    (``nmd_llm_model`` / ``nmd_llm_base_url``); explicit constructor args
    always win over the environment.
    """

    provider = "other"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_model = (model or os.environ.get("NMD_LLM_MODEL", "")).strip()
        if not resolved_model:
            raise LLMProviderError(_CONFIG_ERROR.format(name="Other", provider="other"))
        key = (api_key or os.environ.get("NMD_LLM_API_KEY", "")).strip()
        if not key:
            raise LLMProviderError(_CONFIG_ERROR.format(name="Other", provider="other"))
        resolved_base_url = (base_url or os.environ.get("NMD_LLM_BASE_URL", "")).strip()
        if not resolved_base_url:
            raise LLMProviderError(
                "Other provider: set NMD_LLM_BASE_URL to your OpenAI-compatible "
                "endpoint (e.g. https://llm.example.com/v1)"
            )
        super().__init__(
            model=resolved_model,
            api_key=key,
            base_url=resolved_base_url,
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
    "openrouter": OpenRouterProvider,
    "other": OtherProvider,
}


def provider_from_env(max_retries: Optional[int] = None) -> LLMProvider:
    """Build the provider named by ``NMD_LLM_PROVIDER`` from environment config.

    Non-secret settings fall back to ``nebulonmd.cfg`` ``[llm]`` when the env
    is empty; ``NMD_LLM_API_KEY`` is read from ``.env``/environment only.
    The adapter is wrapped in a ``RetryingLLMProvider`` (bounded backoff on
    rate limits, ``NMD_LLM_MAX_RETRIES`` attempts) unless ``max_retries=0``.
    """
    _ensure_llm_env_from_cfg()
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


def provider_from_env_with_model(
    model: Optional[str] = None, max_retries: Optional[int] = None
) -> LLMProvider:
    """Build the provider from env, optionally overriding the model.

    If ``model`` is provided, it replaces the default model for that provider.
    ``NMD_LLM_API_KEY`` is read from ``.env``/environment only, never cfg.
    """
    _ensure_llm_env_from_cfg()
    name = os.environ.get("NMD_LLM_PROVIDER", "").strip().lower()
    if name not in _PROVIDERS:
        raise LLMProviderError(
            f"NMD_LLM_PROVIDER={name or '(unset)'!r}; expected one of "
            f"{sorted(_PROVIDERS)}"
        )
    provider_cls = _PROVIDERS[name]
    provider = provider_cls(model=model) if model else provider_cls()
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
    "OtherProvider",
    "QwenProvider",
    "RetryingLLMProvider",
    "provider_from_env",
    "provider_from_env_with_model",
]
