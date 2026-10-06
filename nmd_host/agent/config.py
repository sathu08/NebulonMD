"""Step 6 — Agent Runtime configuration (``NMD_AGENT_*`` environment).

The default system prompt lives in ``.prompt`` (Step 6.13 behavior policy);
this module only wires it into ``AgentConfig`` and honours the optional
``NMD_AGENT_SYSTEM_PROMPT`` override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nmd_host.utils.env_helpers import env_int as _env_int

from .prompt import DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs: model, loop bound and recall depth (all env-driven)."""

    model: str = ""
    max_turns: int = 4
    max_recall: int = 5
    temperature: float = 0.0
    max_sessions_per_user: int = 100
    session_ttl_seconds: int = 3600
    durable_sessions: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Step 6.14 — deterministic intent router (eval suggestion #1): run
    # ``recall`` before the first LLM turn for memory questions so tool
    # selection no longer depends on model variance.
    router_enabled: bool = True
    # Suggestion #4 — short-answer path: memory answers in 1-2 sentences
    # from recalled context instead of open-ended reasoning.
    concise_memory_answers: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name, "").strip().lower()
            if not raw:
                return default
            return raw in ("1", "true", "yes", "on")

        return cls(
            model=os.environ.get("NMD_AGENT_MODEL", "").strip(),
            max_turns=_env_int("NMD_AGENT_MAX_TURNS", 4),
            max_recall=_env_int("NMD_AGENT_MAX_RECALL", 5),
            temperature=float(os.environ.get("NMD_AGENT_TEMPERATURE", "0.0")),
            max_sessions_per_user=_env_int("NMD_AGENT_MAX_SESSIONS", 100),
            session_ttl_seconds=_env_int("NMD_AGENT_SESSION_TTL_SECONDS", 3600),
            durable_sessions=(
                os.environ.get("NMD_AGENT_DURABLE_SESSIONS", "").strip().lower()
                in ("1", "true", "yes", "on")
            ),
            system_prompt=os.environ.get(
                "NMD_AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT
            ),
            router_enabled=_env_bool("NMD_AGENT_ROUTER", True),
            concise_memory_answers=_env_bool("NMD_AGENT_CONCISE_ANSWERS", True),
        )


__all__ = ["AgentConfig", "DEFAULT_SYSTEM_PROMPT"]