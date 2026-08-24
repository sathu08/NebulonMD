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

    @classmethod
    def from_env(cls) -> "AgentConfig":
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
        )


__all__ = ["AgentConfig", "DEFAULT_SYSTEM_PROMPT"]