"""nmd_monitor config — env-first, no secrets (mirrors ServiceConfig style)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from nmd_host.utils.env_helpers import env_bool as _env_bool
from nmd_host.utils.env_helpers import env_float as _env_float
from nmd_host.utils.env_helpers import env_int as _env_int


@dataclass(frozen=True)
class MonitorConfig:
    """Runtime switches for the monitor collector."""

    enabled: bool = True
    project: str = "default"
    sample_rate: float = 1.0
    redact: bool = True
    max_body_chars: int = 2000
    retention_days: int = 90

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        rate = _env_float("NMD_MONITOR_SAMPLE_RATE", 1.0)
        rate = min(1.0, max(0.0, rate))
        return cls(
            enabled=_env_bool("NMD_MONITOR_ENABLED", True),
            project=os.environ.get("NMD_MONITOR_PROJECT", "default").strip() or "default",
            sample_rate=rate,
            redact=_env_bool("NMD_MONITOR_REDACT", True),
            max_body_chars=_env_int("NMD_MONITOR_MAX_BODY_CHARS", 2000),
            retention_days=_env_int("NMD_MONITOR_RETENTION_DAYS", 90),
        )


__all__ = ["MonitorConfig"]
