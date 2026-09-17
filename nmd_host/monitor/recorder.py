"""nmd_monitor recorder — sampling + redaction + fail-open capture."""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

from .config import MonitorConfig
from .models import MonitorTrace

logger = logging.getLogger("nmd_host.monitor.recorder")


class MonitorRecorder:
    """Decides what to persist and builds the trace (no I/O itself).

    I/O stays in ``store.py`` so tests can assert on the built trace
    without a backend. Every method is fail-open: never raises.
    """

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self._config = config or MonitorConfig.from_env()

    @property
    def config(self) -> MonitorConfig:
        return self._config

    def should_capture(self) -> bool:
        """Sampling gate (called before building anything expensive)."""
        try:
            if not self._config.enabled:
                return False
            if self._config.sample_rate >= 1.0:
                return True
            if self._config.sample_rate <= 0.0:
                return False
            return random.random() < self._config.sample_rate
        except Exception:
            return True

    def from_chat(
        self,
        execution_trace: Any,
        *,
        user_id: str = "",
        thread_id: str = "",
        request_id: str = "-",
        provider: str = "",
        model: str = "",
        input_text: str = "",
        answer: str = "",
        ok: bool = True,
        error: str = "",
    ) -> Optional[MonitorTrace]:
        """Build a ``MonitorTrace`` from one ``AgentRuntime.chat()`` result."""
        try:
            if not self.should_capture():
                return None
            max_chars = self._config.max_body_chars if self._config.redact else 0
            # redact=False -> keep full bodies (0 means no truncation).
            trace = MonitorTrace.from_execution_trace(
                execution_trace,
                user_id=user_id,
                thread_id=thread_id,
                project=self._config.project,
                request_id=request_id,
                provider=provider,
                model=model,
                input_text=input_text,
                answer=answer,
                max_body_chars=max_chars if max_chars > 0 else 1_000_000,
            )
            trace.ok = ok
            trace.error = (error or "")[:300]
            return trace
        except Exception as exc:  # defensive: monitor never breaks chat
            logger.warning("monitor capture skipped: %s", exc)
            return None


__all__ = ["MonitorRecorder"]
