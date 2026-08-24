"""14.1/14.2 Background scheduler — runs agents on a schedule in-process.

A tiny cron scheduler (5-field expressions, e.g. ``"0 2 * * *"`` for
every night at 2:00 AM). It runs registered jobs on their schedule while
the NebulonMind service is up; job state (last run, last result, error) is
kept in memory for ``/background/status`` — no new persistent store.

Design notes:
* Jobs are plain callables returning a ``dict`` (agent reports).
* ``BackgroundScheduler.start()`` must be awaited from the FastAPI
  lifespan; ``stop()`` cancels the loop. This keeps the scheduler fully
  opt-in and testable without a real clock.
* ``due(job, now)`` is pure and unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("nmd_host.agents.scheduler")


@dataclass
class CronSpec:
    """Minimal 5-field cron: minute hour day-of-month month day-of-week.

    Supports ``*``, exact values, ``a,b,c`` lists and ``*/n`` steps per
    field. Used purely to decide *when a job is due* (minute granularity).
    """

    minute: Any = "*"
    hour: Any = "*"
    dom: Any = "*"
    month: Any = "*"
    dow: Any = "*"

    @classmethod
    def parse(cls, expression: str) -> "CronSpec":
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"cron expression must have 5 fields, got {len(parts)}: {expression!r}"
            )
        return cls(
            minute=_parse_field(parts[0], 0, 59),
            hour=_parse_field(parts[1], 0, 23),
            dom=_parse_field(parts[2], 1, 31),
            month=_parse_field(parts[3], 1, 12),
            dow=_parse_field(parts[4], 0, 6),
        )

    def matches(self, now: "time.struct_time") -> bool:
        return (
            now.tm_min in self.minute
            and now.tm_hour in self.hour
            and now.tm_mday in self.dom
            and now.tm_mon in self.month
            and now.tm_wday in self.dow
        )


def _parse_field(expr: str, low: int, high: int):
    if expr == "*":
        return set(range(low, high + 1))
    values: set = set()
    for part in expr.split(","):
        if part == "*":
            values.update(range(low, high + 1))
        elif "/" in part:
            base, step = part.split("/", 1)
            step = int(step)
            start = low if base == "*" else int(base)
            values.update(range(start, high + 1, step))
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            values.update(range(a, b + 1))
        else:
            values.add(int(part))
    return values


@dataclass
class BackgroundJob:
    """A scheduled job and its in-memory execution state."""

    job_id: str
    fn: Callable[[], Dict[str, Any]]
    cron: CronSpec
    last_run: Optional[float] = None
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    runs: int = 0
    _last_minute: Optional[str] = field(default=None, repr=False)

    def due(self, now: time.struct_time) -> bool:
        key = f"{now.tm_yday}-{now.tm_hour}:{now.tm_min}"
        if self._last_minute == key:
            return False
        return self.cron.matches(now)

    def mark_done(self, now: time.struct_time, result: Optional[Dict], error: Optional[str]):
        key = f"{now.tm_yday}-{now.tm_hour}:{now.tm_min}"
        self._last_minute = key
        self.last_run = time.time()
        self.runs += 1
        if error:
            self.last_error = error
            self.last_result = None
        else:
            self.last_error = None
            self.last_result = result


class BackgroundScheduler:
    """In-process scheduler: register jobs, then start/stop the loop.

    An optional ``store`` (``JobStateStore``, see ``.state_store``) persists
    ``last_run``/``last_result``/``last_error``/``runs`` so a service restarts
    does not reset job history. Persistence is best-effort: any store failure
    is logged and the scheduler keeps its in-memory state.
    """

    def __init__(
        self,
        poll_seconds: float = 30.0,
        store: Optional["JobStateStore"] = None,
    ) -> None:
        self._jobs: Dict[str, BackgroundJob] = {}
        self._poll_seconds = poll_seconds
        self._store = store
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Registration                                                       #
    # ------------------------------------------------------------------ #

    def register(
        self,
        job_id: str,
        fn: Callable[[], Dict[str, Any]],
        schedule: str,
    ) -> None:
        if job_id in self._jobs:
            logger.warning("job %s already registered; skipping duplicate registration", job_id)
            return
        job = BackgroundJob(
            job_id=job_id,
            fn=fn,
            cron=CronSpec.parse(schedule),
        )
        self._jobs[job_id] = job
        logger.info("background job registered: %s (%s)", job_id, schedule)

    def unregister(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    # ------------------------------------------------------------------ #
    # State persistence (best-effort)                                    #
    # ------------------------------------------------------------------ #

    def _hydrate_state(self) -> None:
        store = self._store
        if store is None:
            return
        try:
            states = store.load()
        except Exception as exc:  # best-effort durability
            logger.warning("failed to load persisted job state: %s", exc)
            return
        for job in self._jobs.values():
            state = states.get(job.job_id)
            if not state:
                continue
            job.last_run = state.get("last_run") or job.last_run
            job.last_result = state.get("last_result", job.last_result)
            job.last_error = state.get("last_error", job.last_error)
            job.runs = int(state.get("runs", job.runs))

    def _persist_state(self, job: BackgroundJob) -> None:
        store = self._store
        if store is None:
            return
        state = {
            "last_run": job.last_run,
            "last_result": job.last_result,
            "last_error": job.last_error,
            "runs": job.runs,
        }
        try:
            store.save(job.job_id, state)
        except Exception as exc:  # best-effort durability
            logger.warning(
                "failed to persist state for job %s: %s", job.job_id, exc
            )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._hydrate_state()
        self._task = asyncio.create_task(self._loop())
        logger.info("background scheduler started (poll %ss)", self._poll_seconds)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("background scheduler stopped")

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            for job in list(self._jobs.values()):
                now = time.localtime()
                if job.due(now):
                    await self._run(job, now)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._poll_seconds
                )
            except asyncio.TimeoutError:
                continue

    async def _run(self, job: BackgroundJob, now) -> None:
        logger.info("background job running: %s", job.job_id)
        try:
            result = await asyncio.to_thread(job.fn)
            job.mark_done(now, result, None)
        except Exception as exc:  # jobs must never kill the loop
            logger.exception("background job %s failed", job.job_id)
            job.mark_done(now, None, f"{type(exc).__name__}: {exc}")
        self._persist_state(job)

    # ------------------------------------------------------------------ #
    # Status                                                             #
    # ------------------------------------------------------------------ #

    def run_now(self, job_id: str) -> Dict[str, Any]:
        """Synchronously run one job (used by manual API triggers)."""
        job = self._jobs[job_id]
        result = job.fn()
        job.mark_done(time.localtime(), result, None)
        self._persist_state(job)
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "poll_seconds": self._poll_seconds,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "schedule": _cron_string(job.cron),
                    "last_run": job.last_run,
                    "last_error": job.last_error,
                    "runs": job.runs,
                    "last_result": job.last_result,
                }
                for job in self._jobs.values()
            ],
        }


def _cron_string(cron: CronSpec) -> str:
    return " ".join(
        _render_field(f) for f in (cron.minute, cron.hour, cron.dom, cron.month, cron.dow)
    )


def _render_field(values) -> str:
    if values == set(range(0, 60)):
        return "*"
    if values == set(range(0, 24)):
        return "*"
    if values == set(range(1, 32)):
        return "*"
    if values == set(range(1, 13)):
        return "*"
    if values == set(range(0, 7)):
        return "*"
    return ",".join(str(v) for v in sorted(values))


__all__ = ["BackgroundScheduler", "CronSpec"]