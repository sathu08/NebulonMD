"""Step 14 — Background Agents.

Changes.txt 14.1/14.2: agents that do *memory work* and *scheduled work*
without a user in the loop. They reuse the existing Step 3 lifecycle
pipeline — ``MemoryConsolidator`` for the Memory Agent, the repository +
ingest gate for the Task Agent — so this is a new top layer, not a second
consolidation/summarisation system (the architectural separation the
changes.txt calls for: the user-facing Chat Agent never hears about these).

Nothing here runs on its own: ``BackgroundScheduler`` + the API routes wire
them into the service lifecycle.
"""

from .memory_agent import MemoryAgent
from .scheduler import BackgroundScheduler, CronSpec
from .task_agent import TaskAgent

__all__ = ["BackgroundScheduler", "CronSpec", "MemoryAgent", "TaskAgent"]