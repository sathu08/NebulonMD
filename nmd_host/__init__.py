"""nmd_host — memory layer for AI assistants on top of NebulonDB.

This package is the *client host* for NebulonMind: every store and the
``NebulonMind`` facade talk to the NebulonDB REST API (``NEBULONDB_API_PORT``,
default 6969, see ``.env``) — no ``ndb_host`` Python code is imported.

The package also ships its own service: ``nmd_host.api.server`` exposes
``/api/NebulonMind/...`` endpoints, run with ``python -m nmd_host.serve``
(listens on ``NEBULONDMIND_API_PORT``, default 9696).
"""

from .agent import (
    AgentChatData,
    AgentChatRequest,
    AgentConfig,
    AgentMessage,
    AgentRuntime,
    AgentSession,
    AgentSessionManager,
    AgentToolResult,
    build_memory_toolkit,
)
from .core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    MemoryStatus,
    MemoryType,
    Priority,
    Relationship,
    RetentionPolicy,
)
from .core.registry import (
    InMemoryUserRegistry,
    NebulonDBUserRegistry,
    UserNotFoundError,
    UserRegistry,
)
from .core.repository import (
    InMemoryRepository,
    MemoryRepository,
    NebulonMindRepository,
)
from .core.mind import NebulonMind
from .api.server import create_app
from .intelligence import (
    Conversation,
    LLMExtractor,
    LLMProviderError,
    MemoryCandidate,
    MemoryCategory,
    MemoryDecision,
    MemoryDecisionEngine,
    MemoryIntelligence,
    RuleBasedExtractor,
    provider_from_env,
)
from .lifecycle import (
    ConsolidationDecision,
    DuplicateDetector,
    DuplicateResult,
    ForgettingManager,
    MemoryContextBuilder,
    MemoryLifecycleManager,
    MemoryRanker,
    MemoryRetriever,
    MemoryState,
    RankingConfig,
    RetrievalConfig,
    build_lifecycle_manager,
)

__version__ = "v0.1"

__all__ = [
    "AgentChatData",
    "AgentChatRequest",
    "AgentConfig",
    "AgentMessage",
    "AgentRuntime",
    "AgentSession",
    "AgentSessionManager",
    "AgentToolResult",
    "Classification",
    "ConsolidationDecision",
    "Conversation",
    "DuplicateDetector",
    "DuplicateResult",
    "ForgettingManager",
    "Importance",
    "InMemoryRepository",
    "InMemoryUserRegistry",
    "LLMExtractor",
    "LLMProviderError",
    "Lifecycle",
    "Memory",
    "MemoryCandidate",
    "MemoryCategory",
    "MemoryContent",
    "MemoryContextBuilder",
    "MemoryDecision",
    "MemoryDecisionEngine",
    "MemoryIntelligence",
    "MemoryLifecycleManager",
    "MemoryRanker",
    "MemoryRepository",
    "MemoryRetriever",
    "MemoryState",
    "MemoryStatus",
    "MemoryType",
    "NebulonMind",
    "NebulonDBUserRegistry",
    "NebulonMindRepository",
    "Priority",
    "RankingConfig",
    "Relationship",
    "RetentionPolicy",
    "RetrievalConfig",
    "RuleBasedExtractor",
    "UserNotFoundError",
    "UserRegistry",
    "build_lifecycle_manager",
    "build_memory_toolkit",
    "create_app",
    "provider_from_env",
]
