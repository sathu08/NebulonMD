"""Step 6 — Agent Runtime (memory layer agent over Steps 2/3).

The runtime is a stateless tool-calling loop: an ``LLMProvider`` (Step 2)
decides between plain answers and memory tool calls (remember / recall);
every tool persists or reads exclusively through the user's repository and
lifecycle manager, so NebulonDB remains the only persistent store.
"""

from .config import AgentConfig, DEFAULT_SYSTEM_PROMPT
from .engine import AgentRuntime
from .prompt import build_system_prompt
from .session import AgentSession, AgentSessionManager
from .schemas import (
    AgentChatData,
    AgentChatRequest,
    AgentMessage,
    AgentToolCall,
    AgentToolResult,
    ExecutionTrace,
)
from .tools import (
    AgentTool,
    RecallTool,
    RememberTool,
    build_memory_toolkit,
)
from .trace import LLMSpan, RecallSpan, ToolSpan

__all__ = [
    "AgentChatData",
    "AgentChatRequest",
    "AgentConfig",
    "AgentMessage",
    "AgentRuntime",
    "AgentSession",
    "AgentSessionManager",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "DEFAULT_SYSTEM_PROMPT",
    "ExecutionTrace",
    "LLMSpan",
    "RecallSpan",
    "RecallTool",
    "RememberTool",
    "ToolSpan",
    "build_memory_toolkit",
    "build_system_prompt",
]