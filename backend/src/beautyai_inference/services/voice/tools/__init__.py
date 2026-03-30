"""
Voice Tools Package.

Provides callable tools for LLM during voice conversations.
"""

from .appointment_tools import (
    VoiceTool,
    VoiceToolExecutor,
    ToolCategory,
    VOICE_TOOLS,
    get_tools_for_openai,
    get_tool,
    tool_allows_interruption,
    get_customer_service_system_prompt,
    CUSTOMER_SERVICE_SYSTEM_PROMPT
)

__all__ = [
    "VoiceTool",
    "VoiceToolExecutor",
    "ToolCategory",
    "VOICE_TOOLS",
    "get_tools_for_openai",
    "get_tool",
    "tool_allows_interruption",
    "get_customer_service_system_prompt",
    "CUSTOMER_SERVICE_SYSTEM_PROMPT"
]
