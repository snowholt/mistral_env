"""
Voice services for speech processing functionality.

This module provides speech-to-text and text-to-speech capabilities
using optimized models and services.

Includes:
- Voice conversation management
- Conversation state machine with interruption support
- Utterance queue for concurrent speech handling
- Interruptible streaming TTS
- Voice pipeline orchestration
- Voice tools for LLM (appointment booking, customer management)
"""

# Import conversation services
from .conversation.simple_voice_service import SimpleVoiceService

# Conversation state management
from .conversation_state import (
    ConversationState,
    ConversationStateManager,
    InterruptionType,
    create_conversation_state_manager
)

# Utterance queue for speech during tool execution
from .utterance_queue import (
    UtteranceQueueService,
    UtterancePriority,
    PendingUtterance,
    QueueConfig,
    MergeStrategy,
    create_utterance_queue
)

# Streaming TTS with interruption support
from .streaming_tts import (
    InterruptibleTTSStream,
    SentenceStreamingTTS,
    StreamConfig,
    StreamState,
    CancellationToken,
    create_interruptible_tts_stream,
    stream_tts_with_interruption
)

# Voice pipeline orchestration
from .pipeline_orchestrator import (
    VoicePipelineOrchestrator,
    PipelineConfig,
    PipelineMetrics,
    create_voice_pipeline
)

# Voice tools for LLM
from .tools import (
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
    # Simple voice service
    "SimpleVoiceService",
    
    # Conversation state
    "ConversationState",
    "ConversationStateManager",
    "InterruptionType",
    "create_conversation_state_manager",
    
    # Utterance queue
    "UtteranceQueueService",
    "UtterancePriority",
    "PendingUtterance",
    "QueueConfig",
    "MergeStrategy",
    "create_utterance_queue",
    
    # Streaming TTS
    "InterruptibleTTSStream",
    "SentenceStreamingTTS",
    "StreamConfig",
    "StreamState",
    "CancellationToken",
    "create_interruptible_tts_stream",
    "stream_tts_with_interruption",
    
    # Pipeline orchestrator
    "VoicePipelineOrchestrator",
    "PipelineConfig",
    "PipelineMetrics",
    "create_voice_pipeline",
    
    # Voice tools
    "VoiceTool",
    "VoiceToolExecutor",
    "ToolCategory",
    "VOICE_TOOLS",
    "get_tools_for_openai",
    "get_tool",
    "tool_allows_interruption",
    "get_customer_service_system_prompt",
    "CUSTOMER_SERVICE_SYSTEM_PROMPT",
]
