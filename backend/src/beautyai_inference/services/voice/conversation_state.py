"""
Conversation State Machine for Voice Pipeline with Interruption Support.

Manages the conversation flow states and provides proper handling for:
- TTS playback interruption (barge-in)
- Tool call execution with concurrent listening
- Utterance queuing when model is busy

Author: BeautyAI Framework
Date: January 2026
"""

import asyncio
import logging
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Awaitable
from collections import deque

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Voice conversation pipeline states."""
    IDLE = auto()              # Waiting for user input
    LISTENING = auto()         # Actively receiving user speech
    PROCESSING_STT = auto()    # Transcribing user speech
    PROCESSING_LLM = auto()    # Generating LLM response (interruptible)
    EXECUTING_TOOL = auto()    # Executing tool/DB call (NOT interruptible, but continues listening)
    SYNTHESIZING_TTS = auto()  # Generating TTS audio
    PLAYING_TTS = auto()       # Playing TTS audio (interruptible via barge-in)
    INTERRUPTED = auto()       # User interrupted during playback


class InterruptionType(Enum):
    """Types of interruption events."""
    NONE = auto()
    TTS_BARGE_IN = auto()      # User spoke during TTS playback -> stop TTS
    TOOL_CONCURRENT = auto()   # User spoke during tool execution -> queue utterance
    LLM_CANCEL = auto()        # User spoke during LLM generation -> can cancel


@dataclass
class QueuedUtterance:
    """Represents a queued user utterance during tool execution."""
    text: str
    timestamp: float = field(default_factory=time.time)
    audio_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMetrics:
    """Metrics for conversation state transitions."""
    state_changes: int = 0
    tts_interruptions: int = 0
    tool_concurrent_utterances: int = 0
    llm_cancellations: int = 0
    total_latency_ms: float = 0.0
    utterances_processed: int = 0
    session_start: float = field(default_factory=time.time)


class ConversationStateManager:
    """
    Manages conversation state with interruption support.
    
    Key behaviors:
    - TTS playback can be interrupted (barge-in) -> stops TTS, processes new speech
    - Tool execution continues while listening -> queues utterances
    - LLM generation can be cancelled if user speaks again
    
    Usage:
        state_manager = ConversationStateManager(session_id)
        
        # Set callbacks for state actions
        state_manager.on_tts_interrupt = async_stop_tts_fn
        state_manager.on_tool_complete = async_process_queue_fn
        
        # Handle user speech
        await state_manager.handle_user_speech(transcribed_text)
    """
    
    def __init__(
        self,
        session_id: str,
        max_queue_size: int = 5,
        tool_timeout_ms: int = 30000,
        enable_llm_cancellation: bool = True
    ):
        """
        Initialize conversation state manager.
        
        Args:
            session_id: Unique session identifier
            max_queue_size: Maximum utterances to queue during tool execution
            tool_timeout_ms: Timeout for tool execution before forcing state reset
            enable_llm_cancellation: Allow LLM generation to be cancelled by user speech
        """
        self.session_id = session_id
        self.max_queue_size = max_queue_size
        self.tool_timeout_ms = tool_timeout_ms
        self.enable_llm_cancellation = enable_llm_cancellation
        
        # State tracking
        self._state = ConversationState.IDLE
        self._previous_state = ConversationState.IDLE
        self._state_lock = asyncio.Lock()
        self._state_changed_at = time.time()
        
        # Utterance queue for speech during tool execution
        self._utterance_queue: deque[QueuedUtterance] = deque(maxlen=max_queue_size)
        self._queue_lock = asyncio.Lock()
        
        # Active task tracking
        self._active_tool_task: Optional[asyncio.Task] = None
        self._active_llm_task: Optional[asyncio.Task] = None
        self._active_tts_task: Optional[asyncio.Task] = None
        
        # Cancellation flags
        self._llm_cancel_requested = False
        self._tts_interrupt_requested = False
        
        # Callbacks (set by endpoint)
        self.on_state_change: Optional[Callable[[ConversationState, ConversationState], Awaitable[None]]] = None
        self.on_tts_interrupt: Optional[Callable[[], Awaitable[None]]] = None
        self.on_llm_cancel: Optional[Callable[[], Awaitable[None]]] = None
        self.on_queue_process: Optional[Callable[[List[QueuedUtterance]], Awaitable[None]]] = None
        
        # Metrics
        self.metrics = ConversationMetrics()
        
        logger.info(f"[ConvState] Initialized for session {session_id}")

    @property
    def state(self) -> ConversationState:
        """Get current conversation state."""
        return self._state

    @property
    def is_idle(self) -> bool:
        """Check if conversation is idle (ready for input)."""
        return self._state == ConversationState.IDLE

    @property
    def is_interruptible(self) -> bool:
        """Check if current state allows interruption."""
        return self._state in {
            ConversationState.PLAYING_TTS,
            ConversationState.PROCESSING_LLM,
            ConversationState.SYNTHESIZING_TTS,
        }

    @property 
    def is_tool_executing(self) -> bool:
        """Check if a tool is currently executing."""
        return self._state == ConversationState.EXECUTING_TOOL

    @property
    def has_queued_utterances(self) -> bool:
        """Check if there are queued utterances."""
        return len(self._utterance_queue) > 0

    @property
    def queue_size(self) -> int:
        """Get number of queued utterances."""
        return len(self._utterance_queue)

    async def transition_to(self, new_state: ConversationState) -> bool:
        """
        Transition to a new state with validation.
        
        Args:
            new_state: Target state to transition to
            
        Returns:
            True if transition was successful
        """
        async with self._state_lock:
            old_state = self._state
            
            # Validate transition
            if not self._is_valid_transition(old_state, new_state):
                logger.warning(
                    f"[ConvState] Invalid transition: {old_state.name} -> {new_state.name}"
                )
                return False
            
            self._previous_state = old_state
            self._state = new_state
            self._state_changed_at = time.time()
            self.metrics.state_changes += 1
            
            logger.debug(f"[ConvState] {old_state.name} -> {new_state.name}")
            
            # Trigger callback
            if self.on_state_change:
                try:
                    await self.on_state_change(old_state, new_state)
                except Exception as e:
                    logger.error(f"[ConvState] State change callback error: {e}")
            
            return True

    def _is_valid_transition(self, from_state: ConversationState, to_state: ConversationState) -> bool:
        """Validate state transition."""
        # Define valid transitions
        valid_transitions = {
            ConversationState.IDLE: {
                ConversationState.LISTENING,
            },
            ConversationState.LISTENING: {
                ConversationState.PROCESSING_STT,
                ConversationState.IDLE,  # Timeout/cancel
            },
            ConversationState.PROCESSING_STT: {
                ConversationState.PROCESSING_LLM,
                ConversationState.IDLE,  # Empty transcription
            },
            ConversationState.PROCESSING_LLM: {
                ConversationState.EXECUTING_TOOL,
                ConversationState.SYNTHESIZING_TTS,
                ConversationState.IDLE,  # Cancelled
                ConversationState.INTERRUPTED,  # User interrupted
            },
            ConversationState.EXECUTING_TOOL: {
                ConversationState.SYNTHESIZING_TTS,
                ConversationState.PROCESSING_LLM,  # Tool result needs more LLM processing
                ConversationState.IDLE,  # Tool complete, no response needed
            },
            ConversationState.SYNTHESIZING_TTS: {
                ConversationState.PLAYING_TTS,
                ConversationState.IDLE,  # TTS failed
                ConversationState.INTERRUPTED,  # User interrupted
            },
            ConversationState.PLAYING_TTS: {
                ConversationState.IDLE,  # Playback complete
                ConversationState.INTERRUPTED,  # Barge-in
            },
            ConversationState.INTERRUPTED: {
                ConversationState.LISTENING,  # Process new speech
                ConversationState.PROCESSING_STT,  # Already have speech
                ConversationState.IDLE,
            },
        }
        
        # Allow transitions from any state to IDLE (error recovery)
        if to_state == ConversationState.IDLE:
            return True
        
        return to_state in valid_transitions.get(from_state, set())

    async def handle_user_speech(
        self,
        text: str,
        audio_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InterruptionType:
        """
        Handle incoming user speech based on current state.
        
        Args:
            text: Transcribed speech text
            audio_data: Optional raw audio data
            metadata: Optional metadata (language, confidence, etc.)
            
        Returns:
            Type of interruption that occurred
        """
        current_state = self._state
        interruption = InterruptionType.NONE
        
        logger.info(f"[ConvState] User speech received in state {current_state.name}: '{text[:50]}...'")
        
        if current_state == ConversationState.PLAYING_TTS:
            # TTS barge-in: stop playback and process new speech
            interruption = InterruptionType.TTS_BARGE_IN
            self.metrics.tts_interruptions += 1
            self._tts_interrupt_requested = True
            
            logger.info(f"[ConvState] TTS barge-in detected!")
            
            if self.on_tts_interrupt:
                try:
                    await self.on_tts_interrupt()
                except Exception as e:
                    logger.error(f"[ConvState] TTS interrupt callback error: {e}")
            
            await self.transition_to(ConversationState.INTERRUPTED)
            
        elif current_state == ConversationState.EXECUTING_TOOL:
            # Tool execution: queue the utterance for later
            interruption = InterruptionType.TOOL_CONCURRENT
            self.metrics.tool_concurrent_utterances += 1
            
            async with self._queue_lock:
                utterance = QueuedUtterance(
                    text=text,
                    audio_data=audio_data,
                    metadata=metadata or {}
                )
                self._utterance_queue.append(utterance)
                
            logger.info(
                f"[ConvState] Queued utterance during tool execution "
                f"(queue size: {len(self._utterance_queue)})"
            )
            
        elif current_state == ConversationState.PROCESSING_LLM and self.enable_llm_cancellation:
            # LLM generation: can cancel and process new speech
            interruption = InterruptionType.LLM_CANCEL
            self.metrics.llm_cancellations += 1
            self._llm_cancel_requested = True
            
            logger.info(f"[ConvState] LLM cancellation requested")
            
            if self.on_llm_cancel:
                try:
                    await self.on_llm_cancel()
                except Exception as e:
                    logger.error(f"[ConvState] LLM cancel callback error: {e}")
            
            await self.transition_to(ConversationState.INTERRUPTED)
            
        elif current_state == ConversationState.SYNTHESIZING_TTS:
            # TTS synthesis: can interrupt and process new speech
            interruption = InterruptionType.TTS_BARGE_IN
            self.metrics.tts_interruptions += 1
            
            logger.info(f"[ConvState] TTS synthesis interrupted")
            await self.transition_to(ConversationState.INTERRUPTED)
            
        return interruption

    async def on_tool_execution_complete(self) -> List[QueuedUtterance]:
        """
        Called when tool execution completes.
        Returns queued utterances that need processing.
        
        Returns:
            List of queued utterances (may be empty)
        """
        async with self._queue_lock:
            queued = list(self._utterance_queue)
            self._utterance_queue.clear()
        
        if queued:
            logger.info(f"[ConvState] Tool complete, returning {len(queued)} queued utterances")
            
            if self.on_queue_process:
                try:
                    await self.on_queue_process(queued)
                except Exception as e:
                    logger.error(f"[ConvState] Queue process callback error: {e}")
        
        return queued

    async def check_llm_cancellation(self) -> bool:
        """
        Check if LLM generation should be cancelled.
        Call this periodically during LLM streaming.
        
        Returns:
            True if cancellation was requested
        """
        if self._llm_cancel_requested:
            self._llm_cancel_requested = False
            return True
        return False

    async def check_tts_interrupt(self) -> bool:
        """
        Check if TTS playback should be interrupted.
        Call this periodically during TTS playback.
        
        Returns:
            True if interruption was requested
        """
        if self._tts_interrupt_requested:
            self._tts_interrupt_requested = False
            return True
        return False

    def reset(self):
        """Reset state manager to initial state."""
        self._state = ConversationState.IDLE
        self._previous_state = ConversationState.IDLE
        self._utterance_queue.clear()
        self._llm_cancel_requested = False
        self._tts_interrupt_requested = False
        self._active_tool_task = None
        self._active_llm_task = None
        self._active_tts_task = None
        
        logger.info(f"[ConvState] State manager reset for session {self.session_id}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status for debugging/monitoring."""
        return {
            "session_id": self.session_id,
            "state": self._state.name,
            "previous_state": self._previous_state.name,
            "is_interruptible": self.is_interruptible,
            "is_tool_executing": self.is_tool_executing,
            "queue_size": self.queue_size,
            "llm_cancel_pending": self._llm_cancel_requested,
            "tts_interrupt_pending": self._tts_interrupt_requested,
            "metrics": {
                "state_changes": self.metrics.state_changes,
                "tts_interruptions": self.metrics.tts_interruptions,
                "tool_concurrent_utterances": self.metrics.tool_concurrent_utterances,
                "llm_cancellations": self.metrics.llm_cancellations,
                "session_duration_sec": time.time() - self.metrics.session_start,
            }
        }


# Factory function for creating state managers
def create_conversation_state_manager(
    session_id: str,
    **kwargs
) -> ConversationStateManager:
    """Create a conversation state manager with default configuration."""
    return ConversationStateManager(session_id, **kwargs)
