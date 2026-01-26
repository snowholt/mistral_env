"""
Voice Pipeline Orchestrator with Interruption Support.

Coordinates the complete voice conversation flow:
STT -> LLM -> TTS with proper interruption handling.

Handles:
- TTS barge-in (stop playback, process new speech)
- Tool call concurrent listening (queue utterances)
- LLM cancellation (cancel generation for new speech)
- State machine transitions
- Voice tools execution (appointment booking, customer management)

Author: BeautyAI Framework
Date: January 2026
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Awaitable, List
from enum import Enum, auto

from .conversation_state import (
    ConversationStateManager,
    ConversationState,
    InterruptionType,
    create_conversation_state_manager
)
from .utterance_queue import (
    UtteranceQueueService,
    UtterancePriority,
    PendingUtterance,
    create_utterance_queue
)
from .streaming_tts import (
    InterruptibleTTSStream,
    SentenceStreamingTTS,
    StreamConfig,
    CancellationToken
)
from .tools import (
    VoiceToolExecutor,
    get_tools_for_openai,
    get_tool,
    tool_allows_interruption,
    get_customer_service_system_prompt
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for voice pipeline."""
    # Timing
    turn_silence_ms: int = 2000              # Silence before processing turn
    barge_in_threshold_ms: int = 200         # Min speech to trigger barge-in
    llm_cancel_threshold_ms: int = 500       # Min speech to cancel LLM
    
    # Features
    enable_barge_in: bool = True             # Allow TTS interruption
    enable_llm_cancellation: bool = True     # Allow LLM generation cancellation
    enable_concurrent_listening: bool = True  # Listen during tool execution
    enable_sentence_streaming: bool = True   # Stream TTS sentence by sentence
    enable_voice_tools: bool = True          # Enable voice tools (appointment, customer)
    
    # Queue
    max_queue_size: int = 5                  # Max queued utterances
    
    # TTS
    tts_chunk_size_ms: int = 40              # TTS chunk size
    tts_sample_rate: int = 16000             # TTS sample rate
    
    # Tool executor
    api_base_url: str = "http://localhost:8000"  # API base URL for tools


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance."""
    sessions_started: int = 0
    total_utterances: int = 0
    total_responses: int = 0
    tts_interruptions: int = 0
    llm_cancellations: int = 0
    tool_concurrent_utterances: int = 0
    avg_response_latency_ms: float = 0.0
    avg_first_byte_latency_ms: float = 0.0


class VoicePipelineOrchestrator:
    """
    Orchestrates the complete voice conversation pipeline.
    
    Provides unified handling for:
    - User speech detection and processing
    - LLM response generation with cancellation
    - TTS synthesis with barge-in support
    - Tool execution with concurrent listening
    
    Usage:
        orchestrator = VoicePipelineOrchestrator(
            session_id="123",
            stt_model=whisper,
            llm_model=llm,
            tts_model=tts
        )
        
        # Set up data channel callback
        orchestrator.set_send_callback(data_channel.send)
        
        # Process user speech
        await orchestrator.on_user_speech(audio_data)
        
        # Handle barge-in
        if orchestrator.is_speaking:
            result = await orchestrator.handle_interrupt(new_speech_text)
    """
    
    def __init__(
        self,
        session_id: str,
        stt_model: Any = None,
        llm_model: Any = None,
        tts_model: Any = None,
        config: Optional[PipelineConfig] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        enable_customer_service_mode: bool = False
    ):
        """
        Initialize voice pipeline orchestrator.
        
        Args:
            session_id: Unique session identifier
            stt_model: Speech-to-text model
            llm_model: Language model
            tts_model: Text-to-speech model
            config: Pipeline configuration
            loop: Event loop for sync operations
            enable_customer_service_mode: Enable customer service tools (appointment booking)
        """
        self.session_id = session_id
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.config = config or PipelineConfig()
        self.loop = loop or asyncio.get_event_loop()
        self.enable_customer_service_mode = enable_customer_service_mode
        
        # State management
        self.state_manager = create_conversation_state_manager(
            session_id,
            enable_llm_cancellation=self.config.enable_llm_cancellation
        )
        
        # Utterance queue
        self.utterance_queue = create_utterance_queue(
            session_id,
            max_size=self.config.max_queue_size
        )
        
        # Tool executor for voice tools
        self._tool_executor: Optional[VoiceToolExecutor] = None
        if self.config.enable_voice_tools:
            self._tool_executor = VoiceToolExecutor(
                base_url=self.config.api_base_url
            )
        
        # Active operations
        self._llm_task: Optional[asyncio.Task] = None
        self._tts_stream: Optional[InterruptibleTTSStream] = None
        self._tool_task: Optional[asyncio.Task] = None
        self._llm_cancel_token = CancellationToken()
        
        # Transcript buffer
        self._transcript_buffer: List[str] = []
        self._turn_timer: Optional[asyncio.Task] = None
        
        # Callbacks
        self._send_callback: Optional[Callable[[str], None]] = None
        self._on_response_complete: Optional[Callable[[str], Awaitable[None]]] = None
        
        # Language
        self.language = "en"
        
        # Customer context (for tool calls)
        self._customer_context: Dict[str, Any] = {}
        
        # Metrics
        self.metrics = PipelineMetrics()
        
        # Set up state manager callbacks
        self._setup_state_callbacks()
        
        logger.info(f"[Pipeline] Initialized for session {session_id}, customer_service={enable_customer_service_mode}")

    def _setup_state_callbacks(self):
        """Set up callbacks for state transitions."""
        async def on_state_change(old_state: ConversationState, new_state: ConversationState):
            logger.debug(f"[Pipeline] State: {old_state.name} -> {new_state.name}")
            self._send_state_update(new_state)
        
        async def on_tts_interrupt():
            if self._tts_stream:
                await self._tts_stream.interrupt()
                logger.info("[Pipeline] TTS interrupted")
        
        async def on_llm_cancel():
            self._llm_cancel_token.cancel()
            if self._llm_task:
                self._llm_task.cancel()
                logger.info("[Pipeline] LLM generation cancelled")
        
        async def on_queue_process(utterances: List[PendingUtterance]):
            # Process queued utterances after tool completion
            if utterances:
                merged = " Also, ".join([u.text for u in utterances])
                logger.info(f"[Pipeline] Processing {len(utterances)} queued utterances")
                self._transcript_buffer.append(merged)
                await self._process_turn()
        
        self.state_manager.on_state_change = on_state_change
        self.state_manager.on_tts_interrupt = on_tts_interrupt
        self.state_manager.on_llm_cancel = on_llm_cancel
        self.state_manager.on_queue_process = on_queue_process

    def set_send_callback(self, callback: Callable[[str], None]):
        """Set callback for sending messages to client (data channel)."""
        self._send_callback = callback

    def set_response_complete_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set callback for when response is complete."""
        self._on_response_complete = callback

    def set_language(self, language: str):
        """Set the conversation language."""
        self.language = language

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        return self.state_manager.state == ConversationState.PLAYING_TTS

    @property
    def is_processing(self) -> bool:
        """Check if pipeline is processing (LLM, TTS, or tool)."""
        return self.state_manager.state in {
            ConversationState.PROCESSING_LLM,
            ConversationState.SYNTHESIZING_TTS,
            ConversationState.PLAYING_TTS,
            ConversationState.EXECUTING_TOOL
        }

    @property
    def current_state(self) -> ConversationState:
        """Get current pipeline state."""
        return self.state_manager.state

    async def on_speech_start(self):
        """Called when user starts speaking."""
        if self.state_manager.state == ConversationState.IDLE:
            await self.state_manager.transition_to(ConversationState.LISTENING)
            self._send_message({"type": "state", "state": "listening"})

    async def on_speech_end(self, transcribed_text: str):
        """
        Called when user finishes speaking with transcription.
        
        Args:
            transcribed_text: The transcribed speech
        """
        if not transcribed_text or not transcribed_text.strip():
            return
        
        text = transcribed_text.strip()
        current_state = self.state_manager.state
        
        logger.info(f"[Pipeline] User said: '{text[:50]}...' (state: {current_state.name})")
        
        # Handle based on current state
        if current_state == ConversationState.PLAYING_TTS:
            # Barge-in: stop TTS and process new speech
            if self.config.enable_barge_in:
                result = await self.state_manager.handle_user_speech(text)
                self.metrics.tts_interruptions += 1
                logger.info(f"[Pipeline] Barge-in detected! Interruption type: {result.name}")
                
                # Clear buffer and add new speech
                self._transcript_buffer = [text]
                await self._schedule_turn_processing()
                return
        
        elif current_state == ConversationState.EXECUTING_TOOL:
            # Tool execution: queue the utterance
            if self.config.enable_concurrent_listening:
                await self.utterance_queue.enqueue(
                    text=text,
                    priority=UtterancePriority.HIGH,
                    language=self.language
                )
                self.metrics.tool_concurrent_utterances += 1
                logger.info(f"[Pipeline] Queued utterance during tool execution")
                
                # Also send to client that we heard them
                self._send_message({
                    "type": "queued_utterance",
                    "text": text,
                    "queue_size": self.utterance_queue.size
                })
                return
        
        elif current_state == ConversationState.PROCESSING_LLM:
            # LLM generation: can cancel
            if self.config.enable_llm_cancellation:
                result = await self.state_manager.handle_user_speech(text)
                self.metrics.llm_cancellations += 1
                logger.info(f"[Pipeline] LLM cancelled, processing new speech")
                
                self._transcript_buffer = [text]
                await self._schedule_turn_processing()
                return
        
        elif current_state == ConversationState.SYNTHESIZING_TTS:
            # TTS synthesis: interrupt
            result = await self.state_manager.handle_user_speech(text)
            self._transcript_buffer = [text]
            await self._schedule_turn_processing()
            return
        
        # Normal flow: add to buffer and schedule
        await self.state_manager.transition_to(ConversationState.PROCESSING_STT)
        self._transcript_buffer.append(text)
        
        # Send transcription to client
        self._send_message({
            "type": "transcription",
            "text": text,
            "role": "user"
        })
        
        await self._schedule_turn_processing()

    async def _schedule_turn_processing(self):
        """Schedule turn processing after silence period."""
        # Cancel existing timer
        if self._turn_timer:
            self._turn_timer.cancel()
        
        # Schedule new timer
        self._turn_timer = asyncio.create_task(
            self._wait_and_process_turn()
        )

    async def _wait_and_process_turn(self):
        """Wait for silence then process the turn."""
        try:
            await asyncio.sleep(self.config.turn_silence_ms / 1000.0)
            await self._process_turn()
        except asyncio.CancelledError:
            pass

    async def _process_turn(self):
        """Process the complete turn: LLM -> TTS."""
        if not self._transcript_buffer:
            return
        
        full_text = " ".join(self._transcript_buffer)
        self._transcript_buffer = []
        self._turn_timer = None
        
        self.metrics.total_utterances += 1
        turn_start = time.time()
        
        logger.info(f"[Pipeline] Processing turn: '{full_text[:80]}...'")
        
        # Update state
        await self.state_manager.transition_to(ConversationState.PROCESSING_LLM)
        self._send_message({"type": "state", "state": "processing"})
        self._send_message({"type": "mic_control", "action": "mute"})
        
        try:
            # Reset cancellation token
            self._llm_cancel_token.reset()
            
            # Generate LLM response
            response_text = await self._generate_llm_response(full_text)
            
            if not response_text or self._llm_cancel_token.is_cancelled:
                logger.info("[Pipeline] LLM response cancelled or empty")
                await self.state_manager.transition_to(ConversationState.IDLE)
                self._send_message({"type": "state", "state": "listening"})
                self._send_message({"type": "mic_control", "action": "unmute"})
                return
            
            # Log response
            self.metrics.total_responses += 1
            logger.info(f"[Pipeline] AI response: '{response_text[:80]}...'")
            
            # Synthesize and stream TTS
            await self.state_manager.transition_to(ConversationState.SYNTHESIZING_TTS)
            self._send_message({"type": "state", "state": "synthesizing"})
            
            await self._stream_tts_response(response_text)
            
            # Calculate latency
            latency_ms = (time.time() - turn_start) * 1000
            self._update_latency_metrics(latency_ms)
            
            # Complete
            await self.state_manager.transition_to(ConversationState.IDLE)
            
            # Check for queued utterances
            if self.utterance_queue.size > 0:
                queued = await self.state_manager.on_tool_execution_complete()
            
            # Callback
            if self._on_response_complete:
                await self._on_response_complete(response_text)
            
        except asyncio.CancelledError:
            logger.info("[Pipeline] Turn processing cancelled")
            await self.state_manager.transition_to(ConversationState.IDLE)
        except Exception as e:
            logger.error(f"[Pipeline] Turn processing error: {e}")
            await self.state_manager.transition_to(ConversationState.IDLE)
        finally:
            self._send_message({"type": "state", "state": "listening"})
            self._send_message({"type": "mic_control", "action": "unmute"})

    async def _generate_llm_response(self, user_text: str) -> Optional[str]:
        """Generate LLM response with cancellation support."""
        if not self.llm_model:
            return None
        
        # Build prompt
        system_prompt = (
            "You are a helpful AI assistant having a voice conversation. "
            "Respond naturally and conversationally. "
            "When listing items, use words like 'first', 'second', 'next', 'also', 'finally' instead of numbers. "
            "Keep responses concise and suitable for spoken dialogue."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n/no_think {user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        full_response = ""
        
        try:
            queue = asyncio.Queue()
            
            def generate_sync():
                try:
                    if not self.llm_model.model:
                        self.llm_model.load_model()
                    
                    generator = self.llm_model.model.create_completion(
                        prompt,
                        max_tokens=512,
                        stop=["<|im_end|>"],
                        stream=True
                    )
                    
                    for chunk in generator:
                        if self._llm_cancel_token.is_cancelled:
                            break
                        self.loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    
                    self.loop.call_soon_threadsafe(queue.put_nowait, None)
                    
                except Exception as e:
                    logger.error(f"[Pipeline] LLM generation error: {e}")
                    self.loop.call_soon_threadsafe(queue.put_nowait, None)
            
            # Run generation in thread
            self.loop.run_in_executor(None, generate_sync)
            
            # Stream chunks from queue
            import re
            last_sent_length = 0
            
            while True:
                # Check for cancellation
                if self._llm_cancel_token.is_cancelled:
                    break
                
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check cancellation during timeout
                    if await self.state_manager.check_llm_cancellation():
                        break
                    continue
                
                if chunk is None:
                    break
                
                delta = chunk["choices"][0]["text"]
                full_response += delta
                
                # Clean and send partial response
                clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
                clean_response = re.sub(r'<think>.*$', '', clean_response, flags=re.DOTALL)
                clean_response = clean_response.replace("</think>", "")
                
                if len(clean_response) > last_sent_length:
                    new_content = clean_response[last_sent_length:]
                    last_sent_length = len(clean_response)
                    
                    if new_content.strip():
                        self._send_message({
                            "type": "response_chunk",
                            "text": new_content,
                            "role": "assistant"
                        })
            
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"[Pipeline] LLM response error: {e}")
            return None

    async def _stream_tts_response(self, text: str):
        """Stream TTS response with interruption support."""
        if not self.tts_model:
            return
        
        # Clean text for TTS
        from ...utils.transcription_cleaner import clean_llm_response_for_tts
        tts_text = clean_llm_response_for_tts(text, language=self.language)
        
        if not tts_text:
            return
        
        logger.info(f"[Pipeline] Streaming TTS: {len(tts_text)} chars")
        
        # Create config
        stream_config = StreamConfig(
            chunk_size_ms=self.config.tts_chunk_size_ms,
            target_sample_rate=self.config.tts_sample_rate,
            sentence_based=self.config.enable_sentence_streaming
        )
        
        # Create interruptible stream
        self._tts_stream = InterruptibleTTSStream(
            tts_engine=self.tts_model,
            text=tts_text,
            language=self.language,
            config=stream_config,
            on_first_chunk=self._on_tts_first_chunk
        )
        
        # Transition to playing
        await self.state_manager.transition_to(ConversationState.PLAYING_TTS)
        self._send_message({"type": "state", "state": "speaking"})
        
        # Stream chunks
        import base64
        chunk_count = 0
        
        try:
            async for chunk in self._tts_stream.stream():
                # Check for interruption
                if await self.state_manager.check_tts_interrupt():
                    logger.info("[Pipeline] TTS interrupted by barge-in")
                    break
                
                # Send chunk to client
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                self._send_message({
                    "type": "tts_chunk",
                    "audio_base64": chunk_b64,
                    "sequence": chunk_count,
                    "format": "pcm16"
                })
                chunk_count += 1
            
            # Send end marker
            self._send_message({
                "type": "tts_complete",
                "chunks_sent": chunk_count
            })
            
            logger.info(f"[Pipeline] TTS complete: {chunk_count} chunks")
            
        except asyncio.CancelledError:
            logger.info("[Pipeline] TTS stream cancelled")
        except Exception as e:
            logger.error(f"[Pipeline] TTS streaming error: {e}")
        finally:
            self._tts_stream = None

    async def _on_tts_first_chunk(self):
        """Callback when first TTS chunk is ready."""
        metrics = self._tts_stream.metrics if self._tts_stream else None
        if metrics:
            ttfb = metrics.first_chunk_ms
            self._send_message({
                "type": "tts_start",
                "time_to_first_chunk_ms": ttfb
            })

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Any]]] = None
    ) -> Any:
        """
        Execute a tool call while continuing to listen for user speech.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_executor: Optional custom executor (uses built-in if not provided)
            
        Returns:
            Tool execution result
        """
        # Check if tool allows interruption
        allows_interrupt = tool_allows_interruption(tool_name)
        
        await self.state_manager.transition_to(ConversationState.EXECUTING_TOOL)
        self._send_message({
            "type": "tool_call",
            "tool": tool_name,
            "args": tool_args,
            "status": "executing",
            "allows_interruption": allows_interrupt
        })
        
        try:
            # Use built-in executor if available and no custom one provided
            if tool_executor:
                result = await tool_executor(tool_name, tool_args)
            elif self._tool_executor:
                result = await self._tool_executor.execute(
                    tool_name, 
                    tool_args,
                    session_id=self.session_id
                )
            else:
                logger.warning(f"[Pipeline] No tool executor available for {tool_name}")
                result = {"success": False, "error": "Tool executor not configured"}
            
            # Update customer context if we got customer info
            if result.get("success") and result.get("customer"):
                self._customer_context["customer"] = result["customer"]
                logger.info(f"[Pipeline] Updated customer context: {result['customer'].get('id')}")
            
            # Process any queued utterances
            queued = await self.state_manager.on_tool_execution_complete()
            
            self._send_message({
                "type": "tool_call",
                "tool": tool_name,
                "status": "complete",
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Tool execution error: {e}")
            self._send_message({
                "type": "tool_call",
                "tool": tool_name,
                "status": "error",
                "error": str(e)
            })
            raise
        finally:
            await self.state_manager.transition_to(ConversationState.PROCESSING_LLM)
    
    async def execute_voice_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a voice tool using the built-in tool executor.
        
        Convenience method for LLM function calling.
        
        Args:
            tool_name: Name of the voice tool
            tool_args: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._tool_executor:
            return {"success": False, "error": "Tool executor not configured"}
        
        return await self.execute_tool(tool_name, tool_args)
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get voice tools in OpenAI function calling format.
        
        Use this to pass available tools to the LLM.
        
        Returns:
            List of tool definitions
        """
        if not self.config.enable_voice_tools:
            return []
        return get_tools_for_openai()
    
    def get_customer_service_system_prompt(self) -> str:
        """
        Get the system prompt for customer service mode.
        
        Returns:
            System prompt string
        """
        return get_customer_service_system_prompt()
    
    def get_customer_context(self) -> Dict[str, Any]:
        """Get current customer context from tool calls."""
        return self._customer_context.copy()
    
    def set_customer_context(self, context: Dict[str, Any]):
        """Set customer context (e.g., from previous session)."""
        self._customer_context.update(context)

    def _send_message(self, message: Dict[str, Any]):
        """Send message to client via callback."""
        if self._send_callback:
            try:
                self._send_callback(json.dumps(message))
            except Exception as e:
                logger.error(f"[Pipeline] Send callback error: {e}")
        else:
            logger.debug(f"[Pipeline] Message (no callback): {message}")

    def _send_state_update(self, state: ConversationState):
        """Send state update to client."""
        state_mapping = {
            ConversationState.IDLE: "idle",
            ConversationState.LISTENING: "listening",
            ConversationState.PROCESSING_STT: "transcribing",
            ConversationState.PROCESSING_LLM: "thinking",
            ConversationState.EXECUTING_TOOL: "executing_tool",
            ConversationState.SYNTHESIZING_TTS: "synthesizing",
            ConversationState.PLAYING_TTS: "speaking",
            ConversationState.INTERRUPTED: "interrupted"
        }
        
        state_name = state_mapping.get(state, state.name.lower())
        self._send_message({"type": "state", "state": state_name})

    def _update_latency_metrics(self, latency_ms: float):
        """Update average latency metrics."""
        n = self.metrics.total_responses
        if n == 1:
            self.metrics.avg_response_latency_ms = latency_ms
        else:
            # Running average
            self.metrics.avg_response_latency_ms = (
                (self.metrics.avg_response_latency_ms * (n - 1) + latency_ms) / n
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "session_id": self.session_id,
            "state": self.state_manager.state.name,
            "is_speaking": self.is_speaking,
            "is_processing": self.is_processing,
            "queue_size": self.utterance_queue.size,
            "language": self.language,
            "metrics": {
                "total_utterances": self.metrics.total_utterances,
                "total_responses": self.metrics.total_responses,
                "tts_interruptions": self.metrics.tts_interruptions,
                "llm_cancellations": self.metrics.llm_cancellations,
                "avg_latency_ms": self.metrics.avg_response_latency_ms
            },
            "state_details": self.state_manager.get_status()
        }

    def reset(self):
        """Reset the pipeline to initial state."""
        self.state_manager.reset()
        self._transcript_buffer.clear()
        self._llm_cancel_token.reset()
        self._tts_stream = None
        
        if self._turn_timer:
            self._turn_timer.cancel()
            self._turn_timer = None
        
        logger.info(f"[Pipeline] Reset for session {self.session_id}")


# Factory function
def create_voice_pipeline(
    session_id: str,
    stt_model: Any = None,
    llm_model: Any = None,
    tts_model: Any = None,
    language: str = "en",
    enable_customer_service_mode: bool = False,
    **config_kwargs
) -> VoicePipelineOrchestrator:
    """
    Create a voice pipeline with standard configuration.
    
    Args:
        session_id: Session identifier
        stt_model: Speech-to-text model
        llm_model: Language model
        tts_model: Text-to-speech model
        language: Conversation language (en, ar)
        enable_customer_service_mode: Enable appointment booking tools
        **config_kwargs: Additional PipelineConfig options
        
    Returns:
        Configured VoicePipelineOrchestrator
    """
    config = PipelineConfig(**config_kwargs)
    pipeline = VoicePipelineOrchestrator(
        session_id=session_id,
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        config=config,
        enable_customer_service_mode=enable_customer_service_mode
    )
    pipeline.set_language(language)
    return pipeline
