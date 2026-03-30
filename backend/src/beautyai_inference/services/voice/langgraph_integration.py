"""
LangGraph Integration for Voice Pipeline.

Bridges the LangGraph workflow with the existing VoicePipelineOrchestrator,
providing:
- Workflow-driven intent detection (replaces manual detect_intent())
- Graph-based tool orchestration
- State persistence across conversation turns
- Streaming response generation

Author: BeautyAI Framework
Date: January 2026
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncIterator, Callable, Awaitable, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .graph import VoiceWorkflow, create_voice_workflow, VoiceAgentState, Intent
from .graph.nodes.response_node import ResponseGenerator, set_response_generator
from .tools.appointment_tools import VoiceToolExecutor

logger = logging.getLogger(__name__)


class LangGraphVoiceAdapter:
    """
    Adapts the LangGraph workflow to the voice pipeline interface.
    
    This class:
    1. Provides drop-in replacement for detect_intent() and execute_tool()
    2. Manages conversation state via LangGraph
    3. Generates responses using the graph workflow
    4. Integrates with existing TTS streaming
    
    Usage:
        adapter = LangGraphVoiceAdapter(session_id="123")
        
        # Process user speech through the graph
        result = await adapter.process_speech(transcript, language="ar")
        
        # Get response text for TTS
        response_text = result["response_text"]
        
        # Check what tools were called
        tool_results = result["tool_results"]
    """
    
    def __init__(
        self,
        session_id: str,
        language: str = "ar",
        llm_generator: Optional[Callable[[List, str], Awaitable[str]]] = None,
        enable_checkpoints: bool = True
    ):
        """
        Initialize the LangGraph voice adapter.
        
        Args:
            session_id: Unique session identifier
            language: Default language (ar/en)
            llm_generator: Optional LLM generator for response node
            enable_checkpoints: Enable state persistence
        """
        self.session_id = session_id
        self.language = language
        self.llm_generator = llm_generator
        
        # Create workflow
        self.workflow = create_voice_workflow(enable_checkpoints=enable_checkpoints)
        
        # Set up response generator with LLM if provided
        if llm_generator:
            response_gen = ResponseGenerator(
                llm_generator=llm_generator,
                use_templates_only=False
            )
            set_response_generator(response_gen)
        
        # Track last state for context
        self._last_state: Optional[VoiceAgentState] = None
        
        logger.info(f"[LangGraphAdapter] Initialized for session {session_id}")
    
    async def process_speech(
        self,
        transcript: str,
        language: Optional[str] = None,
        interrupt_flag: bool = False
    ) -> Dict[str, Any]:
        """
        Process user speech through the LangGraph workflow.
        
        This replaces the manual detect_intent() + execute_tool() pattern
        with a graph-based approach.
        
        Args:
            transcript: User's transcribed speech
            language: Language override
            interrupt_flag: Whether user interrupted previous turn
            
        Returns:
            Dict containing:
                - response_text: Text for TTS
                - intent: Detected intent
                - tool_results: Results from any tool calls
                - customer_info: Customer data (if available)
                - needs_confirmation: Whether waiting for user confirmation
        """
        lang = language or self.language
        
        logger.info(f"[LangGraphAdapter] Processing: '{transcript[:60]}...'")
        
        # Run the workflow
        result = await self.workflow.process_turn(
            session_id=self.session_id,
            transcript=transcript,
            language=lang,
            interrupt_flag=interrupt_flag
        )
        
        # Cache state
        self._last_state = result
        
        # Extract key fields
        return {
            "response_text": result.get("response_text", ""),
            "intent": result.get("detected_intent", Intent.UNCLEAR),
            "tool_results": result.get("tool_results", []),
            "customer_info": result.get("customer_info"),
            "available_slots": result.get("available_slots"),
            "pending_booking": result.get("pending_booking"),
            "needs_confirmation": result.get("needs_confirmation", False),
            "confirmation_type": result.get("confirmation_type"),
            "error": result.get("error"),
            "messages": result.get("messages", []),
        }
    
    async def process_speech_stream(
        self,
        transcript: str,
        language: Optional[str] = None,
        interrupt_flag: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process speech with streaming updates.
        
        Yields intermediate states as the graph executes,
        useful for real-time UI updates.
        
        Args:
            transcript: User's transcribed speech
            language: Language override
            interrupt_flag: Whether user interrupted
            
        Yields:
            Dict with node updates
        """
        lang = language or self.language
        
        async for event in self.workflow.process_turn_stream(
            session_id=self.session_id,
            transcript=transcript,
            language=lang,
            interrupt_flag=interrupt_flag
        ):
            yield event
            
            # Update cached state
            if event.get("is_final"):
                self._last_state = event.get("state")
    
    def detect_intent(self, transcript: str) -> str:
        """
        Synchronous intent detection (for compatibility).
        
        Note: This is a simplified version that uses pattern matching only.
        For full intent detection, use process_speech() instead.
        
        Args:
            transcript: User's speech
            
        Returns:
            Intent string
        """
        from .graph.nodes.router_node import detect_intent_patterns
        
        intent = detect_intent_patterns(transcript)
        if intent:
            return intent.value
        return Intent.GENERAL_QUERY.value
    
    def get_customer_info(self) -> Optional[Dict[str, Any]]:
        """Get current customer info from state."""
        if self._last_state:
            return self._last_state.get("customer_info")
        return None
    
    def get_available_slots(self) -> Optional[List[Dict[str, Any]]]:
        """Get available slots from state."""
        if self._last_state:
            return self._last_state.get("available_slots")
        return None
    
    def is_awaiting_confirmation(self) -> bool:
        """Check if waiting for user confirmation."""
        if self._last_state:
            return self._last_state.get("needs_confirmation", False)
        return False
    
    def get_confirmation_type(self) -> Optional[str]:
        """Get the type of confirmation needed."""
        if self._last_state:
            return self._last_state.get("confirmation_type")
        return None
    
    def reset(self):
        """Reset conversation state."""
        self.workflow.reset_session(self.session_id, self.language)
        self._last_state = None
        logger.info(f"[LangGraphAdapter] Reset session {self.session_id}")
    
    def get_state(self) -> Optional[VoiceAgentState]:
        """Get current conversation state."""
        return self._last_state or self.workflow.get_session_state(self.session_id)


async def process_with_langgraph(
    session_id: str,
    transcript: str,
    language: str = "ar",
    interrupt_flag: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to process speech through LangGraph.
    
    Creates adapter if needed and processes the turn.
    
    Args:
        session_id: Session ID
        transcript: User's speech
        language: Language code
        interrupt_flag: Whether user interrupted
        
    Returns:
        Processing result dict
    """
    adapter = LangGraphVoiceAdapter(session_id, language)
    return await adapter.process_speech(transcript, language, interrupt_flag)


def detect_intent_with_patterns(transcript: str) -> Intent:
    """
    Pattern-based intent detection (sync, no graph needed).
    
    Useful for quick routing decisions before running full graph.
    
    Args:
        transcript: User's speech
        
    Returns:
        Detected Intent enum
    """
    from .graph.nodes.router_node import detect_intent_patterns
    
    intent = detect_intent_patterns(transcript)
    return intent if intent else Intent.GENERAL_QUERY


class LangGraphPipelineIntegration:
    """
    Integration layer between LangGraph and VoicePipelineOrchestrator.
    
    Replaces the manual intent detection and tool execution
    with LangGraph workflow while preserving:
    - TTS interruption handling
    - Audio streaming
    - State machine transitions
    
    Usage in webrtc_voice.py:
        integration = LangGraphPipelineIntegration(session_id)
        
        # In the speech processing callback:
        result = await integration.handle_speech(
            transcript=transcribed_text,
            language=language,
            send_callback=data_channel.send
        )
        
        response_text = result["response_text"]
        # ... stream to TTS
    """
    
    def __init__(
        self,
        session_id: str,
        language: str = "ar",
        tool_executor: Optional[VoiceToolExecutor] = None,
        llm_model: Any = None
    ):
        """
        Initialize pipeline integration.
        
        Args:
            session_id: Session identifier
            language: Default language
            tool_executor: Optional tool executor (created if not provided)
            llm_model: Optional LLM model for response generation
        """
        self.session_id = session_id
        self.language = language
        self.tool_executor = tool_executor or VoiceToolExecutor()
        self.llm_model = llm_model
        
        # Create LLM generator wrapper if model provided
        llm_generator = None
        if llm_model:
            llm_generator = self._create_llm_generator(llm_model)
        
        # Create adapter
        self.adapter = LangGraphVoiceAdapter(
            session_id=session_id,
            language=language,
            llm_generator=llm_generator,
            enable_checkpoints=True
        )
        
        logger.info(f"[LangGraphIntegration] Initialized for {session_id}")
    
    def _create_llm_generator(self, llm_model) -> Callable:
        """Create async LLM generator from model."""
        async def generate(messages: List, language: str) -> str:
            """Generate response using LLM model."""
            # Build prompt from messages
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
                elif isinstance(msg, HumanMessage):
                    prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
            
            prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"
            
            try:
                if hasattr(llm_model, 'model') and llm_model.model:
                    result = llm_model.model.create_completion(
                        prompt,
                        max_tokens=256,
                        stop=["<|im_end|>"]
                    )
                    return result["choices"][0]["text"].strip()
                return ""
            except Exception as e:
                logger.error(f"[LangGraphIntegration] LLM error: {e}")
                return ""
        
        return generate
    
    async def handle_speech(
        self,
        transcript: str,
        language: Optional[str] = None,
        interrupt_flag: bool = False,
        send_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Handle speech input through LangGraph workflow.
        
        Args:
            transcript: Transcribed user speech
            language: Language override
            interrupt_flag: Whether user interrupted
            send_callback: Callback for sending state updates
            
        Returns:
            Result dict with response_text, tool_results, etc.
        """
        import json
        
        lang = language or self.language
        
        # Send state update
        if send_callback:
            try:
                send_callback(json.dumps({
                    "type": "state",
                    "state": "processing_intent"
                }))
            except Exception:
                pass
        
        # Process through graph
        result = await self.adapter.process_speech(
            transcript=transcript,
            language=lang,
            interrupt_flag=interrupt_flag
        )
        
        # Send tool results if any
        if send_callback and result.get("tool_results"):
            for tool_result in result["tool_results"]:
                try:
                    send_callback(json.dumps({
                        "type": "tool_result",
                        "tool": tool_result.get("tool_name"),
                        "success": tool_result.get("success"),
                        "data": tool_result.get("data", {})
                    }))
                except Exception:
                    pass
        
        return result
    
    async def handle_speech_stream(
        self,
        transcript: str,
        language: Optional[str] = None,
        interrupt_flag: bool = False,
        send_callback: Optional[Callable[[str], None]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle speech with streaming graph updates.
        
        Yields node-level updates for real-time processing.
        
        Args:
            transcript: Transcribed speech
            language: Language override
            interrupt_flag: Whether interrupted
            send_callback: State update callback
            
        Yields:
            Node update dicts
        """
        import json
        
        lang = language or self.language
        
        async for event in self.adapter.process_speech_stream(
            transcript=transcript,
            language=lang,
            interrupt_flag=interrupt_flag
        ):
            # Forward state updates
            if send_callback:
                try:
                    send_callback(json.dumps({
                        "type": "graph_event",
                        "node": event.get("node"),
                        "is_final": event.get("is_final", False)
                    }))
                except Exception:
                    pass
            
            yield event
    
    def get_response_for_tts(self) -> str:
        """Get the latest response text for TTS synthesis."""
        state = self.adapter.get_state()
        if state:
            return state.get("response_text", "")
        return ""
    
    def get_customer_context(self) -> Dict[str, Any]:
        """Get customer context for tool calls."""
        info = self.adapter.get_customer_info()
        return {"customer": info} if info else {}
    
    def reset(self):
        """Reset the integration state."""
        self.adapter.reset()


# Singleton storage for integrations per session
_integrations: Dict[str, LangGraphPipelineIntegration] = {}


def get_or_create_integration(
    session_id: str,
    language: str = "ar",
    llm_model: Any = None
) -> LangGraphPipelineIntegration:
    """
    Get or create a LangGraph integration for a session.
    
    Args:
        session_id: Session identifier
        language: Default language
        llm_model: Optional LLM model
        
    Returns:
        LangGraphPipelineIntegration instance
    """
    if session_id not in _integrations:
        _integrations[session_id] = LangGraphPipelineIntegration(
            session_id=session_id,
            language=language,
            llm_model=llm_model
        )
    return _integrations[session_id]


def clear_integration(session_id: str):
    """Clear integration for a session."""
    if session_id in _integrations:
        del _integrations[session_id]
