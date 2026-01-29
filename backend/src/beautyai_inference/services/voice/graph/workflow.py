"""
Voice Agent Workflow using LangGraph.

Defines the StateGraph that orchestrates the voice conversation flow:
- Router -> Customer/Booking/Response nodes -> Response node

Supports:
- Conditional routing based on intent
- State persistence with checkpoints
- Interruption handling
- Streaming response generation

Author: BeautyAI Framework
Date: January 2026
"""

import logging
from typing import Dict, Any, Optional, Literal, AsyncIterator, Callable, Awaitable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import VoiceAgentState, Intent, create_initial_state
from .nodes import router_node, customer_node, booking_node, response_node
from .nodes.router_node import get_next_node

logger = logging.getLogger(__name__)


def route_after_router(state: VoiceAgentState) -> Literal["customer", "booking", "response"]:
    """
    Conditional edge function: determines next node after router.
    
    Args:
        state: Current voice agent state
        
    Returns:
        Name of next node to execute
    """
    return get_next_node(state)


def route_after_customer(state: VoiceAgentState) -> Literal["booking", "response"]:
    """
    Conditional edge function: determines next node after customer.
    
    If customer is registered and intent was booking-related, go to booking.
    Otherwise go to response.
    
    Args:
        state: Current voice agent state
        
    Returns:
        Name of next node
    """
    intent = state.get("detected_intent")
    customer_info = state.get("customer_info") or {}
    
    # If we just verified customer and they're registered, check original intent
    if customer_info.get("is_registered"):
        # If user was trying to book, continue to booking
        transcript = state.get("current_transcript", "").lower()
        if any(word in transcript for word in ["book", "appointment", "موعد", "حجز"]):
            return "booking"
    
    return "response"


def route_after_booking(state: VoiceAgentState) -> Literal["response"]:
    """
    Conditional edge function: after booking, always go to response.
    
    Args:
        state: Current voice agent state
        
    Returns:
        "response"
    """
    return "response"


def should_end(state: VoiceAgentState) -> Literal["end", "continue"]:
    """
    Check if conversation should end.
    
    Args:
        state: Current voice agent state
        
    Returns:
        "end" or "continue"
    """
    intent = state.get("detected_intent")
    if intent == Intent.FAREWELL:
        return "end"
    return "continue"


def build_voice_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for voice conversations.
    
    Graph structure:
        START -> router -> [customer | booking | response]
        customer -> response
        booking -> response
        response -> END
    
    Returns:
        Configured StateGraph
    """
    # Create the graph with our state schema
    workflow = StateGraph(VoiceAgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("booking", booking_node)
    workflow.add_node("response", response_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "customer": "customer",
            "booking": "booking",
            "response": "response",
        }
    )
    
    # Add edges from customer node
    workflow.add_conditional_edges(
        "customer",
        route_after_customer,
        {
            "booking": "booking",
            "response": "response",
        }
    )
    
    # Booking always goes to response
    workflow.add_edge("booking", "response")
    
    # Response node ends the graph
    workflow.add_edge("response", END)
    
    return workflow


class VoiceWorkflow:
    """
    High-level wrapper around the LangGraph voice workflow.
    
    Provides:
    - State initialization
    - Session management with checkpoints
    - Interrupt handling
    - Streaming support
    
    Usage:
        workflow = VoiceWorkflow()
        result = await workflow.process_turn(session_id, "Hello, I'm John Doe")
        print(result["response_text"])
    """
    
    def __init__(
        self,
        enable_checkpoints: bool = True,
        checkpoint_ttl_seconds: int = 3600
    ):
        """
        Initialize voice workflow.
        
        Args:
            enable_checkpoints: Enable state persistence between turns
            checkpoint_ttl_seconds: TTL for checkpoint data
        """
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_ttl_seconds = checkpoint_ttl_seconds
        
        # Build the graph
        graph = build_voice_workflow()
        
        # Add checkpointer for state persistence
        if enable_checkpoints:
            self.checkpointer = MemorySaver()
            self.app = graph.compile(checkpointer=self.checkpointer)
        else:
            self.checkpointer = None
            self.app = graph.compile()
        
        # Session state cache (for fast access)
        self._session_states: Dict[str, VoiceAgentState] = {}
        
        logger.info("[VoiceWorkflow] Initialized with checkpoints=%s", enable_checkpoints)
    
    async def process_turn(
        self,
        session_id: str,
        transcript: str,
        language: str = "ar",
        interrupt_flag: bool = False
    ) -> VoiceAgentState:
        """
        Process a single conversation turn.
        
        Args:
            session_id: Unique session identifier
            transcript: User's transcribed speech
            language: Language code (ar/en)
            interrupt_flag: Whether user interrupted previous turn
            
        Returns:
            Final state after processing
        """
        logger.info(f"[VoiceWorkflow] Processing turn for {session_id}: '{transcript[:50]}...'")
        
        # Get or create session state
        if session_id in self._session_states:
            state = self._session_states[session_id].copy()
        else:
            state = create_initial_state(session_id, language)
        
        # Update state with new turn data
        state["current_transcript"] = transcript
        state["language"] = language
        state["interrupt_flag"] = interrupt_flag
        state["tool_results"] = []  # Clear previous turn's results
        state["response_text"] = ""  # Clear previous response
        state["error"] = None
        
        # Run the graph
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            result = await self.app.ainvoke(state, config)
            
            # Cache the result for next turn
            self._session_states[session_id] = result
            
            logger.info(f"[VoiceWorkflow] Turn complete, response: '{result.get('response_text', '')[:50]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"[VoiceWorkflow] Error processing turn: {e}")
            
            # Return error state
            state["error"] = str(e)
            state["response_text"] = self._get_error_response(language)
            return state
    
    async def process_turn_stream(
        self,
        session_id: str,
        transcript: str,
        language: str = "ar",
        interrupt_flag: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a turn with streaming updates.
        
        Yields intermediate states as the graph executes,
        allowing for real-time UI updates and TTS streaming.
        
        Args:
            session_id: Unique session identifier
            transcript: User's transcribed speech
            language: Language code (ar/en)
            interrupt_flag: Whether user interrupted
            
        Yields:
            Dict with node name and state updates
        """
        logger.info(f"[VoiceWorkflow] Streaming turn for {session_id}")
        
        # Get or create session state
        if session_id in self._session_states:
            state = self._session_states[session_id].copy()
        else:
            state = create_initial_state(session_id, language)
        
        # Update state
        state["current_transcript"] = transcript
        state["language"] = language
        state["interrupt_flag"] = interrupt_flag
        state["tool_results"] = []
        state["response_text"] = ""
        state["error"] = None
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            async for event in self.app.astream(state, config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    yield {
                        "node": node_name,
                        "state": node_state,
                        "is_final": node_name == "response",
                    }
                    
                    # Update session state
                    if isinstance(node_state, dict):
                        self._session_states[session_id] = {
                            **self._session_states.get(session_id, state),
                            **node_state
                        }
                        
        except Exception as e:
            logger.error(f"[VoiceWorkflow] Stream error: {e}")
            yield {
                "node": "error",
                "state": {"error": str(e), "response_text": self._get_error_response(language)},
                "is_final": True,
            }
    
    def get_session_state(self, session_id: str) -> Optional[VoiceAgentState]:
        """Get current state for a session."""
        return self._session_states.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear session state."""
        if session_id in self._session_states:
            del self._session_states[session_id]
            logger.info(f"[VoiceWorkflow] Cleared session: {session_id}")
    
    def reset_session(self, session_id: str, language: str = "ar"):
        """Reset session to initial state."""
        self._session_states[session_id] = create_initial_state(session_id, language)
        logger.info(f"[VoiceWorkflow] Reset session: {session_id}")
    
    def _get_error_response(self, language: str) -> str:
        """Get error response message."""
        if language == "ar":
            return "عذراً، حدث خطأ. هل يمكنك المحاولة مرة أخرى؟"
        return "Sorry, something went wrong. Could you try again?"


# Global workflow instance
_voice_workflow: Optional[VoiceWorkflow] = None


def create_voice_workflow(
    enable_checkpoints: bool = True,
    **kwargs
) -> VoiceWorkflow:
    """
    Create or get the voice workflow singleton.
    
    Args:
        enable_checkpoints: Enable state persistence
        **kwargs: Additional workflow configuration
        
    Returns:
        VoiceWorkflow instance
    """
    global _voice_workflow
    
    if _voice_workflow is None:
        _voice_workflow = VoiceWorkflow(
            enable_checkpoints=enable_checkpoints,
            **kwargs
        )
    
    return _voice_workflow


def get_voice_workflow() -> Optional[VoiceWorkflow]:
    """Get the current voice workflow instance."""
    return _voice_workflow


async def process_voice_turn(
    session_id: str,
    transcript: str,
    language: str = "ar",
    interrupt_flag: bool = False
) -> VoiceAgentState:
    """
    Convenience function to process a voice turn.
    
    Creates workflow if needed and processes the turn.
    
    Args:
        session_id: Voice session ID
        transcript: User's speech text
        language: Language code
        interrupt_flag: Whether user interrupted
        
    Returns:
        Final state with response_text
    """
    workflow = create_voice_workflow()
    return await workflow.process_turn(
        session_id=session_id,
        transcript=transcript,
        language=language,
        interrupt_flag=interrupt_flag
    )
