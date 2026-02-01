"""
Voice Agent State Definition for LangGraph.

Defines the state schema that flows through the graph, holding:
- Message history
- Customer information
- Detected intent
- Slot preferences and pending bookings
- Interruption flags

Author: BeautyAI Framework
Date: January 2026
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal, Annotated
from enum import Enum
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from operator import add


class Intent(str, Enum):
    """User intent categories for routing."""
    GREETING = "greeting"
    CHECK_CUSTOMER = "check_customer"
    REGISTER_CUSTOMER = "register_customer"
    LIST_SLOTS = "list_slots"
    BOOK_APPOINTMENT = "book_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    CHECK_APPOINTMENTS = "check_appointments"
    GENERAL_QUERY = "general_query"
    FAREWELL = "farewell"
    UNCLEAR = "unclear"


@dataclass
class CustomerInfo:
    """Customer information extracted during conversation."""
    customer_id: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    preferred_language: str = "ar"
    is_registered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone": self.phone,
            "email": self.email,
            "preferred_language": self.preferred_language,
            "is_registered": self.is_registered,
        }
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CustomerInfo"]:
        if not data:
            return None
        return cls(**data)


@dataclass  
class SlotPreferences:
    """User's appointment preferences."""
    preferred_date: Optional[str] = None  # YYYY-MM-DD
    preferred_time: Optional[str] = None  # HH:MM
    days_ahead: int = 7
    service_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_date": self.preferred_date,
            "preferred_time": self.preferred_time,
            "days_ahead": self.days_ahead,
            "service_type": self.service_type,
        }


@dataclass
class PendingBooking:
    """Pending booking awaiting confirmation."""
    time_slot_id: Optional[int] = None
    time_slot_date: Optional[str] = None
    time_slot_time: Optional[str] = None
    service_type: str = "consultation"
    notes: Optional[str] = None
    awaiting_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_slot_id": self.time_slot_id,
            "time_slot_date": self.time_slot_date,
            "time_slot_time": self.time_slot_time,
            "service_type": self.service_type,
            "notes": self.notes,
            "awaiting_confirmation": self.awaiting_confirmation,
        }


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "message": self.message,
        }


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer that appends messages (with deduplication by id if present)."""
    # Simple append for now
    return left + right


class VoiceAgentState(TypedDict, total=False):
    """
    State schema for the Voice Agent LangGraph workflow.
    
    This state flows through all nodes and accumulates information
    throughout the conversation turn.
    
    Attributes:
        messages: Chat message history (accumulated)
        current_transcript: Latest user speech transcription
        detected_intent: Classified intent from router
        customer_info: Customer data (lookup/registration results)
        slot_preferences: User's appointment preferences
        pending_booking: Booking awaiting confirmation
        available_slots: List of available time slots
        tool_results: Results from tool executions in this turn
        response_text: Generated response text for TTS
        language: Detected/preferred language (ar/en)
        session_id: Voice session identifier
        interrupt_flag: Whether user interrupted during processing
        needs_confirmation: Whether we're waiting for user confirmation
        error: Error message if something went wrong
    """
    # Message history (uses add_messages reducer for accumulation)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current turn data
    current_transcript: str
    detected_intent: Intent
    
    # Customer context
    customer_info: Optional[Dict[str, Any]]
    
    # Booking context  
    slot_preferences: Optional[Dict[str, Any]]
    pending_booking: Optional[Dict[str, Any]]
    available_slots: Optional[List[Dict[str, Any]]]
    
    # Tool execution
    tool_results: List[Dict[str, Any]]
    
    # Response generation
    response_text: str
    language: Literal["ar", "en"]
    
    # Session metadata
    session_id: str
    interrupt_flag: bool
    needs_confirmation: bool
    confirmation_type: Optional[str]  # "booking", "registration", "cancellation"
    
    # Error handling
    error: Optional[str]


def create_initial_state(
    session_id: str,
    language: str = "ar",
    system_message: Optional[str] = None
) -> VoiceAgentState:
    """
    Create initial state for a new voice conversation.
    
    Args:
        session_id: Unique session identifier
        language: Default language (ar/en)
        system_message: Optional custom system prompt
        
    Returns:
        Initialized VoiceAgentState
    """
    messages: List[BaseMessage] = []
    
    if system_message:
        messages.append(SystemMessage(content=system_message))
    
    return VoiceAgentState(
        messages=messages,
        current_transcript="",
        detected_intent=Intent.UNCLEAR,
        customer_info=None,
        slot_preferences=None,
        pending_booking=None,
        available_slots=None,
        tool_results=[],
        response_text="",
        language=language,
        session_id=session_id,
        interrupt_flag=False,
        needs_confirmation=False,
        confirmation_type=None,
        error=None,
    )


def get_customer_from_state(state: VoiceAgentState) -> Optional[CustomerInfo]:
    """Extract CustomerInfo from state dict."""
    return CustomerInfo.from_dict(state.get("customer_info"))


def set_customer_in_state(state: VoiceAgentState, customer: CustomerInfo) -> VoiceAgentState:
    """Update customer_info in state."""
    state["customer_info"] = customer.to_dict()
    return state
