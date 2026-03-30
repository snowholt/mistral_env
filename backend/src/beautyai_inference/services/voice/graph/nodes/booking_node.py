"""
Booking Node for Voice Agent.

Handles appointment-related operations:
- Listing available slots (list_available_slots)
- Booking appointments (book_appointment)
- Cancelling appointments (cancel_appointment)
- Checking customer appointments (get_customer_appointments)

Author: BeautyAI Framework
Date: January 2026
"""

import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage

from ..state import VoiceAgentState, Intent, ToolResult
from ...tools.appointment_tools import VoiceToolExecutor
from .customer_node import get_tool_executor

logger = logging.getLogger(__name__)


async def booking_node(state: VoiceAgentState) -> VoiceAgentState:
    """
    LangGraph node that handles booking operations.
    
    Based on the detected intent:
    - LIST_SLOTS: Lists available appointment slots
    - BOOK_APPOINTMENT: Books an appointment
    - CANCEL_APPOINTMENT: Cancels an existing appointment
    - CHECK_APPOINTMENTS: Shows customer's appointments
    
    Args:
        state: Current voice agent state
        
    Returns:
        Updated state with booking info and tool_results
    """
    intent = state.get("detected_intent")
    session_id = state.get("session_id", "unknown")
    
    executor = get_tool_executor()
    tool_results = state.get("tool_results", [])
    
    logger.info(f"[BookingNode] Handling intent: {intent}")
    
    if intent == Intent.LIST_SLOTS:
        return await _handle_list_slots(state, executor, tool_results, session_id)
    
    elif intent == Intent.BOOK_APPOINTMENT:
        return await _handle_book_appointment(state, executor, tool_results, session_id)
    
    elif intent == Intent.CANCEL_APPOINTMENT:
        return await _handle_cancel_appointment(state, executor, tool_results, session_id)
    
    elif intent == Intent.CHECK_APPOINTMENTS:
        return await _handle_check_appointments(state, executor, tool_results, session_id)
    
    # Unexpected intent, pass through
    logger.warning(f"[BookingNode] Unexpected intent: {intent}")
    return state


async def _handle_list_slots(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle listing available appointment slots."""
    slot_preferences = state.get("slot_preferences") or {}
    language = state.get("language", "ar")
    
    logger.info(f"[BookingNode] Listing slots with preferences: {slot_preferences}")
    
    # Execute tool
    result = await executor.execute(
        tool_name="list_available_slots",
        parameters={
            "date": slot_preferences.get("preferred_date"),
            "days_ahead": slot_preferences.get("days_ahead", 7),
        },
        session_id=session_id
    )
    
    # Create tool result
    tool_result = ToolResult(
        tool_name="list_available_slots",
        success=result.get("success", False),
        data=result,
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    available_slots = result.get("available_slots", [])
    
    if available_slots:
        logger.info(f"[BookingNode] Found {len(available_slots)} available slots")
        
        # Format slots for response
        formatted = _format_slots_for_response(available_slots, language)
        
        return {
            **state,
            "available_slots": available_slots,
            "tool_results": tool_results,
            "needs_confirmation": True,
            "confirmation_type": "slot_selection",
            # Include formatted slots in response context
            "_slots_formatted": formatted,
        }
    else:
        logger.info("[BookingNode] No available slots found")
        
        return {
            **state,
            "available_slots": [],
            "tool_results": tool_results,
            "error": "no_slots_available",
        }


async def _handle_book_appointment(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle booking an appointment."""
    customer_info = state.get("customer_info") or {}
    pending_booking = state.get("pending_booking") or {}
    language = state.get("language", "ar")
    
    # Validate customer is registered
    customer_id = customer_info.get("customer_id")
    if not customer_id:
        logger.warning("[BookingNode] No customer ID for booking")
        return {
            **state,
            "error": "customer_required",
            "response_text": _get_customer_required_message(language),
        }
    
    # Validate we have a slot selected
    time_slot_id = pending_booking.get("time_slot_id")
    if not time_slot_id:
        # Check if we have available slots to select from
        available_slots = state.get("available_slots", [])
        if available_slots:
            logger.info("[BookingNode] No slot selected, asking user to choose")
            return {
                **state,
                "needs_confirmation": True,
                "confirmation_type": "slot_selection",
            }
        else:
            # Need to list slots first
            logger.info("[BookingNode] No slots available, listing first")
            return {
                **state,
                "detected_intent": Intent.LIST_SLOTS,
            }
    
    logger.info(f"[BookingNode] Booking appointment: customer={customer_id}, slot={time_slot_id}")
    
    # Execute tool
    result = await executor.execute(
        tool_name="book_appointment",
        parameters={
            "customer_id": customer_id,
            "time_slot_id": time_slot_id,
            "service_type": pending_booking.get("service_type", "consultation"),
            "notes": pending_booking.get("notes"),
        },
        session_id=session_id
    )
    
    # Create tool result
    tool_result = ToolResult(
        tool_name="book_appointment",
        success=result.get("success", False),
        data=result,
        error=result.get("error"),
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    if result.get("success"):
        appointment = result.get("appointment", {})
        logger.info(f"[BookingNode] Booking successful: ID={appointment.get('id')}")
        
        return {
            **state,
            "tool_results": tool_results,
            "pending_booking": None,  # Clear pending
            "available_slots": None,  # Clear slots
            "needs_confirmation": False,
            "_booking_success": True,
            "_booked_appointment": appointment,
        }
    else:
        logger.error(f"[BookingNode] Booking failed: {result.get('error')}")
        
        return {
            **state,
            "tool_results": tool_results,
            "error": result.get("error", "Booking failed"),
        }


async def _handle_cancel_appointment(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle cancelling an appointment."""
    language = state.get("language", "ar")
    
    # We need the appointment ID - might be in state or need to be extracted
    # For now, we'll need the user to confirm which appointment
    customer_info = state.get("customer_info") or {}
    customer_id = customer_info.get("customer_id")
    
    if not customer_id:
        return {
            **state,
            "error": "customer_required",
            "response_text": _get_customer_required_message(language),
        }
    
    # Get customer's appointments first to show options
    result = await executor.execute(
        tool_name="get_customer_appointments",
        parameters={
            "customer_id": customer_id,
            "include_cancelled": False,
        },
        session_id=session_id
    )
    
    tool_result = ToolResult(
        tool_name="get_customer_appointments",
        success=result.get("success", False),
        data=result,
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    appointments = result.get("appointments", [])
    
    if not appointments:
        return {
            **state,
            "tool_results": tool_results,
            "error": "no_appointments",
        }
    
    # For simplicity, cancel the first/most recent appointment
    # In production, would ask user to select
    appointment_to_cancel = appointments[0]
    
    cancel_result = await executor.execute(
        tool_name="cancel_appointment",
        parameters={
            "appointment_id": appointment_to_cancel.get("id"),
        },
        session_id=session_id
    )
    
    cancel_tool_result = ToolResult(
        tool_name="cancel_appointment",
        success=cancel_result.get("success", False),
        data=cancel_result,
        error=cancel_result.get("error"),
        message=cancel_result.get("message")
    )
    tool_results.append(cancel_tool_result.to_dict())
    
    if cancel_result.get("success"):
        logger.info(f"[BookingNode] Cancellation successful")
        return {
            **state,
            "tool_results": tool_results,
            "_cancellation_success": True,
        }
    else:
        return {
            **state,
            "tool_results": tool_results,
            "error": cancel_result.get("error", "Cancellation failed"),
        }


async def _handle_check_appointments(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle checking customer's appointments."""
    customer_info = state.get("customer_info") or {}
    language = state.get("language", "ar")
    
    customer_id = customer_info.get("customer_id")
    if not customer_id:
        return {
            **state,
            "error": "customer_required",
            "response_text": _get_customer_required_message(language),
        }
    
    logger.info(f"[BookingNode] Getting appointments for customer: {customer_id}")
    
    result = await executor.execute(
        tool_name="get_customer_appointments",
        parameters={
            "customer_id": customer_id,
            "include_cancelled": False,
        },
        session_id=session_id
    )
    
    tool_result = ToolResult(
        tool_name="get_customer_appointments",
        success=result.get("success", False),
        data=result,
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    appointments = result.get("appointments", [])
    
    return {
        **state,
        "tool_results": tool_results,
        "_customer_appointments": appointments,
    }


def _format_slots_for_response(slots: List[Dict[str, Any]], language: str) -> str:
    """Format available slots for voice response."""
    if not slots:
        if language == "ar":
            return "عذراً، لا توجد مواعيد متاحة في الوقت الحالي."
        return "Sorry, there are no available appointments at the moment."
    
    # Group by date
    slots_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for slot in slots[:10]:  # Limit to first 10 for voice
        date = slot.get("date", "Unknown")
        if date not in slots_by_date:
            slots_by_date[date] = []
        slots_by_date[date].append(slot)
    
    lines = []
    if language == "ar":
        lines.append("المواعيد المتاحة هي:")
        for date, date_slots in slots_by_date.items():
            times = [s.get("start_time", "")[:5] for s in date_slots]
            lines.append(f"{date}: {', '.join(times)}")
        lines.append("أي موعد تفضل؟")
    else:
        lines.append("Available appointments are:")
        for date, date_slots in slots_by_date.items():
            times = [s.get("start_time", "")[:5] for s in date_slots]
            lines.append(f"{date}: {', '.join(times)}")
        lines.append("Which one would you prefer?")
    
    return " ".join(lines)


def _get_customer_required_message(language: str) -> str:
    """Get message when customer is not registered."""
    if language == "ar":
        return "عذراً، أحتاج للتعرف عليك أولاً. ما هو اسمك الكامل؟"
    return "Sorry, I need to know who you are first. What's your full name?"
