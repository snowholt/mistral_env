"""
Router Node for Voice Agent.

Classifies user intent using lightweight pattern matching and optional LLM routing.
Routes to appropriate handler: customer, booking, or response nodes.

Author: BeautyAI Framework
Date: January 2026
"""

import re
import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage

from ..state import VoiceAgentState, Intent

logger = logging.getLogger(__name__)


# Intent patterns for quick classification (no LLM needed)
INTENT_PATTERNS: Dict[Intent, List[str]] = {
    Intent.GREETING: [
        r"\b(hello|hi|hey|مرحبا|السلام|اهلا|صباح|مساء)\b",
        r"^(hi|hello|hey|مرحبا|اهلا|السلام عليكم)$",
    ],
    Intent.FAREWELL: [
        r"\b(bye|goodbye|شكرا|مع السلامة|الله يسلمك)\b",
        r"\b(thank you|thanks|شكراً)\b.*\b(bye|leaving|مع السلامة)\b",
    ],
    Intent.CHECK_CUSTOMER: [
        r"\b(check|verify|find|search|lookup|ابحث|تحقق)\b.*\b(customer|name|client|عميل|اسم)\b",
        r"\b(my name is|اسمي|انا)\b",
        r"\b(i am|i'm|انا)\b\s+\w+\s+\w+",  # "I am FirstName LastName"
    ],
    Intent.REGISTER_CUSTOMER: [
        r"\b(register|signup|create|new|تسجيل|جديد)\b.*\b(customer|account|client|عميل|حساب)\b",
        r"\b(not registered|لست مسجل|مو مسجل)\b",
    ],
    Intent.LIST_SLOTS: [
        r"\b(available|free|open|متاح|فاضي)\b.*\b(slot|time|appointment|موعد|وقت)\b",
        r"\b(show|list|what|when|متى|ايش)\b.*\b(available|appointments|slots|مواعيد)\b",
        r"\b(book|appointment|موعد|حجز)\b",  # Asking about booking triggers slot listing first
    ],
    Intent.BOOK_APPOINTMENT: [
        r"\b(book|reserve|confirm|schedule|احجز|اكد|جدول)\b.*\b(appointment|slot|this|هذا|الموعد)\b",
        r"\b(yes|نعم|اي|تمام|اوك)\b.*\b(book|confirm|احجز|اكد)\b",
        r"^(yes|نعم|اي|تمام|اوك|ok|okay)$",  # Simple confirmation
    ],
    Intent.CANCEL_APPOINTMENT: [
        r"\b(cancel|remove|delete|الغاء|الغي|احذف)\b.*\b(appointment|booking|موعد|حجز)\b",
    ],
    Intent.CHECK_APPOINTMENTS: [
        r"\b(my|show my|check my|مواعيدي|عندي)\b.*\b(appointments|bookings|مواعيد|حجوزات)\b",
        r"\b(what|when|متى)\b.*\b(my appointment|موعدي)\b",
    ],
}


def detect_intent_patterns(text: str) -> Optional[Intent]:
    """
    Detect intent using regex patterns.
    
    Args:
        text: User's transcribed speech
        
    Returns:
        Detected Intent or None if no pattern matches
    """
    text_lower = text.lower().strip()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE):
                logger.debug(f"Pattern matched: {intent.value} via '{pattern}'")
                return intent
    
    return None


def detect_intent_context(
    text: str,
    state: VoiceAgentState
) -> Optional[Intent]:
    """
    Detect intent based on conversation context.
    
    Args:
        text: User's transcribed speech
        state: Current conversation state
        
    Returns:
        Detected Intent based on context or None
    """
    text_lower = text.lower().strip()
    
    # If we're waiting for confirmation
    if state.get("needs_confirmation"):
        confirmation_type = state.get("confirmation_type")
        
        # Check for affirmative response
        if re.search(r"\b(yes|نعم|اي|تمام|اوك|ok|okay|sure|confirm|اكد)\b", text_lower):
            if confirmation_type == "booking":
                return Intent.BOOK_APPOINTMENT
            elif confirmation_type == "registration":
                return Intent.REGISTER_CUSTOMER
            elif confirmation_type == "cancellation":
                return Intent.CANCEL_APPOINTMENT
        
        # Check for negative response
        if re.search(r"\b(no|لا|مو|cancel|الغ|never mind)\b", text_lower):
            return Intent.GENERAL_QUERY
    
    # If we have customer info but no booking, asking about times = list slots
    customer_info = state.get("customer_info")
    if customer_info and customer_info.get("is_registered"):
        if re.search(r"\b(book|موعد|حجز|appointment|time|وقت)\b", text_lower):
            # Check if we have available slots, then might be confirming
            if state.get("available_slots") and re.search(r"\b(this|that|first|second|هذا|ذاك|الاول|الثاني)\b", text_lower):
                return Intent.BOOK_APPOINTMENT
            return Intent.LIST_SLOTS
    
    # If user mentions a name and we don't have customer info
    if not customer_info:
        # Pattern: "my name is X Y" or "I am X Y"
        name_match = re.search(
            r"(?:my name is|i am|i'm|اسمي|انا)\s+(\w+)\s+(\w+)",
            text_lower,
            re.IGNORECASE | re.UNICODE
        )
        if name_match:
            return Intent.CHECK_CUSTOMER
    
    return None


def extract_name_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Extract first and last name from user text.
    
    Args:
        text: User's speech
        
    Returns:
        Dict with first_name and last_name if found
    """
    patterns = [
        # English patterns
        r"(?:my name is|i am|i'm)\s+(\w+)\s+(\w+)",
        r"(?:this is|it's)\s+(\w+)\s+(\w+)",
        # Arabic patterns
        r"(?:اسمي|انا)\s+(\w+)\s+(\w+)",
        # Just two words (fallback for names only)
        r"^(\w{2,})\s+(\w{2,})$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            return {
                "first_name": match.group(1).strip(),
                "last_name": match.group(2).strip()
            }
    
    return None


def extract_date_preference(text: str) -> Optional[str]:
    """
    Extract date preference from user text.
    
    Args:
        text: User's speech
        
    Returns:
        Date string (YYYY-MM-DD) or relative date keyword
    """
    import datetime
    
    text_lower = text.lower()
    today = datetime.date.today()
    
    # Check for relative dates
    if re.search(r"\b(today|اليوم)\b", text_lower):
        return today.strftime("%Y-%m-%d")
    
    if re.search(r"\b(tomorrow|غدا|بكره)\b", text_lower):
        return (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Check for day names (simplified)
    days_en = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    days_ar = ["الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعة", "السبت", "الاحد"]
    
    for i, (en, ar) in enumerate(zip(days_en, days_ar)):
        if en in text_lower or ar in text_lower:
            # Calculate days until that weekday
            current_weekday = today.weekday()
            target_weekday = i
            days_ahead = (target_weekday - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week if today
            return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    return None


def extract_slot_selection(text: str, available_slots: List[Dict[str, Any]]) -> Optional[int]:
    """
    Extract slot selection from user text.
    
    Args:
        text: User's speech
        available_slots: List of available slots
        
    Returns:
        Selected slot ID or None
    """
    if not available_slots:
        return None
    
    text_lower = text.lower()
    
    # Check for ordinal selection
    ordinals = {
        "first": 0, "1st": 0, "الاول": 0, "واحد": 0, "1": 0,
        "second": 1, "2nd": 1, "الثاني": 1, "اثنين": 1, "2": 1,
        "third": 2, "3rd": 2, "الثالث": 2, "ثلاثة": 2, "3": 2,
        "fourth": 3, "4th": 3, "الرابع": 3, "اربعة": 3, "4": 3,
        "fifth": 4, "5th": 4, "الخامس": 4, "خمسة": 4, "5": 4,
    }
    
    for word, index in ordinals.items():
        if word in text_lower and index < len(available_slots):
            return available_slots[index].get("id")
    
    # Check for time mention (e.g., "10 AM", "الساعة 10")
    time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|صباحا|مساء)?", text_lower)
    if time_match:
        hour = int(time_match.group(1))
        minute = time_match.group(2) or "00"
        period = time_match.group(3)
        
        # Convert to 24-hour
        if period in ("pm", "مساء") and hour < 12:
            hour += 12
        elif period in ("am", "صباحا") and hour == 12:
            hour = 0
        
        target_time = f"{hour:02d}:{minute}"
        
        for slot in available_slots:
            if slot.get("start_time", "").startswith(target_time):
                return slot.get("id")
    
    return None


async def router_node(state: VoiceAgentState) -> VoiceAgentState:
    """
    LangGraph node that routes user input to appropriate handler.
    
    This node:
    1. Extracts the current transcript
    2. Detects intent using patterns + context
    3. Extracts relevant entities (names, dates, slot selections)
    4. Updates state with detected intent and extracted data
    
    Args:
        state: Current voice agent state
        
    Returns:
        Updated state with detected_intent and extracted entities
    """
    transcript = state.get("current_transcript", "").strip()
    
    if not transcript:
        logger.warning("[Router] Empty transcript received")
        return {
            **state,
            "detected_intent": Intent.UNCLEAR,
            "error": "Empty transcript"
        }
    
    logger.info(f"[Router] Processing: '{transcript[:80]}...'")
    
    # Add user message to history
    new_messages = [HumanMessage(content=transcript)]
    
    # Step 1: Try pattern-based detection
    intent = detect_intent_patterns(transcript)
    
    # Step 2: Try context-based detection if no pattern match
    if intent is None:
        intent = detect_intent_context(transcript, state)
    
    # Step 3: Default to general query
    if intent is None:
        intent = Intent.GENERAL_QUERY
    
    logger.info(f"[Router] Detected intent: {intent.value}")
    
    # Extract entities based on intent
    updates: Dict[str, Any] = {
        "messages": new_messages,
        "detected_intent": intent,
        "error": None,
    }
    
    # Extract customer name if checking/registering
    if intent in (Intent.CHECK_CUSTOMER, Intent.REGISTER_CUSTOMER):
        name_data = extract_name_from_text(transcript)
        if name_data:
            current_customer = state.get("customer_info") or {}
            updates["customer_info"] = {
                **current_customer,
                "first_name": name_data["first_name"],
                "last_name": name_data["last_name"],
            }
            logger.info(f"[Router] Extracted name: {name_data}")
    
    # Extract date preference if listing slots
    if intent == Intent.LIST_SLOTS:
        date_pref = extract_date_preference(transcript)
        if date_pref:
            current_prefs = state.get("slot_preferences") or {}
            updates["slot_preferences"] = {
                **current_prefs,
                "preferred_date": date_pref,
            }
            logger.info(f"[Router] Extracted date preference: {date_pref}")
    
    # Extract slot selection if booking
    if intent == Intent.BOOK_APPOINTMENT:
        available_slots = state.get("available_slots", [])
        slot_id = extract_slot_selection(transcript, available_slots)
        if slot_id:
            # Find the slot details
            selected_slot = next((s for s in available_slots if s.get("id") == slot_id), None)
            updates["pending_booking"] = {
                "time_slot_id": slot_id,
                "time_slot_date": selected_slot.get("date") if selected_slot else None,
                "time_slot_time": selected_slot.get("start_time") if selected_slot else None,
                "awaiting_confirmation": False,
            }
            logger.info(f"[Router] Selected slot: {slot_id}")
    
    # Clear confirmation state if user is asking something new
    if intent not in (Intent.BOOK_APPOINTMENT, Intent.REGISTER_CUSTOMER, Intent.CANCEL_APPOINTMENT):
        if state.get("needs_confirmation"):
            updates["needs_confirmation"] = False
            updates["confirmation_type"] = None
    
    return {**state, **updates}


def get_next_node(state: VoiceAgentState) -> str:
    """
    Determine which node to route to based on detected intent.
    
    Used as conditional edge function in the graph.
    
    Args:
        state: Current voice agent state
        
    Returns:
        Name of next node to execute
    """
    intent = state.get("detected_intent", Intent.UNCLEAR)
    
    # Route to customer node for customer-related intents
    if intent in (Intent.CHECK_CUSTOMER, Intent.REGISTER_CUSTOMER):
        return "customer"
    
    # Route to booking node for appointment-related intents
    if intent in (Intent.LIST_SLOTS, Intent.BOOK_APPOINTMENT, Intent.CANCEL_APPOINTMENT, Intent.CHECK_APPOINTMENTS):
        return "booking"
    
    # Everything else goes to response node
    return "response"
