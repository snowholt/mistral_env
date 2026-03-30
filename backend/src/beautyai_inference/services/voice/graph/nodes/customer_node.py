"""
Customer Node for Voice Agent.

Handles customer-related operations:
- Checking if customer exists (check_customer)
- Registering new customers (register_customer)

Author: BeautyAI Framework
Date: January 2026
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage

from ..state import VoiceAgentState, Intent, ToolResult
from ...tools.appointment_tools import VoiceToolExecutor

logger = logging.getLogger(__name__)

# Singleton executor instance
_tool_executor: Optional[VoiceToolExecutor] = None


def get_tool_executor(base_url: str = "http://localhost:8000") -> VoiceToolExecutor:
    """Get or create tool executor singleton."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = VoiceToolExecutor(base_url=base_url)
    return _tool_executor


async def customer_node(state: VoiceAgentState) -> VoiceAgentState:
    """
    LangGraph node that handles customer operations.
    
    Based on the detected intent:
    - CHECK_CUSTOMER: Looks up customer by name
    - REGISTER_CUSTOMER: Registers new customer
    
    The node uses the VoiceToolExecutor to make API calls
    and updates state with customer information.
    
    Args:
        state: Current voice agent state
        
    Returns:
        Updated state with customer_info and tool_results
    """
    intent = state.get("detected_intent")
    customer_info = state.get("customer_info") or {}
    session_id = state.get("session_id", "unknown")
    
    executor = get_tool_executor()
    tool_results = state.get("tool_results", [])
    
    logger.info(f"[CustomerNode] Handling intent: {intent}")
    
    if intent == Intent.CHECK_CUSTOMER:
        return await _handle_check_customer(state, executor, customer_info, tool_results, session_id)
    
    elif intent == Intent.REGISTER_CUSTOMER:
        return await _handle_register_customer(state, executor, customer_info, tool_results, session_id)
    
    # Unexpected intent, pass through
    logger.warning(f"[CustomerNode] Unexpected intent: {intent}")
    return state


async def _handle_check_customer(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    customer_info: Dict[str, Any],
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle customer lookup."""
    first_name = customer_info.get("first_name")
    last_name = customer_info.get("last_name")
    
    # Validate we have the required info
    if not first_name or not last_name:
        logger.warning("[CustomerNode] Missing name for customer check")
        return {
            **state,
            "needs_confirmation": True,
            "confirmation_type": "name_required",
            "response_text": _get_name_prompt(state.get("language", "ar")),
        }
    
    logger.info(f"[CustomerNode] Checking customer: {first_name} {last_name}")
    
    # Execute tool
    result = await executor.execute(
        tool_name="check_customer",
        parameters={
            "first_name": first_name,
            "last_name": last_name,
            "phone": customer_info.get("phone"),
        },
        session_id=session_id
    )
    
    # Create tool result
    tool_result = ToolResult(
        tool_name="check_customer",
        success=result.get("success", False),
        data=result,
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    # Update customer info based on result
    if result.get("found"):
        customer_data = result.get("customer", {})
        updated_customer = {
            **customer_info,
            "customer_id": customer_data.get("id"),
            "first_name": customer_data.get("first_name", first_name),
            "last_name": customer_data.get("last_name", last_name),
            "phone": customer_data.get("phone"),
            "email": customer_data.get("email"),
            "preferred_language": customer_data.get("preferred_language", "ar"),
            "is_registered": True,
        }
        logger.info(f"[CustomerNode] Customer found: ID={customer_data.get('id')}")
        
        return {
            **state,
            "customer_info": updated_customer,
            "tool_results": tool_results,
            "detected_intent": Intent.GENERAL_QUERY,  # Continue to response
        }
    else:
        # Customer not found - ask about registration
        logger.info("[CustomerNode] Customer not found, prompting for registration")
        
        return {
            **state,
            "customer_info": {
                **customer_info,
                "is_registered": False,
            },
            "tool_results": tool_results,
            "needs_confirmation": True,
            "confirmation_type": "registration",
        }


async def _handle_register_customer(
    state: VoiceAgentState,
    executor: VoiceToolExecutor,
    customer_info: Dict[str, Any],
    tool_results: list,
    session_id: str
) -> VoiceAgentState:
    """Handle new customer registration."""
    first_name = customer_info.get("first_name")
    last_name = customer_info.get("last_name")
    
    # Validate we have the required info
    if not first_name or not last_name:
        logger.warning("[CustomerNode] Missing name for registration")
        return {
            **state,
            "needs_confirmation": True,
            "confirmation_type": "name_required",
            "response_text": _get_name_prompt(state.get("language", "ar")),
        }
    
    logger.info(f"[CustomerNode] Registering customer: {first_name} {last_name}")
    
    # Execute tool
    result = await executor.execute(
        tool_name="register_customer",
        parameters={
            "first_name": first_name,
            "last_name": last_name,
            "phone": customer_info.get("phone"),
            "email": customer_info.get("email"),
            "preferred_language": state.get("language", "ar"),
        },
        session_id=session_id
    )
    
    # Create tool result
    tool_result = ToolResult(
        tool_name="register_customer",
        success=result.get("success", False),
        data=result,
        error=result.get("error"),
        message=result.get("message")
    )
    tool_results.append(tool_result.to_dict())
    
    if result.get("success"):
        customer_data = result.get("customer", {})
        updated_customer = {
            **customer_info,
            "customer_id": customer_data.get("id"),
            "first_name": customer_data.get("first_name", first_name),
            "last_name": customer_data.get("last_name", last_name),
            "phone": customer_data.get("phone"),
            "email": customer_data.get("email"),
            "is_registered": True,
        }
        logger.info(f"[CustomerNode] Registration successful: ID={customer_data.get('id')}")
        
        return {
            **state,
            "customer_info": updated_customer,
            "tool_results": tool_results,
            "needs_confirmation": False,
            "confirmation_type": None,
        }
    else:
        # Registration failed
        logger.error(f"[CustomerNode] Registration failed: {result.get('error')}")
        
        return {
            **state,
            "tool_results": tool_results,
            "error": result.get("error", "Registration failed"),
        }


def _get_name_prompt(language: str) -> str:
    """Get prompt asking for customer name."""
    if language == "ar":
        return "من فضلك، ما هو اسمك الكامل؟"
    return "Could you please tell me your full name?"


def _get_registration_prompt(first_name: str, language: str) -> str:
    """Get prompt asking about registration."""
    if language == "ar":
        return f"عذراً {first_name}، لم أجد حسابك في النظام. هل تريد التسجيل؟"
    return f"Sorry {first_name}, I couldn't find your account. Would you like to register?"


def _get_registration_success(first_name: str, language: str) -> str:
    """Get registration success message."""
    if language == "ar":
        return f"تم تسجيلك بنجاح يا {first_name}! كيف يمكنني مساعدتك اليوم؟"
    return f"You've been registered successfully, {first_name}! How can I help you today?"
