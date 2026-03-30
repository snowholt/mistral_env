"""
Response Node for Voice Agent.

Generates natural language responses based on:
- Current conversation state
- Tool execution results
- Customer context
- Language preferences

Uses the LLM to generate contextual responses for TTS synthesis.

Author: BeautyAI Framework
Date: January 2026
"""

import logging
from typing import Dict, Any, Optional, List, AsyncIterator, Callable, Awaitable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from ..state import VoiceAgentState, Intent

logger = logging.getLogger(__name__)


# Response templates for common scenarios (fallback if LLM unavailable)
RESPONSE_TEMPLATES = {
    "greeting": {
        "ar": "أهلاً وسهلاً! أنا المساعد الصوتي لعيادة كيسي. كيف يمكنني مساعدتك اليوم؟",
        "en": "Hello! I'm the voice assistant for Kesay Clinic. How can I help you today?",
    },
    "farewell": {
        "ar": "شكراً لتواصلك معنا. مع السلامة!",
        "en": "Thank you for contacting us. Goodbye!",
    },
    "customer_found": {
        "ar": "مرحباً {first_name}! سعيدة بالتحدث معك مجدداً. كيف يمكنني مساعدتك؟",
        "en": "Hello {first_name}! Nice to speak with you again. How can I help?",
    },
    "customer_not_found": {
        "ar": "عذراً {first_name}، لم أجد حسابك. هل تريد التسجيل معنا؟",
        "en": "Sorry {first_name}, I couldn't find your account. Would you like to register?",
    },
    "registration_success": {
        "ar": "تم تسجيلك بنجاح يا {first_name}! هل تريد حجز موعد؟",
        "en": "You're now registered, {first_name}! Would you like to book an appointment?",
    },
    "slots_available": {
        "ar": "المواعيد المتاحة: {slots}. أي موعد تفضل؟",
        "en": "Available times: {slots}. Which one would you prefer?",
    },
    "no_slots": {
        "ar": "عذراً، لا توجد مواعيد متاحة في الفترة المطلوبة. هل تريد البحث في فترة أخرى؟",
        "en": "Sorry, no appointments available for that period. Would you like to search another time?",
    },
    "booking_success": {
        "ar": "تم حجز موعدك بنجاح يوم {date} الساعة {time}. سنراك قريباً!",
        "en": "Your appointment is confirmed for {date} at {time}. See you soon!",
    },
    "booking_failed": {
        "ar": "عذراً، لم أتمكن من تأكيد الحجز. هل تريد المحاولة مرة أخرى؟",
        "en": "Sorry, I couldn't confirm the booking. Would you like to try again?",
    },
    "cancellation_success": {
        "ar": "تم إلغاء موعدك بنجاح.",
        "en": "Your appointment has been cancelled successfully.",
    },
    "customer_required": {
        "ar": "أحتاج للتعرف عليك أولاً. ما هو اسمك الكامل من فضلك؟",
        "en": "I need to know who you are first. What's your full name, please?",
    },
    "unclear": {
        "ar": "عذراً، لم أفهم طلبك. هل يمكنك إعادة صياغته؟",
        "en": "Sorry, I didn't understand. Could you rephrase that?",
    },
    "error": {
        "ar": "عذراً، حدث خطأ. هل يمكنك المحاولة مرة أخرى؟",
        "en": "Sorry, something went wrong. Could you try again?",
    },
}


# Type for LLM generation callback
LLMGeneratorType = Callable[[List[BaseMessage], str], Awaitable[str]]
LLMStreamGeneratorType = Callable[[List[BaseMessage], str], AsyncIterator[str]]


class ResponseGenerator:
    """
    Generates responses using LLM or templates.
    
    Can be configured with a custom LLM generator for integration
    with the existing inference pipeline.
    """
    
    def __init__(
        self,
        llm_generator: Optional[LLMGeneratorType] = None,
        llm_stream_generator: Optional[LLMStreamGeneratorType] = None,
        use_templates_only: bool = False
    ):
        """
        Initialize response generator.
        
        Args:
            llm_generator: Async function that takes messages and returns response
            llm_stream_generator: Async generator that yields response tokens
            use_templates_only: If True, only use templates (no LLM)
        """
        self.llm_generator = llm_generator
        self.llm_stream_generator = llm_stream_generator
        self.use_templates_only = use_templates_only
    
    async def generate(
        self,
        state: VoiceAgentState,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response based on current state.
        
        Args:
            state: Current voice agent state
            system_prompt: Optional system prompt override
            
        Returns:
            Generated response text
        """
        # Check if we already have a response set
        if state.get("response_text"):
            return state["response_text"]
        
        language = state.get("language", "ar")
        intent = state.get("detected_intent", Intent.UNCLEAR)
        
        # Try template-based response first for common cases
        template_response = self._get_template_response(state, intent, language)
        
        if self.use_templates_only or not self.llm_generator:
            return template_response
        
        # Build context for LLM
        messages = self._build_llm_context(state, system_prompt)
        
        try:
            response = await self.llm_generator(messages, language)
            return response.strip() if response else template_response
        except Exception as e:
            logger.error(f"[ResponseGenerator] LLM generation failed: {e}")
            return template_response
    
    async def generate_stream(
        self,
        state: VoiceAgentState,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream response tokens for TTS.
        
        Args:
            state: Current voice agent state
            system_prompt: Optional system prompt override
            
        Yields:
            Response tokens
        """
        if not self.llm_stream_generator or self.use_templates_only:
            # Fall back to full response
            response = await self.generate(state, system_prompt)
            yield response
            return
        
        messages = self._build_llm_context(state, system_prompt)
        
        try:
            async for token in self.llm_stream_generator(messages, state.get("language", "ar")):
                yield token
        except Exception as e:
            logger.error(f"[ResponseGenerator] LLM streaming failed: {e}")
            yield await self.generate(state, system_prompt)
    
    def _get_template_response(
        self,
        state: VoiceAgentState,
        intent: Intent,
        language: str
    ) -> str:
        """Get template-based response."""
        customer_info = state.get("customer_info") or {}
        tool_results = state.get("tool_results", [])
        first_name = customer_info.get("first_name", "")
        
        # Check for errors first
        error = state.get("error")
        if error:
            if error == "no_slots_available":
                return RESPONSE_TEMPLATES["no_slots"][language]
            elif error == "customer_required":
                return RESPONSE_TEMPLATES["customer_required"][language]
            return RESPONSE_TEMPLATES["error"][language]
        
        # Check for booking success
        if state.get("_booking_success"):
            booking = state.get("_booked_appointment", {})
            slot = booking.get("time_slot", {})
            return RESPONSE_TEMPLATES["booking_success"][language].format(
                date=slot.get("date", ""),
                time=slot.get("start_time", "")
            )
        
        # Check for cancellation success
        if state.get("_cancellation_success"):
            return RESPONSE_TEMPLATES["cancellation_success"][language]
        
        # Check needs_confirmation states
        if state.get("needs_confirmation"):
            confirmation_type = state.get("confirmation_type")
            
            if confirmation_type == "registration":
                return RESPONSE_TEMPLATES["customer_not_found"][language].format(
                    first_name=first_name
                )
            elif confirmation_type == "slot_selection":
                formatted_slots = state.get("_slots_formatted", "")
                if formatted_slots:
                    return formatted_slots
                return RESPONSE_TEMPLATES["slots_available"][language].format(slots="")
        
        # Intent-based templates
        if intent == Intent.GREETING:
            if customer_info.get("is_registered"):
                return RESPONSE_TEMPLATES["customer_found"][language].format(
                    first_name=first_name
                )
            return RESPONSE_TEMPLATES["greeting"][language]
        
        elif intent == Intent.FAREWELL:
            return RESPONSE_TEMPLATES["farewell"][language]
        
        elif intent == Intent.CHECK_CUSTOMER:
            # Check last tool result
            for result in reversed(tool_results):
                if result.get("tool_name") == "check_customer":
                    if result.get("data", {}).get("found"):
                        return RESPONSE_TEMPLATES["customer_found"][language].format(
                            first_name=first_name
                        )
                    else:
                        return RESPONSE_TEMPLATES["customer_not_found"][language].format(
                            first_name=first_name
                        )
        
        elif intent == Intent.REGISTER_CUSTOMER:
            for result in reversed(tool_results):
                if result.get("tool_name") == "register_customer":
                    if result.get("success"):
                        return RESPONSE_TEMPLATES["registration_success"][language].format(
                            first_name=first_name
                        )
        
        elif intent == Intent.LIST_SLOTS:
            formatted_slots = state.get("_slots_formatted")
            if formatted_slots:
                return formatted_slots
            available_slots = state.get("available_slots", [])
            if not available_slots:
                return RESPONSE_TEMPLATES["no_slots"][language]
        
        elif intent == Intent.UNCLEAR:
            return RESPONSE_TEMPLATES["unclear"][language]
        
        # Default fallback
        return RESPONSE_TEMPLATES["greeting"][language]
    
    def _build_llm_context(
        self,
        state: VoiceAgentState,
        system_prompt: Optional[str]
    ) -> List[BaseMessage]:
        """Build message context for LLM."""
        messages: List[BaseMessage] = []
        
        # System prompt (minimal for context window efficiency)
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        else:
            # Minimal system prompt based on current context
            language = state.get("language", "ar")
            messages.append(SystemMessage(content=self._get_minimal_system_prompt(state, language)))
        
        # Add relevant conversation history (last few turns)
        history = state.get("messages", [])
        for msg in history[-6:]:  # Last 3 turns
            messages.append(msg)
        
        # Add tool results as context
        tool_results = state.get("tool_results", [])
        if tool_results:
            result_summary = self._summarize_tool_results(tool_results)
            messages.append(SystemMessage(content=f"Tool results: {result_summary}"))
        
        return messages
    
    def _get_minimal_system_prompt(self, state: VoiceAgentState, language: str) -> str:
        """Get minimal system prompt for current context."""
        customer_info = state.get("customer_info") or {}
        intent = state.get("detected_intent", Intent.UNCLEAR)
        
        if language == "ar":
            prompt = "أنت مساعد صوتي ودود لعيادة تجميل. أجب بإيجاز ووضوح."
            if customer_info.get("is_registered"):
                prompt += f" اسم العميل: {customer_info.get('first_name', '')}."
            if intent == Intent.LIST_SLOTS:
                prompt += " ساعد في اختيار موعد مناسب."
            elif intent == Intent.BOOK_APPOINTMENT:
                prompt += " أكد تفاصيل الحجز."
        else:
            prompt = "You are a friendly voice assistant for a beauty clinic. Be brief and clear."
            if customer_info.get("is_registered"):
                prompt += f" Customer: {customer_info.get('first_name', '')}."
            if intent == Intent.LIST_SLOTS:
                prompt += " Help choose a suitable appointment."
            elif intent == Intent.BOOK_APPOINTMENT:
                prompt += " Confirm booking details."
        
        return prompt
    
    def _summarize_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Summarize tool results for context."""
        summaries = []
        for result in tool_results:
            name = result.get("tool_name", "unknown")
            success = result.get("success", False)
            message = result.get("message", "")
            summaries.append(f"{name}:{'OK' if success else 'FAIL'},{message[:50]}")
        return "; ".join(summaries)


# Global response generator instance
_response_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    """Get or create response generator singleton."""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator(use_templates_only=True)
    return _response_generator


def set_response_generator(generator: ResponseGenerator):
    """Set custom response generator (for LLM integration)."""
    global _response_generator
    _response_generator = generator


async def response_node(state: VoiceAgentState) -> VoiceAgentState:
    """
    LangGraph node that generates the final response.
    
    Uses the ResponseGenerator to create a natural language
    response for TTS synthesis.
    
    Args:
        state: Current voice agent state
        
    Returns:
        Updated state with response_text
    """
    generator = get_response_generator()
    
    logger.info(f"[ResponseNode] Generating response for intent: {state.get('detected_intent')}")
    
    try:
        response_text = await generator.generate(state)
        
        logger.info(f"[ResponseNode] Generated: '{response_text[:80]}...'")
        
        # Add AI message to history
        new_messages = [AIMessage(content=response_text)]
        
        return {
            **state,
            "messages": new_messages,
            "response_text": response_text,
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"[ResponseNode] Generation failed: {e}")
        
        language = state.get("language", "ar")
        fallback = RESPONSE_TEMPLATES["error"][language]
        
        return {
            **state,
            "response_text": fallback,
            "error": str(e),
        }
