"""
Prompt Building Service

This service handles prompt construction for the BeautyAI chat system,
including system prompts, language-specific instructions, and conversation history.
"""
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptBuildingService:
    """
    Service for building prompts for BeautyAI chat system.
    
    Handles system prompts, language detection, conversation history management,
    and context window optimization.
    """
    
    def __init__(self):
        self.system_prompts = {
            "ar": """أنت طبيب متخصص في الطب التجميلي والعلاجات غير الجراحية. يجب عليك الإجابة باللغة العربية فقط. 
قدم معلومات طبية دقيقة ومفيدة حول العلاجات التجميلية مثل البوتوكس والفيلر. 
اجعل إجاباتك واضحة ومختصرة ومناسبة للمرضى العرب.
مهم جداً: أجب باللغة العربية فقط ولا تستخدم أي لغة أخرى.""",
            
            "es": "Eres un médico especializado en medicina estética y tratamientos no quirúrgicos. Debes responder solo en español. Proporciona información médica precisa y útil sobre tratamientos estéticos como botox y rellenos. Haz que tus respuestas sean claras, concisas y apropiadas para pacientes hispanohablantes.",
            
            "fr": "Vous êtes un médecin spécialisé en médecine esthétique et en traitements non chirurgicaux. Vous devez répondre uniquement en français. Fournissez des informations médicales précises et utiles sur les traitements esthétiques comme le botox et les fillers. Rendez vos réponses claires, concises et appropriées pour les patients francophones.",
            
            "de": "Sie sind ein Arzt, der sich auf ästhetische Medizin und nicht-chirurgische Behandlungen spezialisiert hat. Sie müssen nur auf Deutsch antworten. Stellen Sie präzise und nützliche medizinische Informationen über ästhetische Behandlungen wie Botox und Filler zur Verfügung. Machen Sie Ihre Antworten klar, prägnant und angemessen für deutschsprachige Patienten.",
            
            "en": "You are a doctor specialized in aesthetic medicine and non-surgical treatments. You must respond only in English. Provide accurate and useful medical information about aesthetic treatments like botox and fillers. Make your responses clear, concise, and appropriate for English-speaking patients."
        }
        
        self.language_reinforcement = {
            "ar": [
                ("User: من فضلك أجب باللغة العربية فقط", "Assistant: سأجيب باللغة العربية.")
            ]
        }
    
    def build_prompt(
        self,
        message: str,
        language: str = "ar",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build a complete prompt for the chat model.
        
        Args:
            message: The user's current message
            language: The response language ('ar', 'es', 'fr', 'de', 'en')
            conversation_history: Previous conversation history
            generation_config: Model generation configuration
            
        Returns:
            Tuple of (built_prompt, trimmed_history)
        """
        if generation_config is None:
            generation_config = {}
        
        # Start building prompt parts
        prompt_parts = []
        
        # Add system prompt
        system_prompt = self.system_prompts.get(language, self.system_prompts["en"])
        prompt_parts.append(f"System: {system_prompt}")
        
        # Add language reinforcement if needed
        if language in self.language_reinforcement:
            for user_msg, assistant_msg in self.language_reinforcement[language]:
                prompt_parts.append(user_msg)
                prompt_parts.append(assistant_msg)
        
        # Manage conversation history with context window optimization
        trimmed_history = self._manage_conversation_history(
            conversation_history or [],
            message,
            prompt_parts,
            generation_config
        )
        
        # Add conversation history to prompt
        for msg in trimmed_history:
            if msg.get("role") == "user":
                prompt_parts.append(f"User: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                prompt_parts.append(f"Assistant: {msg.get('content', '')}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        # Build final prompt
        prompt = "\n".join(prompt_parts)
        
        return prompt, trimmed_history
    
    def _manage_conversation_history(
        self,
        conversation_history: List[Dict[str, str]],
        current_message: str,
        prompt_parts: List[str],
        generation_config: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Manage conversation history to fit within context window.
        
        Args:
            conversation_history: Full conversation history
            current_message: Current user message
            prompt_parts: Current prompt parts (system, etc.)
            generation_config: Model generation configuration
            
        Returns:
            Trimmed conversation history
        """
        if not conversation_history:
            return []
        
        full_history = conversation_history[:]
        
        # 1. Hard cap on number of turns
        max_turns = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "6"))
        if max_turns > 0:
            full_history = self._trim_by_turns(full_history, max_turns)
        
        # 2. Token estimation based trimming
        max_length = int(generation_config.get("max_length", os.getenv("CHAT_MAX_LENGTH", "512")))
        context_limit = int(generation_config.get("context_window", os.getenv("CHAT_CONTEXT_WINDOW", "4096")))
        safety_margin = int(os.getenv("CHAT_CONTEXT_SAFETY_MARGIN", "160"))
        
        full_history = self._trim_by_tokens(
            full_history,
            current_message,
            prompt_parts,
            max_length,
            context_limit,
            safety_margin
        )
        
        return full_history
    
    def _trim_by_turns(self, history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
        """Trim history by number of conversation turns."""
        if not history:
            return []
        
        kept = []
        user_count = 0
        assistant_count = 0
        turns = 0
        
        for msg in reversed(history):
            kept.append(msg)
            if msg.get("role") == "assistant":
                assistant_count += 1
            elif msg.get("role") == "user":
                user_count += 1
            
            # A completed turn counted when we have both a user and assistant
            if user_count > 0 and assistant_count > 0:
                turns += 1
                user_count = 0
                assistant_count = 0
            
            if turns >= max_turns:
                break
        
        # Reverse back to chronological order
        trimmed_history = list(reversed(kept))
        
        if len(trimmed_history) < len(history):
            logger.debug(
                "[prompt] History turn cap applied (max_turns=%d) trimmed=%d->%d",
                max_turns, len(history), len(trimmed_history)
            )
        
        return trimmed_history
    
    def _trim_by_tokens(
        self,
        history: List[Dict[str, str]],
        current_message: str,
        prompt_parts: List[str],
        max_length: int,
        context_limit: int,
        safety_margin: int
    ) -> List[Dict[str, str]]:
        """Trim history by estimated token count."""
        if not history:
            return history
        
        def est_tokens_text(txt: str) -> int:
            """Heuristic token estimator: roughly 1 token ~ 4 characters."""
            if not txt:
                return 0
            # Blend word-based and char-based estimates, take max for safety
            word_est = len(txt.split())
            char_est = max(1, len(txt) // 4)
            return max(word_est, char_est)
        
        def build_hist_parts(hist: List[Dict[str, str]]) -> List[str]:
            """Build history parts for token estimation."""
            tmp = []
            for m in hist:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    tmp.append(f"User: {content}")
                elif role == "assistant":
                    tmp.append(f"Assistant: {content}")
            return tmp
        
        def total_estimate(hist: List[Dict[str, str]]) -> int:
            """Estimate total tokens for prompt + history + current message."""
            parts = prompt_parts[:]  # system prompt parts
            parts.extend(build_hist_parts(hist))
            parts.append(f"User: {current_message}")
            parts.append("Assistant:")
            return sum(est_tokens_text(p) for p in parts)
        
        trimmed_by_tokens = False
        while history:
            est = total_estimate(history)
            if est + max_length >= context_limit - safety_margin and len(history) > 1:
                # Remove oldest non-system message
                removed_idx = None
                for idx, h in enumerate(history):
                    if h.get("role") != "system":
                        removed_idx = idx
                        break
                if removed_idx is not None:
                    history.pop(removed_idx)
                    trimmed_by_tokens = True
                    continue
            break
        
        if trimmed_by_tokens:
            logger.info(
                "[prompt] History trimmed by token estimate to fit context (limit=%d, safety=%d). Remaining=%d",
                context_limit, safety_margin, len(history)
            )
        
        return history
    
    def get_system_prompt(self, language: str) -> str:
        """Get system prompt for a specific language."""
        return self.system_prompts.get(language, self.system_prompts["en"])
    
    def add_system_prompt(self, language: str, prompt: str) -> None:
        """Add or update system prompt for a language."""
        self.system_prompts[language] = prompt
        logger.info(f"Added/updated system prompt for language: {language}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        word_est = len(text.split())
        char_est = max(1, len(text) // 4)
        return max(word_est, char_est)


# Global instance for easy access
_shared_prompt_service = None


def get_shared_prompt_builder() -> PromptBuildingService:
    """
    Get the shared PromptBuildingService instance.
    
    Returns:
        PromptBuildingService: The shared singleton instance
    """
    global _shared_prompt_service
    if _shared_prompt_service is None:
        _shared_prompt_service = PromptBuildingService()
    return _shared_prompt_service