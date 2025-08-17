"""
Chat service for interactive model conversations.

This service handles interactive chat functionality including:
- Starting new chat sessions
- Managing conversation history
- Handling special chat commands
- Real-time parameter adjustment
- Streaming response support
"""
import logging
import os
import sys
import time
from typing import Dict, Any, Optional, Generator, List
from uuid import uuid4

from ..base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import clear_terminal_screen
from ...utils.language_detection import language_detector, suggest_response_language
from .content_filter_service import ContentFilterService

logger = logging.getLogger(__name__)


class ChatService(BaseService):
    """Service for interactive chat functionality."""
    
    def __init__(self, content_filter_strictness: str = "disabled"):  # TEMPORARILY DISABLE
        super().__init__()
        self.model_manager = ModelManager()
        self.content_filter = ContentFilterService(strictness_level=content_filter_strictness)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._default_model_loaded = False
        self._default_model_name = None

    def chat_fast(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 192,
        language: str = "auto",
        **generation_config,
    ) -> Dict[str, Any]:
        """Low-latency wrapper that forcibly disables thinking mode.

        This is used by voice (simple + streaming) pipelines where any hidden
        reasoning blocks (<think>...</think>) add latency and produce noisy
        TTS output if leaked.

        Guarantees:
        - thinking_mode False
        - enable_thinking flag stripped / overridden to False
        - conservative max_length (default 192)
        - stable low temperature defaults unless explicitly overridden
        """
        # Remove any external attempts to enable thinking
        generation_config.pop("enable_thinking", None)
        generation_config.pop("thinking_mode", None)
        # Provide latency-oriented defaults if caller didn't override
        generation_config.setdefault("temperature", 0.3)
        generation_config.setdefault("top_p", 0.8)
        return self.chat(
            message=message,
            conversation_history=conversation_history,
            max_length=max_length,
            language=language,
            thinking_mode=False,
            **generation_config,
        )
    
    def load_default_model_from_config(self) -> bool:
        """
        Load the fastest model (qwen3-unsloth-q4ks) for persistent 24/7 service.
        This ensures the model stays loaded on GPU for instant responses.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from pathlib import Path
            
            # Use the fastest model for real-time voice conversation
            model_name = "qwen3-unsloth-q4ks"
            
            # Get model configuration using absolute path
            app_config = AppConfig()
            config_dir = Path(__file__).parent.parent.parent / "config"
            models_file = config_dir / "model_registry.json"
            app_config.models_file = str(models_file)
            app_config.load_model_registry()
            
            model_config = app_config.model_registry.get_model(model_name)
            if not model_config:
                logger.error(f"Model configuration for '{model_name}' not found in registry")
                return False
            if not model_config:
                logger.error(f"Model configuration for '{model_name}' not found in registry")
                return False
            
            logger.info(f"Loading fastest model for 24/7 service: {model_name} ({model_config.model_id})")
            
            # Load the model and keep it persistent on GPU
            model = self._ensure_model_loaded(model_name, model_config)
            if model is not None:
                self._default_model_loaded = True
                self._default_model_name = model_name
                logger.info(f"✅ Fastest model loaded successfully and ready on GPU: {model_name}")
                return True
            else:
                logger.error(f"Failed to load fastest model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load default model from registry: {e}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a model for chat inference.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get model configuration using absolute path
            from pathlib import Path
            app_config = AppConfig()
            
            # Use absolute path to avoid working directory issues
            config_dir = Path(__file__).parent.parent.parent / "config"
            models_file = config_dir / "model_registry.json"
            app_config.models_file = str(models_file)
            
            app_config.load_model_registry()
            
            model_config = app_config.model_registry.get_model(model_name)
            if not model_config:
                logger.error(f"Model configuration for '{model_name}' not found.")
                return False
            
            # Load the model using the model manager
            model = self._ensure_model_loaded(model_name, model_config)
            return model is not None
            
        except Exception as e:
            logger.error(f"Failed to load chat model '{model_name}': {e}")
            return False
    
    def chat(self, message: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
             max_length: int = 256, language: str = "auto", thinking_mode: bool = False,
             **generation_config) -> Dict[str, Any]:
        """
        Single chat inference for API/service integration with automatic language detection.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            max_length: Maximum response length
            language: Response language ("auto" for automatic detection, or specific code)
            thinking_mode: Enable thinking mode
            **generation_config: Additional generation parameters
            
        Returns:
            Dict with success status and response
        """
        try:
            # Try to use the persistent default model first
            if self._default_model_loaded and self._default_model_name:
                model_name = self._default_model_name
                model = self.model_manager.get_loaded_model(model_name)
                logger.info(f"Using persistent default model: {model_name}")
            else:
                # Fallback to any loaded model
                loaded_models = list(self.model_manager._loaded_models.keys())
                if not loaded_models:
                    # Try to load default model if none are loaded
                    if self.load_default_model_from_config():
                        model_name = self._default_model_name
                        model = self.model_manager.get_loaded_model(model_name)
                    else:
                        return {"success": False, "error": "No model loaded and failed to load default", "response": None}
                else:
                    model_name = loaded_models[0]
                    model = self.model_manager.get_loaded_model(model_name)
            
            if not model:
                return {"success": False, "error": f"Model {model_name} not available", "response": None}
            
            # 🔍 Automatic Language Detection
            if language == "auto":
                detected_language = suggest_response_language(message, conversation_history)
                logger.info(f"🌍 Auto-detected language: {detected_language} for message: '{message[:50]}...'")
                response_language = detected_language
            else:
                response_language = language
                logger.info(f"🌍 Using specified language: {response_language}")
            
            # Build prompt with system message for language and behavior
            prompt_parts = []
            
            # Add system prompt based on detected/specified language
            if response_language == "ar":
                system_prompt = """أنت طبيب متخصص في الطب التجميلي والعلاجات غير الجراحية. يجب عليك الإجابة باللغة العربية فقط. 
قدم معلومات طبية دقيقة ومفيدة حول العلاجات التجميلية مثل البوتوكس والفيلر. 
اجعل إجاباتك واضحة ومختصرة ومناسبة للمرضى العرب.
مهم جداً: أجب باللغة العربية فقط ولا تستخدم أي لغة أخرى."""
                prompt_parts.append(f"System: {system_prompt}")
                
                # Add extra Arabic instruction as user message to reinforce
                prompt_parts.append("User: من فضلك أجب باللغة العربية فقط")
                prompt_parts.append("Assistant: سأجيب باللغة العربية.")
            elif response_language == "es":
                system_prompt = "Eres un médico especializado en medicina estética y tratamientos no quirúrgicos. Debes responder solo en español. Proporciona información médica precisa y útil sobre tratamientos estéticos como botox y rellenos. Haz que tus respuestas sean claras, concisas y apropiadas para pacientes hispanohablantes."
                prompt_parts.append(f"System: {system_prompt}")
            elif response_language == "fr":
                system_prompt = "Vous êtes un médecin spécialisé en médecine esthétique et en traitements non chirurgicaux. Vous devez répondre uniquement en français. Fournissez des informations médicales précises et utiles sur les traitements esthétiques comme le botox et les fillers. Rendez vos réponses claires, concises et appropriées pour les patients francophones."
                prompt_parts.append(f"System: {system_prompt}")
            elif response_language == "de":
                system_prompt = "Sie sind ein Arzt, der sich auf ästhetische Medizin und nicht-chirurgische Behandlungen spezialisiert hat. Sie müssen nur auf Deutsch antworten. Stellen Sie präzise und nützliche medizinische Informationen über ästhetische Behandlungen wie Botox und Filler zur Verfügung. Machen Sie Ihre Antworten klar, prägnant und angemessen für deutschsprachige Patienten."
                prompt_parts.append(f"System: {system_prompt}")
            else:  # Default to English
                system_prompt = "You are a doctor specialized in aesthetic medicine and non-surgical treatments. You must respond only in English. Provide accurate and useful medical information about aesthetic treatments like botox and fillers. Make your responses clear, concise, and appropriate for English-speaking patients."
                prompt_parts.append(f"System: {system_prompt}")
            
            # Add conversation history (will be pruned if exceeds context window)
            full_history: List[Dict[str, str]] = conversation_history[:] if conversation_history else []

            # ================= Context / history management ==============================
            # We apply three layers of control:
            # 1. Hard cap on number of turns (user+assistant pairs) via CHAT_MAX_HISTORY_TURNS
            # 2. Token estimation based trimming (conservative over-estimation) BEFORE generation
            # 3. Retry logic: if backend still throws context window error, aggressively shrink to last turn

            # 1. Hard cap on turns
            max_turns = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "6"))  # number of (user,assistant) pairs to retain
            if max_turns > 0 and full_history:
                # Count from end keeping most recent pairs; treat any lone user at end as a partial turn
                kept: List[Dict[str, str]] = []
                user_count = 0
                assistant_count = 0
                turns = 0
                for msg in reversed(full_history):
                    kept.append(msg)
                    if msg.get("role") == "assistant":
                        assistant_count += 1
                    elif msg.get("role") == "user":
                        user_count += 1
                    # A completed turn counted when we have both a user and assistant in kept slice
                    if user_count > 0 and assistant_count > 0:
                        turns += 1
                        user_count = 0
                        assistant_count = 0
                    if turns >= max_turns:
                        break
                # Reverse back to chronological order
                trimmed_history = list(reversed(kept))
                if len(trimmed_history) < len(full_history):
                    logger.debug(
                        "[chat] History turn cap applied (max_turns=%d) trimmed=%d->%d", 
                        max_turns, len(full_history), len(trimmed_history)
                    )
                    full_history = trimmed_history

            # 2. Token estimation based trimming
            context_limit = int(generation_config.get("context_window", os.getenv("CHAT_CONTEXT_WINDOW", "4096")))
            safety_margin = int(os.getenv("CHAT_CONTEXT_SAFETY_MARGIN", "160"))  # more conservative headroom

            # Heuristic token estimator: roughly 1 token ~ 4 characters (fallback if spaces minimal)
            def est_tokens_text(txt: str) -> int:
                if not txt:
                    return 0
                # Blend word-based and char-based estimates, take max for safety
                word_est = len(txt.split())
                char_est = max(1, len(txt) // 4)
                return max(word_est, char_est)

            def build_hist_parts(hist: List[Dict[str, str]]) -> List[str]:
                tmp: List[str] = []
                for m in hist:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "user":
                        tmp.append(f"User: {content}")
                    elif role == "assistant":
                        tmp.append(f"Assistant: {content}")
                return tmp

            def total_estimate(hist: List[Dict[str, str]]) -> int:
                parts = [prompt_parts[0]]  # system
                parts.extend(build_hist_parts(hist))
                parts.append(f"User: {message}")
                # Add assistant prefix placeholder
                parts.append("Assistant:")
                return sum(est_tokens_text(p) for p in parts)

            trimmed_by_tokens = False
            if full_history:
                while True:
                    est = total_estimate(full_history)
                    if est + max_length >= context_limit - safety_margin and len(full_history) > 1:
                        # Remove oldest non-system
                        removed_idx = None
                        for idx, h in enumerate(full_history):
                            if h.get("role") != "system":
                                removed_idx = idx
                                break
                        if removed_idx is not None:
                            full_history.pop(removed_idx)
                            trimmed_by_tokens = True
                            continue
                    break
            if trimmed_by_tokens:
                logger.info(
                    "[chat] History trimmed by token estimate to fit context (limit=%d, safety=%d). Remaining=%d", 
                    context_limit, safety_margin, len(full_history)
                )

            # Now append (possibly trimmed) history to prompt_parts
            if full_history:
                for msg in full_history:
                    if msg.get("role") == "user":
                        prompt_parts.append(f"User: {msg.get('content', '')}")
                    elif msg.get("role") == "assistant":
                        prompt_parts.append(f"Assistant: {msg.get('content', '')}")
            
            prompt_parts.append(f"User: {message}")
            prompt_parts.append("Assistant:")
            
            prompt = "\n".join(prompt_parts)

            # Final safety: if our conservative estimate still breaches context limit, shrink history aggressively
            try:
                context_limit = int(generation_config.get("context_window", os.getenv("CHAT_CONTEXT_WINDOW", "4096")))
            except Exception:
                context_limit = 4096
            safety_margin = int(os.getenv("CHAT_CONTEXT_SAFETY_MARGIN", "160"))
            # Reuse estimator
            def _quick_est(txt: str) -> int:
                return max(len(txt)//4, len(txt.split()))
            est_tokens = _quick_est(prompt)
            if est_tokens >= (context_limit - safety_margin):
                # Keep only system + last assistant (if any) + current user
                preserved_hist: List[Dict[str,str]] = []
                if full_history:
                    # take last assistant if exists
                    for m in reversed(full_history):
                        if m.get("role") == "assistant":
                            preserved_hist.insert(0, m)
                            break
                # rebuild prompt
                prompt_parts = [prompt_parts[0]]  # system
                for m in preserved_hist:
                    prompt_parts.append(f"Assistant: {m.get('content','')}")
                prompt_parts.append(f"User: {message}")
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)
                est_tokens = _quick_est(prompt)
                if est_tokens >= (context_limit - safety_margin):
                    # As last resort, shorten max_new_tokens and proceed; log warning
                    logger.warning(
                        "[chat] Prompt estimate %d still near/over limit (%d); forcing minimal generation window",
                        est_tokens, context_limit
                    )
                    generation_config["max_new_tokens"] = min(int(generation_config.get("max_new_tokens", max_length)), 96)
            
            # Generate response with optimized parameters for language consistency
            generation_params = {
                "max_new_tokens": max_length,
                "temperature": generation_config.get("temperature", 0.3),  # Lower temperature for more consistent language
                "top_p": generation_config.get("top_p", 0.8),
                "do_sample": generation_config.get("do_sample", True),
                "repetition_penalty": generation_config.get("repetition_penalty", 1.1),
                **{k: v for k, v in generation_config.items() if k not in ["max_new_tokens", "temperature", "top_p", "do_sample", "repetition_penalty"]}
            }
            
            logger.info(f"Generating response for language: {response_language}")
            logger.debug(f"Full prompt: {prompt}")
            
            def _apply_budget(prompt_text: str, params: Dict[str, Any]) -> Dict[str, Any]:
                """Adjust max_new_tokens to fit within context window using conservative estimate.

                We use the same heuristic estimator; if still risky we clamp tokens to a small window.
                """
                try:
                    ctx_limit = int(generation_config.get("context_window", os.getenv("CHAT_CONTEXT_WINDOW", "4096")))
                except Exception:
                    ctx_limit = 4096
                safety = int(os.getenv("CHAT_CONTEXT_SAFETY_MARGIN", "160"))
                # Estimate existing prompt tokens
                est_prompt_tokens = max(len(prompt_text)//4, len(prompt_text.split()))
                # Allowed budget for new tokens (keep a small guard band)
                avail = max(32, ctx_limit - safety - est_prompt_tokens)
                original = params.get("max_new_tokens", max_length)
                if avail < original:
                    logger.info(
                        "[chat] Reducing max_new_tokens %d -> %d (est_prompt=%d ctx=%d safety=%d)",
                        original, avail, est_prompt_tokens, ctx_limit, safety,
                    )
                    params = dict(params)
                    params["max_new_tokens"] = avail
                # Absolute upper bound guard
                if params.get("max_new_tokens", 0) > 512:
                    params["max_new_tokens"] = 512
                return params

            generation_params = _apply_budget(prompt, generation_params)

            def _minimal_rebuild() -> None:
                """Rebuild prompt to minimal form: system + (last assistant) + user."""
                nonlocal prompt, generation_params, full_history, prompt_parts
                minimal_hist: List[Dict[str, str]] = []
                for m in reversed(full_history):
                    if m.get("role") == "assistant":
                        minimal_hist.insert(0, m)
                        break
                full_history = minimal_hist
                base_system = prompt_parts[0]
                prompt_parts = [base_system]
                for msg in full_history:
                    if msg.get("role") == "assistant":
                        prompt_parts.append(f"Assistant: {msg.get('content','')}")
                prompt_parts.append(f"User: {message}")
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)
                generation_params = _apply_budget(prompt, generation_params)

            # Primary generation attempt with safety budgeting
            try:
                response = model.generate(prompt, **generation_params)
            except ValueError as ve:
                ve_text = str(ve).lower()
                if "exceed context window" in ve_text or "exceeds context window" in ve_text or "context window" in ve_text:
                    logger.warning("[chat] Context window exceeded on first attempt (err=%s); retrying minimal prompt", ve)
                    _minimal_rebuild()
                    try:
                        response = model.generate(prompt, **generation_params)
                    except Exception as ve2:  # second failure – return fallback response gracefully
                        logger.error("[chat] Retry after trimming failed: %s", ve2)
                        fallback = self._clean_response("", response_language)
                        return {"success": True, "response": fallback, "detected_language": response_language, "language_confidence": 0.2}
                else:
                    # Unrelated ValueError, propagate to outer handler
                    raise
            except Exception as gen_ex:
                # Any other exception (backend error). Provide graceful fallback without raising.
                logger.error("[chat] Generation backend exception: %s", gen_ex)
                fallback = self._clean_response("", response_language)
                return {"success": True, "response": fallback, "detected_language": response_language, "language_confidence": 0.2}

            # Defensive: sometimes backends return None/empty early
            if response is None:
                response = ""
            
            if response and response.strip():
                # Clean the response to remove any thinking content or unwanted text
                cleaned_response = self._clean_response(response.strip(), response_language)

                # ================= Strict Language Enforcement (Arabic) =================
                # If the caller explicitly requested Arabic (not auto) but model produced
                # predominantly non-Arabic characters, we attempt a second-pass translation
                # generation OR fallback to a canned Arabic greeting to preserve UX
                try:
                    enforce = os.getenv("CHAT_STRICT_LANGUAGE_ENFORCEMENT", "1") == "1"
                except Exception:
                    enforce = True
                if enforce and language == "ar":
                    arabic_chars = sum(1 for c in cleaned_response if '\u0600' <= c <= '\u06FF')
                    ratio_ar = arabic_chars / max(1, len(cleaned_response))
                    if ratio_ar < 0.25:  # mismatch threshold
                        logger.warning("[chat] Arabic requested but ratio_ar=%.3f < 0.25; attempting enforced translation", ratio_ar)
                        try:
                            # Build lightweight translation prompt (avoid full history to reduce latency)
                            translation_prompt = (
                                "System: أنت مترجم طبي محترف. ترجم النص التالي إلى العربية الفصحى الحديثة بدقة ووضوح وبدون أي تفسير إضافي.\n"
                                f"User: {cleaned_response}\nAssistant:"
                            )
                            # Constrain tokens to original budget
                            trans_params = {
                                "max_new_tokens": min(max_length, int(len(cleaned_response) * 1.2 // 4) + 48),
                                "temperature": 0.2,
                                "top_p": 0.8,
                                "do_sample": False,
                            }
                            translated = model.generate(translation_prompt, **trans_params) or ""
                            if translated.strip():
                                # Basic verification the result is Arabic heavy
                                arabic_chars2 = sum(1 for c in translated if '\u0600' <= c <= '\u06FF')
                                ratio_ar2 = arabic_chars2 / max(1, len(translated))
                                if ratio_ar2 >= ratio_ar:  # accept if improvement
                                    cleaned_response = self._clean_response(translated.strip(), "ar")
                                    response_language = "ar"
                                    logger.info("[chat] Enforced translation applied ratio_ar2=%.3f", ratio_ar2)
                        except Exception as te:  # pragma: no cover - best effort
                            logger.warning("[chat] translation enforcement failed: %s", te)
                        # Final guard: if still not Arabic enough, override with Arabic fallback greeting
                        if sum(1 for c in cleaned_response if '\u0600' <= c <= '\u06FF') / max(1, len(cleaned_response)) < 0.15:
                            cleaned_response = "أهلاً وسهلاً! كيف يمكنني مساعدتك في المجال التجميلي اليوم؟"
                            response_language = "ar"
                            logger.info("[chat] Applied Arabic fallback greeting after failed enforcement")

                logger.info(f"Generated response (first 100 chars): {cleaned_response[:100]}")
                return {
                    "success": True, 
                    "response": cleaned_response,
                    "detected_language": response_language,
                    "language_confidence": 1.0 if language != "auto" else 0.8  # Add confidence score
                }
            else:
                # Provide a language-specific fallback to avoid hard failure upstream
                fallback = self._clean_response("", response_language)
                return {"success": True, "response": fallback, "detected_language": response_language, "language_confidence": 0.3}
                
        except Exception as e:
            import traceback
            logger.error(f"Chat inference error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e), "response": None}
    
    def _clean_response(self, response: str, language: str = "ar") -> str:
        """
        Clean the model response to ensure proper language and remove unwanted content.
        
        Args:
            response: Raw model response
            language: Target language
            
        Returns:
            str: Cleaned response
        """
        # Remove any thinking tags or content
        import re
        
        # Remove thinking blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'</?think>', '', response, flags=re.IGNORECASE)
        response = re.sub(r'</?thinking>', '', response, flags=re.IGNORECASE)
        
        # Remove common thinking patterns in English
        thinking_patterns = [
            r"Let me think.*?(?=\n|$)",
            r"I need to think.*?(?=\n|$)",
            r"First, let me.*?(?=\n|$)",
            r"Okay, the user is asking.*?(?=\n|$)",
            r"Alright, the user.*?(?=\n|$)",
            r"The user.*?(?=\n|$)",
            r"From what I remember.*?(?=\n|$)",
            r".*?is asking about.*?(?=\n|$)",
            r"Since I.*?(?=\n|$)",
            r"I should.*?(?=\n|$)",
            r"Looking at.*?(?=\n|$)",
            r"Based on.*?(?=\n|$)",
            r"Given that.*?(?=\n|$)"
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove Arabic thinking patterns
        arabic_thinking_patterns = [
            r"دعني أفكر.*?(?=\n|$)",
            r"أحتاج إلى التفكير.*?(?=\n|$)",
            r"المستخدم يسأل.*?(?=\n|$)",
            r"بناءً على.*?(?=\n|$)"
        ]
        
        for pattern in arabic_thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.MULTILINE)
        
        # Clean up whitespace
        response = re.sub(r'\n\s*\n', '\n', response)
        response = response.strip()
        
        # Extract the actual response part (look for direct medical advice)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip meta-commentary and keep actual medical responses
            if any(skip_phrase in line.lower() for skip_phrase in [
                "the user", "i need to", "let me", "okay,", "alright,", "since i", "looking at",
                "based on", "given that", "from what", "i should", "المستخدم", "دعني", "أحتاج"
            ]):
                continue
            if line and not line.startswith(('User:', 'Assistant:', 'System:')):
                cleaned_lines.append(line)
        
        if cleaned_lines:
            response = '\n'.join(cleaned_lines)
        
        # If response is empty or too short, provide a default in the correct language
        if not response or len(response.strip()) < 10:
            logger.warning("Response was empty after cleaning, using fallback")
            if language == "ar":
                return "أهلاً وسهلاً! كيف يمكنني مساعدتك في المجال التجميلي اليوم؟"
            elif language == "es":
                return "¡Hola! ¿Cómo puedo ayudarte con tratamientos estéticos hoy?"
            elif language == "fr":
                return "Bonjour! Comment puis-je vous aider avec les traitements esthétiques aujourd'hui?"
            elif language == "de":
                return "Hallo! Wie kann ich Ihnen heute bei ästhetischen Behandlungen helfen?"
            else:  # English default
                return "Hello! How can I help you with aesthetic treatments today?"
        
        return response
    
    def start_chat(self, model_name: str, model_config: ModelConfig, 
                   generation_config: Dict[str, Any]) -> int:
        """
        Start interactive chat with a model.
        
        Args:
            model_name: Name of the model to use
            model_config: Model configuration
            generation_config: Generation parameters
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
            # Create a new session ID
            session_id = f"chat_{model_name}_{int(time.time())}"
            
            # Set up the interactive chat loop
            clear_terminal_screen()
            print(f"\n🤖 BeautyAI Chat - Model: {model_name} ({model_config.model_id})")
            print("=" * 60)
            print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
            print("Type 'clear' to start a new conversation")
            print("Type 'system <message>' to set a system message")
            print("Type '/help' to see additional commands")
            print("=" * 60)
            
            chat_history = []
            system_message = None
            
            # Store session information
            self.active_sessions[session_id] = {
                "model_name": model_name,
                "history": chat_history,
                "system_message": system_message,
                "start_time": time.time(),
                "config": generation_config
            }
            
            try:
                while True:
                    try:
                        # Get user input
                        user_input = input("\n👤 You: ")
                    except EOFError:
                        print("\n✅ Chat ended (EOF detected).")
                        break
                    except KeyboardInterrupt:
                        print("\n✅ Chat ended (interrupted).")
                        break
                    
                    # Handle special commands
                    if user_input.lower() in ["exit", "quit"]:
                        print("\n✅ Chat ended.")
                        break
                        
                    elif user_input.lower() == "clear":
                        chat_history = []
                        system_message = None
                        self.active_sessions[session_id]["history"] = chat_history
                        self.active_sessions[session_id]["system_message"] = system_message
                        clear_terminal_screen()
                        print("Conversation history cleared.")
                        continue
                        
                    elif user_input.strip() == "":
                        continue
                        
                    elif user_input.lower() == "/help":
                        self._show_chat_help()
                        continue
                        
                    elif user_input.lower() == "/params":
                        self._show_generation_params(generation_config)
                        continue
                        
                    elif user_input.lower().startswith("/temp "):
                        try:
                            new_temp = float(user_input[6:])
                            if 0.0 <= new_temp <= 2.0:
                                generation_config["temperature"] = new_temp
                                self.active_sessions[session_id]["config"] = generation_config
                                print(f"Temperature set to: {new_temp}")
                            else:
                                print("Temperature must be between 0.0 and 2.0")
                        except ValueError:
                            print("Invalid temperature value. Must be a number between 0.0 and 2.0")
                        continue
                        
                    elif user_input.lower().startswith("/tokens "):
                        try:
                            new_tokens = int(user_input[8:])
                            if new_tokens > 0:
                                generation_config["max_new_tokens"] = new_tokens
                                self.active_sessions[session_id]["config"] = generation_config
                                print(f"Max tokens set to: {new_tokens}")
                            else:
                                print("Max tokens must be a positive integer")
                        except ValueError:
                            print("Invalid token value. Must be a positive integer")
                        continue
                        
                    elif user_input.lower().startswith("system "):
                        system_message = user_input[7:]  # Remove "system " prefix
                        print(f"System message set: {system_message}")
                        self.active_sessions[session_id]["system_message"] = system_message
                        
                        # Update conversation with system message
                        if chat_history and chat_history[0]["role"] == "system":
                            chat_history[0] = {"role": "system", "content": system_message}
                        else:
                            chat_history.insert(0, {"role": "system", "content": system_message})
                        continue
                    
                    # Content filtering check
                    filter_result = self.content_filter.filter_content(user_input, language='ar')
                    if not filter_result.is_allowed:
                        print(f"\n🚫 {filter_result.suggested_response}")
                        if filter_result.filter_reason:
                            logger.info(f"Blocked user input due to: {filter_result.filter_reason}")
                        continue
                    
                    # Add user message to conversation
                    chat_history.append({"role": "user", "content": user_input})
                    
                    # Stream response if available
                    print("\n🤖 Model: ", end="")
                    sys.stdout.flush()
                    
                    response = ""
                    streaming = hasattr(model, "chat_stream")
                    
                    start_time = time.time()
                    
                    if streaming:
                        for token in model.chat_stream(chat_history, callback=lambda x: None, **generation_config):
                            # Ensure token is a string
                            token_str = str(token) if not isinstance(token, str) else token
                            print(token_str, end="")
                            sys.stdout.flush()
                            response += token_str
                    else:
                        response = model.chat(chat_history, **generation_config)
                        # Ensure response is a string
                        if not isinstance(response, str):
                            response = str(response)
                        print(response)
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # Add to chat history - ensure response is a string
                    if not isinstance(response, str):
                        response = str(response)
                    chat_history.append({"role": "assistant", "content": response})
                    
                    # Update session history
                    self.active_sessions[session_id]["history"] = chat_history
                    
                    # Print some stats
                    tokens_generated = len(response.split())
                    print(f"\n[Generated ~{tokens_generated} tokens in {generation_time:.2f}s, "
                          f"{tokens_generated/generation_time:.1f} tokens/sec]")
                    
            except KeyboardInterrupt:
                print("\n\n✅ Chat ended.")
            
            # Clean up session when done
            self._cleanup_session(session_id)
            return 0
            
        except Exception as e:
            import traceback
            logger.error(f"Full traceback for chat error: {traceback.format_exc()}")
            return self._handle_error(e, f"Failed to start chat with model {model_name}")
    
    def load_session_chat(self, model_name: str, model_config: ModelConfig, 
                         generation_config: Dict[str, Any], chat_history: list,
                         system_message: Optional[str] = None) -> int:
        """
        Continue a chat session from loaded history.
        
        Args:
            model_name: Name of the model to use
            model_config: Model configuration
            generation_config: Generation parameters
            chat_history: Previous conversation history
            system_message: Optional system message
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
            # Create a new session ID
            session_id = f"chat_{model_name}_{int(time.time())}"
            
            # Store session information
            self.active_sessions[session_id] = {
                "model_name": model_name,
                "history": chat_history,
                "system_message": system_message,
                "start_time": time.time(),
                "config": generation_config
            }
            
            # Set up the interactive chat loop
            clear_terminal_screen()
            print(f"\n🤖 BeautyAI Chat - Model: {model_name} ({model_config.model_id})")
            print(f"Loaded session with {len(chat_history)} messages")
            print("=" * 60)
            print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
            print("Type 'clear' to start a new conversation")
            print("Type 'system <message>' to set a system message")
            print("Type '/help' to see additional commands")
            print("=" * 60)
            
            # Display last few messages for context
            num_context_messages = min(4, len(chat_history))
            if num_context_messages > 0:
                print("\n=== Previous Messages ===")
                for msg in chat_history[-num_context_messages:]:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    print(f"\n{role_icon} {msg['role'].capitalize()}: {msg['content']}")
                print("\n" + "=" * 60)
            
            try:
                while True:
                    # Get user input
                    user_input = input("\n👤 You: ")
                    
                    # Handle special commands (same as start_chat)
                    if user_input.lower() in ["exit", "quit"]:
                        print("\n✅ Chat ended.")
                        break
                        
                    elif user_input.lower() == "clear":
                        chat_history = []
                        system_message = None
                        self.active_sessions[session_id]["history"] = chat_history
                        self.active_sessions[session_id]["system_message"] = system_message
                        clear_terminal_screen()
                        print("Conversation history cleared.")
                        continue
                        
                    elif user_input.strip() == "":
                        continue
                        
                    elif user_input.lower() == "/help":
                        self._show_chat_help()
                        continue
                        
                    elif user_input.lower() == "/params":
                        self._show_generation_params(generation_config)
                        continue
                        
                    elif user_input.lower().startswith("/temp "):
                        try:
                            new_temp = float(user_input[6:])
                            if 0.0 <= new_temp <= 2.0:
                                generation_config["temperature"] = new_temp
                                self.active_sessions[session_id]["config"] = generation_config
                                print(f"Temperature set to: {new_temp}")
                            else:
                                print("Temperature must be between 0.0 and 2.0")
                        except ValueError:
                            print("Invalid temperature value. Must be a number between 0.0 and 2.0")
                        continue
                        
                    elif user_input.lower().startswith("/tokens "):
                        try:
                            new_tokens = int(user_input[8:])
                            if new_tokens > 0:
                                generation_config["max_new_tokens"] = new_tokens
                                self.active_sessions[session_id]["config"] = generation_config
                                print(f"Max tokens set to: {new_tokens}")
                            else:
                                print("Max tokens must be a positive integer")
                        except ValueError:
                            print("Invalid token value. Must be a positive integer")
                        continue
                        
                    elif user_input.lower().startswith("system "):
                        system_message = user_input[7:]  # Remove "system " prefix
                        print(f"System message set: {system_message}")
                        self.active_sessions[session_id]["system_message"] = system_message
                        
                        # Update conversation with system message
                        if chat_history and chat_history[0]["role"] == "system":
                            chat_history[0] = {"role": "system", "content": system_message}
                        else:
                            chat_history.insert(0, {"role": "system", "content": system_message})
                        continue
                    
                    # Content filtering check
                    filter_result = self.content_filter.filter_content(user_input, language='ar')
                    if not filter_result.is_allowed:
                        print(f"\n🚫 {filter_result.suggested_response}")
                        if filter_result.filter_reason:
                            logger.info(f"Blocked user input due to: {filter_result.filter_reason}")
                        continue
                    
                    # Continue with normal message processing, identical to start_chat
                    chat_history.append({"role": "user", "content": user_input})
                    
                    print("\n🤖 Model: ", end="")
                    sys.stdout.flush()
                    
                    response = ""
                    streaming = hasattr(model, "chat_stream")
                    
                    start_time = time.time()
                    
                    if streaming:
                        for token in model.chat_stream(chat_history, callback=lambda x: None, **generation_config):
                            # Ensure token is a string
                            token_str = str(token) if not isinstance(token, str) else token
                            print(token_str, end="")
                            sys.stdout.flush()
                            response += token_str
                    else:
                        response = model.chat(chat_history, **generation_config)
                        # Ensure response is a string
                        if not isinstance(response, str):
                            response = str(response)
                        print(response)
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # Add to chat history - ensure response is a string
                    if not isinstance(response, str):
                        response = str(response)
                    chat_history.append({"role": "assistant", "content": response})
                    
                    # Update session history
                    self.active_sessions[session_id]["history"] = chat_history
                    
                    # Print some stats
                    tokens_generated = len(response.split())
                    print(f"\n[Generated ~{tokens_generated} tokens in {generation_time:.2f}s, "
                          f"{tokens_generated/generation_time:.1f} tokens/sec]")
                    
            except KeyboardInterrupt:
                print("\n\n✅ Chat ended.")
            
            # Clean up session when done
            self._cleanup_session(session_id)
            return 0
            
        except Exception as e:
            return self._handle_error(e, f"Failed to continue chat session")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session.
        
        Args:
            session_id: The ID of the session to retrieve.
            
        Returns:
            Dict containing session information or None if not found.
        """
        return self.active_sessions.get(session_id)
    
    def list_active_sessions(self) -> list:
        """
        List all active sessions.
        
        Returns:
            List of dictionaries with session information.
        """
        return [
            {
                "session_id": sid,
                "model": info["model_name"],
                "duration": time.time() - info["start_time"],
                "messages": len(info["history"])
            }
            for sid, info in self.active_sessions.items()
        ]
    
    def _ensure_model_loaded(self, model_name: str, model_config: ModelConfig):
        """
        Ensure the model is loaded and return it.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            The loaded model
        """
        if not self.model_manager.is_model_loaded(model_name):
            print(f"Loading model '{model_name}'...")
            self.model_manager.load_model(model_config)
            print(f"Model loaded successfully.")
        else:
            print(f"Using already loaded model '{model_name}'.")
            
        return self.model_manager.get_loaded_model(model_name)
    
    def _show_chat_help(self):
        """Display help message for chat commands."""
        help_text = """
Available Commands:
------------------
exit, quit      - End the chat session
clear           - Clear conversation history
system <msg>    - Set a system message
/help           - Show this help message
/params         - Show current generation parameters
/temp <value>   - Set temperature (0.0-2.0)
/tokens <value> - Set max tokens for response
"""
        print(help_text)
    
    def _show_generation_params(self, config: Dict[str, Any]):
        """Display current generation parameters."""
        print("\nCurrent Generation Parameters:")
        print("-" * 30)
        for param, value in config.items():
            print(f"{param}: {value}")
        print()
    
    def _cleanup_session(self, session_id: str) -> None:
        """
        Clean up a session.
        
        Args:
            session_id: ID of the session to clean up
        """
        if session_id in self.active_sessions:
            # Save session history if needed
            # (future feature: could save to disk or database)
            del self.active_sessions[session_id]
    
    def set_content_filter_strictness(self, strictness: str) -> None:
        """
        Set content filter strictness level.
        
        Args:
            strictness: One of "strict", "balanced", "relaxed", "disabled"
        """
        self.content_filter.set_strictness_level(strictness)
        logger.info(f"Content filter strictness set to: {strictness}")
