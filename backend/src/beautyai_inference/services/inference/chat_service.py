"""
Refactored Chat Service

This service provides a streamlined chat interface using shared components
for model management, content filtering, prompt building, and session management.
"""
import logging
import time
import os
from copy import deepcopy
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Core imports
from ...config.config_manager import ModelConfig, AppConfig
from ...utils.language_detection import detect_language
from ..shared import (
    get_shared_model_manager,
    get_shared_content_filter,
    get_shared_prompt_builder,
    get_shared_session_manager
)

logger = logging.getLogger(__name__)


class ChatService:
    """
    Streamlined chat service using shared components.
    
    This service coordinates between the model manager, content filter,
    prompt builder, and session manager to provide chat functionality.
    """
    
    def __init__(self):
        self.model_manager = get_shared_model_manager()
        self.content_filter = get_shared_content_filter()
        self.prompt_builder = get_shared_prompt_builder()
        self.session_manager = get_shared_session_manager()
        self._model_registry_path = Path(__file__).parent.parent.parent / "config" / "model_registry.json"
        self.default_model_name: Optional[str] = None
        self.default_model_config: Optional[ModelConfig] = None
        self.default_generation_config: Dict[str, Any] = {}
        self._initialize_default_model_metadata()
        
        # Default fallback messages for different languages
        self.fallback_messages = {
            'ar': "مرحباً! أنا طبيب متخصص في الطب التجميلي. كيف يمكنني مساعدتك اليوم؟",
            'en': "Hello! I'm a doctor specialized in aesthetic medicine. How can I help you today?",
            'es': "¡Hola! Soy médico especializado en medicina estética. ¿Cómo puedo ayudarte hoy?",
            'fr': "Bonjour! Je suis médecin spécialisé en médecine esthétique. Comment puis-je vous aider aujourd'hui?",
            'de': "Hallo! Ich bin Arzt für ästhetische Medizin. Wie kann ich Ihnen heute helfen?"
        }
    
    def _initialize_default_model_metadata(self) -> None:
        """Resolve and cache the default chat model metadata from the registry."""
        try:
            app_config = AppConfig()
            app_config.models_file = str(self._model_registry_path)
            app_config.load_model_registry()

            registry = app_config.model_registry
            preferred_model_name = os.getenv("BEAUTYAI_DEFAULT_CHAT_MODEL", "").strip()

            if not preferred_model_name:
                preferred_model_name = registry.default_model or "qwen3-unsloth-q4ks"

            model_config = registry.get_model(preferred_model_name)

            if model_config is None and preferred_model_name != "qwen3-unsloth-q4ks":
                model_config = registry.get_model("qwen3-unsloth-q4ks")
                if model_config:
                    logger.info("Falling back to qwen3-unsloth-q4ks for default chat model")
                    preferred_model_name = "qwen3-unsloth-q4ks"

            if model_config is None:
                available_models = registry.list_models()
                if available_models:
                    fallback_name = available_models[0]
                    model_config = registry.get_model(fallback_name)
                    preferred_model_name = fallback_name
                    logger.warning(
                        "Default chat model not found; using first available model: %s",
                        fallback_name
                    )

            if model_config is None:
                logger.error("No chat models available in registry; using minimal defaults")
                self.default_model_name = None
                self.default_model_config = None
                self.default_generation_config = self._build_generation_config(None)
                return

            self.default_model_name = preferred_model_name or getattr(model_config, "name", None)
            self.default_model_config = model_config
            self.default_generation_config = self._build_generation_config(model_config)

            logger.debug(
                "Cached default chat model metadata: name=%s, model_id=%s",
                self.default_model_name,
                self.default_model_config.model_id
            )

        except Exception as exc:
            logger.error("Failed to initialize default chat model metadata: %s", exc)
            self.default_model_name = None
            self.default_model_config = None
            self.default_generation_config = self._build_generation_config(None)

    def _build_generation_config(self, model_config: Optional[ModelConfig]) -> Dict[str, Any]:
        """Construct default generation parameters tuned for fast chat responses."""
        base_config: Dict[str, Any] = {
            "max_new_tokens": 256,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
            "thinking_mode": False
        }

        if model_config:
            if getattr(model_config, "max_new_tokens", None):
                base_config["max_new_tokens"] = min(model_config.max_new_tokens, 512)

            if model_config.custom_generation_params:
                for key, value in model_config.custom_generation_params.items():
                    if value is not None:
                        base_config[key] = value

        # Always enforce voice requirements
        base_config["thinking_mode"] = False
        return base_config

    def get_default_model_info(self) -> Tuple[Optional[str], Optional[ModelConfig]]:
        """Return the cached default chat model name and configuration."""
        return self.default_model_name, self.default_model_config

    def get_default_generation_config(self) -> Dict[str, Any]:
        """Provide a copy of the cached default generation configuration."""
        if not self.default_generation_config:
            self.default_generation_config = self._build_generation_config(self.default_model_config)
        return deepcopy(self.default_generation_config)

    def chat(
        self,
        message: str,
        model_name: str,
        model_config: ModelConfig,
        generation_config: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        response_language: Optional[str] = None,
        session_id: Optional[str] = None,
        disable_content_filter: bool = False
    ) -> Tuple[str, str, List[Dict[str, str]], str]:
        """
        Process a chat message and generate a response.
        
        Args:
            message: User's message
            model_name: Model identifier
            model_config: Model configuration
            generation_config: Generation parameters
            conversation_history: Previous conversation history
            response_language: Preferred response language
            session_id: Optional session ID for session management
            disable_content_filter: Whether to bypass content filtering
            
        Returns:
            Tuple of (response, detected_language, updated_history, session_id)
        """
        try:
            # Detect language BEFORE attempting model load so fallback matches user language
            if response_language is None or response_language == "auto":
                try:
                    detected_lang, _ = detect_language(message)
                    response_language = detected_lang
                except Exception:
                    # Safe fallback to English if detection fails
                    response_language = "en"
                logger.info(f"🌍 Detected language (pre-load): {response_language}")
            else:
                logger.info(f"🌍 Using provided language: {response_language}")

            # Copy generation config and strip non-serializable/persistent hints
            if generation_config is None:
                generation_config = {}
            persistent_model = generation_config.get('_persistent_model_instance')
            use_persistent = generation_config.get('_use_persistent_model', False)
            generation_config = deepcopy({
                k: v for k, v in generation_config.items()
                if k not in ('_persistent_model_instance', '_use_persistent_model')
            })

            # Ensure model is loaded (after language detection so fallback uses detected language)
            if use_persistent and persistent_model is not None:
                model = persistent_model
            else:
                model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                logger.error("Failed to load model; returning language-specific fallback response")
                return self._get_fallback_response(response_language), response_language or "ar", [], session_id or ""
            
            # Content filtering (if not disabled)
            if not disable_content_filter:
                filter_result = self.content_filter.filter_content(message, response_language)
                if not filter_result.is_allowed:
                    logger.warning(f"Content filtered: {filter_result.filter_reason}")
                    return (
                        filter_result.suggested_response or self._get_fallback_response(response_language),
                        response_language,
                        conversation_history or [],
                        session_id or ""
                    )
            else:
                logger.info("Content filtering disabled - bypassing filter check")
            
            # Create or update session
            if session_id:
                session = self.session_manager.get_session(session_id)
                if session:
                    conversation_history = session.history
                else:
                    # Session not found, create new one
                    session_id = self.session_manager.create_session(
                        model_name=model_name,
                        language=response_language,
                        config=generation_config
                    )
            else:
                session_id = self.session_manager.create_session(
                    model_name=model_name,
                    language=response_language,
                    config=generation_config
                )
            
            # Apply model-specific system prompt if available
            # Check for safeguard bypass (dev/testing only)
            disable_safeguards = generation_config.get('disable_system_prompt_safeguards', False)
            env_disable = os.getenv('DISABLE_SYSTEM_PROMPT_SAFEGUARDS', '').lower() in ('1', 'true', 'yes')
            
            if disable_safeguards or env_disable:
                logger.warning("🔓 DEVELOPER MODE: System prompt safeguards DISABLED (testing only)")
                # Reset to default prompts (no Kesay Clinics restrictions)
                self.prompt_builder.reset_to_default_prompts()
            elif hasattr(model_config, 'system_prompt') and model_config.system_prompt:
                self.prompt_builder.apply_model_system_prompt(
                    system_prompt=model_config.system_prompt,
                    language=response_language
                )
                logger.info(f"✅ Applied model-specific system prompt for language: {response_language}")
            
            # Build prompt
            prompt, trimmed_history = self.prompt_builder.build_prompt(
                message=message,
                language=response_language,
                conversation_history=conversation_history,
                generation_config=generation_config
            )
            
            # Log debug information
            self._log_debug_info(message, response_language, prompt, generation_config)
            
            # Generate response
            start_time = time.time()
            
            try:
                response = model.generate(prompt, **generation_config)
                generation_time = time.time() - start_time
                
                # Clean the response
                response = self._clean_response(response)
                
                # Validate response is not empty after cleaning
                if not response or response.strip() == "":
                    logger.warning("Empty response after cleaning, using fallback")
                    response = self._get_fallback_response(response_language)
                
                # Update conversation history
                updated_history = trimmed_history[:]
                updated_history.append({"role": "user", "content": message})
                updated_history.append({"role": "assistant", "content": response})
                
                # Update session
                self.session_manager.update_session_history(session_id, updated_history)
                
                # Log performance metrics
                tokens_generated = len(response.split())
                logger.info(f"Generated ~{tokens_generated} tokens in {generation_time:.2f}s, "
                           f"{tokens_generated/generation_time:.1f} tokens/sec")
                
                return response, response_language, updated_history, session_id
                
            except Exception as generation_error:
                logger.error(f"Generation error: {generation_error}")
                # Return fallback response
                response = self._get_fallback_response(response_language)
                updated_history = (conversation_history or [])[:]
                updated_history.append({"role": "user", "content": message})
                updated_history.append({"role": "assistant", "content": response})
                
                return response, response_language, updated_history, session_id or ""
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return self._get_fallback_response(response_language), "ar", [], ""
    
    def start_interactive_chat(
        self,
        model_name: str,
        model_config: ModelConfig,
        generation_config: Dict[str, Any]
    ) -> int:
        """
        Start an interactive chat session.
        
        Args:
            model_name: Model identifier
            model_config: Model configuration
            generation_config: Generation parameters
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Ensure model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
            
            # Create session
            session_id = self.session_manager.create_session(
                model_name=model_name,
                language="ar",  # Default language
                config=generation_config
            )
            
            # Interactive chat loop
            self._run_interactive_loop(session_id, model_name, model_config, generation_config)
            
            # Cleanup session
            self.session_manager.delete_session(session_id)
            return 0
            
        except Exception as e:
            logger.error(f"Interactive chat error: {e}")
            return 1
    
    def load_default_model_from_config(self) -> bool:
        """
        Load the fastest model for persistent service.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.default_model_config is None:
                self._initialize_default_model_metadata()

            if self.default_model_config is None:
                logger.error("Model configuration for default chat model not found in registry")
                return False

            resolved_name = self.default_model_name or getattr(self.default_model_config, "name", "default")
            logger.info(
                "Loading fastest model for service: %s (%s)",
                resolved_name,
                self.default_model_config.model_id
            )

            model = self.model_manager.load_model(self.default_model_config)
            if model:
                logger.info("✅ Successfully loaded default model: %s", resolved_name)
                self._ensure_default_metadata_populated(resolved_name, self.default_model_config)
                return True

            logger.error("❌ Failed to load default model: %s", resolved_name)
            return False

        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            return False

    def _ensure_default_metadata_populated(
        self,
        model_name: str,
        model_config: Optional[ModelConfig]
    ) -> None:
        """Backfill cached default metadata if it was missing previously."""
        if self.default_model_name is None and model_name:
            self.default_model_name = model_name

        if self.default_model_config is None and model_config is not None:
            self.default_model_config = model_config

        if not self.default_generation_config and model_config is not None:
            self.default_generation_config = self._build_generation_config(model_config)
    
    def _ensure_model_loaded(self, model_name: str, model_config: ModelConfig):
        """Ensure the specified model is loaded."""
        try:
            # Correct usage: ModelManager.load_model expects only ModelConfig
            return self.model_manager.load_model(model_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def _get_fallback_response(self, language: Optional[str] = None) -> str:
        """Get fallback response for the specified language."""
        # Prefer detected / requested language; fall back to English, then Arabic
        if language and language in self.fallback_messages:
            return self.fallback_messages[language]
        if 'en' in self.fallback_messages:
            return self.fallback_messages['en']
        return self.fallback_messages.get('ar', 'Hello! How can I help you today?')
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the model response to remove unwanted content.
        
        Args:
            response: Raw model response
            
        Returns:
            str: Cleaned response
        """
        import re
        
        # Remove thinking blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up whitespace
        response = re.sub(r'\n\s*\n', '\n', response)
        response = response.strip()
        
        return response
    
    def _log_debug_info(self, message: str, language: str, prompt: str, config: Dict[str, Any]):
        """Log debug information about the chat request."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"💬 Message: {message[:100]}...")
            logger.debug(f"🌍 Language: {language}")
            logger.debug(f"📝 Prompt length: {len(prompt)} chars")
            logger.debug(f"⚙️ Config: {config}")
    
    def _run_interactive_loop(
        self,
        session_id: str,
        model_name: str,
        model_config: ModelConfig,
        generation_config: Dict[str, Any]
    ):
        """Run the interactive chat loop."""
        # Clear screen and show header
        self._clear_terminal_screen()
        print(f"\n🤖 BeautyAI Chat - Model: {model_name} ({model_config.model_id})")
        print("=" * 60)
        print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
        print("Type 'clear' to start a new conversation")
        print("Type 'system <message>' to set a system message")
        print("Type '/help' to see additional commands")
        print("=" * 60)
        
        try:
            while True:
                try:
                    user_input = input("\n👤 You: ")
                except (EOFError, KeyboardInterrupt):
                    print("\n✅ Chat ended.")
                    break
                
                # Handle special commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\n✅ Chat ended.")
                    break
                elif user_input.lower() == "clear":
                    self.session_manager.clear_session_history(session_id)
                    self._clear_terminal_screen()
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
                    self._handle_temperature_command(user_input, generation_config, session_id)
                    continue
                elif user_input.lower().startswith("/tokens "):
                    self._handle_tokens_command(user_input, generation_config, session_id)
                    continue
                elif user_input.lower().startswith("system "):
                    system_message = user_input[7:]
                    self.session_manager.update_session_system_message(session_id, system_message)
                    print(f"System message set: {system_message}")
                    continue
                
                # Process chat message
                session = self.session_manager.get_session(session_id)
                conversation_history = session.history if session else []
                
                response, language, updated_history, _ = self.chat(
                    message=user_input,
                    model_name=model_name,
                    model_config=model_config,
                    generation_config=generation_config,
                    conversation_history=conversation_history,
                    session_id=session_id
                )
                
                print(f"\n🤖 Assistant: {response}")
                
        except KeyboardInterrupt:
            print("\n\n✅ Chat ended.")
    
    def _clear_terminal_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_chat_help(self):
        """Show chat help information."""
        print("\n📚 Chat Commands:")
        print("  exit, quit     - End the chat")
        print("  clear          - Clear conversation history")
        print("  system <msg>   - Set system message")
        print("  /help          - Show this help")
        print("  /params        - Show current generation parameters")
        print("  /temp <value>  - Set temperature (0.0-2.0)")
        print("  /tokens <num>  - Set max tokens")
    
    def _show_generation_params(self, config: Dict[str, Any]):
        """Show current generation parameters."""
        print("\n⚙️ Current Generation Parameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    def _handle_temperature_command(self, command: str, config: Dict[str, Any], session_id: str):
        """Handle temperature setting command."""
        try:
            new_temp = float(command[6:])
            if 0.0 <= new_temp <= 2.0:
                config["temperature"] = new_temp
                self.session_manager.update_session_config(session_id, config)
                print(f"Temperature set to: {new_temp}")
            else:
                print("Temperature must be between 0.0 and 2.0")
        except ValueError:
            print("Invalid temperature value. Must be a number between 0.0 and 2.0")
    
    def _handle_tokens_command(self, command: str, config: Dict[str, Any], session_id: str):
        """Handle max tokens setting command."""
        try:
            new_tokens = int(command[8:])
            if new_tokens > 0:
                config["max_new_tokens"] = new_tokens
                self.session_manager.update_session_config(session_id, config)
                print(f"Max tokens set to: {new_tokens}")
            else:
                print("Max tokens must be a positive integer")
        except ValueError:
            print("Invalid token value. Must be a positive integer")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session = self.session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "model_name": session.model_name,
                "language": session.language,
                "message_count": len(session.history),
                "duration": session.get_duration(),
                "has_system_message": session.system_message is not None
            }
        return None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active chat sessions."""
        return self.session_manager.list_sessions()
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a specific session."""
        return self.session_manager.delete_session(session_id)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "session_stats": self.session_manager.get_session_stats(),
            "filter_stats": self.content_filter.get_filter_stats(),
            "model_stats": {
                "loaded_models": len(self.model_manager.list_loaded_models()),
                "loaded_model_names": self.model_manager.list_loaded_models()
            }
        }