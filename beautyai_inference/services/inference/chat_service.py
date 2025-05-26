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
import sys
import time
from typing import Dict, Any, Optional, Generator
from uuid import uuid4

from ..base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import clear_terminal_screen

logger = logging.getLogger(__name__)


class ChatService(BaseService):
    """Service for interactive chat functionality."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
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
                    # Get user input
                    user_input = input("\n👤 You: ")
                    
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
                            print(token, end="")
                            sys.stdout.flush()
                            response += token
                    else:
                        response = model.chat(chat_history, **generation_config)
                        print(response)
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # Add to chat history
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
                    
                    # Continue with normal message processing, identical to start_chat
                    chat_history.append({"role": "user", "content": user_input})
                    
                    print("\n🤖 Model: ", end="")
                    sys.stdout.flush()
                    
                    response = ""
                    streaming = hasattr(model, "chat_stream")
                    
                    start_time = time.time()
                    
                    if streaming:
                        for token in model.chat_stream(chat_history, callback=lambda x: None, **generation_config):
                            print(token, end="")
                            sys.stdout.flush()
                            response += token
                    else:
                        response = model.chat(chat_history, **generation_config)
                        print(response)
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # Add to chat history
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
