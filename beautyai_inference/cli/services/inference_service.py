"""
Inference service for unified CLI.
"""
import logging
import sys
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Callable, Generator, Iterator
from pathlib import Path

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig
from ...core.model_factory import ModelFactory
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_info, get_gpu_memory_stats, clear_terminal_screen

logger = logging.getLogger(__name__)


class InferenceService(BaseService):
    """Service for inference operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        self.model_manager = ModelManager()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_chat(self, args):
        """
        Start interactive chat with a model.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Configure generation parameters
        generation_config = self._get_generation_config(args, model_config)
        
        # Ensure the model is loaded
        try:
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
        except Exception as e:
            return self._handle_error(e, f"Failed to load model {model_name}")
        
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
                streaming = hasattr(model, "generate_streaming")
                
                start_time = time.time()
                
                if streaming:
                    for token in model.generate_streaming(user_input, generation_config, chat_history):
                        print(token, end="")
                        sys.stdout.flush()
                        response += token
                else:
                    response = model.generate(user_input, generation_config, chat_history)
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
    
    def run_test(self, args):
        """
        Run a simple test with the model.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Get prompt and other arguments
        prompt = getattr(args, "prompt", "Hello, how are you today?")
        max_tokens = getattr(args, "max_tokens", model_config.max_new_tokens)
        temperature = getattr(args, "temperature", model_config.temperature)
        
        # Configure generation parameters
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": model_config.top_p,
            "do_sample": model_config.do_sample,
        }
        
        print(f"\n=== Testing {model_name} ===")
        print(f"Model ID: {model_config.model_id}")
        print(f"Engine: {model_config.engine_type}")
        print(f"Quantization: {model_config.quantization or 'none'}")
        print(f"Generation parameters: {generation_config}")
        
        # Ensure the model is loaded
        try:
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
            
        except Exception as e:
            return self._handle_error(e, f"Failed to load model {model_name}")
        
        # Run the test
        print("\n=== Test Prompt ===")
        print(prompt)
        print("\n=== Response ===")
        
        start_time = time.time()
        
        # Create a minimal context for stateless models
        chat_history = []
        response = model.generate(prompt, generation_config, chat_history)
        
        end_time = time.time()
        
        print(response)
        
        generation_time = end_time - start_time
        tokens_generated = len(response.split())
        
        print(f"\n=== Performance ===")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Tokens generated: ~{tokens_generated}")
        print(f"Tokens per second: ~{tokens_generated/generation_time:.2f}")
        
        return 0
    
    def run_benchmark(self, args):
        """
        Run a benchmark on the model.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Get benchmark parameters
        num_runs = getattr(args, "num_runs", 3)
        input_lengths_str = getattr(args, "input_lengths", "10,100,1000")
        output_length = getattr(args, "output_length", 100)
        output_file = getattr(args, "output_file", None)
        
        # Parse input lengths
        try:
            input_lengths = [int(x) for x in input_lengths_str.split(',')]
        except ValueError:
            return self._handle_error(ValueError("Invalid input lengths format"), 
                                      "Input lengths must be comma-separated integers")
        
        print(f"\n=== Benchmarking {model_name} ===")
        print(f"Model ID: {model_config.model_id}")
        print(f"Engine: {model_config.engine_type}")
        print(f"Quantization: {model_config.quantization or 'none'}")
        print(f"Number of runs: {num_runs}")
        print(f"Input lengths: {input_lengths}")
        print(f"Output length: {output_length}")
        
        # Ensure the model is loaded
        try:
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
        except Exception as e:
            return self._handle_error(e, f"Failed to load model {model_name}")
        
        # Configure generation parameters (optimized for benchmark)
        generation_config = {
            "max_new_tokens": output_length,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
        
        # Run the benchmark for each input length
        all_results = {}
        
        for input_length in input_lengths:
            print(f"\n--- Testing input length: {input_length} tokens ---")
            
            # Generate benchmark prompt
            prompt = "Hello, " * (input_length // 2)  # Simple repeating pattern to reach desired length
            
            # Run the benchmark multiple times
            results = []
            
            for i in range(num_runs):
                print(f"Run {i+1}/{num_runs}...")
                
                # Clear GPU memory stats before test
                start_memory = get_gpu_memory_stats()
                
                # Run generation
                start_time = time.time()
                _ = model.generate(prompt, generation_config)
                end_time = time.time()
                
                # Calculate results
                generation_time = end_time - start_time
                end_memory = get_gpu_memory_stats()
                
                memory_used_before = start_memory[0]["memory_used_mb"] if start_memory else 0
                memory_used_after = end_memory[0]["memory_used_mb"] if end_memory else 0
                memory_delta = memory_used_after - memory_used_before
                
                # Append results
                results.append({
                    "run_id": i+1,
                    "time": generation_time,
                    "tokens_per_sec": output_length / generation_time,
                    "memory_before": memory_used_before,
                    "memory_after": memory_used_after,
                    "memory_delta": memory_delta
                })
                
                print(f"Time: {generation_time:.2f}s, Tokens/s: {output_length/generation_time:.2f}")
            
            # Calculate average metrics for this input length
            avg_time = sum(result["time"] for result in results) / len(results)
            avg_tokens_per_sec = sum(result["tokens_per_sec"] for result in results) / len(results)
            avg_memory_delta = sum(result["memory_delta"] for result in results) / len(results)
            
            print(f"\nAverage for {input_length} tokens input:")
            print(f"  Generation time: {avg_time:.2f}s")
            print(f"  Tokens per second: {avg_tokens_per_sec:.2f}")
            print(f"  Memory increase: {avg_memory_delta:.2f} MB")
            
            all_results[input_length] = {
                "runs": results,
                "summary": {
                    "avg_time": avg_time,
                    "avg_tokens_per_sec": avg_tokens_per_sec,
                    "avg_memory_delta": avg_memory_delta
                }
            }
        
        # Save results to file if requested
        if output_file:
            try:
                results_obj = {
                    "model": model_name,
                    "model_id": model_config.model_id,
                    "engine": model_config.engine_type,
                    "quantization": model_config.quantization,
                    "output_length": output_length,
                    "timestamp": int(time.time()),
                    "results": all_results
                }
                
                with open(output_file, 'w') as f:
                    json.dump(results_obj, f, indent=2)
                    
                print(f"\nResults saved to {output_file}")
                
            except Exception as e:
                return self._handle_error(e, f"Failed to save benchmark results to {output_file}")
        
        print("\n=== Benchmark Complete ===")
        return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session.
        
        Args:
            session_id: The ID of the session to retrieve.
            
        Returns:
            Dict containing session information or None if not found.
        """
        return self.active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
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
    
    def save_session(self, args):
        """
        Save the current session to a file.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        session_id = getattr(args, "session_id", None)
        output_file = getattr(args, "output_file", None)
        
        if not session_id:
            sessions = self.list_active_sessions()
            if not sessions:
                print("No active sessions found.")
                return 1
            
            print("Active sessions:")
            for i, session in enumerate(sessions):
                print(f"{i+1}. {session['session_id']} - {session['model']} ({session['messages']} messages)")
            
            try:
                choice = int(input("Enter session number to save: "))
                if choice < 1 or choice > len(sessions):
                    return self._handle_error(ValueError("Invalid session choice"), "Please choose a valid session number")
                session_id = sessions[choice-1]["session_id"]
            except (ValueError, IndexError):
                return self._handle_error(ValueError("Invalid input"), "Please enter a valid number")
        
        session = self.get_session_info(session_id)
        if not session:
            return self._handle_error(ValueError(f"Session {session_id} not found"), "Session not found")
        
        if not output_file:
            output_file = f"beautyai_session_{session_id}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    "session_id": session_id,
                    "model": session["model_name"],
                    "timestamp": int(time.time()),
                    "start_time": session["start_time"],
                    "system_message": session["system_message"],
                    "history": session["history"],
                    "config": session.get("config", {})
                }, f, indent=2)
            
            print(f"Session saved to {output_file}")
            return 0
        except Exception as e:
            return self._handle_error(e, f"Failed to save session to {output_file}")
    
    def load_session(self, args):
        """
        Load a session from a file and continue the conversation.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        input_file = getattr(args, "input_file", None)
        if not input_file:
            return self._handle_error(ValueError("No session file specified"), "Please specify a session file to load")
        
        try:
            with open(input_file, 'r') as f:
                session_data = json.load(f)
        except Exception as e:
            return self._handle_error(e, f"Failed to load session from {input_file}")
        
        # Modify args to use the model from the session
        args.model_name = session_data.get("model")
        
        # Start a new chat with the loaded session data
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Configure generation parameters from session data
        generation_config = session_data.get("config", {})
        if not generation_config:
            generation_config = self._get_generation_config(args, model_config)
        
        # Ensure the model is loaded
        try:
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
        except Exception as e:
            return self._handle_error(e, f"Failed to load model {model_name}")
        
        # Create a new session ID
        session_id = f"chat_{model_name}_{int(time.time())}"
        
        # Load chat history from session file
        chat_history = session_data.get("history", [])
        system_message = session_data.get("system_message")
        
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
                streaming = hasattr(model, "generate_streaming")
                
                start_time = time.time()
                
                if streaming:
                    for token in model.generate_streaming(user_input, generation_config, chat_history):
                        print(token, end="")
                        sys.stdout.flush()
                        response += token
                else:
                    response = model.generate(user_input, generation_config, chat_history)
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
            self.model_manager.load_model(model_name, model_config)
            print(f"Model loaded successfully.")
        else:
            print(f"Using already loaded model '{model_name}'.")
            
        return self.model_manager.get_model(model_name)
    
    def _get_generation_config(self, args, model_config: ModelConfig) -> Dict[str, Any]:
        """
        Get generation configuration from arguments and model config.
        
        Args:
            args: Command-line arguments
            model_config: Model configuration
            
        Returns:
            Dictionary with generation parameters
        """
        return {
            "max_new_tokens": getattr(args, "max_tokens", model_config.max_new_tokens),
            "temperature": getattr(args, "temperature", model_config.temperature),
            "top_p": getattr(args, "top_p", model_config.top_p),
            "do_sample": getattr(args, "do_sample", model_config.do_sample),
            "top_k": getattr(args, "top_k", model_config.top_k),
            "repetition_penalty": getattr(args, "repetition_penalty", model_config.repetition_penalty),
        }
    
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
    
    def _get_model_configuration(self, args):
        """
        Get the model configuration based on arguments.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Tuple of (model_name, model_config) or (None, None) if not found
        """
        # Get model configuration - from named model or direct config
        model_name = getattr(args, "model_name", None)
        if not self.app_config:
            self._load_config(args)
        models = self.app_config.model_registry.models
        
        # Case 1: Using a model from registry
        if model_name:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found in registry.")
                return None, None
            return model_name, models[model_name]
        
        # Case 2: Using direct model ID or default model
        model_id = getattr(args, "model", None)
        
        # If no model_id specified, use default model from registry
        if not model_id:
            default_model_name = self.app_config.model_registry.default_model
            if not default_model_name or default_model_name not in models:
                print(f"Error: No model specified and no valid default model set.")
                return None, None
            return default_model_name, models[default_model_name]
        
        # Create a temporary model configuration using direct arguments
        temp_model_name = f"temp_{int(time.time())}"
        temp_model_config = ModelConfig(
            model_id=model_id,
            engine_type=getattr(args, "engine", "transformers"),
            quantization=getattr(args, "quantization", "4bit"),
            dtype=getattr(args, "dtype", "float16"),
            name=temp_model_name,
            temperature=getattr(args, "temperature", 0.7),
            top_p=getattr(args, "top_p", 0.95),
            top_k=getattr(args, "top_k", 50),
            repetition_penalty=getattr(args, "repetition_penalty", 1.1),
            max_new_tokens=getattr(args, "max_tokens", 1024),
            do_sample=getattr(args, "do_sample", True),
        )
        
        return temp_model_name, temp_model_config
    
    def _load_config(self, args):
        """
        Load the configuration.
        
        Args:
            args: Command-line arguments
            
        Returns:
            The loaded configuration
        """
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        if config_file:
            # Load configuration from file
            self.app_config = AppConfig.load_from_file(config_file)
            # Override models file if specified
            if models_file:
                self.app_config.models_file = models_file
        else:
            # Load default configuration from file
            from pathlib import Path
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            if default_config_path.exists():
                self.app_config = AppConfig.load_from_file(default_config_path)
                # Ensure the models_file path is absolute for the default config
                if self.app_config.models_file and not Path(self.app_config.models_file).is_absolute():
                    config_dir = Path(__file__).parent.parent.parent / "config"
                    self.app_config.models_file = str(config_dir / "model_registry.json")
            else:
                # Fallback to empty configuration
                self.app_config = AppConfig()
            # Set models file if specified
            if models_file:
                self.app_config.models_file = models_file
        
        # Load model registry
        self.app_config.load_model_registry()
        
        return self.app_config
