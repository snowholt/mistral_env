"""
Inference service for unified CLI.
"""
import logging
import sys
import time
from typing import Dict, Any, Optional, List

from .base_service import BaseService
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
    
    def start_chat(self, args):
        """Start interactive chat with a model."""
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Configure generation parameters
        generation_config = {
            "max_new_tokens": getattr(args, "max_tokens", model_config.max_new_tokens),
            "temperature": getattr(args, "temperature", model_config.temperature),
            "top_p": getattr(args, "top_p", model_config.top_p),
            "do_sample": model_config.do_sample,
        }
        
        # Ensure the model is loaded
        try:
            if not self.model_manager.is_model_loaded(model_name):
                print(f"Loading model '{model_name}'...")
                self.model_manager.load_model(model_name, model_config)
                print(f"Model loaded successfully.")
            else:
                print(f"Using already loaded model '{model_name}'.")
                
            model = self.model_manager.get_model(model_name)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            logger.exception(f"Failed to load model {model_name}")
            return 1
        
        # Set up the interactive chat loop
        clear_terminal_screen()
        print(f"\n🤖 BeautyAI Chat - Model: {model_name} ({model_config.model_id})")
        print("=" * 60)
        print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
        print("=" * 60)
        
        chat_history = []
        
        try:
            while True:
                # Get user input
                user_input = input("\n👤 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("\n✅ Chat ended.")
                    break
                
                # Stream response if available
                print("\n🤖 Model: ", end="")
                sys.stdout.flush()
                
                response = ""
                streaming = hasattr(model, "generate_streaming")
                
                start_time = time.time()
                
                if streaming:
                    for token in model.generate_streaming(user_input, generation_config):
                        print(token, end="")
                        sys.stdout.flush()
                        response += token
                else:
                    response = model.generate(user_input, generation_config)
                    print(response)
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Add to chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})
                
                # Print some stats
                tokens_generated = len(response.split())
                print(f"\n[Generated ~{tokens_generated} tokens in {generation_time:.2f}s, "
                      f"{tokens_generated/generation_time:.1f} tokens/sec]")
                
        except KeyboardInterrupt:
            print("\n\n✅ Chat ended.")
    
    def run_test(self, args):
        """Run a simple test with the model."""
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
            if not self.model_manager.is_model_loaded(model_name):
                print(f"\nLoading model...")
                self.model_manager.load_model(model_name, model_config)
                print(f"Model loaded successfully.")
            else:
                print(f"\nUsing already loaded model.")
                
            model = self.model_manager.get_model(model_name)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            logger.exception(f"Failed to load model {model_name}")
            return 1
        
        # Run the test
        print("\n=== Test Prompt ===")
        print(prompt)
        print("\n=== Response ===")
        
        start_time = time.time()
        response = model.generate(prompt, generation_config)
        end_time = time.time()
        
        print(response)
        
        generation_time = end_time - start_time
        tokens_generated = len(response.split())
        
        print(f"\n=== Performance ===")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Tokens generated: ~{tokens_generated}")
        print(f"Tokens per second: ~{tokens_generated/generation_time:.2f}")
    
    def run_benchmark(self, args):
        """Run a benchmark on the model."""
        self._load_config(args)
        
        # Get model configuration
        model_name, model_config = self._get_model_configuration(args)
        if model_name is None or model_config is None:
            return 1
        
        # Get benchmark parameters
        num_runs = getattr(args, "num_runs", 3)
        prompt_length = getattr(args, "prompt_length", 100)
        output_length = getattr(args, "output_length", 100)
        
        print(f"\n=== Benchmarking {model_name} ===")
        print(f"Model ID: {model_config.model_id}")
        print(f"Engine: {model_config.engine_type}")
        print(f"Quantization: {model_config.quantization or 'none'}")
        print(f"Number of runs: {num_runs}")
        print(f"Prompt length: {prompt_length}")
        print(f"Output length: {output_length}")
        
        # Ensure the model is loaded
        try:
            if not self.model_manager.is_model_loaded(model_name):
                print(f"\nLoading model...")
                self.model_manager.load_model(model_name, model_config)
                print(f"Model loaded successfully.")
            else:
                print(f"\nUsing already loaded model.")
                
            model = self.model_manager.get_model(model_name)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            logger.exception(f"Failed to load model {model_name}")
            return 1
        
        # Generate benchmark prompt
        prompt = "Hello, " * (prompt_length // 2)  # Simple repeating pattern to reach desired length
        
        # Configure generation parameters
        generation_config = {
            "max_new_tokens": output_length,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
        
        # Run the benchmark
        results = []
        
        for i in range(num_runs):
            print(f"\nRun {i+1}/{num_runs}...")
            
            # Clear GPU memory stats before test
            start_memory = get_gpu_memory_stats()
            
            # Run generation
            start_time = time.time()
            _ = model.generate(prompt, generation_config)
            end_time = time.time()
            
            # Calculate results
            generation_time = end_time - start_time
            end_memory = get_gpu_memory_stats()
            
            # Append results
            results.append({
                "run_id": i+1,
                "time": generation_time,
                "tokens_per_sec": output_length / generation_time,
                "memory_before": start_memory[0]["memory_used_mb"] if start_memory else 0,
                "memory_after": end_memory[0]["memory_used_mb"] if end_memory else 0,
                "memory_delta": (end_memory[0]["memory_used_mb"] - start_memory[0]["memory_used_mb"]) 
                    if start_memory and end_memory else 0
            })
            
            print(f"Time: {generation_time:.2f}s, Tokens/s: {output_length/generation_time:.2f}")
        
        # Calculate average metrics
        avg_time = sum(result["time"] for result in results) / len(results)
        avg_tokens_per_sec = sum(result["tokens_per_sec"] for result in results) / len(results)
        
        print("\n=== Benchmark Results ===")
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Average tokens per second: {avg_tokens_per_sec:.2f}")
        print("\nDetailed runs:")
        
        for result in results:
            print(f"Run {result['run_id']}: {result['time']:.2f}s, {result['tokens_per_sec']:.2f} tokens/s, "
                  f"Memory delta: {result['memory_delta']:.2f} MB")

    def _get_model_configuration(self, args):
        """Get the model configuration based on arguments."""
        # Get model configuration - from named model or direct config
        model_name = getattr(args, "model_name", None)
        models = self.app_config.get_models()
        
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
            default_model_name = self.app_config.default_model_name
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
            name=temp_model_name
        )
        
        return temp_model_name, temp_model_config
    
    def _load_config(self, args):
        """Load the configuration."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        self.app_config = AppConfig(
            config_file=config_file,
            models_file=models_file
        )
