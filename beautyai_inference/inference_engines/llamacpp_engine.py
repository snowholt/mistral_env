"""
Inference engine implementation based on llama.cpp.
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional

try:
    from llama_cpp import Llama, ChatCompletionMessage
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import get_gpu_memory_stats

logger = logging.getLogger(__name__)


class LlamaCppEngine(ModelInterface):
    """Inference engine implementation using llama.cpp."""
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the engine with a model configuration."""
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Install it with 'pip install llama-cpp-python[server]'")
            
        self.config = model_config
        self.model = None
    
    def load_model(self) -> None:
        """Load the model into memory."""
        start_time = time.time()
        logger.info(f"Loading GGUF model: {self.config.model_id}")
        
        # Find the GGUF model file
        model_path = self._find_gguf_model_path()
        if not model_path:
            raise FileNotFoundError(f"Could not find GGUF model file for {self.config.model_id}")
        
        logger.info(f"Loading GGUF file: {model_path}")
        
        # Configure GPU layers (use all layers for GPU acceleration)
        n_gpu_layers = -1  # Use all layers on GPU
        
        # Configure memory usage (conservative settings for 24GB GPU)
        n_ctx = 4096  # Context length
        n_batch = 512  # Batch size
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                verbose=False,
                use_mmap=True,  # Use memory mapping for efficiency
                use_mlock=False,  # Don't lock all memory
            )
            
            loading_time = time.time() - start_time
            logger.info(f"GGUF model loaded in {loading_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model {model_path}: {e}")
            raise
    
    def _find_gguf_model_path(self) -> Optional[str]:
        """Find the GGUF model file path."""
        # Check if model_filename is specified in config
        if hasattr(self.config, 'model_filename') and self.config.model_filename:
            filename = self.config.model_filename
        else:
            # Default GGUF filename patterns
            filename = "devstralQ4_K_M.gguf"  # Default for Devstral
        
        # Common paths to check
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Pattern for HuggingFace cache
        model_id_safe = self.config.model_id.replace("/", "--")
        
        # Search patterns
        search_patterns = [
            f"{cache_dir}/models--{model_id_safe}/snapshots/*/",
            f"{cache_dir}/models--{model_id_safe.replace('_', '--')}/snapshots/*/",
        ]
        
        for pattern_dir in search_patterns:
            import glob
            dirs = glob.glob(pattern_dir)
            for dir_path in dirs:
                full_path = os.path.join(dir_path, filename)
                if os.path.exists(full_path):
                    logger.info(f"Found GGUF model at: {full_path}")
                    return full_path
        
        # Also check if it's a direct path
        if os.path.exists(self.config.model_id):
            return self.config.model_id
            
        logger.error(f"Could not find GGUF file {filename} for model {self.config.model_id}")
        return None
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        logger.info(f"Unloading llama.cpp model: {self.config.model_id}")
        
        try:
            if hasattr(self, 'model') and self.model is not None:
                # llama.cpp doesn't have explicit unload, but we can delete the instance
                del self.model
                self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"llama.cpp model {self.config.model_id} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading llama.cpp model {self.config.model_id}: {e}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format a single prompt for the model."""
        return f"<s>[INST] {prompt} [/INST]"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        if not self.model:
            self.load_model()
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Generate response
        response = self.model(
            formatted_prompt,
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", getattr(self.config, 'temperature', 0.1)),
            top_p=kwargs.get("top_p", getattr(self.config, 'top_p', 0.95)),
            echo=False,
            stop=["</s>", "[INST]", "[/INST]"],
        )
        
        # Extract response text
        if response and 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['text'].strip()
        else:
            return ""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        if not self.model:
            self.load_model()
        
        # Convert messages to llama.cpp format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(ChatCompletionMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        # Generate response using chat completion
        try:
            response = self.model.create_chat_completion(
                messages=formatted_messages,
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", getattr(self.config, 'temperature', 0.1)),
                top_p=kwargs.get("top_p", getattr(self.config, 'top_p', 0.95)),
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content'].strip()
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"Chat completion failed, falling back to simple generation: {e}")
            # Fallback to simple generation
            conversation_text = self._format_conversation(messages)
            return self.generate(conversation_text, **kwargs)
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation for simple generation."""
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                conversation += f"System: {content}\n\n"
            elif role == "user":
                conversation += f"User: {content}\n\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n\n"
        
        conversation += "Assistant:"
        return conversation
    
    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run a benchmark on the model."""
        if not self.model:
            self.load_model()
        
        # Prepare inputs
        formatted_prompt = self._format_prompt(prompt)
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        
        # Warmup run
        _ = self.model(formatted_prompt, max_tokens=10)
        
        # Actual benchmark
        start_time = time.time()
        response = self.model(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,  # Use greedy decoding for benchmarking
            top_p=1.0,
            echo=False,
        )
        end_time = time.time()
        
        # Calculate stats
        generated_text = response['choices'][0]['text'] if response['choices'] else ""
        output_tokens = len(generated_text.split())  # Rough estimate
        input_tokens = len(prompt.split())  # Rough estimate
        inference_time = end_time - start_time
        tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_stats": self.get_memory_stats(),
        }
    
    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """
        Streaming chat (basic implementation).
        """
        # For now, just call regular chat and simulate streaming
        result = self.chat(messages, **kwargs)
        
        if callback:
            for char in result:
                callback(char)
                time.sleep(0.01)  # Small delay for UI responsiveness
        
        return result
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return get_gpu_memory_stats()
