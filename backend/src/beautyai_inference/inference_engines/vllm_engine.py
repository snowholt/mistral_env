"""
Inference engine implementation based on vLLM.
"""
import torch
import time
import logging
from typing import List, Dict, Any, Optional, Union

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import get_gpu_memory_stats

logger = logging.getLogger(__name__)


class VLLMEngine(ModelInterface):
    """Inference engine implementation using vLLM."""
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the engine with a model configuration."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install it with 'pip install vllm'")
            
        self.config = model_config
        self.model = None
    
    def load_model(self) -> None:
        """Load the model into memory."""
        start_time = time.time()
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Configure quantization if specified
        quantization = None
        if self.config.quantization in ["awq", "squeezellm"]:
            quantization = self.config.quantization
        
        # Special handling for Mistral models (including Devstral)
        model_id_lower = self.config.model_id.lower()
        mistral_model_params = {}
        max_model_len = 8192  # Default
        
        if "mistral" in model_id_lower or "devstral" in model_id_lower:
            logger.info("Setting Mistral-specific parameters for vLLM")
            mistral_model_params = {
                "tokenizer_mode": "mistral",
                "trust_remote_code": True
            }
            
            # For large Mistral models, reduce max_model_len to save memory
            if "24b" in model_id_lower or "large" in model_id_lower:
                max_model_len = 4096
                logger.info(f"Large Mistral model detected, reducing max_model_len to {max_model_len}")
        
        # Load model with vLLM
        try:
            self.model = LLM(
                model=self.config.model_id,
                quantization=quantization,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=max_model_len,
                **mistral_model_params
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory when loading {self.config.model_id}")
            logger.error("Try reducing gpu_memory_utilization or using a smaller model")
            raise RuntimeError(f"Insufficient GPU memory to load {self.config.model_id}. "
                             f"Available memory may be too low. Consider using a quantized version.") from e
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_id}: {e}")
            raise
        
        loading_time = time.time() - start_time
        logger.info(f"Model loaded in {loading_time:.2f} seconds")
    
    def unload_model(self) -> None:
        """Unload the model from memory and free resources."""
        logger.info(f"Unloading vLLM model: {self.config.model_id}")
        
        try:
            # vLLM doesn't have a direct unload method, but we can delete the instance
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"vLLM model {self.config.model_id} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading vLLM model {self.config.model_id}: {e}")
            # Even if there's an error, try to clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _format_prompt(self, prompt: str) -> str:
        """Format a single prompt for the model."""
        return f"<s>[INST] {prompt} [/INST]"
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation for the model."""
        system_msg = None
        formatted_messages = "<s>"
        
        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]["content"]
            messages = messages[1:]
        
        # Add system message if present
        if system_msg:
            formatted_messages += f"[INST] {system_msg}\n\n"
        else:
            formatted_messages += "[INST] "
            
        # Add conversation history
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                if i == 0 and not system_msg:
                    formatted_messages += f"{content} [/INST]"
                else:
                    formatted_messages += f"{content} [/INST]"
            elif role == "assistant":
                formatted_messages += f" {content} </s><s>[INST] "
        
        return formatted_messages
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        if not self.model:
            self.load_model()
        
        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
        )
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Generate response
        outputs = self.model.generate([formatted_prompt], sampling_params)
        
        # Extract response
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        if not self.model:
            self.load_model()
        
        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
        )
        
        # Format conversation
        formatted_prompt = self._format_conversation(messages)
        
        # Generate response
        outputs = self.model.generate([formatted_prompt], sampling_params)
        
        # Extract response
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run a benchmark on the model."""
        if not self.model:
            self.load_model()
        
        # Prepare inputs
        formatted_prompt = self._format_prompt(prompt)
        
        # Setup sampling parameters
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        sampling_params = SamplingParams(
            temperature=0.0,  # Use greedy decoding for benchmarking
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        
        # Warmup run
        _ = self.model.generate([formatted_prompt], SamplingParams(max_tokens=10))
        
        # Actual benchmark
        start_time = time.time()
        outputs = self.model.generate([formatted_prompt], sampling_params)
        end_time = time.time()
        
        # Calculate stats
        generated_text = outputs[0].outputs[0].text
        output_tokens = len(generated_text.split())  # Rough estimate
        input_tokens = len(prompt.split())  # Rough estimate
        inference_time = end_time - start_time
        tokens_per_second = output_tokens / inference_time
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_stats": self.get_memory_stats(),
        }
    
    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """
        This is a pseudo-streaming implementation since vLLM doesn't support 
        token-by-token streaming in the same way as the Transformers API.
        """
        result = self.chat(messages, **kwargs)
        
        # Simulate streaming by yielding characters
        if callback:
            for char in result:
                callback(char)
                time.sleep(0.001)  # Small delay for UI responsiveness
        
        return result
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return get_gpu_memory_stats()
