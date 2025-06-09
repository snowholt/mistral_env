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
        
        # Enhanced CUDA detection and configuration
        import torch
        has_cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {has_cuda}")
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            gpu_memory_gb = gpu_memory_mb / 1024
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory_gb:.1f}GB")
            
            # Calculate optimal GPU layers based on model size and GPU memory
            # For GGUF models, typically each layer uses ~100-200MB for 14B models
            # Leave some memory for KV cache and operations
            available_gpu_memory_gb = gpu_memory_gb * 0.85  # Use 85% of GPU memory
            
            # Use all layers if we have enough memory, otherwise calculate optimal layers
            n_gpu_layers = -1  # Use all layers by default
            
            logger.info(f"Using all layers on GPU (n_gpu_layers={n_gpu_layers})")
        else:
            n_gpu_layers = 0
            logger.warning("CUDA not available, using CPU-only mode")
        
        # Configure optimal parameters for maximum speed
        n_ctx = getattr(self.config, 'max_seq_len', 4096)  # Optimal context for speed/quality balance
        n_batch = 2048  # Much larger batch size for RTX 4090 (was 1024)
        n_threads = 8   # Optimal for most CPUs (was 16) 
        n_threads_batch = 8  # Match main threads for consistency
        
        # GPU-specific settings optimized for RTX 4090
        gpu_settings = {}
        if has_cuda:
            gpu_settings.update({
                "main_gpu": 0,  # Use GPU 0 as primary
                "tensor_split": None,  # Don't split tensors (single GPU)
                "low_vram": False,  # We have 24GB, don't use low VRAM mode
                "mul_mat_q": True,  # Use quantized matrix multiplication for speed
                "flash_attn": True,  # Enable flash attention for speed
                "split_mode": 0,  # Don't split model across devices
                "offload_kqv": True,  # Offload KQV to GPU for speed
                "cont_batching": True,  # Enable continuous batching
            })
        
        # Model loading parameters optimized for maximum performance
        model_params = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_threads": n_threads,
            "n_threads_batch": n_threads_batch,  # Added for batch processing
            "verbose": False,  # Disable verbose for cleaner output and speed
            "use_mmap": True,  # Use memory mapping for efficiency
            "use_mlock": False,  # Don't lock memory (can cause issues)
            "rope_freq_base": 10000.0,  # RoPE frequency base
            "rope_freq_scale": 1.0,  # RoPE frequency scale
            "f16_kv": True,  # Use half precision for KV cache (faster)
            "logits_all": False,  # Only compute needed logits (faster)
            "vocab_only": False,  # Don't load vocab only
            "embedding": False,  # Not using embeddings
            "last_n_tokens_size": 64,  # Smaller buffer for speed
            **gpu_settings
        }
        
        try:
            logger.info(f"Initializing Llama with {n_gpu_layers} GPU layers, context: {n_ctx}")
            self.model = Llama(**model_params)
            
            loading_time = time.time() - start_time
            logger.info(f"✅ GGUF model loaded successfully in {loading_time:.2f} seconds")
            
            # Check GPU memory usage after loading
            if has_cuda and n_gpu_layers != 0:
                try:
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    logger.info(f"GPU memory used: {gpu_memory_used:.2f}GB, cached: {gpu_memory_cached:.2f}GB")
                    
                    if gpu_memory_used > 0.5:  # If more than 500MB is used
                        logger.info("✅ Model successfully loaded on GPU!")
                    else:
                        logger.warning("⚠️ Low GPU memory usage - model may be running on CPU")
                except Exception as mem_e:
                    logger.warning(f"Could not check GPU memory usage: {mem_e}")
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Failed to load GGUF model: {e}")
            
            # Enhanced error handling based on common CUDA issues
            if "cuda" in error_msg or "gpu" in error_msg:
                logger.warning("GPU loading failed, attempting CPU fallback...")
                
                # Try with fewer GPU layers first
                if n_gpu_layers == -1:
                    logger.info("Trying with limited GPU layers...")
                    try:
                        model_params["n_gpu_layers"] = 20  # Try with 20 layers
                        self.model = Llama(**model_params)
                        logger.info("✅ Model loaded with limited GPU layers")
                        return
                    except Exception:
                        logger.info("Limited GPU layers failed, trying CPU-only...")
                
                # Final fallback to CPU-only
                try:
                    model_params.update({
                        "n_gpu_layers": 0,
                        "main_gpu": 0,
                        "tensor_split": None,
                        "low_vram": False,
                        "mul_mat_q": False,
                    })
                    
                    self.model = Llama(**model_params)
                    logger.warning("⚠️ Model loaded in CPU-only mode as fallback")
                    
                except Exception as e2:
                    logger.error(f"Failed to load model even in CPU mode: {e2}")
                    raise
            else:
                # Non-CUDA related error
                raise
    
    def _find_gguf_model_path(self) -> Optional[str]:
        """Find the GGUF model file path."""
        # Check if model_filename is specified in config
        if hasattr(self.config, 'model_filename') and self.config.model_filename:
            filename = self.config.model_filename
            logger.info(f"Looking for specific model file: {filename}")
        else:
            # Default GGUF filename patterns based on model
            if "devstral" in self.config.model_id.lower():
                filename = "devstralQ4_K_M.gguf"
            elif "bee1reason" in self.config.model_id.lower():
                if "i1-q4_k_s" in str(getattr(self.config, 'quantization', '')).lower():
                    filename = "Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf"
                else:
                    filename = "Bee1reason-arabic-Qwen-14B-Q4_K_M.gguf"
            else:
                filename = "*.gguf"  # Fallback pattern
        
        # Common paths to check
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Pattern for HuggingFace cache (handle various naming conventions)
        model_id_safe = self.config.model_id.replace("/", "--")
        
        # Search patterns with various possible cache directory names
        search_patterns = [
            f"{cache_dir}/models--{model_id_safe}/snapshots/*/",
            f"{cache_dir}/models--{model_id_safe.replace('_', '--')}/snapshots/*/",
            f"{cache_dir}/models--{model_id_safe.replace('-', '--')}/snapshots/*/",
        ]
        
        import glob
        
        # Search for specific filename first
        if filename != "*.gguf":
            for pattern_dir in search_patterns:
                dirs = glob.glob(pattern_dir)
                for dir_path in dirs:
                    full_path = os.path.join(dir_path, filename)
                    if os.path.exists(full_path):
                        logger.info(f"Found GGUF model at: {full_path}")
                        return full_path
        
        # If specific filename not found, search for any GGUF files
        logger.info("Specific filename not found, searching for any GGUF files...")
        for pattern_dir in search_patterns:
            dirs = glob.glob(pattern_dir)
            for dir_path in dirs:
                gguf_files = glob.glob(os.path.join(dir_path, "*.gguf"))
                if gguf_files:
                    # Sort by file size (largest first) - usually the main model
                    gguf_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                    
                    # Prefer Q4_K_M quantization if available
                    for gguf_file in gguf_files:
                        if "Q4_K_M" in os.path.basename(gguf_file):
                            logger.info(f"Found preferred Q4_K_M model: {gguf_file}")
                            return gguf_file
                    
                    # Otherwise use the first (largest) file
                    logger.info(f"Found GGUF model: {gguf_files[0]}")
                    return gguf_files[0]
        
        # Also check if it's a direct path
        if os.path.exists(self.config.model_id):
            return self.config.model_id
            
        logger.error(f"Could not find any GGUF files for model {self.config.model_id}")
        logger.info(f"Searched in patterns: {search_patterns}")
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
        """Generate text from a prompt with optimized settings for speed."""
        if not self.model:
            self.load_model()
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Optimized generation parameters for maximum speed
        response = self.model(
            formatted_prompt,
            max_tokens=kwargs.get("max_new_tokens", min(self.config.max_new_tokens, 256)),  # Reduced for speed
            temperature=kwargs.get("temperature", getattr(self.config, 'temperature', 0.1)),
            top_p=kwargs.get("top_p", getattr(self.config, 'top_p', 0.9)),  # Reduced for speed
            top_k=kwargs.get("top_k", 20),  # Reduced top_k for faster sampling
            repeat_penalty=kwargs.get("repeat_penalty", 1.05),  # Reduced for speed
            echo=False,
            stop=["</s>", "[INST]", "[/INST]", "User:", "\n\n\n"],  # More stop tokens
            # Aggressive speed optimizations
            stream=False,  # Disable streaming for faster batch processing
            tfs_z=1.0,     # TFS disabled for speed
            typical_p=1.0,  # Typical sampling disabled
            mirostat_mode=0,  # Disable mirostat for speed
            frequency_penalty=0.0,  # Disable for speed
            presence_penalty=0.0,   # Disable for speed
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
        
        # Generate response using chat completion with speed optimizations
        try:
            response = self.model.create_chat_completion(
                messages=formatted_messages,
                max_tokens=kwargs.get("max_new_tokens", min(self.config.max_new_tokens, 256)),  # Reduced for speed
                temperature=kwargs.get("temperature", getattr(self.config, 'temperature', 0.1)),
                top_p=kwargs.get("top_p", getattr(self.config, 'top_p', 0.9)),  # Slightly reduced for speed
                top_k=kwargs.get("top_k", 20),  # Reduced top_k for faster sampling
                repeat_penalty=kwargs.get("repeat_penalty", 1.05),  # Reduced for speed
                # Aggressive speed optimizations
                stream=False,
                tfs_z=1.0,
                typical_p=1.0,
                mirostat_mode=0,
                # Additional speed parameters
                stop=["User:", "Human:", "\n\n\n"],  # Stop tokens for faster generation
                frequency_penalty=0.0,
                presence_penalty=0.0,
                # Disable logit bias for speed
                logit_bias={},
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
