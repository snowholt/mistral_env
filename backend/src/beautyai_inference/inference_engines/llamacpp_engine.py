"""
Inference engine implementation based on llama.cpp.

This implementation is optimized for NVIDIA RTX 4090 (24GB VRAM) to achieve
50-100+ tokens per second with the following key optimizations:

1. **Batch Size Optimization**: n_batch=4096 (increased from 2048) to fully
   utilize the RTX 4090's massive parallel processing capabilities.

2. **Thread Optimization**: n_threads=16 for modern multi-core CPUs to maximize
   parallel preprocessing and batch handling.

3. **Context Size Optimization**: n_ctx=2048 (reduced from 4096) for faster
   inference while maintaining sufficient context for most use cases.

4. **Aggressive Sampling Parameters**: Lower top_k (10 vs 20) and top_p (0.8 vs 0.9)
   for faster token sampling without significant quality loss.

5. **GPU Memory Utilization**: All layers on GPU (-1) with optimized settings
   for flash attention, continuous batching, and quantized matrix operations.

These settings are specifically tuned for GGUF models on RTX 4090 hardware.
For other GPUs or quality requirements, parameters may need adjustment.
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
        """Load the model into memory with optimized settings for RTX 4090."""
        start_time = time.time()
        logger.info(f"Loading GGUF model: {self.config.model_id}")
        
        # Find the GGUF model file
        model_path = self._find_gguf_model_path()
        if not model_path:
            raise FileNotFoundError(f"Could not find GGUF model file for {self.config.model_id}")
        
        # If the resolved path doesn't end with .gguf, create a temporary copy with proper extension
        if not model_path.endswith('.gguf'):
            import tempfile
            import shutil
            
            # Create temporary file with .gguf extension
            temp_dir = tempfile.mkdtemp()
            temp_model_path = os.path.join(temp_dir, f"{self.config.model_filename or 'model.gguf'}")
            
            logger.info(f"Copying model from blob to temporary location: {temp_model_path}")
            shutil.copy2(model_path, temp_model_path)
            model_path = temp_model_path
        
        logger.info(f"Loading GGUF file: {model_path}")
        
        # Enhanced CUDA detection and configuration
        import torch
        has_cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {has_cuda}")
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory_gb:.1f}GB")
            
            # Use config value or default to all layers on GPU
            n_gpu_layers = getattr(self.config, 'n_gpu_layers', -1)
            logger.info(f"Using {n_gpu_layers} layers on GPU (from config)")
        else:
            n_gpu_layers = 0
            logger.warning("CUDA not available, using CPU-only mode")
        
        # Use config values or optimized defaults for RTX 4090
        n_ctx = getattr(self.config, 'n_ctx', 1024)  # Use config or reduced context size for speed
        n_batch = getattr(self.config, 'n_batch', 8192)  # Use config or maximum batch size for RTX 4090
        n_threads = getattr(self.config, 'n_threads', 24)  # Use config or maximum CPU threads
        n_threads_batch = n_threads  # Match main threads
        
        logger.info(f"Model parameters from config: n_ctx={n_ctx}, n_batch={n_batch}, n_threads={n_threads}")
        
        # GPU-specific settings optimized for RTX 4090
        gpu_settings = {}
        if has_cuda:
            gpu_settings.update({
                "main_gpu": 0,
                "tensor_split": None,
                "low_vram": False,
                "mul_mat_q": True,
                "flash_attn": True,
                "split_mode": 0,
                "offload_kqv": True,
                "cont_batching": True,
            })
        
                # Model loading parameters - use minimal parameters for better compatibility
        model_params = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": False,
        }
        
        # Add only essential parameters that are known to work
        if has_cuda and n_gpu_layers != 0:  # -1 means all layers, > 0 means specific number
            # Use full configuration values for ULTRA speed optimization
            model_params.update({
                "n_batch": n_batch,  # Use config value for maximum speed
                "n_threads": n_threads,  # Use config value for optimal threading
            })
            logger.info(f"Using OPTIMIZED GPU parameters: n_gpu_layers={n_gpu_layers}, n_batch={n_batch}, n_threads={n_threads}")
        else:
            # CPU-only parameters - still use config values
            model_params.update({
                "n_batch": n_batch,
                "n_threads": n_threads,
            })
        
        def _attempt_load(params: dict, label: str) -> bool:
            try:
                logger.info(f"Initializing Llama ({label}) with n_gpu_layers={params.get('n_gpu_layers')} n_ctx={params.get('n_ctx')}")
                for key, value in params.items():
                    logger.debug(f"  {key}: {value}")
                self.model = Llama(**params)
                return True
            except Exception as e:  # pragma: no cover - depends on local environment
                logger.error(f"Load attempt '{label}' failed: {e}")
                return False

        # Primary attempt
        primary_ok = _attempt_load(model_params, "primary")
        if not primary_ok:
            # Fallback strategy matrix
            fallbacks: List[dict] = []
            # 1. Reduce context
            if model_params.get("n_ctx", 0) > 2048:
                fb = dict(model_params)
                fb["n_ctx"] = 2048
                fallbacks.append(fb)
            # 2. CPU-only minimal (detect GPU issues or unsupported quantization kernels)
            if has_cuda:
                fb = dict(model_params)
                fb["n_gpu_layers"] = 0
                fallbacks.append(fb)
            # 3. Smaller batch & threads
            fb = dict(model_params)
            fb["n_batch"] = 256
            fb["n_threads"] = min(fb.get("n_threads", 4), 8)
            fallbacks.append(fb)
            # 4. Combined minimal (CPU + reduced ctx)
            if has_cuda:
                fb = dict(model_params)
                fb["n_gpu_layers"] = 0
                fb["n_ctx"] = 2048
                fb["n_batch"] = 128
                fb["n_threads"] = 4
                fallbacks.append(fb)

            # 5. If quantization is *_Q4_K_S try alternate *_Q4_K_M file in same dir
            alt_path = None
            if "Q4_K_S" in os.path.basename(model_path):
                directory = os.path.dirname(model_path)
                for fname in os.listdir(directory):
                    if fname.endswith(".gguf") and "Q4_K_M" in fname:
                        alt_path = os.path.join(directory, fname)
                        break
                if alt_path:
                    fb = dict(model_params)
                    fb["model_path"] = alt_path
                    fallbacks.append(fb)
                    logger.warning(f"Adding alternate quantization fallback file: {alt_path}")

            for idx, fb in enumerate(fallbacks, 1):
                if _attempt_load(fb, f"fallback#{idx}"):
                    model_params = fb  # record winning params
                    primary_ok = True
                    break

        if not primary_ok:
            raise RuntimeError(
                "All llama.cpp model load attempts failed. See prior log entries for diagnostics. "
                "Consider updating llama-cpp-python or using a different quantization (e.g. Q4_K_M)."
            )

        loading_time = time.time() - start_time
        logger.info(f"✅ GGUF model loaded successfully in {loading_time:.2f} seconds (params={ {k: model_params[k] for k in ['model_path','n_gpu_layers','n_ctx']} })")
        if has_cuda and model_params.get("n_gpu_layers", 0) != 0:
            logger.info("✅ Model successfully loaded on GPU (llama.cpp independent allocator)")
            logger.info("   Verify with: nvidia-smi | grep $(basename ${model_params['model_path']}) || true")
    
    def _find_gguf_model_path(self) -> Optional[str]:
        """Find the GGUF model file path."""
        # Check if explicit model_path is provided in config
        if hasattr(self.config, 'model_path') and self.config.model_path:
            if os.path.exists(self.config.model_path):
                logger.info(f"Using explicit model path: {self.config.model_path}")
                return self.config.model_path
            else:
                logger.warning(f"Explicit model path not found: {self.config.model_path}")
        
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
                        # Keep original path if it ends with .gguf (even if it's a symlink)
                        if full_path.endswith('.gguf'):
                            logger.info(f"Found GGUF model at: {full_path}")
                            return full_path
                        else:
                            # Resolve symlinks only if original doesn't end with .gguf
                            resolved_path = os.path.realpath(full_path)
                            logger.info(f"Found GGUF model at: {full_path}")
                            logger.info(f"Resolved symlink to: {resolved_path}")
                            return resolved_path
        
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
                            if gguf_file.endswith('.gguf'):
                                logger.info(f"Found preferred Q4_K_M model: {gguf_file}")
                                return gguf_file
                            else:
                                resolved_path = os.path.realpath(gguf_file)
                                logger.info(f"Found preferred Q4_K_M model: {gguf_file}")
                                logger.info(f"Resolved symlink to: {resolved_path}")
                                return resolved_path
                    
                    # Otherwise use the first (largest) file
                    first_file = gguf_files[0]
                    if first_file.endswith('.gguf'):
                        logger.info(f"Found GGUF model: {first_file}")
                        return first_file
                    else:
                        resolved_path = os.path.realpath(first_file)
                        logger.info(f"Found GGUF model: {first_file}")
                        logger.info(f"Resolved symlink to: {resolved_path}")
                        return resolved_path
        
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
        """Generate text from a prompt with optimized settings for maximum speed."""
        if not self.model:
            self.load_model()
        
        # **NEW: Handle enable_thinking parameter**
        enable_thinking = kwargs.get('enable_thinking', True)  # Default to True for backward compatibility
        if enable_thinking is None:
            enable_thinking = True
        
        # **NEW: Modify prompt based on thinking mode**
        if not enable_thinking:
            # For no-thinking mode, explicitly instruct the model
            modified_prompt = f"{prompt}\n\nPlease respond directly without showing your thinking process or reasoning steps."
        else:
            # For thinking mode, allow normal behavior
            modified_prompt = prompt
        
        # Format prompt
        formatted_prompt = self._format_prompt(modified_prompt)
        
        # Optimized generation parameters for MAXIMUM speed on RTX 4090
        
        # Get temperature from kwargs, config, or custom_generation_params
        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = getattr(self.config, 'temperature', None)
            if temperature is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                temperature = self.config.custom_generation_params.get('temperature', 0.01)
            if temperature is None:
                temperature = 0.01
        
        # Ensure temperature is not None and is a valid float
        try:
            temperature = float(temperature) if temperature is not None else 0.01
            # Validate temperature range
            if temperature <= 0:
                temperature = 0.01
            elif temperature > 2.0:
                temperature = 2.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid temperature value: {temperature}, using default 0.01")
            temperature = 0.01
        
        # Get top_p from kwargs, config, or custom_generation_params
        top_p = kwargs.get("top_p")
        if top_p is None:
            top_p = getattr(self.config, 'top_p', None)
            if top_p is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                top_p = self.config.custom_generation_params.get('top_p', 0.5)
            if top_p is None:
                top_p = 0.5
        
        # Ensure top_p is not None and is a valid float
        try:
            top_p = float(top_p) if top_p is not None else 0.5
            # Validate top_p range
            if top_p <= 0:
                top_p = 0.1
            elif top_p > 1.0:
                top_p = 1.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid top_p value: {top_p}, using default 0.5")
            top_p = 0.5
        
        # Get top_k from kwargs or custom_generation_params
        top_k = kwargs.get("top_k")
        if top_k is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
            top_k = self.config.custom_generation_params.get('top_k', 1)
        if top_k is None:
            top_k = 1
            
        # Ensure top_k is not None and is a valid int
        try:
            top_k = int(top_k) if top_k is not None else 1
            # Validate top_k range
            if top_k <= 0:
                top_k = 1
            elif top_k > 100:
                top_k = 100
        except (TypeError, ValueError):
            logger.warning(f"Invalid top_k value: {top_k}, using default 1")
            top_k = 1
            
        # Get repeat_penalty from kwargs or custom_generation_params  
        repeat_penalty = kwargs.get("repeat_penalty", kwargs.get("repetition_penalty"))
        if repeat_penalty is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
            repeat_penalty = self.config.custom_generation_params.get('repetition_penalty', 1.0)
        if repeat_penalty is None:
            repeat_penalty = 1.0
        
        # Ensure repeat_penalty is not None and is a valid float
        try:
            repeat_penalty = float(repeat_penalty) if repeat_penalty is not None else 1.0
            # Validate repeat_penalty range
            if repeat_penalty <= 0:
                repeat_penalty = 1.0
            elif repeat_penalty > 2.0:
                repeat_penalty = 2.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid repeat_penalty value: {repeat_penalty}, using default 1.0")
            repeat_penalty = 1.0
        
        response = self.model(
            formatted_prompt,
            max_tokens=kwargs.get("max_new_tokens", min(self.config.max_new_tokens, 64)),  # Extremely short for speed
            temperature=temperature,  # Near-greedy
            top_p=top_p,  # Ultra-aggressive for speed
            top_k=top_k,  # Maximum speed - nearly greedy
            repeat_penalty=repeat_penalty,  # Disabled for speed
            echo=False,
            stop=["</s>", "[INST]", "[/INST]", "User:", "\n\n\n"],
            # Ultra-aggressive speed optimizations
            stream=False,
            tfs_z=1.0,     # TFS disabled for speed
            typical_p=1.0,  # Typical sampling disabled
            mirostat_mode=0,  # Disable mirostat for speed
            frequency_penalty=0.0,  # Disable for speed
            presence_penalty=0.0,   # Disable for speed
        )
        
        # Extract response text
        if response and 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text'].strip()
            
            # **NEW: Post-process response based on thinking mode**
            if not enable_thinking:
                # Remove thinking blocks if thinking mode is disabled
                response_text = self._clean_thinking_blocks(response_text)
            
            return response_text
        else:
            return ""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation with reliable parameters."""
        if not self.model:
            self.load_model()
        
        # Convert messages to llama.cpp format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(ChatCompletionMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        # Use more reliable parameters instead of ultra-aggressive ones
        try:
            max_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 
                                                                 min(self.config.max_new_tokens, 256)))
            
            # Get temperature from kwargs, config, or custom_generation_params
            temperature = kwargs.get("temperature")
            if temperature is None:
                temperature = getattr(self.config, 'temperature', None)
                if temperature is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                    temperature = self.config.custom_generation_params.get('temperature', 0.7)
                if temperature is None:
                    temperature = 0.7
            
            # Ensure temperature is not None and is a valid float
            try:
                temperature = float(temperature) if temperature is not None else 0.7
                # Validate temperature range
                if temperature <= 0:
                    temperature = 0.1
                elif temperature > 2.0:
                    temperature = 2.0
            except (TypeError, ValueError):
                logger.warning(f"Invalid temperature value: {temperature}, using default 0.7")
                temperature = 0.7
            
            # Get top_p from kwargs, config, or custom_generation_params
            top_p = kwargs.get("top_p")
            if top_p is None:
                top_p = getattr(self.config, 'top_p', None)
                if top_p is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                    top_p = self.config.custom_generation_params.get('top_p', 0.9)
                if top_p is None:
                    top_p = 0.9
            
            # Ensure top_p is not None and is a valid float
            try:
                top_p = float(top_p) if top_p is not None else 0.9
                # Validate top_p range
                if top_p <= 0:
                    top_p = 0.1
                elif top_p > 1.0:
                    top_p = 1.0
            except (TypeError, ValueError):
                logger.warning(f"Invalid top_p value: {top_p}, using default 0.9")
                top_p = 0.9
            
            # Get top_k from kwargs or custom_generation_params
            top_k = kwargs.get("top_k")
            if top_k is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                top_k = self.config.custom_generation_params.get('top_k', 40)
            if top_k is None:
                top_k = 40
                
            # Ensure top_k is not None and is a valid int
            try:
                top_k = int(top_k) if top_k is not None else 40
                # Validate top_k range
                if top_k <= 0:
                    top_k = 1
                elif top_k > 100:
                    top_k = 100
            except (TypeError, ValueError):
                logger.warning(f"Invalid top_k value: {top_k}, using default 40")
                top_k = 40
                
            # Get repeat_penalty from kwargs or custom_generation_params
            repeat_penalty = kwargs.get("repeat_penalty", kwargs.get("repetition_penalty"))
            if repeat_penalty is None and hasattr(self.config, 'custom_generation_params') and self.config.custom_generation_params:
                repeat_penalty = self.config.custom_generation_params.get('repetition_penalty', 1.1)
            if repeat_penalty is None:
                repeat_penalty = 1.1
            
            # Ensure repeat_penalty is not None and is a valid float
            try:
                repeat_penalty = float(repeat_penalty) if repeat_penalty is not None else 1.1
                # Validate repeat_penalty range
                if repeat_penalty <= 0:
                    repeat_penalty = 1.0
                elif repeat_penalty > 2.0:
                    repeat_penalty = 2.0
            except (TypeError, ValueError):
                logger.warning(f"Invalid repeat_penalty value: {repeat_penalty}, using default 1.1")
                repeat_penalty = 1.1
            
            logger.info(f"Chat parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")
            
            response = self.model.create_chat_completion(
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stream=False,
                stop=["</s>", "[INST]", "[/INST]", "User:", "Human:"],
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content'].strip()
                logger.info(f"Chat response length: {len(content)} chars")
                return content
            else:
                logger.warning("Empty response from chat completion")
                return ""
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            # Fallback to simple generation
            try:
                conversation_text = self._format_conversation(messages)
                return self.generate(conversation_text, **kwargs)
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}")
                return "I apologize, but I encountered an error generating a response."
    
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
        Streaming chat - optimized for speed (no artificial delays).
        """
        # Call regular chat with the same optimized parameters
        result = self.chat(messages, **kwargs)
        
        if callback:
            # Stream result without artificial delays for maximum performance
            for char in result:
                callback(char)
                # Remove the artificial delay that was slowing down the chat interface
                # time.sleep(0.01)  # <-- This was the bottleneck!
        
        return result
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return get_gpu_memory_stats()
    
    def _clean_thinking_blocks(self, text: str) -> str:
        """Remove thinking blocks from text when thinking mode is disabled."""
        import re
        
        # Remove <think>...</think> blocks (case insensitive, multiline)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
