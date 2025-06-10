"""
Inference engine implementation based on Hugging Face Transformers.
"""
import torch
import time
import logging
import warnings
from threading import Thread
from typing import List, Dict, Any, Optional, Union, Tuple
import os

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TextIteratorStreamer,
    AutoConfig,
    MistralConfig,
)

# Import registration utilities
from transformers.models.auto import modeling_auto

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import get_gpu_memory_stats, time_function

logger = logging.getLogger(__name__)


def suppress_transformers_warnings():
    """Suppress common Transformers warnings about generation parameters."""
    # Filter out specific warnings about generation parameters
    warnings.filterwarnings("ignore", message=".*generation flags.*")
    warnings.filterwarnings("ignore", message=".*generation config.*")
    warnings.filterwarnings("ignore", message=".*not valid.*")
    warnings.filterwarnings("ignore", message=".*may be ignored.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    # Set environment variable to reduce verbosity
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def register_mistral3_modules():
    """Register Mistral3 modules with the transformers library."""
    try:
        logger.info("Attempting to register Mistral3 configuration and model classes")
        from transformers import Mistral3Config, Mistral3ForCausalLM

        # Register the Mistral3Config with AutoConfig
        if not hasattr(Mistral3Config, "_register_for_auto_class") or not hasattr(modeling_auto, "AUTO_MODEL_FOR_CAUSAL_LM_MAPPING"):
            logger.warning("Could not register Mistral3Config: missing registration methods")
            return False

        # Don't re-register if already registered
        if "Mistral3Config" in str(modeling_auto.AUTO_MODEL_FOR_CAUSAL_LM_MAPPING):
            logger.info("Mistral3Config already registered with AutoModel classes")
            return True

        # Register for both causal LM and base model
        Mistral3Config._register_for_auto_class("AutoModelForCausalLM")
        modeling_auto.AUTO_MODEL_FOR_CAUSAL_LM_MAPPING.register(Mistral3Config, Mistral3ForCausalLM)
        logger.info("Successfully registered Mistral3 modules with transformers")
        return True
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to register Mistral3 modules: {e}")
        return False


class TransformersEngine(ModelInterface):
    """Inference engine implementation using Hugging Face Transformers."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the engine with a model configuration."""
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Content filter will be injected when needed
        self.content_filter = None

        # Register Mistral3 modules if we're loading a Mistral3 model
        if "mistral" in self.config.model_id.lower() and "3" in self.config.model_id:
            register_mistral3_modules()

    def load_model(self) -> None:
        """Load the model into memory."""
        start_time = time.time()
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Suppress warnings early
        suppress_transformers_warnings()
        
        # Setup download optimization
        self._setup_download_optimization()
        
        # Check if model is cached, if not pre-download with optimizations
        if not self._check_model_cached(self.config.model_id):
            logger.info("Model not cached, starting optimized download...")
            self._pre_download_model(self.config.model_id)

        # Configure quantization if specified
        quantization_config = None
        if self.config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.config.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Load tokenizer with trust_remote_code=True and model-specific settings
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }

        # Add Mistral-specific parameters if it's likely a Mistral model
        if "mistral" in self.config.model_id.lower():
            logger.info("Detected Mistral model, using Mistral-specific tokenizer settings")
            # For newer Mistral models like Devstral, try different approaches
            if "devstral" in self.config.model_id.lower() or "2505" in self.config.model_id.lower():
                # These models may need special tokenizer handling
                tokenizer_kwargs.update({
                    "use_fast": True,
                    "padding_side": "left"
                })

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                **tokenizer_kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with standard settings: {e}")
            logger.info("Trying to load tokenizer with alternative settings...")
            
            # Try with minimal settings
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    use_fast=False  # Force slow tokenizer for compatibility
                )
            except Exception as e2:
                logger.error(f"Failed to load tokenizer with alternative settings: {e2}")
                
                # Last resort: try to use a compatible tokenizer
                try:
                    logger.info("Trying to use LlamaTokenizer as fallback...")
                    from transformers import LlamaTokenizer
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        "meta-llama/Llama-2-7b-hf",  # Use a known working tokenizer
                        trust_remote_code=True
                    )
                    logger.warning("Using fallback tokenizer - results may not be optimal")
                except Exception as e3:
                    logger.error(f"All tokenizer loading methods failed: {e3}")
                    raise RuntimeError(f"Could not load any compatible tokenizer for {self.config.model_id}")

        # First check the model configuration to handle special cases
        try:
            # Set stronger quantization for large models
            model_id_lower = self.config.model_id.lower()
            if "24b" in model_id_lower or "70b" in model_id_lower:
                logger.info("Large model detected, using stronger 4-bit quantization settings")
                # Override with more aggressive quantization for large models
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_has_fp16_weight=True,
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=None,
                    llm_int8_enable_fp32_cpu_offload=True
                )

            # For Mistral3 models, try to register the configuration classes
            if "mistral" in model_id_lower and "3" in model_id_lower:
                logger.info("Attempting to handle Mistral3 model")
                register_mistral3_modules()

                # Set environment variables that might help with Mistral3 loading
                os.environ["HF_ALLOW_CODE_EVAL"] = "1" 
                os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"

            # Load the configuration first to check what kind of model it is
            config = AutoConfig.from_pretrained(
                self.config.model_id,
                trust_remote_code=True
            )
            logger.info(f"Detected model configuration: {config.__class__.__name__}")

            # Choose the right model class based on the architecture
            architecture = getattr(self.config, 'model_architecture', 'causal_lm')

            # For sequence-to-sequence models, use AutoModelForSeq2SeqLM
            if architecture == 'seq2seq_lm':
                logger.info(f"Loading as a sequence-to-sequence model: {self.config.model_id}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_id,
                    config=config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            # For causal language models, use AutoModelForCausalLM with original methods
            else:
                logger.info(f"Loading as a causal language model: {self.config.model_id}")

                # Try different loading approaches in sequence
                loading_methods = [
                    # Method 1: Standard loading with trust_remote_code
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        config=config,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=dtype,
                    ),

                    # Method 2: Loading with specific mistral format settings
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        quantization_config=quantization_config,
                        device_map="auto", 
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        load_format="mistral",
                        tokenizer_mode="mistral",
                    )
                ]

                # Try each loading method until one succeeds
                last_error = None
                for i, load_method in enumerate(loading_methods):
                    try:
                        logger.info(f"Trying model loading method {i+1}...")
                        self.model = load_method()
                        logger.info(f"Model loaded successfully using method {i+1}")
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Method {i+1} failed: {e}")

                # If all methods failed, try a last resort approach
                if self.model is None:
                    logger.warning("All standard loading methods failed, trying last resort approach")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        revision="main",
                        use_safetensors=True,
                        offload_folder="offload"
                    )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model '{self.config.model_id}'. Original error: {e}")

        # Create generator pipeline with appropriate task based on architecture
        pipeline_task = "text-generation"
        if getattr(self.config, 'model_architecture', 'causal_lm') == 'seq2seq_lm':
            pipeline_task = "text2text-generation"
            logger.info(f"Using text2text-generation pipeline for sequence-to-sequence model")
        else:
            logger.info(f"Using text-generation pipeline for causal language model")

        # Create pipeline without generation parameters (they should be passed during inference)
        suppress_transformers_warnings()  # Suppress warnings during pipeline creation
        
        self.generator = pipeline(
            pipeline_task,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        loading_time = time.time() - start_time
        logger.info(f"Model loaded in {loading_time:.2f} seconds")

    def unload_model(self) -> None:
        """Unload the model from memory and free resources."""
        logger.info(f"Unloading model: {self.config.model_id}")
        
        try:
            # Clear pipeline first
            if hasattr(self, 'generator') and self.generator is not None:
                del self.generator
                self.generator = None
            
            # Move model to CPU and clear from GPU
            if hasattr(self, 'model') and self.model is not None:
                # Move to CPU first to free GPU memory
                self.model.cpu()
                # Delete the model
                del self.model
                self.model = None
            
            # Clear tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Model {self.config.model_id} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading model {self.config.model_id}: {e}")
            # Even if there's an error, try to clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_content_filter(self, content_filter):
        """Inject content filter service to avoid circular imports."""
        self.content_filter = content_filter

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        if not self.model or not self.tokenizer:
            self.load_model()

        # Override config parameters with kwargs
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        do_sample = kwargs.get("do_sample", self.config.do_sample)

        # Merge with custom generation params if available
        generation_params = {}
        if self.config.custom_generation_params:
            generation_params.update(self.config.custom_generation_params)

        # Override with method kwargs
        generation_params.update({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        })

        # Filter out unsupported parameters for transformers pipeline
        # Supported parameters for text-generation pipeline
        supported_params = {
            'max_new_tokens', 'max_length', 'min_length', 'temperature', 
            'top_p', 'repetition_penalty', 'length_penalty', 'num_beams', 
            'early_stopping', 'do_sample', 'pad_token_id', 'eos_token_id', 
            'use_cache', 'num_return_sequences'
        }

        # Filter parameters and log warnings for unsupported ones
        # Filter out unsupported parameters for transformers pipeline
        # Comprehensive list of supported parameters for text-generation pipeline
        supported_params = {
            'max_new_tokens', 'max_length', 'min_length', 'temperature', 
            'top_p', 'repetition_penalty', 'length_penalty', 'num_beams', 
            'early_stopping', 'do_sample', 'pad_token_id', 'eos_token_id', 
            'use_cache', 'num_return_sequences'
        }

        # Special handling for enable_thinking - not passed to generation but used in chat template
        enable_thinking = generation_params.get('enable_thinking', False)
        
        filtered_params = {}
        unsupported_params = []
        for param, value in generation_params.items():
            if param in supported_params:
                # Additional model-specific filtering
                if param == 'temperature' and value == 0:
                    # Some models don't handle temperature=0 well, use a small value
                    filtered_params[param] = 0.01
                elif param == 'do_sample' and value is False and 'temperature' in generation_params:
                    # When do_sample=False, temperature should be ignored
                    continue
                else:
                    filtered_params[param] = value
            elif param == 'enable_thinking':
                # This parameter is handled separately for chat template formatting
                continue
            else:
                unsupported_params.append(param)
                logger.debug(f"Ignoring unsupported generation parameter: {param}={value}")
        
        # Only log warnings for unexpected unsupported parameters
        if unsupported_params:
            # Common parameters that users might try but aren't supported by transformers
            common_unsupported = {'top_k', 'presence_penalty', 'frequency_penalty', 'logit_bias', 'stream'}
            unexpected_unsupported = set(unsupported_params) - common_unsupported
            
            if unexpected_unsupported:
                logger.warning(f"Ignored unexpected unsupported generation parameters for Transformers engine: {list(unexpected_unsupported)}. "
                              f"Supported parameters: {sorted(supported_params)}")
            
            # Only log info for commonly unsupported params if in debug mode
            if common_unsupported.intersection(set(unsupported_params)):
                logger.debug(f"Note: Parameters {list(common_unsupported.intersection(set(unsupported_params)))} are not supported by Transformers engine (use vLLM for these)")

        architecture = getattr(self.config, 'model_architecture', 'causal_lm')

        # Handle differently based on model architecture
        if architecture == 'seq2seq_lm':
            # For seq2seq models, just use the prompt directly
            formatted_prompt = prompt

            # Generate response with warning suppression
            suppress_transformers_warnings()
            response = self.generator(
                formatted_prompt,
                **filtered_params
            )[0]["generated_text"]

            return response
        else:
            # For causal LMs, format with chat template
            try:
                # Try using chat template first with enable_thinking from custom params
                enable_thinking = False
                if self.config.custom_generation_params:
                    enable_thinking = self.config.custom_generation_params.get('enable_thinking', False)
                
                if enable_thinking is not None:
                    formatted_prompt = self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt}
                    ], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt}
                    ], tokenize=False, add_generation_prompt=True)
            except (AttributeError, NotImplementedError, TypeError):
                # Fallback for models without chat templates or that don't support enable_thinking
                formatted_prompt = f"User: {prompt}\nAssistant:"

            # Generate response
            generated_output = self.generator(
                formatted_prompt,
                **filtered_params
            )
            
            # Ensure we get the generated text as a string
            if isinstance(generated_output, list) and len(generated_output) > 0:
                response = generated_output[0].get("generated_text", "")
            else:
                response = str(generated_output)
            
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)

            # Extract assistant's reply
            if "[/INST]" in response:
                assistant_reply = response.split("[/INST]")[-1].strip()
            elif "Assistant:" in response:
                assistant_reply = response.split("Assistant:")[-1].strip()
            else:
                # For models that just continue the text
                assistant_reply = response[len(formatted_prompt):].strip()

            # Ensure the result is a string
            if not isinstance(assistant_reply, str):
                assistant_reply = str(assistant_reply)

            return assistant_reply

    def _format_conversation(self, messages: List[Dict[str, str]], enable_thinking: bool = None) -> str:
        """Format a conversation for the model."""
        architecture = getattr(self.config, 'model_architecture', 'causal_lm')

        if architecture == 'seq2seq_lm':
            # For seq2seq models, just use the last user message as the input
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return msg.get("content", "")
            return ""  # Fallback to empty if no user message found
        else:
            # For causal LMs, use the chat template
            try:
                # Use enable_thinking from custom_generation_params if not explicitly provided
                if enable_thinking is None and self.config.custom_generation_params:
                    enable_thinking = self.config.custom_generation_params.get('enable_thinking', False)
                
                # Apply chat template with enable_thinking parameter for Qwen3 models
                if enable_thinking is not None:
                    return self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking
                    )
                else:
                    return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (AttributeError, NotImplementedError, TypeError):
                # Fallback for models without chat templates or that don't support enable_thinking
                formatted = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted += f"User: {content}\n"
                    elif role == "assistant":
                        formatted += f"Assistant: {content}\n"
                    else:
                        formatted += f"{role}: {content}\n"
                formatted += "Assistant:"
                return formatted

    def _extract_last_reply(self, response: str, formatted_prompt: str) -> str:
        """Extract the last reply from a conversation."""
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)
        if not isinstance(formatted_prompt, str):
            formatted_prompt = str(formatted_prompt)
            
        architecture = getattr(self.config, 'model_architecture', 'causal_lm')

        if architecture == 'seq2seq_lm':
            # For seq2seq models, just return the entire response
            return response

        # For causal LMs
        if "[/INST]" in response:
            parts = response.split("[/INST]")
            if len(parts) > 1:
                last_part = parts[-1]
                last_exchanges = last_part.split("<s>")
                assistant_reply = last_exchanges[-1].strip()
                if assistant_reply.startswith("[ASST]"):
                    assistant_reply = assistant_reply[len("[ASST]"):].strip()
                return assistant_reply
        elif "Assistant:" in response:
            parts = response.split("Assistant:")
            if len(parts) > 1:
                return parts[-1].strip()

        # Fallback: return everything after the prompt
        fallback_result = response[len(formatted_prompt):].strip() if response.startswith(formatted_prompt) else response
        return fallback_result

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        if not self.model or not self.tokenizer:
            self.load_model()

        # Extract the last user message for content filtering
        last_user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                last_user_message = message.get('content', '')
                break
        
        # Apply content filtering if available
        if last_user_message and self.content_filter:
            filter_result = self.content_filter.filter_content(last_user_message)
            if not filter_result.is_allowed:
                logger.warning(f"Content filtered: {filter_result.filter_reason}")
                logger.debug(f"Filtered content: {last_user_message[:100]}...")
                return filter_result.suggested_response or "I apologize, but I cannot assist with that request."

        # Format conversation  
        enable_thinking = generation_params.get('enable_thinking', False)
        formatted_prompt = self._format_conversation(messages, enable_thinking)

        # Merge parameters from config and kwargs
        generation_params = {}
        if self.config.custom_generation_params:
            generation_params.update(self.config.custom_generation_params)

        # Override with method kwargs
        generation_params.update({
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        })

        # Filter out unsupported parameters for transformers pipeline
        # Comprehensive list of supported parameters for text-generation pipeline
        supported_params = {
            'max_new_tokens', 'max_length', 'min_length', 'temperature', 
            'top_p', 'repetition_penalty', 'length_penalty', 'num_beams', 
            'early_stopping', 'do_sample', 'pad_token_id', 'eos_token_id', 
            'use_cache', 'num_return_sequences'
        }

        filtered_params = {}
        unsupported_params = []
        for param, value in generation_params.items():
            if param in supported_params:
                # Additional model-specific filtering
                if param == 'temperature' and value == 0:
                    # Some models don't handle temperature=0 well, use a small value
                    filtered_params[param] = 0.01
                elif param == 'do_sample' and value is False and 'temperature' in generation_params:
                    # When do_sample=False, temperature should be ignored
                    continue
                else:
                    filtered_params[param] = value
            else:
                unsupported_params.append(param)
                logger.debug(f"Ignoring unsupported generation parameter: {param}={value}")
        
        # Only log warnings for unexpected unsupported parameters
        if unsupported_params:
            # Common parameters that users might try but aren't supported by transformers
            common_unsupported = {'top_k', 'presence_penalty', 'frequency_penalty', 'logit_bias', 'stream'}
            unexpected_unsupported = set(unsupported_params) - common_unsupported
            
            if unexpected_unsupported:
                logger.warning(f"Ignored unexpected unsupported generation parameters for Transformers engine: {list(unexpected_unsupported)}. "
                              f"Supported parameters: {sorted(supported_params)}")
            
            # Only log info for commonly unsupported params if in debug mode
            if common_unsupported.intersection(set(unsupported_params)):
                logger.debug(f"Note: Parameters {list(common_unsupported.intersection(set(unsupported_params)))} are not supported by Transformers engine (use vLLM for these)")

        # Generate response with warning suppression
        suppress_transformers_warnings()
        generated_output = self.generator(
            formatted_prompt,
            **filtered_params
        )
        
        # Ensure we get the generated text as a string
        if isinstance(generated_output, list) and len(generated_output) > 0:
            response = generated_output[0].get("generated_text", "")
        else:
            response = str(generated_output)
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)

        # Extract assistant's reply
        extracted_reply = self._extract_last_reply(response, formatted_prompt)
        
        # Ensure the final result is a string
        if not isinstance(extracted_reply, str):
            extracted_reply = str(extracted_reply)
            
        return extracted_reply

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Stream a chat response token by token."""
        if not self.model or not self.tokenizer:
            self.load_model()

        # Format conversation
        enable_thinking = False
        if self.config.custom_generation_params:
            enable_thinking = self.config.custom_generation_params.get('enable_thinking', False)
        formatted_prompt = self._format_conversation(messages, enable_thinking)
        architecture = getattr(self.config, 'model_architecture', 'causal_lm')

        # Create inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # For seq2seq models, we need to handle the streaming differently
        if architecture == 'seq2seq_lm':
            # Simple implementation for seq2seq (non-streaming)
            logger.info("Streaming not fully supported for sequence-to-sequence models")

            # Generate directly
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=kwargs.get("do_sample", self.config.do_sample),
            )

            # Decode the output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Call the callback with the entire output if provided
            if callback:
                callback(decoded_output)

            return decoded_output

        # Create streamer for causal LMs
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Setup generation parameters
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "streamer": streamer,
        }

        # Generate in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Collect response
        collected_response = ""
        for token in streamer:
            collected_response += token
            if callback:
                callback(token)

        return collected_response

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run a benchmark on the model."""
        if not self.model or not self.tokenizer:
            self.load_model()

        architecture = getattr(self.config, 'model_architecture', 'causal_lm')

        # Prepare inputs
        if architecture == 'seq2seq_lm':
            formatted_prompt = prompt
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_len = inputs.input_ids.shape[1]

        # Timing parameters
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)

        # Warmup run
        with torch.no_grad():
            _ = self.model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False
            )

        # Actual benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            output = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False  # For consistent benchmarking
            )

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate stats
        output_len = output.shape[1] - input_len
        inference_time = end_time - start_time
        tokens_per_second = output_len / inference_time

        return {
            "input_tokens": input_len,
            "output_tokens": output_len,
            "inference_time": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_stats": self.get_memory_stats(),
            "model_architecture": architecture
        }

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return get_gpu_memory_stats()

    def _setup_download_optimization(self) -> None:
        """Setup environment variables for optimized downloads."""
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HUB_PARALLEL_DOWNLOADS", "4")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
        
        # Install hf_transfer if not available
        try:
            import hf_transfer
        except ImportError:
            logger.info("Installing hf_transfer for faster downloads...")
            import subprocess
            subprocess.check_call(["pip", "install", "hf_transfer"])
            logger.info("hf_transfer installed successfully")

    def _pre_download_model(self, model_id: str) -> bool:
        """
        Pre-download model files using optimized settings.
        
        Args:
            model_id: The model identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from huggingface_hub import snapshot_download
            import time
            
            logger.info(f"Pre-downloading model files for {model_id}...")
            start_time = time.time()
            
            # Setup optimization before download
            self._setup_download_optimization()
            
            # Download with optimized settings
            snapshot_download(
                model_id,
                cache_dir=None,  # Use default cache
                resume_download=True,
                local_files_only=False,
                use_auth_token=True if os.getenv("HUGGING_FACE_HUB_TOKEN") else None
            )
            
            end_time = time.time()
            logger.info(f"Pre-download completed in {end_time - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.warning(f"Pre-download failed: {e}")
            return False

    def _check_model_cached(self, model_id: str) -> bool:
        """
        Check if model is already fully cached.
        
        Args:
            model_id: The model identifier
            
        Returns:
            bool: True if model is cached, False otherwise
        """
        try:
            from transformers.utils import is_offline_mode
            from transformers import AutoConfig
            
            # Try to load config locally only
            AutoConfig.from_pretrained(model_id, local_files_only=True)
            return True
        except Exception:
            return False
