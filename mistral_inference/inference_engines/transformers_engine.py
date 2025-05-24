"""
Inference engine implementation based on Hugging Face Transformers.
"""
import torch
import time
import logging
from threading import Thread
from typing import List, Dict, Any, Optional, Union, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TextIteratorStreamer,
)

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import get_gpu_memory_stats, time_function

logger = logging.getLogger(__name__)


class TransformersEngine(ModelInterface):
    """Inference engine implementation using Hugging Face Transformers."""
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the engine with a model configuration."""
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.generator = None
    
    def load_model(self) -> None:
        """Load the model into memory."""
        start_time = time.time()
        logger.info(f"Loading model: {self.config.model_id}")
        
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
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        
        # Create generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        loading_time = time.time() - start_time
        logger.info(f"Model loaded in {loading_time:.2f} seconds")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Override config parameters with kwargs
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        do_sample = kwargs.get("do_sample", self.config.do_sample)
        
        # Format prompt
        formatted_prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": prompt}
        ], tokenize=False)
        
        # Generate response
        response = self.generator(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )[0]["generated_text"]
        
        # Extract assistant's reply
        assistant_reply = response.split("[/INST]")[-1].strip()
        return assistant_reply
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation for the model."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def _extract_last_reply(self, response: str) -> str:
        """Extract the last reply from a conversation."""
        parts = response.split("[/INST]")
        if len(parts) > 1:
            last_part = parts[-1]
            last_exchanges = last_part.split("<s>")
            assistant_reply = last_exchanges[-1].strip()
            if assistant_reply.startswith("[ASST]"):
                assistant_reply = assistant_reply[len("[ASST]"):].strip()
            return assistant_reply
        return "Error: Could not parse the response"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Format conversation
        formatted_prompt = self._format_conversation(messages)
        
        # Generate response
        response = self.generator(
            formatted_prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            do_sample=kwargs.get("do_sample", self.config.do_sample),
        )[0]["generated_text"]
        
        # Extract assistant's reply
        return self._extract_last_reply(response)
    
    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Stream a chat response token by token."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Format conversation
        formatted_prompt = self._format_conversation(messages)
        
        # Create inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Create streamer
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
        
        # Prepare inputs
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
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return get_gpu_memory_stats()
