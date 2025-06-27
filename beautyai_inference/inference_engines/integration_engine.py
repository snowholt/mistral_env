"""
Integration Engine for BeautyAI Framework.
Combines multiple inference engines for enhanced flexibility and performance.
"""

import logging
from typing import List, Dict, Any

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from .transformers_engine import TransformersEngine
from .vllm_engine import VLLMEngine
from .oute_tts_engine import OuteTTSEngine

logger = logging.getLogger(__name__)

class IntegrationEngine(ModelInterface):
    """Integration Engine combining Transformers, vLLM, and OuteTTS engines."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the engine with a model configuration."""
        self.config = model_config
        self.transformers_engine = None
        self.vllm_engine = None
        self.oute_tts_engine = None

        # Initialize sub-engines based on configuration
        if model_config.engine_type == "transformers":
            self.transformers_engine = TransformersEngine(model_config)
        elif model_config.engine_type == "vllm":
            self.vllm_engine = VLLMEngine(model_config)
        elif model_config.engine_type == "oute_tts":
            self.oute_tts_engine = OuteTTSEngine(model_config)
        else:
            raise ValueError(f"Unsupported engine type: {model_config.engine_type}")

    def load_model(self) -> None:
        """Load the model into memory."""
        if self.transformers_engine:
            self.transformers_engine.load_model()
        if self.vllm_engine:
            self.vllm_engine.load_model()
        if self.oute_tts_engine:
            self.oute_tts_engine.load_model()

    def unload_model(self) -> None:
        """Unload the model from memory and free resources."""
        if self.transformers_engine:
            self.transformers_engine.unload_model()
        if self.vllm_engine:
            self.vllm_engine.unload_model()
        if self.oute_tts_engine:
            self.oute_tts_engine.unload_model()

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt or speech from text."""
        if self.transformers_engine:
            return self.transformers_engine.generate(prompt, **kwargs)
        if self.vllm_engine:
            return self.vllm_engine.generate(prompt, **kwargs)
        if self.oute_tts_engine:
            return self.oute_tts_engine.generate(prompt, **kwargs)
        raise RuntimeError("No engine available for generation.")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        if self.transformers_engine:
            return self.transformers_engine.chat(messages, **kwargs)
        if self.vllm_engine:
            return self.vllm_engine.chat(messages, **kwargs)
        if self.oute_tts_engine:
            return self.oute_tts_engine.chat(messages, **kwargs)
        raise RuntimeError("No engine available for chat.")

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run a benchmark on the model."""
        if self.transformers_engine:
            return self.transformers_engine.benchmark(prompt, **kwargs)
        if self.vllm_engine:
            return self.vllm_engine.benchmark(prompt, **kwargs)
        if self.oute_tts_engine:
            return self.oute_tts_engine.benchmark(prompt, **kwargs)
        raise RuntimeError("No engine available for benchmarking.")

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Stream a chat response token by token."""
        if self.transformers_engine:
            return self.transformers_engine.chat_stream(messages, callback, **kwargs)
        if self.vllm_engine:
            return self.vllm_engine.chat_stream(messages, callback, **kwargs)
        if self.oute_tts_engine:
            return self.oute_tts_engine.chat_stream(messages, callback, **kwargs)
        raise RuntimeError("No engine available for chat streaming.")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if self.transformers_engine:
            return self.transformers_engine.get_memory_stats()
        if self.vllm_engine:
            return self.vllm_engine.get_memory_stats()
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_memory_stats()
        raise RuntimeError("No engine available for memory stats.")

    # TTS-specific methods (delegate to OuteTTS engine)
    def text_to_speech(self, text: str, **kwargs) -> str:
        """Convert text to speech (only available with OuteTTS engine)."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.text_to_speech(text, **kwargs)
        raise RuntimeError("TTS functionality requires OuteTTS engine")

    def text_to_speech_bytes(self, text: str, **kwargs) -> bytes:
        """Convert text to speech and return as bytes (only available with OuteTTS engine)."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.text_to_speech_bytes(text, **kwargs)
        raise RuntimeError("TTS functionality requires OuteTTS engine")

    def text_to_speech_stream(self, text: str, **kwargs):
        """Convert text to speech and return as stream (only available with OuteTTS engine)."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.text_to_speech_stream(text, **kwargs)
        raise RuntimeError("TTS functionality requires OuteTTS engine")

    def get_supported_languages(self) -> List[str]:
        """Get supported languages (only available with OuteTTS engine)."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_supported_languages()
        return []

    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers (only available with OuteTTS engine)."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_available_speakers(language)
        return []

    def is_tts_engine(self) -> bool:
        """Check if this is a TTS engine."""
        return self.oute_tts_engine is not None

    def get_engine_type(self) -> str:
        """Get the type of the active engine."""
        if self.transformers_engine:
            return "transformers"
        elif self.vllm_engine:
            return "vllm"
        elif self.oute_tts_engine:
            return "oute_tts"
        return "unknown"
