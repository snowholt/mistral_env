"""
Factory for creating model instances based on configuration.
"""
import logging
from typing import Optional

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..inference_engines.transformers_engine import TransformersEngine
from ..inference_engines.vllm_engine import VLLMEngine

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_config: ModelConfig) -> ModelInterface:
        """Create a model instance based on the provided configuration."""
        engine_type = model_config.engine_type.lower()
        model_id = model_config.model_id.lower()
        
        # Detect model architecture type for special handling
        model_architecture = "causal_lm"  # Default assumption
        
        # Check for sequence-to-sequence models
        if "t5" in model_id or "byt5" in model_id or "bart" in model_id or "pegasus" in model_id:
            model_architecture = "seq2seq_lm"
            logger.info(f"Detected a sequence-to-sequence model: {model_config.model_id}")
            
        # Update model config with detected architecture
        model_config.model_architecture = model_architecture
        
        # Special case for Mistral3 models
        if "mistral" in model_id and "3" in model_id:
            logger.info(f"Detected a Mistral3 model: {model_config.model_id}")
            logger.info("These models work best with vLLM. Attempting to use VLLMEngine.")
            try:
                return VLLMEngine(model_config)
            except ImportError:
                logger.warning("vLLM not available for Mistral3 model, trying transformers engine")
                return TransformersEngine(model_config)
                
        # Special case for Qwen3 models
        elif "qwen" in model_id and "3" in model_id:
            logger.info(f"Detected a Qwen3 model: {model_config.model_id}")
            logger.info("Using transformers engine for Qwen3 model with optimizations.")
            return TransformersEngine(model_config)
        
        # Normal cases
        if engine_type == "transformers":
            logger.info(f"Creating TransformersEngine for model: {model_config.model_id}")
            return TransformersEngine(model_config)
        
        elif engine_type == "vllm":
            # vLLM only supports causal LMs, so warn if trying to use with seq2seq
            if model_architecture == "seq2seq_lm":
                logger.warning(f"vLLM does not support sequence-to-sequence models like {model_config.model_id}. Falling back to TransformersEngine.")
                return TransformersEngine(model_config)
                
            logger.info(f"Creating VLLMEngine for model: {model_config.model_id}")
            try:
                return VLLMEngine(model_config)
            except ImportError:
                logger.warning("vLLM not available, falling back to TransformersEngine")
                return TransformersEngine(model_config)
        
        else:
            logger.warning(f"Unknown engine type: {engine_type}, using TransformersEngine")
            return TransformersEngine(model_config)
