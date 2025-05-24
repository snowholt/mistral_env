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
        
        if engine_type == "transformers":
            logger.info(f"Creating TransformersEngine for model: {model_config.model_id}")
            return TransformersEngine(model_config)
        
        elif engine_type == "vllm":
            logger.info(f"Creating VLLMEngine for model: {model_config.model_id}")
            try:
                return VLLMEngine(model_config)
            except ImportError:
                logger.warning("vLLM not available, falling back to TransformersEngine")
                return TransformersEngine(model_config)
        
        else:
            logger.warning(f"Unknown engine type: {engine_type}, using TransformersEngine")
            return TransformersEngine(model_config)
