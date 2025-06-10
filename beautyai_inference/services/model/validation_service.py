"""
Model configuration validation service.

This service handles:
- Validating model configurations before adding to registry
- Checking model compatibility with backends
- Validating model parameters and settings
- Architecture-specific validation rules
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..base.base_service import BaseService
from ...config.config_manager import ModelConfig

logger = logging.getLogger(__name__)


class ModelValidationService(BaseService):
    """Service for validating model configurations."""
    
    # Supported engines and their capabilities
    SUPPORTED_ENGINES = {
        'transformers': {
            'architectures': ['causal_lm', 'seq2seq'],
            'quantization': ['4bit', '8bit', 'none'],
            'dtypes': ['float16', 'bfloat16', 'float32']
        },
        'vllm': {
            'architectures': ['causal_lm'],  # vLLM only supports causal LM
            'quantization': ['awq', 'squeezellm', 'none'],
            'dtypes': ['float16', 'bfloat16']
        },
        'llama.cpp': {
            'architectures': ['causal_lm'],  # llama.cpp only supports causal LM
            'quantization': ['Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_K', 'none'],
            'dtypes': ['float16', 'float32']
        }
    }
    
    # Model ID patterns for different providers
    MODEL_ID_PATTERNS = {
        'huggingface': r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$',
        'local': r'^(/|\.\.?/|~/).*',
        'simple': r'^[a-zA-Z0-9_.-]+$'
    }
    
    def __init__(self):
        super().__init__()
    
    def validate_model_config(self, model_config: ModelConfig) -> Tuple[bool, List[str]]:
        """
        Validate a complete model configuration.
        
        Args:
            model_config: The model configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic field validation
        basic_errors = self._validate_basic_fields(model_config)
        errors.extend(basic_errors)
        
        # Engine compatibility validation
        engine_errors = self._validate_engine_compatibility(model_config)
        errors.extend(engine_errors)
        
        # Model ID format validation
        id_errors = self._validate_model_id(model_config.model_id)
        errors.extend(id_errors)
        
        # Architecture validation
        arch_errors = self._validate_architecture(model_config)
        errors.extend(arch_errors)
        
        # Generation parameters validation
        param_errors = self._validate_generation_params(model_config)
        errors.extend(param_errors)
        
        return len(errors) == 0, errors
    
    def validate_engine_compatibility(self, engine_type: str, model_architecture: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that an engine supports a specific model architecture.
        
        Args:
            engine_type: The inference engine (transformers, vllm)
            model_architecture: The model architecture (causal_lm, seq2seq)
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        if engine_type not in self.SUPPORTED_ENGINES:
            return False, f"Unsupported engine: {engine_type}"
        
        engine_info = self.SUPPORTED_ENGINES[engine_type]
        if model_architecture not in engine_info['architectures']:
            return False, f"Engine '{engine_type}' does not support '{model_architecture}' architecture"
        
        return True, None
    
    def validate_quantization_config(self, engine_type: str, quantization: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate quantization configuration for a specific engine.
        
        Args:
            engine_type: The inference engine
            quantization: The quantization method
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not quantization or quantization == 'none':
            return True, None
        
        if engine_type not in self.SUPPORTED_ENGINES:
            return False, f"Unsupported engine: {engine_type}"
        
        engine_info = self.SUPPORTED_ENGINES[engine_type]
        if quantization not in engine_info['quantization']:
            supported = ', '.join(engine_info['quantization'])
            return False, f"Engine '{engine_type}' does not support '{quantization}' quantization. Supported: {supported}"
        
        return True, None
    
    def suggest_engine_fallback(self, model_architecture: str, preferred_engine: str) -> Optional[str]:
        """
        Suggest an alternative engine if the preferred one is incompatible.
        
        Args:
            model_architecture: The model architecture
            preferred_engine: The originally requested engine
            
        Returns:
            Alternative engine name or None if no fallback available
        """
        for engine, info in self.SUPPORTED_ENGINES.items():
            if engine != preferred_engine and model_architecture in info['architectures']:
                return engine
        return None
    
    def _validate_basic_fields(self, model_config: ModelConfig) -> List[str]:
        """Validate basic required fields."""
        errors = []
        
        if not model_config.name or not model_config.name.strip():
            errors.append("Model name is required and cannot be empty")
        
        if not model_config.model_id or not model_config.model_id.strip():
            errors.append("Model ID is required and cannot be empty")
        
        if not model_config.engine_type:
            errors.append("Engine type is required")
        
        return errors
    
    def _validate_engine_compatibility(self, model_config: ModelConfig) -> List[str]:
        """Validate engine and architecture compatibility."""
        errors = []
        
        # Check if engine is supported
        if model_config.engine_type not in self.SUPPORTED_ENGINES:
            supported_engines = ', '.join(self.SUPPORTED_ENGINES.keys())
            errors.append(f"Unsupported engine '{model_config.engine_type}'. Supported: {supported_engines}")
            return errors  # Skip further validation if engine is unsupported
        
        # Check architecture compatibility
        is_compatible, error_msg = self.validate_engine_compatibility(
            model_config.engine_type, 
            model_config.model_architecture
        )
        if not is_compatible:
            # Suggest fallback
            fallback = self.suggest_engine_fallback(model_config.model_architecture, model_config.engine_type)
            if fallback:
                error_msg += f". Consider using '{fallback}' engine instead."
            errors.append(error_msg)
        
        # Check quantization compatibility
        is_valid, quant_error = self.validate_quantization_config(
            model_config.engine_type, 
            model_config.quantization
        )
        if not is_valid:
            errors.append(quant_error)
        
        # Check dtype compatibility
        engine_info = self.SUPPORTED_ENGINES[model_config.engine_type]
        if model_config.dtype and model_config.dtype not in engine_info['dtypes']:
            supported_dtypes = ', '.join(engine_info['dtypes'])
            errors.append(f"Engine '{model_config.engine_type}' does not support dtype '{model_config.dtype}'. Supported: {supported_dtypes}")
        
        return errors
    
    def _validate_model_id(self, model_id: str) -> List[str]:
        """Validate model ID format."""
        errors = []
        
        # Check if it matches any known pattern
        import re
        valid_pattern = False
        
        for pattern_name, pattern in self.MODEL_ID_PATTERNS.items():
            if re.match(pattern, model_id):
                valid_pattern = True
                break
        
        if not valid_pattern:
            errors.append(f"Invalid model ID format: '{model_id}'. Expected formats: HuggingFace (owner/model), local path, or simple name")
        
        return errors
    
    def _validate_architecture(self, model_config: ModelConfig) -> List[str]:
        """Validate model architecture."""
        errors = []
        
        valid_architectures = set()
        for engine_info in self.SUPPORTED_ENGINES.values():
            valid_architectures.update(engine_info['architectures'])
        
        if model_config.model_architecture not in valid_architectures:
            supported = ', '.join(sorted(valid_architectures))
            errors.append(f"Unsupported model architecture '{model_config.model_architecture}'. Supported: {supported}")
        
        return errors
    
    def _validate_generation_params(self, model_config: ModelConfig) -> List[str]:
        """Validate custom generation parameters."""
        errors = []
        
        if not model_config.custom_generation_params:
            return errors
        
        # Validate common generation parameters
        valid_params = {
            'max_new_tokens', 'max_length', 'min_length', 'temperature', 
            'top_k', 'top_p', 'repetition_penalty', 'length_penalty',
            'num_beams', 'early_stopping', 'do_sample', 'pad_token_id',
            'eos_token_id', 'use_cache', 'enable_thinking'
        }
        
        for param_name, param_value in model_config.custom_generation_params.items():
            if param_name not in valid_params:
                logger.warning(f"Unknown generation parameter: {param_name}")
            
            # Type validation for numeric parameters
            numeric_params = {'max_new_tokens', 'max_length', 'min_length', 'top_k', 'num_beams'}
            if param_name in numeric_params and not isinstance(param_value, int):
                errors.append(f"Parameter '{param_name}' must be an integer, got {type(param_value).__name__}")
            
            float_params = {'temperature', 'top_p', 'repetition_penalty', 'length_penalty'}
            if param_name in float_params and not isinstance(param_value, (int, float)):
                errors.append(f"Parameter '{param_name}' must be a number, got {type(param_value).__name__}")
            
            bool_params = {'do_sample', 'early_stopping', 'use_cache', 'enable_thinking'}
            if param_name in bool_params and not isinstance(param_value, bool):
                errors.append(f"Parameter '{param_name}' must be a boolean, got {type(param_value).__name__}")
            
            # Value range validation
            if param_name == 'temperature' and (param_value < 0 or param_value > 2):
                errors.append("Temperature must be between 0 and 2")
            
            if param_name == 'top_p' and (param_value < 0 or param_value > 1):
                errors.append("top_p must be between 0 and 1")
        
        return errors
