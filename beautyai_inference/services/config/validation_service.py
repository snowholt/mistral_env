"""
Configuration validation service for BeautyAI.

This service provides comprehensive configuration validation capabilities:
- JSON schema validation
- Engine-specific configuration compatibility checks
- Quantization method compatibility validation
- File path validation
- Cross-configuration dependency validation
"""
import logging
import os
from typing import Dict, Any, Optional, List
import jsonschema

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class ValidationService(BaseService):
    """Service for validating configuration against schema and compatibility rules.
    
    Provides comprehensive validation including schema compliance,
    engine compatibility, and cross-configuration dependency checks.
    """
    
    def __init__(self):
        super().__init__()
        self.config_schema: Dict[str, Any] = self._get_config_schema()
    
    def validate_config(self, app_config: AppConfig) -> Dict[str, Any]:
        """
        Validate configuration against schema and check for compatibility issues.
        
        This method performs comprehensive validation of both global configuration
        and model-specific configurations in the registry. It checks for:
        
        - JSON schema compliance
        - Engine-specific configuration compatibilities
        - Quantization method compatibilities
        - File path validations
        - Cross-configuration dependencies
        
        Args:
            app_config: The application configuration to validate
            
        Returns:
            Dict with validation results containing 'valid' flag and 'errors' list
        """
        try:
            # Validate main application config
            config_dict = app_config.get_config()
            validation_result = self._validate_against_schema(config_dict, self.config_schema)
            
            # Additional validation for file paths
            paths_valid = True
            path_errors = []
            
            # Check cache directory
            cache_dir = app_config.cache_dir
            if cache_dir and not os.path.isdir(cache_dir) and not os.access(os.path.dirname(cache_dir), os.W_OK):
                paths_valid = False
                path_errors.append(f"Cache directory '{cache_dir}' is not writeable")
            
            # Check models file location
            models_file = app_config.models_file
            if not os.path.exists(os.path.dirname(models_file)) and not os.access(os.path.dirname(os.path.dirname(models_file)), os.W_OK):
                paths_valid = False
                path_errors.append(f"Models file directory '{os.path.dirname(models_file)}' is not writeable")
                
            # Validate models
            models_valid = True
            model_errors = []
            
            for model_name, model_config in app_config.model_registry.models.items():
                try:
                    model_dict = model_config.to_dict()
                    model_result = self._validate_model_config(model_dict)
                    if not model_result["valid"]:
                        models_valid = False
                        model_errors.append({
                            "model": model_name,
                            "errors": model_result["errors"]
                        })
                        
                    # Additional model-specific validations
                    if model_config.engine_type == "transformers":
                        if model_config.quantization in ["awq", "squeezellm"]:
                            models_valid = False
                            model_errors.append({
                                "model": model_name,
                                "errors": [f"Quantization method '{model_config.quantization}' is only supported with vLLM engine"]
                            })
                            
                    if model_config.engine_type == "llama.cpp" and model_config.quantization in ["4bit", "8bit"]:
                        models_valid = False
                        model_errors.append({
                            "model": model_name,
                            "errors": [f"Quantization method '{model_config.quantization}' is not compatible with llama.cpp engine"]
                        })
                        
                except Exception as e:
                    models_valid = False
                    model_errors.append({
                        "model": model_name,
                        "errors": [f"Validation error: {str(e)}"]
                    })
            
            # Check if default model exists in registry
            default_model = app_config.model_registry.default_model
            if default_model and default_model not in app_config.model_registry.models:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Default model '{default_model}' not found in model registry")
            
            # Consolidate results
            overall_valid = validation_result["valid"] and models_valid and paths_valid
            all_errors = validation_result["errors"] + path_errors
            
            for model_error in model_errors:
                all_errors.extend([f"{model_error['model']}: {error}" for error in model_error["errors"]])
            
            return {
                "valid": overall_valid,
                "errors": all_errors,
                "model_errors": model_errors,
                "path_errors": path_errors,
                "summary": {
                    "total_models": len(app_config.model_registry.models),
                    "default_model": app_config.model_registry.default_model,
                    "log_level": app_config.log_level,
                    "cache_dir": app_config.cache_dir or 'Default'
                }
            }
                
        except Exception as e:
            logger.exception("Failed to validate configuration")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "model_errors": [],
                "path_errors": [],
                "summary": {}
            }
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for configuration validation."""
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string"},
                        "engine_type": {
                            "type": "string",
                            "enum": ["transformers", "vllm", "llama.cpp"]
                        },
                        "device": {"type": "string"},
                        "device_map": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "object"}
                            ]
                        },
                        "torch_dtype": {
                            "type": "string",
                            "enum": ["auto", "float16", "bfloat16", "float32"]
                        },
                        "quantization": {
                            "type": "string",
                            "enum": ["none", "4bit", "8bit", "awq", "squeezellm", "gptq"]
                        },
                        "trust_remote_code": {"type": "boolean"},
                        "use_flash_attention": {"type": "boolean"},
                        "max_memory": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "object"}
                            ]
                        },
                        "generation_config": {
                            "type": "object",
                            "properties": {
                                "max_new_tokens": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 8192
                                },
                                "temperature": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 2.0
                                },
                                "top_p": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "top_k": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100
                                },
                                "repetition_penalty": {
                                    "type": "number",
                                    "minimum": 0.5,
                                    "maximum": 2.0
                                },
                                "do_sample": {"type": "boolean"},
                                "use_cache": {"type": "boolean"}
                            },
                            "additionalProperties": False
                        },
                        "vllm_config": {
                            "type": "object",
                            "properties": {
                                "tensor_parallel_size": {
                                    "type": "integer",
                                    "minimum": 1
                                },
                                "gpu_memory_utilization": {
                                    "type": "number",
                                    "minimum": 0.1,
                                    "maximum": 1.0
                                },
                                "max_model_len": {
                                    "type": "integer",
                                    "minimum": 512
                                }
                            },
                            "additionalProperties": False
                        }
                    },
                    "required": ["model_id"],
                    "additionalProperties": False
                },
                "cache_dir": {"type": "string"},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                },
                "models_file": {"type": "string"},
                "default_engine": {
                    "type": "string",
                    "enum": ["transformers", "vllm", "llama.cpp"]
                }
            },
            "additionalProperties": False
        }
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against JSON schema."""
        try:
            jsonschema.validate(config, schema)
            return {"valid": True, "errors": []}
        except jsonschema.ValidationError as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {e.message}"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a model configuration dictionary."""
        errors = []
        
        # Check required fields
        required_fields = ["model_id"]
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Missing required field: {field}")
        
        # Validate engine compatibility
        engine_type = model_config.get("engine_type", "transformers")
        quantization = model_config.get("quantization", "none")
        
        # Check engine-specific quantization compatibility
        if engine_type == "transformers" and quantization in ["awq", "squeezellm"]:
            errors.append(f"Quantization '{quantization}' not supported with transformers engine")
        
        if engine_type == "vllm" and quantization in ["4bit", "8bit"]:
            errors.append(f"Quantization '{quantization}' not supported with vLLM engine")
        
        # Validate generation config ranges
        gen_config = model_config.get("generation_config", {})
        if isinstance(gen_config, dict):
            if "temperature" in gen_config:
                temp = gen_config["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors.append("Temperature must be between 0 and 2")
            
            if "top_p" in gen_config:
                top_p = gen_config["top_p"]
                if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                    errors.append("top_p must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
