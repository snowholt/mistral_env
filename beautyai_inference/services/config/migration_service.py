"""
Configuration migration service for BeautyAI.

This service provides configuration format migration capabilities:
- Detecting configuration format compatibility issues
- Migrating configuration between different versions
- Handling legacy configuration parameters
- Ensuring backward compatibility during upgrades
"""
import logging
import argparse
from typing import Dict, Any, List, Tuple

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class MigrationService(BaseService):
    """Service for migrating configuration between different format versions.
    
    Handles migration of configuration files from older formats to current
    standards, ensuring compatibility and fixing deprecated parameters.
    """
    
    def __init__(self):
        super().__init__()
    
    def migrate_config(self, app_config: AppConfig, backup_service=None) -> Dict[str, Any]:
        """
        Migrate configuration to the current format standard.
        
        This method checks for compatibility issues in the configuration,
        optionally creates a backup, performs necessary migrations, and
        returns the migration results.
        
        Args:
            app_config: The application configuration to migrate
            backup_service: Optional backup service for creating pre-migration backup
            
        Returns:
            Dict with migration results containing 'success', 'issues_fixed', and 'config'
        """
        try:
            # Check if current configuration format needs migration
            config_dict = app_config.get_config()
            needs_migration, migration_issues = self._check_migration_needed(config_dict, app_config)
            
            if not needs_migration:
                return {
                    "success": True,
                    "needs_migration": False,
                    "issues_fixed": [],
                    "config": config_dict,
                    "message": "Configuration format is up to date. No migration needed."
                }
            
            # Perform migration steps
            migrated_config = self._migrate_config_format(config_dict, app_config)
            
            # Update app_config with migrated data
            app_config.update_from_dict(migrated_config)
            
            return {
                "success": True,
                "needs_migration": True,
                "issues_fixed": migration_issues,
                "config": migrated_config,
                "message": f"Configuration successfully migrated. Fixed {len(migration_issues)} issues."
            }
            
        except Exception as e:
            logger.exception("Failed to migrate configuration")
            return {
                "success": False,
                "needs_migration": True,
                "issues_fixed": [],
                "config": config_dict,
                "message": f"Migration failed: {str(e)}"
            }
    
    def check_migration_needed(self, app_config: AppConfig) -> Tuple[bool, List[str]]:
        """
        Check if configuration needs migration to new format.
        
        Args:
            app_config: The application configuration to check
            
        Returns:
            Tuple[bool, List[str]]: (needs_migration, list of migration issues)
        """
        config_dict = app_config.get_config()
        return self._check_migration_needed(config_dict, app_config)
    
    def _check_migration_needed(self, config: Dict[str, Any], app_config: AppConfig) -> Tuple[bool, List[str]]:
        """
        Check if configuration needs migration to new format.
        
        This method identifies specific configuration issues that require 
        migration to the current format standard.
        
        Args:
            config: Current configuration dictionary
            app_config: The application configuration object
            
        Returns:
            Tuple[bool, List[str]]: (needs_migration, list of migration issues)
        """
        migration_issues = []
        
        # Check for old-style configuration keys
        if "engines" in config and "preferred_engine" not in config:
            migration_issues.append("Convert 'engines' list to 'preferred_engine'")
        
        # Check for deprecated model configuration keys
        model_config = config.get("model", {})
        if "quantized" in model_config and "quantization" not in model_config:
            migration_issues.append("Convert boolean 'quantized' to string 'quantization'")
            
        # Check for missing fields that are now required
        if "model_architecture" not in model_config:
            migration_issues.append("Add 'model_architecture' field to model configuration")
            
        # Check for legacy engine names
        if model_config.get("engine_type") == "transformer":
            migration_issues.append("Update 'transformer' to 'transformers'")
            
        # Check for deprecated parameters
        if "max_length" in model_config and "max_new_tokens" not in model_config:
            migration_issues.append("Convert 'max_length' to 'max_new_tokens'")
            
        # Check model registry format as well
        models = app_config.model_registry.models
        for name, model in models.items():
            model_dict = model.to_dict()
            
            # Check for missing model architecture
            if not model_dict.get("model_architecture"):
                migration_issues.append(f"Add 'model_architecture' to model '{name}'")
                
            # Check for deprecated engine types
            if model_dict.get("engine_type") == "transformer":
                migration_issues.append(f"Update engine type for model '{name}' (transformer → transformers)")
                
            # Check for invalid quantization settings
            if model_dict.get("engine_type") == "vllm" and model_dict.get("quantization") in ["4bit", "8bit"]:
                migration_issues.append(f"Update incompatible quantization for model '{name}'")
                
        return len(migration_issues) > 0, migration_issues
    
    def _migrate_config_format(self, config: Dict[str, Any], app_config: AppConfig) -> Dict[str, Any]:
        """
        Migrate configuration to current format.
        
        This method performs various transformations to update configuration
        from older formats to the current standard format.
        
        Args:
            config: Current configuration dictionary
            app_config: The application configuration object
            
        Returns:
            Dict[str, Any]: Migrated configuration
        """
        migrated = config.copy()
        
        # Update root level configuration
        if "engines" in migrated and "preferred_engine" not in migrated:
            engines = migrated.pop("engines")
            if engines and isinstance(engines, list) and len(engines) > 0:
                migrated["preferred_engine"] = engines[0]
        
        # Update model configuration
        if "model" in migrated:
            model = migrated["model"].copy()
            
            # Convert legacy quantization format
            if "quantized" in model and "quantization" not in model:
                quantized = model.pop("quantized")
                model["quantization"] = "4bit" if quantized else None
            
            # Add model_architecture if missing
            if "model_architecture" not in model:
                # Attempt to infer from model_id or default to causal_lm
                model_id = model.get("model_id", "").lower()
                if any(t5_name in model_id for t5_name in ["t5", "flan-t5", "mt5", "ul2"]):
                    model["model_architecture"] = "seq2seq_lm"
                else:
                    model["model_architecture"] = "causal_lm"
                    
            # Fix legacy engine name
            if model.get("engine_type") == "transformer":
                model["engine_type"] = "transformers"
                
            # Convert legacy max_length to max_new_tokens
            if "max_length" in model and "max_new_tokens" not in model:
                model["max_new_tokens"] = model.pop("max_length")
                
            # Ensure correct dtype values
            if "dtype" in model and model["dtype"] not in ["float16", "float32", "bfloat16"]:
                if model["dtype"] == "fp16":
                    model["dtype"] = "float16"
                elif model["dtype"] == "fp32":
                    model["dtype"] = "float32"
                elif model["dtype"] == "bf16":
                    model["dtype"] = "bfloat16"
                else:
                    model["dtype"] = "float16"  # Default to float16 if unknown
                    
            migrated["model"] = model
        
        # Migrate model registry
        if hasattr(app_config, "model_registry") and app_config.model_registry:
            for name, model in app_config.model_registry.models.items():
                model_dict = model.to_dict()
                
                # Add model architecture if missing
                if not model_dict.get("model_architecture"):
                    # Try to infer from model_id
                    model_id = model_dict.get("model_id", "").lower()
                    if any(t5_name in model_id for t5_name in ["t5", "flan-t5", "mt5", "ul2"]):
                        model.model_architecture = "seq2seq_lm"
                    else:
                        model.model_architecture = "causal_lm"
                
                # Fix legacy engine name
                if model_dict.get("engine_type") == "transformer":
                    model.engine_type = "transformers"
                    
                # Fix incompatible quantization settings
                if model_dict.get("engine_type") == "vllm" and model_dict.get("quantization") in ["4bit", "8bit"]:
                    if model_dict.get("quantization") == "4bit":
                        model.quantization = "awq"
                    else:
                        model.quantization = "none"
        
        return migrated
