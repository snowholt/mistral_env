"""
Configuration service for unified CLI.

This service provides comprehensive configuration management capabilities including:
- Displaying configuration values
- Setting and updating configuration values
- Validating configuration against schema
- Resetting configuration to defaults 
- Migrating configuration between formats
- Backing up and restoring configuration
"""
import logging
import json
import os
import shutil
import datetime
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import jsonschema

from .base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class ConfigService(BaseService):
    """Service for managing configuration operations.
    
    Provides comprehensive configuration management including configuration
    display, setting, validation, backup, and migration.
    """
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        self.config_schema: Dict[str, Any] = self._get_config_schema()
        # Keep track of config backups
        self.backup_history: List[Dict[str, str]] = []
    
    def show_config(self, args):
        """Show the current configuration."""
        self._load_config(args)
        
        # Display global configuration
        print("\n=== Global Configuration ===")
        print(f"Config File:     {self.config_file_path or 'Default (none specified)'}")
        print(f"Models File:     {self.app_config.models_file}")
        print(f"Default Model:   {self.app_config.model_registry.default_model or 'None'}")
        print(f"Cache Directory: {self.app_config.cache_dir}")
        
        # Display all other configuration
        config_dict = self.app_config.to_dict()
        if config_dict:
            print("\n=== Current Model Configuration ===")
            model_config = config_dict.get("model", {})
            for key, value in model_config.items():
                if key not in ["name", "description"]:  # Skip metadata fields
                    print(f"{key}: {value}")
        
        print()
    
    def set_config(self, args):
        """Set a configuration value."""
        self._load_config(args)
        
        key = args.key
        value = args.value
        
        # Handle special values
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "none":
            value = None
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
            value = float(value)
        
        # Validate key
        if key in ["models", "default_model"]:
            print(f"Error: Cannot directly modify '{key}'. Use model commands instead.")
            return 1
            
        # Set the configuration
        self.app_config.set_config(key, value)
        self.app_config.save_config()
        
        print(f"Configuration '{key}' updated to '{value}'.")
    
    def reset_config(self, args):
        """Reset configuration to defaults."""
        self._load_config(args)
        
        try:
            # Create backup of current config
            config_file = self.config_file_path
            if config_file and os.path.exists(config_file):
                backup_file = f"{config_file}.backup"
                with open(config_file, "r") as f_in:
                    with open(backup_file, "w") as f_out:
                        f_out.write(f_in.read())
                print(f"Created backup of current config at {backup_file}")
            
            # Reset to defaults (but keep models)
            models = self.app_config.model_registry.models
            default_model = self.app_config.model_registry.default_model
            
            self.app_config.reset_to_defaults()
            
            # Restore models
            for name, model in models.items():
                self.app_config.add_model_config(model)
            
            if default_model:
                self.app_config.model_registry.set_default_model(default_model)
                
            self.app_config.save_config()
            self.app_config.save_model_registry()
            
            print(f"Configuration reset to defaults. Models have been preserved.")
            
        except Exception as e:
            print(f"Error resetting configuration: {str(e)}")
            logger.exception("Failed to reset configuration")
            return 1

    def validate_config(self, args):
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
            args: Command line arguments
            
        Returns:
            int: 0 for success, 1 for validation failure
        """
        self._load_config(args)
        
        try:
            # Validate main application config
            config_dict = self.app_config.get_config()
            validation_result = self._validate_against_schema(config_dict, self.config_schema)
            
            # Additional validation for file paths
            paths_valid = True
            path_errors = []
            
            # Check cache directory
            cache_dir = self.app_config.cache_dir
            if cache_dir and not os.path.isdir(cache_dir) and not os.access(os.path.dirname(cache_dir), os.W_OK):
                paths_valid = False
                path_errors.append(f"Cache directory '{cache_dir}' is not writeable")
            
            # Check models file location
            models_file = self.app_config.models_file
            if not os.path.exists(os.path.dirname(models_file)) and not os.access(os.path.dirname(os.path.dirname(models_file)), os.W_OK):
                paths_valid = False
                path_errors.append(f"Models file directory '{os.path.dirname(models_file)}' is not writeable")
                
            # Validate models
            models_valid = True
            model_errors = []
            
            for model_name, model_config in self.app_config.model_registry.models.items():
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
            default_model = self.app_config.model_registry.default_model
            if default_model and default_model not in self.app_config.model_registry.models:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Default model '{default_model}' not found in model registry")
            
            # Display validation results
            if validation_result["valid"] and models_valid and paths_valid:
                print("✅ Configuration is valid.")
                
                # Show summary information
                print(f"\n=== Configuration Summary ===")
                print(f"Total models: {len(self.app_config.model_registry.models)}")
                print(f"Default model: {self.app_config.model_registry.default_model}")
                print(f"Log level: {self.app_config.log_level}")
                print(f"Cache directory: {self.app_config.cache_dir or 'Default'}")
                
                return 0
            else:
                print("❌ Configuration has validation errors:")
                
                if not validation_result["valid"]:
                    print("\nMain configuration errors:")
                    for error in validation_result["errors"]:
                        print(f"  - {error}")
                
                if not paths_valid:
                    print("\nPath validation errors:")
                    for error in path_errors:
                        print(f"  - {error}")
                
                if not models_valid:
                    print("\nModel configuration errors:")
                    for model_error in model_errors:
                        print(f"  {model_error['model']}:")
                        for error in model_error["errors"]:
                            print(f"    - {error}")
                
                # Add suggestions for fixing errors
                print("\nSuggestions to fix errors:")
                if not validation_result["valid"]:
                    print("  - Check schema requirements in configuration files")
                    print("  - Run 'beautyai config migrate' to fix legacy configuration formats")
                if not paths_valid:
                    print("  - Ensure all directories are properly set up and have write permissions")
                if not models_valid:
                    print("  - Check model configurations for incompatible settings")
                    print("  - Run 'beautyai config validate --fix' to automatically fix minor issues")
                
                return 1
                
        except Exception as e:
            print(f"Error validating configuration: {str(e)}")
            logger.exception("Failed to validate configuration")
            return 1
    
    def backup_config(self, args):
        """
        Back up configuration files to a specified directory.
        
        This method creates timestamped backups of configuration files,
        including the main configuration and model registry files.
        It also maintains a history of backups and can create compressed
        archives of configurations if requested.
        
        Args:
            args: Command line arguments containing backup options
            
        Returns:
            int: 0 for success, 1 for failure
        """
        self._load_config(args)
        
        try:
            # Create backup directory if it doesn't exist
            backup_dir = getattr(args, 'backup_dir', "backups")
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True, parents=True)
            
            # Generate backup timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get backup label if provided
            backup_label = getattr(args, 'label', '')
            label_suffix = f"_{backup_label}" if backup_label else ""
            
            # Backup main configuration
            config_file = self.config_file_path or "default_config.json"
            config_backup = backup_path / f"config{label_suffix}_{timestamp}.json"
            
            if os.path.exists(config_file):
                shutil.copy2(config_file, config_backup)
                print(f"Configuration backed up to {config_backup}")
            else:
                print(f"Warning: Configuration file {config_file} not found, skipping backup")
            
            # Backup models configuration
            models_file = self.app_config.models_file
            models_backup = backup_path / f"models{label_suffix}_{timestamp}.json"
            
            if os.path.exists(models_file):
                shutil.copy2(models_file, models_backup)
                print(f"Models configuration backed up to {models_backup}")
            else:
                print(f"Warning: Models file {models_file} not found, skipping backup")
            
            # Add to backup history
            backup_record = {
                "timestamp": timestamp,
                "label": backup_label or "auto-backup",
                "config_file": str(config_backup) if os.path.exists(config_file) else None,
                "models_file": str(models_backup) if os.path.exists(models_file) else None
            }
            self.backup_history.append(backup_record)
            
            # Create compressed archive if requested
            if getattr(args, 'compress', False):
                import tarfile
                archive_path = backup_path / f"beautyai_config_backup{label_suffix}_{timestamp}.tar.gz"
                
                with tarfile.open(archive_path, "w:gz") as tar:
                    if os.path.exists(config_file):
                        tar.add(config_backup, arcname=os.path.basename(config_backup))
                    if os.path.exists(models_file):
                        tar.add(models_backup, arcname=os.path.basename(models_backup))
                
                print(f"Compressed backup created at {archive_path}")
            
            # Clean up old backups if requested
            if getattr(args, 'keep_count', 0) > 0:
                keep_count = getattr(args, 'keep_count')
                self._cleanup_old_backups(backup_path, keep_count)
                
            print(f"Backup completed successfully.")
            return 0
            
        except Exception as e:
            print(f"Error backing up configuration: {str(e)}")
            logger.exception("Failed to back up configuration")
            return 1
            
    def _cleanup_old_backups(self, backup_path: Path, keep_count: int) -> None:
        """
        Clean up old backups, keeping only the most recent ones.
        
        Args:
            backup_path: Path to backup directory
            keep_count: Number of most recent backups to keep
        """
        # Find all config backup files
        config_backups = list(backup_path.glob("config_*.json"))
        models_backups = list(backup_path.glob("models_*.json"))
        archives = list(backup_path.glob("beautyai_config_backup_*.tar.gz"))
        
        # Sort by modification time (oldest first)
        config_backups.sort(key=lambda x: os.path.getmtime(x))
        models_backups.sort(key=lambda x: os.path.getmtime(x))
        archives.sort(key=lambda x: os.path.getmtime(x))
        
        # Delete oldest backups beyond keep_count
        if len(config_backups) > keep_count:
            for old_backup in config_backups[:-keep_count]:
                os.remove(old_backup)
                print(f"Removed old backup: {old_backup}")
                
        if len(models_backups) > keep_count:
            for old_backup in models_backups[:-keep_count]:
                os.remove(old_backup)
                print(f"Removed old backup: {old_backup}")
                
        if len(archives) > keep_count:
            for old_archive in archives[:-keep_count]:
                os.remove(old_archive)
                print(f"Removed old archive: {old_archive}")
                
        print(f"Kept {keep_count} most recent backups")
    
    def restore_config(self, args):
        """
        Restore configuration from backup files.
        
        This method restores configuration from previously created backups.
        It automatically creates a safety backup of current configuration
        before proceeding, and validates the restored configuration after
        completion.
        
        Args:
            args: Command line arguments with restore options
            
        Returns:
            int: 0 for success, 1 for failure
        """
        try:
            config_file = args.config_file
            models_file = getattr(args, 'models_file', None)
            
            if not os.path.exists(config_file):
                print(f"Error: Config backup file {config_file} not found")
                return 1
            
            # Determine target paths
            target_config = getattr(args, 'target_config', None)
            if not target_config:
                # Use default location
                target_config = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            
            # Verify backup file is a valid JSON
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Backup file {config_file} is not valid JSON")
                return 1
                
            # Backup current configuration before restoring
            if os.path.exists(target_config):
                backup_suffix = f".pre_restore.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_file = f"{target_config}{backup_suffix}"
                shutil.copy2(target_config, backup_file)
                print(f"Current configuration backed up to {backup_file}")
                
            # Restore main configuration
            shutil.copy2(config_file, target_config)
            print(f"Configuration restored from {config_file}")
            
            # Restore models if provided
            if models_file:
                if not os.path.exists(models_file):
                    print(f"Warning: Models backup file {models_file} not found, skipping model restore")
                else:
                    # Verify models file is valid JSON
                    try:
                        with open(models_file, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error: Models backup file {models_file} is not valid JSON")
                        # Rollback main config restore
                        if os.path.exists(backup_file):
                            shutil.copy2(backup_file, target_config)
                            print(f"Rolled back configuration restore due to invalid models file")
                        return 1
                    
                    # Load config to determine models file path
                    self.app_config = AppConfig(config_file=target_config)
                    target_models = self.app_config.models_file
                    
                    # Create directory structure if needed
                    os.makedirs(os.path.dirname(target_models), exist_ok=True)
                    
                    # Backup current models file
                    if os.path.exists(target_models):
                        backup_models = f"{target_models}{backup_suffix}"
                        shutil.copy2(target_models, backup_models)
                        print(f"Current models backed up to {backup_models}")
                    
                    # Restore models file
                    shutil.copy2(models_file, target_models)
                    print(f"Models configuration restored from {models_file}")
            
            # Load and validate the restored configuration
            if getattr(args, 'validate', True):
                print("\nValidating restored configuration...")
                
                # Create validation args
                validation_args = argparse.Namespace()
                validation_args.config = target_config
                if models_file:
                    validation_args.models_file = target_models
                
                # Check if validation passes
                self.app_config = None  # Clear existing config
                validation_result = self.validate_config(validation_args)
                
                if validation_result != 0:
                    print("\nWarning: Restored configuration has validation errors.")
                    if getattr(args, 'auto_migrate', False):
                        print("Attempting to automatically migrate configuration format...")
                        migrate_args = argparse.Namespace()
                        migrate_args.config = target_config
                        migrate_args.models_file = target_models if models_file else None
                        migrate_args.verbose = True
                        self.migrate_config(migrate_args)
                    else:
                        print("You can fix this by running 'beautyai config migrate' or restore a different backup.")
            
            print(f"\nRestore completed successfully.")
            return 0
                
        except Exception as e:
            print(f"Error restoring configuration: {str(e)}")
            logger.exception("Failed to restore configuration")
            return 1

    def migrate_config(self, args):
        """
        Migrate configuration to the current format standard.
        
        This method checks for compatibility issues in the configuration,
        creates a backup of the current configuration, performs necessary
        migrations, and validates the result.
        
        Args:
            args: Command line arguments
            
        Returns:
            int: 0 for success, 1 for failure
        """
        self._load_config(args)
        
        try:
            # Check if current configuration format needs migration
            config_dict = self.app_config.get_config()
            needs_migration, migration_issues = self._check_migration_needed(config_dict)
            
            if not needs_migration:
                print("✅ Configuration format is up to date. No migration needed.")
                return 0
            
            # Create backup if requested or by default
            if getattr(args, 'backup', True):
                # Create backup args
                backup_args = argparse.Namespace()
                backup_args.backup_dir = "migration_backups"
                backup_args.label = "pre_migration"
                backup_args.compress = True
                backup_args.config = args.config if hasattr(args, 'config') else None
                backup_args.models_file = args.models_file if hasattr(args, 'models_file') else None
                
                # Create backup
                backup_result = self.backup_config(backup_args)
                if backup_result != 0:
                    print("Warning: Failed to create backup before migration")
                    
                    # Confirm if user wants to continue without backup
                    if not getattr(args, 'force', False):
                        response = input("Continue migration without backup? (y/n): ").lower().strip()
                        if response != 'y':
                            print("Migration cancelled.")
                            return 1
            
            print("\n=== Migrating Configuration ===")
            print(f"Found {len(migration_issues)} issues to fix:")
            for issue in migration_issues:
                print(f"  - {issue}")
            
            # Perform migration steps
            migrated_config = self._migrate_config_format(config_dict)
            
            # Update app_config with migrated data
            self.app_config.update_from_dict(migrated_config)
            
            # Save migrated config
            self.app_config.save_config()
            self.app_config.save_model_registry()
            
            # Validate the migrated configuration
            print("\nValidating migrated configuration...")
            validation_args = argparse.Namespace()
            validation_args.config = self.config_file_path
            validation_args.models_file = self.app_config.models_file
            validation_result = self.validate_config(validation_args)
            
            if validation_result == 0:
                print("\n✅ Configuration successfully migrated to new format.")
                
                # Show summary of changes
                if getattr(args, 'verbose', False):
                    print("\nMigration changes summary:")
                    for issue in migration_issues:
                        print(f"✓ Fixed: {issue}")
                    
                    # Show any configuration updates
                    if hasattr(self.app_config, 'model'):
                        print("\nUpdated model configuration:")
                        model_dict = self.app_config.model.to_dict()
                        for key, value in model_dict.items():
                            print(f"  {key}: {value}")
            else:
                print("\n⚠️ Configuration migrated but validation found issues.")
                print("Please check the validation output and fix any remaining issues manually.")
            
            return 0
            
        except Exception as e:
            print(f"Error migrating configuration: {str(e)}")
            logger.exception("Failed to migrate configuration")
            return 1
    
    def _load_config(self, args):
        """Load the configuration."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        # Store the config file path for reference
        self.config_file_path = config_file
        
        if config_file:
            # Load configuration from file
            self.app_config = AppConfig.load_from_file(config_file)
            # Override models file if specified
            if models_file:
                self.app_config.models_file = models_file
        else:
            # Create default configuration
            self.app_config = AppConfig()
            # Set models file if specified
            if models_file:
                self.app_config.models_file = models_file
        
        # Load model registry
        self.app_config.load_model_registry()

    def _get_config_schema(self) -> Dict[str, Any]:
        """
        Get the comprehensive configuration schema for validation.
        
        This schema defines all valid configuration properties and their
        constraints, including data types, allowed values, and requirements.
        
        Returns:
            Dict[str, Any]: Complete JSON schema for configuration validation
        """
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "description": "Default model configuration",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Hugging Face model ID or path to local model"
                        },
                        "engine_type": {
                            "type": "string",
                            "enum": ["transformers", "vllm", "llama.cpp"],
                            "description": "Inference engine to use"
                        },
                        "quantization": {
                            "type": ["string", "null"], 
                            "enum": ["4bit", "8bit", "awq", "squeezellm", "Q4_K_M", "Q5_K_M", "Q8_0", "none", None],
                            "description": "Quantization method for reducing model size"
                        },
                        "dtype": {
                            "type": "string", 
                            "enum": ["float16", "float32", "bfloat16"],
                            "description": "Floating point precision type"
                        },
                        "max_new_tokens": {
                            "type": "integer", 
                            "minimum": 1,
                            "description": "Maximum new tokens to generate"
                        },
                        "temperature": {
                            "type": "number", 
                            "minimum": 0.0, 
                            "maximum": 2.0,
                            "description": "Sampling temperature (higher = more creative)"
                        },
                        "top_p": {
                            "type": "number", 
                            "minimum": 0.0, 
                            "maximum": 1.0,
                            "description": "Nucleus sampling parameter"
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Top-k sampling parameter"
                        },
                        "repetition_penalty": {
                            "type": "number",
                            "minimum": 0.0,
                            "description": "Penalty for repetition in generated text"
                        },
                        "do_sample": {
                            "type": "boolean",
                            "description": "Whether to use sampling (vs greedy decoding)"
                        },
                        "gpu_memory_utilization": {
                            "type": "number", 
                            "minimum": 0.0, 
                            "maximum": 1.0,
                            "description": "GPU memory utilization target (vLLM only)"
                        },
                        "tensor_parallel_size": {
                            "type": "integer", 
                            "minimum": 1,
                            "description": "Number of GPUs for tensor parallelism (vLLM only)"
                        },
                        "name": {
                            "type": "string",
                            "description": "Friendly name for the model configuration"
                        },
                        "description": {
                            "type": ["string", "null"],
                            "description": "Optional description of the model"
                        },
                        "model_architecture": {
                            "type": "string", 
                            "enum": ["causal_lm", "seq2seq_lm"],
                            "description": "Model architecture type"
                        },
                        "documentation": {
                            "type": ["object", "null"],
                            "additionalProperties": {"type": "string"},
                            "description": "Additional documentation for the model"
                        },
                        "custom_generation_params": {
                            "type": ["object", "null"],
                            "additionalProperties": True,
                            "description": "Custom parameters for text generation"
                        }
                    },
                    "required": ["model_id", "engine_type", "name"]
                },
                "cache_dir": {
                    "type": ["string", "null"],
                    "description": "Cache directory for model files"
                },
                "log_level": {
                    "type": "string", 
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "description": "Logging verbosity level"
                },
                "models_file": {
                    "type": "string",
                    "description": "Path to model registry file"
                },
                "preferred_engine": {
                    "type": ["string", "null"],
                    "enum": ["transformers", "vllm", "llama.cpp", None],
                    "description": "Preferred inference engine when multiple are available"
                },
                "auto_migration": {
                    "type": "boolean",
                    "description": "Whether to automatically migrate configuration format"
                },
                "max_memory": {
                    "type": ["integer", "null"],
                    "description": "Maximum memory limit in MB"
                },
                "device_map": {
                    "type": ["string", "null", "object"],
                    "description": "Device mapping for multi-GPU setups"
                },
            },
            "required": ["model", "log_level"],
            "additionalProperties": True
        }
        return schema
        
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against JSON schema."""
        result = {"valid": True, "errors": []}
        
        try:
            jsonschema.validate(instance=config, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            result["valid"] = False
            # Convert validation error to user-friendly message
            path = ".".join(str(p) for p in e.path) if e.path else "root"
            message = e.message
            result["errors"].append(f"At {path}: {message}")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))
            
        return result
        
    def _validate_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a model configuration."""
        model_schema = {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "engine_type": {"type": "string", "enum": ["transformers", "vllm"]},
                "quantization": {"type": ["string", "null"], "enum": ["4bit", "8bit", "awq", "squeezellm", "none", None]},
                "dtype": {"type": "string", "enum": ["float16", "float32", "bfloat16"]},
                "max_new_tokens": {"type": "integer", "minimum": 1},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "do_sample": {"type": "boolean"},
                "gpu_memory_utilization": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "tensor_parallel_size": {"type": "integer", "minimum": 1},
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "model_architecture": {"type": "string", "enum": ["causal_lm", "seq2seq_lm"]},
                "documentation": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"}
                },
                "custom_generation_params": {
                    "type": ["object", "null"],
                    "additionalProperties": True
                }
            },
            "required": ["model_id", "engine_type", "name"]
        }
        
        # Handle vLLM-specific validations
        if model_config.get("engine_type") == "vllm":
            if model_config.get("model_architecture") == "seq2seq_lm":
                return {
                    "valid": False,
                    "errors": ["vLLM engine does not support seq2seq_lm architecture models. Use transformers engine instead."]
                }
            
            if model_config.get("quantization") in ["4bit", "8bit"]:
                return {
                    "valid": False, 
                    "errors": [f"{model_config.get('quantization')} quantization is not supported with vLLM. Use 'awq', 'squeezellm', or 'none' instead."]
                }
        
        # Validate against schema
        return self._validate_against_schema(model_config, model_schema)
    
    def _check_migration_needed(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if configuration needs migration to new format.
        
        This method identifies specific configuration issues that require 
        migration to the current format standard.
        
        Args:
            config: Current configuration dictionary
            
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
        models = self.app_config.model_registry.models
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
    
    def _migrate_config_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate configuration to current format.
        
        This method performs various transformations to update configuration
        from older formats to the current standard format.
        
        Args:
            config: Current configuration dictionary
            
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
        if hasattr(self.app_config, "model_registry") and self.app_config.model_registry:
            for name, model in self.app_config.model_registry.models.items():
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
