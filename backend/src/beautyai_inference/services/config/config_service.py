"""
Core configuration service for BeautyAI.

This service provides basic configuration management operations:
- Displaying current configuration values
- Setting and updating configuration values  
- Resetting configuration to defaults
- Loading configuration from files

This is the core service that other config services depend on.
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class ConfigService(BaseService):
    """Core configuration service for basic config operations.
    
    Handles fundamental configuration management including loading,
    displaying, setting values, and resetting to defaults.
    """
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
    
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

    def _load_config(self, args):
        """Load configuration from args or defaults."""
        config_file = getattr(args, 'config', None)
        models_file = getattr(args, 'models_file', None)
        
        # Store config file path for later use
        self.config_file_path = config_file
        
        # Use existing app_config if already loaded and paths match
        if (self.app_config and 
            getattr(self.app_config, 'config_file', None) == config_file and
            getattr(self.app_config, 'models_file', None) == models_file):
            return
        
        # Load fresh config
        try:
            self.app_config = AppConfig.from_file(
                config_file=config_file,
                models_file=models_file
            )
            if config_file:
                self.app_config.config_file = config_file
            if models_file:
                self.app_config.models_file = models_file
                
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            # Fall back to default config
            self.app_config = AppConfig()
            if config_file:
                self.app_config.config_file = config_file
            if models_file:
                self.app_config.models_file = models_file
    
    def get_config(self) -> Optional[AppConfig]:
        """Get the current app configuration."""
        return self.app_config
    
    def load_config_from_args(self, args) -> AppConfig:
        """Load and return configuration from arguments."""
        self._load_config(args)
        return self.app_config
