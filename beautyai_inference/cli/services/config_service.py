"""
Configuration service for unified CLI.
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ...config.config_manager import AppConfig

logger = logging.getLogger(__name__)


class ConfigService(BaseService):
    """Service for managing configuration operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
    
    def show_config(self, args):
        """Show the current configuration."""
        self._load_config(args)
        
        # Display global configuration
        print("\n=== Global Configuration ===")
        print(f"Config File:     {self.app_config.config_file}")
        print(f"Models File:     {self.app_config.models_file}")
        print(f"Default Model:   {self.app_config.default_model_name or 'None'}")
        print(f"Cache Directory: {self.app_config.cache_dir}")
        
        # Display all other configuration
        config_dict = self.app_config.get_config()
        if config_dict:
            print("\n=== Custom Configuration ===")
            for key, value in config_dict.items():
                if key not in ["models", "default_model"]:
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
            config_file = self.app_config.config_file
            if os.path.exists(config_file):
                backup_file = f"{config_file}.backup"
                with open(config_file, "r") as f_in:
                    with open(backup_file, "w") as f_out:
                        f_out.write(f_in.read())
                print(f"Created backup of current config at {backup_file}")
            
            # Reset to defaults (but keep models)
            models = self.app_config.get_models()
            default_model = self.app_config.default_model_name
            
            self.app_config.reset_to_defaults()
            
            # Restore models
            for name, model in models.items():
                self.app_config.add_model(model)
            
            if default_model:
                self.app_config.set_default_model(default_model)
                
            self.app_config.save_config()
            self.app_config.save_models()
            
            print(f"Configuration reset to defaults. Models have been preserved.")
            
        except Exception as e:
            print(f"Error resetting configuration: {str(e)}")
            logger.exception("Failed to reset configuration")
            return 1
    
    def _load_config(self, args):
        """Load the configuration."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        self.app_config = AppConfig(
            config_file=config_file,
            models_file=models_file
        )
