"""
Model registry service for unified CLI.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistryService(BaseService):
    """Service for managing model registry operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        
    def list_models(self, args):
        """List all models in the registry."""
        self._load_config(args)
        
        models = self.app_config.model_registry.models
        if not models:
            print("No models found in registry.")
            return
            
        default_model = self.app_config.model_registry.default_model
        
        print("\n{:<40} {:<18} {:<12} {:<10}".format(
            "MODEL NAME", "ENGINE", "QUANT", "DEFAULT"))
        print("-" * 85)
        
        for name, model in models.items():
            is_default = "✓" if name == default_model else ""
            print("{:<40} {:<18} {:<12} {:<10}".format(
                name, 
                model.engine_type, 
                model.quantization or "none",
                is_default
            ))
        print()
    
    def add_model(self, args):
        """Add a new model to the registry."""
        self._load_config(args)
        
        name = args.name
        if name in self.app_config.model_registry.models:
            print(f"Error: Model with name '{name}' already exists. Use 'update' to modify.")
            return 1
        
        model_config = ModelConfig(
            model_id=args.model_id,
            engine_type=args.engine,
            quantization=args.quantization,
            dtype=args.dtype,
            name=name,
            description=args.description
        )
        
        self.app_config.add_model_config(model_config)
        
        if args.default:
            self.app_config.model_registry.set_default_model(name)
            self.app_config.save_model_registry()
            print(f"Added model '{name}' to registry and set as default.")
        else:
            print(f"Added model '{name}' to registry.")
    
    def show_model(self, args):
        """Show details for a specific model."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.model_registry.models
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
            
        model = models[name]
        is_default = name == self.app_config.model_registry.default_model
        
        print(f"\nModel: {name} {'(default)' if is_default else ''}")
        print("-" * 40)
        print(f"Model ID:         {model.model_id}")
        print(f"Engine:           {model.engine_type}")
        print(f"Quantization:     {model.quantization or 'none'}")
        print(f"Data type:        {model.dtype}")
        print(f"Architecture:     {model.model_architecture}")
        
        if model.description:
            print(f"\nDescription: {model.description}")
        
        if model.custom_generation_params:
            print("\nGeneration Parameters:")
            for key, value in model.custom_generation_params.items():
                print(f"  {key}: {value}")
        
        print()
    
    def update_model(self, args):
        """Update an existing model in the registry."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.model_registry.models
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
            
        model = models[name]
        
        # Update only the fields that were provided
        if args.model_id:
            model.model_id = args.model_id
        
        if args.engine:
            model.engine_type = args.engine
            
        if args.quantization:
            model.quantization = args.quantization
            
        if args.dtype:
            model.dtype = args.dtype
            
        if args.description:
            model.description = args.description
        
        # Update the model in the registry
        self.app_config.model_registry.add_model(model)  # This will overwrite existing
        self.app_config.save_model_registry()
        
        if args.default:
            self.app_config.model_registry.set_default_model(name)
            self.app_config.save_model_registry()
            print(f"Updated model '{name}' and set as default.")
        else:
            print(f"Updated model '{name}'.")
    
    def remove_model(self, args):
        """Remove a model from the registry."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.model_registry.models
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
        
        if name == self.app_config.model_registry.default_model:
            print(f"Warning: Removing default model '{name}'.")
            self.app_config.model_registry.default_model = None
            
        self.app_config.model_registry.remove_model(name)
        self.app_config.save_model_registry()
        print(f"Removed model '{name}' from registry.")
    
    def set_default_model(self, args):
        """Set a model as the default."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.model_registry.models
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
            
        self.app_config.model_registry.set_default_model(name)
        self.app_config.save_model_registry()
        print(f"Set '{name}' as the default model.")
    
    def _load_config(self, args):
        """Load the configuration."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        if config_file:
            # Load configuration from file
            self.app_config = AppConfig.load_from_file(config_file)
            # Override models file if specified
            if models_file:
                self.app_config.models_file = models_file
        else:
            # Load default configuration from file
            from pathlib import Path
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            if default_config_path.exists():
                self.app_config = AppConfig.load_from_file(default_config_path)
                # Ensure the models_file path is absolute for the default config
                if self.app_config.models_file and not Path(self.app_config.models_file).is_absolute():
                    config_dir = Path(__file__).parent.parent.parent / "config"
                    self.app_config.models_file = str(config_dir / "model_registry.json")
            else:
                # Fallback to empty configuration
                self.app_config = AppConfig()
            # Set models file if specified
            if models_file:
                self.app_config.models_file = models_file
        
        # Load model registry
        self.app_config.load_model_registry()
