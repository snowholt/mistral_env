"""
CLI handler for model management commands.

This handler provides a thin routing layer between CLI arguments
and the model management services. It handles argument parsing
and formatting output for user consumption.
"""
import logging
from typing import Dict, Any, Optional
from argparse import Namespace

from ...services.model import RegistryService, ModelLifecycleService, ModelValidationService
from ...config.config_manager import ModelConfig

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handler for routing model management CLI commands to services."""
    
    def __init__(self):
        self.registry_service = RegistryService()
        self.lifecycle_service = ModelLifecycleService()
        self.validation_service = ModelValidationService()
    
    def configure_services(self, config_data: Dict[str, Any]):
        """Configure all model services with the provided configuration."""
        self.registry_service.configure(config_data)
        self.lifecycle_service.configure(config_data)
        self.validation_service.configure(config_data)
    
    # Model Registry Commands
    
    def list_models(self, args: Namespace) -> int:
        """List all models in the registry."""
        try:
            models = self.registry_service.list_models()
            
            if not models:
                print("No models found in registry.")
                return 0
            
            print("\n{:<40} {:<18} {:<12} {:<10}".format(
                "MODEL NAME", "ENGINE", "QUANT", "DEFAULT"))
            print("-" * 85)
            
            for model in models:
                is_default = "✓" if model['is_default'] else ""
                print("{:<40} {:<18} {:<12} {:<10}".format(
                    model['name'], 
                    model['engine_type'], 
                    model['quantization'],
                    is_default
                ))
            print()
            return 0
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return 1
    
    def add_model(self, args: Namespace) -> int:
        """Add a new model to the registry."""
        try:
            # Create model configuration
            model_config = ModelConfig(
                model_id=args.model_id,
                engine_type=args.engine,
                quantization=args.quantization,
                dtype=args.dtype,
                name=args.name,
                description=args.description
            )
            
            # Validate the configuration
            is_valid, errors = self.validation_service.validate_model_config(model_config)
            if not is_valid:
                print("❌ Model configuration validation failed:")
                for error in errors:
                    print(f"   • {error}")
                return 1
            
            # Check if model already exists
            if self.registry_service.model_exists(args.name):
                print(f"Error: Model with name '{args.name}' already exists. Use 'update' to modify.")
                return 1
            
            # Add the model
            success = self.registry_service.add_model(model_config, set_as_default=args.default)
            
            if success:
                if args.default:
                    print(f"✅ Added model '{args.name}' to registry and set as default.")
                else:
                    print(f"✅ Added model '{args.name}' to registry.")
                return 0
            else:
                print(f"❌ Failed to add model '{args.name}' to registry.")
                return 1
                
        except Exception as e:
            print(f"Error adding model: {e}")
            return 1
    
    def show_model(self, args: Namespace) -> int:
        """Show details for a specific model."""
        try:
            model = self.registry_service.get_model(args.name)
            
            if not model:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
            
            default_model = self.registry_service.get_default_model()
            is_default = default_model and model.name == default_model.name
            
            print(f"\nModel: {model.name} {'(default)' if is_default else ''}")
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
            return 0
            
        except Exception as e:
            print(f"Error showing model details: {e}")
            return 1
    
    def update_model(self, args: Namespace) -> int:
        """Update an existing model in the registry."""
        try:
            if not self.registry_service.model_exists(args.name):
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
            
            # Prepare updates dictionary
            updates = {}
            if args.model_id:
                updates['model_id'] = args.model_id
            if args.engine:
                updates['engine_type'] = args.engine
            if args.quantization:
                updates['quantization'] = args.quantization
            if args.dtype:
                updates['dtype'] = args.dtype
            if args.description:
                updates['description'] = args.description
            
            # If we have updates, validate the resulting configuration
            if updates:
                current_model = self.registry_service.get_model(args.name)
                
                # Create a test configuration for validation
                test_config = ModelConfig(
                    name=current_model.name,
                    model_id=updates.get('model_id', current_model.model_id),
                    engine_type=updates.get('engine_type', current_model.engine_type),
                    quantization=updates.get('quantization', current_model.quantization),
                    dtype=updates.get('dtype', current_model.dtype),
                    description=updates.get('description', current_model.description),
                    model_architecture=current_model.model_architecture,
                    custom_generation_params=current_model.custom_generation_params
                )
                
                # Validate the updated configuration
                is_valid, errors = self.validation_service.validate_model_config(test_config)
                if not is_valid:
                    print("❌ Updated model configuration validation failed:")
                    for error in errors:
                        print(f"   • {error}")
                    return 1
            
            # Update the model
            success = self.registry_service.update_model(args.name, updates, set_as_default=args.default)
            
            if success:
                if args.default:
                    print(f"✅ Updated model '{args.name}' and set as default.")
                else:
                    print(f"✅ Updated model '{args.name}'.")
                return 0
            else:
                print(f"❌ Failed to update model '{args.name}'.")
                return 1
                
        except Exception as e:
            print(f"Error updating model: {e}")
            return 1
    
    def remove_model(self, args: Namespace) -> int:
        """Remove a model from the registry."""
        try:
            if not self.registry_service.model_exists(args.name):
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
            
            # Check if it's the default model
            default_model = self.registry_service.get_default_model()
            if default_model and args.name == default_model.name:
                print(f"Warning: Removing default model '{args.name}'.")
            
            # Remove the model
            clear_cache = getattr(args, 'clear_cache', False)
            success = self.registry_service.remove_model(args.name, clear_cache=clear_cache)
            
            if success:
                cache_msg = " and cleared cache" if clear_cache else ""
                print(f"✅ Removed model '{args.name}' from registry{cache_msg}.")
                return 0
            else:
                print(f"❌ Failed to remove model '{args.name}' from registry.")
                return 1
                
        except Exception as e:
            print(f"Error removing model: {e}")
            return 1
    
    def set_default_model(self, args: Namespace) -> int:
        """Set a model as the default."""
        try:
            success = self.registry_service.set_default_model(args.name)
            
            if success:
                print(f"✅ Set '{args.name}' as the default model.")
                return 0
            else:
                print(f"❌ Model '{args.name}' not found in registry.")
                return 1
                
        except Exception as e:
            print(f"Error setting default model: {e}")
            return 1
    
    # Model Lifecycle Commands
    
    def load_model(self, args: Namespace) -> int:
        """Load a model into memory."""
        try:
            model_config = self.registry_service.get_model(args.name)
            
            if not model_config:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
            
            success, error_msg = self.lifecycle_service.load_model(model_config, show_progress=True)
            
            if success:
                print(f"✅ Model '{args.name}' loaded into memory.")
                return 0
            else:
                print(f"❌ Failed to load model '{args.name}': {error_msg}")
                return 1
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
    
    def unload_model(self, args: Namespace) -> int:
        """Unload a model from memory."""
        try:
            success, error_msg = self.lifecycle_service.unload_model(args.name, show_progress=True)
            
            if success:
                print(f"✅ Model '{args.name}' unloaded from memory.")
                return 0
            else:
                print(f"❌ Failed to unload model '{args.name}': {error_msg}")
                return 1
                
        except Exception as e:
            print(f"Error unloading model: {e}")
            return 1
    
    def unload_all_models(self, args: Namespace) -> int:
        """Unload all models from memory."""
        try:
            success, errors = self.lifecycle_service.unload_all_models(show_progress=True)
            
            if success:
                print("✅ All models unloaded from memory.")
                return 0
            else:
                print("❌ Failed to unload all models:")
                for error in errors:
                    print(f"   • {error}")
                return 1
                
        except Exception as e:
            print(f"Error unloading all models: {e}")
            return 1
    
    def list_loaded_models(self, args: Namespace) -> int:
        """List all loaded models."""
        try:
            loaded_models = self.lifecycle_service.list_loaded_models()
            
            if not loaded_models:
                print("No models are currently loaded in memory.")
                return 0
            
            print(f"\n📋 Loaded Models ({len(loaded_models)}):")
            print("-" * 40)
            for model in loaded_models:
                print(f"  • {model['name']} ({model['status']})")
            print()
            return 0
            
        except Exception as e:
            print(f"Error listing loaded models: {e}")
            return 1
    
    def show_status(self, args: Namespace) -> int:
        """Show memory status and loaded models."""
        try:
            status = self.lifecycle_service.get_memory_status()
            
            print("\n📊 System Status")
            print("=" * 50)
            
            # Loaded models
            print(f"Loaded Models: {status['loaded_models_count']}")
            if status['loaded_models']:
                for model_name in status['loaded_models']:
                    print(f"  • {model_name}")
            
            # GPU memory
            if status['has_gpu'] and status['gpu_memory']:
                print("\nGPU Memory:")
                for i, gpu in enumerate(status['gpu_memory']):
                    total = gpu.get('memory_total', 0)
                    used = gpu.get('memory_used', 0)
                    
                    if total > 0:
                        usage_percent = (used / total) * 100
                        from ...utils.memory_utils import format_size
                        print(f"  GPU {i}: {format_size(used)}/{format_size(total)} ({usage_percent:.1f}% used)")
                    else:
                        print(f"  GPU {i}: Status unavailable")
            else:
                print("\nGPU: Not available or not detected")
            
            print()
            return 0
            
        except Exception as e:
            print(f"Error getting status: {e}")
            return 1
    
    def clear_cache(self, args: Namespace) -> int:
        """Clear cache for a specific model."""
        try:
            model_config = self.registry_service.get_model(args.name)
            
            if not model_config:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
            
            success, error_msg = self.lifecycle_service.clear_model_cache(model_config.model_id)
            
            if success:
                print(f"✅ Cache cleared for model '{args.name}'.")
                return 0
            else:
                print(f"❌ Failed to clear cache for model '{args.name}': {error_msg}")
                return 1
                
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 1
