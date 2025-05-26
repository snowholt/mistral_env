"""
CLI adapter for unified CLI to use new refactored services.

This adapter bridges the gap between the old CLI interface (which expects args objects)
and the new refactored services (which expect structured parameters).
"""
import logging
from typing import Dict, Any, Optional
import argparse

from ...services.model import RegistryService, ModelLifecycleService
from ...services.inference import ChatService, TestService, BenchmarkService, SessionService
from ...services.config import ConfigService, ValidationService, MigrationService, BackupService
from ...services.system import MemoryService, CacheService, StatusService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class UnifiedCLIAdapter:
    """Adapter to bridge CLI interface with new refactored services."""
    
    def __init__(self):
        # Initialize new services
        self.registry_service = RegistryService()
        self.lifecycle_service = ModelLifecycleService()
        self.chat_service = ChatService()
        self.test_service = TestService()
        self.benchmark_service = BenchmarkService()
        self.session_service = SessionService()
        
        # Initialize specialized config services
        self.config_service = ConfigService()
        self.validation_service = ValidationService()
        self.migration_service = MigrationService()
        self.backup_service = BackupService()
        
        # Initialize system services
        self.memory_service = MemoryService()
        self.cache_service = CacheService()
        self.status_service = StatusService()
        
    def _load_config(self, args) -> AppConfig:
        """Load configuration from args."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        if config_file:
            app_config = AppConfig.load_from_file(config_file)
            if models_file:
                app_config.models_file = models_file
        else:
            # Load default configuration
            from pathlib import Path
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            if default_config_path.exists():
                app_config = AppConfig.load_from_file(default_config_path)
                if app_config.models_file and not Path(app_config.models_file).is_absolute():
                    config_dir = Path(__file__).parent.parent.parent / "config"
                    app_config.models_file = str(config_dir / "model_registry.json")
            else:
                app_config = AppConfig()
                
        # Set custom models file if provided
        if models_file:
            app_config.models_file = models_file
            
        # Load model registry
        try:
            app_config.load_model_registry()
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
            
        return app_config
    
    # Model Registry Operations
    def list_models(self, args) -> int:
        """List all models in the registry."""
        try:
            app_config = self._load_config(args)
            result = self.registry_service.list_models(app_config)
            
            models = result["models"]
            default_model = result["default_model"]
            
            if not models:
                print("No models found in registry.")
                return 0
                
            print(f"\n{'':<40} {'ENGINE':<18} {'QUANT':<12} {'DEFAULT':<10}")
            print("-" * 85)
            
            for name, model in models.items():
                is_default = "✓" if name == default_model else ""
                print(f"{name:<40} {model.engine_type:<18} {model.quantization or 'none':<12} {is_default:<10}")
            print()
            
            return 0
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            print(f"Error: Failed to list models - {e}")
            return 1
    
    def add_model(self, args) -> int:
        """Add a new model to the registry."""
        try:
            app_config = self._load_config(args)
            
            model_config = ModelConfig(
                model_id=args.model_id,
                engine_type=args.engine,
                quantization=getattr(args, 'quantization', None),
                dtype=getattr(args, 'dtype', 'float16'),
                name=args.name,
                description=getattr(args, 'description', None)
            )
            
            set_as_default = getattr(args, 'default', False)
            success = self.registry_service.add_model(app_config, model_config, set_as_default)
            
            if success:
                if set_as_default:
                    print(f"Added model '{args.name}' to registry and set as default.")
                else:
                    print(f"Added model '{args.name}' to registry.")
                return 0
            else:
                print(f"Error: Model with name '{args.name}' already exists. Use 'update' to modify.")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to add model: {e}")
            print(f"Error: Failed to add model - {e}")
            return 1
    
    def show_model(self, args) -> int:
        """Show details for a specific model."""
        try:
            app_config = self._load_config(args)
            model_config = self.registry_service.get_model(app_config, args.name)
            
            if not model_config:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
                
            is_default = args.name == app_config.model_registry.default_model
            
            print(f"\nModel: {args.name} {'(default)' if is_default else ''}")
            print("-" * 40)
            print(f"Model ID:         {model_config.model_id}")
            print(f"Engine:           {model_config.engine_type}")
            print(f"Quantization:     {model_config.quantization or 'none'}")
            print(f"Data type:        {model_config.dtype}")
            print(f"Architecture:     {model_config.model_architecture}")
            
            if model_config.description:
                print(f"\nDescription: {model_config.description}")
            
            if model_config.custom_generation_params:
                print("\nGeneration Parameters:")
                for key, value in model_config.custom_generation_params.items():
                    print(f"  {key}: {value}")
            
            print()
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show model: {e}")
            print(f"Error: Failed to show model - {e}")
            return 1
    
    def update_model(self, args) -> int:
        """Update an existing model in the registry."""
        try:
            app_config = self._load_config(args)
            
            # Build updates dictionary
            updates = {}
            if hasattr(args, 'model_id') and args.model_id:
                updates['model_id'] = args.model_id
            if hasattr(args, 'engine') and args.engine:
                updates['engine_type'] = args.engine
            if hasattr(args, 'quantization') and args.quantization:
                updates['quantization'] = args.quantization
            if hasattr(args, 'dtype') and args.dtype:
                updates['dtype'] = args.dtype
            if hasattr(args, 'description') and args.description:
                updates['description'] = args.description
            
            set_as_default = getattr(args, 'default', False)
            success = self.registry_service.update_model(app_config, args.name, updates, set_as_default)
            
            if success:
                if set_as_default:
                    print(f"Updated model '{args.name}' and set as default.")
                else:
                    print(f"Updated model '{args.name}'.")
                return 0
            else:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            print(f"Error: Failed to update model - {e}")
            return 1
    
    def remove_model(self, args) -> int:
        """Remove a model from the registry."""
        try:
            app_config = self._load_config(args)
            
            clear_cache = getattr(args, 'clear_cache', False)
            success = self.registry_service.remove_model(app_config, args.name, clear_cache)
            
            if success:
                print(f"Removed model '{args.name}' from registry.")
                return 0
            else:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            print(f"Error: Failed to remove model - {e}")
            return 1
    
    def set_default_model(self, args) -> int:
        """Set a model as the default."""
        try:
            app_config = self._load_config(args)
            success = self.registry_service.set_default_model(app_config, args.name)
            
            if success:
                print(f"Set '{args.name}' as the default model.")
                return 0
            else:
                print(f"Error: Model '{args.name}' not found in registry.")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to set default model: {e}")
            print(f"Error: Failed to set default model - {e}")
            return 1
    
    # Lifecycle Operations - delegate to lifecycle service
    def load_model(self, args) -> int:
        """Load a model into memory."""
        return self.lifecycle_service.load_model(args)
    
    def unload_model(self, args) -> int:
        """Unload a model from memory."""
        return self.lifecycle_service.unload_model(args)
    
    def unload_all_models(self, args) -> int:
        """Unload all models from memory."""
        return self.lifecycle_service.unload_all_models(args)
    
    def list_loaded_models(self, args) -> int:
        """List all loaded models."""
        return self.lifecycle_service.list_loaded_models(args)
    
    def show_status(self, args) -> int:
        """Show system status."""
        try:
            status = self.status_service.get_comprehensive_status()
            formatted_status = self.status_service.format_status_display(status)
            print(formatted_status)
            return 0
        except Exception as e:
            logger.error(f"Failed to show status: {e}")
            print(f"Error: Failed to show status - {e}")
            return 1
    
    def clear_cache(self, args) -> int:
        """Clear model cache."""
        try:
            model_id = getattr(args, 'model_id', None)
            
            if model_id:
                # Clear cache for specific model
                success = self.cache_service.clear_model_cache(model_id)
                if success:
                    cache_info = self.cache_service.get_model_cache_info(model_id)
                    print(f"✅ Successfully cleared cache for {model_id}")
                    if cache_info.exists:
                        print(f"   Freed: {cache_info.size_human}")
                    return 0
                else:
                    print(f"❌ Failed to clear cache for {model_id}")
                    return 1
            else:
                # Clear all caches
                if hasattr(args, 'all') and args.all:
                    # Get stats before clearing
                    cache_stats = self.cache_service.get_total_cache_size()
                    success = self.cache_service.clear_all_cache()
                    
                    if success:
                        print(f"✅ Successfully cleared all model caches")
                        print(f"   Freed: {cache_stats['total_size_human']}")
                        return 0
                    else:
                        print("❌ Failed to clear all caches")
                        return 1
                else:
                    print("Error: Must specify --model-id or --all")
                    return 1
                    
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            print(f"Error: Failed to clear cache - {e}")
            return 1
    
    # Inference Operations - delegate to specialized services
    def start_chat(self, args) -> int:
        """Start interactive chat."""
        return self.chat_service.start_chat(args)
    
    def run_test(self, args) -> int:
        """Run model test."""
        return self.test_service.run_test(args)
    
    def run_benchmark(self, args) -> int:
        """Run performance benchmark."""
        return self.benchmark_service.run_benchmark(args)
    
    def save_session(self, args) -> int:
        """Save chat session."""
        return self.session_service.save_session(args)
    
    def load_session(self, args) -> int:
        """Load chat session."""
        return self.session_service.load_session(args)
    
    # Config Operations
    def show_config(self, args) -> int:
        """Show configuration."""
        return self.config_service.show_config(args)
    
    def set_config(self, args) -> int:
        """Set configuration value."""
        return self.config_service.set_config(args)
    
    def reset_config(self, args) -> int:
        """Reset configuration."""
        return self.config_service.reset_config(args)
    
    def validate_config(self, args) -> int:
        """Validate configuration."""
        return self.validation_service.validate_config(args)
    
    def backup_config(self, args) -> int:
        """Backup configuration."""
        return self.backup_service.backup_config(args)
    
    def restore_config(self, args) -> int:
        """Restore configuration."""
        return self.backup_service.restore_config(args)
    
    def migrate_config(self, args) -> int:
        """Migrate configuration."""
        return self.migration_service.migrate_config(args)
