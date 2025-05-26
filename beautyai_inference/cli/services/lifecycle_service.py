"""
Lifecycle service for unified CLI.

This service manages the lifecycle of models including loading, unloading, 
status reporting and cache management. It integrates with the ModelManager
singleton for centralized model management.
"""
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from ...services.base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_memory_stats, format_size

logger = logging.getLogger(__name__)


class LifecycleService(BaseService):
    """Service for managing model lifecycle operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        self.model_manager = ModelManager()
    
    def load_model(self, args) -> int:
        """
        Load a model into memory with progress reporting.
        
        Args:
            args: Command line arguments containing model name
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            self._ensure_config_loaded(args)
            
            name = args.name
            
            # Check if model exists in registry
            if not self.config.model_registry.has_model(name):
                return self._handle_error(
                    ValueError(f"Model '{name}' not found in registry"),
                    f"Model '{name}' not found in registry"
                )
            
            # Check if model is already loaded
            if self.model_manager.is_model_loaded(name):
                print(f"Model '{name}' is already loaded.")
                return 0
            
            # Get model configuration
            model_config = self.config.model_registry.get_model(name)
            
            print(f"Loading model '{name}' ({model_config.model_id})...")
            
            # Show model details before loading
            self._print_model_details(name, model_config)
            
            # Estimate memory requirements
            memory_estimate = self._estimate_memory_requirements(model_config)
            
            # Check available memory
            available_memory = self._check_available_memory()
            
            # Warn if memory might be an issue
            if memory_estimate > available_memory * 0.8:  # 80% threshold
                print(f"\n⚠️  WARNING: This model may require more memory than available.")
                print(f"   Estimated: {format_size(memory_estimate)}, Available: {format_size(available_memory)}")
                print("   Consider using a more efficient quantization method if loading fails.\n")
                
                if not getattr(args, 'force', False):
                    confirmation = input("Do you want to continue loading? (y/n): ").lower()
                    if confirmation != 'y':
                        print("Model loading cancelled.")
                        return 0
            
            # Load the model with progress indication
            self.model_manager.load_model(model_config)
            
            return self._success_message(f"Model '{name}' loaded successfully.")
            
        except Exception as e:
            return self._handle_error(e, f"Failed to load model '{args.name}'")
            
    
    def unload_model(self, args) -> int:
        """
        Unload a model from memory with error handling.
        
        Args:
            args: Command line arguments containing model name
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            name = args.name
            
            # Check if model is loaded
            if not self.model_manager.is_model_loaded(name):
                return self._handle_error(
                    ValueError(f"Model '{name}' is not loaded"),
                    f"Model '{name}' is not loaded"
                )
            
            print(f"Unloading model '{name}'...")
            
            # Unload with progress indication
            if self.config_data.get('verbose'):
                print("Releasing GPU memory and cleaning up resources...")
                
            self.model_manager.unload_model(name)
            
            # Show memory status after unloading
            if self.config_data.get('verbose'):
                self._print_memory_status()
                
            return self._success_message(f"Model '{name}' unloaded successfully.")
            
        except Exception as e:
            return self._handle_error(e, f"Failed to unload model '{args.name}'")
            
    
    def unload_all_models(self, args) -> int:
        """
        Unload all loaded models from memory with progress reporting.
        
        Args:
            args: Command line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Get loaded models
            loaded_models = self.model_manager.list_loaded_models()
            
            if not loaded_models:
                return self._success_message("No models are currently loaded.")
            
            # Show models that will be unloaded
            print(f"Unloading {len(loaded_models)} model(s)...")
            
            if self.config_data.get('verbose'):
                print("The following models will be unloaded:")
                for model_name in loaded_models:
                    print(f"  - {model_name}")
            
            # Unload all models
            self.model_manager.unload_all_models()
            
            # Display memory status after unloading if verbose
            if self.config_data.get('verbose'):
                self._print_memory_status()
                
            return self._success_message("All models unloaded successfully.")
            
        except Exception as e:
            return self._handle_error(e, "Failed to unload all models")
            
    
    def list_loaded_models(self, args) -> int:
        """
        List all loaded models with detailed formatting.
        
        Args:
            args: Command line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Get loaded models
            loaded_models = self.model_manager.list_loaded_models()
            
            if not loaded_models:
                return self._success_message("No models are currently loaded.")
                
            # Print header
            print("\n===== Loaded Models =====")
            print("\n{:<20} {:<30} {:<15} {:<15} {:<15}".format(
                "MODEL NAME", "MODEL ID", "ENGINE", "QUANTIZATION", "MEMORY USAGE"))
            print("-" * 95)
            
            # Get memory stats for better reporting
            memory_stats = get_gpu_memory_stats()
            
            # Print models with details
            for model_name in loaded_models:
                model = self.model_manager.get_model(model_name)
                if model:
                    # Estimate memory usage (actual implementation would be more accurate)
                    memory_usage = "Unknown"
                    if hasattr(model, "get_memory_usage"):
                        try:
                            mem = model.get_memory_usage()
                            memory_usage = format_size(mem)
                        except:
                            pass
                            
                    print("{:<20} {:<30} {:<15} {:<15} {:<15}".format(
                        model_name,
                        model.config.model_id,
                        model.config.engine_type,
                        model.config.quantization or "none",
                        memory_usage
                    ))
            print()
            
            # Show GPU memory stats
            if memory_stats:
                self._print_memory_status()
                
            return 0
                
        except Exception as e:
            return self._handle_error(e, "Failed to list loaded models")
            
    
    def show_status(self, args) -> int:
        """
        Show comprehensive system status including memory usage and loaded models.
        
        Args:
            args: Command line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Get loaded models
            loaded_models = self.model_manager.list_loaded_models()
            
            print("\n===== System Status =====")
            
            # GPU status
            memory_stats = get_gpu_memory_stats()
            if memory_stats:
                print("\nGPU Status:")
                for i, stats in enumerate(memory_stats):
                    print(f"\n  GPU {i}: {stats.get('name', 'Unknown')}")
                    print(f"    Memory Used:     {stats.get('memory_used_mb', 0):.2f} MB / "
                          f"{stats.get('memory_total_mb', 0):.2f} MB "
                          f"({stats.get('memory_used_percent', 0):.1f}%)")
                    print(f"    Memory Free:     {stats.get('memory_free_mb', 0):.2f} MB")
                    print(f"    Utilization:     {stats.get('gpu_utilization', 0):.1f}%")
                    
                    # Show memory chart
                    if 'memory_used_percent' in stats:
                        used_percent = stats['memory_used_percent']
                        bar_length = 30
                        filled_length = int(bar_length * used_percent / 100)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"    Memory Usage:    [{bar}] {used_percent:.1f}%")
            else:
                print("\nGPU Status: No GPU detected or drivers not available")
                
            # Show system info
            print("\nSystem Information:")
            import platform
            print(f"  Platform:        {platform.system()} {platform.release()}")
            print(f"  Python Version:  {platform.python_version()}")
            
            try:
                import torch
                print(f"  PyTorch Version: {torch.__version__}")
                print(f"  CUDA Available:  {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  CUDA Version:    {torch.version.cuda}")
                    print(f"  Device Count:    {torch.cuda.device_count()}")
                    
                    # Show all devices
                    for i in range(torch.cuda.device_count()):
                        print(f"    Device {i}:       {torch.cuda.get_device_name(i)}")
            except ImportError:
                print("  PyTorch:         Not installed")
            
            # Show loaded models
            print("\nLoaded Models:")
            if loaded_models:
                for model_name in loaded_models:
                    model = self.model_manager.get_model(model_name)
                    engine_type = model.config.engine_type if model else "unknown"
                    quantization = model.config.quantization if model else "unknown"
                    
                    print(f"  - {model_name}")
                    if model:
                        print(f"    ID:            {model.config.model_id}")
                        print(f"    Engine:        {engine_type}")
                        print(f"    Quantization:  {quantization or 'none'}")
                        
                        # Show model specific stats if available
                        if hasattr(model, "get_stats"):
                            try:
                                stats = model.get_stats()
                                for key, value in stats.items():
                                    print(f"    {key}:  {value}")
                            except:
                                pass
            else:
                print("  No models currently loaded.")
                
            print()
            return 0
            
        except Exception as e:
            return self._handle_error(e, "Failed to show system status")
            
    
    def clear_cache(self, args) -> int:
        """
        Clear model cache from disk with progress reporting.
        
        Args:
            args: Command line arguments containing model name
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            self._ensure_config_loaded(args)
            
            name = args.name
            
            # Check if model exists in registry
            if not self.config.model_registry.has_model(name):
                return self._handle_error(
                    ValueError(f"Model '{name}' not found in registry"),
                    f"Model '{name}' not found in registry"
                )
            
            # Get model configuration
            model_config = self.config.model_registry.get_model(name)
            
            print(f"Clearing cache for model '{name}' ({model_config.model_id})...")
            
            # Check if model is loaded
            if self.model_manager.is_model_loaded(name):
                print(f"⚠️  Warning: Model '{name}' is currently loaded. Clearing cache may affect performance.")
                if not getattr(args, 'force', False):
                    confirmation = input("Do you want to continue? (y/n): ").lower()
                    if confirmation != 'y':
                        print("Operation cancelled.")
                        return 0
            
            # Clear cache with feedback
            import os
            from transformers import cached_file
            from pathlib import Path
            
            # Attempt to find the cache directory location
            cache_dir = os.getenv('TRANSFORMERS_CACHE', None)
            if not cache_dir:
                home_dir = Path.home()
                cache_dir = home_dir / '.cache' / 'huggingface'
            
            if os.path.exists(cache_dir):
                print(f"Cache directory: {cache_dir}")
                
            # Clear cache
            result = self.model_manager.clear_model_cache(model_config.model_id)
            
            if result:
                return self._success_message(f"Cache for model '{name}' cleared successfully.")
            else:
                return self._success_message(f"No cache found for model '{name}'.")
                
        except Exception as e:
            return self._handle_error(e, f"Failed to clear cache for model '{args.name}'")
    
    def _ensure_config_loaded(self, args):
        """
        Ensure configuration is loaded properly.
        
        Args:
            args: Command line arguments
        """
        if self.config is None:
            config_file = getattr(args, "config", None)
            models_file = getattr(args, "models_file", None)
            
            config_data = {
                'config_file': config_file,
                'models_file': models_file,
                'verbose': getattr(args, "verbose", False),
            }
            
            self.configure(config_data)
    
    def _print_model_details(self, name: str, model_config: ModelConfig) -> None:
        """
        Print detailed information about a model configuration.
        
        Args:
            name: Model name
            model_config: Model configuration
        """
        print("\nModel Details:")
        print(f"  Name:          {name}")
        print(f"  Model ID:      {model_config.model_id}")
        print(f"  Engine:        {model_config.engine_type}")
        print(f"  Quantization:  {model_config.quantization or 'none'}")
        print(f"  Data Type:     {model_config.dtype}")
        if model_config.description:
            print(f"  Description:   {model_config.description}")
        print()

    def _estimate_memory_requirements(self, model_config: ModelConfig) -> int:
        """
        Estimate memory requirements for a model in bytes.
        
        Args:
            model_config: Model configuration
            
        Returns:
            int: Estimated memory requirements in bytes
        """
        # Extract model size from ID if possible
        import re
        
        # Base memory requirements by model size (very rough estimates)
        base_sizes = {
            "7b": 14 * 1024 * 1024 * 1024,  # ~14GB for 7B models
            "14b": 28 * 1024 * 1024 * 1024,  # ~28GB for 14B models
            "30b": 60 * 1024 * 1024 * 1024,  # ~60GB for 30B models
            "70b": 140 * 1024 * 1024 * 1024,  # ~140GB for 70B models
        }
        
        # Default size if we can't determine
        default_size = 16 * 1024 * 1024 * 1024  # 16GB
        
        # Try to extract size from model ID
        size_match = re.search(r'(\d+)[bB]', model_config.model_id)
        model_size = size_match.group(1).lower() + "b" if size_match else None
        
        # Get base size
        memory_estimate = base_sizes.get(model_size, default_size)
        
        # Apply quantization factor
        if model_config.quantization == "4bit":
            memory_estimate = int(memory_estimate * 0.25)  # ~1/4 of original
        elif model_config.quantization == "8bit":
            memory_estimate = int(memory_estimate * 0.5)   # ~1/2 of original
        elif model_config.quantization in ["awq", "squeezellm"]:
            memory_estimate = int(memory_estimate * 0.2)   # ~1/5 of original
            
        return memory_estimate

    def _check_available_memory(self) -> int:
        """
        Check available GPU memory.
        
        Returns:
            int: Available memory in bytes
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                stats = torch.cuda.get_device_properties(device)
                total_memory = stats.total_memory
                
                # Get current usage
                reserved_memory = torch.cuda.memory_reserved(device)
                allocated_memory = torch.cuda.memory_allocated(device)
                
                # Calculate available memory
                available = total_memory - allocated_memory
                
                return available
            
        except (ImportError, RuntimeError):
            pass
            
        # Return a conservative estimate if we can't determine
        # 8GB as safety fallback
        return 8 * 1024 * 1024 * 1024
        
    def _print_memory_status(self) -> None:
        """Print current memory status."""
        memory_stats = get_gpu_memory_stats()
        
        if not memory_stats:
            print("\nGPU Memory: Not available")
            return
            
        print("\nGPU Memory Status:")
        for i, stats in enumerate(memory_stats):
            print(f"  GPU {i}: {stats.get('name', 'Unknown')}")
            print(f"    Used:  {stats.get('memory_used_mb', 0):.2f} MB / {stats.get('memory_total_mb', 0):.2f} MB")
            print(f"    Free:  {stats.get('memory_free_mb', 0):.2f} MB")
            
            # Add visual meter
            if 'memory_used_percent' in stats:
                used_percent = stats['memory_used_percent']
                bar_length = 20
                filled_length = int(bar_length * used_percent / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"    Usage: [{bar}] {used_percent:.1f}%")
        print()

