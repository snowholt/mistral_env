#!/usr/bin/env python3
"""
CLI tool for managing loaded models.
DEPRECATED: This command is deprecated. Please use 'beautyai manage lifecycle' instead.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

from ..config.config_manager import AppConfig
from ..core.model_manager import ModelManager
from ..utils.memory_utils import get_gpu_memory_stats
from .argument_config import add_backward_compatible_args, ArgumentValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track legacy command usage
USAGE_LOG_FILE = Path.home() / ".beautyai" / "legacy_usage.log"


def log_legacy_usage(command: str, args: list):
    """Log usage of legacy command for future cleanup analysis."""
    try:
        USAGE_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(USAGE_LOG_FILE, "a") as f:
            import datetime
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"{timestamp},{command},{' '.join(args)}\n")
    except Exception:
        # Silently fail if logging doesn't work
        pass


def show_deprecation_warning():
    """Show deprecation warning with migration guidance."""
    warning_msg = """
🚨 DEPRECATION WARNING 🚨

The 'beautyai-model-management' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-model-management [options]
  NEW: beautyai manage lifecycle [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help

This warning can be suppressed by setting BEAUTYAI_SUPPRESS_WARNINGS=1
"""
    
    if not sys.environ.get("BEAUTYAI_SUPPRESS_WARNINGS"):
        print(warning_msg, file=sys.stderr)
        warnings.warn(
            "beautyai-model-management is deprecated. Use 'beautyai manage lifecycle' instead.",
            DeprecationWarning,
            stacklevel=2
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manage loaded models")
    
    # Add standardized arguments with backward compatibility (global args only)
    add_backward_compatible_args(
        parser,
        include_model=False,  # We handle model selection manually for subcommands
        include_generation=False,
        include_system=False
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List loaded models
    list_parser = subparsers.add_parser("list-loaded", help="List all loaded models")
    
    # Load a model
    load_parser = subparsers.add_parser("load", help="Load a model into memory")
    load_parser.add_argument("name", help="Name of the model to load")
    
    # Unload a model
    unload_parser = subparsers.add_parser("unload", help="Unload a model from memory")
    unload_parser.add_argument("name", help="Name of the model to unload")
    
    # Unload all models
    unload_all_parser = subparsers.add_parser("unload-all", help="Unload all models from memory")
    
    # Memory status
    status_parser = subparsers.add_parser("status", help="Show memory status")
    
    # Clear cache
    clear_cache_parser = subparsers.add_parser("clear-cache", help="Clear model cache from disk")
    clear_cache_parser.add_argument("name", help="Name of the model to clear cache for")
    
    # Enable auto-completion if available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass  # Auto-completion not available
    
    return parser.parse_args()


def main():
    """
    Main entry point for the model management CLI.
    DEPRECATED: Redirects to unified CLI.
    """
    # Log legacy usage
    log_legacy_usage("beautyai-model-management", sys.argv[1:])
    
    # Show deprecation warning
    show_deprecation_warning()
    
    # Redirect to unified CLI
    try:
        from .unified_cli import main as unified_main
        
        # Modify sys.argv to match unified CLI format
        # Convert: beautyai-model-management [args] -> beautyai manage lifecycle [args]
        original_argv = sys.argv.copy()
        sys.argv = ["beautyai", "manage", "lifecycle"] + sys.argv[1:]
        
        # Call the unified CLI
        return unified_main()
        
    except Exception as e:
        # Fallback to original implementation if unified CLI fails
        logger.warning(f"Failed to redirect to unified CLI: {e}")
        logger.info("Falling back to legacy implementation...")
        
        # Restore original argv
        sys.argv = original_argv
        
        # Execute legacy implementation
        return _legacy_main()


def _legacy_main():
    """Legacy main implementation kept for fallback."""
    args = parse_arguments()
    
    if not args.command:
        print("No command specified. Use --help to see available commands.")
        return 1
    
    model_manager = ModelManager()
    
    # Handle list-loaded command
    if args.command == "list-loaded":
        loaded_models = model_manager.list_loaded_models()
        
        if not loaded_models:
            print("No models currently loaded")
            return 0
        
        print("\n===== Loaded Models =====")
        for i, model_name in enumerate(loaded_models, 1):
            print(f"{i}. {model_name}")
        
        # Print memory usage
        gpu_memory = get_gpu_memory_stats()
        if gpu_memory:
            print("\n===== GPU Memory Usage =====")
            for key, value in gpu_memory.items():
                print(f"{key}: {value:.2f} GB")
        else:
            print("\n===== GPU Memory Usage =====")
            print("GPU not available")
        
        return 0
    
    # Handle unload command
    if args.command == "unload":
        if model_manager.unload_model(args.name):
            print(f"Model '{args.name}' unloaded successfully")
            return 0
        else:
            print(f"Failed to unload model '{args.name}'")
            return 1
    
    # Handle unload-all command
    if args.command == "unload-all":
        model_manager.unload_all_models()
        print("All models unloaded successfully")
        return 0
    
    # Handle status command
    if args.command == "status":
        loaded_models = model_manager.list_loaded_models()
        
        print("\n===== Memory Status =====")
        
        # GPU memory
        gpu_memory = get_gpu_memory_stats()
        if gpu_memory:
            print("\nGPU Memory:")
            for key, value in gpu_memory.items():
                print(f"  {key}: {value:.2f} GB")
        else:
            print("\nGPU Memory: Not available")
        
        # Loaded models
        print(f"\nLoaded Models: {len(loaded_models)}")
        for model_name in loaded_models:
            print(f"  - {model_name}")
        
        return 0
    
    # Commands that need config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = AppConfig.load_from_file(config_path)
    
    # Set custom models file if provided
    if args.models_file:
        config.models_file = args.models_file
    
    # Load model registry
    config.load_model_registry()
    
    # Handle load command
    if args.command == "load":
        model_config = config.model_registry.get_model(args.name)
        if not model_config:
            print(f"Model '{args.name}' not found in registry")
            return 1
        
        if model_manager.is_model_loaded(args.name):
            print(f"Model '{args.name}' is already loaded")
            return 0
        
        print(f"Loading model '{args.name}'...")
        try:
            model_manager.load_model(model_config)
            print(f"Model '{args.name}' loaded successfully")
            return 0
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
    
    # Handle clear-cache command
    if args.command == "clear-cache":
        model_config = config.model_registry.get_model(args.name)
        if not model_config:
            print(f"Model '{args.name}' not found in registry")
            return 1
        
        print(f"Clearing cache for model '{args.name}'...")
        try:
            if model_manager.clear_model_cache(model_config.model_id):
                print(f"Cache cleared successfully for model '{args.name}'")
                return 0
            else:
                print(f"No cache found for model '{args.name}'")
                return 0
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 1
    
    print(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
