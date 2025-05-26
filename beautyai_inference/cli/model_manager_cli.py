#!/usr/bin/env python3
"""
CLI tool for managing model configurations.
DEPRECATED: This command is deprecated. Please use 'beautyai manage models' instead.
"""
import argparse
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

from ..config.config_manager import AppConfig, ModelConfig
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

The 'beautyai-model-manager' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-model-manager [options]
  NEW: beautyai model [options]

All arguments and functionality remain the same.

For more information: https://github.com/BeautyAI/inference-framework
"""
    
    warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
    print(warning_msg)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manage model configurations")
    
    # Add standardized arguments with backward compatibility (global args only)
    add_backward_compatible_args(
        parser,
        include_model=False,  # We handle model args manually for subcommands
        include_generation=False,
        include_system=False
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models
    list_parser = subparsers.add_parser("list", help="List all available models")
    
    # Show model details
    show_parser = subparsers.add_parser("show", help="Show details of a specific model")
    show_parser.add_argument("name", help="Name of the model")
    
    # Add model
    add_parser = subparsers.add_parser("add", help="Add a new model configuration")
    add_parser.add_argument("--name", required=True, help="Name for the model configuration")
    add_parser.add_argument("--model-id", required=True, help="Model ID (e.g., Qwen/Qwen3-14B)")
    add_parser.add_argument("--engine", choices=["transformers", "vllm"], default="transformers", 
                           help="Inference engine (default: transformers)")
    add_parser.add_argument("--quantization", choices=["4bit", "8bit", "awq", "squeezellm", "none"], 
                           default="4bit", help="Quantization method (default: 4bit)")
    add_parser.add_argument("--dtype", default="float16", help="Data type (default: float16)")
    add_parser.add_argument("--description", help="Description of the model")
    add_parser.add_argument("--default", action="store_true", help="Set as default model")
    
    # Update model
    update_parser = subparsers.add_parser("update", help="Update an existing model configuration")
    update_parser.add_argument("name", help="Name of the model to update")
    update_parser.add_argument("--model-id", help="Model ID (e.g., Qwen/Qwen3-14B)")
    update_parser.add_argument("--engine", choices=["transformers", "vllm"], help="Inference engine")
    update_parser.add_argument("--description", help="Description of the model")
    update_parser.add_argument("--default", action="store_true", help="Set as default model")
    
    # Remove model
    remove_parser = subparsers.add_parser("remove", help="Remove a model configuration")
    remove_parser.add_argument("name", help="Name of the model to remove")
    remove_parser.add_argument("--clear-cache", action="store_true", help="Also clear model cache from disk")
    
    # Set default model
    default_parser = subparsers.add_parser("set-default", help="Set default model")
    default_parser.add_argument("name", help="Name of the model to set as default")
    
    # Enable auto-completion if available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass  # Auto-completion not available
    
    return parser.parse_args()


def list_models(config: AppConfig):
    """List all available models in the registry."""
    print("\n===== Available Models =====")
    models = config.model_registry.list_models()
    default_model = config.model_registry.default_model
    
    if not models:
        print("No models found in the registry.")
        return
    
    for i, model_name in enumerate(models, 1):
        model = config.model_registry.get_model(model_name)
        default_mark = " (default)" if model_name == default_model else ""
        print(f"{i}. {model_name}{default_mark}")
        print(f"   - ID: {model.model_id}")
        print(f"   - Engine: {model.engine_type}")
        print(f"   - Quantization: {model.quantization or 'None'}")
        if model.description:
            print(f"   - Description: {model.description}")
        print()


def show_model(config: AppConfig, name: str):
    """Show details of a specific model."""
    model = config.model_registry.get_model(name)
    if not model:
        print(f"Model '{name}' not found in registry.")
        return
    
    print(f"\n===== Model: {name} =====")
    print(f"Model ID: {model.model_id}")
    print(f"Engine: {model.engine_type}")
    print(f"Quantization: {model.quantization}")
    print(f"Data Type: {model.dtype}")
    print(f"Max New Tokens: {model.max_new_tokens}")
    print(f"Temperature: {model.temperature}")
    print(f"Top P: {model.top_p}")
    print(f"Do Sample: {model.do_sample}")
    print(f"GPU Memory Utilization: {model.gpu_memory_utilization}")
    print(f"Tensor Parallel Size: {model.tensor_parallel_size}")
    if model.description:
        print(f"Description: {model.description}")
    
    if name == config.model_registry.default_model:
        print("This is the default model.")


def add_model(config: AppConfig, args):
    """Add a new model configuration."""
    # Check if model with this name already exists
    if config.model_registry.get_model(args.name):
        print(f"Model with name '{args.name}' already exists. Use 'update' command to modify it.")
        return False
    
    # Create new model config
    model_config = ModelConfig(
        name=args.name,
        model_id=args.model_id,
        engine_type=args.engine,
        quantization=None if args.quantization == "none" else args.quantization,
        dtype=args.dtype,
        description=args.description
    )
    
    # Add to registry
    config.add_model_config(model_config, set_as_default=args.default)
    print(f"Model '{args.name}' added to registry.")
    
    if args.default:
        print(f"Model '{args.name}' set as default.")
    
    return True


def update_model(config: AppConfig, args):
    """Update an existing model configuration."""
    # Get existing model
    model = config.model_registry.get_model(args.name)
    if not model:
        print(f"Model '{args.name}' not found in registry.")
        return False
    
    # Update fields if provided
    if args.model_id:
        model.model_id = args.model_id
    
    if args.engine:
        model.engine_type = args.engine
    
    if args.quantization:
        model.quantization = None if args.quantization == "none" else args.quantization
    
    if args.dtype:
        model.dtype = args.dtype
    
    if args.description:
        model.description = args.description
    
    # Update model in registry
    config.model_registry.add_model(model)
    config.save_model_registry()
    
    print(f"Model '{args.name}' updated.")
    
    # Set as default if requested
    if args.default:
        config.model_registry.set_default_model(args.name)
        config.save_model_registry()
        print(f"Model '{args.name}' set as default.")
    
    return True


def remove_model(config: AppConfig, args):
    """Remove a model configuration."""
    name = args.name
    clear_cache = getattr(args, 'clear_cache', False)
    
    if name == config.model_registry.default_model:
        print(f"Cannot remove default model '{name}'. Set another model as default first.")
        return False
    
    if config.model_registry.remove_model(name, clear_cache=clear_cache):
        config.save_model_registry()
        if clear_cache:
            print(f"Model '{name}' removed from registry and cache cleared.")
        else:
            print(f"Model '{name}' removed from registry.")
        return True
    else:
        print(f"Model '{name}' not found in registry.")
        return False


def set_default_model(config: AppConfig, name: str):
    """Set default model."""
    if config.model_registry.set_default_model(name):
        config.save_model_registry()
        print(f"Model '{name}' set as default.")
        return True
    else:
        print(f"Model '{name}' not found in registry.")
        return False


def main():
    """
    Main entry point for the model manager CLI.
    DEPRECATED: Redirects to unified CLI.
    """
    # Log legacy usage
    log_legacy_usage("beautyai-model-manager", sys.argv[1:])
    
    # Show deprecation warning
    show_deprecation_warning()
    
    # Redirect to unified CLI using subprocess
    try:
        import subprocess
        
        # Create the new command: beautyai model [args]
        new_cmd = ["beautyai", "model"] + sys.argv[1:]
        
        # Execute the unified CLI command
        process = subprocess.run(new_cmd, check=True)
        sys.exit(process.returncode)
        
    except subprocess.CalledProcessError as e:
        # Handle command failure
        sys.exit(e.returncode)
    except Exception as e:
        # Fallback to original implementation if redirection fails
        logger.warning(f"Failed to redirect to unified CLI: {e}")
        logger.info("Falling back to legacy implementation...")
        
        # Execute legacy implementation
        return _legacy_main()


def _legacy_main():
    """Legacy main implementation kept for fallback."""
    args = parse_arguments()
    
    if not args.command:
        print("No command specified. Use --help to see available commands.")
        return 1
    
    # Load configuration
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
    
    # Handle commands
    if args.command == "list":
        list_models(config)
    
    elif args.command == "show":
        show_model(config, args.name)
    
    elif args.command == "add":
        if not add_model(config, args):
            return 1
    
    elif args.command == "update":
        if not update_model(config, args):
            return 1
    
    elif args.command == "remove":
        if not remove_model(config, args):
            return 1
    
    elif args.command == "set-default":
        if not set_default_model(config, args.name):
            return 1
    
    return 0


if __name__ == "__main__":
    show_deprecation_warning()
    sys.exit(main())
