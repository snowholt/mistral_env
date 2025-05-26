#!/usr/bin/env python3
"""
Test CLI for running inference with BeautyAI models.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_factory import ModelFactory
from ..utils.memory_utils import get_gpu_info, get_gpu_memory_stats, clear_terminal_screen
from .argument_config import add_backward_compatible_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BeautyAI model inference")
    
    # Add standardized arguments with backward compatibility
    add_backward_compatible_args(
        parser,
        include_model=True,
        include_generation=True,
        include_system=False
    )
    
    # Add legacy config argument if not already added
    try:
        parser.add_argument(
            "--config",
            type=str,
            help="Path to a JSON configuration file",
        )
    except argparse.ArgumentError:
        # Argument already exists from global args
        pass
    
    # Enable auto-completion if available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass  # Auto-completion not available
    
    return parser.parse_args()


def main():
    """Main entry point for the test CLI."""
    args = parse_arguments()
    
    # Load configuration
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        config = AppConfig.load_from_file(config_path)
        
        # Set the absolute path to the model registry file in the same directory as the config
        config.models_file = str(config_path.parent / "model_registry.json")
        
        # Load model registry
        config.load_model_registry()
        
        # Check if a model name was specified
        if args.model_name:
            selected_model = config.model_registry.get_model(args.model_name)
            if selected_model:
                config.model = selected_model
                print(f"Using model '{args.model_name}' from registry")
            else:
                logger.error(f"Model '{args.model_name}' not found in registry")
                print(f"Available models: {config.model_registry.list_models()}")
                sys.exit(1)
    else:
        # Create configuration from arguments
        model_config = ModelConfig(
            model_id=args.model,
            engine_type=args.engine,
            quantization=None if args.quantization == "none" else args.quantization,
        )
        
        config = AppConfig(model=model_config)
    
    # Print basic information
    # Get basic GPU info
    try:
        gpu_info = get_gpu_info()
        print(f"\nGPU Info: {gpu_info}\n")
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
    
    try:
        # Create model
        model = ModelFactory.create_model(config.model)
        
        # Get model architecture information
        model_architecture = getattr(config.model, 'model_architecture', 'causal_lm')
        
        # Print model information
        print(f"\nLoading model: {config.model.model_id}")
        print(f"Engine: {config.model.engine_type}")
        print(f"Quantization: {config.model.quantization}")
        print(f"Model architecture: {model_architecture}")
        print(f"Model dtype: {config.model.dtype}")
        
        # Load the model
        model.load_model()
        
        # Chat loop
        print("\n----- Model loaded successfully! -----")
        print("Enter your prompts (type 'exit' to quit):")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                print("\nGenerating response...")
                
                # Handle differently based on model architecture
                if model_architecture == 'seq2seq_lm':
                    print(f"\nUsing sequence-to-sequence model: {config.model.model_id}")
                    response = model.generate(user_input)
                    print(f"\nModel: {response}")
                else:
                    # Use chat interface for causal LMs
                    response = model.chat([{"role": "user", "content": user_input}])
                    print(f"\nModel: {response}")
                
                # Show memory usage
                mem_stats = model.get_memory_stats()
                print("\nGPU Memory:")
                for k, v in mem_stats.items():
                    print(f"  {k}: {v:.2f} GB")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError during generation: {e}")
                print("\nPlease try another prompt or model.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Provide more helpful error messages for common issues
        if "not found in the Hugging Face" in str(e):
            print("\nThe specified model was not found. Please check the model ID for typos.")
        elif "trust_remote_code" in str(e):
            print("\nThis model requires setting trust_remote_code=True. This is automatically handled by our system, but may indicate the model requires special handling.")
        elif "seq2seq" in str(e) and "causal" in str(e):
            print("\nThere might be an architecture mismatch. If you're using a sequence-to-sequence model (like T5 or ByT5), make sure the system correctly identifies it as such.")
        elif "CUDA out of memory" in str(e):
            print("\nThe model is too large for your GPU memory. Try using a smaller model or a stronger quantization setting.")
            
        # Print memory usage if available
        try:
            mem_stats = get_gpu_memory_stats()
            print("\nCurrent GPU Memory:")
            for k, v in mem_stats.items():
                print(f"  {k}: {v:.2f} GB")
        except:
            pass

if __name__ == "__main__":
    main()
