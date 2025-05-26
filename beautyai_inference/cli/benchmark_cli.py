#!/usr/bin/env python3
"""
Benchmark CLI for measuring performance of BeautyAI models.
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_factory import ModelFactory
from ..utils.memory_utils import get_gpu_info, get_system_memory_stats
from .argument_config import add_backward_compatible_args, ArgumentValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark BeautyAI model")
    
    # Add standardized arguments with backward compatibility
    add_backward_compatible_args(
        parser,
        include_model=True,
        include_generation=True,
        include_system=False
    )
    
    # Benchmark specific settings
    benchmark_group = parser.add_argument_group("Benchmark Settings")
    benchmark_group.add_argument(
        "--input-lengths", 
        type=str, 
        default="10,100,1000",
        help="Comma-separated list of input token lengths to test (default: 10,100,1000)"
    )
    
    benchmark_group.add_argument(
        "--output-length",
        type=int,
        default=200,
        help="Number of output tokens to generate (default: 200)",
    )

    benchmark_group.add_argument(
        "--save-model",
        action="store_true",
        help="Save the current model configuration to the registry"
    )
    
    # Enable auto-completion if available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass  # Auto-completion not available
    
    return parser.parse_args()


def create_prompts(input_lengths: List[int]) -> Dict[int, str]:
    """Create prompts with different lengths."""
    prompts = {}
    for length in input_lengths:
        # Create a prompt with approximately the specified number of tokens
        words_needed = max(1, length // 2)  # Roughly 2 tokens per word
        prompt = "I am a language model trained by BeautyAI AI. " * words_needed
        prompts[length] = prompt
    return prompts


def list_available_models(config: AppConfig):
    """List all available models in the registry."""
    print("\n===== Available Models =====")
    models = config.model_registry.list_models()
    default_model = config.model_registry.default_model
    
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


def main():
    """Main entry point for the benchmark CLI."""
    args = parse_arguments()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = AppConfig.load_from_file(config_path)
    
    # Set custom models file if provided
    if args.models_file:
        config.models_file = args.models_file
    
    # Load model registry
    config.load_model_registry()
    
    # If requested to list models, do that and exit
    if args.list_models:
        list_available_models(config)
        sys.exit(0)
    
    # If a model name is provided, try to load from registry
    if args.model_name:
        if not config.switch_model(args.model_name):
            logger.error(f"Model '{args.model_name}' not found in registry.")
            print("\nAvailable models:")
            for name in config.model_registry.list_models():
                print(f"- {name}")
            sys.exit(1)
    
    # If direct model ID is provided, update configuration
    if args.model:
        config.model.model_id = args.model
    
    # Update other model parameters if provided
    if args.engine:
        config.model.engine_type = args.engine
    
    if args.quantization:
        config.model.quantization = None if args.quantization == "none" else args.quantization
    
    # Save model to registry if requested
    if args.save_model:
        # Create a unique name if none exists
        if config.model.name == "default":
            model_short_name = config.model.model_id.split("/")[-1].lower()
            config.model.name = f"{model_short_name}-{config.model.quantization or 'fp16'}"
        
        config.add_model_config(config.model)
        print(f"Model '{config.model.name}' saved to registry.")
    
    # Parse input lengths
    try:
        input_lengths = [int(x) for x in args.input_lengths.split(",")]
    except ValueError:
        logger.error("Invalid input lengths. Should be comma-separated integers.")
        sys.exit(1)
    
    # Print benchmark information
    gpu_info = get_gpu_info()
    if not gpu_info["is_available"]:
        logger.error("No GPU available, this script requires a CUDA-capable GPU.")
        sys.exit(1)
    
    device_name = gpu_info["device_name"]
    vram_gb = gpu_info["total_memory"]
    
    print(f"===== BeautyAI Model Benchmark =====")
    print(f"Model: {config.model.model_id}")
    print(f"Engine: {config.model.engine_type}")
    print(f"GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Quantization: {config.model.quantization}")
    print("=" * 50)
    
    # Create and load model
    model = ModelFactory.create_model(config.model)
    
    # Measure model loading time
    print("\n1. Model Loading Benchmark")
    import time
    start_time = time.time()
    model.load_model()
    loading_time = time.time() - start_time
    print(f"Model loading time: {loading_time:.2f} seconds")
    
    # Memory usage after loading
    print("\n2. Memory Usage After Loading")
    memory_stats = model.get_memory_stats()
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f} GB")
    
    # Inference speed tests with different input lengths
    print("\n3. Inference Speed Benchmarks")
    
    results = []
    prompts = create_prompts(input_lengths)
    
    for input_len, prompt in prompts.items():
        print(f"\nTesting with ~{input_len} input tokens...")
        
        benchmark_result = model.benchmark(
            prompt, 
            max_new_tokens=args.output_length
        )
        
        print(f"Generated {benchmark_result['output_tokens']} tokens in {benchmark_result['inference_time']:.2f} seconds")
        print(f"Speed: {benchmark_result['tokens_per_second']:.2f} tokens/second")
        
        results.append({
            "input_tokens": input_len,
            "output_tokens": benchmark_result['output_tokens'],
            "time": benchmark_result['inference_time'],
            "tokens_per_second": benchmark_result['tokens_per_second']
        })
    
    # System information
    system_info = {
        "gpu": gpu_info,
        "memory": get_system_memory_stats(),
        "model": config.model.model_id,
        "engine": config.model.engine_type,
        "quantization": config.model.quantization,
    }
    
    # Summary
    print("\n===== Benchmark Summary =====")
    print(f"Model: {config.model.model_id}")
    print(f"Engine: {config.model.engine_type}")
    print(f"Quantization: {config.model.quantization}")
    print("\nPerformance by input length:")
    for result in results:
        print(f"Input: ~{result['input_tokens']} tokens, " +
              f"Output: {result['output_tokens']} tokens, " +
              f"Speed: {result['tokens_per_second']:.2f} tokens/sec")
    
    # Save results if output file specified
    if args.output_file:
        output_data = {
            "system_info": system_info,
            "loading_time": loading_time,
            "memory_after_loading": memory_stats,
            "inference_results": results
        }
        
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nBenchmark results saved to {output_path}")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
