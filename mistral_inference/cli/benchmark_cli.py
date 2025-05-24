#!/usr/bin/env python3
"""
Benchmark CLI for measuring performance of Mistral models.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Mistral model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        help="Model ID to use (default: mistralai/Mistral-Small-3.1-24B-Instruct-2503)",
    )
    
    parser.add_argument(
        "--engine", 
        type=str, 
        choices=["transformers", "vllm"], 
        default="transformers",
        help="Inference engine to use (default: transformers)"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "awq", "squeezellm", "none"],
        default="4bit",
        help="Quantization method (default: 4bit)",
    )
    
    parser.add_argument(
        "--input-lengths", 
        type=str, 
        default="10,100,1000",
        help="Comma-separated list of input token lengths to test (default: 10,100,1000)"
    )
    
    parser.add_argument(
        "--output-length",
        type=int,
        default=200,
        help="Number of output tokens to generate (default: 200)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save benchmark results as JSON",
    )
    
    return parser.parse_args()


def create_prompts(input_lengths: List[int]) -> Dict[int, str]:
    """Create prompts with different lengths."""
    prompts = {}
    for length in input_lengths:
        # Create a prompt with approximately the specified number of tokens
        words_needed = max(1, length // 2)  # Roughly 2 tokens per word
        prompt = "I am a language model trained by Mistral AI. " * words_needed
        prompts[length] = prompt
    return prompts


def main():
    """Main entry point for the benchmark CLI."""
    args = parse_arguments()
    
    # Parse input lengths
    try:
        input_lengths = [int(x) for x in args.input_lengths.split(",")]
    except ValueError:
        logger.error("Invalid input lengths. Should be comma-separated integers.")
        sys.exit(1)
    
    # Load configuration
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        config = AppConfig.load_from_file(config_path)
    else:
        # Create configuration from arguments
        model_config = ModelConfig(
            model_id=args.model,
            engine_type=args.engine,
            quantization=None if args.quantization == "none" else args.quantization,
        )
        
        config = AppConfig(model=model_config)
    
    # Print benchmark information
    gpu_info = get_gpu_info()
    if not gpu_info["is_available"]:
        logger.error("No GPU available, this script requires a CUDA-capable GPU.")
        sys.exit(1)
    
    device_name = gpu_info["device_name"]
    vram_gb = gpu_info["total_memory"]
    
    print(f"===== Mistral Model Benchmark =====")
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
