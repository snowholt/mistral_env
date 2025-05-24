#!/usr/bin/env python3
"""
Test CLI for running inference with Mistral models.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_factory import ModelFactory
from ..utils.memory_utils import get_gpu_info, get_gpu_memory_stats, clear_terminal_screen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Mistral model inference")
    
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
        "--config",
        type=str,
        help="Path to a JSON configuration file",
    )
    
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
    else:
        # Create configuration from arguments
        model_config = ModelConfig(
            model_id=args.model,
            engine_type=args.engine,
            quantization=None if args.quantization == "none" else args.quantization,
        )
        
        config = AppConfig(model=model_config)
    
    # Print basic information
    gpu_info = get_gpu_info()
    if gpu_info["is_available"]:
        print(f"Using GPU: {gpu_info['device_name']}")
        print(f"Total GPU memory: {gpu_info['total_memory']:.2f} GB")
    else:
        print("No GPU available, using CPU")
        return
    
    # Create and load model
    model = ModelFactory.create_model(config.model)
    model.load_model()
    
    print("Model loaded successfully!")
    
    # Test with a simple prompt
    prompt = "Explain quantum computing in simple terms."
    
    print("\n==== Generating response ====")
    print(f"Prompt: {prompt}")
    
    # Generate response
    response = model.generate(prompt)
    
    print("\n==== Model Response ====")
    print(response)
    
    # Test a conversational exchange
    print("\n==== Testing Conversation ====")
    conversation = [
        {"role": "user", "content": "What are three interesting facts about the moon?"}
    ]
    
    response = model.chat(conversation)
    
    print("\nUser: What are three interesting facts about the moon?")
    print(f"Assistant: {response}")
    
    # Add the assistant's response to the conversation
    conversation.append({"role": "assistant", "content": response})
    
    # Add a follow-up question
    conversation.append({"role": "user", "content": "Can humans live on the moon?"})
    
    response = model.chat(conversation)
    
    print("\nUser: Can humans live on the moon?")
    print(f"Assistant: {response}")
    
    # Print memory usage
    memory_stats = model.get_memory_stats()
    print("\n==== Memory Usage Stats ====")
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f} GB")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
