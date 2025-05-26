#!/usr/bin/env python3
"""
Chat CLI for interactive conversation with BeautyAI models.
"""
import argparse
import logging
import sys
from pathlib import Path

from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_factory import ModelFactory
from ..utils.memory_utils import get_gpu_info, clear_terminal_screen
from .argument_config import add_backward_compatible_args, ArgumentValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with BeautyAI model")
    
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


def print_assistant_response(token):
    """Print a token from the assistant."""
    print(token, end="", flush=True)


def main():
    """Main entry point for the chat CLI."""
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
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
        
        config = AppConfig(model=model_config)
    
    # Print basic information
    gpu_info = get_gpu_info()
    if not gpu_info["is_available"]:
        logger.error("No GPU available, this script requires a CUDA-capable GPU.")
        sys.exit(1)
    
    device_name = gpu_info["device_name"]
    vram_gb = gpu_info["total_memory"]
    
    print(f"Using GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Loading model: {config.model.model_id}")
    print("This may take a few minutes...")
    
    # Create and load model
    model = ModelFactory.create_model(config.model)
    model.load_model()
    
    print("Model loaded successfully!")
    
    # Initialize conversation
    conversation = []
    
    print("\n" + "="*50)
    print(f"Interactive Chat with {config.model.model_id}")
    print("Type 'exit', 'quit', or 'q' to end the chat")
    print("Type 'clear' to start a new conversation")
    print("Type 'system <message>' to set a system message")
    print("=" * 50 + "\n")
    
    # Start chat loop
    system_message = None
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Ending chat session. Goodbye!")
                break
            
            elif user_input.lower() == "clear":
                conversation = []
                system_message = None
                clear_terminal_screen()
                print("Conversation history cleared.")
                continue
            
            elif user_input.strip() == "":
                continue
            
            elif user_input.lower().startswith("system "):
                system_message = user_input[7:]  # Remove "system " prefix
                print(f"System message set: {system_message}")
                
                # Update conversation with system message
                if conversation and conversation[0]["role"] == "system":
                    conversation[0] = {"role": "system", "content": system_message}
                else:
                    conversation.insert(0, {"role": "system", "content": system_message})
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Generate and print response
            print("\nAssistant: ", end="", flush=True)
            
            response = model.chat_stream(
                conversation, 
                callback=print_assistant_response,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            print("\n")
            
            # Add assistant response to conversation history
            conversation.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Continuing with a new prompt...")


if __name__ == "__main__":
    main()
