#!/usr/bin/env python3
"""
Unified CLI for BeautyAI - Single entry point for all model management and inference operations.

This unified CLI consolidates all functionality from:
- beautyai-models (model registry management)
- beautyai-manage (model lifecycle management) 
- beautyai-chat (interactive chat interface)
- beautyai-test (simple model testing)
- beautyai-benchmark (performance benchmarking)

Usage:
    beautyai-manage model list                 # List models in registry
    beautyai-manage model add --name my-model  # Add model to registry
    beautyai-manage system load my-model       # Load model into memory
    beautyai-manage run chat                   # Start interactive chat
    beautyai-manage run test                   # Run model test
    beautyai-manage run benchmark              # Run performance benchmark
    beautyai-manage config show                # Show current configuration
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Callable, Any

# Import service classes (to be created)
from .services.model_registry_service import ModelRegistryService
from .services.lifecycle_service import LifecycleService
from .services.inference_service import InferenceService
from .services.config_service import ConfigService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedCLI:
    """Unified command-line interface for BeautyAI."""
    
    def __init__(self):
        self.model_registry_service = ModelRegistryService()
        self.lifecycle_service = LifecycleService()
        self.inference_service = InferenceService()
        self.config_service = ConfigService()
        
        # Command routing table
        self.command_map: Dict[str, Dict[str, Callable]] = {
            'model': {
                'list': self.model_registry_service.list_models,
                'add': self.model_registry_service.add_model,
                'show': self.model_registry_service.show_model,
                'update': self.model_registry_service.update_model,
                'remove': self.model_registry_service.remove_model,
                'set-default': self.model_registry_service.set_default_model,
            },
            'system': {
                'load': self.lifecycle_service.load_model,
                'unload': self.lifecycle_service.unload_model,
                'unload-all': self.lifecycle_service.unload_all_models,
                'list-loaded': self.lifecycle_service.list_loaded_models,
                'status': self.lifecycle_service.show_status,
                'clear-cache': self.lifecycle_service.clear_cache,
            },
            'run': {
                'chat': self.inference_service.start_chat,
                'test': self.inference_service.run_test,
                'benchmark': self.inference_service.run_benchmark,
            },
            'config': {
                'show': self.config_service.show_config,
                'set': self.config_service.set_config,
                'reset': self.config_service.reset_config,
            },
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            prog='beautyai-manage',
            description='Unified CLI for BeautyAI model management and inference',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_help_examples()
        )
        
        # Global options
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        
        parser.add_argument(
            '--models-file',
            type=str,
            help='Path to model registry file'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version='BeautyAI CLI v1.0.0'
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(
            dest='command_group',
            help='Command groups',
            metavar='COMMAND'
        )
        
        # Model management commands
        self._add_model_commands(subparsers)
        
        # System lifecycle commands
        self._add_system_commands(subparsers)
        
        # Inference commands
        self._add_run_commands(subparsers)
        
        # Configuration commands
        self._add_config_commands(subparsers)
        
        return parser

    def _add_model_commands(self, subparsers):
        """Add model registry management commands."""
        model_parser = subparsers.add_parser(
            'model',
            help='Model registry management',
            description='Manage model configurations in the registry'
        )
        model_subparsers = model_parser.add_subparsers(
            dest='model_command',
            help='Model commands'
        )
        
        # List models
        list_parser = model_subparsers.add_parser('list', help='List all models in registry')
        
        # Show model details
        show_parser = model_subparsers.add_parser('show', help='Show model details')
        show_parser.add_argument('name', help='Model name')
        
        # Add model
        add_parser = model_subparsers.add_parser('add', help='Add new model configuration')
        add_parser.add_argument('--name', required=True, help='Model name')
        add_parser.add_argument('--model-id', required=True, help='Model ID (e.g., Qwen/Qwen3-14B)')
        add_parser.add_argument('--engine', choices=['transformers', 'vllm'], 
                               default='transformers', help='Inference engine')
        add_parser.add_argument('--quantization', choices=['4bit', '8bit', 'awq', 'squeezellm', 'none'],
                               default='4bit', help='Quantization method')
        add_parser.add_argument('--dtype', default='float16', help='Data type')
        add_parser.add_argument('--description', help='Model description')
        add_parser.add_argument('--default', action='store_true', help='Set as default model')
        
        # Update model
        update_parser = model_subparsers.add_parser('update', help='Update model configuration')
        update_parser.add_argument('name', help='Model name')
        update_parser.add_argument('--model-id', help='Model ID')
        update_parser.add_argument('--engine', choices=['transformers', 'vllm'], help='Inference engine')
        update_parser.add_argument('--quantization', choices=['4bit', '8bit', 'awq', 'squeezellm', 'none'],
                                  help='Quantization method')
        update_parser.add_argument('--dtype', help='Data type')
        update_parser.add_argument('--description', help='Model description')
        update_parser.add_argument('--default', action='store_true', help='Set as default model')
        
        # Remove model
        remove_parser = model_subparsers.add_parser('remove', help='Remove model from registry')
        remove_parser.add_argument('name', help='Model name')
        remove_parser.add_argument('--clear-cache', action='store_true', 
                                  help='Also clear model cache from disk')
        
        # Set default model
        default_parser = model_subparsers.add_parser('set-default', help='Set default model')
        default_parser.add_argument('name', help='Model name')

    def _add_system_commands(self, subparsers):
        """Add system lifecycle management commands."""
        system_parser = subparsers.add_parser(
            'system',
            help='Model lifecycle management',
            description='Manage models in memory and system resources'
        )
        system_subparsers = system_parser.add_subparsers(
            dest='system_command',
            help='System commands'
        )
        
        # Load model
        load_parser = system_subparsers.add_parser('load', help='Load model into memory')
        load_parser.add_argument('name', help='Model name')
        
        # Unload model
        unload_parser = system_subparsers.add_parser('unload', help='Unload model from memory')
        unload_parser.add_argument('name', help='Model name')
        
        # Unload all models
        unload_all_parser = system_subparsers.add_parser('unload-all', help='Unload all models')
        
        # List loaded models
        list_loaded_parser = system_subparsers.add_parser('list-loaded', help='List loaded models')
        
        # System status
        status_parser = system_subparsers.add_parser('status', help='Show system status')
        
        # Clear cache
        clear_cache_parser = system_subparsers.add_parser('clear-cache', help='Clear model cache')
        clear_cache_parser.add_argument('name', help='Model name')

    def _add_run_commands(self, subparsers):
        """Add inference operation commands."""
        run_parser = subparsers.add_parser(
            'run',
            help='Inference operations',
            description='Run inference operations like chat, test, and benchmark'
        )
        run_subparsers = run_parser.add_subparsers(
            dest='run_command',
            help='Inference commands'
        )
        
        # Chat interface
        chat_parser = run_subparsers.add_parser('chat', help='Start interactive chat')
        chat_parser.add_argument('--model', help='Model ID to use')
        chat_parser.add_argument('--model-name', help='Model name from registry')
        chat_parser.add_argument('--engine', choices=['transformers', 'vllm'], help='Inference engine')
        chat_parser.add_argument('--quantization', choices=['4bit', '8bit', 'awq', 'squeezellm', 'none'],
                                help='Quantization method')
        chat_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
        chat_parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum tokens to generate')
        
        # Test interface
        test_parser = run_subparsers.add_parser('test', help='Run model test')
        test_parser.add_argument('--model', help='Model ID to use')
        test_parser.add_argument('--model-name', help='Model name from registry')
        test_parser.add_argument('--engine', choices=['transformers', 'vllm'], help='Inference engine')
        test_parser.add_argument('--quantization', choices=['4bit', '8bit', 'awq', 'squeezellm', 'none'],
                                help='Quantization method')
        
        # Benchmark interface
        benchmark_parser = run_subparsers.add_parser('benchmark', help='Run performance benchmark')
        benchmark_parser.add_argument('--model', help='Model ID to use')
        benchmark_parser.add_argument('--model-name', help='Model name from registry')
        benchmark_parser.add_argument('--engine', choices=['transformers', 'vllm'], help='Inference engine')
        benchmark_parser.add_argument('--quantization', choices=['4bit', '8bit', 'awq', 'squeezellm', 'none'],
                                     help='Quantization method')
        benchmark_parser.add_argument('--input-lengths', default='10,100,1000',
                                     help='Comma-separated input lengths')
        benchmark_parser.add_argument('--output-length', type=int, default=200,
                                     help='Output length for benchmark')
        benchmark_parser.add_argument('--output-file', help='Save results to file')
        benchmark_parser.add_argument('--save-model', action='store_true',
                                     help='Save model configuration to registry')

    def _add_config_commands(self, subparsers):
        """Add configuration management commands."""
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management',
            description='Manage application configuration'
        )
        config_subparsers = config_parser.add_subparsers(
            dest='config_command',
            help='Configuration commands'
        )
        
        # Show configuration
        show_parser = config_subparsers.add_parser('show', help='Show current configuration')
        
        # Set configuration
        set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')
        
        # Reset configuration
        reset_parser = config_subparsers.add_parser('reset', help='Reset to default configuration')
        reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset')

    def _get_help_examples(self) -> str:
        """Get help examples for the CLI."""
        return """
Examples:
  # Model registry management
  beautyai-manage model list
  beautyai-manage model add --name my-qwen --model-id Qwen/Qwen3-14B
  beautyai-manage model show my-qwen
  beautyai-manage model set-default my-qwen
  
  # System lifecycle management
  beautyai-manage system load my-qwen
  beautyai-manage system status
  beautyai-manage system unload my-qwen
  
  # Inference operations
  beautyai-manage run chat --model-name my-qwen
  beautyai-manage run test --model Qwen/Qwen3-14B
  beautyai-manage run benchmark --model-name my-qwen --output-file results.json
  
  # Configuration management
  beautyai-manage config show
  beautyai-manage config set default_engine vllm

For backward compatibility, old commands still work:
  beautyai-models list        -> beautyai-manage model list
  beautyai-manage load model  -> beautyai-manage system load model
  beautyai-chat              -> beautyai-manage run chat
  beautyai-test              -> beautyai-manage run test
  beautyai-benchmark         -> beautyai-manage run benchmark
"""

    def route_command(self, args: argparse.Namespace) -> int:
        """Route command to appropriate service."""
        try:
            # Set up global configuration
            self._setup_global_config(args)
            
            # Route to appropriate command group
            if args.command_group == 'model':
                return self._route_model_command(args)
            elif args.command_group == 'system':
                return self._route_system_command(args)
            elif args.command_group == 'run':
                return self._route_run_command(args)
            elif args.command_group == 'config':
                return self._route_config_command(args)
            else:
                print("No command specified. Use --help to see available commands.")
                return 1
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _setup_global_config(self, args: argparse.Namespace):
        """Set up global configuration for all services."""
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Pass configuration to all services
        config_data = {
            'config_file': args.config,
            'models_file': args.models_file,
            'verbose': args.verbose,
        }
        
        self.model_registry_service.configure(config_data)
        self.lifecycle_service.configure(config_data)
        self.inference_service.configure(config_data)
        self.config_service.configure(config_data)

    def _route_model_command(self, args: argparse.Namespace) -> int:
        """Route model registry commands."""
        command = args.model_command
        if not command:
            print("No model command specified. Use 'beautyai-manage model --help' for options.")
            return 1
        
        handler = self.command_map['model'].get(command)
        if handler:
            return handler(args)
        else:
            print(f"Unknown model command: {command}")
            return 1

    def _route_system_command(self, args: argparse.Namespace) -> int:
        """Route system lifecycle commands."""
        command = args.system_command
        if not command:
            print("No system command specified. Use 'beautyai-manage system --help' for options.")
            return 1
        
        handler = self.command_map['system'].get(command)
        if handler:
            return handler(args)
        else:
            print(f"Unknown system command: {command}")
            return 1

    def _route_run_command(self, args: argparse.Namespace) -> int:
        """Route inference operation commands."""
        command = args.run_command
        if not command:
            print("No run command specified. Use 'beautyai-manage run --help' for options.")
            return 1
        
        handler = self.command_map['run'].get(command)
        if handler:
            return handler(args)
        else:
            print(f"Unknown run command: {command}")
            return 1

    def _route_config_command(self, args: argparse.Namespace) -> int:
        """Route configuration management commands."""
        command = args.config_command
        if not command:
            print("No config command specified. Use 'beautyai-manage config --help' for options.")
            return 1
        
        handler = self.command_map['config'].get(command)
        if handler:
            return handler(args)
        else:
            print(f"Unknown config command: {command}")
            return 1


def main():
    """Main entry point for the unified CLI."""
    cli = UnifiedCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    return cli.route_command(args)


if __name__ == "__main__":
    sys.exit(main())
