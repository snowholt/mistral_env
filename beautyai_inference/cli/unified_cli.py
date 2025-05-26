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
    beautyai model list                 # List models in registry
    beautyai model add --name my-model  # Add model to registry
    beautyai system load my-model       # Load model into memory
    beautyai run chat                   # Start interactive chat
    beautyai run test                   # Run model test
    beautyai run benchmark              # Run performance benchmark
    beautyai config show                # Show current configuration
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
from .argument_config import (
    StandardizedArgumentParser, 
    StandardizedArguments,
    ArgumentDefinition,
    ArgumentGroup,
    add_backward_compatible_args
)
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
                'save-session': self.inference_service.save_session,
                'load-session': self.inference_service.load_session,
            },
            'config': {
                'show': self.config_service.show_config,
                'set': self.config_service.set_config,
                'reset': self.config_service.reset_config,
                'validate': self.config_service.validate_config,
                'backup': self.config_service.backup_config,
                'restore': self.config_service.restore_config,
                'migrate': self.config_service.migrate_config,
            },
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all subcommands."""
        parser = StandardizedArgumentParser(
            prog_name='beautyai',
            description='Unified CLI for BeautyAI model management and inference',
            add_global_args=True
        ).parser
        
        # Override epilog with examples
        parser.epilog = self._get_help_examples()
        
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
        # Add standardized output formatting args
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name in ['--format', '--output-file']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                if arg_def.default:
                    kwargs['default'] = arg_def.default
                list_parser.add_argument(arg_def.name, **kwargs)
        
        # Show model details
        show_parser = model_subparsers.add_parser('show', help='Show model details')
        show_parser.add_argument('name', help='Model name')
        # Add output formatting
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name in ['--format']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                if arg_def.default:
                    kwargs['default'] = arg_def.default
                show_parser.add_argument(arg_def.name, **kwargs)
        
        # Add model
        add_parser = model_subparsers.add_parser('add', help='Add new model configuration')
        add_parser.add_argument('--name', required=True, help='Model name')
        
        # Add standardized model arguments
        for arg_def in StandardizedArguments.MODEL_SELECTION_GROUP.arguments:
            if arg_def.name == '--model':
                add_parser.add_argument('--model-id', required=True, type=str, 
                                       help='Model ID (e.g., Qwen/Qwen3-14B)')
            elif arg_def.name in ['--engine', '--quantization', '--dtype']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                if arg_def.default:
                    kwargs['default'] = arg_def.default
                add_parser.add_argument(arg_def.name, **kwargs)
        
        add_parser.add_argument('--description', help='Model description')
        add_parser.add_argument('--default', action='store_true', help='Set as default model')
        
        # Update model  
        update_parser = model_subparsers.add_parser('update', help='Update model configuration')
        update_parser.add_argument('name', help='Model name')
        update_parser.add_argument('--model-id', help='Model ID')
        
        # Add standardized model arguments (optional for update)
        for arg_def in StandardizedArguments.MODEL_SELECTION_GROUP.arguments:
            if arg_def.name in ['--engine', '--quantization', '--dtype']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                # Don't set default for update command
                update_parser.add_argument(arg_def.name, **kwargs)
        
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
        # Add system configuration args
        for arg_def in StandardizedArguments.SYSTEM_GROUP.arguments:
            if arg_def.name in ['--gpu-memory-utilization', '--tensor-parallel-size', '--force-cpu']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.action:
                    kwargs['action'] = arg_def.action
                    kwargs.pop('type', None)
                if arg_def.default and not arg_def.action:
                    kwargs['default'] = arg_def.default
                load_parser.add_argument(arg_def.name, **kwargs)
        
        # Unload model
        unload_parser = system_subparsers.add_parser('unload', help='Unload model from memory')
        unload_parser.add_argument('name', help='Model name')
        
        # Unload all models
        unload_all_parser = system_subparsers.add_parser('unload-all', help='Unload all models')
        
        # List loaded models
        list_loaded_parser = system_subparsers.add_parser('list-loaded', help='List loaded models')
        # Add output formatting
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name in ['--format']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                if arg_def.default:
                    kwargs['default'] = arg_def.default
                list_loaded_parser.add_argument(arg_def.name, **kwargs)
        
        # System status
        status_parser = system_subparsers.add_parser('status', help='Show system status')
        # Add output formatting
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name in ['--format']:
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                if arg_def.default:
                    kwargs['default'] = arg_def.default
                status_parser.add_argument(arg_def.name, **kwargs)
        
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
        
        # Helper function to add standardized arguments to a parser
        def add_model_and_generation_args(parser, include_all_generation=True):
            # Model selection arguments
            for arg_def in StandardizedArguments.MODEL_SELECTION_GROUP.arguments:
                if arg_def.name in ['--model', '--model-name', '--engine', '--quantization']:
                    kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                    if arg_def.choices:
                        kwargs['choices'] = arg_def.choices
                    # Don't set defaults for run commands, let services handle defaults
                    parser.add_argument(arg_def.name, **kwargs)
            
            # Generation arguments  
            for arg_def in StandardizedArguments.GENERATION_GROUP.arguments:
                if include_all_generation or arg_def.name in ['--temperature', '--max-tokens', '--top-p', '--top-k', '--repetition-penalty', '--stream']:
                    kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                    if arg_def.action:
                        kwargs['action'] = arg_def.action
                        kwargs.pop('type', None)
                    if arg_def.default and not arg_def.action:
                        kwargs['default'] = arg_def.default
                    parser.add_argument(arg_def.name, **kwargs)
        
        # Chat interface
        chat_parser = run_subparsers.add_parser('chat', help='Start interactive chat')
        add_model_and_generation_args(chat_parser)
        
        # Test interface
        test_parser = run_subparsers.add_parser('test', help='Run model test')
        add_model_and_generation_args(test_parser, include_all_generation=False)
        test_parser.add_argument('--prompt', help='Test prompt')
        
        # Benchmark interface
        benchmark_parser = run_subparsers.add_parser('benchmark', help='Run performance benchmark')
        add_model_and_generation_args(benchmark_parser, include_all_generation=False)
        benchmark_parser.add_argument('--input-lengths', default='10,100,1000',
                                     help='Comma-separated input lengths')
        benchmark_parser.add_argument('--output-length', type=int, default=200,
                                     help='Output length for benchmark')
        benchmark_parser.add_argument('--num-runs', type=int, default=3,
                                     help='Number of benchmark runs per input length')
        # Add output file argument
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name == '--output-file':
                kwargs = {'type': arg_def.arg_type, 'help': arg_def.help_text}
                benchmark_parser.add_argument(arg_def.name, **kwargs)
        
        # Session management - save current session
        save_session_parser = run_subparsers.add_parser('save-session', help='Save chat session to file')
        save_session_parser.add_argument('--session-id', help='Session ID to save')
        for arg_def in StandardizedArguments.OUTPUT_GROUP.arguments:
            if arg_def.name == '--output-file':
                kwargs = {'type': arg_def.arg_type, 'help': 'Output file path for session'}
                save_session_parser.add_argument(arg_def.name, **kwargs)
        
        # Session management - load saved session
        load_session_parser = run_subparsers.add_parser('load-session', help='Load chat session from file')
        load_session_parser.add_argument('--input-file', required=True, help='Session file to load')
        # Add model override arguments
        for arg_def in StandardizedArguments.MODEL_SELECTION_GROUP.arguments:
            if arg_def.name in ['--model-name', '--engine', '--quantization']:
                kwargs = {'type': arg_def.arg_type, 'help': f'Override {arg_def.help_text.lower()}'}
                if arg_def.choices:
                    kwargs['choices'] = arg_def.choices
                load_session_parser.add_argument(arg_def.name, **kwargs)

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
        
        # Validate configuration
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration against schema')
        
        # Backup configuration
        backup_parser = config_subparsers.add_parser('backup', help='Backup configuration files')
        backup_parser.add_argument('--backup-dir', help='Directory to store backups (default: backups)')
        backup_parser.add_argument('--label', help='Label to identify this backup')
        backup_parser.add_argument('--compress', action='store_true', help='Create compressed archive of backups')
        backup_parser.add_argument('--keep-count', type=int, help='Keep only N most recent backups')
        
        # Restore configuration
        restore_parser = config_subparsers.add_parser('restore', help='Restore configuration from backup')
        restore_parser.add_argument('config_file', help='Configuration file to restore from')
        restore_parser.add_argument('--models-file', help='Models file to restore from')
        restore_parser.add_argument('--target-config', help='Target configuration file (defaults to current)')
        restore_parser.add_argument('--no-validate', dest='validate', action='store_false', 
                                   help='Skip validation after restore')
        restore_parser.add_argument('--auto-migrate', action='store_true', 
                                   help='Automatically migrate if restored config is outdated')
        
        # Migrate configuration
        migrate_parser = config_subparsers.add_parser('migrate', help='Migrate configuration to new format')
        migrate_parser.add_argument('--verbose', action='store_true', help='Show detailed migration steps')
        migrate_parser.add_argument('--backup', action='store_true', help='Create backup before migration')

    def _get_help_examples(self) -> str:
        """Get help examples for the CLI."""
        return """
Examples:
  # Model registry management
  beautyai model list
  beautyai model add --name my-qwen --model-id Qwen/Qwen3-14B
  beautyai model show my-qwen
  beautyai model set-default my-qwen
  
  # System lifecycle management
  beautyai system load my-qwen
  beautyai system status
  beautyai system unload my-qwen
  
  # Inference operations
  beautyai run chat --model-name my-qwen
  beautyai run test --model Qwen/Qwen3-14B
  beautyai run benchmark --model-name my-qwen --output-file results.json
  beautyai run save-session --output-file session.json
  beautyai run load-session --input-file session.json
  
  # Configuration management
  beautyai config show
  beautyai config set default_engine vllm
  beautyai config validate
  beautyai config backup --backup-dir my_backups
  beautyai config restore backups/config_20250525_120000.json
  beautyai config migrate --verbose

For backward compatibility, old commands still work:
  beautyai-models list        -> beautyai model list
  beautyai-manage load model  -> beautyai system load model
  beautyai-chat              -> beautyai run chat
  beautyai-test              -> beautyai run test
  beautyai-benchmark         -> beautyai run benchmark
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
