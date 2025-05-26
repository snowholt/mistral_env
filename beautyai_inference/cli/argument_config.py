#!/usr/bin/env python3
"""
Standardized argument configuration for BeautyAI CLI commands.

This module provides consistent argument patterns, global options, validation,
and auto-completion support across all CLI operations.
"""
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field

# Import auto-completion support
try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ArgumentDefinition:
    """Definition for a standardized CLI argument."""
    name: str
    short_name: Optional[str] = None
    arg_type: type = str
    default: Any = None
    choices: Optional[List[str]] = None
    required: bool = False
    help_text: str = ""
    action: Optional[str] = None
    nargs: Optional[Union[str, int]] = None
    metavar: Optional[str] = None
    dest: Optional[str] = None
    validator: Optional[Callable[[Any], bool]] = None
    completion_options: Optional[List[str]] = None
    completer: Optional[Callable] = None  # Custom argcomplete completer function


@dataclass 
class ArgumentGroup:
    """Group of related arguments."""
    name: str
    description: str
    arguments: List[ArgumentDefinition] = field(default_factory=list)


# Auto-completion helper functions
def config_file_completer(prefix, parsed_args, **kwargs):
    """Completer for configuration files."""
    config_extensions = ['.json', '.yaml', '.yml']
    candidates = []
    
    # Look in current directory and config directory
    search_dirs = [Path('.'), Path('./config'), Path('./beautyai_inference/config')]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for file_path in search_dir.glob('*'):
                if file_path.is_file() and file_path.suffix in config_extensions:
                    candidates.append(str(file_path))
    
    return [c for c in candidates if c.startswith(prefix)]


def model_registry_completer(prefix, parsed_args, **kwargs):
    """Completer for model registry files."""
    registry_files = ['model_registry.json', 'models.json', 'registry.json']
    candidates = []
    
    # Look in current directory and config directory
    search_dirs = [Path('.'), Path('./config'), Path('./beautyai_inference/config')]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for filename in registry_files:
                file_path = search_dir / filename
                if file_path.exists():
                    candidates.append(str(file_path))
    
    return [c for c in candidates if c.startswith(prefix)]


def model_name_completer(prefix, parsed_args, **kwargs):
    """Completer for model names from registry."""
    try:
        from ..config.config_manager import AppConfig
        
        # Try to load model registry
        config_file = getattr(parsed_args, 'config', None)
        models_file = getattr(parsed_args, 'models_file', None)
        
        try:
            config = AppConfig(config_file=config_file, models_file=models_file)
            model_names = list(config.models.keys())
            return [name for name in model_names if name.startswith(prefix)]
        except Exception:
            # Fallback to common model names
            return [name for name in ['qwen-14b', 'mistral-7b', 'default'] if name.startswith(prefix)]
    except ImportError:
        return []


def model_id_completer(prefix, parsed_args, **kwargs):
    """Completer for Hugging Face model IDs."""
    common_models = [
        'Qwen/Qwen3-14B',
        'Qwen/Qwen3-7B', 
        'mistralai/Mistral-7B-Instruct-v0.2',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'microsoft/DialoGPT-medium',
        'google/flan-t5-base',
        'google/flan-t5-large'
    ]
    
    return [model for model in common_models if model.startswith(prefix)]


class StandardizedArguments:
    """
    Central repository for standardized CLI arguments.
    
    This class provides consistent argument definitions that can be reused
    across all CLI commands, ensuring backward compatibility and consistency.
    """
    
    # Global options that apply to all commands
    GLOBAL_ARGUMENTS = [
        ArgumentDefinition(
            name="--config",
            arg_type=str,
            help_text="Path to configuration file",
            validator=lambda x: Path(x).exists() if x else True,
            completion_options=["config.json", "default_config.json"],
            completer=config_file_completer
        ),
        ArgumentDefinition(
            name="--models-file", 
            arg_type=str,
            help_text="Path to model registry file",
            validator=lambda x: Path(x).exists() if x else True,
            completion_options=["model_registry.json", "models.json"],
            completer=model_registry_completer
        ),
        ArgumentDefinition(
            name="--verbose",
            short_name="-v",
            action="store_true",
            help_text="Enable verbose logging"
        ),
        ArgumentDefinition(
            name="--quiet",
            short_name="-q", 
            action="store_true",
            help_text="Suppress non-essential output"
        ),
        ArgumentDefinition(
            name="--log-level",
            arg_type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help_text="Set logging level (default: INFO)"
        ),
        ArgumentDefinition(
            name="--no-color",
            action="store_true",
            help_text="Disable colored output"
        ),
        ArgumentDefinition(
            name="--version",
            action="version",
            help_text="Show version information"
        )
    ]
    
    # Model selection arguments
    MODEL_SELECTION_GROUP = ArgumentGroup(
        name="Model Selection",
        description="Options for selecting and configuring models",
        arguments=[
            ArgumentDefinition(
                name="--model",
                arg_type=str,
                help_text="Model ID to use (e.g., Qwen/Qwen3-14B)",
                completion_options=["Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.2"],
                completer=model_id_completer
            ),
            ArgumentDefinition(
                name="--model-name",
                arg_type=str, 
                help_text="Name of model from registry to use",
                validator=lambda x: True,  # Will be validated against registry
                completer=model_name_completer
            ),
            ArgumentDefinition(
                name="--engine",
                arg_type=str,
                choices=["transformers", "vllm"],
                default="transformers",
                help_text="Inference engine to use (default: transformers)"
            ),
            ArgumentDefinition(
                name="--quantization",
                arg_type=str,
                choices=["4bit", "8bit", "awq", "squeezellm", "none"],
                default="4bit", 
                help_text="Quantization method (default: 4bit)"
            ),
            ArgumentDefinition(
                name="--dtype",
                arg_type=str,
                choices=["float16", "float32", "bfloat16"],
                default="float16",
                help_text="Data type for model weights (default: float16)"
            )
        ]
    )
    
    # Generation parameters
    GENERATION_GROUP = ArgumentGroup(
        name="Generation Parameters",
        description="Parameters for text generation and inference",
        arguments=[
            ArgumentDefinition(
                name="--max-tokens",
                arg_type=int,
                default=512,
                help_text="Maximum number of tokens to generate (default: 512)",
                validator=lambda x: x > 0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--temperature",
                arg_type=float,
                default=0.7,
                help_text="Sampling temperature (default: 0.7)",
                validator=lambda x: 0.0 <= x <= 2.0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--top-p",
                arg_type=float,
                default=0.95,
                help_text="Top-p (nucleus) sampling parameter (default: 0.95)",
                validator=lambda x: 0.0 <= x <= 1.0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--top-k",
                arg_type=int,
                help_text="Top-k sampling parameter",
                validator=lambda x: x > 0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--repetition-penalty",
                arg_type=float,
                default=1.0,
                help_text="Repetition penalty (default: 1.0)",
                validator=lambda x: x > 0.0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--do-sample",
                action="store_true",
                help_text="Enable sampling (vs greedy decoding)"
            ),
            ArgumentDefinition(
                name="--stream",
                action="store_true", 
                help_text="Enable streaming output"
            )
        ]
    )
    
    # System configuration
    SYSTEM_GROUP = ArgumentGroup(
        name="System Configuration", 
        description="System-level configuration options",
        arguments=[
            ArgumentDefinition(
                name="--gpu-memory-utilization",
                arg_type=float,
                default=0.9,
                help_text="GPU memory utilization for vLLM (default: 0.9)",
                validator=lambda x: 0.1 <= x <= 1.0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--tensor-parallel-size",
                arg_type=int,
                default=1,
                help_text="Tensor parallel size for vLLM (default: 1)",
                validator=lambda x: x > 0 if x is not None else True
            ),
            ArgumentDefinition(
                name="--force-cpu",
                action="store_true",
                help_text="Force CPU-only inference"
            ),
            ArgumentDefinition(
                name="--trust-remote-code",
                action="store_true",
                help_text="Trust remote code for model loading"
            )
        ]
    )
    
    # Output and formatting
    OUTPUT_GROUP = ArgumentGroup(
        name="Output and Formatting",
        description="Options for output formatting and saving",
        arguments=[
            ArgumentDefinition(
                name="--output-file",
                arg_type=str,
                help_text="Path to save output/results as JSON",
                validator=lambda x: Path(x).parent.exists() if x else True
            ),
            ArgumentDefinition(
                name="--format",
                arg_type=str,
                choices=["json", "yaml", "table", "plain"],
                default="table",
                help_text="Output format (default: table)"
            ),
            ArgumentDefinition(
                name="--save-model",
                action="store_true",
                help_text="Save current model configuration to registry"
            )
        ]
    )


class ArgumentValidator:
    """Validates CLI arguments and provides helpful error messages."""
    
    @staticmethod
    def validate_argument(arg_def: ArgumentDefinition, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate an argument value against its definition.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None and not arg_def.required:
            return True, None
            
        if arg_def.required and value is None:
            return False, f"Argument {arg_def.name} is required"
        
        # Type validation
        if arg_def.arg_type and value is not None:
            try:
                if arg_def.arg_type != type(value):
                    arg_def.arg_type(value)
            except (ValueError, TypeError):
                return False, f"Argument {arg_def.name} must be of type {arg_def.arg_type.__name__}"
        
        # Choices validation
        if arg_def.choices and value not in arg_def.choices:
            return False, f"Argument {arg_def.name} must be one of: {', '.join(arg_def.choices)}"
        
        # Custom validator
        if arg_def.validator and not arg_def.validator(value):
            return False, f"Argument {arg_def.name} failed validation"
        
        return True, None
    
    @staticmethod
    def validate_all_arguments(args: argparse.Namespace, argument_groups: List[ArgumentGroup]) -> List[str]:
        """
        Validate all arguments in the namespace.
        
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        
        for group in argument_groups:
            for arg_def in group.arguments:
                # Get argument value from namespace
                attr_name = arg_def.dest or arg_def.name.lstrip('-').replace('-', '_')
                value = getattr(args, attr_name, None)
                
                is_valid, error_msg = ArgumentValidator.validate_argument(arg_def, value)
                if not is_valid:
                    errors.append(error_msg)
        
        return errors


class StandardizedArgumentParser:
    """
    Enhanced argument parser with standardized arguments and validation.
    """
    
    def __init__(self, 
                 prog_name: str,
                 description: str,
                 add_global_args: bool = True,
                 formatter_class=argparse.RawDescriptionHelpFormatter):
        """
        Initialize standardized argument parser.
        
        Args:
            prog_name: Program name
            description: Program description
            add_global_args: Whether to add global arguments
            formatter_class: Argument parser formatter class
        """
        self.parser = argparse.ArgumentParser(
            prog=prog_name,
            description=description,
            formatter_class=formatter_class
        )
        
        self.argument_groups: List[ArgumentGroup] = []
        
        if add_global_args:
            self.add_global_arguments()
    
    def add_global_arguments(self):
        """Add standard global arguments to parser."""
        for arg_def in StandardizedArguments.GLOBAL_ARGUMENTS:
            self._add_argument(arg_def)
    
    def add_argument_group(self, group: ArgumentGroup):
        """Add an argument group to the parser."""
        parser_group = self.parser.add_argument_group(group.name, group.description)
        
        for arg_def in group.arguments:
            self._add_argument_to_group(parser_group, arg_def)
        
        self.argument_groups.append(group)
    
    def add_custom_argument(self, arg_def: ArgumentDefinition, group_name: Optional[str] = None):
        """Add a custom argument to parser."""
        if group_name:
            # Find existing group or create new one
            group = None
            for g in self.argument_groups:
                if g.name == group_name:
                    group = g
                    break
            
            if not group:
                group = ArgumentGroup(group_name, f"{group_name} options")
                self.argument_groups.append(group)
            
            group.arguments.append(arg_def)
            
            # Find parser group
            parser_group = None
            for action_group in self.parser._action_groups:
                if action_group.title == group_name:
                    parser_group = action_group
                    break
            
            if not parser_group:
                parser_group = self.parser.add_argument_group(group_name)
            
            self._add_argument_to_group(parser_group, arg_def)
        else:
            self._add_argument(arg_def)
    
    def _add_argument(self, arg_def: ArgumentDefinition):
        """Add argument to main parser."""
        self._add_argument_to_group(self.parser, arg_def)
    
    def _add_argument_to_group(self, group, arg_def: ArgumentDefinition):
        """Add argument to a specific group."""
        kwargs = {
            'type': arg_def.arg_type if arg_def.action not in ['store_true', 'store_false', 'version'] else None,
            'default': arg_def.default,
            'help': arg_def.help_text,
        }
        
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        if arg_def.choices:
            kwargs['choices'] = arg_def.choices
        if arg_def.required:
            kwargs['required'] = arg_def.required
        if arg_def.action:
            kwargs['action'] = arg_def.action
            if arg_def.action == 'version':
                kwargs['version'] = 'BeautyAI CLI v1.0.0'
        if arg_def.nargs:
            kwargs['nargs'] = arg_def.nargs
        if arg_def.metavar:
            kwargs['metavar'] = arg_def.metavar
        if arg_def.dest:
            kwargs['dest'] = arg_def.dest
        
        # Don't pass completer as kwarg to avoid TypeError when argcomplete is not available
        
        # Add argument with or without short name
        if arg_def.short_name:
            action = group.add_argument(arg_def.short_name, arg_def.name, **kwargs)
        else:
            action = group.add_argument(arg_def.name, **kwargs)
        
        # Set completer on the action object if argcomplete is available
        if ARGCOMPLETE_AVAILABLE and arg_def.completer:
            action.completer = arg_def.completer
        elif ARGCOMPLETE_AVAILABLE and not arg_def.completer and arg_def.completion_options:
            action.completer = lambda prefix, **kwargs: [opt for opt in arg_def.completion_options if opt.startswith(prefix)]
    
    def parse_args(self, args=None):
        """Parse arguments with validation and auto-completion support."""
        # Enable auto-completion if available
        if ARGCOMPLETE_AVAILABLE:
            try:
                import argcomplete
                argcomplete.autocomplete(self.parser)
            except Exception as e:
                logger.debug(f"Auto-completion setup failed: {e}")
        
        parsed_args = self.parser.parse_args(args)
        
        # Validate arguments
        errors = ArgumentValidator.validate_all_arguments(parsed_args, self.argument_groups)
        
        if errors:
            print("❌ Argument validation errors:")
            for error in errors:
                print(f"  - {error}")
            self.parser.exit(1)
        
        # Set up logging based on arguments
        self._setup_logging(parsed_args)
        
        return parsed_args
    
    def _setup_logging(self, args: argparse.Namespace):
        """Set up logging based on parsed arguments."""
        log_level = getattr(args, 'log_level', 'INFO')
        verbose = getattr(args, 'verbose', False)
        quiet = getattr(args, 'quiet', False)
        
        if verbose:
            log_level = 'DEBUG'
        elif quiet:
            log_level = 'WARNING'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(levelname)s: %(message)s'
        )


def create_model_management_parser() -> StandardizedArgumentParser:
    """Create parser for model management commands."""
    parser = StandardizedArgumentParser(
        prog_name="beautyai model",
        description="Manage model configurations in the registry"
    )
    
    parser.add_argument_group(StandardizedArguments.MODEL_SELECTION_GROUP)
    parser.add_argument_group(StandardizedArguments.OUTPUT_GROUP)
    
    return parser


def create_inference_parser() -> StandardizedArgumentParser:
    """Create parser for inference commands."""
    parser = StandardizedArgumentParser(
        prog_name="beautyai run",
        description="Run inference operations with BeautyAI models"
    )
    
    parser.add_argument_group(StandardizedArguments.MODEL_SELECTION_GROUP)
    parser.add_argument_group(StandardizedArguments.GENERATION_GROUP)
    parser.add_argument_group(StandardizedArguments.SYSTEM_GROUP)
    parser.add_argument_group(StandardizedArguments.OUTPUT_GROUP)
    
    return parser


def create_system_parser() -> StandardizedArgumentParser:
    """Create parser for system management commands."""
    parser = StandardizedArgumentParser(
        prog_name="beautyai system",
        description="Manage model lifecycle and system resources"
    )
    
    parser.add_argument_group(StandardizedArguments.MODEL_SELECTION_GROUP)
    parser.add_argument_group(StandardizedArguments.SYSTEM_GROUP)
    parser.add_argument_group(StandardizedArguments.OUTPUT_GROUP)
    
    return parser


# Backward compatibility helpers
def get_legacy_model_args() -> List[ArgumentDefinition]:
    """Get model arguments in legacy format for backward compatibility."""
    return StandardizedArguments.MODEL_SELECTION_GROUP.arguments


def get_legacy_generation_args() -> List[ArgumentDefinition]:
    """Get generation arguments in legacy format for backward compatibility.""" 
    return StandardizedArguments.GENERATION_GROUP.arguments


def add_backward_compatible_args(parser: argparse.ArgumentParser, 
                                include_model: bool = True,
                                include_generation: bool = True,
                                include_system: bool = False):
    """
    Add backward compatible arguments to existing parser.
    
    This function helps maintain compatibility with existing CLI scripts
    while gradually migrating to the standardized system.
    """
    if include_model:
        for arg_def in get_legacy_model_args():
            # Convert ArgumentDefinition back to add_argument call
            kwargs = {
                'type': arg_def.arg_type if arg_def.action not in ['store_true', 'store_false'] else None,
                'default': arg_def.default,
                'help': arg_def.help_text,
            }
            
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            if arg_def.choices:
                kwargs['choices'] = arg_def.choices
            if arg_def.required:
                kwargs['required'] = arg_def.required
            if arg_def.action:
                kwargs['action'] = arg_def.action
            
            # Don't pass completer as kwarg to avoid TypeError when argcomplete is not available
            
            try:
                if arg_def.short_name:
                    action = parser.add_argument(arg_def.short_name, arg_def.name, **kwargs)
                else:
                    action = parser.add_argument(arg_def.name, **kwargs)
                
                # Set completer on the action object if argcomplete is available
                if ARGCOMPLETE_AVAILABLE and arg_def.completer:
                    action.completer = arg_def.completer
                elif ARGCOMPLETE_AVAILABLE and not arg_def.completer and arg_def.completion_options:
                    action.completer = lambda prefix, **kwargs: [opt for opt in arg_def.completion_options if opt.startswith(prefix)]
            except argparse.ArgumentError:
                # Argument already exists, skip
                pass
    
    if include_generation:
        for arg_def in get_legacy_generation_args():
            kwargs = {
                'type': arg_def.arg_type if arg_def.action not in ['store_true', 'store_false'] else None,
                'default': arg_def.default,
                'help': arg_def.help_text,
            }
            
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            if arg_def.choices:
                kwargs['choices'] = arg_def.choices
            if arg_def.action:
                kwargs['action'] = arg_def.action
            
            # Don't pass completer as kwarg to avoid TypeError when argcomplete is not available
            
            try:
                if arg_def.short_name:
                    action = parser.add_argument(arg_def.short_name, arg_def.name, **kwargs)
                else:
                    action = parser.add_argument(arg_def.name, **kwargs)
                
                # Set completer on the action object if argcomplete is available
                if ARGCOMPLETE_AVAILABLE and arg_def.completer:
                    action.completer = arg_def.completer
                elif ARGCOMPLETE_AVAILABLE and not arg_def.completer and arg_def.completion_options:
                    action.completer = lambda prefix, **kwargs: [opt for opt in arg_def.completion_options if opt.startswith(prefix)]
            except argparse.ArgumentError:
                # Argument already exists, skip
                pass
    
    if include_system:
        for arg_def in StandardizedArguments.SYSTEM_GROUP.arguments:
            kwargs = {
                'type': arg_def.arg_type if arg_def.action not in ['store_true', 'store_false'] else None,
                'default': arg_def.default,
                'help': arg_def.help_text,
            }
            
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            if arg_def.choices:
                kwargs['choices'] = arg_def.choices
            if arg_def.action:
                kwargs['action'] = arg_def.action
            
            # Don't pass completer as kwarg to avoid TypeError when argcomplete is not available
            
            try:
                if arg_def.short_name:
                    action = parser.add_argument(arg_def.short_name, arg_def.name, **kwargs)
                else:
                    action = parser.add_argument(arg_def.name, **kwargs)
                
                # Set completer on the action object if argcomplete is available
                if ARGCOMPLETE_AVAILABLE and arg_def.completer:
                    action.completer = arg_def.completer
                elif ARGCOMPLETE_AVAILABLE and not arg_def.completer and arg_def.completion_options:
                    action.completer = lambda prefix, **kwargs: [opt for opt in arg_def.completion_options if opt.startswith(prefix)]
            except argparse.ArgumentError:
                # Argument already exists, skip
                pass
