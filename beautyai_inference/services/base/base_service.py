"""
Base service class for BeautyAI services.

This module provides the foundational BaseService class that all other services
inherit from. It handles common functionality like configuration loading,
error handling, and logging setup.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ...config.config_manager import AppConfig

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base class for all BeautyAI services.
    
    Provides common functionality for configuration management, error handling,
    and logging that all services need. This class is designed to be CLI-agnostic
    and suitable for use in future API endpoints.
    """
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.config_data: Dict[str, Any] = {}
        self.config_file_path: Optional[str] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def configure(self, config_data: Dict[str, Any]):
        """
        Configure the service with global settings.
        
        Args:
            config_data: Dictionary containing configuration options including
                        config_file path and models_file path
        """
        self.config_data = config_data
        
        # Load application configuration
        config_file = config_data.get('config_file')
        self.config_file_path = config_file
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                self.config = AppConfig.load_from_file(config_path)
            else:
                self.logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        else:
            # Use default configuration - updated path for new structure
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            if default_config_path.exists():
                self.config = AppConfig.load_from_file(default_config_path)
                # Ensure the models_file path is absolute for the default config
                if self.config.models_file and not Path(self.config.models_file).is_absolute():
                    config_dir = Path(__file__).parent.parent.parent / "config"
                    self.config.models_file = str(config_dir / "model_registry.json")
            else:
                self.logger.warning("No configuration file found, using minimal configuration")
                from ...config.config_manager import ModelConfig
                self.config = AppConfig(model=ModelConfig())
        
        # Set custom models file if provided
        models_file = config_data.get('models_file')
        if models_file:
            self.config.models_file = models_file
        
        # Load model registry
        try:
            self.config.load_model_registry()
        except Exception as e:
            self.logger.warning(f"Could not load model registry: {e}")
    
    def _handle_error(self, error: Exception, message: str) -> int:
        """
        Handle service errors consistently.
        
        Args:
            error: The exception that occurred
            message: User-friendly error message
            
        Returns:
            int: Error exit code (1)
        """
        self.logger.error(f"{message}: {error}")
        if self.config_data.get('verbose'):
            import traceback
            traceback.print_exc()
        print(f"Error: {message}")
        return 1
    
    def _success_message(self, message: str) -> int:
        """
        Print success message and return success code.
        
        Args:
            message: Success message to display
            
        Returns:
            int: Success exit code (0)
        """
        print(message)
        return 0
