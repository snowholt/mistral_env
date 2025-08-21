"""
Shared ModelManager Service

This module provides a shared singleton instance of ModelManager to ensure
all services use the same model instances for efficiency and consistency.
"""

from typing import Optional
from ...core.model_manager import ModelManager


class ModelManagerService:
    """
    Shared singleton service for ModelManager access.
    
    This ensures all inference services use the same ModelManager instance,
    preventing duplicate model loading and improving memory efficiency.
    """
    
    _instance: Optional['ModelManagerService'] = None
    _model_manager: Optional[ModelManager] = None
    
    def __new__(cls) -> 'ModelManagerService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model_manager is None:
            self._model_manager = ModelManager()
    
    @property
    def model_manager(self) -> ModelManager:
        """Get the shared ModelManager instance."""
        return self._model_manager
    
    def get_model_manager(self) -> ModelManager:
        """Get the shared ModelManager instance (alternative method)."""
        return self._model_manager


# Global instance for easy access
_shared_service = ModelManagerService()


def get_shared_model_manager() -> ModelManager:
    """
    Get the shared ModelManager instance.
    
    Returns:
        ModelManager: The shared singleton instance
    """
    return _shared_service.model_manager