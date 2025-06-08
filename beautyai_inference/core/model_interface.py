"""
Base class for model implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class ModelInterface(ABC):
    """Abstract base class for all model implementations."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory and free resources."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response in a conversation."""
        pass
    
    @abstractmethod
    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run a benchmark on the model."""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        pass
