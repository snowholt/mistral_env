"""
Core components for model initialization and inference.

Modules:
- model_manager: Singleton model lifecycle management
- persistent_model_manager: Voice service model preloading with LLM pool support
- model_factory: Factory pattern for inference engine creation
- inference_router: LLM pool routing and request tracking
- redis_client: Redis client for distributed state management
- cluster_coordinator: Master-slave cluster coordination
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "get_persistent_model_manager",
    "get_inference_router",
    "get_redis_client",
    "get_cluster_coordinator",
]


def get_persistent_model_manager():
    """Get the persistent model manager instance."""
    from .persistent_model_manager import get_persistent_model_manager as _get
    return _get()


def get_inference_router(**kwargs):
    """Get the inference router instance."""
    from .inference_router import get_inference_router as _get
    return _get(**kwargs)


async def get_redis_client(config=None):
    """Get the Redis client instance."""
    from .redis_client import get_redis_client as _get
    return await _get(config)


async def get_cluster_coordinator(config=None):
    """Get the cluster coordinator instance."""
    from .cluster_coordinator import get_cluster_coordinator as _get
    return await _get(config)
