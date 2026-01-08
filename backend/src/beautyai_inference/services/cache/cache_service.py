"""
High-level caching service for application data.

Provides typed caching methods for common use cases with automatic serialization.
"""

import logging
from typing import Optional, Any, TypeVar, Type, Callable
from datetime import timedelta
import json
from functools import wraps

from pydantic import BaseModel

from .redis_client import RedisClient, get_redis

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheService:
    """
    Application-level caching service.
    
    Provides high-level caching patterns for:
    - User sessions
    - API responses
    - Computed data
    - Temporary tokens
    """
    
    # Default TTLs for different cache types
    TTL_SHORT = timedelta(minutes=5)
    TTL_MEDIUM = timedelta(minutes=30)
    TTL_LONG = timedelta(hours=2)
    TTL_DAY = timedelta(days=1)
    
    def __init__(self, redis: RedisClient):
        self.redis = redis
    
    # ========================================================================
    # User Session Cache
    # ========================================================================
    
    async def cache_user_session(
        self,
        user_id: int,
        session_data: dict,
        ttl: timedelta = TTL_LONG,
    ) -> bool:
        """Cache user session data (permissions, preferences, etc.)."""
        key = f"session:user:{user_id}"
        return await self.redis.set_json(key, session_data, expire=ttl)
    
    async def get_user_session(self, user_id: int) -> Optional[dict]:
        """Get cached user session data."""
        key = f"session:user:{user_id}"
        return await self.redis.get_json(key)
    
    async def invalidate_user_session(self, user_id: int) -> bool:
        """Invalidate user session cache (on logout, role change, etc.)."""
        key = f"session:user:{user_id}"
        deleted = await self.redis.delete(key)
        return deleted > 0
    
    # ========================================================================
    # Customer Data Cache
    # ========================================================================
    
    async def cache_customer_config(
        self,
        customer_id: int,
        config: dict,
        ttl: timedelta = TTL_MEDIUM,
    ) -> bool:
        """Cache customer configuration (widget settings, agent config, etc.)."""
        key = f"customer:config:{customer_id}"
        return await self.redis.set_json(key, config, expire=ttl)
    
    async def get_customer_config(self, customer_id: int) -> Optional[dict]:
        """Get cached customer configuration."""
        key = f"customer:config:{customer_id}"
        return await self.redis.get_json(key)
    
    async def invalidate_customer_config(self, customer_id: int) -> bool:
        """Invalidate customer config cache (on settings change)."""
        key = f"customer:config:{customer_id}"
        deleted = await self.redis.delete(key)
        return deleted > 0
    
    # ========================================================================
    # API Response Cache
    # ========================================================================
    
    async def cache_api_response(
        self,
        cache_key: str,
        response: Any,
        ttl: timedelta = TTL_SHORT,
    ) -> bool:
        """Cache API response data."""
        key = f"api:response:{cache_key}"
        return await self.redis.set_json(key, response, expire=ttl)
    
    async def get_api_response(self, cache_key: str) -> Optional[Any]:
        """Get cached API response."""
        key = f"api:response:{cache_key}"
        return await self.redis.get_json(key)
    
    # ========================================================================
    # Temporary Token Cache
    # ========================================================================
    
    async def store_token(
        self,
        token_type: str,
        token: str,
        data: dict,
        ttl: timedelta,
    ) -> bool:
        """Store temporary token with associated data (verification, reset, etc.)."""
        key = f"token:{token_type}:{token}"
        return await self.redis.set_json(key, data, expire=ttl)
    
    async def get_token_data(
        self,
        token_type: str,
        token: str,
    ) -> Optional[dict]:
        """Get data associated with temporary token."""
        key = f"token:{token_type}:{token}"
        return await self.redis.get_json(key)
    
    async def consume_token(
        self,
        token_type: str,
        token: str,
    ) -> Optional[dict]:
        """Get token data and delete it (single-use tokens)."""
        data = await self.get_token_data(token_type, token)
        if data:
            key = f"token:{token_type}:{token}"
            await self.redis.delete(key)
        return data
    
    # ========================================================================
    # Widget Token Validation Cache
    # ========================================================================
    
    async def cache_widget_token(
        self,
        token_hash: str,
        customer_id: int,
        settings: dict,
        ttl: timedelta = TTL_MEDIUM,
    ) -> bool:
        """Cache validated widget token data."""
        key = f"widget:token:{token_hash}"
        data = {"customer_id": customer_id, "settings": settings}
        return await self.redis.set_json(key, data, expire=ttl)
    
    async def get_widget_token(self, token_hash: str) -> Optional[dict]:
        """Get cached widget token data."""
        key = f"widget:token:{token_hash}"
        return await self.redis.get_json(key)
    
    # ========================================================================
    # Usage Tracking
    # ========================================================================
    
    async def increment_usage(
        self,
        customer_id: int,
        metric: str,
        amount: int = 1,
    ) -> int:
        """Increment usage counter for a customer metric."""
        # Use current month as the time bucket
        from datetime import datetime
        month_key = datetime.utcnow().strftime("%Y-%m")
        key = f"usage:{customer_id}:{metric}:{month_key}"
        
        # Increment and ensure key expires after 35 days (cleanup buffer)
        for _ in range(amount):
            await self.redis.incr(key)
        
        # Set expiration if this is a new key
        ttl = await self.redis.ttl(key)
        if ttl < 0:
            await self.redis.expire(key, 35 * 24 * 60 * 60)  # 35 days
        
        return await self.redis.get_int(key)
    
    async def get_usage(
        self,
        customer_id: int,
        metric: str,
        month_key: Optional[str] = None,
    ) -> int:
        """Get usage count for a customer metric."""
        from datetime import datetime
        month_key = month_key or datetime.utcnow().strftime("%Y-%m")
        key = f"usage:{customer_id}:{metric}:{month_key}"
        return await self.redis.get_int(key)
    
    # ========================================================================
    # Pydantic Model Caching
    # ========================================================================
    
    async def cache_model(
        self,
        key: str,
        model: BaseModel,
        ttl: timedelta = TTL_MEDIUM,
    ) -> bool:
        """Cache a Pydantic model instance."""
        return await self.redis.set_json(key, model.model_dump(), expire=ttl)
    
    async def get_model(
        self,
        key: str,
        model_class: Type[T],
    ) -> Optional[T]:
        """Get cached Pydantic model instance."""
        data = await self.redis.get_json(key)
        if data is None:
            return None
        try:
            return model_class.model_validate(data)
        except Exception:
            return None
    
    # ========================================================================
    # Lock Pattern (for preventing race conditions)
    # ========================================================================
    
    async def acquire_lock(
        self,
        lock_name: str,
        ttl: int = 30,
    ) -> bool:
        """
        Try to acquire a distributed lock.
        
        Returns True if lock was acquired, False if already held.
        """
        key = f"lock:{lock_name}"
        # SET NX (only if not exists) with expiration
        if not self.redis._client:
            return True  # No Redis = no locking (development mode)
        
        result = await self.redis._client.set(
            self.redis._key(key),
            "1",
            nx=True,
            ex=ttl,
        )
        return result is not None
    
    async def release_lock(self, lock_name: str) -> bool:
        """Release a distributed lock."""
        key = f"lock:{lock_name}"
        deleted = await self.redis.delete(key)
        return deleted > 0


def cached(
    key_template: str,
    ttl: timedelta = CacheService.TTL_MEDIUM,
):
    """
    Decorator for caching function results.
    
    Usage:
        @cached("user:{user_id}:profile", ttl=timedelta(minutes=30))
        async def get_user_profile(user_id: int, cache: CacheService):
            # This will only run if cache miss
            return await expensive_computation(user_id)
    
    Note: Function must accept `cache: CacheService` as a parameter.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find cache service in arguments
            cache: Optional[CacheService] = kwargs.get("cache")
            if cache is None:
                # Try to find in positional args
                for arg in args:
                    if isinstance(arg, CacheService):
                        cache = arg
                        break
            
            if cache is None:
                # No cache available, just run function
                return await func(*args, **kwargs)
            
            # Build cache key from template
            cache_key = key_template.format(**kwargs)
            
            # Try cache first
            cached_result = await cache.redis.get_json(f"cache:{cache_key}")
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Cache miss - compute and store
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)
            
            await cache.redis.set_json(f"cache:{cache_key}", result, expire=ttl)
            
            return result
        
        return wrapper
    return decorator
