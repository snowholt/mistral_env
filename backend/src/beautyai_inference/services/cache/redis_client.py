"""
Redis client with connection pooling for caching and rate limiting.

Configured for production deployment with Alibaba Cloud Redis.
"""

import os
import json
import logging
from urllib.parse import urlparse
from typing import Optional, Any, Union
from datetime import timedelta
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Async Redis client wrapper with connection pooling.
    
    Provides methods for caching, rate limiting, and pub/sub.
    """
    
    _instance: Optional["RedisClient"] = None
    _pool: Optional[ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    
    def __init__(self):
        """Initialize Redis connection settings from environment."""
        self.url = os.getenv("REDIS_URL")
        if self.url:
            parsed = urlparse(self.url)
            self.ssl = parsed.scheme == "rediss"
            self.host = parsed.hostname or "localhost"
            self.port = parsed.port or 6379
            self.password = parsed.password
            self.db = int(parsed.path.lstrip("/") or 0)
        else:
            self.host = os.getenv("REDIS_HOST", "localhost")
            self.port = int(os.getenv("REDIS_PORT", "6379"))
            self.password = os.getenv("REDIS_PASSWORD", None)
            self.db = int(os.getenv("REDIS_DB", "0"))
            self.ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
        
        # Connection pool settings
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        self.socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
        self.socket_connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))
        
        # Key prefix for namespace isolation
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "beautyai:")
    
    @classmethod
    async def get_instance(cls) -> "RedisClient":
        """Get or create singleton Redis client instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.connect()
        return cls._instance
    
    async def connect(self) -> None:
        """Establish Redis connection pool."""
        if self._client is not None:
            return
        
        try:
            # Build connection URL
            if self.url:
                url = self.url
            else:
                protocol = "rediss" if self.ssl else "redis"
                auth = f":{self.password}@" if self.password else ""
                url = f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
            
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                url,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=True,
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info(f"✅ Redis connected: {self.host}:{self.port}/{self.db}")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self._client = None
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connections gracefully."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis disconnected")
    
    def _key(self, key: str) -> str:
        """Add namespace prefix to key."""
        return f"{self.key_prefix}{key}"
    
    # ========================================================================
    # Basic Operations
    # ========================================================================
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self._client:
            return None
        return await self._client.get(self._key(key))
    
    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value with optional expiration (seconds or timedelta)."""
        if not self._client:
            return False
        
        if isinstance(expire, timedelta):
            expire = int(expire.total_seconds())
        
        return await self._client.set(self._key(key), value, ex=expire)
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self._client:
            return 0
        prefixed_keys = [self._key(k) for k in keys]
        return await self._client.delete(*prefixed_keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._client:
            return False
        return await self._client.exists(self._key(key)) > 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration in seconds."""
        if not self._client:
            return False
        return await self._client.expire(self._key(key), seconds)
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL in seconds. Returns -1 if no expiry, -2 if key doesn't exist."""
        if not self._client:
            return -2
        return await self._client.ttl(self._key(key))
    
    # ========================================================================
    # JSON Operations
    # ========================================================================
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get and parse JSON value."""
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value as JSON."""
        return await self.set(key, json.dumps(value), expire)
    
    # ========================================================================
    # Counter Operations (for rate limiting)
    # ========================================================================
    
    async def incr(self, key: str) -> int:
        """Increment counter and return new value."""
        if not self._client:
            return 0
        return await self._client.incr(self._key(key))
    
    async def incr_expire(self, key: str, expire: int) -> int:
        """Increment counter and set expiration atomically."""
        if not self._client:
            return 0
        
        prefixed_key = self._key(key)
        pipeline = self._client.pipeline()
        pipeline.incr(prefixed_key)
        pipeline.expire(prefixed_key, expire)
        results = await pipeline.execute()
        return results[0]  # Return the incremented value
    
    async def get_int(self, key: str) -> int:
        """Get value as integer."""
        value = await self.get(key)
        if value is None:
            return 0
        try:
            return int(value)
        except ValueError:
            return 0
    
    # ========================================================================
    # Hash Operations (for structured caching)
    # ========================================================================
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value."""
        if not self._client:
            return None
        return await self._client.hget(self._key(key), field)
    
    async def hset(self, key: str, field: str, value: str) -> int:
        """Set hash field value."""
        if not self._client:
            return 0
        return await self._client.hset(self._key(key), field, value)
    
    async def hgetall(self, key: str) -> dict:
        """Get all hash fields and values."""
        if not self._client:
            return {}
        return await self._client.hgetall(self._key(key))
    
    async def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields."""
        if not self._client:
            return 0
        return await self._client.hdel(self._key(key), *fields)
    
    # ========================================================================
    # Set Operations (for tracking unique items)
    # ========================================================================
    
    async def sadd(self, key: str, *values: str) -> int:
        """Add values to set."""
        if not self._client:
            return 0
        return await self._client.sadd(self._key(key), *values)
    
    async def sismember(self, key: str, value: str) -> bool:
        """Check if value is in set."""
        if not self._client:
            return False
        return await self._client.sismember(self._key(key), value)
    
    async def smembers(self, key: str) -> set:
        """Get all set members."""
        if not self._client:
            return set()
        return await self._client.smembers(self._key(key))
    
    async def scard(self, key: str) -> int:
        """Get set cardinality (size)."""
        if not self._client:
            return 0
        return await self._client.scard(self._key(key))
    
    # ========================================================================
    # Pub/Sub (for real-time notifications)
    # ========================================================================
    
    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        if not self._client:
            return 0
        return await self._client.publish(self._key(channel), message)
    
    @asynccontextmanager
    async def subscribe(self, *channels: str):
        """Subscribe to channels. Use as async context manager."""
        if not self._client:
            yield None
            return
        
        pubsub = self._client.pubsub()
        try:
            prefixed_channels = [self._key(c) for c in channels]
            await pubsub.subscribe(*prefixed_channels)
            yield pubsub
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health_check(self) -> dict:
        """Check Redis health and return status."""
        if not self._client:
            return {"status": "disconnected", "error": "No client"}
        
        try:
            await self._client.ping()
            info = await self._client.info("memory")
            return {
                "status": "healthy",
                "host": f"{self.host}:{self.port}",
                "used_memory_human": info.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global instance getter for FastAPI dependency injection
async def get_redis() -> RedisClient:
    """
    FastAPI dependency for Redis client.
    
    Usage:
        @router.get("/example")
        async def example(redis: RedisClient = Depends(get_redis)):
            await redis.set("key", "value", expire=60)
    """
    return await RedisClient.get_instance()
