"""
Redis-based rate limiting for API endpoints.

Implements sliding window rate limiting with configurable limits per route/user.
"""

import logging
from typing import Optional, Callable
from datetime import datetime
from functools import wraps

from fastapi import Request, HTTPException, status, Depends

from .redis_client import RedisClient, get_redis

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        limit: int,
        window: int,
        retry_after: int
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit of {limit} requests per {window} seconds exceeded",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    
    Supports different limits for:
    - IP address (default)
    - User ID (authenticated)
    - Widget token (chat widget)
    - Custom keys
    """
    
    def __init__(
        self,
        requests: int = 60,
        window: int = 60,
        key_prefix: str = "ratelimit:",
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests: Maximum number of requests allowed
            window: Time window in seconds
            key_prefix: Redis key prefix for this limiter
            key_func: Optional function to extract rate limit key from request
        """
        self.requests = requests
        self.window = window
        self.key_prefix = key_prefix
        self.key_func = key_func or self._default_key_func
    
    @staticmethod
    def _default_key_func(request: Request) -> str:
        """Default key function using client IP."""
        # Try to get real IP from proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP in chain (client IP)
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_redis_key(self, identifier: str) -> str:
        """Build Redis key for rate limiting."""
        # Include current time window in key for sliding window
        window_start = int(datetime.utcnow().timestamp()) // self.window
        return f"{self.key_prefix}{identifier}:{window_start}"
    
    async def is_allowed(
        self,
        redis: RedisClient,
        identifier: str
    ) -> tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit.
        
        Returns:
            Tuple of (allowed, remaining, retry_after)
        """
        key = self._get_redis_key(identifier)
        
        # Increment counter and set expiration atomically
        current = await redis.incr_expire(key, self.window)
        
        remaining = max(0, self.requests - current)
        
        if current > self.requests:
            # Calculate retry-after time
            ttl = await redis.ttl(key)
            retry_after = max(1, ttl) if ttl > 0 else self.window
            return False, 0, retry_after
        
        return True, remaining, 0
    
    async def check(
        self,
        request: Request,
        redis: RedisClient
    ) -> dict:
        """
        Check rate limit for request.
        
        Raises RateLimitExceeded if limit exceeded.
        Returns rate limit info dict otherwise.
        """
        identifier = self.key_func(request)
        allowed, remaining, retry_after = await self.is_allowed(redis, identifier)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {identifier}")
            raise RateLimitExceeded(
                limit=self.requests,
                window=self.window,
                retry_after=retry_after,
            )
        
        return {
            "X-RateLimit-Limit": str(self.requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(datetime.utcnow().timestamp()) + self.window),
        }


# ============================================================================
# Common Rate Limiters
# ============================================================================

# Default API rate limiter: 100 requests per minute per IP
default_limiter = RateLimiter(requests=100, window=60, key_prefix="ratelimit:api:")

# Strict limiter for sensitive endpoints: 10 requests per minute
auth_limiter = RateLimiter(requests=10, window=60, key_prefix="ratelimit:auth:")

# Widget limiter: 60 requests per minute per IP
widget_limiter = RateLimiter(requests=60, window=60, key_prefix="ratelimit:widget:")

# Admin limiter: 200 requests per minute (more lenient for admins)
admin_limiter = RateLimiter(requests=200, window=60, key_prefix="ratelimit:admin:")


# ============================================================================
# FastAPI Dependency
# ============================================================================

def rate_limit_dependency(
    limiter: Optional[RateLimiter] = None,
    requests: int = 60,
    window: int = 60,
):
    """
    Create a FastAPI dependency for rate limiting.
    
    Usage:
        @router.get("/endpoint", dependencies=[Depends(rate_limit_dependency())])
        async def endpoint():
            ...
        
        # Or with custom limits:
        @router.post("/auth", dependencies=[Depends(rate_limit_dependency(requests=5, window=60))])
        async def auth():
            ...
    """
    _limiter = limiter or RateLimiter(requests=requests, window=window)
    
    async def _rate_limit(
        request: Request,
        redis: RedisClient = Depends(get_redis),
    ) -> dict:
        return await _limiter.check(request, redis)
    
    return _rate_limit


# Convenience dependencies for common use cases
async def rate_limit_default(
    request: Request,
    redis: RedisClient = Depends(get_redis),
) -> dict:
    """Default rate limit: 100/minute per IP."""
    return await default_limiter.check(request, redis)


async def rate_limit_auth(
    request: Request,
    redis: RedisClient = Depends(get_redis),
) -> dict:
    """Auth rate limit: 10/minute per IP."""
    return await auth_limiter.check(request, redis)


async def rate_limit_widget(
    request: Request,
    redis: RedisClient = Depends(get_redis),
) -> dict:
    """Widget rate limit: 60/minute per IP."""
    return await widget_limiter.check(request, redis)
