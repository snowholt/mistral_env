"""
Redis caching and rate limiting services.
"""

# Import everything explicitly to avoid circular import issues
import sys

try:
    from .redis_client import RedisClient, get_redis
    _redis_ok = True
except Exception as e:
    _redis_ok = False
    print(f"Warning: Could not import from redis_client: {e}", file=sys.stderr)
    RedisClient = None
    get_redis = None

try:
    from .rate_limiter import RateLimiter, rate_limit_dependency, rate_limit_auth
    _rate_limiter_ok = True
except Exception as e:
    _rate_limiter_ok = False
    print(f"Warning: Could not import from rate_limiter: {e}", file=sys.stderr)
    RateLimiter = None
    rate_limit_dependency = None
    rate_limit_auth = None

try:
    from .cache_service import CacheService
    _cache_service_ok = True
except Exception as e:
    _cache_service_ok = False
    print(f"Warning: Could not import from cache_service: {e}", file=sys.stderr)
    CacheService = None

__all__ = [
    "RedisClient",
    "get_redis",
    "RateLimiter",
    "rate_limit_dependency",
    "rate_limit_auth",
    "CacheService",
]
