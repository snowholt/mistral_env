"""
OTP (One-Time Password) generation and verification for 2FA.

Used for secure operations like WhatsApp account connection.
Stores OTPs in Redis with 5-minute TTL.
"""

import secrets
import logging
from typing import Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

# OTP Configuration
OTP_LENGTH = 6
OTP_TTL_SECONDS = 300  # 5 minutes
OTP_KEY_PREFIX = "otp:"
OTP_ATTEMPTS_KEY_PREFIX = "otp_attempts:"
MAX_OTP_ATTEMPTS = 5


async def generate_otp(
    user_id: int,
    purpose: str = "whatsapp_connect"
) -> str:
    """
    Generate a 6-digit OTP for a user and store in Redis.
    
    Args:
        user_id: The user ID requesting OTP
        purpose: Purpose of OTP (e.g., "whatsapp_connect", "sensitive_action")
        
    Returns:
        6-digit OTP string
    """
    from ..services.cache import get_redis
    
    # Generate secure 6-digit code
    otp = "".join([str(secrets.randbelow(10)) for _ in range(OTP_LENGTH)])
    
    # Store in Redis with TTL
    redis = await get_redis()
    key = f"{OTP_KEY_PREFIX}{purpose}:{user_id}"
    await redis.set(key, otp, expire=OTP_TTL_SECONDS)
    
    # Reset attempt counter
    attempts_key = f"{OTP_ATTEMPTS_KEY_PREFIX}{purpose}:{user_id}"
    await redis.delete(attempts_key)
    
    logger.info(f"OTP generated for user {user_id} ({purpose})")
    return otp


async def verify_otp(
    user_id: int,
    otp_code: str,
    purpose: str = "whatsapp_connect"
) -> bool:
    """
    Verify an OTP code for a user.
    
    Args:
        user_id: The user ID
        otp_code: The OTP code to verify
        purpose: Purpose of OTP verification
        
    Returns:
        True if OTP is valid, False otherwise
    """
    from ..services.cache import get_redis
    
    redis = await get_redis()
    key = f"{OTP_KEY_PREFIX}{purpose}:{user_id}"
    attempts_key = f"{OTP_ATTEMPTS_KEY_PREFIX}{purpose}:{user_id}"
    
    # Check attempt count
    attempts = await redis.get_int(attempts_key)
    if attempts >= MAX_OTP_ATTEMPTS:
        logger.warning(f"OTP max attempts exceeded for user {user_id} ({purpose})")
        return False
    
    # Increment attempts
    await redis.incr_expire(attempts_key, OTP_TTL_SECONDS)
    
    # Get stored OTP
    stored_otp = await redis.get(key)
    
    if stored_otp is None:
        logger.debug(f"No OTP found for user {user_id} ({purpose})")
        return False
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(stored_otp, otp_code):
        logger.debug(f"OTP mismatch for user {user_id} ({purpose})")
        return False
    
    # OTP is valid - delete it (single use)
    await redis.delete(key, attempts_key)
    logger.info(f"OTP verified successfully for user {user_id} ({purpose})")
    return True


async def invalidate_otp(
    user_id: int,
    purpose: str = "whatsapp_connect"
) -> None:
    """
    Invalidate (delete) an OTP for a user.
    
    Args:
        user_id: The user ID
        purpose: Purpose of OTP
    """
    from ..services.cache import get_redis
    
    redis = await get_redis()
    key = f"{OTP_KEY_PREFIX}{purpose}:{user_id}"
    attempts_key = f"{OTP_ATTEMPTS_KEY_PREFIX}{purpose}:{user_id}"
    await redis.delete(key, attempts_key)
    logger.debug(f"OTP invalidated for user {user_id} ({purpose})")


async def get_otp_ttl(
    user_id: int,
    purpose: str = "whatsapp_connect"
) -> int:
    """
    Get remaining TTL for an OTP.
    
    Args:
        user_id: The user ID
        purpose: Purpose of OTP
        
    Returns:
        Remaining seconds, -1 if no expiry, -2 if key doesn't exist
    """
    from ..services.cache import get_redis
    
    redis = await get_redis()
    key = f"{OTP_KEY_PREFIX}{purpose}:{user_id}"
    return await redis.ttl(key)


async def has_pending_otp(
    user_id: int,
    purpose: str = "whatsapp_connect"
) -> bool:
    """
    Check if user has a pending (unexpired) OTP.
    
    Args:
        user_id: The user ID
        purpose: Purpose of OTP
        
    Returns:
        True if pending OTP exists
    """
    from ..services.cache import get_redis
    
    redis = await get_redis()
    key = f"{OTP_KEY_PREFIX}{purpose}:{user_id}"
    return await redis.exists(key)


# ============================================
# OTPService class for FastAPI dependency injection
# ============================================

class OTPService:
    """
    OTP Service wrapper for FastAPI dependency injection.
    
    Provides class-based access to OTP functions.
    """
    
    async def generate_otp(
        self,
        user_id: str,
        purpose: str = "whatsapp_connect"
    ) -> str:
        """Generate OTP for user."""
        return await generate_otp(int(user_id), purpose)
    
    async def verify_otp(
        self,
        user_id: str,
        code: str,
        purpose: str = "whatsapp_connect"
    ) -> bool:
        """Verify OTP code."""
        return await verify_otp(int(user_id), code, purpose)
    
    async def invalidate_otp(
        self,
        user_id: str,
        purpose: str = "whatsapp_connect"
    ) -> None:
        """Invalidate OTP."""
        await invalidate_otp(int(user_id), purpose)
    
    async def has_pending(
        self,
        user_id: str,
        purpose: str = "whatsapp_connect"
    ) -> bool:
        """Check if pending OTP exists."""
        return await has_pending_otp(int(user_id), purpose)
    
    async def get_ttl(
        self,
        user_id: str,
        purpose: str = "whatsapp_connect"
    ) -> int:
        """Get remaining TTL for OTP."""
        return await get_otp_ttl(int(user_id), purpose)


# Singleton instance
_otp_service: OTPService | None = None


async def get_otp_service() -> OTPService:
    """
    FastAPI dependency to get OTP service instance.
    
    Returns:
        OTPService instance (singleton)
    """
    global _otp_service
    if _otp_service is None:
        _otp_service = OTPService()
    return _otp_service
