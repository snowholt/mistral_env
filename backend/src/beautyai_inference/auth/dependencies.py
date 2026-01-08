"""
FastAPI authentication dependencies.

Provides dependency injection for protected endpoints including:
- Regular user authentication via JWT
- Guest user authentication via access token
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database.connection import get_db
from ..database.models import User, GuestUser
from .jwt_handler import verify_token, TokenType, JWTPayload

logger = logging.getLogger(__name__)

# OAuth2 scheme for bearer token extraction
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/whatsapp/auth/login",
    auto_error=True
)

# Optional version that doesn't raise on missing token
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl="/api/v1/whatsapp/auth/login",
    auto_error=False
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.
    
    Extracts and validates JWT from Authorization header,
    then fetches the user from database.
    
    Args:
        token: JWT access token from Authorization header
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException 401: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify token
    payload = verify_token(token, TokenType.ACCESS)
    if payload is None:
        logger.warning("Invalid or expired access token")
        raise credentials_exception
    
    # Fetch user from database
    result = await db.execute(
        select(User).where(User.id == payload.user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        logger.warning(f"User {payload.user_id} from token not found in database")
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get the current active user.
    
    Same as get_current_user but also checks that user is active.
    For guest users, also checks expiration and usage limits.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        Active User object
        
    Raises:
        HTTPException 403: If user is not active or guest limits reached
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    # Unified Guest Check
    if current_user.is_guest():
        if current_user.is_expired():
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Demo access has expired. Please upgrade your plan."
            )
        if current_user.is_limit_reached():
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Demo conversation limit reached. Please upgrade your plan."
            )

    return current_user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Dependency to optionally get the current user.
    
    Returns None if no token provided or token is invalid.
    Useful for endpoints that work both authenticated and anonymously.
    
    Args:
        token: Optional JWT access token
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not token:
        return None
    
    payload = verify_token(token, TokenType.ACCESS)
    if payload is None:
        return None
    
    result = await db.execute(
        select(User).where(User.id == payload.user_id)
    )
    return result.scalar_one_or_none()


# ============================================
# Guest User Authentication
# ============================================

async def get_guest_user_by_token(
    token: str,
    db: AsyncSession
) -> Optional[GuestUser]:
    """
    Get guest user by access token.
    
    Args:
        token: Guest access token
        db: Database session
        
    Returns:
        GuestUser object if found and valid, None otherwise
    """
    result = await db.execute(
        select(GuestUser).where(GuestUser.access_token == token)
    )
    guest_user = result.scalar_one_or_none()
    
    if guest_user and guest_user.can_access_demo():
        return guest_user
    
    return None


async def get_current_guest_user(
    token: str,
    db: AsyncSession = Depends(get_db)
) -> GuestUser:
    """
    Dependency to get the current authenticated guest user.
    
    Used for guest-only endpoints (demo access).
    
    Args:
        token: Guest access token from custom header or query param
        db: Database session
        
    Returns:
        GuestUser object
        
    Raises:
        HTTPException 401: If token is invalid
        HTTPException 403: If guest access has expired or limits reached
    """
    guest_user = await get_guest_user_by_token(token, db)
    
    if guest_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired guest access token"
        )
    
    if not guest_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guest access has been disabled by administrator"
        )
    
    if guest_user.is_expired():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Demo access has expired. Your trial period has ended. Please contact us to upgrade."
        )
    
    if guest_user.is_limit_reached():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Usage limit reached. You have used all {guest_user.max_conversations} demo conversations. Please contact us to upgrade."
        )
    
    return guest_user
