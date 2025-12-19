"""
FastAPI authentication dependencies.

Provides dependency injection for protected endpoints.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database.connection import get_db
from ..database.models import User
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
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        Active User object
        
    Raises:
        HTTPException 403: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
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
