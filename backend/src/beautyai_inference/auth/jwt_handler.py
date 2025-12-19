"""
JWT token creation and verification.

Uses python-jose for JWT handling with HS256 algorithm.
"""

import os
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from jose import jwt, JWTError, ExpiredSignatureError

logger = logging.getLogger(__name__)

# Configuration from environment
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))  # 1 hour
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))  # 7 days


class TokenType(str, Enum):
    """Type of JWT token."""
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class JWTPayload:
    """Decoded JWT payload structure."""
    user_id: int
    email: str
    token_type: TokenType
    exp: datetime
    iat: datetime
    
    @classmethod
    def from_dict(cls, data: dict) -> "JWTPayload":
        """Create JWTPayload from decoded token dict."""
        return cls(
            user_id=data.get("sub"),
            email=data.get("email", ""),
            token_type=TokenType(data.get("type", "access")),
            exp=datetime.fromtimestamp(data.get("exp", 0)),
            iat=datetime.fromtimestamp(data.get("iat", 0)),
        )


def create_access_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT access token.
    
    Args:
        user_id: User ID to encode in token
        email: User email to encode in token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": TokenType.ACCESS.value,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.debug(f"Created access token for user {user_id}, expires at {expire}")
    return token


def create_refresh_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT refresh token.
    
    Args:
        user_id: User ID to encode in token
        email: User email to encode in token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": TokenType.REFRESH.value,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.debug(f"Created refresh token for user {user_id}, expires at {expire}")
    return token


def decode_token(token: str) -> Optional[dict]:
    """
    Decode a JWT token without verification.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload dict or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"Failed to decode token: {e}")
        return None


def verify_token(token: str, expected_type: TokenType = TokenType.ACCESS) -> Optional[JWTPayload]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        expected_type: Expected token type (access or refresh)
        
    Returns:
        JWTPayload if valid, None if invalid or expired
        
    Raises:
        JWTError: If token is malformed
        ExpiredSignatureError: If token has expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Verify token type
        token_type = payload.get("type")
        if token_type != expected_type.value:
            logger.warning(f"Token type mismatch: expected {expected_type.value}, got {token_type}")
            return None
        
        # Verify required fields
        if not payload.get("sub"):
            logger.warning("Token missing 'sub' (user_id) claim")
            return None
        
        return JWTPayload.from_dict(payload)
        
    except ExpiredSignatureError:
        logger.debug("Token has expired")
        return None
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None
