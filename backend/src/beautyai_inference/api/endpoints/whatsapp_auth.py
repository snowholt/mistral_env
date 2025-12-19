"""
WhatsApp Manager Authentication API endpoints.

Provides user registration, login, and token refresh for the SaaS platform.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ...database.connection import get_db
from ...database.models import User, Customer
from ...auth.password import hash_password, verify_password
from ...auth.jwt_handler import create_access_token, create_refresh_token, verify_token, TokenType
from ...auth.dependencies import get_current_user, get_current_active_user

logger = logging.getLogger(__name__)

whatsapp_auth_router = APIRouter(prefix="/api/v1/whatsapp/auth", tags=["whatsapp-auth"])


# ============================================
# Request/Response Models
# ============================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    full_name: str = Field(..., min_length=2, max_length=255)
    business_name: Optional[str] = Field(None, description="Optional business name to create initial customer")


class RegisterResponse(BaseModel):
    """User registration response."""
    success: bool
    message: str
    user_id: int
    customer_id: Optional[int] = None
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    """Login response with tokens."""
    success: bool
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: dict


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class RefreshResponse(BaseModel):
    """Token refresh response."""
    success: bool
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User profile response."""
    id: int
    email: str
    full_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    customers: list


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ============================================
# API Endpoints
# ============================================

@whatsapp_auth_router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a user and optionally an initial customer (business).
    Returns JWT tokens for immediate login.
    """
    logger.info(f"Registration attempt for email: {request.email}")
    
    # Check if email already exists
    existing = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        email=request.email.lower(),
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        is_active=True,
        is_verified=False  # Email verification would be added later
    )
    db.add(user)
    await db.flush()  # Get user.id
    
    # Create initial customer if business name provided
    customer_id = None
    if request.business_name:
        customer = Customer(
            user_id=user.id,
            name=request.business_name,
            email=request.email.lower()
        )
        db.add(customer)
        await db.flush()
        customer_id = customer.id
    
    await db.commit()
    
    # Generate tokens
    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id, user.email)
    
    logger.info(f"User registered successfully: {user.id} ({user.email})")
    
    return RegisterResponse(
        success=True,
        message="Registration successful",
        user_id=user.id,
        customer_id=customer_id,
        access_token=access_token,
        refresh_token=refresh_token
    )


@whatsapp_auth_router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password.
    
    Uses OAuth2 password flow for compatibility with OpenAPI/Swagger UI.
    Returns JWT access and refresh tokens.
    """
    logger.info(f"Login attempt for: {form_data.username}")
    
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == form_data.username.lower())
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Generate tokens
    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id, user.email)
    
    logger.info(f"User logged in: {user.id} ({user.email})")
    
    return LoginResponse(
        success=True,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=3600,  # 1 hour for access token
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_verified": user.is_verified
        }
    )


@whatsapp_auth_router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Returns a new access token if the refresh token is valid.
    """
    payload = verify_token(request.refresh_token, TokenType.REFRESH)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Verify user still exists and is active
    result = await db.execute(
        select(User).where(User.id == payload.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated"
        )
    
    # Generate new access token
    access_token = create_access_token(user.id, user.email)
    
    return RefreshResponse(
        success=True,
        access_token=access_token,
        expires_in=3600
    )


@whatsapp_auth_router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's profile.
    
    Returns user info including associated customers (businesses).
    Protected endpoint - requires valid JWT.
    """
    # Eager load customers if not already loaded
    await db.refresh(current_user, ["customers"])
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        customers=[
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in current_user.customers
        ]
    )


@whatsapp_auth_router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user's password.
    
    Requires current password for verification.
    """
    # Verify current password
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.password_hash = hash_password(request.new_password)
    await db.commit()
    
    logger.info(f"Password changed for user: {current_user.id}")
    
    return {"success": True, "message": "Password changed successfully"}


@whatsapp_auth_router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """
    Logout current user.
    
    Note: With JWT, logout is typically handled client-side by discarding tokens.
    This endpoint exists for API completeness and future token blacklisting.
    """
    logger.info(f"User logged out: {current_user.id}")
    
    # In a production system, you might want to:
    # 1. Add the token to a blacklist (Redis)
    # 2. Invalidate all refresh tokens for the user
    
    return {"success": True, "message": "Logged out successfully"}
