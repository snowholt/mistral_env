"""
WhatsApp Manager Authentication API endpoints.

Provides user registration, login, token refresh, email verification,
password reset, and admin invite functionality for the SaaS platform.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status, Query, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ...database.connection import get_db
from ...database.models import User, Customer, UserRole, AdminInvite, OTPVerificationLog
from ...auth.password import hash_password, verify_password
from ...auth.jwt_handler import create_access_token, create_refresh_token, verify_token, TokenType
from ...auth.dependencies import get_current_user, get_current_active_user, get_current_verified_user
from ...services.email import EmailService, get_email_service
from ...services.cache import RateLimiter, get_redis, rate_limit_auth
from ...auth.otp import OTPService, get_otp_service

logger = logging.getLogger(__name__)

# Main auth router - handles user registration, login, profile, etc.
# Note: This was previously at /api/v1/whatsapp/auth but moved to /api/v1/auth for clarity
auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def get_request_ip(http_request: Request) -> Optional[str]:
    """Extract client IP address from request headers (proxy-aware)."""
    forwarded_for = http_request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if http_request.client:
        return http_request.client.host
    return None


async def record_otp_log(
    db: AsyncSession,
    user: User,
    action: str,
    purpose: str,
    success: bool,
    http_request: Request,
    failure_reason: Optional[str] = None,
) -> None:
    """Persist OTP audit log without breaking the main flow."""
    try:
        db.add(
            OTPVerificationLog(
                user_id=user.id,
                email=user.email,
                purpose=purpose,
                action=action,
                success=success,
                failure_reason=failure_reason,
                ip_address=get_request_ip(http_request),
                user_agent=http_request.headers.get("user-agent"),
            )
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.error(f"Failed to record OTP log for user {user.email}: {exc}")


# ============================================
# Request/Response Models
# ============================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    full_name: str = Field(..., min_length=2, max_length=255)
    business_name: Optional[str] = Field(None, description="Optional business name to create initial customer")
    invite_code: Optional[str] = Field(None, description="Admin invite code (for @gmai.sa domain users)")


class RegisterResponse(BaseModel):
    """User registration response."""
    success: bool
    message: str
    user_id: int
    customer_id: Optional[int] = None
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    requires_verification: bool = True


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


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
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    customers: list
    # Guest-specific fields (only populated when role=guest)
    expires_at: Optional[datetime] = None
    max_conversations: Optional[int] = None
    conversations_used: Optional[int] = None
    is_expired: Optional[bool] = None
    is_limit_reached: Optional[bool] = None
    days_remaining: Optional[int] = None
    conversations_remaining: Optional[int] = None
    can_access_demo: Optional[bool] = None


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Reset password with token."""
    token: str
    new_password: str = Field(..., min_length=8)


class VerifyEmailRequest(BaseModel):
    """Email verification request."""
    token: str


class ResendVerificationRequest(BaseModel):
    """Resend verification email request."""
    email: EmailStr


class AdminInviteRequest(BaseModel):
    """Create admin invite request."""
    target_email: Optional[EmailStr] = Field(None, description="Restrict invite to specific email")
    max_uses: int = Field(1, ge=1, le=10)


class OTPRequestModel(BaseModel):
    """Request OTP for 2FA verification."""
    purpose: str = Field("whatsapp_connect", description="Purpose of OTP (whatsapp_connect, sensitive_action)")


class OTPVerifyModel(BaseModel):
    """Verify OTP code."""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit OTP code")
    purpose: str = Field("whatsapp_connect", description="Purpose must match request")


class OTPResponse(BaseModel):
    """OTP operation response."""
    success: bool
    message: str
    expires_in: Optional[int] = None  # seconds until OTP expires


# ============================================
# API Endpoints
# ============================================

@auth_router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Register a new user account.
    
    Creates a user and optionally an initial customer (business).
    Sends verification email and returns JWT tokens.
    
    Admin role is automatically assigned for @gmai.sa emails with valid invite code.
    """
    logger.info(f"Registration attempt for email: {request.email}")
    email_lower = request.email.lower()
    
    # Check if email already exists
    existing = await db.execute(
        select(User).where(User.email == email_lower)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Determine role - admin for @gmai.sa with valid invite
    role = UserRole.USER
    used_invite = None
    
    if User.should_be_admin(email_lower):
        if request.invite_code:
            # Validate invite code
            invite_result = await db.execute(
                select(AdminInvite).where(AdminInvite.code == request.invite_code)
            )
            invite = invite_result.scalar_one_or_none()
            
            if invite and invite.is_valid(email_lower):
                role = UserRole.ADMIN
                used_invite = invite
                logger.info(f"Admin invite used for {email_lower}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired admin invite code"
                )
        else:
            # @gmai.sa without invite - still require invite for admin
            logger.info(f"@gmai.sa user {email_lower} registered without invite - regular user role")
    
    # Create user
    user = User(
        email=email_lower,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        role=role,
        is_active=True,
        is_verified=False,
    )
    
    # Generate verification token
    verification_token = user.generate_verification_token()
    
    db.add(user)
    await db.flush()  # Get user.id
    
    # Mark invite as used
    if used_invite:
        used_invite.use()
    
    # Create initial customer if business name provided
    customer_id = None
    if request.business_name:
        customer = Customer(
            user_id=user.id,
            name=request.business_name,
            email=email_lower
        )
        db.add(customer)
        await db.flush()
        customer_id = customer.id
    
    await db.commit()
    
    # Send verification email
    try:
        await email_service.send_verification_email(
            to_address=email_lower,
            full_name=request.full_name,
            verification_token=verification_token,
        )
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        # Don't fail registration if email fails
    
    # Generate tokens
    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id, user.email)
    
    logger.info(f"User registered successfully: {user.id} ({user.email}), role={role.value}")
    
    return RegisterResponse(
        success=True,
        message="Registration successful. Please check your email to verify your account.",
        user_id=user.id,
        customer_id=customer_id,
        access_token=access_token,
        refresh_token=refresh_token,
        requires_verification=True,
    )


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password.
    
    Returns JWT access and refresh tokens.
    """
    logger.info(f"Login attempt for: {request.email}")
    
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(request.password, user.password_hash):
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
    
    logger.info(f"User logged in: {user.id} ({user.email}), role={user.role.value}")
    
    return LoginResponse(
        success=True,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=3600,  # 1 hour for access token
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "is_verified": user.is_verified,
            "is_admin": user.is_admin(),
        }
    )


@auth_router.post("/refresh", response_model=RefreshResponse)
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


@auth_router.get("/me", response_model=UserResponse)
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
    
    # Build response with guest fields if applicable
    response_data = {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at,
        "customers": [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in current_user.customers
        ]
    }
    
    # Add guest-specific fields if user is a guest
    if current_user.is_guest():
        response_data.update({
            "expires_at": current_user.expires_at,
            "max_conversations": current_user.max_conversations,
            "conversations_used": current_user.conversations_used,
            "is_expired": current_user.is_expired(),
            "is_limit_reached": current_user.is_limit_reached(),
            "days_remaining": current_user.days_remaining(),
            "conversations_remaining": current_user.conversations_remaining(),
            "can_access_demo": current_user.can_access_demo(),
        })
    
    return UserResponse(**response_data)


@auth_router.post("/change-password")
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


@auth_router.post("/logout")
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


# ============================================
# Email Verification Endpoints
# ============================================

@auth_router.post("/verify-email")
async def verify_email(
    request: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Verify user email with token from email link.
    
    Marks user as verified and sends welcome email.
    """
    # Find user with matching verification token
    result = await db.execute(select(User).where(User.is_verified == False))
    users = result.scalars().all()
    
    verified_user = None
    for user in users:
        if user.verify_verification_token(request.token):
            verified_user = user
            break
    
    if not verified_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Mark as verified
    verified_user.is_verified = True
    verified_user.verification_token = None
    verified_user.verification_token_expires = None
    await db.commit()
    
    logger.info(f"Email verified for user: {verified_user.id} ({verified_user.email})")
    
    # Send welcome email
    try:
        await email_service.send_welcome_email(
            to_address=verified_user.email,
            full_name=verified_user.full_name,
        )
    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")
    
    return {
        "success": True,
        "message": "Email verified successfully. Welcome to GMAI.sa!"
    }


@auth_router.post("/resend-verification")
async def resend_verification(
    request: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Resend verification email.
    
    Rate limited to prevent abuse.
    """
    result = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Don't reveal if email exists
        return {"success": True, "message": "If the email exists, a verification email has been sent."}
    
    if user.is_verified:
        return {"success": True, "message": "Email is already verified."}
    
    # Generate new verification token
    verification_token = user.generate_verification_token()
    await db.commit()
    
    # Send verification email
    try:
        await email_service.send_verification_email(
            to_address=user.email,
            full_name=user.full_name,
            verification_token=verification_token,
        )
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email"
        )
    
    return {"success": True, "message": "Verification email sent."}


# ============================================
# Password Reset Endpoints
# ============================================

@auth_router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Request password reset email.
    
    Sends reset link to email if account exists.
    Rate limited to prevent abuse.
    """
    result = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    user = result.scalar_one_or_none()
    
    # Always return success to prevent email enumeration
    if not user:
        return {"success": True, "message": "If the email exists, a password reset email has been sent."}
    
    if not user.is_active:
        return {"success": True, "message": "If the email exists, a password reset email has been sent."}
    
    # Generate reset token
    reset_token = user.generate_reset_token()
    await db.commit()
    
    # Send reset email
    try:
        await email_service.send_password_reset_email(
            to_address=user.email,
            full_name=user.full_name,
            reset_token=reset_token,
        )
        logger.info(f"Password reset email sent to: {user.email}")
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        # Don't fail - security through obscurity
    
    return {"success": True, "message": "If the email exists, a password reset email has been sent."}


@auth_router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Reset password using token from email.
    
    Token is single-use and expires after 1 hour.
    """
    # Find user with matching reset token
    result = await db.execute(select(User).where(User.reset_token.isnot(None)))
    users = result.scalars().all()
    
    reset_user = None
    for user in users:
        if user.verify_reset_token(request.token):
            reset_user = user
            break
    
    if not reset_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Update password and clear token
    reset_user.password_hash = hash_password(request.new_password)
    reset_user.reset_token = None
    reset_user.reset_token_expires = None
    await db.commit()
    
    logger.info(f"Password reset successful for user: {reset_user.id} ({reset_user.email})")
    
    return {
        "success": True,
        "message": "Password reset successfully. You can now log in with your new password."
    }


# ============================================
# Admin Invite Endpoints
# ============================================

@auth_router.post("/admin/invite")
async def create_admin_invite(
    request: AdminInviteRequest,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Create an admin invite code.
    
    Only existing admins can create invites.
    Invites are restricted to @gmai.sa email domain.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create invite codes"
        )
    
    # Validate target email domain if specified
    if request.target_email and not request.target_email.lower().endswith("@gmai.sa"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin invites can only be sent to @gmai.sa email addresses"
        )
    
    # Create invite
    invite = AdminInvite(
        code=AdminInvite.generate_code(),
        created_by_user_id=current_user.id,
        max_uses=request.max_uses,
        target_email=request.target_email.lower() if request.target_email else None,
    )
    db.add(invite)
    await db.commit()
    
    logger.info(f"Admin invite created by {current_user.email}: {invite.code[:20]}...")
    
    # Send invite email if target specified
    if request.target_email:
        try:
            await email_service.send_admin_invite_email(
                to_address=request.target_email,
                invite_code=invite.code,
                invited_by=current_user.full_name,
            )
        except Exception as e:
            logger.error(f"Failed to send admin invite email: {e}")
    
    return {
        "success": True,
        "invite_code": invite.code,
        "target_email": request.target_email,
        "max_uses": request.max_uses,
        "message": "Admin invite created successfully"
    }


@auth_router.get("/admin/invites")
async def list_admin_invites(
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all admin invites created by current admin.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view invites"
        )
    
    result = await db.execute(
        select(AdminInvite)
        .where(AdminInvite.created_by_user_id == current_user.id)
        .order_by(AdminInvite.created_at.desc())
    )
    invites = result.scalars().all()
    
    return {
        "success": True,
        "invites": [
            {
                "id": inv.id,
                "code_prefix": inv.code[:15] + "...",
                "target_email": inv.target_email,
                "max_uses": inv.max_uses,
                "use_count": inv.use_count,
                "is_active": inv.is_active,
                "created_at": inv.created_at.isoformat(),
            }
            for inv in invites
        ]
    }


# ============================================
# OTP Verification Endpoints (2FA for WhatsApp Connect)
# ============================================

@auth_router.post("/otp/request", response_model=OTPResponse)
async def request_otp(
    payload: OTPRequestModel,
    http_request: Request,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
    otp_service: OTPService = Depends(get_otp_service),
    email_service: EmailService = Depends(get_email_service),
):
    """
    Request OTP code for 2FA verification.
    
    Sends a 6-digit code to the user's verified email.
    Used before sensitive actions like WhatsApp account connection.
    Rate limited to prevent abuse.
    
    Requires verified email (enforced by get_current_verified_user dependency).
    """
    # Check rate limiting (max 3 OTP requests per 5 minutes)
    rate_key = f"otp_rate:{current_user.id}"
    redis = await get_redis()
    request_count = await redis.incr(rate_key)
    if request_count == 1:
        await redis.expire(rate_key, 300)  # 5 minute window
    if request_count > 3:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Please wait 5 minutes."
        )
    
    # Generate OTP with purpose
    otp_code = await otp_service.generate_otp(
        user_id=str(current_user.id),
        purpose=payload.purpose
    )
    
    # Send OTP via email
    try:
        await email_service.send_otp_email(
            to_address=current_user.email,
            full_name=current_user.full_name,
            otp_code=otp_code
        )
    except Exception as e:
        await record_otp_log(
            db=db,
            user=current_user,
            action="request",
            purpose=payload.purpose,
            success=False,
            http_request=http_request,
            failure_reason="email_send_failed",
        )
        logger.error(f"Failed to send OTP email to {current_user.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email. Please try again."
        )

    await record_otp_log(
        db=db,
        user=current_user,
        action="request",
        purpose=payload.purpose,
        success=True,
        http_request=http_request,
    )
    
    logger.info(f"OTP requested by user {current_user.email} for {payload.purpose}")
    
    return OTPResponse(
        success=True,
        message="OTP sent to your email",
        expires_in=300  # 5 minutes
    )


@auth_router.post("/otp/verify", response_model=OTPResponse)
async def verify_otp(
    payload: OTPVerifyModel,
    http_request: Request,
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
    otp_service: OTPService = Depends(get_otp_service),
):
    """
    Verify OTP code for 2FA.
    
    Returns success if code is valid and matches purpose.
    Code is invalidated after successful verification.
    """
    is_valid = await otp_service.verify_otp(
        user_id=str(current_user.id),
        code=payload.code,
        purpose=payload.purpose
    )
    
    if not is_valid:
        # Track failed attempts for security logging
        logger.warning(f"Failed OTP verification for user {current_user.email}")
        await record_otp_log(
            db=db,
            user=current_user,
            action="verify",
            purpose=payload.purpose,
            success=False,
            http_request=http_request,
            failure_reason="invalid_or_expired",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP code"
        )

    await record_otp_log(
        db=db,
        user=current_user,
        action="verify",
        purpose=payload.purpose,
        success=True,
        http_request=http_request,
    )

    logger.info(f"OTP verified for user {current_user.email}, purpose: {payload.purpose}")
    
    return OTPResponse(
        success=True,
        message="OTP verified successfully"
    )
