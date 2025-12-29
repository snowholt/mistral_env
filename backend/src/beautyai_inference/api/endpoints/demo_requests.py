"""
Demo Request API endpoints.

Provides endpoints for:
- Public demo request submission (contact form)
- Admin demo request management (list, approve, reject, update)
- Guest user creation and access control

Public endpoints:
- POST /api/v1/demo-requests - Submit demo request

Admin endpoints:
- GET /api/v1/admin/demo-requests - List all demo requests
- GET /api/v1/admin/demo-requests/{id} - Get demo request details
- PATCH /api/v1/admin/demo-requests/{id}/approve - Approve and grant demo access
- PATCH /api/v1/admin/demo-requests/{id}/reject - Reject demo request
- PATCH /api/v1/admin/demo-requests/{id} - Update demo request (notes, follow-up)
"""

import os
import logging
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, status, Query, Body
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from sqlalchemy.orm import selectinload

from ...database.connection import get_db
from ...database.models import (
    User, DemoRequest, GuestUser, DemoRequestStatus
)
from ...auth.dependencies import get_current_active_user, get_current_guest_user
from ...auth.jwt_handler import create_access_token
from ...services.email import get_email_service

logger = logging.getLogger(__name__)

# Get admin notification email from environment (configurable)
DEMO_ADMIN_EMAIL = os.getenv("DEMO_ADMIN_NOTIFICATION_EMAIL", "admin@gmai.sa")

demo_router = APIRouter(tags=["demo_requests"])
guest_auth_router = APIRouter(prefix="/api/v1/auth/guest", tags=["guest_auth"])


# ============================================
# Admin Auth Dependency
# ============================================

async def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency to require admin role for endpoint access."""
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# ============================================
# Request/Response Models
# ============================================

class DemoRequestCreate(BaseModel):
    """Request body for submitting a demo request."""
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: Optional[str] = Field(None, max_length=50)
    company: Optional[str] = Field(None, max_length=255)
    company_size: Optional[str] = Field(None, max_length=50)
    message: Optional[str] = None


class DemoRequestResponse(BaseModel):
    """Response model for demo request."""
    id: int
    first_name: str
    last_name: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    company_size: Optional[str]
    message: Optional[str]
    status: str
    admin_notes: Optional[str]
    assigned_to_admin_id: Optional[int]
    scheduled_follow_up: Optional[datetime]
    submitted_at: datetime
    reviewed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    # Relationships
    assigned_to_admin: Optional[dict] = None
    guest_user: Optional[dict] = None
    
    class Config:
        from_attributes = True


class DemoRequestSummary(BaseModel):
    """Summary model for listing demo requests."""
    id: int
    full_name: str
    email: str
    company: Optional[str]
    company_size: Optional[str]
    status: str
    submitted_at: datetime
    reviewed_at: Optional[datetime]
    assigned_to_admin_email: Optional[str] = None
    has_guest_user: bool = False


class DemoRequestUpdate(BaseModel):
    """Request body for updating demo request admin notes and follow-up."""
    admin_notes: Optional[str] = None
    assigned_to_admin_id: Optional[int] = None
    scheduled_follow_up: Optional[datetime] = None


class DemoApprovalRequest(BaseModel):
    """Request body for approving demo request and setting limits."""
    demo_duration_days: int = Field(default=7, ge=1, le=90, description="Number of days until demo expires")
    max_conversations: int = Field(default=10, ge=1, le=100, description="Maximum number of conversations allowed")
    admin_notes: Optional[str] = None


class DemoRejectionRequest(BaseModel):
    """Request body for rejecting demo request."""
    admin_notes: Optional[str] = Field(..., min_length=1, description="Reason for rejection")


class GuestUserResponse(BaseModel):
    """Response model for guest user."""
    id: int
    email: str
    access_token: str
    expires_at: datetime
    max_conversations: int
    conversations_used: int
    is_active: bool
    granted_at: datetime
    last_used_at: Optional[datetime]
    days_remaining: int
    conversations_remaining: int
    can_access_demo: bool
    
    class Config:
        from_attributes = True


# ============================================
# Public Endpoints
# ============================================

@demo_router.post(
    "/api/v1/demo-requests",
    response_model=DemoRequestResponse,
    status_code=status.HTTP_201_CREATED
)
async def submit_demo_request(
    request: DemoRequestCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a demo request (public endpoint).
    
    Called from the website contact/request demo form.
    Creates a pending demo request that admins can approve/reject.
    """
    # Check if email already has a pending or approved request
    existing_query = select(DemoRequest).where(
        and_(
            DemoRequest.email == request.email,
            DemoRequest.status.in_([DemoRequestStatus.PENDING, DemoRequestStatus.APPROVED])
        )
    )
    existing = await db.execute(existing_query)
    existing_request = existing.scalar_one_or_none()
    
    if existing_request:
        if existing_request.status == DemoRequestStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You already have a pending demo request. Please wait for admin approval."
            )
        elif existing_request.status == DemoRequestStatus.APPROVED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You already have an approved demo request. Please check your email for access instructions."
            )
    
    # Create new demo request
    demo_request = DemoRequest(
        first_name=request.first_name,
        last_name=request.last_name,
        email=request.email,
        phone=request.phone,
        company=request.company,
        company_size=request.company_size,
        message=request.message,
        status=DemoRequestStatus.PENDING,
    )
    
    db.add(demo_request)
    await db.commit()
    await db.refresh(demo_request)
    
    logger.info(f"New demo request submitted: {demo_request.email} (ID: {demo_request.id})")
    
    # Send confirmation email to requester
    try:
        email_service = await get_email_service()
        await email_service.send_demo_request_confirmation(
            to_address=demo_request.email,
            full_name=demo_request.full_name(),
        )
        logger.info(f"Confirmation email sent to {demo_request.email}")
    except Exception as e:
        logger.error(f"Failed to send confirmation email: {e}")
        # Don't fail the request if email fails
    
    # Send notification email to admin
    try:
        email_service = await get_email_service()
        await email_service.send_demo_request_admin_notification(
            admin_email=DEMO_ADMIN_EMAIL,
            requester_name=demo_request.full_name(),
            requester_email=demo_request.email,
            company=demo_request.company or "",
            company_size=demo_request.company_size or "",
            message=demo_request.message or "",
            demo_request_id=demo_request.id,
        )
        logger.info(f"Admin notification email sent to {DEMO_ADMIN_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to send admin notification email: {e}")
        # Don't fail the request if email fails
    
    return DemoRequestResponse(
        id=demo_request.id,
        first_name=demo_request.first_name,
        last_name=demo_request.last_name,
        email=demo_request.email,
        phone=demo_request.phone,
        company=demo_request.company,
        company_size=demo_request.company_size,
        message=demo_request.message,
        status=demo_request.status.value,
        admin_notes=demo_request.admin_notes,
        assigned_to_admin_id=demo_request.assigned_to_admin_id,
        scheduled_follow_up=demo_request.scheduled_follow_up,
        submitted_at=demo_request.submitted_at,
        reviewed_at=demo_request.reviewed_at,
        created_at=demo_request.created_at,
        updated_at=demo_request.updated_at,
    )


# ============================================
# Admin Endpoints
# ============================================

@demo_router.get(
    "/api/v1/admin/demo-requests",
    response_model=List[DemoRequestSummary]
)
async def list_demo_requests(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[str] = Query(None, regex="^(pending|approved|rejected)$"),
    search: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    List all demo requests (admin only).
    
    Supports filtering by status and searching by name/email.
    """
    query = select(DemoRequest).options(
        selectinload(DemoRequest.assigned_to_admin),
        selectinload(DemoRequest.guest_user)
    )
    
    # Apply status filter
    if status_filter:
        query = query.where(DemoRequest.status == DemoRequestStatus(status_filter))
    
    # Apply search
    if search:
        search_term = f"%{search}%"
        query = query.where(
            (DemoRequest.first_name.ilike(search_term)) |
            (DemoRequest.last_name.ilike(search_term)) |
            (DemoRequest.email.ilike(search_term)) |
            (DemoRequest.company.ilike(search_term))
        )
    
    # Order by submission date (newest first)
    query = query.order_by(desc(DemoRequest.submitted_at))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    demo_requests = result.scalars().all()
    
    # Convert to summary format
    summaries = []
    for dr in demo_requests:
        summaries.append(DemoRequestSummary(
            id=dr.id,
            full_name=dr.full_name(),
            email=dr.email,
            company=dr.company,
            company_size=dr.company_size,
            status=dr.status.value,
            submitted_at=dr.submitted_at,
            reviewed_at=dr.reviewed_at,
            assigned_to_admin_email=dr.assigned_to_admin.email if dr.assigned_to_admin else None,
            has_guest_user=dr.guest_user is not None
        ))
    
    return summaries


@demo_router.get(
    "/api/v1/admin/demo-requests/{request_id}",
    response_model=DemoRequestResponse
)
async def get_demo_request(
    request_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a specific demo request (admin only).
    """
    query = select(DemoRequest).where(DemoRequest.id == request_id).options(
        selectinload(DemoRequest.assigned_to_admin),
        selectinload(DemoRequest.guest_user)
    )
    
    result = await db.execute(query)
    demo_request = result.scalar_one_or_none()
    
    if not demo_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo request with ID {request_id} not found"
        )
    
    # Build response
    response_data = {
        "id": demo_request.id,
        "first_name": demo_request.first_name,
        "last_name": demo_request.last_name,
        "email": demo_request.email,
        "phone": demo_request.phone,
        "company": demo_request.company,
        "company_size": demo_request.company_size,
        "message": demo_request.message,
        "status": demo_request.status.value,
        "admin_notes": demo_request.admin_notes,
        "assigned_to_admin_id": demo_request.assigned_to_admin_id,
        "scheduled_follow_up": demo_request.scheduled_follow_up,
        "submitted_at": demo_request.submitted_at,
        "reviewed_at": demo_request.reviewed_at,
        "created_at": demo_request.created_at,
        "updated_at": demo_request.updated_at,
    }
    
    if demo_request.assigned_to_admin:
        response_data["assigned_to_admin"] = {
            "id": demo_request.assigned_to_admin.id,
            "email": demo_request.assigned_to_admin.email,
            "full_name": demo_request.assigned_to_admin.full_name,
        }
    
    if demo_request.guest_user:
        guest = demo_request.guest_user
        response_data["guest_user"] = {
            "id": guest.id,
            "email": guest.email,
            "expires_at": guest.expires_at,
            "max_conversations": guest.max_conversations,
            "conversations_used": guest.conversations_used,
            "is_active": guest.is_active,
            "days_remaining": guest.days_remaining(),
            "conversations_remaining": guest.conversations_remaining(),
            "can_access_demo": guest.can_access_demo(),
        }
    
    return DemoRequestResponse(**response_data)


@demo_router.patch(
    "/api/v1/admin/demo-requests/{request_id}/approve",
    response_model=GuestUserResponse
)
async def approve_demo_request(
    request_id: int,
    approval: DemoApprovalRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Approve a demo request and create guest user with demo access (admin only).
    
    Creates a GuestUser account with specified limits and sends access email.
    """
    # Get demo request
    query = select(DemoRequest).where(DemoRequest.id == request_id).options(
        selectinload(DemoRequest.guest_user)
    )
    result = await db.execute(query)
    demo_request = result.scalar_one_or_none()
    
    if not demo_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo request with ID {request_id} not found"
        )
    
    if demo_request.status == DemoRequestStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Demo request is already approved"
        )
    
    if demo_request.guest_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Guest user already exists for this demo request"
        )
    
    # Update demo request status
    demo_request.status = DemoRequestStatus.APPROVED
    demo_request.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if approval.admin_notes:
        demo_request.admin_notes = approval.admin_notes
    
    # Create guest user
    access_token = GuestUser.generate_access_token()
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=approval.demo_duration_days)
    
    guest_user = GuestUser(
        demo_request_id=demo_request.id,
        email=demo_request.email,
        access_token=access_token,
        expires_at=expires_at,
        max_conversations=approval.max_conversations,
        conversations_used=0,
        is_active=True,
    )
    
    db.add(guest_user)
    await db.commit()
    await db.refresh(guest_user)
    await db.refresh(demo_request)
    
    logger.info(
        f"Demo request {request_id} approved by admin {admin.email}. "
        f"Guest user created: {guest_user.email} (expires in {approval.demo_duration_days} days, "
        f"{approval.max_conversations} conversations max)"
    )
    
    # Send demo access granted email to guest
    try:
        email_service = await get_email_service()
        await email_service.send_demo_access_granted(
            to_address=guest_user.email,
            full_name=demo_request.full_name(),
            access_token=guest_user.access_token,
            expires_days=approval.demo_duration_days,
            max_conversations=approval.max_conversations,
        )
        logger.info(f"Demo access email sent to {guest_user.email}")
    except Exception as e:
        logger.error(f"Failed to send demo access email: {e}")
        # Don't fail the request if email fails
    
    return GuestUserResponse(
        id=guest_user.id,
        email=guest_user.email,
        access_token=guest_user.access_token,
        expires_at=guest_user.expires_at,
        max_conversations=guest_user.max_conversations,
        conversations_used=guest_user.conversations_used,
        is_active=guest_user.is_active,
        granted_at=guest_user.granted_at,
        last_used_at=guest_user.last_used_at,
        days_remaining=guest_user.days_remaining(),
        conversations_remaining=guest_user.conversations_remaining(),
        can_access_demo=guest_user.can_access_demo(),
    )


@demo_router.patch(
    "/api/v1/admin/demo-requests/{request_id}/reject",
    response_model=DemoRequestResponse
)
async def reject_demo_request(
    request_id: int,
    rejection: DemoRejectionRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Reject a demo request with reason (admin only).
    """
    query = select(DemoRequest).where(DemoRequest.id == request_id)
    result = await db.execute(query)
    demo_request = result.scalar_one_or_none()
    
    if not demo_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo request with ID {request_id} not found"
        )
    
    if demo_request.status == DemoRequestStatus.REJECTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Demo request is already rejected"
        )
    
    # Update demo request status
    demo_request.status = DemoRequestStatus.REJECTED
    demo_request.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    demo_request.admin_notes = rejection.admin_notes
    
    await db.commit()
    await db.refresh(demo_request)
    
    logger.info(f"Demo request {request_id} rejected by admin {admin.email}")
    
    # TODO: Send rejection email to requester (optional, be considerate)
    
    return DemoRequestResponse(
        id=demo_request.id,
        first_name=demo_request.first_name,
        last_name=demo_request.last_name,
        email=demo_request.email,
        phone=demo_request.phone,
        company=demo_request.company,
        company_size=demo_request.company_size,
        message=demo_request.message,
        status=demo_request.status.value,
        admin_notes=demo_request.admin_notes,
        assigned_to_admin_id=demo_request.assigned_to_admin_id,
        scheduled_follow_up=demo_request.scheduled_follow_up,
        submitted_at=demo_request.submitted_at,
        reviewed_at=demo_request.reviewed_at,
        created_at=demo_request.created_at,
        updated_at=demo_request.updated_at,
    )


@demo_router.patch(
    "/api/v1/admin/demo-requests/{request_id}",
    response_model=DemoRequestResponse
)
async def update_demo_request(
    request_id: int,
    update: DemoRequestUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Update demo request admin notes, assignment, or follow-up schedule (admin only).
    """
    query = select(DemoRequest).where(DemoRequest.id == request_id).options(
        selectinload(DemoRequest.assigned_to_admin),
        selectinload(DemoRequest.guest_user)
    )
    result = await db.execute(query)
    demo_request = result.scalar_one_or_none()
    
    if not demo_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo request with ID {request_id} not found"
        )
    
    # Update fields
    if update.admin_notes is not None:
        demo_request.admin_notes = update.admin_notes
    
    if update.assigned_to_admin_id is not None:
        # Verify admin user exists
        if update.assigned_to_admin_id:
            admin_query = select(User).where(
                and_(
                    User.id == update.assigned_to_admin_id,
                    User.role == UserRole.ADMIN
                )
            )
            admin_result = await db.execute(admin_query)
            admin_user = admin_result.scalar_one_or_none()
            if not admin_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Admin user with ID {update.assigned_to_admin_id} not found"
                )
        demo_request.assigned_to_admin_id = update.assigned_to_admin_id
    
    if update.scheduled_follow_up is not None:
        demo_request.scheduled_follow_up = update.scheduled_follow_up
    
    await db.commit()
    await db.refresh(demo_request)
    
    logger.info(f"Demo request {request_id} updated by admin {admin.email}")
    
    # Build response
    response_data = {
        "id": demo_request.id,
        "first_name": demo_request.first_name,
        "last_name": demo_request.last_name,
        "email": demo_request.email,
        "phone": demo_request.phone,
        "company": demo_request.company,
        "company_size": demo_request.company_size,
        "message": demo_request.message,
        "status": demo_request.status.value,
        "admin_notes": demo_request.admin_notes,
        "assigned_to_admin_id": demo_request.assigned_to_admin_id,
        "scheduled_follow_up": demo_request.scheduled_follow_up,
        "submitted_at": demo_request.submitted_at,
        "reviewed_at": demo_request.reviewed_at,
        "created_at": demo_request.created_at,
        "updated_at": demo_request.updated_at,
    }
    
    if demo_request.assigned_to_admin:
        response_data["assigned_to_admin"] = {
            "id": demo_request.assigned_to_admin.id,
            "email": demo_request.assigned_to_admin.email,
            "full_name": demo_request.assigned_to_admin.full_name,
        }
    
    if demo_request.guest_user:
        guest = demo_request.guest_user
        response_data["guest_user"] = {
            "id": guest.id,
            "email": guest.email,
            "expires_at": guest.expires_at,
            "max_conversations": guest.max_conversations,
            "conversations_used": guest.conversations_used,
            "is_active": guest.is_active,
            "days_remaining": guest.days_remaining(),
            "conversations_remaining": guest.conversations_remaining(),
            "can_access_demo": guest.can_access_demo(),
        }
    
    return DemoRequestResponse(**response_data)


# Admin endpoint to disable guest user access
@demo_router.patch(
    "/api/v1/admin/guest-users/{guest_id}/disable",
    response_model=GuestUserResponse
)
async def disable_guest_user(
    guest_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Disable guest user access (admin only).
    
    Used when admin wants to revoke demo access before expiry.
    """
    query = select(GuestUser).where(GuestUser.id == guest_id)
    result = await db.execute(query)
    guest_user = result.scalar_one_or_none()
    
    if not guest_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guest user with ID {guest_id} not found"
        )
    
    guest_user.is_active = False
    
    await db.commit()
    await db.refresh(guest_user)
    
    logger.info(f"Guest user {guest_id} ({guest_user.email}) disabled by admin {admin.email}")
    
    return GuestUserResponse(
        id=guest_user.id,
        email=guest_user.email,
        access_token=guest_user.access_token,
        expires_at=guest_user.expires_at,
        max_conversations=guest_user.max_conversations,
        conversations_used=guest_user.conversations_used,
        is_active=guest_user.is_active,
        granted_at=guest_user.granted_at,
        last_used_at=guest_user.last_used_at,
        days_remaining=guest_user.days_remaining(),
        conversations_remaining=guest_user.conversations_remaining(),
        can_access_demo=guest_user.can_access_demo(),
    )


# ============================================
# Guest Authentication Endpoints
# ============================================

class GuestLoginRequest(BaseModel):
    """Guest login request."""
    email: EmailStr
    access_token: str = Field(..., min_length=32, description="Access token sent via email")


class GuestLoginResponse(BaseModel):
    """Guest login response."""
    success: bool
    message: str
    jwt_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    guest_user: GuestUserResponse


@guest_auth_router.post("/login", response_model=GuestLoginResponse)
async def guest_login(
    request: GuestLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Guest user login with email and access token.
    
    Validates the guest access token and returns a JWT for API access.
    """
    # Find guest user by email and token
    query = select(GuestUser).where(
        (GuestUser.email == request.email) &
        (GuestUser.access_token == request.access_token)
    )
    result = await db.execute(query)
    guest_user = result.scalar_one_or_none()
    
    if not guest_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or access token"
        )
    
    if not guest_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your demo access has been disabled. Please contact support for more information."
        )
    
    if guest_user.is_expired():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your demo access has expired. Please contact us to upgrade to a full account."
        )
    
    # Create JWT token for guest (includes guest_id in sub field)
    jwt_token = create_access_token(
        user_id=guest_user.id,
        email=guest_user.email
    )
    
    logger.info(f"Guest user {guest_user.email} logged in successfully")
    
    return GuestLoginResponse(
        success=True,
        message="Login successful",
        jwt_token=jwt_token,
        token_type="bearer",
        expires_in=3600,  # 1 hour
        guest_user=GuestUserResponse(
            id=guest_user.id,
            email=guest_user.email,
            access_token=guest_user.access_token,
            expires_at=guest_user.expires_at,
            max_conversations=guest_user.max_conversations,
            conversations_used=guest_user.conversations_used,
            is_active=guest_user.is_active,
            granted_at=guest_user.granted_at,
            last_used_at=guest_user.last_used_at,
            days_remaining=guest_user.days_remaining(),
            conversations_remaining=guest_user.conversations_remaining(),
            can_access_demo=guest_user.can_access_demo(),
        )
    )


@guest_auth_router.get("/me", response_model=GuestUserResponse)
async def get_guest_profile(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get current guest user profile and usage stats.
    
    Requires guest access token in query parameter.
    """
    guest_user = await get_current_guest_user(token, db)
    
    return GuestUserResponse(
        id=guest_user.id,
        email=guest_user.email,
        access_token=guest_user.access_token,
        expires_at=guest_user.expires_at,
        max_conversations=guest_user.max_conversations,
        conversations_used=guest_user.conversations_used,
        is_active=guest_user.is_active,
        granted_at=guest_user.granted_at,
        last_used_at=guest_user.last_used_at,
        days_remaining=guest_user.days_remaining(),
        conversations_remaining=guest_user.conversations_remaining(),
        can_access_demo=guest_user.can_access_demo(),
    )


@guest_auth_router.post("/increment-usage")
async def increment_guest_usage(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Increment guest user conversation usage counter.
    
    Called after each successful demo conversation.
    Requires guest access token.
    """
    guest_user = await get_current_guest_user(token, db)
    
    # Increment usage
    guest_user.increment_usage()
    
    await db.commit()
    await db.refresh(guest_user)
    
    logger.info(
        f"Guest user {guest_user.email} usage incremented: "
        f"{guest_user.conversations_used}/{guest_user.max_conversations}"
    )
    
    return {
        "success": True,
        "conversations_used": guest_user.conversations_used,
        "conversations_remaining": guest_user.conversations_remaining(),
        "max_conversations": guest_user.max_conversations,
        "days_remaining": guest_user.days_remaining(),
    }


@guest_auth_router.get("/validate-access")
async def validate_guest_access(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Validate guest access token and return usage status.
    
    Used by demo interface to check if guest can access demo.
    Returns 403 if access denied with appropriate error message.
    """
    guest_user = await get_current_guest_user(token, db)
    
    return {
        "valid": True,
        "email": guest_user.email,
        "conversations_used": guest_user.conversations_used,
        "conversations_remaining": guest_user.conversations_remaining(),
        "max_conversations": guest_user.max_conversations,
        "days_remaining": guest_user.days_remaining(),
        "expires_at": guest_user.expires_at.isoformat(),
        "can_access_demo": guest_user.can_access_demo(),
    }


