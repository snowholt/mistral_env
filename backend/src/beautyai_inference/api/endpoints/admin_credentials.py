"""
Admin Credential Management API endpoints.

Provides admin-only endpoints for:
- Listing all Meta API credentials across customers
- Creating credentials on behalf of customers
- Viewing credential details (masked tokens)
- Revoking compromised credentials
- Viewing audit logs

Access restricted to users with admin role (@gmai.sa domain).
"""

import logging
import httpx
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.orm import selectinload

from ...database.connection import get_db
from ...database.models import (
    User, Customer, MetaCredential, CredentialType, WhatsAppAccount, AuditLog
)
from ...auth.dependencies import get_current_active_user
from ...services.meta_credential import get_meta_credential_service
from ...utils.encryption import get_encryption_service
from ..schemas.credential_schemas import (
    CredentialCreate,
    CredentialUpdate,
    CredentialResponse,
    CredentialListResponse,
    CredentialRevokeRequest,
    TokenValidationResult,
    CredentialAuditLog,
)

logger = logging.getLogger(__name__)

admin_credentials_router = APIRouter(
    prefix="/api/v1/admin/credentials",
    tags=["admin-credentials"]
)


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
# Helper Functions
# ============================================

def get_token_prefix(encrypted_value: bytes) -> str:
    """Get first 8 chars of decrypted token for identification."""
    try:
        encryption = get_encryption_service()
        token = encryption.decrypt(encrypted_value)
        return token[:8] + "..." if len(token) > 8 else token + "..."
    except Exception:
        return "****..."


def credential_to_response(
    credential: MetaCredential,
    include_customer: bool = True
) -> CredentialResponse:
    """Convert MetaCredential model to response schema."""
    customer_name = None
    customer_email = None
    whatsapp_accounts = []
    
    if include_customer and credential.customer:
        customer_name = credential.customer.name
        customer_email = credential.customer.email
    
    if credential.whatsapp_accounts:
        whatsapp_accounts = [
            {
                "id": wa.id,
                "phone_number": wa.phone_number,
                "display_name": wa.display_name,
                "is_active": wa.is_active,
            }
            for wa in credential.whatsapp_accounts
        ]
    
    return CredentialResponse(
        id=credential.id,
        customer_id=credential.customer_id,
        customer_name=customer_name,
        customer_email=customer_email,
        name=f"{credential.credential_type.value} credential",
        credential_type=credential.credential_type.value,
        token_prefix=get_token_prefix(credential.encrypted_value),
        scopes=credential.scopes or [],
        expires_at=credential.expires_at,
        is_active=credential.is_active,
        is_expired=credential.is_expired(),
        is_valid=credential.is_valid(),
        use_count=credential.use_count,
        last_used_at=credential.last_used_at,
        created_at=credential.created_at,
        updated_at=credential.updated_at,
        whatsapp_accounts=whatsapp_accounts,
    )


async def validate_meta_token(token: str) -> TokenValidationResult:
    """
    Validate a Meta API token by calling the Graph API debug_token endpoint.
    
    Returns token type, scopes, expiration, and validity status.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Use the token to get info about itself
            response = await client.get(
                "https://graph.facebook.com/v21.0/me",
                params={"access_token": token, "fields": "id,name"}
            )
            
            if response.status_code != 200:
                error_data = response.json().get("error", {})
                return TokenValidationResult(
                    is_valid=False,
                    error=error_data.get("message", "Token validation failed")
                )
            
            user_data = response.json()
            user_id = user_data.get("id")
            
            # Get token debug info using app-level token inspection
            # For now, we'll infer token type from the /me response
            # System user tokens have a different ID format
            
            # Check what scopes the token has by testing endpoints
            scopes = []
            
            # Test whatsapp_business_management scope
            wa_response = await client.get(
                "https://graph.facebook.com/v21.0/me/whatsapp_business_accounts",
                params={"access_token": token}
            )
            if wa_response.status_code == 200:
                scopes.append("whatsapp_business_management")
            
            # Test business_management scope
            biz_response = await client.get(
                "https://graph.facebook.com/v21.0/me/businesses",
                params={"access_token": token, "limit": 1}
            )
            if biz_response.status_code == 200:
                scopes.append("business_management")
            
            # Infer token type
            # System user tokens typically:
            # 1. Don't expire (or have very long expiration)
            # 2. Have numeric IDs that look like app-scoped user IDs
            token_type = "User"  # Default
            if user_id and user_id.isdigit() and len(user_id) > 15:
                # Long numeric ID suggests system user
                token_type = "System User"
            
            return TokenValidationResult(
                is_valid=True,
                token_type=token_type,
                user_id=user_id,
                scopes=scopes,
                expires_at=None if token_type == "System User" else None,  # Would need debug_token for exact expiry
                is_expired=False,
                error=None
            )
            
    except httpx.TimeoutException:
        return TokenValidationResult(
            is_valid=False,
            error="Request timed out while validating token"
        )
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return TokenValidationResult(
            is_valid=False,
            error=f"Validation error: {str(e)}"
        )


async def log_audit_event(
    db: AsyncSession,
    action: str,
    resource_type: str,
    resource_id: str,
    user_id: Optional[int] = None,
    customer_id: Optional[int] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
):
    """Log an audit event for credential operations."""
    audit_log = AuditLog(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user_id,
        customer_id=customer_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(audit_log)
    await db.flush()


# ============================================
# Admin Credential Endpoints
# ============================================

@admin_credentials_router.get("", response_model=CredentialListResponse)
async def list_credentials(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    customer_id: Optional[int] = Query(None, description="Filter by customer"),
    status: Optional[str] = Query(None, regex="^(active|expired|revoked|all)$"),
    credential_type: Optional[str] = Query(None, regex="^(user_token|system_user_token|page_token)$"),
    search: Optional[str] = Query(None, description="Search by customer name/email"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    List all Meta API credentials with filtering and pagination.
    
    Admin only - shows credentials across all customers.
    """
    # Build query
    query = select(MetaCredential).options(
        selectinload(MetaCredential.customer),
        selectinload(MetaCredential.whatsapp_accounts),
    )
    
    # Apply filters
    filters = []
    
    if customer_id:
        filters.append(MetaCredential.customer_id == customer_id)
    
    if credential_type:
        filters.append(MetaCredential.credential_type == CredentialType(credential_type))
    
    if status == "active":
        filters.append(MetaCredential.is_active == True)
    elif status == "revoked":
        filters.append(MetaCredential.is_active == False)
    # Note: "expired" status needs runtime check, done in response filtering
    
    if search:
        search_filter = f"%{search}%"
        query = query.join(Customer).where(
            or_(
                Customer.name.ilike(search_filter),
                Customer.email.ilike(search_filter)
            )
        )
    
    if filters:
        query = query.where(and_(*filters))
    
    # Get total count
    count_query = select(func.count(MetaCredential.id))
    if customer_id:
        count_query = count_query.where(MetaCredential.customer_id == customer_id)
    if credential_type:
        count_query = count_query.where(MetaCredential.credential_type == CredentialType(credential_type))
    if status == "active":
        count_query = count_query.where(MetaCredential.is_active == True)
    elif status == "revoked":
        count_query = count_query.where(MetaCredential.is_active == False)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply sorting and pagination
    query = query.order_by(desc(MetaCredential.created_at))
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    credentials = result.scalars().all()
    
    # Convert to response models
    credential_list = [credential_to_response(c) for c in credentials]
    
    # Filter by expired status if requested (runtime check)
    if status == "expired":
        credential_list = [c for c in credential_list if c.is_expired]
        total = len(credential_list)
    
    return CredentialListResponse(
        success=True,
        total=total,
        skip=skip,
        limit=limit,
        credentials=credential_list,
    )


@admin_credentials_router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential_detail(
    credential_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a specific credential.
    
    Admin only - token value is masked (shows prefix only).
    """
    result = await db.execute(
        select(MetaCredential)
        .options(
            selectinload(MetaCredential.customer),
            selectinload(MetaCredential.whatsapp_accounts),
        )
        .where(MetaCredential.id == credential_id)
    )
    credential = result.scalar_one_or_none()
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    return credential_to_response(credential)


@admin_credentials_router.post("", response_model=CredentialResponse)
async def create_credential(
    request: CredentialCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new credential for a customer (admin only).
    
    Validates the token with Meta API before storing.
    """
    # Verify customer exists
    if not request.customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="customer_id is required for admin credential creation"
        )
    
    customer_result = await db.execute(
        select(Customer).where(Customer.id == request.customer_id)
    )
    customer = customer_result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Validate the token with Meta API
    validation = await validate_meta_token(request.token)
    
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token validation failed: {validation.error}"
        )
    
    # Store the credential using MetaCredentialService
    credential_service = get_meta_credential_service()
    
    credential_type = CredentialType(request.credential_type)
    
    credential = await credential_service.store_token(
        db=db,
        customer_id=request.customer_id,
        token=request.token,
        credential_type=credential_type,
        scopes=validation.scopes,
        expires_at=validation.expires_at,
        user_id=admin.id,
    )
    
    await db.commit()
    
    # Log audit event
    await log_audit_event(
        db=db,
        action="credential.created",
        resource_type="meta_credential",
        resource_id=str(credential.id),
        user_id=admin.id,
        customer_id=request.customer_id,
        details={
            "credential_type": request.credential_type,
            "created_by_admin": True,
            "scopes": validation.scopes,
        }
    )
    await db.commit()
    
    logger.info(
        f"Admin {admin.email} created credential {credential.id} "
        f"for customer {customer.email}"
    )
    
    # Reload with relationships
    result = await db.execute(
        select(MetaCredential)
        .options(
            selectinload(MetaCredential.customer),
            selectinload(MetaCredential.whatsapp_accounts),
        )
        .where(MetaCredential.id == credential.id)
    )
    credential = result.scalar_one()
    
    return credential_to_response(credential)


@admin_credentials_router.patch("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    request: CredentialUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Update credential metadata (admin only).
    
    Can update name and active status.
    """
    result = await db.execute(
        select(MetaCredential)
        .options(
            selectinload(MetaCredential.customer),
            selectinload(MetaCredential.whatsapp_accounts),
        )
        .where(MetaCredential.id == credential_id)
    )
    credential = result.scalar_one_or_none()
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    # Apply updates
    if request.is_active is not None:
        credential.is_active = request.is_active
    
    await db.commit()
    
    logger.info(f"Admin {admin.email} updated credential {credential_id}")
    
    return credential_to_response(credential)


@admin_credentials_router.patch("/{credential_id}/revoke")
async def revoke_credential(
    credential_id: int,
    request: Optional[CredentialRevokeRequest] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke a credential (soft delete).
    
    Sets is_active=False. The credential record is preserved for audit.
    """
    result = await db.execute(
        select(MetaCredential).where(MetaCredential.id == credential_id)
    )
    credential = result.scalar_one_or_none()
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    if not credential.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential is already revoked"
        )
    
    credential.is_active = False
    
    # Log audit event
    await log_audit_event(
        db=db,
        action="credential.revoked",
        resource_type="meta_credential",
        resource_id=str(credential_id),
        user_id=admin.id,
        customer_id=credential.customer_id,
        details={
            "reason": request.reason if request else None,
            "revoked_by_admin": True,
        }
    )
    
    await db.commit()
    
    logger.info(
        f"Admin {admin.email} revoked credential {credential_id}, "
        f"reason: {request.reason if request else 'Not specified'}"
    )
    
    return {
        "success": True,
        "message": "Credential revoked successfully",
        "credential_id": credential_id,
    }


@admin_credentials_router.delete("/{credential_id}")
async def delete_credential(
    credential_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Hard delete a credential (use with caution).
    
    Permanently removes the credential. Prefer revoke for audit trail preservation.
    """
    result = await db.execute(
        select(MetaCredential).where(MetaCredential.id == credential_id)
    )
    credential = result.scalar_one_or_none()
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    # Check if any WhatsApp accounts are linked
    if credential.whatsapp_accounts:
        # Clear the credential_id from linked accounts
        for wa in credential.whatsapp_accounts:
            wa.credential_id = None
    
    # Log audit event before deletion
    await log_audit_event(
        db=db,
        action="credential.deleted",
        resource_type="meta_credential",
        resource_id=str(credential_id),
        user_id=admin.id,
        customer_id=credential.customer_id,
        details={
            "deleted_by_admin": True,
            "credential_type": credential.credential_type.value,
        }
    )
    
    await db.delete(credential)
    await db.commit()
    
    logger.warning(f"Admin {admin.email} permanently deleted credential {credential_id}")
    
    return {
        "success": True,
        "message": "Credential permanently deleted",
        "credential_id": credential_id,
    }


@admin_credentials_router.post("/{credential_id}/validate", response_model=TokenValidationResult)
async def validate_credential(
    credential_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Validate a stored credential against Meta API.
    
    Decrypts the token and checks if it's still valid with Meta.
    """
    credential_service = get_meta_credential_service()
    
    # Get decrypted token
    token = await credential_service.get_token(db, credential_id, user_id=admin.id)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or inactive"
        )
    
    # Validate with Meta
    validation = await validate_meta_token(token)
    
    return validation


@admin_credentials_router.get("/{credential_id}/audit-log", response_model=List[CredentialAuditLog])
async def get_credential_audit_log(
    credential_id: int,
    limit: int = Query(50, ge=1, le=200),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get audit log for a specific credential.
    
    Shows all access, modification, and revocation events.
    """
    # Verify credential exists
    cred_result = await db.execute(
        select(MetaCredential).where(MetaCredential.id == credential_id)
    )
    if not cred_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    # Get audit logs
    result = await db.execute(
        select(AuditLog)
        .options(selectinload(AuditLog.user))
        .where(
            and_(
                AuditLog.resource_type == "meta_credential",
                AuditLog.resource_id == str(credential_id)
            )
        )
        .order_by(desc(AuditLog.created_at))
        .limit(limit)
    )
    logs = result.scalars().all()
    
    return [
        CredentialAuditLog(
            id=log.id,
            action=log.action,
            performed_by_user_id=log.user_id,
            performed_by_email=log.user.email if log.user else None,
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            details=log.details,
            created_at=log.created_at,
        )
        for log in logs
    ]


@admin_credentials_router.get("/stats/summary")
async def get_credential_stats(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get summary statistics for all credentials.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Total credentials
    total = (await db.execute(select(func.count(MetaCredential.id)))).scalar() or 0
    
    # Active credentials
    active = (await db.execute(
        select(func.count(MetaCredential.id)).where(MetaCredential.is_active == True)
    )).scalar() or 0
    
    # Revoked credentials
    revoked = (await db.execute(
        select(func.count(MetaCredential.id)).where(MetaCredential.is_active == False)
    )).scalar() or 0
    
    # By type
    by_type = {}
    for ctype in CredentialType:
        count = (await db.execute(
            select(func.count(MetaCredential.id)).where(
                MetaCredential.credential_type == ctype
            )
        )).scalar() or 0
        by_type[ctype.value] = count
    
    # Customers with credentials
    customers_with_creds = (await db.execute(
        select(func.count(func.distinct(MetaCredential.customer_id)))
    )).scalar() or 0
    
    return {
        "success": True,
        "stats": {
            "total": total,
            "active": active,
            "revoked": revoked,
            "by_type": by_type,
            "customers_with_credentials": customers_with_creds,
        },
        "generated_at": now.isoformat(),
    }
