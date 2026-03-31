"""
Pydantic schemas for Meta API credential management.

Provides request/response models for:
- Admin credential CRUD operations
- Customer token submission and status
- Token validation results
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class CredentialCreate(BaseModel):
    """Request to create a new credential (admin or customer)."""
    
    customer_id: Optional[int] = Field(
        None,
        description="Customer ID (required for admin creating on behalf of customer)"
    )
    token: str = Field(
        ...,
        min_length=10,
        description="Meta API token (System User or User Access Token)"
    )
    name: Optional[str] = Field(
        "Meta API Token",
        max_length=100,
        description="Friendly name for the credential"
    )
    credential_type: str = Field(
        "system_user_token",
        description="Type: 'system_user_token' (permanent) or 'user_token' (temporary)"
    )
    
    @field_validator('credential_type')
    @classmethod
    def validate_credential_type(cls, v: str) -> str:
        valid_types = {'user_token', 'system_user_token', 'page_token'}
        if v not in valid_types:
            raise ValueError(f"credential_type must be one of: {valid_types}")
        return v


class CredentialUpdate(BaseModel):
    """Request to update a credential (admin only)."""
    
    name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class TokenValidationRequest(BaseModel):
    """Request to validate a Meta API token."""
    
    token: str = Field(..., min_length=10, description="Token to validate")


class TokenValidationResult(BaseModel):
    """Result of validating a Meta API token against Meta Graph API."""
    
    is_valid: bool
    token_type: Optional[str] = None  # "User", "System User", "Page"
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    is_expired: bool = False
    error: Optional[str] = None


class CredentialResponse(BaseModel):
    """Response for a single credential (token value never exposed)."""
    
    id: int
    customer_id: int
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    
    name: str
    credential_type: str
    token_prefix: str = Field(..., description="First 8 chars of token + '...'")
    
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    
    is_active: bool
    is_expired: bool
    is_valid: bool  # Combined: is_active AND NOT is_expired
    
    # Usage tracking
    use_count: int
    last_used_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Linked WhatsApp accounts
    whatsapp_accounts: List[dict] = Field(default_factory=list)


class CredentialListResponse(BaseModel):
    """Paginated list of credentials."""
    
    success: bool = True
    total: int
    skip: int
    limit: int
    credentials: List[CredentialResponse]


class CustomerTokenStatus(BaseModel):
    """Token status response for customer dashboard."""
    
    has_token: bool
    credential_id: Optional[int] = None
    token_prefix: Optional[str] = None
    token_type: Optional[str] = None
    
    status: str  # "connected", "expired", "revoked", "not_connected"
    status_label: str  # Human-readable status
    status_color: str  # "green", "yellow", "red", "gray"
    
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    expires_in_days: Optional[int] = None
    
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    
    # Linked account info
    whatsapp_account_id: Optional[int] = None
    phone_number: Optional[str] = None
    display_name: Optional[str] = None


class CustomerTokenSubmit(BaseModel):
    """Request for customer to submit their own token."""
    
    token: str = Field(
        ...,
        min_length=10,
        description="System User token from Meta Business Settings"
    )
    name: Optional[str] = Field(
        "My WhatsApp Token",
        max_length=100,
        description="Friendly name for the token"
    )


class CustomerTokenSubmitResponse(BaseModel):
    """Response after customer submits a token."""
    
    success: bool
    message: str
    
    # Validation results
    validation: Optional[TokenValidationResult] = None
    
    # Created credential (if successful)
    credential_id: Optional[int] = None
    token_prefix: Optional[str] = None
    token_type: Optional[str] = None
    
    # Error details (if failed)
    error_code: Optional[str] = None
    error_detail: Optional[str] = None


class CredentialRevokeRequest(BaseModel):
    """Request to revoke a credential."""
    
    reason: Optional[str] = Field(None, max_length=500, description="Reason for revocation")


class CredentialAuditLog(BaseModel):
    """Audit log entry for credential operations."""
    
    id: int
    action: str  # "created", "accessed", "revoked", "deleted"
    performed_by_user_id: Optional[int] = None
    performed_by_email: Optional[str] = None
    
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    details: Optional[dict] = None
    created_at: datetime
