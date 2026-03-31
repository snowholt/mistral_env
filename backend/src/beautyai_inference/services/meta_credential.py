"""
Meta Credential Service for secure token management.

Provides encrypted storage, retrieval, and lifecycle management
for Meta API tokens (WhatsApp Business, Business Manager, etc.).

Usage:
    from beautyai_inference.services.meta_credential import MetaCredentialService
    
    # Store a token
    credential = await service.store_token(
        db=db,
        customer_id=1,
        token="EAA...",
        credential_type=CredentialType.USER_TOKEN,
        scopes=["whatsapp_business_management"]
    )
    
    # Retrieve and decrypt
    token = await service.get_token(db, credential_id=credential.id)
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from beautyai_inference.database.models import (
    MetaCredential,
    CredentialType,
    WhatsAppAccount,
)
from beautyai_inference.utils.encryption import get_encryption_service

logger = logging.getLogger(__name__)


class MetaCredentialService:
    """
    Service for managing encrypted Meta API credentials.
    
    Features:
    - Encrypt tokens at rest using Fernet
    - Track key versions for rotation
    - Automatic audit logging on access
    - Usage tracking (last_used_at, use_count)
    """
    
    def __init__(self, audit_service: Optional['AuditService'] = None):
        """
        Initialize credential service.
        
        Args:
            audit_service: Optional AuditService for logging access events.
        """
        self._encryption = get_encryption_service()
        self._audit_service = audit_service
    
    async def store_token(
        self,
        db: AsyncSession,
        customer_id: int,
        token: str,
        credential_type: CredentialType = CredentialType.USER_TOKEN,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        user_id: Optional[int] = None,
        request_meta: Optional[dict] = None,
    ) -> MetaCredential:
        """
        Encrypt and store a new Meta API token.
        
        Args:
            db: Database session.
            customer_id: Customer (tenant) ID.
            token: Plaintext token to encrypt.
            credential_type: Type of credential (user_token, system_user_token, etc.).
            scopes: OAuth scopes granted to this token.
            expires_at: Token expiration datetime.
            user_id: User performing the action (for audit).
            request_meta: Request context for audit (ip_address, user_agent, etc.).
            
        Returns:
            Created MetaCredential instance.
        """
        # Encrypt the token
        encrypted_value, key_version = self._encryption.encrypt_with_version(token)
        
        # Create credential record
        credential = MetaCredential(
            customer_id=customer_id,
            credential_type=credential_type,
            encrypted_value=encrypted_value,
            encryption_key_version=key_version,
            created_by_user_id=user_id,
            scopes=scopes,
            expires_at=expires_at,
            is_active=True,
        )
        
        db.add(credential)
        await db.flush()  # Get the ID without committing
        
        logger.info(
            f"Stored encrypted credential id={credential.id} "
            f"type={credential_type.value} customer_id={customer_id}"
        )
        
        # Audit log
        if self._audit_service:
            await self._audit_service.log(
                db=db,
                action="credential.created",
                resource_type="meta_credential",
                resource_id=str(credential.id),
                customer_id=customer_id,
                user_id=user_id,
                metadata={
                    "credential_type": credential_type.value,
                    "scopes": scopes,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                },
                **(request_meta or {}),
            )
        
        return credential
    
    async def get_token(
        self,
        db: AsyncSession,
        credential_id: int,
        user_id: Optional[int] = None,
        request_meta: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Retrieve and decrypt a token by credential ID.
        
        Args:
            db: Database session.
            credential_id: MetaCredential ID.
            user_id: User performing the action (for audit).
            request_meta: Request context for audit.
            
        Returns:
            Decrypted token string, or None if not found/inactive.
        """
        result = await db.execute(
            select(MetaCredential).where(MetaCredential.id == credential_id)
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            logger.warning(f"Credential id={credential_id} not found")
            return None
        
        if not credential.is_valid():
            logger.warning(
                f"Credential id={credential_id} is invalid "
                f"(active={credential.is_active}, expired={credential.is_expired()})"
            )
            return None
        
        # Decrypt the token
        try:
            token = self._encryption.decrypt(credential.encrypted_value)
        except ValueError as e:
            logger.error(f"Failed to decrypt credential id={credential_id}: {e}")
            return None
        
        # Record usage
        credential.record_use()
        await db.flush()
        
        logger.debug(f"Retrieved credential id={credential_id} for customer_id={credential.customer_id}")
        
        # Audit log
        if self._audit_service:
            await self._audit_service.log(
                db=db,
                action="credential.accessed",
                resource_type="meta_credential",
                resource_id=str(credential_id),
                customer_id=credential.customer_id,
                user_id=user_id,
                metadata={"credential_type": credential.credential_type.value},
                **(request_meta or {}),
            )
        
        return token
    
    async def get_token_for_whatsapp_account(
        self,
        db: AsyncSession,
        whatsapp_account_id: int,
        user_id: Optional[int] = None,
        request_meta: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Get decrypted token for a WhatsApp account.
        
        Falls back to deprecated access_token field if no credential_id.
        
        Args:
            db: Database session.
            whatsapp_account_id: WhatsAppAccount ID.
            user_id: User performing the action (for audit).
            request_meta: Request context for audit.
            
        Returns:
            Decrypted token string.
        """
        result = await db.execute(
            select(WhatsAppAccount).where(WhatsAppAccount.id == whatsapp_account_id)
        )
        account = result.scalar_one_or_none()
        
        if not account:
            logger.warning(f"WhatsAppAccount id={whatsapp_account_id} not found")
            return None
        
        # Use encrypted credential if available
        if account.credential_id:
            return await self.get_token(
                db=db,
                credential_id=account.credential_id,
                user_id=user_id,
                request_meta=request_meta,
            )
        
        # Fallback to deprecated plaintext token
        logger.warning(
            f"WhatsAppAccount id={whatsapp_account_id} using deprecated plaintext token. "
            "Run migration script to encrypt."
        )
        return account.access_token
    
    async def revoke_token(
        self,
        db: AsyncSession,
        credential_id: int,
        user_id: Optional[int] = None,
        request_meta: Optional[dict] = None,
    ) -> bool:
        """
        Revoke (deactivate) a credential.
        
        Args:
            db: Database session.
            credential_id: MetaCredential ID.
            user_id: User performing the action (for audit).
            request_meta: Request context for audit.
            
        Returns:
            True if credential was found and revoked.
        """
        result = await db.execute(
            select(MetaCredential).where(MetaCredential.id == credential_id)
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            logger.warning(f"Credential id={credential_id} not found for revocation")
            return False
        
        credential.is_active = False
        await db.flush()
        
        logger.info(f"Revoked credential id={credential_id}")
        
        # Audit log
        if self._audit_service:
            await self._audit_service.log(
                db=db,
                action="credential.revoked",
                resource_type="meta_credential",
                resource_id=str(credential_id),
                customer_id=credential.customer_id,
                user_id=user_id,
                **(request_meta or {}),
            )
        
        return True
    
    async def get_active_credentials_for_customer(
        self,
        db: AsyncSession,
        customer_id: int,
        credential_type: Optional[CredentialType] = None,
    ) -> List[MetaCredential]:
        """
        List active credentials for a customer.
        
        Args:
            db: Database session.
            customer_id: Customer ID.
            credential_type: Optional filter by type.
            
        Returns:
            List of MetaCredential instances (without decrypted values).
        """
        query = select(MetaCredential).where(
            MetaCredential.customer_id == customer_id,
            MetaCredential.is_active == True,
        )
        
        if credential_type:
            query = query.where(MetaCredential.credential_type == credential_type)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def link_credential_to_whatsapp_account(
        self,
        db: AsyncSession,
        whatsapp_account_id: int,
        credential_id: int,
    ) -> bool:
        """
        Link a credential to a WhatsApp account.
        
        Args:
            db: Database session.
            whatsapp_account_id: WhatsAppAccount ID.
            credential_id: MetaCredential ID.
            
        Returns:
            True if link was successful.
        """
        result = await db.execute(
            select(WhatsAppAccount).where(WhatsAppAccount.id == whatsapp_account_id)
        )
        account = result.scalar_one_or_none()
        
        if not account:
            logger.warning(f"WhatsAppAccount id={whatsapp_account_id} not found")
            return False
        
        account.credential_id = credential_id
        await db.flush()
        
        logger.info(
            f"Linked credential id={credential_id} to "
            f"WhatsAppAccount id={whatsapp_account_id}"
        )
        return True


# Global service instance
_meta_credential_service: Optional[MetaCredentialService] = None


def get_meta_credential_service(
    audit_service: Optional['AuditService'] = None
) -> MetaCredentialService:
    """
    Get global MetaCredentialService instance.
    
    Args:
        audit_service: Optional AuditService for logging.
        
    Returns:
        MetaCredentialService instance.
    """
    global _meta_credential_service
    if _meta_credential_service is None:
        _meta_credential_service = MetaCredentialService(audit_service=audit_service)
    return _meta_credential_service


def initialize_meta_credential_service(
    audit_service: Optional['AuditService'] = None
) -> MetaCredentialService:
    """
    Initialize global MetaCredentialService with custom settings.
    
    Args:
        audit_service: AuditService for logging access events.
        
    Returns:
        Initialized MetaCredentialService instance.
    """
    global _meta_credential_service
    _meta_credential_service = MetaCredentialService(audit_service=audit_service)
    return _meta_credential_service


# Type hint import for circular dependency
if False:  # TYPE_CHECKING
    from beautyai_inference.services.audit import AuditService
