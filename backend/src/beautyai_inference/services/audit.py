"""
Audit Service for compliance and security logging.

Provides structured audit logging for sensitive operations:
- Credential access/creation/revocation
- Authentication events
- Configuration changes
- Data exports

Usage:
    from beautyai_inference.services.audit import AuditService, get_audit_service
    
    audit = get_audit_service()
    await audit.log(
        db=db,
        action="credential.accessed",
        resource_type="meta_credential",
        resource_id="123",
        customer_id=1,
        user_id=5,
        ip_address="192.168.1.1"
    )
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from fastapi import Request
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from beautyai_inference.database.models import AuditLog

logger = logging.getLogger(__name__)


# Standard action constants
class AuditAction:
    """Standard audit action names."""
    # Credentials
    CREDENTIAL_CREATED = "credential.created"
    CREDENTIAL_ACCESSED = "credential.accessed"
    CREDENTIAL_REVOKED = "credential.revoked"
    CREDENTIAL_ROTATED = "credential.rotated"
    
    # Authentication
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_PASSWORD_CHANGED = "auth.password.changed"
    AUTH_PASSWORD_RESET = "auth.password.reset"
    
    # WhatsApp
    WHATSAPP_ACCOUNT_CONNECTED = "whatsapp.account.connected"
    WHATSAPP_ACCOUNT_DISCONNECTED = "whatsapp.account.disconnected"
    WHATSAPP_MESSAGE_SENT = "whatsapp.message.sent"
    
    # Configuration
    CONFIG_UPDATED = "config.updated"
    AGENT_CONFIG_UPDATED = "agent_config.updated"
    
    # Data operations
    DATA_EXPORTED = "data.exported"
    DATA_DELETED = "data.deleted"


class AuditService:
    """
    Service for recording security and compliance audit logs.
    
    Features:
    - Structured logging with action/resource tracking
    - Request context extraction (IP, user-agent)
    - Batch insert support for high-volume logging
    - Query helpers for audit log retrieval
    """
    
    async def log(
        self,
        db: AsyncSession,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        customer_id: Optional[int] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> AuditLog:
        """
        Record an audit log entry.
        
        Args:
            db: Database session.
            action: Action identifier (e.g., "credential.accessed").
            resource_type: Type of resource (e.g., "meta_credential").
            resource_id: ID of the affected resource.
            customer_id: Customer (tenant) ID.
            user_id: User performing the action.
            metadata: Additional context data.
            ip_address: Client IP address.
            user_agent: Client user-agent string.
            request_id: Request correlation ID.
            request: FastAPI Request object (extracts IP/user-agent automatically).
            
        Returns:
            Created AuditLog instance.
        """
        # Extract context from request if provided
        if request:
            if not ip_address:
                ip_address = self._extract_ip(request)
            if not user_agent:
                user_agent = request.headers.get("user-agent")
            if not request_id:
                request_id = request.headers.get("x-request-id")
        
        audit_log = AuditLog(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            customer_id=customer_id,
            user_id=user_id,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )
        
        db.add(audit_log)
        await db.flush()  # Get ID without committing
        
        logger.debug(
            f"Audit: {action} on {resource_type}/{resource_id} "
            f"by user_id={user_id} customer_id={customer_id}"
        )
        
        return audit_log
    
    async def log_from_request(
        self,
        db: AsyncSession,
        request: Request,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        customer_id: Optional[int] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """
        Convenience method to log with request context extraction.
        
        Args:
            db: Database session.
            request: FastAPI Request object.
            action: Action identifier.
            resource_type: Type of resource.
            resource_id: ID of the affected resource.
            customer_id: Customer (tenant) ID.
            user_id: User performing the action.
            metadata: Additional context data.
            
        Returns:
            Created AuditLog instance.
        """
        return await self.log(
            db=db,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            customer_id=customer_id,
            user_id=user_id,
            metadata=metadata,
            request=request,
        )
    
    async def log_batch(
        self,
        db: AsyncSession,
        entries: List[Dict[str, Any]],
    ) -> List[AuditLog]:
        """
        Record multiple audit entries in batch.
        
        Args:
            db: Database session.
            entries: List of audit entry dictionaries with keys:
                     action, resource_type, resource_id (optional),
                     customer_id (optional), user_id (optional),
                     metadata (optional), ip_address (optional),
                     user_agent (optional), request_id (optional).
                     
        Returns:
            List of created AuditLog instances.
        """
        audit_logs = []
        for entry in entries:
            audit_log = AuditLog(
                action=entry["action"],
                resource_type=entry["resource_type"],
                resource_id=entry.get("resource_id"),
                customer_id=entry.get("customer_id"),
                user_id=entry.get("user_id"),
                metadata=entry.get("metadata"),
                ip_address=entry.get("ip_address"),
                user_agent=entry.get("user_agent"),
                request_id=entry.get("request_id"),
            )
            db.add(audit_log)
            audit_logs.append(audit_log)
        
        await db.flush()
        logger.debug(f"Batch logged {len(audit_logs)} audit entries")
        return audit_logs
    
    async def get_logs_for_customer(
        self,
        db: AsyncSession,
        customer_id: int,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Retrieve audit logs for a customer.
        
        Args:
            db: Database session.
            customer_id: Customer ID.
            action: Optional filter by action.
            resource_type: Optional filter by resource type.
            limit: Maximum number of records.
            offset: Pagination offset.
            
        Returns:
            List of AuditLog instances.
        """
        conditions = [AuditLog.customer_id == customer_id]
        
        if action:
            conditions.append(AuditLog.action == action)
        if resource_type:
            conditions.append(AuditLog.resource_type == resource_type)
        
        query = (
            select(AuditLog)
            .where(and_(*conditions))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_logs_for_user(
        self,
        db: AsyncSession,
        user_id: int,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Retrieve audit logs for a user.
        
        Args:
            db: Database session.
            user_id: User ID.
            action: Optional filter by action.
            limit: Maximum number of records.
            offset: Pagination offset.
            
        Returns:
            List of AuditLog instances.
        """
        conditions = [AuditLog.user_id == user_id]
        
        if action:
            conditions.append(AuditLog.action == action)
        
        query = (
            select(AuditLog)
            .where(and_(*conditions))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_logs_for_resource(
        self,
        db: AsyncSession,
        resource_type: str,
        resource_id: str,
        limit: int = 100,
    ) -> List[AuditLog]:
        """
        Retrieve audit logs for a specific resource.
        
        Args:
            db: Database session.
            resource_type: Resource type.
            resource_id: Resource ID.
            limit: Maximum number of records.
            
        Returns:
            List of AuditLog instances.
        """
        query = (
            select(AuditLog)
            .where(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id,
            )
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_credential_access(
        self,
        db: AsyncSession,
        credential_id: int,
        hours: int = 24,
    ) -> List[AuditLog]:
        """
        Get recent access logs for a credential.
        
        Useful for detecting unusual access patterns.
        
        Args:
            db: Database session.
            credential_id: MetaCredential ID.
            hours: Time window in hours.
            
        Returns:
            List of access AuditLog instances.
        """
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
        
        query = (
            select(AuditLog)
            .where(
                AuditLog.resource_type == "meta_credential",
                AuditLog.resource_id == str(credential_id),
                AuditLog.action == AuditAction.CREDENTIAL_ACCESSED,
                AuditLog.created_at >= cutoff,
            )
            .order_by(AuditLog.created_at.desc())
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    def _extract_ip(self, request: Request) -> Optional[str]:
        """Extract client IP from request, handling proxies."""
        # Check X-Forwarded-For header (nginx/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return None


# Global service instance
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """
    Get global AuditService instance.
    
    Returns:
        AuditService instance.
    """
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


def initialize_audit_service() -> AuditService:
    """
    Initialize global AuditService.
    
    Returns:
        Initialized AuditService instance.
    """
    global _audit_service
    _audit_service = AuditService()
    return _audit_service
