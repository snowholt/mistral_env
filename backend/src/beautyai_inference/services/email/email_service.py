"""
Email service using Alibaba Cloud DirectMail.

Provides async email sending for:
- Email verification
- Password reset
- Notifications
- Transactional emails
"""

import os
import logging
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, List
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class EmailService:
    """
    Alibaba Cloud DirectMail email service.
    
    Uses Alibaba's HTTP API for sending transactional emails.
    PDPL compliant - all data stays in Saudi Arabia region.
    """
    
    def __init__(self):
        """Initialize DirectMail configuration from environment."""
        self.access_key_id = os.getenv("ALICLOUD_ACCESS_KEY_ID", "")
        self.access_key_secret = os.getenv("ALICLOUD_ACCESS_KEY_SECRET", "")
        
        # DirectMail specific settings
        self.region = os.getenv("DIRECTMAIL_REGION", "me-central-1")  # Saudi Arabia
        self.endpoint = os.getenv(
            "DIRECTMAIL_ENDPOINT",
            f"https://dm.{self.region}.aliyuncs.com"
        )
        
        # Sender configuration
        self.sender_address = os.getenv("DIRECTMAIL_SENDER", "noreply@gmai.sa")
        self.sender_name = os.getenv("DIRECTMAIL_SENDER_NAME", "GMAI.sa")
        
        # Application URLs for email links
        self.app_base_url = os.getenv("APP_BASE_URL", "https://gmai.sa")
        
        # Development mode
        self.dev_mode = os.getenv("EMAIL_DEV_MODE", "false").lower() == "true"
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _sign_request(self, params: dict) -> str:
        """
        Sign request parameters using Alibaba Cloud signature method.
        
        Uses HMAC-SHA1 signature as per DirectMail API spec.
        """
        # Sort parameters by key
        sorted_params = sorted(params.items())
        
        # Build canonical query string
        canonical_query = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted_params
        )
        
        # Build string to sign
        string_to_sign = f"POST&{urllib.parse.quote('/', safe='')}&{urllib.parse.quote(canonical_query, safe='')}"
        
        # Calculate signature
        key = f"{self.access_key_secret}&"
        signature = base64.b64encode(
            hmac.new(
                key.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                hashlib.sha1
            ).digest()
        ).decode("utf-8")
        
        return signature
    
    async def send_email(
        self,
        to_address: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        reply_to: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict:
        """
        Send a single email via DirectMail.
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            html_body: HTML content
            text_body: Plain text content (optional)
            reply_to: Reply-to address (optional)
            tag: Tag for tracking (optional)
        
        Returns:
            Response dict with status and message_id
        """
        # Development mode - log instead of sending
        if self.dev_mode:
            logger.info(f"📧 [DEV MODE] Email to {to_address}")
            logger.info(f"   Subject: {subject}")
            logger.debug(f"   Body: {html_body[:200]}...")
            return {
                "success": True,
                "message_id": f"dev-{uuid4()}",
                "dev_mode": True,
            }
        
        # Validate configuration
        if not self.access_key_id or not self.access_key_secret:
            logger.error("DirectMail credentials not configured")
            return {
                "success": False,
                "error": "Email service not configured",
            }
        
        try:
            # Build request parameters
            params = {
                # Common parameters
                "Format": "JSON",
                "Version": "2015-11-23",
                "AccessKeyId": self.access_key_id,
                "SignatureMethod": "HMAC-SHA1",
                "SignatureVersion": "1.0",
                "SignatureNonce": str(uuid4()),
                "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "RegionId": self.region,
                
                # Action-specific parameters
                "Action": "SingleSendMail",
                "AccountName": self.sender_address,
                "AddressType": "1",  # 1 = Send from sender address
                "FromAlias": self.sender_name,
                "ReplyToAddress": "true" if reply_to else "false",
                "ToAddress": to_address,
                "Subject": subject,
                "HtmlBody": html_body,
            }
            
            if text_body:
                params["TextBody"] = text_body
            
            if reply_to:
                params["ReplyTo"] = reply_to
            
            if tag:
                params["TagName"] = tag
            
            # Sign request
            params["Signature"] = self._sign_request(params)
            
            # Send request
            client = await self._get_client()
            response = await client.post(
                self.endpoint,
                data=params,
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and "EnvId" in response_data:
                logger.info(f"✅ Email sent to {to_address}, EnvId: {response_data.get('EnvId')}")
                return {
                    "success": True,
                    "message_id": response_data.get("EnvId"),
                    "request_id": response_data.get("RequestId"),
                }
            else:
                logger.error(f"❌ Email failed: {response_data}")
                return {
                    "success": False,
                    "error": response_data.get("Message", "Unknown error"),
                    "code": response_data.get("Code"),
                }
        
        except Exception as e:
            logger.exception(f"Email send error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    # ========================================================================
    # Template Methods
    # ========================================================================
    
    async def send_verification_email(
        self,
        to_address: str,
        full_name: str,
        verification_token: str,
    ) -> dict:
        """Send email verification email."""
        from .templates import EmailTemplates
        
        verification_url = f"{self.app_base_url}/auth/verify-email?token={verification_token}"
        
        html_body = EmailTemplates.verification_email(
            full_name=full_name,
            verification_url=verification_url,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="تأكيد بريدك الإلكتروني - Verify Your Email | GMAI.sa",
            html_body=html_body,
            tag="verification",
        )
    
    async def send_password_reset_email(
        self,
        to_address: str,
        full_name: str,
        reset_token: str,
    ) -> dict:
        """Send password reset email."""
        from .templates import EmailTemplates
        
        reset_url = f"{self.app_base_url}/auth/reset-password?token={reset_token}"
        
        html_body = EmailTemplates.password_reset_email(
            full_name=full_name,
            reset_url=reset_url,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="إعادة تعيين كلمة المرور - Reset Your Password | GMAI.sa",
            html_body=html_body,
            tag="password_reset",
        )
    
    async def send_welcome_email(
        self,
        to_address: str,
        full_name: str,
    ) -> dict:
        """Send welcome email after verification."""
        from .templates import EmailTemplates
        
        dashboard_url = f"{self.app_base_url}/app"
        
        html_body = EmailTemplates.welcome_email(
            full_name=full_name,
            dashboard_url=dashboard_url,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="مرحباً بك في GMAI.sa - Welcome to GMAI.sa!",
            html_body=html_body,
            tag="welcome",
        )
    
    async def send_admin_invite_email(
        self,
        to_address: str,
        invite_code: str,
        invited_by: str,
    ) -> dict:
        """Send admin invite email (for @gmai.sa domain only)."""
        from .templates import EmailTemplates
        
        invite_url = f"{self.app_base_url}/auth/register?invite={invite_code}"
        
        html_body = EmailTemplates.admin_invite_email(
            invite_url=invite_url,
            invited_by=invited_by,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="دعوة للانضمام كمسؤول - Admin Invite | GMAI.sa",
            html_body=html_body,
            tag="admin_invite",
        )


# Singleton instance
_email_service: Optional[EmailService] = None


async def get_email_service() -> EmailService:
    """Get or create singleton email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
