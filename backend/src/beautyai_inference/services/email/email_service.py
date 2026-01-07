"""
Email service using Microsoft Graph API (OAuth2).

Provides async email sending for:
- Email verification
- Password reset
- Notifications
- Transactional emails

Uses MSAL for authentication and Graph API for sending.
"""

import os
import logging
from typing import Optional
from uuid import uuid4
import time
from email.message import EmailMessage

import httpx
import msal
import aiosmtplib

logger = logging.getLogger(__name__)


class EmailService:
    """
    Microsoft Graph API Email Service.
    
    Uses OAuth2 (Client Credentials Flow) to authenticate and send emails via Graph API.
    """
    
    def __init__(self):
        """Initialize Graph API configuration from environment."""
        # Azure AD Configuration
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        # Sender configuration
        self.sender_address = os.getenv("SMTP_SENDER", "info@gmai.sa")

        # SMTP fallback configuration (used when Azure AD is not configured)
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.smtp_starttls = os.getenv("SMTP_STARTTLS", "true").lower() == "true"
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "false").lower() == "true"
        self.smtp_from_name = os.getenv("SMTP_FROM_NAME", "GMAI.sa")
        
        # Application URLs for email links
        self.app_base_url = os.getenv("APP_BASE_URL", "https://gmai.sa")
        
        # Development mode
        self.dev_mode = os.getenv("EMAIL_DEV_MODE", "false").lower() == "true"
        
        # MSAL App
        self._msal_app: Optional[msal.ConfidentialClientApplication] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        
        # HTTP Client
        self._client: Optional[httpx.AsyncClient] = None

    def _azure_is_configured(self) -> bool:
        return bool(self.tenant_id and self.client_id and self.client_secret)

    def _smtp_is_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_port and self.sender_address)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close any underlying HTTP clients."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def _get_msal_app(self) -> msal.ConfidentialClientApplication:
        """Get or create MSAL application instance."""
        if self._msal_app is None:
            if not self._azure_is_configured():
                raise ValueError("Azure AD credentials not configured")
                
            self._msal_app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                client_credential=self.client_secret,
            )
        return self._msal_app
    
    async def _get_access_token(self) -> str:
        """Get valid access token for Graph API."""
        # Check if token is valid (with 5 minute buffer)
        if self._access_token and time.time() < self._token_expires_at - 300:
            return self._access_token
            
        app = self._get_msal_app()
        
        # Acquire token
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        
        if "access_token" in result:
            self._access_token = result["access_token"]
            # Calculate expiration (default is usually 1 hour)
            expires_in = result.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            return self._access_token
        else:
            error = result.get("error")
            desc = result.get("error_description")
            logger.error(f"Failed to acquire token: {error} - {desc}")
            raise Exception(f"Authentication failed: {desc}")
    
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
        Send a single email via Microsoft Graph API.
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            html_body: HTML content
            text_body: Plain text content (optional)
            reply_to: Reply-to address (optional)
            tag: Tag for tracking (optional - logged only)
        
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
        
        try:
            # Prefer Microsoft Graph when configured; otherwise fall back to SMTP.
            if self._azure_is_configured():
                return await self._send_via_graph(
                    to_address=to_address,
                    subject=subject,
                    html_body=html_body,
                    reply_to=reply_to,
                    tag=tag,
                )

            if self._smtp_is_configured():
                return await self._send_via_smtp(
                    to_address=to_address,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    reply_to=reply_to,
                    tag=tag,
                )

            raise ValueError(
                "Email service not configured: set Azure AD env vars (AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET) "
                "or SMTP env vars (SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD)."
            )

        except Exception as e:
            logger.exception(f"Email send error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _send_via_graph(
        self,
        to_address: str,
        subject: str,
        html_body: str,
        reply_to: Optional[str],
        tag: Optional[str],
    ) -> dict:
        token = await self._get_access_token()
            
        # Construct Graph API payload
        email_msg = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content": html_body,
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to_address,
                        }
                    }
                ],
            },
            "saveToSentItems": "false",
        }
            
        if reply_to:
            email_msg["message"]["replyTo"] = [
                {
                    "emailAddress": {
                        "address": reply_to,
                    }
                }
            ]
            
        # Send via Graph API (application permissions)
        endpoint = f"https://graph.microsoft.com/v1.0/users/{self.sender_address}/sendMail"

        client = await self._get_client()
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=email_msg,
        )

        if response.status_code == 202:
            logger.info(f"✅ Email sent via Graph to {to_address} (Tag: {tag})")
            return {
                "success": True,
                "message_id": str(uuid4()),  # Graph API async send doesn't return ID immediately
                "provider": "graph",
            }

        error_text = response.text
        logger.error(f"❌ Graph API Error ({response.status_code}): {error_text}")
        return {
            "success": False,
            "error": f"Graph API Error: {response.status_code}",
            "details": error_text,
            "provider": "graph",
        }

    async def _send_via_smtp(
        self,
        to_address: str,
        subject: str,
        html_body: str,
        text_body: Optional[str],
        reply_to: Optional[str],
        tag: Optional[str],
    ) -> dict:
        if not self.smtp_host:
            raise ValueError("SMTP_HOST is not configured")
        if self.smtp_password is None and self.smtp_username is not None:
            raise ValueError("SMTP_PASSWORD is not configured")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"{self.smtp_from_name} <{self.sender_address}>"
        msg["To"] = to_address
        if reply_to:
            msg["Reply-To"] = reply_to

        if text_body:
            msg.set_content(text_body)
            msg.add_alternative(html_body, subtype="html")
        else:
            # Some providers still want a plain part; keep it minimal.
            msg.set_content("This email requires an HTML-capable client.")
            msg.add_alternative(html_body, subtype="html")

        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                start_tls=self.smtp_starttls,
                use_tls=self.smtp_use_tls,
                sender=self.sender_address,
                recipients=[to_address],
            )
            logger.info(f"✅ Email sent via SMTP to {to_address} (Tag: {tag})")
            return {
                "success": True,
                "message_id": str(uuid4()),
                "provider": "smtp",
            }
        except Exception as e:
            logger.exception(f"SMTP send error: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "smtp",
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
    
    async def send_demo_request_confirmation(
        self,
        to_address: str,
        full_name: str,
    ) -> dict:
        """Send demo request confirmation email to requester."""
        from .templates import EmailTemplates
        
        html_body = EmailTemplates.demo_request_confirmation_email(
            full_name=full_name,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="تأكيد طلب التجربة - Demo Request Confirmed | GMAI.sa",
            html_body=html_body,
            tag="demo_request_confirmation",
        )
    
    async def send_demo_request_admin_notification(
        self,
        admin_email: str,
        requester_name: str,
        requester_email: str,
        company: str,
        company_size: str,
        message: str,
        demo_request_id: int,
    ) -> dict:
        """Send new demo request notification to admin."""
        from .templates import EmailTemplates
        
        html_body = EmailTemplates.demo_request_admin_notification_email(
            requester_name=requester_name,
            requester_email=requester_email,
            company=company,
            company_size=company_size,
            message=message,
            demo_request_id=demo_request_id,
            admin_panel_url=self.app_base_url,
        )
        
        return await self.send_email(
            to_address=admin_email,
            subject=f"طلب تجربة جديد من {requester_name} - New Demo Request | GMAI.sa",
            html_body=html_body,
            tag="demo_request_admin_notification",
        )
    
    async def send_demo_access_granted(
        self,
        to_address: str,
        full_name: str,
        access_token: str,
        expires_days: int,
        max_conversations: int,
    ) -> dict:
        """Send demo access granted email with login credentials."""
        from .templates import EmailTemplates
        
        login_url = f"{self.app_base_url}/demo/login"
        
        html_body = EmailTemplates.demo_access_granted_email(
            full_name=full_name,
            access_token=access_token,
            login_url=login_url,
            expires_days=expires_days,
            max_conversations=max_conversations,
        )
        
        return await self.send_email(
            to_address=to_address,
            subject="🎉 تم منح الوصول إلى التجربة - Demo Access Granted | GMAI.sa",
            html_body=html_body,
            tag="demo_access_granted",
        )


# Singleton instance
_email_service: Optional[EmailService] = None


async def get_email_service() -> EmailService:
    """Get or create singleton email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
