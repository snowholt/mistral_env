"""
Email services for BeautyAI platform.
"""

from .email_service import EmailService, get_email_service
from .templates import EmailTemplates

__all__ = [
    "EmailService",
    "get_email_service",
    "EmailTemplates",
]
