"""
WhatsApp Manager Authentication Module.

Provides JWT-based authentication for the WhatsApp Manager SaaS platform.
"""

from .jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    JWTPayload,
    TokenType
)

from .password import (
    hash_password,
    verify_password
)

from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_optional_user,
    oauth2_scheme
)

__all__ = [
    # JWT
    'create_access_token',
    'create_refresh_token',
    'verify_token',
    'decode_token',
    'JWTPayload',
    'TokenType',
    # Password
    'hash_password',
    'verify_password',
    # Dependencies
    'get_current_user',
    'get_current_active_user',
    'get_optional_user',
    'oauth2_scheme',
]
