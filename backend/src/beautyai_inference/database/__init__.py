"""
Database module for WhatsApp Manager.

Provides SQLAlchemy async database connection and session management.
"""

from .connection import (
    engine,
    async_session_maker,
    get_db,
    init_db,
    Base
)

from .models import (
    User,
    Customer,
    WhatsAppAccount,
    AgentConfig,
    Conversation,
    Message,
    MessageSource,
    MessageStatus
)

__all__ = [
    # Connection
    'engine',
    'async_session_maker',
    'get_db',
    'init_db',
    'Base',
    # Models
    'User',
    'Customer',
    'WhatsAppAccount',
    'AgentConfig',
    'Conversation',
    'Message',
    'MessageSource',
    'MessageStatus',
]
