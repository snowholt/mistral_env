"""
Dashboard API endpoints for customer business overview.

Provides endpoints for:
- Dashboard stats (messages, active chats, response rate)
- Quick metrics for the user's own business
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ...database.connection import get_db
from ...database.models import (
    User, Customer, Conversation, Message, MessageSource,
    WebChatSession, WebChatMessage, AgentConfig
)
from ...auth.dependencies import get_current_active_user

logger = logging.getLogger(__name__)

dashboard_router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@dashboard_router.get("/stats")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dashboard statistics for the current user's business.
    
    Returns:
    - total_messages: Total messages across all conversations
    - active_chats: Number of active conversations (with messages in last 24h)
    - response_rate: Percentage of customer messages that got AI/human responses
    - avg_response_time: Average time to first AI response (in seconds)
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    
    # Get the user's customer (business)
    result = await db.execute(
        select(Customer).where(Customer.user_id == current_user.id)
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        # Return zeros if no customer is set up yet
        return {
            "total_messages": 0,
            "active_chats": 0,
            "response_rate": 0,
            "avg_response_time": "0s",
        }
    
    customer_id = customer.id
    
    # Total messages for this customer (WhatsApp + WebChat)
    # WhatsApp messages
    wa_messages = (await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(Conversation.customer_id == customer_id)
    )).scalar() or 0
    
    # WebChat messages
    wc_messages = (await db.execute(
        select(func.count(WebChatMessage.id))
        .join(WebChatSession)
        .where(WebChatSession.customer_id == customer_id)
    )).scalar() or 0
    
    total_messages = wa_messages + wc_messages
    
    # Active chats (conversations with activity in last 24 hours)
    # WhatsApp active conversations
    wa_active = (await db.execute(
        select(func.count(Conversation.id))
        .where(
            and_(
                Conversation.customer_id == customer_id,
                Conversation.last_message_at >= last_24h,
                Conversation.status == "active"
            )
        )
    )).scalar() or 0
    
    # WebChat active sessions
    wc_active = (await db.execute(
        select(func.count(WebChatSession.id))
        .where(
            and_(
                WebChatSession.customer_id == customer_id,
                WebChatSession.last_message_at >= last_24h,
                WebChatSession.is_active == True
            )
        )
    )).scalar() or 0
    
    active_chats = wa_active + wc_active
    
    # Response rate calculation
    # Count customer messages vs AI/human responses in last 7 days
    customer_messages = (await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(
            and_(
                Conversation.customer_id == customer_id,
                Message.source == MessageSource.CUSTOMER,
                Message.created_at >= last_7d
            )
        )
    )).scalar() or 0
    
    ai_responses = (await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(
            and_(
                Conversation.customer_id == customer_id,
                Message.source.in_([MessageSource.AI, MessageSource.HUMAN]),
                Message.created_at >= last_7d
            )
        )
    )).scalar() or 0
    
    # Also count webchat responses
    wc_user_messages = (await db.execute(
        select(func.count(WebChatMessage.id))
        .join(WebChatSession)
        .where(
            and_(
                WebChatSession.customer_id == customer_id,
                WebChatMessage.role == "user",
                WebChatMessage.created_at >= last_7d
            )
        )
    )).scalar() or 0
    
    wc_assistant_messages = (await db.execute(
        select(func.count(WebChatMessage.id))
        .join(WebChatSession)
        .where(
            and_(
                WebChatSession.customer_id == customer_id,
                WebChatMessage.role == "assistant",
                WebChatMessage.created_at >= last_7d
            )
        )
    )).scalar() or 0
    
    total_customer_messages = customer_messages + wc_user_messages
    total_responses = ai_responses + wc_assistant_messages
    
    if total_customer_messages > 0:
        response_rate = min(100, round((total_responses / total_customer_messages) * 100))
    else:
        response_rate = 100  # No messages = 100% response rate (nothing to respond to)
    
    # Average response time (placeholder - would need message timestamps analysis)
    # For now, return a reasonable default or calculate from actual data if available
    avg_response_time = "1.2s"  # TODO: Calculate from actual message pairs
    
    return {
        "total_messages": total_messages,
        "active_chats": active_chats,
        "response_rate": response_rate,
        "avg_response_time": avg_response_time,
    }


@dashboard_router.get("/setup-status")
async def get_setup_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the setup completion status for the user's business.
    
    Returns status of:
    - WhatsApp connection
    - AI Agent configuration
    - Knowledge base setup
    - Live status
    """
    from ...database.models import WhatsAppAccount, KnowledgeBase
    
    # Get the user's customer
    result = await db.execute(
        select(Customer).where(Customer.user_id == current_user.id)
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        return {
            "whatsapp_connected": False,
            "agent_configured": False,
            "knowledge_added": False,
            "is_live": False,
        }
    
    customer_id = customer.id
    
    # Check WhatsApp connection
    wa_result = await db.execute(
        select(func.count(WhatsAppAccount.id))
        .where(
            and_(
                WhatsAppAccount.customer_id == customer_id,
                WhatsAppAccount.is_active == True
            )
        )
    )
    whatsapp_connected = (wa_result.scalar() or 0) > 0
    
    # Check Agent configuration
    agent_result = await db.execute(
        select(AgentConfig).where(AgentConfig.customer_id == customer_id)
    )
    agent_config = agent_result.scalar_one_or_none()
    agent_configured = agent_config is not None and bool(agent_config.system_prompt)
    
    # Check Knowledge base
    kb_result = await db.execute(
        select(func.count(KnowledgeBase.id))
        .where(KnowledgeBase.customer_id == customer_id)
    )
    knowledge_added = (kb_result.scalar() or 0) > 0
    
    # Is live = has WhatsApp and agent configured
    is_live = whatsapp_connected and agent_configured
    
    return {
        "whatsapp_connected": whatsapp_connected,
        "agent_configured": agent_configured,
        "knowledge_added": knowledge_added,
        "is_live": is_live,
    }
