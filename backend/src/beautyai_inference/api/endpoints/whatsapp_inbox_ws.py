"""
WhatsApp Inbox WebSocket Handler.

Provides real-time updates for the chat inbox using WebSocket connections.
Authenticated connections receive updates for their conversations only.
"""

import os
import json
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...database.connection import get_db_context
from ...database.models import User, Customer, Conversation
from ...auth.jwt_handler import verify_token, TokenType

logger = logging.getLogger(__name__)

whatsapp_inbox_ws_router = APIRouter(tags=["whatsapp-inbox-ws"])


# ============================================
# Connection Manager
# ============================================

@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    user_id: int
    customer_ids: Set[int] = field(default_factory=set)
    subscribed_conversations: Set[int] = field(default_factory=set)


class InboxConnectionManager:
    """
    Manages WebSocket connections for the inbox.
    
    Supports:
    - Per-user connections
    - Subscription to specific conversations
    - Broadcasting to relevant clients
    """
    
    def __init__(self):
        # user_id -> list of connections (user can have multiple tabs)
        self.user_connections: Dict[int, list[ConnectionInfo]] = {}
        # conversation_id -> set of user_ids subscribed
        self.conversation_subscribers: Dict[int, Set[int]] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: int,
        customer_ids: Set[int]
    ) -> ConnectionInfo:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        conn_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            customer_ids=customer_ids
        )
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(conn_info)
        
        logger.info(f"WebSocket connected: user {user_id}")
        return conn_info
    
    def disconnect(self, conn_info: ConnectionInfo):
        """Remove a WebSocket connection."""
        user_id = conn_info.user_id
        
        if user_id in self.user_connections:
            self.user_connections[user_id] = [
                c for c in self.user_connections[user_id]
                if c.websocket != conn_info.websocket
            ]
            
            # Clean up empty user entry
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from conversation subscriptions
        for conv_id in conn_info.subscribed_conversations:
            if conv_id in self.conversation_subscribers:
                self.conversation_subscribers[conv_id].discard(user_id)
        
        logger.info(f"WebSocket disconnected: user {user_id}")
    
    def subscribe_conversation(self, conn_info: ConnectionInfo, conversation_id: int):
        """Subscribe a connection to a conversation's updates."""
        conn_info.subscribed_conversations.add(conversation_id)
        
        if conversation_id not in self.conversation_subscribers:
            self.conversation_subscribers[conversation_id] = set()
        self.conversation_subscribers[conversation_id].add(conn_info.user_id)
    
    def unsubscribe_conversation(self, conn_info: ConnectionInfo, conversation_id: int):
        """Unsubscribe from a conversation's updates."""
        conn_info.subscribed_conversations.discard(conversation_id)
        
        if conversation_id in self.conversation_subscribers:
            self.conversation_subscribers[conversation_id].discard(conn_info.user_id)
    
    async def send_to_user(self, user_id: int, message: dict):
        """Send a message to all connections of a user."""
        if user_id not in self.user_connections:
            return
        
        dead_connections = []
        for conn_info in self.user_connections[user_id]:
            try:
                await conn_info.websocket.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to send to user {user_id}: {e}")
                dead_connections.append(conn_info)
        
        # Clean up dead connections
        for conn_info in dead_connections:
            self.disconnect(conn_info)
    
    async def broadcast_to_conversation(self, conversation_id: int, message: dict):
        """Broadcast a message to all users subscribed to a conversation."""
        if conversation_id not in self.conversation_subscribers:
            return
        
        for user_id in list(self.conversation_subscribers[conversation_id]):
            await self.send_to_user(user_id, message)
    
    async def broadcast_to_customer(self, customer_id: int, message: dict):
        """Broadcast a message to all users who own a customer."""
        for user_id, connections in list(self.user_connections.items()):
            for conn_info in connections:
                if customer_id in conn_info.customer_ids:
                    try:
                        await conn_info.websocket.send_json(message)
                    except Exception:
                        pass


# Global connection manager instance
inbox_manager = InboxConnectionManager()


# ============================================
# Public broadcast function (used by webhook)
# ============================================

async def broadcast_message(conversation_id: int, message: dict):
    """
    Broadcast a message event to subscribed clients.
    
    Called from whatsapp_webhook.py when new messages arrive.
    """
    await inbox_manager.broadcast_to_conversation(conversation_id, message)


async def broadcast_to_customer_users(customer_id: int, message: dict):
    """
    Broadcast to all users who own a customer.
    
    Useful for new conversation notifications.
    """
    await inbox_manager.broadcast_to_customer(customer_id, message)


# ============================================
# WebSocket Endpoint
# ============================================

@whatsapp_inbox_ws_router.websocket("/api/v1/whatsapp/inbox/ws")
async def inbox_websocket(
    websocket: WebSocket,
    token: str = Query(...)
):
    """
    WebSocket endpoint for real-time inbox updates.
    
    Authentication: Pass JWT token as query parameter.
    
    Client messages:
    - {"type": "subscribe", "conversation_id": 123}
    - {"type": "unsubscribe", "conversation_id": 123}
    - {"type": "ping"}
    
    Server messages:
    - {"type": "new_message", "conversation_id": 123, "message": {...}}
    - {"type": "status_update", "conversation_id": 123, "message_id": 456, "status": "read"}
    - {"type": "pong"}
    - {"type": "error", "message": "..."}
    """
    # Verify JWT token
    payload = verify_token(token, TokenType.ACCESS)
    if not payload:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return
    
    user_id = payload.user_id
    
    # Get user's customer IDs for authorization
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Customer.id).where(Customer.user_id == user_id)
            )
            customer_ids = set(row[0] for row in result.fetchall())
    except Exception as e:
        logger.error(f"Failed to get customer IDs: {e}")
        await websocket.close(code=4002, reason="Database error")
        return
    
    # Accept connection
    conn_info = await inbox_manager.connect(websocket, user_id, customer_ids)
    
    try:
        # Send connection success message
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "customer_count": len(customer_ids)
        })
        
        # Message handling loop
        while True:
            try:
                data = await websocket.receive_json()
                await handle_client_message(conn_info, data, customer_ids)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected normally: user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        inbox_manager.disconnect(conn_info)


async def handle_client_message(
    conn_info: ConnectionInfo,
    data: dict,
    customer_ids: Set[int]
):
    """Handle incoming messages from WebSocket client."""
    msg_type = data.get("type")
    
    if msg_type == "ping":
        await conn_info.websocket.send_json({"type": "pong"})
    
    elif msg_type == "subscribe":
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            await conn_info.websocket.send_json({
                "type": "error",
                "message": "conversation_id required"
            })
            return
        
        # Verify user owns this conversation
        if await verify_conversation_access(conversation_id, customer_ids):
            inbox_manager.subscribe_conversation(conn_info, conversation_id)
            await conn_info.websocket.send_json({
                "type": "subscribed",
                "conversation_id": conversation_id
            })
        else:
            await conn_info.websocket.send_json({
                "type": "error",
                "message": "Access denied to conversation"
            })
    
    elif msg_type == "unsubscribe":
        conversation_id = data.get("conversation_id")
        if conversation_id:
            inbox_manager.unsubscribe_conversation(conn_info, conversation_id)
            await conn_info.websocket.send_json({
                "type": "unsubscribed",
                "conversation_id": conversation_id
            })
    
    else:
        await conn_info.websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        })


async def verify_conversation_access(
    conversation_id: int,
    customer_ids: Set[int]
) -> bool:
    """Verify user has access to a conversation."""
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Conversation.customer_id)
                .where(Conversation.id == conversation_id)
            )
            row = result.fetchone()
            
            if row and row[0] in customer_ids:
                return True
            return False
    except Exception as e:
        logger.error(f"Error verifying conversation access: {e}")
        return False
