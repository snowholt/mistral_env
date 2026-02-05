"""
WhatsApp Webhook Handler.

PUBLIC endpoint for receiving incoming WhatsApp messages from Meta.
Integrates with local LLM for AI-powered auto-replies.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import httpx

from ...database.connection import get_db_context
from ...database.models import (
    WhatsAppAccount, Customer, AgentConfig,
    Conversation, Message, MessageSource, MessageStatus
)
from ...services.meta_credential import get_meta_credential_service
from ...services.audit import get_audit_service

logger = logging.getLogger(__name__)

whatsapp_webhook_router = APIRouter(prefix="/api/v1/whatsapp/webhook", tags=["whatsapp-webhook"])

# Configuration
META_API_BASE = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com/v21.0")
WEBHOOK_VERIFY_TOKEN = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "beautyai_webhook_verify_token")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/inference/chat")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-unsloth-q4ks")
LLM_PRESET = os.getenv("LLM_PRESET", "qwen_optimized")


# ============================================
# Webhook Verification (GET)
# ============================================

@whatsapp_webhook_router.get("")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default="")
):
    """
    Meta webhook verification endpoint.
    
    Meta sends a GET request to verify webhook URL ownership.
    We must return the challenge value if the verify token matches.
    
    PUBLIC - No authentication required.
    """
    logger.info(f"Webhook verification: mode={hub_mode}, token={hub_verify_token[:10]}...")
    
    if hub_mode == "subscribe" and hub_verify_token == WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return int(hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")


# ============================================
# Incoming Message Handler (POST)
# ============================================

@whatsapp_webhook_router.post("")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle incoming WhatsApp webhook events.
    
    Processes incoming messages and triggers AI responses.
    
    PUBLIC - No authentication (Meta signs requests with app secret).
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    logger.debug(f"Webhook received: {body}")
    
    # Extract entry data
    entry = body.get("entry", [])
    
    for ent in entry:
        changes = ent.get("changes", [])
        
        for change in changes:
            if change.get("field") == "messages":
                value = change.get("value", {})
                
                # Extract phone_number_id from metadata
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id")
                
                if not phone_number_id:
                    logger.warning("No phone_number_id in webhook")
                    continue
                
                # Process messages
                messages = value.get("messages", [])
                for msg in messages:
                    # Process in background to respond quickly to Meta
                    background_tasks.add_task(
                        process_incoming_message,
                        phone_number_id=phone_number_id,
                        message_data=msg,
                        contacts=value.get("contacts", [])
                    )
                
                # Process status updates
                statuses = value.get("statuses", [])
                for status_update in statuses:
                    background_tasks.add_task(
                        process_status_update,
                        status_data=status_update
                    )
    
    # Always return 200 quickly to acknowledge receipt
    return {"status": "received"}


# ============================================
# Message Processing
# ============================================

async def process_incoming_message(
    phone_number_id: str,
    message_data: Dict[str, Any],
    contacts: List[Dict[str, Any]]
):
    """
    Process an incoming WhatsApp message.
    
    1. Find the WhatsApp account by phone_number_id
    2. Create/find conversation with sender
    3. Store incoming message
    4. Check if AI is enabled
    5. Generate AI response using LLM
    6. Send response via WhatsApp API
    7. Store AI response message
    """
    logger.info(f"Processing message for phone_number_id: {phone_number_id}")
    
    try:
        async with get_db_context() as db:
            # Find WhatsApp account
            result = await db.execute(
                select(WhatsAppAccount)
                .options(selectinload(WhatsAppAccount.customer))
                .where(WhatsAppAccount.phone_number_id == phone_number_id)
            )
            whatsapp_account = result.scalar_one_or_none()
            
            if not whatsapp_account:
                logger.warning(f"No WhatsApp account found for {phone_number_id}")
                return
            
            # Extract message details
            sender_phone = message_data.get("from", "")
            message_type = message_data.get("type", "text")
            whatsapp_message_id = message_data.get("id", "")
            
            # Get message content based on type
            if message_type == "text":
                content = message_data.get("text", {}).get("body", "")
            elif message_type == "image":
                content = "[Image received]"
            elif message_type == "audio":
                content = "[Audio message received]"
            elif message_type == "document":
                content = "[Document received]"
            else:
                content = f"[{message_type} received]"
            
            if not content:
                logger.warning("Empty message content")
                return
            
            # Get sender name from contacts
            sender_name = None
            for contact in contacts:
                if contact.get("wa_id") == sender_phone:
                    profile = contact.get("profile", {})
                    sender_name = profile.get("name")
                    break
            
            # Find or create conversation
            result = await db.execute(
                select(Conversation).where(
                    Conversation.whatsapp_account_id == whatsapp_account.id,
                    Conversation.contact_phone == sender_phone
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                # Auto-create conversation
                conversation = Conversation(
                    customer_id=whatsapp_account.customer_id,
                    whatsapp_account_id=whatsapp_account.id,
                    contact_phone=sender_phone,
                    contact_name=sender_name,
                    status="active"
                )
                db.add(conversation)
                await db.flush()
                logger.info(f"Created new conversation: {conversation.id}")
            else:
                # Update contact name if we have it now
                if sender_name and not conversation.contact_name:
                    conversation.contact_name = sender_name
            
            # Store incoming message
            incoming_message = Message(
                conversation_id=conversation.id,
                content=content,
                source=MessageSource.CUSTOMER,
                status=MessageStatus.DELIVERED,
                whatsapp_message_id=whatsapp_message_id
            )
            db.add(incoming_message)
            
            # Update conversation
            conversation.last_message_at = datetime.utcnow()
            conversation.unread_count += 1
            
            await db.commit()
            
            # Notify WebSocket clients (will be implemented in inbox_ws)
            await notify_new_message(conversation.id, incoming_message)
            
            # Check if AI should respond
            if not await should_ai_respond(db, conversation, whatsapp_account.customer_id):
                logger.info(f"AI disabled for conversation {conversation.id}")
                return
            
            # Generate and send AI response
            await generate_and_send_ai_response(
                db=db,
                conversation=conversation,
                whatsapp_account=whatsapp_account,
                incoming_content=content
            )
    
    except Exception as e:
        logger.error(f"Error processing incoming message: {e}", exc_info=True)


async def should_ai_respond(
    db: AsyncSession,
    conversation: Conversation,
    customer_id: int
) -> bool:
    """Check if AI should respond to this conversation."""
    
    # Check conversation-level pause
    if conversation.ai_paused_until and conversation.ai_paused_until > datetime.utcnow():
        logger.debug(f"Conversation {conversation.id} AI paused until {conversation.ai_paused_until}")
        return False
    
    # Check agent config
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.customer_id == customer_id)
    )
    agent_config = result.scalar_one_or_none()
    
    if not agent_config:
        logger.debug(f"No agent config for customer {customer_id}")
        return False
    
    if not agent_config.is_ai_active():
        logger.debug(f"AI not active for customer {customer_id}")
        return False
    
    return True


async def generate_and_send_ai_response(
    db: AsyncSession,
    conversation: Conversation,
    whatsapp_account: WhatsAppAccount,
    incoming_content: str
):
    """Generate AI response using LLM and send via WhatsApp."""
    
    try:
        # Get agent config for system prompt
        result = await db.execute(
            select(AgentConfig).where(AgentConfig.customer_id == conversation.customer_id)
        )
        agent_config = result.scalar_one_or_none()
        
        if not agent_config:
            logger.warning(f"No agent config for customer {conversation.customer_id}")
            return
        
        # Get conversation history for context
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(10)  # Last 10 messages for context
        )
        recent_messages = result.scalars().all()
        
        # Build chat history in reverse (oldest first)
        chat_history = []
        for msg in reversed(recent_messages):
            if msg.source == MessageSource.CUSTOMER:
                chat_history.append({"role": "user", "content": msg.content})
            else:
                chat_history.append({"role": "assistant", "content": msg.content})
        
        # Call LLM API
        ai_response = await call_llm_api(
            system_prompt=agent_config.system_prompt,
            message=incoming_content,
            chat_history=chat_history[:-1],  # Exclude the current message (it's passed separately)
            session_id=f"wa_conv_{conversation.id}"
        )
        
        if not ai_response:
            logger.error("Empty response from LLM")
            return
        
        # Get access token from encrypted vault
        credential_service = get_meta_credential_service(audit_service=get_audit_service())
        access_token = await credential_service.get_token_for_whatsapp_account(
            db=db,
            whatsapp_account_id=whatsapp_account.id,
        )
        
        if not access_token:
            logger.error(f"No valid access token for WhatsApp account {whatsapp_account.id}")
            return
        
        # Send via WhatsApp
        whatsapp_message_id = await send_whatsapp_message(
            phone_number_id=whatsapp_account.phone_number_id,
            access_token=access_token,
            recipient=conversation.contact_phone,
            message=ai_response
        )
        
        # Store AI response
        ai_message = Message(
            conversation_id=conversation.id,
            content=ai_response,
            source=MessageSource.AI,
            status=MessageStatus.SENT,
            whatsapp_message_id=whatsapp_message_id
        )
        db.add(ai_message)
        
        # Update conversation
        conversation.last_message_at = datetime.utcnow()
        
        await db.commit()
        
        # Notify WebSocket clients
        await notify_new_message(conversation.id, ai_message)
        
        logger.info(f"AI response sent for conversation {conversation.id}")
    
    except Exception as e:
        logger.error(f"Error generating AI response: {e}", exc_info=True)


async def call_llm_api(
    system_prompt: str,
    message: str,
    chat_history: List[Dict[str, str]],
    session_id: str
) -> Optional[str]:
    """
    Call the local LLM API for chat completion.
    
    Uses the BeautyAI inference endpoint.
    """
    try:
        # Prepend system prompt to chat history
        full_history = [{"role": "system", "content": system_prompt}] + chat_history
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                LLM_API_URL,
                json={
                    "model_name": LLM_MODEL_NAME,
                    "message": message,
                    "chat_history": full_history,
                    "session_id": session_id,
                    "preset": LLM_PRESET,
                    "disable_content_filter": True,
                    "max_new_tokens": 512
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            if not data.get("success"):
                logger.error(f"LLM API returned error: {data.get('error')}")
                return None
            
            return data.get("response", "")
    
    except httpx.TimeoutException:
        logger.error("LLM API timeout")
        return None
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None


async def send_whatsapp_message(
    phone_number_id: str,
    access_token: str,
    recipient: str,
    message: str
) -> str:
    """Send a message via WhatsApp Cloud API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{META_API_BASE}/{phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {"body": message}
            }
        )
        
        if response.status_code != 200:
            logger.error(f"WhatsApp API error: {response.text}")
            return ""
        
        data = response.json()
        return data.get("messages", [{}])[0].get("id", "")


async def process_status_update(status_data: Dict[str, Any]):
    """Process message status updates (sent, delivered, read)."""
    whatsapp_message_id = status_data.get("id")
    status = status_data.get("status")  # sent, delivered, read, failed
    
    if not whatsapp_message_id or not status:
        return
    
    logger.debug(f"Status update: {whatsapp_message_id} -> {status}")
    
    try:
        status_map = {
            "sent": MessageStatus.SENT,
            "delivered": MessageStatus.DELIVERED,
            "read": MessageStatus.READ,
            "failed": MessageStatus.FAILED
        }
        
        new_status = status_map.get(status)
        if not new_status:
            return
        
        async with get_db_context() as db:
            result = await db.execute(
                select(Message).where(Message.whatsapp_message_id == whatsapp_message_id)
            )
            message = result.scalar_one_or_none()
            
            if message:
                message.status = new_status
                await db.commit()
                logger.debug(f"Updated message {message.id} status to {status}")
    
    except Exception as e:
        logger.error(f"Error processing status update: {e}")


# ============================================
# WebSocket Notification Placeholder
# ============================================

async def notify_new_message(conversation_id: int, message: Message):
    """
    Notify WebSocket clients about a new message.
    
    This is a placeholder - actual implementation will be in whatsapp_inbox_ws.py
    """
    # Import here to avoid circular imports
    try:
        from .whatsapp_inbox_ws import broadcast_message
        await broadcast_message(conversation_id, {
            "type": "new_message",
            "conversation_id": conversation_id,
            "message": {
                "id": message.id,
                "content": message.content,
                "source": message.source.value,
                "status": message.status.value,
                "created_at": message.created_at.isoformat() if message.created_at else None
            }
        })
    except ImportError:
        # WebSocket module not yet loaded
        pass
    except Exception as e:
        logger.debug(f"WebSocket notification skipped: {e}")
