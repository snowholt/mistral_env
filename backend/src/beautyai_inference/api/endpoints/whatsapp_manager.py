"""
WhatsApp Manager API endpoints.

Provides Meta Embedded Signup, Agent Configuration, and Inbox management APIs.
All endpoints are protected and require JWT authentication.
"""

import os
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc
from sqlalchemy.orm import selectinload
import httpx

from ...database.connection import get_db
from ...database.models import (
    User, Customer, WhatsAppAccount, AgentConfig, 
    Conversation, Message, MessageSource, MessageStatus
)
from ...auth.dependencies import get_current_active_user

logger = logging.getLogger(__name__)

whatsapp_manager_router = APIRouter(prefix="/api/v1/whatsapp", tags=["whatsapp-manager"])

# Environment config
META_API_BASE = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com/v21.0")
META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")


# ============================================
# Request/Response Models
# ============================================

class CustomerCreate(BaseModel):
    """Create a new customer (business)."""
    name: str = Field(..., min_length=2, max_length=255)
    email: str


class CustomerResponse(BaseModel):
    """Customer response model."""
    id: int
    name: str
    email: str
    created_at: datetime
    has_whatsapp: bool
    has_agent_config: bool


class MetaSignupInitRequest(BaseModel):
    """Initialize Meta Embedded Signup."""
    customer_id: int


class MetaSignupCompleteRequest(BaseModel):
    """Complete Meta Embedded Signup with OAuth code."""
    customer_id: int
    code: str  # OAuth authorization code from Meta
    phone_number_id: Optional[str] = None  # Sometimes passed directly


class WhatsAppAccountResponse(BaseModel):
    """WhatsApp account response."""
    id: int
    phone_number_id: str
    waba_id: Optional[str]
    display_name: Optional[str]
    phone_number: Optional[str]
    is_active: bool
    verified_at: Optional[datetime]
    created_at: datetime


class AgentConfigRequest(BaseModel):
    """Agent configuration request."""
    customer_id: int
    business_name: str = Field(..., min_length=2, max_length=255)
    tone: str = Field(default="professional", pattern="^(professional|friendly|casual|formal)$")
    behavior_rules: Optional[str] = Field(None, max_length=2000)
    custom_instructions: Optional[str] = Field(None, max_length=5000, description="Override template with custom prompt")
    ai_pause_duration_minutes: int = Field(default=30, ge=5, le=1440)


class AgentConfigResponse(BaseModel):
    """Agent configuration response."""
    id: int
    customer_id: int
    business_name: str
    tone: str
    behavior_rules: Optional[str]
    custom_instructions: Optional[str]
    system_prompt: str
    ai_enabled: bool
    ai_pause_until: Optional[datetime]
    ai_pause_duration_minutes: int
    created_at: datetime
    updated_at: datetime


class ConversationResponse(BaseModel):
    """Conversation response model."""
    id: int
    contact_phone: str
    contact_name: Optional[str]
    status: str
    last_message_at: Optional[datetime]
    unread_count: int
    last_message_preview: Optional[str] = None
    created_at: datetime


class MessageResponse(BaseModel):
    """Message response model."""
    id: int
    content: str
    media_url: Optional[str]
    media_type: Optional[str]
    source: str
    status: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    """Send a manual message request."""
    content: str = Field(..., min_length=1, max_length=4096)
    media_url: Optional[str] = None
    media_type: Optional[str] = None


class AIControlRequest(BaseModel):
    """Control AI for a conversation or customer."""
    action: str = Field(..., pattern="^(pause|resume|toggle)$")
    pause_minutes: Optional[int] = Field(None, ge=5, le=1440)


# ============================================
# Customer Management
# ============================================

@whatsapp_manager_router.get("/customers", response_model=List[CustomerResponse])
async def list_customers(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all customers (businesses) for the current user."""
    result = await db.execute(
        select(Customer)
        .options(selectinload(Customer.whatsapp_accounts), selectinload(Customer.agent_config))
        .where(Customer.user_id == current_user.id)
        .order_by(Customer.created_at.desc())
    )
    customers = result.scalars().all()
    
    return [
        CustomerResponse(
            id=c.id,
            name=c.name,
            email=c.email,
            created_at=c.created_at,
            has_whatsapp=len(c.whatsapp_accounts) > 0,
            has_agent_config=c.agent_config is not None
        )
        for c in customers
    ]


@whatsapp_manager_router.post("/customers", response_model=CustomerResponse)
async def create_customer(
    request: CustomerCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new customer (business)."""
    customer = Customer(
        user_id=current_user.id,
        name=request.name,
        email=request.email
    )
    db.add(customer)
    await db.commit()
    await db.refresh(customer)
    
    logger.info(f"Customer created: {customer.id} by user {current_user.id}")
    
    return CustomerResponse(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        created_at=customer.created_at,
        has_whatsapp=False,
        has_agent_config=False
    )


# ============================================
# Meta Embedded Signup
# ============================================

@whatsapp_manager_router.post("/meta/init-signup")
async def init_meta_signup(
    request: MetaSignupInitRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initialize Meta Embedded Signup flow.
    
    Returns the configuration needed for the frontend to launch
    the Meta Facebook Login SDK popup.
    """
    # Verify customer ownership
    result = await db.execute(
        select(Customer).where(
            Customer.id == request.customer_id,
            Customer.user_id == current_user.id
        )
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found or not owned by you"
        )
    
    # Return Meta SDK configuration
    return {
        "success": True,
        "config": {
            "app_id": META_APP_ID,
            "config_id": os.getenv("META_CONFIG_ID", ""),
            "redirect_uri": os.getenv("OAUTH_REDIRECT_URI", ""),
            "scope": "whatsapp_business_management,whatsapp_business_messaging",
            "response_type": "code",
            "customer_id": customer.id
        }
    }


@whatsapp_manager_router.post("/meta/complete-signup", response_model=WhatsAppAccountResponse)
async def complete_meta_signup(
    request: MetaSignupCompleteRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete Meta Embedded Signup with OAuth authorization code.
    
    Exchanges the code for access token and stores WhatsApp account credentials.
    """
    # Verify customer ownership
    result = await db.execute(
        select(Customer).where(
            Customer.id == request.customer_id,
            Customer.user_id == current_user.id
        )
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    try:
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.get(
                f"{META_API_BASE}/oauth/access_token",
                params={
                    "client_id": META_APP_ID,
                    "client_secret": META_APP_SECRET,
                    "code": request.code,
                    "redirect_uri": os.getenv("OAUTH_REDIRECT_URI", "")
                }
            )
            
            if token_response.status_code != 200:
                logger.error(f"Meta token exchange failed: {token_response.text}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code"
                )
            
            token_data = token_response.json()
            access_token = token_data.get("access_token")
            
            # Get WhatsApp Business Account info
            # In production, you'd fetch the actual WABA_ID and phone numbers
            # For now, we'll use the phone_number_id from the request or generate one
            phone_number_id = request.phone_number_id or f"temp_{customer.id}"
            
            # Debug exchange for development
            if os.getenv("NODE_ENV") == "development":
                access_token = access_token or f"dev_token_{customer.id}"
                phone_number_id = request.phone_number_id or f"dev_phone_{customer.id}"
    
    except httpx.RequestError as e:
        logger.error(f"Meta API request failed: {e}")
        # For development, create with placeholder values
        if os.getenv("NODE_ENV") == "development":
            access_token = f"dev_token_{customer.id}"
            phone_number_id = request.phone_number_id or f"dev_phone_{customer.id}"
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to connect to Meta API"
            )
    
    # Create WhatsApp account record
    whatsapp_account = WhatsAppAccount(
        customer_id=customer.id,
        phone_number_id=phone_number_id,
        waba_id=None,  # Would be fetched from Meta API
        access_token=access_token,
        display_name=customer.name,
        is_active=True,
        verified_at=datetime.utcnow()
    )
    db.add(whatsapp_account)
    await db.commit()
    await db.refresh(whatsapp_account)
    
    logger.info(f"WhatsApp account created: {whatsapp_account.id} for customer {customer.id}")
    
    return WhatsAppAccountResponse(
        id=whatsapp_account.id,
        phone_number_id=whatsapp_account.phone_number_id,
        waba_id=whatsapp_account.waba_id,
        display_name=whatsapp_account.display_name,
        phone_number=whatsapp_account.phone_number,
        is_active=whatsapp_account.is_active,
        verified_at=whatsapp_account.verified_at,
        created_at=whatsapp_account.created_at
    )


@whatsapp_manager_router.get("/accounts", response_model=List[WhatsAppAccountResponse])
async def list_whatsapp_accounts(
    customer_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List WhatsApp accounts for user's customers."""
    query = (
        select(WhatsAppAccount)
        .join(Customer)
        .where(Customer.user_id == current_user.id)
    )
    
    if customer_id:
        query = query.where(WhatsAppAccount.customer_id == customer_id)
    
    result = await db.execute(query)
    accounts = result.scalars().all()
    
    return [
        WhatsAppAccountResponse(
            id=a.id,
            phone_number_id=a.phone_number_id,
            waba_id=a.waba_id,
            display_name=a.display_name,
            phone_number=a.phone_number,
            is_active=a.is_active,
            verified_at=a.verified_at,
            created_at=a.created_at
        )
        for a in accounts
    ]


# ============================================
# Agent Configuration
# ============================================

@whatsapp_manager_router.post("/agents/configure", response_model=AgentConfigResponse)
async def configure_agent(
    request: AgentConfigRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Configure or update AI agent for a customer.
    
    Creates/updates the agent configuration including system prompt.
    The system prompt is compiled from template fields or custom instructions.
    """
    # Verify customer ownership
    result = await db.execute(
        select(Customer).where(
            Customer.id == request.customer_id,
            Customer.user_id == current_user.id
        )
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Compile system prompt
    system_prompt = AgentConfig.compile_system_prompt(
        business_name=request.business_name,
        tone=request.tone,
        behavior_rules=request.behavior_rules,
        custom_instructions=request.custom_instructions
    )
    
    # Check for existing config
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.customer_id == customer.id)
    )
    existing_config = result.scalar_one_or_none()
    
    if existing_config:
        # Update existing
        existing_config.business_name = request.business_name
        existing_config.tone = request.tone
        existing_config.behavior_rules = request.behavior_rules
        existing_config.custom_instructions = request.custom_instructions
        existing_config.system_prompt = system_prompt
        existing_config.ai_pause_duration_minutes = request.ai_pause_duration_minutes
        agent_config = existing_config
    else:
        # Create new
        agent_config = AgentConfig(
            customer_id=customer.id,
            business_name=request.business_name,
            tone=request.tone,
            behavior_rules=request.behavior_rules,
            custom_instructions=request.custom_instructions,
            system_prompt=system_prompt,
            ai_pause_duration_minutes=request.ai_pause_duration_minutes,
            ai_enabled=True
        )
        db.add(agent_config)
    
    await db.commit()
    await db.refresh(agent_config)
    
    logger.info(f"Agent config {'updated' if existing_config else 'created'}: {agent_config.id}")
    
    return AgentConfigResponse(
        id=agent_config.id,
        customer_id=agent_config.customer_id,
        business_name=agent_config.business_name,
        tone=agent_config.tone,
        behavior_rules=agent_config.behavior_rules,
        custom_instructions=agent_config.custom_instructions,
        system_prompt=agent_config.system_prompt,
        ai_enabled=agent_config.ai_enabled,
        ai_pause_until=agent_config.ai_pause_until,
        ai_pause_duration_minutes=agent_config.ai_pause_duration_minutes,
        created_at=agent_config.created_at,
        updated_at=agent_config.updated_at
    )


@whatsapp_manager_router.get("/agents/config/{customer_id}", response_model=AgentConfigResponse)
async def get_agent_config(
    customer_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get agent configuration for a customer."""
    result = await db.execute(
        select(AgentConfig)
        .join(Customer)
        .where(
            AgentConfig.customer_id == customer_id,
            Customer.user_id == current_user.id
        )
    )
    agent_config = result.scalar_one_or_none()
    
    if not agent_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent configuration not found"
        )
    
    return AgentConfigResponse(
        id=agent_config.id,
        customer_id=agent_config.customer_id,
        business_name=agent_config.business_name,
        tone=agent_config.tone,
        behavior_rules=agent_config.behavior_rules,
        custom_instructions=agent_config.custom_instructions,
        system_prompt=agent_config.system_prompt,
        ai_enabled=agent_config.ai_enabled,
        ai_pause_until=agent_config.ai_pause_until,
        ai_pause_duration_minutes=agent_config.ai_pause_duration_minutes,
        created_at=agent_config.created_at,
        updated_at=agent_config.updated_at
    )


@whatsapp_manager_router.post("/agents/config/{customer_id}/ai-control")
async def control_ai(
    customer_id: int,
    request: AIControlRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Control AI (pause/resume) for a customer's agent."""
    result = await db.execute(
        select(AgentConfig)
        .join(Customer)
        .where(
            AgentConfig.customer_id == customer_id,
            Customer.user_id == current_user.id
        )
    )
    agent_config = result.scalar_one_or_none()
    
    if not agent_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent configuration not found"
        )
    
    if request.action == "pause":
        minutes = request.pause_minutes or agent_config.ai_pause_duration_minutes
        agent_config.pause_ai(minutes)
        message = f"AI paused for {minutes} minutes"
    elif request.action == "resume":
        agent_config.resume_ai()
        message = "AI resumed"
    elif request.action == "toggle":
        agent_config.ai_enabled = not agent_config.ai_enabled
        message = f"AI {'enabled' if agent_config.ai_enabled else 'disabled'}"
    
    await db.commit()
    
    return {
        "success": True,
        "message": message,
        "ai_enabled": agent_config.ai_enabled,
        "ai_pause_until": agent_config.ai_pause_until.isoformat() if agent_config.ai_pause_until else None
    }


# ============================================
# Inbox Management
# ============================================

@whatsapp_manager_router.get("/inbox/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    customer_id: Optional[int] = None,
    status_filter: Optional[str] = Query(None, pattern="^(active|archived|blocked)$"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List conversations for user's customers."""
    query = (
        select(Conversation)
        .join(Customer)
        .where(Customer.user_id == current_user.id)
        .order_by(desc(Conversation.last_message_at))
        .limit(limit)
        .offset(offset)
    )
    
    if customer_id:
        query = query.where(Conversation.customer_id == customer_id)
    if status_filter:
        query = query.where(Conversation.status == status_filter)
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    
    # Get last message preview for each conversation
    responses = []
    for conv in conversations:
        # Get last message
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(desc(Message.created_at))
            .limit(1)
        )
        last_msg = msg_result.scalar_one_or_none()
        
        responses.append(ConversationResponse(
            id=conv.id,
            contact_phone=conv.contact_phone,
            contact_name=conv.contact_name,
            status=conv.status,
            last_message_at=conv.last_message_at,
            unread_count=conv.unread_count,
            last_message_preview=last_msg.content[:100] if last_msg else None,
            created_at=conv.created_at
        ))
    
    return responses


@whatsapp_manager_router.get("/inbox/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    limit: int = Query(50, ge=1, le=200),
    before_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get messages for a conversation (paginated)."""
    # Verify conversation ownership
    result = await db.execute(
        select(Conversation)
        .join(Customer)
        .where(
            Conversation.id == conversation_id,
            Customer.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Mark as read
    conversation.unread_count = 0
    
    # Get messages
    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at))
        .limit(limit)
    )
    
    if before_id:
        query = query.where(Message.id < before_id)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    await db.commit()
    
    return [
        MessageResponse(
            id=m.id,
            content=m.content,
            media_url=m.media_url,
            media_type=m.media_type,
            source=m.source.value,
            status=m.status.value,
            created_at=m.created_at
        )
        for m in reversed(messages)  # Return in chronological order
    ]


@whatsapp_manager_router.post("/inbox/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_manual_message(
    conversation_id: int,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a manual message (human intervention).
    
    This pauses AI auto-replies for the configured duration.
    """
    # Verify conversation ownership and get related data
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.whatsapp_account))
        .join(Customer)
        .where(
            Conversation.id == conversation_id,
            Customer.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get agent config to know pause duration
    config_result = await db.execute(
        select(AgentConfig).where(AgentConfig.customer_id == conversation.customer_id)
    )
    agent_config = config_result.scalar_one_or_none()
    
    # Pause AI for this conversation
    if agent_config:
        from datetime import timedelta
        conversation.ai_paused_until = datetime.utcnow() + timedelta(
            minutes=agent_config.ai_pause_duration_minutes
        )
    
    # Create message record
    message = Message(
        conversation_id=conversation.id,
        content=request.content,
        media_url=request.media_url,
        media_type=request.media_type,
        source=MessageSource.HUMAN,
        status=MessageStatus.PENDING
    )
    db.add(message)
    
    # Update conversation
    conversation.last_message_at = datetime.utcnow()
    
    await db.flush()
    
    # Send via WhatsApp API
    try:
        whatsapp_message_id = await send_whatsapp_message(
            phone_number_id=conversation.whatsapp_account.phone_number_id,
            access_token=conversation.whatsapp_account.access_token,
            recipient=conversation.contact_phone,
            message=request.content
        )
        message.whatsapp_message_id = whatsapp_message_id
        message.status = MessageStatus.SENT
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        message.status = MessageStatus.FAILED
    
    await db.commit()
    await db.refresh(message)
    
    return MessageResponse(
        id=message.id,
        content=message.content,
        media_url=message.media_url,
        media_type=message.media_type,
        source=message.source.value,
        status=message.status.value,
        created_at=message.created_at
    )


# ============================================
# WhatsApp API Helper
# ============================================

async def send_whatsapp_message(
    phone_number_id: str,
    access_token: str,
    recipient: str,
    message: str
) -> str:
    """
    Send a message via WhatsApp Cloud API.
    
    Returns the WhatsApp message ID.
    """
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
            raise Exception(f"WhatsApp API error: {response.status_code}")
        
        data = response.json()
        return data.get("messages", [{}])[0].get("id", "")
