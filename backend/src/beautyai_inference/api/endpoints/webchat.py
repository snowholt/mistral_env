"""
Web Chat Widget API endpoints.

Provides public endpoints for embedding chat widget on customer websites:
- Session creation (rate-limited by IP)
- Message sending with AI response
- Session management

Authentication is via widget tokens (API keys) rather than JWT.
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, status, Request, Header
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ...database.connection import get_db
from ...database.models import (
    Customer, WidgetToken, WebChatSession, WebChatMessage,
    AgentConfig, UsageEvent, UsageEventType,
)
from ...services.cache import get_redis, RedisClient, RateLimiter

logger = logging.getLogger(__name__)

webchat_router = APIRouter(prefix="/api/v1/webchat", tags=["webchat"])

# Rate limiter for widget endpoints (60 requests per minute per IP)
widget_rate_limiter = RateLimiter(requests=60, window=60, key_prefix="ratelimit:widget:")


# ============================================
# Request/Response Models
# ============================================

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""
    widget_token: str = Field(..., description="Widget API token from customer dashboard")
    visitor_name: Optional[str] = Field(None, max_length=255)
    visitor_email: Optional[str] = Field(None, max_length=255)
    page_url: Optional[str] = Field(None, description="URL where widget is loaded")


class CreateSessionResponse(BaseModel):
    """Response with new session details."""
    success: bool
    session_token: str
    greeting_message: str
    expires_at: datetime
    widget_config: dict


class SendMessageRequest(BaseModel):
    """Request to send a message in a session."""
    session_token: str
    message: str = Field(..., min_length=1, max_length=4000)


class SendMessageResponse(BaseModel):
    """Response with AI reply."""
    success: bool
    message_id: int
    ai_response: str
    input_tokens: int
    output_tokens: int


class SessionHistoryResponse(BaseModel):
    """Response with chat history."""
    success: bool
    messages: List[dict]


# ============================================
# Helper Functions
# ============================================

async def validate_widget_token(
    token: str,
    request: Request,
    db: AsyncSession,
    redis: RedisClient,
) -> tuple[WidgetToken, Customer]:
    """
    Validate widget token and return token + customer.
    
    Checks:
    - Token exists and is active
    - Token is not expired
    - Domain is whitelisted (if applicable)
    """
    token_hash = WidgetToken.hash_token(token)
    
    # Try cache first
    cached = await redis.get_json(f"widget:token:{token_hash}")
    if cached:
        customer_id = cached.get("customer_id")
        # Still need to fetch from DB for relationships
        result = await db.execute(
            select(Customer)
            .options(selectinload(Customer.agent_config))
            .where(Customer.id == customer_id)
        )
        customer = result.scalar_one_or_none()
        if not customer:
            raise HTTPException(status_code=401, detail="Invalid widget token")
        
        # Fetch token for domain check
        token_result = await db.execute(
            select(WidgetToken).where(WidgetToken.token_hash == token_hash)
        )
        widget_token = token_result.scalar_one_or_none()
        if not widget_token:
            raise HTTPException(status_code=401, detail="Invalid widget token")
        
        return widget_token, customer
    
    # Fetch from database
    result = await db.execute(
        select(WidgetToken)
        .options(selectinload(WidgetToken.customer).selectinload(Customer.agent_config))
        .where(WidgetToken.token_hash == token_hash)
    )
    widget_token = result.scalar_one_or_none()
    
    if not widget_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid widget token"
        )
    
    if not widget_token.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Widget token is expired or inactive"
        )
    
    # Check domain whitelist
    origin = request.headers.get("origin", "")
    if origin:
        # Extract domain from origin
        try:
            from urllib.parse import urlparse
            domain = urlparse(origin).netloc
            if not widget_token.is_domain_allowed(domain):
                logger.warning(f"Domain {domain} not allowed for widget token {widget_token.token_prefix}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Domain not whitelisted for this widget"
                )
        except Exception as e:
            logger.error(f"Error checking domain: {e}")
    
    customer = widget_token.customer
    if not customer or not customer.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Customer account is inactive"
        )
    
    # Cache the token validation
    await redis.set_json(
        f"widget:token:{token_hash}",
        {"customer_id": customer.id},
        expire=timedelta(minutes=30)
    )
    
    # Update token usage stats
    widget_token.last_used_at = datetime.now()
    widget_token.request_count += 1
    
    return widget_token, customer


async def validate_session(
    session_token: str,
    db: AsyncSession,
) -> WebChatSession:
    """Validate session token and return session."""
    result = await db.execute(
        select(WebChatSession)
        .options(selectinload(WebChatSession.customer).selectinload(Customer.agent_config))
        .where(WebChatSession.session_token == session_token)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )
    
    if not session.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired"
        )
    
    return session


# ============================================
# API Endpoints
# ============================================

@webchat_router.post("/session", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    """
    Create a new chat session for website visitor.
    
    Rate limited by IP address.
    Returns session token and widget configuration.
    """
    # Rate limiting
    await widget_rate_limiter.check(http_request, redis)
    
    # Validate widget token
    widget_token, customer = await validate_widget_token(
        request.widget_token, http_request, db, redis
    )
    
    # Create session
    session = WebChatSession(
        customer_id=customer.id,
        session_token=WebChatSession.generate_session_token(),
        visitor_name=request.visitor_name,
        visitor_email=request.visitor_email,
        page_url=request.page_url,
        referrer=http_request.headers.get("referer"),
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None,
        expires_at=datetime.now() + timedelta(hours=2),
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    logger.info(f"Web chat session created for customer {customer.id}: {session.id}")
    
    # Build widget config (colors, logo - per requirements, limited customization)
    widget_config = {
        "primary_color": customer.widget_primary_color,
        "secondary_color": customer.widget_secondary_color,
        "logo_url": customer.widget_logo_url,
        "business_name": customer.agent_config.business_name if customer.agent_config else customer.name,
    }
    
    return CreateSessionResponse(
        success=True,
        session_token=session.session_token,
        greeting_message=customer.widget_greeting_message,
        expires_at=session.expires_at,
        widget_config=widget_config,
    )


@webchat_router.post("/message", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    """
    Send a message and receive AI response.
    
    Uses the customer's agent configuration for AI behavior.
    Rate limited by IP address.
    """
    # Rate limiting
    await widget_rate_limiter.check(http_request, redis)
    
    # Validate session
    session = await validate_session(request.session_token, db)
    customer = session.customer
    
    # Save user message
    user_message = WebChatMessage(
        session_id=session.id,
        content=request.message,
        role="user",
    )
    db.add(user_message)
    
    # Update session activity
    session.last_message_at = datetime.now()
    # Extend session expiration on activity
    session.expires_at = datetime.now() + timedelta(hours=2)
    
    await db.flush()
    
    # Build conversation history for context
    await db.refresh(session, ["messages"])
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in session.messages[-20:]  # Last 20 messages for context
    ]
    
    # Generate AI response
    try:
        ai_response, input_tokens, output_tokens = await generate_ai_response(
            customer=customer,
            conversation_history=conversation_history,
        )
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        ai_response = "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."  # Arabic fallback
        input_tokens = 0
        output_tokens = 0
    
    # Save AI response
    ai_message = WebChatMessage(
        session_id=session.id,
        content=ai_response,
        role="assistant",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    db.add(ai_message)
    
    # Record usage event
    usage_event = UsageEvent(
        customer_id=customer.id,
        event_type=UsageEventType.WEBCHAT_MESSAGE,
        quantity=1,
        event_metadata={
            "session_id": session.id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )
    db.add(usage_event)
    
    # Update subscription usage if available
    if customer.subscription:
        customer.subscription.messages_used += 1
        customer.subscription.tokens_used += (input_tokens + output_tokens)
    
    await db.commit()
    await db.refresh(ai_message)
    
    logger.info(f"Web chat message processed for session {session.id}: {input_tokens}+{output_tokens} tokens")
    
    return SendMessageResponse(
        success=True,
        message_id=ai_message.id,
        ai_response=ai_response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@webchat_router.get("/session/{session_token}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get chat history for a session.
    
    Used to restore conversation on page reload.
    """
    session = await validate_session(session_token, db)
    
    await db.refresh(session, ["messages"])
    
    return SessionHistoryResponse(
        success=True,
        messages=[
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in session.messages
        ]
    )


@webchat_router.post("/session/{session_token}/end")
async def end_session(
    session_token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    End a chat session explicitly.
    """
    session = await validate_session(session_token, db)
    
    session.is_active = False
    await db.commit()
    
    logger.info(f"Web chat session ended: {session.id}")
    
    return {"success": True, "message": "Session ended"}


# ============================================
# AI Response Generation
# ============================================

async def generate_ai_response(
    customer: Customer,
    conversation_history: List[dict],
) -> tuple[str, int, int]:
    """
    Generate AI response using customer's agent config with persistent Qwen3-14B model.
    
    Returns: (response_text, input_tokens, output_tokens)
    """
    # Get system prompt from agent config or use default
    agent_config = customer.agent_config
    if agent_config:
        system_prompt = agent_config.system_prompt
    else:
        system_prompt = f"""أنت مساعد ذكاء اصطناعي لشركة {customer.name}.

مهمتك:
- الرد بطريقة ودية ومهنية باللغة العربية
- تقديم معلومات دقيقة ومفيدة
- مساعدة العملاء في استفساراتهم
- كن واضحاً ومختصراً في الردود"""
    
    # Build messages for LLM
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    
    # Use persistent Qwen3-14B model directly
    try:
        from ...core.persistent_model_manager import get_persistent_model_manager
        from ...inference_engines.llamacpp_engine import LlamaCppEngine
        
        # Get persistent model manager
        persistent_mgr = get_persistent_model_manager()
        
        # Get the persistent LLM model (Qwen3-14B)
        llm_model = persistent_mgr.get_llm_model()
        
        if llm_model and isinstance(llm_model, LlamaCppEngine):
            logger.info(f"Using persistent Qwen3-14B model for webchat")
            
            # Generate response using persistent model chat method
            content = llm_model.chat(
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                enable_thinking=False,  # Explicitly disable thinking mode
            )
            
            # Llama.cpp chat() returns just the string content
            # Token counts not available from chat method
            input_tokens = 0  # Could be estimated
            output_tokens = 0  # Could be estimated
            
            return content, input_tokens, output_tokens
        else:
            logger.warning("Persistent LLM model not available, using fallback")
            raise Exception("Persistent model not loaded")
            
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        # Fallback response
        return (
            "عذراً، حدث خطأ مؤقت. يرجى المحاولة مرة أخرى.",
            0,
            0
        )
