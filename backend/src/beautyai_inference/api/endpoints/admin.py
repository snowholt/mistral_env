"""
Admin API endpoints for GMAI.sa platform management.

Provides admin-only endpoints for:
- Customer management (list, view, update, suspend)
- Usage metrics and analytics
- Subscription and billing management
- System monitoring

Access restricted to users with admin role (@gmai.sa domain with invite).
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from sqlalchemy.orm import selectinload

from ...database.connection import get_db
from ...database.models import (
    User, Customer, WhatsAppAccount, Subscription, Plan,
    UsageEvent, UsageEventType, UserRole, Conversation, Message
)
from ...auth.dependencies import get_current_active_user
from ...services.system import StatusService

logger = logging.getLogger(__name__)

admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
status_service = StatusService()


# ============================================
# Admin Auth Dependency
# ============================================

async def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency to require admin role for endpoint access."""
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# ============================================
# Response Models
# ============================================

class CustomerSummary(BaseModel):
    id: int
    name: str
    email: str
    user_email: Optional[str]
    is_active: bool
    whatsapp_accounts_count: int
    subscription_status: Optional[str]
    plan_name: Optional[str]
    created_at: datetime
    total_messages: int
    total_conversations: int


class CustomerDetail(BaseModel):
    id: int
    name: str
    email: str
    timezone: str
    locale: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    user: Optional[dict]
    whatsapp_accounts: List[dict]
    subscription: Optional[dict]
    agent_config: Optional[dict]
    
    # Stats
    total_conversations: int
    total_messages: int
    messages_this_month: int


class UsageMetrics(BaseModel):
    total_customers: int
    active_customers: int
    total_users: int
    verified_users: int
    
    total_whatsapp_accounts: int
    active_whatsapp_accounts: int
    
    total_conversations: int
    total_messages: int
    messages_today: int
    messages_this_week: int
    messages_this_month: int
    
    # By type
    messages_by_source: dict
    
    # Trends
    customer_growth_30d: int
    message_growth_30d: float  # percentage
    uptime_hours: Optional[float] = None


class RevenueMetrics(BaseModel):
    monthly_recurring_revenue: float
    total_subscriptions: int
    active_subscriptions: int
    trial_subscriptions: int
    churned_subscriptions_30d: int
    
    revenue_by_plan: List[dict]


class CustomerUpdateRequest(BaseModel):
    """Request to update customer settings."""
    is_active: Optional[bool] = None
    name: Optional[str] = None


# ============================================
# Customer Management Endpoints
# ============================================

@admin_router.get("/customers")
async def list_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    sort_by: str = Query("created_at", regex="^(created_at|name|email)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    List all customers with pagination and filtering.
    """
    # Build query
    query = select(Customer).options(
        selectinload(Customer.user),
        selectinload(Customer.whatsapp_accounts),
        selectinload(Customer.subscription).selectinload(Subscription.plan),
    )
    
    # Apply filters
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (Customer.name.ilike(search_filter)) |
            (Customer.email.ilike(search_filter))
        )
    
    if is_active is not None:
        query = query.where(Customer.is_active == is_active)
    
    # Apply sorting
    sort_column = getattr(Customer, sort_by)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    # Get total count
    count_query = select(func.count(Customer.id))
    if search:
        count_query = count_query.where(
            (Customer.name.ilike(search_filter)) |
            (Customer.email.ilike(search_filter))
        )
    if is_active is not None:
        count_query = count_query.where(Customer.is_active == is_active)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    customers = result.scalars().all()
    
    # Build response
    customer_list = []
    for c in customers:
        # Get message count for this customer
        msg_count = await db.execute(
            select(func.count(Message.id))
            .join(Conversation)
            .where(Conversation.customer_id == c.id)
        )
        total_messages = msg_count.scalar() or 0
        
        conv_count = await db.execute(
            select(func.count(Conversation.id))
            .where(Conversation.customer_id == c.id)
        )
        total_conversations = conv_count.scalar() or 0
        
        customer_list.append({
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "user_email": c.user.email if c.user else None,
            "is_active": c.is_active,
            "whatsapp_accounts_count": len(c.whatsapp_accounts),
            "subscription_status": c.subscription.status.value if c.subscription else None,
            "plan_name": c.subscription.plan.name if c.subscription and c.subscription.plan else None,
            "created_at": c.created_at.isoformat(),
            "total_messages": total_messages,
            "total_conversations": total_conversations,
        })
    
    return {
        "success": True,
        "total": total,
        "skip": skip,
        "limit": limit,
        "customers": customer_list,
    }


@admin_router.get("/customers/{customer_id}")
async def get_customer_detail(
    customer_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a specific customer.
    """
    result = await db.execute(
        select(Customer)
        .options(
            selectinload(Customer.user),
            selectinload(Customer.whatsapp_accounts),
            selectinload(Customer.subscription).selectinload(Subscription.plan),
            selectinload(Customer.agent_config),
        )
        .where(Customer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Get stats
    total_conv = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.customer_id == customer_id)
    )
    total_conversations = total_conv.scalar() or 0
    
    total_msg = await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(Conversation.customer_id == customer_id)
    )
    total_messages = total_msg.scalar() or 0
    
    # Messages this month
    month_start = datetime.now(timezone.utc).replace(tzinfo=None).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_msg = await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(
            and_(
                Conversation.customer_id == customer_id,
                Message.created_at >= month_start
            )
        )
    )
    messages_this_month = month_msg.scalar() or 0
    
    return {
        "success": True,
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "email": customer.email,
            "timezone": customer.timezone,
            "locale": customer.locale,
            "is_active": customer.is_active,
            "created_at": customer.created_at.isoformat(),
            "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
            
            "user": {
                "id": customer.user.id,
                "email": customer.user.email,
                "full_name": customer.user.full_name,
                "is_verified": customer.user.is_verified,
                "created_at": customer.user.created_at.isoformat(),
            } if customer.user else None,
            
            "whatsapp_accounts": [
                {
                    "id": wa.id,
                    "phone_number": wa.phone_number,
                    "display_name": wa.display_name,
                    "is_active": wa.is_active,
                    "created_at": wa.created_at.isoformat(),
                }
                for wa in customer.whatsapp_accounts
            ],
            
            "subscription": {
                "id": customer.subscription.id,
                "status": customer.subscription.status.value,
                "plan_name": customer.subscription.plan.name,
                "plan_price": float(customer.subscription.plan.price_monthly),
                "current_period_end": customer.subscription.current_period_end.isoformat(),
                "messages_used": customer.subscription.messages_used,
                "message_limit": customer.subscription.plan.message_limit,
            } if customer.subscription else None,
            
            "agent_config": {
                "business_name": customer.agent_config.business_name,
                "tone": customer.agent_config.tone,
                "ai_enabled": customer.agent_config.ai_enabled,
            } if customer.agent_config else None,
            
            "stats": {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "messages_this_month": messages_this_month,
            }
        }
    }


@admin_router.patch("/customers/{customer_id}")
async def update_customer(
    customer_id: int,
    request: CustomerUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Update customer settings (activate/deactivate, rename).
    """
    result = await db.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Apply updates
    if request.is_active is not None:
        customer.is_active = request.is_active
        logger.info(f"Admin {admin.email} {'activated' if request.is_active else 'deactivated'} customer {customer_id}")
    
    if request.name is not None:
        customer.name = request.name
    
    await db.commit()
    
    return {
        "success": True,
        "message": "Customer updated successfully",
        "customer_id": customer_id,
    }


# ============================================
# Usage Metrics Endpoints
# ============================================

@admin_router.get("/metrics/usage")
async def get_usage_metrics(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get platform-wide usage metrics and analytics.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = now - timedelta(days=30)
    
    # Customer metrics
    total_customers = (await db.execute(select(func.count(Customer.id)))).scalar() or 0
    active_customers = (await db.execute(
        select(func.count(Customer.id)).where(Customer.is_active == True)
    )).scalar() or 0
    
    # User metrics
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    verified_users = (await db.execute(
        select(func.count(User.id)).where(User.is_verified == True)
    )).scalar() or 0
    
    # WhatsApp account metrics
    total_wa = (await db.execute(select(func.count(WhatsAppAccount.id)))).scalar() or 0
    active_wa = (await db.execute(
        select(func.count(WhatsAppAccount.id)).where(WhatsAppAccount.is_active == True)
    )).scalar() or 0
    
    # Message metrics
    total_conversations = (await db.execute(select(func.count(Conversation.id)))).scalar() or 0
    total_messages = (await db.execute(select(func.count(Message.id)))).scalar() or 0
    
    messages_today = (await db.execute(
        select(func.count(Message.id)).where(Message.created_at >= today_start)
    )).scalar() or 0
    
    messages_this_week = (await db.execute(
        select(func.count(Message.id)).where(Message.created_at >= week_start)
    )).scalar() or 0
    
    messages_this_month = (await db.execute(
        select(func.count(Message.id)).where(Message.created_at >= month_start)
    )).scalar() or 0
    
    # Messages by source
    from ...database.models import MessageSource
    messages_by_source = {}
    for source in MessageSource:
        count = (await db.execute(
            select(func.count(Message.id)).where(Message.source == source)
        )).scalar() or 0
        messages_by_source[source.value] = count
    
    # Growth metrics
    new_customers_30d = (await db.execute(
        select(func.count(Customer.id)).where(Customer.created_at >= thirty_days_ago)
    )).scalar() or 0
    
    # Message growth (compare this week to previous week)
    prev_week_start = week_start - timedelta(days=7)
    messages_prev_week = (await db.execute(
        select(func.count(Message.id)).where(
            and_(
                Message.created_at >= prev_week_start,
                Message.created_at < week_start
            )
        )
    )).scalar() or 0
    
    if messages_prev_week > 0:
        message_growth = ((messages_this_week - messages_prev_week) / messages_prev_week) * 100
    else:
        message_growth = 100.0 if messages_this_week > 0 else 0.0
    
    # Get system uptime
    try:
        status = status_service.get_comprehensive_status()
        uptime_hours = round(status.uptime_seconds / 3600, 2)
    except Exception:
        uptime_hours = 0.0

    return {
        "success": True,
        "metrics": {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "total_users": total_users,
            "verified_users": verified_users,
            "total_whatsapp_accounts": total_wa,
            "active_whatsapp_accounts": active_wa,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "messages_today": messages_today,
            "messages_this_week": messages_this_week,
            "messages_this_month": messages_this_month,
            "messages_by_source": messages_by_source,
            "customer_growth_30d": new_customers_30d,
            "message_growth_30d": round(message_growth, 2),
            "uptime_hours": uptime_hours,
        },
        "generated_at": now.isoformat(),
    }


@admin_router.get("/metrics/revenue")
async def get_revenue_metrics(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get revenue and subscription metrics.
    """
    from ...database.models import SubscriptionStatus
    
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    thirty_days_ago = now - timedelta(days=30)
    
    # Subscription counts
    total_subs = (await db.execute(select(func.count(Subscription.id)))).scalar() or 0
    
    active_subs = (await db.execute(
        select(func.count(Subscription.id))
        .where(Subscription.status == SubscriptionStatus.ACTIVE)
    )).scalar() or 0
    
    trial_subs = (await db.execute(
        select(func.count(Subscription.id))
        .where(Subscription.status == SubscriptionStatus.TRIAL)
    )).scalar() or 0
    
    # Churned in last 30 days
    churned_subs = (await db.execute(
        select(func.count(Subscription.id))
        .where(
            and_(
                Subscription.status == SubscriptionStatus.CANCELED,
                Subscription.canceled_at >= thirty_days_ago
            )
        )
    )).scalar() or 0
    
    # MRR calculation
    mrr_result = await db.execute(
        select(func.sum(Plan.price_monthly))
        .join(Subscription)
        .where(Subscription.status == SubscriptionStatus.ACTIVE)
    )
    mrr = float(mrr_result.scalar() or 0)
    
    # Revenue by plan
    plan_revenue = await db.execute(
        select(
            Plan.name,
            Plan.price_monthly,
            func.count(Subscription.id).label("subscriber_count")
        )
        .join(Subscription)
        .where(Subscription.status == SubscriptionStatus.ACTIVE)
        .group_by(Plan.id, Plan.name, Plan.price_monthly)
    )
    
    revenue_by_plan = [
        {
            "plan_name": row.name,
            "price": float(row.price_monthly),
            "subscribers": row.subscriber_count,
            "revenue": float(row.price_monthly * row.subscriber_count),
        }
        for row in plan_revenue
    ]
    
    return {
        "success": True,
        "revenue": {
            "monthly_recurring_revenue": mrr,
            "total_subscriptions": total_subs,
            "active_subscriptions": active_subs,
            "trial_subscriptions": trial_subs,
            "churned_subscriptions_30d": churned_subs,
            "revenue_by_plan": revenue_by_plan,
        },
        "generated_at": now.isoformat(),
    }


# ============================================
# User Management Endpoints
# ============================================

@admin_router.get("/users")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[str] = Query(None, regex="^(user|admin)$"),
    is_verified: Optional[bool] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    List all users with pagination and filtering.
    """
    query = select(User)
    
    if role:
        query = query.where(User.role == UserRole(role))
    
    if is_verified is not None:
        query = query.where(User.is_verified == is_verified)
    
    query = query.order_by(desc(User.created_at))
    
    # Get total
    count_query = select(func.count(User.id))
    if role:
        count_query = count_query.where(User.role == UserRole(role))
    if is_verified is not None:
        count_query = count_query.where(User.is_verified == is_verified)
    
    total = (await db.execute(count_query)).scalar() or 0
    
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    users = result.scalars().all()
    
    return {
        "success": True,
        "total": total,
        "skip": skip,
        "limit": limit,
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role.value,
                "is_active": u.is_active,
                "is_verified": u.is_verified,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ],
    }


@admin_router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str = Query(..., regex="^(user|admin)$"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Update user role. Can only promote/demote users.
    Admin cannot demote themselves.
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Only allow admin role for @gmai.sa emails
    if role == "admin" and not User.should_be_admin(user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin role can only be assigned to @gmai.sa email addresses"
        )
    
    user.role = UserRole(role)
    await db.commit()
    
    logger.info(f"Admin {admin.email} changed role of user {user_id} to {role}")
    
    return {
        "success": True,
        "message": f"User role updated to {role}",
        "user_id": user_id,
    }


# ============================================
# Frontend Compatibility Aliases
# ============================================

@admin_router.get("/businesses")
async def list_businesses_alias(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    sort_by: str = Query("created_at", regex="^(created_at|name|email)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Alias for /customers to match frontend expectations."""
    return await list_customers(
        skip=skip,
        limit=limit,
        search=search,
        is_active=is_active,
        sort_by=sort_by,
        sort_order=sort_order,
        admin=admin,
        db=db
    )

@admin_router.get("/metrics")
async def get_metrics_alias(
    range: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get combined system and platform metrics for the admin dashboard.
    Returns data in the format expected by the frontend.
    """
    from ...services.system import MemoryService
    import psutil
    import shutil
    
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    
    # Initialize services
    memory_service = MemoryService()
    
    # Get system status
    try:
        status = status_service.get_comprehensive_status()
        uptime_hours = round(status.uptime_seconds / 3600, 2)
        memory_status = status.memory_status
    except Exception:
        uptime_hours = 0.0
        memory_status = None
    
    # System metrics
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    # Memory metrics
    if memory_status:
        memory_total_gb = memory_status.system_stats.get("total_gb", 0)
        memory_used_gb = memory_status.system_stats.get("used_gb", 0)
        memory_usage = memory_status.system_stats.get("percent", 0)
    else:
        mem = psutil.virtual_memory()
        memory_total_gb = round(mem.total / (1024**3), 2)
        memory_used_gb = round(mem.used / (1024**3), 2)
        memory_usage = mem.percent
    
    # Disk metrics
    disk = shutil.disk_usage("/")
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_used_gb = round(disk.used / (1024**3), 2)
    disk_usage = round((disk.used / disk.total) * 100, 1)
    
    # GPU metrics
    gpu_usage = 0.0
    gpu_memory_usage = 0.0
    gpu_memory_total_gb = 0.0
    gpu_memory_used_gb = 0.0
    
    if memory_status and memory_status.has_gpu and memory_status.gpu_stats:
        gpu_stat = memory_status.gpu_stats[0]
        gpu_usage = round(gpu_stat.get("gpu_utilization", 0), 1)
        gpu_memory_total_gb = round(gpu_stat.get("total_memory", 0) / (1024**3), 2)
        gpu_memory_used_gb = round(gpu_stat.get("memory_used", 0) / (1024**3), 2)
        if gpu_memory_total_gb > 0:
            gpu_memory_usage = round((gpu_memory_used_gb / gpu_memory_total_gb) * 100, 1)
    
    # Platform metrics from database
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    total_businesses = (await db.execute(select(func.count(Customer.id)))).scalar() or 0
    
    # Active users (users who have logged in today - simplified to verified users for now)
    active_users_24h = (await db.execute(
        select(func.count(User.id)).where(User.is_verified == True)
    )).scalar() or 0
    
    # Messages today and this week
    total_messages_today = (await db.execute(
        select(func.count(Message.id)).where(Message.created_at >= today_start)
    )).scalar() or 0
    
    total_messages_week = (await db.execute(
        select(func.count(Message.id)).where(Message.created_at >= week_start)
    )).scalar() or 0
    
    # Knowledge base stats (if table exists)
    try:
        from ...database.models import Document, Chunk
        total_kb_documents = (await db.execute(select(func.count(Document.id)))).scalar() or 0
        total_kb_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0
    except Exception:
        total_kb_documents = 0
        total_kb_chunks = 0
    
    # Voice sessions (placeholder - would need actual tracking)
    total_voice_sessions = 0
    
    # API metrics (placeholder - would need request tracking)
    api_requests_today = 0
    avg_response_time_ms = 150  # Placeholder
    error_rate_percent = 0.0  # Placeholder
    
    return {
        "system": {
            "cpu_usage": round(cpu_usage, 1),
            "memory_usage": round(memory_usage, 1),
            "memory_total_gb": memory_total_gb,
            "memory_used_gb": memory_used_gb,
            "disk_usage": disk_usage,
            "disk_total_gb": disk_total_gb,
            "disk_used_gb": disk_used_gb,
            "gpu_usage": gpu_usage,
            "gpu_memory_usage": gpu_memory_usage,
            "gpu_memory_total_gb": gpu_memory_total_gb,
            "gpu_memory_used_gb": gpu_memory_used_gb,
            "uptime_hours": uptime_hours,
        },
        "platform": {
            "total_users": total_users,
            "active_users_24h": active_users_24h,
            "total_businesses": total_businesses,
            "total_messages_today": total_messages_today,
            "total_messages_week": total_messages_week,
            "total_voice_sessions": total_voice_sessions,
            "avg_response_time_ms": avg_response_time_ms,
            "total_kb_documents": total_kb_documents,
            "total_kb_chunks": total_kb_chunks,
            "api_requests_today": api_requests_today,
            "error_rate_percent": error_rate_percent,
        },
        "timestamp": now.isoformat(),
    }


# ============================================
# GPU Benchmarking Endpoints
# ============================================

class BenchmarkRequest(BaseModel):
    """Request for GPU benchmarking."""
    prompt: str = Field(
        default="What are the benefits of healthy skincare?",
        description="Prompt to use for benchmarking"
    )
    max_tokens: int = Field(default=100, ge=10, le=500)
    num_runs: int = Field(default=3, ge=1, le=10, description="Number of benchmark runs")


class BenchmarkResult(BaseModel):
    """Single benchmark result."""
    run_number: int
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


class BenchmarkResponse(BaseModel):
    """Benchmark response with aggregated stats."""
    model_name: str
    prompt: str
    num_runs: int
    results: List[BenchmarkResult]
    average_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float
    total_tokens_generated: int
    total_time_seconds: float
    gpu_name: Optional[str] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


def _get_gpu_info() -> tuple:
    """Get GPU name and memory info."""
    gpu_name = None
    gpu_memory_used_gb = None
    gpu_memory_total_gb = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_gb = round(memory_info.used / (1024**3), 2)
        gpu_memory_total_gb = round(memory_info.total / (1024**3), 2)
        pynvml.nvmlShutdown()
        return gpu_name, gpu_memory_used_gb, gpu_memory_total_gb
    except Exception:
        # NVML isn't always available in minimal/prod deployments.
        # Fall back to torch.cuda so the UI still shows meaningful VRAM values.
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = gpu_name

                total_bytes = None
                free_bytes = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                except Exception:
                    try:
                        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
                    except Exception:
                        total_bytes = None

                if total_bytes is not None:
                    gpu_memory_total_gb = round(total_bytes / (1024**3), 2)

                if total_bytes is not None and free_bytes is not None:
                    used_bytes = max(0, int(total_bytes - free_bytes))
                    gpu_memory_used_gb = round(used_bytes / (1024**3), 2)
                else:
                    # Process-only fallback (still better than 0)
                    try:
                        gpu_memory_used_gb = round(float(torch.cuda.memory_reserved(0)) / (1024**3), 2)
                    except Exception:
                        gpu_memory_used_gb = gpu_memory_used_gb
        except Exception:
            pass
    return gpu_name, gpu_memory_used_gb, gpu_memory_total_gb


@admin_router.post("/benchmark/gpu")
async def run_gpu_benchmark(
    request: BenchmarkRequest = BenchmarkRequest(),
    admin: User = Depends(require_admin),
):
    """
    Run GPU benchmarking to measure tokens per second.
    
    This endpoint runs inference on the loaded model multiple times
    and calculates token generation speed metrics.
    If no model is loaded, returns GPU info with zero performance metrics.
    """
    import time
    
    # Get GPU info first
    gpu_name, gpu_memory_used_gb, gpu_memory_total_gb = _get_gpu_info()
    
    # Get the loaded LLM model (must be loaded in *this* process)
    model_instance = None
    model_name = "No model loaded"

    try:
        from ...core.model_manager import ModelManager

        def _looks_like_llm(loaded_name: str, instance: object) -> bool:
            lowered = (loaded_name or "").lower()
            config = getattr(instance, "config", None)
            engine_type = (getattr(config, "engine_type", None) or "").lower()
            model_id = (getattr(config, "model_id", None) or "").lower()

            if engine_type in {"llamacpp", "transformers", "vllm"}:
                return True

            # Filter out known non-LLM workloads
            if "whisper" in lowered or "whisper" in model_id:
                return False
            if "tts" in lowered or "tts" in model_id:
                return False

            # Fallback: if it has a usable benchmark() method, treat as candidate
            return callable(getattr(instance, "benchmark", None))

        model_manager = ModelManager()
        loaded_models = model_manager.list_loaded_models()

        # Prefer an LLM-looking model; otherwise fall back to any loaded model with benchmark()
        for candidate_name in loaded_models:
            candidate_instance = model_manager.get_loaded_model(candidate_name)
            if candidate_instance and _looks_like_llm(candidate_name, candidate_instance):
                model_instance = candidate_instance
                model_name = candidate_name
                break

        if model_instance is None:
            for candidate_name in loaded_models:
                candidate_instance = model_manager.get_loaded_model(candidate_name)
                if candidate_instance and callable(getattr(candidate_instance, "benchmark", None)):
                    model_instance = candidate_instance
                    model_name = candidate_name
                    break

        if model_instance is None:
            logger.warning(
                "No suitable LLM model found for benchmark (loaded_models=%s)",
                loaded_models,
            )

    except Exception as e:
        logger.warning(f"Error accessing model for benchmark: {e}")

    # If no model is loaded/available, return empty results with GPU info
    if model_instance is None:
        return {
            "tokens_per_second": 0,
            "total_tokens": 0,
            "inference_time_seconds": 0,
            "model_name": model_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "runs": 0,
            "gpu_name": gpu_name or "Unknown",
            "gpu_memory_used_gb": gpu_memory_used_gb or 0,
            "gpu_memory_total_gb": gpu_memory_total_gb or 0,
            "detailed_results": [],
            "average_tokens_per_second": 0,
            "min_tokens_per_second": 0,
            "max_tokens_per_second": 0,
            "error": "No suitable LLM model is currently loaded in this process. Load an LLM model to run benchmarks.",
        }
    
    # Run benchmarks
    results = []
    
    for run_num in range(1, request.num_runs + 1):
        try:
            # Prefer the model's built-in benchmark() implementation for accurate timing.
            # Most LLM engines accept max_new_tokens.
            bench = model_instance.benchmark(
                request.prompt,
                max_new_tokens=request.max_tokens,
            )

            elapsed = float(bench.get("inference_time") or 0)
            tokens_generated = int(bench.get("output_tokens") or 0)
            tokens_per_second = float(bench.get("tokens_per_second") or 0)

            # Fallback safety if an engine returns partial stats
            if elapsed <= 0 and tokens_per_second > 0 and tokens_generated > 0:
                elapsed = tokens_generated / tokens_per_second
            
            results.append(BenchmarkResult(
                run_number=run_num,
                tokens_generated=tokens_generated,
                time_seconds=round(elapsed, 3),
                tokens_per_second=round(tokens_per_second, 2)
            ))
            
        except Exception as e:
            logger.error(f"Benchmark run {run_num} failed: {e}")
            # Add a failed result
            results.append(BenchmarkResult(
                run_number=run_num,
                tokens_generated=0,
                time_seconds=0,
                tokens_per_second=0
            ))
    
    # Calculate aggregates
    valid_results = [r for r in results if r.tokens_per_second > 0]
    
    if valid_results:
        avg_tps = sum(r.tokens_per_second for r in valid_results) / len(valid_results)
        min_tps = min(r.tokens_per_second for r in valid_results)
        max_tps = max(r.tokens_per_second for r in valid_results)
    else:
        avg_tps = min_tps = max_tps = 0
    
    total_tokens = sum(r.tokens_generated for r in results)
    total_time = sum(r.time_seconds for r in results)
    
    # Return format that matches frontend expectations
    return {
        "tokens_per_second": round(avg_tps, 2),
        "total_tokens": total_tokens,
        "inference_time_seconds": round(total_time, 3),
        "model_name": model_name,
        "prompt_tokens": int(len(request.prompt.split()) * 2),  # Approximate
        "completion_tokens": total_tokens,
        "runs": request.num_runs,
        "gpu_name": gpu_name or "Unknown",
        "gpu_memory_used_gb": gpu_memory_used_gb or 0,
        "gpu_memory_total_gb": gpu_memory_total_gb or 0,
        # Also include detailed results for advanced use
        "detailed_results": [r.model_dump() for r in results],
        "average_tokens_per_second": round(avg_tps, 2),
        "min_tokens_per_second": round(min_tps, 2),
        "max_tokens_per_second": round(max_tps, 2),
    }


# ============================================
# Model Management Endpoints
# ============================================

@admin_router.post("/unload-all-models")
async def unload_all_models(
    admin: User = Depends(require_admin),
):
    """
    Unload all persistent models to free GPU VRAM.
    
    This is useful before loading large models like PersonaPlex that require
    most of the GPU memory. Unloads: Whisper, LLM, TTS models.
    
    Returns memory stats before and after cleanup.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager, cleanup_persistent_models
    from ...utils.memory_utils import get_gpu_memory_stats, clear_gpu_memory
    import gc
    
    # Get memory before cleanup
    gpu_before = get_gpu_memory_stats()
    memory_before = gpu_before[0] if gpu_before else {}
    
    # Get current model status
    manager = get_persistent_model_manager()
    models_status_before = manager.check_models_ready()
    
    logger.info(f"🧹 Admin {admin.email} requested unload of all models")
    logger.info(f"   Models before: {list(manager._preloaded_models.keys())}")
    
    # Perform cleanup
    success = await cleanup_persistent_models()
    
    # Force additional cleanup
    gc.collect()
    clear_gpu_memory()
    
    # Get memory after cleanup
    gpu_after = get_gpu_memory_stats()
    memory_after = gpu_after[0] if gpu_after else {}
    
    # Calculate freed memory
    freed_mb = (memory_before.get('memory_used_mb', 0) - 
                memory_after.get('memory_used_mb', 0))
    
    logger.info(f"✅ Model cleanup complete. Freed {freed_mb:.0f}MB VRAM")
    
    return {
        "success": success,
        "message": f"All models unloaded. Freed {freed_mb:.0f}MB GPU memory.",
        "models_unloaded": list(models_status_before.keys()),
        "memory_before": {
            "used_mb": memory_before.get('memory_used_mb', 0),
            "free_mb": memory_before.get('memory_free_mb', 0),
            "total_mb": memory_before.get('memory_total_mb', 0),
        },
        "memory_after": {
            "used_mb": memory_after.get('memory_used_mb', 0),
            "free_mb": memory_after.get('memory_free_mb', 0),
            "total_mb": memory_after.get('memory_total_mb', 0),
        },
        "freed_mb": freed_mb,
        "admin": admin.email,
    }


@admin_router.get("/models/status")
async def get_models_status(
    admin: User = Depends(require_admin),
):
    """
    Get status of all loaded models and GPU memory usage.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager
    from ...utils.memory_utils import get_gpu_memory_stats
    
    manager = get_persistent_model_manager()
    
    # Get model status
    models_ready = manager.check_models_ready()
    init_stats = manager.get_initialization_stats()
    
    # Get GPU memory
    gpu_stats = get_gpu_memory_stats()
    gpu_info = gpu_stats[0] if gpu_stats else {}
    
    return {
        "initialized": manager.is_initialized(),
        "models": models_ready,
        "preloaded_models": init_stats.get("preloaded_models", []),
        "startup_time_seconds": init_stats.get("startup_time_seconds"),
        "llm_pool": init_stats.get("llm_pool", {}),
        "gpu_memory": {
            "used_mb": gpu_info.get("memory_used_mb", 0),
            "free_mb": gpu_info.get("memory_free_mb", 0),
            "total_mb": gpu_info.get("memory_total_mb", 0),
            "utilization_percent": gpu_info.get("gpu_utilization", 0),
        }
    }


@admin_router.post("/models/unload/{model_id}")
async def unload_specific_model(
    model_id: str,
    admin: User = Depends(require_admin),
):
    """
    Unload a specific model by ID to free GPU VRAM.
    
    Model IDs are typically: stt, llm, tts, tts_fallback, etc.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager
    from ...utils.memory_utils import get_gpu_memory_stats, clear_gpu_memory
    import gc
    
    # Get memory before
    gpu_before = get_gpu_memory_stats()
    memory_before = gpu_before[0] if gpu_before else {}
    
    manager = get_persistent_model_manager()
    
    # Check if model exists
    if not manager.is_model_loaded(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' is not currently loaded"
        )
    
    logger.info(f"🧹 Admin {admin.email} requested unload of model: {model_id}")
    
    # Unload the specific model
    try:
        success = await manager.unload_model(model_id)
    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )
    
    # Force cleanup
    gc.collect()
    clear_gpu_memory()
    
    # Get memory after
    gpu_after = get_gpu_memory_stats()
    memory_after = gpu_after[0] if gpu_after else {}
    
    freed_mb = (memory_before.get('memory_used_mb', 0) - 
                memory_after.get('memory_used_mb', 0))
    
    logger.info(f"✅ Model {model_id} unloaded. Freed {freed_mb:.0f}MB VRAM")
    
    return {
        "success": success,
        "message": f"Model '{model_id}' unloaded. Freed {freed_mb:.0f}MB GPU memory.",
        "model_id": model_id,
        "memory_before": {
            "used_mb": memory_before.get('memory_used_mb', 0),
            "free_mb": memory_before.get('memory_free_mb', 0),
            "total_mb": memory_before.get('memory_total_mb', 0),
        },
        "memory_after": {
            "used_mb": memory_after.get('memory_used_mb', 0),
            "free_mb": memory_after.get('memory_free_mb', 0),
            "total_mb": memory_after.get('memory_total_mb', 0),
        },
        "freed_mb": freed_mb,
    }


@admin_router.get("/benchmark/gpu/quick")
async def quick_gpu_benchmark(
    admin: User = Depends(require_admin),
):
    """
    Quick GPU benchmark that returns current tokens/sec performance.
    
    Uses a single run for fast results. Use POST /benchmark/gpu for detailed benchmarks.
    If no model is loaded, returns GPU info with null performance metrics.
    """
    gpu_name, gpu_memory_used_gb, gpu_memory_total_gb = _get_gpu_info()
    
    # Try to run actual benchmark
    try:
        request = BenchmarkRequest(
            prompt="Hello, how are you?",
            max_tokens=50,
            num_runs=1
        )
        
        result = await run_gpu_benchmark(request, admin)
        
        # run_gpu_benchmark returns a dict now, so use dict access
        return {
            "tokens_per_second": result.get("tokens_per_second", 0),
            "total_tokens": result.get("total_tokens", 0),
            "inference_time_seconds": result.get("inference_time_seconds", 0),
            "model_name": result.get("model_name", "Unknown"),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "runs": result.get("runs", 0),
            "gpu_name": result.get("gpu_name") or gpu_name or "Unknown",
            "gpu_memory_used_gb": result.get("gpu_memory_used_gb") or gpu_memory_used_gb or 0,
            "gpu_memory_total_gb": result.get("gpu_memory_total_gb") or gpu_memory_total_gb or 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": result.get("error"),  # Pass through any error message
        }
    except Exception as e:
        # Any error - return GPU info only
        logger.warning(f"Quick benchmark failed: {e}")
        return {
            "tokens_per_second": 0,
            "total_tokens": 0,
            "inference_time_seconds": 0,
            "model_name": "No model loaded",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "runs": 0,
            "gpu_name": gpu_name or "Unknown",
            "gpu_memory_used_gb": gpu_memory_used_gb or 0,
            "gpu_memory_total_gb": gpu_memory_total_gb or 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e) if str(e) else "Benchmark failed",
        }

