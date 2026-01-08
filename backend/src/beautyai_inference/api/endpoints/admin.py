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
