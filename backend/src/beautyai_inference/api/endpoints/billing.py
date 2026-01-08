"""
Billing API endpoints for subscription management.

Endpoints:
- GET /billing/plans - List available plans
- GET /billing/subscription - Get current subscription
- POST /billing/checkout - Create checkout session
- POST /billing/portal - Create billing portal session
- GET /billing/invoices - List invoices
- POST /billing/cancel - Cancel subscription
- POST /billing/reactivate - Reactivate subscription
- POST /billing/webhook - Stripe webhook handler
"""

import os
import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.connection import get_db
from ...database.models import User, Subscription, SubscriptionStatus, Plan
from ..endpoints.whatsapp_auth import get_current_user
from ...services.billing import StripeService, get_stripe_service, handle_stripe_webhook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PlanResponse(BaseModel):
    """Plan details."""
    id: int
    name: str
    display_name: str
    description: Optional[str]
    price_monthly: float
    price_yearly: Optional[float]
    stripe_price_id_monthly: Optional[str]
    stripe_price_id_yearly: Optional[str]
    features: dict
    limits: dict
    is_active: bool


class SubscriptionResponse(BaseModel):
    """Current subscription details."""
    id: int
    status: str
    plan_name: Optional[str]
    stripe_price_id: Optional[str]
    current_period_start: Optional[datetime]
    current_period_end: Optional[datetime]
    cancel_at_period_end: bool
    trial_end: Optional[datetime]
    canceled_at: Optional[datetime]


class CheckoutRequest(BaseModel):
    """Request to create checkout session."""
    price_id: str = Field(..., description="Stripe price ID")


class CheckoutResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str


class PortalResponse(BaseModel):
    """Billing portal session response."""
    portal_url: str


class InvoiceResponse(BaseModel):
    """Invoice details."""
    id: str
    status: str
    amount_due: float
    amount_paid: float
    currency: str
    created: datetime
    invoice_pdf: Optional[str]


class CancelRequest(BaseModel):
    """Subscription cancellation request."""
    immediately: bool = Field(default=False, description="Cancel immediately vs end of period")


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


# ============================================================================
# Plan Endpoints
# ============================================================================

@router.get("/plans", response_model=List[PlanResponse])
async def list_plans(
    db: AsyncSession = Depends(get_db),
):
    """
    List all available subscription plans.
    
    Public endpoint - no authentication required.
    """
    result = await db.execute(
        select(Plan)
        .where(Plan.is_active == True)
        .order_by(Plan.price_monthly)
    )
    plans = result.scalars().all()
    
    return [
        PlanResponse(
            id=p.id,
            name=p.name,
            display_name=p.display_name,
            description=p.description,
            price_monthly=float(p.price_monthly),
            price_yearly=float(p.price_yearly) if p.price_yearly else None,
            stripe_price_id_monthly=p.stripe_price_id_monthly,
            stripe_price_id_yearly=p.stripe_price_id_yearly,
            features=p.features or {},
            limits=p.limits or {},
            is_active=p.is_active,
        )
        for p in plans
    ]


# ============================================================================
# Subscription Endpoints
# ============================================================================

@router.get("/subscription", response_model=Optional[SubscriptionResponse])
async def get_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's subscription.
    
    Returns null if no subscription exists.
    """
    result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == current_user.id)
        .order_by(Subscription.created_at.desc())
        .limit(1)
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        return None
    
    # Get plan name if available
    plan_name = None
    if subscription.plan_id:
        plan_result = await db.execute(
            select(Plan).where(Plan.id == subscription.plan_id)
        )
        plan = plan_result.scalar_one_or_none()
        if plan:
            plan_name = plan.display_name
    
    return SubscriptionResponse(
        id=subscription.id,
        status=subscription.status.value,
        plan_name=plan_name,
        stripe_price_id=subscription.stripe_price_id,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        cancel_at_period_end=subscription.cancel_at_period_end,
        trial_end=subscription.trial_end,
        canceled_at=subscription.canceled_at,
    )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """
    Create a Stripe Checkout session for subscription signup.
    
    Returns a URL to redirect the user to.
    """
    # Ensure user has Stripe customer ID
    if not current_user.stripe_customer_id:
        # Create Stripe customer
        stripe_customer_id = await stripe_service.create_customer(
            email=current_user.email,
            name=current_user.full_name or current_user.email,
            metadata={"user_id": str(current_user.id)},
        )
        # Update user (this should be done in a service, but for brevity)
        current_user.stripe_customer_id = stripe_customer_id
    
    try:
        checkout_url = await stripe_service.create_checkout_session(
            stripe_customer_id=current_user.stripe_customer_id,
            price_id=request.price_id,
            metadata={"user_id": str(current_user.id)},
        )
        return CheckoutResponse(checkout_url=checkout_url)
    except Exception as e:
        logger.error(f"Checkout session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/portal", response_model=PortalResponse)
async def create_portal_session(
    current_user: User = Depends(get_current_user),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """
    Create a Stripe Billing Portal session.
    
    Allows users to manage payment methods, view invoices, etc.
    """
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")
    
    try:
        portal_url = await stripe_service.create_portal_session(
            stripe_customer_id=current_user.stripe_customer_id,
        )
        return PortalResponse(portal_url=portal_url)
    except Exception as e:
        logger.error(f"Portal session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create portal session")


@router.get("/invoices", response_model=List[InvoiceResponse])
async def list_invoices(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """List user's invoices from Stripe."""
    if not current_user.stripe_customer_id:
        return []
    
    try:
        invoices = await stripe_service.list_invoices(
            stripe_customer_id=current_user.stripe_customer_id,
            limit=limit,
        )
        
        return [
            InvoiceResponse(
                id=inv["id"],
                status=inv.get("status", "unknown"),
                amount_due=inv.get("amount_due", 0) / 100,
                amount_paid=inv.get("amount_paid", 0) / 100,
                currency=inv.get("currency", "sar").upper(),
                created=datetime.fromtimestamp(inv.get("created", 0), tz=timezone.utc),
                invoice_pdf=inv.get("invoice_pdf"),
            )
            for inv in invoices
        ]
    except Exception as e:
        logger.error(f"Invoice listing failed: {e}")
        return []


@router.post("/cancel", response_model=MessageResponse)
async def cancel_subscription(
    request: CancelRequest,
    current_user: User = Depends(get_current_user),
    stripe_service: StripeService = Depends(get_stripe_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel current subscription.
    
    By default, cancels at end of billing period.
    Set immediately=true to cancel right away.
    """
    # Get active subscription
    result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == current_user.id)
        .where(Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]))
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription or not subscription.stripe_subscription_id:
        raise HTTPException(status_code=404, detail="No active subscription found")
    
    try:
        await stripe_service.cancel_subscription(
            subscription_id=subscription.stripe_subscription_id,
            at_period_end=not request.immediately,
        )
        
        if request.immediately:
            return MessageResponse(message="Subscription canceled immediately")
        else:
            return MessageResponse(message="Subscription will cancel at end of billing period")
    except Exception as e:
        logger.error(f"Subscription cancellation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")


@router.post("/reactivate", response_model=MessageResponse)
async def reactivate_subscription(
    current_user: User = Depends(get_current_user),
    stripe_service: StripeService = Depends(get_stripe_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Reactivate a subscription that was scheduled to cancel.
    
    Only works if subscription hasn't actually ended yet.
    """
    # Get subscription scheduled to cancel
    result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == current_user.id)
        .where(Subscription.status == SubscriptionStatus.ACTIVE)
        .where(Subscription.cancel_at_period_end == True)
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription or not subscription.stripe_subscription_id:
        raise HTTPException(status_code=404, detail="No subscription to reactivate")
    
    try:
        await stripe_service.reactivate_subscription(
            subscription_id=subscription.stripe_subscription_id,
        )
        return MessageResponse(message="Subscription reactivated successfully")
    except Exception as e:
        logger.error(f"Subscription reactivation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to reactivate subscription")


# ============================================================================
# Webhook Endpoint
# ============================================================================

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    stripe_service: StripeService = Depends(get_stripe_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Stripe webhook endpoint.
    
    This endpoint receives events from Stripe about subscription changes,
    payment events, etc.
    """
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
    
    # Get raw body
    body = await request.body()
    
    try:
        # Verify signature and parse event
        event = stripe_service.verify_webhook_signature(body, stripe_signature)
    except ValueError as e:
        logger.warning(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Process event
    result = await handle_stripe_webhook(event, db)
    
    return {"received": True, **result}
