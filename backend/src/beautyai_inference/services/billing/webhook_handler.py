"""
Stripe webhook event handler.

Processes Stripe events for subscription lifecycle management.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ...database.models import Subscription, SubscriptionStatus, User, Customer, UsageEvent, UsageEventType

logger = logging.getLogger(__name__)


class StripeWebhookHandler:
    """
    Handles Stripe webhook events and updates database accordingly.
    
    Supported events:
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.paid
    - invoice.payment_failed
    - checkout.session.completed
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session."""
        self.db = db
    
    async def handle_event(self, event: dict) -> dict:
        """
        Main entry point for handling Stripe events.
        
        Returns: Result dict with status and message
        """
        event_type = event.get("type", "")
        event_id = event.get("id", "")
        data = event.get("data", {}).get("object", {})
        
        logger.info(f"Processing Stripe event: {event_type} ({event_id})")
        
        # Route to appropriate handler
        handlers = {
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_invoice_payment_failed,
            "checkout.session.completed": self._handle_checkout_completed,
        }
        
        handler = handlers.get(event_type)
        if handler:
            try:
                result = await handler(data)
                return {"status": "success", "event": event_type, "result": result}
            except Exception as e:
                logger.error(f"Error handling {event_type}: {e}")
                return {"status": "error", "event": event_type, "error": str(e)}
        else:
            logger.debug(f"Unhandled event type: {event_type}")
            return {"status": "ignored", "event": event_type}
    
    # ========================================================================
    # Subscription Events
    # ========================================================================
    
    async def _handle_subscription_created(self, data: dict) -> dict:
        """Handle new subscription creation."""
        stripe_subscription_id = data.get("id")
        stripe_customer_id = data.get("customer")
        status = self._map_stripe_status(data.get("status"))
        
        # Get price/plan info
        items = data.get("items", {}).get("data", [])
        price_id = items[0]["price"]["id"] if items else None
        
        # Find user by Stripe customer ID
        user = await self._get_user_by_stripe_customer(stripe_customer_id)
        if not user:
            logger.warning(f"No user found for Stripe customer: {stripe_customer_id}")
            return {"user_found": False}
        
        # Check if subscription already exists
        existing = await self.db.execute(
            select(Subscription).where(Subscription.stripe_subscription_id == stripe_subscription_id)
        )
        if existing.scalar_one_or_none():
            logger.info(f"Subscription already exists: {stripe_subscription_id}")
            return {"action": "already_exists"}
        
        # Create subscription record
        subscription = Subscription(
            user_id=user.id,
            stripe_subscription_id=stripe_subscription_id,
            stripe_price_id=price_id,
            status=status,
            current_period_start=self._timestamp_to_datetime(data.get("current_period_start")),
            current_period_end=self._timestamp_to_datetime(data.get("current_period_end")),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
        )
        self.db.add(subscription)
        await self.db.commit()
        
        logger.info(f"Subscription created in DB: {stripe_subscription_id} for user {user.id}")
        return {"action": "created", "subscription_id": subscription.id}
    
    async def _handle_subscription_updated(self, data: dict) -> dict:
        """Handle subscription updates (plan changes, status changes)."""
        stripe_subscription_id = data.get("id")
        status = self._map_stripe_status(data.get("status"))
        
        # Get price/plan info
        items = data.get("items", {}).get("data", [])
        price_id = items[0]["price"]["id"] if items else None
        
        # Update subscription
        result = await self.db.execute(
            update(Subscription)
            .where(Subscription.stripe_subscription_id == stripe_subscription_id)
            .values(
                status=status,
                stripe_price_id=price_id,
                current_period_start=self._timestamp_to_datetime(data.get("current_period_start")),
                current_period_end=self._timestamp_to_datetime(data.get("current_period_end")),
                cancel_at_period_end=data.get("cancel_at_period_end", False),
                canceled_at=self._timestamp_to_datetime(data.get("canceled_at")),
            )
            .returning(Subscription.id)
        )
        await self.db.commit()
        
        row = result.first()
        if row:
            logger.info(f"Subscription updated: {stripe_subscription_id}")
            return {"action": "updated", "subscription_id": row.id}
        else:
            logger.warning(f"Subscription not found for update: {stripe_subscription_id}")
            return {"action": "not_found"}
    
    async def _handle_subscription_deleted(self, data: dict) -> dict:
        """Handle subscription cancellation/deletion."""
        stripe_subscription_id = data.get("id")
        
        result = await self.db.execute(
            update(Subscription)
            .where(Subscription.stripe_subscription_id == stripe_subscription_id)
            .values(
                status=SubscriptionStatus.CANCELED,
                canceled_at=datetime.now(timezone.utc),
            )
            .returning(Subscription.id)
        )
        await self.db.commit()
        
        row = result.first()
        if row:
            logger.info(f"Subscription canceled: {stripe_subscription_id}")
            return {"action": "canceled", "subscription_id": row.id}
        else:
            logger.warning(f"Subscription not found for deletion: {stripe_subscription_id}")
            return {"action": "not_found"}
    
    # ========================================================================
    # Invoice Events
    # ========================================================================
    
    async def _handle_invoice_paid(self, data: dict) -> dict:
        """Handle successful invoice payment."""
        stripe_subscription_id = data.get("subscription")
        stripe_customer_id = data.get("customer")
        amount_paid = data.get("amount_paid", 0) / 100  # Convert from cents
        
        if not stripe_subscription_id:
            # One-time payment, not subscription
            return {"action": "one_time_payment"}
        
        # Update subscription status to active if it was past_due
        await self.db.execute(
            update(Subscription)
            .where(Subscription.stripe_subscription_id == stripe_subscription_id)
            .where(Subscription.status == SubscriptionStatus.PAST_DUE)
            .values(status=SubscriptionStatus.ACTIVE)
        )
        
        # Record payment event
        user = await self._get_user_by_stripe_customer(stripe_customer_id)
        if user:
            # Get customer
            customer_result = await self.db.execute(
                select(Customer).where(Customer.user_id == user.id)
            )
            customer = customer_result.scalar_one_or_none()
            
            if customer:
                event = UsageEvent(
                    customer_id=customer.id,
                    event_type=UsageEventType.OTHER,
                    event_data={
                        "type": "invoice_paid",
                        "amount": amount_paid,
                        "invoice_id": data.get("id"),
                    },
                )
                self.db.add(event)
        
        await self.db.commit()
        
        logger.info(f"Invoice paid for subscription: {stripe_subscription_id}, amount: {amount_paid}")
        return {"action": "payment_recorded", "amount": amount_paid}
    
    async def _handle_invoice_payment_failed(self, data: dict) -> dict:
        """Handle failed invoice payment."""
        stripe_subscription_id = data.get("subscription")
        
        if not stripe_subscription_id:
            return {"action": "ignored"}
        
        # Update subscription status to past_due
        await self.db.execute(
            update(Subscription)
            .where(Subscription.stripe_subscription_id == stripe_subscription_id)
            .values(status=SubscriptionStatus.PAST_DUE)
        )
        await self.db.commit()
        
        logger.warning(f"Invoice payment failed for subscription: {stripe_subscription_id}")
        return {"action": "marked_past_due"}
    
    # ========================================================================
    # Checkout Events
    # ========================================================================
    
    async def _handle_checkout_completed(self, data: dict) -> dict:
        """Handle completed checkout session."""
        mode = data.get("mode")
        stripe_customer_id = data.get("customer")
        stripe_subscription_id = data.get("subscription")
        
        if mode == "subscription" and stripe_subscription_id:
            # Subscription checkout - the subscription.created event will handle this
            logger.info(f"Checkout completed for subscription: {stripe_subscription_id}")
            return {"action": "subscription_checkout"}
        
        return {"action": "checkout_completed", "mode": mode}
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    async def _get_user_by_stripe_customer(self, stripe_customer_id: str) -> Optional[User]:
        """Find user by their Stripe customer ID."""
        result = await self.db.execute(
            select(User).where(User.stripe_customer_id == stripe_customer_id)
        )
        return result.scalar_one_or_none()
    
    def _map_stripe_status(self, stripe_status: str) -> SubscriptionStatus:
        """Map Stripe subscription status to our enum."""
        mapping = {
            "active": SubscriptionStatus.ACTIVE,
            "trialing": SubscriptionStatus.TRIALING,
            "past_due": SubscriptionStatus.PAST_DUE,
            "canceled": SubscriptionStatus.CANCELED,
            "unpaid": SubscriptionStatus.PAST_DUE,
            "incomplete": SubscriptionStatus.PAST_DUE,
            "incomplete_expired": SubscriptionStatus.CANCELED,
        }
        return mapping.get(stripe_status, SubscriptionStatus.ACTIVE)
    
    def _timestamp_to_datetime(self, timestamp: Optional[int]) -> Optional[datetime]:
        """Convert Unix timestamp to datetime."""
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)


async def handle_stripe_webhook(event: dict, db: AsyncSession) -> dict:
    """
    Convenience function to handle Stripe webhook event.
    
    Usage:
        result = await handle_stripe_webhook(event, db_session)
    """
    handler = StripeWebhookHandler(db)
    return await handler.handle_event(event)
