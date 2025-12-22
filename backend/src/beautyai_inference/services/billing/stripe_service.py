"""
Stripe service for subscription management and billing.

Handles:
- Customer creation in Stripe
- Subscription management (create, update, cancel)
- Payment method setup
- Invoice management
"""

import os
import logging
from typing import Optional, List
from datetime import datetime, timezone

import stripe
from stripe import StripeError

logger = logging.getLogger(__name__)


class StripeService:
    """
    Stripe integration service for BeautyAI SaaS billing.
    
    All prices are in SAR (Saudi Riyal).
    """
    
    def __init__(self):
        """Initialize Stripe with API keys from environment."""
        self.api_key = os.getenv("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
        self.publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
        
        # Currency
        self.currency = os.getenv("STRIPE_CURRENCY", "sar").lower()
        
        # URLs for checkout
        self.success_url = os.getenv("STRIPE_SUCCESS_URL", "https://gmai.sa/app/billing?success=true")
        self.cancel_url = os.getenv("STRIPE_CANCEL_URL", "https://gmai.sa/app/billing?canceled=true")
        
        # Development mode
        self.dev_mode = os.getenv("STRIPE_DEV_MODE", "false").lower() == "true"
        
        if self.api_key:
            stripe.api_key = self.api_key
        else:
            logger.warning("Stripe API key not configured")
    
    def _check_configured(self) -> None:
        """Raise error if Stripe is not configured."""
        if not self.api_key:
            raise ValueError("Stripe API key not configured")
    
    # ========================================================================
    # Customer Management
    # ========================================================================
    
    async def create_customer(
        self,
        email: str,
        name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new Stripe customer.
        
        Returns: Stripe customer ID
        """
        self._check_configured()
        
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata or {},
            )
            logger.info(f"Stripe customer created: {customer.id} for {email}")
            return customer.id
        except StripeError as e:
            logger.error(f"Stripe customer creation failed: {e}")
            raise
    
    async def get_customer(self, customer_id: str) -> Optional[dict]:
        """Get Stripe customer details."""
        self._check_configured()
        
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return dict(customer)
        except StripeError as e:
            logger.error(f"Stripe customer retrieval failed: {e}")
            return None
    
    async def update_customer(
        self,
        customer_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Update Stripe customer."""
        self._check_configured()
        
        update_data = {}
        if email:
            update_data["email"] = email
        if name:
            update_data["name"] = name
        if metadata:
            update_data["metadata"] = metadata
        
        try:
            customer = stripe.Customer.modify(customer_id, **update_data)
            return dict(customer)
        except StripeError as e:
            logger.error(f"Stripe customer update failed: {e}")
            raise
    
    # ========================================================================
    # Subscription Management
    # ========================================================================
    
    async def create_checkout_session(
        self,
        stripe_customer_id: str,
        price_id: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a Stripe Checkout Session for subscription signup.
        
        Returns: Checkout session URL
        """
        self._check_configured()
        
        try:
            session = stripe.checkout.Session.create(
                customer=stripe_customer_id,
                payment_method_types=["card"],
                line_items=[
                    {
                        "price": price_id,
                        "quantity": 1,
                    }
                ],
                mode="subscription",
                success_url=self.success_url,
                cancel_url=self.cancel_url,
                metadata=metadata or {},
                allow_promotion_codes=True,
            )
            logger.info(f"Checkout session created: {session.id}")
            return session.url
        except StripeError as e:
            logger.error(f"Checkout session creation failed: {e}")
            raise
    
    async def create_subscription(
        self,
        stripe_customer_id: str,
        price_id: str,
        trial_days: int = 0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Create a subscription directly (for API-based signup).
        
        Returns: Subscription data
        """
        self._check_configured()
        
        try:
            subscription_data = {
                "customer": stripe_customer_id,
                "items": [{"price": price_id}],
                "metadata": metadata or {},
            }
            
            if trial_days > 0:
                subscription_data["trial_period_days"] = trial_days
            
            subscription = stripe.Subscription.create(**subscription_data)
            logger.info(f"Subscription created: {subscription.id}")
            return dict(subscription)
        except StripeError as e:
            logger.error(f"Subscription creation failed: {e}")
            raise
    
    async def get_subscription(self, subscription_id: str) -> Optional[dict]:
        """Get subscription details."""
        self._check_configured()
        
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return dict(subscription)
        except StripeError as e:
            logger.error(f"Subscription retrieval failed: {e}")
            return None
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> dict:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            at_period_end: If True, cancel at end of billing period. If False, cancel immediately.
        """
        self._check_configured()
        
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
            
            logger.info(f"Subscription canceled: {subscription_id}, at_period_end={at_period_end}")
            return dict(subscription)
        except StripeError as e:
            logger.error(f"Subscription cancellation failed: {e}")
            raise
    
    async def update_subscription(
        self,
        subscription_id: str,
        price_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Update subscription (change plan, etc.)."""
        self._check_configured()
        
        update_data = {}
        
        if price_id:
            # Get current subscription to find item ID
            sub = stripe.Subscription.retrieve(subscription_id)
            item_id = sub["items"]["data"][0]["id"]
            update_data["items"] = [{"id": item_id, "price": price_id}]
            update_data["proration_behavior"] = "create_prorations"
        
        if metadata:
            update_data["metadata"] = metadata
        
        try:
            subscription = stripe.Subscription.modify(subscription_id, **update_data)
            logger.info(f"Subscription updated: {subscription_id}")
            return dict(subscription)
        except StripeError as e:
            logger.error(f"Subscription update failed: {e}")
            raise
    
    async def reactivate_subscription(self, subscription_id: str) -> dict:
        """Reactivate a subscription that was scheduled to cancel."""
        self._check_configured()
        
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=False,
            )
            logger.info(f"Subscription reactivated: {subscription_id}")
            return dict(subscription)
        except StripeError as e:
            logger.error(f"Subscription reactivation failed: {e}")
            raise
    
    # ========================================================================
    # Billing Portal
    # ========================================================================
    
    async def create_portal_session(
        self,
        stripe_customer_id: str,
        return_url: Optional[str] = None,
    ) -> str:
        """
        Create a Stripe Billing Portal session.
        
        Allows customers to manage their subscription, payment methods, and invoices.
        
        Returns: Portal session URL
        """
        self._check_configured()
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=stripe_customer_id,
                return_url=return_url or self.success_url,
            )
            return session.url
        except StripeError as e:
            logger.error(f"Portal session creation failed: {e}")
            raise
    
    # ========================================================================
    # Invoices
    # ========================================================================
    
    async def list_invoices(
        self,
        stripe_customer_id: str,
        limit: int = 10,
    ) -> List[dict]:
        """List customer invoices."""
        self._check_configured()
        
        try:
            invoices = stripe.Invoice.list(
                customer=stripe_customer_id,
                limit=limit,
            )
            return [dict(inv) for inv in invoices.data]
        except StripeError as e:
            logger.error(f"Invoice listing failed: {e}")
            return []
    
    async def get_upcoming_invoice(
        self,
        stripe_customer_id: str,
    ) -> Optional[dict]:
        """Get upcoming invoice for customer."""
        self._check_configured()
        
        try:
            invoice = stripe.Invoice.upcoming(customer=stripe_customer_id)
            return dict(invoice)
        except StripeError as e:
            # No upcoming invoice is not an error
            if "No upcoming invoices" in str(e):
                return None
            logger.error(f"Upcoming invoice retrieval failed: {e}")
            return None
    
    # ========================================================================
    # Webhook Signature Verification
    # ========================================================================
    
    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> dict:
        """
        Verify Stripe webhook signature and parse event.
        
        Raises:
            ValueError: If signature is invalid
        """
        if not self.webhook_secret:
            raise ValueError("Webhook secret not configured")
        
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.webhook_secret,
            )
            return dict(event)
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            raise ValueError("Invalid webhook signature")
    
    # ========================================================================
    # Usage Reporting (for metered billing)
    # ========================================================================
    
    async def report_usage(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
    ) -> dict:
        """
        Report usage for metered billing.
        
        For future usage-based pricing.
        """
        self._check_configured()
        
        try:
            usage_record = stripe.SubscriptionItem.create_usage_record(
                subscription_item_id,
                quantity=quantity,
                timestamp=timestamp or int(datetime.now(timezone.utc).timestamp()),
                action="increment",
            )
            return dict(usage_record)
        except StripeError as e:
            logger.error(f"Usage reporting failed: {e}")
            raise


# Singleton instance
_stripe_service: Optional[StripeService] = None


def get_stripe_service() -> StripeService:
    """Get or create singleton Stripe service instance."""
    global _stripe_service
    if _stripe_service is None:
        _stripe_service = StripeService()
    return _stripe_service
