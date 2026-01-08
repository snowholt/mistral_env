#!/usr/bin/env python3
"""
Create test accounts for BeautyAI SaaS Platform.

Creates:
1. Admin account: nariman@gmai.sa (password: Admin@123456)
2. Customer account: customer@test.com (password: Customer@123456)

Usage:
    python scripts/create_test_accounts.py
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from beautyai_inference.database.models import User, UserRole, Customer, Plan, Subscription, SubscriptionStatus
from beautyai_inference.auth.password import hash_password


# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai"
)


async def create_test_accounts():
    """Create test admin and customer accounts."""
    
    print("🚀 Creating test accounts...")
    print(f"📡 Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    
    # Create async engine and session
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        async with session.begin():
            # ===========================================
            # 1. Create Admin Account: nariman@gmai.sa
            # ===========================================
            admin_email = "nariman@gmai.sa"
            admin_password = "Admin@123456"
            
            # Check if admin already exists
            result = await session.execute(
                select(User).where(User.email == admin_email)
            )
            existing_admin = result.scalar_one_or_none()
            
            if existing_admin:
                print(f"⚠️  Admin account already exists: {admin_email}")
                # Update password and role
                await session.execute(
                    User.__table__.update()
                    .where(User.email == admin_email)
                    .values(
                        password_hash=hash_password(admin_password),
                        is_active=True,
                        is_verified=True,
                        role='admin'  # Use lowercase string for PostgreSQL enum
                    )
                )
                admin_user = existing_admin
            else:
                # Insert directly with raw SQL to handle enum properly
                from sqlalchemy import text
                result = await session.execute(
                    text("""
                        INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                        VALUES (:email, :password_hash, :full_name, 'admin', :is_active, :is_verified)
                        RETURNING id
                    """),
                    {
                        "email": admin_email,
                        "password_hash": hash_password(admin_password),
                        "full_name": "Nariman Admin",
                        "is_active": True,
                        "is_verified": True,
                    }
                )
                admin_id = result.scalar()
                print(f"✅ Created admin account: {admin_email} (ID: {admin_id})")
                
                # Fetch the user for later use
                result = await session.execute(
                    select(User).where(User.email == admin_email)
                )
                admin_user = result.scalar_one_or_none()
            
            # ===========================================
            # 2. Create Customer Account: customer@test.com
            # ===========================================
            customer_email = "customer@test.com"
            customer_password = "Customer@123456"
            
            # Check if customer already exists
            result = await session.execute(
                select(User).where(User.email == customer_email)
            )
            existing_customer = result.scalar_one_or_none()
            
            if existing_customer:
                print(f"⚠️  Customer account already exists: {customer_email}")
                await session.execute(
                    User.__table__.update()
                    .where(User.email == customer_email)
                    .values(
                        password_hash=hash_password(customer_password),
                        is_active=True,
                        is_verified=True,
                    )
                )
                customer_user = existing_customer
            else:
                # Insert customer with raw SQL
                from sqlalchemy import text
                result = await session.execute(
                    text("""
                        INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                        VALUES (:email, :password_hash, :full_name, 'user', :is_active, :is_verified)
                        RETURNING id
                    """),
                    {
                        "email": customer_email,
                        "password_hash": hash_password(customer_password),
                        "full_name": "Test Customer",
                        "is_active": True,
                        "is_verified": True,
                    }
                )
                customer_id = result.scalar()
                print(f"✅ Created customer account: {customer_email} (ID: {customer_id})")
                
                # Fetch the user for later use
                result = await session.execute(
                    select(User).where(User.email == customer_email)
                )
                customer_user = result.scalar_one()
            
            # ===========================================
            # 3. Create a Business (Customer entity) for the customer user
            # ===========================================
            result = await session.execute(
                select(Customer).where(Customer.user_id == customer_user.id)
            )
            existing_business = result.scalar_one_or_none()
            
            if existing_business:
                print(f"⚠️  Business already exists for customer: {existing_business.business_name}")
                customer_business = existing_business
            else:
                customer_business = Customer(
                    user_id=customer_user.id,
                    business_name="Test Business Co.",
                    business_phone="+966500000000",
                    business_email="business@test.com",
                    is_active=True,
                )
                session.add(customer_business)
                print(f"✅ Created business: Test Business Co.")
            
            await session.flush()
            
            # ===========================================
            # 4. Create a Free Plan if not exists
            # ===========================================
            result = await session.execute(
                select(Plan).where(Plan.name == "Free Trial")
            )
            free_plan = result.scalar_one_or_none()
            
            if not free_plan:
                free_plan = Plan(
                    name="Free Trial",
                    stripe_price_id="price_free_trial",
                    price_monthly=0,
                    price_yearly=0,
                    max_whatsapp_accounts=1,
                    max_monthly_messages=1000,
                    max_knowledge_base_mb=50,
                    features={"whatsapp": True, "webchat": True, "analytics": False},
                    is_active=True,
                )
                session.add(free_plan)
                print(f"✅ Created Free Trial plan")
                await session.flush()
            
            # ===========================================
            # 5. Create Subscription for the customer
            # ===========================================
            result = await session.execute(
                select(Subscription).where(Subscription.customer_id == customer_business.id)
            )
            existing_subscription = result.scalar_one_or_none()
            
            if not existing_subscription:
                from datetime import datetime, timezone, timedelta
                subscription = Subscription(
                    customer_id=customer_business.id,
                    plan_id=free_plan.id,
                    status=SubscriptionStatus.TRIAL,
                    current_period_start=datetime.now(timezone.utc),
                    current_period_end=datetime.now(timezone.utc) + timedelta(days=14),
                    trial_ends_at=datetime.now(timezone.utc) + timedelta(days=14),
                )
                session.add(subscription)
                print(f"✅ Created trial subscription for customer")
            else:
                print(f"⚠️  Subscription already exists for customer")
    
    await engine.dispose()
    
    print("\n" + "="*60)
    print("🎉 Test accounts ready!")
    print("="*60)
    print("\n📋 ADMIN ACCOUNT (for Admin Dashboard):")
    print(f"   Email:    nariman@gmai.sa")
    print(f"   Password: Admin@123456")
    print(f"   Role:     ADMIN")
    print(f"   URL:      https://portal.gmai.sa/login")
    print(f"   Dashboard: https://portal.gmai.sa/app/admin/customers")
    print("\n📋 CUSTOMER ACCOUNT (for Customer Dashboard):")
    print(f"   Email:    customer@test.com")
    print(f"   Password: Customer@123456")
    print(f"   Role:     USER")
    print(f"   URL:      https://portal.gmai.sa/login")
    print(f"   Dashboard: https://portal.gmai.sa/app")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(create_test_accounts())
