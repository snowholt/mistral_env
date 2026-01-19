#!/usr/bin/env python3
"""
Create test accounts for BeautyAI SaaS Platform using raw SQL.

Creates:
1. Admin account: nariman@gmai.sa (password: Admin@123456)
2. Customer account: customer@test.com (password: Customer@123456)
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from beautyai_inference.auth.password import hash_password

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai"
)


async def create_test_accounts():
    """Create test accounts using raw SQL."""
    
    print("🚀 Creating test accounts...")
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    # Hash passwords
    admin_password_hash = hash_password("Admin@123456")
    customer_password_hash = hash_password("Customer@123456")
    
    async with engine.begin() as conn:
        # 1. Create Admin Account
        admin_email = "nariman@gmai.sa"
        
        # Check if exists
        result = await conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": admin_email}
        )
        existing = result.scalar()
        
        if existing:
            print(f"⚠️  Admin account exists (ID: {existing}), updating...")
            await conn.execute(
                text("""
                    UPDATE users 
                    SET password_hash = :password_hash, 
                        role = 'admin', 
                        is_active = true, 
                        is_verified = true
                    WHERE email = :email
                """),
                {"email": admin_email, "password_hash": admin_password_hash}
            )
        else:
            result = await conn.execute(
                text("""
                    INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                    VALUES (:email, :password_hash, :full_name, 'admin', true, true)
                    RETURNING id
                """),
                {
                    "email": admin_email,
                    "password_hash": admin_password_hash,
                    "full_name": "Nariman Admin"
                }
            )
            admin_id = result.scalar()
            print(f"✅ Created admin account: {admin_email} (ID: {admin_id})")
        
        # 2. Create Customer Account
        customer_email = "customer@test.com"
        
        result = await conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": customer_email}
        )
        existing = result.scalar()
        
        if existing:
            print(f"⚠️  Customer account exists (ID: {existing}), updating...")
            await conn.execute(
                text("""
                    UPDATE users 
                    SET password_hash = :password_hash,
                        is_active = true, 
                        is_verified = true
                    WHERE email = :email
                """),
                {"email": customer_email, "password_hash": customer_password_hash}
            )
            customer_id = existing
        else:
            result = await conn.execute(
                text("""
                    INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                    VALUES (:email, :password_hash, :full_name, 'user', true, true)
                    RETURNING id
                """),
                {
                    "email": customer_email,
                    "password_hash": customer_password_hash,
                    "full_name": "Test Customer"
                }
            )
            customer_id = result.scalar()
            print(f"✅ Created customer account: {customer_email} (ID: {customer_id})")
        
        # 3. Create a Free Plan if not exists
        result = await conn.execute(
            text("SELECT id FROM plans WHERE name = 'Free Trial'")
        )
        plan_id = result.scalar()
        
        if not plan_id:
            result = await conn.execute(
                text("""
                    INSERT INTO plans (name, stripe_price_id, price_monthly, price_yearly, 
                                       max_whatsapp_accounts, max_monthly_messages, max_knowledge_base_mb,
                                       features, is_active)
                    VALUES ('Free Trial', 'price_free_trial', 0, 0, 1, 1000, 50, 
                            '{"whatsapp": true, "webchat": true, "analytics": false}'::jsonb, true)
                    RETURNING id
                """)
            )
            plan_id = result.scalar()
            print(f"✅ Created Free Trial plan (ID: {plan_id})")
        else:
            print(f"⚠️  Free Trial plan exists (ID: {plan_id})")
        
        # 4. Create a Business (Customer entity) for the customer user
        result = await conn.execute(
            text("SELECT id FROM customers WHERE user_id = :user_id"),
            {"user_id": customer_id}
        )
        business_id = result.scalar()
        
        if not business_id:
            result = await conn.execute(
                text("""
                    INSERT INTO customers (user_id, business_name, business_phone, business_email, is_active)
                    VALUES (:user_id, 'Test Business Co.', '+966500000000', 'business@test.com', true)
                    RETURNING id
                """),
                {"user_id": customer_id}
            )
            business_id = result.scalar()
            print(f"✅ Created business: Test Business Co. (ID: {business_id})")
        else:
            print(f"⚠️  Business exists (ID: {business_id})")
        
        # 5. Create Subscription for the business
        result = await conn.execute(
            text("SELECT id FROM subscriptions WHERE customer_id = :customer_id"),
            {"customer_id": business_id}
        )
        sub_id = result.scalar()
        
        if not sub_id:
            result = await conn.execute(
                text("""
                    INSERT INTO subscriptions (customer_id, plan_id, status, 
                                               current_period_start, current_period_end, trial_ends_at)
                    VALUES (:customer_id, :plan_id, 'trial',
                            NOW(), NOW() + INTERVAL '14 days', NOW() + INTERVAL '14 days')
                    RETURNING id
                """),
                {"customer_id": business_id, "plan_id": plan_id}
            )
            sub_id = result.scalar()
            print(f"✅ Created trial subscription (ID: {sub_id})")
        else:
            print(f"⚠️  Subscription exists (ID: {sub_id})")
    
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
