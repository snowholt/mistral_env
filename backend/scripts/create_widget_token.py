#!/usr/bin/env python3
"""
Create a widget token for the portal website to connect chatbot to real AI.

Usage:
    python scripts/create_widget_token.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai"


async def create_widget_token():
    """Create a widget token for portal.gmai.sa"""
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        # Get or create customer
        result = await conn.execute(
            text("SELECT id FROM users WHERE email = 'customer@test.com' LIMIT 1")
        )
        user = result.first()
        
        if not user:
            print("❌ No user found! Please run simple_create_users.py first")
            await engine.dispose()
            return
        
        user_id = user[0]
        
        # Check if customer entity exists
        result = await conn.execute(
            text("SELECT id, name FROM customers WHERE user_id = :user_id LIMIT 1"),
            {"user_id": user_id}
        )
        customer = result.first()
        
        if not customer:
            # Create customer entity
            result = await conn.execute(
                text("""
                    INSERT INTO customers (user_id, name, email, widget_greeting_message, is_active)
                    VALUES (:user_id, :name, :email, :greeting, true)
                    RETURNING id, name
                """),
                {
                    "user_id": user_id,
                    "name": "Genius AI",
                    "email": "info@gmai.sa",
                    "greeting": "مرحباً! 👋 كيف يمكنني مساعدتك اليوم؟"
                }
            )
            customer = result.first()
            print(f"✅ Created customer: {customer[1]} (ID: {customer[0]})")
        
        customer_id = customer[0]
        business_name = customer[1]
        
        print(f"📋 Customer: {business_name} (ID: {customer_id})")
        
        # Check if widget token already exists
        result = await conn.execute(
            text("SELECT id, token_prefix, is_active FROM widget_tokens WHERE customer_id = :customer_id LIMIT 1"),
            {"customer_id": customer_id}
        )
        existing = result.first()
        
        if existing:
            widget_id, token_prefix, is_active = existing
            print(f"\n⚠️  Widget token already exists!")
            print(f"Widget ID: {widget_id}")
            print(f"Token Prefix: {token_prefix}...")
            print(f"Active: {is_active}")
            print(f"\n❌ Cannot retrieve full token (stored as hash for security)")
            print(f"If you need a new token, delete the old one first:")
            print(f"   DELETE FROM widget_tokens WHERE id = {widget_id};")
            token = f"{token_prefix}[EXISTING_TOKEN_HASH_CANNOT_BE_RETRIEVED]"
        else:
            # Generate secure token using proper method
            import secrets
            import hashlib
            token = f"wt_{secrets.token_urlsafe(32)}"
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            token_prefix = token[:8]  # Only first 8 chars for prefix (wt_xxxxx)
            
            # Create widget token
            result = await conn.execute(
                text("""
                    INSERT INTO widget_tokens (customer_id, token_hash, token_prefix, name, domain_whitelist, is_active)
                    VALUES (:customer_id, :token_hash, :token_prefix, :name, :domains, true)
                    RETURNING id
                """),
                {
                    "customer_id": customer_id,
                    "token_hash": token_hash,
                    "token_prefix": token_prefix,
                    "name": "Portal Website - portal.gmai.sa",
                    "domains": ['portal.gmai.sa', 'gmai.sa', 'www.gmai.sa', 'localhost']
                }
            )
            
            widget_id = result.scalar()
            
            print(f"\n✅ Widget token created successfully!")
            print(f"Widget ID: {widget_id}")
            print(f"Token: {token}")
            print(f"Token Prefix: {token_prefix}")
            print(f"Allowed Domains: portal.gmai.sa, gmai.sa, www.gmai.sa, localhost")
            print(f"\n⚠️  IMPORTANT: Copy this token NOW! It cannot be retrieved later.")
        
        # Check if agent config exists
        result = await conn.execute(
            text("SELECT id, system_prompt FROM agent_configs WHERE customer_id = :customer_id LIMIT 1"),
            {"customer_id": customer_id}
        )
        agent = result.first()
        
        if not agent:
            # Create default agent config
            system_prompt = """أنت مساعد ذكاء اصطناعي لشركة Genius AI المتخصصة في حلول خدمة العملاء بالذكاء الاصطناعي.

معلومات الشركة:
- اسم الشركة: Genius AI
- الخدمات: وكلاء AI صوتيين، وكلاء WhatsApp الأذكياء، روبوتات الدردشة الذكية
- المنتجات:
  1. Voice AI Agent (S.I.N.A) - وكيل صوتي يتحدث باللهجة السعودية
  2. WhatsApp Smart Agent - أتمتة المحادثات على واتساب
  3. Subject Matter Expert LLM (S.I.N.A Chatbot) - روبوت دردشة للمواقع

المزايا الرئيسية:
- دعم على مدار الساعة (24/7)
- تكامل سريع مع الأنظمة الحالية
- أمان على مستوى البنوك
- ردود فورية ذكية
- لوحة تحليلات شاملة

معلومات التواصل:
- البريد الإلكتروني: info@gmai.sa
- الهاتف/واتساب: ‎+966 54 466 9879
- الموقع: الرياض، السعودية - حي الأندلس، بندر بن عبدالعزيز

مهمتك:
- الرد بطريقة ودية ومهنية
- تقديم معلومات دقيقة عن خدماتنا
- مساعدة العملاء في فهم كيف يمكن للذكاء الاصطناعي تحسين خدمة عملائهم
- تشجيع العملاء على طلب عرض توضيحي

تعليمات:
- استخدم اللغة العربية بشكل أساسي (اللهجة السعودية عند الاقتضاء)
- كن واضحاً ومختصراً في الردود
- إذا لم تعرف إجابة سؤال معين، أوجه العميل للتواصل المباشر مع الفريق"""

            result = await conn.execute(
                text("""
                    INSERT INTO agent_configs (customer_id, business_name, system_prompt, 
                                               model_name, temperature, max_tokens)
                    VALUES (:customer_id, :business_name, :system_prompt, 
                            'qwen3-unsloth-q4ks', 0.7, 500)
                    RETURNING id
                """),
                {
                    "customer_id": customer_id,
                    "business_name": "Genius AI",
                    "system_prompt": system_prompt
                }
            )
            agent_id = result.scalar()
            print(f"\n✅ Agent config created (ID: {agent_id})")
            print("Model: qwen3-unsloth-q4ks (Qwen3-14B)")
            print("Temperature: 0.7")
            print("Max Tokens: 500")
        else:
            print(f"\n✅ Agent config already exists (ID: {agent[0]})")
            print(f"System prompt: {agent[1][:100]}...")
    
    await engine.dispose()
    
    print("\n" + "="*70)
    print("🎉 NEXT STEPS:")
    print("="*70)
    print("\n1. Copy the widget token above")
    print("2. Edit: _website_snapshot/gmai.sa/gmai.sa/src/App.tsx")
    print("3. Replace: widgetToken=\"demo\"")
    print(f"4. With: widgetToken=\"{token if not existing else existing[0]}\"")
    print("5. Rebuild frontend: cd _website_snapshot/gmai.sa/gmai.sa && npm run build")
    print("6. Deploy: sudo cp -r dist/* /var/www/portal.gmai.sa/html/")
    print("7. Test the chatbot - it will now use REAL AI (Qwen3-14B)!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(create_widget_token())
