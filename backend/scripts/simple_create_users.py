#!/usr/bin/env python3
"""Simple script to create test accounts - minimal version."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from beautyai_inference.auth.password import hash_password

DATABASE_URL = "postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai"


async def main():
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    admin_password_hash = hash_password("Admin@123456")
    customer_password_hash = hash_password("Customer@123456")
    
    async with engine.begin() as conn:
        # Delete existing test users to start fresh
        await conn.execute(text("DELETE FROM users WHERE email IN ('nariman@gmai.sa', 'customer@test.com')"))
        
        # Create Admin
        result = await conn.execute(
            text("""
                INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                VALUES (:email, :password_hash, :full_name, 'admin', true, true)
                RETURNING id
            """),
            {"email": "nariman@gmai.sa", "password_hash": admin_password_hash, "full_name": "Nariman Admin"}
        )
        admin_id = result.scalar()
        print(f"Admin created: nariman@gmai.sa (ID: {admin_id})")
        
        # Create Customer
        result = await conn.execute(
            text("""
                INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
                VALUES (:email, :password_hash, :full_name, 'user', true, true)
                RETURNING id
            """),
            {"email": "customer@test.com", "password_hash": customer_password_hash, "full_name": "Test Customer"}
        )
        customer_id = result.scalar()
        print(f"Customer created: customer@test.com (ID: {customer_id})")
        
        # Verify
        result = await conn.execute(text("SELECT id, email, role, is_verified FROM users"))
        print("\nAll users:")
        for row in result:
            print(f"  {row}")
    
    await engine.dispose()
    
    print("\n" + "="*50)
    print("ADMIN: nariman@gmai.sa / Admin@123456")
    print("CUSTOMER: customer@test.com / Customer@123456")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
