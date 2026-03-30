
import asyncio
import os
import sys
from datetime import timedelta

# Add backend directory to sys.path
sys.path.append(os.path.abspath("backend/src"))
sys.path.append(os.path.abspath("backend"))

# Load env vars first
if os.path.exists("backend/.env"):
    from dotenv import load_dotenv
    load_dotenv("backend/.env")

from beautyai_inference.auth.jwt_handler import create_access_token
from beautyai_inference.database.connection import get_db_context
from beautyai_inference.database.models import User
from sqlalchemy import select

async def main():
    # 1. Get User
    print("Connecting to DB...")
    async with get_db_context() as session:
        print("Querying user...")
        result = await session.execute(select(User).where(User.email == "nariman@gmai.sa"))
        user = result.scalar_one_or_none()
        
        if not user:
            print("User not found!")
            return

        print(f"User: {user.email}")
        print(f"ID: {user.id}")
        print(f"Active: {user.is_active}")
        print(f"Verified: {user.is_verified}")

        # 2. Generate Token
        token = create_access_token(user.id, user.email, expires_delta=timedelta(minutes=-10))
        print(f"\nToken generated (EXPIRED).")

        # 3. Test Endpoint
        import httpx
        url = "http://localhost:8000/api/v1/auth/otp/request"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"purpose": "whatsapp_connect"}
        
        print(f"\nMaking request to {url}...")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers)
                print(f"Status: {resp.status_code}")
                print(f"Response: {resp.text[:500]}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
