#!/usr/bin/env python3
"""Extend user demo access."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta

async def main():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    engine = create_async_engine('postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai')
    
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    days_to_extend = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    async with engine.begin() as conn:
        # Calculate new expiry date (from now)
        new_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=days_to_extend)
        
        # Update the user
        await conn.execute(text(
            'UPDATE users SET expires_at = :expires_at WHERE id = :user_id'
        ), {"user_id": user_id, "expires_at": new_expires_at})
        
        print(f"✅ User {user_id} extended by {days_to_extend} days")
        print(f"   New expiry: {new_expires_at}")
        
        # Verify the update
        result = await conn.execute(text(
            'SELECT id, email, expires_at, max_conversations, conversations_used FROM users WHERE id = :user_id'
        ), {"user_id": user_id})
        row = result.fetchone()
        if row:
            print(f"\n=== Updated Status ===")
            print(f"Email: {row[1]}")
            print(f"Expires at: {row[2]}")
            print(f"Conversations: {row[4]}/{row[3]} used")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
