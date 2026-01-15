#!/usr/bin/env python3
"""Check and update user demo status."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta

async def main():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    engine = create_async_engine('postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai')
    
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    
    async with engine.connect() as conn:
        # Check current status
        result = await conn.execute(text(
            'SELECT id, email, role, expires_at, max_conversations, conversations_used, is_active '
            'FROM users WHERE id = :user_id'
        ), {"user_id": user_id})
        row = result.fetchone()
        
        if row:
            print(f"=== User Status ===")
            print(f"User ID: {row[0]}")
            print(f"Email: {row[1]}")
            print(f"Role: {row[2]}")
            print(f"Expires at: {row[3]}")
            print(f"Max conversations: {row[4]}")
            print(f"Conversations used: {row[5]}")
            print(f"Is active: {row[6]}")
            
            # Check if expired
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if row[3] and row[3] < now:
                print(f"\n⚠️  EXPIRED! (expired {(now - row[3]).days} days ago)")
            
            # Check if limit reached
            if row[4] and row[5] >= row[4]:
                print(f"\n⚠️  LIMIT REACHED! ({row[5]}/{row[4]} conversations used)")
        else:
            print(f"User {user_id} not found")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
