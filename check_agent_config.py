#!/usr/bin/env python3
"""Quick script to check agent_configs table."""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, '/home/lumi/beautyai/backend/src')

async def main():
    # Use the database connection from the app
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    # Get DB URL from environment or use default
    db_url = os.getenv(
        'DATABASE_URL', 
        'postgresql+asyncpg://beautyai:vHHC6U-dpJrCRjCd@localhost/beautyai'
    )
    
    engine = create_async_engine(db_url)
    
    async with engine.begin() as conn:
        # Query agent_configs - get FULL system_prompt
        result = await conn.execute(text('''
            SELECT 
                ac.id,
                ac.customer_id,
                ac.business_name,
                ac.ai_enabled,
                ac.system_prompt,
                c.name as customer_name
            FROM agent_configs ac
            JOIN customers c ON c.id = ac.customer_id
            ORDER BY ac.id
            LIMIT 10
        '''))
        rows = result.fetchall()
        
        print(f"\n{'='*80}")
        print(f"Found {len(rows)} agent configurations:")
        print(f"{'='*80}\n")
        
        for row in rows:
            print(f"ID: {row[0]}, Customer ID: {row[1]}")
            print(f"Business Name: {row[2]}")
            print(f"Customer Name: {row[5]}")
            print(f"AI Enabled: {row[3]}")
            print(f"\n--- System Prompt ---")
            print(row[4] if row[4] else "(No system prompt)")
            print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
