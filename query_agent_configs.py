#!/usr/bin/env python3
"""Query agent_configs table and print system prompts."""

import os
import sys
import traceback

# Use absolute path for output
OUTPUT_PATH = '/home/lumi/beautyai/reports/agent_configs_query_result.txt'

def main():
    result_text = []
    try:
        sys.path.insert(0, '/home/lumi/beautyai/backend/src')
        
        import asyncio
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        
        async def run_query():
            db_url = 'postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai'
            engine = create_async_engine(db_url)
            
            async with engine.begin() as conn:
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
                return result.fetchall()
        
        rows = asyncio.run(run_query())
        
        result_text.append('=' * 80)
        result_text.append(f'Found {len(rows)} agent configurations')
        result_text.append('=' * 80)
        result_text.append('')
        
        for row in rows:
            result_text.append(f'ID: {row[0]}, Customer ID: {row[1]}')
            result_text.append(f'Business Name: {row[2]}')
            result_text.append(f'Customer Name: {row[5]}')
            result_text.append(f'AI Enabled: {row[3]}')
            result_text.append('')
            result_text.append('--- System Prompt ---')
            result_text.append(row[4] if row[4] else '(No system prompt)')
            result_text.append('=' * 80)
            result_text.append('')
        
        # Write to file
        with open(OUTPUT_PATH, 'w') as f:
            f.write('\n'.join(result_text))
        
        print(f"SUCCESS: Results written to {OUTPUT_PATH}")
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        result_text.append(error_msg)
        with open(OUTPUT_PATH, 'w') as f:
            f.write('\n'.join(result_text))
        print(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
