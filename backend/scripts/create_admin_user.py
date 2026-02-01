#!/usr/bin/env python3
"""
Create an admin user directly in the database.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/create_admin_user.py --email admin@gmai.sa --password YourPassword123! --name "Admin User"
    
Or interactive mode:
    python scripts/create_admin_user.py
"""
import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
# If run from backend/scripts/:
project_root = Path(__file__).parent.parent
# If run directly from backend/:
if Path.cwd().name == "backend" and (Path.cwd() / "src").exists():
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load .env before other imports
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from passlib.context import CryptContext
from sqlalchemy import text
from beautyai_inference.database.connection import get_db_context

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto', bcrypt__rounds=12)


async def create_admin(email: str, password: str, full_name: str, role: str = "admin"):
    """Create or update a user in the database."""
    password_hash = pwd_context.hash(password)
    
    async with get_db_context() as db:
        await db.execute(text('''
            INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified, created_at, updated_at)
            VALUES (:email, :password_hash, :full_name, :role, true, true, NOW(), NOW())
            ON CONFLICT (email) DO UPDATE SET
                password_hash = :password_hash,
                full_name = :full_name,
                role = :role,
                is_active = true,
                is_verified = true,
                updated_at = NOW()
        '''), {
            'email': email,
            'password_hash': password_hash,
            'full_name': full_name,
            'role': role
        })
    print(f'✅ Created/updated {role} user: {email}')


def main():
    parser = argparse.ArgumentParser(description='Create an admin user in the database')
    parser.add_argument('--email', '-e', help='User email address')
    parser.add_argument('--password', '-p', help='User password')
    parser.add_argument('--name', '-n', help='Full name')
    parser.add_argument('--role', '-r', default='admin', choices=['admin', 'user', 'guest'],
                        help='User role (default: admin)')
    
    args = parser.parse_args()
    
    # Interactive mode if args not provided
    email = args.email or input('Email: ').strip()
    password = args.password or input('Password: ').strip()
    full_name = args.name or input('Full Name: ').strip()
    role = args.role
    
    if not email or not password or not full_name:
        print('❌ Email, password, and name are required')
        sys.exit(1)
    
    print(f'\n📝 Creating {role} user:')
    print(f'   Email: {email}')
    print(f'   Name: {full_name}')
    print(f'   Role: {role}')
    
    asyncio.run(create_admin(email, password, full_name, role))


if __name__ == '__main__':
    main()
