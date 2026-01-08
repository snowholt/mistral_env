import asyncio
import sys
from sqlalchemy import select
from beautyai_inference.database.connection import get_db_context
from beautyai_inference.database.models import User, GuestUser
from beautyai_inference.auth.password import verify_password

async def check_user(email, password):
    async with get_db_context() as db:
        print(f"Checking for email: {email}")
        
        # Check User table
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            print(f"Found in User table: ID={user.id}, Role={user.role}")
            print(f"Is Active: {user.is_active}")
            if user.password_hash:
                is_valid = verify_password(password, user.password_hash)
                print(f"Password Valid: {is_valid}")
            else:
                print("No password hash set.")
        else:
            print("Not found in User table.")

        # Check GuestUser table
        result = await db.execute(select(GuestUser).where(GuestUser.email == email))
        guest = result.scalar_one_or_none()
        
        if guest:
            print(f"Found in GuestUser table: ID={guest.id}, Active={guest.is_active}")
            print(f"Activated: {guest.is_activated}")
            if guest.password_hash:
                is_valid = verify_password(password, guest.password_hash)
                print(f"Password Valid (Guest Table): {is_valid}")
            else:
                print("No password hash set in GuestUser table.")
        else:
            print("Not found in GuestUser table.")

if __name__ == "__main__":
    email = "jafari.nariman@gmail.com"
    password = "4yZ37u4+U)Byr!"
    asyncio.run(check_user(email, password))
