import asyncio
import logging
import os
import sys

# Add backend source to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend/src"))

from sqlalchemy import select
from beautyai_inference.database.connection import get_db_context
from beautyai_inference.database.models import User, GuestUser, UserRole, DemoRequest

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_guests_to_users():
    """
    Migrate existing GuestUser records to the unified User table.
    """
    async with get_db_context() as db:
        logger.info("Starting migration of GuestUsers to Users table...")
        
        # 1. Fetch all GuestUsers
        result = await db.execute(select(GuestUser))
        guest_users = result.scalars().all()
        
        migrated_count = 0
        skipped_count = 0
        
        for guest in guest_users:
            logger.info(f"Processing guest: {guest.email}")
            
            # 2. Check if user already exists
            user_result = await db.execute(select(User).where(User.email == guest.email))
            existing_user = user_result.scalar_one_or_none()
            
            if existing_user:
                logger.warning(f"User {guest.email} already exists in User table. Skipping creation.")
                # Optional: Update existing user with guest details if needed?
                # For now, we assume if they exist in User table, they are handled.
                # But we might want to ensure they have the guest fields set if they are just a user?
                if existing_user.role != UserRole.ADMIN: # Don't downgrade admins
                    logger.info(f"Updating existing user {existing_user.id} with guest details.")
                    existing_user.demo_request_id = guest.demo_request_id
                    existing_user.expires_at = guest.expires_at
                    existing_user.max_conversations = guest.max_conversations
                    existing_user.conversations_used = guest.conversations_used
                    existing_user.role = UserRole.GUEST # Ensure they count as guest for checks
                skipped_count += 1
                continue
                
            # 3. Create new User from GuestUser
            try:
                # We need to fetch the full name from the demo request
                demo_req_result = await db.execute(select(DemoRequest).where(DemoRequest.id == guest.demo_request_id))
                demo_request = demo_req_result.scalar_one_or_none()
                full_name = demo_request.full_name() if demo_request else "Guest User"
                
                new_user = User(
                    email=guest.email,
                    # If guest has a password set (activated), use it. 
                    # If not, they can't login anyway, but we migrate them.
                    password_hash=guest.password_hash or "pending_activation", 
                    full_name=full_name,
                    role=UserRole.GUEST,
                    is_active=guest.is_active,
                    is_verified=guest.is_activated, # Map Is Activated -> Is Verified
                    
                    # Guest specific fields
                    demo_request_id=guest.demo_request_id,
                    expires_at=guest.expires_at,
                    max_conversations=guest.max_conversations,
                    conversations_used=guest.conversations_used,
                    
                    # Migration flag or similar could be useful, but unnecessary now
                )
                
                # If guest had a setup token, maybe we should move it to verification_token?
                # The auth code we wrote supports checking `verification_token` for unified users.
                if guest.setup_token and not guest.is_activated:
                    new_user.verification_token = guest.setup_token
                    new_user.verification_token_expires = guest.setup_token_expires
                
                db.add(new_user)
                migrated_count += 1
                logger.info(f"Migrated {guest.email} -> User ID (pending commit)")
                
            except Exception as e:
                logger.error(f"Failed to migrate {guest.email}: {e}")
        
        # Commit all changes
        await db.commit()
        logger.info(f"Migration complete. Migrated: {migrated_count}, Skipped/Updated: {skipped_count}")

if __name__ == "__main__":
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL environment variable not set.")
        sys.exit(1)
        
    asyncio.run(migrate_guests_to_users())
