#!/usr/bin/env python3
"""
Token Migration Script: Migrate plaintext access_tokens to encrypted vault.

This script:
1. Reads existing WhatsAppAccount records with plaintext access_token
2. Encrypts each token using MetaCredentialService
3. Creates MetaCredential records
4. Links WhatsAppAccount.credential_id to the new encrypted credential
5. Logs all operations for audit trail

Usage:
    cd backend
    source venv/bin/activate
    python scripts/migrate_tokens_to_vault.py [--dry-run] [--verbose]

Options:
    --dry-run   Preview changes without committing
    --verbose   Show detailed progress

Safety:
    - Idempotent: Safe to run multiple times
    - Skips accounts already linked to credentials
    - Does NOT delete original access_token (requires separate step)
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add backend/src to path for imports
backend_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(backend_src))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from beautyai_inference.database.models import (
    WhatsAppAccount,
    MetaCredential,
    CredentialType,
    AuditLog,
)
from beautyai_inference.utils.encryption import get_encryption_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment."""
    # Check for async URL first
    url = os.getenv("DATABASE_URL_ASYNC")
    if url:
        return url
    
    # Fall back to sync URL and convert
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    # Default for development
    return "postgresql+asyncpg://beautyai:beautyai123@localhost:5432/beautyai"


async def migrate_tokens(dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Migrate plaintext tokens to encrypted MetaCredential vault.
    
    Args:
        dry_run: If True, preview changes without committing.
        verbose: If True, show detailed progress.
        
    Returns:
        Migration statistics dictionary.
    """
    stats = {
        "total_accounts": 0,
        "already_migrated": 0,
        "migrated": 0,
        "failed": 0,
        "errors": [],
    }
    
    # Create async engine
    database_url = get_database_url()
    logger.info(f"Connecting to database...")
    
    engine = create_async_engine(database_url, echo=verbose)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Get encryption service
    encryption = get_encryption_service()
    logger.info(f"Encryption service initialized (key version: {encryption.key_version})")
    
    async with async_session() as db:
        # Get all WhatsApp accounts
        result = await db.execute(select(WhatsAppAccount))
        accounts = list(result.scalars().all())
        stats["total_accounts"] = len(accounts)
        
        logger.info(f"Found {len(accounts)} WhatsApp accounts to process")
        
        for account in accounts:
            try:
                # Skip if already migrated
                if account.credential_id is not None:
                    stats["already_migrated"] += 1
                    if verbose:
                        logger.info(
                            f"  [SKIP] Account id={account.id} already has "
                            f"credential_id={account.credential_id}"
                        )
                    continue
                
                # Skip if no access_token
                if not account.access_token:
                    logger.warning(f"  [SKIP] Account id={account.id} has no access_token")
                    stats["failed"] += 1
                    stats["errors"].append({
                        "account_id": account.id,
                        "error": "No access_token to migrate",
                    })
                    continue
                
                if verbose:
                    logger.info(
                        f"  [MIGRATE] Account id={account.id} "
                        f"phone_number_id={account.phone_number_id}"
                    )
                
                if not dry_run:
                    # Encrypt the token
                    encrypted_value, key_version = encryption.encrypt_with_version(
                        account.access_token
                    )
                    
                    # Create MetaCredential
                    credential = MetaCredential(
                        customer_id=account.customer_id,
                        credential_type=CredentialType.USER_TOKEN,
                        encrypted_value=encrypted_value,
                        encryption_key_version=key_version,
                        scopes=["whatsapp_business_management", "whatsapp_business_messaging"],
                        is_active=True,
                    )
                    db.add(credential)
                    await db.flush()  # Get credential.id
                    
                    # Link to WhatsApp account
                    account.credential_id = credential.id
                    
                    # Create audit log
                    audit_log = AuditLog(
                        action="credential.migrated",
                        resource_type="meta_credential",
                        resource_id=str(credential.id),
                        customer_id=account.customer_id,
                        metadata={
                            "whatsapp_account_id": account.id,
                            "phone_number_id": account.phone_number_id,
                            "source": "migrate_tokens_to_vault.py",
                        },
                    )
                    db.add(audit_log)
                    
                    logger.info(
                        f"  [OK] Migrated account id={account.id} → "
                        f"credential id={credential.id}"
                    )
                else:
                    logger.info(
                        f"  [DRY-RUN] Would migrate account id={account.id}"
                    )
                
                stats["migrated"] += 1
                
            except Exception as e:
                logger.error(f"  [ERROR] Account id={account.id}: {e}")
                stats["failed"] += 1
                stats["errors"].append({
                    "account_id": account.id,
                    "error": str(e),
                })
        
        # Commit all changes
        if not dry_run:
            await db.commit()
            logger.info("Changes committed to database")
        else:
            logger.info("Dry-run mode: No changes committed")
    
    await engine.dispose()
    return stats


def print_summary(stats: dict, dry_run: bool) -> None:
    """Print migration summary."""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total WhatsApp accounts:  {stats['total_accounts']}")
    print(f"Already migrated (skip):  {stats['already_migrated']}")
    print(f"{'Would migrate' if dry_run else 'Migrated'}:           {stats['migrated']}")
    print(f"Failed:                   {stats['failed']}")
    
    if stats["errors"]:
        print("\nErrors:")
        for err in stats["errors"]:
            print(f"  - Account {err['account_id']}: {err['error']}")
    
    if dry_run:
        print("\n⚠️  DRY-RUN MODE: No changes were made")
        print("    Run without --dry-run to apply changes")
    else:
        print("\n✅ Migration complete!")
        print("   Next steps:")
        print("   1. Verify tokens work: test sending a WhatsApp message")
        print("   2. After validation, consider removing plaintext access_token column")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate WhatsApp access tokens to encrypted vault"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress",
    )
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Running in DRY-RUN mode (no changes will be made)")
    
    try:
        stats = asyncio.run(migrate_tokens(dry_run=args.dry_run, verbose=args.verbose))
        print_summary(stats, args.dry_run)
        
        # Exit with error code if any failures
        if stats["failed"] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nMigration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
