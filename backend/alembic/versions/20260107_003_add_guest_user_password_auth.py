"""
Add Password-Based Authentication for Guest Users

Revision ID: 003_guest_password
Revises: 002_demo_request
Create Date: 2026-01-07

Adds secure password-based authentication for guest demo accounts:
- password_hash: Bcrypt hashed password for login
- is_activated: Whether the account has been activated with a password
- setup_token: Short-lived token for account activation (replaces direct access_token login)
- setup_token_expires: Expiration time for setup token (1 hour default)

Flow:
1. Admin approves demo request → GuestUser created with setup_token
2. User clicks email link → validates token, sets password  
3. After password set → setup_token invalidated, is_activated=True
4. Subsequent logins use email + password
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "003_guest_password"
down_revision: Union[str, None] = "002_demo_request"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add password authentication fields to guest_users
    op.add_column(
        "guest_users",
        sa.Column("password_hash", sa.String(255), nullable=True)
    )
    
    op.add_column(
        "guest_users",
        sa.Column("is_activated", sa.Boolean(), nullable=False, server_default="false")
    )
    
    # Add setup token fields (short-lived, for account activation)
    op.add_column(
        "guest_users",
        sa.Column("setup_token", sa.String(255), nullable=True)
    )
    
    op.add_column(
        "guest_users",
        sa.Column("setup_token_expires", sa.DateTime(), nullable=True)
    )
    
    # Create index on setup_token for fast lookup
    op.create_index(
        "ix_guest_users_setup_token",
        "guest_users",
        ["setup_token"],
        unique=True,
        postgresql_where=sa.text("setup_token IS NOT NULL")
    )


def downgrade() -> None:
    # Remove index first
    op.drop_index("ix_guest_users_setup_token", "guest_users")
    
    # Remove columns
    op.drop_column("guest_users", "setup_token_expires")
    op.drop_column("guest_users", "setup_token")
    op.drop_column("guest_users", "is_activated")
    op.drop_column("guest_users", "password_hash")
