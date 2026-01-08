"""
Add Demo Request and Guest User Models

Revision ID: 002_demo_request
Revises: 001_initial
Create Date: 2024-12-29

Adds demo request functionality:
- DemoRequest: Contact form submissions for demo access
- GuestUser: Limited-access guest accounts for demos
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_demo_request"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create demo request status enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE demo_request_status AS ENUM ('pending', 'approved', 'rejected');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    # Create demo_requests table
    op.create_table(
        "demo_requests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("company", sa.String(255), nullable=True),
        sa.Column("company_size", sa.String(50), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("status", postgresql.ENUM("pending", "approved", "rejected", name="demo_request_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("admin_notes", sa.Text(), nullable=True),
        sa.Column("assigned_to_admin_id", sa.Integer(), nullable=True),
        sa.Column("scheduled_follow_up", sa.DateTime(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["assigned_to_admin_id"], ["users.id"], ondelete="SET NULL"),
    )
    
    # Create indexes for demo_requests
    op.create_index("ix_demo_requests_email", "demo_requests", ["email"])
    op.create_index("ix_demo_requests_status", "demo_requests", ["status"])
    
    # Create guest_users table
    op.create_table(
        "guest_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("demo_request_id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("access_token", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("max_conversations", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("conversations_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("granted_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["demo_request_id"], ["demo_requests.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("demo_request_id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("access_token"),
    )
    
    # Create indexes for guest_users
    op.create_index("ix_guest_users_demo_request_id", "guest_users", ["demo_request_id"])
    op.create_index("ix_guest_users_email", "guest_users", ["email"])
    op.create_index("ix_guest_users_access_token", "guest_users", ["access_token"])


def downgrade() -> None:
    # Drop tables
    op.drop_index("ix_guest_users_access_token", "guest_users")
    op.drop_index("ix_guest_users_email", "guest_users")
    op.drop_index("ix_guest_users_demo_request_id", "guest_users")
    op.drop_table("guest_users")
    
    op.drop_index("ix_demo_requests_status", "demo_requests")
    op.drop_index("ix_demo_requests_email", "demo_requests")
    op.drop_table("demo_requests")
    
    # Drop enum type
    op.execute("DROP TYPE IF EXISTS demo_request_status")
