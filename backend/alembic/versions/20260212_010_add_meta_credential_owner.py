"""
Add created_by_user_id to meta_credentials

Revision ID: 010_add_meta_credential_owner
Revises: 009_make_access_token_nullable
Create Date: 2026-02-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "010_add_meta_credential_owner"
down_revision: Union[str, None] = "009_make_access_token_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add created_by_user_id to meta_credentials and backfill ownership."""
    bind = op.get_bind()

    column_exists = bind.execute(
        sa.text(
            """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'meta_credentials'
                AND column_name = 'created_by_user_id'
            )
            """
        )
    ).scalar()

    if not column_exists:
        op.add_column(
            "meta_credentials",
            sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        )
        op.create_foreign_key(
            "fk_meta_credentials_created_by_user_id",
            "meta_credentials",
            "users",
            ["created_by_user_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_index(
            "ix_meta_credentials_created_by_user_id",
            "meta_credentials",
            ["created_by_user_id"],
        )

    # Backfill ownership from customers.user_id where possible
    bind.execute(
        sa.text(
            """
            UPDATE meta_credentials mc
            SET created_by_user_id = c.user_id
            FROM customers c
            WHERE mc.customer_id = c.id
              AND mc.created_by_user_id IS NULL
              AND c.user_id IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    """Remove created_by_user_id from meta_credentials."""
    bind = op.get_bind()

    column_exists = bind.execute(
        sa.text(
            """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'meta_credentials'
                AND column_name = 'created_by_user_id'
            )
            """
        )
    ).scalar()

    if column_exists:
        op.drop_index("ix_meta_credentials_created_by_user_id", table_name="meta_credentials")
        op.drop_constraint(
            "fk_meta_credentials_created_by_user_id",
            "meta_credentials",
            type_="foreignkey",
        )
        op.drop_column("meta_credentials", "created_by_user_id")
