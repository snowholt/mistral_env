"""
Make WhatsAppAccount.access_token nullable

This migration allows WhatsApp accounts to use encrypted credentials
(via MetaCredential) without requiring plaintext access_token storage.

Revision ID: 009_make_access_token_nullable
Revises: ee7ef5badf68
Create Date: 2026-02-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '009_make_access_token_nullable'
down_revision: Union[str, None] = 'ee7ef5badf68'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make access_token column nullable for encrypted-only storage."""
    # Allow NULL for accounts that use only encrypted credential storage
    op.alter_column(
        'whatsapp_accounts',
        'access_token',
        existing_type=sa.Text(),
        nullable=True,
    )


def downgrade() -> None:
    """Revert to non-nullable access_token (requires data migration)."""
    # Note: This downgrade may fail if there are rows with NULL access_token.
    # Before running this downgrade, ensure all accounts have access_token populated.
    op.alter_column(
        'whatsapp_accounts',
        'access_token',
        existing_type=sa.Text(),
        nullable=False,
    )
