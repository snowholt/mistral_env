"""unify_users_table

Revision ID: 004_unify_users
Revises: 003_guest_password
Create Date: 2026-01-09 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004_unify_users'
down_revision: Union[str, None] = '003_guest_password'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Add 'guest' to UserRole enum
    # Alembic/Postgres enum handling is tricky.
    # We use execute with autocommit block for ALTER TYPE
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE userrole ADD VALUE 'guest'")

    # 2. Add new columns to the users table
    op.add_column('users', sa.Column('demo_request_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_users_demo_request_id', 'users', 'demo_requests', ['demo_request_id'], ['id'], ondelete='SET NULL')
    
    op.add_column('users', sa.Column('expires_at', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('max_conversations', sa.Integer(), server_default='10', nullable=True))
    op.add_column('users', sa.Column('conversations_used', sa.Integer(), server_default='0', nullable=True))

    # 3. Create index for performance
    op.create_index('ix_users_demo_request_id', 'users', ['demo_request_id'], unique=False)


def downgrade() -> None:
    # 1. Drop index
    op.drop_index('ix_users_demo_request_id', table_name='users')
    
    # 2. Drop columns
    op.drop_column('users', 'conversations_used')
    op.drop_column('users', 'max_conversations')
    op.drop_column('users', 'expires_at')
    
    # Drop constraint before column
    op.drop_constraint('fk_users_demo_request_id', 'users', type_='foreignkey')
    op.drop_column('users', 'demo_request_id')
    
    # NOTE: Postgres does not support removing enum values easily (requires recreating type).
    # We will skip removing 'guest' from userrole enum in downgrade to maintain data integrity safe.
