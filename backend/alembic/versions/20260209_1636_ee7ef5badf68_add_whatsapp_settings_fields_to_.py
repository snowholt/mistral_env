"""
Add WhatsApp settings fields to AgentConfig

Revision ID: ee7ef5badf68
Revises: 008_security_models
Create Date: 2026-02-09 16:36:20.025998+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'ee7ef5badf68'
down_revision: Union[str, None] = '008_security_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'agent_configs',
        sa.Column(
            'max_response_length',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('500'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'response_delay_seconds',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('2'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'email_notifications',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('true'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'notify_on_new_conversation',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('true'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'notify_on_inactivity',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'inactivity_threshold_minutes',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('30'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column(
            'business_hours_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'agent_configs',
        sa.Column('outside_hours_message', sa.Text(), nullable=True),
    )

    op.alter_column('agent_configs', 'max_response_length', server_default=None)
    op.alter_column('agent_configs', 'response_delay_seconds', server_default=None)
    op.alter_column('agent_configs', 'email_notifications', server_default=None)
    op.alter_column('agent_configs', 'notify_on_new_conversation', server_default=None)
    op.alter_column('agent_configs', 'notify_on_inactivity', server_default=None)
    op.alter_column('agent_configs', 'inactivity_threshold_minutes', server_default=None)
    op.alter_column('agent_configs', 'business_hours_enabled', server_default=None)


def downgrade() -> None:
    op.drop_column('agent_configs', 'outside_hours_message')
    op.drop_column('agent_configs', 'business_hours_enabled')
    op.drop_column('agent_configs', 'inactivity_threshold_minutes')
    op.drop_column('agent_configs', 'notify_on_inactivity')
    op.drop_column('agent_configs', 'notify_on_new_conversation')
    op.drop_column('agent_configs', 'email_notifications')
    op.drop_column('agent_configs', 'response_delay_seconds')
    op.drop_column('agent_configs', 'max_response_length')
