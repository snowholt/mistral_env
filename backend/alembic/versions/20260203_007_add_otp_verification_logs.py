"""Add OTP verification audit logs table

Revision ID: 007_otp_verification_logs
Revises: 006_demo_appointments
Create Date: 2026-02-03

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '007_otp_verification_logs'
down_revision = '006_demo_appointments'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'otp_verification_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('purpose', sa.String(length=50), nullable=False, server_default='whatsapp_connect'),
        sa.Column('action', sa.String(length=20), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('failure_reason', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_otp_verification_logs_email', 'otp_verification_logs', ['email'])
    op.create_index('ix_otp_verification_logs_user_id', 'otp_verification_logs', ['user_id'])
    op.create_index('ix_otp_logs_user_created', 'otp_verification_logs', ['user_id', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_otp_logs_user_created', table_name='otp_verification_logs')
    op.drop_index('ix_otp_verification_logs_user_id', table_name='otp_verification_logs')
    op.drop_index('ix_otp_verification_logs_email', table_name='otp_verification_logs')
    op.drop_table('otp_verification_logs')
