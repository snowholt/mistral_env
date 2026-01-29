"""Add demo appointment tables for voice demo

Revision ID: 006_demo_appointments
Revises: 20260109_005_add_wizard_config_tables
Create Date: 2026-01-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_demo_appointments'
down_revision = '005_wizard_config'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create appointment status enum (checkfirst=True handles if it already exists)
    bind = op.get_bind()
    # Check if enum already exists
    result = bind.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'appointmentstatus')"
    ))
    enum_exists = result.scalar()
    
    if not enum_exists:
        appointment_status = postgresql.ENUM(
            'pending', 'confirmed', 'cancelled', 'completed', 'no_show',
            name='appointmentstatus',
            create_type=True
        )
        appointment_status.create(bind, checkfirst=True)
    
    # Create demo_customers table
    op.create_table(
        'demo_customers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('phone', sa.String(50), nullable=True),
        sa.Column('email', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('preferred_language', sa.String(10), nullable=False, server_default='ar'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_demo_customers_phone', 'demo_customers', ['phone'])
    op.create_index('ix_demo_customers_email', 'demo_customers', ['email'])
    
    # Create demo_time_slots table
    op.create_table(
        'demo_time_slots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('start_time', sa.String(10), nullable=False),
        sa.Column('end_time', sa.String(10), nullable=False),
        sa.Column('duration_minutes', sa.Integer(), nullable=False, server_default='30'),
        sa.Column('max_bookings', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('current_bookings', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_available', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_demo_time_slots_date', 'demo_time_slots', ['date'])
    op.create_index('ix_demo_slots_date_time', 'demo_time_slots', ['date', 'start_time'])
    
    # Create demo_appointments table
    # Use postgresql.ENUM with create_type=False since we already created the enum above
    status_enum = postgresql.ENUM(
        'pending', 'confirmed', 'cancelled', 'completed', 'no_show',
        name='appointmentstatus',
        create_type=False
    )
    op.create_table(
        'demo_appointments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('time_slot_id', sa.Integer(), nullable=False),
        sa.Column('service_type', sa.String(100), nullable=False, server_default='consultation'),
        sa.Column('status', status_enum, nullable=False, server_default='pending'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('voice_session_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.Column('confirmed_at', sa.DateTime(), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['customer_id'], ['demo_customers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['time_slot_id'], ['demo_time_slots.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_demo_appointments_customer_id', 'demo_appointments', ['customer_id'])
    op.create_index('ix_demo_appointments_time_slot_id', 'demo_appointments', ['time_slot_id'])


def downgrade() -> None:
    # Drop tables
    op.drop_table('demo_appointments')
    op.drop_table('demo_time_slots')
    op.drop_table('demo_customers')
    
    # Drop enum
    appointment_status = postgresql.ENUM(
        'pending', 'confirmed', 'cancelled', 'completed', 'no_show',
        name='appointmentstatus'
    )
    appointment_status.drop(op.get_bind(), checkfirst=True)
