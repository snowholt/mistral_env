"""Add wizard configuration tables for enhanced agent setup.

Revision ID: 005_wizard_config
Revises: 004_unify_users
Create Date: 2026-01-09

This migration adds:
- New columns to agent_configs for wizard-based configuration
- business_services table
- business_products table  
- business_locations table
- business_promotions table
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '005_wizard_config'
down_revision = '004_unify_users'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to agent_configs table
    op.add_column('agent_configs', sa.Column('business_description', sa.Text(), nullable=True))
    op.add_column('agent_configs', sa.Column('business_type', sa.String(20), nullable=True, server_default='services'))
    op.add_column('agent_configs', sa.Column('supported_language', sa.String(20), nullable=True, server_default='both'))
    op.add_column('agent_configs', sa.Column('website_url', sa.String(500), nullable=True))
    op.add_column('agent_configs', sa.Column('booking_enabled', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('agent_configs', sa.Column('booking_link', sa.String(500), nullable=True))
    op.add_column('agent_configs', sa.Column('business_policies', sa.Text(), nullable=True))
    op.add_column('agent_configs', sa.Column('wizard_completed', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('agent_configs', sa.Column('wizard_current_step', sa.Integer(), nullable=True, server_default='1'))

    # Create business_services table
    op.create_table(
        'business_services',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_config_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('price_display', sa.String(100), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('warranty', sa.String(255), nullable=True),
        sa.Column('is_bookable', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('sort_order', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['agent_config_id'], ['agent_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_business_services_agent_config_id', 'business_services', ['agent_config_id'])

    # Create business_products table
    op.create_table(
        'business_products',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_config_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('price_min', sa.Float(), nullable=True),
        sa.Column('price_max', sa.Float(), nullable=True),
        sa.Column('price_range', sa.String(100), nullable=True),
        sa.Column('warranty', sa.String(255), nullable=True),
        sa.Column('shipping_cost', sa.String(100), nullable=True),
        sa.Column('is_available', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('sort_order', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['agent_config_id'], ['agent_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_business_products_agent_config_id', 'business_products', ['agent_config_id'])

    # Create business_locations table
    op.create_table(
        'business_locations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_config_id', sa.Integer(), nullable=False),
        sa.Column('branch_name', sa.String(255), nullable=False),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('city', sa.String(100), nullable=True),
        sa.Column('google_maps_link', sa.String(500), nullable=True),
        sa.Column('working_hours', sa.Text(), nullable=True),
        sa.Column('working_hours_json', postgresql.JSON(), nullable=True),
        sa.Column('contact_number', sa.String(50), nullable=True),
        sa.Column('extension', sa.String(20), nullable=True),
        sa.Column('whatsapp_number', sa.String(50), nullable=True),
        sa.Column('is_main_branch', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('sort_order', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['agent_config_id'], ['agent_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_business_locations_agent_config_id', 'business_locations', ['agent_config_id'])

    # Create business_promotions table
    op.create_table(
        'business_promotions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_config_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('discount_type', sa.String(50), nullable=True),
        sa.Column('discount_value', sa.String(100), nullable=True),
        sa.Column('terms_conditions', sa.Text(), nullable=True),
        sa.Column('promo_code', sa.String(50), nullable=True),
        sa.Column('valid_from', sa.DateTime(), nullable=True),
        sa.Column('valid_until', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('sort_order', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['agent_config_id'], ['agent_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_business_promotions_agent_config_id', 'business_promotions', ['agent_config_id'])


def downgrade():
    # Drop tables
    op.drop_index('ix_business_promotions_agent_config_id', table_name='business_promotions')
    op.drop_table('business_promotions')
    
    op.drop_index('ix_business_locations_agent_config_id', table_name='business_locations')
    op.drop_table('business_locations')
    
    op.drop_index('ix_business_products_agent_config_id', table_name='business_products')
    op.drop_table('business_products')
    
    op.drop_index('ix_business_services_agent_config_id', table_name='business_services')
    op.drop_table('business_services')
    
    # Remove columns from agent_configs
    op.drop_column('agent_configs', 'wizard_current_step')
    op.drop_column('agent_configs', 'wizard_completed')
    op.drop_column('agent_configs', 'business_policies')
    op.drop_column('agent_configs', 'booking_link')
    op.drop_column('agent_configs', 'booking_enabled')
    op.drop_column('agent_configs', 'website_url')
    op.drop_column('agent_configs', 'supported_language')
    op.drop_column('agent_configs', 'business_type')
    op.drop_column('agent_configs', 'business_description')
