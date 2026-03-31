"""Add security models (MetaCredential, AuditLog) for token encryption

Revision ID: 008_security_models
Revises: 007_otp_verification_logs
Create Date: 2026-02-05

Security hardening phase 1:
- MetaCredential: Encrypted vault for Meta API tokens
- AuditLog: Compliance audit trail
- WhatsAppAccount.credential_id: FK to encrypted credential
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008_security_models'
down_revision = '007_otp_verification_logs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    
    # =========================================================================
    # Create credentialtype enum
    # =========================================================================
    credentialtype_enum = postgresql.ENUM(
        'user_token', 'system_user_token', 'page_token',
        name='credentialtype',
        create_type=False
    )
    
    # Check if enum exists
    enum_exists = bind.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'credentialtype')")
    ).scalar()
    
    if not enum_exists:
        credentialtype_enum.create(bind, checkfirst=True)
    
    # =========================================================================
    # Create meta_credentials table
    # =========================================================================
    meta_credentials_exists = bind.execute(
        sa.text("SELECT to_regclass('public.meta_credentials')")
    ).scalar()
    
    if not meta_credentials_exists:
        op.create_table(
            'meta_credentials',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('customer_id', sa.Integer(), nullable=False),
            sa.Column('credential_type', credentialtype_enum, nullable=False, server_default='user_token'),
            sa.Column('encrypted_value', sa.LargeBinary(), nullable=False),
            sa.Column('encryption_key_version', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('scopes', postgresql.ARRAY(sa.String(length=100)), nullable=True),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('last_used_at', sa.DateTime(), nullable=True),
            sa.Column('use_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_meta_credentials_customer_id', 'meta_credentials', ['customer_id'])
        op.create_index('ix_meta_credentials_customer_type', 'meta_credentials', ['customer_id', 'credential_type'])
    else:
        # Ensure indexes exist if table already exists
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_meta_credentials_customer_id "
            "ON meta_credentials (customer_id)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_meta_credentials_customer_type "
            "ON meta_credentials (customer_id, credential_type)"
        ))
    
    # =========================================================================
    # Create audit_logs table
    # =========================================================================
    audit_logs_exists = bind.execute(
        sa.text("SELECT to_regclass('public.audit_logs')")
    ).scalar()
    
    if not audit_logs_exists:
        op.create_table(
            'audit_logs',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('customer_id', sa.Integer(), nullable=True),
            sa.Column('user_id', sa.Integer(), nullable=True),
            sa.Column('action', sa.String(length=100), nullable=False),
            sa.Column('resource_type', sa.String(length=100), nullable=False),
            sa.Column('resource_id', sa.String(length=100), nullable=True),
            sa.Column('details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.Column('ip_address', sa.String(length=45), nullable=True),
            sa.Column('user_agent', sa.Text(), nullable=True),
            sa.Column('request_id', sa.String(length=100), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ondelete='SET NULL'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_audit_logs_customer_id', 'audit_logs', ['customer_id'])
        op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
        op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
        op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'])
        op.create_index('ix_audit_logs_customer_created', 'audit_logs', ['customer_id', 'created_at'])
        op.create_index('ix_audit_logs_action_created', 'audit_logs', ['action', 'created_at'])
    else:
        # Ensure indexes exist if table already exists
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_customer_id ON audit_logs (customer_id)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_user_id ON audit_logs (user_id)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_action ON audit_logs (action)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs (created_at)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_customer_created ON audit_logs (customer_id, created_at)"
        ))
        bind.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_action_created ON audit_logs (action, created_at)"
        ))
    
    # =========================================================================
    # Add credential_id column to whatsapp_accounts
    # =========================================================================
    # Check if column exists
    column_exists = bind.execute(
        sa.text("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'whatsapp_accounts' 
                AND column_name = 'credential_id'
            )
        """)
    ).scalar()
    
    if not column_exists:
        op.add_column(
            'whatsapp_accounts',
            sa.Column('credential_id', sa.Integer(), nullable=True)
        )
        op.create_foreign_key(
            'fk_whatsapp_accounts_credential_id',
            'whatsapp_accounts',
            'meta_credentials',
            ['credential_id'],
            ['id'],
            ondelete='SET NULL'
        )
        op.create_index('ix_whatsapp_accounts_credential_id', 'whatsapp_accounts', ['credential_id'])


def downgrade() -> None:
    bind = op.get_bind()
    
    # Remove credential_id from whatsapp_accounts
    column_exists = bind.execute(
        sa.text("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'whatsapp_accounts' 
                AND column_name = 'credential_id'
            )
        """)
    ).scalar()
    
    if column_exists:
        op.drop_index('ix_whatsapp_accounts_credential_id', table_name='whatsapp_accounts')
        op.drop_constraint('fk_whatsapp_accounts_credential_id', 'whatsapp_accounts', type_='foreignkey')
        op.drop_column('whatsapp_accounts', 'credential_id')
    
    # Drop audit_logs table
    audit_logs_exists = bind.execute(
        sa.text("SELECT to_regclass('public.audit_logs')")
    ).scalar()
    
    if audit_logs_exists:
        op.drop_index('ix_audit_logs_action_created', table_name='audit_logs')
        op.drop_index('ix_audit_logs_customer_created', table_name='audit_logs')
        op.drop_index('ix_audit_logs_created_at', table_name='audit_logs')
        op.drop_index('ix_audit_logs_action', table_name='audit_logs')
        op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
        op.drop_index('ix_audit_logs_customer_id', table_name='audit_logs')
        op.drop_table('audit_logs')
    
    # Drop meta_credentials table
    meta_credentials_exists = bind.execute(
        sa.text("SELECT to_regclass('public.meta_credentials')")
    ).scalar()
    
    if meta_credentials_exists:
        op.drop_index('ix_meta_credentials_customer_type', table_name='meta_credentials')
        op.drop_index('ix_meta_credentials_customer_id', table_name='meta_credentials')
        op.drop_table('meta_credentials')
    
    # Drop credentialtype enum
    enum_exists = bind.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'credentialtype')")
    ).scalar()
    
    if enum_exists:
        op.execute("DROP TYPE IF EXISTS credentialtype")
