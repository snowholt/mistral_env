"""
Alembic migration environment for BeautyAI SaaS Platform.

Supports both sync and async database connections.
"""

import os
import sys
from logging.config import fileConfig

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import pool, create_engine
from sqlalchemy.engine import Connection

from alembic import context

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models to ensure all tables are registered
from src.beautyai_inference.database.models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for autogenerate support
target_metadata = Base.metadata


def get_database_url() -> str:
    """Get database URL from environment, converting to sync driver."""
    url = os.getenv(
        "DATABASE_URL_SYNC",
        os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/beautyai")
    )
    
    # Convert async URL to sync URL for Alembic
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
    elif "postgresql+asyncpg" in url:
        url = url.replace("postgresql+asyncpg", "postgresql")
    
    # Handle Heroku-style postgres:// URLs
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() will emit the given string to the script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    """
    url = get_database_url()
    
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
