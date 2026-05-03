"""
Async SQLAlchemy database engine and session factory.

Local:      SQLite   (default)  — chatbot.db in project root
Production: swap DATABASE_URL to postgresql+asyncpg://...
"""

import os
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

# Ensure the directory for a SQLite file exists before the engine is created.
if settings.database_url.startswith("sqlite"):
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    db_dir = os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else "."
    os.makedirs(db_dir, exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables. Safe to call multiple times (uses CREATE IF NOT EXISTS)."""
    from backend.auth import models  # noqa: F401 — registers User with Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session
